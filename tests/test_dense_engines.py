"""Dense 3D on modern GPUs: finding the engines, saying why one cannot run, never faking.

Two dense engines sit behind the COLMAP sparse model: COLMAP patch-match stereo (GPU
only) and OpenMVS (GPU or CPU, plus a textured mesh). These tests pin the decisions
that matter to an operator, without needing either binary:

* which COLMAP / OpenMVS is used, and that a configured path that is wrong is reported
  rather than silently swapped for another install;
* that a crash of a too-new CUDA build is explained (COLMAP 4.2.1 is built with CUDA
  13.2 and dies with 0xC0000409 under a CUDA 13.1 driver, printing no CUDA error);
* that OpenMVS never runs with its default tower mode, which appends synthetic points;
* that an unavailable engine leaves the run sparse and says so.

The real runs are in test_dense_engines_integration.py, gated on the binaries.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core import openmvs
from core import reconstruction_colmap as rc
from core.settings import AppSettings, load_app_settings, save_app_settings, validate_app_settings


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Settings and probe caches in a scratch folder, nothing from the environment."""
    folder = tmp_path / "config"
    folder.mkdir()
    monkeypatch.setattr("core.settings.user_config_dir", lambda: folder)
    for name in ("ODK_COLMAP", "ODK_OPENMVS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("core.reconstruction_colmap.shutil.which", lambda _name: None)
    monkeypatch.setattr("core.openmvs.shutil.which", lambda _name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-local-install"))
    return folder


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")
    return path


def _save(**values) -> None:
    settings = load_app_settings()
    for key, value in values.items():
        setattr(settings, key, value)
    save_app_settings(settings)


def _openmvs_folder(root: Path, layout: str = "") -> Path:
    base = root / layout if layout else root
    for tool in openmvs.TOOLS:
        _exe(base / openmvs._exe_name(tool))
    return root


# ---------------------------------------------------------------------------
# COLMAP: the configured binary goes into settings and is reported with its origin
# ---------------------------------------------------------------------------


class TestColmapBinaryComesFromSettings:
    def test_the_setting_is_used_when_no_env_var_is_set(self, config_dir, tmp_path):
        exe = _exe(tmp_path / "colmap-4.0.4" / "bin" / "colmap.exe")
        _save(colmap_binary=str(tmp_path / "colmap-4.0.4"))
        assert rc.colmap_executable_source() == (str(exe), "settings:colmap_binary")

    def test_the_env_var_wins_over_the_setting(self, config_dir, tmp_path, monkeypatch):
        _exe(tmp_path / "a" / "bin" / "colmap.exe")
        env_exe = _exe(tmp_path / "b" / "bin" / "colmap.exe")
        _save(colmap_binary=str(tmp_path / "a"))
        monkeypatch.setenv("ODK_COLMAP", str(env_exe))
        assert rc.colmap_executable_source() == (str(env_exe), "env:ODK_COLMAP")

    def test_a_wrong_setting_is_not_replaced_by_another_colmap(self, config_dir, tmp_path,
                                                               monkeypatch):
        monkeypatch.setattr("core.reconstruction_colmap.shutil.which",
                            lambda _name: str(_exe(tmp_path / "path" / "colmap.exe")))
        _save(colmap_binary=str(tmp_path / "missing"))
        assert rc.colmap_executable_source() == ("", "settings:colmap_binary")
        assert rc.colmap_executable() is None

    def test_capabilities_name_the_source_and_why_dense_is_off(self, config_dir, tmp_path):
        _save(colmap_binary=str(tmp_path / "missing"))
        caps = rc.engine_capabilities(probe_dense=False)
        assert caps["colmap_binary"] == "" and caps["colmap_binary_source"] == "settings:colmap_binary"
        assert caps["dense_stereo"] is False
        assert "settings:colmap_binary" in caps["dense_stereo_reason"]
        assert caps["dense_engines"] == {"colmap": False, "openmvs": False}

    def test_capabilities_report_openmvs_and_the_chosen_engine(self, config_dir):
        caps = rc.engine_capabilities(probe_dense=False)
        assert caps["openmvs"] is False
        assert "OpenMVS is not configured" in caps["openmvs_reason"]
        assert caps["openmvs_download_url"].startswith("https://github.com/cdcseacave/openMVS")
        assert caps["dense_engine"] == "colmap"


# ---------------------------------------------------------------------------
# A crashing CUDA build is explained, not reported as its last log line
# ---------------------------------------------------------------------------


class TestCrashesAreExplained:
    def test_windows_fail_fast_is_named(self):
        text = rc.describe_exit_code(3221226505)
        assert "0xC0000409" in text and "crashed" in text

    def test_negative_ntstatus_and_signals(self):
        assert "0xC0000005" in rc.describe_exit_code(-1073741819)
        assert "signal 11" in rc.describe_exit_code(-11)

    def test_an_ordinary_error_exit_is_not_a_crash(self):
        assert rc.describe_exit_code(0) == "" and rc.describe_exit_code(1) == ""

    def test_a_crash_from_a_newer_toolkit_than_the_driver_gets_the_cause(self, monkeypatch):
        monkeypatch.setattr(rc, "binary_cuda_toolchain", lambda _b: "13.2")
        monkeypatch.setattr(rc, "driver_cuda_version", lambda: ("592.15", "13.1"))
        advice = rc.explain_dense_failure(
            "colmap.exe", "COLMAP patch_match_stereo failed on this machine: the process "
                          "crashed with exit code 0xC0000409 (fail-fast abort)")
        assert "compiled with CUDA 13.2" in advice and "supports CUDA only up to 13.1" in advice

    def test_a_crash_without_a_version_gap_is_not_blamed_on_the_driver(self, monkeypatch):
        monkeypatch.setattr(rc, "binary_cuda_toolchain", lambda _b: "12.9")
        monkeypatch.setattr(rc, "driver_cuda_version", lambda: ("592.15", "13.1"))
        assert rc.explain_dense_failure("colmap.exe", "patch_match_stereo crashed with ...") == ""


# ---------------------------------------------------------------------------
# OpenMVS discovery and capability
# ---------------------------------------------------------------------------


class TestOpenMVSDiscovery:
    def test_not_configured_says_where_to_get_it(self, config_dir):
        tools, folder, origin, reason = openmvs.locate()
        assert tools is None and folder == "" and origin == ""
        assert openmvs.DOWNLOAD_URL in reason and openmvs.ENV_VAR in reason

    def test_the_release_archive_layout_is_found(self, config_dir, tmp_path, monkeypatch):
        root = _openmvs_folder(tmp_path / "openmvs", layout="vc17/x64/Release")
        monkeypatch.setenv(openmvs.ENV_VAR, str(root))
        tools, folder, origin, reason = openmvs.locate()
        assert reason == "" and origin == f"env:{openmvs.ENV_VAR}"
        assert Path(tools["DensifyPointCloud"]).parent == root / "vc17" / "x64" / "Release"

    def test_the_setting_is_used_and_a_folder_without_tools_is_refused(self, config_dir, tmp_path):
        _save(openmvs_dir=str(tmp_path / "empty"))
        tools, _folder, origin, reason = openmvs.locate()
        assert tools is None and origin == "settings:openmvs_dir"
        assert "were not found" in reason
        _openmvs_folder(tmp_path / "empty")
        assert openmvs.locate()[0] is not None

    def test_status_without_tools_is_honest(self, config_dir):
        report = openmvs.status(probe_run=False)
        assert report["openmvs"] is False and report["openmvs_gpu"] is False

    def test_cuda_device_is_only_passed_to_a_cuda_build(self, monkeypatch):
        tools = {"DensifyPointCloud": "x"}
        monkeypatch.setattr(openmvs, "cuda_toolkit", lambda _t: "")
        assert openmvs.device_args(tools, gpu=True) == []
        monkeypatch.setattr(openmvs, "cuda_toolkit", lambda _t: "13.1")
        assert openmvs.device_args(tools, gpu=True) == ["--cuda-device", "-1"]
        assert openmvs.device_args(tools, gpu=False) == ["--cuda-device", "-2"]


class TestOpenMVSProbeFallsBackToTheCpu:
    @pytest.fixture
    def tools(self, config_dir, tmp_path, monkeypatch):
        root = _openmvs_folder(tmp_path / "openmvs")
        monkeypatch.setattr(openmvs, "cuda_toolkit", lambda _t: "13.1")
        monkeypatch.setattr(rc, "_dense_probe_key", lambda b: f"key|{b}")
        return openmvs.find_tools(root)

    def test_gpu_failure_then_cpu_success_is_a_cpu_engine(self, tools, monkeypatch):
        """The RTX 5060 case: the 2.4.0 CUDA archive has no sm_120 kernels."""
        calls = []

        def fake(_tools, gpu, timeout=600.0):
            calls.append(gpu)
            if gpu:
                return {"ok": False, "reason": "CUDA error ... named symbol not found (code 500)"}
            return {"ok": True, "reason": "densified the test scene"}

        monkeypatch.setattr(openmvs, "_run_probe", fake)
        result = openmvs.probe(tools)
        assert calls == [True, False]
        assert result["ok"] is True and result["gpu"] is False
        assert "named symbol not found" in result["reason"]
        # Cached: a second call does not re-run either probe.
        assert openmvs.probe(tools)["gpu"] is False and calls == [True, False]

    def test_nothing_works_is_reported_as_unavailable(self, tools, monkeypatch):
        monkeypatch.setattr(openmvs, "_run_probe",
                            lambda _t, gpu, timeout=600.0: {"ok": False, "reason": f"gpu={gpu}"})
        result = openmvs.probe(tools, use_cache=False)
        assert result["ok"] is False and "gpu=False" in result["reason"]

    def test_run_false_does_not_guess(self, tools):
        result = openmvs.probe(tools, run=False)
        assert result == {"ok": False, "gpu": False, "probed": False,
                          "reason": "OpenMVS has not been tested on this machine yet."}


# ---------------------------------------------------------------------------
# OpenMVS pipeline commands
# ---------------------------------------------------------------------------


class _FakeOpenMVS:
    """Stands in for the four executables: records each call and writes its outputs."""

    def __init__(self, fail_tool: str = ""):
        self.calls: list[list[str]] = []
        self.fail_tool = fail_tool

    def __call__(self, command, cwd=None, **_kwargs):
        self.calls.append(list(command))
        tool = Path(command[0]).stem
        folder = Path(cwd)
        if tool == self.fail_tool:
            (folder / f"{tool}-1.log").write_text(
                "10:00:00 [App     ] error: unable to open file 'x'\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 1, "", "")
        outputs = {
            "InterfaceCOLMAP": ["scene.mvs"],
            "DensifyPointCloud": ["scene_dense.mvs", "scene_dense.ply"],
            "ReconstructMesh": ["scene_dense_mesh.ply"],
            "TextureMesh": ["scene_dense_mesh_texture.obj", "scene_dense_mesh_texture.mtl",
                            "scene_dense_mesh_texture0.png"],
        }[tool]
        for name in outputs:
            text = "map_Kd scene_dense_mesh_texture0.png\n" if name.endswith(".mtl") else "x"
            (folder / name).write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(openmvs, "cuda_toolkit", lambda _t: "")
    return {tool: str(tmp_path / "bin" / openmvs._exe_name(tool)) for tool in openmvs.TOOLS}


class TestOpenMVSCommands:
    def test_densify_never_uses_tower_mode(self, tmp_path, fake_tools):
        """Tower mode appends a synthetic cylinder of points: not a measurement."""
        runner = _FakeOpenMVS()
        openmvs.densify_and_texture(fake_tools, tmp_path / "dense", tmp_path / "out",
                                    gpu=False, runner=runner)
        densify = next(c for c in runner.calls if Path(c[0]).stem == "DensifyPointCloud")
        assert densify[densify.index("--tower-mode") + 1] == "0"

    def test_the_full_chain_returns_the_textured_mesh(self, tmp_path, fake_tools):
        runner = _FakeOpenMVS()
        paths = openmvs.densify_and_texture(fake_tools, tmp_path / "dense", tmp_path / "out",
                                            gpu=False, max_resolution=2400, runner=runner)
        assert [Path(c[0]).stem for c in runner.calls] == list(openmvs.TOOLS)
        densify = runner.calls[1]
        assert densify[densify.index("--max-resolution") + 1] == "2400"
        assert "--cuda-device" not in densify  # a CPU build rejects the option
        assert Path(paths["textured_obj"]).name == "scene_dense_mesh_texture.obj"
        assert Path(paths["texture_image"]).name == "scene_dense_mesh_texture0.png"

    def test_a_failing_step_is_named_with_its_log_line(self, tmp_path, fake_tools):
        runner = _FakeOpenMVS(fail_tool="DensifyPointCloud")
        with pytest.raises(openmvs.OpenMVSFailed) as caught:
            openmvs.densify_and_texture(fake_tools, tmp_path / "dense", tmp_path / "out",
                                        gpu=False, runner=runner)
        assert "DensifyPointCloud" in str(caught.value)
        assert "unable to open file" in str(caught.value)


# ---------------------------------------------------------------------------
# The reconstruction engine: unavailable stays sparse, available is reported
# ---------------------------------------------------------------------------


class _Model:
    def write(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _progress(_p, _m):
    return None


class TestTheEngineStaysSparseWhenDenseCannotRun:
    def test_openmvs_not_configured_is_a_warning_and_no_points(self, config_dir, tmp_path):
        engine = rc.ColmapReconstructor(dense=True, dense_engine="openmvs")
        xyz, rgb = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape == (0, 3) and rgb.shape == (0, 3)
        assert any(w.startswith("Dense MVS skipped: OpenMVS is not configured")
                   for w in engine.warnings), engine.warnings
        assert engine.dense_report == {"engine": "openmvs", "ran": False}

    def test_openmvs_that_failed_its_probe_is_not_run(self, config_dir, tmp_path, monkeypatch):
        root = _openmvs_folder(tmp_path / "openmvs")
        monkeypatch.setenv(openmvs.ENV_VAR, str(root))
        monkeypatch.setattr(openmvs, "probe", lambda _t, **_k: {
            "ok": False, "gpu": False, "reason": "crashed"})
        called = []
        monkeypatch.setattr(openmvs, "densify_and_texture", lambda *a, **k: called.append(1))
        engine = rc.ColmapReconstructor(dense=True, dense_engine="openmvs")
        xyz, _ = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape[0] == 0 and not called
        assert "OpenMVS cannot densify on this machine: crashed" in " ".join(engine.warnings)

    def test_colmap_engine_without_a_binary_stays_sparse(self, config_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(rc, "pycolmap_has_cuda", lambda: False)
        engine = rc.ColmapReconstructor(dense=True, dense_engine="colmap")
        xyz, _ = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape[0] == 0
        assert "Dense MVS skipped" in " ".join(engine.warnings)

    def test_dense_off_runs_nothing(self, config_dir, tmp_path):
        engine = rc.ColmapReconstructor(dense=False, dense_engine="openmvs")
        xyz, _ = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape[0] == 0 and engine.warnings == []


class TestTheOpenMVSEngineRuns:
    def test_the_dense_cloud_and_textured_mesh_are_reported(self, config_dir, tmp_path,
                                                            monkeypatch):
        root = _openmvs_folder(tmp_path / "openmvs")
        monkeypatch.setenv(openmvs.ENV_VAR, str(root))
        monkeypatch.setattr(openmvs, "probe", lambda _t, **_k: {
            "ok": True, "gpu": False, "reason": "CPU: densified the test scene."})
        cloud = tmp_path / "work" / "openmvs" / "scene_dense.ply"

        def densify(tools, undistorted, out_dir, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_ply_with_lists(cloud, np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1.5]], float))
            assert kwargs["gpu"] is False and kwargs["max_resolution"] == 2400
            return {"dense_cloud": str(cloud), "textured_obj": str(out_dir / "m.obj"),
                    "textured_mtl": str(out_dir / "m.mtl"), "texture_image": str(out_dir / "t.png")}

        monkeypatch.setattr(openmvs, "densify_and_texture", densify)
        engine = rc.ColmapReconstructor(profile="standard", dense=True, dense_engine="openmvs")
        monkeypatch.setattr(engine, "_undistort_for_dense", lambda *a: "")
        xyz, rgb = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape == (3, 3) and rgb[2].tolist() == [30, 40, 50]
        assert engine.dense_report["ran"] is True and engine.dense_report["gpu"] is False
        assert engine.dense_report["textured_obj"].endswith("m.obj")
        assert any(w.startswith("OpenMVS densified on the CPU") for w in engine.warnings)

    def test_an_undistortion_failure_stays_sparse(self, config_dir, tmp_path, monkeypatch):
        root = _openmvs_folder(tmp_path / "openmvs")
        monkeypatch.setenv(openmvs.ENV_VAR, str(root))
        monkeypatch.setattr(openmvs, "probe", lambda _t, **_k: {"ok": True, "gpu": True})
        engine = rc.ColmapReconstructor(dense=True, dense_engine="openmvs")
        monkeypatch.setattr(engine, "_undistort_for_dense", lambda *a: "no undistorter")
        xyz, _ = engine._run_dense(tmp_path, tmp_path / "work", _Model(), _progress)
        assert xyz.shape[0] == 0
        assert "Dense MVS skipped: no undistorter" in " ".join(engine.warnings)


# ---------------------------------------------------------------------------
# Engine choice: settings, validation, factory, server options
# ---------------------------------------------------------------------------


class TestEngineChoice:
    def test_default_is_colmap_and_the_setting_changes_it(self, config_dir):
        assert rc.normalize_dense_engine(None) == "colmap"
        _save(dense_engine="openmvs")
        assert rc.normalize_dense_engine(None) == "openmvs"
        assert rc.normalize_dense_engine("COLMAP") == "colmap"

    def test_an_unknown_engine_is_refused(self, config_dir):
        with pytest.raises(ValueError):
            rc.normalize_dense_engine("meshroom")
        messages = validate_app_settings(AppSettings(dense_engine="meshroom"))
        assert any(m.field == "dense_engine" for m in messages)

    def test_the_factory_passes_the_engine(self, config_dir):
        if not rc.colmap_available():
            pytest.skip("pycolmap is not installed")
        engine = rc.build_reconstructor("colmap", dense_engine="openmvs")
        assert engine.dense_engine == "openmvs"

    def test_cloud_jobs_carry_the_engine(self):
        from core.reconstruction_remote import job_options

        assert job_options(dense_engine="openmvs") == {"dense_engine": "openmvs"}
        assert job_options() == {}

    def test_the_server_accepts_and_maps_the_option(self):
        from services.api.routers.processing import RECONSTRUCTION_OPTIONS, _reconstruction_settings

        assert "dense_engine" in RECONSTRUCTION_OPTIONS
        assert _reconstruction_settings({"dense_engine": "openmvs"}) == {"dense_engine": "openmvs"}


class TestTheDesktopApi:
    @pytest.fixture
    def api(self, config_dir, tmp_path):
        from cockpit_harness import make_api

        return make_api(tmp_path)

    def test_a_folder_without_openmvs_is_refused_and_nothing_saved(self, api, tmp_path):
        result = api.set_openmvs_dir(str(tmp_path))
        assert result["ok"] is False and "were not found" in result["error"]
        assert load_app_settings().openmvs_dir == ""

    def test_a_missing_colmap_is_refused(self, api, tmp_path):
        result = api.set_colmap_binary(str(tmp_path / "nothing"))
        assert result["ok"] is False and "No colmap executable" in result["error"]

    def test_a_saved_colmap_is_tested_not_assumed(self, api, tmp_path, monkeypatch):
        _exe(tmp_path / "c" / "bin" / "colmap.exe")
        monkeypatch.setattr(rc, "native_dense_probe", lambda b, run=True: {
            "ok": False, "probed": True, "reason": "crashed with 0xC0000409"})
        result = api.set_colmap_binary(str(tmp_path / "c"))
        assert result["ok"] and result["colmap_binary_source"] == "settings:colmap_binary"
        assert result["dense_stereo"] is False and "0xC0000409" in result["dense_stereo_reason"]
        assert api.dense_engine_settings()["colmap_binary"] == str(tmp_path / "c")

    def test_the_engine_choice_is_saved_and_validated(self, api):
        assert api.set_dense_engine("openmvs")["ok"]
        assert load_app_settings().dense_engine == "openmvs"
        assert api.set_dense_engine("nerf")["ok"] is False

    def test_the_capability_answer_is_for_the_chosen_engine(self, api, monkeypatch):
        # COLMAP dense works, OpenMVS is not configured: with OpenMVS chosen the answer
        # must be "no" and name OpenMVS's reason, not borrow COLMAP's "yes".
        base = {"dense_stereo": True, "dense_stereo_reason": "ran",
                "openmvs": False, "openmvs_gpu": False,
                "openmvs_reason": "OpenMVS is not configured.",
                "dense_engines": {"colmap": True, "openmvs": False}}
        monkeypatch.setattr(rc, "engine_capabilities",
                            lambda probe_dense=True: dict(base, dense_engine="openmvs"))
        result = api.reconstruction_capabilities()
        assert result["dense_available"] is False and result["dense_engine"] == "openmvs"
        assert "NOT available" in result["note"] and "not configured" in result["note"]
        monkeypatch.setattr(rc, "engine_capabilities",
                            lambda probe_dense=True: dict(base, dense_engine="colmap"))
        result = api.reconstruction_capabilities()
        assert result["dense_available"] is True and "is available" in result["note"]
        assert result["dense_engines"] == {"colmap": True, "openmvs": False}


# ---------------------------------------------------------------------------
# OpenMVS clouds carry per-point view lists
# ---------------------------------------------------------------------------


def _write_ply_with_lists(path: Path, xyz: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property list uint8 uint32 view_indices\n"
        "property list uint8 float32 view_weights\nend_header\n"
    ).encode("ascii")
    body = b""
    for index, (x, y, z) in enumerate(xyz):
        views = list(range(index + 1))
        body += struct.pack("<fffBBB", x, y, z, 10 + 10 * index, 20 + 10 * index, 30 + 10 * index)
        body += struct.pack("<B", len(views)) + struct.pack(f"<{len(views)}I", *views)
        body += struct.pack("<B", len(views)) + struct.pack(f"<{len(views)}f", *[0.5] * len(views))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)


def test_a_ply_with_view_lists_is_read_exactly(tmp_path):
    xyz = np.array([[0.5, -1.25, 3.0], [10.0, 20.0, 30.0], [-7.5, 0.0, 1.5]])
    path = tmp_path / "scene_dense.ply"
    _write_ply_with_lists(path, xyz)
    read_xyz, read_rgb = rc._read_ply(path)
    assert np.allclose(read_xyz, xyz)
    assert read_rgb.tolist() == [[10, 20, 30], [20, 30, 40], [30, 40, 50]]
