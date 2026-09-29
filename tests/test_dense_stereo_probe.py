"""Dense stereo is reported from a run, not from a file existing.

engine_capabilities() said dense_stereo=True whenever a native COLMAP binary was found,
and the UI showed "Dense MVS: yes". On an RTX 5060 the installed CUDA build has no
sm_120 kernels, so every patch-match run died with "the provided PTX was compiled with
an unsupported toolchain" -- after the whole dataset had been undistorted. The report
is now the result of running patch-match once on a tiny scene, with the reason.

The real-binary test runs whatever COLMAP this machine has. The other two stand in for
the external binary only (a script that answers like COLMAP does), so the decision
logic is exercised on both outcomes regardless of the GPU in the test machine.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap

import pytest

from core import reconstruction_colmap as rc


@pytest.fixture(autouse=True)
def isolated_probe_cache(tmp_path, monkeypatch):
    """The probe caches per user; tests must neither read nor write the real cache."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "appdata"))
    yield


def _fake_colmap(tmp_path: Path, behaviour: str) -> Path:
    """A stand-in executable that behaves like COLMAP's two dense subcommands."""
    script = tmp_path / "fake_colmap.py"
    script.write_text(textwrap.dedent(f"""
        import sys, pathlib
        args = sys.argv[1:]
        command = args[0]
        opts = dict(zip(args[1::2], args[2::2]))
        if command == "image_undistorter":
            out = pathlib.Path(opts["--output_path"])
            (out / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
            sys.exit(0)
        if command == "patch_match_stereo":
            if {behaviour!r} == "ptx":
                print("E0928 cudacc.cc:59] CUDA error at patch_match_cuda.cu:1683 - "
                      "the provided PTX was compiled with an unsupported toolchain.", file=sys.stderr)
                sys.exit(1)
            ws = pathlib.Path(opts["--workspace_path"])
            (ws / "stereo" / "depth_maps" / "a.png.geometric.bin").write_bytes(b"depth")
            sys.exit(0)
        sys.exit(2)
    """), encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / "colmap.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="ascii")
    else:
        wrapper = tmp_path / "colmap"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="ascii")
        wrapper.chmod(0o755)
    return wrapper


def test_a_build_that_cannot_run_patch_match_is_reported_unavailable_with_the_reason(tmp_path, monkeypatch):
    binary = _fake_colmap(tmp_path, "ptx")
    monkeypatch.setenv("ODK_COLMAP", str(binary))
    monkeypatch.setattr(rc, "pycolmap_has_cuda", lambda: False)

    caps = rc.engine_capabilities()

    assert caps["colmap_binary"] == str(binary)
    assert caps["dense_stereo"] is False, "a binary that exists is not a binary that runs"
    assert "unsupported toolchain" in caps["dense_stereo_reason"]


def test_a_build_that_writes_depth_maps_is_reported_available(tmp_path, monkeypatch):
    binary = _fake_colmap(tmp_path, "works")
    monkeypatch.setenv("ODK_COLMAP", str(binary))
    monkeypatch.setattr(rc, "pycolmap_has_cuda", lambda: False)

    caps = rc.engine_capabilities()

    assert caps["dense_stereo"] is True
    assert caps["dense_stereo_probed"] is True


def test_the_answer_is_cached_and_not_guessed_before_the_first_probe(tmp_path, monkeypatch):
    binary = _fake_colmap(tmp_path, "ptx")
    monkeypatch.setattr(rc, "pycolmap_has_cuda", lambda: False)

    before = rc.native_dense_probe(str(binary), run=False)
    assert before["ok"] is False and before["probed"] is False
    assert "not been tested" in before["reason"]

    first = rc.native_dense_probe(str(binary))
    cached = rc.native_dense_probe(str(binary), run=False)
    assert cached["probed"] is True and cached["ok"] == first["ok"] is False


def test_the_dense_stage_skips_with_the_probe_reason_instead_of_undistorting_everything(tmp_path, monkeypatch):
    binary = _fake_colmap(tmp_path, "ptx")
    monkeypatch.setenv("ODK_COLMAP", str(binary))
    monkeypatch.setattr(rc, "pycolmap_has_cuda", lambda: False)
    engine = rc.ColmapReconstructor(profile="standard", dense=True)

    xyz, _rgb = engine._run_dense(tmp_path, tmp_path / "work", sparse_model=None, progress=lambda *_: None)

    assert xyz.shape[0] == 0
    assert len(engine.warnings) == 1, "one outcome, one warning -- not a contradictory pair"
    assert "unsupported toolchain" in engine.warnings[0]
    assert not (tmp_path / "work" / "dense").exists(), "nothing should be undistorted for a build that cannot stereo"


def test_this_machines_colmap_is_probed_for_real():
    """Runs the installed binary. On the RTX 5060 dev box this reports the PTX failure."""
    binary = rc.colmap_executable()
    if not binary:
        pytest.skip("No native COLMAP binary on this machine.")
    probe = rc.native_dense_probe(binary, use_cache=False)
    assert probe["probed"] is True
    assert isinstance(probe["ok"], bool)
    assert probe["reason"], "the answer must say why"
    caps = rc.engine_capabilities()
    assert caps["dense_stereo"] == (probe["ok"] or (rc._pycolmap_has_dense() and rc.pycolmap_has_cuda()))


def _stamped_binary(tmp_path: Path, release: str) -> Path:
    """A file carrying the stamp nvcc writes into what it compiles."""
    binary = tmp_path / "bin" / "colmap.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\0" * 4096 + f"Cuda compilation tools, release {release}, V{release}.51".encode()
                       + b"\0" * 4096)
    return binary


def test_the_toolkit_is_read_from_the_binary_and_the_bat_launcher_resolves_to_it(tmp_path):
    binary = _stamped_binary(tmp_path, "13.2")
    (tmp_path / "COLMAP.bat").write_text("@echo off\r\n", encoding="ascii")
    assert rc.binary_cuda_toolchain(str(binary)) == "13.2"
    assert rc.binary_cuda_toolchain(str(tmp_path / "COLMAP.bat")) == "13.2"
    assert rc.binary_cuda_toolchain(str(tmp_path / "missing.exe")) == ""


def test_a_toolkit_newer_than_the_driver_is_named_as_the_cause_with_the_fix(tmp_path, monkeypatch):
    """The RTX 5060 box: official COLMAP 4.1.1 is CUDA 13.2, driver 592.15 supports 13.1."""
    binary = _stamped_binary(tmp_path, "13.2")
    monkeypatch.setattr(rc, "driver_cuda_version", lambda: ("592.15", "13.1"))
    advice = rc.explain_dense_failure(str(binary), "CUDA error ... unsupported toolchain.")
    assert "compiled with CUDA 13.2" in advice
    assert "driver 592.15 supports CUDA only up to 13.1" in advice
    assert "ODK_COLMAP" in advice and "INSTALLATION.md" in advice


def test_an_unknown_cause_is_not_guessed(tmp_path, monkeypatch):
    binary = tmp_path / "colmap.exe"
    binary.write_bytes(b"no stamp here")
    monkeypatch.setattr(rc, "driver_cuda_version", lambda: ("", ""))
    advice = rc.explain_dense_failure(str(binary), "unsupported toolchain")
    assert "13." not in advice and "compiled with CUDA" not in advice
    assert rc.explain_dense_failure(str(binary), "out of memory") == ""


def test_the_probe_reason_carries_the_explanation(tmp_path, monkeypatch):
    binary = _fake_colmap(tmp_path, "ptx")
    monkeypatch.setattr(rc, "binary_cuda_toolchain", lambda _b: "13.2")
    monkeypatch.setattr(rc, "driver_cuda_version", lambda: ("592.15", "13.1"))
    result = rc.native_dense_probe(str(binary), use_cache=False)
    assert result["ok"] is False
    assert "unsupported toolchain" in result["reason"]
    assert "Cause: this COLMAP was compiled with CUDA 13.2" in result["reason"]
