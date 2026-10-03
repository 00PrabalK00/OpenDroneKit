"""OpenMVS as a dense engine: densify a COLMAP model and build a textured mesh.

OpenMVS (AGPL-3.0, https://github.com/cdcseacave/openMVS) is run as separate
executables the operator installs; nothing of it is linked into or shipped with
OpenDroneKit. It reads the undistorted COLMAP workspace, so the sparse model, the
georeferencing and every raster product stay exactly as the COLMAP engine makes them --
only the point cloud they are built from changes, and a textured OBJ is added.

Capability is decided by running the tools, as with COLMAP patch-match: a two-view
synthetic scene is densified once (cached per install and GPU/driver) and the plane it
recovers is checked against the plane that was drawn. GPU is tried first and CPU second,
because the official 2.4.0 CUDA archive initialises an RTX 50-series card and then fails
with "named symbol not found" -- it carries no sm_120 kernels -- while its CPU path works.

Two OpenMVS defaults are overridden on purpose:

* ``--tower-mode 0``. The default (4) *appends a cylinder of synthetic points* at the
  centre of the region of interest when it thinks the scene is a tower. Those points
  are not measurements, and a DSM built through them would show a structure that
  nothing observed.
* ``--cuda-device -2`` (CPU) whenever the GPU probe failed, so a GPU the build cannot
  drive is not retried on every survey.

The mesh is an interpolated surface (ReconstructMesh closes small holes and smooths);
it is a visual model. Measurements and rasters come from the fused point cloud.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ENV_VAR = "ODK_OPENMVS_DIR"
TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh")
DOWNLOAD_URL = "https://github.com/cdcseacave/openMVS/releases"
_PROBE_VERSION = 1
# Overrides applied to every DensifyPointCloud run; see the module docstring.
DENSIFY_SAFETY_ARGS = ("--tower-mode", "0")


class OpenMVSUnavailable(RuntimeError):
    """The OpenMVS tools are not configured, not found, or cannot run here."""


class OpenMVSFailed(RuntimeError):
    """An OpenMVS step ran and failed; the message names the step and its log line."""


def _exe_name(tool: str) -> str:
    return f"{tool}.exe" if os.name == "nt" else tool


def find_tools(folder: str | Path) -> dict[str, str] | None:
    """The four tools in `folder` (or the usual sub-folders of a release archive)."""
    base = Path(folder)
    for candidate in (base, base / "bin", base / "vc17" / "x64" / "Release",
                      base / "x64" / "Release"):
        tools = {tool: candidate / _exe_name(tool) for tool in TOOLS}
        if all(path.is_file() for path in tools.values()):
            return {tool: str(path) for tool, path in tools.items()}
    return None


def configured_dir() -> tuple[str, str]:
    """(folder, origin): ODK_OPENMVS_DIR, then the openmvs_dir setting, then PATH."""
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return override, f"env:{ENV_VAR}"
    try:
        from .settings import load_app_settings

        configured = str(load_app_settings().openmvs_dir or "").strip()
    except Exception:  # noqa: BLE001 - unreadable settings must not hide a PATH install
        configured = ""
    if configured:
        return configured, "settings:openmvs_dir"
    found = shutil.which("DensifyPointCloud")
    if found:
        return str(Path(found).parent), "PATH"
    return "", ""


def locate() -> tuple[dict[str, str] | None, str, str, str]:
    """(tools, folder, origin, reason-if-missing)."""
    folder, origin = configured_dir()
    if not folder:
        return None, "", "", (
            f"OpenMVS is not configured. Download it from {DOWNLOAD_URL} (Windows: "
            "OpenMVS_Windows_x64.zip), extract it, and set the folder in Settings "
            f"(openmvs_dir) or the {ENV_VAR} environment variable.")
    tools = find_tools(folder)
    if tools is None:
        missing = ", ".join(_exe_name(tool) for tool in TOOLS)
        return None, folder, origin, (
            f"OpenMVS tools were not found in {folder} ({origin}); expected {missing}.")
    return tools, folder, origin, ""


def cuda_toolkit(tools: dict[str, str]) -> str:
    """The CUDA release DensifyPointCloud was compiled with, or "" for a CPU build."""
    from .reconstruction_colmap import binary_cuda_toolchain

    return binary_cuda_toolchain(tools["DensifyPointCloud"])


def device_args(tools: dict[str, str], gpu: bool) -> list[str]:
    """--cuda-device for a CUDA build (-1 best GPU, -2 CPU); nothing for a CPU build.

    The CPU-only Windows archive does not know the option and exits 1 without writing a
    log when it is given, so it must not be passed there.
    """
    if not cuda_toolkit(tools):
        return []
    return ["--cuda-device", "-1" if gpu else "-2"]


def _version(tools: dict[str, str]) -> str:
    """OpenMVS writes its banner to a log, not stdout; run --help in a scratch folder."""
    with tempfile.TemporaryDirectory(prefix="odk_openmvs_ver_") as tmp:
        try:
            subprocess.run([tools["InterfaceCOLMAP"], "--help"], cwd=tmp,
                           capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return ""
        for log in Path(tmp).glob("*.log"):
            match = re.search(r"OpenMVS \S+ v([\d.]+)", log.read_text("utf-8", "replace"))
            if match:
                return match.group(1)
    return ""


def _log_text(folder: Path, tool: str) -> str:
    logs = sorted(folder.glob(f"{tool}-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1].read_text("utf-8", "replace") if logs else ""


def _log_error(text: str) -> str:
    for line in text.splitlines():
        lowered = line.lower()
        if "error" in lowered and "errors" not in lowered:
            return line.split("]", 1)[-1].strip()
    return ""


def run_tool(tools: dict[str, str], tool: str, args: Sequence[str], cwd: Path, *,
             timeout: float | None = None,
             runner: Callable[..., subprocess.CompletedProcess] | None = None
             ) -> subprocess.CompletedProcess:
    """Run one OpenMVS tool in `cwd` (where it writes its log); raise OpenMVSFailed."""
    from .reconstruction_colmap import describe_exit_code

    command = [tools[tool], *args]
    run = runner or subprocess.run
    try:
        done = run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise OpenMVSFailed(f"OpenMVS {tool} did not finish within {timeout:.0f} s.") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenMVSFailed(f"OpenMVS {tool} could not run: {exc}") from exc
    if done.returncode != 0:
        detail = _log_error(_log_text(cwd, tool)) or describe_exit_code(done.returncode) \
            or f"exit code {done.returncode}"
        raise OpenMVSFailed(f"OpenMVS {tool} failed: {detail}")
    return done


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def _probe_cache_file() -> Path:
    from .settings import user_config_dir

    return user_config_dir() / "openmvs_probe.json"


def _probe_key(tools: dict[str, str]) -> str:
    from .reconstruction_colmap import _dense_probe_key

    # Same identity rule as the COLMAP probe: the binary and the GPU/driver it ran on.
    return f"openmvs{_PROBE_VERSION}|" + _dense_probe_key(tools["DensifyPointCloud"])


def _run_probe(tools: dict[str, str], gpu: bool, timeout: float = 600.0) -> dict[str, Any]:
    """Densify the synthetic two-view plane; ok only if the recovered depth is right."""
    from .reconstruction_colmap import _read_ply, write_probe_scene

    with tempfile.TemporaryDirectory(prefix="odk_openmvs_probe_") as tmp:
        root = Path(tmp)
        write_probe_scene(root, size=256, grid=10, baseline=0.4)
        try:
            run_tool(tools, "InterfaceCOLMAP",
                     ["-i", ".", "-o", "scene.mvs", "--image-folder", "images"],
                     root, timeout=timeout)
            run_tool(tools, "DensifyPointCloud",
                     ["-i", "scene.mvs", "-o", "scene_dense.mvs", "--resolution-level", "0",
                      "--min-resolution", "128", "--max-resolution", "256",
                      "--fusion-filter", "1", *device_args(tools, gpu),
                      *DENSIFY_SAFETY_ARGS],
                     root, timeout=timeout)
        except OpenMVSFailed as exc:
            return {"ok": False, "reason": str(exc)}
        cloud = root / "scene_dense.ply"
        if not cloud.is_file():
            return {"ok": False, "reason": "OpenMVS DensifyPointCloud wrote no point cloud."}
        xyz, _rgb = _read_ply(cloud)
        if xyz.shape[0] < 1000:
            return {"ok": False, "reason": f"OpenMVS densified only {xyz.shape[0]} points "
                                           "on the test scene."}
        depth = float(np.median(xyz[:, 2]))
        if abs(depth - 2.0) > 0.1:
            return {"ok": False, "reason": f"OpenMVS put the test plane at depth {depth:.2f} "
                                           "instead of 2.00; its output cannot be trusted."}
    return {"ok": True, "reason": f"OpenMVS densified the test scene ({xyz.shape[0]:,} points "
                                  f"at depth {depth:.3f} of 2.000)."}


def probe(tools: dict[str, str], *, run: bool = True, use_cache: bool = True) -> dict[str, Any]:
    """{"ok", "gpu", "reason", "probed"}: GPU densify first, then CPU, cached."""
    key = _probe_key(tools)
    cache_file = _probe_cache_file()
    cache: dict[str, Any] = {}
    if use_cache and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    if use_cache and isinstance(cache.get(key), dict):
        return dict(cache[key], probed=True)
    if not run:
        return {"ok": False, "gpu": False, "probed": False,
                "reason": "OpenMVS has not been tested on this machine yet."}

    gpu_result: dict[str, Any] = {"ok": False, "reason": "CPU-only OpenMVS build."}
    if cuda_toolkit(tools):
        gpu_result = _run_probe(tools, gpu=True)
    if gpu_result["ok"]:
        result = {"ok": True, "gpu": True, "reason": f"GPU: {gpu_result['reason']}"}
    else:
        cpu_result = _run_probe(tools, gpu=False)
        if cpu_result["ok"]:
            result = {"ok": True, "gpu": False,
                      "reason": f"CPU: {cpu_result['reason']} GPU densify unavailable: "
                                f"{gpu_result['reason']}"}
        else:
            result = {"ok": False, "gpu": False,
                      "reason": f"{cpu_result['reason']} (GPU: {gpu_result['reason']})"}
    if use_cache:
        cache[key] = result
        try:
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError:
            pass
    return dict(result, probed=True)


def status(probe_run: bool = True) -> dict[str, Any]:
    """What Environment Capabilities shows for OpenMVS."""
    tools, folder, origin, reason = locate()
    report: dict[str, Any] = {
        "openmvs": False, "openmvs_gpu": False, "openmvs_dir": folder,
        "openmvs_configured_by": origin, "openmvs_version": "",
        "openmvs_cuda_toolkit": "", "openmvs_reason": reason,
        "openmvs_download_url": DOWNLOAD_URL, "openmvs_env_var": ENV_VAR,
    }
    if tools is None:
        return report
    report["openmvs_version"] = _version(tools)
    report["openmvs_cuda_toolkit"] = cuda_toolkit(tools)
    result = probe(tools, run=probe_run)
    report.update(openmvs=bool(result.get("ok")), openmvs_gpu=bool(result.get("gpu")),
                  openmvs_reason=str(result.get("reason", "")),
                  openmvs_probed=bool(result.get("probed")))
    return report


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def densify_and_texture(tools: dict[str, str], undistorted: Path, out_dir: Path, *,
                        gpu: bool, max_resolution: int = 2560, texture: bool = True,
                        progress: Callable[[int, str], None] | None = None,
                        runner: Callable[..., subprocess.CompletedProcess] | None = None,
                        timeout: float | None = None) -> dict[str, str]:
    """COLMAP undistorted workspace -> dense cloud, mesh and textured OBJ.

    `undistorted` is the output of COLMAP's image_undistorter (images/ + sparse/).
    Returns the paths written; raises OpenMVSFailed naming the step that failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tick = progress or (lambda _p, _m: None)
    device = device_args(tools, gpu)
    images = (undistorted / "images").resolve()

    tick(62, "OpenMVS: importing the COLMAP model")
    run_tool(tools, "InterfaceCOLMAP",
             ["-i", str(undistorted.resolve()), "-o", "scene.mvs", "--image-folder", str(images)],
             out_dir, runner=runner, timeout=timeout)
    tick(64, f"OpenMVS: densifying on the {'GPU' if gpu else 'CPU'}")
    run_tool(tools, "DensifyPointCloud",
             ["-i", "scene.mvs", "-o", "scene_dense.mvs", "--resolution-level", "0",
              "--max-resolution", str(int(max_resolution)), *device,
              *DENSIFY_SAFETY_ARGS],
             out_dir, runner=runner, timeout=timeout)
    paths = {"dense_cloud": str(out_dir / "scene_dense.ply"),
             "dense_scene": str(out_dir / "scene_dense.mvs")}
    if not Path(paths["dense_cloud"]).is_file():
        raise OpenMVSFailed("OpenMVS DensifyPointCloud finished but wrote no point cloud.")
    if not texture:
        return paths
    # The dense cloud is the measurement; the mesh is a visual model built from it.
    # A meshing or texturing failure therefore costs the textured OBJ, not the cloud:
    # it is reported in paths["mesh_error"] and the caller still builds every product
    # from OpenMVS's own cloud.
    try:
        paths.update(_mesh_and_texture(tools, out_dir, device, tick, runner, timeout))
    except OpenMVSFailed as exc:
        paths["mesh_error"] = str(exc)
    return paths


# ReconstructMesh --min-point-distance attempts, in pixels. None is OpenMVS's default
# (1.5). A 77-image, 14.7-million-point Aukerman cloud crashed the default Delaunay
# graph-cut on a 16 GB laptop (0xC0000409 after weighting 41 M cells); 4 px merged it to
# 6.5 M -> ~1 M mesh vertices and finished in 4 minutes at 2.4 GB. The cloud itself is
# untouched; only the mesh is coarser, and the value used is reported.
MESH_MIN_POINT_DISTANCES: tuple[float | None, ...] = (None, 4.0)


def _mesh_and_texture(tools: dict[str, str], out_dir: Path, device: list[str],
                      tick: Callable[[int, str], None],
                      runner: Callable[..., subprocess.CompletedProcess] | None,
                      timeout: float | None) -> dict[str, str]:
    mesh = out_dir / "scene_dense_mesh.ply"
    failures: list[str] = []
    used: float | None = None
    for distance in MESH_MIN_POINT_DISTANCES:
        extra = [] if distance is None else ["--min-point-distance", f"{distance:g}"]
        label = "default point spacing" if distance is None else f"points merged within {distance:g} px"
        tick(68, f"OpenMVS: meshing the dense cloud ({label})")
        try:
            run_tool(tools, "ReconstructMesh",
                     ["-i", "scene_dense.mvs", "-p", "scene_dense.ply", "-o", mesh.name,
                      *extra, *device],
                     out_dir, runner=runner, timeout=timeout)
        except OpenMVSFailed as exc:
            failures.append(f"{label}: {exc}")
            continue
        if mesh.is_file():
            used = distance
            break
        failures.append(f"{label}: ReconstructMesh finished but wrote no mesh")
    else:
        raise OpenMVSFailed("OpenMVS could not mesh the dense cloud (" + "; ".join(failures) + ")")
    tick(70, "OpenMVS: texturing the mesh")
    run_tool(tools, "TextureMesh",
             ["-i", "scene_dense.mvs", "-m", mesh.name,
              "-o", "scene_dense_mesh_texture.obj", "--export-type", "obj",
              *device],
             out_dir, runner=runner, timeout=timeout)
    obj = out_dir / "scene_dense_mesh_texture.obj"
    if not obj.is_file():
        raise OpenMVSFailed("OpenMVS TextureMesh finished but wrote no OBJ.")
    paths = {"mesh": str(mesh), "textured_obj": str(obj),
             "mesh_min_point_distance_px": "default" if used is None else f"{used:g}"}
    if failures:
        paths["mesh_retries"] = "; ".join(failures)
    mtl = obj.with_suffix(".mtl")
    if mtl.is_file():
        paths["textured_mtl"] = str(mtl)
        for line in mtl.read_text("utf-8", "replace").splitlines():
            if line.strip().lower().startswith("map_kd"):
                image = out_dir / line.split(None, 1)[1].strip()
                if image.is_file():
                    paths["texture_image"] = str(image)
                    break
    return paths
