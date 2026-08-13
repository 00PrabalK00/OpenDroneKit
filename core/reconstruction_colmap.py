"""COLMAP-backed photogrammetry engine producing georeferenced GIS artifacts.

This is the accurate path: real SIFT matching, incremental SfM with bundle adjustment,
optional dense MVS, Poisson meshing, and true-orthophoto / DSM / DTM rasters written as
Cloud-Optimized GeoTIFFs in an automatically selected UTM CRS.

The legacy `CustomDroneReconstructor` remains available as a zero-dependency fallback.
Where a capability is genuinely unavailable in the current environment (no CUDA COLMAP
for dense stereo, no geotags for georeferencing) this engine records an explicit warning
rather than substituting a fabricated result.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from . import geo
from .reconstruction import (
    SUPPORTED_EXTENSIONS,
    ReconstructionResult,
    _write_ascii_ply,
    _write_obj_mesh,
    available_reconstruction_profiles,
    _normalize_profile,
)

try:  # pragma: no cover - environment dependent
    import pycolmap

    _HAS_PYCOLMAP = True
except Exception:  # noqa: BLE001
    pycolmap = None  # type: ignore[assignment]
    _HAS_PYCOLMAP = False

try:  # pragma: no cover
    import open3d as o3d

    _HAS_OPEN3D = True
except Exception:  # noqa: BLE001
    o3d = None  # type: ignore[assignment]
    _HAS_OPEN3D = False


# Per-profile COLMAP tuning. Values map onto SIFT extraction limits, matching strategy,
# and the ground resolution of the raster products.
COLMAP_PROFILES: dict[str, dict[str, Any]] = {
    "fast_preview": {
        "max_image_size": 1600,
        "max_num_features": 4096,
        "matcher": "sequential",
        "sequential_overlap": 8,
        "dense": False,
        "raster_px_per_m": 8.0,
        "mesh_depth": 8,
    },
    "standard": {
        "max_image_size": 2400,
        "max_num_features": 8192,
        "matcher": "auto",
        "sequential_overlap": 12,
        "dense": True,
        "raster_px_per_m": 16.0,
        "mesh_depth": 9,
    },
    "inspection_high_accuracy": {
        "max_image_size": 3600,
        "max_num_features": 16384,
        "matcher": "exhaustive",
        "sequential_overlap": 20,
        "dense": True,
        "raster_px_per_m": 32.0,
        "mesh_depth": 10,
    },
}

# Beyond this many images an exhaustive O(n^2) match becomes the dominant cost, so the
# auto matcher switches to sequential + loop detection.
EXHAUSTIVE_MATCH_LIMIT = 60


def colmap_available() -> bool:
    """True when the pycolmap bindings import successfully."""
    return _HAS_PYCOLMAP


def colmap_executable() -> str | None:
    """Locate a native colmap binary, which is what enables GPU dense stereo."""
    found = shutil.which("colmap")
    if found:
        return found
    for candidate in (
        Path("C:/Program Files/COLMAP/COLMAP.bat"),
        Path("C:/Program Files/COLMAP/bin/colmap.exe"),
        Path("/usr/local/bin/colmap"),
        Path("/usr/bin/colmap"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def engine_capabilities() -> dict[str, Any]:
    """Report exactly what this environment can do, for honest UI and log reporting."""
    caps = geo.geo_capabilities()
    caps.update(
        {
            "pycolmap": _HAS_PYCOLMAP,
            "open3d": _HAS_OPEN3D,
            "colmap_binary": colmap_executable() or "",
            "pycolmap_cuda": pycolmap_has_cuda(),
            # Dense patch-match stereo needs CUDA, from either the bindings or a native
            # build. Without it only the sparse cloud is available.
            "dense_stereo": bool(colmap_executable()) or (_pycolmap_has_dense() and pycolmap_has_cuda()),
        }
    )
    return caps


def _pycolmap_has_dense() -> bool:
    return bool(_HAS_PYCOLMAP and hasattr(pycolmap, "patch_match_stereo"))


def pycolmap_has_cuda() -> bool:
    """Whether this pycolmap build can use the GPU.

    The published wheels are usually CPU-only, which means SIFT and dense patch-match
    stereo both run on the CPU or are unavailable. Knowing this up front avoids
    promising dense output the environment cannot deliver.
    """
    if not _HAS_PYCOLMAP:
        return False
    flag = getattr(pycolmap, "has_cuda", None)
    if isinstance(flag, bool):
        return flag
    if callable(flag):
        try:
            return bool(flag())
        except Exception:  # noqa: BLE001
            return False
    return False


# ---------------------------------------------------------------------------
# pycolmap compatibility shims
#
# The pycolmap API changed shape across 0.4 / 0.6 / 3.x (qvec+tvec -> cam_from_world
# rigid transforms). These accessors keep the engine working across those versions
# instead of pinning to one.
# ---------------------------------------------------------------------------


def _image_name(image: Any) -> str:
    return str(getattr(image, "name", ""))


def _rigid_pose(image: Any) -> Any:
    """The world->camera Rigid3d, whether pycolmap exposes it as a property or method.

    pycolmap 4.x made `cam_from_world` a method; earlier releases used a property.
    Reading it without calling yields a bound method, which silently fails every
    downstream attribute access.
    """
    pose = getattr(image, "cam_from_world", None)
    if pose is None:
        return None
    if callable(pose):
        try:
            return pose()
        except Exception:  # noqa: BLE001
            return None
    return pose


def _camera_center(image: Any) -> np.ndarray | None:
    """World-space camera centre for a registered image, across pycolmap versions."""
    for attr in ("projection_center", "center"):
        fn = getattr(image, attr, None)
        if callable(fn):
            try:
                return np.asarray(fn(), dtype=float).reshape(3)
            except Exception:  # noqa: BLE001
                pass
    pose = _rigid_pose(image)
    if pose is not None:
        try:
            return np.asarray(pose.inverse().translation, dtype=float).reshape(3)
        except Exception:  # noqa: BLE001
            pass
    qvec = getattr(image, "qvec", None)
    tvec = getattr(image, "tvec", None)
    if qvec is not None and tvec is not None:
        rotation = _quat_to_matrix(np.asarray(qvec, dtype=float))
        return -rotation.T @ np.asarray(tvec, dtype=float).reshape(3)
    return None


def _cam_from_world_matrix(image: Any) -> np.ndarray | None:
    """3x4 world->camera extrinsic matrix for a registered image."""
    pose = _rigid_pose(image)
    if pose is not None:
        try:
            return np.asarray(pose.matrix(), dtype=float).reshape(3, 4)
        except Exception:  # noqa: BLE001
            try:
                rotation = np.asarray(pose.rotation.matrix(), dtype=float).reshape(3, 3)
                translation = np.asarray(pose.translation, dtype=float).reshape(3, 1)
                return np.hstack([rotation, translation])
            except Exception:  # noqa: BLE001
                pass
    qvec = getattr(image, "qvec", None)
    tvec = getattr(image, "tvec", None)
    if qvec is not None and tvec is not None:
        rotation = _quat_to_matrix(np.asarray(qvec, dtype=float))
        translation = np.asarray(tvec, dtype=float).reshape(3, 1)
        return np.hstack([rotation, translation])
    return None


def _quat_to_matrix(qvec: np.ndarray) -> np.ndarray:
    """COLMAP stores quaternions as (w, x, y, z)."""
    w, x, y, z = [float(v) for v in np.asarray(qvec, dtype=float).reshape(4)]
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _camera_matrix(camera: Any) -> np.ndarray:
    """Pinhole K for a COLMAP camera, tolerating each supported camera model's layout."""
    try:
        return np.asarray(camera.calibration_matrix(), dtype=float).reshape(3, 3)
    except Exception:  # noqa: BLE001
        pass
    params = np.asarray(getattr(camera, "params", []), dtype=float)
    width = float(getattr(camera, "width", 0) or 0)
    height = float(getattr(camera, "height", 0) or 0)
    if params.size >= 4:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif params.size == 3:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx = fy = 0.9 * max(width, height)
        cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _map_values(container: Any) -> list[Any]:
    """Values of a pycolmap map container.

    pycolmap 4.x returns `ImageMap` / `Point3DMap` / `CameraMap`, which are dict-like
    but not `dict` subclasses. Iterating them yields keys, so an `isinstance(x, dict)`
    check silently produces integers instead of objects.
    """
    if container is None:
        return []
    values = getattr(container, "values", None)
    if callable(values):
        return list(values())
    return list(container)


def _map_items(container: Any) -> dict[int, Any]:
    """Key/value pairs of a pycolmap map container, keyed by integer id."""
    if container is None:
        return {}
    items = getattr(container, "items", None)
    if callable(items):
        return {int(key): value for key, value in items()}
    return {int(getattr(value, "camera_id", index)): value for index, value in enumerate(container)}


def _registered_images(reconstruction: Any) -> list[Any]:
    values = _map_values(getattr(reconstruction, "images", None))
    registered = []
    for image in values:
        flag = getattr(image, "registered", None)
        has_pose = getattr(image, "has_pose", None)
        if flag is False or has_pose is False:
            continue
        registered.append(image)
    return registered


def _mean_reprojection_error(reconstruction: Any) -> float | None:
    for attr in ("compute_mean_reprojection_error", "compute_mean_reproj_error"):
        fn = getattr(reconstruction, attr, None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:  # noqa: BLE001
                continue
    return None


def _points_and_colors(reconstruction: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract the sparse cloud as (N,3) XYZ and (N,3) uint8 RGB."""
    items = _map_values(getattr(reconstruction, "points3D", None))
    xyz: list[list[float]] = []
    rgb: list[list[int]] = []
    for point in items:
        try:
            xyz.append([float(v) for v in np.asarray(point.xyz, dtype=float).reshape(3)])
            color = np.asarray(getattr(point, "color", [200, 200, 200]), dtype=int).reshape(3)
            rgb.append([int(v) for v in color])
        except Exception:  # noqa: BLE001
            continue
    if not xyz:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz, dtype=float), np.asarray(rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ColmapReconstructor:
    """Georeferenced reconstruction engine with a drop-in `reconstruct()` signature."""

    def __init__(
        self,
        profile: str = "standard",
        *,
        use_gpu: bool = True,
        dense: bool | None = None,
        max_image_size: int | None = None,
        raster_px_per_m: float | None = None,
        target_epsg: int | None = None,
    ):
        self.profile = _normalize_profile(profile)
        settings = COLMAP_PROFILES.get(self.profile, COLMAP_PROFILES["standard"])
        self.settings = dict(settings)
        if max_image_size is not None:
            self.settings["max_image_size"] = int(max_image_size)
        if raster_px_per_m is not None:
            self.settings["raster_px_per_m"] = float(raster_px_per_m)
        if dense is not None:
            self.settings["dense"] = bool(dense)
        self.use_gpu = bool(use_gpu)
        self.target_epsg = target_epsg
        self.warnings: list[str] = []

    # -- helpers ---------------------------------------------------------

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @staticmethod
    def _image_paths(image_dir: str | Path) -> list[Path]:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        images = [p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        images.sort()
        return images

    # -- SfM -------------------------------------------------------------

    def _run_sfm(self, image_dir: Path, work_dir: Path, image_count: int, progress: Callable[[int, str], None]) -> Any:
        """Feature extraction, matching, and incremental mapping with bundle adjustment."""
        database_path = work_dir / "database.db"
        sparse_dir = work_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if database_path.exists():
            database_path.unlink()

        progress(8, "Extracting SIFT features")
        extract_kwargs: dict[str, Any] = {
            "database_path": str(database_path),
            "image_path": str(image_dir),
        }
        extraction_options = self._extraction_options()
        if extraction_options is not None:
            extract_kwargs["extraction_options"] = extraction_options
        device = self._device()
        if device is not None:
            extract_kwargs["device"] = device
        try:
            pycolmap.extract_features(**extract_kwargs)  # type: ignore[union-attr]
        except TypeError:
            # pycolmap <4 named this `sift_options`; <0.5 took positional args only.
            try:
                pycolmap.extract_features(  # type: ignore[union-attr]
                    database_path=str(database_path),
                    image_path=str(image_dir),
                    sift_options=extraction_options,
                )
            except TypeError:
                pycolmap.extract_features(str(database_path), str(image_dir))  # type: ignore[union-attr]

        matcher = str(self.settings.get("matcher", "auto"))
        if matcher == "auto":
            matcher = "exhaustive" if image_count <= EXHAUSTIVE_MATCH_LIMIT else "sequential"

        progress(24, f"Matching features ({matcher})")
        self._run_matcher(matcher, database_path, device)

        progress(45, "Incremental mapping with bundle adjustment")
        try:
            reconstructions = pycolmap.incremental_mapping(  # type: ignore[union-attr]
                database_path=str(database_path),
                image_path=str(image_dir),
                output_path=str(sparse_dir),
            )
        except TypeError:
            reconstructions = pycolmap.incremental_mapping(  # type: ignore[union-attr]
                str(database_path), str(image_dir), str(sparse_dir)
            )

        if not reconstructions:
            raise RuntimeError(
                "COLMAP could not register any images. The dataset may lack overlap, "
                "be motion blurred, or be too small for structure-from-motion."
            )

        models = list(reconstructions.values()) if isinstance(reconstructions, dict) else list(reconstructions)
        # Multiple disconnected models mean the imagery split into separate clusters;
        # the largest is the usable one and the split is worth reporting.
        best = max(models, key=lambda model: len(_registered_images(model)))
        if len(models) > 1:
            self._warn(
                f"COLMAP produced {len(models)} disconnected models; using the largest "
                f"({len(_registered_images(best))} of {image_count} images). Increase overlap to merge them."
            )
        return best

    def _worker_threads(self) -> int:
        """Cap COLMAP's worker threads.

        COLMAP defaults to one thread per core. On a laptop with many cores and
        20 MP imagery each thread holds a full-resolution pyramid, which exhausts
        memory and crashes the extractor outright.
        """
        import os

        cores = os.cpu_count() or 4
        return max(1, min(8, cores - 2))

    def _extraction_options(self) -> Any:
        """Build feature-extraction options across the 4.x and legacy layouts.

        In pycolmap 4.x the SIFT knobs moved onto a nested `.sift` member; older
        releases exposed them flat on SiftExtractionOptions.
        """
        max_image_size = int(self.settings["max_image_size"])
        max_features = int(self.settings["max_num_features"])

        factory = getattr(pycolmap, "FeatureExtractionOptions", None)
        if factory is not None:
            try:
                options = factory()
                options.max_image_size = max_image_size
                if hasattr(options, "sift"):
                    options.sift.max_num_features = max_features
                if hasattr(options, "use_gpu"):
                    options.use_gpu = bool(self.use_gpu)
                if hasattr(options, "num_threads"):
                    options.num_threads = self._worker_threads()
                return options
            except Exception:  # noqa: BLE001
                pass

        legacy = getattr(pycolmap, "SiftExtractionOptions", None)
        if legacy is None:
            return None
        try:
            options = legacy()
            if hasattr(options, "max_image_size"):
                options.max_image_size = max_image_size
            options.max_num_features = max_features
            return options
        except Exception:  # noqa: BLE001
            return None

    def _matching_options(self) -> Any:
        """Feature-matching options, mainly to enable GPU matching when available."""
        factory = getattr(pycolmap, "FeatureMatchingOptions", None)
        if factory is None:
            return None
        try:
            options = factory()
            if hasattr(options, "use_gpu"):
                options.use_gpu = bool(self.use_gpu)
            if hasattr(options, "num_threads"):
                options.num_threads = self._worker_threads()
            return options
        except Exception:  # noqa: BLE001
            return None

    def _pairing_options(self, matcher: str) -> Any:
        """Pairing options; sequential matching needs an explicit overlap window."""
        if matcher != "sequential":
            return None
        factory = getattr(pycolmap, "SequentialPairingOptions", None)
        if factory is None:
            return None
        try:
            options = factory()
            if hasattr(options, "overlap"):
                options.overlap = int(self.settings.get("sequential_overlap", 12))
            # Loop detection recovers links between passes that sequential order misses.
            if hasattr(options, "loop_detection"):
                options.loop_detection = True
            return options
        except Exception:  # noqa: BLE001
            return None

    def _device(self) -> Any:
        device_enum = getattr(pycolmap, "Device", None)
        if device_enum is None:
            return None
        try:
            return device_enum.auto if self.use_gpu else device_enum.cpu
        except Exception:  # noqa: BLE001
            return None

    def _run_matcher(self, matcher: str, database_path: Path, device: Any) -> None:
        """Run the chosen matcher, degrading keyword-by-keyword across API versions."""
        name = "match_exhaustive" if matcher == "exhaustive" else "match_sequential"
        fn = getattr(pycolmap, name, None) or getattr(pycolmap, "match_exhaustive")  # type: ignore[union-attr]

        attempts: list[dict[str, Any]] = []
        full: dict[str, Any] = {}
        matching_options = self._matching_options()
        if matching_options is not None:
            full["matching_options"] = matching_options
        pairing_options = self._pairing_options(matcher)
        if pairing_options is not None:
            full["pairing_options"] = pairing_options
        if full:
            attempts.append(full)
        if device is not None:
            attempts.append({"device": device})
        attempts.append({})

        last_error: TypeError | None = None
        for kwargs in attempts:
            try:
                fn(str(database_path), **kwargs)
                return
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    # -- dense -----------------------------------------------------------

    def _run_dense(
        self, image_dir: Path, work_dir: Path, sparse_model: Any, progress: Callable[[int, str], None]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense MVS via the native COLMAP binary. Returns empty arrays when unavailable."""
        empty = (np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8))
        if not self.settings.get("dense", True):
            return empty

        # The Python bindings can run the whole dense stage when the wheel was built
        # with CUDA, which avoids requiring a separate COLMAP install.
        native = self._run_dense_pycolmap(image_dir, work_dir, sparse_model, progress)
        if native is not None:
            return native

        binary = colmap_executable()
        if not binary:
            self._warn(
                "Dense MVS skipped: pycolmap has no CUDA support in this build and no native "
                "COLMAP binary was found on PATH. Derived products come from the sparse cloud. "
                "Install COLMAP with CUDA for full-resolution dense output."
            )
            return empty

        dense_dir = work_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        sparse_export = work_dir / "sparse_for_dense"
        sparse_export.mkdir(parents=True, exist_ok=True)
        try:
            sparse_model.write(str(sparse_export))
        except Exception as exc:  # noqa: BLE001
            self._warn(f"Dense MVS skipped: could not export sparse model ({exc}).")
            return empty

        progress(62, "Dense stereo (image undistortion)")
        steps = [
            [binary, "image_undistorter", "--image_path", str(image_dir), "--input_path", str(sparse_export),
             "--output_path", str(dense_dir), "--output_type", "COLMAP"],
            [binary, "patch_match_stereo", "--workspace_path", str(dense_dir),
             "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"],
            [binary, "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP",
             "--input_type", "geometric", "--output_path", str(dense_dir / "fused.ply")],
        ]
        for index, command in enumerate(steps):
            progress(62 + index * 6, f"Dense stereo step {index + 1}/3")
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=7200)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._warn(f"Dense MVS failed at step {index + 1}: {exc}. Falling back to the sparse cloud.")
                return empty
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()
                detail = tail[-1] if tail else f"exit code {completed.returncode}"
                self._warn(f"Dense MVS failed at step {index + 1}: {detail}. Falling back to the sparse cloud.")
                return empty

        fused = dense_dir / "fused.ply"
        if not fused.exists():
            self._warn("Dense MVS produced no fused cloud; falling back to the sparse cloud.")
            return empty
        return _read_ply(fused)

    def _run_dense_pycolmap(
        self, image_dir: Path, work_dir: Path, sparse_model: Any, progress: Callable[[int, str], None]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Dense MVS through the pycolmap bindings.

        Returns None (not empty arrays) when the bindings cannot do it, so the caller
        can still try the native binary. Patch-match stereo requires CUDA, so on a
        CPU-only wheel this exits quickly without producing a misleading result.
        """
        required = ("undistort_images", "patch_match_stereo", "stereo_fusion")
        if not all(hasattr(pycolmap, name) for name in required):
            return None
        if not pycolmap_has_cuda():
            # patch_match_stereo is CUDA-only; calling it on a CPU wheel aborts the
            # process rather than raising, so the check must happen before the call.
            self._warn(
                "Dense MVS unavailable: the installed pycolmap wheel was built without CUDA. "
                "Install a native CUDA COLMAP build to enable dense reconstruction."
            )
            return None

        dense_dir = work_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        fused = dense_dir / "fused.ply"

        try:
            progress(62, "Dense stereo (undistorting images)")
            pycolmap.undistort_images(  # type: ignore[union-attr]
                output_path=str(dense_dir),
                input_path=str(work_dir / "sparse" / "0") if (work_dir / "sparse" / "0").exists() else str(work_dir / "sparse"),
                image_path=str(image_dir),
            )
            progress(66, "Dense stereo (patch match)")
            pycolmap.patch_match_stereo(workspace_path=str(dense_dir))  # type: ignore[union-attr]
            progress(70, "Dense stereo (fusion)")
            pycolmap.stereo_fusion(output_path=str(fused), workspace_path=str(dense_dir))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "cuda" in message or "not supported" in message or "without cuda" in message:
                self._warn(
                    "Dense MVS unavailable: this pycolmap build has no CUDA support. "
                    "Products come from the sparse cloud."
                )
                return None
            self._warn(f"Dense MVS via pycolmap failed ({exc}); trying the native COLMAP binary.")
            return None

        if not fused.exists():
            return None
        points, colors = _read_ply(fused)
        if points.shape[0] == 0:
            return None
        progress(71, f"Dense cloud: {points.shape[0]:,} points")
        return points, colors

    # -- rasters ---------------------------------------------------------

    def _raster_pixel_size(self, points_world: np.ndarray, *, georeferenced: bool) -> float:
        """Choose a raster resolution the point cloud can actually fill.

        The profile's requested ground resolution assumes a dense cloud. Applied to a
        sparse cloud it produces a DSM that is almost entirely holes, which then
        propagates into an empty orthophoto. The density floor keeps roughly one point
        per cell, and the requested resolution is used only when the data supports it.
        """
        planar = points_world[:, :2]
        span_x = float(planar[:, 0].max() - planar[:, 0].min())
        span_y = float(planar[:, 1].max() - planar[:, 1].min())
        area = max(span_x * span_y, 1e-9)
        density_floor = math.sqrt(area / max(points_world.shape[0], 1))

        if not georeferenced:
            # Without a CRS the units are arbitrary, so resolution can only be relative.
            return max(density_floor, max(span_x, span_y) / 1024.0, 1e-9)

        requested = 1.0 / max(float(self.settings.get("raster_px_per_m", 16.0)), 1e-6)
        if density_floor > requested:
            self._warn(
                f"Raster resolution relaxed from {requested:.3f} to {density_floor:.3f} m/px: "
                f"{points_world.shape[0]:,} points cannot fill a finer grid. "
                "Enable dense MVS for full-resolution products."
            )
            return density_floor
        return requested

    def _rasterize(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        pixel_size: float,
    ) -> dict[str, Any] | None:
        """Bin a projected point cloud into DSM, DTM, and a colour raster.

        `points_world` is (N,3) in projected CRS metres. DSM takes the maximum Z per
        cell, DTM the minimum-Z of a locally-filtered ground set. Empty cells are
        recorded as nodata rather than silently interpolated over.
        """
        if points_world.shape[0] < 16:
            return None

        east = points_world[:, 0]
        north = points_world[:, 1]
        up = points_world[:, 2]

        # Trim gross outliers before sizing the grid, or a single stray point
        # inflates the raster to gigabytes.
        keep = np.ones(points_world.shape[0], dtype=bool)
        for axis in (east, north, up):
            low, high = np.percentile(axis, [0.5, 99.5])
            span = max(high - low, 1e-6)
            keep &= (axis >= low - 0.25 * span) & (axis <= high + 0.25 * span)
        if int(keep.sum()) < 16:
            keep = np.ones(points_world.shape[0], dtype=bool)
        east, north, up = east[keep], north[keep], up[keep]
        colors = colors[keep] if colors.shape[0] == points_world.shape[0] else colors

        west, east_max = float(east.min()), float(east.max())
        south, north_max = float(north.min()), float(north.max())
        width = max(2, int(math.ceil((east_max - west) / pixel_size)) + 1)
        height = max(2, int(math.ceil((north_max - south) / pixel_size)) + 1)

        # Cap the raster so a huge site or a tiny pixel size cannot exhaust memory.
        max_dimension = 8192
        if max(width, height) > max_dimension:
            scale = max(width, height) / max_dimension
            pixel_size = float(pixel_size * scale)
            width = max(2, int(math.ceil((east_max - west) / pixel_size)) + 1)
            height = max(2, int(math.ceil((north_max - south) / pixel_size)) + 1)
            self._warn(
                f"Raster resolution reduced to {pixel_size:.3f} m/px to keep the output under "
                f"{max_dimension}x{max_dimension}."
            )

        col = np.clip(((east - west) / pixel_size).astype(np.int64), 0, width - 1)
        # Row 0 is the northern edge, matching the GeoTIFF north-up convention.
        row = np.clip(((north_max - north) / pixel_size).astype(np.int64), 0, height - 1)
        flat = row * width + col
        cells = width * height

        dsm = np.full(cells, -np.inf, dtype=np.float64)
        np.maximum.at(dsm, flat, up)
        dtm_min = np.full(cells, np.inf, dtype=np.float64)
        np.minimum.at(dtm_min, flat, up)

        filled = np.isfinite(dsm)
        dsm[~filled] = np.nan
        dtm_min[~np.isfinite(dtm_min)] = np.nan

        color_sum = np.zeros((cells, 3), dtype=np.float64)
        color_count = np.zeros(cells, dtype=np.float64)
        if colors.shape[0] == up.shape[0] and colors.size:
            for channel in range(3):
                np.add.at(color_sum[:, channel], flat, colors[:, channel].astype(np.float64))
            np.add.at(color_count, flat, 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            color_mean = np.where(color_count[:, None] > 0, color_sum / np.maximum(color_count[:, None], 1.0), 0.0)

        dsm_grid = dsm.reshape(height, width)
        dtm_grid = dtm_min.reshape(height, width)
        ortho = color_mean.reshape(height, width, 3)
        valid = filled.reshape(height, width)

        return {
            "dsm": dsm_grid,
            "dtm": _ground_filter(dtm_grid, pixel_size),
            "ortho": ortho,
            "valid": valid,
            "west": west,
            "north": north_max,
            "pixel_size": float(pixel_size),
            "width": width,
            "height": height,
        }

    def _true_orthophoto(
        self,
        raster: dict[str, Any],
        images: Sequence[Any],
        cameras: dict[int, Any],
        image_dir: Path,
        anchor: geo.GeoAnchor | None,
        progress: Callable[[int, str], None],
    ) -> np.ndarray | None:
        """Ortho-rectify source imagery onto the DSM surface.

        Every raster cell with a height is lifted to a 3-D ground point, projected into
        each registered image, and coloured from the view closest to nadir. This is a
        real true-orthophoto: geometry comes from the DSM and colour from the source
        pixels, so the result is metrically correct rather than a translation mosaic.
        """
        if anchor is None:
            return None
        dsm = raster["dsm"]
        height, width = dsm.shape
        valid = np.isfinite(dsm)
        if not valid.any():
            return None

        rows, cols = np.nonzero(valid)
        east = raster["west"] + (cols + 0.5) * raster["pixel_size"]
        north = raster["north"] - (rows + 0.5) * raster["pixel_size"]
        up = dsm[rows, cols]
        world = np.stack([east, north, up], axis=1)

        # Ground points live in the projected CRS; the cameras live in SfM space, so
        # invert the anchor to bring the surface back into the reconstruction frame.
        rotation = anchor.rotation_matrix
        local = world - anchor.origin_vector - anchor.translation_vector
        sfm_points = (local @ rotation) / max(anchor.scale, 1e-12)

        canvas = np.zeros((height, width, 3), dtype=np.float32)
        score_map = np.full((height, width), -np.inf, dtype=np.float32)

        total = max(1, len(images))
        for index, image in enumerate(images):
            if index % 5 == 0:
                progress(78 + int(8.0 * index / total), f"Ortho-rectifying {index + 1}/{total}")
            name = _image_name(image)
            source = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
            if source is None:
                continue
            extrinsic = _cam_from_world_matrix(image)
            camera = cameras.get(getattr(image, "camera_id", -1))
            if extrinsic is None or camera is None:
                continue
            k_matrix = _camera_matrix(camera)
            cam_width = int(getattr(camera, "width", source.shape[1]) or source.shape[1])
            cam_height = int(getattr(camera, "height", source.shape[0]) or source.shape[0])
            if cam_width != source.shape[1] or cam_height != source.shape[0]:
                source = cv2.resize(source, (cam_width, cam_height), interpolation=cv2.INTER_AREA)

            rot = extrinsic[:, :3]
            trans = extrinsic[:, 3]
            camera_points = sfm_points @ rot.T + trans
            in_front = camera_points[:, 2] > 1e-6
            if not in_front.any():
                continue

            projected = camera_points[in_front] @ k_matrix.T
            pixels_x = projected[:, 0] / projected[:, 2]
            pixels_y = projected[:, 1] / projected[:, 2]
            inside = (
                (pixels_x >= 0) & (pixels_x < cam_width - 1) & (pixels_y >= 0) & (pixels_y < cam_height - 1)
            )
            if not inside.any():
                continue

            selector = np.nonzero(in_front)[0][inside]
            sample_x = pixels_x[inside].astype(np.int64)
            sample_y = pixels_y[inside].astype(np.int64)
            bgr = source[sample_y, sample_x]

            # Nadir-most view wins: the optical axis in world space versus straight down.
            optical_axis_world = rot.T @ np.array([0.0, 0.0, 1.0])
            axis_projected = anchor.scale * (rotation @ optical_axis_world)
            norm = float(np.linalg.norm(axis_projected)) or 1.0
            nadir_score = float(-axis_projected[2] / norm)
            # Slight preference for shorter ranges, which resolve more detail.
            depth = camera_points[in_front][inside][:, 2]
            scores = np.full(depth.shape[0], nadir_score, dtype=np.float32) - 0.001 * depth.astype(np.float32)

            target_rows = rows[selector]
            target_cols = cols[selector]
            better = scores > score_map[target_rows, target_cols]
            if not better.any():
                continue
            canvas[target_rows[better], target_cols[better]] = bgr[better][:, ::-1]
            score_map[target_rows[better], target_cols[better]] = scores[better]

        if not np.isfinite(score_map).any():
            return None
        return canvas

    # -- main ------------------------------------------------------------

    def reconstruct(
        self,
        image_dir: str | Path,
        crack_mask_dir: str | Path | None = None,
        output_dir: str | Path = "reconstruction",
        profile: str | None = None,
        execution_mode: str = "local",
        use_cache: bool = True,
        cloud_endpoint: str = "",
        progress_callback: Callable[[int, str], None] | None = None,
        **_ignored: Any,
    ) -> ReconstructionResult:
        """Run the full COLMAP pipeline and write georeferenced artifacts."""
        if not _HAS_PYCOLMAP:
            raise RuntimeError(
                "pycolmap is not installed. Install it with `pip install pycolmap`, "
                "or run the pipeline with --engine custom to use the dependency-free reconstructor."
            )
        if profile:
            self.profile = _normalize_profile(profile)
            self.settings = dict(COLMAP_PROFILES.get(self.profile, COLMAP_PROFILES["standard"]))
        self.warnings = []

        def progress(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(int(max(0, min(100, percent))), message)

        images_root = Path(image_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = out_dir / "colmap"
        work_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self._image_paths(images_root)
        if len(image_paths) < 3:
            raise ValueError(f"Need at least 3 images for reconstruction; found {len(image_paths)}")

        progress(2, f"Reading EXIF for {len(image_paths)} images")
        fixes = geo.collect_gps_fixes(image_paths)
        camera_exif = geo.read_exif_camera(image_paths[0])
        try:
            _k, intrinsics_method = geo.intrinsics_from_exif(camera_exif)
        except ValueError:
            intrinsics_method = "unavailable"

        if execution_mode and execution_mode.lower() == "cloud":
            self._warn(
                "Cloud reconstruction is not implemented in this build; the run executed locally. "
                "No data was sent to a remote endpoint."
            )

        sparse = self._run_sfm(images_root, work_dir, len(image_paths), progress)
        registered = _registered_images(sparse)
        reprojection_error = _mean_reprojection_error(sparse)
        progress(56, f"Registered {len(registered)}/{len(image_paths)} images")

        if len(registered) < len(image_paths):
            self._warn(
                f"{len(image_paths) - len(registered)} of {len(image_paths)} images failed to register. "
                "They are excluded from all downstream products."
            )

        centers: dict[str, list[float]] = {}
        for image in registered:
            center = _camera_center(image)
            if center is not None:
                centers[_image_name(image)] = [float(v) for v in center]

        anchor: geo.GeoAnchor | None = None
        if len(fixes) < 3:
            self._warn(
                f"Only {len(fixes)} of {len(image_paths)} images carry GPS EXIF. "
                "Outputs are in arbitrary SfM units and have no CRS."
            )
        elif not geo.geo_capabilities()["pyproj"]:
            self._warn("pyproj is not installed, so the reconstruction could not be georeferenced.")
        else:
            anchor = geo.anchor_from_cameras(centers, fixes, epsg=self.target_epsg)
            if anchor is None:
                self._warn("Fewer than 3 registered images had geotags; georeferencing skipped.")
            elif not math.isfinite(anchor.rmse_m) or anchor.rmse_m > 25.0:
                self._warn(
                    f"Georeferencing fit is poor (RMSE {anchor.rmse_m:.1f} m across "
                    f"{anchor.inlier_count}/{anchor.sample_count} cameras). Treat absolute positions with caution."
                )

        sparse_xyz, sparse_rgb = _points_and_colors(sparse)
        dense_xyz, dense_rgb = self._run_dense(images_root, work_dir, sparse, progress)
        if dense_xyz.shape[0] > sparse_xyz.shape[0]:
            cloud_xyz, cloud_rgb = dense_xyz, dense_rgb
        else:
            cloud_xyz, cloud_rgb = sparse_xyz, sparse_rgb

        progress(72, f"Writing point cloud ({cloud_xyz.shape[0]:,} points)")
        cloud_path = out_dir / "reconstruction.ply"
        _write_ascii_ply(cloud_path, cloud_xyz, cloud_rgb if cloud_rgb.shape[0] == cloud_xyz.shape[0] else None)

        artifacts = self._write_products(
            out_dir=out_dir,
            image_dir=images_root,
            sparse=sparse,
            registered=registered,
            cloud_xyz=cloud_xyz,
            cloud_rgb=cloud_rgb,
            anchor=anchor,
            fixes=fixes,
            progress=progress,
        )

        pose_path = out_dir / "camera_poses.json"
        camera_map_for_export = _map_items(getattr(sparse, "cameras", None))
        pose_payload = {
            "engine": "colmap",
            "registered": len(registered),
            "total_images": len(image_paths),
            "mean_reprojection_error_px": reprojection_error,
            "intrinsics_method": intrinsics_method,
            # Full extrinsics and intrinsics are exported so downstream stages (defect
            # back-projection, measurement) can reproject without re-running COLMAP.
            "cameras": [
                _camera_record(image, camera_map_for_export.get(int(getattr(image, "camera_id", -1))))
                for image in registered
            ],
        }
        pose_path.write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

        anchor_path = ""
        if anchor is not None:
            anchor_path = str(out_dir / "geo_anchor.json")
            Path(anchor_path).write_text(json.dumps(anchor.to_dict(), indent=2), encoding="utf-8")

        twin_path = out_dir / "digital_twin.json"
        twin_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "engine": "colmap",
            "profile": self.profile,
            "crs_epsg": anchor.epsg if anchor else None,
            "georeferenced": anchor is not None,
            "frame_count": len(image_paths),
            "registered_images": len(registered),
            "point_count": int(cloud_xyz.shape[0]),
            "dense": bool(dense_xyz.shape[0] > 0),
            "mean_reprojection_error_px": reprojection_error,
            "geo_rmse_m": anchor.rmse_m if anchor else None,
            "ground_sample_distance_m": artifacts.get("gsd_m"),
            "artifacts": {key: value for key, value in artifacts.items() if isinstance(value, str)},
            "warnings": list(self.warnings),
        }
        twin_path.write_text(json.dumps(twin_payload, indent=2), encoding="utf-8")

        progress(100, "Reconstruction complete")
        return ReconstructionResult(
            frame_count=len(image_paths),
            processed_pairs=len(registered),
            failed_pairs=max(0, len(image_paths) - len(registered)),
            total_points=int(cloud_xyz.shape[0]),
            crack_points=0,
            point_cloud_path=str(cloud_path),
            crack_cloud_path=None,
            camera_pose_path=str(pose_path),
            orthomosaic_path=artifacts.get("orthomosaic_png", ""),
            dsm_path=artifacts.get("dsm_png", ""),
            dtm_path=artifacts.get("dtm_png", ""),
            mesh_path=artifacts.get("mesh", ""),
            textured_mesh_obj_path=artifacts.get("mesh_obj", ""),
            textured_mesh_mtl_path="",
            texture_image_path=artifacts.get("orthomosaic_png", ""),
            digital_twin_path=str(twin_path),
            processing_profile=self.profile,
            execution_mode_requested=str(execution_mode),
            execution_mode_used="local",
            cache_hits=0,
            cache_misses=0,
            feature_cache_dir=str(work_dir),
            match_cache_dir=str(work_dir),
            warnings=list(self.warnings),
            engine="colmap",
            crs_epsg=anchor.epsg if anchor else None,
            geo_anchor_path=anchor_path,
            geo_rmse_m=anchor.rmse_m if anchor else None,
            geo_inlier_cameras=anchor.inlier_count if anchor else 0,
            geotagged_images=len(fixes),
            intrinsics_method=intrinsics_method,
            reprojection_error_px=reprojection_error,
            registered_images=len(registered),
            dense_point_count=int(dense_xyz.shape[0]),
            orthomosaic_cog_path=artifacts.get("orthomosaic_cog", ""),
            dsm_cog_path=artifacts.get("dsm_cog", ""),
            dtm_cog_path=artifacts.get("dtm_cog", ""),
            hillshade_path=artifacts.get("hillshade_cog", ""),
            camera_track_geojson_path=artifacts.get("camera_track", ""),
            ground_sample_distance_m=artifacts.get("gsd_m"),
        )

    def _write_products(
        self,
        *,
        out_dir: Path,
        image_dir: Path,
        sparse: Any,
        registered: Sequence[Any],
        cloud_xyz: np.ndarray,
        cloud_rgb: np.ndarray,
        anchor: geo.GeoAnchor | None,
        fixes: dict[str, geo.GpsFix],
        progress: Callable[[int, str], None],
    ) -> dict[str, Any]:
        """Write mesh, rasters, and vector products. Returns a path map for the result."""
        artifacts: dict[str, Any] = {}

        progress(74, "Building mesh")
        mesh_paths = self._build_mesh(out_dir, cloud_xyz, cloud_rgb)
        artifacts.update(mesh_paths)

        if cloud_xyz.shape[0] < 16:
            self._warn(
                f"Only {cloud_xyz.shape[0]} 3-D points were triangulated, which is too few for "
                "raster or orthophoto products. Check image overlap and texture."
            )
            return artifacts

        if anchor is not None:
            world_points = anchor.apply(cloud_xyz, absolute=True)
        else:
            world_points = cloud_xyz
            self._warn("Rasters are written without a CRS because the run is not georeferenced.")

        pixel_size = self._raster_pixel_size(world_points, georeferenced=anchor is not None)

        progress(76, "Rasterizing DSM/DTM")
        raster = self._rasterize(world_points, cloud_rgb, pixel_size)
        if raster is None:
            self._warn("Too few points to build raster products.")
            return artifacts

        artifacts["gsd_m"] = raster["pixel_size"] if anchor is not None else None

        ortho = None
        camera_map = _map_items(getattr(sparse, "cameras", None))
        if anchor is not None and registered:
            ortho = self._true_orthophoto(raster, registered, camera_map, image_dir, anchor, progress)
        if ortho is None:
            ortho = raster["ortho"]
            if anchor is not None:
                self._warn("True-orthophoto generation failed; the point-colour raster was used instead.")

        ortho_uint8 = np.clip(np.nan_to_num(ortho), 0, 255).astype(np.uint8)
        ortho_png = out_dir / "orthomosaic.png"
        cv2.imwrite(str(ortho_png), ortho_uint8[:, :, ::-1])
        artifacts["orthomosaic_png"] = str(ortho_png)

        dsm = raster["dsm"]
        dtm = raster["dtm"]
        artifacts["dsm_png"] = str(_write_elevation_png(out_dir / "dsm.png", dsm))
        artifacts["dtm_png"] = str(_write_elevation_png(out_dir / "dtm.png", dtm))

        if anchor is not None and geo.geo_capabilities()["rasterio"]:
            progress(88, "Writing Cloud-Optimized GeoTIFFs")
            common = {
                "epsg": anchor.epsg,
                "west": raster["west"],
                "north": raster["north"],
                "pixel_size": raster["pixel_size"],
            }
            try:
                artifacts["dsm_cog"] = geo.write_geotiff(
                    out_dir / "dsm.tif", np.nan_to_num(dsm, nan=-9999.0).astype(np.float32), nodata=-9999.0, **common
                )
                artifacts["dtm_cog"] = geo.write_geotiff(
                    out_dir / "dtm.tif", np.nan_to_num(dtm, nan=-9999.0).astype(np.float32), nodata=-9999.0, **common
                )
                artifacts["orthomosaic_cog"] = geo.write_geotiff(
                    out_dir / "orthomosaic.tif", np.moveaxis(ortho_uint8, 2, 0), nodata=0, **common
                )
                artifacts["hillshade_cog"] = geo.write_geotiff(
                    out_dir / "dsm_hillshade.tif",
                    geo.hillshade(np.nan_to_num(dsm, nan=float(np.nanmin(dsm)) if np.isfinite(dsm).any() else 0.0),
                                  raster["pixel_size"]),
                    nodata=0,
                    **common,
                )
            except Exception as exc:  # noqa: BLE001
                self._warn(f"GeoTIFF export failed: {exc}. PNG products were still written.")
        elif anchor is not None:
            self._warn("rasterio is not installed, so no GeoTIFF was written. Only PNG previews exist.")

        if anchor is not None:
            track = self._camera_track_geojson(registered, fixes, anchor)
            if track is not None:
                artifacts["camera_track"] = geo.write_geojson(out_dir / "camera_track.geojson", track)

        return artifacts

    def _build_mesh(self, out_dir: Path, points: np.ndarray, colors: np.ndarray) -> dict[str, str]:
        """Poisson surface reconstruction. Skipped with a warning when Open3D is absent."""
        if points.shape[0] < 100:
            self._warn("Too few points to build a mesh.")
            return {}
        if not _HAS_OPEN3D:
            self._warn("Open3D is not installed, so no surface mesh was produced. Install with `pip install open3d`.")
            return {}
        try:
            cloud = o3d.geometry.PointCloud()  # type: ignore[union-attr]
            cloud.points = o3d.utility.Vector3dVector(points)  # type: ignore[union-attr]
            if colors.shape[0] == points.shape[0]:
                cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)  # type: ignore[union-attr]
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(  # type: ignore[union-attr]
                    radius=_auto_normal_radius(np.asarray(cloud.points)), max_nn=30
                )
            )
            cloud.orient_normals_consistent_tangent_plane(30)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(  # type: ignore[union-attr]
                cloud, depth=int(self.settings.get("mesh_depth", 9))
            )
            # Poisson extrapolates a closed surface well past the data; trimming the
            # lowest-density vertices removes those invented regions.
            density = np.asarray(densities)
            if density.size:
                mesh.remove_vertices_by_mask(density < np.quantile(density, 0.05))
            mesh.compute_vertex_normals()

            ply_path = out_dir / "mesh.ply"
            obj_path = out_dir / "mesh.obj"
            o3d.io.write_triangle_mesh(str(ply_path), mesh)  # type: ignore[union-attr]
            _write_obj_mesh(obj_path, np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            return {"mesh": str(ply_path), "mesh_obj": str(obj_path)}
        except Exception as exc:  # noqa: BLE001
            self._warn(f"Mesh generation failed: {exc}.")
            return {}

    def _camera_track_geojson(
        self, registered: Sequence[Any], fixes: dict[str, geo.GpsFix], anchor: geo.GeoAnchor
    ) -> list[dict] | None:
        """Camera positions as GeoJSON points, carrying the per-image GPS residual."""
        features: list[dict] = []
        for image in registered:
            name = _image_name(image)
            center = _camera_center(image)
            if center is None:
                continue
            lonlat = anchor.to_wgs84(center.reshape(1, 3))[0]
            properties: dict[str, Any] = {"image": name, "altitude_m": float(lonlat[2])}
            fix = fixes.get(name)
            if fix is not None:
                properties["exif_latitude"] = fix.latitude
                properties["exif_longitude"] = fix.longitude
                properties["gps_residual_m"] = round(
                    geo.haversine_m(float(lonlat[1]), float(lonlat[0]), fix.latitude, fix.longitude), 3
                )
            features.append(geo.point_feature(float(lonlat[0]), float(lonlat[1]), properties, alt=float(lonlat[2])))
        return features or None


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------


def _camera_record(image: Any, camera: Any) -> dict[str, Any]:
    """Serialize one registered view: centre, world->camera extrinsics, and K."""
    center = _camera_center(image)
    extrinsic = _cam_from_world_matrix(image)
    record: dict[str, Any] = {
        "image": _image_name(image),
        "camera_id": int(getattr(image, "camera_id", -1)),
        "center": None if center is None else [float(v) for v in center],
    }
    if extrinsic is not None:
        record["rotation_cam_from_world"] = [[float(v) for v in row] for row in extrinsic[:, :3]]
        record["translation_cam_from_world"] = [float(v) for v in extrinsic[:, 3]]
    if camera is not None:
        k_matrix = _camera_matrix(camera)
        record["intrinsics"] = [[float(v) for v in row] for row in k_matrix]
        record["width"] = int(getattr(camera, "width", 0) or 0)
        record["height"] = int(getattr(camera, "height", 0) or 0)
        record["model"] = str(getattr(camera, "model_name", getattr(camera, "model", "")) or "")
    return record


def _auto_normal_radius(points: np.ndarray) -> float:
    """Pick a normal-estimation radius from the cloud extent, not a magic constant."""
    if points.shape[0] < 2:
        return 1.0
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(extent / 200.0, 1e-4)


def _ground_filter(min_grid: np.ndarray, pixel_size: float) -> np.ndarray:
    """Approximate a bare-earth DTM by morphological opening of the minimum surface.

    A full cloth-simulation filter needs a dense cloud; this opening removes vegetation
    and structures at the scale of a few metres, which is the useful case for inspection
    sites, then restores the values that were already ground.
    """
    grid = np.array(min_grid, dtype=np.float32)
    mask = ~np.isfinite(grid)
    if mask.all():
        return grid
    fill = float(np.nanmax(grid[np.isfinite(grid)]))
    grid[mask] = fill

    kernel_size = max(3, int(round(5.0 / max(pixel_size, 1e-6))) | 1)
    kernel_size = min(kernel_size, 51)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel)
    opened[mask] = np.nan
    return opened


def _write_elevation_png(path: Path, elevation: np.ndarray) -> Path:
    """Write a colour-mapped elevation preview alongside the authoritative GeoTIFF."""
    finite = np.isfinite(elevation)
    image = np.zeros(elevation.shape, dtype=np.uint8)
    if finite.any():
        values = elevation[finite]
        low, high = float(values.min()), float(values.max())
        span = max(high - low, 1e-9)
        image[finite] = np.clip(((values - low) / span) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~finite] = (0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), colored)
    return path


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal PLY reader for COLMAP's fused output (binary little-endian or ASCII)."""
    with open(path, "rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        vertex_count = 0
        properties: list[tuple[str, str]] = []
        fmt = "ascii"
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) >= 3:
                properties.append((parts[1], parts[2]))

        if vertex_count <= 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)

        numpy_types = {
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        }

        if fmt.startswith("binary"):
            endian = "<" if "little" in fmt else ">"
            dtype = np.dtype([(name, endian + numpy_types.get(kind, "f4")) for kind, name in properties])
            data = np.frombuffer(handle.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count)
        else:
            rows = []
            for _ in range(vertex_count):
                line = handle.readline().decode("ascii", errors="ignore").split()
                if not line:
                    break
                rows.append([float(v) for v in line])
            if not rows:
                return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
            names = [name for _kind, name in properties]
            array = np.asarray(rows, dtype=float)
            data = {name: array[:, index] for index, name in enumerate(names) if index < array.shape[1]}

    def column(name: str) -> np.ndarray | None:
        try:
            return np.asarray(data[name], dtype=float)
        except Exception:  # noqa: BLE001
            return None

    xs, ys, zs = column("x"), column("y"), column("z")
    if xs is None or ys is None or zs is None:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    xyz = np.stack([xs, ys, zs], axis=1)

    reds, greens, blues = column("red"), column("green"), column("blue")
    if reds is None or greens is None or blues is None:
        rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
    else:
        rgb = np.clip(np.stack([reds, greens, blues], axis=1), 0, 255).astype(np.uint8)
    return xyz, rgb


def build_reconstructor(engine: str, **kwargs: Any):
    """Factory used by the pipeline and CLI to pick an engine by name.

    `auto` prefers COLMAP when it is importable and falls back to the custom engine
    otherwise, so a machine without the optional dependencies still produces output.
    """
    from .reconstruction import CustomDroneReconstructor

    name = str(engine or "auto").strip().lower()
    if name == "auto":
        name = "colmap" if _HAS_PYCOLMAP else "custom"
    if name == "colmap":
        if not _HAS_PYCOLMAP:
            raise RuntimeError("pycolmap is not installed; use --engine custom or install pycolmap.")
        return ColmapReconstructor(
            profile=kwargs.get("profile", "standard"),
            use_gpu=kwargs.get("use_gpu", True),
            dense=kwargs.get("dense"),
            max_image_size=kwargs.get("max_image_size"),
            raster_px_per_m=kwargs.get("raster_px_per_m"),
            target_epsg=kwargs.get("target_epsg"),
        )
    if name == "custom":
        return CustomDroneReconstructor(
            profile=kwargs.get("profile", "standard"),
            execution_mode=kwargs.get("execution_mode", "local"),
            use_cache=kwargs.get("use_cache", True),
            cloud_endpoint=kwargs.get("cloud_endpoint", ""),
        )
    raise ValueError(f"Unknown reconstruction engine {engine!r}; expected one of: auto, colmap, custom")
