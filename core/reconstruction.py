"""Custom photogrammetry pipeline (SfM -> lightweight MVS) for drone imagery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .detection import detect_cracks, detect_metal_defects, detect_solar_defects, detect_structural_defects
from .propagation import CrackPropagationForecaster


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


RECONSTRUCTION_PROFILES: dict[str, dict[str, float | int]] = {
    "fast_preview": {
        "max_features": 2400,
        "ratio_test": 0.80,
        "min_inliers": 30,
        "max_pose_matches": 1200,
        "max_match_descriptors": 1800,
        "flann_trigger_product": 4_000_000,
        "crosscheck_trigger_product": 9_000_000,
        "max_points_default": 80_000,
        "mosaic_scale": 0.45,
        "dsm_grid_size": 0.16,
    },
    "standard": {
        "max_features": 4500,
        "ratio_test": 0.75,
        "min_inliers": 50,
        "max_pose_matches": 1800,
        "max_match_descriptors": 2600,
        "flann_trigger_product": 6_500_000,
        "crosscheck_trigger_product": 14_000_000,
        "max_points_default": 150_000,
        "mosaic_scale": 0.65,
        "dsm_grid_size": 0.09,
    },
    "inspection_high_accuracy": {
        "max_features": 9000,
        "ratio_test": 0.72,
        "min_inliers": 65,
        "max_pose_matches": 2600,
        "max_match_descriptors": 3400,
        "flann_trigger_product": 9_500_000,
        "crosscheck_trigger_product": 20_000_000,
        "max_points_default": 400_000,
        "mosaic_scale": 1.0,
        "dsm_grid_size": 0.05,
    },
}


def available_reconstruction_profiles() -> list[str]:
    return sorted(RECONSTRUCTION_PROFILES.keys())


def _normalize_profile(name: str) -> str:
    value = str(name or "standard").strip().lower()
    aliases = {
        "fast": "fast_preview",
        "preview": "fast_preview",
        "standard": "standard",
        "default": "standard",
        "high": "inspection_high_accuracy",
        "high_accuracy": "inspection_high_accuracy",
        "inspection": "inspection_high_accuracy",
        "inspection_high_accuracy": "inspection_high_accuracy",
    }
    normalized = aliases.get(value, value)
    return normalized if normalized in RECONSTRUCTION_PROFILES else "standard"


def _normalize_execution_mode(mode: str) -> str:
    value = str(mode or "local").strip().lower()
    if value in {"cloud", "local"}:
        return value
    return "local"


def _hash_text(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


def _path_fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except Exception:
        return f"{path.resolve()}|missing"


@dataclass
class ReconstructionResult:
    frame_count: int
    processed_pairs: int
    failed_pairs: int
    total_points: int
    crack_points: int
    point_cloud_path: str
    crack_cloud_path: str | None
    camera_pose_path: str
    orthomosaic_path: str
    dsm_path: str
    dtm_path: str
    mesh_path: str
    textured_mesh_obj_path: str
    textured_mesh_mtl_path: str
    texture_image_path: str
    digital_twin_path: str
    processing_profile: str
    execution_mode_requested: str
    execution_mode_used: str
    cache_hits: int
    cache_misses: int
    feature_cache_dir: str
    match_cache_dir: str
    cloud_request_path: str = ""
    cloud_response_path: str = ""
    risk_point_cloud_path: str = ""
    point_link_path: str = ""
    critical_points_path: str = ""
    warnings: list[str] = field(default_factory=list)

    # Engine identity and georeferencing. Populated by the COLMAP engine; the custom
    # engine leaves them empty, which is how callers tell an unreferenced run apart
    # from a real-world-anchored one.
    engine: str = "custom"
    crs_epsg: int | None = None
    geo_anchor_path: str = ""
    geo_rmse_m: float | None = None
    geo_inlier_cameras: int = 0
    geotagged_images: int = 0
    intrinsics_method: str = "heuristic_0.9_max_dim"
    reprojection_error_px: float | None = None
    registered_images: int = 0
    dense_point_count: int = 0
    orthomosaic_cog_path: str = ""
    dsm_cog_path: str = ""
    dtm_cog_path: str = ""
    hillshade_path: str = ""
    defects_geojson_path: str = ""
    camera_track_geojson_path: str = ""
    ground_sample_distance_m: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _write_ascii_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if colors_rgb is not None and colors_rgb.size != 0:
        colors = np.clip(colors_rgb, 0, 255).astype(np.uint8)
    else:
        colors = None

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if colors is not None:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")

        if colors is not None:
            for p, c in zip(points, colors):
                handle.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                handle.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _write_obj_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by CustomDroneReconstructor\n")
        for v in vertices:
            handle.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for f in faces:
            handle.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")


def _write_textured_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_file_name: str,
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.parent.mkdir(parents=True, exist_ok=True)

    with mtl_path.open("w", encoding="utf-8") as handle:
        handle.write("newmtl ortho_texture\n")
        handle.write("Ka 1.000 1.000 1.000\n")
        handle.write("Kd 1.000 1.000 1.000\n")
        handle.write("Ks 0.000 0.000 0.000\n")
        handle.write(f"map_Kd {texture_file_name}\n")

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by CustomDroneReconstructor\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        for v in vertices:
            handle.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for uv in uvs:
            handle.write(f"vt {float(uv[0]):.6f} {float(uv[1]):.6f}\n")
        handle.write("usemtl ortho_texture\n")
        for f in faces:
            i1, i2, i3 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
            handle.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")


def _save_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


def _normalize_to_colormap(image: np.ndarray, colormap: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, colormap)


def _risk_level_from_score(score: float) -> str:
    s = float(np.clip(score, 0.0, 1.0))
    if s < 0.25:
        return "green"
    if s < 0.55:
        return "yellow"
    return "red"


def _risk_color_rgb(level: str) -> tuple[int, int, int]:
    if level == "red":
        return (255, 36, 36)
    if level == "yellow":
        return (245, 196, 33)
    return (52, 190, 84)


class CustomDroneReconstructor:
    """Incremental SfM-based reconstructor with profile modes, cache, and derivative products."""

    def __init__(
        self,
        max_features: int | None = None,
        ratio_test: float | None = None,
        min_inliers: int | None = None,
        max_points: int | None = None,
        profile: str = "standard",
        execution_mode: str = "local",
        use_cache: bool = True,
        cloud_endpoint: str = "",
        cloud_timeout_s: float = 8.0,
    ):
        self.profile = _normalize_profile(profile)
        p = RECONSTRUCTION_PROFILES[self.profile]

        self.max_features = int(max_features if max_features is not None else p["max_features"])
        self.ratio_test = float(ratio_test if ratio_test is not None else p["ratio_test"])
        self.min_inliers = int(min_inliers if min_inliers is not None else p["min_inliers"])
        self.max_points = int(max_points if max_points is not None else p["max_points_default"])
        self.max_pose_matches = int(max(self.min_inliers * 2, p.get("max_pose_matches", 1800)))
        self.max_match_descriptors = int(
            max(
                self.max_pose_matches,
                min(self.max_features, int(p.get("max_match_descriptors", 3000))),
            )
        )
        self.flann_trigger_product = int(max(1_000_000, int(p.get("flann_trigger_product", 7_000_000))))
        self.crosscheck_trigger_product = int(
            max(self.flann_trigger_product + 1, int(p.get("crosscheck_trigger_product", 15_000_000)))
        )

        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.use_cache = bool(use_cache)
        self.cloud_endpoint = str(cloud_endpoint or "").strip()
        self.cloud_timeout_s = float(max(1.0, cloud_timeout_s))

        self.mosaic_scale = float(np.clip(float(p["mosaic_scale"]), 0.2, 1.0))
        self.dsm_grid_size = float(max(1e-4, p["dsm_grid_size"]))

        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.matcher_crosscheck = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # FLANN-LSH is much faster than pure BF on large ORB descriptor sets.
        self.matcher_flann = cv2.FlannBasedMatcher(
            {
                "algorithm": 6,  # FLANN_INDEX_LSH
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            },
            {"checks": 32},
        )

        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _image_paths(image_dir: str | Path) -> list[Path]:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        images = [p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        images.sort()
        return images

    @staticmethod
    def _intrinsics(width: int, height: int) -> np.ndarray:
        focal = 0.9 * float(max(width, height))
        cx, cy = width / 2.0, height / 2.0
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _mask_for_image(mask_dir: Path | None, image_path: Path) -> np.ndarray | None:
        if mask_dir is None:
            return None
        candidates = [
            mask_dir / f"{image_path.stem}_crack_mask.png",
            mask_dir / f"{image_path.stem}.png",
            mask_dir / image_path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        return None

    def _register_cache(self, hit: bool) -> None:
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1


    def _feature_cache_file(self, image_path: Path, cache_dir: Path) -> Path:
        fingerprint = _path_fingerprint(image_path)
        key = _hash_text(
            fingerprint,
            f"profile={self.profile}",
            f"max_features={self.max_features}",
        )[:16]
        return cache_dir / f"{image_path.stem}_{key}.npz"

    def _pair_cache_file(self, p1: Path, p2: Path, cache_dir: Path) -> Path:
        fingerprint = _hash_text(
            _path_fingerprint(p1),
            _path_fingerprint(p2),
            f"ratio={self.ratio_test:.6f}",
            f"min_inliers={self.min_inliers}",
            f"max_pose_matches={self.max_pose_matches}",
            f"max_match_descriptors={self.max_match_descriptors}",
            f"flann_trigger_product={self.flann_trigger_product}",
            f"crosscheck_trigger_product={self.crosscheck_trigger_product}",
            f"profile={self.profile}",
        )[:16]
        return cache_dir / f"{p1.stem}_{p2.stem}_{fingerprint}.npz"

    @staticmethod
    def _downsample_descriptors(
        pts: np.ndarray,
        des: np.ndarray,
        cap: int,
        seed_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(des) <= cap:
            return pts, des
        seed = int(_hash_text(seed_key)[:8], 16)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(des), size=int(cap), replace=False)
        idx.sort()
        return np.asarray(pts[idx], dtype=np.float32), np.asarray(des[idx], dtype=np.uint8)

    def _detect_or_load_features(
        self,
        gray: np.ndarray,
        image_path: Path,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_file = self._feature_cache_file(image_path, cache_dir)
        if self.use_cache and cache_file.exists():
            try:
                blob = np.load(str(cache_file), allow_pickle=False)
                pts = np.asarray(blob["pts"], dtype=np.float32)
                des = np.asarray(blob["des"], dtype=np.uint8)
                self._register_cache(True)
                return pts, des
            except Exception:
                pass

        kps, des = self.orb.detectAndCompute(gray, None)
        if des is None or kps is None or len(kps) == 0:
            pts = np.empty((0, 2), dtype=np.float32)
            des = np.empty((0, 32), dtype=np.uint8)
        else:
            pts = np.asarray([kp.pt for kp in kps], dtype=np.float32)
            des = np.asarray(des, dtype=np.uint8)

        if self.use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.savez_compressed(str(cache_file), pts=pts, des=des)
            except Exception:
                pass

        self._register_cache(False)
        return pts, des

    def _match_or_load_points(
        self,
        pts1: np.ndarray,
        des1: np.ndarray,
        path1: Path,
        pts2: np.ndarray,
        des2: np.ndarray,
        path2: Path,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_file = self._pair_cache_file(path1, path2, cache_dir)
        if self.use_cache and cache_file.exists():
            try:
                blob = np.load(str(cache_file), allow_pickle=False)
                m1 = np.asarray(blob["pts1"], dtype=np.float64)
                m2 = np.asarray(blob["pts2"], dtype=np.float64)
                self._register_cache(True)
                return m1, m2
            except Exception:
                pass

        if des1.size == 0 or des2.size == 0:
            self._register_cache(False)
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        # Hard cap descriptor count per pair to prevent pathological runtime stalls.
        pts1_use, des1_use = self._downsample_descriptors(
            pts1,
            des1,
            cap=self.max_match_descriptors,
            seed_key=f"{path1.name}|{path2.name}|a",
        )
        pts2_use, des2_use = self._downsample_descriptors(
            pts2,
            des2,
            cap=self.max_match_descriptors,
            seed_key=f"{path1.name}|{path2.name}|b",
        )
        match_product = int(len(des1_use) * len(des2_use))

        raw: list[Any]
        if match_product >= self.crosscheck_trigger_product:
            try:
                raw_cc = self.matcher_crosscheck.match(des1_use, des2_use)
            except Exception:
                raw_cc = []
            # Normalize to the same structure used by the ratio loop.
            raw = [[m, m] for m in raw_cc]
        elif match_product >= self.flann_trigger_product:
            try:
                raw = self.matcher_flann.knnMatch(des1_use, des2_use, k=2)
            except Exception:
                raw = self.matcher_bf.knnMatch(des1_use, des2_use, k=2)
        else:
            raw = self.matcher_bf.knnMatch(des1_use, des2_use, k=2)

        q_idx: list[int] = []
        t_idx: list[int] = []
        match_dist: list[float] = []
        for pair in raw:
            if len(pair) != 2:
                continue
            m, n = pair
            if match_product >= self.crosscheck_trigger_product:
                # Cross-check fallback: no ratio test available.
                q_idx.append(int(m.queryIdx))
                t_idx.append(int(m.trainIdx))
                match_dist.append(float(m.distance))
                continue
            if m.distance < self.ratio_test * n.distance:
                q_idx.append(int(m.queryIdx))
                t_idx.append(int(m.trainIdx))
                match_dist.append(float(m.distance))

        if not q_idx:
            self._register_cache(False)
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        if len(q_idx) > self.max_pose_matches:
            order = np.argsort(np.asarray(match_dist, dtype=np.float32))
            keep = order[: int(self.max_pose_matches)]
            q_idx = [q_idx[int(i)] for i in keep.tolist()]
            t_idx = [t_idx[int(i)] for i in keep.tolist()]

        matched1 = np.asarray(pts1_use[q_idx], dtype=np.float64)
        matched2 = np.asarray(pts2_use[t_idx], dtype=np.float64)

        if self.use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.savez_compressed(str(cache_file), pts1=matched1, pts2=matched2)
            except Exception:
                pass

        self._register_cache(False)
        return matched1, matched2

    def _request_cloud_run(
        self,
        image_dir: str | Path,
        output_dir: Path,
        image_count: int,
    ) -> tuple[str, str, list[str]]:
        """Record that cloud processing was requested. Nothing is transmitted.

        Cloud reconstruction is not implemented in this build. The previous version
        POSTed the absolute path of the operator's imagery, the image count, and the
        processing profile to a configured endpoint, saved whatever came back, and
        then discarded it and reconstructed locally regardless. That transmitted
        information about the survey off the machine while delivering nothing, and it
        contradicted this toolkit's offline-first guarantee.

        The request is now written locally as a record of intent, and the caller
        reports that the run executed locally.
        """
        request_payload = {
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "image_dir": str(Path(image_dir).resolve()),
            "image_count": int(image_count),
            "processing_profile": self.profile,
            "execution_mode": "cloud_requested",
            "max_points": int(self.max_points),
            "status": "not_submitted",
            "reason": (
                "Cloud reconstruction is not implemented in this build. This file is a "
                "local record only; no data was sent anywhere."
            ),
        }
        request_path = output_dir / "cloud_request.json"
        _save_json(request_path, request_payload)

        warnings = [
            "Cloud reconstruction is not implemented in this build. The run executed "
            "locally and no data left this machine."
        ]
        if self.cloud_endpoint:
            warnings.append(
                f"The configured cloud endpoint ({self.cloud_endpoint}) was not contacted."
            )
        return str(request_path), "", warnings

    def _densify_point_cloud(self, points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the cloud unchanged. Retained so callers and profiles keep working.

        This previously cloned every point two or three times (per profile) with
        Gaussian jitter and presented the result as multi-view-stereo densification.
        It added no information: the copies carry no new observations, and the only
        effect was to multiply the reported point count while blurring the surface by
        the jitter sigma. Real densification needs patch-match stereo, which lives in
        the COLMAP engine and requires a CUDA build.
        """
        return points, colors

    def _downsample_points(self, points: np.ndarray, colors: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        if len(points) <= limit:
            return points, colors
        rng = np.random.default_rng(42)
        sel = rng.choice(len(points), size=int(limit), replace=False)
        return points[sel], colors[sel]

    def _build_orthomosaic(self, image_paths: list[Path], output_path: Path) -> tuple[str, list[dict]]:
        if not image_paths:
            return "", []

        frames: list[np.ndarray] = []
        names: list[str] = []
        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if self.mosaic_scale < 0.999:
                w = max(1, int(round(img.shape[1] * self.mosaic_scale)))
                h = max(1, int(round(img.shape[0] * self.mosaic_scale)))
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(img)
            names.append(p.name)

        if not frames:
            return "", []

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
        positions = np.zeros((len(frames), 2), dtype=np.float64)

        # Pairs whose correlation failed get an assumed offset so the strip still
        # renders, but that offset is invented and the count is reported so the
        # mosaic is never mistaken for a measured one.
        self._mosaic_guessed_offsets = 0

        for i in range(1, len(frames)):
            shift, response = cv2.phaseCorrelate(grays[i - 1], grays[i])
            dx = float(shift[0])
            dy = float(shift[1])
            if response < 0.03:
                # No usable correlation between these frames. 35% of frame width is a
                # stand-in for typical forward overlap, not a measurement.
                dx = float(frames[i - 1].shape[1] * 0.35)
                dy = 0.0
                self._mosaic_guessed_offsets += 1
            max_dx = float(frames[i].shape[1] * 0.85)
            max_dy = float(frames[i].shape[0] * 0.85)
            dx = float(np.clip(dx, -max_dx, max_dx))
            dy = float(np.clip(dy, -max_dy, max_dy))
            positions[i] = positions[i - 1] + np.array([dx, -dy], dtype=np.float64)

        widths = np.asarray([f.shape[1] for f in frames], dtype=np.float64)
        heights = np.asarray([f.shape[0] for f in frames], dtype=np.float64)

        min_x = float(np.min(positions[:, 0]))
        min_y = float(np.min(positions[:, 1]))
        max_x = float(np.max(positions[:, 0] + widths))
        max_y = float(np.max(positions[:, 1] + heights))

        pad = 40
        canvas_w = max(64, int(np.ceil(max_x - min_x + 2 * pad)))
        canvas_h = max(64, int(np.ceil(max_y - min_y + 2 * pad)))

        # Prevent pathological mosaics from producing huge canvases that stall UI workflows.
        max_canvas_dim = 12000
        if canvas_w > max_canvas_dim or canvas_h > max_canvas_dim:
            scale = min(
                float(max_canvas_dim) / float(max(canvas_w, 1)),
                float(max_canvas_dim) / float(max(canvas_h, 1)),
            )
            scale = float(np.clip(scale, 0.15, 1.0))
            if scale < 0.999:
                scaled_frames: list[np.ndarray] = []
                for img in frames:
                    new_w = max(1, int(round(img.shape[1] * scale)))
                    new_h = max(1, int(round(img.shape[0] * scale)))
                    scaled_frames.append(cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA))
                frames = scaled_frames
                positions = positions * scale
                widths = np.asarray([f.shape[1] for f in frames], dtype=np.float64)
                heights = np.asarray([f.shape[0] for f in frames], dtype=np.float64)
                min_x = float(np.min(positions[:, 0]))
                min_y = float(np.min(positions[:, 1]))
                max_x = float(np.max(positions[:, 0] + widths))
                max_y = float(np.max(positions[:, 1] + heights))
                canvas_w = max(64, int(np.ceil(max_x - min_x + 2 * pad)))
                canvas_h = max(64, int(np.ceil(max_y - min_y + 2 * pad)))

        offset = np.array([pad - min_x, pad - min_y], dtype=np.float64)

        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        footprints: list[dict] = []
        for img, name, pos in zip(frames, names, positions):
            x = int(round(pos[0] + offset[0]))
            y = int(round(pos[1] + offset[1]))
            h, w = img.shape[:2]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(canvas_w, x + w)
            y1 = min(canvas_h, y + h)
            if x1 <= x0 or y1 <= y0:
                continue

            ix0 = x0 - x
            iy0 = y0 - y
            ix1 = ix0 + (x1 - x0)
            iy1 = iy0 + (y1 - y0)

            patch = img[iy0:iy1, ix0:ix1].astype(np.float32)
            accum[y0:y1, x0:x1] += patch
            weights[y0:y1, x0:x1] += 1.0

            footprints.append(
                {
                    "image": name,
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                }
            )

        output = np.zeros_like(accum, dtype=np.uint8)
        mask = weights > 0
        if np.any(mask):
            output[mask] = np.clip(accum[mask] / weights[mask, None], 0, 255).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)
        return str(output_path), footprints


    def _build_dsm_dtm(self, points: np.ndarray, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        dsm_png = output_dir / "dsm.png"
        dtm_png = output_dir / "dtm.png"
        dsm_npy = output_dir / "dsm.npy"
        dtm_npy = output_dir / "dtm.npy"

        if len(points) == 0:
            empty = np.zeros((64, 64), dtype=np.float32)
            cv2.imwrite(str(dsm_png), _normalize_to_colormap(empty, cv2.COLORMAP_JET))
            cv2.imwrite(str(dtm_png), _normalize_to_colormap(empty, cv2.COLORMAP_VIRIDIS))
            np.save(str(dsm_npy), empty)
            np.save(str(dtm_npy), empty)
            return {
                "dsm_path": str(dsm_png),
                "dtm_path": str(dtm_png),
                "dsm_npy_path": str(dsm_npy),
                "dtm_npy_path": str(dtm_npy),
                "origin_xy": [0.0, 0.0],
                "grid_size": self.dsm_grid_size,
                "dsm_array": empty,
            }

        xy = points[:, :2].astype(np.float64)
        z = points[:, 2].astype(np.float64)

        min_x, min_y = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
        max_x, max_y = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))

        grid = float(self.dsm_grid_size)
        gx = max(2, int(np.ceil((max_x - min_x) / grid)) + 1)
        gy = max(2, int(np.ceil((max_y - min_y) / grid)) + 1)

        max_dim = 1600
        if gx > max_dim or gy > max_dim:
            factor = max(gx / max_dim, gy / max_dim)
            grid *= float(factor)
            gx = max(2, int(np.ceil((max_x - min_x) / grid)) + 1)
            gy = max(2, int(np.ceil((max_y - min_y) / grid)) + 1)

        idx_x = np.clip(np.floor((xy[:, 0] - min_x) / grid).astype(int), 0, gx - 1)
        idx_y = np.clip(np.floor((xy[:, 1] - min_y) / grid).astype(int), 0, gy - 1)

        dsm = np.full((gy, gx), -np.inf, dtype=np.float32)
        dtm = np.full((gy, gx), np.inf, dtype=np.float32)

        for x, y, zz in zip(idx_x, idx_y, z):
            zf = float(zz)
            if zf > float(dsm[y, x]):
                dsm[y, x] = zf
            if zf < float(dtm[y, x]):
                dtm[y, x] = zf

        valid_dsm = np.isfinite(dsm)
        valid_dtm = np.isfinite(dtm)

        dsm_fill = float(np.median(dsm[valid_dsm])) if np.any(valid_dsm) else 0.0
        dtm_fill = float(np.median(dtm[valid_dtm])) if np.any(valid_dtm) else 0.0
        dsm[~valid_dsm] = dsm_fill
        dtm[~valid_dtm] = dtm_fill

        cv2.imwrite(str(dsm_png), _normalize_to_colormap(dsm, cv2.COLORMAP_JET))
        cv2.imwrite(str(dtm_png), _normalize_to_colormap(dtm, cv2.COLORMAP_VIRIDIS))
        np.save(str(dsm_npy), dsm)
        np.save(str(dtm_npy), dtm)

        return {
            "dsm_path": str(dsm_png),
            "dtm_path": str(dtm_png),
            "dsm_npy_path": str(dsm_npy),
            "dtm_npy_path": str(dtm_npy),
            "origin_xy": [min_x, min_y],
            "grid_size": grid,
            "dsm_array": dsm,
        }

    def _build_meshes(
        self,
        dsm: np.ndarray,
        origin_xy: tuple[float, float],
        grid_size: float,
        output_dir: Path,
        texture_path: str,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / "mesh.obj"
        textured_obj_path = output_dir / "textured_mesh.obj"
        textured_mtl_path = output_dir / "textured_mesh.mtl"

        if dsm.size == 0:
            _write_obj_mesh(mesh_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
            return {
                "mesh_path": str(mesh_path),
                "textured_mesh_obj_path": "",
                "textured_mesh_mtl_path": "",
                "texture_image_path": "",
            }

        h, w = dsm.shape
        step = max(1, int(np.ceil(max(h, w) / 680.0)))
        rows = np.arange(0, h, step, dtype=int)
        cols = np.arange(0, w, step, dtype=int)

        if len(rows) < 2 or len(cols) < 2:
            _write_obj_mesh(mesh_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
            return {
                "mesh_path": str(mesh_path),
                "textured_mesh_obj_path": "",
                "textured_mesh_mtl_path": "",
                "texture_image_path": "",
            }

        origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])

        vertices: list[list[float]] = []
        uvs: list[list[float]] = []
        idx_map = -np.ones((len(rows), len(cols)), dtype=np.int32)

        for ri, r in enumerate(rows):
            for ci, c in enumerate(cols):
                z = float(dsm[r, c])
                x = origin_x + float(c) * float(grid_size)
                y = origin_y + float(r) * float(grid_size)
                idx = len(vertices)
                idx_map[ri, ci] = idx
                vertices.append([x, y, z])
                u = float(c / max(w - 1, 1))
                v = float(1.0 - r / max(h - 1, 1))
                uvs.append([u, v])

        faces: list[list[int]] = []
        for ri in range(len(rows) - 1):
            for ci in range(len(cols) - 1):
                a = int(idx_map[ri, ci])
                b = int(idx_map[ri, ci + 1])
                c1 = int(idx_map[ri + 1, ci])
                d = int(idx_map[ri + 1, ci + 1])
                faces.append([a, b, d])
                faces.append([a, d, c1])

        verts_np = np.asarray(vertices, dtype=np.float32)
        uvs_np = np.asarray(uvs, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)

        _write_obj_mesh(mesh_path, verts_np, faces_np)

        texture_image_path = ""
        if texture_path and Path(texture_path).exists():
            texture_image_path = str(texture_path)
        _write_textured_obj(
            textured_obj_path,
            textured_mtl_path,
            texture_file_name=Path(texture_image_path).name if texture_image_path else "",
            vertices=verts_np,
            uvs=uvs_np,
            faces=faces_np,
        )

        return {
            "mesh_path": str(mesh_path),
            "textured_mesh_obj_path": str(textured_obj_path),
            "textured_mesh_mtl_path": str(textured_mtl_path),
            "texture_image_path": texture_image_path,
        }

    def reconstruct(
        self,
        image_dir: str | Path,
        crack_mask_dir: str | Path | None = None,
        output_dir: str | Path = "reconstruction_output",
        profile: str | None = None,
        execution_mode: str | None = None,
        use_cache: bool | None = None,
        cloud_endpoint: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> ReconstructionResult:
        def _emit_progress(percent: int, message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(int(max(0, min(100, int(percent)))), str(message))
            except Exception:
                pass

        _emit_progress(1, "Scanning input images")
        if profile is not None:
            self.profile = _normalize_profile(profile)
        if execution_mode is not None:
            self.execution_mode = _normalize_execution_mode(execution_mode)
        if use_cache is not None:
            self.use_cache = bool(use_cache)
        if cloud_endpoint is not None:
            self.cloud_endpoint = str(cloud_endpoint or "").strip()

        self._cache_hits = 0
        self._cache_misses = 0

        images = self._image_paths(image_dir)
        if len(images) < 2:
            raise ValueError("Need at least 2 images for reconstruction.")
        _emit_progress(3, f"Loaded {len(images)} frames")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = Path(crack_mask_dir) if crack_mask_dir else None

        cache_root = out_dir / "cache"
        feature_cache_dir = cache_root / "features"
        match_cache_dir = cache_root / "matches"
        if self.use_cache:
            feature_cache_dir.mkdir(parents=True, exist_ok=True)
            match_cache_dir.mkdir(parents=True, exist_ok=True)

        execution_requested = self.execution_mode
        execution_used = "local"
        cloud_request_path = ""
        cloud_response_path = ""
        warnings: list[str] = []

        if execution_requested == "cloud":
            _emit_progress(4, "Cloud requested; running locally (cloud not implemented)")
            cloud_request_path, cloud_response_path, cloud_warnings = self._request_cloud_run(
                image_dir=image_dir,
                output_dir=out_dir,
                image_count=len(images),
            )
            warnings.extend(cloud_warnings)
            execution_used = "local_cloud_unavailable"
        else:
            execution_used = "local"

        _emit_progress(5, "Initializing camera model")
        first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise FileNotFoundError(f"Could not read first image: {images[0]}")

        h, w = first.shape[:2]
        k = self._intrinsics(w, h)

        image_analysis_cache: dict[str, dict[str, Any]] = {}
        forecaster = CrackPropagationForecaster(growth_px_per_step=3.5, lateral_growth=1)

        def _analysis_for_image(image_path: Path, image_bgr: np.ndarray) -> dict[str, Any]:
            key = str(image_path.resolve())
            cached = image_analysis_cache.get(key)
            if cached is not None:
                return cached

            crack = detect_cracks(image_bgr)
            metal = detect_metal_defects(image_bgr)
            structural = detect_structural_defects(
                image_bgr,
                model_key="structural_multiclass_detector",
                use_model=True,
            )
            solar = detect_solar_defects(
                image_bgr,
                model_key="solar_defect_detector",
                use_model=True,
            )

            crack_bin = (crack.mask > 0).astype(np.uint8)
            metal_bin = (metal.mask > 0).astype(np.uint8)
            risk_map = 0.68 * crack_bin.astype(np.float32) + 0.32 * metal_bin.astype(np.float32)
            model_map = np.zeros_like(risk_map, dtype=np.float32)

            structural_weights = {
                "crack": 1.0,
                "delamination": 0.95,
                "rebar_exposure": 0.90,
                "moisture_intrusion": 0.88,
                "spalling": 0.84,
                "corrosion": 0.80,
                "efflorescence": 0.55,
            }
            for label, mask in structural.mask_by_class.items():
                if mask is None or mask.size == 0:
                    continue
                weight = float(structural_weights.get(str(label), 0.60))
                model_map = np.maximum(model_map, (mask > 0).astype(np.float32) * weight)

            solar_weights = {
                "hotspot": 0.92,
                "cell_crack": 0.86,
                "string_fault": 0.90,
                "delamination": 0.76,
                "soiling": 0.40,
            }
            for label, mask in solar.mask_by_class.items():
                if mask is None or mask.size == 0:
                    continue
                weight = float(solar_weights.get(str(label), 0.55))
                model_map = np.maximum(model_map, (mask > 0).astype(np.float32) * weight)

            risk_map = np.maximum(risk_map, model_map)
            risk_map = cv2.GaussianBlur(risk_map, (0, 0), sigmaX=3.0, sigmaY=3.0)
            peak = float(np.max(risk_map)) if risk_map.size else 0.0
            if peak > 1e-7:
                risk_map = np.clip(risk_map / peak, 0.0, 1.0)
            else:
                risk_map = np.zeros_like(risk_map, dtype=np.float32)

            growth_potential = 0.0
            try:
                _, forecast_metrics = forecaster.forecast(crack.mask, steps=4)
                if forecast_metrics:
                    base_ratio = float(forecast_metrics[0].crack_ratio)
                    final_ratio = float(forecast_metrics[-1].crack_ratio)
                    delta = max(0.0, final_ratio - base_ratio)
                    growth_potential = float(np.clip(delta / 0.22, 0.0, 1.0))
            except Exception:
                growth_potential = 0.0

            summary = {
                "source_image": key,
                "crack_ratio": float(crack.crack_ratio),
                "metal_ratio": float(metal.defect_ratio),
                "growth_potential": float(growth_potential),
                "crack_pixels": int(crack.crack_pixels),
                "metal_pixels": int(metal.defect_pixels),
                "structural_defect_ratio": float(structural.defect_ratio),
                "solar_defect_ratio": float(solar.defect_ratio),
                "structural_model_used": str(structural.model_used),
                "solar_model_used": str(solar.model_used),
                "structural_model_available": bool(structural.model_available),
                "solar_model_available": bool(solar.model_available),
            }
            payload = {
                "crack_mask": crack_bin,
                "metal_mask": metal_bin,
                "risk_map": risk_map.astype(np.float32),
                "summary": summary,
            }
            image_analysis_cache[key] = payload
            return payload

        r_prev = np.eye(3, dtype=np.float64)
        t_prev = np.zeros((3, 1), dtype=np.float64)

        camera_poses: list[dict] = [
            {
                "image": images[0].name,
                "R": r_prev.tolist(),
                "t": t_prev.reshape(-1).tolist(),
            }
        ]
        pair_stats: list[dict] = []

        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        base_links: list[dict[str, Any]] = []
        base_risk_scores: list[float] = []
        base_criticality_scores: list[float] = []
        crack_points: list[np.ndarray] = []

        processed_pairs = 0
        failed_pairs = 0

        total_pairs = max(1, len(images) - 1)
        for idx in range(len(images) - 1):
            pair_base = 6 + int((58.0 * float(idx)) / float(total_pairs))
            pair_next = 6 + int((58.0 * float(idx + 1)) / float(total_pairs))
            pair_span = max(1, pair_next - pair_base)
            pct_load = int(pair_base)
            pct_detect = int(min(64, pair_base + max(1, pair_span // 4)))
            pct_match = int(min(64, pair_base + max(2, pair_span // 2)))
            pct_pose = int(min(64, pair_base + max(3, (3 * pair_span) // 4)))
            pct_triang = int(min(64, pair_next - 1))

            _emit_progress(pct_load, f"Pair {idx + 1}/{total_pairs}: loading images")
            p1 = images[idx]
            p2 = images[idx + 1]

            img1 = cv2.imread(str(p1), cv2.IMREAD_COLOR)
            img2 = cv2.imread(str(p2), cv2.IMREAD_COLOR)
            if img1 is None or img2 is None:
                failed_pairs += 1
                continue

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            _emit_progress(pct_detect, f"Pair {idx + 1}/{total_pairs}: detecting features")
            pts1, des1 = self._detect_or_load_features(gray1, p1, feature_cache_dir)
            pts2, des2 = self._detect_or_load_features(gray2, p2, feature_cache_dir)
            if len(pts1) < 20 or len(pts2) < 20:
                failed_pairs += 1
                continue

            _emit_progress(
                pct_match,
                (
                    f"Pair {idx + 1}/{total_pairs}: matching features "
                    f"({len(des1)}x{len(des2)}, cap {self.max_match_descriptors})"
                ),
            )
            m1, m2 = self._match_or_load_points(pts1, des1, p1, pts2, des2, p2, match_cache_dir)
            if len(m1) < self.min_inliers:
                failed_pairs += 1
                continue

            _emit_progress(pct_pose, f"Pair {idx + 1}/{total_pairs}: estimating pose")
            e, inlier_mask = cv2.findEssentialMat(
                m1,
                m2,
                k,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if e is None or inlier_mask is None:
                failed_pairs += 1
                continue

            _, r_rel, t_rel, pose_mask = cv2.recoverPose(e, m1, m2, k)
            if pose_mask is None:
                failed_pairs += 1
                continue

            inliers = pose_mask.ravel().astype(bool)
            inlier_count = int(inliers.sum())
            if inlier_count < self.min_inliers:
                failed_pairs += 1
                continue

            pts1_in = m1[inliers]
            pts2_in = m2[inliers]

            r_curr = r_rel @ r_prev
            t_curr = r_rel @ t_prev + t_rel

            _emit_progress(pct_triang, f"Pair {idx + 1}/{total_pairs}: triangulating")
            pmat1 = k @ np.hstack([r_prev, t_prev])
            pmat2 = k @ np.hstack([r_curr, t_curr])
            tri_h = cv2.triangulatePoints(pmat1, pmat2, pts1_in.T, pts2_in.T)
            xyz = (tri_h[:3, :] / np.clip(tri_h[3:4, :], 1e-12, None)).T

            valid = np.isfinite(xyz).all(axis=1)
            depth1 = (r_prev @ xyz.T + t_prev).T[:, 2]
            depth2 = (r_curr @ xyz.T + t_curr).T[:, 2]
            valid &= depth1 > 0
            valid &= depth2 > 0
            valid &= np.linalg.norm(xyz, axis=1) < 1e5
            if int(valid.sum()) == 0:
                failed_pairs += 1
                continue

            xyz_valid = xyz[valid].astype(np.float32)
            pts1_valid = pts1_in[valid]
            pts2_valid = pts2_in[valid]

            px = np.clip(np.round(pts1_valid[:, 0]).astype(int), 0, img1.shape[1] - 1)
            py = np.clip(np.round(pts1_valid[:, 1]).astype(int), 0, img1.shape[0] - 1)
            rgb = img1[py, px][:, ::-1]

            analysis = _analysis_for_image(p1, img1)
            crack_map = np.asarray(analysis["crack_mask"], dtype=np.uint8)
            metal_map = np.asarray(analysis["metal_mask"], dtype=np.uint8)
            risk_map = np.asarray(analysis["risk_map"], dtype=np.float32)
            growth_potential = float(analysis["summary"].get("growth_potential", 0.0))
            model_pressure = float(
                np.clip(
                    0.6 * float(analysis["summary"].get("structural_defect_ratio", 0.0))
                    + 0.4 * float(analysis["summary"].get("solar_defect_ratio", 0.0)),
                    0.0,
                    1.0,
                )
            )

            local_scores = np.clip(risk_map[py, px], 0.0, 1.0)
            crack_hits = (crack_map[py, px] > 0).astype(np.float32)
            metal_hits = (metal_map[py, px] > 0).astype(np.float32)
            binary_score = crack_hits * 1.0 + metal_hits * 0.7
            risk_scores = np.clip(
                0.54 * local_scores + 0.20 * binary_score + 0.14 * growth_potential + 0.12 * model_pressure,
                0.0,
                1.0,
            )
            criticality_scores = np.clip(0.46 * risk_scores + 0.34 * growth_potential + 0.20 * model_pressure, 0.0, 1.0)

            all_points.append(xyz_valid)
            all_colors.append(rgb.astype(np.uint8))
            base_risk_scores.extend(float(v) for v in risk_scores.tolist())
            base_criticality_scores.extend(float(v) for v in criticality_scores.tolist())
            summary = dict(analysis["summary"])
            for x_pix, y_pix, score, crit, crack_flag, metal_flag in zip(
                px.tolist(),
                py.tolist(),
                risk_scores.tolist(),
                criticality_scores.tolist(),
                crack_hits.tolist(),
                metal_hits.tolist(),
            ):
                base_links.append(
                    {
                        "source_image": str(p1.resolve()),
                        "source_image_name": p1.name,
                        "paired_with": p2.name,
                        "pixel_xy": [int(x_pix), int(y_pix)],
                        "risk_score": float(score),
                        "criticality_score": float(crit),
                        "defect_flags": {
                            "crack": bool(crack_flag > 0.5),
                            "metal": bool(metal_flag > 0.5),
                        },
                        "image_summary": summary,
                    }
                )
            processed_pairs += 1

            mask1 = self._mask_for_image(mask_dir, p1)
            mask2 = self._mask_for_image(mask_dir, p2)
            if mask1 is not None or mask2 is not None:
                crack_sel = np.zeros(len(pts1_valid), dtype=bool)
                if mask1 is not None:
                    m1x = np.clip(np.round(pts1_valid[:, 0]).astype(int), 0, mask1.shape[1] - 1)
                    m1y = np.clip(np.round(pts1_valid[:, 1]).astype(int), 0, mask1.shape[0] - 1)
                    crack_sel |= mask1[m1y, m1x] > 0
                if mask2 is not None:
                    m2x = np.clip(np.round(pts2_valid[:, 0]).astype(int), 0, mask2.shape[1] - 1)
                    m2y = np.clip(np.round(pts2_valid[:, 1]).astype(int), 0, mask2.shape[0] - 1)
                    crack_sel |= mask2[m2y, m2x] > 0
                if crack_sel.any():
                    crack_points.append(xyz_valid[crack_sel])

            pair_stats.append(
                {
                    "pair": [p1.name, p2.name],
                    "match_count": int(len(m1)),
                    "inlier_count": int(inlier_count),
                    "triangulated_points": int(len(xyz_valid)),
                }
            )

            r_prev, t_prev = r_curr, t_curr
            camera_poses.append(
                {
                    "image": p2.name,
                    "R": r_curr.tolist(),
                    "t": t_curr.reshape(-1).tolist(),
                }
            )

        _emit_progress(66, "Aggregating triangulated points")
        if all_points:
            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)

        risk_scores_np = np.asarray(base_risk_scores, dtype=np.float32)
        criticality_scores_np = np.asarray(base_criticality_scores, dtype=np.float32)
        if len(base_links) != len(points) or len(risk_scores_np) != len(points) or len(criticality_scores_np) != len(points):
            n = int(min(len(points), len(base_links), len(risk_scores_np), len(criticality_scores_np)))
            points = points[:n]
            colors = colors[:n]
            base_links = base_links[:n]
            risk_scores_np = risk_scores_np[:n]
            criticality_scores_np = criticality_scores_np[:n]

        points, colors = self._densify_point_cloud(points, colors)

        if len(points) > self.max_points:
            n_before = int(len(points))
            rng = np.random.default_rng(42)
            sel = rng.choice(n_before, size=int(self.max_points), replace=False)
            points = points[sel]
            colors = colors[sel]
            if len(base_links) == n_before:
                base_links = [base_links[int(i)] for i in sel.tolist()]
            if len(risk_scores_np) == n_before:
                risk_scores_np = risk_scores_np[sel]
            if len(criticality_scores_np) == n_before:
                criticality_scores_np = criticality_scores_np[sel]

        if len(base_links) != len(points):
            base_links = [
                {
                    "source_image": "",
                    "source_image_name": "",
                    "paired_with": "",
                    "pixel_xy": [0, 0],
                    "risk_score": 0.0,
                    "criticality_score": 0.0,
                    "defect_flags": {"crack": False, "metal": False},
                    "image_summary": {},
                }
                for _ in range(len(points))
            ]
        if len(risk_scores_np) != len(points):
            risk_scores_np = np.zeros((len(points),), dtype=np.float32)
        if len(criticality_scores_np) != len(points):
            criticality_scores_np = np.clip(0.7 * risk_scores_np, 0.0, 1.0).astype(np.float32)

        _emit_progress(74, "Writing point clouds and camera poses")
        risk_levels = [_risk_level_from_score(float(v)) for v in risk_scores_np.tolist()]
        risk_colors = (
            np.asarray([_risk_color_rgb(level) for level in risk_levels], dtype=np.uint8)
            if len(points) > 0
            else np.empty((0, 3), dtype=np.uint8)
        )

        point_links: list[dict[str, Any]] = []
        for idx_point, (meta, xyz, risk_score, crit_score, level) in enumerate(
            zip(base_links, points, risk_scores_np.tolist(), criticality_scores_np.tolist(), risk_levels)
        ):
            row = dict(meta)
            row["point_index"] = int(idx_point)
            row["xyz"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            row["risk_score"] = float(risk_score)
            row["risk_level"] = str(level)
            row["criticality_score"] = float(crit_score)
            point_links.append(row)

        if crack_points:
            crack_xyz = np.concatenate(crack_points, axis=0).astype(np.float32)
            if len(crack_xyz) > self.max_points:
                rng = np.random.default_rng(7)
                sel = rng.choice(len(crack_xyz), size=self.max_points, replace=False)
                crack_xyz = crack_xyz[sel]
        else:
            crack_xyz = np.empty((0, 3), dtype=np.float32)

        cloud_path = out_dir / "custom_reconstruction.ply"
        risk_cloud_path = out_dir / "custom_reconstruction_risk.ply"
        crack_cloud_path = out_dir / "crack_points.ply"
        pose_path = out_dir / "camera_poses.json"
        point_link_map_path = out_dir / "point_link_map.json"
        critical_points_path = out_dir / "critical_points.json"

        _write_ascii_ply(cloud_path, points, colors)
        _write_ascii_ply(risk_cloud_path, points, risk_colors)
        crack_path_out: str | None = None
        if len(crack_xyz) > 0:
            crack_color = np.tile(np.array([[255, 32, 32]], dtype=np.uint8), (len(crack_xyz), 1))
            _write_ascii_ply(crack_cloud_path, crack_xyz, crack_color)
            crack_path_out = str(crack_cloud_path)

        _save_json(
            pose_path,
            {
                "frames": camera_poses,
                "pairs": pair_stats,
                "intrinsics": k.tolist(),
            },
        )

        image_summary_payload: dict[str, dict[str, Any]] = {}
        for key, payload in image_analysis_cache.items():
            summary = payload.get("summary", {})
            if isinstance(summary, dict):
                image_summary_payload[str(key)] = dict(summary)
        _save_json(
            point_link_map_path,
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "point_count": int(len(point_links)),
                "risk_thresholds": {"green_max": 0.25, "yellow_max": 0.55},
                "predictive_model": "heuristic_future_risk_v1",
                "image_summaries": image_summary_payload,
                "points": point_links,
            },
        )

        critical_indices = np.where(criticality_scores_np >= 0.72)[0]
        if len(critical_indices) == 0 and len(points) > 0:
            top_k = min(180, max(1, int(len(points) // 40)))
            critical_indices = np.argsort(-criticality_scores_np)[:top_k]
        if len(critical_indices) > 600:
            order = np.argsort(-criticality_scores_np[critical_indices])
            critical_indices = critical_indices[order[:600]]

        critical_points_payload: list[dict[str, Any]] = []
        for rank, idx_point in enumerate(critical_indices.tolist(), start=1):
            if idx_point < 0 or idx_point >= len(point_links):
                continue
            link = point_links[idx_point]
            reason: list[str] = []
            flags = link.get("defect_flags", {}) if isinstance(link.get("defect_flags", {}), dict) else {}
            if bool(flags.get("crack", False)):
                reason.append("active_crack")
            if bool(flags.get("metal", False)):
                reason.append("surface_metal_anomaly")
            img_summary = link.get("image_summary", {}) if isinstance(link.get("image_summary", {}), dict) else {}
            growth = float(img_summary.get("growth_potential", 0.0))
            if growth >= 0.2:
                reason.append("high_growth_potential")
            if not reason:
                reason = ["aggregated_risk"]

            critical_points_payload.append(
                {
                    "rank": int(rank),
                    "point_index": int(idx_point),
                    "xyz": link.get("xyz", [0.0, 0.0, 0.0]),
                    "source_image": str(link.get("source_image", "")),
                    "pixel_xy": link.get("pixel_xy", [0, 0]),
                    "risk_level": str(link.get("risk_level", "green")),
                    "risk_score": float(link.get("risk_score", 0.0)),
                    "criticality_score": float(link.get("criticality_score", 0.0)),
                    "reason": reason,
                }
            )

        _save_json(
            critical_points_path,
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "predictive_model": "heuristic_future_risk_v1",
                "critical_count": int(len(critical_points_payload)),
                "points": critical_points_payload,
            },
        )

        _emit_progress(84, "Generating orthomosaic")
        orthomosaic_path, footprints = self._build_orthomosaic(images, out_dir / "orthomosaic.png")
        guessed_offsets = int(getattr(self, "_mosaic_guessed_offsets", 0))
        if guessed_offsets:
            warnings.append(
                f"Orthomosaic: {guessed_offsets} of {max(0, len(images) - 1)} frame pairs "
                "could not be correlated, so an assumed forward overlap was substituted. "
                "Those sections of the mosaic are not positioned from image content."
            )
        _emit_progress(90, "Generating DSM/DTM")
        dem = self._build_dsm_dtm(points, out_dir)
        dsm_array = dem.pop("dsm_array")

        texture_for_mesh = orthomosaic_path if orthomosaic_path else dem.get("dsm_path", "")
        _emit_progress(94, "Generating mesh")
        meshes = self._build_meshes(
            dsm=dsm_array,
            origin_xy=(float(dem["origin_xy"][0]), float(dem["origin_xy"][1])),
            grid_size=float(dem["grid_size"]),
            output_dir=out_dir,
            texture_path=texture_for_mesh,
        )

        _emit_progress(98, "Writing digital twin metadata")
        digital_twin = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            # Sparse SfM only. This engine performs no multi-view stereo; the label
            # used to say "sfm_mvs_custom" while the only densification present was
            # jittered copies of the sparse points.
            "pipeline": "sfm_sparse_custom",
            "processing_profile": self.profile,
            "execution_mode_requested": execution_requested,
            "execution_mode_used": execution_used,
            "frame_count": len(images),
            "processed_pairs": processed_pairs,
            "failed_pairs": failed_pairs,
            "cache": {
                "enabled": bool(self.use_cache),
                "hits": int(self._cache_hits),
                "misses": int(self._cache_misses),
            },
            "artifacts": {
                "point_cloud": str(cloud_path),
                "risk_point_cloud": str(risk_cloud_path),
                "crack_cloud": crack_path_out or "",
                "point_links": str(point_link_map_path),
                "critical_points": str(critical_points_path),
                "camera_poses": str(pose_path),
                "orthomosaic": orthomosaic_path,
                "dsm": dem.get("dsm_path", ""),
                "dtm": dem.get("dtm_path", ""),
                "mesh": meshes.get("mesh_path", ""),
                "textured_mesh_obj": meshes.get("textured_mesh_obj_path", ""),
                "textured_mesh_mtl": meshes.get("textured_mesh_mtl_path", ""),
                "texture": meshes.get("texture_image_path", ""),
            },
            "footprints": footprints,
            "warnings": warnings,
        }
        twin_path = out_dir / "digital_twin.json"
        _save_json(twin_path, digital_twin)
        _emit_progress(100, "Reconstruction complete")

        return ReconstructionResult(
            frame_count=len(images),
            processed_pairs=processed_pairs,
            failed_pairs=failed_pairs,
            total_points=int(len(points)),
            crack_points=int(len(crack_xyz)),
            point_cloud_path=str(cloud_path),
            crack_cloud_path=crack_path_out,
            camera_pose_path=str(pose_path),
            orthomosaic_path=orthomosaic_path,
            dsm_path=str(dem.get("dsm_path", "")),
            dtm_path=str(dem.get("dtm_path", "")),
            mesh_path=str(meshes.get("mesh_path", "")),
            textured_mesh_obj_path=str(meshes.get("textured_mesh_obj_path", "")),
            textured_mesh_mtl_path=str(meshes.get("textured_mesh_mtl_path", "")),
            texture_image_path=str(meshes.get("texture_image_path", "")),
            digital_twin_path=str(twin_path),
            processing_profile=self.profile,
            execution_mode_requested=execution_requested,
            execution_mode_used=execution_used,
            cache_hits=int(self._cache_hits),
            cache_misses=int(self._cache_misses),
            feature_cache_dir=str(feature_cache_dir),
            match_cache_dir=str(match_cache_dir),
            cloud_request_path=cloud_request_path,
            cloud_response_path=cloud_response_path,
            risk_point_cloud_path=str(risk_cloud_path),
            point_link_path=str(point_link_map_path),
            critical_points_path=str(critical_points_path),
            warnings=warnings,
        )
