"""Dependency-free incremental structure-from-motion for the custom engine.

The custom engine used to chain two-view essential-matrix solves: ORB features, one
pose per consecutive pair, and every pair's translation normalised to unit length. On
a real aerial survey ORB found 15-25 matches per pair over grass, so 13 of 14 pairs
failed, and the pairs that did solve could not agree on scale anyway -- the "cloud" was
two points and the "orthomosaic" was the frames pasted in a strip at an assumed overlap.

This is a small but real incremental SfM, built only on OpenCV and NumPy:

* SIFT features selected like COLMAP's (largest scales), RootSIFT, exact mutual
  nearest-neighbour matching with Lowe's ratio test; ORB only without SIFT.
* An initial pair from a verified essential matrix, triangulated.
* Every further image registered by PnP against points already in the model, so all
  views share one scale, then new points triangulated and filtered by reprojection
  error and parallax.
* Sparse bundle adjustment (SciPy least-squares) as the model grows and at the end.
  Intrinsics come from EXIF and are held fixed, with no lens distortion model; the mean
  reprojection error is measured and reported so the result's quality is stated.

An image that cannot be registered is reported, never guessed into place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np


MAX_DETECT_DIMENSION = 2000
# Match every pair up to this many images; beyond it, a window along capture order.
EXHAUSTIVE_LIMIT = 40
# MAGSAC++ where this OpenCV has it: markedly more stable essential matrices than plain
# RANSAC on low-texture aerial pairs.
_ROBUST = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
SEQUENTIAL_WINDOW = 8
MIN_PNP_CORRESPONDENCES = 12
MAX_REPROJECTION_PX = 4.0
MIN_PARALLAX_DEG = 1.0


@dataclass
class SfmModel:
    """What the incremental solve produced, in its own (arbitrary-scale) frame."""

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    # Per point: (source image index, x, y, paired image index), in full-resolution pixels.
    observations: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    tracks: list[list[tuple[int, int]]] = field(default_factory=list)
    pair_stats: list[dict[str, Any]] = field(default_factory=list)
    unregistered: list[int] = field(default_factory=list)
    detector: str = "sift"
    mean_reprojection_error_px: float | None = None

    def center(self, index: int) -> np.ndarray:
        rotation, translation = self.poses[index]
        return (-rotation.T @ translation.reshape(3)).reshape(3)


def create_detector(max_features: int) -> tuple[Any, str]:
    """SIFT tuned like COLMAP's extractor; ORB only when this OpenCV has no SIFT.

    OpenCV's default contrast threshold (0.04) is twice COLMAP's effective one and its
    nfeatures cap keeps the strongest responses, which over grass are tiny blobs that do
    not repeat between views: 8 verified matches on a pair COLMAP matched 207 times.
    Detection is uncapped here and `detect` keeps the largest-scale features instead,
    as COLMAP does.
    """
    factory = getattr(cv2, "SIFT_create", None)
    if factory is not None:
        return factory(contrastThreshold=0.02), "sift"
    return cv2.ORB_create(nfeatures=int(max_features)), "orb"


def detect(detector: Any, gray: np.ndarray, max_features: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints in full-resolution pixel coordinates, detected on a bounded image."""
    height, width = gray.shape[:2]
    scale = min(1.0, MAX_DETECT_DIMENSION / float(max(height, width)))
    work = gray if scale >= 0.999 else cv2.resize(
        gray, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = detector.detectAndCompute(work, None)
    if descriptors is None or not keypoints:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)
    order = np.argsort([-kp.size for kp in keypoints], kind="stable")[: int(max_features)]
    points = np.asarray([keypoints[i].pt for i in order], dtype=np.float32) / float(scale)
    descriptors = np.asarray(descriptors)[order]
    if descriptors.dtype != np.uint8:
        # RootSIFT: Hellinger distance on SIFT histograms matches markedly better than L2.
        descriptors = descriptors.astype(np.float32)
        descriptors = np.sqrt(descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7))
    return points, descriptors.astype(descriptors.dtype)


def match(des1: np.ndarray, des2: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact mutual nearest neighbours passing Lowe's ratio test, as index arrays.

    Approximate FLANN search with few checks lost most true neighbours on 8000-feature
    sets; an exact search is a matrix product, done in row blocks to bound memory.
    """
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if len(des1) < 2 or len(des2) < 2:
        return empty
    if des1.dtype == np.uint8:
        raw = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(des1, des2, k=2)
        pairs = [(p[0].queryIdx, p[0].trainIdx) for p in raw
                 if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        if not pairs:
            return empty
        arr = np.asarray(pairs, dtype=np.int64)
        _, first = np.unique(arr[:, 1], return_index=True)
        return arr[first, 0], arr[first, 1]

    a = des1.astype(np.float32)
    b = des2.astype(np.float32)
    b_t = b.T.copy()
    b_sq = np.sum(b * b, axis=1)
    best_j = np.empty(len(a), dtype=np.int64)
    best_d = np.empty(len(a), dtype=np.float32)
    second_d = np.empty(len(a), dtype=np.float32)
    col_best_d = np.full(len(b), np.inf, dtype=np.float32)
    col_best_i = np.full(len(b), -1, dtype=np.int64)
    block = 2048
    for start in range(0, len(a), block):
        chunk = a[start:start + block]
        dist = np.sum(chunk * chunk, axis=1, keepdims=True) + b_sq[None, :] - 2.0 * (chunk @ b_t)
        np.maximum(dist, 0.0, out=dist)
        two = np.argpartition(dist, 1, axis=1)[:, :2]
        rows = np.arange(len(chunk))
        d0 = dist[rows, two[:, 0]]
        d1 = dist[rows, two[:, 1]]
        swap = d1 < d0
        best_j[start:start + len(chunk)] = np.where(swap, two[:, 1], two[:, 0])
        best_d[start:start + len(chunk)] = np.where(swap, d1, d0)
        second_d[start:start + len(chunk)] = np.where(swap, d0, d1)
        cols = np.argmin(dist, axis=0)
        values = dist[cols, np.arange(len(b))]
        better = values < col_best_d
        col_best_d[better] = values[better]
        col_best_i[better] = cols[better] + start
    query = np.arange(len(a), dtype=np.int64)
    ratio_ok = np.sqrt(best_d) < ratio * np.sqrt(second_d)
    mutual = col_best_i[best_j] == query
    keep = ratio_ok & mutual
    return query[keep], best_j[keep]


class IncrementalSfm:
    """Register an ordered image sequence into one consistently scaled model."""

    def __init__(
        self,
        images: Sequence[Path],
        k_matrix: np.ndarray,
        *,
        max_features: int = 8000,
        ratio: float = 0.8,
        min_inliers: int = 50,
        cache_dir: Path | None = None,
        progress: Callable[[int, str], None] | None = None,
        cache_hit: Callable[[bool], None] | None = None,
    ):
        self.images = list(images)
        self.k = np.asarray(k_matrix, dtype=np.float64)
        self.ratio = float(ratio)
        self.min_inliers = int(min_inliers)
        self.cache_dir = cache_dir
        self.progress = progress or (lambda _p, _m: None)
        self.cache_hit = cache_hit or (lambda _hit: None)
        self.detector, self.detector_name = create_detector(max_features)
        # Thresholds are defined at detection resolution and scaled to the full-resolution
        # pixels keypoints are stored in. A fixed 1.5 px on a 4896 px frame detected at
        # 2000 px is 0.6 px where the features were found, which rejected real matches.
        self.pixel_scale = 1.0
        if self.images:
            first = cv2.imread(str(self.images[0]), cv2.IMREAD_GRAYSCALE)
            if first is not None:
                self.pixel_scale = max(1.0, max(first.shape[:2]) / float(MAX_DETECT_DIMENSION))
        self.epipolar_px = 1.0 * self.pixel_scale  # ~1 px where features were found
        self.reprojection_px = MAX_REPROJECTION_PX * self.pixel_scale
        self.max_features = int(max_features)
        self.features: list[tuple[np.ndarray, np.ndarray]] = []
        self._matches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._color_cache: dict[int, np.ndarray] = {}

        self.poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.points: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.obs: list[tuple[int, float, float, int]] = []
        self.tracks: list[list[tuple[int, int]]] = []
        self.kp_to_point: list[dict[int, int]] = []
        self.pair_stats: list[dict[str, Any]] = []

    # -- features and matches ---------------------------------------------

    def _cache_file(self, index: int, suffix: str) -> Path | None:
        if self.cache_dir is None:
            return None
        path = self.images[index]
        try:
            stat = path.stat()
            stamp = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
        except OSError:
            return None
        key = hashlib.sha1(
            f"v2|{stamp}|{self.detector_name}|{self.max_features}|{MAX_DETECT_DIMENSION}|{suffix}".encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"{path.stem}_{key}.npz"

    def _load_features(self) -> None:
        total = max(1, len(self.images))
        for index, path in enumerate(self.images):
            self.progress(6 + int(20.0 * index / total), f"Detecting {self.detector_name.upper()} features {index + 1}/{total}")
            cached = self._cache_file(index, "features")
            if cached is not None and cached.exists():
                try:
                    blob = np.load(str(cached), allow_pickle=False)
                    self.features.append((blob["pts"], blob["des"]))
                    self.cache_hit(True)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                self.features.append((np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)))
                continue
            pts, des = detect(self.detector, gray, self.max_features)
            self.features.append((pts, des))
            self.cache_hit(False)
            if cached is not None:
                try:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(str(cached), pts=pts, des=des)
                except OSError:
                    pass
        self.kp_to_point = [dict() for _ in self.images]

    def _pair(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Matches between images i and j, verified against an essential matrix."""
        key = (i, j)
        if key in self._matches:
            return self._matches[key]
        pts_i, des_i = self.features[i]
        pts_j, des_j = self.features[j]
        qi, tj = match(des_i, des_j, self.ratio)
        if len(qi) >= 8:
            e_matrix, mask = cv2.findEssentialMat(
                pts_i[qi].astype(np.float64), pts_j[tj].astype(np.float64), self.k,
                method=_ROBUST, prob=0.999, threshold=self.epipolar_px,
            )
            if e_matrix is not None and mask is not None:
                keep = mask.ravel().astype(bool)
                qi, tj = qi[keep], tj[keep]
            else:
                qi, tj = qi[:0], tj[:0]
        self._matches[key] = (qi, tj)
        return qi, tj

    def _color(self, index: int) -> np.ndarray | None:
        if index not in self._color_cache:
            if len(self._color_cache) >= 4:
                self._color_cache.pop(next(iter(self._color_cache)))
            self._color_cache[index] = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)
        return self._color_cache[index]

    # -- geometry ------------------------------------------------------------

    def _projection(self, index: int) -> np.ndarray:
        rotation, translation = self.poses[index]
        return self.k @ np.hstack([rotation, translation.reshape(3, 1)])

    def _reprojection_error(self, index: int, xyz: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        rotation, translation = self.poses[index]
        cam = xyz @ rotation.T + translation.reshape(1, 3)
        proj = cam @ self.k.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = proj[:, :2] / proj[:, 2:3]
        return np.where(cam[:, 2] > 0, np.linalg.norm(uv - pixels, axis=1), np.inf)

    def _triangulate(self, i: int, j: int, qi: np.ndarray, tj: np.ndarray) -> int:
        """Add points for matches whose features are not yet in the model."""
        fresh = np.asarray(
            [a not in self.kp_to_point[i] and b not in self.kp_to_point[j] for a, b in zip(qi, tj)],
            dtype=bool,
        )
        if not fresh.any():
            return 0
        qi, tj = qi[fresh], tj[fresh]
        pix_i = self.features[i][0][qi].astype(np.float64)
        pix_j = self.features[j][0][tj].astype(np.float64)
        homogeneous = cv2.triangulatePoints(self._projection(i), self._projection(j), pix_i.T, pix_j.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = (homogeneous[:3] / homogeneous[3:4]).T
        good = np.isfinite(xyz).all(axis=1)
        good &= self._reprojection_error(i, xyz, pix_i) < self.reprojection_px
        good &= self._reprojection_error(j, xyz, pix_j) < self.reprojection_px
        ray_i = xyz - self.center(i)
        ray_j = xyz - self.center(j)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.sum(ray_i * ray_j, axis=1) / (
                np.linalg.norm(ray_i, axis=1) * np.linalg.norm(ray_j, axis=1))
        good &= np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > MIN_PARALLAX_DEG
        if not good.any():
            return 0
        color = self._color(i)
        added = 0
        for point, a, b, (x, y) in zip(xyz[good], qi[good], tj[good], pix_i[good]):
            pid = len(self.points)
            self.points.append(point)
            if color is not None:
                cy = int(np.clip(round(y), 0, color.shape[0] - 1))
                cx = int(np.clip(round(x), 0, color.shape[1] - 1))
                self.colors.append(color[cy, cx][::-1].astype(np.uint8))
            else:
                self.colors.append(np.array([200, 200, 200], dtype=np.uint8))
            self.obs.append((i, float(x), float(y), j))
            self.tracks.append([(i, int(a)), (j, int(b))])
            self.kp_to_point[i][int(a)] = pid
            self.kp_to_point[j][int(b)] = pid
            added += 1
        return added

    def center(self, index: int) -> np.ndarray:
        rotation, translation = self.poses[index]
        return (-rotation.T @ translation.reshape(3)).reshape(3)

    def _candidate_pairs(self) -> list[tuple[int, int]]:
        """Pairs worth matching: all of them for a small survey, a sequence window beyond.

        Consecutive-only matching was one reason the old engine failed: a lawnmower
        survey's neighbours across flight lines are far apart in capture order.
        """
        count = len(self.images)
        if count <= EXHAUSTIVE_LIMIT:
            return [(i, j) for i in range(count) for j in range(i + 1, count)]
        return [(i, j) for i in range(count) for j in range(i + 1, min(count, i + SEQUENTIAL_WINDOW + 1))]

    def _match_all(self) -> None:
        pairs = self._candidate_pairs()
        total = max(1, len(pairs))
        for step, (i, j) in enumerate(pairs):
            if step % 10 == 0:
                self.progress(26 + int(4.0 * step / total), f"Matching image pairs {step + 1}/{total}")
            self._pair(i, j)

    def _verified(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Verified matches between i and j, oriented (features of i, features of j)."""
        if (i, j) in self._matches:
            return self._matches[(i, j)]
        if (j, i) in self._matches:
            qj, ti = self._matches[(j, i)]
            return ti, qj
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def _initialize(self) -> int | None:
        """Seed the model from the best-supported pair with real baseline."""
        ranked = sorted(self._matches, key=lambda key: -len(self._matches[key][0]))
        for a, b in ranked:
            qi, tj = self._matches[(a, b)]
            if len(qi) < self.min_inliers:
                break
            pix_a = self.features[a][0][qi].astype(np.float64)
            pix_b = self.features[b][0][tj].astype(np.float64)
            e_matrix, _ = cv2.findEssentialMat(pix_a, pix_b, self.k, method=_ROBUST, prob=0.999, threshold=self.epipolar_px)
            if e_matrix is None or e_matrix.shape != (3, 3):
                continue
            inliers, rotation, translation, _mask = cv2.recoverPose(e_matrix, pix_a, pix_b, self.k)
            if int(inliers) < self.min_inliers:
                continue
            self.poses[a] = (np.eye(3), np.zeros(3))
            self.poses[b] = (rotation, translation.reshape(3))
            added = self._triangulate(a, b, qi, tj)
            # Too few points with parallax means the pair is close to a pure rotation or
            # a very short baseline; seeding from it gives a degenerate model.
            if added < max(MIN_PNP_CORRESPONDENCES, int(0.25 * int(inliers))):
                self.poses.clear()
                self.points.clear(); self.colors.clear(); self.obs.clear(); self.tracks.clear()
                self.kp_to_point = [dict() for _ in self.images]
                continue
            self.pair_stats.append({
                "pair": [self.images[a].name, self.images[b].name],
                "match_count": int(len(qi)), "inlier_count": int(inliers),
                "triangulated_points": int(added), "method": "essential_matrix",
            })
            return a
        return None

    def _correspondences(self, j: int) -> int:
        """How many model points image j observes through verified matches."""
        seen: set[int] = set()
        for ref in self.poses:
            qr, tj = self._verified(ref, j)
            for a, b in zip(qr, tj):
                if int(a) in self.kp_to_point[ref]:
                    seen.add(int(b))
        return len(seen)

    def _register(self, j: int, references: Sequence[int]) -> bool:
        """Place image j by PnP against model points seen in its reference images."""
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        seen: set[int] = set()
        pair_matches: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ref in references:
            qr, tj = self._verified(ref, j)
            if len(qr) == 0:
                continue
            pair_matches[ref] = (qr, tj)
            for a, b in zip(qr, tj):
                pid = self.kp_to_point[ref].get(int(a))
                if pid is None or int(b) in seen:
                    continue
                seen.add(int(b))
                object_points.append(self.points[pid])
                image_points.append(self.features[j][0][int(b)])
        if len(object_points) < MIN_PNP_CORRESPONDENCES:
            return False
        obj = np.asarray(object_points, dtype=np.float64)
        img = np.asarray(image_points, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.k, None, iterationsCount=2000,
            reprojectionError=self.reprojection_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < MIN_PNP_CORRESPONDENCES:
            return False
        idx = inliers.ravel()
        ok, rvec, tvec = cv2.solvePnP(obj[idx], img[idx], self.k, None, rvec, tvec,
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return False
        rotation, _ = cv2.Rodrigues(rvec)
        self.poses[j] = (rotation, tvec.reshape(3))

        added_total = 0
        for ref, (qr, tj) in pair_matches.items():
            # Extend existing tracks so later views can see them through j as well.
            for a, b in zip(qr, tj):
                pid = self.kp_to_point[ref].get(int(a))
                if pid is not None and int(b) not in self.kp_to_point[j]:
                    self.kp_to_point[j][int(b)] = pid
                    self.tracks[pid].append((j, int(b)))
            added = self._triangulate(ref, j, qr, tj) if ref < j else self._triangulate(j, ref, tj, qr)
            added_total += added
            self.pair_stats.append({
                "pair": [self.images[ref].name, self.images[j].name],
                "match_count": int(len(qr)), "inlier_count": int(len(idx)),
                "triangulated_points": int(added), "method": "pnp",
            })
        return True

    # -- bundle adjustment ----------------------------------------------------

    def _bundle_adjust(self, max_nfev: int = 40) -> None:
        """Jointly refine every pose and point against every observation.

        Without it each PnP pose inherits the previous poses' error and the model drifts
        (7.7 px mean reprojection on 15 Aukerman frames). This is the sparse
        Levenberg-Marquardt of the SciPy bundle-adjustment example, with a Huber loss and
        the first camera held fixed to pin the gauge. Intrinsics are not refined.
        """
        try:
            from scipy.optimize import least_squares
            from scipy.sparse import lil_matrix
        except ImportError:
            return
        cameras = sorted(self.poses)
        if len(cameras) < 3 or not self.points:
            return
        fixed = cameras[0]
        free = [c for c in cameras if c != fixed]
        cam_slot = {c: i for i, c in enumerate(free)}
        cam_ids, point_ids, pixels = [], [], []
        for pid, track in enumerate(self.tracks):
            for image_index, kp in track:
                if image_index in self.poses:
                    cam_ids.append(image_index)
                    point_ids.append(pid)
                    pixels.append(self.features[image_index][0][kp])
        if len(pixels) < 50:
            return
        cam_ids_arr = np.asarray(cam_ids)
        point_ids_arr = np.asarray(point_ids)
        pixels_arr = np.asarray(pixels, dtype=np.float64)
        n_cam, n_pts = len(free), len(self.points)

        def pack() -> np.ndarray:
            cams = []
            for c in free:
                rotation, translation = self.poses[c]
                cams.append(np.concatenate([cv2.Rodrigues(rotation)[0].ravel(), translation.reshape(3)]))
            return np.concatenate([np.asarray(cams).ravel(), np.asarray(self.points).ravel()])

        fixed_rt = np.concatenate([cv2.Rodrigues(self.poses[fixed][0])[0].ravel(),
                                   self.poses[fixed][1].reshape(3)])
        slot_of_obs = np.asarray([cam_slot.get(c, -1) for c in cam_ids_arr])
        fx, fy = self.k[0, 0], self.k[1, 1]
        cx, cy = self.k[0, 2], self.k[1, 2]

        def residuals(x: np.ndarray) -> np.ndarray:
            cam_params = x[: n_cam * 6].reshape(n_cam, 6)
            pts = x[n_cam * 6:].reshape(n_pts, 3)
            rt = np.where(slot_of_obs[:, None] >= 0, cam_params[np.maximum(slot_of_obs, 0)], fixed_rt[None, :])
            rvec, tvec = rt[:, :3], rt[:, 3:]
            p = pts[point_ids_arr]
            theta = np.linalg.norm(rvec, axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                axis = np.where(theta > 0, rvec / theta, 0.0)
            dot = np.sum(p * axis, axis=1, keepdims=True)
            cos, sin = np.cos(theta), np.sin(theta)
            rotated = cos * p + sin * np.cross(axis, p) + dot * (1 - cos) * axis
            cam = rotated + tvec
            z = np.where(np.abs(cam[:, 2:3]) < 1e-9, 1e-9, cam[:, 2:3])
            u = fx * cam[:, 0:1] / z + cx
            v = fy * cam[:, 1:2] / z + cy
            return (np.hstack([u, v]) - pixels_arr).ravel()

        sparsity = lil_matrix((2 * len(pixels_arr), n_cam * 6 + n_pts * 3), dtype=int)
        rows = np.arange(len(pixels_arr))
        has_cam = slot_of_obs >= 0
        for k in range(6):
            sparsity[2 * rows[has_cam], slot_of_obs[has_cam] * 6 + k] = 1
            sparsity[2 * rows[has_cam] + 1, slot_of_obs[has_cam] * 6 + k] = 1
        for k in range(3):
            sparsity[2 * rows, n_cam * 6 + point_ids_arr * 3 + k] = 1
            sparsity[2 * rows + 1, n_cam * 6 + point_ids_arr * 3 + k] = 1

        result = least_squares(
            residuals, pack(), jac_sparsity=sparsity, x_scale="jac", method="trf",
            loss="huber", f_scale=self.reprojection_px / 2.0, ftol=1e-4, max_nfev=max_nfev,
        )
        cam_params = result.x[: n_cam * 6].reshape(n_cam, 6)
        for c, params in zip(free, cam_params):
            self.poses[c] = (cv2.Rodrigues(params[:3].reshape(3, 1))[0], params[3:].copy())
        new_points = result.x[n_cam * 6:].reshape(n_pts, 3)
        self.points = [row.copy() for row in new_points]

        # Observations that still disagree after adjustment are outliers; drop them so
        # they neither colour the cloud nor anchor the next registration.
        errors = np.abs(residuals(result.x)).reshape(-1, 2).max(axis=1)
        bad = errors > self.reprojection_px
        if bad.any():
            for pid, cam in zip(point_ids_arr[bad], cam_ids_arr[bad]):
                self.tracks[pid] = [(i, kp) for i, kp in self.tracks[pid] if i != cam]
            self._drop_weak_points()

    def _drop_weak_points(self) -> None:
        """Remove points left with fewer than two observations, keeping indices consistent."""
        keep = [pid for pid, track in enumerate(self.tracks) if len(track) >= 2]
        if len(keep) == len(self.tracks):
            return
        remap = {old: new for new, old in enumerate(keep)}
        self.points = [self.points[i] for i in keep]
        self.colors = [self.colors[i] for i in keep]
        self.tracks = [self.tracks[i] for i in keep]
        obs = []
        for i in keep:
            source, x, y, paired = self.obs[i]
            track_images = {img for img, _kp in self.tracks[remap[i]]}
            if source not in track_images:
                img, kp = self.tracks[remap[i]][0]
                x, y = (float(v) for v in self.features[img][0][kp])
                source = img
            if paired not in track_images or paired == source:
                paired = next(img for img, _kp in self.tracks[remap[i]] if img != source)
            obs.append((source, x, y, paired))
        self.obs = obs
        self.kp_to_point = [dict() for _ in self.images]
        for pid, track in enumerate(self.tracks):
            for img, kp in track:
                self.kp_to_point[img][int(kp)] = pid

    # -- driver --------------------------------------------------------------

    def run(self) -> SfmModel:
        self._load_features()
        count = len(self.images)
        self._match_all()
        self.progress(30, "Finding an initial image pair")
        start = self._initialize()
        if start is None:
            return SfmModel(unregistered=list(range(count)), detector=self.detector_name)

        # Next-best-view: always add the image that sees the most of the model, which
        # is what keeps a grid survey connected across flight lines.
        failed: set[int] = set()
        while True:
            pending = [j for j in range(count) if j not in self.poses and j not in failed]
            if not pending:
                break
            scored = sorted(((self._correspondences(j), j) for j in pending), reverse=True)
            support, j = scored[0]
            if support < MIN_PNP_CORRESPONDENCES:
                break
            self.progress(32 + int(32.0 * len(self.poses) / max(1, count)),
                          f"Registering image {len(self.poses) + 1}/{count}: {self.images[j].name}")
            references = sorted(self.poses, key=lambda ref: -len(self._verified(ref, j)[0]))[:6]
            if not self._register(j, references):
                failed.add(j)
            else:
                # A new view can make an earlier failure solvable.
                failed.clear()
                if len(self.poses) in (4, 8, 16, 32, 64, 128):
                    self.progress(32 + int(32.0 * len(self.poses) / max(1, count)),
                                  f"Bundle adjusting {len(self.poses)} views")
                    self._bundle_adjust(max_nfev=20)

        self.progress(64, f"Final bundle adjustment of {len(self.poses)} views")
        self._bundle_adjust()

        points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        errors: list[float] = []
        for pid, track in enumerate(self.tracks):
            for image_index, kp in track:
                if image_index in self.poses:
                    pixel = self.features[image_index][0][kp].reshape(1, 2).astype(np.float64)
                    err = self._reprojection_error(image_index, points[pid:pid + 1], pixel)[0]
                    if np.isfinite(err):
                        errors.append(float(err))
        return SfmModel(
            poses=dict(self.poses),
            points=points,
            colors=np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3),
            observations=np.asarray(self.obs, dtype=np.float64).reshape(-1, 4),
            tracks=self.tracks,
            pair_stats=self.pair_stats,
            unregistered=[i for i in range(count) if i not in self.poses],
            detector=self.detector_name,
            mean_reprojection_error_px=float(np.mean(errors)) if errors else None,
        )


def level_frame(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and origin that put the dominant plane at z=0 with the cameras above it.

    Used only when no geotags are available. The result has no CRS and arbitrary units,
    but "up" is up, so a DSM of it is a height map rather than a depth map from the
    first camera.
    """
    centroid = np.median(points, axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred[: min(len(centred), 20000)], full_matrices=False)
    normal = vt[-1]
    if np.mean((centers - centroid) @ normal) < 0:
        normal = -normal
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z_axis)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(normal, z_axis))
    if sine < 1e-12:
        rotation = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        rotation, _ = cv2.Rodrigues(axis / sine * np.arctan2(sine, cosine))
    return rotation, centroid
