"""Fast on-site coverage and quality validation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import cos, radians
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from mission import (
    AssetReferenceFrame,
    MissionConstraints,
    MissionPlanner,
    export_flight_recipe,
    export_geojson,
    export_qgc_wpl,
    load_flight_recipe,
)


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class CoverageValidationConfig:
    """Thresholds used during fast ingest quality validation."""

    sharpness_threshold: float = 120.0
    motion_blur_threshold: float = 0.45
    lens_blur_ratio_threshold: float = 0.5
    exposure_clip_threshold: float = 0.12
    overlap_threshold: float = 0.14
    expected_overlap_threshold: float = 0.22
    max_gap_suggestions: int = 12
    min_gap_component_px: int = 1600
    thumbnail_width: int = 180
    canvas_size_px: int = 960


@dataclass
class _LayoutTransform:
    min_x: float
    max_y: float
    scale: float
    pad: float
    canvas_size: int


@dataclass
class _SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * (pts @ self.rotation) + self.translation


def _safe_imread(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _resize_with_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if w == width:
        return image
    scale = float(width) / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def _to_gray_small(image: np.ndarray, width: int = 720) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _resize_with_width(gray, width)


def _laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _edge_sharpness_ratio(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 1.0

    cx0 = int(w * 0.25)
    cx1 = int(w * 0.75)
    cy0 = int(h * 0.25)
    cy1 = int(h * 0.75)
    center = gray[cy0:cy1, cx0:cx1]

    top = gray[: max(1, int(h * 0.2)), :]
    bottom = gray[h - max(1, int(h * 0.2)) :, :]
    left = gray[:, : max(1, int(w * 0.2))]
    right = gray[:, w - max(1, int(w * 0.2)) :]

    center_score = _laplacian_var(center) if center.size > 0 else 1.0
    edge_scores = [
        _laplacian_var(top) if top.size > 0 else 0.0,
        _laplacian_var(bottom) if bottom.size > 0 else 0.0,
        _laplacian_var(left) if left.size > 0 else 0.0,
        _laplacian_var(right) if right.size > 0 else 0.0,
    ]
    edge_score = float(np.mean(edge_scores))
    return float(edge_score / max(center_score, 1e-6))


def _motion_blur_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(gx)))
    ey = float(np.mean(np.abs(gy)))
    return float(abs(ex - ey) / max(ex + ey, 1e-6))


def _quality_metrics(gray: np.ndarray, cfg: CoverageValidationConfig) -> dict:
    sharpness = _laplacian_var(gray)
    lens_ratio = _edge_sharpness_ratio(gray)
    motion = _motion_blur_score(gray)

    dark_clip = float(np.mean(gray <= 10))
    bright_clip = float(np.mean(gray >= 245))
    mean_luma = float(gray.mean())
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))

    blur = sharpness < cfg.sharpness_threshold
    lens_blur = lens_ratio < cfg.lens_blur_ratio_threshold
    motion_blur = motion > cfg.motion_blur_threshold and sharpness < cfg.sharpness_threshold * 1.4
    underexp = dark_clip > cfg.exposure_clip_threshold
    overexp = bright_clip > cfg.exposure_clip_threshold

    flags = []
    if blur:
        flags.append("blur")
    if lens_blur:
        flags.append("lens_blur")
    if motion_blur:
        flags.append("motion_blur")
    if underexp:
        flags.append("under_exposed")
    if overexp:
        flags.append("over_exposed")

    return {
        "sharpness": float(sharpness),
        "lens_blur_ratio": float(lens_ratio),
        "motion_blur_score": float(motion),
        "mean_luma": mean_luma,
        "p10_luma": p10,
        "p90_luma": p90,
        "dark_clip_pct": dark_clip,
        "bright_clip_pct": bright_clip,
        "quality_ok": len(flags) == 0,
        "quality_flags": flags,
    }


def _orb_overlap(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, int, int]:
    orb = cv2.ORB_create(nfeatures=1400, fastThreshold=8)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    k1 = len(kp1) if kp1 is not None else 0
    k2 = len(kp2) if kp2 is not None else 0
    keypoints = min(k1, k2)

    if des1 is None or des2 is None or keypoints == 0:
        return 0.0, 0, keypoints

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)

    good = 0
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < 0.76 * n.distance:
            good += 1

    score = float(good / max(keypoints, 1))
    return score, int(good), int(keypoints)


def _phase_shift(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, float]:
    a = prev_gray.astype(np.float32)
    b = curr_gray.astype(np.float32)
    shift, response = cv2.phaseCorrelate(a, b)
    dx = float(shift[0])
    dy = float(shift[1])
    return np.array([dx, -dy], dtype=np.float64), float(response)


def _draw_thumbnail(canvas_acc: np.ndarray, canvas_w: np.ndarray, thumb: np.ndarray, cx: int, cy: int) -> None:
    h, w = thumb.shape[:2]
    half_w = w // 2
    half_h = h // 2

    y0 = max(0, cy - half_h)
    y1 = min(canvas_acc.shape[0], cy + half_h)
    x0 = max(0, cx - half_w)
    x1 = min(canvas_acc.shape[1], cx + half_w)

    if x1 <= x0 or y1 <= y0:
        return

    tx0 = max(0, half_w - cx)
    ty0 = max(0, half_h - cy)
    tx1 = tx0 + (x1 - x0)
    ty1 = ty0 + (y1 - y0)

    patch = thumb[ty0:ty1, tx0:tx1].astype(np.float32)
    canvas_acc[y0:y1, x0:x1] += patch
    canvas_w[y0:y1, x0:x1] += 1.0


def _build_layout(positions: np.ndarray, canvas_size: int, pad: float = 90.0) -> tuple[np.ndarray, _LayoutTransform]:
    if len(positions) == 0:
        return np.zeros((0, 2), dtype=np.float64), _LayoutTransform(0.0, 0.0, 1.0, pad, canvas_size)

    min_x = float(np.min(positions[:, 0]))
    max_x = float(np.max(positions[:, 0]))
    min_y = float(np.min(positions[:, 1]))
    max_y = float(np.max(positions[:, 1]))

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    span = max(span_x, span_y)
    scale = float(max(1e-6, (canvas_size - 2.0 * pad) / span))

    mapped = np.zeros_like(positions, dtype=np.float64)
    mapped[:, 0] = (positions[:, 0] - min_x) * scale + pad
    mapped[:, 1] = (max_y - positions[:, 1]) * scale + pad
    return mapped, _LayoutTransform(min_x=min_x, max_y=max_y, scale=scale, pad=pad, canvas_size=canvas_size)


def _canvas_to_local(point_xy: np.ndarray, tf: _LayoutTransform) -> np.ndarray:
    x = (float(point_xy[0]) - tf.pad) / max(tf.scale, 1e-6) + tf.min_x
    y = tf.max_y - (float(point_xy[1]) - tf.pad) / max(tf.scale, 1e-6)
    return np.array([x, y], dtype=np.float64)


def _draw_expected_region(canvas_size: int, mapped_positions: np.ndarray, footprint: tuple[int, int]) -> np.ndarray:
    expected = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if len(mapped_positions) == 0:
        return expected

    radius = max(8, int(round(max(footprint[0], footprint[1]) * 0.45)))
    line_width = max(4, int(round(radius * 1.1)))

    pts = np.round(mapped_positions).astype(np.int32)
    for p in pts:
        cv2.circle(expected, (int(p[0]), int(p[1])), radius, 255, thickness=-1)

    for i in range(1, len(pts)):
        a = tuple(pts[i - 1])
        b = tuple(pts[i])
        cv2.line(expected, a, b, 255, thickness=line_width)

    expected = cv2.dilate(expected, np.ones((11, 11), dtype=np.uint8), iterations=1)
    return expected


def _dedup_suggestions(suggestions: list[dict], radius_px: float, max_items: int) -> list[dict]:
    if not suggestions:
        return []
    suggestions = sorted(suggestions, key=lambda row: float(row.get("priority", 0.0)), reverse=True)
    kept: list[dict] = []

    for row in suggestions:
        p = np.asarray(row.get("canvas_xy", [0.0, 0.0]), dtype=np.float64)
        merged = False
        for cur in kept:
            q = np.asarray(cur.get("canvas_xy", [0.0, 0.0]), dtype=np.float64)
            if float(np.linalg.norm(p - q)) <= radius_px:
                cur_priority = float(cur.get("priority", 0.0))
                cur["priority"] = float(max(cur_priority, float(row.get("priority", 0.0))))
                reasons = list(cur.get("reasons", []))
                for r in row.get("reasons", []):
                    if r not in reasons:
                        reasons.append(r)
                cur["reasons"] = reasons
                related = list(cur.get("related_images", []))
                for name in row.get("related_images", []):
                    if name not in related:
                        related.append(name)
                cur["related_images"] = related
                merged = True
                break
        if not merged:
            kept.append(row)
        if len(kept) >= max_items:
            break
    return kept


def _fit_similarity(src_xy: np.ndarray, dst_xy: np.ndarray) -> _SimilarityTransform | None:
    if len(src_xy) < 2 or len(dst_xy) < 2:
        return None

    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    src0 = src_xy - src_mean
    dst0 = dst_xy - dst_mean

    src_norm = float(np.linalg.norm(src0))
    dst_norm = float(np.linalg.norm(dst0))
    if src_norm < 1e-6 or dst_norm < 1e-6:
        return None

    m = src0.T @ dst0
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if float(np.linalg.det(r)) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt

    scale = float(dst_norm / src_norm)
    t = dst_mean - scale * (src_mean @ r)
    return _SimilarityTransform(scale=scale, rotation=r, translation=t)


def _local_to_world(points_xy: np.ndarray, frame: AssetReferenceFrame) -> np.ndarray:
    theta = radians(float(frame.yaw_deg))
    c = cos(theta)
    s = np.sin(theta)

    enu_x = points_xy[:, 0] * c - points_xy[:, 1] * s
    enu_y = points_xy[:, 0] * s + points_xy[:, 1] * c

    lon0 = radians(float(frame.origin_lon))
    lat0 = radians(float(frame.origin_lat))

    lon = enu_x / (cos(lat0) * EARTH_RADIUS_M) + lon0
    lat = enu_y / EARTH_RADIUS_M + lat0

    return np.column_stack([np.degrees(lon), np.degrees(lat)])


def _suggestion_hull_local(points_xy: np.ndarray, margin_m: float = 6.0) -> np.ndarray:
    if len(points_xy) == 0:
        return np.asarray([[0.0, 0.0], [margin_m, 0.0], [0.0, margin_m]], dtype=np.float64)

    if len(points_xy) >= 3:
        hull = cv2.convexHull(points_xy.astype(np.float32))
        hull = hull.reshape(-1, 2).astype(np.float64)
    elif len(points_xy) == 2:
        a = points_xy[0]
        b = points_xy[1]
        mid = (a + b) * 0.5
        v = b - a
        n = np.array([-v[1], v[0]], dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            n = np.array([0.0, 1.0], dtype=np.float64)
        else:
            n = n / norm
        hull = np.vstack([a, b, mid + n * margin_m])
    else:
        c = points_xy[0]
        hull = np.vstack(
            [
                c + np.array([-margin_m, -margin_m]),
                c + np.array([margin_m, -margin_m]),
                c + np.array([margin_m, margin_m]),
                c + np.array([-margin_m, margin_m]),
            ]
        )

    center = hull.mean(axis=0)
    expanded = center + (hull - center) * 1.15
    return expanded.astype(np.float64)


class CoverageValidator:
    """Fast ingest validator for quality, overlap, and on-site re-fly guidance."""

    def __init__(self, config: CoverageValidationConfig | None = None):
        self.config = config or CoverageValidationConfig()
        self._planner = MissionPlanner()

    def validate(
        self,
        image_paths: Iterable[Path],
        output_dir: str | Path,
        reference_recipe_path: str = "",
    ) -> dict:
        paths = [Path(p) for p in image_paths]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images: list[np.ndarray] = []
        grays: list[np.ndarray] = []
        thumbs: list[np.ndarray] = []
        rows: list[dict] = []
        skipped_images: list[str] = []

        for p in paths:
            img = _safe_imread(p)
            if img is None:
                skipped_images.append(p.name)
                continue

            gray = _to_gray_small(img)
            q = _quality_metrics(gray, self.config)
            rows.append(
                {
                    "image": p.name,
                    "valid": True,
                    **q,
                    "overlap_prev": 0.0,
                    "overlap_next": 0.0,
                    "low_overlap": False,
                }
            )
            images.append(img)
            grays.append(gray)
            thumbs.append(_resize_with_width(img, self.config.thumbnail_width))

        overlap_edges: list[dict] = []
        missing_pairs: list[dict] = []
        positions = np.zeros((len(grays), 2), dtype=np.float64)

        if len(grays) > 1:
            step_fallback = float(max(grays[0].shape[1] * 0.25, 40.0))
            for i in range(1, len(grays)):
                score, matches, keypoints = _orb_overlap(grays[i - 1], grays[i])
                shift, response = _phase_shift(grays[i - 1], grays[i])

                shift_norm = float(np.linalg.norm(shift))
                max_step = float(max(grays[i].shape[1] * 0.6, 80.0))
                if response < 0.05 or shift_norm < 1.0:
                    shift = np.array([step_fallback, 0.0], dtype=np.float64)
                elif shift_norm > max_step:
                    shift *= max_step / max(shift_norm, 1e-6)

                positions[i] = positions[i - 1] + shift

                overlap_edges.append(
                    {
                        "index_prev": i - 1,
                        "index_curr": i,
                        "image_prev": rows[i - 1].get("image", ""),
                        "image_curr": rows[i].get("image", ""),
                        "score": float(score),
                        "matches": int(matches),
                        "keypoints": int(keypoints),
                        "phase_response": float(response),
                        "shift_px": [float(shift[0]), float(shift[1])],
                    }
                )

                rows[i - 1]["overlap_next"] = float(score)
                rows[i]["overlap_prev"] = float(score)

                if score < self.config.overlap_threshold:
                    missing_pairs.append(
                        {
                            "index_prev": i - 1,
                            "index_curr": i,
                            "image_prev": rows[i - 1].get("image", ""),
                            "image_curr": rows[i].get("image", ""),
                            "score": float(score),
                        }
                    )

        for row in rows:
            prev_o = float(row.get("overlap_prev", 1.0))
            next_o = float(row.get("overlap_next", 1.0))
            low = min(prev_o, next_o) < self.config.overlap_threshold
            row["low_overlap"] = bool(low)
            if low and "overlap_gap" not in row["quality_flags"]:
                row["quality_flags"] = list(row.get("quality_flags", [])) + ["overlap_gap"]
                row["quality_ok"] = False

        mapped, tf = _build_layout(positions, canvas_size=self.config.canvas_size_px)
        mosaic, heatmap, counts, expected = self._render_preview_artifacts(thumbs, mapped)

        mosaic_path = out_dir / "preview_mosaic.png"
        heatmap_path = out_dir / "coverage_heatmap.png"
        expected_path = out_dir / "coverage_expected_mask.png"
        cv2.imwrite(str(mosaic_path), mosaic)
        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(str(expected_path), expected)

        suggestions = self._build_gap_suggestions(
            rows=rows,
            mapped_positions=mapped,
            counts=counts,
            expected=expected,
            transform=tf,
            missing_pairs=missing_pairs,
        )

        quality_fail = sum(1 for row in rows if not bool(row.get("quality_ok", False)))
        coverage_complete = float(np.mean(counts[expected > 0] > 0) * 100.0) if np.any(expected > 0) else 100.0

        result = {
            "config": asdict(self.config),
            "images": rows,
            "skipped_images": skipped_images,
            "overlap_pairs": overlap_edges,
            "missing_pairs": missing_pairs,
            "quality_fail_count": int(quality_fail),
            "coverage_completion_pct": float(coverage_complete),
            "preview_model_path": str(mosaic_path),
            "coverage_heatmap_path": str(heatmap_path),
            "coverage_expected_path": str(expected_path),
            "gap_suggestions": suggestions,
            "gap_suggestion_count": int(len(suggestions)),
            "gap_mission": {},
        }

        gap_payload = self._build_gap_mission(result, reference_recipe_path=reference_recipe_path, output_dir=out_dir)
        if gap_payload:
            result["gap_mission"] = gap_payload

        quality_json_path = out_dir / "coverage_validation.json"
        with quality_json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        result["coverage_json_path"] = str(quality_json_path)
        return result

    def _render_preview_artifacts(
        self,
        thumbnails: list[np.ndarray],
        mapped_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = self.config.canvas_size_px
        accum = np.zeros((size, size, 3), dtype=np.float32)
        weights = np.zeros((size, size), dtype=np.float32)
        counts = np.zeros((size, size), dtype=np.float32)

        if not thumbnails:
            empty = np.zeros((size, size, 3), dtype=np.uint8)
            return empty, empty, counts, np.zeros((size, size), dtype=np.uint8)

        thumb_h, thumb_w = thumbnails[0].shape[:2]
        half_w = max(1, thumb_w // 2)
        half_h = max(1, thumb_h // 2)

        for thumb, pt in zip(thumbnails, mapped_positions):
            cx = int(round(pt[0]))
            cy = int(round(pt[1]))
            _draw_thumbnail(accum, weights, thumb, cx, cy)

            y0 = max(0, cy - half_h)
            y1 = min(size, cy + half_h)
            x0 = max(0, cx - half_w)
            x1 = min(size, cx + half_w)
            if x1 > x0 and y1 > y0:
                counts[y0:y1, x0:x1] += 1.0

        preview = np.zeros_like(accum, dtype=np.uint8)
        mask = weights > 0
        preview[mask] = np.clip(accum[mask] / weights[mask, None], 0, 255).astype(np.uint8)

        expected = _draw_expected_region(size, mapped_positions, (thumb_w, thumb_h))
        norm = cv2.normalize(counts, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)

        gap_mask = np.zeros_like(expected)
        gap_mask[(expected > 0) & (counts <= 0.0)] = 255
        heat[gap_mask > 0] = (0, 0, 255)
        return preview, heat, counts, expected

    def _build_gap_suggestions(
        self,
        rows: list[dict],
        mapped_positions: np.ndarray,
        counts: np.ndarray,
        expected: np.ndarray,
        transform: _LayoutTransform,
        missing_pairs: list[dict],
    ) -> list[dict]:
        suggestions: list[dict] = []

        if len(mapped_positions) > 0:
            gap_mask = np.zeros_like(expected)
            gap_mask[(expected > 0) & (counts <= 0.0)] = 255
            comp, labels, stats, centroids = cv2.connectedComponentsWithStats(gap_mask, connectivity=8)
            for idx in range(1, comp):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < self.config.min_gap_component_px:
                    continue
                cx = float(centroids[idx, 0])
                cy = float(centroids[idx, 1])
                local = _canvas_to_local(np.array([cx, cy], dtype=np.float64), transform)
                suggestions.append(
                    {
                        "kind": "coverage_gap",
                        "priority": float(area),
                        "canvas_xy": [cx, cy],
                        "local_xy": [float(local[0]), float(local[1])],
                        "reasons": ["coverage_gap"],
                        "related_images": [],
                    }
                )

        for miss in missing_pairs:
            i0 = int(miss.get("index_prev", 0))
            i1 = int(miss.get("index_curr", 0))
            if i0 < 0 or i1 < 0 or i0 >= len(mapped_positions) or i1 >= len(mapped_positions):
                continue
            mid = (mapped_positions[i0] + mapped_positions[i1]) * 0.5
            local = _canvas_to_local(mid, transform)
            suggestions.append(
                {
                    "kind": "overlap_gap",
                    "priority": float(1.0 - float(miss.get("score", 0.0))) * 5000.0,
                    "canvas_xy": [float(mid[0]), float(mid[1])],
                    "local_xy": [float(local[0]), float(local[1])],
                    "reasons": ["overlap_gap"],
                    "related_images": [str(miss.get("image_prev", "")), str(miss.get("image_curr", ""))],
                }
            )

        for idx, row in enumerate(rows):
            if bool(row.get("quality_ok", True)):
                continue
            if idx >= len(mapped_positions):
                continue
            pt = mapped_positions[idx]
            local = _canvas_to_local(pt, transform)
            suggestions.append(
                {
                    "kind": "quality_retake",
                    "priority": float(2200.0 + 100.0 * len(row.get("quality_flags", []))),
                    "canvas_xy": [float(pt[0]), float(pt[1])],
                    "local_xy": [float(local[0]), float(local[1])],
                    "reasons": list(row.get("quality_flags", [])) or ["quality_issue"],
                    "related_images": [str(row.get("image", ""))],
                }
            )

        merged = _dedup_suggestions(suggestions, radius_px=36.0, max_items=self.config.max_gap_suggestions)

        for row in merged:
            cx, cy = row["canvas_xy"]
            nearest = int(np.argmin(np.linalg.norm(mapped_positions - np.array([cx, cy], dtype=np.float64), axis=1))) if len(mapped_positions) > 0 else 0
            row["suggested_yaw_deg"] = float(self._estimate_yaw(mapped_positions, nearest))
        return merged

    @staticmethod
    def _estimate_yaw(mapped_positions: np.ndarray, index: int) -> float:
        if len(mapped_positions) < 2:
            return 0.0
        i0 = max(0, index - 1)
        i1 = min(len(mapped_positions) - 1, index + 1)
        if i0 == i1:
            i0 = max(0, i1 - 1)
        a = mapped_positions[i0]
        b = mapped_positions[i1]
        vec = b - a
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(-vec[1], vec[0])))

    def _build_gap_mission(
        self,
        result: dict,
        reference_recipe_path: str,
        output_dir: Path,
    ) -> dict:
        if not reference_recipe_path:
            return {}
        recipe_path = Path(reference_recipe_path)
        if not recipe_path.exists():
            return {}

        suggestions = result.get("gap_suggestions", [])
        if not suggestions:
            return {}

        try:
            base_recipe = load_flight_recipe(recipe_path)
        except Exception:
            return {}

        try:
            compiled = self._planner.compile_recipe(base_recipe, repeat_enabled=True)
        except Exception:
            return {}

        base_local = np.asarray(
            [
                [float(p.get("x_m", 0.0)), float(p.get("y_m", 0.0))]
                for p in compiled.repeat_anchor.get("capture_poses_local", [])
            ],
            dtype=np.float64,
        )
        obs_local = np.asarray([row.get("local_xy", [0.0, 0.0]) for row in suggestions], dtype=np.float64)

        n_obs = int(result.get("gap_suggestion_count", len(obs_local)))
        if n_obs <= 0:
            return {}

        if len(base_local) >= 2 and len(obs_local) >= 2:
            k = min(len(base_local), len(obs_local), 24)
            idx_obs = np.linspace(0, len(obs_local) - 1, k).astype(int)
            idx_base = np.linspace(0, len(base_local) - 1, k).astype(int)
            sim = _fit_similarity(obs_local[idx_obs], base_local[idx_base])
        else:
            sim = None

        if sim is None:
            center = np.mean(base_local, axis=0) if len(base_local) else np.array([0.0, 0.0], dtype=np.float64)
            spread = np.std(base_local, axis=0).mean() if len(base_local) else 10.0
            mission_local = center + (obs_local - np.mean(obs_local, axis=0)) * max(float(spread), 8.0) / max(float(np.std(obs_local) or 1.0), 1.0)
        else:
            mission_local = sim.map_points(obs_local)

        polygon_local = _suggestion_hull_local(mission_local, margin_m=max(5.0, base_recipe.constraints.standoff_m))
        polygon_world = _local_to_world(polygon_local, base_recipe.asset_frame)
        polygon_lonlat = [[float(p[0]), float(p[1])] for p in polygon_world.tolist()]

        constraints = MissionConstraints(
            geofence=base_recipe.constraints.geofence,
            min_altitude_m=base_recipe.constraints.min_altitude_m,
            max_altitude_m=base_recipe.constraints.max_altitude_m,
            standoff_m=base_recipe.constraints.standoff_m,
            rth_altitude_m=base_recipe.constraints.rth_altitude_m,
            rth_action=base_recipe.constraints.rth_action,
            obstacle_avoidance_profile=base_recipe.constraints.obstacle_avoidance_profile,
        )

        gap_recipe = self._planner.build_flight_recipe(
            polygon_lonlat=polygon_lonlat,
            altitude_m=float(base_recipe.metadata.get("altitude_m", max(base_recipe.constraints.min_altitude_m, 40.0))),
            front_overlap_pct=base_recipe.coverage.front_overlap_pct,
            side_overlap_pct=base_recipe.coverage.side_overlap_pct,
            mode="smart_adaptive",
            camera=str(base_recipe.metadata.get("camera", "custom")),
            constraints=constraints,
            asset_frame=base_recipe.asset_frame,
            recipe_version=int(base_recipe.version) + 1,
            metadata={
                "parent_recipe_id": base_recipe.recipe_id,
                "purpose": "coverage_gap_refly",
                "generated_by": "coverage_validator",
                "gap_suggestion_count": int(len(suggestions)),
            },
        )
        gap_plan = self._planner.compile_recipe(gap_recipe, repeat_enabled=True)

        recipe_out = output_dir / "gap_mission_recipe.json"
        qgc_out = output_dir / "gap_mission.waypoints"
        geojson_out = output_dir / "gap_mission.geojson"

        export_flight_recipe(recipe_out, gap_recipe)
        export_qgc_wpl(qgc_out, gap_plan)
        export_geojson(geojson_out, gap_plan)

        suggestions_out = output_dir / "gap_suggestions.json"
        with suggestions_out.open("w", encoding="utf-8") as handle:
            json.dump(suggestions, handle, indent=2)

        return {
            "base_recipe_path": str(recipe_path),
            "recipe_path": str(recipe_out),
            "qgc_wpl_path": str(qgc_out),
            "geojson_path": str(geojson_out),
            "suggestions_path": str(suggestions_out),
            "suggestion_count": int(len(suggestions)),
            "waypoint_count": int(len(gap_plan.waypoints)),
            "autopilot_command_count": int(len(gap_plan.autopilot_commands)),
        }
