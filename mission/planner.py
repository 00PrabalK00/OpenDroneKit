"""Mission planning utilities with versioned flight recipes and compiler output."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from math import cos, pi, radians, sin, sqrt
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EARTH_RADIUS_M = 6_378_137.0


CAMERA_PRESETS = {
    "mavic2pro": {
        "sensor_w_mm": 13.2,
        "sensor_h_mm": 8.8,
        "focal_mm": 10.26,
        "image_w_px": 5472,
    },
    "phantom4rtk": {
        "sensor_w_mm": 13.2,
        "sensor_h_mm": 8.8,
        "focal_mm": 8.8,
        "image_w_px": 5472,
    },
    "custom": {
        "sensor_w_mm": 13.2,
        "sensor_h_mm": 8.8,
        "focal_mm": 10.0,
        "image_w_px": 4000,
    },
}


DOUBLE_GRID_MIN_FRONT_OVERLAP_PCT = 85.0
DOUBLE_GRID_MIN_SIDE_OVERLAP_PCT = 80.0
DOUBLE_GRID_DEFAULT_CROSS_ANGLE_DEG = 90.0
FACADE_MAPPING_MIN_FRONT_OVERLAP_PCT = 85.0
FACADE_MAPPING_MIN_SIDE_OVERLAP_PCT = 80.0
FACADE_MAPPING_SPACING_SCALE = 0.75
FACADE_MAPPING_NORMAL_GIMBAL_DEG = 0.0
FACADE_MAPPING_OBLIQUE_GIMBAL_DEG = -20.0


@dataclass
class AssetReferenceFrame:
    """Anchor frame for repeatable mission geometry."""

    asset_id: str
    origin_lon: float
    origin_lat: float
    yaw_deg: float = 0.0
    coordinate_source: str = "survey_polygon"
    reference_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissionConstraints:
    """Safety and operational constraints."""

    geofence: list[list[float]]
    min_altitude_m: float
    max_altitude_m: float
    standoff_m: float
    rth_altitude_m: float
    no_fly_polygons: list[list[list[float]]] = field(default_factory=list)
    rth_action: str = "return_home"
    obstacle_avoidance_profile: str = "balanced"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageExpectation:
    """Expected capture quality/coverage constraints."""

    front_overlap_pct: float
    side_overlap_pct: float
    minimum_coverage_pct: float = 95.0
    required_viewpoints: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissionPrimitive:
    """Primitive used by mission compiler."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlightRecipe:
    """Versioned recipe that can be compiled to autopilot commands."""

    recipe_id: str
    version: int
    template: str
    asset_frame: AssetReferenceFrame
    primitives: list[MissionPrimitive]
    constraints: MissionConstraints
    coverage: CoverageExpectation
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissionPlan:
    polygon: list[list[float]]
    waypoints: list[list[float]]
    altitude_m: float
    front_overlap_pct: float
    side_overlap_pct: float
    camera: str
    mode: str
    path_distance_m: float
    estimated_time_min: float
    estimated_gsd_cm: float
    source: str
    geojson: dict
    kmz_path: str = ""
    template: str = "grid"
    recipe_version: int = 1
    flight_recipe: dict = field(default_factory=dict)
    autopilot_commands: list[dict] = field(default_factory=list)
    safety_constraints: dict = field(default_factory=dict)
    repeat_enabled: bool = False
    repeat_anchor: dict = field(default_factory=dict)
    expected_coverage: dict = field(default_factory=dict)
    safety_adjustments: dict = field(default_factory=dict)
    flight_direction_deg: float = 0.0
    gimbal_tilt_deg: float = -90.0
    ground_offset_m: float = 0.0
    terrain_follow_enabled: bool = False
    terrain_follow_mode: str = "agl"
    terrain_normal_camera_enabled: bool = False
    terrain_normal_gain: float = 1.0
    terrain_normal_yaw_align: bool = False
    terrain_model_type: str = "flat"
    terrain_model_source: str = "none"
    terrain_source_path: str = ""
    line_spacing_m: float = 0.0
    capture_spacing_m: float = 0.0
    capture_interval_s: float = 0.0
    double_grid_cross_angle_deg: float = 0.0
    camera_policy: dict = field(default_factory=dict)
    camera_direction_deg: float = 0.0
    camera_direction_locked: bool = False
    inspection_dwell_s: float = 0.0
    facade_top_altitude_m: float = 0.0
    facade_bottom_altitude_m: float = 0.0
    facade_standoff_m: float = 0.0
    facade_rotate_points_180: bool = False
    facade_capture_profile: str = "custom"
    smooth_motion_profile: str = ""
    linear_segmentation_enabled: bool = False
    linear_max_segment_length_m: float = 0.0
    linear_segment_count: int = 0
    linear_path_length_m: float = 0.0
    lateral_standoff_m: float = 0.0
    lateral_target_side: str = "right"
    lateral_yaw_offset_deg: float = 0.0
    lateral_path_length_m: float = 0.0
    tower_top_altitude_m: float = 0.0
    tower_bottom_altitude_m: float = 0.0
    tower_object_radius_m: float = 0.0
    tower_flight_radius_m: float = 0.0
    tower_orbit_count: int = 0
    tower_resume_enabled: bool = False
    tower_safe_rth_altitude_m: float = 0.0
    solar_row_angle_deg: float = 0.0
    solar_sensor_profile: str = "rgb"
    solar_orientation_mode: str = "row_aligned"
    magnetic_tie_line_spacing_m: float = 0.0
    magnetic_smoothing_radius_m: float = 0.0
    orbit_radius_m: float = 0.0
    orbit_level_count: int = 0
    orbit_vertical_step_m: float = 0.0
    orbit_poi_yaw_lock: bool = True
    panorama_overlap_pct: float = 0.0
    panorama_multi_row_enabled: bool = False
    panorama_row_count: int = 0
    panorama_pitch_step_deg: float = 0.0
    panorama_yaw_step_deg: float = 0.0
    panorama_yaw_count: int = 0
    bubble_overlap_pct: float = 0.0
    bubble_pitch_step_deg: float = 0.0
    bubble_top_pitch_deg: float = 0.0
    bubble_bottom_pitch_deg: float = 0.0
    bubble_pitch_count: int = 0
    bubble_yaw_step_deg: float = 0.0
    bubble_yaw_count: int = 0
    waypoint_heading_mode: str = "tangent"
    waypoint_fixed_yaw_deg: float = 0.0
    waypoint_turn_radius_m: float = 0.0
    waypoint_smoothing_enabled: bool = False
    waypoint_capture_enabled: bool = True
    waypoint_path_length_m: float = 0.0
    no_fly_polygon_count: int = 0
    linked_segment_count: int = 0
    linked_transition_count: int = 0
    linked_dry_run_ok: bool = False
    wind_speed_m_s: float = 0.0
    wind_direction_deg: float = 0.0
    wind_gust_m_s: float = 0.0
    wind_adjusted_speed_m_s: float = 0.0
    wind_penalty_pct: float = 0.0
    facade_curvature_alignment: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _CapturePose:
    x_m: float
    y_m: float
    alt_m: float
    yaw_deg: float
    gimbal_pitch_deg: float
    primitive: str
    trigger: bool = True
    dwell_s: float = 0.0
    camera_yaw_locked: bool = False


def _normalize_template(mode: str) -> str:
    value = str(mode or "grid").strip().lower()
    table = {
        "grid": "grid",
        "double_grid": "double_grid",
        "roof": "roof_inspection",
        "inspection": "roof_inspection",
        "roof_inspection": "roof_inspection",
        "horizontal_inspection": "roof_inspection",
        "corridor": "corridor",
        "orbit": "orbit",
        "panorama": "panorama",
        "pano": "panorama",
        "spherical": "panorama",
        "360_bubble": "bubble_360",
        "360 bubble": "bubble_360",
        "bubble_360": "bubble_360",
        "bubble360": "bubble_360",
        "bubble": "bubble_360",
        "tower": "tower_mapping",
        "tower_mapping": "tower_mapping",
        "cell_tower": "tower_mapping",
        "stack": "tower_mapping",
        "dome": "tower_mapping",
        "solar": "solar_inspection",
        "solar_inspection": "solar_inspection",
        "solar_rows": "solar_inspection",
        "magnetic": "magnetic_mapping",
        "magnetic_mapping": "magnetic_mapping",
        "mag": "magnetic_mapping",
        "waypoint": "waypoints",
        "waypoints": "waypoints",
        "advanced_waypoints": "waypoints",
        "facade": "facade",
        "vertical": "facade",
        "vertical_scan": "facade",
        "facade_mapping": "facade_mapping",
        "vertical_mapping": "facade_mapping",
        "facade_3d": "facade_mapping",
        "linear": "linear_inspection",
        "linear_inspection": "linear_inspection",
        "pipeline": "linear_inspection",
        "rail": "linear_inspection",
        "river": "linear_inspection",
        "lateral": "lateral_capture",
        "lateral_capture": "lateral_capture",
        "lateral capture": "lateral_capture",
        "lateral profile": "lateral_capture",
        "profile_capture": "lateral_capture",
        "side_profile": "lateral_capture",
        "sideways": "lateral_capture",
        "linked_mission": "linked_mission",
        "linked mission": "linked_mission",
        "linked_route": "linked_mission",
        "smart": "smart_adaptive",
        "adaptive": "smart_adaptive",
        "smart_adaptive": "smart_adaptive",
    }
    return table.get(value, "grid")


def _normalize_facade_capture_profile(value: str | None) -> str:
    v = str(value or "custom").strip().lower()
    table = {
        "normal": "normal",
        "normal_0deg": "normal",
        "oblique": "oblique",
        "oblique_down": "oblique",
        "custom": "custom",
    }
    return table.get(v, "custom")


def _ensure_closed(coords_lonlat: Iterable[Iterable[float]]) -> list[list[float]]:
    pts = [[float(p[0]), float(p[1])] for p in coords_lonlat]
    if len(pts) < 3:
        raise ValueError("Polygon requires at least 3 points.")
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _ensure_line(coords_lonlat: Iterable[Iterable[float]]) -> list[list[float]]:
    pts = [[float(p[0]), float(p[1])] for p in coords_lonlat if len(p) >= 2]
    if len(pts) < 2:
        raise ValueError("Line requires at least 2 points.")
    cleaned = [pts[0]]
    for pt in pts[1:]:
        if abs(pt[0] - cleaned[-1][0]) > 1e-12 or abs(pt[1] - cleaned[-1][1]) > 1e-12:
            cleaned.append(pt)
    if len(cleaned) < 2:
        raise ValueError("Line requires at least 2 unique points.")
    return cleaned


def _ensure_closed_xy(points: Iterable[Iterable[float]]) -> list[list[float]]:
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3:
        raise ValueError("Local polygon requires at least 3 points.")
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _lonlat_to_xy(coords_lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lon = np.radians(coords_lonlat[:, 0])
    lat = np.radians(coords_lonlat[:, 1])
    lon0r = radians(lon0)
    lat0r = radians(lat0)
    x = (lon - lon0r) * cos(lat0r) * EARTH_RADIUS_M
    y = (lat - lat0r) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def _xy_to_lonlat(xy: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lon0r = radians(lon0)
    lat0r = radians(lat0)
    lon = xy[:, 0] / (cos(lat0r) * EARTH_RADIUS_M) + lon0r
    lat = xy[:, 1] / EARTH_RADIUS_M + lat0r
    return np.column_stack([np.degrees(lon), np.degrees(lat)])


def _rotate_xy(points_xy: np.ndarray, yaw_deg: float) -> np.ndarray:
    theta = np.radians(yaw_deg)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    return points_xy @ r.T


def _world_to_local(points_lonlat: list[list[float]], frame: AssetReferenceFrame) -> np.ndarray:
    arr = np.asarray(points_lonlat, dtype=np.float64)
    enu = _lonlat_to_xy(arr, lon0=frame.origin_lon, lat0=frame.origin_lat)
    return _rotate_xy(enu, -float(frame.yaw_deg))


def _local_to_world(points_local_xy: np.ndarray, frame: AssetReferenceFrame) -> np.ndarray:
    enu = _rotate_xy(points_local_xy, float(frame.yaw_deg))
    return _xy_to_lonlat(enu, lon0=frame.origin_lon, lat0=frame.origin_lat)


def _line_intersections_at_y(poly_xy: np.ndarray, y: float) -> list[float]:
    xs: list[float] = []
    for i in range(len(poly_xy) - 1):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[i + 1]
        if (y1 <= y < y2) or (y2 <= y < y1):
            if abs(y2 - y1) < 1e-9:
                continue
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    return xs


def _line_intersections_at_x(poly_xy: np.ndarray, x: float) -> list[float]:
    ys: list[float] = []
    for i in range(len(poly_xy) - 1):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[i + 1]
        if (x1 <= x < x2) or (x2 <= x < x1):
            if abs(x2 - x1) < 1e-9:
                continue
            t = (x - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    ys.sort()
    return ys


def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    for i in range(len(polygon_xy) - 1):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[i + 1]
        crosses = (y1 > y) != (y2 > y)
        if not crosses:
            continue
        denom = y2 - y1
        if abs(denom) < 1e-12:
            continue
        x_at_y = (x2 - x1) * (y - y1) / denom + x1
        if x < x_at_y:
            inside = not inside
    return inside


def _closest_point_on_segment(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    ab = b_xy - a_xy
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a_xy.copy()
    t = float(np.dot(point_xy - a_xy, ab) / denom)
    t = min(1.0, max(0.0, t))
    return a_xy + t * ab


def _project_inside_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if _point_in_polygon(point_xy, polygon_xy):
        return point_xy

    best = None
    best_d = float("inf")
    for i in range(len(polygon_xy) - 1):
        cand = _closest_point_on_segment(point_xy, polygon_xy[i], polygon_xy[i + 1])
        d = float(np.sum((cand - point_xy) ** 2))
        if d < best_d:
            best_d = d
            best = cand

    if best is None:
        return point_xy

    centroid = polygon_xy[:-1].mean(axis=0)
    nudged = best + (centroid - best) * 0.03
    if _point_in_polygon(nudged, polygon_xy):
        return nudged
    if _point_in_polygon(best, polygon_xy):
        return best
    return centroid


def _point_on_segment(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray, tol: float = 1e-9) -> bool:
    ap = point_xy - a_xy
    ab = b_xy - a_xy
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    if abs(cross) > tol:
        return False
    dot = float(np.dot(ap, ab))
    if dot < -tol:
        return False
    denom = float(np.dot(ab, ab))
    if dot > denom + tol:
        return False
    return True


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, tol: float = 1e-9) -> bool:
    def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if (o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol):
        if (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol):
            return True

    if abs(o1) <= tol and _point_on_segment(b1, a1, a2, tol=tol):
        return True
    if abs(o2) <= tol and _point_on_segment(b2, a1, a2, tol=tol):
        return True
    if abs(o3) <= tol and _point_on_segment(a1, b1, b2, tol=tol):
        return True
    if abs(o4) <= tol and _point_on_segment(a2, b1, b2, tol=tol):
        return True
    return False


def _segment_intersects_polygon(start_xy: np.ndarray, end_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    if len(polygon_xy) < 4:
        return False
    if _point_in_polygon(start_xy, polygon_xy) or _point_in_polygon(end_xy, polygon_xy):
        return True
    for i in range(len(polygon_xy) - 1):
        a = np.asarray(polygon_xy[i], dtype=np.float64)
        b = np.asarray(polygon_xy[i + 1], dtype=np.float64)
        if _segments_intersect(start_xy, end_xy, a, b):
            return True
    return False


def _project_outside_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray, margin_m: float = 1.0) -> np.ndarray:
    if len(polygon_xy) < 4:
        return point_xy
    if not _point_in_polygon(point_xy, polygon_xy):
        return point_xy

    best = None
    best_d = float("inf")
    for i in range(len(polygon_xy) - 1):
        cand = _closest_point_on_segment(point_xy, polygon_xy[i], polygon_xy[i + 1])
        d = float(np.sum((cand - point_xy) ** 2))
        if d < best_d:
            best_d = d
            best = cand
    if best is None:
        return point_xy

    centroid = np.asarray(polygon_xy[:-1], dtype=np.float64).mean(axis=0)
    direction = best - centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = point_xy - centroid
        norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array([1.0, 0.0], dtype=np.float64)
        norm = 1.0
    direction /= norm

    out = best + direction * max(0.5, float(margin_m))
    if not _point_in_polygon(out, polygon_xy):
        return out

    scale = max(0.5, float(margin_m))
    for _ in range(6):
        scale *= 1.5
        out = best + direction * scale
        if not _point_in_polygon(out, polygon_xy):
            return out
    return out


def _detour_points_around_polygon(start_xy: np.ndarray, end_xy: np.ndarray, polygon_xy: np.ndarray, margin_m: float = 2.0) -> list[np.ndarray]:
    if len(polygon_xy) < 4:
        return []
    if not _segment_intersects_polygon(start_xy, end_xy, polygon_xy):
        return []

    body = np.asarray(polygon_xy[:-1], dtype=np.float64)
    min_x = float(np.min(body[:, 0])) - float(margin_m)
    max_x = float(np.max(body[:, 0])) + float(margin_m)
    min_y = float(np.min(body[:, 1])) - float(margin_m)
    max_y = float(np.max(body[:, 1])) + float(margin_m)
    corners = [
        np.array([min_x, min_y], dtype=np.float64),
        np.array([max_x, min_y], dtype=np.float64),
        np.array([max_x, max_y], dtype=np.float64),
        np.array([min_x, max_y], dtype=np.float64),
    ]

    nodes = [np.asarray(start_xy, dtype=np.float64), np.asarray(end_xy, dtype=np.float64), *corners]
    n = len(nodes)
    inf = float("inf")
    dist = [inf] * n
    prev = [-1] * n
    visited = [False] * n
    dist[0] = 0.0

    for _ in range(n):
        cur = -1
        cur_d = inf
        for i in range(n):
            if not visited[i] and dist[i] < cur_d:
                cur = i
                cur_d = dist[i]
        if cur < 0:
            break
        if cur == 1:
            break
        visited[cur] = True

        for nxt in range(n):
            if nxt == cur or visited[nxt]:
                continue
            if _segment_intersects_polygon(nodes[cur], nodes[nxt], polygon_xy):
                continue
            w = float(np.linalg.norm(nodes[nxt] - nodes[cur]))
            cand = dist[cur] + w
            if cand < dist[nxt]:
                dist[nxt] = cand
                prev[nxt] = cur

    if not np.isfinite(dist[1]):
        return []

    path_idx: list[int] = []
    cur = 1
    while cur >= 0:
        path_idx.append(cur)
        if cur == 0:
            break
        cur = prev[cur]
    path_idx.reverse()
    if not path_idx or path_idx[0] != 0 or path_idx[-1] != 1:
        return []
    return [nodes[i] for i in path_idx[1:-1]]


def _sample_segment(start_xy: np.ndarray, end_xy: np.ndarray, spacing_m: float) -> list[np.ndarray]:
    spacing = max(0.5, float(spacing_m))
    delta = end_xy - start_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return [start_xy.copy()]
    count = max(1, int(np.ceil(length / spacing)))
    return [start_xy + delta * (i / count) for i in range(count + 1)]


def _bearing_deg(start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    vec = end_xy - start_xy
    if float(np.linalg.norm(vec)) < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _wrap_deg(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _estimate_gsd_cm(altitude_m: float, camera: str) -> float:
    p = CAMERA_PRESETS.get(camera.lower(), CAMERA_PRESETS["custom"])
    gsd_m = altitude_m * (p["sensor_w_mm"] / p["focal_mm"]) / p["image_w_px"]
    return float(gsd_m * 100.0)


def _estimate_footprint_m(altitude_m: float, camera: str) -> tuple[float, float]:
    p = CAMERA_PRESETS.get(camera.lower(), CAMERA_PRESETS["custom"])
    fw = altitude_m * (p["sensor_w_mm"] / p["focal_mm"])
    fh = altitude_m * (p["sensor_h_mm"] / p["focal_mm"])
    return float(fw), float(fh)


def _effective_footprint_m(altitude_m: float, camera: str, gimbal_pitch_deg: float) -> tuple[float, float]:
    fp_w, fp_h = _estimate_footprint_m(altitude_m, camera)
    nadir_delta = abs(float(gimbal_pitch_deg) + 90.0)
    nadir_delta = min(75.0, max(0.0, nadir_delta))
    oblique_scale = 1.0 / max(0.35, cos(radians(nadir_delta)))
    return fp_w, float(fp_h * oblique_scale)


def _horizontal_fov_deg(camera: str) -> float:
    p = CAMERA_PRESETS.get(camera.lower(), CAMERA_PRESETS["custom"])
    sensor_w = float(p["sensor_w_mm"])
    focal = max(1e-6, float(p["focal_mm"]))
    return float(np.degrees(2.0 * np.arctan(sensor_w / (2.0 * focal))))


def _recommended_shutter_s(speed_m_s: float, gsd_cm: float, blur_px_limit: float = 0.7) -> float:
    speed = max(0.5, float(speed_m_s))
    gsd_m = max(1e-4, float(gsd_cm) / 100.0)
    max_motion_m = gsd_m * max(0.2, float(blur_px_limit))
    raw = max_motion_m / speed
    return float(np.clip(raw, 1.0 / 2000.0, 1.0 / 250.0))


def _double_grid_camera_policy(speed_m_s: float, gsd_cm: float) -> dict[str, Any]:
    shutter_s = _recommended_shutter_s(speed_m_s=speed_m_s, gsd_cm=gsd_cm, blur_px_limit=0.7)
    return {
        "profile": "photogrammetry_locked_exposure",
        "capture_dataset": "3d_modelling_double_grid",
        "exposure_mode": "manual_locked",
        "white_balance_mode": "manual_locked",
        "focus_mode": "manual_locked",
        "iso_max": 200,
        "min_shutter_s": shutter_s,
        "exposure_change_tolerance_ev": 0.1,
    }


def _facade_mapping_camera_policy(speed_m_s: float, gsd_cm: float, capture_profile: str) -> dict[str, Any]:
    shutter_s = _recommended_shutter_s(speed_m_s=speed_m_s, gsd_cm=gsd_cm, blur_px_limit=0.6)
    return {
        "profile": "facade_photogrammetry_locked_exposure",
        "capture_dataset": "3d_facade_mapping",
        "capture_profile": _normalize_facade_capture_profile(capture_profile),
        "exposure_mode": "manual_locked",
        "white_balance_mode": "manual_locked",
        "focus_mode": "manual_locked",
        "iso_max": 200,
        "min_shutter_s": shutter_s,
        "exposure_change_tolerance_ev": 0.1,
    }


def _fit_terrain_plane(local_polygon_closed: list[list[float]], elevations_closed: list[float]) -> dict[str, Any]:
    if len(local_polygon_closed) < 4 or len(elevations_closed) < 4:
        return {"type": "flat", "source": "insufficient_samples", "z_ref_m": 0.0}

    xy = np.asarray(local_polygon_closed[:-1], dtype=np.float64)
    z = np.asarray(elevations_closed[:-1], dtype=np.float64)
    if len(xy) < 3 or len(z) < 3:
        return {"type": "flat", "source": "insufficient_samples", "z_ref_m": 0.0}

    a = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy), dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    return {
        "type": "plane",
        "source": "polygon_vertex_elevations",
        "slope_x_m_per_m": float(coef[0]),
        "slope_y_m_per_m": float(coef[1]),
        "offset_m": float(coef[2]),
        "z_ref_m": float(np.mean(z)),
        "sample_count": int(len(z)),
    }


def _terrain_elevation_m(point_xy: np.ndarray, terrain_model: dict[str, Any] | None) -> float | None:
    if not isinstance(terrain_model, dict):
        return None

    model_type = str(terrain_model.get("type", "flat")).strip().lower()
    x = float(point_xy[0])
    y = float(point_xy[1])
    if model_type == "plane":
        slope_x = float(terrain_model.get("slope_x_m_per_m", 0.0))
        slope_y = float(terrain_model.get("slope_y_m_per_m", 0.0))
        offset = float(terrain_model.get("offset_m", 0.0))
        return float(slope_x * x + slope_y * y + offset)

    if model_type == "grid":
        z_grid = terrain_model.get("z_grid")
        if not isinstance(z_grid, list) or not z_grid:
            return None
        try:
            z_arr = np.asarray(z_grid, dtype=np.float64)
            if z_arr.ndim != 2 or z_arr.shape[0] < 1 or z_arr.shape[1] < 1:
                return None
            x0 = float(terrain_model.get("x0_m", 0.0))
            y0 = float(terrain_model.get("y0_m", 0.0))
            dx = float(max(1e-6, terrain_model.get("dx_m", 1.0)))
            dy = float(max(1e-6, terrain_model.get("dy_m", 1.0)))

            fx = (x - x0) / dx
            fy = (y - y0) / dy
            ix = int(np.floor(fx))
            iy = int(np.floor(fy))
            tx = float(fx - ix)
            ty = float(fy - iy)

            if ix < 0 or iy < 0 or ix >= z_arr.shape[1] - 1 or iy >= z_arr.shape[0] - 1:
                ix = int(np.clip(ix, 0, z_arr.shape[1] - 1))
                iy = int(np.clip(iy, 0, z_arr.shape[0] - 1))
                return float(z_arr[iy, ix])

            z00 = float(z_arr[iy, ix])
            z10 = float(z_arr[iy, ix + 1])
            z01 = float(z_arr[iy + 1, ix])
            z11 = float(z_arr[iy + 1, ix + 1])
            z0 = z00 * (1.0 - tx) + z10 * tx
            z1 = z01 * (1.0 - tx) + z11 * tx
            return float(z0 * (1.0 - ty) + z1 * ty)
        except Exception:
            return None

    samples = terrain_model.get("samples_local")
    if isinstance(samples, list) and len(samples) >= 3:
        weighted = 0.0
        weight_sum = 0.0
        for sample in samples:
            if not isinstance(sample, (list, tuple)) or len(sample) < 3:
                continue
            sx = float(sample[0])
            sy = float(sample[1])
            sz = float(sample[2])
            d = sqrt((x - sx) ** 2 + (y - sy) ** 2)
            w = 1.0 / max(d, 0.25)
            weighted += w * sz
            weight_sum += w
        if weight_sum > 0.0:
            return float(weighted / weight_sum)

    return None


def _terrain_gradient(point_xy: np.ndarray, terrain_model: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(terrain_model, dict):
        return 0.0, 0.0

    model_type = str(terrain_model.get("type", "flat")).strip().lower()
    if model_type == "plane":
        return (
            float(terrain_model.get("slope_x_m_per_m", 0.0)),
            float(terrain_model.get("slope_y_m_per_m", 0.0)),
        )

    if model_type == "samples":
        samples = terrain_model.get("samples_local")
        if isinstance(samples, list) and len(samples) >= 3:
            rows: list[list[float]] = []
            zs: list[float] = []
            x = float(point_xy[0])
            y = float(point_xy[1])
            nearest = sorted(
                (
                    sample
                    for sample in samples
                    if isinstance(sample, (list, tuple)) and len(sample) >= 3
                ),
                key=lambda s: (float(s[0]) - x) ** 2 + (float(s[1]) - y) ** 2,
            )[: min(12, len(samples))]
            for sample in nearest:
                sx = float(sample[0])
                sy = float(sample[1])
                sz = float(sample[2])
                rows.append([sx, sy, 1.0])
                zs.append(sz)
            if len(rows) >= 3:
                try:
                    a = np.asarray(rows, dtype=np.float64)
                    z = np.asarray(zs, dtype=np.float64)
                    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
                    return float(coef[0]), float(coef[1])
                except Exception:
                    pass

    step = float(max(0.5, terrain_model.get("gradient_step_m", 2.0)))
    p = np.asarray(point_xy, dtype=np.float64)
    ex = np.array([step, 0.0], dtype=np.float64)
    ey = np.array([0.0, step], dtype=np.float64)
    z_px = _terrain_elevation_m(p + ex, terrain_model)
    z_mx = _terrain_elevation_m(p - ex, terrain_model)
    z_py = _terrain_elevation_m(p + ey, terrain_model)
    z_my = _terrain_elevation_m(p - ey, terrain_model)
    if z_px is None or z_mx is None or z_py is None or z_my is None:
        return 0.0, 0.0
    dz_dx = float((z_px - z_mx) / (2.0 * step))
    dz_dy = float((z_py - z_my) / (2.0 * step))
    return dz_dx, dz_dy


def _terrain_delta_m(point_xy: np.ndarray, terrain_model: dict[str, Any] | None) -> float:
    if not isinstance(terrain_model, dict):
        return 0.0
    if str(terrain_model.get("follow_mode", "agl")).strip().lower() == "amsl":
        return 0.0

    z_here = _terrain_elevation_m(point_xy, terrain_model)
    if z_here is not None:
        z_ref = float(terrain_model.get("z_ref_m", z_here))
        return float(z_here - z_ref)
    return 0.0


def _normalize_terrain_samples_local(samples: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(samples, list):
        return out
    for sample in samples:
        if isinstance(sample, dict):
            x = sample.get("x_m")
            y = sample.get("y_m")
            z = sample.get("elevation_m", sample.get("z_m"))
            if x is None or y is None or z is None:
                continue
            out.append([float(x), float(y), float(z)])
        elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
            out.append([float(sample[0]), float(sample[1]), float(sample[2])])
    return out


def _downsample_terrain_samples(
    samples_local: list[list[float]],
    max_points: int = 20000,
) -> list[list[float]]:
    if len(samples_local) <= max_points:
        return samples_local
    step = max(1, int(np.ceil(len(samples_local) / float(max_points))))
    return [samples_local[i] for i in range(0, len(samples_local), step)]


def _terrain_model_from_samples_local(
    samples_local: list[list[float]],
    source: str,
) -> dict[str, Any] | None:
    normalized = _normalize_terrain_samples_local(samples_local)
    if len(normalized) < 3:
        return None
    normalized = _downsample_terrain_samples(normalized)
    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    z_ref = float(np.nanmean(arr[:, 2]))
    return {
        "type": "samples",
        "source": source,
        "samples_local": normalized,
        "z_ref_m": z_ref,
        "sample_count": int(len(normalized)),
    }


def _load_terrain_samples_from_csv(path: str, frame: AssetReferenceFrame) -> dict[str, Any] | None:
    src = Path(path)
    if not src.exists():
        return None
    delim = "\t" if src.suffix.lower() == ".tsv" else ","

    try:
        with src.open("r", encoding="utf-8", newline="") as fh:
            head = fh.readline()
            fh.seek(0)
            has_header = any(ch.isalpha() for ch in head)
            if has_header:
                reader = csv.DictReader(fh, delimiter=delim)
                rows = [dict(r) for r in reader if isinstance(r, dict)]
            else:
                reader_raw = csv.reader(fh, delimiter=delim)
                rows = [{"c0": r[0], "c1": r[1], "c2": r[2]} for r in reader_raw if len(r) >= 3]
    except Exception:
        return None

    if not rows:
        return None

    def _pick(row: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            for rk, rv in row.items():
                k = str(rk or "").strip().lower().replace("-", "_").replace(" ", "_")
                if k == key:
                    try:
                        return float(rv)
                    except Exception:
                        return None
        return None

    world_pts: list[list[float]] = []
    local_samples: list[list[float]] = []
    for row in rows:
        lon = _pick(row, ["lon", "longitude"])
        lat = _pick(row, ["lat", "latitude"])
        z = _pick(row, ["elevation_m", "elevation", "z_m", "z", "height_m", "height"])
        if lon is not None and lat is not None and z is not None:
            world_pts.append([float(lon), float(lat), float(z)])
            continue

        x = _pick(row, ["x_m", "x"])
        y = _pick(row, ["y_m", "y"])
        if x is not None and y is not None and z is not None:
            local_samples.append([float(x), float(y), float(z)])
            continue

        # Headerless fallback (first 3 columns).
        c0 = _pick(row, ["c0"])
        c1 = _pick(row, ["c1"])
        c2 = _pick(row, ["c2"])
        if c0 is None or c1 is None or c2 is None:
            continue
        if -180.0 <= c0 <= 180.0 and -90.0 <= c1 <= 90.0:
            world_pts.append([float(c0), float(c1), float(c2)])
        else:
            local_samples.append([float(c0), float(c1), float(c2)])

    if len(world_pts) >= 3:
        world_xy = [[p[0], p[1]] for p in world_pts]
        elev = [float(p[2]) for p in world_pts]
        local = _world_to_local(world_xy, frame)
        samples_local = [[float(pt[0]), float(pt[1]), elev[i]] for i, pt in enumerate(local)]
        return _terrain_model_from_samples_local(samples_local, source=f"terrain_csv:{src.name}")

    if len(local_samples) >= 3:
        return _terrain_model_from_samples_local(local_samples, source=f"terrain_csv:{src.name}")
    return None


def _load_terrain_samples_from_ascii_grid(path: str, frame: AssetReferenceFrame) -> dict[str, Any] | None:
    src = Path(path)
    if not src.exists():
        return None
    try:
        text = src.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return None

    header: dict[str, float] = {}
    data_start = 0
    for idx, line in enumerate(lines[:20]):
        parts = line.split()
        if len(parts) < 2:
            continue
        key = str(parts[0]).strip().lower()
        if key in {"ncols", "nrows", "xllcorner", "xllcenter", "yllcorner", "yllcenter", "cellsize", "nodata_value"}:
            try:
                header[key] = float(parts[1])
                data_start = idx + 1
                continue
            except Exception:
                pass
        # stop when first non-header line is reached
        if idx > 1:
            break

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    cell = float(header.get("cellsize", 0.0))
    if ncols <= 0 or nrows <= 0 or cell <= 0.0:
        return None

    data_lines = lines[data_start : data_start + nrows]
    if len(data_lines) < nrows:
        return None

    try:
        grid = np.asarray([[float(v) for v in ln.split()] for ln in data_lines], dtype=np.float64)
    except Exception:
        return None
    if grid.shape[0] != nrows:
        return None
    if grid.shape[1] != ncols:
        # accept ragged rows only if truncation still leaves enough columns
        if grid.shape[1] < 2:
            return None
        ncols = int(grid.shape[1])

    nodata = header.get("nodata_value")
    if nodata is not None:
        grid = np.where(np.isclose(grid, float(nodata)), np.nan, grid)

    x0 = header.get("xllcorner")
    y0 = header.get("yllcorner")
    x_center = header.get("xllcenter")
    y_center = header.get("yllcenter")
    if x0 is None and x_center is None:
        return None
    if y0 is None and y_center is None:
        return None

    if x0 is None:
        x0 = float(x_center) - 0.5 * cell
    if y0 is None:
        y0 = float(y_center) - 0.5 * cell

    # ESRI ASCII grid is stored top->bottom. Convert each cell center.
    rr, cc = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    x_vals = float(x0) + (cc.astype(np.float64) + 0.5) * cell
    y_vals = float(y0) + ((nrows - 1 - rr).astype(np.float64) + 0.5) * cell
    z_vals = grid.astype(np.float64)

    valid = np.isfinite(z_vals)
    if not np.any(valid):
        return None
    x_flat = x_vals[valid].reshape(-1)
    y_flat = y_vals[valid].reshape(-1)
    z_flat = z_vals[valid].reshape(-1)

    step = max(1, int(np.ceil(np.sqrt(len(z_flat) / 20000.0))))
    x_flat = x_flat[::step]
    y_flat = y_flat[::step]
    z_flat = z_flat[::step]

    if len(z_flat) < 3:
        return None

    # If grid coordinates look geodetic, convert via frame; else treat as local meters.
    if (
        np.nanmin(x_flat) >= -180.0
        and np.nanmax(x_flat) <= 180.0
        and np.nanmin(y_flat) >= -90.0
        and np.nanmax(y_flat) <= 90.0
    ):
        world_xy = [[float(x_flat[i]), float(y_flat[i])] for i in range(len(z_flat))]
        local_xy = _world_to_local(world_xy, frame)
        samples_local = [[float(pt[0]), float(pt[1]), float(z_flat[i])] for i, pt in enumerate(local_xy)]
    else:
        samples_local = [[float(x_flat[i]), float(y_flat[i]), float(z_flat[i])] for i in range(len(z_flat))]

    return _terrain_model_from_samples_local(samples_local, source=f"terrain_ascii:{src.name}")


def _load_terrain_samples_from_raster(path: str, frame: AssetReferenceFrame) -> dict[str, Any] | None:
    src = Path(path)
    if not src.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import transform as rio_transform
    except Exception:
        return None

    try:
        with rasterio.open(src) as ds:
            band = ds.read(1, masked=True)
            if band.ndim != 2 or band.size < 4:
                return None
            rows, cols = band.shape
            stride = max(1, int(np.ceil(np.sqrt((rows * cols) / 25000.0))))
            rr = np.arange(0, rows, stride, dtype=np.int64)
            cc = np.arange(0, cols, stride, dtype=np.int64)
            row_grid, col_grid = np.meshgrid(rr, cc, indexing="ij")

            values = np.asarray(band[row_grid, col_grid], dtype=np.float64).reshape(-1)
            if np.ma.is_masked(band):
                mask = ~np.asarray(np.ma.getmaskarray(band[row_grid, col_grid]), dtype=bool).reshape(-1)
            else:
                mask = np.ones(values.shape[0], dtype=bool)
            mask &= np.isfinite(values)
            if not np.any(mask):
                return None

            xs, ys = rasterio.transform.xy(
                ds.transform,
                row_grid.reshape(-1).tolist(),
                col_grid.reshape(-1).tolist(),
                offset="center",
            )
            x_arr = np.asarray(xs, dtype=np.float64)
            y_arr = np.asarray(ys, dtype=np.float64)
            z_arr = values

            x_arr = x_arr[mask]
            y_arr = y_arr[mask]
            z_arr = z_arr[mask]
            if len(z_arr) < 3:
                return None

            if ds.crs is not None:
                try:
                    epsg = ds.crs.to_epsg()
                except Exception:
                    epsg = None
                if epsg != 4326:
                    lons, lats = rio_transform(ds.crs, "EPSG:4326", x_arr.tolist(), y_arr.tolist())
                    lon_arr = np.asarray(lons, dtype=np.float64)
                    lat_arr = np.asarray(lats, dtype=np.float64)
                else:
                    lon_arr = x_arr
                    lat_arr = y_arr
            else:
                # Without CRS metadata, only accept lon/lat-like coordinate ranges.
                if (
                    np.nanmin(x_arr) < -180.0
                    or np.nanmax(x_arr) > 180.0
                    or np.nanmin(y_arr) < -90.0
                    or np.nanmax(y_arr) > 90.0
                ):
                    return None
                lon_arr = x_arr
                lat_arr = y_arr

            valid_geo = (
                np.isfinite(lon_arr)
                & np.isfinite(lat_arr)
                & (lon_arr >= -180.0)
                & (lon_arr <= 180.0)
                & (lat_arr >= -90.0)
                & (lat_arr <= 90.0)
            )
            if not np.any(valid_geo):
                return None

            lon_arr = lon_arr[valid_geo]
            lat_arr = lat_arr[valid_geo]
            z_arr = z_arr[valid_geo]
            if len(z_arr) < 3:
                return None

            world_xy = [[float(lon_arr[i]), float(lat_arr[i])] for i in range(len(z_arr))]
            local_xy = _world_to_local(world_xy, frame)
            samples_local = [[float(pt[0]), float(pt[1]), float(z_arr[i])] for i, pt in enumerate(local_xy)]
            return _terrain_model_from_samples_local(samples_local, source=f"terrain_raster:{src.name}")
    except Exception:
        return None


def _load_terrain_model_from_path(path: str, frame: AssetReferenceFrame) -> dict[str, Any] | None:
    source = str(path or "").strip()
    if not source:
        return None
    ext = Path(source).suffix.lower()

    if ext in {".csv", ".tsv"}:
        model = _load_terrain_samples_from_csv(source, frame=frame)
        if isinstance(model, dict):
            return model

    if ext in {".asc", ".grd"}:
        model = _load_terrain_samples_from_ascii_grid(source, frame=frame)
        if isinstance(model, dict):
            return model

    if ext in {".tif", ".tiff", ".geotiff"}:
        model = _load_terrain_samples_from_raster(source, frame=frame)
        if isinstance(model, dict):
            return model

    try:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    model_type = str(payload.get("type", "samples")).strip().lower()
    if model_type == "grid":
        grid = payload.get("z_grid", payload.get("elevations"))
        if not isinstance(grid, list) or not grid:
            return None
        x0_m = payload.get("x0_m")
        y0_m = payload.get("y0_m")
        if x0_m is None or y0_m is None:
            lon0 = payload.get("origin_lon", payload.get("lon0"))
            lat0 = payload.get("origin_lat", payload.get("lat0"))
            if lon0 is None or lat0 is None:
                return None
            local0 = _world_to_local([[float(lon0), float(lat0)]], frame)[0]
            x0_m = float(local0[0])
            y0_m = float(local0[1])
        z_arr = np.asarray(grid, dtype=np.float64)
        if z_arr.ndim != 2 or z_arr.shape[0] < 1 or z_arr.shape[1] < 1:
            return None
        z_ref = float(payload.get("z_ref_m", float(np.nanmean(z_arr))))
        return {
            "type": "grid",
            "source": f"terrain_file:{Path(source).name}",
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "dx_m": float(max(1e-6, payload.get("dx_m", 1.0))),
            "dy_m": float(max(1e-6, payload.get("dy_m", 1.0))),
            "z_grid": z_arr.tolist(),
            "z_ref_m": z_ref,
            "gradient_step_m": float(max(0.5, payload.get("gradient_step_m", 2.0))),
        }

    samples_local = _normalize_terrain_samples_local(payload.get("samples_local"))
    if not samples_local:
        samples_world = payload.get("samples", payload.get("samples_lonlatz"))
        if isinstance(samples_world, list):
            world_pts: list[list[float]] = []
            elevations: list[float] = []
            for sample in samples_world:
                if isinstance(sample, dict):
                    lon = sample.get("lon")
                    lat = sample.get("lat")
                    z = sample.get("elevation_m", sample.get("z_m"))
                    if lon is None or lat is None or z is None:
                        continue
                    world_pts.append([float(lon), float(lat)])
                    elevations.append(float(z))
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    world_pts.append([float(sample[0]), float(sample[1])])
                    elevations.append(float(sample[2]))
            if world_pts:
                local = _world_to_local(world_pts, frame)
                samples_local = [[float(pt[0]), float(pt[1]), elevations[i]] for i, pt in enumerate(local)]
    if len(samples_local) >= 3:
        z_ref = float(payload.get("z_ref_m", float(np.mean(np.asarray(samples_local, dtype=np.float64)[:, 2]))))
        return {
            "type": "samples",
            "source": f"terrain_file:{Path(source).name}",
            "samples_local": samples_local,
            "z_ref_m": z_ref,
            "sample_count": int(len(samples_local)),
        }
    return None


def _build_terrain_model(
    terrain_follow_enabled: bool,
    terrain_follow_mode: str,
    local_polygon_closed: list[list[float]],
    raw_vertices: list[list[float]],
    frame: AssetReferenceFrame,
    terrain_source_path: str = "",
    terrain_samples_lonlatz: Iterable[Iterable[float]] | None = None,
    terrain_model_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not terrain_follow_enabled:
        return {"type": "flat", "source": "disabled", "z_ref_m": 0.0}
    follow_mode = str(terrain_follow_mode or "agl").strip().lower()
    if follow_mode not in {"agl", "amsl"}:
        follow_mode = "agl"

    if isinstance(terrain_model_override, dict) and terrain_model_override:
        model = dict(terrain_model_override)
        model.setdefault("type", "samples")
        model.setdefault("source", "override")
        model.setdefault("z_ref_m", 0.0)
        model["follow_mode"] = follow_mode
        if str(model.get("type", "")).lower() == "samples":
            samples_local = _normalize_terrain_samples_local(model.get("samples_local"))
            if samples_local:
                model["samples_local"] = samples_local
                model["sample_count"] = int(len(samples_local))
                if "z_ref_m" not in model:
                    arr = np.asarray(samples_local, dtype=np.float64)
                    model["z_ref_m"] = float(np.mean(arr[:, 2]))
        return model

    model_from_file = _load_terrain_model_from_path(terrain_source_path, frame=frame)
    if isinstance(model_from_file, dict):
        model_from_file["follow_mode"] = follow_mode
        return model_from_file

    if terrain_samples_lonlatz is not None:
        world_pts: list[list[float]] = []
        elevations: list[float] = []
        for sample in terrain_samples_lonlatz:
            row = list(sample)
            if len(row) < 3:
                continue
            world_pts.append([float(row[0]), float(row[1])])
            elevations.append(float(row[2]))
        if len(world_pts) >= 3:
            local = _world_to_local(world_pts, frame)
            samples_local = [[float(pt[0]), float(pt[1]), elevations[i]] for i, pt in enumerate(local)]
            return {
                "type": "samples",
                "source": "explicit_world_samples",
                "samples_local": samples_local,
                "z_ref_m": float(np.mean(np.asarray(elevations, dtype=np.float64))),
                "sample_count": int(len(samples_local)),
                "follow_mode": follow_mode,
            }

    elevations_open: list[float] = []
    for row in raw_vertices:
        if len(row) < 3:
            elevations_open = []
            break
        elevations_open.append(float(row[2]))
    if len(elevations_open) >= 3:
        if len(local_polygon_closed) == len(elevations_open):
            elevations_closed = elevations_open
        else:
            elevations_closed = elevations_open + [elevations_open[0]]
        model = _fit_terrain_plane(local_polygon_closed, elevations_closed)
        model["follow_mode"] = follow_mode
        return model

    return {"type": "flat", "source": "missing_terrain_source", "z_ref_m": 0.0, "follow_mode": follow_mode}


def _wind_adjusted_speed(
    requested_speed_m_s: float,
    heading_deg: float,
    wind_speed_m_s: float,
    wind_direction_deg: float,
    wind_gust_m_s: float,
) -> dict[str, float]:
    requested = float(max(0.5, requested_speed_m_s))
    wind_speed = float(max(0.0, wind_speed_m_s))
    gust = float(max(0.0, wind_gust_m_s))
    if wind_speed <= 0.01 and gust <= 0.01:
        return {
            "requested_speed_m_s": requested,
            "effective_speed_m_s": requested,
            "crosswind_m_s": 0.0,
            "headwind_m_s": 0.0,
            "penalty_pct": 0.0,
        }

    rel = radians(_wrap_deg(float(wind_direction_deg) - float(heading_deg)))
    cross = abs(sin(rel)) * wind_speed
    head = max(0.0, cos(rel)) * wind_speed
    gust_factor = max(0.0, gust - wind_speed)
    penalty = min(0.65, 0.018 * cross + 0.010 * head + 0.012 * gust_factor)
    effective = max(0.8, requested * (1.0 - penalty))
    return {
        "requested_speed_m_s": requested,
        "effective_speed_m_s": float(effective),
        "crosswind_m_s": float(cross),
        "headwind_m_s": float(head),
        "penalty_pct": float(100.0 * penalty),
    }


def _apply_terrain_normal_attitude(
    x_m: float,
    y_m: float,
    yaw_deg: float,
    gimbal_pitch_deg: float,
    terrain_model: dict[str, Any] | None,
    enabled: bool,
    gain: float = 1.0,
    yaw_align: bool = False,
) -> tuple[float, float]:
    if not enabled or not isinstance(terrain_model, dict):
        return float(yaw_deg), float(gimbal_pitch_deg)
    dz_dx, dz_dy = _terrain_gradient(np.asarray([x_m, y_m], dtype=np.float64), terrain_model)
    if abs(dz_dx) < 1e-9 and abs(dz_dy) < 1e-9:
        return float(yaw_deg), float(gimbal_pitch_deg)

    yaw = float(yaw_deg)
    if yaw_align:
        yaw = _wrap_deg(float(np.degrees(np.arctan2(dz_dy, dz_dx))))

    yaw_rad = radians(yaw)
    slope_along_view = float(dz_dx * cos(yaw_rad) + dz_dy * sin(yaw_rad))
    pitch_comp_deg = float(np.degrees(np.arctan(slope_along_view)))
    pitch = float(np.clip(float(gimbal_pitch_deg) + float(gain) * pitch_comp_deg, -120.0, 30.0))
    return yaw, pitch


def _distance_m(waypoints_lonlat: list[list[float]]) -> float:
    if len(waypoints_lonlat) < 2:
        return 0.0
    arr = np.asarray(waypoints_lonlat, dtype=np.float64)
    lat = np.radians(arr[:, 1])
    lon = np.radians(arr[:, 0])
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return float((EARTH_RADIUS_M * c).sum())


def _distance_xy(points_xy: Iterable[Iterable[float]]) -> float:
    pts = np.asarray(list(points_xy), dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def _line_buffer_polygon(line_lonlat: list[list[float]], buffer_m: float = 25.0) -> list[list[float]]:
    line = _ensure_line(line_lonlat)
    arr = np.asarray(line, dtype=np.float64)
    lon0 = float(arr[:, 0].mean())
    lat0 = float(arr[:, 1].mean())
    xy = _lonlat_to_xy(arr, lon0=lon0, lat0=lat0)
    pad = max(5.0, float(buffer_m))
    min_x, min_y = float(np.min(xy[:, 0])) - pad, float(np.min(xy[:, 1])) - pad
    max_x, max_y = float(np.max(xy[:, 0])) + pad, float(np.max(xy[:, 1])) + pad
    rect_xy = np.asarray(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ],
        dtype=np.float64,
    )
    rect_ll = _xy_to_lonlat(rect_xy, lon0=lon0, lat0=lat0)
    return [[float(p[0]), float(p[1])] for p in rect_ll.tolist()]


def _tower_buffer_polygon(center_lonlat: Iterable[float], radius_m: float = 35.0, points: int = 24) -> list[list[float]]:
    center = list(center_lonlat)
    if len(center) < 2:
        raise ValueError("Tower center requires [lon, lat].")
    lon0 = float(center[0])
    lat0 = float(center[1])
    radius = max(5.0, float(radius_m))
    count = max(8, int(points))
    angles = np.linspace(0.0, 2.0 * pi, count, endpoint=False, dtype=np.float64)
    circle_xy = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    circle_ll = _xy_to_lonlat(circle_xy, lon0=lon0, lat0=lat0)
    polygon = [[float(p[0]), float(p[1])] for p in circle_ll.tolist()]
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(list(polygon[0]))
    return polygon


def _geojson_from_plan(
    polygon: list[list[float]],
    waypoints: list[list[float]],
    world_poses: list[dict] | None = None,
    no_fly_polygons: list[list[list[float]]] | None = None,
) -> dict:
    features = [
        {
            "type": "Feature",
            "properties": {"role": "survey_polygon"},
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
        }
    ]
    for nf in no_fly_polygons or []:
        if isinstance(nf, list) and len(nf) >= 3:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"role": "no_fly_polygon"},
                    "geometry": {"type": "Polygon", "coordinates": [nf]},
                }
            )
    if waypoints:
        features.append(
            {
                "type": "Feature",
                "properties": {"role": "flight_path"},
                "geometry": {"type": "LineString", "coordinates": waypoints},
            }
        )
        for i, pt in enumerate(waypoints):
            props = {"role": "waypoint", "index": i}
            if world_poses and i < len(world_poses):
                props.update(
                    {
                        "primitive": world_poses[i].get("primitive", ""),
                        "yaw_deg": world_poses[i].get("yaw_deg"),
                        "gimbal_pitch_deg": world_poses[i].get("gimbal_pitch_deg"),
                        "trigger": bool(world_poses[i].get("trigger", True)),
                        "dwell_s": float(world_poses[i].get("dwell_s", 0.0)),
                        "camera_yaw_locked": bool(world_poses[i].get("camera_yaw_locked", False)),
                    }
                )
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": pt},
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _extract_waypoints_from_geojson(geojson: dict) -> list[list[float]]:
    if not isinstance(geojson, dict):
        return []
    features = geojson.get("features", [])
    for ft in features:
        geom = ft.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if coords:
                rows: list[list[float]] = []
                for p in coords:
                    if not isinstance(p, list) or len(p) < 2:
                        continue
                    row = [float(p[0]), float(p[1])]
                    if len(p) >= 3:
                        row.append(float(p[2]))
                    rows.append(row)
                if rows:
                    return rows

    pts = []
    for ft in features:
        geom = ft.get("geometry", {})
        if geom.get("type") == "Point":
            c = geom.get("coordinates", [])
            if len(c) >= 2:
                idx = ft.get("properties", {}).get("index")
                row = [float(c[0]), float(c[1])]
                if len(c) >= 3:
                    row.append(float(c[2]))
                pts.append((idx if isinstance(idx, int) else 10**9, row))
    if pts:
        pts.sort(key=lambda x: x[0])
        return [p for _, p in pts]
    return []


def _extract_world_poses_from_geojson(geojson: dict) -> list[dict[str, Any]]:
    if not isinstance(geojson, dict):
        return []
    poses: list[tuple[int, dict[str, Any]]] = []
    for ft in geojson.get("features", []):
        if not isinstance(ft, dict):
            continue
        geom = ft.get("geometry", {})
        props = ft.get("properties", {}) if isinstance(ft.get("properties"), dict) else {}
        if geom.get("type") != "Point":
            continue
        if str(props.get("role", "")) != "waypoint":
            continue
        coords = geom.get("coordinates", [])
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        idx = int(props.get("index", len(poses)))
        alt = float(coords[2]) if len(coords) >= 3 else float(props.get("alt_m", 0.0))
        poses.append(
            (
                idx,
                {
                    "lon": float(coords[0]),
                    "lat": float(coords[1]),
                    "alt_m": alt,
                    "yaw_deg": float(props.get("yaw_deg", 0.0)),
                    "gimbal_pitch_deg": float(props.get("gimbal_pitch_deg", -90.0)),
                    "primitive": str(props.get("primitive", "linked_segment")),
                    "trigger": bool(props.get("trigger", True)),
                    "dwell_s": float(props.get("dwell_s", 0.0)),
                    "camera_yaw_locked": bool(props.get("camera_yaw_locked", False)),
                },
            )
        )
    poses.sort(key=lambda row: row[0])
    return [p for _, p in poses]


def _asset_frame_from_dict(data: dict[str, Any]) -> AssetReferenceFrame:
    return AssetReferenceFrame(
        asset_id=str(data.get("asset_id") or "asset"),
        origin_lon=float(data.get("origin_lon", 0.0)),
        origin_lat=float(data.get("origin_lat", 0.0)),
        yaw_deg=float(data.get("yaw_deg", 0.0)),
        coordinate_source=str(data.get("coordinate_source") or "survey_polygon"),
        reference_note=str(data.get("reference_note") or ""),
    )


def _constraints_from_dict(
    data: dict[str, Any],
    fallback_polygon: list[list[float]],
    default_altitude_m: float,
) -> MissionConstraints:
    geofence_input = data.get("geofence") if isinstance(data, dict) else None
    geofence = _ensure_closed(geofence_input) if geofence_input else _ensure_closed(fallback_polygon)

    min_alt = float(
        data.get("min_altitude_m")
        if "min_altitude_m" in data
        else data.get("min_alt_m", max(5.0, default_altitude_m * 0.5))
    )
    max_alt = float(
        data.get("max_altitude_m")
        if "max_altitude_m" in data
        else data.get("max_alt_m", max(min_alt + 5.0, default_altitude_m * 1.7))
    )
    if max_alt < min_alt + 1.0:
        max_alt = min_alt + 1.0

    standoff = float(data.get("standoff_m", 8.0))
    raw_no_fly = (
        data.get("no_fly_polygons")
        if "no_fly_polygons" in data
        else data.get("no_fly_zones", data.get("obstacles", []))
    )
    no_fly_polygons: list[list[list[float]]] = []
    if isinstance(raw_no_fly, list):
        for poly in raw_no_fly:
            if not isinstance(poly, list):
                continue
            try:
                no_fly_polygons.append(_ensure_closed(poly))
            except Exception:
                continue
    rth_alt = float(data.get("rth_altitude_m", max(max_alt + 10.0, default_altitude_m + 20.0)))
    return MissionConstraints(
        geofence=geofence,
        min_altitude_m=max(5.0, min_alt),
        max_altitude_m=max(6.0, max_alt),
        standoff_m=max(0.0, standoff),
        rth_altitude_m=max(10.0, rth_alt),
        no_fly_polygons=no_fly_polygons,
        rth_action=str(data.get("rth_action") or "return_home"),
        obstacle_avoidance_profile=str(data.get("obstacle_avoidance_profile") or "balanced"),
    )


def _coverage_from_dict(data: dict[str, Any]) -> CoverageExpectation:
    return CoverageExpectation(
        front_overlap_pct=float(data.get("front_overlap_pct", 80.0)),
        side_overlap_pct=float(data.get("side_overlap_pct", 70.0)),
        minimum_coverage_pct=float(data.get("minimum_coverage_pct", 95.0)),
        required_viewpoints=int(data.get("required_viewpoints", 0)),
    )


def _primitive_from_dict(data: dict[str, Any]) -> MissionPrimitive:
    params = data.get("params", {})
    return MissionPrimitive(kind=str(data.get("kind") or "grid"), params=params if isinstance(params, dict) else {})


def _flight_recipe_from_dict(payload: dict[str, Any]) -> FlightRecipe:
    frame = _asset_frame_from_dict(payload.get("asset_frame", {}))
    constraints = _constraints_from_dict(
        payload.get("constraints", {}),
        fallback_polygon=payload.get("constraints", {}).get("geofence")
        or [[frame.origin_lon, frame.origin_lat], [frame.origin_lon + 1e-6, frame.origin_lat], [frame.origin_lon, frame.origin_lat + 1e-6]],
        default_altitude_m=60.0,
    )
    coverage = _coverage_from_dict(payload.get("coverage", {}))
    primitives_data = payload.get("primitives", [])
    primitives = []
    for item in primitives_data if isinstance(primitives_data, list) else []:
        if isinstance(item, dict):
            primitives.append(_primitive_from_dict(item))
    if not primitives:
        primitives = [MissionPrimitive(kind="grid", params={})]

    return FlightRecipe(
        recipe_id=str(payload.get("recipe_id") or "recipe-unknown"),
        version=int(payload.get("version", 1)),
        template=_normalize_template(str(payload.get("template", "grid"))),
        asset_frame=frame,
        primitives=primitives,
        constraints=constraints,
        coverage=coverage,
        created_at_utc=str(payload.get("created_at_utc") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )

class MissionPlanner:
    """Recipe-driven mission planner with repeatable asset-frame capture."""

    def __init__(self):
        pass

    def line_buffer_geofence(
        self,
        line_lonlat: Iterable[Iterable[float]],
        buffer_m: float = 25.0,
    ) -> list[list[float]]:
        return _line_buffer_polygon(_ensure_line(line_lonlat), buffer_m=buffer_m)

    def tower_buffer_geofence(
        self,
        center_lonlat: Iterable[float],
        flight_radius_m: float,
        padding_m: float = 15.0,
    ) -> list[list[float]]:
        radius = max(5.0, float(flight_radius_m) + max(0.0, float(padding_m)))
        return _tower_buffer_polygon(center_lonlat, radius_m=radius, points=24)

    def derive_asset_frame(
        self,
        polygon_lonlat: Iterable[Iterable[float]],
        asset_id: str | None = None,
        coordinate_source: str = "survey_polygon",
    ) -> AssetReferenceFrame:
        polygon = _ensure_closed(polygon_lonlat)
        coords = np.asarray(polygon[:-1], dtype=np.float64)
        lon0 = float(coords[:, 0].mean())
        lat0 = float(coords[:, 1].mean())

        yaw = 0.0
        if len(polygon) >= 3:
            edge = np.asarray([polygon[0], polygon[1]], dtype=np.float64)
            edge_xy = _lonlat_to_xy(edge, lon0=lon0, lat0=lat0)
            vec = edge_xy[1] - edge_xy[0]
            if float(np.linalg.norm(vec)) > 1e-6:
                yaw = float(np.degrees(np.arctan2(vec[1], vec[0])))

        if not asset_id:
            payload = json.dumps([[round(p[0], 7), round(p[1], 7)] for p in polygon], sort_keys=True)
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
            asset_id = f"asset-{digest}"

        return AssetReferenceFrame(
            asset_id=asset_id,
            origin_lon=lon0,
            origin_lat=lat0,
            yaw_deg=yaw,
            coordinate_source=coordinate_source,
        )

    def build_flight_recipe(
        self,
        polygon_lonlat: Iterable[Iterable[float]] | None,
        altitude_m: float = 60.0,
        front_overlap_pct: float = 80.0,
        side_overlap_pct: float = 70.0,
        mode: str = "grid",
        camera: str = "mavic2pro",
        flight_direction_deg: float = 0.0,
        camera_direction_deg: float | None = None,
        gimbal_tilt_deg: float = -90.0,
        inspection_dwell_s: float = 1.5,
        facade_top_altitude_m: float | None = None,
        facade_bottom_altitude_m: float | None = None,
        facade_standoff_m: float | None = None,
        facade_rotate_points_180: bool = False,
        facade_capture_profile: str = "custom",
        linear_path_lonlat: Iterable[Iterable[float]] | None = None,
        linear_segmentation_enabled: bool = True,
        linear_max_segment_length_m: float = 1500.0,
        lateral_target_path_lonlat: Iterable[Iterable[float]] | None = None,
        lateral_standoff_m: float = 10.0,
        lateral_target_side: str = "right",
        waypoint_path_lonlat: Iterable[Iterable[float]] | None = None,
        waypoint_heading_mode: str = "tangent",
        waypoint_fixed_yaw_deg: float = 0.0,
        waypoint_poi_lonlat: Iterable[float] | None = None,
        waypoint_enable_smoothing: bool = False,
        waypoint_turn_radius_m: float = 6.0,
        waypoint_capture_enabled: bool = True,
        orbit_center_lonlat: Iterable[float] | None = None,
        orbit_radius_m: float | None = None,
        orbit_level_count: int | None = None,
        orbit_vertical_step_m: float | None = None,
        orbit_poi_yaw_lock: bool = True,
        orbit_poi_lonlat: Iterable[float] | None = None,
        panorama_center_lonlat: Iterable[float] | None = None,
        panorama_overlap_pct: float = 35.0,
        panorama_multi_row_enabled: bool = False,
        panorama_row_count: int = 1,
        panorama_pitch_step_deg: float = 12.0,
        bubble_center_lonlat: Iterable[float] | None = None,
        bubble_overlap_pct: float = 35.0,
        bubble_pitch_step_deg: float = 12.0,
        bubble_top_pitch_deg: float = 20.0,
        bubble_bottom_pitch_deg: float = -90.0,
        tower_center_lonlat: Iterable[float] | None = None,
        tower_top_altitude_m: float | None = None,
        tower_bottom_altitude_m: float | None = None,
        tower_object_radius_m: float = 2.0,
        tower_flight_radius_m: float | None = None,
        tower_resume_enabled: bool = True,
        solar_row_angle_deg: float | None = None,
        solar_sensor_profile: str = "rgb",
        solar_orientation_mode: str = "row_aligned",
        solar_rows_lonlat: Iterable[Iterable[Iterable[float]]] | None = None,
        magnetic_tie_line_spacing_m: float = 50.0,
        magnetic_smoothing_radius_m: float = 8.0,
        ground_offset_m: float = 0.0,
        terrain_follow_enabled: bool = False,
        terrain_source_path: str = "",
        terrain_samples_lonlatz: Iterable[Iterable[float]] | None = None,
        terrain_model_override: dict[str, Any] | None = None,
        terrain_follow_mode: str = "agl",
        terrain_normal_camera_enabled: bool = False,
        terrain_normal_gain: float = 1.0,
        terrain_normal_yaw_align: bool = False,
        wind_speed_m_s: float = 0.0,
        wind_direction_deg: float = 0.0,
        wind_gust_m_s: float = 0.0,
        facade_curvature_alignment: bool = False,
        facade_curve_path_lonlat: Iterable[Iterable[float]] | None = None,
        constraints: MissionConstraints | dict | None = None,
        asset_frame: AssetReferenceFrame | dict | None = None,
        recipe_version: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> FlightRecipe:
        raw_vertices = [list(p) for p in (polygon_lonlat or [])]
        linear_path_world: list[list[float]] | None = None
        lateral_target_world: list[list[float]] | None = None
        waypoint_path_world: list[list[float]] | None = None
        waypoint_poi_world: list[float] | None = None
        orbit_center_world: list[float] | None = None
        orbit_poi_world: list[float] | None = None
        panorama_center_world: list[float] | None = None
        bubble_center_world: list[float] | None = None
        tower_center_world: list[float] | None = None
        solar_rows_world: list[list[list[float]]] = []
        facade_curve_world: list[list[float]] | None = None
        altitude_m = float(max(5.0, altitude_m))
        requested_front_overlap_pct = float(np.clip(front_overlap_pct, 20.0, 95.0))
        requested_side_overlap_pct = float(np.clip(side_overlap_pct, 20.0, 95.0))
        flight_direction_deg = _wrap_deg(float(flight_direction_deg))
        camera_direction_locked = camera_direction_deg is not None
        camera_direction_value = _wrap_deg(float(camera_direction_deg)) if camera_direction_deg is not None else flight_direction_deg
        gimbal_tilt_deg = float(np.clip(gimbal_tilt_deg, -120.0, 30.0))
        inspection_dwell_s = float(np.clip(inspection_dwell_s, 0.0, 30.0))
        ground_offset_m = float(np.clip(ground_offset_m, -100.0, 300.0))
        terrain_follow_enabled = bool(terrain_follow_enabled)
        terrain_source_path = str(terrain_source_path or "").strip()
        terrain_follow_mode = str(terrain_follow_mode or "agl").strip().lower()
        if terrain_follow_mode not in {"agl", "amsl"}:
            terrain_follow_mode = "agl"
        terrain_normal_camera_enabled = bool(terrain_normal_camera_enabled)
        terrain_normal_gain = float(np.clip(float(terrain_normal_gain), 0.0, 3.0))
        terrain_normal_yaw_align = bool(terrain_normal_yaw_align)
        wind_speed_m_s = float(max(0.0, wind_speed_m_s))
        wind_direction_deg = _wrap_deg(float(wind_direction_deg))
        wind_gust_m_s = float(max(0.0, wind_gust_m_s))
        facade_curvature_alignment = bool(facade_curvature_alignment)
        camera = str(camera or "custom").lower()
        template = _normalize_template(mode)
        facade_rotate_points_180 = bool(facade_rotate_points_180)
        facade_capture_profile = _normalize_facade_capture_profile(facade_capture_profile)
        linear_segmentation_enabled = bool(linear_segmentation_enabled)
        linear_max_segment_length_m = float(max(100.0, linear_max_segment_length_m))
        lateral_standoff_m = float(max(0.5, lateral_standoff_m))
        lateral_target_side = str(lateral_target_side or "right").strip().lower()
        if lateral_target_side not in {"left", "right"}:
            lateral_target_side = "right"
        waypoint_heading_mode = str(waypoint_heading_mode or "tangent").strip().lower()
        if waypoint_heading_mode not in {"tangent", "fixed", "poi"}:
            waypoint_heading_mode = "tangent"
        waypoint_fixed_yaw_deg = _wrap_deg(float(waypoint_fixed_yaw_deg))
        waypoint_enable_smoothing = bool(waypoint_enable_smoothing)
        waypoint_turn_radius_m = float(max(0.0, waypoint_turn_radius_m))
        waypoint_capture_enabled = bool(waypoint_capture_enabled)
        orbit_radius_m = float(max(1.0, orbit_radius_m)) if orbit_radius_m is not None else None
        orbit_level_count = int(max(1, orbit_level_count)) if orbit_level_count is not None else None
        orbit_vertical_step_m = float(max(0.5, orbit_vertical_step_m)) if orbit_vertical_step_m is not None else None
        orbit_poi_yaw_lock = bool(orbit_poi_yaw_lock)
        panorama_overlap_pct = float(np.clip(panorama_overlap_pct, 5.0, 90.0))
        panorama_multi_row_enabled = bool(panorama_multi_row_enabled)
        panorama_row_count = int(max(1, panorama_row_count))
        panorama_pitch_step_deg = float(np.clip(abs(panorama_pitch_step_deg), 1.0, 45.0))
        bubble_overlap_pct = float(np.clip(bubble_overlap_pct, 5.0, 90.0))
        bubble_pitch_step_deg = float(np.clip(abs(bubble_pitch_step_deg), 1.0, 45.0))
        bubble_top_pitch_deg = float(np.clip(bubble_top_pitch_deg, -120.0, 30.0))
        bubble_bottom_pitch_deg = float(np.clip(bubble_bottom_pitch_deg, -120.0, 30.0))
        if bubble_top_pitch_deg < bubble_bottom_pitch_deg:
            bubble_top_pitch_deg, bubble_bottom_pitch_deg = bubble_bottom_pitch_deg, bubble_top_pitch_deg
        tower_object_radius_m = float(max(0.5, tower_object_radius_m))
        tower_resume_enabled = bool(tower_resume_enabled)
        solar_sensor_profile = str(solar_sensor_profile or "rgb").strip().lower()
        if solar_sensor_profile not in {"rgb", "thermal"}:
            solar_sensor_profile = "rgb"
        solar_orientation_mode = str(solar_orientation_mode or "row_aligned").strip().lower()
        if solar_orientation_mode not in {"row_aligned", "path_aligned"}:
            solar_orientation_mode = "row_aligned"
        magnetic_tie_line_spacing_m = float(max(5.0, magnetic_tie_line_spacing_m))
        magnetic_smoothing_radius_m = float(max(0.0, magnetic_smoothing_radius_m))

        if facade_curve_path_lonlat is not None:
            try:
                facade_curve_world = _ensure_line(facade_curve_path_lonlat)
            except Exception:
                facade_curve_world = None

        if template == "linear_inspection":
            if linear_path_lonlat is not None:
                linear_path_world = _ensure_line(linear_path_lonlat)
            elif len(raw_vertices) >= 2:
                linear_path_world = _ensure_line(raw_vertices)
            else:
                raise ValueError("Linear inspection requires a line with at least two points.")
            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                polygon = _line_buffer_polygon(linear_path_world, buffer_m=25.0)
        elif template == "lateral_capture":
            if lateral_target_path_lonlat is not None:
                lateral_target_world = _ensure_line(lateral_target_path_lonlat)
            elif linear_path_lonlat is not None:
                lateral_target_world = _ensure_line(linear_path_lonlat)
            elif len(raw_vertices) >= 2:
                lateral_target_world = _ensure_line(raw_vertices)
            else:
                raise ValueError("Lateral capture requires a target line with at least two points.")
            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                polygon = _line_buffer_polygon(
                    lateral_target_world,
                    buffer_m=max(15.0, lateral_standoff_m + 10.0),
                )
        elif template == "waypoints":
            if waypoint_path_lonlat is not None:
                waypoint_path_world = _ensure_line(waypoint_path_lonlat)
            elif linear_path_lonlat is not None:
                waypoint_path_world = _ensure_line(linear_path_lonlat)
            elif len(raw_vertices) >= 2:
                waypoint_path_world = _ensure_line(raw_vertices)
            else:
                raise ValueError("Advanced waypoints requires a waypoint polyline with at least two points.")
            if waypoint_poi_lonlat is not None:
                poi = list(waypoint_poi_lonlat)
                if len(poi) >= 2:
                    waypoint_poi_world = [float(poi[0]), float(poi[1])]
            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                polygon = _line_buffer_polygon(waypoint_path_world, buffer_m=25.0)
        elif template == "orbit":
            if orbit_center_lonlat is not None:
                center = list(orbit_center_lonlat)
                if len(center) < 2:
                    raise ValueError("Orbit center requires [lon, lat].")
                orbit_center_world = [float(center[0]), float(center[1])]
            elif len(raw_vertices) >= 1:
                arr = np.asarray(raw_vertices, dtype=np.float64)
                orbit_center_world = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
            else:
                raise ValueError("Orbit mission requires a center point or geometry.")

            if orbit_poi_lonlat is not None:
                poi = list(orbit_poi_lonlat)
                if len(poi) >= 2:
                    orbit_poi_world = [float(poi[0]), float(poi[1])]

            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                fallback_radius = float(orbit_radius_m) if orbit_radius_m is not None else 25.0
                polygon = _tower_buffer_polygon(orbit_center_world, radius_m=fallback_radius + 15.0, points=24)
        elif template == "panorama":
            if panorama_center_lonlat is not None:
                center = list(panorama_center_lonlat)
                if len(center) < 2:
                    raise ValueError("Panorama center requires [lon, lat].")
                panorama_center_world = [float(center[0]), float(center[1])]
            elif len(raw_vertices) >= 1:
                arr = np.asarray(raw_vertices, dtype=np.float64)
                panorama_center_world = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
            else:
                raise ValueError("Panorama mission requires a center point or geometry.")

            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                polygon = _tower_buffer_polygon(panorama_center_world, radius_m=20.0, points=20)
        elif template == "bubble_360":
            if bubble_center_lonlat is not None:
                center = list(bubble_center_lonlat)
                if len(center) < 2:
                    raise ValueError("360 bubble center requires [lon, lat].")
                bubble_center_world = [float(center[0]), float(center[1])]
            elif len(raw_vertices) >= 1:
                arr = np.asarray(raw_vertices, dtype=np.float64)
                bubble_center_world = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
            else:
                raise ValueError("360 bubble mission requires a center point or geometry.")

            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                polygon = _tower_buffer_polygon(bubble_center_world, radius_m=20.0, points=20)
        elif template == "tower_mapping":
            if tower_center_lonlat is not None:
                center = list(tower_center_lonlat)
                if len(center) < 2:
                    raise ValueError("Tower center requires [lon, lat].")
                tower_center_world = [float(center[0]), float(center[1])]
            elif len(raw_vertices) >= 1:
                arr = np.asarray(raw_vertices, dtype=np.float64)
                tower_center_world = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
            else:
                raise ValueError("Tower mapping requires a center point or geometry.")

            if len(raw_vertices) >= 3:
                polygon = _ensure_closed(raw_vertices)
            else:
                fallback_radius = float(tower_flight_radius_m) if tower_flight_radius_m is not None else max(18.0, tower_object_radius_m + 10.0)
                polygon = _tower_buffer_polygon(tower_center_world, radius_m=fallback_radius + 15.0, points=24)
        elif template in {"solar_inspection", "magnetic_mapping"}:
            polygon = _ensure_closed(raw_vertices)
        else:
            polygon = _ensure_closed(raw_vertices)

        if template in {"facade", "facade_mapping"} and facade_curve_world is None:
            if linear_path_lonlat is not None:
                try:
                    facade_curve_world = _ensure_line(linear_path_lonlat)
                except Exception:
                    facade_curve_world = None

        if template == "solar_inspection":
            if solar_rows_lonlat is not None:
                for row in solar_rows_lonlat:
                    try:
                        solar_rows_world.append(_ensure_line(row))
                    except Exception:
                        continue
            elif linear_path_lonlat is not None:
                try:
                    solar_rows_world = [_ensure_line(linear_path_lonlat)]
                except Exception:
                    solar_rows_world = []

        cross_angle_deg = DOUBLE_GRID_DEFAULT_CROSS_ANGLE_DEG
        if isinstance(metadata, dict) and "double_grid_cross_angle_deg" in metadata:
            cross_angle_deg = float(np.clip(float(metadata.get("double_grid_cross_angle_deg", cross_angle_deg)), 30.0, 150.0))

        front_overlap_pct = requested_front_overlap_pct
        side_overlap_pct = requested_side_overlap_pct
        overlap_floor_applied = False
        if template == "double_grid":
            front_overlap_pct = max(front_overlap_pct, DOUBLE_GRID_MIN_FRONT_OVERLAP_PCT)
            side_overlap_pct = max(side_overlap_pct, DOUBLE_GRID_MIN_SIDE_OVERLAP_PCT)
            overlap_floor_applied = (
                front_overlap_pct > requested_front_overlap_pct or side_overlap_pct > requested_side_overlap_pct
            )
        if template == "facade_mapping":
            front_overlap_pct = max(front_overlap_pct, FACADE_MAPPING_MIN_FRONT_OVERLAP_PCT)
            side_overlap_pct = max(side_overlap_pct, FACADE_MAPPING_MIN_SIDE_OVERLAP_PCT)
            overlap_floor_applied = (
                front_overlap_pct > requested_front_overlap_pct or side_overlap_pct > requested_side_overlap_pct
            )
            if facade_capture_profile == "normal":
                gimbal_tilt_deg = FACADE_MAPPING_NORMAL_GIMBAL_DEG
            elif facade_capture_profile == "oblique":
                gimbal_tilt_deg = FACADE_MAPPING_OBLIQUE_GIMBAL_DEG

        if asset_frame is None:
            if template == "tower_mapping" and tower_center_world is not None:
                payload = json.dumps([round(tower_center_world[0], 7), round(tower_center_world[1], 7)], sort_keys=True)
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
                frame = AssetReferenceFrame(
                    asset_id=f"tower-{digest}",
                    origin_lon=float(tower_center_world[0]),
                    origin_lat=float(tower_center_world[1]),
                    yaw_deg=0.0,
                    coordinate_source="tower_center",
                )
            elif template == "orbit" and orbit_center_world is not None:
                payload = json.dumps([round(orbit_center_world[0], 7), round(orbit_center_world[1], 7)], sort_keys=True)
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
                frame = AssetReferenceFrame(
                    asset_id=f"orbit-{digest}",
                    origin_lon=float(orbit_center_world[0]),
                    origin_lat=float(orbit_center_world[1]),
                    yaw_deg=0.0,
                    coordinate_source="orbit_center",
                )
            elif template == "panorama" and panorama_center_world is not None:
                payload = json.dumps([round(panorama_center_world[0], 7), round(panorama_center_world[1], 7)], sort_keys=True)
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
                frame = AssetReferenceFrame(
                    asset_id=f"panorama-{digest}",
                    origin_lon=float(panorama_center_world[0]),
                    origin_lat=float(panorama_center_world[1]),
                    yaw_deg=0.0,
                    coordinate_source="panorama_center",
                )
            elif template == "bubble_360" and bubble_center_world is not None:
                payload = json.dumps([round(bubble_center_world[0], 7), round(bubble_center_world[1], 7)], sort_keys=True)
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
                frame = AssetReferenceFrame(
                    asset_id=f"bubble-{digest}",
                    origin_lon=float(bubble_center_world[0]),
                    origin_lat=float(bubble_center_world[1]),
                    yaw_deg=0.0,
                    coordinate_source="bubble_center",
                )
            else:
                frame = self.derive_asset_frame(polygon)
        elif isinstance(asset_frame, AssetReferenceFrame):
            frame = asset_frame
        else:
            frame = _asset_frame_from_dict(dict(asset_frame))

        constraints_obj = self._coerce_constraints(constraints, polygon=polygon, default_altitude_m=altitude_m)

        if facade_bottom_altitude_m is None:
            bottom_altitude_m = max(constraints_obj.min_altitude_m, altitude_m - 20.0)
        else:
            bottom_altitude_m = float(np.clip(float(facade_bottom_altitude_m), constraints_obj.min_altitude_m, constraints_obj.max_altitude_m))
        if facade_top_altitude_m is None:
            top_altitude_m = min(constraints_obj.max_altitude_m, altitude_m + 20.0)
        else:
            top_altitude_m = float(np.clip(float(facade_top_altitude_m), constraints_obj.min_altitude_m, constraints_obj.max_altitude_m))
        if top_altitude_m < bottom_altitude_m + 0.5:
            top_altitude_m = min(constraints_obj.max_altitude_m, bottom_altitude_m + 0.5)
            if top_altitude_m < bottom_altitude_m + 0.5:
                bottom_altitude_m = max(constraints_obj.min_altitude_m, top_altitude_m - 0.5)
        if facade_standoff_m is None:
            facade_standoff = float(max(0.5, constraints_obj.standoff_m))
        else:
            facade_standoff = float(max(0.5, facade_standoff_m))

        if tower_bottom_altitude_m is None:
            tower_bottom_alt = max(constraints_obj.min_altitude_m, altitude_m - 15.0)
        else:
            tower_bottom_alt = float(np.clip(float(tower_bottom_altitude_m), constraints_obj.min_altitude_m, constraints_obj.max_altitude_m))
        if tower_top_altitude_m is None:
            tower_top_alt = min(constraints_obj.max_altitude_m, altitude_m + 25.0)
        else:
            tower_top_alt = float(np.clip(float(tower_top_altitude_m), constraints_obj.min_altitude_m, constraints_obj.max_altitude_m))
        if tower_top_alt < tower_bottom_alt + 0.5:
            tower_top_alt = min(constraints_obj.max_altitude_m, tower_bottom_alt + 0.5)
            if tower_top_alt < tower_bottom_alt + 0.5:
                tower_bottom_alt = max(constraints_obj.min_altitude_m, tower_top_alt - 0.5)

        if tower_flight_radius_m is None:
            tower_flight_radius = max(tower_object_radius_m + constraints_obj.standoff_m, tower_object_radius_m + 4.0)
        else:
            tower_flight_radius = max(float(tower_flight_radius_m), tower_object_radius_m + 2.0)
        tower_safe_rth_altitude_m = max(float(constraints_obj.rth_altitude_m), tower_top_alt + 10.0)
        if template == "tower_mapping" and tower_safe_rth_altitude_m > float(constraints_obj.rth_altitude_m):
            constraints_obj = replace(constraints_obj, rth_altitude_m=tower_safe_rth_altitude_m)

        local_polygon = _world_to_local(polygon, frame)
        local_polygon_closed = _ensure_closed_xy(local_polygon.tolist())
        linear_path_local = _world_to_local(linear_path_world, frame).tolist() if linear_path_world is not None else None
        lateral_target_local = (
            _world_to_local(lateral_target_world, frame).tolist()
            if lateral_target_world is not None
            else None
        )
        waypoint_path_local = _world_to_local(waypoint_path_world, frame).tolist() if waypoint_path_world is not None else None
        waypoint_poi_local = (
            _world_to_local([waypoint_poi_world], frame)[0].tolist()
            if waypoint_poi_world is not None
            else None
        )
        orbit_center_local = (
            _world_to_local([orbit_center_world], frame)[0].tolist()
            if orbit_center_world is not None
            else [0.0, 0.0]
        )
        orbit_poi_local = (
            _world_to_local([orbit_poi_world], frame)[0].tolist()
            if orbit_poi_world is not None
            else None
        )
        panorama_center_local = (
            _world_to_local([panorama_center_world], frame)[0].tolist()
            if panorama_center_world is not None
            else [0.0, 0.0]
        )
        bubble_center_local = (
            _world_to_local([bubble_center_world], frame)[0].tolist()
            if bubble_center_world is not None
            else [0.0, 0.0]
        )
        tower_center_local = (
            _world_to_local([tower_center_world], frame)[0].tolist()
            if tower_center_world is not None
            else [0.0, 0.0]
        )
        solar_rows_local = [
            _world_to_local(row, frame).tolist()
            for row in solar_rows_world
            if len(row) >= 2
        ]
        facade_curve_local = (
            _world_to_local(facade_curve_world, frame).tolist()
            if facade_curve_world is not None and len(facade_curve_world) >= 2
            else None
        )
        terrain_model = _build_terrain_model(
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_follow_mode=terrain_follow_mode,
            local_polygon_closed=local_polygon_closed,
            raw_vertices=raw_vertices,
            frame=frame,
            terrain_source_path=terrain_source_path,
            terrain_samples_lonlatz=terrain_samples_lonlatz,
            terrain_model_override=terrain_model_override,
        )

        effective_altitude_m = max(5.0, altitude_m + ground_offset_m)
        if template in {"facade", "facade_mapping"}:
            facade_range_m = max(2.0, facade_standoff)
            fp_w, fp_h = _estimate_footprint_m(facade_range_m, camera)
            line_spacing_m = max(0.5, fp_w * (1.0 - side_overlap_pct / 100.0))
            capture_spacing_m = max(0.5, fp_h * (1.0 - front_overlap_pct / 100.0))
            if template == "facade_mapping":
                line_spacing_m = max(0.5, line_spacing_m * FACADE_MAPPING_SPACING_SCALE)
                capture_spacing_m = max(0.5, capture_spacing_m * FACADE_MAPPING_SPACING_SCALE)
        elif template == "linear_inspection":
            fp_w, fp_h = _effective_footprint_m(effective_altitude_m, camera, gimbal_tilt_deg)
            spacing = max(0.75, fp_h * (1.0 - front_overlap_pct / 100.0))
            line_spacing_m = spacing
            capture_spacing_m = spacing
        elif template == "lateral_capture":
            fp_w, fp_h = _effective_footprint_m(effective_altitude_m, camera, gimbal_tilt_deg)
            spacing = max(0.75, fp_h * (1.0 - front_overlap_pct / 100.0))
            line_spacing_m = spacing
            capture_spacing_m = spacing
        elif template == "waypoints":
            if waypoint_path_local is not None and len(waypoint_path_local) >= 2:
                mean_leg = _distance_xy(waypoint_path_local) / max(1, len(waypoint_path_local) - 1)
                spacing = max(0.75, float(mean_leg))
            else:
                spacing = 3.0
            line_spacing_m = spacing
            capture_spacing_m = spacing
        elif template == "tower_mapping":
            tower_capture_range_m = max(2.0, tower_flight_radius - tower_object_radius_m)
            fp_w, fp_h = _estimate_footprint_m(tower_capture_range_m, camera)
            line_spacing_m = max(0.75, fp_w * (1.0 - side_overlap_pct / 100.0))
            capture_spacing_m = max(0.75, fp_h * (1.0 - front_overlap_pct / 100.0))
        elif template == "solar_inspection":
            fp_w, fp_h = _effective_footprint_m(effective_altitude_m, camera, gimbal_tilt_deg)
            line_spacing_m = max(0.75, fp_w * (1.0 - side_overlap_pct / 100.0))
            capture_spacing_m = max(0.75, fp_h * (1.0 - front_overlap_pct / 100.0))
        elif template == "magnetic_mapping":
            fp_w, fp_h = _effective_footprint_m(effective_altitude_m, camera, gimbal_tilt_deg)
            line_spacing_m = max(2.0, fp_w * (1.0 - side_overlap_pct / 100.0))
            capture_spacing_m = max(2.0, fp_h * (1.0 - front_overlap_pct / 100.0))
        elif template == "panorama":
            line_spacing_m = 0.0
            capture_spacing_m = 0.0
        elif template == "bubble_360":
            line_spacing_m = 0.0
            capture_spacing_m = 0.0
        else:
            fp_w, fp_h = _effective_footprint_m(effective_altitude_m, camera, gimbal_tilt_deg)
            line_spacing_m = max(0.75, fp_w * (1.0 - side_overlap_pct / 100.0))
            capture_spacing_m = max(0.75, fp_h * (1.0 - front_overlap_pct / 100.0))

        if template == "orbit":
            if orbit_level_count is None:
                if requested_front_overlap_pct >= 85.0 or requested_side_overlap_pct >= 80.0:
                    orbit_level_count_eff = 3
                elif requested_front_overlap_pct >= 70.0 or requested_side_overlap_pct >= 65.0:
                    orbit_level_count_eff = 2
                else:
                    orbit_level_count_eff = 1
            else:
                orbit_level_count_eff = int(np.clip(orbit_level_count, 1, 8))
            orbit_vertical_step_eff = (
                float(max(0.5, orbit_vertical_step_m))
                if orbit_vertical_step_m is not None
                else float(max(0.75, capture_spacing_m))
            )
        else:
            orbit_level_count_eff = 1
            orbit_vertical_step_eff = float(max(0.75, capture_spacing_m))

        primitives = self._build_primitives(
            template=template,
            local_polygon=local_polygon_closed,
            camera_name=camera,
            altitude_m=altitude_m,
            line_step_m=line_spacing_m,
            point_step_m=capture_spacing_m,
            constraints=constraints_obj,
            flight_direction_deg=flight_direction_deg,
            camera_direction_deg=camera_direction_value,
            camera_direction_locked=camera_direction_locked,
            gimbal_pitch_deg=gimbal_tilt_deg,
            inspection_dwell_s=inspection_dwell_s,
            facade_top_altitude_m=top_altitude_m,
            facade_bottom_altitude_m=bottom_altitude_m,
            facade_standoff_m=facade_standoff,
            facade_rotate_points_180=facade_rotate_points_180,
            facade_capture_profile=facade_capture_profile,
            linear_path_local=linear_path_local,
            linear_segmentation_enabled=linear_segmentation_enabled,
            linear_max_segment_length_m=linear_max_segment_length_m,
            lateral_target_local=lateral_target_local,
            lateral_standoff_m=lateral_standoff_m,
            lateral_target_side=lateral_target_side,
            waypoint_path_local=waypoint_path_local,
            waypoint_heading_mode=waypoint_heading_mode,
            waypoint_fixed_yaw_deg=waypoint_fixed_yaw_deg,
            waypoint_poi_local=waypoint_poi_local,
            waypoint_enable_smoothing=waypoint_enable_smoothing,
            waypoint_turn_radius_m=waypoint_turn_radius_m,
            waypoint_capture_enabled=waypoint_capture_enabled,
            orbit_center_local=orbit_center_local,
            orbit_radius_m=orbit_radius_m,
            orbit_level_count=orbit_level_count_eff,
            orbit_vertical_step_m=orbit_vertical_step_eff,
            orbit_poi_yaw_lock=orbit_poi_yaw_lock,
            orbit_poi_local=orbit_poi_local,
            panorama_center_local=panorama_center_local,
            panorama_overlap_pct=panorama_overlap_pct,
            panorama_multi_row_enabled=panorama_multi_row_enabled,
            panorama_row_count=panorama_row_count,
            panorama_pitch_step_deg=panorama_pitch_step_deg,
            bubble_center_local=bubble_center_local,
            bubble_overlap_pct=bubble_overlap_pct,
            bubble_pitch_step_deg=bubble_pitch_step_deg,
            bubble_top_pitch_deg=bubble_top_pitch_deg,
            bubble_bottom_pitch_deg=bubble_bottom_pitch_deg,
            tower_center_local=tower_center_local,
            tower_top_altitude_m=tower_top_alt,
            tower_bottom_altitude_m=tower_bottom_alt,
            tower_object_radius_m=tower_object_radius_m,
            tower_flight_radius_m=tower_flight_radius,
            tower_resume_enabled=tower_resume_enabled,
            solar_row_angle_deg=solar_row_angle_deg,
            solar_sensor_profile=solar_sensor_profile,
            solar_orientation_mode=solar_orientation_mode,
            solar_rows_local=solar_rows_local,
            magnetic_tie_line_spacing_m=magnetic_tie_line_spacing_m,
            magnetic_smoothing_radius_m=magnetic_smoothing_radius_m,
            ground_offset_m=ground_offset_m,
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_follow_mode=terrain_follow_mode,
            terrain_model=terrain_model,
            terrain_normal_camera_enabled=terrain_normal_camera_enabled,
            terrain_normal_gain=terrain_normal_gain,
            terrain_normal_yaw_align=terrain_normal_yaw_align,
            facade_curvature_alignment=facade_curvature_alignment,
            facade_curve_local=facade_curve_local,
            double_grid_cross_angle_deg=cross_angle_deg,
        )

        coverage = CoverageExpectation(
            front_overlap_pct=front_overlap_pct,
            side_overlap_pct=side_overlap_pct,
            minimum_coverage_pct=95.0,
            required_viewpoints=0,
        )

        recipe_seed = {
            "template": template,
            "polygon_local": [[round(p[0], 2), round(p[1], 2)] for p in local_polygon_closed],
            "altitude_m": round(altitude_m, 2),
            "front_overlap_pct": round(front_overlap_pct, 2),
            "side_overlap_pct": round(side_overlap_pct, 2),
            "camera": camera,
            "flight_direction_deg": round(flight_direction_deg, 2),
            "camera_direction_deg": round(camera_direction_value, 2),
            "camera_direction_locked": bool(camera_direction_locked),
            "gimbal_tilt_deg": round(gimbal_tilt_deg, 2),
            "inspection_dwell_s": round(inspection_dwell_s, 2),
            "facade_top_altitude_m": round(top_altitude_m, 2),
            "facade_bottom_altitude_m": round(bottom_altitude_m, 2),
            "facade_standoff_m": round(facade_standoff, 2),
            "facade_rotate_points_180": bool(facade_rotate_points_180),
            "facade_capture_profile": facade_capture_profile,
            "linear_segmentation_enabled": bool(linear_segmentation_enabled),
            "linear_max_segment_length_m": round(linear_max_segment_length_m, 2),
            "linear_path_points": [[round(p[0], 7), round(p[1], 7)] for p in linear_path_world] if linear_path_world else [],
            "lateral_path_points": [[round(p[0], 7), round(p[1], 7)] for p in lateral_target_world] if lateral_target_world else [],
            "lateral_standoff_m": round(lateral_standoff_m, 2),
            "lateral_target_side": lateral_target_side,
            "lateral_yaw_offset_deg": float(90.0 if lateral_target_side == "left" else -90.0),
            "waypoint_path_points": [[round(p[0], 7), round(p[1], 7)] for p in waypoint_path_world] if waypoint_path_world else [],
            "waypoint_heading_mode": waypoint_heading_mode,
            "waypoint_fixed_yaw_deg": round(waypoint_fixed_yaw_deg, 2),
            "waypoint_poi_lonlat": [round(float(waypoint_poi_world[0]), 7), round(float(waypoint_poi_world[1]), 7)]
            if waypoint_poi_world is not None
            else [],
            "waypoint_enable_smoothing": bool(waypoint_enable_smoothing),
            "waypoint_turn_radius_m": round(waypoint_turn_radius_m, 2),
            "waypoint_capture_enabled": bool(waypoint_capture_enabled),
            "orbit_center_lonlat": [round(float(orbit_center_world[0]), 7), round(float(orbit_center_world[1]), 7)]
            if orbit_center_world is not None
            else [],
            "orbit_radius_m": round(float(orbit_radius_m if orbit_radius_m is not None else 0.0), 2),
            "orbit_level_count": int(orbit_level_count_eff),
            "orbit_vertical_step_m": round(float(orbit_vertical_step_eff), 2),
            "orbit_poi_yaw_lock": bool(orbit_poi_yaw_lock),
            "orbit_poi_lonlat": [round(float(orbit_poi_world[0]), 7), round(float(orbit_poi_world[1]), 7)]
            if orbit_poi_world is not None
            else [],
            "panorama_center_lonlat": [round(float(panorama_center_world[0]), 7), round(float(panorama_center_world[1]), 7)]
            if panorama_center_world is not None
            else [],
            "panorama_overlap_pct": round(panorama_overlap_pct, 2),
            "panorama_multi_row_enabled": bool(panorama_multi_row_enabled),
            "panorama_row_count": int(panorama_row_count),
            "panorama_pitch_step_deg": round(panorama_pitch_step_deg, 2),
            "bubble_center_lonlat": [round(float(bubble_center_world[0]), 7), round(float(bubble_center_world[1]), 7)]
            if bubble_center_world is not None
            else [],
            "bubble_overlap_pct": round(bubble_overlap_pct, 2),
            "bubble_pitch_step_deg": round(bubble_pitch_step_deg, 2),
            "bubble_top_pitch_deg": round(bubble_top_pitch_deg, 2),
            "bubble_bottom_pitch_deg": round(bubble_bottom_pitch_deg, 2),
            "tower_center_lonlat": [round(float(tower_center_world[0]), 7), round(float(tower_center_world[1]), 7)]
            if tower_center_world is not None
            else [],
            "tower_top_altitude_m": round(tower_top_alt, 2),
            "tower_bottom_altitude_m": round(tower_bottom_alt, 2),
            "tower_object_radius_m": round(tower_object_radius_m, 2),
            "tower_flight_radius_m": round(tower_flight_radius, 2),
            "tower_resume_enabled": bool(tower_resume_enabled),
            "tower_safe_rth_altitude_m": round(tower_safe_rth_altitude_m, 2),
            "solar_row_angle_deg": round(float(solar_row_angle_deg if solar_row_angle_deg is not None else flight_direction_deg), 2),
            "solar_sensor_profile": solar_sensor_profile,
            "solar_orientation_mode": solar_orientation_mode,
            "solar_rows_lonlat": [
                [[round(float(p[0]), 7), round(float(p[1]), 7)] for p in row]
                for row in solar_rows_world
            ],
            "magnetic_tie_line_spacing_m": round(magnetic_tie_line_spacing_m, 2),
            "magnetic_smoothing_radius_m": round(magnetic_smoothing_radius_m, 2),
            "ground_offset_m": round(ground_offset_m, 2),
            "terrain_follow_enabled": bool(terrain_follow_enabled),
            "terrain_follow_mode": str(terrain_follow_mode),
            "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
            "terrain_normal_gain": round(float(terrain_normal_gain), 3),
            "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
            "terrain_source_path": terrain_source_path,
            "wind_speed_m_s": round(float(wind_speed_m_s), 3),
            "wind_direction_deg": round(float(wind_direction_deg), 3),
            "wind_gust_m_s": round(float(wind_gust_m_s), 3),
            "facade_curvature_alignment": bool(facade_curvature_alignment),
            "double_grid_cross_angle_deg": round(cross_angle_deg, 2),
            "requested_front_overlap_pct": round(requested_front_overlap_pct, 2),
            "requested_side_overlap_pct": round(requested_side_overlap_pct, 2),
            "overlap_floor_applied": bool(overlap_floor_applied),
            "terrain_model": {
                "type": str(terrain_model.get("type", "flat")),
                "source": str(terrain_model.get("source", "none")),
                "slope_x_m_per_m": round(float(terrain_model.get("slope_x_m_per_m", 0.0)), 6),
                "slope_y_m_per_m": round(float(terrain_model.get("slope_y_m_per_m", 0.0)), 6),
                "offset_m": round(float(terrain_model.get("offset_m", 0.0)), 3),
            },
            "version": int(recipe_version),
        }
        digest = hashlib.sha1(json.dumps(recipe_seed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        recipe_id = f"fr-{digest}"

        recipe_meta: dict[str, Any] = {
            "camera": camera,
            "line_step_m": line_spacing_m,
            "point_step_m": capture_spacing_m,
            "line_spacing_m": line_spacing_m,
            "capture_spacing_m": capture_spacing_m,
            "flight_direction_deg": flight_direction_deg,
            "camera_direction_deg": camera_direction_value,
            "camera_direction_locked": bool(camera_direction_locked),
            "gimbal_tilt_deg": gimbal_tilt_deg,
            "inspection_dwell_s": inspection_dwell_s,
            "facade_top_altitude_m": top_altitude_m,
            "facade_bottom_altitude_m": bottom_altitude_m,
            "facade_standoff_m": facade_standoff,
            "facade_rotate_points_180": bool(facade_rotate_points_180),
            "facade_capture_profile": facade_capture_profile,
            "linear_segmentation_enabled": bool(linear_segmentation_enabled),
            "linear_max_segment_length_m": linear_max_segment_length_m,
            "linear_path_points": linear_path_world or [],
            "lateral_path_points": lateral_target_world or [],
            "facade_curve_path_points": facade_curve_world or [],
            "lateral_standoff_m": lateral_standoff_m,
            "lateral_target_side": lateral_target_side,
            "lateral_yaw_offset_deg": float(90.0 if lateral_target_side == "left" else -90.0),
            "waypoint_path_points": waypoint_path_world or [],
            "waypoint_heading_mode": waypoint_heading_mode,
            "waypoint_fixed_yaw_deg": waypoint_fixed_yaw_deg,
            "waypoint_poi_lonlat": waypoint_poi_world or [],
            "waypoint_enable_smoothing": bool(waypoint_enable_smoothing),
            "waypoint_turn_radius_m": waypoint_turn_radius_m,
            "waypoint_capture_enabled": bool(waypoint_capture_enabled),
            "orbit_center_lonlat": orbit_center_world or [],
            "orbit_radius_m": float(orbit_radius_m if orbit_radius_m is not None else 0.0),
            "orbit_level_count": int(orbit_level_count_eff),
            "orbit_vertical_step_m": float(orbit_vertical_step_eff),
            "orbit_poi_yaw_lock": bool(orbit_poi_yaw_lock),
            "orbit_poi_lonlat": orbit_poi_world or [],
            "panorama_center_lonlat": panorama_center_world or [],
            "panorama_overlap_pct": panorama_overlap_pct,
            "panorama_multi_row_enabled": bool(panorama_multi_row_enabled),
            "panorama_row_count": int(panorama_row_count),
            "panorama_pitch_step_deg": panorama_pitch_step_deg,
            "bubble_center_lonlat": bubble_center_world or [],
            "bubble_overlap_pct": bubble_overlap_pct,
            "bubble_pitch_step_deg": bubble_pitch_step_deg,
            "bubble_top_pitch_deg": bubble_top_pitch_deg,
            "bubble_bottom_pitch_deg": bubble_bottom_pitch_deg,
            "tower_center_lonlat": tower_center_world or [],
            "tower_top_altitude_m": tower_top_alt,
            "tower_bottom_altitude_m": tower_bottom_alt,
            "tower_object_radius_m": tower_object_radius_m,
            "tower_flight_radius_m": tower_flight_radius,
            "tower_resume_enabled": bool(tower_resume_enabled),
            "tower_safe_rth_altitude_m": float(tower_safe_rth_altitude_m),
            "solar_row_angle_deg": float(solar_row_angle_deg if solar_row_angle_deg is not None else flight_direction_deg),
            "solar_sensor_profile": solar_sensor_profile,
            "solar_orientation_mode": solar_orientation_mode,
            "solar_rows_lonlat": solar_rows_world,
            "magnetic_tie_line_spacing_m": magnetic_tie_line_spacing_m,
            "magnetic_smoothing_radius_m": magnetic_smoothing_radius_m,
            "ground_offset_m": ground_offset_m,
            "terrain_follow_enabled": bool(terrain_follow_enabled),
            "terrain_follow_mode": str(terrain_follow_mode),
            "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
            "terrain_normal_gain": float(terrain_normal_gain),
            "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
            "terrain_source_path": terrain_source_path,
            "terrain_model": terrain_model,
            "wind_speed_m_s": float(wind_speed_m_s),
            "wind_direction_deg": float(wind_direction_deg),
            "wind_gust_m_s": float(wind_gust_m_s),
            "facade_curvature_alignment": bool(facade_curvature_alignment),
            "double_grid_cross_angle_deg": float(cross_angle_deg),
            "requested_front_overlap_pct": float(requested_front_overlap_pct),
            "requested_side_overlap_pct": float(requested_side_overlap_pct),
            "photogrammetry_overlap_front_floor_pct": float(DOUBLE_GRID_MIN_FRONT_OVERLAP_PCT),
            "photogrammetry_overlap_side_floor_pct": float(DOUBLE_GRID_MIN_SIDE_OVERLAP_PCT),
            "overlap_floor_applied": bool(overlap_floor_applied),
            "generated_from": "mission_planner",
        }
        if template == "double_grid":
            recipe_meta["camera_policy_profile"] = "photogrammetry_locked_exposure"
            recipe_meta["capture_dataset"] = "3d_modelling_double_grid"
            recipe_meta["camera_policy"] = _double_grid_camera_policy(
                speed_m_s=5.0,
                gsd_cm=_estimate_gsd_cm(effective_altitude_m, camera),
            )
        if template == "facade_mapping":
            recipe_meta["camera_policy_profile"] = "facade_photogrammetry_locked_exposure"
            recipe_meta["capture_dataset"] = "3d_facade_mapping"
            recipe_meta["smooth_motion_profile"] = "facade_mapping"
            recipe_meta["recommended_speed_m_s"] = 2.5
            recipe_meta["camera_policy"] = _facade_mapping_camera_policy(
                speed_m_s=2.5,
                gsd_cm=_estimate_gsd_cm(max(2.0, facade_standoff), camera),
                capture_profile=facade_capture_profile,
            )
        if template == "linear_inspection":
            recipe_meta["capture_dataset"] = "linear_inspection"
            recipe_meta["smooth_motion_profile"] = "stop_and_capture"
        if template == "lateral_capture":
            recipe_meta["capture_dataset"] = "lateral_profile_capture"
            recipe_meta["smooth_motion_profile"] = "sideways_profile_tracking"
        if template == "orbit":
            recipe_meta["capture_dataset"] = "orbit_general"
            recipe_meta["smooth_motion_profile"] = "circular_orbit_stack"
        if template == "panorama":
            recipe_meta["capture_dataset"] = "panorama_capture"
            recipe_meta["smooth_motion_profile"] = "yaw_sweep"
        if template == "bubble_360":
            recipe_meta["capture_dataset"] = "360_bubble_capture"
            recipe_meta["smooth_motion_profile"] = "yaw_pitch_spherical_sweep"
        if template == "waypoints":
            recipe_meta["capture_dataset"] = "advanced_waypoints"
            recipe_meta["smooth_motion_profile"] = "curved_path" if waypoint_enable_smoothing else "direct_path"
        if template == "tower_mapping":
            recipe_meta["capture_dataset"] = "tower_mapping"
            recipe_meta["smooth_motion_profile"] = "orbital_vertical_step"
            recipe_meta["resume_strategy"] = "battery_change_rejoin_next_orbit"
        if template == "solar_inspection":
            recipe_meta["capture_dataset"] = "solar_row_aligned_inspection"
            recipe_meta["smooth_motion_profile"] = "row_aligned"
        if template == "magnetic_mapping":
            recipe_meta["capture_dataset"] = "magnetic_lawnmower_tielines"
            recipe_meta["smooth_motion_profile"] = "curvature_turns"
        if metadata:
            recipe_meta.update(metadata)

        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return FlightRecipe(
            recipe_id=recipe_id,
            version=max(1, int(recipe_version)),
            template=template,
            asset_frame=frame,
            primitives=primitives,
            constraints=constraints_obj,
            coverage=coverage,
            created_at_utc=created,
            metadata=recipe_meta,
        )

    def compile_recipe(
        self,
        recipe: FlightRecipe | dict,
        speed_m_s: float = 5.0,
        camera: str | None = None,
        repeat_enabled: bool = False,
        asset_frame: AssetReferenceFrame | dict | None = None,
        constraints_override: MissionConstraints | dict | None = None,
    ) -> MissionPlan:
        recipe_obj = self._coerce_recipe(recipe)

        if asset_frame is not None:
            if isinstance(asset_frame, AssetReferenceFrame):
                recipe_obj = replace(recipe_obj, asset_frame=asset_frame)
            else:
                recipe_obj = replace(recipe_obj, asset_frame=_asset_frame_from_dict(dict(asset_frame)))

        if constraints_override is not None:
            default_alt = self._default_recipe_altitude(recipe_obj)
            merged_constraints = self._coerce_constraints(
                constraints_override,
                polygon=recipe_obj.constraints.geofence,
                default_altitude_m=default_alt,
            )
            recipe_obj = replace(recipe_obj, constraints=merged_constraints)

        local_poses: list[_CapturePose] = []
        for primitive in recipe_obj.primitives:
            local_poses.extend(self._compile_primitive(primitive))

        if not local_poses:
            geofence_local = _world_to_local(recipe_obj.constraints.geofence, recipe_obj.asset_frame)
            geofence_local = geofence_local[:-1] if len(geofence_local) > 1 else geofence_local
            default_alt = self._default_recipe_altitude(recipe_obj)
            for pt in geofence_local:
                local_poses.append(
                    _CapturePose(
                        x_m=float(pt[0]),
                        y_m=float(pt[1]),
                        alt_m=float(default_alt),
                        yaw_deg=0.0,
                        gimbal_pitch_deg=-90.0,
                        primitive="fallback",
                        trigger=True,
                    )
                )

        constrained_poses, safety_adjustments = self._apply_constraints(local_poses, recipe_obj)
        world_poses = self._local_to_world_poses(constrained_poses, recipe_obj.asset_frame)

        waypoints = [[float(p["lon"]), float(p["lat"]), float(p["alt_m"])] for p in world_poses]
        distance = _distance_m(waypoints)
        template_name = _normalize_template(recipe_obj.template)
        speed_requested = float(max(0.5, speed_m_s))
        if template_name == "facade_mapping":
            speed_cap = float(max(0.5, float(recipe_obj.metadata.get("recommended_speed_m_s", 2.5))))
            speed_nominal = min(speed_requested, speed_cap)
        else:
            speed_nominal = speed_requested
        wind_speed = float(max(0.0, recipe_obj.metadata.get("wind_speed_m_s", 0.0)))
        wind_direction = float(recipe_obj.metadata.get("wind_direction_deg", 0.0))
        wind_gust = float(max(0.0, recipe_obj.metadata.get("wind_gust_m_s", 0.0)))
        wind_heading = float(recipe_obj.metadata.get("flight_direction_deg", 0.0))
        wind_eval = _wind_adjusted_speed(
            requested_speed_m_s=float(speed_nominal),
            heading_deg=float(wind_heading),
            wind_speed_m_s=float(wind_speed),
            wind_direction_deg=float(wind_direction),
            wind_gust_m_s=float(wind_gust),
        )
        speed = float(max(0.5, wind_eval.get("effective_speed_m_s", speed_nominal)))
        capture_count = sum(1 for p in world_poses if bool(p.get("trigger", True)))
        dwell_total_s = float(sum(float(p.get("dwell_s", 0.0)) for p in world_poses))
        est_time = float(distance / speed / 60.0 + (capture_count * 0.5 + dwell_total_s) / 60.0)

        camera_name = str(camera or recipe_obj.metadata.get("camera") or "custom").lower()
        mean_alt = float(np.mean([p["alt_m"] for p in world_poses])) if world_poses else self._default_recipe_altitude(recipe_obj)
        gsd_cm = _estimate_gsd_cm(mean_alt, camera_name)
        mapping_config = self._mapping_capture_config(
            recipe_obj=recipe_obj,
            speed_m_s=speed,
            estimated_gsd_cm=gsd_cm,
        )
        linear_segments: set[int] = set()
        tower_levels: set[int] = set()
        has_linear = False
        has_tower = False
        for pose in world_poses:
            primitive_name = str(pose.get("primitive", ""))
            if primitive_name.startswith("linear_inspection"):
                has_linear = True
                if "_seg" in primitive_name:
                    try:
                        linear_segments.add(int(primitive_name.split("_seg", 1)[1]))
                    except Exception:
                        pass
            if primitive_name.startswith("tower_mapping"):
                has_tower = True
                if "_level" in primitive_name:
                    try:
                        tower_levels.add(int(primitive_name.split("_level", 1)[1]))
                    except Exception:
                        pass
        if linear_segments:
            mapping_config["linear_segment_count"] = float(len(linear_segments))
        elif has_linear:
            mapping_config["linear_segment_count"] = 1.0
        if tower_levels:
            mapping_config["tower_orbit_count"] = float(len(tower_levels))
        elif has_tower and float(mapping_config.get("tower_orbit_count", 0.0)) <= 0.0:
            mapping_config["tower_orbit_count"] = 1.0
        mapping_config["wind_speed_m_s"] = float(wind_speed)
        mapping_config["wind_direction_deg"] = float(wind_direction)
        mapping_config["wind_gust_m_s"] = float(wind_gust)
        mapping_config["wind_adjusted_speed_m_s"] = float(speed)
        mapping_config["wind_penalty_pct"] = float(wind_eval.get("penalty_pct", 0.0))

        autopilot_commands = self._build_autopilot_commands(
            world_poses=world_poses,
            constraints=recipe_obj.constraints,
            speed_m_s=speed,
            repeat_enabled=bool(repeat_enabled),
            continuous_capture=bool(mapping_config["continuous_capture"]),
            capture_spacing_m=float(mapping_config["capture_spacing_m"]),
            capture_interval_s=float(mapping_config["capture_interval_s"]),
            camera_policy=dict(mapping_config.get("camera_policy") or {}),
            wind_model={
                "wind_speed_m_s": float(wind_speed),
                "wind_direction_deg": float(wind_direction),
                "wind_gust_m_s": float(wind_gust),
                "crosswind_m_s": float(wind_eval.get("crosswind_m_s", 0.0)),
                "headwind_m_s": float(wind_eval.get("headwind_m_s", 0.0)),
                "penalty_pct": float(wind_eval.get("penalty_pct", 0.0)),
            },
        )
        coverage_report = self._coverage_report(recipe_obj, capture_count)

        repeat_anchor = {
            "asset_frame": recipe_obj.asset_frame.to_dict(),
            "pose_tolerance": {
                "position_m": 1.25,
                "yaw_deg": 3.0,
                "gimbal_pitch_deg": 2.0,
            },
            "capture_poses_local": [
                {
                    "x_m": float(p.x_m),
                    "y_m": float(p.y_m),
                    "alt_m": float(p.alt_m),
                    "yaw_deg": float(p.yaw_deg),
                    "gimbal_pitch_deg": float(p.gimbal_pitch_deg),
                    "primitive": p.primitive,
                    "dwell_s": float(p.dwell_s),
                    "camera_yaw_locked": bool(p.camera_yaw_locked),
                }
                for p in constrained_poses
            ],
        }

        geojson = _geojson_from_plan(
            polygon=recipe_obj.constraints.geofence,
            waypoints=waypoints,
            world_poses=world_poses,
            no_fly_polygons=recipe_obj.constraints.no_fly_polygons,
        )

        return MissionPlan(
            polygon=recipe_obj.constraints.geofence,
            waypoints=waypoints,
            altitude_m=mean_alt,
            front_overlap_pct=recipe_obj.coverage.front_overlap_pct,
            side_overlap_pct=recipe_obj.coverage.side_overlap_pct,
            camera=camera_name,
            mode=recipe_obj.template,
            path_distance_m=distance,
            estimated_time_min=est_time,
            estimated_gsd_cm=gsd_cm,
            source="recipe_compiler",
            geojson=geojson,
            kmz_path="",
            template=recipe_obj.template,
            recipe_version=recipe_obj.version,
            flight_recipe=recipe_obj.to_dict(),
            autopilot_commands=autopilot_commands,
            safety_constraints=recipe_obj.constraints.to_dict(),
            repeat_enabled=bool(repeat_enabled),
            repeat_anchor=repeat_anchor,
            expected_coverage=coverage_report,
            safety_adjustments=safety_adjustments,
            flight_direction_deg=float(mapping_config["flight_direction_deg"]),
            camera_direction_deg=float(mapping_config.get("camera_direction_deg", mapping_config["flight_direction_deg"])),
            camera_direction_locked=bool(mapping_config.get("camera_direction_locked", False)),
            gimbal_tilt_deg=float(mapping_config["gimbal_tilt_deg"]),
            inspection_dwell_s=float(mapping_config.get("inspection_dwell_s", 0.0)),
            ground_offset_m=float(mapping_config["ground_offset_m"]),
            terrain_follow_enabled=bool(mapping_config["terrain_follow_enabled"]),
            terrain_follow_mode=str(mapping_config.get("terrain_follow_mode", "agl")),
            terrain_normal_camera_enabled=bool(mapping_config.get("terrain_normal_camera_enabled", False)),
            terrain_normal_gain=float(mapping_config.get("terrain_normal_gain", 1.0)),
            terrain_normal_yaw_align=bool(mapping_config.get("terrain_normal_yaw_align", False)),
            terrain_model_type=str(mapping_config.get("terrain_model_type", "flat")),
            terrain_model_source=str(mapping_config.get("terrain_model_source", "none")),
            terrain_source_path=str(mapping_config.get("terrain_source_path", "")),
            line_spacing_m=float(mapping_config["line_spacing_m"]),
            capture_spacing_m=float(mapping_config["capture_spacing_m"]),
            capture_interval_s=float(mapping_config["capture_interval_s"]),
            double_grid_cross_angle_deg=float(mapping_config.get("double_grid_cross_angle_deg", 0.0)),
            camera_policy=dict(mapping_config.get("camera_policy") or {}),
            facade_top_altitude_m=float(mapping_config.get("facade_top_altitude_m", 0.0)),
            facade_bottom_altitude_m=float(mapping_config.get("facade_bottom_altitude_m", 0.0)),
            facade_standoff_m=float(mapping_config.get("facade_standoff_m", 0.0)),
            facade_rotate_points_180=bool(mapping_config.get("facade_rotate_points_180", False)),
            facade_capture_profile=str(mapping_config.get("facade_capture_profile", "custom")),
            smooth_motion_profile=str(mapping_config.get("smooth_motion_profile", "")),
            linear_segmentation_enabled=bool(mapping_config.get("linear_segmentation_enabled", False)),
            linear_max_segment_length_m=float(mapping_config.get("linear_max_segment_length_m", 0.0)),
            linear_segment_count=int(round(float(mapping_config.get("linear_segment_count", 0.0)))),
            linear_path_length_m=float(mapping_config.get("linear_path_length_m", 0.0)),
            lateral_standoff_m=float(mapping_config.get("lateral_standoff_m", 0.0)),
            lateral_target_side=str(mapping_config.get("lateral_target_side", "right")),
            lateral_yaw_offset_deg=float(mapping_config.get("lateral_yaw_offset_deg", 0.0)),
            lateral_path_length_m=float(mapping_config.get("lateral_path_length_m", 0.0)),
            tower_top_altitude_m=float(mapping_config.get("tower_top_altitude_m", 0.0)),
            tower_bottom_altitude_m=float(mapping_config.get("tower_bottom_altitude_m", 0.0)),
            tower_object_radius_m=float(mapping_config.get("tower_object_radius_m", 0.0)),
            tower_flight_radius_m=float(mapping_config.get("tower_flight_radius_m", 0.0)),
            tower_orbit_count=int(round(float(mapping_config.get("tower_orbit_count", 0.0)))),
            tower_resume_enabled=bool(mapping_config.get("tower_resume_enabled", False)),
            tower_safe_rth_altitude_m=float(mapping_config.get("tower_safe_rth_altitude_m", 0.0)),
            solar_row_angle_deg=float(mapping_config.get("solar_row_angle_deg", 0.0)),
            solar_sensor_profile=str(mapping_config.get("solar_sensor_profile", "rgb")),
            solar_orientation_mode=str(mapping_config.get("solar_orientation_mode", "row_aligned")),
            magnetic_tie_line_spacing_m=float(mapping_config.get("magnetic_tie_line_spacing_m", 0.0)),
            magnetic_smoothing_radius_m=float(mapping_config.get("magnetic_smoothing_radius_m", 0.0)),
            orbit_radius_m=float(mapping_config.get("orbit_radius_m", 0.0)),
            orbit_level_count=int(round(float(mapping_config.get("orbit_level_count", 0.0)))),
            orbit_vertical_step_m=float(mapping_config.get("orbit_vertical_step_m", 0.0)),
            orbit_poi_yaw_lock=bool(mapping_config.get("orbit_poi_yaw_lock", True)),
            panorama_overlap_pct=float(mapping_config.get("panorama_overlap_pct", 0.0)),
            panorama_multi_row_enabled=bool(mapping_config.get("panorama_multi_row_enabled", False)),
            panorama_row_count=int(round(float(mapping_config.get("panorama_row_count", 0.0)))),
            panorama_pitch_step_deg=float(mapping_config.get("panorama_pitch_step_deg", 0.0)),
            panorama_yaw_step_deg=float(mapping_config.get("panorama_yaw_step_deg", 0.0)),
            panorama_yaw_count=int(round(float(mapping_config.get("panorama_yaw_count", 0.0)))),
            bubble_overlap_pct=float(mapping_config.get("bubble_overlap_pct", 0.0)),
            bubble_pitch_step_deg=float(mapping_config.get("bubble_pitch_step_deg", 0.0)),
            bubble_top_pitch_deg=float(mapping_config.get("bubble_top_pitch_deg", 0.0)),
            bubble_bottom_pitch_deg=float(mapping_config.get("bubble_bottom_pitch_deg", 0.0)),
            bubble_pitch_count=int(round(float(mapping_config.get("bubble_pitch_count", 0.0)))),
            bubble_yaw_step_deg=float(mapping_config.get("bubble_yaw_step_deg", 0.0)),
            bubble_yaw_count=int(round(float(mapping_config.get("bubble_yaw_count", 0.0)))),
            waypoint_heading_mode=str(mapping_config.get("waypoint_heading_mode", "tangent")),
            waypoint_fixed_yaw_deg=float(mapping_config.get("waypoint_fixed_yaw_deg", 0.0)),
            waypoint_turn_radius_m=float(mapping_config.get("waypoint_turn_radius_m", 0.0)),
            waypoint_smoothing_enabled=bool(mapping_config.get("waypoint_smoothing_enabled", False)),
            waypoint_capture_enabled=bool(mapping_config.get("waypoint_capture_enabled", True)),
            waypoint_path_length_m=float(mapping_config.get("waypoint_path_length_m", 0.0)),
            no_fly_polygon_count=int(len(recipe_obj.constraints.no_fly_polygons)),
            linked_segment_count=int(recipe_obj.metadata.get("linked_segment_count", 0)),
            linked_transition_count=int(recipe_obj.metadata.get("linked_transition_count", 0)),
            linked_dry_run_ok=bool(recipe_obj.metadata.get("linked_dry_run_ok", False)),
            wind_speed_m_s=float(mapping_config.get("wind_speed_m_s", 0.0)),
            wind_direction_deg=float(mapping_config.get("wind_direction_deg", 0.0)),
            wind_gust_m_s=float(mapping_config.get("wind_gust_m_s", 0.0)),
            wind_adjusted_speed_m_s=float(mapping_config.get("wind_adjusted_speed_m_s", speed)),
            wind_penalty_pct=float(mapping_config.get("wind_penalty_pct", 0.0)),
            facade_curvature_alignment=bool(mapping_config.get("facade_curvature_alignment", False)),
        )

    def generate(
        self,
        polygon_lonlat: Iterable[Iterable[float]] | None = None,
        altitude_m: float = 60.0,
        front_overlap_pct: float = 80.0,
        side_overlap_pct: float = 70.0,
        speed_m_s: float = 5.0,
        mode: str = "grid",
        camera: str = "mavic2pro",
        flight_direction_deg: float = 0.0,
        camera_direction_deg: float | None = None,
        gimbal_tilt_deg: float = -90.0,
        inspection_dwell_s: float = 1.5,
        facade_top_altitude_m: float | None = None,
        facade_bottom_altitude_m: float | None = None,
        facade_standoff_m: float | None = None,
        facade_rotate_points_180: bool = False,
        facade_capture_profile: str = "custom",
        linear_path_lonlat: Iterable[Iterable[float]] | None = None,
        linear_segmentation_enabled: bool = True,
        linear_max_segment_length_m: float = 1500.0,
        lateral_target_path_lonlat: Iterable[Iterable[float]] | None = None,
        lateral_standoff_m: float = 10.0,
        lateral_target_side: str = "right",
        waypoint_path_lonlat: Iterable[Iterable[float]] | None = None,
        waypoint_heading_mode: str = "tangent",
        waypoint_fixed_yaw_deg: float = 0.0,
        waypoint_poi_lonlat: Iterable[float] | None = None,
        waypoint_enable_smoothing: bool = False,
        waypoint_turn_radius_m: float = 6.0,
        waypoint_capture_enabled: bool = True,
        orbit_center_lonlat: Iterable[float] | None = None,
        orbit_radius_m: float | None = None,
        orbit_level_count: int | None = None,
        orbit_vertical_step_m: float | None = None,
        orbit_poi_yaw_lock: bool = True,
        orbit_poi_lonlat: Iterable[float] | None = None,
        panorama_center_lonlat: Iterable[float] | None = None,
        panorama_overlap_pct: float = 35.0,
        panorama_multi_row_enabled: bool = False,
        panorama_row_count: int = 1,
        panorama_pitch_step_deg: float = 12.0,
        bubble_center_lonlat: Iterable[float] | None = None,
        bubble_overlap_pct: float = 35.0,
        bubble_pitch_step_deg: float = 12.0,
        bubble_top_pitch_deg: float = 20.0,
        bubble_bottom_pitch_deg: float = -90.0,
        tower_center_lonlat: Iterable[float] | None = None,
        tower_top_altitude_m: float | None = None,
        tower_bottom_altitude_m: float | None = None,
        tower_object_radius_m: float = 2.0,
        tower_flight_radius_m: float | None = None,
        tower_resume_enabled: bool = True,
        solar_row_angle_deg: float | None = None,
        solar_sensor_profile: str = "rgb",
        solar_orientation_mode: str = "row_aligned",
        solar_rows_lonlat: Iterable[Iterable[Iterable[float]]] | None = None,
        magnetic_tie_line_spacing_m: float = 50.0,
        magnetic_smoothing_radius_m: float = 8.0,
        ground_offset_m: float = 0.0,
        terrain_follow_enabled: bool = False,
        terrain_source_path: str = "",
        terrain_samples_lonlatz: Iterable[Iterable[float]] | None = None,
        terrain_model_override: dict[str, Any] | None = None,
        terrain_follow_mode: str = "agl",
        terrain_normal_camera_enabled: bool = False,
        terrain_normal_gain: float = 1.0,
        terrain_normal_yaw_align: bool = False,
        wind_speed_m_s: float = 0.0,
        wind_direction_deg: float = 0.0,
        wind_gust_m_s: float = 0.0,
        facade_curvature_alignment: bool = False,
        facade_curve_path_lonlat: Iterable[Iterable[float]] | None = None,
        repeat_recipe: FlightRecipe | dict | None = None,
        enable_repeat: bool = False,
        constraints: MissionConstraints | dict | None = None,
        asset_frame: AssetReferenceFrame | dict | None = None,
    ) -> MissionPlan:
        if repeat_recipe is not None:
            recipe_obj = self._coerce_recipe(repeat_recipe)
            if asset_frame is not None:
                if isinstance(asset_frame, AssetReferenceFrame):
                    recipe_obj = replace(recipe_obj, asset_frame=asset_frame)
                else:
                    recipe_obj = replace(recipe_obj, asset_frame=_asset_frame_from_dict(dict(asset_frame)))
            return self.compile_recipe(
                recipe=recipe_obj,
                speed_m_s=speed_m_s,
                camera=camera,
                repeat_enabled=True,
                constraints_override=constraints,
            )

        normalized_mode = _normalize_template(mode)
        if polygon_lonlat is None and normalized_mode not in {"linear_inspection", "lateral_capture", "tower_mapping", "waypoints", "orbit", "panorama", "bubble_360"}:
            raise ValueError("polygon_lonlat is required for this mission mode.")

        recipe = self.build_flight_recipe(
            polygon_lonlat=polygon_lonlat,
            altitude_m=altitude_m,
            front_overlap_pct=front_overlap_pct,
            side_overlap_pct=side_overlap_pct,
            mode=mode,
            camera=camera,
            flight_direction_deg=flight_direction_deg,
            camera_direction_deg=camera_direction_deg,
            gimbal_tilt_deg=gimbal_tilt_deg,
            inspection_dwell_s=inspection_dwell_s,
            facade_top_altitude_m=facade_top_altitude_m,
            facade_bottom_altitude_m=facade_bottom_altitude_m,
            facade_standoff_m=facade_standoff_m,
            facade_rotate_points_180=facade_rotate_points_180,
            facade_capture_profile=facade_capture_profile,
            linear_path_lonlat=linear_path_lonlat,
            linear_segmentation_enabled=linear_segmentation_enabled,
            linear_max_segment_length_m=linear_max_segment_length_m,
            lateral_target_path_lonlat=lateral_target_path_lonlat,
            lateral_standoff_m=lateral_standoff_m,
            lateral_target_side=lateral_target_side,
            waypoint_path_lonlat=waypoint_path_lonlat,
            waypoint_heading_mode=waypoint_heading_mode,
            waypoint_fixed_yaw_deg=waypoint_fixed_yaw_deg,
            waypoint_poi_lonlat=waypoint_poi_lonlat,
            waypoint_enable_smoothing=waypoint_enable_smoothing,
            waypoint_turn_radius_m=waypoint_turn_radius_m,
            waypoint_capture_enabled=waypoint_capture_enabled,
            orbit_center_lonlat=orbit_center_lonlat,
            orbit_radius_m=orbit_radius_m,
            orbit_level_count=orbit_level_count,
            orbit_vertical_step_m=orbit_vertical_step_m,
            orbit_poi_yaw_lock=orbit_poi_yaw_lock,
            orbit_poi_lonlat=orbit_poi_lonlat,
            panorama_center_lonlat=panorama_center_lonlat,
            panorama_overlap_pct=panorama_overlap_pct,
            panorama_multi_row_enabled=panorama_multi_row_enabled,
            panorama_row_count=panorama_row_count,
            panorama_pitch_step_deg=panorama_pitch_step_deg,
            bubble_center_lonlat=bubble_center_lonlat,
            bubble_overlap_pct=bubble_overlap_pct,
            bubble_pitch_step_deg=bubble_pitch_step_deg,
            bubble_top_pitch_deg=bubble_top_pitch_deg,
            bubble_bottom_pitch_deg=bubble_bottom_pitch_deg,
            tower_center_lonlat=tower_center_lonlat,
            tower_top_altitude_m=tower_top_altitude_m,
            tower_bottom_altitude_m=tower_bottom_altitude_m,
            tower_object_radius_m=tower_object_radius_m,
            tower_flight_radius_m=tower_flight_radius_m,
            tower_resume_enabled=tower_resume_enabled,
            solar_row_angle_deg=solar_row_angle_deg,
            solar_sensor_profile=solar_sensor_profile,
            solar_orientation_mode=solar_orientation_mode,
            solar_rows_lonlat=solar_rows_lonlat,
            magnetic_tie_line_spacing_m=magnetic_tie_line_spacing_m,
            magnetic_smoothing_radius_m=magnetic_smoothing_radius_m,
            ground_offset_m=ground_offset_m,
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_source_path=terrain_source_path,
            terrain_samples_lonlatz=terrain_samples_lonlatz,
            terrain_model_override=terrain_model_override,
            terrain_follow_mode=terrain_follow_mode,
            terrain_normal_camera_enabled=terrain_normal_camera_enabled,
            terrain_normal_gain=terrain_normal_gain,
            terrain_normal_yaw_align=terrain_normal_yaw_align,
            wind_speed_m_s=wind_speed_m_s,
            wind_direction_deg=wind_direction_deg,
            wind_gust_m_s=wind_gust_m_s,
            facade_curvature_alignment=facade_curvature_alignment,
            facade_curve_path_lonlat=facade_curve_path_lonlat,
            constraints=constraints,
            asset_frame=asset_frame,
        )
        return self.compile_recipe(
            recipe=recipe,
            speed_m_s=speed_m_s,
            camera=camera,
            repeat_enabled=bool(enable_repeat),
        )

    def _merge_linked_constraints(
        self,
        recipes: list[FlightRecipe],
        constraints_override: MissionConstraints | dict | None = None,
    ) -> MissionConstraints:
        all_world_points: list[list[float]] = []
        min_alt = float("inf")
        max_alt = 0.0
        standoff = 0.0
        rth = 10.0
        oa_profile = "balanced"
        no_fly: list[list[list[float]]] = []

        for recipe in recipes:
            for pt in recipe.constraints.geofence:
                if isinstance(pt, list) and len(pt) >= 2:
                    all_world_points.append([float(pt[0]), float(pt[1])])
            min_alt = min(min_alt, float(recipe.constraints.min_altitude_m))
            max_alt = max(max_alt, float(recipe.constraints.max_altitude_m))
            standoff = max(standoff, float(recipe.constraints.standoff_m))
            rth = max(rth, float(recipe.constraints.rth_altitude_m))
            if str(recipe.constraints.obstacle_avoidance_profile).strip():
                oa_profile = str(recipe.constraints.obstacle_avoidance_profile)
            for poly in recipe.constraints.no_fly_polygons:
                try:
                    no_fly.append(_ensure_closed(poly))
                except Exception:
                    continue

        if not all_world_points:
            raise ValueError("Linked mission requires recipes with geofence geometry.")

        world_arr = np.asarray(all_world_points, dtype=np.float64)
        lon0 = float(np.mean(world_arr[:, 0]))
        lat0 = float(np.mean(world_arr[:, 1]))
        world_xy = _lonlat_to_xy(world_arr, lon0=lon0, lat0=lat0)
        pad = max(20.0, standoff + 10.0)
        min_x = float(np.min(world_xy[:, 0])) - pad
        max_x = float(np.max(world_xy[:, 0])) + pad
        min_y = float(np.min(world_xy[:, 1])) - pad
        max_y = float(np.max(world_xy[:, 1])) + pad
        geofence_xy = np.asarray(
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ],
            dtype=np.float64,
        )
        geofence = [[float(p[0]), float(p[1])] for p in _xy_to_lonlat(geofence_xy, lon0=lon0, lat0=lat0).tolist()]

        if not np.isfinite(min_alt):
            min_alt = 30.0
        if max_alt < min_alt + 1.0:
            max_alt = min_alt + 1.0

        merged = MissionConstraints(
            geofence=geofence,
            min_altitude_m=float(max(5.0, min_alt)),
            max_altitude_m=float(max(6.0, max_alt)),
            standoff_m=float(max(0.0, standoff)),
            rth_altitude_m=float(max(10.0, rth)),
            no_fly_polygons=no_fly,
            rth_action="return_home",
            obstacle_avoidance_profile=oa_profile,
        )
        if constraints_override is None:
            return merged

        override = self._coerce_constraints(
            constraints_override,
            polygon=merged.geofence,
            default_altitude_m=float(max(5.0, min_alt)),
        )
        if not override.no_fly_polygons and merged.no_fly_polygons:
            override = replace(override, no_fly_polygons=merged.no_fly_polygons)
        return override

    def _linked_order_indices(self, segments: list[list[_CapturePose]]) -> list[int]:
        valid = [i for i, seg in enumerate(segments) if seg]
        if not valid:
            return []
        order = [valid[0]]
        remaining = set(valid[1:])
        while remaining:
            current = order[-1]
            current_end = np.asarray([segments[current][-1].x_m, segments[current][-1].y_m], dtype=np.float64)
            next_idx = min(
                remaining,
                key=lambda idx: float(
                    np.linalg.norm(
                        current_end - np.asarray([segments[idx][0].x_m, segments[idx][0].y_m], dtype=np.float64)
                    )
                ),
            )
            order.append(next_idx)
            remaining.remove(next_idx)
        return order

    def generate_linked_mission(
        self,
        recipes: Iterable[FlightRecipe | dict],
        speed_m_s: float = 5.0,
        camera: str = "custom",
        optimize_order: bool = True,
        constraints: MissionConstraints | dict | None = None,
        simulate_dry_run: bool = True,
    ) -> MissionPlan:
        recipe_objs: list[FlightRecipe] = [self._coerce_recipe(r) for r in recipes]
        if len(recipe_objs) < 2:
            raise ValueError("Linked mission requires at least 2 mission recipes.")

        segment_world_poses: list[list[dict[str, Any]]] = []
        for recipe_obj in recipe_objs:
            camera_name = str(recipe_obj.metadata.get("camera") or camera or "custom")
            segment_plan = self.compile_recipe(
                recipe=recipe_obj,
                speed_m_s=float(max(0.5, speed_m_s)),
                camera=camera_name,
                repeat_enabled=False,
            )
            world_poses = _extract_world_poses_from_geojson(segment_plan.geojson)
            if not world_poses and segment_plan.waypoints:
                world_poses = [
                    {
                        "lon": float(pt[0]),
                        "lat": float(pt[1]),
                        "alt_m": float(pt[2]) if len(pt) >= 3 else float(segment_plan.altitude_m),
                        "yaw_deg": 0.0,
                        "gimbal_pitch_deg": -90.0,
                        "primitive": "linked_segment",
                        "trigger": True,
                        "dwell_s": 0.0,
                        "camera_yaw_locked": False,
                    }
                    for pt in segment_plan.waypoints
                ]
            if world_poses:
                segment_world_poses.append(world_poses)

        if len(segment_world_poses) < 2:
            raise ValueError("Linked mission requires at least 2 segments with valid waypoints.")

        merged_constraints = self._merge_linked_constraints(recipe_objs, constraints_override=constraints)
        linked_ids = [str(r.recipe_id) for r in recipe_objs]
        linked_digest = hashlib.sha1(json.dumps(linked_ids, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        frame = self.derive_asset_frame(
            polygon_lonlat=merged_constraints.geofence,
            asset_id=f"linked-{linked_digest}",
            coordinate_source="linked_mission",
        )

        segment_local: list[list[_CapturePose]] = []
        for seg in segment_world_poses:
            pts = [[float(p["lon"]), float(p["lat"])] for p in seg]
            local_xy = _world_to_local(pts, frame)
            local_poses: list[_CapturePose] = []
            for item, xy in zip(seg, local_xy):
                local_poses.append(
                    _CapturePose(
                        x_m=float(xy[0]),
                        y_m=float(xy[1]),
                        alt_m=float(max(1.0, item.get("alt_m", 60.0))),
                        yaw_deg=_wrap_deg(float(item.get("yaw_deg", 0.0)) - float(frame.yaw_deg)),
                        gimbal_pitch_deg=float(item.get("gimbal_pitch_deg", -90.0)),
                        primitive=str(item.get("primitive", "linked_segment")),
                        trigger=bool(item.get("trigger", True)),
                        dwell_s=float(item.get("dwell_s", 0.0)),
                        camera_yaw_locked=bool(item.get("camera_yaw_locked", False)),
                    )
                )
            segment_local.append(local_poses)

        order = self._linked_order_indices(segment_local) if optimize_order else [i for i, s in enumerate(segment_local) if s]
        ordered_segments = [segment_local[i] for i in order]
        ordered_recipe_ids = [linked_ids[i] for i in order]

        no_fly_local: list[np.ndarray] = []
        for poly in merged_constraints.no_fly_polygons:
            try:
                local = _world_to_local(poly, frame)
                no_fly_local.append(np.asarray(_ensure_closed_xy(local.tolist()), dtype=np.float64))
            except Exception:
                continue

        linked_local_poses: list[_CapturePose] = []
        transition_count = 0
        obstacle_hits_dry_run = 0
        altitude_violations_dry_run = 0
        for seg_idx, seg in enumerate(ordered_segments):
            if not seg:
                continue
            if linked_local_poses:
                prev = linked_local_poses[-1]
                start = seg[0]
                transition_alt = float(
                    np.clip(
                        max(float(prev.alt_m), float(start.alt_m), float(merged_constraints.min_altitude_m) + 5.0),
                        float(merged_constraints.min_altitude_m),
                        float(merged_constraints.max_altitude_m),
                    )
                )
                if transition_alt < merged_constraints.min_altitude_m - 1e-6 or transition_alt > merged_constraints.max_altitude_m + 1e-6:
                    altitude_violations_dry_run += 1

                start_xy = np.asarray([prev.x_m, prev.y_m], dtype=np.float64)
                end_xy = np.asarray([start.x_m, start.y_m], dtype=np.float64)
                for obstacle in no_fly_local:
                    if _segment_intersects_polygon(start_xy, end_xy, obstacle):
                        obstacle_hits_dry_run += 1
                        break

                trans_primitive = f"linked_transition_{transition_count + 1}"
                if abs(float(prev.alt_m) - transition_alt) > 0.25:
                    linked_local_poses.append(
                        _CapturePose(
                            x_m=float(prev.x_m),
                            y_m=float(prev.y_m),
                            alt_m=transition_alt,
                            yaw_deg=float(prev.yaw_deg),
                            gimbal_pitch_deg=float(prev.gimbal_pitch_deg),
                            primitive=trans_primitive,
                            trigger=False,
                            dwell_s=0.0,
                            camera_yaw_locked=False,
                        )
                    )

                travel_yaw = _bearing_deg(
                    np.asarray([linked_local_poses[-1].x_m, linked_local_poses[-1].y_m], dtype=np.float64),
                    np.asarray([start.x_m, start.y_m], dtype=np.float64),
                )
                linked_local_poses.append(
                    _CapturePose(
                        x_m=float(start.x_m),
                        y_m=float(start.y_m),
                        alt_m=transition_alt,
                        yaw_deg=float(travel_yaw),
                        gimbal_pitch_deg=float(start.gimbal_pitch_deg),
                        primitive=trans_primitive,
                        trigger=False,
                        dwell_s=0.0,
                        camera_yaw_locked=False,
                    )
                )
                if abs(float(start.alt_m) - transition_alt) > 0.25:
                    linked_local_poses.append(
                        _CapturePose(
                            x_m=float(start.x_m),
                            y_m=float(start.y_m),
                            alt_m=float(start.alt_m),
                            yaw_deg=float(start.yaw_deg),
                            gimbal_pitch_deg=float(start.gimbal_pitch_deg),
                            primitive=trans_primitive,
                            trigger=False,
                            dwell_s=0.0,
                            camera_yaw_locked=False,
                        )
                    )
                transition_count += 1

            for pose in seg:
                linked_local_poses.append(
                    replace(
                        pose,
                        primitive=f"linked_seg{seg_idx + 1}:{pose.primitive}",
                    )
                )

        if not linked_local_poses:
            raise ValueError("Linked mission contained no valid capture poses.")

        primitive_payload = [
            {
                "x_m": float(p.x_m),
                "y_m": float(p.y_m),
                "alt_m": float(p.alt_m),
                "yaw_deg": float(p.yaw_deg),
                "gimbal_pitch_deg": float(p.gimbal_pitch_deg),
                "primitive": str(p.primitive),
                "trigger": bool(p.trigger),
                "dwell_s": float(p.dwell_s),
                "camera_yaw_locked": bool(p.camera_yaw_locked),
            }
            for p in linked_local_poses
        ]

        front_overlap = float(np.mean([r.coverage.front_overlap_pct for r in recipe_objs]))
        side_overlap = float(np.mean([r.coverage.side_overlap_pct for r in recipe_objs]))
        coverage = CoverageExpectation(
            front_overlap_pct=front_overlap,
            side_overlap_pct=side_overlap,
            minimum_coverage_pct=95.0,
            required_viewpoints=max(1, int(sum(1 for p in linked_local_poses if p.trigger))),
        )

        seed = {
            "recipe_ids": ordered_recipe_ids,
            "transition_count": int(transition_count),
            "dry_run": bool(simulate_dry_run),
        }
        recipe_id = f"fr-linked-{hashlib.sha1(json.dumps(seed, sort_keys=True).encode('utf-8')).hexdigest()[:12]}"
        metadata: dict[str, Any] = {
            "camera": str(camera or "custom").lower(),
            "generated_from": "mission_planner_linked",
            "capture_dataset": "linked_mission",
            "smooth_motion_profile": "segment_transition",
            "linked_segment_recipe_ids": ordered_recipe_ids,
            "linked_segment_count": int(len(ordered_recipe_ids)),
            "linked_transition_count": int(transition_count),
            "linked_ordering_strategy": "nearest_neighbor" if optimize_order else "input_order",
            "linked_order_indices": [int(i) for i in order],
            "linked_dry_run_ok": bool((altitude_violations_dry_run == 0) and (obstacle_hits_dry_run == 0)),
            "linked_dry_run_altitude_violations": int(altitude_violations_dry_run),
            "linked_dry_run_obstacle_hits": int(obstacle_hits_dry_run),
            "linked_dry_run_simulated": bool(simulate_dry_run),
        }

        linked_recipe = FlightRecipe(
            recipe_id=recipe_id,
            version=1,
            template="linked_mission",
            asset_frame=frame,
            primitives=[
                MissionPrimitive(
                    kind="linked_route",
                    params={
                        "poses_local": primitive_payload,
                        "motion_profile": "segment_transition",
                        "continuous_capture": False,
                    },
                )
            ],
            constraints=merged_constraints,
            coverage=coverage,
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            metadata=metadata,
        )
        return self.compile_recipe(
            recipe=linked_recipe,
            speed_m_s=float(max(0.5, speed_m_s)),
            camera=str(camera or "custom"),
            repeat_enabled=False,
        )

    def _coerce_constraints(
        self,
        constraints: MissionConstraints | dict | None,
        polygon: list[list[float]],
        default_altitude_m: float,
    ) -> MissionConstraints:
        if isinstance(constraints, MissionConstraints):
            geofence = constraints.geofence if constraints.geofence else polygon
            no_fly_polygons: list[list[list[float]]] = []
            for poly in constraints.no_fly_polygons:
                try:
                    no_fly_polygons.append(_ensure_closed(poly))
                except Exception:
                    continue
            return MissionConstraints(
                geofence=_ensure_closed(geofence),
                min_altitude_m=max(5.0, float(constraints.min_altitude_m)),
                max_altitude_m=max(float(constraints.max_altitude_m), float(constraints.min_altitude_m) + 1.0),
                standoff_m=max(0.0, float(constraints.standoff_m)),
                rth_altitude_m=max(10.0, float(constraints.rth_altitude_m)),
                no_fly_polygons=no_fly_polygons,
                rth_action=str(constraints.rth_action or "return_home"),
                obstacle_avoidance_profile=str(constraints.obstacle_avoidance_profile or "balanced"),
            )

        if isinstance(constraints, dict):
            return _constraints_from_dict(constraints, fallback_polygon=polygon, default_altitude_m=default_altitude_m)

        return _constraints_from_dict({}, fallback_polygon=polygon, default_altitude_m=default_altitude_m)

    def _coerce_recipe(self, recipe: FlightRecipe | dict) -> FlightRecipe:
        if isinstance(recipe, FlightRecipe):
            return recipe
        if isinstance(recipe, dict):
            return _flight_recipe_from_dict(recipe)
        raise TypeError("recipe must be FlightRecipe or dict")

    def _default_recipe_altitude(self, recipe: FlightRecipe) -> float:
        for primitive in recipe.primitives:
            alt = primitive.params.get("altitude_m")
            if alt is not None:
                return float(alt)
            levels = primitive.params.get("altitude_levels_m")
            if isinstance(levels, list) and levels:
                return float(levels[0])
        return float(max(5.0, recipe.constraints.min_altitude_m))

    def _mapping_capture_config(
        self,
        recipe_obj: FlightRecipe,
        speed_m_s: float,
        estimated_gsd_cm: float,
    ) -> dict[str, float | bool | dict[str, Any]]:
        template = _normalize_template(recipe_obj.template)
        allow_continuous = template in {"grid", "double_grid", "facade_mapping", "solar_inspection", "lateral_capture"}
        cross_angle_default = (
            float(recipe_obj.metadata.get("double_grid_cross_angle_deg", DOUBLE_GRID_DEFAULT_CROSS_ANGLE_DEG))
            if template == "double_grid"
            else 0.0
        )
        config: dict[str, float | bool | dict[str, Any]] = {
            "flight_direction_deg": float(recipe_obj.metadata.get("flight_direction_deg", 0.0)),
            "camera_direction_deg": float(recipe_obj.metadata.get("camera_direction_deg", recipe_obj.metadata.get("flight_direction_deg", 0.0))),
            "camera_direction_locked": bool(recipe_obj.metadata.get("camera_direction_locked", False)),
            "gimbal_tilt_deg": float(recipe_obj.metadata.get("gimbal_tilt_deg", -90.0)),
            "inspection_dwell_s": float(recipe_obj.metadata.get("inspection_dwell_s", 0.0)),
            "facade_top_altitude_m": float(recipe_obj.metadata.get("facade_top_altitude_m", 0.0)),
            "facade_bottom_altitude_m": float(recipe_obj.metadata.get("facade_bottom_altitude_m", 0.0)),
            "facade_standoff_m": float(recipe_obj.metadata.get("facade_standoff_m", 0.0)),
            "facade_rotate_points_180": bool(recipe_obj.metadata.get("facade_rotate_points_180", False)),
            "facade_capture_profile": str(recipe_obj.metadata.get("facade_capture_profile", "custom")),
            "smooth_motion_profile": str(recipe_obj.metadata.get("smooth_motion_profile", "")),
            "linear_segmentation_enabled": bool(recipe_obj.metadata.get("linear_segmentation_enabled", False)),
            "linear_max_segment_length_m": float(recipe_obj.metadata.get("linear_max_segment_length_m", 0.0)),
            "linear_segment_count": 0.0,
            "linear_path_length_m": 0.0,
            "lateral_standoff_m": float(recipe_obj.metadata.get("lateral_standoff_m", 0.0)),
            "lateral_target_side": str(recipe_obj.metadata.get("lateral_target_side", "right")),
            "lateral_yaw_offset_deg": float(recipe_obj.metadata.get("lateral_yaw_offset_deg", 0.0)),
            "lateral_path_length_m": 0.0,
            "waypoint_heading_mode": str(recipe_obj.metadata.get("waypoint_heading_mode", "tangent")),
            "waypoint_fixed_yaw_deg": float(recipe_obj.metadata.get("waypoint_fixed_yaw_deg", 0.0)),
            "waypoint_turn_radius_m": float(recipe_obj.metadata.get("waypoint_turn_radius_m", 0.0)),
            "waypoint_smoothing_enabled": bool(recipe_obj.metadata.get("waypoint_enable_smoothing", False)),
            "waypoint_capture_enabled": bool(recipe_obj.metadata.get("waypoint_capture_enabled", True)),
            "waypoint_path_length_m": 0.0,
            "orbit_radius_m": float(recipe_obj.metadata.get("orbit_radius_m", 0.0)),
            "orbit_level_count": float(recipe_obj.metadata.get("orbit_level_count", 0.0)),
            "orbit_vertical_step_m": float(recipe_obj.metadata.get("orbit_vertical_step_m", 0.0)),
            "orbit_poi_yaw_lock": bool(recipe_obj.metadata.get("orbit_poi_yaw_lock", True)),
            "panorama_overlap_pct": float(recipe_obj.metadata.get("panorama_overlap_pct", 0.0)),
            "panorama_multi_row_enabled": bool(recipe_obj.metadata.get("panorama_multi_row_enabled", False)),
            "panorama_row_count": float(recipe_obj.metadata.get("panorama_row_count", 0.0)),
            "panorama_pitch_step_deg": float(recipe_obj.metadata.get("panorama_pitch_step_deg", 0.0)),
            "panorama_yaw_step_deg": 0.0,
            "panorama_yaw_count": 0.0,
            "bubble_overlap_pct": float(recipe_obj.metadata.get("bubble_overlap_pct", 0.0)),
            "bubble_pitch_step_deg": float(recipe_obj.metadata.get("bubble_pitch_step_deg", 0.0)),
            "bubble_top_pitch_deg": float(recipe_obj.metadata.get("bubble_top_pitch_deg", 0.0)),
            "bubble_bottom_pitch_deg": float(recipe_obj.metadata.get("bubble_bottom_pitch_deg", 0.0)),
            "bubble_pitch_count": 0.0,
            "bubble_yaw_step_deg": 0.0,
            "bubble_yaw_count": 0.0,
            "tower_top_altitude_m": float(recipe_obj.metadata.get("tower_top_altitude_m", 0.0)),
            "tower_bottom_altitude_m": float(recipe_obj.metadata.get("tower_bottom_altitude_m", 0.0)),
            "tower_object_radius_m": float(recipe_obj.metadata.get("tower_object_radius_m", 0.0)),
            "tower_flight_radius_m": float(recipe_obj.metadata.get("tower_flight_radius_m", 0.0)),
            "tower_orbit_count": 0.0,
            "tower_resume_enabled": bool(recipe_obj.metadata.get("tower_resume_enabled", False)),
            "tower_safe_rth_altitude_m": float(recipe_obj.metadata.get("tower_safe_rth_altitude_m", 0.0)),
            "solar_row_angle_deg": float(recipe_obj.metadata.get("solar_row_angle_deg", recipe_obj.metadata.get("flight_direction_deg", 0.0))),
            "solar_sensor_profile": str(recipe_obj.metadata.get("solar_sensor_profile", "rgb")),
            "solar_orientation_mode": str(recipe_obj.metadata.get("solar_orientation_mode", "row_aligned")),
            "magnetic_tie_line_spacing_m": float(recipe_obj.metadata.get("magnetic_tie_line_spacing_m", 0.0)),
            "magnetic_smoothing_radius_m": float(recipe_obj.metadata.get("magnetic_smoothing_radius_m", 0.0)),
            "ground_offset_m": float(recipe_obj.metadata.get("ground_offset_m", 0.0)),
            "terrain_follow_enabled": bool(recipe_obj.metadata.get("terrain_follow_enabled", False)),
            "terrain_follow_mode": str(recipe_obj.metadata.get("terrain_follow_mode", "agl")),
            "terrain_normal_camera_enabled": bool(recipe_obj.metadata.get("terrain_normal_camera_enabled", False)),
            "terrain_normal_gain": float(recipe_obj.metadata.get("terrain_normal_gain", 1.0)),
            "terrain_normal_yaw_align": bool(recipe_obj.metadata.get("terrain_normal_yaw_align", False)),
            "terrain_source_path": str(recipe_obj.metadata.get("terrain_source_path", "")),
            "terrain_model_type": str((recipe_obj.metadata.get("terrain_model") or {}).get("type", "flat"))
            if isinstance(recipe_obj.metadata.get("terrain_model"), dict)
            else "flat",
            "terrain_model_source": str((recipe_obj.metadata.get("terrain_model") or {}).get("source", "none"))
            if isinstance(recipe_obj.metadata.get("terrain_model"), dict)
            else "none",
            "line_spacing_m": float(recipe_obj.metadata.get("line_spacing_m", 0.0)),
            "capture_spacing_m": float(recipe_obj.metadata.get("capture_spacing_m", 0.0)),
            "capture_interval_s": 0.0,
            "continuous_capture": False,
            "double_grid_cross_angle_deg": cross_angle_default,
            "recommended_speed_m_s": float(recipe_obj.metadata.get("recommended_speed_m_s", 0.0)),
            "wind_speed_m_s": float(recipe_obj.metadata.get("wind_speed_m_s", 0.0)),
            "wind_direction_deg": float(recipe_obj.metadata.get("wind_direction_deg", 0.0)),
            "wind_gust_m_s": float(recipe_obj.metadata.get("wind_gust_m_s", 0.0)),
            "wind_adjusted_speed_m_s": 0.0,
            "wind_penalty_pct": 0.0,
            "facade_curvature_alignment": bool(recipe_obj.metadata.get("facade_curvature_alignment", False)),
            "camera_policy": {},
        }

        primary_heading = float(config["flight_direction_deg"])
        secondary_heading: float | None = None
        seen_grid_passes = 0
        for primitive in recipe_obj.primitives:
            kind = _normalize_template(primitive.kind)
            if kind not in {
                "grid",
                "double_grid",
                "roof_inspection",
                "facade",
                "facade_mapping",
                "orbit",
                "panorama",
                "bubble_360",
                "linear_inspection",
                "lateral_capture",
                "waypoints",
                "tower_mapping",
                "solar_inspection",
                "magnetic_mapping",
                "linked_mission",
            }:
                continue

            params = primitive.params
            heading = float(params.get("flight_direction_deg", config.get("flight_direction_deg", 0.0)))
            if kind in {"grid", "double_grid"}:
                seen_grid_passes += 1
                if seen_grid_passes == 1:
                    primary_heading = heading
                elif secondary_heading is None:
                    secondary_heading = heading

            config["flight_direction_deg"] = float(
                params.get("flight_direction_deg", config.get("flight_direction_deg", 0.0))
            )
            config["camera_direction_deg"] = float(
                params.get("camera_direction_deg", config.get("camera_direction_deg", config["flight_direction_deg"]))
            )
            config["camera_direction_locked"] = bool(
                params.get("camera_direction_locked", config.get("camera_direction_locked", False))
            )
            config["gimbal_tilt_deg"] = float(params.get("gimbal_pitch_deg", config.get("gimbal_tilt_deg", -90.0)))
            config["inspection_dwell_s"] = float(
                params.get("inspection_dwell_s", config.get("inspection_dwell_s", 0.0))
            )
            config["facade_top_altitude_m"] = float(
                params.get("top_altitude_m", config.get("facade_top_altitude_m", 0.0))
            )
            config["facade_bottom_altitude_m"] = float(
                params.get("bottom_altitude_m", config.get("facade_bottom_altitude_m", 0.0))
            )
            config["facade_standoff_m"] = float(
                params.get("standoff_m", config.get("facade_standoff_m", 0.0))
            )
            config["facade_rotate_points_180"] = bool(
                params.get("rotate_points_180", config.get("facade_rotate_points_180", False))
            )
            config["facade_capture_profile"] = str(
                params.get("capture_profile", config.get("facade_capture_profile", "custom"))
            )
            config["smooth_motion_profile"] = str(
                params.get("motion_profile", config.get("smooth_motion_profile", ""))
            )
            config["linear_segmentation_enabled"] = bool(
                params.get("linear_segmentation_enabled", config.get("linear_segmentation_enabled", False))
            )
            config["linear_max_segment_length_m"] = float(
                params.get("linear_max_segment_length_m", config.get("linear_max_segment_length_m", 0.0))
            )
            config["lateral_standoff_m"] = float(
                params.get("standoff_m", config.get("lateral_standoff_m", 0.0))
            )
            config["lateral_target_side"] = str(
                params.get("target_side", config.get("lateral_target_side", "right"))
            )
            config["lateral_yaw_offset_deg"] = float(
                params.get("yaw_offset_deg", config.get("lateral_yaw_offset_deg", 0.0))
            )
            config["waypoint_heading_mode"] = str(
                params.get("heading_mode", config.get("waypoint_heading_mode", "tangent"))
            )
            config["waypoint_fixed_yaw_deg"] = float(
                params.get("fixed_yaw_deg", config.get("waypoint_fixed_yaw_deg", 0.0))
            )
            config["waypoint_turn_radius_m"] = float(
                params.get("turn_radius_m", config.get("waypoint_turn_radius_m", 0.0))
            )
            config["waypoint_smoothing_enabled"] = bool(
                params.get("enable_smoothing", config.get("waypoint_smoothing_enabled", False))
            )
            config["waypoint_capture_enabled"] = bool(
                params.get("capture_enabled", config.get("waypoint_capture_enabled", True))
            )
            config["orbit_radius_m"] = float(
                params.get("radius_m", config.get("orbit_radius_m", 0.0))
            )
            config["orbit_poi_yaw_lock"] = bool(
                params.get("yaw_lock_to_poi", config.get("orbit_poi_yaw_lock", True))
            )
            config["panorama_multi_row_enabled"] = bool(
                params.get("panorama_multi_row_enabled", config.get("panorama_multi_row_enabled", False))
            )
            config["panorama_overlap_pct"] = float(
                params.get("panorama_overlap_pct", config.get("panorama_overlap_pct", 0.0))
            )
            config["panorama_pitch_step_deg"] = float(
                params.get("panorama_pitch_step_deg", config.get("panorama_pitch_step_deg", 0.0))
            )
            config["bubble_overlap_pct"] = float(
                params.get("bubble_overlap_pct", config.get("bubble_overlap_pct", 0.0))
            )
            config["bubble_pitch_step_deg"] = float(
                params.get("bubble_pitch_step_deg", config.get("bubble_pitch_step_deg", 0.0))
            )
            config["bubble_top_pitch_deg"] = float(
                params.get("bubble_top_pitch_deg", config.get("bubble_top_pitch_deg", 0.0))
            )
            config["bubble_bottom_pitch_deg"] = float(
                params.get("bubble_bottom_pitch_deg", config.get("bubble_bottom_pitch_deg", 0.0))
            )
            config["tower_object_radius_m"] = float(
                params.get("object_radius_m", config.get("tower_object_radius_m", 0.0))
            )
            config["tower_flight_radius_m"] = float(
                params.get("radius_m", config.get("tower_flight_radius_m", 0.0))
            )
            config["tower_resume_enabled"] = bool(
                params.get("tower_resume_enabled", config.get("tower_resume_enabled", False))
            )
            config["solar_row_angle_deg"] = float(
                params.get("row_angle_deg", config.get("solar_row_angle_deg", config.get("flight_direction_deg", 0.0)))
            )
            config["solar_sensor_profile"] = str(
                params.get("sensor_profile", config.get("solar_sensor_profile", "rgb"))
            )
            config["solar_orientation_mode"] = str(
                params.get("orientation_mode", config.get("solar_orientation_mode", "row_aligned"))
            )
            config["magnetic_tie_line_spacing_m"] = float(
                params.get("tie_line_spacing_m", config.get("magnetic_tie_line_spacing_m", 0.0))
            )
            config["magnetic_smoothing_radius_m"] = float(
                params.get("turn_smoothing_radius_m", config.get("magnetic_smoothing_radius_m", 0.0))
            )
            config["ground_offset_m"] = float(params.get("ground_offset_m", config.get("ground_offset_m", 0.0)))
            config["terrain_follow_enabled"] = bool(
                params.get("terrain_follow_enabled", config.get("terrain_follow_enabled", False))
            )
            config["terrain_follow_mode"] = str(
                params.get("terrain_follow_mode", config.get("terrain_follow_mode", "agl"))
            )
            config["terrain_normal_camera_enabled"] = bool(
                params.get("terrain_normal_camera_enabled", config.get("terrain_normal_camera_enabled", False))
            )
            config["terrain_normal_gain"] = float(
                params.get("terrain_normal_gain", config.get("terrain_normal_gain", 1.0))
            )
            config["terrain_normal_yaw_align"] = bool(
                params.get("terrain_normal_yaw_align", config.get("terrain_normal_yaw_align", False))
            )
            config["line_spacing_m"] = float(
                params.get(
                    "line_spacing_m",
                    params.get("line_step_m", config.get("line_spacing_m", 0.0)),
                )
            )
            config["capture_spacing_m"] = float(
                params.get(
                    "capture_spacing_m",
                    params.get("point_step_m", config.get("capture_spacing_m", 0.0)),
                )
            )
            config["continuous_capture"] = bool(params.get("continuous_capture", False)) and allow_continuous
            if kind == "linear_inspection":
                line_local = params.get("line_local")
                if isinstance(line_local, list) and len(line_local) >= 2:
                    try:
                        config["linear_path_length_m"] = float(_distance_xy(line_local))
                    except Exception:
                        pass
            if kind == "waypoints":
                path_local = params.get("path_local")
                if isinstance(path_local, list) and len(path_local) >= 2:
                    try:
                        config["waypoint_path_length_m"] = float(_distance_xy(path_local))
                    except Exception:
                        pass
            if kind == "lateral_capture":
                target_line_local = params.get("target_line_local")
                if isinstance(target_line_local, list) and len(target_line_local) >= 2:
                    try:
                        config["lateral_path_length_m"] = float(_distance_xy(target_line_local))
                    except Exception:
                        pass
            if kind == "orbit":
                levels = params.get("altitude_levels_m")
                if isinstance(levels, list) and levels:
                    try:
                        config["orbit_level_count"] = float(len(levels))
                        if len(levels) >= 2:
                            steps = [abs(float(levels[i]) - float(levels[i - 1])) for i in range(1, len(levels))]
                            config["orbit_vertical_step_m"] = float(np.mean(np.asarray(steps, dtype=np.float64)))
                    except Exception:
                        pass
            if kind == "panorama":
                try:
                    config["panorama_yaw_step_deg"] = float(params.get("yaw_step_deg", config.get("panorama_yaw_step_deg", 0.0)))
                    config["panorama_yaw_count"] = float(params.get("yaw_count", config.get("panorama_yaw_count", 0.0)))
                    row_pitches = params.get("row_pitches_deg", [])
                    if isinstance(row_pitches, list):
                        config["panorama_row_count"] = float(len(row_pitches))
                except Exception:
                    pass
            if kind == "bubble_360":
                try:
                    config["bubble_yaw_step_deg"] = float(params.get("yaw_step_deg", config.get("bubble_yaw_step_deg", 0.0)))
                    config["bubble_yaw_count"] = float(params.get("yaw_count", config.get("bubble_yaw_count", 0.0)))
                    row_pitches = params.get("row_pitches_deg", [])
                    if isinstance(row_pitches, list):
                        config["bubble_pitch_count"] = float(len(row_pitches))
                        if row_pitches:
                            config["bubble_top_pitch_deg"] = float(max(row_pitches))
                            config["bubble_bottom_pitch_deg"] = float(min(row_pitches))
                except Exception:
                    pass
            if kind == "tower_mapping":
                levels = params.get("altitude_levels_m")
                if isinstance(levels, list) and levels:
                    try:
                        config["tower_orbit_count"] = float(len(levels))
                        config["tower_top_altitude_m"] = float(max(levels))
                        config["tower_bottom_altitude_m"] = float(min(levels))
                    except Exception:
                        pass
            if kind in {"facade", "facade_mapping"}:
                config["facade_curvature_alignment"] = bool(
                    params.get("curvature_alignment", config.get("facade_curvature_alignment", False))
                )

        if secondary_heading is not None:
            config["double_grid_cross_angle_deg"] = abs(_wrap_deg(secondary_heading - primary_heading))
        if seen_grid_passes > 0:
            config["flight_direction_deg"] = _wrap_deg(primary_heading)

        speed = max(0.5, float(speed_m_s))
        spacing = max(0.0, float(config["capture_spacing_m"]))
        config["capture_interval_s"] = float(spacing / speed) if spacing > 0.0 else 0.0

        if template == "double_grid":
            camera_policy = _double_grid_camera_policy(speed_m_s=speed, gsd_cm=estimated_gsd_cm)
            existing_policy = recipe_obj.metadata.get("camera_policy")
            if isinstance(existing_policy, dict):
                merged_policy = dict(existing_policy)
                merged_policy.update(camera_policy)
                camera_policy = merged_policy
            config["camera_policy"] = camera_policy
        if template == "facade_mapping":
            capture_profile = _normalize_facade_capture_profile(str(config.get("facade_capture_profile", "custom")))
            camera_policy = _facade_mapping_camera_policy(
                speed_m_s=float(config.get("recommended_speed_m_s", 0.0)) or speed,
                gsd_cm=estimated_gsd_cm,
                capture_profile=capture_profile,
            )
            existing_policy = recipe_obj.metadata.get("camera_policy")
            if isinstance(existing_policy, dict):
                merged_policy = dict(existing_policy)
                merged_policy.update(camera_policy)
                camera_policy = merged_policy
            config["camera_policy"] = camera_policy

        return config

    def _build_primitives(
        self,
        template: str,
        local_polygon: list[list[float]],
        camera_name: str,
        altitude_m: float,
        line_step_m: float,
        point_step_m: float,
        constraints: MissionConstraints,
        flight_direction_deg: float = 0.0,
        camera_direction_deg: float = 0.0,
        camera_direction_locked: bool = False,
        gimbal_pitch_deg: float = -90.0,
        inspection_dwell_s: float = 1.5,
        facade_top_altitude_m: float | None = None,
        facade_bottom_altitude_m: float | None = None,
        facade_standoff_m: float | None = None,
        facade_rotate_points_180: bool = False,
        facade_capture_profile: str = "custom",
        linear_path_local: list[list[float]] | None = None,
        linear_segmentation_enabled: bool = True,
        linear_max_segment_length_m: float = 1500.0,
        lateral_target_local: list[list[float]] | None = None,
        lateral_standoff_m: float = 10.0,
        lateral_target_side: str = "right",
        waypoint_path_local: list[list[float]] | None = None,
        waypoint_heading_mode: str = "tangent",
        waypoint_fixed_yaw_deg: float = 0.0,
        waypoint_poi_local: list[float] | None = None,
        waypoint_enable_smoothing: bool = False,
        waypoint_turn_radius_m: float = 6.0,
        waypoint_capture_enabled: bool = True,
        orbit_center_local: list[float] | None = None,
        orbit_radius_m: float | None = None,
        orbit_level_count: int = 1,
        orbit_vertical_step_m: float = 3.0,
        orbit_poi_yaw_lock: bool = True,
        orbit_poi_local: list[float] | None = None,
        panorama_center_local: list[float] | None = None,
        panorama_overlap_pct: float = 35.0,
        panorama_multi_row_enabled: bool = False,
        panorama_row_count: int = 1,
        panorama_pitch_step_deg: float = 12.0,
        bubble_center_local: list[float] | None = None,
        bubble_overlap_pct: float = 35.0,
        bubble_pitch_step_deg: float = 12.0,
        bubble_top_pitch_deg: float = 20.0,
        bubble_bottom_pitch_deg: float = -90.0,
        tower_center_local: list[float] | None = None,
        tower_top_altitude_m: float | None = None,
        tower_bottom_altitude_m: float | None = None,
        tower_object_radius_m: float = 2.0,
        tower_flight_radius_m: float | None = None,
        tower_resume_enabled: bool = True,
        solar_row_angle_deg: float | None = None,
        solar_sensor_profile: str = "rgb",
        solar_orientation_mode: str = "row_aligned",
        solar_rows_local: list[list[list[float]]] | None = None,
        magnetic_tie_line_spacing_m: float = 50.0,
        magnetic_smoothing_radius_m: float = 8.0,
        ground_offset_m: float = 0.0,
        terrain_follow_enabled: bool = False,
        terrain_follow_mode: str = "agl",
        terrain_model: dict[str, Any] | None = None,
        terrain_normal_camera_enabled: bool = False,
        terrain_normal_gain: float = 1.0,
        terrain_normal_yaw_align: bool = False,
        facade_curvature_alignment: bool = False,
        facade_curve_local: list[list[float]] | None = None,
        double_grid_cross_angle_deg: float = DOUBLE_GRID_DEFAULT_CROSS_ANGLE_DEG,
    ) -> list[MissionPrimitive]:
        template = _normalize_template(template)
        polygon_arr = np.asarray(local_polygon[:-1], dtype=np.float64)
        center = polygon_arr.mean(axis=0)
        max_radius = float(np.linalg.norm(polygon_arr - center, axis=1).max()) if len(polygon_arr) else 10.0
        orbit_radius = max(max_radius + constraints.standoff_m, constraints.standoff_m + 8.0)

        if template == "grid":
            return [
                MissionPrimitive(
                    kind="grid",
                    params={
                        "polygon_local": local_polygon,
                        "line_step_m": line_step_m,
                        "point_step_m": point_step_m,
                        "line_spacing_m": line_step_m,
                        "capture_spacing_m": point_step_m,
                        "altitude_m": altitude_m,
                        "flight_direction_deg": float(flight_direction_deg),
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": True,
                        "double_pass": False,
                    },
                )
            ]

        if template == "double_grid":
            cross = float(np.clip(double_grid_cross_angle_deg, 30.0, 150.0))
            primary_params = {
                "polygon_local": local_polygon,
                "line_step_m": line_step_m,
                "point_step_m": point_step_m,
                "line_spacing_m": line_step_m,
                "capture_spacing_m": point_step_m,
                "altitude_m": altitude_m,
                "flight_direction_deg": float(flight_direction_deg),
                "gimbal_pitch_deg": float(gimbal_pitch_deg),
                "ground_offset_m": float(ground_offset_m),
                "terrain_follow_enabled": bool(terrain_follow_enabled),
                "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                "continuous_capture": True,
                "double_pass": False,
                "grid_pass_index": 1,
                "capture_set": "primary",
            }
            secondary_params = dict(primary_params)
            secondary_params.update(
                {
                    "flight_direction_deg": _wrap_deg(float(flight_direction_deg) + cross),
                    "grid_pass_index": 2,
                    "capture_set": "cross",
                }
            )
            return [
                MissionPrimitive(kind="grid", params=primary_params),
                MissionPrimitive(kind="grid", params=secondary_params),
            ]

        if template == "roof_inspection":
            return [
                MissionPrimitive(
                    kind="roof_inspection",
                    params={
                        "polygon_local": local_polygon,
                        "line_step_m": line_step_m,
                        "point_step_m": point_step_m,
                        "line_spacing_m": line_step_m,
                        "capture_spacing_m": point_step_m,
                        "altitude_m": altitude_m,
                        "flight_direction_deg": float(flight_direction_deg),
                        "camera_direction_deg": float(camera_direction_deg),
                        "camera_direction_locked": bool(camera_direction_locked),
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "inspection_dwell_s": float(inspection_dwell_s),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": False,
                        "stop_and_capture": True,
                    },
                )
            ]

        if template == "linear_inspection":
            if not linear_path_local or len(linear_path_local) < 2:
                linear_path_local = [local_polygon[0], local_polygon[1]]
            return [
                MissionPrimitive(
                    kind="linear_inspection",
                    params={
                        "line_local": linear_path_local,
                        "capture_spacing_m": max(0.75, point_step_m),
                        "altitude_m": altitude_m,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "inspection_dwell_s": float(inspection_dwell_s),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "linear_segmentation_enabled": bool(linear_segmentation_enabled),
                        "linear_max_segment_length_m": float(max(100.0, linear_max_segment_length_m)),
                        "continuous_capture": False,
                        "stop_and_capture": True,
                    },
                )
            ]

        if template == "lateral_capture":
            if not lateral_target_local or len(lateral_target_local) < 2:
                lateral_target_local = linear_path_local if linear_path_local and len(linear_path_local) >= 2 else [local_polygon[0], local_polygon[1]]
            side = str(lateral_target_side or "right").strip().lower()
            if side not in {"left", "right"}:
                side = "right"
            yaw_offset_deg = 90.0 if side == "left" else -90.0
            return [
                MissionPrimitive(
                    kind="lateral_capture",
                    params={
                        "target_line_local": lateral_target_local,
                        "capture_spacing_m": max(0.75, point_step_m),
                        "altitude_m": altitude_m,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "standoff_m": float(max(0.5, lateral_standoff_m)),
                        "target_side": side,
                        "yaw_offset_deg": float(yaw_offset_deg),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": True,
                        "motion_profile": "lateral_profile",
                    },
                )
            ]

        if template == "waypoints":
            if not waypoint_path_local or len(waypoint_path_local) < 2:
                waypoint_path_local = [local_polygon[0], local_polygon[1]]
            return [
                MissionPrimitive(
                    kind="waypoints",
                    params={
                        "path_local": waypoint_path_local,
                        "altitude_m": altitude_m,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "inspection_dwell_s": float(inspection_dwell_s),
                        "heading_mode": str(waypoint_heading_mode),
                        "fixed_yaw_deg": float(waypoint_fixed_yaw_deg),
                        "poi_local": list(waypoint_poi_local) if waypoint_poi_local is not None else None,
                        "enable_smoothing": bool(waypoint_enable_smoothing),
                        "turn_radius_m": float(max(0.0, waypoint_turn_radius_m)),
                        "capture_enabled": bool(waypoint_capture_enabled),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": False,
                        "motion_profile": "curved_waypoint_path" if waypoint_enable_smoothing else "direct_waypoint_path",
                    },
                )
            ]

        if template == "corridor":
            axis_start, axis_end, width = self._corridor_axis(local_polygon)
            lane_spacing = max(3.0, line_step_m)
            lane_count = max(1, int(np.ceil(width / lane_spacing)) + 1)
            return [
                MissionPrimitive(
                    kind="corridor",
                    params={
                        "axis_start_local": axis_start.tolist(),
                        "axis_end_local": axis_end.tolist(),
                        "lane_count": lane_count,
                        "lane_spacing_m": lane_spacing,
                        "capture_spacing_m": point_step_m,
                        "altitude_m": altitude_m,
                    },
                )
            ]

        if template == "orbit":
            center_local_xy = (
                [float(orbit_center_local[0]), float(orbit_center_local[1])]
                if isinstance(orbit_center_local, list) and len(orbit_center_local) >= 2
                else [0.0, 0.0]
            )
            radius = max(1.0, float(orbit_radius_m if orbit_radius_m is not None else orbit_radius))
            level_count = max(1, int(orbit_level_count))
            vertical_step = max(0.5, float(orbit_vertical_step_m))
            base_alt = float(np.clip(altitude_m, constraints.min_altitude_m, constraints.max_altitude_m))
            if level_count <= 1:
                levels = [base_alt]
            else:
                half_span = 0.5 * vertical_step * float(level_count - 1)
                top = min(constraints.max_altitude_m, base_alt + half_span)
                bottom = max(constraints.min_altitude_m, base_alt - half_span)
                if top <= bottom + 1e-6:
                    levels = [base_alt]
                else:
                    levels = np.linspace(top, bottom, level_count, dtype=np.float64).tolist()

            poi_local_xy = (
                [float(orbit_poi_local[0]), float(orbit_poi_local[1])]
                if isinstance(orbit_poi_local, list) and len(orbit_poi_local) >= 2
                else center_local_xy
            )
            points_per_orbit = max(24, int(np.ceil((2.0 * pi * radius) / max(point_step_m, 1.0))))
            return [
                MissionPrimitive(
                    kind="orbit",
                    params={
                        "center_local": center_local_xy,
                        "radius_m": radius,
                        "points_per_orbit": points_per_orbit,
                        "altitude_levels_m": levels,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "yaw_lock_to_poi": bool(orbit_poi_yaw_lock),
                        "poi_local": poi_local_xy,
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "motion_profile": "orbit_stack" if len(levels) > 1 else "orbit_single",
                        "continuous_capture": False,
                    },
                )
            ]

        if template == "panorama":
            center_local_xy = (
                [float(panorama_center_local[0]), float(panorama_center_local[1])]
                if isinstance(panorama_center_local, list) and len(panorama_center_local) >= 2
                else [0.0, 0.0]
            )
            overlap_pct = float(np.clip(panorama_overlap_pct, 5.0, 90.0))
            hfov_deg = max(10.0, _horizontal_fov_deg(str(camera_name or "custom")))
            yaw_step_deg = max(2.0, hfov_deg * (1.0 - overlap_pct / 100.0))
            yaw_count = max(3, int(np.ceil(360.0 / yaw_step_deg)))
            yaw_start_deg = _wrap_deg(float(flight_direction_deg))
            multi_row = bool(panorama_multi_row_enabled)
            row_count = max(1, int(panorama_row_count if multi_row else 1))
            pitch_step = float(np.clip(abs(panorama_pitch_step_deg), 1.0, 45.0))
            base_pitch = float(np.clip(gimbal_pitch_deg, -120.0, 30.0))
            if row_count <= 1:
                row_pitches = [base_pitch]
            else:
                span = pitch_step * float(row_count - 1)
                top = base_pitch + 0.5 * span
                row_pitches = [float(np.clip(top - i * pitch_step, -120.0, 30.0)) for i in range(row_count)]

            return [
                MissionPrimitive(
                    kind="panorama",
                    params={
                        "center_local": center_local_xy,
                        "altitude_m": altitude_m,
                        "yaw_start_deg": yaw_start_deg,
                        "yaw_step_deg": yaw_step_deg,
                        "yaw_count": yaw_count,
                        "row_pitches_deg": row_pitches,
                        "panorama_overlap_pct": overlap_pct,
                        "panorama_multi_row_enabled": bool(multi_row),
                        "panorama_pitch_step_deg": float(pitch_step),
                        "inspection_dwell_s": float(inspection_dwell_s),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": False,
                        "motion_profile": "yaw_sweep_multi_row" if row_count > 1 else "yaw_sweep_single_row",
                    },
                )
            ]

        if template == "bubble_360":
            center_local_xy = (
                [float(bubble_center_local[0]), float(bubble_center_local[1])]
                if isinstance(bubble_center_local, list) and len(bubble_center_local) >= 2
                else [0.0, 0.0]
            )
            overlap_pct = float(np.clip(bubble_overlap_pct, 5.0, 90.0))
            hfov_deg = max(10.0, _horizontal_fov_deg(str(camera_name or "custom")))
            yaw_step_deg = max(2.0, hfov_deg * (1.0 - overlap_pct / 100.0))
            yaw_count = max(3, int(np.ceil(360.0 / yaw_step_deg)))
            yaw_start_deg = _wrap_deg(float(flight_direction_deg))
            top_pitch = float(np.clip(bubble_top_pitch_deg, -120.0, 30.0))
            bottom_pitch = float(np.clip(bubble_bottom_pitch_deg, -120.0, 30.0))
            if top_pitch < bottom_pitch:
                top_pitch, bottom_pitch = bottom_pitch, top_pitch
            pitch_step = float(np.clip(abs(bubble_pitch_step_deg), 1.0, 45.0))
            if top_pitch <= bottom_pitch + 1e-6:
                row_pitches = [top_pitch]
            else:
                row_pitches: list[float] = []
                pitch = top_pitch
                while pitch > bottom_pitch + 1e-6:
                    row_pitches.append(float(np.clip(pitch, -120.0, 30.0)))
                    pitch -= pitch_step
                row_pitches.append(float(np.clip(bottom_pitch, -120.0, 30.0)))

            return [
                MissionPrimitive(
                    kind="bubble_360",
                    params={
                        "center_local": center_local_xy,
                        "altitude_m": altitude_m,
                        "yaw_start_deg": yaw_start_deg,
                        "yaw_step_deg": yaw_step_deg,
                        "yaw_count": yaw_count,
                        "row_pitches_deg": row_pitches,
                        "bubble_overlap_pct": overlap_pct,
                        "bubble_pitch_step_deg": float(pitch_step),
                        "bubble_top_pitch_deg": float(max(row_pitches)),
                        "bubble_bottom_pitch_deg": float(min(row_pitches)),
                        "inspection_dwell_s": float(inspection_dwell_s),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": False,
                        "motion_profile": "yaw_pitch_spherical_sweep",
                    },
                )
            ]

        if template == "tower_mapping":
            center_local_xy = (
                [float(tower_center_local[0]), float(tower_center_local[1])]
                if isinstance(tower_center_local, list) and len(tower_center_local) >= 2
                else [0.0, 0.0]
            )
            top_alt = float(tower_top_altitude_m) if tower_top_altitude_m is not None else min(
                constraints.max_altitude_m, altitude_m + 25.0
            )
            bottom_alt = float(tower_bottom_altitude_m) if tower_bottom_altitude_m is not None else max(
                constraints.min_altitude_m, altitude_m - 15.0
            )
            if top_alt < bottom_alt + 0.5:
                top_alt = bottom_alt + 0.5

            object_radius = float(max(0.5, tower_object_radius_m))
            flight_radius = float(
                max(
                    object_radius + 2.0,
                    tower_flight_radius_m if tower_flight_radius_m is not None else object_radius + constraints.standoff_m,
                )
            )
            circum_step_m = max(0.75, line_step_m)
            vertical_step_m = max(0.75, point_step_m)
            points_per_orbit = max(18, int(np.ceil((2.0 * pi * flight_radius) / circum_step_m)))
            altitude_span = max(0.5, top_alt - bottom_alt)
            level_count = max(2, int(np.ceil(altitude_span / vertical_step_m)) + 1)
            altitude_levels = np.linspace(top_alt, bottom_alt, level_count, dtype=np.float64).tolist()
            target_alt = float((top_alt + bottom_alt) * 0.5)

            return [
                MissionPrimitive(
                    kind="tower_mapping",
                    params={
                        "center_local": center_local_xy,
                        "radius_m": flight_radius,
                        "object_radius_m": object_radius,
                        "points_per_orbit": points_per_orbit,
                        "altitude_levels_m": altitude_levels,
                        "target_altitude_m": target_alt,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "tower_resume_enabled": bool(tower_resume_enabled),
                        "continuous_capture": False,
                        "motion_profile": "orbital_vertical_step",
                    },
                )
            ]

        if template == "solar_inspection":
            row_angle = _wrap_deg(float(solar_row_angle_deg if solar_row_angle_deg is not None else flight_direction_deg))
            sensor_mode = str(solar_sensor_profile or "rgb").strip().lower()
            if sensor_mode not in {"rgb", "thermal"}:
                sensor_mode = "rgb"
            row_constraints = list(solar_rows_local or [])
            return [
                MissionPrimitive(
                    kind="solar_inspection",
                    params={
                        "polygon_local": local_polygon,
                        "row_angle_deg": row_angle,
                        "flight_direction_deg": row_angle,
                        "line_spacing_m": max(0.75, line_step_m),
                        "capture_spacing_m": max(0.75, point_step_m),
                        "altitude_m": altitude_m,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "sensor_profile": sensor_mode,
                        "rows_local": row_constraints,
                        "row_constraint_count": int(len(row_constraints)),
                        "row_snap_tolerance_m": max(0.2, float(max(0.75, line_step_m)) * 0.35),
                        "orientation_mode": str(solar_orientation_mode or "row_aligned"),
                        "yaw_policy": "row_locked_thermal" if sensor_mode == "thermal" else "row_locked_rgb",
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": True,
                        "motion_profile": "row_aligned",
                    },
                )
            ]

        if template == "magnetic_mapping":
            tie_spacing = max(5.0, float(magnetic_tie_line_spacing_m))
            smooth_radius = max(0.0, float(magnetic_smoothing_radius_m))
            return [
                MissionPrimitive(
                    kind="magnetic_mapping",
                    params={
                        "polygon_local": local_polygon,
                        "flight_direction_deg": float(flight_direction_deg),
                        "line_spacing_m": max(2.0, line_step_m),
                        "capture_spacing_m": max(2.0, point_step_m),
                        "tie_line_spacing_m": tie_spacing,
                        "turn_smoothing_radius_m": smooth_radius,
                        "altitude_m": altitude_m,
                        "gimbal_pitch_deg": float(gimbal_pitch_deg),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "continuous_capture": False,
                        "motion_profile": "curvature_turns",
                    },
                )
            ]

        if template in {"facade", "facade_mapping"}:
            baseline_start, baseline_end = self._facade_baseline(local_polygon)
            baseline_curve = self._facade_curve(
                local_polygon=local_polygon,
                curve_local=facade_curve_local,
                curvature_alignment=facade_curvature_alignment,
            )
            top_alt = float(facade_top_altitude_m) if facade_top_altitude_m is not None else min(
                constraints.max_altitude_m, altitude_m + 20.0
            )
            bottom_alt = float(facade_bottom_altitude_m) if facade_bottom_altitude_m is not None else max(
                constraints.min_altitude_m, altitude_m - 20.0
            )
            if top_alt < bottom_alt + 0.5:
                top_alt = bottom_alt + 0.5
            standoff = float(max(0.5, facade_standoff_m if facade_standoff_m is not None else constraints.standoff_m))

            capture_profile = _normalize_facade_capture_profile(facade_capture_profile)
            if template == "facade_mapping":
                if capture_profile == "normal":
                    mapped_gimbal = FACADE_MAPPING_NORMAL_GIMBAL_DEG
                elif capture_profile == "oblique":
                    mapped_gimbal = FACADE_MAPPING_OBLIQUE_GIMBAL_DEG
                else:
                    mapped_gimbal = float(gimbal_pitch_deg)
                horizontal_spacing = max(0.5, line_step_m)
                vertical_spacing = max(0.5, point_step_m)
                motion_profile = "smooth"
                continuous_capture = True
                primitive_kind = "facade_mapping"
            else:
                mapped_gimbal = float(gimbal_pitch_deg)
                horizontal_spacing = max(1.0, line_step_m)
                vertical_spacing = max(1.0, point_step_m)
                motion_profile = "inspection"
                continuous_capture = False
                primitive_kind = "facade"

            return [
                MissionPrimitive(
                    kind=primitive_kind,
                    params={
                        "polygon_local": local_polygon,
                        "baseline_start_local": baseline_start.tolist(),
                        "baseline_end_local": baseline_end.tolist(),
                        "baseline_curve_local": baseline_curve.tolist() if isinstance(baseline_curve, np.ndarray) else None,
                        "curvature_alignment": bool(facade_curvature_alignment and isinstance(baseline_curve, np.ndarray)),
                        "standoff_m": standoff,
                        "top_altitude_m": top_alt,
                        "bottom_altitude_m": bottom_alt,
                        "horizontal_spacing_m": horizontal_spacing,
                        "vertical_spacing_m": vertical_spacing,
                        "flight_direction_deg": float(flight_direction_deg),
                        "gimbal_pitch_deg": mapped_gimbal,
                        "capture_profile": capture_profile,
                        "motion_profile": motion_profile,
                        "primitive_name": primitive_kind,
                        "rotate_points_180": bool(facade_rotate_points_180),
                        "continuous_capture": bool(continuous_capture),
                        "ground_offset_m": float(ground_offset_m),
                        "terrain_follow_enabled": bool(terrain_follow_enabled),
                        "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                        "facade_curvature_alignment": bool(facade_curvature_alignment),
                    },
                )
            ]

        levels = self._altitude_levels(
            altitude_m=altitude_m,
            constraints=constraints,
            min_levels=3,
            max_levels=6,
        )
        points_per_orbit = max(24, int(np.ceil((2.0 * pi * orbit_radius) / max(point_step_m, 1.0))))
        primitives = [
            MissionPrimitive(
                kind="orbit",
                params={
                    "center_local": [0.0, 0.0],
                    "radius_m": orbit_radius,
                    "points_per_orbit": points_per_orbit,
                    "altitude_levels_m": levels[: max(2, min(3, len(levels)))],
                },
            ),
            MissionPrimitive(
                kind="facade",
                params={
                    "polygon_local": local_polygon,
                    "standoff_m": constraints.standoff_m,
                    "altitude_levels_m": levels,
                    "horizontal_spacing_m": max(2.5, point_step_m * 0.9),
                },
            ),
            MissionPrimitive(
                kind="grid",
                params={
                    "polygon_local": local_polygon,
                    "line_step_m": max(1.0, line_step_m * 0.75),
                    "point_step_m": max(1.0, point_step_m * 0.75),
                    "line_spacing_m": max(1.0, line_step_m * 0.75),
                    "capture_spacing_m": max(1.0, point_step_m * 0.75),
                    "altitude_m": altitude_m,
                    "flight_direction_deg": float(flight_direction_deg),
                    "gimbal_pitch_deg": float(gimbal_pitch_deg),
                    "ground_offset_m": float(ground_offset_m),
                    "terrain_follow_enabled": bool(terrain_follow_enabled),
                    "terrain_model": dict(terrain_model or {}),
                        "terrain_follow_mode": str(terrain_follow_mode),
                        "terrain_normal_camera_enabled": bool(terrain_normal_camera_enabled),
                        "terrain_normal_gain": float(terrain_normal_gain),
                        "terrain_normal_yaw_align": bool(terrain_normal_yaw_align),
                    "continuous_capture": True,
                    "double_pass": True,
                },
            ),
        ]

        complexity = max(0, len(local_polygon) - 6)
        if complexity >= 3:
            primitives.append(
                MissionPrimitive(
                    kind="orbit",
                    params={
                        "center_local": [0.0, 0.0],
                        "radius_m": orbit_radius * 1.2,
                        "points_per_orbit": max(24, points_per_orbit),
                        "altitude_levels_m": [levels[min(len(levels) - 1, 1)] if levels else altitude_m],
                    },
                )
            )
        return primitives

    def _altitude_levels(
        self,
        altitude_m: float,
        constraints: MissionConstraints,
        min_levels: int,
        max_levels: int,
    ) -> list[float]:
        lo = max(constraints.min_altitude_m, altitude_m * 0.75)
        hi = min(constraints.max_altitude_m, max(altitude_m + 8.0, lo + 4.0))
        if hi <= lo + 0.5:
            return [float(np.clip(altitude_m, lo, hi if hi > lo else lo + 1.0))]
        rough_count = int(np.ceil((hi - lo) / 8.0)) + 1
        count = max(min_levels, min(max_levels, rough_count))
        levels = np.linspace(lo, hi, count)
        return [float(v) for v in levels]

    def _corridor_axis(self, local_polygon: list[list[float]]) -> tuple[np.ndarray, np.ndarray, float]:
        arr = np.asarray(local_polygon[:-1], dtype=np.float64)
        if len(arr) < 2:
            return np.array([0.0, 0.0]), np.array([1.0, 0.0]), 5.0

        center = arr.mean(axis=0)
        shifted = arr - center
        cov = np.cov(shifted.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.array([1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        perp = np.array([-axis[1], axis[0]])

        along = shifted @ axis
        across = shifted @ perp
        start = center + axis * float(along.min())
        end = center + axis * float(along.max())
        width = float(max(1.0, across.max() - across.min()))
        return start, end, width

    def _facade_baseline(self, local_polygon: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(local_polygon[:-1], dtype=np.float64)
        if len(arr) < 2:
            return np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64)

        start = arr[0]
        end = arr[1]
        if float(np.linalg.norm(end - start)) >= 0.5:
            return start.astype(np.float64), end.astype(np.float64)

        best_i, best_j = 0, 1
        best_d = 0.0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = float(np.linalg.norm(arr[j] - arr[i]))
                if d > best_d:
                    best_d = d
                    best_i, best_j = i, j
        return arr[best_i].astype(np.float64), arr[best_j].astype(np.float64)

    def _facade_curve(
        self,
        local_polygon: list[list[float]],
        curve_local: list[list[float]] | None = None,
        curvature_alignment: bool = False,
    ) -> np.ndarray | None:
        if not curvature_alignment:
            return None
        if curve_local is not None:
            try:
                line = np.asarray(_ensure_line(curve_local), dtype=np.float64)
                if len(line) >= 2:
                    return line
            except Exception:
                pass
        arr = np.asarray(local_polygon[:-1], dtype=np.float64)
        if len(arr) < 3:
            return None
        # Use user-drawn polygon boundary order as a curvature proxy for facade alignment.
        return arr

    def _compile_primitive(self, primitive: MissionPrimitive) -> list[_CapturePose]:
        kind = _normalize_template(primitive.kind)
        if kind in {"grid", "double_grid"}:
            return self._compile_grid_primitive(primitive.params)
        if kind == "roof_inspection":
            return self._compile_roof_inspection_primitive(primitive.params)
        if kind == "linear_inspection":
            return self._compile_linear_inspection_primitive(primitive.params)
        if kind == "lateral_capture":
            return self._compile_lateral_capture_primitive(primitive.params)
        if kind == "linked_mission":
            return self._compile_linked_route_primitive(primitive.params)
        if kind == "waypoints":
            return self._compile_waypoints_primitive(primitive.params)
        if kind == "tower_mapping":
            return self._compile_tower_mapping_primitive(primitive.params)
        if kind == "solar_inspection":
            return self._compile_solar_inspection_primitive(primitive.params)
        if kind == "magnetic_mapping":
            return self._compile_magnetic_mapping_primitive(primitive.params)
        if kind == "corridor":
            return self._compile_corridor_primitive(primitive.params)
        if kind == "orbit":
            return self._compile_orbit_primitive(primitive.params)
        if kind == "panorama":
            return self._compile_panorama_primitive(primitive.params)
        if kind == "bubble_360":
            return self._compile_bubble_primitive(primitive.params)
        if kind in {"facade", "facade_mapping"}:
            return self._compile_facade_primitive(primitive.params)
        if kind == "smart_adaptive":
            return self._compile_grid_primitive(primitive.params)
        return []

    def _compile_grid_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        polygon_local = params.get("polygon_local", [])
        if not polygon_local:
            return []
        poly = np.asarray(_ensure_closed_xy(polygon_local), dtype=np.float64)

        altitude = float(params.get("altitude_m", 60.0))
        line_step = max(0.75, float(params.get("line_spacing_m", params.get("line_step_m", 5.0))))
        point_step = max(0.75, float(params.get("capture_spacing_m", params.get("point_step_m", 5.0))))
        flight_direction = _wrap_deg(float(params.get("flight_direction_deg", 0.0)))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        double_pass = bool(params.get("double_pass", False))
        altitude_base = max(1.0, altitude + ground_offset)

        poses = self._compile_grid_pass(
            poly_local_closed=poly,
            altitude_m=altitude_base,
            line_step_m=line_step,
            capture_spacing_m=point_step,
            flight_direction_deg=flight_direction,
            gimbal_pitch_deg=gimbal_pitch,
            primitive_name="grid",
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
        )

        if double_pass:
            poses.extend(
                self._compile_grid_pass(
                    poly_local_closed=poly,
                    altitude_m=altitude_base,
                    line_step_m=line_step,
                    capture_spacing_m=point_step,
                    flight_direction_deg=_wrap_deg(flight_direction + 90.0),
                    gimbal_pitch_deg=gimbal_pitch,
                    primitive_name="grid_cross",
                    terrain_follow_enabled=terrain_follow_enabled,
                    terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
                )
            )

        return poses

    def _compile_roof_inspection_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        polygon_local = params.get("polygon_local", [])
        if not polygon_local:
            return []
        poly = np.asarray(_ensure_closed_xy(polygon_local), dtype=np.float64)

        altitude = float(params.get("altitude_m", 60.0))
        line_step = max(0.75, float(params.get("line_spacing_m", params.get("line_step_m", 5.0))))
        point_step = max(0.75, float(params.get("capture_spacing_m", params.get("point_step_m", 5.0))))
        flight_direction = _wrap_deg(float(params.get("flight_direction_deg", 0.0)))
        camera_direction = _wrap_deg(float(params.get("camera_direction_deg", flight_direction)))
        camera_direction_locked = bool(params.get("camera_direction_locked", False))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        dwell_s = float(np.clip(float(params.get("inspection_dwell_s", 1.5)), 0.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        altitude_base = max(1.0, altitude + ground_offset)

        sampled = self._compile_grid_pass(
            poly_local_closed=poly,
            altitude_m=altitude_base,
            line_step_m=line_step,
            capture_spacing_m=point_step,
            flight_direction_deg=flight_direction,
            gimbal_pitch_deg=gimbal_pitch,
            primitive_name="roof_inspection",
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
        )
        if not sampled:
            return []

        out: list[_CapturePose] = []
        for pose in sampled:
            yaw = camera_direction if camera_direction_locked else float(pose.yaw_deg)
            out.append(
                replace(
                    pose,
                    yaw_deg=yaw,
                    primitive="roof_inspection",
                    dwell_s=dwell_s,
                    camera_yaw_locked=camera_direction_locked,
                    trigger=True,
                )
            )
        return out

    def _compile_linear_inspection_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        line_local = params.get("line_local", [])
        if not line_local or len(line_local) < 2:
            return []
        line = np.asarray(_ensure_line(line_local), dtype=np.float64)

        capture_spacing = max(0.75, float(params.get("capture_spacing_m", 5.0)))
        altitude = float(params.get("altitude_m", 60.0))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -60.0)), -120.0, 30.0))
        dwell_s = float(np.clip(float(params.get("inspection_dwell_s", 1.5)), 0.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        segmentation_enabled = bool(params.get("linear_segmentation_enabled", True))
        max_seg_len = float(max(100.0, float(params.get("linear_max_segment_length_m", 1500.0))))
        base_alt = max(1.0, altitude + ground_offset)

        poses: list[_CapturePose] = []
        cumulative_m = 0.0
        prev = line[0]
        for seg_idx in range(len(line) - 1):
            a = line[seg_idx]
            b = line[seg_idx + 1]
            seg_len = float(np.linalg.norm(b - a))
            if seg_len < 1e-6:
                continue
            sampled = _sample_segment(a, b, capture_spacing)
            if seg_idx > 0 and sampled:
                sampled = sampled[1:]

            for i, pt in enumerate(sampled):
                if poses:
                    cumulative_m += float(np.linalg.norm(np.asarray(pt, dtype=np.float64) - prev))
                pt_np = np.asarray(pt, dtype=np.float64)
                if i < len(sampled) - 1:
                    next_pt = np.asarray(sampled[i + 1], dtype=np.float64)
                    yaw = _bearing_deg(pt_np, next_pt)
                else:
                    yaw = _bearing_deg(a, b)

                terrain_delta = (
                    _terrain_delta_m(pt_np, terrain_model if isinstance(terrain_model, dict) else None)
                    if terrain_follow_enabled
                    else 0.0
                )
                seg_number = int(cumulative_m // max_seg_len) + 1 if segmentation_enabled else 1
                primitive_name = f"linear_inspection_seg{seg_number}" if segmentation_enabled else "linear_inspection"
                poses.append(
                    _CapturePose(
                        x_m=float(pt_np[0]),
                        y_m=float(pt_np[1]),
                        alt_m=max(1.0, float(base_alt + terrain_delta)),
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=float(gimbal_pitch),
                        primitive=primitive_name,
                        trigger=True,
                        dwell_s=dwell_s,
                        camera_yaw_locked=True,
                    )
                )
                prev = pt_np

        return poses

    def _compile_lateral_capture_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        target_line_local = params.get("target_line_local", [])
        if not target_line_local or len(target_line_local) < 2:
            return []
        line = np.asarray(_ensure_line(target_line_local), dtype=np.float64)

        capture_spacing = max(0.75, float(params.get("capture_spacing_m", 5.0)))
        altitude = float(params.get("altitude_m", 60.0))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -60.0)), -120.0, 30.0))
        standoff_m = max(0.5, float(params.get("standoff_m", 10.0)))
        target_side = str(params.get("target_side", "right")).strip().lower()
        if target_side not in {"left", "right"}:
            target_side = "right"
        offset_sign = -1.0 if target_side == "left" else 1.0
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        base_alt = max(1.0, altitude + ground_offset)

        poses: list[_CapturePose] = []
        for seg_idx in range(len(line) - 1):
            a = line[seg_idx]
            b = line[seg_idx + 1]
            seg_vec = b - a
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                continue
            tangent_default = seg_vec / seg_len
            sampled = _sample_segment(a, b, capture_spacing)
            if seg_idx > 0 and sampled:
                sampled = sampled[1:]
            if not sampled:
                continue

            for i, asset_pt in enumerate(sampled):
                asset_np = np.asarray(asset_pt, dtype=np.float64)
                if i < len(sampled) - 1:
                    next_pt = np.asarray(sampled[i + 1], dtype=np.float64)
                    step_vec = next_pt - asset_np
                    step_len = float(np.linalg.norm(step_vec))
                    tangent = step_vec / step_len if step_len > 1e-6 else tangent_default
                else:
                    tangent = tangent_default
                left_normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
                flight_pt = asset_np + left_normal * (offset_sign * standoff_m)
                yaw = _bearing_deg(flight_pt, asset_np)

                terrain_delta = (
                    _terrain_delta_m(flight_pt, terrain_model if isinstance(terrain_model, dict) else None)
                    if terrain_follow_enabled
                    else 0.0
                )
                poses.append(
                    _CapturePose(
                        x_m=float(flight_pt[0]),
                        y_m=float(flight_pt[1]),
                        alt_m=max(1.0, float(base_alt + terrain_delta)),
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=float(gimbal_pitch),
                        primitive="lateral_capture",
                        trigger=True,
                        dwell_s=0.0,
                        camera_yaw_locked=True,
                    )
                )

        return poses

    def _compile_linked_route_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        raw = params.get("poses_local", [])
        if not isinstance(raw, list) or not raw:
            return []
        poses: list[_CapturePose] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                poses.append(
                    _CapturePose(
                        x_m=float(item.get("x_m", 0.0)),
                        y_m=float(item.get("y_m", 0.0)),
                        alt_m=max(1.0, float(item.get("alt_m", 60.0))),
                        yaw_deg=float(item.get("yaw_deg", 0.0)),
                        gimbal_pitch_deg=float(item.get("gimbal_pitch_deg", -90.0)),
                        primitive=str(item.get("primitive", "linked_route")),
                        trigger=bool(item.get("trigger", True)),
                        dwell_s=float(np.clip(float(item.get("dwell_s", 0.0)), 0.0, 30.0)),
                        camera_yaw_locked=bool(item.get("camera_yaw_locked", False)),
                    )
                )
            except Exception:
                continue
        return poses

    def _compile_waypoints_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        path_local = params.get("path_local", [])
        if not path_local or len(path_local) < 2:
            return []
        path = np.asarray(_ensure_line(path_local), dtype=np.float64)

        altitude = float(params.get("altitude_m", 60.0))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        dwell_s = float(np.clip(float(params.get("inspection_dwell_s", 1.5)), 0.0, 30.0))
        heading_mode = str(params.get("heading_mode", "tangent")).strip().lower()
        if heading_mode not in {"tangent", "fixed", "poi"}:
            heading_mode = "tangent"
        fixed_yaw_deg = _wrap_deg(float(params.get("fixed_yaw_deg", 0.0)))
        poi_local = params.get("poi_local")
        poi_xy = (
            np.asarray([float(poi_local[0]), float(poi_local[1])], dtype=np.float64)
            if isinstance(poi_local, list) and len(poi_local) >= 2
            else None
        )
        if heading_mode == "poi" and poi_xy is None:
            heading_mode = "tangent"

        enable_smoothing = bool(params.get("enable_smoothing", False))
        turn_radius_m = max(0.0, float(params.get("turn_radius_m", 0.0)))
        capture_enabled = bool(params.get("capture_enabled", True))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        altitude_base = max(1.0, altitude + ground_offset)

        if enable_smoothing and turn_radius_m > 0.0 and len(path) >= 3:
            path_points = self._smooth_waypoint_path(path, turn_radius_m)
        else:
            path_points = [(pt, True) for pt in path]

        poses: list[_CapturePose] = []
        for idx, (pt, is_action_waypoint) in enumerate(path_points):
            if heading_mode == "fixed":
                yaw = fixed_yaw_deg
                yaw_locked = True
            elif heading_mode == "poi" and poi_xy is not None:
                yaw = _bearing_deg(pt, poi_xy)
                yaw_locked = True
            else:
                yaw_locked = False
                if idx < len(path_points) - 1:
                    yaw = _bearing_deg(pt, path_points[idx + 1][0])
                elif idx > 0:
                    yaw = _bearing_deg(path_points[idx - 1][0], pt)
                else:
                    yaw = fixed_yaw_deg

            terrain_delta = (
                _terrain_delta_m(pt, terrain_model if isinstance(terrain_model, dict) else None)
                if terrain_follow_enabled
                else 0.0
            )
            do_capture = bool(capture_enabled and is_action_waypoint)
            poses.append(
                _CapturePose(
                    x_m=float(pt[0]),
                    y_m=float(pt[1]),
                    alt_m=max(1.0, float(altitude_base + terrain_delta)),
                    yaw_deg=float(yaw),
                    gimbal_pitch_deg=float(gimbal_pitch),
                    primitive="waypoints" if is_action_waypoint else "waypoints_curve",
                    trigger=do_capture,
                    dwell_s=float(dwell_s if do_capture else 0.0),
                    camera_yaw_locked=bool(yaw_locked),
                )
            )

        return poses

    def _smooth_waypoint_path(
        self,
        path_xy: np.ndarray,
        radius_m: float,
    ) -> list[tuple[np.ndarray, bool]]:
        if len(path_xy) < 3 or radius_m <= 1e-6:
            return [(pt, True) for pt in path_xy]

        out: list[tuple[np.ndarray, bool]] = [(path_xy[0], True)]
        for i in range(1, len(path_xy) - 1):
            a = np.asarray(path_xy[i - 1], dtype=np.float64)
            b = np.asarray(path_xy[i], dtype=np.float64)
            c = np.asarray(path_xy[i + 1], dtype=np.float64)
            v1 = b - a
            v2 = c - b
            l1 = float(np.linalg.norm(v1))
            l2 = float(np.linalg.norm(v2))
            if l1 < 1e-6 or l2 < 1e-6:
                out.append((b, True))
                continue
            u1 = v1 / l1
            u2 = v2 / l2
            turn_angle = float(np.degrees(np.arccos(np.clip(float(np.dot(u1, u2)), -1.0, 1.0))))
            if turn_angle < 15.0:
                out.append((b, True))
                continue

            d = min(float(radius_m), 0.4 * l1, 0.4 * l2)
            before = b - u1 * d
            after = b + u2 * d
            mid = (before + after) * 0.5
            out.append((before, False))
            out.append((mid, False))
            out.append((b, True))
            out.append((after, False))

        out.append((path_xy[-1], True))
        return out

    def _compile_tower_mapping_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        center = np.asarray(params.get("center_local", [0.0, 0.0]), dtype=np.float64)
        radius = max(1.0, float(params.get("radius_m", 12.0)))
        object_radius = max(0.5, float(params.get("object_radius_m", 2.0)))
        points_per_orbit = max(12, int(params.get("points_per_orbit", 36)))
        levels = params.get("altitude_levels_m", [60.0])
        if not isinstance(levels, list) or not levels:
            levels = [60.0]
        target_alt = float(params.get("target_altitude_m", float(np.mean(levels))))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        resume_enabled = bool(params.get("tower_resume_enabled", True))
        tower_range = max(2.0, radius - object_radius)

        poses: list[_CapturePose] = []
        for level_idx, level in enumerate(levels):
            alt = float(level) + ground_offset
            reverse = level_idx % 2 == 1
            orbit_indices = range(points_per_orbit - 1, -1, -1) if reverse else range(points_per_orbit)
            primitive_name = f"tower_mapping_level{level_idx + 1}" if resume_enabled else "tower_mapping"

            if gimbal_pitch <= -89.0:
                auto_pitch = float(np.degrees(np.arctan2(target_alt - float(level), tower_range)))
                pitch = float(np.clip(auto_pitch, -120.0, 30.0))
            else:
                pitch = gimbal_pitch

            for i in orbit_indices:
                theta = 2.0 * pi * (i / points_per_orbit)
                x = float(center[0] + radius * np.cos(theta))
                y = float(center[1] + radius * np.sin(theta))
                yaw = float(np.degrees(np.arctan2(float(center[1]) - y, float(center[0]) - x)))
                pt = np.asarray([x, y], dtype=np.float64)
                terrain_delta = (
                    _terrain_delta_m(pt, terrain_model if isinstance(terrain_model, dict) else None)
                    if terrain_follow_enabled
                    else 0.0
                )
                poses.append(
                    _CapturePose(
                        x_m=x,
                        y_m=y,
                        alt_m=max(1.0, float(alt + terrain_delta)),
                        yaw_deg=yaw,
                        gimbal_pitch_deg=pitch,
                        primitive=primitive_name,
                        trigger=True,
                        dwell_s=0.0,
                        camera_yaw_locked=True,
                    )
                )

        return poses

    def _compile_solar_inspection_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        polygon_local = params.get("polygon_local", [])
        if not polygon_local:
            return []
        poly = np.asarray(_ensure_closed_xy(polygon_local), dtype=np.float64)

        row_angle = _wrap_deg(float(params.get("row_angle_deg", 0.0)))
        line_spacing = max(0.75, float(params.get("line_spacing_m", 4.0)))
        capture_spacing = max(0.75, float(params.get("capture_spacing_m", 4.0)))
        altitude = float(params.get("altitude_m", 60.0))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        sensor_profile = str(params.get("sensor_profile", "rgb")).strip().lower()
        if sensor_profile not in {"rgb", "thermal"}:
            sensor_profile = "rgb"
        orientation_mode = str(params.get("orientation_mode", "row_aligned")).strip().lower()
        if orientation_mode not in {"row_aligned", "path_aligned"}:
            orientation_mode = "row_aligned"
        row_snap_tolerance = max(0.2, float(params.get("row_snap_tolerance_m", line_spacing * 0.35)))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")

        aligned_poly = _rotate_xy(poly, -row_angle)
        min_y = float(np.min(aligned_poly[:, 1]))
        max_y = float(np.max(aligned_poly[:, 1]))

        row_targets: list[float] = []
        rows_local = params.get("rows_local", [])
        if isinstance(rows_local, list):
            for row in rows_local:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                try:
                    row_arr = np.asarray(_ensure_line(row), dtype=np.float64)
                except Exception:
                    continue
                aligned_row = _rotate_xy(row_arr, -row_angle)
                row_targets.append(float(np.mean(aligned_row[:, 1])))

        phase = self._solar_line_phase(min_y=min_y, line_spacing=line_spacing, row_targets=row_targets)
        start_y = min_y + phase
        while start_y > min_y + 1e-6:
            start_y -= line_spacing
        candidate_ys = np.arange(start_y, max_y + line_spacing * 0.5, line_spacing, dtype=np.float64)
        if len(candidate_ys) == 0:
            candidate_ys = np.asarray([min_y], dtype=np.float64)
        ys, _row_snap_stats = self._snap_solar_lines(
            candidate_ys=candidate_ys,
            row_targets=row_targets,
            min_y=min_y,
            max_y=max_y,
            line_spacing=line_spacing,
            snap_tolerance=row_snap_tolerance,
        )

        rows: list[list[list[np.ndarray]]] = []
        for y in ys:
            xs = _line_intersections_at_y(aligned_poly, float(y))
            if len(xs) < 2:
                continue
            segments: list[list[np.ndarray]] = []
            for i in range(0, len(xs) - 1, 2):
                x1 = float(xs[i])
                x2 = float(xs[i + 1])
                if x2 <= x1 + 1e-6:
                    continue
                start = np.array([x1, float(y)], dtype=np.float64)
                end = np.array([x2, float(y)], dtype=np.float64)
                sampled = _sample_segment(start, end, spacing_m=capture_spacing)
                if len(sampled) >= 2:
                    segments.append(sampled)
            if segments:
                rows.append(segments)

        poses: list[_CapturePose] = []
        prev_end_aligned: np.ndarray | None = None
        altitude_base = max(1.0, altitude + ground_offset)
        for row_idx, row_segments in enumerate(rows):
            row_ordered = sorted(row_segments, key=lambda seg: float(seg[0][0]))
            if row_idx % 2 == 1:
                row_ordered = list(reversed(row_ordered))

            for segment in row_ordered:
                forward = segment
                backward = list(reversed(segment))
                if prev_end_aligned is None:
                    chosen = forward
                else:
                    d_forward = float(np.linalg.norm(prev_end_aligned - forward[0]))
                    d_backward = float(np.linalg.norm(prev_end_aligned - backward[0]))
                    chosen = forward if d_forward <= d_backward else backward

                aligned_strip = np.asarray(chosen, dtype=np.float64)
                local_strip = _rotate_xy(aligned_strip, row_angle)
                if len(local_strip) < 2:
                    continue
                travel_yaw = _bearing_deg(local_strip[0], local_strip[-1])
                row_forward_yaw = _wrap_deg(row_angle)
                row_reverse_yaw = _wrap_deg(row_angle + 180.0)
                if abs(_wrap_deg(travel_yaw - row_forward_yaw)) <= abs(_wrap_deg(travel_yaw - row_reverse_yaw)):
                    row_locked_yaw = row_forward_yaw
                else:
                    row_locked_yaw = row_reverse_yaw

                yaw = float(travel_yaw) if orientation_mode == "path_aligned" else float(row_locked_yaw)
                camera_lock = True
                if sensor_profile == "thermal":
                    pitch = float(np.clip(min(gimbal_pitch, -85.0), -100.0, -70.0))
                else:
                    pitch = float(np.clip(gimbal_pitch, -100.0, -55.0))
                for pt in local_strip:
                    terrain_delta = (
                        _terrain_delta_m(pt, terrain_model if isinstance(terrain_model, dict) else None)
                        if terrain_follow_enabled
                        else 0.0
                    )
                    poses.append(
                        _CapturePose(
                            x_m=float(pt[0]),
                            y_m=float(pt[1]),
                            alt_m=max(1.0, float(altitude_base + terrain_delta)),
                            yaw_deg=float(yaw),
                            gimbal_pitch_deg=float(pitch),
                            primitive="solar_inspection",
                            trigger=True,
                            dwell_s=0.0,
                            camera_yaw_locked=camera_lock,
                        )
                    )
                prev_end_aligned = aligned_strip[-1]

        return poses

    def _compile_magnetic_mapping_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        polygon_local = params.get("polygon_local", [])
        if not polygon_local:
            return []
        poly = np.asarray(_ensure_closed_xy(polygon_local), dtype=np.float64)

        altitude = float(params.get("altitude_m", 60.0))
        line_spacing = max(2.0, float(params.get("line_spacing_m", 12.0)))
        capture_spacing = max(2.0, float(params.get("capture_spacing_m", 12.0)))
        tie_spacing = max(5.0, float(params.get("tie_line_spacing_m", line_spacing * 3.0)))
        turn_smoothing_radius = max(0.0, float(params.get("turn_smoothing_radius_m", 8.0)))
        flight_direction = _wrap_deg(float(params.get("flight_direction_deg", 0.0)))
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        altitude_base = max(1.0, altitude + ground_offset)

        main = self._compile_grid_pass(
            poly_local_closed=poly,
            altitude_m=altitude_base,
            line_step_m=line_spacing,
            capture_spacing_m=capture_spacing,
            flight_direction_deg=flight_direction,
            gimbal_pitch_deg=gimbal_pitch,
            primitive_name="magnetic_main",
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
        )
        tie = self._compile_grid_pass(
            poly_local_closed=poly,
            altitude_m=altitude_base,
            line_step_m=tie_spacing,
            capture_spacing_m=capture_spacing,
            flight_direction_deg=_wrap_deg(flight_direction + 90.0),
            gimbal_pitch_deg=gimbal_pitch,
            primitive_name="magnetic_tie",
            terrain_follow_enabled=terrain_follow_enabled,
            terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
        )
        path = main + tie
        if turn_smoothing_radius > 0.0:
            path = self._apply_turn_smoothing(path, turn_smoothing_radius)
        return path

    def _solar_line_phase(self, min_y: float, line_spacing: float, row_targets: list[float]) -> float:
        if not row_targets or line_spacing <= 1e-6:
            return 0.0
        mods = sorted(
            {
                float((row_y - min_y) % line_spacing)
                for row_y in row_targets
            }
        )
        if not mods:
            return 0.0

        best_phase = float(mods[0])
        best_score = float("inf")
        for phase in mods:
            start = float(min_y + phase)
            score = 0.0
            for row_y in row_targets:
                delta = abs(((float(row_y) - start + 0.5 * line_spacing) % line_spacing) - 0.5 * line_spacing)
                score += float(delta)
            if score < best_score:
                best_score = score
                best_phase = float(phase)
        return best_phase

    def _snap_solar_lines(
        self,
        candidate_ys: np.ndarray,
        row_targets: list[float],
        min_y: float,
        max_y: float,
        line_spacing: float,
        snap_tolerance: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        values = sorted(
            {
                float(np.clip(v, min_y, max_y))
                for v in np.asarray(candidate_ys, dtype=np.float64).tolist()
            }
        )
        if not values:
            values = [float(np.clip(min_y, min_y, max_y))]

        targets = sorted(float(np.clip(t, min_y, max_y)) for t in row_targets)
        snapped_hits = 0
        inserted = 0
        shifted = 0
        total_shift = 0.0

        for target in targets:
            nearest_idx = min(range(len(values)), key=lambda i: abs(values[i] - target))
            delta = float(target - values[nearest_idx])
            if abs(delta) <= snap_tolerance:
                snapped_hits += 1
                if abs(delta) > 1e-6:
                    total_shift += abs(delta)
                    values[nearest_idx] = float(target)
                    shifted += 1
            else:
                values.append(float(target))
                inserted += 1

        values.sort()
        min_gap = max(0.2, float(line_spacing) * 0.1)
        dedup: list[float] = []
        for y in values:
            if not dedup or abs(y - dedup[-1]) >= min_gap:
                dedup.append(float(y))
                continue
            if not targets:
                continue
            prev = dedup[-1]
            prev_err = min(abs(prev - t) for t in targets)
            cur_err = min(abs(y - t) for t in targets)
            if cur_err < prev_err:
                dedup[-1] = float(y)

        filled: list[float] = []
        for y in dedup:
            if not filled:
                filled.append(float(y))
                continue
            while y - filled[-1] > line_spacing * 1.8:
                filled.append(float(filled[-1] + line_spacing))
            filled.append(float(y))

        final = np.asarray(
            [float(np.clip(v, min_y, max_y)) for v in filled],
            dtype=np.float64,
        )
        if len(final) == 0:
            final = np.asarray([float(np.clip(min_y, min_y, max_y))], dtype=np.float64)
        final = np.unique(np.round(final, 6))
        return final, {
            "row_targets": float(len(targets)),
            "snapped_hits": float(snapped_hits),
            "shifted": float(shifted),
            "inserted": float(inserted),
            "avg_shift_m": float(total_shift / max(1, shifted)),
        }

    def _apply_turn_smoothing(self, poses: list[_CapturePose], radius_m: float) -> list[_CapturePose]:
        if len(poses) < 3 or radius_m <= 1e-6:
            return poses

        smoothed: list[_CapturePose] = [poses[0]]
        for i in range(1, len(poses) - 1):
            prev = poses[i - 1]
            cur = poses[i]
            nxt = poses[i + 1]
            if cur.trigger is False:
                smoothed.append(cur)
                continue

            a = np.asarray([prev.x_m, prev.y_m], dtype=np.float64)
            b = np.asarray([cur.x_m, cur.y_m], dtype=np.float64)
            c = np.asarray([nxt.x_m, nxt.y_m], dtype=np.float64)
            v1 = b - a
            v2 = c - b
            l1 = float(np.linalg.norm(v1))
            l2 = float(np.linalg.norm(v2))
            if l1 < 1e-6 or l2 < 1e-6:
                smoothed.append(cur)
                continue
            u1 = v1 / l1
            u2 = v2 / l2
            turn_angle = float(np.degrees(np.arccos(np.clip(float(np.dot(u1, u2)), -1.0, 1.0))))
            if turn_angle < 35.0:
                smoothed.append(cur)
                continue

            d = min(float(radius_m), 0.4 * l1, 0.4 * l2)
            before = b - u1 * d
            after = b + u2 * d
            yaw_before = _bearing_deg(a, before)
            yaw_after = _bearing_deg(before, after)
            smoothed.append(
                replace(
                    cur,
                    x_m=float(before[0]),
                    y_m=float(before[1]),
                    yaw_deg=float(yaw_before),
                    primitive="magnetic_turn",
                    trigger=False,
                    camera_yaw_locked=False,
                )
            )
            smoothed.append(
                replace(
                    cur,
                    x_m=float(after[0]),
                    y_m=float(after[1]),
                    yaw_deg=float(yaw_after),
                    primitive="magnetic_turn",
                    trigger=False,
                    camera_yaw_locked=False,
                )
            )

        smoothed.append(poses[-1])
        return smoothed

    def _compile_grid_pass(
        self,
        poly_local_closed: np.ndarray,
        altitude_m: float,
        line_step_m: float,
        capture_spacing_m: float,
        flight_direction_deg: float,
        gimbal_pitch_deg: float,
        primitive_name: str,
        terrain_follow_enabled: bool,
        terrain_model: dict[str, Any] | None,
    ) -> list[_CapturePose]:
        if len(poly_local_closed) < 4:
            return []

        aligned_poly = _rotate_xy(poly_local_closed, -float(flight_direction_deg))
        min_y = float(np.min(aligned_poly[:, 1]))
        max_y = float(np.max(aligned_poly[:, 1]))
        step = max(0.75, float(line_step_m))

        ys = np.arange(min_y, max_y + step * 0.5, step, dtype=np.float64)
        if len(ys) == 0:
            ys = np.asarray([min_y], dtype=np.float64)
        elif ys[-1] < max_y - step * 0.25:
            ys = np.append(ys, max_y)

        rows: list[list[list[np.ndarray]]] = []
        for y in ys:
            xs = _line_intersections_at_y(aligned_poly, float(y))
            if len(xs) < 2:
                continue
            segments: list[list[np.ndarray]] = []
            for i in range(0, len(xs) - 1, 2):
                x1 = float(xs[i])
                x2 = float(xs[i + 1])
                if x2 <= x1 + 1e-6:
                    continue
                start = np.array([x1, float(y)], dtype=np.float64)
                end = np.array([x2, float(y)], dtype=np.float64)
                sampled = _sample_segment(start, end, spacing_m=max(0.75, float(capture_spacing_m)))
                if len(sampled) >= 2:
                    segments.append(sampled)
            if segments:
                rows.append(segments)

        poses: list[_CapturePose] = []
        prev_end_aligned: np.ndarray | None = None
        for row_idx, row_segments in enumerate(rows):
            row_ordered = sorted(row_segments, key=lambda seg: float(seg[0][0]))
            if row_idx % 2 == 1:
                row_ordered = list(reversed(row_ordered))

            for segment in row_ordered:
                forward = segment
                backward = list(reversed(segment))

                if prev_end_aligned is None:
                    chosen = forward
                else:
                    d_forward = float(np.linalg.norm(prev_end_aligned - forward[0]))
                    d_backward = float(np.linalg.norm(prev_end_aligned - backward[0]))
                    chosen = forward if d_forward <= d_backward else backward

                aligned_strip = np.asarray(chosen, dtype=np.float64)
                local_strip = _rotate_xy(aligned_strip, float(flight_direction_deg))
                if len(local_strip) < 2:
                    continue

                yaw = _bearing_deg(local_strip[0], local_strip[-1])
                for pt in local_strip:
                    terrain_delta = _terrain_delta_m(pt, terrain_model) if terrain_follow_enabled else 0.0
                    poses.append(
                        _CapturePose(
                            x_m=float(pt[0]),
                            y_m=float(pt[1]),
                            alt_m=max(1.0, float(altitude_m + terrain_delta)),
                            yaw_deg=yaw,
                            gimbal_pitch_deg=float(gimbal_pitch_deg),
                            primitive=primitive_name,
                            trigger=True,
                        )
                    )
                prev_end_aligned = aligned_strip[-1]

        return poses

    def _compile_corridor_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        start = np.asarray(params.get("axis_start_local", [0.0, 0.0]), dtype=np.float64)
        end = np.asarray(params.get("axis_end_local", [1.0, 0.0]), dtype=np.float64)
        lane_count = max(1, int(params.get("lane_count", 1)))
        lane_spacing = max(1.0, float(params.get("lane_spacing_m", 4.0)))
        capture_spacing = max(1.0, float(params.get("capture_spacing_m", 4.0)))
        altitude = float(params.get("altitude_m", 60.0))

        axis = end - start
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            return []
        axis = axis / length
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        offsets = np.linspace(-(lane_count - 1) * lane_spacing / 2.0, (lane_count - 1) * lane_spacing / 2.0, lane_count)
        poses: list[_CapturePose] = []
        for idx, off in enumerate(offsets):
            a = start + normal * float(off)
            b = end + normal * float(off)
            sampled = _sample_segment(a, b, capture_spacing)
            if idx % 2 == 1:
                sampled = list(reversed(sampled))
            if len(sampled) < 2:
                continue
            yaw = _bearing_deg(sampled[0], sampled[-1])
            for pt in sampled:
                poses.append(
                    _CapturePose(
                        x_m=float(pt[0]),
                        y_m=float(pt[1]),
                        alt_m=altitude,
                        yaw_deg=yaw,
                        gimbal_pitch_deg=-80.0,
                        primitive="corridor",
                        trigger=True,
                    )
                )
        return poses

    def _compile_orbit_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        center = np.asarray(params.get("center_local", [0.0, 0.0]), dtype=np.float64)
        radius = max(1.0, float(params.get("radius_m", 10.0)))
        points_per_orbit = max(12, int(params.get("points_per_orbit", 36)))
        levels = params.get("altitude_levels_m", [60.0])
        if not isinstance(levels, list) or not levels:
            levels = [60.0]
        gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -35.0)), -120.0, 30.0))
        yaw_lock_to_poi = bool(params.get("yaw_lock_to_poi", True))
        poi_local = params.get("poi_local")
        poi = (
            np.asarray([float(poi_local[0]), float(poi_local[1])], dtype=np.float64)
            if isinstance(poi_local, list) and len(poi_local) >= 2
            else center
        )
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")

        poses: list[_CapturePose] = []
        for level in levels:
            alt = float(level) + ground_offset
            for i in range(points_per_orbit):
                theta = 2.0 * pi * (i / points_per_orbit)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                pt = np.asarray([float(x), float(y)], dtype=np.float64)
                if yaw_lock_to_poi:
                    yaw = _bearing_deg(pt, poi)
                else:
                    next_theta = 2.0 * pi * (((i + 1) % points_per_orbit) / points_per_orbit)
                    next_pt = np.asarray(
                        [
                            float(center[0] + radius * np.cos(next_theta)),
                            float(center[1] + radius * np.sin(next_theta)),
                        ],
                        dtype=np.float64,
                    )
                    yaw = _bearing_deg(pt, next_pt)
                terrain_delta = (
                    _terrain_delta_m(pt, terrain_model if isinstance(terrain_model, dict) else None)
                    if terrain_follow_enabled
                    else 0.0
                )
                poses.append(
                    _CapturePose(
                        x_m=float(pt[0]),
                        y_m=float(pt[1]),
                        alt_m=max(1.0, float(alt + terrain_delta)),
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=float(gimbal_pitch),
                        primitive="orbit",
                        trigger=True,
                        camera_yaw_locked=bool(yaw_lock_to_poi),
                    )
                )
        return poses

    def _compile_panorama_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        center = np.asarray(params.get("center_local", [0.0, 0.0]), dtype=np.float64)
        altitude = float(params.get("altitude_m", 60.0))
        yaw_start = _wrap_deg(float(params.get("yaw_start_deg", 0.0)))
        yaw_step = max(1.0, float(params.get("yaw_step_deg", 15.0)))
        yaw_count = max(3, int(params.get("yaw_count", 24)))
        row_pitches = params.get("row_pitches_deg", [-15.0])
        if not isinstance(row_pitches, list) or not row_pitches:
            row_pitches = [-15.0]
        dwell_s = float(np.clip(float(params.get("inspection_dwell_s", 0.6)), 0.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        base_alt = max(1.0, altitude + ground_offset)

        poses: list[_CapturePose] = []
        terrain_delta = (
            _terrain_delta_m(center, terrain_model if isinstance(terrain_model, dict) else None)
            if terrain_follow_enabled
            else 0.0
        )
        final_alt = max(1.0, float(base_alt + terrain_delta))
        for row_idx, pitch in enumerate(row_pitches):
            pitch_clamped = float(np.clip(float(pitch), -120.0, 30.0))
            reverse = row_idx % 2 == 1
            yaw_indices = range(yaw_count - 1, -1, -1) if reverse else range(yaw_count)
            for i in yaw_indices:
                yaw = _wrap_deg(yaw_start + float(i) * yaw_step)
                poses.append(
                    _CapturePose(
                        x_m=float(center[0]),
                        y_m=float(center[1]),
                        alt_m=final_alt,
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=pitch_clamped,
                        primitive=f"panorama_row{row_idx + 1}",
                        trigger=True,
                        dwell_s=dwell_s,
                        camera_yaw_locked=True,
                    )
                )
        return poses

    def _compile_bubble_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        center = np.asarray(params.get("center_local", [0.0, 0.0]), dtype=np.float64)
        altitude = float(params.get("altitude_m", 60.0))
        yaw_start = _wrap_deg(float(params.get("yaw_start_deg", 0.0)))
        yaw_step = max(1.0, float(params.get("yaw_step_deg", 15.0)))
        yaw_count = max(3, int(params.get("yaw_count", 24)))
        row_pitches = params.get("row_pitches_deg", [-15.0])
        if not isinstance(row_pitches, list) or not row_pitches:
            row_pitches = [-15.0]
        dwell_s = float(np.clip(float(params.get("inspection_dwell_s", 0.6)), 0.0, 30.0))
        ground_offset = float(params.get("ground_offset_m", 0.0))
        terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
        terrain_model = params.get("terrain_model")
        base_alt = max(1.0, altitude + ground_offset)

        poses: list[_CapturePose] = []
        terrain_delta = (
            _terrain_delta_m(center, terrain_model if isinstance(terrain_model, dict) else None)
            if terrain_follow_enabled
            else 0.0
        )
        final_alt = max(1.0, float(base_alt + terrain_delta))
        for row_idx, pitch in enumerate(row_pitches):
            pitch_clamped = float(np.clip(float(pitch), -120.0, 30.0))
            reverse = row_idx % 2 == 1
            yaw_indices = range(yaw_count - 1, -1, -1) if reverse else range(yaw_count)
            for i in yaw_indices:
                yaw = _wrap_deg(yaw_start + float(i) * yaw_step)
                poses.append(
                    _CapturePose(
                        x_m=float(center[0]),
                        y_m=float(center[1]),
                        alt_m=final_alt,
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=pitch_clamped,
                        primitive=f"bubble_row{row_idx + 1}",
                        trigger=True,
                        dwell_s=dwell_s,
                        camera_yaw_locked=True,
                    )
                )
        return poses

    def _compile_facade_primitive(self, params: dict[str, Any]) -> list[_CapturePose]:
        polygon_local = params.get("polygon_local", [])
        if not polygon_local:
            return []
        poly = np.asarray(_ensure_closed_xy(polygon_local), dtype=np.float64)
        boundary = poly[:-1]
        if len(boundary) < 2:
            return []
        primitive_name = str(params.get("primitive_name", "facade"))

        baseline_start = params.get("baseline_start_local")
        baseline_end = params.get("baseline_end_local")
        if isinstance(baseline_start, list) and isinstance(baseline_end, list) and len(baseline_start) >= 2 and len(baseline_end) >= 2:
            start = np.asarray([float(baseline_start[0]), float(baseline_start[1])], dtype=np.float64)
            end = np.asarray([float(baseline_end[0]), float(baseline_end[1])], dtype=np.float64)
            standoff = max(0.5, float(params.get("standoff_m", 8.0)))
            horizontal_spacing = max(0.75, float(params.get("horizontal_spacing_m", 4.0)))
            vertical_spacing = max(0.75, float(params.get("vertical_spacing_m", 4.0)))
            top_alt = float(params.get("top_altitude_m", 80.0))
            bottom_alt = float(params.get("bottom_altitude_m", 40.0))
            rotate_points_180 = bool(params.get("rotate_points_180", False))
            gimbal_pitch = float(np.clip(float(params.get("gimbal_pitch_deg", -90.0)), -120.0, 30.0))
            ground_offset = float(params.get("ground_offset_m", 0.0))
            terrain_follow_enabled = bool(params.get("terrain_follow_enabled", False))
            terrain_model = params.get("terrain_model")
            terrain_normal_enabled = bool(params.get("terrain_normal_camera_enabled", False))
            terrain_normal_gain = float(np.clip(float(params.get("terrain_normal_gain", 1.0)), 0.0, 3.0))
            terrain_normal_yaw_align = bool(params.get("terrain_normal_yaw_align", False))
            curvature_alignment = bool(params.get("curvature_alignment", params.get("facade_curvature_alignment", False)))
            baseline_curve_local = params.get("baseline_curve_local")

            top = max(top_alt, bottom_alt)
            bottom = min(top_alt, bottom_alt)
            if top < bottom + 0.5:
                top = bottom + 0.5

            centroid = boundary.mean(axis=0)
            column_pairs: list[tuple[np.ndarray, np.ndarray]] = []

            if curvature_alignment and isinstance(baseline_curve_local, list) and len(baseline_curve_local) >= 2:
                try:
                    curve = np.asarray(_ensure_line(baseline_curve_local), dtype=np.float64)
                except Exception:
                    curve = np.empty((0, 2), dtype=np.float64)
                if len(curve) >= 2:
                    for seg_idx in range(len(curve) - 1):
                        a = curve[seg_idx]
                        b = curve[seg_idx + 1]
                        seg_vec = b - a
                        seg_len = float(np.linalg.norm(seg_vec))
                        if seg_len < 1e-6:
                            continue
                        axis = seg_vec / seg_len
                        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
                        mid = (a + b) * 0.5
                        if float(np.dot(mid - centroid, normal)) < 0.0:
                            normal = -normal
                        if rotate_points_180:
                            normal = -normal
                        sampled = _sample_segment(a, b, horizontal_spacing)
                        if seg_idx > 0 and sampled:
                            sampled = sampled[1:]
                        for base in sampled:
                            facade_point = np.asarray(base, dtype=np.float64)
                            flight_point = facade_point + normal * standoff
                            column_pairs.append((flight_point, facade_point))

            if not column_pairs:
                axis = end - start
                length = float(np.linalg.norm(axis))
                if length < 1e-6:
                    return []
                axis = axis / length
                normal = np.array([-axis[1], axis[0]], dtype=np.float64)
                mid = (start + end) * 0.5
                if float(np.dot(mid - centroid, normal)) < 0.0:
                    normal = -normal
                if rotate_points_180:
                    normal = -normal
                column_points = _sample_segment(start, end, horizontal_spacing)
                if not column_points:
                    return []
                for base in column_points:
                    facade_point = np.asarray(base, dtype=np.float64)
                    flight_point = facade_point + normal * standoff
                    column_pairs.append((flight_point, facade_point))

            span = max(0.5, top - bottom)
            level_count = max(1, int(np.ceil(span / vertical_spacing)))
            altitudes = np.linspace(top, bottom, level_count + 1)
            target_alt_center = 0.5 * (top + bottom)

            poses: list[_CapturePose] = []
            for flight_point, facade_point in column_pairs:
                yaw = _bearing_deg(flight_point, facade_point)

                for alt in altitudes:
                    terrain_delta = (
                        _terrain_delta_m(flight_point, terrain_model if isinstance(terrain_model, dict) else None)
                        if terrain_follow_enabled
                        else 0.0
                    )
                    final_alt = max(1.0, float(alt + ground_offset + terrain_delta))
                    if gimbal_pitch <= -89.0:
                        pitch = float(np.degrees(np.arctan2(target_alt_center - alt, standoff)))
                    else:
                        pitch = gimbal_pitch
                    yaw_out, pitch_out = _apply_terrain_normal_attitude(
                        x_m=float(flight_point[0]),
                        y_m=float(flight_point[1]),
                        yaw_deg=float(yaw),
                        gimbal_pitch_deg=float(np.clip(pitch, -85.0, 10.0)),
                        terrain_model=terrain_model if isinstance(terrain_model, dict) else None,
                        enabled=terrain_normal_enabled,
                        gain=terrain_normal_gain,
                        yaw_align=terrain_normal_yaw_align,
                    )
                    poses.append(
                        _CapturePose(
                            x_m=float(flight_point[0]),
                            y_m=float(flight_point[1]),
                            alt_m=final_alt,
                            yaw_deg=float(yaw_out),
                            gimbal_pitch_deg=float(np.clip(pitch_out, -85.0, 10.0)),
                            primitive=primitive_name,
                            trigger=True,
                        )
                    )
            return poses

        levels = params.get("altitude_levels_m", [60.0])
        if not isinstance(levels, list) or not levels:
            levels = [60.0]
        alt_levels = [float(v) for v in levels]

        standoff = max(0.0, float(params.get("standoff_m", 8.0)))
        spacing = max(1.0, float(params.get("horizontal_spacing_m", 4.0)))
        center = boundary.mean(axis=0)

        facade_points: list[np.ndarray] = []
        for i in range(len(poly) - 1):
            a = poly[i]
            b = poly[i + 1]
            sampled = _sample_segment(a, b, spacing)
            for pt in sampled[:-1]:
                direction = pt - center
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    edge = b - a
                    if float(np.linalg.norm(edge)) < 1e-9:
                        direction = np.array([1.0, 0.0], dtype=np.float64)
                    else:
                        direction = np.array([-edge[1], edge[0]], dtype=np.float64)
                        direction /= max(float(np.linalg.norm(direction)), 1e-6)
                    norm = float(np.linalg.norm(direction))
                direction = direction / max(norm, 1e-6)
                facade_points.append(pt + direction * standoff)

        if not facade_points:
            return []

        poses: list[_CapturePose] = []
        for idx, point in enumerate(facade_points):
            alt_seq = alt_levels if idx % 2 == 0 else list(reversed(alt_levels))
            yaw = float(np.degrees(np.arctan2(center[1] - point[1], center[0] - point[0])))
            for alt in alt_seq:
                poses.append(
                    _CapturePose(
                        x_m=float(point[0]),
                        y_m=float(point[1]),
                        alt_m=float(alt),
                        yaw_deg=yaw,
                        gimbal_pitch_deg=-45.0,
                        primitive=primitive_name,
                        trigger=True,
                    )
                )
        return poses

    def _apply_constraints(
        self,
        poses: list[_CapturePose],
        recipe: FlightRecipe,
    ) -> tuple[list[_CapturePose], dict]:
        constraints = recipe.constraints
        geofence_local = _world_to_local(constraints.geofence, recipe.asset_frame)
        if len(geofence_local) >= 4:
            geofence_local = np.asarray(_ensure_closed_xy(geofence_local.tolist()), dtype=np.float64)
        else:
            geofence_local = np.empty((0, 2), dtype=np.float64)
        no_fly_local: list[np.ndarray] = []
        for poly in constraints.no_fly_polygons:
            try:
                local = _world_to_local(poly, recipe.asset_frame)
                no_fly_local.append(np.asarray(_ensure_closed_xy(local.tolist()), dtype=np.float64))
            except Exception:
                continue

        adjustments = {
            "altitude_clamps": 0,
            "standoff_adjustments": 0,
            "geofence_projections": 0,
            "no_fly_projections": 0,
            "obstacle_detours": 0,
            "terrain_normal_adjustments": 0,
        }

        terrain_model = recipe.metadata.get("terrain_model")
        if not isinstance(terrain_model, dict):
            terrain_model = {}
        terrain_normal_enabled = bool(recipe.metadata.get("terrain_normal_camera_enabled", False))
        terrain_normal_gain = float(np.clip(float(recipe.metadata.get("terrain_normal_gain", 1.0)), 0.0, 3.0))
        terrain_normal_yaw_align = bool(recipe.metadata.get("terrain_normal_yaw_align", False))

        output: list[_CapturePose] = []
        for pose in poses:
            x = float(pose.x_m)
            y = float(pose.y_m)
            alt = float(pose.alt_m)
            yaw = float(pose.yaw_deg)
            gimbal = float(pose.gimbal_pitch_deg)

            clamped_alt = float(np.clip(alt, constraints.min_altitude_m, constraints.max_altitude_m))
            if abs(clamped_alt - alt) > 1e-6:
                adjustments["altitude_clamps"] += 1
            alt = clamped_alt

            radius = sqrt(x * x + y * y)
            if constraints.standoff_m > 0.0 and radius < constraints.standoff_m:
                if radius < 1e-6:
                    x = constraints.standoff_m
                    y = 0.0
                else:
                    scale = constraints.standoff_m / radius
                    x *= scale
                    y *= scale
                adjustments["standoff_adjustments"] += 1

            if len(geofence_local) >= 4:
                pt = np.array([x, y], dtype=np.float64)
                if not _point_in_polygon(pt, geofence_local):
                    projected = _project_inside_polygon(pt, geofence_local)
                    x = float(projected[0])
                    y = float(projected[1])
                    adjustments["geofence_projections"] += 1

            if no_fly_local:
                pt = np.array([x, y], dtype=np.float64)
                for obstacle in no_fly_local:
                    if _point_in_polygon(pt, obstacle):
                        projected = _project_outside_polygon(
                            pt,
                            obstacle,
                            margin_m=max(1.5, float(constraints.standoff_m) * 0.2),
                        )
                        x = float(projected[0])
                        y = float(projected[1])
                        pt = projected
                        adjustments["no_fly_projections"] += 1

            yaw_adj, gimbal_adj = _apply_terrain_normal_attitude(
                x_m=x,
                y_m=y,
                yaw_deg=yaw,
                gimbal_pitch_deg=gimbal,
                terrain_model=terrain_model,
                enabled=terrain_normal_enabled,
                gain=terrain_normal_gain,
                yaw_align=terrain_normal_yaw_align,
            )
            if abs(yaw_adj - yaw) > 1e-6 or abs(gimbal_adj - gimbal) > 1e-6:
                adjustments["terrain_normal_adjustments"] += 1

            output.append(
                replace(
                    pose,
                    x_m=x,
                    y_m=y,
                    alt_m=alt,
                    yaw_deg=float(yaw_adj),
                    gimbal_pitch_deg=float(gimbal_adj),
                )
            )

        if len(output) < 2 or not no_fly_local:
            return output, adjustments

        detoured: list[_CapturePose] = [output[0]]
        for pose in output[1:]:
            prev = detoured[-1]
            start = np.asarray([prev.x_m, prev.y_m], dtype=np.float64)
            end = np.asarray([pose.x_m, pose.y_m], dtype=np.float64)
            detour_nodes: list[np.ndarray] = []
            for obstacle in no_fly_local:
                if _segment_intersects_polygon(start, end, obstacle):
                    detour_nodes = _detour_points_around_polygon(
                        start,
                        end,
                        obstacle,
                        margin_m=max(2.0, float(constraints.standoff_m) * 0.25),
                    )
                    if detour_nodes:
                        break
            if detour_nodes:
                for node in detour_nodes:
                    x = float(node[0])
                    y = float(node[1])
                    pt = np.array([x, y], dtype=np.float64)
                    if len(geofence_local) >= 4 and not _point_in_polygon(pt, geofence_local):
                        projected = _project_inside_polygon(pt, geofence_local)
                        x = float(projected[0])
                        y = float(projected[1])
                        pt = projected
                    for obstacle in no_fly_local:
                        if _point_in_polygon(pt, obstacle):
                            projected = _project_outside_polygon(
                                pt,
                                obstacle,
                                margin_m=max(1.5, float(constraints.standoff_m) * 0.2),
                            )
                            x = float(projected[0])
                            y = float(projected[1])
                            pt = projected

                    prev_xy = np.asarray([detoured[-1].x_m, detoured[-1].y_m], dtype=np.float64)
                    yaw = _bearing_deg(prev_xy, np.asarray([x, y], dtype=np.float64))
                    detour_alt = float(
                        np.clip(
                            max(float(prev.alt_m), float(pose.alt_m)),
                            float(constraints.min_altitude_m),
                            float(constraints.max_altitude_m),
                        )
                    )
                    detoured.append(
                        _CapturePose(
                            x_m=x,
                            y_m=y,
                            alt_m=detour_alt,
                            yaw_deg=float(yaw),
                            gimbal_pitch_deg=float(prev.gimbal_pitch_deg),
                            primitive="obstacle_detour",
                            trigger=False,
                            dwell_s=0.0,
                            camera_yaw_locked=False,
                        )
                    )
                adjustments["obstacle_detours"] += int(len(detour_nodes))

            detoured.append(pose)

        return detoured, adjustments

    def _local_to_world_poses(
        self,
        poses: list[_CapturePose],
        frame: AssetReferenceFrame,
    ) -> list[dict]:
        if not poses:
            return []
        local_points = np.asarray([[p.x_m, p.y_m] for p in poses], dtype=np.float64)
        world_points = _local_to_world(local_points, frame)

        out: list[dict] = []
        for pose, ll in zip(poses, world_points):
            out.append(
                {
                    "lon": float(ll[0]),
                    "lat": float(ll[1]),
                    "alt_m": float(pose.alt_m),
                    "yaw_deg": _wrap_deg(float(pose.yaw_deg) + float(frame.yaw_deg)),
                    "gimbal_pitch_deg": float(pose.gimbal_pitch_deg),
                    "primitive": pose.primitive,
                    "trigger": bool(pose.trigger),
                    "dwell_s": float(pose.dwell_s),
                    "camera_yaw_locked": bool(pose.camera_yaw_locked),
                    "x_local_m": float(pose.x_m),
                    "y_local_m": float(pose.y_m),
                }
            )
        return out

    def _build_autopilot_commands(
        self,
        world_poses: list[dict],
        constraints: MissionConstraints,
        speed_m_s: float,
        repeat_enabled: bool,
        continuous_capture: bool = False,
        capture_spacing_m: float = 0.0,
        capture_interval_s: float = 0.0,
        camera_policy: dict[str, Any] | None = None,
        wind_model: dict[str, Any] | None = None,
    ) -> list[dict]:
        commands: list[dict] = []
        seq = 0

        commands.append(
            {
                "seq": seq,
                "command": "SET_MISSION_POLICY",
                "repeat_mode": bool(repeat_enabled),
                "obstacle_avoidance_profile": constraints.obstacle_avoidance_profile,
                "source": "recipe_compiler",
            }
        )
        seq += 1

        commands.append(
            {
                "seq": seq,
                "command": "SET_GEOFENCE",
                "polygon": constraints.geofence,
            }
        )
        seq += 1

        if constraints.no_fly_polygons:
            commands.append(
                {
                    "seq": seq,
                    "command": "SET_NO_FLY_ZONES",
                    "polygons": constraints.no_fly_polygons,
                }
            )
            seq += 1

        commands.append(
            {
                "seq": seq,
                "command": "SET_ALTITUDE_BAND",
                "min_altitude_m": constraints.min_altitude_m,
                "max_altitude_m": constraints.max_altitude_m,
                "standoff_m": constraints.standoff_m,
            }
        )
        seq += 1

        if isinstance(wind_model, dict) and (
            float(wind_model.get("wind_speed_m_s", 0.0)) > 0.0
            or float(wind_model.get("wind_gust_m_s", 0.0)) > 0.0
        ):
            commands.append(
                {
                    "seq": seq,
                    "command": "SET_WIND_MODEL",
                    "wind_speed_m_s": float(wind_model.get("wind_speed_m_s", 0.0)),
                    "wind_direction_deg": float(wind_model.get("wind_direction_deg", 0.0)),
                    "wind_gust_m_s": float(wind_model.get("wind_gust_m_s", 0.0)),
                    "crosswind_m_s": float(wind_model.get("crosswind_m_s", 0.0)),
                    "headwind_m_s": float(wind_model.get("headwind_m_s", 0.0)),
                    "penalty_pct": float(wind_model.get("penalty_pct", 0.0)),
                }
            )
            seq += 1

        if isinstance(camera_policy, dict) and camera_policy:
            commands.append(
                {
                    "seq": seq,
                    "command": "SET_CAMERA_POLICY",
                    "policy": dict(camera_policy),
                }
            )
            seq += 1

        if bool(continuous_capture) and float(capture_spacing_m) > 0.0:
            commands.append(
                {
                    "seq": seq,
                    "command": "SET_CAMERA_TRIGGER_DISTANCE",
                    "distance_m": float(capture_spacing_m),
                    "min_interval_s": float(max(0.0, capture_interval_s)),
                    "capture_mode": "continuous_photo",
                }
            )
            seq += 1

        current_linear_segment: int | None = None
        current_tower_level: int | None = None
        for pose in world_poses:
            primitive_name = str(pose.get("primitive", ""))
            if primitive_name.startswith("linear_inspection_seg"):
                try:
                    seg_id = int(primitive_name.split("_seg", 1)[1])
                except Exception:
                    seg_id = None
                if seg_id is not None and seg_id != current_linear_segment:
                    current_linear_segment = seg_id
                    commands.append(
                        {
                            "seq": seq,
                            "command": "MISSION_SPLIT_HINT",
                            "segment_index": int(seg_id),
                            "reason": "battery_aware_linear_inspection_split",
                        }
                    )
                    seq += 1
            if primitive_name.startswith("tower_mapping_level"):
                try:
                    level_id = int(primitive_name.split("_level", 1)[1])
                except Exception:
                    level_id = None
                if level_id is not None and level_id != current_tower_level:
                    current_tower_level = level_id
                    commands.append(
                        {
                            "seq": seq,
                            "command": "MISSION_SPLIT_HINT",
                            "segment_index": int(level_id),
                            "reason": "battery_change_tower_resume",
                        }
                    )
                    seq += 1

            commands.append(
                {
                    "seq": seq,
                    "command": "NAV_WAYPOINT",
                    "frame": "GLOBAL_RELATIVE_ALT",
                    "lat": float(pose["lat"]),
                    "lon": float(pose["lon"]),
                    "alt_m": float(pose["alt_m"]),
                    "yaw_deg": float(pose["yaw_deg"]),
                    "speed_m_s": float(speed_m_s),
                    "primitive": primitive_name,
                }
            )
            seq += 1

            commands.append(
                {
                    "seq": seq,
                    "command": "SET_GIMBAL",
                    "pitch_deg": float(pose.get("gimbal_pitch_deg", -90.0)),
                }
            )
            seq += 1

            if bool(pose.get("camera_yaw_locked", False)):
                commands.append(
                    {
                        "seq": seq,
                        "command": "SET_YAW",
                        "yaw_deg": float(pose.get("yaw_deg", 0.0)),
                    }
                )
                seq += 1

            dwell_s = float(pose.get("dwell_s", 0.0))
            if dwell_s > 0.0:
                commands.append(
                    {
                        "seq": seq,
                        "command": "HOLD_POSITION",
                        "duration_s": dwell_s,
                        "reason": "stabilize_for_inspection_capture",
                    }
                )
                seq += 1

            if not continuous_capture and bool(pose.get("trigger", True)):
                commands.append(
                    {
                        "seq": seq,
                        "command": "CAMERA_CAPTURE",
                        "capture_mode": "single_photo",
                    }
                )
                seq += 1

        if bool(continuous_capture) and float(capture_spacing_m) > 0.0:
            commands.append(
                {
                    "seq": seq,
                    "command": "STOP_CAMERA_TRIGGER",
                }
            )
            seq += 1

        if world_poses:
            home_lon = float(world_poses[0]["lon"])
            home_lat = float(world_poses[0]["lat"])
        elif constraints.geofence:
            home_lon = float(constraints.geofence[0][0])
            home_lat = float(constraints.geofence[0][1])
        else:
            home_lon = 0.0
            home_lat = 0.0

        commands.append(
            {
                "seq": seq,
                "command": "RTH",
                "action": constraints.rth_action,
                "home_lon": home_lon,
                "home_lat": home_lat,
                "rth_altitude_m": float(constraints.rth_altitude_m),
            }
        )
        return commands

    def _coverage_report(self, recipe: FlightRecipe, achieved_viewpoints: int) -> dict:
        required = int(recipe.coverage.required_viewpoints)
        if required <= 0:
            required = max(1, int(np.ceil(achieved_viewpoints * 0.95)))
        coverage_pct = min(100.0, 100.0 * achieved_viewpoints / required) if required > 0 else 100.0
        return {
            "required_viewpoints": int(required),
            "achieved_viewpoints": int(achieved_viewpoints),
            "target_front_overlap_pct": float(recipe.coverage.front_overlap_pct),
            "target_side_overlap_pct": float(recipe.coverage.side_overlap_pct),
            "minimum_coverage_pct": float(recipe.coverage.minimum_coverage_pct),
            "achieved_coverage_pct": float(coverage_pct),
            "meets_target": bool(coverage_pct >= float(recipe.coverage.minimum_coverage_pct)),
        }


def export_geojson(path: str | Path, mission: MissionPlan) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(mission.geojson, handle, indent=2)
    return str(target)


def export_qgc_wpl(path: str | Path, mission: MissionPlan) -> str:
    """Export mission as QGroundControl text mission file (WPL 110)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = ["QGC WPL 110"]
    if not mission.waypoints:
        lon, lat = mission.polygon[0][0], mission.polygon[0][1]
        waypoints = [[lon, lat, mission.altitude_m]]
    else:
        waypoints = mission.waypoints

    for idx, row in enumerate(waypoints):
        lon = float(row[0])
        lat = float(row[1])
        alt = float(row[2]) if len(row) >= 3 else float(mission.altitude_m)
        current = 1 if idx == 0 else 0
        frame = 3
        command = 16
        autocontinue = 1
        lines.append(
            f"{idx}\t{current}\t{frame}\t{command}\t0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{alt:.2f}\t{autocontinue}"
        )

    with target.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return str(target)


def export_flight_recipe(path: str | Path, payload: FlightRecipe | MissionPlan | dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, FlightRecipe):
        data = payload.to_dict()
    elif isinstance(payload, MissionPlan):
        data = payload.flight_recipe
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("payload must be FlightRecipe, MissionPlan, or dict")

    if not isinstance(data, dict) or not data:
        raise ValueError("No flight recipe data available to export.")

    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return str(target)


def load_flight_recipe(path: str | Path) -> FlightRecipe:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Flight recipe file must contain a JSON object.")
    return _flight_recipe_from_dict(payload)


