"""Measurement engine — distance, area, crack length, volume; stored per project."""

from __future__ import annotations

import csv
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Measurement:
    id: str
    project_id: str
    source_type: str          # "image" | "map" | "3d" | "crack_mask"
    source_id: str            # path or id of source
    measurement_type: str     # "distance" | "area" | "crack_length" | "polygon_area" | "volume" | "angle"
    geometry: dict[str, Any]  # GeoJSON-style or pixel coords
    value: float
    unit: str                 # "m" | "m2" | "m3" | "px" | "deg"
    confidence: float | None = None
    label: str = ""
    notes: str = ""
    include_in_report: bool = True
    created_at: str = field(default_factory=_now_iso)
    # The full result the value was taken from -- reference surfaces, CRS, caveats --
    # so a figure in the history can always be traced to how it was measured.
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measurement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Storage ───────────────────────────────────────────────────────────────────

def _store_path(project_root: Path) -> Path:
    path = project_root / "analysis" / "measurements" / "measurements.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_store(project_root: Path) -> list[dict[str, Any]]:
    p = _store_path(project_root)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_store(project_root: Path, items: list[dict[str, Any]]) -> None:
    _store_path(project_root).write_text(json.dumps(items, indent=2), encoding="utf-8")


# ── Calculation helpers ───────────────────────────────────────────────────────

def calculate_pixel_distance(
    p1: tuple[int, int],
    p2: tuple[int, int],
    scale_mm_per_px: float,
) -> float:
    """Euclidean pixel distance converted to metres."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    px = math.sqrt(dx * dx + dy * dy)
    return px * float(scale_mm_per_px) / 1000.0


def calculate_polygon_area_px(
    vertices: list[tuple[int, int]],
    scale_mm_per_px: float,
) -> float:
    """Shoelace formula for polygon area in m²."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area_px2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area_px2 += vertices[i][0] * vertices[j][1]
        area_px2 -= vertices[j][0] * vertices[i][1]
    area_px2 = abs(area_px2) / 2.0
    m_per_px = float(scale_mm_per_px) / 1000.0
    return area_px2 * (m_per_px ** 2)


def calculate_geo_area(polygon_coords: list[tuple[float, float]]) -> float:
    """Approximate area in m² from lat/lon polygon (equirectangular)."""
    if len(polygon_coords) < 3:
        return 0.0
    lat0 = sum(p[0] for p in polygon_coords) / len(polygon_coords)
    R = 6_371_000.0
    lat_m = math.radians(1.0) * R
    lon_m = math.radians(1.0) * R * math.cos(math.radians(lat0))
    verts = [(p[1] * lon_m, p[0] * lat_m) for p in polygon_coords]
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1]
        area -= verts[j][0] * verts[i][1]
    return abs(area) / 2.0


def calculate_crack_length_from_mask(mask: "np.ndarray", scale_mm_per_px: float) -> float:
    """Estimate crack length by skeletonizing the binary mask.

    Without scikit-image there is no skeleton, and dividing the mask's pixel count by a
    constant is a guess at the crack's width presented as its length. Refused instead.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError(
            "Crack length needs scikit-image to skeletonise the mask "
            "(`pip install scikit-image`); it is not estimated without it."
        ) from exc
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    crack_px = skeleton_length_px(skeleton)

    return crack_px * float(scale_mm_per_px) / 1000.0


# Skeleton pixels are sampled every this many steps along a branch and joined by straight
# chords. Counting the pixels instead makes a 45-degree crack 29% short (a diagonal step
# is sqrt(2) px, not 1), and counting 1 / sqrt(2) per step makes any other angle up to 8%
# long (the staircase of an 8-connected line). Chords over a few pixels follow the line.
SKELETON_CHORD_PX = 8


def skeleton_length_px(skeleton: "np.ndarray", chord_px: int = SKELETON_CHORD_PX) -> float:
    """Length in pixels of a one-pixel-wide skeleton, measured along its branches.

    The skeleton is walked as a graph (4-neighbours, and diagonal neighbours only where
    no 4-neighbour path already joins them, so a staircase corner is not counted twice).
    Each branch between end points / junctions -- and each closed loop -- is measured as
    a polyline through every ``chord_px``-th pixel and its last pixel.
    """
    sk = np.asarray(skeleton).astype(bool)
    ys, xs = np.nonzero(sk)
    if ys.size == 0:
        return 0.0
    on = set(zip(ys.tolist(), xs.tolist()))

    def neighbours(p: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = p
        out = [q for q in ((y, x + 1), (y + 1, x), (y, x - 1), (y - 1, x)) if q in on]
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (y + dy, x + dx)
            if q in on and (y, x + dx) not in on and (y + dy, x) not in on:
                out.append(q)
        return out

    ordered = sorted(on)  # a fixed walking order, so the same mask gives the same number
    adjacent = {p: neighbours(p) for p in ordered}
    walked: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    step = max(1, int(chord_px))

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def polyline_length(path: list[tuple[int, int]]) -> float:
        picks = list(range(0, len(path), step))
        if picks[-1] != len(path) - 1:
            picks.append(len(path) - 1)
        return float(sum(math.hypot(path[i][0] - path[j][0], path[i][1] - path[j][1])
                         for i, j in zip(picks, picks[1:])))

    def walk(start, nxt) -> list[tuple[int, int]]:
        path = [start, nxt]
        walked.add(key(start, nxt))
        prev, cur = start, nxt
        while len(adjacent[cur]) == 2:
            a, b = adjacent[cur]
            following = b if a == prev else a
            if key(cur, following) in walked:
                break
            walked.add(key(cur, following))
            path.append(following)
            prev, cur = cur, following
        return path

    total = 0.0
    # Branches start at end points and junctions (anything without exactly two neighbours).
    for p in ordered:
        if len(adjacent[p]) != 2:
            for q in adjacent[p]:
                if key(p, q) not in walked:
                    total += polyline_length(walk(p, q))
    # What remains are closed loops, where every pixel has two neighbours.
    for p in ordered:
        for q in adjacent[p]:
            if key(p, q) not in walked:
                total += polyline_length(walk(p, q))
    return total


# ── Public API ────────────────────────────────────────────────────────────────

def create_measurement(
    project_root: Path,
    project_id: str,
    source_type: str,
    source_id: str,
    measurement_type: str,
    geometry: dict[str, Any],
    value: float,
    unit: str,
    label: str = "",
    notes: str = "",
    confidence: float | None = None,
    include_in_report: bool = True,
    details: dict[str, Any] | None = None,
) -> Measurement:
    m = Measurement(
        id=str(uuid.uuid4()),
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        measurement_type=measurement_type,
        geometry=geometry,
        value=value,
        unit=unit,
        confidence=confidence,
        label=label,
        notes=notes,
        include_in_report=include_in_report,
        details=dict(details or {}),
    )
    items = _load_store(project_root)
    items.append(m.to_dict())
    _save_store(project_root, items)
    return m


def list_measurements(project_root: Path, project_id: str | None = None) -> list[Measurement]:
    items = _load_store(project_root)
    result = []
    for d in items:
        try:
            m = Measurement.from_dict(d)
            if project_id is None or m.project_id == project_id:
                result.append(m)
        except Exception:
            pass
    return result


def delete_measurement(project_root: Path, measurement_id: str) -> bool:
    items = _load_store(project_root)
    new_items = [d for d in items if d.get("id") != measurement_id]
    if len(new_items) == len(items):
        return False
    _save_store(project_root, new_items)
    return True


def update_measurement(project_root: Path, measurement_id: str, patch: dict[str, Any]) -> Measurement | None:
    items = _load_store(project_root)
    readonly = {"id", "project_id", "created_at"}
    for i, d in enumerate(items):
        if d.get("id") == measurement_id:
            for k, v in patch.items():
                if k not in readonly:
                    d[k] = v
            items[i] = d
            _save_store(project_root, items)
            return Measurement.from_dict(d)
    return None


def export_measurements(
    project_root: Path,
    project_id: str,
    output_format: str = "csv",
) -> Path:
    """Export measurements to CSV or JSON."""
    measurements = list_measurements(project_root, project_id)
    out_dir = project_root / "analysis" / "measurements"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        out_path = out_dir / "measurements_export.json"
        out_path.write_text(
            json.dumps([m.to_dict() for m in measurements], indent=2),
            encoding="utf-8",
        )
        return out_path

    # CSV
    out_path = out_dir / "measurements_export.csv"
    if not measurements:
        out_path.write_text("id,type,value,unit,label,source_type,source_id,created_at\n", encoding="utf-8")
        return out_path

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "measurement_type", "value", "unit", "label",
            "source_type", "source_id", "notes", "include_in_report", "created_at"
        ])
        writer.writeheader()
        for m in measurements:
            writer.writerow({
                "id": m.id,
                "measurement_type": m.measurement_type,
                "value": round(m.value, 6),
                "unit": m.unit,
                "label": m.label,
                "source_type": m.source_type,
                "source_id": m.source_id,
                "notes": m.notes,
                "include_in_report": m.include_in_report,
                "created_at": m.created_at,
            })
    return out_path


def measure_image_distance(
    project_root: Path,
    project_id: str,
    source_id: str,
    p1: tuple[int, int],
    p2: tuple[int, int],
    scale_mm_per_px: float,
    label: str = "",
    details: dict[str, Any] | None = None,
) -> Measurement:
    value = calculate_pixel_distance(p1, p2, scale_mm_per_px)
    return create_measurement(
        project_root=project_root,
        project_id=project_id,
        source_type="image",
        source_id=source_id,
        measurement_type="distance",
        geometry={"type": "LineString", "pixel_coords": [list(p1), list(p2)]},
        value=value,
        unit="m",
        label=label or f"Distance {round(value, 3)} m",
        details=details,
    )


def measure_polygon_area(
    project_root: Path,
    project_id: str,
    source_id: str,
    vertices: list[tuple[int, int]],
    scale_mm_per_px: float,
    label: str = "",
    details: dict[str, Any] | None = None,
) -> Measurement:
    value = calculate_polygon_area_px(vertices, scale_mm_per_px)
    return create_measurement(
        project_root=project_root,
        project_id=project_id,
        source_type="image",
        source_id=source_id,
        measurement_type="area",
        geometry={"type": "Polygon", "pixel_coords": [list(v) for v in vertices]},
        value=value,
        unit="m2",
        label=label or f"Area {round(value, 4)} m²",
        details=details,
    )


def measure_crack_length(
    project_root: Path,
    project_id: str,
    source_id: str,
    mask: "np.ndarray",
    scale_mm_per_px: float,
    label: str = "",
    details: dict[str, Any] | None = None,
) -> Measurement:
    value = calculate_crack_length_from_mask(mask, scale_mm_per_px)
    return create_measurement(
        project_root=project_root,
        project_id=project_id,
        source_type="crack_mask",
        source_id=source_id,
        measurement_type="crack_length",
        geometry={"type": "CrackMask", "source": source_id},
        value=value,
        unit="m",
        label=label or f"Crack length {round(value, 4)} m",
        details=details,
    )
