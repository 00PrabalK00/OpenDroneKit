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
    """Estimate crack length by skeletonizing the binary mask."""
    try:
        from skimage.morphology import skeletonize
        binary = (mask > 0).astype(np.uint8)
        skeleton = skeletonize(binary)
        crack_px = float(np.sum(skeleton))
    except ImportError:
        crack_px = float(np.sum(mask > 0)) / 3.0

    return crack_px * float(scale_mm_per_px) / 1000.0


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
    )


def measure_polygon_area(
    project_root: Path,
    project_id: str,
    source_id: str,
    vertices: list[tuple[int, int]],
    scale_mm_per_px: float,
    label: str = "",
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
    )


def measure_crack_length(
    project_root: Path,
    project_id: str,
    source_id: str,
    mask: "np.ndarray",
    scale_mm_per_px: float,
    label: str = "",
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
    )
