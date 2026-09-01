"""Annotation engine — user marks on images, maps and 3D views; stored per project."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Any, Literal, Sequence

ANNOTATION_TYPES = Literal[
    "point", "line", "polygon", "rectangle", "circle", "freehand", "text"
]

SEVERITY_LEVELS = Literal["critical", "high", "medium", "low", "info"]
ANNOTATION_STATUSES = Literal["open", "in_review", "resolved", "dismissed"]

SUPPORTED_ANNOTATION_TYPES = {
    "point", "line", "polygon", "rectangle", "circle", "freehand", "text",
}
SUPPORTED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
SUPPORTED_STATUSES = {"open", "in_review", "resolved", "dismissed"}
SUPPORTED_SOURCES = {"image", "map", "3d"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Annotation:
    id: str
    project_id: str
    source_type: str      # "image" | "map" | "3d"
    source_id: str        # image path, layer id, or point_index
    annotation_type: str  # see ANNOTATION_TYPES
    geometry: dict[str, Any]
    label: str
    severity: str
    status: str
    note: str | None = None
    #: Free-form labels for grouping. Distinct from `label`, which names WHAT the finding
    #: is, and from `severity`, which says how bad. A tag is how an inspector slices the
    #: set afterwards -- "north elevation", "reflight", "client query" -- and one finding
    #: routinely needs several.
    tags: list[str] = field(default_factory=list)
    include_in_report: bool = True
    created_by: str = "user"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        validate_annotation(
            source_type=self.source_type,
            annotation_type=self.annotation_type,
            geometry=self.geometry,
            label=self.label,
            severity=self.severity,
            status=self.status,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Annotation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Storage ───────────────────────────────────────────────────────────────────

def _store_path(project_root: Path) -> Path:
    path = project_root / "analysis" / "annotations" / "annotations.json"
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


def _point(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two coordinates.")
    point = [float(value[0]), float(value[1])]
    if not all(math.isfinite(number) for number in point):
        raise ValueError(f"{name} coordinates must be finite.")
    return point


def validate_annotation(
    *,
    source_type: str,
    annotation_type: str,
    geometry: dict[str, Any],
    label: str,
    severity: str,
    status: str,
) -> None:
    """Validate the shared core/browser annotation contract."""

    if source_type not in SUPPORTED_SOURCES:
        raise ValueError(f"source_type must be one of: {', '.join(sorted(SUPPORTED_SOURCES))}.")
    if annotation_type not in SUPPORTED_ANNOTATION_TYPES:
        raise ValueError(
            f"annotation_type must be one of: {', '.join(sorted(SUPPORTED_ANNOTATION_TYPES))}."
        )
    if severity not in SUPPORTED_SEVERITIES:
        raise ValueError(f"severity must be one of: {', '.join(sorted(SUPPORTED_SEVERITIES))}.")
    if status not in SUPPORTED_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(SUPPORTED_STATUSES))}.")
    if not str(label).strip():
        raise ValueError("Annotation label is required.")
    if not isinstance(geometry, dict):
        raise ValueError("Annotation geometry must be an object.")

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if annotation_type in {"point", "circle", "text"}:
        if geometry_type != "Point":
            raise ValueError(f"{annotation_type} annotations require Point geometry.")
        _point(coordinates, "Point")
        if annotation_type == "circle":
            radius = float(geometry.get("radius_m", 0))
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Circle annotations require a positive finite radius_m.")
        return

    if annotation_type in {"line", "freehand"}:
        if geometry_type != "LineString" or not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValueError(f"{annotation_type} annotations require a two-point LineString.")
        for index, value in enumerate(coordinates):
            _point(value, f"Line vertex {index + 1}")
        return

    if geometry_type != "Polygon" or not isinstance(coordinates, list) or not coordinates:
        raise ValueError(f"{annotation_type} annotations require Polygon geometry.")
    ring = coordinates[0]
    if not isinstance(ring, list) or len(ring) < 4:
        raise ValueError("Polygon annotations require a closed ring with at least four positions.")
    checked = [_point(value, f"Polygon vertex {index + 1}") for index, value in enumerate(ring)]
    if checked[0] != checked[-1]:
        raise ValueError("Polygon annotation ring must be closed.")
    if annotation_type == "rectangle":
        unique = {tuple(point) for point in checked[:-1]}
        xs = {point[0] for point in unique}
        ys = {point[1] for point in unique}
        if len(unique) != 4 or len(xs) != 2 or len(ys) != 2:
            raise ValueError("Rectangle annotation must have four axis-aligned corners.")


# ── Public API ────────────────────────────────────────────────────────────────

def create_annotation(
    project_root: Path,
    project_id: str,
    source_type: str,
    source_id: str,
    annotation_type: str,
    geometry: dict[str, Any],
    label: str,
    severity: str,
    status: str,
    note: str | None = None,
    include_in_report: bool = True,
    created_by: str = "user",
) -> Annotation:
    a = Annotation(
        id=str(uuid.uuid4()),
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        annotation_type=annotation_type,
        geometry=geometry,
        label=label,
        severity=severity,
        status=status,
        note=note,
        include_in_report=include_in_report,
        created_by=created_by,
    )
    items = _load_store(project_root)
    items.append(a.to_dict())
    _save_store(project_root, items)
    return a


def list_annotations(
    project_root: Path,
    project_id: str | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    tag: str | None = None,
) -> list[Annotation]:
    items = _load_store(project_root)
    result = []
    for d in items:
        try:
            a = Annotation.from_dict(d)
            if project_id and a.project_id != project_id:
                continue
            if source_id and a.source_id != source_id:
                continue
            if source_type and a.source_type != source_type:
                continue
            if tag and normalise_tag(tag) not in [normalise_tag(t) for t in a.tags]:
                continue
            result.append(a)
        except Exception:
            pass
    return result


def update_annotation(
    project_root: Path,
    annotation_id: str,
    patch: dict[str, Any],
) -> Annotation | None:
    items = _load_store(project_root)
    readonly = {"id", "project_id", "created_at", "created_by"}
    for i, d in enumerate(items):
        if d.get("id") == annotation_id:
            for k, v in patch.items():
                if k in Annotation.__dataclass_fields__ and k not in readonly:
                    d[k] = v
            d["updated_at"] = _now_iso()
            validated = Annotation.from_dict(d)
            items[i] = validated.to_dict()
            _save_store(project_root, items)
            return validated
    return None


def normalise_tag(tag: str) -> str:
    """One spelling per tag.

    "North Elevation", "north elevation" and " north elevation " are one tag to an
    inspector and three to a filter, which is how a tag list becomes useless by the
    second survey.
    """
    return " ".join(str(tag).strip().lower().split())


def add_tags(
    project_root: Path,
    annotation_ids: Sequence[str],
    tags: Sequence[str],
) -> dict[str, Any]:
    """Apply tags to many annotations at once.

    The bulk case is the normal case: an inspector reviews forty roof photographs and
    wants them all marked "north elevation". Doing that one at a time is why people stop
    tagging, and an untagged set cannot be filtered, reported on or handed over.

    Reports which ids were not found rather than failing the whole call -- a stale id in a
    selection should not discard the other thirty-nine.
    """
    wanted = [t for t in (normalise_tag(t) for t in tags) if t]
    if not wanted:
        raise ValueError("No tags given.")

    items = _load_store(project_root)
    by_id = {d.get("id"): d for d in items}
    updated: list[str] = []
    missing: list[str] = []

    for annotation_id in annotation_ids:
        record = by_id.get(annotation_id)
        if record is None:
            missing.append(annotation_id)
            continue
        existing = [normalise_tag(t) for t in (record.get("tags") or [])]
        merged = list(dict.fromkeys(existing + wanted))
        if merged != existing:
            record["tags"] = merged
            record["updated_at"] = _now_iso()
            updated.append(annotation_id)

    if updated:
        _save_store(project_root, items)
    return {"updated": updated, "missing": missing, "tags": wanted}


def remove_tags(
    project_root: Path,
    annotation_ids: Sequence[str],
    tags: Sequence[str],
) -> dict[str, Any]:
    """Take tags off many annotations at once."""
    unwanted = {t for t in (normalise_tag(t) for t in tags) if t}
    if not unwanted:
        raise ValueError("No tags given.")

    items = _load_store(project_root)
    by_id = {d.get("id"): d for d in items}
    updated: list[str] = []
    missing: list[str] = []

    for annotation_id in annotation_ids:
        record = by_id.get(annotation_id)
        if record is None:
            missing.append(annotation_id)
            continue
        existing = [normalise_tag(t) for t in (record.get("tags") or [])]
        kept = [t for t in existing if t not in unwanted]
        if kept != existing:
            record["tags"] = kept
            record["updated_at"] = _now_iso()
            updated.append(annotation_id)

    if updated:
        _save_store(project_root, items)
    return {"updated": updated, "missing": missing, "tags": sorted(unwanted)}


def all_tags(project_root: Path, project_id: str | None = None) -> list[dict[str, Any]]:
    """Every tag in use, with how many findings carry it.

    The count is what makes the list usable: a tag on one finding out of four hundred is
    usually a typo, and it shows up here next to the one it should have been.
    """
    counts: dict[str, int] = {}
    for annotation in list_annotations(project_root, project_id=project_id):
        for tag in annotation.tags:
            key = normalise_tag(tag)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return [{"tag": tag, "count": counts[tag]} for tag in sorted(counts)]


def delete_annotation(project_root: Path, annotation_id: str) -> bool:
    items = _load_store(project_root)
    new_items = [d for d in items if d.get("id") != annotation_id]
    if len(new_items) == len(items):
        return False
    _save_store(project_root, new_items)
    return True


def get_annotation(project_root: Path, annotation_id: str) -> Annotation | None:
    for d in _load_store(project_root):
        if d.get("id") == annotation_id:
            try:
                return Annotation.from_dict(d)
            except Exception:
                return None
    return None


def export_annotations(project_root: Path, project_id: str, output_format: str = "json") -> Path:
    annotations = list_annotations(project_root, project_id)
    out_dir = project_root / "analysis" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        import csv
        out_path = out_dir / "annotations_export.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "annotation_type", "label", "severity",
                "source_type", "source_id", "note", "include_in_report", "created_at",
            ])
            writer.writeheader()
            for a in annotations:
                writer.writerow({
                    "id": a.id, "annotation_type": a.annotation_type,
                    "label": a.label, "severity": a.severity or "",
                    "source_type": a.source_type, "source_id": a.source_id,
                    "note": a.note or "", "include_in_report": a.include_in_report,
                    "created_at": a.created_at,
                })
        return out_path

    out_path = out_dir / "annotations_export.json"
    out_path.write_text(
        json.dumps([a.to_dict() for a in annotations], indent=2),
        encoding="utf-8",
    )
    return out_path


def get_report_annotations(project_root: Path, project_id: str) -> list[Annotation]:
    """Return only annotations marked for report inclusion."""
    return [a for a in list_annotations(project_root, project_id) if a.include_in_report]
