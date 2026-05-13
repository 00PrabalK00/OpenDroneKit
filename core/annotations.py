"""Annotation engine — user marks on images, maps and 3D views; stored per project."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ANNOTATION_TYPES = Literal[
    "point", "rectangle", "polygon", "free_text",
    "severity_tag", "defect_confirm", "false_positive", "report_highlight"
]

SEVERITY_LEVELS = Literal["critical", "high", "medium", "low", "info"]


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
    severity: str | None = None
    note: str | None = None
    include_in_report: bool = True
    created_by: str = "user"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

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


# ── Public API ────────────────────────────────────────────────────────────────

def create_annotation(
    project_root: Path,
    project_id: str,
    source_type: str,
    source_id: str,
    annotation_type: str,
    geometry: dict[str, Any],
    label: str,
    severity: str | None = None,
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
                if k not in readonly:
                    d[k] = v
            d["updated_at"] = _now_iso()
            items[i] = d
            _save_store(project_root, items)
            return Annotation.from_dict(d)
    return None


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
