"""Project-contained annotation CRUD for the browser Hub."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.annotations import validate_annotation
from core.detection import detect_structural_defects
from core.models import model_status

from ..audit import record
from ..db import dumps_geometry, get_db, loads_geometry
from ..models import AnnotationRecord, Project, Role, utcnow
from ..security import CurrentUser, require_project_role
from ..storage import StorageError, build_storage

router = APIRouter(tags=["annotations"])


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class AnnotationCreate(BaseModel):
    source_type: str = "map"
    source_id: str = Field(min_length=1, max_length=500)
    annotation_type: str
    geometry: dict[str, Any]
    crs_epsg: int | None = None
    label: str = Field(min_length=1, max_length=500)
    severity: str
    status: str
    note: str = ""
    include_in_report: bool = True


class AnnotationPatch(BaseModel):
    annotation_type: str | None = None
    geometry: dict[str, Any] | None = None
    crs_epsg: int | None = None
    label: str | None = Field(default=None, min_length=1, max_length=500)
    severity: str | None = None
    status: str | None = None
    note: str | None = None
    include_in_report: bool | None = None


class AnnotationOut(BaseModel):
    id: int
    project_id: int
    source_type: str
    source_id: str
    annotation_type: str
    geometry: dict[str, Any]
    crs_epsg: int | None
    label: str
    severity: str
    status: str
    note: str
    include_in_report: bool
    origin: str
    machine_claims: list[dict[str, Any]]
    review_action: str
    parent_ids: list[int]
    reviewed_by: int | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class AnnotationReview(BaseModel):
    action: Literal["accept", "edit", "reclassify"]
    geometry: dict[str, Any] | None = None
    label: str | None = Field(default=None, min_length=1, max_length=500)
    severity: str | None = None
    status: str | None = None
    note: str | None = None


class MergeAnnotations(BaseModel):
    annotation_ids: list[int] = Field(min_length=2)
    annotation_type: str
    geometry: dict[str, Any]
    label: str = Field(min_length=1, max_length=500)
    severity: str
    status: str = "in_review"
    note: str = ""
    include_in_report: bool = True


class SplitPart(BaseModel):
    annotation_type: str
    geometry: dict[str, Any]
    label: str = Field(min_length=1, max_length=500)
    severity: str
    status: str = "in_review"
    note: str = ""
    include_in_report: bool = True


class SplitAnnotation(BaseModel):
    parts: list[SplitPart] = Field(min_length=2)


def _json_list(raw: str | None) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _out(row: AnnotationRecord) -> AnnotationOut:
    return AnnotationOut(
        id=row.id, project_id=row.project_id, source_type=row.source_type,
        source_id=row.source_id, annotation_type=row.annotation_type,
        geometry=loads_geometry(row.geometry_geojson) or {}, crs_epsg=row.crs_epsg,
        label=row.label, severity=row.severity, status=row.status, note=row.note,
        include_in_report=row.include_in_report, created_at=_utc(row.created_at),
        origin=row.origin or "human", machine_claims=_json_list(row.machine_claims_json),
        review_action=row.review_action or "human_drawn",
        parent_ids=[int(value) for value in _json_list(row.parent_ids_json)],
        reviewed_by=row.reviewed_by,
        reviewed_at=_utc(row.reviewed_at) if row.reviewed_at else None,
        updated_at=_utc(row.updated_at),
    )


def _validate(payload: AnnotationCreate) -> None:
    try:
        validate_annotation(
            source_type=payload.source_type, annotation_type=payload.annotation_type,
            geometry=payload.geometry, label=payload.label,
            severity=payload.severity, status=payload.status,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.source_type == "map" and payload.crs_epsg is None:
        raise HTTPException(status_code=422, detail="Map annotations require an explicit CRS.")


@router.get("/projects/{project_id}/annotations/prelabel/models")
def prelabel_models(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Report the installed models and the persistence contract they can satisfy.

    SegFormer is a real installed crack model, but its current detector result exposes
    only an aggregate mask and no per-region score. Inventing one would erase the
    distinction between a threshold and model confidence, so it is not offered for
    persistent pre-labels yet.
    """
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    structural = model_status("structural_multiclass_detector")
    cracks = model_status("crack_segmentation")
    return {
        "models": [
            {
                "model_key": "structural_multiclass_detector",
                "installed": bool(structural.get("exists")),
                "prelabel_supported": bool(structural.get("exists")),
                "output": "confidence-bearing bounding boxes",
            },
            {
                "model_key": "crack_segmentation",
                "installed": bool(cracks.get("exists")),
                "prelabel_supported": False,
                "output": "segmentation mask",
                "reason": (
                    "The detector contract has no per-region model confidence. "
                    "Pre-label persistence refuses to invent one."
                ),
            },
        ]
    }


@router.post("/projects/{project_id}/annotations/prelabel", status_code=201)
async def create_prelabels(
    project_id: int,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    image: UploadFile = File(...),
    model_key: str = Form("structural_multiclass_detector"),
    severity: str = Form("info"),
) -> dict[str, Any]:
    """Run a confidence-bearing installed model and persist reviewable rectangles."""
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.inspector)
    if model_key != "structural_multiclass_detector":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model {model_key!r} cannot create persistent pre-labels. "
                "Only structural_multiclass_detector currently exposes per-label confidence."
            ),
        )
    if severity not in {"critical", "high", "medium", "low", "info"}:
        raise HTTPException(status_code=422, detail="Invalid review-default severity.")

    raw = await image.read(30 * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Pre-label image exceeds the 30 MiB limit.")
    array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise HTTPException(status_code=422, detail="Uploaded bytes are not a decodable image.")

    digest = hashlib.sha256(raw).hexdigest()
    suffix = Path(image.filename or "capture.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        suffix = ".img"
    storage_key = f"projects/{project_id}/annotations/sources/{digest}{suffix}"
    try:
        build_storage().put(storage_key, raw)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=f"Could not retain source image: {exc}") from exc

    result = detect_structural_defects(array, model_key=model_key, use_model=True)
    if not result.model_used.startswith("onnx:") or not result.model_key or not result.model_sha256:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained detector did not run. Heuristic output is intentionally "
                "not persisted as machine pre-labels."
            ),
        )

    created: list[AnnotationRecord] = []
    for hit in result.detections:
        x1, y1, x2, y2 = (int(value) for value in hit.bbox_xyxy)
        geometry = {"type": "Polygon", "coordinates": [[
            [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1],
        ]]}
        claim = {
            "geometry": geometry,
            "label": hit.label,
            "model_key": result.model_key,
            "model_sha256": result.model_sha256,
            "model_used": result.model_used,
            "confidence": float(hit.confidence),
            "confidence_kind": "per_detection_model_score",
            "source_image_sha256": digest,
            "source_storage_key": storage_key,
        }
        candidate = AnnotationCreate(
            source_type="image", source_id=storage_key, annotation_type="rectangle",
            geometry=geometry, label=hit.label, severity=severity, status="in_review",
            note="Machine pre-label awaiting human review.", include_in_report=False,
        )
        _validate(candidate)
        row = AnnotationRecord(
            project_id=project_id, source_type="image", source_id=storage_key,
            annotation_type="rectangle", geometry_geojson=dumps_geometry(geometry),
            crs_epsg=None, label=hit.label, severity=severity, status="in_review",
            note=candidate.note, include_in_report=False, created_by=user.id,
            origin="model", machine_claims_json=json.dumps([claim], sort_keys=True),
            review_action="unreviewed", parent_ids_json="[]",
        )
        db.add(row)
        created.append(row)
    db.flush()
    record(
        db, action="annotation_prelabels_created", user_id=user.id,
        organization_id=project.organization_id, resource=f"project:{project_id}",
        detail={
            "count": len(created), "model_key": result.model_key,
            "model_sha256": result.model_sha256, "source_sha256": digest,
        },
    )
    db.commit()
    return {
        "source": {
            "storage_key": storage_key, "sha256": digest,
            "width_px": int(array.shape[1]), "height_px": int(array.shape[0]),
        },
        "model": {
            "model_key": result.model_key, "model_sha256": result.model_sha256,
            "model_used": result.model_used,
        },
        "prelabels": [_out(row).model_dump(mode="json") for row in created],
        "finding_count": len(created),
    }


@router.get("/projects/{project_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> list[AnnotationOut]:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    rows = db.scalars(
        select(AnnotationRecord).where(AnnotationRecord.project_id == project_id)
        .order_by(AnnotationRecord.id)
    )
    return [_out(row) for row in rows]


@router.post("/projects/{project_id}/annotations", response_model=AnnotationOut, status_code=201)
def create_annotation(
    project_id: int, payload: AnnotationCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.inspector)
    _validate(payload)
    row = AnnotationRecord(
        project_id=project_id, source_type=payload.source_type,
        source_id=payload.source_id, annotation_type=payload.annotation_type,
        geometry_geojson=dumps_geometry(payload.geometry), crs_epsg=payload.crs_epsg,
        label=payload.label.strip(), severity=payload.severity, status=payload.status,
        note=payload.note, include_in_report=payload.include_in_report,
        created_by=user.id,
    )
    db.add(row)
    db.flush()
    record(db, action="annotation_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"annotation:{row.id}",
           detail={"type": row.annotation_type, "severity": row.severity})
    db.commit()
    return _out(row)


def _annotation_or_404(db: Session, annotation_id: int) -> AnnotationRecord:
    row = db.get(AnnotationRecord, annotation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Annotation not found.")
    return row


@router.get("/annotations/{annotation_id}", response_model=AnnotationOut)
def get_annotation(
    annotation_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    row = _annotation_or_404(db, annotation_id)
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.viewer)
    return _out(row)


@router.patch("/annotations/{annotation_id}", response_model=AnnotationOut)
def update_annotation(
    annotation_id: int, patch: AnnotationPatch,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    row = _annotation_or_404(db, annotation_id)
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.inspector)
    changes = patch.model_dump(exclude_unset=True)
    candidate = AnnotationCreate(
        source_type=row.source_type, source_id=row.source_id,
        annotation_type=changes.get("annotation_type", row.annotation_type),
        geometry=changes.get("geometry", loads_geometry(row.geometry_geojson) or {}),
        crs_epsg=changes.get("crs_epsg", row.crs_epsg),
        label=changes.get("label", row.label), severity=changes.get("severity", row.severity),
        status=changes.get("status", row.status), note=changes.get("note", row.note),
        include_in_report=changes.get("include_in_report", row.include_in_report),
    )
    _validate(candidate)
    row.annotation_type = candidate.annotation_type
    row.geometry_geojson = dumps_geometry(candidate.geometry)
    row.crs_epsg = candidate.crs_epsg
    row.label = candidate.label.strip()
    row.severity = candidate.severity
    row.status = candidate.status
    row.note = candidate.note
    row.include_in_report = candidate.include_in_report
    if row.origin == "model" and changes:
        row.review_action = "edited"
        row.reviewed_by = user.id
        row.reviewed_at = utcnow()
    row.updated_at = utcnow()
    record(db, action="annotation_updated", user_id=user.id,
           organization_id=project.organization_id, resource=f"annotation:{row.id}",
           detail={"fields": sorted(changes)})
    db.commit()
    return _out(row)


@router.post("/annotations/{annotation_id}/review", response_model=AnnotationOut)
def review_annotation(
    annotation_id: int, payload: AnnotationReview,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    row = _annotation_or_404(db, annotation_id)
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.inspector)
    if row.origin != "model" or not _json_list(row.machine_claims_json):
        raise HTTPException(status_code=422, detail="Only machine pre-labels use this review workflow.")
    if payload.action == "edit" and payload.geometry is None:
        raise HTTPException(status_code=422, detail="Edit review requires replacement geometry.")
    if payload.action == "reclassify" and payload.label is None:
        raise HTTPException(status_code=422, detail="Reclassify review requires a replacement label.")

    candidate = AnnotationCreate(
        source_type=row.source_type, source_id=row.source_id,
        annotation_type=row.annotation_type,
        geometry=payload.geometry or loads_geometry(row.geometry_geojson) or {},
        crs_epsg=row.crs_epsg, label=payload.label or row.label,
        severity=payload.severity or row.severity,
        status=payload.status or "open", note=payload.note if payload.note is not None else row.note,
        include_in_report=True,
    )
    _validate(candidate)
    row.geometry_geojson = dumps_geometry(candidate.geometry)
    row.label = candidate.label.strip()
    row.severity = candidate.severity
    row.status = candidate.status
    row.note = candidate.note
    row.include_in_report = True
    row.review_action = {
        "accept": "accepted", "edit": "edited", "reclassify": "reclassified",
    }[payload.action]
    row.reviewed_by = user.id
    row.reviewed_at = utcnow()
    row.updated_at = utcnow()
    record(
        db, action=f"annotation_{row.review_action}", user_id=user.id,
        organization_id=project.organization_id, resource=f"annotation:{row.id}",
        detail={"machine_claims_preserved": len(_json_list(row.machine_claims_json))},
    )
    db.commit()
    return _out(row)


@router.post("/projects/{project_id}/annotations/merge", response_model=AnnotationOut, status_code=201)
def merge_annotations(
    project_id: int, payload: MergeAnnotations,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.inspector)
    ids = list(dict.fromkeys(payload.annotation_ids))
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="Merge requires at least two distinct annotations.")
    parents = list(db.scalars(select(AnnotationRecord).where(
        AnnotationRecord.project_id == project_id, AnnotationRecord.id.in_(ids)
    )))
    if len(parents) != len(ids):
        raise HTTPException(status_code=404, detail="One or more merge annotations were not found in this project.")
    first = parents[0]
    if any((row.source_type, row.source_id, row.crs_epsg) !=
           (first.source_type, first.source_id, first.crs_epsg) for row in parents[1:]):
        raise HTTPException(status_code=422, detail="Merged annotations must use the same source and CRS.")
    candidate = AnnotationCreate(
        source_type=first.source_type, source_id=first.source_id,
        annotation_type=payload.annotation_type, geometry=payload.geometry,
        crs_epsg=first.crs_epsg, label=payload.label, severity=payload.severity,
        status=payload.status, note=payload.note, include_in_report=payload.include_in_report,
    )
    _validate(candidate)
    claims = [claim for parent in parents for claim in _json_list(parent.machine_claims_json)]
    now = utcnow()
    for parent in parents:
        parent.review_action = "merged_source"
        parent.reviewed_by = user.id
        parent.reviewed_at = now
        parent.updated_at = now
        parent.status = "resolved"
    row = AnnotationRecord(
        project_id=project_id, source_type=first.source_type, source_id=first.source_id,
        annotation_type=candidate.annotation_type, geometry_geojson=dumps_geometry(candidate.geometry),
        crs_epsg=first.crs_epsg, label=candidate.label.strip(), severity=candidate.severity,
        status=candidate.status, note=candidate.note, include_in_report=candidate.include_in_report,
        created_by=user.id, origin="model" if claims else "human",
        machine_claims_json=json.dumps(claims, sort_keys=True), review_action="merged",
        parent_ids_json=json.dumps(ids), reviewed_by=user.id, reviewed_at=now,
    )
    db.add(row)
    db.flush()
    record(
        db, action="annotations_merged", user_id=user.id,
        organization_id=project.organization_id, resource=f"annotation:{row.id}",
        detail={"parent_ids": ids, "machine_claims_preserved": len(claims)},
    )
    db.commit()
    return _out(row)


@router.post("/annotations/{annotation_id}/split", response_model=list[AnnotationOut], status_code=201)
def split_annotation(
    annotation_id: int, payload: SplitAnnotation,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> list[AnnotationOut]:
    parent = _annotation_or_404(db, annotation_id)
    project = _project_or_404(db, parent.project_id)
    require_project_role(db, user, project, Role.inspector)
    claims = _json_list(parent.machine_claims_json)
    candidates: list[AnnotationCreate] = []
    for part in payload.parts:
        candidate = AnnotationCreate(
            source_type=parent.source_type, source_id=parent.source_id,
            annotation_type=part.annotation_type, geometry=part.geometry,
            crs_epsg=parent.crs_epsg, label=part.label, severity=part.severity,
            status=part.status, note=part.note, include_in_report=part.include_in_report,
        )
        _validate(candidate)
        candidates.append(candidate)
    now = utcnow()
    parent.review_action = "split_source"
    parent.reviewed_by = user.id
    parent.reviewed_at = now
    parent.updated_at = now
    parent.status = "resolved"
    children: list[AnnotationRecord] = []
    for candidate in candidates:
        row = AnnotationRecord(
            project_id=parent.project_id, source_type=parent.source_type,
            source_id=parent.source_id, annotation_type=candidate.annotation_type,
            geometry_geojson=dumps_geometry(candidate.geometry), crs_epsg=parent.crs_epsg,
            label=candidate.label.strip(), severity=candidate.severity, status=candidate.status,
            note=candidate.note, include_in_report=candidate.include_in_report,
            created_by=user.id, origin="model" if claims else "human",
            machine_claims_json=json.dumps(claims, sort_keys=True),
            review_action="split_child", parent_ids_json=json.dumps([parent.id]),
            reviewed_by=user.id, reviewed_at=now,
        )
        db.add(row)
        children.append(row)
    db.flush()
    record(
        db, action="annotation_split", user_id=user.id,
        organization_id=project.organization_id, resource=f"annotation:{parent.id}",
        detail={"child_ids": [row.id for row in children], "machine_claims_preserved": len(claims)},
    )
    db.commit()
    return [_out(row) for row in children]


@router.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(
    annotation_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> None:
    row = _annotation_or_404(db, annotation_id)
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.inspector)
    db.delete(row)
    record(db, action="annotation_deleted", user_id=user.id,
           organization_id=project.organization_id, resource=f"annotation:{annotation_id}")
    db.commit()
