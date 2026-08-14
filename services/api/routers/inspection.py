"""Defects, annotations and measurements.

Every defect stores the model version and confidence that produced it, and whether a
human has reviewed it. An AI prediction is never treated as verified: the reviewer's
decision is a separate field from the model's, so a report can always distinguish
"the model said so" from "an inspector agreed".

Measurements are computed by the same `core.dsm_analysis` the desktop uses, against
the georeferenced rasters a run produced. Where no georeferenced raster exists, the
endpoint refuses rather than returning a number in pixel units.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db, dumps_geometry, loads_geometry
from ..models import Defect, Job, Project, Role
from ..security import CurrentUser, require_role

router = APIRouter(tags=["inspection"])

# The default library. Organisations extend this; nothing here is hard-coded into
# detection, so a custom category is a first-class citizen.
DEFAULT_CATEGORIES = [
    "crack", "spalling", "corrosion", "water_ponding", "surface_deformation",
    "coating_failure", "leak", "rust", "missing_component", "loose_component",
    "structural_damage", "thermal_anomaly", "insulation_issue", "moisture_anomaly",
]

SEVERITIES = ["low", "medium", "high", "critical"]
REVIEW_STATES = ["unreviewed", "accepted", "rejected", "reclassified"]


class DefectCreate(BaseModel):
    category: str = Field(min_length=1, max_length=80)
    severity: str = "medium"
    description: str = ""
    longitude: float | None = None
    latitude: float | None = None
    altitude_m: float | None = None
    geometry: dict[str, Any] | None = None
    crs_epsg: int = 4326
    area_m2: float | None = None
    length_m: float | None = None
    # Provenance. A defect created by a model must carry which model and how sure.
    source: str = "human"
    model_key: str = ""
    model_sha256: str = ""
    confidence: float | None = None


class DefectReview(BaseModel):
    decision: str = Field(description="accepted | rejected | reclassified")
    category: str = ""
    severity: str = ""
    note: str = ""


class DefectOut(BaseModel):
    id: int
    project_id: int
    job_id: int | None
    category: str
    severity: str
    description: str
    longitude: float | None
    latitude: float | None
    altitude_m: float | None
    geometry: dict[str, Any] | None
    crs_epsg: int
    area_m2: float | None
    length_m: float | None
    source: str
    model_key: str
    model_sha256: str
    confidence: float | None
    review_state: str
    reviewed_by: int | None
    reviewed_at: datetime | None
    review_note: str
    created_at: datetime


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _defect_out(defect: Defect) -> DefectOut:
    return DefectOut(
        id=defect.id, project_id=defect.project_id, job_id=defect.job_id,
        category=defect.category, severity=defect.severity, description=defect.description,
        longitude=defect.longitude, latitude=defect.latitude, altitude_m=defect.altitude_m,
        geometry=loads_geometry(defect.geometry_geojson), crs_epsg=defect.crs_epsg,
        area_m2=defect.area_m2, length_m=defect.length_m, source=defect.source,
        model_key=defect.model_key, model_sha256=defect.model_sha256,
        confidence=defect.confidence, review_state=defect.review_state,
        reviewed_by=defect.reviewed_by, reviewed_at=defect.reviewed_at,
        review_note=defect.review_note, created_at=defect.created_at,
    )


@router.get("/defect-categories")
def defect_categories() -> dict[str, Any]:
    """The default library. Custom categories are accepted on creation."""
    return {
        "categories": DEFAULT_CATEGORIES,
        "severities": SEVERITIES,
        "note": "Organisations may use any category string; this list is only the default.",
    }


@router.get("/projects/{project_id}/defects", response_model=list[DefectOut])
def list_defects(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
    review_state: str = "", category: str = "",
) -> list[DefectOut]:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.viewer)

    statement = select(Defect).where(Defect.project_id == project_id)
    if review_state:
        statement = statement.where(Defect.review_state == review_state)
    if category:
        statement = statement.where(Defect.category == category)
    return [_defect_out(defect) for defect in db.scalars(statement)]


@router.post("/projects/{project_id}/defects", response_model=DefectOut, status_code=201)
def create_defect(
    project_id: int, payload: DefectCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> DefectOut:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.inspector)

    if payload.severity not in SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown severity {payload.severity!r}. Use one of: {', '.join(SEVERITIES)}.",
        )
    if payload.source == "model" and not payload.model_key:
        # A model-sourced finding without its model is unattributable.
        raise HTTPException(
            status_code=422,
            detail="A defect with source 'model' must name the model_key that produced it.",
        )

    defect = Defect(
        project_id=project_id, category=payload.category, severity=payload.severity,
        description=payload.description, longitude=payload.longitude,
        latitude=payload.latitude, altitude_m=payload.altitude_m,
        geometry_geojson=dumps_geometry(payload.geometry), crs_epsg=payload.crs_epsg,
        area_m2=payload.area_m2, length_m=payload.length_m, source=payload.source,
        model_key=payload.model_key, model_sha256=payload.model_sha256,
        confidence=payload.confidence,
        # A model prediction starts unreviewed, always. It is never born accepted.
        review_state="unreviewed",
        created_by=user.id,
    )
    db.add(defect)
    db.flush()
    record(db, action="defect_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"defect:{defect.id}",
           detail={"category": defect.category, "source": defect.source})
    db.commit()
    return _defect_out(defect)


@router.post("/defects/{defect_id}/review", response_model=DefectOut)
def review_defect(
    defect_id: int, payload: DefectReview,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> DefectOut:
    """Record a human decision, keeping the model's original claim intact.

    The reviewer's category and severity overwrite the displayed values, but
    `model_key`, `model_sha256` and `confidence` are left untouched so the record
    still shows what the model asserted and how sure it was.
    """
    defect = db.get(Defect, defect_id)
    if defect is None:
        raise HTTPException(status_code=404, detail="Defect not found.")
    project = _project_or_404(db, defect.project_id)
    require_role(db, user, project.organization_id, Role.inspector)

    if payload.decision not in REVIEW_STATES or payload.decision == "unreviewed":
        raise HTTPException(
            status_code=422,
            detail=f"decision must be one of: accepted, rejected, reclassified.",
        )
    if payload.decision == "reclassified" and not payload.category:
        raise HTTPException(
            status_code=422, detail="Reclassifying requires the new category."
        )

    from .. import models as model_module

    defect.review_state = payload.decision
    defect.reviewed_by = user.id
    defect.reviewed_at = model_module.utcnow()
    defect.review_note = payload.note
    if payload.category:
        defect.category = payload.category
    if payload.severity:
        if payload.severity not in SEVERITIES:
            raise HTTPException(status_code=422, detail=f"Unknown severity {payload.severity!r}.")
        defect.severity = payload.severity

    record(db, action="defect_reviewed", user_id=user.id,
           organization_id=project.organization_id, resource=f"defect:{defect.id}",
           detail={"decision": payload.decision})
    db.commit()
    return _defect_out(defect)


@router.get("/projects/{project_id}/defects/summary")
def defect_summary(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Counts and quantities, separating what a model claimed from what a human confirmed."""
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.viewer)

    defects = list(db.scalars(select(Defect).where(Defect.project_id == project_id)))
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_review: dict[str, int] = {}
    measured_area = 0.0
    measured_length = 0.0
    unmeasured = 0

    for defect in defects:
        by_category[defect.category] = by_category.get(defect.category, 0) + 1
        by_severity[defect.severity] = by_severity.get(defect.severity, 0) + 1
        by_review[defect.review_state] = by_review.get(defect.review_state, 0) + 1
        if defect.area_m2:
            measured_area += defect.area_m2
        if defect.length_m:
            measured_length += defect.length_m
        if not defect.area_m2 and not defect.length_m:
            unmeasured += 1

    confirmed = [d for d in defects if d.review_state == "accepted"]
    return {
        "total": len(defects),
        "human_confirmed": len(confirmed),
        "awaiting_review": by_review.get("unreviewed", 0),
        "by_category": dict(sorted(by_category.items())),
        "by_severity": {s: by_severity.get(s, 0) for s in SEVERITIES},
        "by_review_state": {s: by_review.get(s, 0) for s in REVIEW_STATES},
        "total_area_m2": round(measured_area, 4),
        "total_length_m": round(measured_length, 3),
        "unmeasured_count": unmeasured,
        # Stated rather than left to be inferred: a total over partly unmeasured
        # defects is not the total extent of the damage.
        "note": (
            f"{unmeasured} defect(s) carry no measured extent, so the area and length "
            "totals cover only the measured ones."
        ) if unmeasured else "",
    }


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


class VolumeRequest(BaseModel):
    job_id: int
    polygon_xy: list[list[float]] | None = None
    base_elevation_m: float | None = None


@router.post("/measurements/volume")
def measure_volume(
    payload: VolumeRequest, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Cut and fill from a run's DSM, in cubic metres.

    Refuses when the run produced no georeferenced raster. A volume computed on an
    unreferenced surface would be in pixel units while looking like cubic metres.
    """
    job = db.get(Job, payload.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    project = _project_or_404(db, job.project_id)
    require_role(db, user, project.organization_id, Role.analyst)

    if job.status != "done":
        raise HTTPException(
            status_code=409, detail=f"Job {job.id} is {job.status}; no outputs to measure."
        )

    try:
        artifacts = json.loads(job.artifacts_json or "[]")
    except json.JSONDecodeError:
        artifacts = []

    def find(name: str) -> str:
        for path in artifacts:
            candidate = Path(path)
            if candidate.name.lower().startswith(name) and candidate.suffix.lower() in {".tif", ".tiff"}:
                return path
        return ""

    dsm = find("dsm")
    if not dsm:
        raise HTTPException(
            status_code=422,
            detail=(
                "This run produced no georeferenced DSM, so no volume can be measured. "
                "Reconstruct with the COLMAP engine to obtain metric rasters."
            ),
        )

    from core.dsm_analysis import NotGeoreferenced, estimate_volume

    try:
        result = estimate_volume(
            dsm, dtm_path=find("dtm") or None,
            polygon_xy=payload.polygon_xy, base_elevation_m=payload.base_elevation_m,
        )
    except NotGeoreferenced as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record(db, action="volume_measured", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{job.id}")
    db.commit()
    return result
