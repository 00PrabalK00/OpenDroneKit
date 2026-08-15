"""Measurements, report snapshots and AI inference-job resources.

These endpoints persist real request and project data. They do not claim that an AI
worker ran: a submitted AI job remains ``pending_worker`` until an inference worker
claims it. Likewise, a measurement is accepted only with geometry, CRS, source and
method; a bare number cannot enter the API looking like a survey measurement.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import dumps_geometry, get_db, loads_geometry
from ..models import Dataset, Defect, Job, Measurement, Project, Report, Role, utcnow
from ..security import CurrentUser, require_project_role

router = APIRouter(tags=["resources"])

MEASUREMENT_UNITS = {
    "distance": "m",
    "perimeter": "m",
    "area": "m2",
    "volume": "m3",
    "temperature": "celsius",
    "count": "count",
}
METRIC_GEOMETRY_KINDS = {"distance", "perimeter", "area", "volume"}
AI_TASKS = {"detection", "segmentation", "classification", "anomaly"}


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _utc(value: datetime | None) -> datetime | None:
    """Keep SQLite reloads and freshly created rows on the same UTC wire format."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class MeasurementCreate(BaseModel):
    kind: str
    value: float
    unit: str
    geometry: dict[str, Any]
    crs_epsg: int
    source_ref: str = Field(min_length=1, max_length=1000)
    method: str = Field(min_length=1, max_length=1000)
    job_id: int | None = None


class MeasurementOut(BaseModel):
    id: int
    project_id: int
    job_id: int | None
    kind: str
    value: float
    unit: str
    geometry: dict[str, Any]
    crs_epsg: int
    source_ref: str
    method: str
    created_at: datetime


def _measurement_out(row: Measurement) -> MeasurementOut:
    return MeasurementOut(
        id=row.id, project_id=row.project_id, job_id=row.job_id,
        kind=row.kind, value=row.value, unit=row.unit,
        geometry=loads_geometry(row.geometry_geojson) or {}, crs_epsg=row.crs_epsg,
        source_ref=row.source_ref, method=row.method, created_at=_utc(row.created_at),
    )


def _validate_measurement(payload: MeasurementCreate) -> None:
    expected = MEASUREMENT_UNITS.get(payload.kind)
    if expected is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown measurement kind. Supported: {', '.join(sorted(MEASUREMENT_UNITS))}.",
        )
    if payload.unit != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Measurement kind {payload.kind!r} uses unit {expected!r}, not {payload.unit!r}.",
        )
    if not math.isfinite(payload.value):
        raise HTTPException(status_code=422, detail="Measurement value must be finite.")
    if payload.kind != "temperature" and payload.value < 0:
        raise HTTPException(status_code=422, detail="This measurement cannot be negative.")
    if not payload.geometry.get("type") or "coordinates" not in payload.geometry:
        raise HTTPException(status_code=422, detail="Measurement requires GeoJSON geometry.")
    try:
        from pyproj import CRS

        crs = CRS.from_epsg(payload.crs_epsg)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Measurement CRS is not usable.") from exc
    if payload.kind in METRIC_GEOMETRY_KINDS and not crs.is_projected:
        raise HTTPException(
            status_code=422,
            detail="Distance, perimeter, area and volume require a projected CRS; degrees are refused.",
        )


@router.get("/projects/{project_id}/measurements", response_model=list[MeasurementOut])
def list_measurements(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> list[MeasurementOut]:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    rows = db.scalars(
        select(Measurement).where(Measurement.project_id == project_id).order_by(Measurement.id)
    )
    return [_measurement_out(row) for row in rows]


@router.post(
    "/projects/{project_id}/measurements", response_model=MeasurementOut, status_code=201,
)
def create_measurement(
    project_id: int, payload: MeasurementCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> MeasurementOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.analyst)
    _validate_measurement(payload)
    if payload.job_id is not None:
        job = db.get(Job, payload.job_id)
        if job is None or job.project_id != project_id:
            raise HTTPException(status_code=404, detail="Job not found in this project.")
        if job.crs_epsg is not None and job.crs_epsg != payload.crs_epsg:
            raise HTTPException(status_code=422, detail="Measurement CRS does not match its job.")
    row = Measurement(
        project_id=project_id, job_id=payload.job_id, kind=payload.kind,
        value=payload.value, unit=payload.unit,
        geometry_geojson=dumps_geometry(payload.geometry), crs_epsg=payload.crs_epsg,
        source_ref=payload.source_ref.strip(), method=payload.method.strip(),
        created_by=user.id,
    )
    db.add(row)
    db.flush()
    record(db, action="measurement_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"measurement:{row.id}",
           detail={"kind": row.kind, "unit": row.unit, "source_ref": row.source_ref})
    db.commit()
    return _measurement_out(row)


@router.get("/measurements/{measurement_id}", response_model=MeasurementOut)
def get_measurement(
    measurement_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> MeasurementOut:
    row = db.get(Measurement, measurement_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Measurement not found.")
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.viewer)
    return _measurement_out(row)


class ReportCreate(BaseModel):
    title: str = Field(default="Inspection report", min_length=1, max_length=240)
    include_unreviewed: bool = False


class ReportOut(BaseModel):
    id: int
    project_id: int
    title: str
    format: str
    status: str
    payload: dict[str, Any]
    created_at: datetime


def _report_out(row: Report) -> ReportOut:
    try:
        payload = json.loads(row.payload_json)
    except json.JSONDecodeError:
        payload = {"status": "unavailable", "reason": "Stored report payload is invalid."}
    return ReportOut(
        id=row.id, project_id=row.project_id, title=row.title, format=row.format,
        status=row.status, payload=payload, created_at=_utc(row.created_at),
    )


@router.get("/projects/{project_id}/reports", response_model=list[ReportOut])
def list_reports(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> list[ReportOut]:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    rows = db.scalars(select(Report).where(Report.project_id == project_id).order_by(Report.id))
    return [_report_out(row) for row in rows]


@router.post("/projects/{project_id}/reports", response_model=ReportOut, status_code=201)
def create_report(
    project_id: int, request: ReportCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> ReportOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.inspector)
    all_defects = list(db.scalars(select(Defect).where(Defect.project_id == project_id)))
    defects = all_defects if request.include_unreviewed else [
        row for row in all_defects if row.review_state == "accepted"
    ]
    measurements = list(
        db.scalars(select(Measurement).where(Measurement.project_id == project_id))
    )
    payload = {
        "project": {"id": project.id, "name": project.name, "client": project.client},
        "defect_evidence_status": "present" if all_defects else "absent",
        "defect_note": "" if all_defects else (
            "No defect records exist for this project; this is absence of a defect run, "
            "not evidence that zero defects were found."
        ),
        "findings": [
            {
                "id": row.id, "category": row.category, "severity": row.severity,
                "review_state": row.review_state, "source": row.source,
                "confidence": row.confidence,
            } for row in defects
        ],
        "measurements": [
            {
                "id": row.id, "kind": row.kind, "value": row.value, "unit": row.unit,
                "crs_epsg": row.crs_epsg, "source_ref": row.source_ref,
                "method": row.method,
            } for row in measurements
        ],
        "include_unreviewed": request.include_unreviewed,
    }
    row = Report(
        project_id=project_id, title=request.title.strip(), format="structured_json",
        status="complete", payload_json=json.dumps(payload), created_by=user.id,
    )
    db.add(row)
    db.flush()
    record(db, action="report_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"report:{row.id}",
           detail={"title": row.title, "format": row.format})
    db.commit()
    return _report_out(row)


@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(
    report_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> ReportOut:
    row = db.get(Report, report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.viewer)
    return _report_out(row)


class AIJobCreate(BaseModel):
    task: str
    model_key: str = Field(min_length=1, max_length=120)
    input_ref: str = Field(min_length=1, max_length=1000)
    dataset_id: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class AIJobOut(BaseModel):
    id: int
    project_id: int
    dataset_id: int | None
    task: str
    model_key: str
    input_ref: str
    parameters: dict[str, Any]
    status: str
    created_at: datetime
    finished_at: datetime | None
    note: str


def _ai_job_out(row: Job) -> AIJobOut:
    try:
        options = json.loads(row.options_json or "{}")
    except json.JSONDecodeError:
        options = {}
    return AIJobOut(
        id=row.id, project_id=row.project_id, dataset_id=row.dataset_id,
        task=row.kind.removeprefix("ai:"), model_key=str(options.get("model_key", "")),
        input_ref=str(options.get("input_ref", "")),
        parameters=options.get("parameters", {}) if isinstance(options.get("parameters", {}), dict) else {},
        status=row.status, created_at=_utc(row.created_at), finished_at=_utc(row.finished_at),
        note=(
            "Persistent inference request only. No result is claimed until an AI worker "
            "records a terminal status and attributable artifacts."
        ),
    )


@router.get("/projects/{project_id}/ai-jobs", response_model=list[AIJobOut])
def list_ai_jobs(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> list[AIJobOut]:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    rows = db.scalars(
        select(Job).where(Job.project_id == project_id, Job.kind.like("ai:%")).order_by(Job.id)
    )
    return [_ai_job_out(row) for row in rows]


@router.post("/projects/{project_id}/ai-jobs", response_model=AIJobOut, status_code=201)
def create_ai_job(
    project_id: int, request: AIJobCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AIJobOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.engineer)
    if request.task not in AI_TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown AI task. Supported: {', '.join(sorted(AI_TASKS))}.",
        )
    if request.dataset_id is not None:
        dataset = db.get(Dataset, request.dataset_id)
        if dataset is None or dataset.project_id != project_id:
            raise HTTPException(status_code=404, detail="Dataset not found in this project.")
    row = Job(
        project_id=project_id, dataset_id=request.dataset_id, kind=f"ai:{request.task}",
        engine="inference_worker", profile="api",
        options_json=json.dumps({
            "model_key": request.model_key, "input_ref": request.input_ref,
            "parameters": request.parameters,
        }),
        status="pending_worker", percent=0,
        message="Waiting for an inference worker; no inference result exists yet.",
        created_by=user.id,
    )
    db.add(row)
    db.flush()
    record(db, action="ai_job_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{row.id}",
           detail={"task": request.task, "model_key": request.model_key})
    db.commit()
    return _ai_job_out(row)


@router.get("/ai-jobs/{job_id}", response_model=AIJobOut)
def get_ai_job(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AIJobOut:
    row = db.get(Job, job_id)
    if row is None or not row.kind.startswith("ai:"):
        raise HTTPException(status_code=404, detail="AI job not found.")
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.viewer)
    return _ai_job_out(row)


@router.post("/ai-jobs/{job_id}/cancel", response_model=AIJobOut)
def cancel_ai_job(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AIJobOut:
    row = db.get(Job, job_id)
    if row is None or not row.kind.startswith("ai:"):
        raise HTTPException(status_code=404, detail="AI job not found.")
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.engineer)
    if row.status != "pending_worker":
        raise HTTPException(status_code=409, detail=f"AI job is {row.status} and cannot be cancelled.")
    row.status = "cancelled"
    row.message = "Cancelled before an inference worker claimed the request."
    row.finished_at = utcnow()
    record(db, action="ai_job_cancelled", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{row.id}")
    db.commit()
    return _ai_job_out(row)
