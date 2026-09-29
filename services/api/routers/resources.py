"""Measurements, report snapshots and AI inference-job resources.

These endpoints persist real request and project data. An AI job is refused up front
unless the named model is installed and can answer the task; it then stays
``pending_worker`` until an inference worker (services/api/ai_jobs.py) claims it, and
no result is claimed before the worker records one. Likewise, a measurement is accepted only with geometry, CRS, source and
method; a bare number cannot enter the API looking like a survey measurement.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..audit import record
from ..db import dumps_geometry, get_db, loads_geometry
from ..models import Dataset, Defect, Job, Measurement, Project, Report, Role, utcnow
from ..security import CurrentUser, require_project_role
from .events import emit

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


def build_report_payload(
    db: Session, project: Project, *, include_unreviewed: bool = False,
) -> dict[str, Any]:
    """The structured report snapshot, shared by the report endpoint and report jobs."""
    all_defects = list(db.scalars(select(Defect).where(Defect.project_id == project.id)))
    defects = all_defects if include_unreviewed else [
        row for row in all_defects if row.review_state == "accepted"
    ]
    measurements = list(
        db.scalars(select(Measurement).where(Measurement.project_id == project.id))
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
                "confidence": row.confidence, "area_m2": row.area_m2,
                "length_m": row.length_m,
            } for row in defects
        ],
        "measurements": [
            {
                "id": row.id, "kind": row.kind, "value": row.value, "unit": row.unit,
                "crs_epsg": row.crs_epsg, "source_ref": row.source_ref,
                "method": row.method,
            } for row in measurements
        ],
        "include_unreviewed": include_unreviewed,
        "excluded_not_accepted_count": len(all_defects) - len(defects),
    }
    excluded = payload["excluded_not_accepted_count"]
    if excluded:
        # Stated so an empty findings list is never read as "nothing was found".
        payload["findings_note"] = (
            f"{excluded} defect(s) without an accepted review (unreviewed, rejected or "
            "reclassified) were excluded. "
            "Review them, or request the report with include_unreviewed=true."
        )
    return payload


@router.post("/projects/{project_id}/reports", response_model=ReportOut, status_code=201)
def create_report(
    project_id: int, request: ReportCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> ReportOut:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.inspector)
    payload = build_report_payload(db, project, include_unreviewed=request.include_unreviewed)
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
    out = _report_out(row)
    emit(project.organization_id, "report.generated", {
        "report_id": out.id, "project_id": out.project_id, "title": out.title,
        "format": out.format, "status": out.status,
    })
    return out


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
    # "dataset:<id>" for every image in a dataset, "dataset:<id>/frame:<file>" for one,
    # or a file name (or "*") together with dataset_id.
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
    percent: int
    message: str
    error: str
    artifacts: list[str]
    result: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    note: str


def _ai_job_out(row: Job) -> AIJobOut:
    from ..ai_jobs import read_results

    try:
        options = json.loads(row.options_json or "{}")
    except json.JSONDecodeError:
        options = {}
    try:
        artifacts = json.loads(row.artifacts_json or "[]")
    except json.JSONDecodeError:
        artifacts = []
    result = read_results(row)
    if result is not None:
        note = ("Model output from the named model, recorded unreviewed. Download "
                "results.json from artifacts for per-image detail.")
    elif row.status in {"pending_worker", "running", "cancelling"}:
        note = "No result yet. None is claimed until the worker records a terminal status."
    else:
        note = "No result was produced; see error and message."
    return AIJobOut(
        id=row.id, project_id=row.project_id, dataset_id=row.dataset_id,
        task=row.kind.removeprefix("ai:"), model_key=str(options.get("model_key", "")),
        input_ref=str(options.get("input_ref", "")),
        parameters=options.get("parameters", {}) if isinstance(options.get("parameters", {}), dict) else {},
        status=row.status, percent=int(row.percent or 0), message=row.message or "",
        error=row.error or "", artifacts=artifacts if isinstance(artifacts, list) else [],
        result=result, created_at=_utc(row.created_at), started_at=_utc(row.started_at),
        finished_at=_utc(row.finished_at), note=note,
    )


@router.get("/ai-models")
def list_ai_models(user: CurrentUser) -> dict[str, Any]:
    """The models an AI or detection job may name, per task, and whether each can run."""
    from ..inference import model_catalogue  # noqa: PLC0415

    return {"models": model_catalogue()}


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
    """Record an inference request and hand it to a worker.

    Everything that would make the worker refuse -- an unknown model, one that cannot
    answer the task, weights not installed, an input that names no uploaded image -- is
    refused here with the reason, rather than accepted and failed later.
    """
    from ..ai_jobs import dispatch, worker_mode
    from ..inference import InferenceRefused, choose_model, dataset_images, parse_input_ref
    from ..paths import storage_root

    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.engineer)
    if request.task not in AI_TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown AI task. Supported: {', '.join(sorted(AI_TASKS))}.",
        )
    try:
        choose_model(request.task, request.model_key, verify_digest=False)
        dataset_id, selector = parse_input_ref(request.input_ref, request.dataset_id)
    except InferenceRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(status_code=404, detail="Dataset not found in this project.")
    try:
        dataset_images(storage_root() / "datasets" / str(dataset_id), selector)
    except InferenceRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = Job(
        project_id=project_id, dataset_id=dataset_id, kind=f"ai:{request.task}",
        engine="inference_worker", profile="api",
        options_json=json.dumps({
            "model_key": request.model_key, "input_ref": request.input_ref,
            "parameters": request.parameters,
        }),
        status="pending_worker", percent=0,
        message=(
            "Waiting for an inference worker; no inference result exists yet."
            if worker_mode() == "external" else "Queued for the in-process inference worker."
        ),
        created_by=user.id,
    )
    db.add(row)
    db.flush()
    record(db, action="ai_job_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{row.id}",
           detail={"task": request.task, "model_key": request.model_key})
    db.commit()
    out = _ai_job_out(row)
    dispatch(row.id)
    return out


@router.get("/ai-jobs/{job_id}", response_model=AIJobOut)
def get_ai_job(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AIJobOut:
    row = db.get(Job, job_id)
    if row is None or not row.kind.startswith("ai:"):
        raise HTTPException(status_code=404, detail="AI job not found.")
    project = _project_or_404(db, row.project_id)
    require_project_role(db, user, project, Role.viewer)
    db.refresh(row)
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
    # Conditional updates: the worker may claim the job between the read and the write.
    cancelled = db.execute(
        update(Job).where(Job.id == job_id, Job.status == "pending_worker").values(
            status="cancelled", finished_at=utcnow(),
            message="Cancelled before an inference worker claimed the request.",
        )
    ).rowcount == 1
    if not cancelled:
        stopping = db.execute(
            update(Job).where(Job.id == job_id, Job.status == "running").values(
                status="cancelling", message="Cancelling at the next image boundary...",
            )
        ).rowcount == 1
        if not stopping:
            db.rollback()
            db.refresh(row)
            raise HTTPException(status_code=409,
                                detail=f"AI job is {row.status} and cannot be cancelled.")
    record(db, action="ai_job_cancelled", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{row.id}")
    db.commit()
    db.refresh(row)
    return _ai_job_out(row)
