"""Processing jobs.

Reconstruction runs for hours, so it cannot happen inside a request. A job is
submitted, executed on a worker thread, and polled. Progress and cancellation are
cooperative: the worker checks a flag at each checkpoint rather than being killed,
which is what allows a cancelled run to leave the filesystem in a describable state.

Every job records the engine that ran it and the CRS of what it produced. A run whose
outputs carry no CRS is reported as unreferenced rather than being presented as if its
measurements meant something.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db, get_session_factory
from ..models import Dataset, Job, Project, Role
from ..paths import storage_root
from ..security import CurrentUser, require_role

router = APIRouter(tags=["processing"])

# Cancellation flags, keyed by job id. Held in memory because a cancel only makes
# sense for a job this process is currently running.
_CANCEL: dict[int, threading.Event] = {}
_LOCK = threading.Lock()


class JobSubmit(BaseModel):
    kind: str = Field(default="reconstruction")
    dataset_id: int | None = None
    # Engine choice is explicit. "auto" is honest about picking for you; naming an
    # engine that is unavailable produces a refusal, not a silent substitution.
    engine: str = "auto"
    profile: str = "standard"
    options: dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: int
    project_id: int
    dataset_id: int | None
    kind: str
    engine: str
    status: str
    percent: int
    message: str
    error: str
    crs_epsg: int | None
    artifacts: list[str]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


SUPPORTED_KINDS = {"reconstruction", "detection", "measurement", "report"}


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _job_out(job: Job) -> JobOut:
    try:
        artifacts = json.loads(job.artifacts_json) if job.artifacts_json else []
    except json.JSONDecodeError:
        artifacts = []
    return JobOut(
        id=job.id, project_id=job.project_id, dataset_id=job.dataset_id, kind=job.kind,
        engine=job.engine, status=job.status, percent=job.percent, message=job.message,
        error=job.error, crs_epsg=job.crs_epsg, artifacts=artifacts,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
    )


def _run_job(job_id: int) -> None:
    """Execute a job on a worker thread with its own database session.

    The request's session belongs to the request; a background thread that borrowed it
    would be using a connection that closes underneath it.
    """
    session_factory = get_session_factory()
    db = session_factory()
    cancel = _CANCEL.setdefault(job_id, threading.Event())

    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        def progress(percent: int, message: str = "") -> None:
            if cancel.is_set():
                raise _Cancelled()
            job.percent = max(0, min(100, int(percent)))
            if message:
                job.message = message
            db.commit()

        artifacts: list[str] = []
        crs_epsg: int | None = None

        if job.kind == "reconstruction":
            artifacts, crs_epsg = _run_reconstruction(job, progress)
        else:
            # Refusing beats inventing an artifact for a stage that is not built.
            raise NotImplementedError(
                f"Job kind {job.kind!r} is accepted by the schema but has no worker yet."
            )

        job.artifacts_json = json.dumps(artifacts)
        job.crs_epsg = crs_epsg
        job.status = "done"
        job.percent = 100
        job.message = job.message or "Complete"

    except _Cancelled:
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "cancelled"
            job.message = "Cancelled"
    except Exception as exc:  # noqa: BLE001
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            job.log = traceback.format_exc()[-8000:]
    finally:
        job = db.get(Job, job_id)
        if job is not None:
            if job.status in {"queued", "running"}:
                # The worker left without reaching a terminal branch.
                job.status = "failed"
                job.error = job.error or "Worker ended without reporting a result."
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
        db.close()
        with _LOCK:
            _CANCEL.pop(job_id, None)


class _Cancelled(Exception):
    """Raised inside a worker when cancellation was requested."""


def _run_reconstruction(job: Job, progress) -> tuple[list[str], int | None]:
    """Reconstruct a dataset's imagery, returning artifact paths and the CRS."""
    from core.reconstruction_colmap import build_reconstructor

    if job.dataset_id is None:
        raise ValueError("Reconstruction requires a dataset.")

    images = storage_root() / "datasets" / str(job.dataset_id)
    if not images.is_dir() or not any(images.iterdir()):
        raise ValueError(f"Dataset {job.dataset_id} has no uploaded imagery at {images}.")

    output = storage_root() / "runs" / f"job_{job.id}"
    output.mkdir(parents=True, exist_ok=True)

    progress(5, "Starting reconstruction")
    reconstructor = build_reconstructor(job.engine, profile=job.profile)
    result = reconstructor.reconstruct(str(images), str(output))
    progress(95, "Collecting artifacts")

    payload = result.to_dict() if hasattr(result, "to_dict") else {}
    artifacts = [
        str(value) for key, value in payload.items()
        if key.endswith(("_path", "_paths")) and isinstance(value, str) and value
        and Path(value).exists()
    ]
    return artifacts, payload.get("crs_epsg")


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[JobOut]:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    rows = db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()))
    return [_job_out(job) for job in rows]


@router.post("/projects/{project_id}/jobs", response_model=JobOut, status_code=202)
def submit_job(
    project_id: int, payload: JobSubmit,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> JobOut:
    """Queue a job. 202 rather than 201: the work has not happened yet."""
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.engineer)

    if payload.kind not in SUPPORTED_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown job kind {payload.kind!r}. Supported: {', '.join(sorted(SUPPORTED_KINDS))}.",
        )
    if payload.dataset_id is not None:
        dataset = db.get(Dataset, payload.dataset_id)
        if dataset is None or dataset.project_id != project_id:
            raise HTTPException(status_code=404, detail="Dataset not found in this project.")

    job = Job(
        project_id=project_id, dataset_id=payload.dataset_id, kind=payload.kind,
        engine=payload.engine, profile=payload.profile,
        options_json=json.dumps(payload.options), status="queued",
        created_by=user.id,
    )
    db.add(job)
    db.flush()
    record(db, action="job_submitted", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{job.id}",
           detail={"kind": job.kind, "engine": job.engine})
    db.commit()

    with _LOCK:
        _CANCEL[job.id] = threading.Event()
    threading.Thread(target=_run_job, args=(job.id,), name=f"job-{job.id}", daemon=True).start()

    return _job_out(job)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    project = _project_or_404(db, job.project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    db.refresh(job)
    return _job_out(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> JobOut:
    """Request cancellation. The worker stops at its next checkpoint."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    project = _project_or_404(db, job.project_id)
    require_role(db, user, project.organization_id, Role.engineer)

    if job.status not in {"queued", "running"}:
        raise HTTPException(
            status_code=409, detail=f"Job is {job.status} and can no longer be cancelled."
        )

    with _LOCK:
        event = _CANCEL.get(job_id)
    if event is not None:
        event.set()
    job.message = "Cancelling..."
    record(db, action="job_cancelled", user_id=user.id,
           organization_id=project.organization_id, resource=f"job:{job_id}")
    db.commit()
    return _job_out(job)


@router.get("/jobs/{job_id}/log")
def job_log(
    job_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    project = _project_or_404(db, job.project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    return {"job_id": job.id, "status": job.status, "error": job.error, "log": job.log or ""}
