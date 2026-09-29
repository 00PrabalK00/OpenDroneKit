"""Processing jobs.

Reconstruction runs for hours, so it cannot happen inside a request. A job is
submitted, executed on a worker thread (or a Celery worker), and polled. Progress and
cancellation are cooperative: the worker checks a flag at each checkpoint rather than
being killed, which is what allows a cancelled run to leave the filesystem in a
describable state.

Every job records the engine that ran it and the CRS of what it produced. A run whose
outputs carry no CRS is reported as unreferenced rather than being presented as if its
measurements meant something.

Four kinds run here:

``reconstruction``  COLMAP (or the custom engine) over a dataset's images.
``detection``       a registered defect model over a dataset's images; every hit is
                    recorded as an *unreviewed* defect with the model's identity.
``measurement``     distance/area/perimeter on a finished reconstruction's orthomosaic,
                    or volume on its DSM; persisted as a Measurement with its CRS.
``report``          the project's accepted findings and measurements rendered to PDF.

Outputs are recorded as storage-relative references and served by
``GET /jobs/{id}/artifacts/{ref}``.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..audit import record
from ..db import dumps_geometry, get_db, get_session_factory
from ..models import Dataset, Defect, Job, Measurement, Project, Report, Role
from ..paths import from_storage_ref, storage_root, to_storage_ref, within_root
from ..security import CurrentUser, require_role
from .events import emit

router = APIRouter(tags=["processing"])

# Cancellation flags, keyed by job id. Held in memory for jobs this process runs; a
# cancel is also written to the row ("cancelling") for a job running elsewhere.
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
DEFAULT_DETECTION_MODEL = "structural_multiclass_detector"
HEURISTIC_SOURCE = "heuristic"
MEASUREMENT_UNIT = {"distance": "m", "perimeter": "m", "area": "m2", "volume": "m3"}


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
    """Execute a job with its own database session.

    The request's session belongs to the request; a background thread that borrowed it
    would be using a connection that closes underneath it. The same function is the
    body of the Celery task, so a job runs identically in-process and on a worker.
    """
    session_factory = get_session_factory()
    db = session_factory()
    with _LOCK:
        cancel = _CANCEL.setdefault(job_id, threading.Event())
    summary: dict[str, Any] | None = None
    organization_id: int | None = None
    claimed = False

    try:
        # Claimed with a conditional update, so two workers handed the same id (a
        # redelivered Celery message, say) cannot both run it.
        claimed = db.execute(
            update(Job).where(Job.id == job_id, Job.status == "queued")
            .values(status="running", started_at=datetime.now(timezone.utc))
        ).rowcount == 1
        db.commit()
        if not claimed:
            # Unknown, claimed by another worker, or cancelled before it started.
            return
        job = db.get(Job, job_id)

        def progress(percent: int, message: str = "") -> None:
            # The in-memory flag covers this process; the row covers a cancel issued by
            # an API process that is not the one running the job (the Celery case).
            if cancel.is_set():
                raise _Cancelled()
            db.refresh(job, attribute_names=["status"])
            if job.status == "cancelling":
                raise _Cancelled()
            job.percent = max(0, min(100, int(percent)))
            if message:
                job.message = message
            db.commit()

        runner = _RUNNERS.get(job.kind)
        if runner is None:
            raise NotImplementedError(f"Job kind {job.kind!r} has no worker.")
        artifacts, crs_epsg = runner(db, job, progress)
        db.refresh(job, attribute_names=["status"])
        if cancel.is_set() or job.status == "cancelling":
            # An engine that swallows callback exceptions still finishes; its output is
            # not reported as the result of a job the user cancelled.
            raise _Cancelled()

        job.artifacts_json = json.dumps(artifacts)
        job.crs_epsg = crs_epsg
        job.status = "done"
        job.percent = 100
        job.message = job.message or "Complete"

    except _Cancelled:
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "cancelled"
            job.message = "Cancelled"
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            job.log = traceback.format_exc()[-8000:]
    finally:
        job = db.get(Job, job_id) if claimed else None
        if job is not None:
            if job.status in {"queued", "running"}:
                # The worker left without reaching a terminal branch.
                job.status = "failed"
                job.error = job.error or "Worker ended without reporting a result."
            if job.status == "cancelling":
                job.status = "cancelled"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            project = db.get(Project, job.project_id)
            organization_id = project.organization_id if project is not None else None
            summary = _job_out(job).model_dump(mode="json")
        db.close()
        with _LOCK:
            _CANCEL.pop(job_id, None)
    if organization_id is not None and summary is not None and summary["status"] in {"done", "failed"}:
        emit(organization_id,
             "processing.completed" if summary["status"] == "done" else "processing.failed",
             summary)


class _Cancelled(Exception):
    """Raised inside a worker when cancellation was requested."""


def _job_dir(job: Job) -> Path:
    output = storage_root() / "runs" / f"job_{job.id}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _refs(paths: list[str]) -> list[str]:
    """Storage-relative references for the files that exist, de-duplicated in order."""
    refs: list[str] = []
    for value in paths:
        if not value or not Path(value).is_file():
            continue
        ref = to_storage_ref(value)
        if ref is not None and ref not in refs:
            refs.append(ref)
    return refs


def _options(job: Job) -> dict[str, Any]:
    try:
        options = json.loads(job.options_json or "{}")
    except json.JSONDecodeError:
        options = {}
    return options if isinstance(options, dict) else {}


def _dataset_dir(job: Job) -> Path:
    if job.dataset_id is None:
        raise ValueError(f"A {job.kind} job requires a dataset.")
    images = storage_root() / "datasets" / str(job.dataset_id)
    if not images.is_dir() or not any(images.iterdir()):
        raise ValueError(f"Dataset {job.dataset_id} has no uploaded imagery at {images}.")
    return images


def _run_reconstruction(db: Session, job: Job, progress) -> tuple[list[str], int | None]:
    """Reconstruct a dataset's imagery, returning artifact references and the CRS."""
    from core.reconstruction_colmap import build_reconstructor

    images = _dataset_dir(job)
    output = _job_dir(job)

    progress(2, "Starting reconstruction")
    reconstructor = build_reconstructor(job.engine, profile=job.profile)
    # Keywords, not positions: the second positional parameter is crack_mask_dir, and
    # passing the output folder there sent every job's files to ./reconstruction.
    result = reconstructor.reconstruct(
        str(images), output_dir=str(output), profile=job.profile,
        progress_callback=lambda percent, message="": progress(
            2 + int(0.93 * max(0, min(100, int(percent)))), message),
    )
    progress(96, "Collecting artifacts")

    payload = result.to_dict() if hasattr(result, "to_dict") else {}
    paths: list[str] = []
    for key, value in payload.items():
        if not key.endswith(("_path", "_paths")):
            continue
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, str) and item:
                paths.append(item)
    summary = output / "reconstruction_result.json"
    summary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths.append(str(summary))
    return _refs(paths), payload.get("crs_epsg")


def _run_detection(db: Session, job: Job, progress) -> tuple[list[str], int | None]:
    """Run a defect model over a dataset and record each hit as an unreviewed defect."""
    from ..inference import choose_model, dataset_images, run_inference

    options = _options(job)
    choice = choose_model("detection", str(options.get("model_key") or DEFAULT_DETECTION_MODEL))
    images = dataset_images(_dataset_dir(job), str(options.get("image", "*")))
    document = run_inference(
        choice, images, _job_dir(job), parameters=options,
        progress=lambda percent, message="": progress(int(percent * 0.95), message),
    )

    created: list[Defect] = []
    for row in document["images"]:
        for hit in row.get("detections", []):
            x1, y1, x2, y2 = hit["bbox_xyxy"]
            created.append(Defect(
                project_id=job.project_id, job_id=job.id,
                category=str(hit["label"]).strip().lower().replace(" ", "_")[:80],
                severity="medium",
                description=(
                    f"{hit['label']} in {row['image']} at pixels ({x1},{y1})-({x2},{y2}), "
                    f"{hit['area_px']} px. Pixel geometry only: not georeferenced, "
                    "not measured."
                ),
                source=HEURISTIC_SOURCE if choice.is_heuristic else "model",
                model_key=choice.identity()["model_key"], model_sha256=choice.sha256,
                confidence=float(hit["confidence"]), review_state="unreviewed",
                created_by=job.created_by,
            ))
    db.add_all(created)
    job.message = (
        f"{len(created)} finding(s) over {len(images)} image(s), recorded unreviewed "
        f"({document['model_used']})."
    )
    db.commit()
    project = db.get(Project, job.project_id)
    if project is not None:
        from .inspection import _defect_out  # noqa: PLC0415

        for defect in created:
            emit(project.organization_id, "defect.created",
                 _defect_out(defect).model_dump(mode="json"))
    return _refs(document["artifact_paths"]), None


def _run_measurement(db: Session, job: Job, progress) -> tuple[list[str], int | None]:
    """Measure on a finished reconstruction's rasters and persist the measurement."""
    options = _options(job)
    kind = str(options.get("measurement", ""))
    source = db.get(Job, int(options["source_job_id"]))
    raster = _source_raster(source, "dsm" if kind == "volume" else
                            str(options.get("raster", "orthomosaic")))
    progress(20, f"Measuring {kind} on {raster.name}")

    if kind == "volume":
        from core.dsm_analysis import estimate_volume

        polygon = [[float(x), float(y)] for x, y in options["polygon_xy"]]
        dtm = _source_raster(source, "dtm", required=False)
        result = estimate_volume(
            raster, dtm_path=dtm, polygon_xy=polygon,
            base_elevation_m=options.get("base_elevation_m"),
        )
        preferred = result.get("preferred") or {}
        if not preferred:
            raise ValueError("The DSM gave no usable volume over this polygon.")
        value = float(preferred["cut_volume_m3"])
        epsg = result.get("crs_epsg")
        geometry = {"type": "Polygon", "coordinates": [polygon + [polygon[0]]]}
        method = (f"DSM cut volume above the '{preferred['reference']}' reference "
                  f"({result['region_cells']} cells at {result['pixel_size_m']} m)")
        detail: dict[str, Any] = result
    else:
        from core.raster_measurement import measure_area, measure_distance, measure_perimeter

        pixels = [(float(c), float(r)) for c, r in options["pixels"]]
        measured = {"distance": measure_distance, "area": measure_area,
                    "perimeter": measure_perimeter}[kind](raster, pixels)
        unit = str(measured.unit).lower().replace("square ", "")
        if unit not in {"metre", "meter", "metres", "meters", "m"}:
            raise ValueError(f"The raster's linear unit is {measured.unit!r}, not metres.")
        value = float(measured.value)
        epsg = measured.epsg
        world = [[float(x), float(y)] for x, y in measured.vertices_world]
        geometry = ({"type": "Polygon", "coordinates": [world + [world[0]]]} if kind == "area"
                    else {"type": "LineString", "coordinates": world})
        method = f"{kind} on {raster.name} from {len(pixels)} pixel vertices"
        detail = measured.to_dict()
    if epsg is None:
        raise ValueError("The raster carries no EPSG code, so the measurement has no CRS.")

    ref = to_storage_ref(raster) or raster.name
    row = Measurement(
        project_id=job.project_id, job_id=source.id, kind=kind, value=value,
        unit=MEASUREMENT_UNIT[kind], geometry_geojson=dumps_geometry(geometry),
        crs_epsg=int(epsg), source_ref=ref, method=method, created_by=job.created_by,
    )
    db.add(row)
    db.flush()
    out = _job_dir(job) / "measurement.json"
    out.write_text(json.dumps({"measurement_id": row.id, "kind": kind, "value": value,
                               "unit": MEASUREMENT_UNIT[kind], "crs_epsg": epsg,
                               "source": ref, "detail": detail}, indent=2, default=str),
                   encoding="utf-8")
    job.message = f"{kind} = {value:.3f} {MEASUREMENT_UNIT[kind]} (measurement {row.id})"
    db.commit()
    return _refs([str(out)]), int(epsg)


def _source_raster(source: Job | None, name: str, *, required: bool = True) -> Path | None:
    if source is None or source.status != "done":
        raise ValueError("The source job has no finished outputs to measure.")
    try:
        refs = json.loads(source.artifacts_json or "[]")
    except json.JSONDecodeError:
        refs = []
    for ref in refs:
        path = from_storage_ref(ref)
        lower = path.name.lower()
        if lower.startswith(name) and path.suffix.lower() in {".tif", ".tiff"} \
                and "cog" not in lower and path.is_file():
            return path
    if required:
        raise ValueError(
            f"Job {source.id} produced no georeferenced {name} raster. Reconstruct with "
            "the COLMAP engine to obtain metric rasters."
        )
    return None


def _run_report(db: Session, job: Job, progress) -> tuple[list[str], int | None]:
    """Render the project's records into a stored report: JSON snapshot and PDF."""
    from core.report_formats import ReportTemplate, write_pdf

    from .resources import build_report_payload

    options = _options(job)
    project = db.get(Project, job.project_id)
    include_unreviewed = bool(options.get("include_unreviewed", False))
    payload = build_report_payload(db, project, include_unreviewed=include_unreviewed)
    title = str(options.get("title") or "Inspection report")[:240]
    progress(30, "Rendering report")

    output = _job_dir(job)
    json_path = output / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    measurements = {
        f"{m['kind']} #{m['id']} ({m['unit']}, EPSG:{m['crs_epsg']})": m["value"]
        for m in payload["measurements"]
    }
    notes = " ".join(n for n in (payload.get("defect_note"), payload.get("findings_note")) if n)
    pdf_path = Path(write_pdf(output / "report.pdf", {
        "project_name": project.name, "client": project.client,
        "findings": payload["findings"], "measurements": measurements,
        "summary": notes,
    }, ReportTemplate(title=title, client=project.client or "",
                      include_unreviewed=include_unreviewed)))

    refs = _refs([str(json_path), str(pdf_path)])
    payload["artifacts"] = refs
    report = Report(project_id=project.id, title=title, format="pdf", status="complete",
                    payload_json=json.dumps(payload, default=str), created_by=job.created_by)
    db.add(report)
    db.flush()
    job.message = f"Report {report.id} written ({len(payload['findings'])} finding(s))."
    db.commit()
    emit(project.organization_id, "report.generated", {
        "report_id": report.id, "project_id": project.id, "title": title,
        "format": "pdf", "status": "complete", "job_id": job.id, "artifacts": refs,
    })
    return refs, None


_RUNNERS = {
    "reconstruction": _run_reconstruction,
    "detection": _run_detection,
    "measurement": _run_measurement,
    "report": _run_report,
}


def _validate_options(db: Session, project_id: int, payload: JobSubmit) -> None:
    """Refuse at submission what would only fail later on the worker."""
    options = payload.options
    if payload.kind == "detection":
        if payload.dataset_id is None:
            raise HTTPException(status_code=422, detail="A detection job requires dataset_id.")
        from ..inference import InferenceRefused, choose_model

        try:
            choose_model("detection", str(options.get("model_key") or DEFAULT_DETECTION_MODEL),
                         verify_digest=False)
        except InferenceRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.kind == "measurement":
        kind = options.get("measurement")
        if kind not in MEASUREMENT_UNIT:
            raise HTTPException(status_code=422, detail=(
                f"options.measurement must be one of: {', '.join(sorted(MEASUREMENT_UNIT))}."))
        source_id = options.get("source_job_id")
        source = db.get(Job, source_id) if isinstance(source_id, int) else None
        if source is None or source.project_id != project_id or source.kind != "reconstruction":
            raise HTTPException(status_code=422, detail=(
                "options.source_job_id must name a reconstruction job in this project."))
        key, minimum = ("polygon_xy", 3) if kind == "volume" else (
            "pixels", 3 if kind in {"area", "perimeter"} else 2)
        points = options.get(key)
        valid = isinstance(points, list) and len(points) >= minimum and all(
            isinstance(p, (list, tuple)) and len(p) == 2
            and all(isinstance(v, (int, float)) for v in p) for p in points)
        if not valid:
            what = "world x,y in the DSM's CRS" if kind == "volume" else "raster column,row"
            raise HTTPException(status_code=422, detail=(
                f"A {kind} measurement needs options.{key}: at least {minimum} [x, y] "
                f"pairs ({what})."))


def dispatch(job_id: int) -> str:
    """Start a queued job: on a local thread, or on a Celery worker when configured.

    ODK_JOB_BACKEND=celery sends the job id to the broker; any worker sharing this
    database and storage runs it through the same _run_job. The default runs it in this
    process, which is what a single-machine install wants.
    """
    backend = os.environ.get("ODK_JOB_BACKEND", "thread").strip().lower()
    if backend == "celery":
        from services.worker.tasks import enqueue_api_job  # noqa: PLC0415

        enqueue_api_job(job_id)
        return "celery"
    with _LOCK:
        _CANCEL[job_id] = threading.Event()
    threading.Thread(target=_run_job, args=(job_id,), name=f"job-{job_id}", daemon=True).start()
    return "thread"


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
    _validate_options(db, project_id, payload)

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

    out = _job_out(job)
    dispatch(job.id)
    return out


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
    if job.status == "queued":
        # Never started, so there is nothing to stop: it is cancelled now, and a worker
        # that picks it up later finds it no longer queued.
        job.status = "cancelled"
        job.message = "Cancelled before it started"
        job.finished_at = datetime.now(timezone.utc)
    else:
        # Written to the row as well as the in-process flag, so a worker in another
        # process sees it at its next checkpoint.
        job.status = "cancelling"
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


@router.get("/jobs/{job_id}/artifacts/{ref:path}")
def download_artifact(
    job_id: int, ref: str, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> FileResponse:
    """Download one file a job recorded as its output.

    Only references the job itself lists are served, so this cannot be used to read
    arbitrary files under the storage root.
    """
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    project = _project_or_404(db, job.project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    if ref not in _job_out(job).artifacts:
        raise HTTPException(status_code=404, detail="This job recorded no such artifact.")
    path = from_storage_ref(ref)
    if not path.is_file() or not within_root(path):
        raise HTTPException(status_code=410, detail="The artifact is no longer in storage.")
    return FileResponse(path, filename=path.name)
