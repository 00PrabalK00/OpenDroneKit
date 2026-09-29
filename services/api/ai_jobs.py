"""The inference worker behind ``/projects/{id}/ai-jobs``.

An AI job is stored as ``pending_worker`` and claimed by exactly one worker with a
conditional update, so a job is never run twice and never silently dropped. Two ways
to run the worker, chosen by ``ODK_AI_WORKER``:

``inline`` (default)
    The API process starts a thread for each job it accepts, and on startup claims any
    job left ``pending_worker`` by a previous process.
``external``
    The API only records the request; ``python -m services.worker.ai_worker`` (one or
    more, sharing the database and storage) polls for pending jobs and runs them.

Cancellation of a running job is written to the row (``cancelling``) and checked
between images, so it works whichever process holds the job.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from .db import get_session_factory
from .models import Job, Project
from .paths import storage_root, to_storage_ref

PENDING = "pending_worker"


def worker_mode() -> str:
    return os.environ.get("ODK_AI_WORKER", "inline").strip().lower() or "inline"


def output_dir(job_id: int) -> Path:
    return storage_root() / "runs" / f"ai_job_{job_id}"


def dispatch(job_id: int) -> str:
    """Hand a newly recorded job to a worker according to the configured mode."""
    if worker_mode() == "external":
        return "external"
    threading.Thread(target=run_ai_job, args=(job_id,), name=f"ai-job-{job_id}",
                     daemon=True).start()
    return "inline"


def resume_pending() -> list[int]:
    """Start every job still waiting, e.g. after an inline API process restarted."""
    if worker_mode() == "external":
        return []
    session = get_session_factory()()
    try:
        ids = list(session.scalars(
            select(Job.id).where(Job.kind.like("ai:%"), Job.status == PENDING).order_by(Job.id)
        ))
    finally:
        session.close()
    for job_id in ids:
        dispatch(job_id)
    return ids


def claim_next() -> int | None:
    """Claim the oldest pending job for this worker, or None when there is none."""
    session = get_session_factory()()
    try:
        for job_id in session.scalars(
            select(Job.id).where(Job.kind.like("ai:%"), Job.status == PENDING)
            .order_by(Job.id).limit(20)
        ):
            if _claim(session, job_id):
                return job_id
        return None
    finally:
        session.close()


def _claim(session, job_id: int) -> bool:
    claimed = session.execute(
        update(Job).where(Job.id == job_id, Job.status == PENDING).values(
            status="running", started_at=datetime.now(timezone.utc), percent=0,
            message="Claimed by an inference worker.",
        )
    ).rowcount == 1
    session.commit()
    return claimed


class _Cancelled(Exception):
    pass


def run_ai_job(job_id: int, *, already_claimed: bool = False) -> str:
    """Claim (unless already claimed) and run one AI job. Returns its final status."""
    from .inference import (
        InferenceRefused, choose_model, dataset_images, parse_input_ref, run_inference,
    )

    session = get_session_factory()()
    status = "unclaimed"
    organization_id: int | None = None
    event: dict[str, Any] | None = None
    try:
        if not already_claimed and not _claim(session, job_id):
            return status
        job = session.get(Job, job_id)
        options = json.loads(job.options_json or "{}")
        task = job.kind.removeprefix("ai:")

        def progress(percent: int, message: str = "") -> None:
            session.refresh(job, attribute_names=["status"])
            if job.status == "cancelling":
                raise _Cancelled()
            job.percent = max(0, min(99, int(percent)))
            if message:
                job.message = message
            session.commit()

        try:
            choice = choose_model(task, str(options.get("model_key", "")))
            dataset_id, selector = parse_input_ref(str(options.get("input_ref", "")),
                                                   job.dataset_id)
            images = dataset_images(storage_root() / "datasets" / str(dataset_id), selector)
            document = run_inference(choice, images, output_dir(job_id),
                                     parameters=options.get("parameters") or {},
                                     progress=progress)
            refs = [ref for ref in (to_storage_ref(path) for path in document["artifact_paths"])
                    if ref]
            session.refresh(job, attribute_names=["status"])
            if job.status == "cancelling":
                raise _Cancelled()
            job.artifacts_json = json.dumps(refs)
            job.engine = document["model_used"]
            job.status = "done"
            job.percent = 100
            job.message = (f"{task} with {choice.model_key} over {len(images)} image(s): "
                           f"{json.dumps(document['summary'])}")[:2000]
        except _Cancelled:
            session.rollback()
            job = session.get(Job, job_id)
            job.status = "cancelled"
            job.message = "Cancelled while running; partial outputs were discarded."
        except InferenceRefused as exc:
            session.rollback()
            job = session.get(Job, job_id)
            job.status = "failed"
            job.error = str(exc)
            job.message = f"Refused: {exc}"
        except Exception as exc:  # noqa: BLE001 - recorded on the job, never swallowed
            session.rollback()
            job = session.get(Job, job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            job.log = traceback.format_exc()[-8000:]
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
        status = job.status
        project = session.get(Project, job.project_id)
        organization_id = project.organization_id if project is not None else None
        event = {
            "ai_job_id": job.id, "project_id": job.project_id, "task": task,
            "model_key": options.get("model_key", ""), "status": job.status,
            "message": job.message, "error": job.error,
            "artifacts": json.loads(job.artifacts_json or "[]"),
        }
    finally:
        session.close()
    if organization_id is not None and event is not None and status in {"done", "failed"}:
        from .routers.events import emit  # noqa: PLC0415

        emit(organization_id, "ai.completed", event)
    return status


def read_results(job: Job) -> dict[str, Any] | None:
    """The results document a finished job wrote, without the per-image detail."""
    path = output_dir(job.id) / "results.json"
    if job.status != "done" or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {key: document.get(key) for key in
            ("task", "model_key", "model_sha256", "model_used", "image_count", "summary",
             "caveat")}
