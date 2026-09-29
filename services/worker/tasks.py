"""The processing pipeline as a Celery task, with a cancel that crosses machines.

Wrapping run_pipeline in a task is the easy half. The half that matters is
cancellation, and it is where an in-process design quietly stops being correct once
there is more than one process.

core/processing_runs.py cancels through ``_cancel_event``, a module-level dict of
threading.Events keyed by run id. That works when the thing calling stop_processing_run
is the same process running the pipeline. It does not work at all when a worker is on
another machine: the API sets an Event in its own memory, reports the run cancelled, and
the worker carries on reconstructing for another forty minutes. Nobody is told. The
operator sees "cancelled" and the cluster keeps burning.

So cancellation goes through Redis, which both sides can see. The API sets a key; the
worker checks it on every progress callback -- which the pipeline already emits between
stages -- and asks its own local machinery to stop. The local Event is still what stops
the work; Redis is only how the request crosses the gap.

This is deliberately cooperative rather than a kill. A reconstruction interrupted
mid-stage leaves partial artefacts, and a stage boundary is a place where the run's
recorded state and what is on disk agree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

CANCEL_PREFIX = "odk:cancel:"
# Long enough to outlive any run, short enough that a cancelled run's key does not
# linger forever and cancel a later run that reuses the id.
CANCEL_TTL_S = 60 * 60 * 24


def cancel_key(run_id: str) -> str:
    return f"{CANCEL_PREFIX}{run_id}"


def request_cancel(run_id: str) -> bool:
    """Ask whichever worker holds this run to stop. Called by the API side.

    Returns False when the broker cannot be reached, rather than raising: a caller
    deserves to know the request did not land, and an exception here would read as the
    run failing rather than the cancel failing.
    """
    try:
        import redis

        from services.worker.celery_app import broker_url

        client = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
        client.set(cancel_key(run_id), "1", ex=CANCEL_TTL_S)
        return True
    except Exception:  # noqa: BLE001 - reported through the return value
        return False


def cancel_requested(run_id: str) -> bool:
    """Whether a cancel is pending for this run.

    A broker that cannot be reached answers False. That is the safe direction: the
    alternative is treating a network blip as a cancellation and throwing away an
    hour of reconstruction that was going fine.
    """
    try:
        import redis

        from services.worker.celery_app import broker_url

        client = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
        return bool(client.exists(cancel_key(run_id)))
    except Exception:  # noqa: BLE001
        return False


def clear_cancel(run_id: str) -> None:
    try:
        import redis

        from services.worker.celery_app import broker_url

        redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0).delete(cancel_key(run_id))
    except Exception:  # noqa: BLE001
        pass


def execute_run(
    project_root: str,
    run_id: str,
    stages: list[str] | None = None,
    *,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the pipeline, honouring a cancel request that arrives from another process.

    Separated from the Celery task decorator so it can be tested, and run, without a
    broker. A task body that only exists inside a decorator is a task body nobody
    exercises until deployment.
    """
    from core.processing_runs import get_processing_status, run_pipeline, stop_processing_run

    def progress(percent: float, message: str = "") -> None:
        # Checked here rather than in a background thread: the pipeline emits progress
        # between stages, which is exactly where stopping leaves the recorded state and
        # the files on disk agreeing with each other.
        if cancel_requested(run_id):
            stop_processing_run(project_root, run_id)
        if on_progress is not None:
            on_progress(percent, message)

    try:
        run = run_pipeline(project_root, run_id, stages, progress_callback=progress)
        status = getattr(run, "status", "unknown")
    except Exception as exc:  # noqa: BLE001 - recorded in the result, then re-raised
        # Re-raised so Celery marks the task failed and acks_late can return it, but
        # named first so the reason survives in the worker log next to the run id.
        print(f"run {run_id} failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        clear_cancel(run_id)

    return {
        "run_id": run_id,
        "status": status,
        "progress": get_processing_status(project_root, run_id).progress_percent,
    }


def enqueue_api_job(job_id: int, app: Any = None) -> Any:
    """Send an API processing job to the broker. Called by the API when
    ODK_JOB_BACKEND=celery; the worker runs it with the same code the API would."""
    if app is None:
        from services.worker.celery_app import build_app

        app = build_app()
    return app.send_task("odk.api.run_job", args=[int(job_id)])


def run_api_job(job_id: int) -> dict[str, Any]:
    """Body of the API-job task: claim and run one queued job from the shared database."""
    from services.api.db import get_session_factory
    from services.api.models import Job
    from services.api.routers.processing import _run_job

    _run_job(int(job_id))
    session = get_session_factory()()
    try:
        job = session.get(Job, int(job_id))
        return {"job_id": int(job_id), "status": job.status if job is not None else "missing"}
    finally:
        session.close()


def register(app: Any) -> Any:
    """Attach the task to a Celery app.

    A function rather than a decorator at import time, so importing this module does not
    require a broker and the same task can be registered onto a test app.
    """

    @app.task(name="odk.processing.run", bind=True)
    def run_processing(self, project_root: str, run_id: str,
                       stages: list[str] | None = None) -> dict[str, Any]:
        def report(percent: float, message: str = "") -> None:
            # STARTED plus progress, so a polling client can tell a long stage from a
            # stuck one. Without this a running task and a queued task look identical.
            self.update_state(state="PROGRESS",
                              meta={"run_id": run_id, "percent": percent, "message": message})

        return execute_run(project_root, run_id, stages, on_progress=report)

    @app.task(name="odk.api.run_job")
    def run_api_job_task(job_id: int) -> dict[str, Any]:
        return run_api_job(job_id)

    return run_processing


def worker_pool() -> str:
    """The Celery execution pool this host can actually run.

    Celery's default, prefork, does not work on Windows: every task fails inside the
    pool with "ValueError: not enough values to unpack (expected 3, got 0)" and the
    job stays queued forever while the worker looks healthy. A worker started on a
    Windows machine (a site laptop, a workstation joining the queue) never ran a job.
    `solo` runs tasks in the worker process itself, which at the default concurrency
    of 1 is the same thing prefork was doing. ODK_WORKER_POOL overrides either way.
    """
    import sys

    override = os.environ.get("ODK_WORKER_POOL", "").strip()
    if override:
        return override
    return "solo" if sys.platform == "win32" else "prefork"


def worker_main() -> int:
    """Entry point for the worker container."""
    from services.worker.celery_app import build_app

    app = build_app()
    register(app)
    concurrency = os.environ.get("ODK_WORKER_CONCURRENCY", "1")
    # Concurrency 1 by default. Reconstruction is memory-bound, so two workers on one
    # host is usually slower than one and occasionally fatal -- the second job arrives
    # while the first holds the memory, and the OOM killer picks whichever it likes.
    app.worker_main(["worker", "--loglevel=INFO", f"--concurrency={concurrency}",
                     f"--pool={worker_pool()}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
