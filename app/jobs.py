"""Background job runner for long operations.

Reconstruction, the analysis pipeline, and dataset imports take minutes. Running them
on the UI thread would freeze the window, so each runs on a worker thread that reports
progress and can be cancelled. Job records are kept so the UI can poll state without
holding a reference to the thread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
import traceback
from typing import Any, Callable
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCancelled(Exception):
    """Raised inside a job when the user cancels it."""


@dataclass
class Job:
    """State of one background operation, safe to serialize straight to the UI."""

    id: str
    name: str
    status: str = "pending"  # pending | running | done | failed | cancelled
    percent: int = 0
    message: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    result: Any = None
    error: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # The tail is all the UI shows, and full logs of a long run get large.
        payload["log"] = self.log[-200:]
        return payload


class JobManager:
    """Thread-backed job registry with progress reporting and cooperative cancel."""

    def __init__(self, max_history: int = 50):
        self._jobs: dict[str, Job] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._max_history = int(max_history)

    def submit(self, name: str, target: Callable[..., Any], **kwargs: Any) -> str:
        """Start `target` on a worker thread.

        `target` is called with a `progress(percent, message)` callable and a
        `should_cancel()` predicate injected as keyword arguments when it accepts them.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, name=name, status="pending")
        cancel = threading.Event()

        with self._lock:
            self._jobs[job_id] = job
            self._cancels[job_id] = cancel
            self._prune()

        def progress(percent: int, message: str = "") -> None:
            if cancel.is_set():
                raise JobCancelled(f"Job {name} cancelled")
            with self._lock:
                job.percent = int(max(0, min(100, percent)))
                if message:
                    job.message = message
                    job.log.append(f"[{job.percent:3d}%] {message}")

        def should_cancel() -> bool:
            return cancel.is_set()

        def run() -> None:
            with self._lock:
                job.status = "running"
                job.started_utc = _now()
            try:
                job.result = target(progress=progress, should_cancel=should_cancel, **kwargs)
                with self._lock:
                    job.status = "done"
                    job.percent = 100
                    job.message = job.message or "Complete"
            except JobCancelled:
                with self._lock:
                    job.status = "cancelled"
                    job.message = "Cancelled"
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.message = job.error
                    job.log.append(traceback.format_exc())
            finally:
                with self._lock:
                    job.finished_utc = _now()

        thread = threading.Thread(target=run, name=f"job-{name}-{job_id}", daemon=True)
        with self._lock:
            self._threads[job_id] = thread
        thread.start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values() if j.status in {"pending", "running"}]

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. The job stops at its next progress checkpoint."""
        with self._lock:
            event = self._cancels.get(job_id)
            job = self._jobs.get(job_id)
        if event is None or job is None or job.status not in {"pending", "running"}:
            return False
        event.set()
        with self._lock:
            job.message = "Cancelling..."
        return True

    def _prune(self) -> None:
        """Drop the oldest finished jobs once history exceeds the cap."""
        finished = [j for j in self._jobs.values() if j.status in {"done", "failed", "cancelled"}]
        excess = len(self._jobs) - self._max_history
        for job in sorted(finished, key=lambda j: j.finished_utc)[: max(0, excess)]:
            self._jobs.pop(job.id, None)
            self._cancels.pop(job.id, None)
            self._threads.pop(job.id, None)
