"""A bounded, prioritised job queue with retries that admit what they did.

The desktop JobManager starts a thread per submission. That is right for a UI, where
work arrives one click at a time, and wrong for processing: reconstruction is
memory-bound before it is CPU-bound, so eight simultaneous jobs on a four-core box do
not finish sooner -- they finish later, or get OOM-killed halfway through and lose hours
of work that was nearly done.

This queue bounds concurrency, orders by priority, and retries. The retry behaviour is
where most of the care went, because a retry is a claim that the failure was transient
and that claim is usually wrong:

  * Retries are opt-in per job. A deterministic failure retried three times is the same
    failure three times, an hour later.
  * Every attempt is recorded with its own error. A job that failed three times for
    three different reasons is a different problem from one that failed identically, and
    a summary reporting only the last error hides which one you have.
  * A job that exhausts its attempts reports `failed`, never `pending`. Work that will
    never run again must not look like work that has not run yet.

Priority is strict, with first-in-first-out within a level. Strict priority can starve:
a continuous stream of high-priority work will hold low-priority jobs indefinitely. That
is not prevented here, because silently promoting a job the operator marked low is its
own surprise -- instead every job records how long it waited, so starvation is visible
in the queue's own report rather than inferred from a job that never ran.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class JobCancelled(Exception):
    """Raised inside a job target when cancellation has been requested."""


@dataclass
class Attempt:
    """One run of a job, successful or not."""

    number: int
    started_monotonic: float
    finished_monotonic: float | None = None
    error: str = ""

    @property
    def duration_s(self) -> float:
        if self.finished_monotonic is None:
            return 0.0
        return max(0.0, self.finished_monotonic - self.started_monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.number,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


@dataclass
class QueuedJob:
    id: str
    name: str
    priority: int = 100
    max_attempts: int = 1
    status: str = "queued"  # queued | running | done | failed | cancelled
    result: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    submitted_monotonic: float = 0.0
    started_monotonic: float | None = None

    @property
    def error(self) -> str:
        """The last failure, if any. Read `attempts` when the history matters."""
        return self.attempts[-1].error if self.attempts else ""

    def waited_s(self, now: float | None = None) -> float:
        reference = self.started_monotonic
        if reference is None:
            reference = time.monotonic() if now is None else now
        return max(0.0, reference - self.submitted_monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status,
            "attempts_made": len(self.attempts),
            "max_attempts": self.max_attempts,
            "error": self.error,
            "attempt_history": [a.to_dict() for a in self.attempts],
            "waited_s": round(self.waited_s(), 3),
        }


class JobQueue:
    """Runs submitted work across a fixed number of worker threads."""

    def __init__(self, workers: int = 2):
        if int(workers) < 1:
            raise ValueError("A queue needs at least one worker.")
        self._workers = int(workers)
        self._heap: list[tuple[int, int, str]] = []
        self._jobs: dict[str, QueuedJob] = {}
        self._targets: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._sequence = 0
        self._lock = threading.RLock()
        self._work_available = threading.Condition(self._lock)
        self._threads: list[threading.Thread] = []
        self._stopping = False
        self._active = 0

    # -- submission ----------------------------------------------------------

    def submit(self, name: str, target: Callable[..., Any], *,
               priority: int = 100, max_attempts: int = 1, **kwargs: Any) -> str:
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1.")
        job_id = uuid.uuid4().hex[:12]
        job = QueuedJob(
            id=job_id, name=name, priority=int(priority),
            max_attempts=int(max_attempts), submitted_monotonic=time.monotonic(),
        )
        with self._work_available:
            self._sequence += 1
            self._jobs[job_id] = job
            self._targets[job_id] = (target, dict(kwargs))
            self._cancels[job_id] = threading.Event()
            # The sequence number breaks priority ties in submission order, so equal
            # priority behaves first-in-first-out rather than by dictionary chance.
            heapq.heappush(self._heap, (int(priority), self._sequence, job_id))
            self._work_available.notify()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Queued jobs stop immediately; running ones cooperate."""
        with self._work_available:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"done", "failed", "cancelled"}:
                return False
            self._cancels[job_id].set()
            if job.status == "queued":
                job.status = "cancelled"
                self._work_available.notify_all()
            return True

    # -- execution -----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            self._stopping = False
            for index in range(self._workers):
                thread = threading.Thread(
                    target=self._run_worker, name=f"odk-job-worker-{index}", daemon=True
                )
                self._threads.append(thread)
                thread.start()

    def _next_job(self) -> str | None:
        while self._heap:
            _priority, _sequence, job_id = heapq.heappop(self._heap)
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                continue  # cancelled while waiting
            return job_id
        return None

    def _run_worker(self) -> None:
        while True:
            with self._work_available:
                job_id = self._next_job()
                while job_id is None:
                    if self._stopping:
                        return
                    self._work_available.wait(timeout=0.05)
                    if self._stopping and not self._heap:
                        return
                    job_id = self._next_job()
                job = self._jobs[job_id]
                job.status = "running"
                job.started_monotonic = time.monotonic()
                target, kwargs = self._targets[job_id]
                cancel = self._cancels[job_id]
                self._active += 1
            try:
                self._execute(job, target, kwargs, cancel)
            finally:
                with self._work_available:
                    self._active -= 1
                    self._work_available.notify_all()

    def _execute(self, job: QueuedJob, target: Callable[..., Any],
                 kwargs: dict[str, Any], cancel: threading.Event) -> None:
        for number in range(1, job.max_attempts + 1):
            if cancel.is_set():
                with self._lock:
                    job.status = "cancelled"
                return
            attempt = Attempt(number=number, started_monotonic=time.monotonic())
            with self._lock:
                job.attempts.append(attempt)
            try:
                result = target(should_cancel=cancel.is_set, **kwargs) \
                    if _accepts_cancel(target) else target(**kwargs)
            except JobCancelled:
                attempt.finished_monotonic = time.monotonic()
                attempt.error = "cancelled"
                with self._lock:
                    job.status = "cancelled"
                return
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                attempt.finished_monotonic = time.monotonic()
                attempt.error = f"{type(exc).__name__}: {exc}"
                # Fall through to the next attempt, if any remain.
                continue
            attempt.finished_monotonic = time.monotonic()
            with self._lock:
                job.result = result
                job.status = "done"
            return

        with self._lock:
            # Exhausted, not waiting. A job that will never run again must not read as
            # one that has not run yet.
            job.status = "failed"

    # -- inspection ----------------------------------------------------------

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.to_dict()

    def result(self, job_id: str) -> Any:
        with self._lock:
            return self._jobs[job_id].result

    def join(self, timeout: float = 30.0) -> bool:
        """Wait until nothing is queued or running. False if the timeout wins."""
        deadline = time.monotonic() + float(timeout)
        with self._work_available:
            while True:
                pending = any(j.status in {"queued", "running"} for j in self._jobs.values())
                if not pending:
                    return True
                if time.monotonic() >= deadline:
                    return False
                self._work_available.wait(timeout=0.02)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._work_available:
            self._stopping = True
            self._work_available.notify_all()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    def report(self) -> dict[str, Any]:
        """What the queue is doing, including how long work has been waiting.

        The longest wait is reported because strict priority can starve: a steady
        stream of high-priority jobs holds low-priority ones indefinitely, and the only
        symptom is a job that never starts. A number here is how that becomes visible.
        """
        with self._lock:
            jobs = list(self._jobs.values())
            now = time.monotonic()
            queued = [j for j in jobs if j.status == "queued"]
            by_status: dict[str, int] = {}
            for job in jobs:
                by_status[job.status] = by_status.get(job.status, 0) + 1
            return {
                "workers": self._workers,
                "running": sum(1 for j in jobs if j.status == "running"),
                "queued": len(queued),
                "by_status": dict(sorted(by_status.items())),
                "longest_wait_s": round(max((j.waited_s(now) for j in queued), default=0.0), 3),
                "retried_jobs": sum(1 for j in jobs if len(j.attempts) > 1),
            }


def _accepts_cancel(target: Callable[..., Any]) -> bool:
    import inspect

    try:
        return "should_cancel" in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
