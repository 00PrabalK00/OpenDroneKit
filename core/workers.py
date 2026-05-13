"""Reusable worker patterns for long-running core tasks.

The worker layer is intentionally Qt-free so it can be used from the desktop
UI, command-line scripts, tests, and future service processes. It provides:

* cancellable background tasks
* progress/result dataclasses
* callback hooks
* an executor-backed worker pool
* a small stage runner for multi-step pipelines
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import threading
import traceback
import uuid
from typing import Any, Callable, Generic, Iterable, TypeVar

from .events import (
    PROCESSING_CANCELLED,
    PROCESSING_COMPLETED,
    PROCESSING_FAILED,
    PROCESSING_PROGRESS,
    PROCESSING_STARTED,
    publish_event,
)


T = TypeVar("T")

WORKER_PENDING = "pending"
WORKER_RUNNING = "running"
WORKER_COMPLETED = "completed"
WORKER_FAILED = "failed"
WORKER_CANCELLED = "cancelled"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class WorkerCancelled(RuntimeError):
    """Raised when a worker cooperatively stops after cancellation."""


@dataclass(frozen=True)
class WorkerProgress:
    task_id: str
    task_name: str
    status: str = WORKER_RUNNING
    percent: float = 0.0
    message: str = ""
    stage: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status,
            "percent": float(self.percent),
            "message": self.message,
            "stage": self.stage,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class WorkerResult(Generic[T]):
    task_id: str
    task_name: str
    status: str
    value: T | None = None
    error: str = ""
    traceback_text: str = ""
    started_at: str = ""
    finished_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == WORKER_COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status,
            "value": self.value,
            "error": self.error,
            "traceback_text": self.traceback_text,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise WorkerCancelled("Worker task was cancelled.")


ProgressCallback = Callable[[WorkerProgress], None]
ResultCallback = Callable[[WorkerResult[Any]], None]


class WorkerContext:
    """Context passed to worker functions that opt into progress/cancellation."""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        token: CancellationToken,
        emit: Callable[[WorkerProgress], None],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.task_name = task_name
        self.token = token
        self.metadata = dict(metadata or {})
        self._emit = emit

    def is_cancelled(self) -> bool:
        return self.token.is_cancelled()

    def check_cancelled(self) -> None:
        self.token.raise_if_cancelled()

    def progress(
        self,
        percent: float,
        message: str = "",
        stage: str = "",
        payload: dict[str, Any] | None = None,
        status: str = WORKER_RUNNING,
    ) -> WorkerProgress:
        self.check_cancelled()
        progress = WorkerProgress(
            task_id=self.task_id,
            task_name=self.task_name,
            status=status,
            percent=max(0.0, min(100.0, float(percent))),
            message=str(message),
            stage=str(stage),
            payload=dict(payload or {}),
        )
        self._emit(progress)
        return progress


@dataclass(frozen=True)
class WorkerStage:
    id: str
    name: str
    run: Callable[[WorkerContext], Any]
    weight: float = 1.0


class WorkerHandle(Generic[T]):
    """Handle returned by WorkerPool.submit."""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        future: Future[WorkerResult[T]],
        token: CancellationToken,
        progress_callbacks: list[ProgressCallback],
    ) -> None:
        self.task_id = task_id
        self.task_name = task_name
        self._future = future
        self._token = token
        self._progress_callbacks = progress_callbacks

    def cancel(self) -> bool:
        self._token.cancel()
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._token.is_cancelled() or self._future.cancelled()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> WorkerResult[T]:
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def add_done_callback(self, callback: ResultCallback) -> None:
        def _forward(fut: Future[WorkerResult[T]]) -> None:
            try:
                callback(fut.result())
            except Exception:
                pass

        self._future.add_done_callback(_forward)

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        if callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)


class WorkerPool:
    """Thread-pool worker coordinator with progress and event-bus publishing."""

    def __init__(
        self,
        max_workers: int = 4,
        publish_events: bool = True,
        thread_name_prefix: str = "odk-worker",
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.publish_events = bool(publish_events)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._handles: dict[str, WorkerHandle[Any]] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        task_name: str,
        func: Callable[..., T],
        *args: Any,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        use_context: bool | None = None,
        progress_callback: ProgressCallback | None = None,
        result_callback: ResultCallback | None = None,
        **kwargs: Any,
    ) -> WorkerHandle[T]:
        """Submit a task.

        If `use_context` is None, the worker inspects the callable signature and
        passes WorkerContext when the first argument is named context, ctx, or
        worker_context. Set `use_context=True` to force passing it.
        """
        tid = task_id or str(uuid.uuid4())
        token = CancellationToken()
        callbacks: list[ProgressCallback] = []
        if progress_callback is not None:
            callbacks.append(progress_callback)

        future = self._executor.submit(
            self._run_task,
            tid,
            str(task_name),
            func,
            args,
            kwargs,
            token,
            callbacks,
            dict(metadata or {}),
            use_context,
        )
        handle: WorkerHandle[T] = WorkerHandle(tid, str(task_name), future, token, callbacks)
        if result_callback is not None:
            handle.add_done_callback(result_callback)
        with self._lock:
            self._handles[tid] = handle
        future.add_done_callback(lambda _f: self._forget(tid))
        return handle

    def get(self, task_id: str) -> WorkerHandle[Any] | None:
        with self._lock:
            return self._handles.get(task_id)

    def active(self) -> list[WorkerHandle[Any]]:
        with self._lock:
            return list(self._handles.values())

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        if cancel_futures:
            for handle in self.active():
                handle.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _forget(self, task_id: str) -> None:
        with self._lock:
            self._handles.pop(task_id, None)

    def _emit(self, progress: WorkerProgress, callbacks: list[ProgressCallback]) -> None:
        if self.publish_events:
            publish_event(PROCESSING_PROGRESS, progress.to_dict())
        for cb in list(callbacks):
            try:
                cb(progress)
            except Exception:
                pass

    def _run_task(
        self,
        task_id: str,
        task_name: str,
        func: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        token: CancellationToken,
        callbacks: list[ProgressCallback],
        metadata: dict[str, Any],
        use_context: bool | None,
    ) -> WorkerResult[T]:
        started_at = _now_iso()

        def _emit(progress: WorkerProgress) -> None:
            self._emit(progress, callbacks)

        context = WorkerContext(task_id, task_name, token, _emit, metadata=metadata)
        try:
            if self.publish_events:
                publish_event(PROCESSING_STARTED, {"task_id": task_id, "task_name": task_name, **metadata})
            context.progress(0.0, "Started.", status=WORKER_RUNNING)
            value = self._invoke(func, context, args, kwargs, use_context)
            token.raise_if_cancelled()
            context.progress(100.0, "Completed.", status=WORKER_COMPLETED)
            result: WorkerResult[T] = WorkerResult(
                task_id=task_id,
                task_name=task_name,
                status=WORKER_COMPLETED,
                value=value,
                started_at=started_at,
                metadata=metadata,
            )
            if self.publish_events:
                publish_event(PROCESSING_COMPLETED, result.to_dict())
            return result
        except WorkerCancelled as exc:
            result = WorkerResult[T](
                task_id=task_id,
                task_name=task_name,
                status=WORKER_CANCELLED,
                error=str(exc),
                started_at=started_at,
                metadata=metadata,
            )
            _emit(WorkerProgress(task_id, task_name, WORKER_CANCELLED, 0.0, str(exc)))
            if self.publish_events:
                publish_event(PROCESSING_CANCELLED, result.to_dict())
            return result
        except Exception as exc:
            tb = traceback.format_exc()
            result = WorkerResult[T](
                task_id=task_id,
                task_name=task_name,
                status=WORKER_FAILED,
                error=str(exc),
                traceback_text=tb,
                started_at=started_at,
                metadata=metadata,
            )
            _emit(WorkerProgress(task_id, task_name, WORKER_FAILED, 0.0, str(exc)))
            if self.publish_events:
                publish_event(PROCESSING_FAILED, result.to_dict())
            return result

    @staticmethod
    def _invoke(
        func: Callable[..., T],
        context: WorkerContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        use_context: bool | None,
    ) -> T:
        if use_context is True:
            return func(context, *args, **kwargs)
        if use_context is False:
            return func(*args, **kwargs)
        try:
            params = list(inspect.signature(func).parameters.values())
        except (TypeError, ValueError):
            params = []
        if params:
            first = params[0]
            if first.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ) and first.name in {"context", "ctx", "worker_context"}:
                return func(context, *args, **kwargs)
        return func(*args, **kwargs)


def run_stages(context: WorkerContext, stages: Iterable[WorkerStage]) -> list[Any]:
    """Run weighted stages inside an existing worker context."""
    stage_list = list(stages)
    total_weight = sum(max(0.01, float(stage.weight)) for stage in stage_list) or 1.0
    completed_weight = 0.0
    results: list[Any] = []
    for stage in stage_list:
        context.check_cancelled()
        start_pct = 100.0 * completed_weight / total_weight
        context.progress(start_pct, f"Starting {stage.name}.", stage=stage.id)
        result = stage.run(context)
        results.append(result)
        completed_weight += max(0.01, float(stage.weight))
        end_pct = 100.0 * completed_weight / total_weight
        context.progress(end_pct, f"Completed {stage.name}.", stage=stage.id)
    return results


_default_pool: WorkerPool | None = None
_default_pool_lock = threading.RLock()


def get_worker_pool(max_workers: int = 4) -> WorkerPool:
    global _default_pool
    with _default_pool_lock:
        if _default_pool is None:
            _default_pool = WorkerPool(max_workers=max_workers)
        return _default_pool


def submit_worker(
    task_name: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> WorkerHandle[T]:
    return get_worker_pool().submit(task_name, func, *args, **kwargs)


__all__ = [
    "WORKER_PENDING",
    "WORKER_RUNNING",
    "WORKER_COMPLETED",
    "WORKER_FAILED",
    "WORKER_CANCELLED",
    "WorkerCancelled",
    "WorkerProgress",
    "WorkerResult",
    "CancellationToken",
    "WorkerContext",
    "WorkerStage",
    "WorkerHandle",
    "WorkerPool",
    "run_stages",
    "get_worker_pool",
    "submit_worker",
]
