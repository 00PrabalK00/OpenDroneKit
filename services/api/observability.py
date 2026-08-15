"""Dependency-free API observability with explicit operational boundaries.

Request traces use W3C Trace Context and finish as structured JSON log spans.  No
collector is implied: unless an operator ships stdout, traces remain local logs.
Prometheus metrics are process-local, so a multi-worker deployment must scrape every
worker or aggregate at its service layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from sqlalchemy import text

from .db import get_engine, spatial_backend
from .storage import build_storage, describe_storage


TRACEPARENT = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$", re.IGNORECASE
)
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
PROCESS_STARTED = time.time()
TRACE_ID: ContextVar[str] = ContextVar("odk_trace_id", default="")
SPAN_ID: ContextVar[str] = ContextVar("odk_span_id", default="")


class JsonFormatter(logging.Formatter):
    """One JSON object per line for collection by ordinary container runtimes."""

    FIELDS = (
        "event", "trace_id", "span_id", "parent_span_id", "method", "route",
        "path", "status_code", "duration_ms", "client", "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.FIELDS:
            value = getattr(record, field, None)
            if value not in (None, ""):
                payload[field] = value
        payload.setdefault("trace_id", TRACE_ID.get())
        payload.setdefault("span_id", SPAN_ID.get())
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


LOGGER = logging.getLogger("opendronekit.api")


def configure_logging() -> logging.Logger:
    """Install exactly one JSON stdout/stderr handler for this API logger."""
    if not any(getattr(handler, "_odk_json", False) for handler in LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._odk_json = True  # type: ignore[attr-defined]
        LOGGER.addHandler(handler)
    LOGGER.setLevel(getattr(logging, os.environ.get("ODK_LOG_LEVEL", "INFO").upper(), logging.INFO))
    LOGGER.propagate = False
    return LOGGER


class Metrics:
    """Small Prometheus registry; aggregates durations without retaining requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._duration_sum: Counter[tuple[str, str]] = Counter()
        self._duration_count: Counter[tuple[str, str]] = Counter()
        self._duration_buckets: Counter[tuple[str, str, float]] = Counter()
        self._in_flight = 0

    def started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finished(self, method: str, route: str, status_code: int, duration_s: float) -> None:
        key = (method, route)
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._requests[(method, route, int(status_code))] += 1
            self._duration_sum[key] += float(duration_s)
            self._duration_count[key] += 1
            for bucket in BUCKETS:
                if duration_s <= bucket:
                    self._duration_buckets[(method, route, bucket)] += 1

    @staticmethod
    def _label(value: str) -> str:
        return json.dumps(value)[1:-1]

    def render(self) -> str:
        with self._lock:
            requests = dict(self._requests)
            sums = dict(self._duration_sum)
            counts = dict(self._duration_count)
            buckets = dict(self._duration_buckets)
            in_flight = self._in_flight
        lines = [
            "# HELP odk_observability_scope_info Operational scope and delivery contract.",
            "# TYPE odk_observability_scope_info gauge",
            'odk_observability_scope_info{metrics="per_process",traces="json_log_spans",guarantee="best_effort"} 1',
            "# HELP odk_process_start_time_seconds Process start time since Unix epoch.",
            "# TYPE odk_process_start_time_seconds gauge",
            f"odk_process_start_time_seconds {PROCESS_STARTED:.6f}",
            "# HELP odk_http_requests_in_flight Requests currently executing in this worker.",
            "# TYPE odk_http_requests_in_flight gauge",
            f"odk_http_requests_in_flight {in_flight}",
            "# HELP odk_http_requests_total Completed HTTP requests.",
            "# TYPE odk_http_requests_total counter",
        ]
        for (method, route, status_code), count in sorted(requests.items()):
            labels = f'method="{self._label(method)}",route="{self._label(route)}",status="{status_code}"'
            lines.append(f"odk_http_requests_total{{{labels}}} {count}")
        lines.extend((
            "# HELP odk_http_request_duration_seconds Request duration by route.",
            "# TYPE odk_http_request_duration_seconds histogram",
        ))
        for method, route in sorted(counts):
            labels = f'method="{self._label(method)}",route="{self._label(route)}"'
            for bucket in BUCKETS:
                lines.append(
                    f'odk_http_request_duration_seconds_bucket{{{labels},le="{bucket:g}"}} '
                    f'{buckets.get((method, route, bucket), 0)}'
                )
            lines.append(
                f'odk_http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {counts[(method, route)]}'
            )
            lines.append(f"odk_http_request_duration_seconds_sum{{{labels}}} {sums[(method, route)]:.9f}")
            lines.append(f"odk_http_request_duration_seconds_count{{{labels}}} {counts[(method, route)]}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()


def _trace_context(header: str) -> tuple[str, str, str]:
    match = TRACEPARENT.fullmatch((header or "").strip())
    if match and match.group(1) != "0" * 32 and match.group(2) != "0" * 16:
        return match.group(1).lower(), match.group(2).lower(), match.group(3).lower()
    return secrets.token_hex(16), "", "01"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id, parent_span_id, flags = _trace_context(request.headers.get("traceparent", ""))
        span_id = secrets.token_hex(8)
        trace_token, span_token = TRACE_ID.set(trace_id), SPAN_ID.set(span_id)
        started = time.perf_counter()
        status_code = 500
        error_type = ""
        METRICS.started()
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["traceparent"] = f"00-{trace_id}-{span_id}-{flags}"
            response.headers["x-trace-id"] = trace_id
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            duration = time.perf_counter() - started
            route_obj = request.scope.get("route")
            route = getattr(route_obj, "path", request.url.path)
            METRICS.finished(request.method, str(route), status_code, duration)
            extra = {
                "event": "http.request", "trace_id": trace_id, "span_id": span_id,
                "parent_span_id": parent_span_id, "method": request.method,
                "path": request.url.path, "route": str(route), "status_code": status_code,
                "duration_ms": round(duration * 1000.0, 3),
                "client": request.client.host if request.client else "", "error_type": error_type,
            }
            LOGGER.log(logging.ERROR if status_code >= 500 else logging.INFO, "request complete", extra=extra)
            TRACE_ID.reset(trace_token)
            SPAN_ID.reset(span_token)


def database_readiness() -> dict[str, Any]:
    description = spatial_backend()
    try:
        with get_engine().connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar()
        return {"ready": value == 1, **description}
    except Exception as exc:  # noqa: BLE001 - health must report backend failure
        return {
            "ready": False, **description, "error_type": type(exc).__name__,
            "note": "The configured database did not answer SELECT 1.",
        }


def storage_readiness() -> dict[str, Any]:
    description = describe_storage()
    try:
        backend = build_storage()
        if backend.name == "local":
            path = Path(str(description["path"]))
            path.mkdir(parents=True, exist_ok=True)
            ready = path.is_dir() and os.access(path, os.R_OK | os.W_OK)
            if not ready:
                raise PermissionError("configured local storage path is not readable and writable")
        else:
            # A list call proves credentials, endpoint and bucket access without writing.
            backend.list("__odk_health_probe__/")
        return {"ready": True, **description}
    except Exception as exc:  # noqa: BLE001 - health must report configured backend failure
        return {
            "ready": False, **description, "error_type": type(exc).__name__,
            "note": "The configured storage backend is not reachable with current settings.",
        }


def observability_contract() -> dict[str, Any]:
    return {
        "structured_logs": {"format": "json_lines", "destination": "stderr"},
        "metrics": {"format": "prometheus", "path": "/metrics", "scope": "per_process"},
        "tracing": {
            "propagation": "w3c_trace_context", "span_sink": "structured_logs",
            "collector_export": False,
        },
        "guarantee": (
            "Telemetry is best-effort operational evidence. Metrics are local to each "
            "worker and trace spans are not durably exported unless the operator ships logs."
        ),
    }
