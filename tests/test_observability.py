"""Operational evidence through real ASGI requests and real configured backends."""

from __future__ import annotations

import io
import json
import logging
import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_structured_spans_metrics_tracing_and_backend_health(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'observability.db'}")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "objects"))
    monkeypatch.setenv("ODK_STORAGE_BACKEND", "local")
    monkeypatch.setenv("ODK_SECRET_KEY", "observability-test-secret-that-is-long-enough")

    import services.api.db as db_module
    from services.api.observability import JsonFormatter, LOGGER

    db_module._engine = None
    db_module._SessionLocal = None
    from services.api.main import app

    stream = io.StringIO()
    capture = logging.StreamHandler(stream)
    capture.setFormatter(JsonFormatter())
    LOGGER.addHandler(capture)
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_span = "00f067aa0ba902b7"
    try:
        with TestClient(app) as client:
            live = client.get(
                "/health/live", headers={"traceparent": f"00-{trace_id}-{parent_span}-01"},
            )
            assert live.status_code == 200
            assert live.json()["status"] == "alive"
            assert live.json()["scope"] == "this_api_worker"
            returned_trace = live.headers["traceparent"]
            assert re.fullmatch(rf"00-{trace_id}-[0-9a-f]{{16}}-01", returned_trace)
            assert live.headers["x-trace-id"] == trace_id

            ready = client.get("/health/ready")
            assert ready.status_code == 200, ready.text
            assert ready.json()["database"]["ready"] is True
            assert ready.json()["database"]["backend"] == "sqlite"
            assert ready.json()["storage"]["ready"] is True
            assert ready.json()["storage"]["backend"] == "local"

            health = client.get("/health")
            assert health.status_code == 200
            contract = health.json()["observability"]
            assert contract["structured_logs"]["format"] == "json_lines"
            assert contract["metrics"]["scope"] == "per_process"
            assert contract["tracing"]["propagation"] == "w3c_trace_context"
            assert contract["tracing"]["collector_export"] is False
            assert "best-effort" in contract["guarantee"]

            metrics = client.get("/metrics")
            assert metrics.status_code == 200
            assert "text/plain" in metrics.headers["content-type"]
            assert 'odk_observability_scope_info{metrics="per_process"' in metrics.text
            assert 'odk_http_requests_total{method="GET",route="/health/live",status="200"}' in metrics.text
            assert 'odk_http_request_duration_seconds_count{method="GET",route="/health/ready"}' in metrics.text

            # A real bad storage configuration must make readiness fail, not stay green.
            monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
            monkeypatch.delenv("ODK_S3_BUCKET", raising=False)
            not_ready = client.get("/health/ready")
            assert not_ready.status_code == 503
            assert not_ready.json()["status"] == "not_ready"
            assert not_ready.json()["storage"]["ready"] is False
            assert not_ready.json()["storage"]["error_type"] == "StorageError"

        rows = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        request_span = next(
            row for row in rows if row.get("event") == "http.request" and row.get("path") == "/health/live"
        )
        assert request_span["trace_id"] == trace_id
        assert request_span["parent_span_id"] == parent_span
        assert re.fullmatch(r"[0-9a-f]{16}", request_span["span_id"])
        assert request_span["method"] == "GET"
        assert request_span["route"] == "/health/live"
        assert request_span["status_code"] == 200
        assert request_span["duration_ms"] >= 0
    finally:
        LOGGER.removeHandler(capture)
        capture.close()
        db_module._engine = None
        db_module._SessionLocal = None
