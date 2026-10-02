"""The deployment's moving parts, together and for real: PostGIS, Redis, a Celery worker
process and an S3 gateway.

Every other test of these runs one piece at a time -- the broker with no worker, the
worker in-process on Kombu's memory transport, the database without the queue. This one
is the compose stack's shape on one machine: the API (served in this process through
its real routers) records a job in PostGIS and sends its id to Redis; a separate
`python -m services.worker.tasks` process -- the worker container's command -- takes it
off the queue and runs it against the same database and storage; the API then reports
it done, and its readiness probe checks the live database and object store.

It runs only when all three services are given, and skips otherwise: a stack test that
quietly ran on SQLite and an in-memory queue would prove nothing about the stack.

    ODK_TEST_POSTGIS=postgresql+psycopg://postgres:odk@127.0.0.1:55432/odk
    ODK_BROKER_URL=redis://127.0.0.1:56379/0
    ODK_TEST_S3_ENDPOINT=http://127.0.0.1:18333  (+ ODK_S3_ACCESS_KEY / ODK_S3_SECRET_KEY)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("celery")
pytest.importorskip("redis")
boto3 = pytest.importorskip("boto3")

from hub_api_helpers import REPO, project_with_dataset, register, reset_engine, upload, wait  # noqa: E402

DSN = os.environ.get("ODK_TEST_POSTGIS", "")
S3 = os.environ.get("ODK_TEST_S3_ENDPOINT", "")


def _broker_live() -> bool:
    try:
        from services.worker.celery_app import broker_reachable

        return bool(broker_reachable(timeout_s=0.5).get("reachable"))
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not (DSN and S3 and os.environ.get("ODK_BROKER_URL") and _broker_live()),
    reason="needs ODK_TEST_POSTGIS, a reachable ODK_BROKER_URL and ODK_TEST_S3_ENDPOINT",
)


@pytest.fixture
def stack(tmp_path, monkeypatch):
    bucket = f"odk-stack-{uuid.uuid4().hex[:10]}"
    client = boto3.client("s3", endpoint_url=S3, region_name="us-east-1",
                          aws_access_key_id=os.environ.get("ODK_S3_ACCESS_KEY") or None,
                          aws_secret_access_key=os.environ.get("ODK_S3_SECRET_KEY") or None)
    client.create_bucket(Bucket=bucket)
    env = {
        "ODK_DATABASE_URL": DSN,
        "ODK_SECRET_KEY": "stack-test-secret-long-enough-for-hmac-sha256!",
        "ODK_STORAGE_PATH": str(tmp_path / "storage"),
        "ODK_JOB_BACKEND": "celery",
        "ODK_STORAGE_BACKEND": "s3",
        "ODK_S3_ENDPOINT": S3,
        "ODK_S3_BUCKET": bucket,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_engine()
    log = tmp_path / "worker.log"
    with log.open("w", encoding="utf-8") as sink:
        worker = subprocess.Popen(
            [sys.executable, "-m", "services.worker.tasks"], cwd=str(REPO),
            env={**os.environ, **env, "ODK_WORKER_CONCURRENCY": "1"},
            stdout=sink, stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and "ready" not in log.read_text(encoding="utf-8", errors="replace"):
            assert worker.poll() is None, log.read_text(encoding="utf-8", errors="replace")
            time.sleep(0.5)
        yield {"worker": worker, "log": log, "bucket": bucket}
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
        reset_engine()
        for item in client.list_objects_v2(Bucket=bucket).get("Contents", []):
            client.delete_object(Bucket=bucket, Key=item["Key"])
        client.delete_bucket(Bucket=bucket)


def test_a_job_goes_through_postgis_redis_and_a_real_worker_process(stack) -> None:
    from fastapi.testclient import TestClient

    from services.api.main import app

    tag = uuid.uuid4().hex[:8]
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200, ready.text
        headers, org = register(client, email=f"stack-{tag}@example.com", organization=f"Stack {tag}")
        project_id, dataset_id = project_with_dataset(client, headers, org, name=f"Site {tag}")
        upload(client, headers, dataset_id)
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={"kind": "report"})
        assert job.status_code == 202, job.text
        done = wait(client, headers, f"/jobs/{job.json()['id']}", timeout=240)
        health = client.get("/health").json()
    log = stack["log"].read_text(encoding="utf-8", errors="replace")
    assert done["status"] == "done", (done, log[-3000:])
    assert any(ref.endswith(".pdf") for ref in done["artifacts"]), done
    # The worker process ran it, not a thread in this one.
    assert "odk.api.run_job" in log and "succeeded" in log, log[-3000:]
    assert stack["worker"].poll() is None, "the worker died"
    # Health describes the live pieces, not defaults.
    text = str(health)
    assert "postgres" in text.lower(), health
    assert stack["bucket"] in text, health
