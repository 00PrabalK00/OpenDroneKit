"""Job dispatch to Celery, the worker image's command, and where a bare server writes.

* ODK_JOB_BACKEND=celery: the API enqueues the job id and a Celery worker -- here an
  in-process one on Kombu's in-memory transport, standing in for Redis -- runs it
  through the same code path, against the same database and storage.
* The worker image used to sleep forever printing "queue consumer not implemented".
* A server started without ODK_DATABASE_URL / ODK_STORAGE_PATH used to write
  ./opendronekit.db and ./odk_storage into whatever directory it was launched from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

from hub_api_helpers import REPO, configure, project_with_dataset, register, reset_engine, upload, wait  # noqa: E402


def test_a_celery_backed_api_runs_jobs_on_the_worker(tmp_path, monkeypatch):
    pytest.importorskip("celery")
    from celery.contrib.testing.worker import start_worker

    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("ODK_JOB_BACKEND", "celery")
    monkeypatch.setenv("ODK_BROKER_URL", "memory://")
    monkeypatch.setenv("ODK_RESULT_BACKEND", "cache+memory://")

    from services.worker.celery_app import build_app
    from services.worker.tasks import register as register_tasks

    worker_app = build_app("odk-test-worker")
    register_tasks(worker_app)
    from services.api.main import app

    with TestClient(app) as client, start_worker(worker_app, perform_ping_check=False,
                                                 pool="solo", loglevel="WARNING"):
        headers, org = register(client)
        project_id, _ = project_with_dataset(client, headers, org)
        job = client.post(f"/projects/{project_id}/jobs", headers=headers,
                          json={"kind": "report"}).json()
        done = wait(client, headers, f"/jobs/{job['id']}", timeout=60)
    reset_engine()
    assert done["status"] == "done", done
    assert any(ref.endswith(".pdf") for ref in done["artifacts"])


def test_a_job_is_run_once_even_if_delivered_twice(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("ODK_JOB_BACKEND", "celery")
    import services.worker.tasks as tasks

    sent: list[int] = []
    monkeypatch.setattr(tasks, "enqueue_api_job", lambda job_id, app=None: sent.append(job_id))
    from services.api.main import app

    with TestClient(app) as client:
        headers, org = register(client)
        project_id, _ = project_with_dataset(client, headers, org)
        job = client.post(f"/projects/{project_id}/jobs", headers=headers,
                          json={"kind": "report"}).json()
        assert sent == [job["id"]]
        assert client.get(f"/jobs/{job['id']}", headers=headers).json()["status"] == "queued"
        first = tasks.run_api_job(job["id"])
        second = tasks.run_api_job(job["id"])  # a redelivered message
        reports = client.get(f"/projects/{project_id}/reports", headers=headers).json()
    reset_engine()
    assert first["status"] == second["status"] == "done"
    assert len(reports) == 1, "a redelivered job must not run a second time"


def test_the_worker_image_runs_a_consumer_not_a_sleep():
    dockerfile = (REPO / "infrastructure" / "docker" / "Dockerfile.worker").read_text(encoding="utf-8")
    command = re.findall(r"^CMD\s+(.+)$", dockerfile, re.M)[-1]
    assert "services.worker.tasks" in command
    assert "not implemented" not in dockerfile


def test_a_bare_server_keeps_its_data_out_of_the_working_directory(tmp_path, monkeypatch):
    for name in ("ODK_DATABASE_URL", "ODK_STORAGE_PATH"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "hub-home"
    cwd = tmp_path / "launched-from"
    cwd.mkdir()
    monkeypatch.setenv("ODK_HUB_HOME", str(home))
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.chdir(cwd)
    reset_engine()
    from services.api.main import app

    with TestClient(app) as client:
        headers, org = register(client)
        _, dataset_id = project_with_dataset(client, headers, org)
        upload(client, headers, dataset_id)
    reset_engine()
    assert list(cwd.iterdir()) == []
    assert (home / "opendronekit.db").is_file()
    assert any((home / "storage" / "datasets" / str(dataset_id)).iterdir())


def test_completing_an_upload_twice_returns_the_first_result(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    from services.api.main import app

    with TestClient(app) as client:
        headers, org = register(client)
        _, dataset_id = project_with_dataset(client, headers, org)
        first = upload(client, headers, dataset_id)
        again = client.post(f"/uploads/{first['upload_id']}/complete", headers=headers)
        datasets = client.get(f"/projects/{_}/datasets", headers=headers).json()
    reset_engine()
    assert again.status_code == 200
    body = again.json()
    assert body["already_completed"] is True
    assert body["sha256"] == first["sha256"] and body["bytes"] == first["bytes"]
    assert datasets[0]["file_count"] == 1, "a retried completion must not count the file twice"


def test_an_existing_working_directory_database_is_still_used(tmp_path, monkeypatch):
    """Upgrading must not hide records kept in the old ./opendronekit.db default."""
    cwd = tmp_path / "install"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(cwd / 'opendronekit.db').as_posix()}")
    reset_engine()
    from services.api.main import app

    with TestClient(app) as client:
        register(client, "legacy@example.com")
    reset_engine()

    monkeypatch.delenv("ODK_DATABASE_URL")
    monkeypatch.setenv("ODK_HUB_HOME", str(tmp_path / "hub-home"))
    with TestClient(app) as client:
        login = client.post("/auth/login", json={
            "email": "legacy@example.com", "password": "longenough1"})
    reset_engine()
    assert login.status_code == 200
    assert not (tmp_path / "hub-home" / "opendronekit.db").exists()
