"""The Python SDK against the real API over HTTP, not a mock that agrees with it.

``wait_for_job`` once treated "completed" as finished while the server reports
"done", so a job that finished in seconds hung the SDK until its timeout. The previous
SDK test passed because its mock server also said "completed".
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("email_validator")

from hub_api_helpers import FIXTURE_IMAGE, configure, reset_engine  # noqa: E402
from live_api_server import live_api  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    with live_api() as base:
        yield base
    reset_engine()


def _session(base: str):
    from sdk import OpenDroneKitClient

    anonymous = OpenDroneKitClient(base)
    token = anonymous.request("POST", "/auth/register", payload={
        "email": "sdk@example.com", "password": "longenough1", "organization_name": "SDK Co",
    })["access_token"]
    client = OpenDroneKitClient(base, token=token, timeout_s=30)
    org = client.request("GET", "/organizations")[0]["id"]
    return client, org


def test_wait_for_job_returns_when_the_server_says_done(api):
    client, org = _session(api)
    project = client.create_project(org, name="SDK site")
    client.request("POST", f"/projects/{project['id']}/defects",
                   payload={"category": "crack", "severity": "low"})
    job = client.submit_job(project["id"], kind="report", options={"include_unreviewed": True})

    finished = client.wait_for_job(job.id, timeout_s=60, poll_interval_s=0.2)
    assert finished.status == "done"
    pdf = next(ref for ref in finished.payload["artifacts"] if ref.endswith(".pdf"))
    assert client.download_artifact(job.id, pdf).startswith(b"%PDF-")


def test_a_failed_job_is_terminal_too(api):
    client, org = _session(api)
    project = client.create_project(org, name="Empty")
    dataset = client.request("POST", f"/projects/{project['id']}/datasets", payload={"name": "d"})
    job = client.submit_job(project["id"], kind="reconstruction", dataset_id=dataset["id"])
    assert client.wait_for_job(job.id, timeout_s=60, poll_interval_s=0.2).status == "failed"


def test_ai_jobs_run_to_completion_through_the_sdk(api, tmp_path):
    client, org = _session(api)
    project = client.create_project(org, name="AI site")
    dataset = client.request("POST", f"/projects/{project['id']}/datasets", payload={"name": "d"})
    # The fixture image goes where an upload would put it; the upload protocol itself
    # is covered by tests/test_uploads.py.
    target = tmp_path / "storage" / "datasets" / str(dataset["id"])
    target.mkdir(parents=True)
    target.joinpath(FIXTURE_IMAGE.name).write_bytes(FIXTURE_IMAGE.read_bytes())

    job = client.submit_ai_job(project["id"], task="detection", model_key="heuristic",
                               input_ref=f"dataset:{dataset['id']}")
    finished = client.wait_for_ai_job(job.id, timeout_s=60, poll_interval_s=0.2)
    assert finished.status == "done"
    assert finished.payload["result"]["model_used"] == "heuristic"
