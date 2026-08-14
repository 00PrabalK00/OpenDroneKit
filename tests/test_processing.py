"""Processing jobs: submission, polling, cancellation and honest failure.

The interesting cases are the ones where work does not succeed. A job that dies must
say so; a job kind with no worker must be refused rather than reported complete with
nothing behind it; and a reconstruction over an empty dataset must fail with a reason
naming the missing input.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def project(client):
    response = client.post("/auth/register", json={
        "email": "eng@example.com", "password": "longenough1",
        "organization_name": "Processing Co",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                             json={"name": "Recon site"}).json()["id"]
    dataset_id = client.post(f"/projects/{project_id}/datasets", headers=headers,
                             json={"name": "Imagery"}).json()["id"]
    return headers, organization_id, project_id, dataset_id


def wait_for_terminal(client, headers, job_id, timeout=25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.15)
    return client.get(f"/jobs/{job_id}", headers=headers).json()


class TestSubmission:
    def test_a_job_is_accepted_with_202(self, client, project):
        """202, not 201: the work has not happened yet."""
        headers, _, project_id, dataset_id = project
        response = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id})
        assert response.status_code == 202
        assert response.json()["status"] in {"queued", "running", "failed"}

    def test_an_unknown_job_kind_is_refused_with_the_list(self, client, project):
        headers, _, project_id, dataset_id = project
        response = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "telepathy", "dataset_id": dataset_id})
        assert response.status_code == 422
        assert "reconstruction" in response.json()["detail"]

    def test_a_dataset_from_another_project_is_refused(self, client, project):
        headers, organization_id, _, dataset_id = project
        other_project = client.post(f"/organizations/{organization_id}/projects",
                                    headers=headers, json={"name": "Other"}).json()["id"]
        response = client.post(f"/projects/{other_project}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id})
        assert response.status_code == 404

    def test_a_viewer_cannot_submit_a_job(self, client, project):
        headers, organization_id, project_id, dataset_id = project
        client.post("/auth/register", json={
            "email": "look@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "look@example.com", "role": "viewer"})
        viewer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "look@example.com", "password": "longenough1"}).json()["access_token"]}

        response = client.post(f"/projects/{project_id}/jobs", headers=viewer, json={
            "kind": "reconstruction", "dataset_id": dataset_id})
        assert response.status_code == 403


class TestFailureReporting:
    def test_reconstruction_over_an_empty_dataset_fails_with_a_reason(self, client, project):
        """The message must name the missing input, not just say something went wrong."""
        headers, _, project_id, dataset_id = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]

        job = wait_for_terminal(client, headers, job_id)
        assert job["status"] == "failed"
        assert "no uploaded imagery" in job["error"].lower()

    def test_a_failed_job_never_reports_artifacts(self, client, project):
        headers, _, project_id, dataset_id = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]

        job = wait_for_terminal(client, headers, job_id)
        assert job["status"] == "failed"
        assert job["artifacts"] == []
        assert job["crs_epsg"] is None

    def test_a_kind_without_a_worker_fails_rather_than_pretending(self, client, project):
        """A stage that is not built must not report success with nothing behind it."""
        headers, _, project_id, _ = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "report"}).json()["id"]

        job = wait_for_terminal(client, headers, job_id)
        assert job["status"] == "failed"
        assert "no worker" in job["error"].lower()

    def test_a_failed_job_keeps_a_traceback(self, client, project):
        headers, _, project_id, dataset_id = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]
        wait_for_terminal(client, headers, job_id)

        log = client.get(f"/jobs/{job_id}/log", headers=headers).json()
        assert log["status"] == "failed"
        assert "Traceback" in log["log"]


class TestCancellation:
    def test_cancelling_a_finished_job_is_refused(self, client, project):
        headers, _, project_id, dataset_id = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]
        wait_for_terminal(client, headers, job_id)

        response = client.post(f"/jobs/{job_id}/cancel", headers=headers)
        assert response.status_code == 409

    def test_cancelling_an_unknown_job_is_404(self, client, project):
        headers, _, _, _ = project
        assert client.post("/jobs/99999/cancel", headers=headers).status_code == 404


class TestVisibility:
    def test_jobs_are_listed_newest_first(self, client, project):
        headers, _, project_id, dataset_id = project
        first = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]
        second = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]

        listed = client.get(f"/projects/{project_id}/jobs", headers=headers).json()
        ids = [job["id"] for job in listed]
        assert ids.index(second) < ids.index(first)

    def test_a_non_member_cannot_see_a_job(self, client, project):
        headers, _, project_id, dataset_id = project
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]

        outsider_token = client.post("/auth/register", json={
            "email": "outsider@example.com", "password": "longenough1",
            "organization_name": "Elsewhere"}).json()["access_token"]
        outsider = {"Authorization": f"Bearer {outsider_token}"}

        assert client.get(f"/jobs/{job_id}", headers=outsider).status_code == 404

    def test_the_engine_is_recorded_on_the_job(self, client, project):
        """A result must always be traceable to what produced it."""
        headers, _, project_id, dataset_id = project
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id, "engine": "colmap"}).json()
        assert job["engine"] == "colmap"
