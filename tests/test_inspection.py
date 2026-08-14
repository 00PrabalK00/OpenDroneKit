"""Defects, human review, and measurement refusal.

The property worth protecting: an AI prediction is never stored as verified. The
model's claim and the inspector's decision live in separate fields, so a report can
always say which is which, and reviewing a finding never erases what the model
originally asserted.
"""

from __future__ import annotations

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
        "email": "inspector@example.com", "password": "longenough1",
        "organization_name": "Inspection Co",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                             json={"name": "Bridge 7"}).json()["id"]
    return headers, organization_id, project_id


MODEL_DEFECT = {
    "category": "crack", "severity": "high", "source": "model",
    "model_key": "crack_segmentation",
    "model_sha256": "cda3b64828d7352a63bc2e0815c40d9d09d9668a5b358af7134485de87f157a8",
    "confidence": 0.87, "area_m2": 0.42, "length_m": 1.8,
    "longitude": -81.7505, "latitude": 41.3042, "crs_epsg": 4326,
}


class TestDefectLibrary:
    def test_default_categories_are_offered_without_being_mandatory(self, client):
        payload = client.get("/defect-categories").json()
        assert "crack" in payload["categories"]
        assert "spalling" in payload["categories"]
        assert "only the default" in payload["note"]

    def test_a_custom_category_is_accepted(self, client, project):
        """Organisations define their own defects; the list is not a whitelist."""
        headers, _, project_id = project
        response = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "cathodic_protection_failure", "severity": "high"})
        assert response.status_code == 201
        assert response.json()["category"] == "cathodic_protection_failure"

    def test_an_unknown_severity_is_refused(self, client, project):
        headers, _, project_id = project
        response = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "crack", "severity": "apocalyptic"})
        assert response.status_code == 422


class TestProvenance:
    def test_a_model_defect_must_name_its_model(self, client, project):
        """An unattributable model finding cannot be audited later."""
        headers, _, project_id = project
        response = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "crack", "source": "model", "confidence": 0.9})
        assert response.status_code == 422
        assert "model_key" in response.json()["detail"]

    def test_a_model_prediction_is_never_born_accepted(self, client, project):
        headers, _, project_id = project
        defect = client.post(f"/projects/{project_id}/defects", headers=headers,
                             json=MODEL_DEFECT).json()
        assert defect["review_state"] == "unreviewed"
        assert defect["reviewed_by"] is None

    def test_the_model_claim_survives_review(self, client, project):
        """Reviewing records a human decision; it must not erase what the model said."""
        headers, _, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]

        reviewed = client.post(f"/defects/{defect_id}/review", headers=headers, json={
            "decision": "reclassified", "category": "spalling", "note": "actually spall"}).json()

        assert reviewed["category"] == "spalling"
        assert reviewed["review_state"] == "reclassified"
        assert reviewed["model_key"] == "crack_segmentation"
        assert reviewed["confidence"] == pytest.approx(0.87)
        assert reviewed["model_sha256"].startswith("cda3b648")


class TestReview:
    def test_accepting_records_the_reviewer(self, client, project):
        headers, _, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]
        reviewed = client.post(f"/defects/{defect_id}/review", headers=headers,
                               json={"decision": "accepted"}).json()
        assert reviewed["review_state"] == "accepted"
        assert reviewed["reviewed_by"] is not None
        assert reviewed["reviewed_at"] is not None

    def test_rejecting_keeps_the_record(self, client, project):
        """A rejected finding stays visible; deleting it would hide a model's error."""
        headers, _, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]
        client.post(f"/defects/{defect_id}/review", headers=headers,
                    json={"decision": "rejected", "note": "shadow, not a crack"})

        listed = client.get(f"/projects/{project_id}/defects", headers=headers).json()
        assert any(d["id"] == defect_id and d["review_state"] == "rejected" for d in listed)

    def test_reclassifying_without_a_category_is_refused(self, client, project):
        headers, _, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]
        response = client.post(f"/defects/{defect_id}/review", headers=headers,
                               json={"decision": "reclassified"})
        assert response.status_code == 422

    def test_marking_back_to_unreviewed_is_refused(self, client, project):
        headers, _, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]
        response = client.post(f"/defects/{defect_id}/review", headers=headers,
                               json={"decision": "unreviewed"})
        assert response.status_code == 422

    def test_a_viewer_cannot_review(self, client, project):
        headers, organization_id, project_id = project
        defect_id = client.post(f"/projects/{project_id}/defects", headers=headers,
                                json=MODEL_DEFECT).json()["id"]
        client.post("/auth/register", json={
            "email": "onlooker@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "onlooker@example.com", "role": "viewer"})
        viewer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "onlooker@example.com", "password": "longenough1"}).json()["access_token"]}

        assert client.post(f"/defects/{defect_id}/review", headers=viewer,
                           json={"decision": "accepted"}).status_code == 403


class TestSummary:
    def test_confirmed_and_claimed_are_counted_separately(self, client, project):
        headers, _, project_id = project
        first = client.post(f"/projects/{project_id}/defects", headers=headers,
                            json=MODEL_DEFECT).json()["id"]
        client.post(f"/projects/{project_id}/defects", headers=headers, json=MODEL_DEFECT)
        client.post(f"/defects/{first}/review", headers=headers, json={"decision": "accepted"})

        summary = client.get(f"/projects/{project_id}/defects/summary", headers=headers).json()
        assert summary["total"] == 2
        assert summary["human_confirmed"] == 1
        assert summary["awaiting_review"] == 1

    def test_unmeasured_defects_are_declared_not_counted_as_zero(self, client, project):
        """A total over partly unmeasured defects is not the total extent of damage."""
        headers, _, project_id = project
        client.post(f"/projects/{project_id}/defects", headers=headers, json=MODEL_DEFECT)
        client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "corrosion", "severity": "medium"})  # no area or length

        summary = client.get(f"/projects/{project_id}/defects/summary", headers=headers).json()
        assert summary["unmeasured_count"] == 1
        # The caveat must be stated, not left to be inferred from the counts.
        assert summary["note"], "a partly unmeasured total must carry a caveat"
        assert "measured" in summary["note"]
        # The measured defect's area only; the unmeasured one contributes nothing
        # rather than being counted as zero area.
        assert summary["total_area_m2"] == pytest.approx(0.42)

    def test_filtering_by_review_state(self, client, project):
        headers, _, project_id = project
        first = client.post(f"/projects/{project_id}/defects", headers=headers,
                            json=MODEL_DEFECT).json()["id"]
        client.post(f"/projects/{project_id}/defects", headers=headers, json=MODEL_DEFECT)
        client.post(f"/defects/{first}/review", headers=headers, json={"decision": "accepted"})

        accepted = client.get(f"/projects/{project_id}/defects?review_state=accepted",
                              headers=headers).json()
        assert len(accepted) == 1
        assert accepted[0]["id"] == first


class TestMeasurementRefusal:
    def test_volume_on_a_job_with_no_dsm_is_refused_with_a_reason(self, client, project):
        """A volume from an unreferenced surface would be pixel units dressed as m3."""
        headers, _, project_id = project
        dataset_id = client.post(f"/projects/{project_id}/datasets", headers=headers,
                                 json={"name": "Empty"}).json()["id"]
        job_id = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "reconstruction", "dataset_id": dataset_id}).json()["id"]

        import time
        for _ in range(150):
            if client.get(f"/jobs/{job_id}", headers=headers).json()["status"] != "queued":
                break
            time.sleep(0.1)

        response = client.post("/measurements/volume", headers=headers, json={"job_id": job_id})
        # Either the job has not succeeded, or it produced no DSM. Both are refusals
        # with an explanation, never a number.
        assert response.status_code in {409, 422}
        assert "detail" in response.json()

    def test_volume_on_an_unknown_job_is_404(self, client, project):
        headers, _, _ = project
        assert client.post("/measurements/volume", headers=headers,
                           json={"job_id": 99999}).status_code == 404
