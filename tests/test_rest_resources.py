"""REST resources missing from the original Hub surface.

Requests use FastAPI's real validation stack and an isolated SQLite database. The
payloads are persisted and read back; no router, ORM object, or authorization helper
is mocked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'rest.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-rest-resources")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None
    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    db_module._engine = None
    db_module._SessionLocal = None


def _register(client, email, *, organization_name=""):
    response = client.post("/auth/register", json={
        "email": email, "password": "longenough1",
        "organization_name": organization_name,
    })
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _project(client, headers, organization_id, *, restricted=False, name="Site 14"):
    response = client.post(
        f"/organizations/{organization_id}/projects", headers=headers,
        json={"name": name, "client": "Municipal Works", "restricted": restricted},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def project(client):
    headers = _register(client, "owner@rest.example.com", organization_name="Survey Co")
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    row = _project(client, headers, organization_id)
    return headers, organization_id, row["id"]


def _measurement_payload():
    return {
        "kind": "area",
        "value": 48.0,
        "unit": "m2",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500000, 2000000], [500008, 2000000], [500008, 1999994],
                [500000, 1999994], [500000, 2000000],
            ]],
        },
        "crs_epsg": 32643,
        "source_ref": "survey-2026-08-15/orthomosaic.tif#module-A-3",
        "method": "polygon area on the source raster's projected CRS",
    }


class TestRestDefects:
    def test_model_finding_and_human_review_round_trip(self, client, project):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "thermal_anomaly",
            "severity": "high",
            "description": "Module-level temperature deviation candidate",
            "geometry": {"type": "Point", "coordinates": [500004.5, 1999997.5]},
            "crs_epsg": 32643,
            "source": "model",
            "model_key": "solar_det",
            "model_sha256": "a" * 64,
            "confidence": 0.82,
        })
        assert created.status_code == 201, created.text
        defect_id = created.json()["id"]
        assert created.json()["review_state"] == "unreviewed"

        reviewed = client.post(f"/defects/{defect_id}/review", headers=headers, json={
            "decision": "accepted", "note": "Confirmed against radiometric frame",
        })
        assert reviewed.status_code == 200
        listed = client.get(f"/projects/{project_id}/defects", headers=headers).json()
        assert listed[0]["model_sha256"] == "a" * 64
        assert listed[0]["review_state"] == "accepted"


class TestRestMeasurements:
    def test_georeferenced_measurement_is_persisted_and_read_back(self, client, project):
        headers, _, project_id = project
        created = client.post(
            f"/projects/{project_id}/measurements", headers=headers,
            json=_measurement_payload(),
        )
        assert created.status_code == 201, created.text
        measurement = created.json()
        assert measurement["value"] == pytest.approx(48.0)
        assert measurement["crs_epsg"] == 32643
        assert measurement["geometry"]["type"] == "Polygon"
        fetched = client.get(f"/measurements/{measurement['id']}", headers=headers)
        assert fetched.json() == measurement

    def test_metric_measurement_in_degrees_is_refused(self, client, project):
        headers, _, project_id = project
        payload = _measurement_payload()
        payload["crs_epsg"] = 4326
        response = client.post(
            f"/projects/{project_id}/measurements", headers=headers, json=payload,
        )
        assert response.status_code == 422
        assert "projected CRS" in response.json()["detail"]


class TestRestReports:
    def test_report_snapshots_real_project_records(self, client, project):
        headers, _, project_id = project
        measurement = client.post(
            f"/projects/{project_id}/measurements", headers=headers,
            json=_measurement_payload(),
        ).json()
        defect = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "corrosion", "severity": "medium", "source": "human",
            "geometry": {"type": "Point", "coordinates": [500002, 1999998]},
            "crs_epsg": 32643,
        }).json()
        client.post(f"/defects/{defect['id']}/review", headers=headers,
                    json={"decision": "accepted", "note": "Field verified"})

        response = client.post(
            f"/projects/{project_id}/reports", headers=headers,
            json={"title": "Site 14 inspection"},
        )
        assert response.status_code == 201, response.text
        report = response.json()
        assert report["status"] == "complete"
        assert report["format"] == "structured_json"
        assert report["payload"]["findings"][0]["review_state"] == "accepted"
        assert report["payload"]["measurements"][0]["id"] == measurement["id"]
        fetched = client.get(f"/reports/{report['id']}", headers=headers).json()
        assert fetched == report

    def test_empty_report_says_defect_run_absent_not_zero(self, client, project):
        headers, _, project_id = project
        report = client.post(
            f"/projects/{project_id}/reports", headers=headers,
            json={"title": "Before inspection"},
        ).json()
        assert report["payload"]["defect_evidence_status"] == "absent"
        assert "not evidence that zero defects" in report["payload"]["defect_note"]


class TestRestAIJobs:
    def test_inference_request_is_persistent_without_claiming_a_result(self, client, project):
        headers, _, project_id = project
        response = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
            "task": "segmentation",
            "model_key": "solar_det",
            "input_ref": "dataset:17/frame:DJI_0042.JPG",
            "parameters": {"score_threshold": 0.4},
        })
        assert response.status_code == 201, response.text
        job = response.json()
        assert job["status"] == "pending_worker"
        assert "No result is claimed" in job["note"]
        assert job["parameters"] == {"score_threshold": 0.4}
        assert client.get(f"/ai-jobs/{job['id']}", headers=headers).json() == job
        assert client.get(f"/projects/{project_id}/ai-jobs", headers=headers).json() == [job]

        cancelled = client.post(f"/ai-jobs/{job['id']}/cancel", headers=headers)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


class TestRestContainment:
    def test_restricted_project_resources_are_hidden_from_organization_member(
        self, client,
    ):
        owner = _register(client, "owner@restricted.example.com", organization_name="Private Co")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]
        project_id = _project(
            client, owner, organization_id, restricted=True, name="Confidential site",
        )["id"]
        created_measurement = client.post(
            f"/projects/{project_id}/measurements", headers=owner,
            json=_measurement_payload(),
        ).json()
        created_report = client.post(
            f"/projects/{project_id}/reports", headers=owner, json={"title": "Private"},
        ).json()
        created_job = client.post(f"/projects/{project_id}/ai-jobs", headers=owner, json={
            "task": "detection", "model_key": "corrosion_det", "input_ref": "dataset:9",
        }).json()

        _register(client, "engineer@restricted.example.com", organization_name="Personal")
        invited = client.post(
            f"/organizations/{organization_id}/members", headers=owner,
            json={"email": "engineer@restricted.example.com", "role": "engineer"},
        )
        assert invited.status_code == 201
        login = client.post("/auth/login", json={
            "email": "engineer@restricted.example.com", "password": "longenough1",
        }).json()
        member = {"Authorization": f"Bearer {login['access_token']}"}

        assert client.get(f"/projects/{project_id}/defects", headers=member).status_code == 404
        assert client.get(
            f"/measurements/{created_measurement['id']}", headers=member,
        ).status_code == 404
        assert client.get(f"/reports/{created_report['id']}", headers=member).status_code == 404
        assert client.get(f"/ai-jobs/{created_job['id']}", headers=member).status_code == 404

    def test_openapi_documents_every_new_resource(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/projects/{project_id}/defects",
            "/projects/{project_id}/measurements",
            "/measurements/{measurement_id}",
            "/projects/{project_id}/reports",
            "/reports/{report_id}",
            "/projects/{project_id}/ai-jobs",
            "/ai-jobs/{job_id}",
        ):
            assert path in paths
