"""Annotation geometry, metadata and persistence tests against real JSON storage."""

from __future__ import annotations

import json

import pytest

from core.annotations import create_annotation, list_annotations, update_annotation


GEOMETRIES = {
    "point": {"type": "Point", "coordinates": [77.595, 12.975]},
    "line": {"type": "LineString", "coordinates": [[77.595, 12.975], [77.596, 12.976]]},
    "polygon": {"type": "Polygon", "coordinates": [[
        [77.595, 12.975], [77.596, 12.975], [77.596, 12.976], [77.595, 12.975],
    ]]},
    "rectangle": {"type": "Polygon", "coordinates": [[
        [77.595, 12.975], [77.596, 12.975], [77.596, 12.976],
        [77.595, 12.976], [77.595, 12.975],
    ]]},
    "circle": {"type": "Point", "coordinates": [77.595, 12.975], "radius_m": 3.5},
    "freehand": {"type": "LineString", "coordinates": [
        [77.595, 12.975], [77.5952, 12.9753], [77.5958, 12.9757],
    ]},
    "text": {"type": "Point", "coordinates": [77.595, 12.975]},
}


class TestAnnotationGeometry:
    @pytest.mark.parametrize("annotation_type", sorted(GEOMETRIES))
    def test_every_required_geometry_round_trips_through_real_storage(
        self, tmp_path, annotation_type,
    ):
        row = create_annotation(
            tmp_path, "project-1", "map", "orthomosaic-2026-08-15",
            annotation_type, GEOMETRIES[annotation_type], f"{annotation_type} note",
            severity="high", status="open", created_by="inspector@example.com",
        )

        stored = list_annotations(tmp_path, project_id="project-1")
        assert stored[0].id == row.id
        assert stored[0].annotation_type == annotation_type
        assert stored[0].geometry == GEOMETRIES[annotation_type]
        assert stored[0].severity == "high"
        assert stored[0].status == "open"
        payload = json.loads(
            (tmp_path / "analysis" / "annotations" / "annotations.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload[0]["geometry"] == GEOMETRIES[annotation_type]


class TestAnnotationValidation:
    def test_circle_without_a_measured_radius_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="radius_m"):
            create_annotation(
                tmp_path, "p", "map", "ortho", "circle",
                {"type": "Point", "coordinates": [77.5, 12.9]}, "circle",
                severity="medium", status="open",
            )

    def test_rectangle_that_is_not_axis_aligned_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="axis-aligned"):
            create_annotation(
                tmp_path, "p", "map", "ortho", "rectangle",
                {"type": "Polygon", "coordinates": [[
                    [0, 0], [2, 1], [1, 3], [-1, 2], [0, 0],
                ]]}, "skewed", severity="medium", status="open",
            )

    def test_status_and_severity_are_validated_on_update(self, tmp_path):
        row = create_annotation(
            tmp_path, "p", "map", "ortho", "point", GEOMETRIES["point"], "pier",
            severity="medium", status="open",
        )
        updated = update_annotation(tmp_path, row.id, {
            "severity": "critical", "status": "resolved",
        })
        assert updated and updated.severity == "critical" and updated.status == "resolved"
        with pytest.raises(ValueError, match="status"):
            update_annotation(tmp_path, row.id, {"status": "probably"})


class TestAnnotationApi:
    def test_all_shapes_persist_update_and_delete_through_real_http(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("email_validator")
        from fastapi.testclient import TestClient
        import services.api.db as db_module

        monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'annotations.db'}")
        monkeypatch.setenv("ODK_SECRET_KEY", "annotation-test-secret-long-enough")
        db_module._engine = None
        db_module._SessionLocal = None
        from services.api.main import app

        try:
            with TestClient(app) as client:
                token = client.post("/auth/register", json={
                    "email": "annotations@example.com", "password": "longenough1",
                    "organization_name": "Annotation Co",
                }).json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
                project_id = client.post(
                    f"/organizations/{organization_id}/projects", headers=headers,
                    json={"name": "Mapped site"},
                ).json()["id"]

                created = []
                for annotation_type, geometry in GEOMETRIES.items():
                    response = client.post(
                        f"/projects/{project_id}/annotations", headers=headers,
                        json={
                            "source_type": "map", "source_id": "orthomosaic:T1",
                            "annotation_type": annotation_type, "geometry": geometry,
                            "crs_epsg": 4326, "label": f"{annotation_type} evidence",
                            "severity": "high", "status": "open",
                        },
                    )
                    assert response.status_code == 201, response.text
                    created.append(response.json())

                listed = client.get(
                    f"/projects/{project_id}/annotations", headers=headers,
                ).json()
                assert {row["annotation_type"] for row in listed} == set(GEOMETRIES)
                assert all(row["severity"] == "high" and row["status"] == "open" for row in listed)

                annotation_id = created[0]["id"]
                changed = client.patch(
                    f"/annotations/{annotation_id}", headers=headers,
                    json={"severity": "critical", "status": "resolved"},
                )
                assert changed.status_code == 200
                assert changed.json()["severity"] == "critical"
                assert changed.json()["status"] == "resolved"
                assert client.delete(
                    f"/annotations/{annotation_id}", headers=headers,
                ).status_code == 204
                assert client.get(f"/annotations/{annotation_id}", headers=headers).status_code == 404
        finally:
            db_module._engine = None
            db_module._SessionLocal = None
