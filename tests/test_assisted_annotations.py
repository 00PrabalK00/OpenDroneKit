"""Installed-model pre-label and human-review evidence against a real photograph."""

from __future__ import annotations

import hashlib
import pathlib
from pathlib import Path
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient


FIXTURE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"
FIXTURE_SHA256 = "e020561bb21af625a19e820cbd75ce8246505fd4de68dd0ad6a12b57ba28e6ba"


def _rectangle(west: float, north: float, east: float, south: float) -> dict:
    return {"type": "Polygon", "coordinates": [[
        [west, north], [east, north], [east, south], [west, south], [west, north],
    ]]}


def _bounds(annotation: dict) -> tuple[float, float, float, float]:
    ring = annotation["geometry"]["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _detector_weights_present() -> bool:
    """Whether an installed detector's ONNX file is actually on this machine.

    The point of this test is that the PRODUCTION route runs -- real weights, real
    inference, no mock. That makes it unrunnable on a fresh clone, where the weights are
    gitignored as large binaries, and skipping is the only honest answer: mocking the
    detector here would leave a test that passes while checking the opposite of what it
    claims.
    """
    import json

    root = pathlib.Path(__file__).resolve().parents[1] / "models"
    try:
        registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    except OSError:
        return False
    return any(
        (root / entry["path"]).is_file()
        for entry in registry.get("models", {}).values()
        if entry.get("status") == "installed"
        and entry.get("kind", "").startswith("onnx")
        and entry.get("path")
    )


@pytest.mark.skipif(
    not _detector_weights_present(),
    reason="no installed ONNX weights on this machine; this test must not mock the detector",
)
def test_real_model_prelabels_retain_claims_through_every_review_action(tmp_path, monkeypatch):
    """The production ONNX route runs; no detector, file, DB or review call is mocked."""
    assert FIXTURE.is_file()
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256

    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'assisted.db'}")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("ODK_SECRET_KEY", "assisted-annotation-test-secret-long-enough")
    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None
    from services.api.main import app

    try:
        with TestClient(app) as client:
            registered = client.post("/auth/register", json={
                "email": "reviewer@example.com", "password": "longenough1",
                "organization_name": "Review Lab",
            })
            assert registered.status_code == 201, registered.text
            headers = {"Authorization": f"Bearer {registered.json()['access_token']}"}
            organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
            project = client.post(
                f"/organizations/{organization_id}/projects", headers=headers,
                json={"name": "Concrete inspection"},
            )
            assert project.status_code == 201, project.text
            project_id = project.json()["id"]

            catalog = client.get(
                f"/projects/{project_id}/annotations/prelabel/models", headers=headers,
            )
            assert catalog.status_code == 200
            models = {row["model_key"]: row for row in catalog.json()["models"]}
            assert models["structural_multiclass_detector"]["installed"] is True
            assert models["structural_multiclass_detector"]["prelabel_supported"] is True
            assert models["crack_segmentation"]["installed"] is True
            assert models["crack_segmentation"]["prelabel_supported"] is False
            assert "per-region model confidence" in models["crack_segmentation"]["reason"]

            refused = client.post(
                f"/projects/{project_id}/annotations/prelabel", headers=headers,
                data={"model_key": "crack_segmentation"},
                files={"image": (FIXTURE.name, raw, "image/jpeg")},
            )
            assert refused.status_code == 422
            assert "Only structural_multiclass_detector" in refused.json()["detail"]

            response = client.post(
                f"/projects/{project_id}/annotations/prelabel", headers=headers,
                data={"model_key": "structural_multiclass_detector", "severity": "info"},
                files={"image": (FIXTURE.name, raw, "image/jpeg")},
            )
            assert response.status_code == 201, response.text
            batch = response.json()
            assert batch["source"]["sha256"] == FIXTURE_SHA256
            assert batch["source"]["width_px"] == 960
            assert batch["source"]["height_px"] == 720
            assert batch["model"]["model_key"] == "structural_multiclass_detector"
            assert batch["model"]["model_sha256"] == (
                "359817c663f4cb7772f44e4370095c116efb9d0123478bc58aa618884380c445"
            )
            assert batch["model"]["model_used"].startswith("onnx:")
            assert batch["finding_count"] >= 2
            prelabels = batch["prelabels"]
            source_file = tmp_path / "storage" / "objects" / batch["source"]["storage_key"]
            assert source_file.read_bytes() == raw

            for annotation in prelabels:
                assert annotation["origin"] == "model"
                assert annotation["review_action"] == "unreviewed"
                assert annotation["include_in_report"] is False
                assert len(annotation["machine_claims"]) == 1
                claim = annotation["machine_claims"][0]
                assert claim["model_key"] == "structural_multiclass_detector"
                assert claim["model_sha256"] == batch["model"]["model_sha256"]
                assert claim["confidence_kind"] == "per_detection_model_score"
                assert 0.0 < claim["confidence"] <= 1.0
                assert claim["geometry"] == annotation["geometry"]
                assert claim["label"] == annotation["label"]

            first, second = prelabels[:2]
            original_first_claim = first["machine_claims"][0]
            accepted = client.post(
                f"/annotations/{first['id']}/review", headers=headers,
                json={"action": "accept"},
            )
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()["review_action"] == "accepted"
            assert accepted.json()["include_in_report"] is True
            assert accepted.json()["machine_claims"][0] == original_first_claim

            west, north, east, south = _bounds(first)
            edited_geometry = _rectangle(west + 1, north + 1, east - 1, south - 1)
            edited = client.post(
                f"/annotations/{first['id']}/review", headers=headers,
                json={"action": "edit", "geometry": edited_geometry, "note": "Tightened by reviewer"},
            )
            assert edited.status_code == 200, edited.text
            assert edited.json()["review_action"] == "edited"
            assert edited.json()["geometry"] == edited_geometry
            assert edited.json()["machine_claims"][0] == original_first_claim

            reclassified = client.post(
                f"/annotations/{second['id']}/review", headers=headers,
                json={"action": "reclassify", "label": "surface_crack"},
            )
            assert reclassified.status_code == 200, reclassified.text
            assert reclassified.json()["review_action"] == "reclassified"
            assert reclassified.json()["label"] == "surface_crack"
            assert reclassified.json()["machine_claims"][0]["label"] == second["label"]

            b1, b2 = _bounds(edited.json()), _bounds(reclassified.json())
            merged_geometry = _rectangle(
                min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]),
            )
            merged = client.post(
                f"/projects/{project_id}/annotations/merge", headers=headers,
                json={
                    "annotation_ids": [first["id"], second["id"]],
                    "annotation_type": "rectangle", "geometry": merged_geometry,
                    "label": "reviewed crack region", "severity": "medium",
                    "status": "open",
                },
            )
            assert merged.status_code == 201, merged.text
            merged_row = merged.json()
            assert merged_row["review_action"] == "merged"
            assert merged_row["parent_ids"] == [first["id"], second["id"]]
            assert len(merged_row["machine_claims"]) == 2

            mw, mn, me, ms = _bounds(merged_row)
            middle = (mw + me) / 2.0
            split = client.post(
                f"/annotations/{merged_row['id']}/split", headers=headers,
                json={"parts": [
                    {"annotation_type": "rectangle", "geometry": _rectangle(mw, mn, middle, ms),
                     "label": "left crack", "severity": "medium", "status": "open"},
                    {"annotation_type": "rectangle", "geometry": _rectangle(middle, mn, me, ms),
                     "label": "right crack", "severity": "medium", "status": "open"},
                ]},
            )
            assert split.status_code == 201, split.text
            children = split.json()
            assert len(children) == 2
            assert all(row["review_action"] == "split_child" for row in children)
            assert all(row["parent_ids"] == [merged_row["id"]] for row in children)
            assert all(row["machine_claims"] == merged_row["machine_claims"] for row in children)

            parent_after = client.get(f"/annotations/{merged_row['id']}", headers=headers).json()
            assert parent_after["review_action"] == "split_source"
            assert parent_after["status"] == "resolved"
    finally:
        db_module._engine = None
        db_module._SessionLocal = None


def test_existing_annotation_database_gets_idempotent_review_columns(tmp_path, monkeypatch):
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE annotations (
            id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, source_type VARCHAR(20) NOT NULL,
            source_id VARCHAR(500) NOT NULL, annotation_type VARCHAR(20), geometry_geojson TEXT NOT NULL,
            crs_epsg INTEGER, label VARCHAR(500) NOT NULL, severity VARCHAR(20) NOT NULL,
            status VARCHAR(20) NOT NULL, note TEXT, include_in_report BOOLEAN,
            created_by INTEGER, created_at TIMESTAMP, updated_at TIMESTAMP
        )""")
        connection.execute("""INSERT INTO annotations
            (id, project_id, source_type, source_id, annotation_type, geometry_geojson,
             label, severity, status, note, include_in_report)
            VALUES (1, 7, 'image', 'legacy.jpg', 'rectangle', '{}',
                    'legacy mark', 'info', 'open', '', 1)""")

    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{path}")
    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None
    try:
        db_module.init_db()
        with sqlite3.connect(path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(annotations)")}
            assert {
                "origin", "machine_claims_json", "review_action", "parent_ids_json",
                "reviewed_by", "reviewed_at",
            } <= columns
            row = connection.execute(
                "SELECT origin, machine_claims_json, review_action, parent_ids_json "
                "FROM annotations WHERE id = 1"
            ).fetchone()
            assert row == ("human", "[]", "human_drawn", "[]")
        # A second startup proves the migration is idempotent.
        db_module.init_db()
    finally:
        db_module._engine = None
        db_module._SessionLocal = None
