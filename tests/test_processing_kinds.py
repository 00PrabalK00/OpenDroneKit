"""Every job kind the API advertises runs, writes where it says, and can be downloaded.

The audit found ``detection``, ``measurement`` and ``report`` accepted with 202 and then
failed with NotImplementedError, and ``reconstruction`` writing into the server's
working directory because its output folder was passed as ``crack_mask_dir``. These
tests submit each kind through the API and check the product: rows in the database,
files under ``<storage>/runs/job_<id>``, and bytes from the artifact download route.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

from conftest import BLOCK_VOLUME_M3, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M  # noqa: E402
from hub_api_helpers import (  # noqa: E402
    REPO, configure, project_with_dataset, register, reset_engine, upload, wait,
    weights_installed,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)  # so anything written to the cwd would be visible
    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_engine()


@pytest.fixture
def site(client):
    headers, org = register(client)
    project_id, dataset_id = project_with_dataset(client, headers, org)
    return headers, org, project_id, dataset_id


def _storage(tmp_path: Path) -> Path:
    return tmp_path / "storage"


class TestDetectionJob:
    def test_heuristic_detection_records_unreviewed_defects_and_an_overlay(self, client, site, tmp_path):
        headers, _, project_id, dataset_id = site
        upload(client, headers, dataset_id)
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "detection", "dataset_id": dataset_id, "options": {"model_key": "heuristic"},
        })
        assert job.status_code == 202, job.text
        done = wait(client, headers, f"/jobs/{job.json()['id']}")
        assert done["status"] == "done", done
        assert all(ref.startswith(f"runs/job_{done['id']}/") for ref in done["artifacts"])

        defects = client.get(f"/projects/{project_id}/defects", headers=headers).json()
        assert defects, "a cracked concrete photo produced no heuristic findings"
        assert {d["source"] for d in defects} == {"heuristic"}
        assert {d["review_state"] for d in defects} == {"unreviewed"}
        assert all(d["model_key"] == "" for d in defects), "a heuristic must not claim a model"

        results = client.get(f"/jobs/{done['id']}/artifacts/{done['artifacts'][0]}", headers=headers)
        assert results.status_code == 200
        assert json.loads(results.content)["model_used"] == "heuristic"

    @pytest.mark.skipif(not weights_installed("structural_multiclass_detector"),
                        reason="structural_multiclass_detector weights are not installed")
    def test_model_detection_attributes_every_defect_to_the_model_file(self, client, site):
        headers, _, project_id, dataset_id = site
        upload(client, headers, dataset_id)
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "detection", "dataset_id": dataset_id,
            "options": {"model_key": "structural_multiclass_detector"}}).json()
        done = wait(client, headers, f"/jobs/{job['id']}")
        assert done["status"] == "done", done
        from core.models import model_identity

        digest = model_identity("structural_multiclass_detector")["model_sha256"]
        for defect in client.get(f"/projects/{project_id}/defects", headers=headers).json():
            assert defect["source"] == "model"
            assert defect["model_key"] == "structural_multiclass_detector"
            assert defect["model_sha256"] == digest

    def test_a_model_without_weights_is_refused_at_submission(self, client, site):
        headers, _, project_id, dataset_id = site
        response = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "detection", "dataset_id": dataset_id,
            "options": {"model_key": "structural_codebrim_detector"}})
        assert response.status_code == 422
        assert "awaiting_weights" in response.json()["detail"]

    def test_detection_needs_a_dataset(self, client, site):
        headers, _, project_id, _ = site
        response = client.post(f"/projects/{project_id}/jobs", headers=headers,
                               json={"kind": "detection"})
        assert response.status_code == 422


class TestReportJob:
    def test_report_job_writes_a_pdf_and_a_report_row(self, client, site):
        headers, _, project_id, _ = site
        client.post(f"/projects/{project_id}/defects", headers=headers,
                    json={"category": "crack", "severity": "high"})
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "report", "options": {"title": "Bridge 7", "include_unreviewed": True}}).json()
        done = wait(client, headers, f"/jobs/{job['id']}")
        assert done["status"] == "done", done
        pdf_ref = next(ref for ref in done["artifacts"] if ref.endswith(".pdf"))
        pdf = client.get(f"/jobs/{done['id']}/artifacts/{pdf_ref}", headers=headers)
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")

        reports = client.get(f"/projects/{project_id}/reports", headers=headers).json()
        assert reports[-1]["format"] == "pdf" and reports[-1]["title"] == "Bridge 7"
        assert len(reports[-1]["payload"]["findings"]) == 1

    def test_the_default_report_says_what_it_left_out(self, client, site):
        headers, _, project_id, _ = site
        client.post(f"/projects/{project_id}/defects", headers=headers,
                    json={"category": "crack", "severity": "high"})
        report = client.post(f"/projects/{project_id}/reports", headers=headers,
                             json={"title": "Default"}).json()
        assert report["payload"]["findings"] == []
        assert report["payload"]["excluded_not_accepted_count"] == 1
        assert "include_unreviewed" in report["payload"]["findings_note"]


class TestArtifactDownload:
    def test_only_listed_artifacts_are_served(self, client, site):
        headers, _, project_id, _ = site
        job = client.post(f"/projects/{project_id}/jobs", headers=headers,
                          json={"kind": "report"}).json()
        done = wait(client, headers, f"/jobs/{job['id']}")
        for ref in ("../api.db", "runs/job_999/report.pdf", "datasets/1/x.jpg"):
            assert client.get(f"/jobs/{done['id']}/artifacts/{ref}", headers=headers).status_code == 404


def _done_reconstruction(client, headers, project_id, tmp_path, rasters: dict[str, Path]) -> int:
    """A finished reconstruction job whose rasters are real GeoTIFFs in storage.

    Photogrammetry is the external engine here; the measurement code under test reads
    the rasters it would have written, from where the API records them.
    """
    from services.api.db import get_session_factory
    from services.api.models import Job

    session = get_session_factory()()
    try:
        job = Job(project_id=project_id, kind="reconstruction", engine="colmap",
                  status="done", percent=100, crs_epsg=32617)
        session.add(job)
        session.flush()
        folder = _storage(tmp_path) / "runs" / f"job_{job.id}"
        folder.mkdir(parents=True)
        refs = []
        for name, source in rasters.items():
            shutil.copy(source, folder / f"{name}.tif")
            refs.append(f"runs/job_{job.id}/{name}.tif")
        job.artifacts_json = json.dumps(refs)
        session.commit()
        return job.id
    finally:
        session.close()


class TestMeasurementJob:
    def test_volume_on_the_dsm_is_measured_and_persisted(self, client, site, surface_pair, tmp_path):
        headers, _, project_id, _ = site
        source = _done_reconstruction(client, headers, project_id, tmp_path, surface_pair)
        polygon = [[ORIGIN_X, ORIGIN_Y - 40], [ORIGIN_X + 40, ORIGIN_Y - 40],
                   [ORIGIN_X + 40, ORIGIN_Y], [ORIGIN_X, ORIGIN_Y]]
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "measurement",
            "options": {"measurement": "volume", "source_job_id": source, "polygon_xy": polygon},
        })
        assert job.status_code == 202, job.text
        done = wait(client, headers, f"/jobs/{job.json()['id']}")
        assert done["status"] == "done", done
        assert done["crs_epsg"] == 32617

        rows = client.get(f"/projects/{project_id}/measurements", headers=headers).json()
        assert len(rows) == 1
        assert rows[0]["kind"] == "volume" and rows[0]["unit"] == "m3"
        assert rows[0]["value"] == pytest.approx(BLOCK_VOLUME_M3, rel=0.02)
        assert rows[0]["source_ref"] == f"runs/job_{source}/dsm.tif"

    def test_area_on_the_orthomosaic_uses_its_pixel_size(self, client, site, surface_pair, tmp_path):
        headers, _, project_id, _ = site
        source = _done_reconstruction(client, headers, project_id, tmp_path,
                                      {"orthomosaic": surface_pair["dsm"]})
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "measurement", "options": {
                "measurement": "area", "source_job_id": source,
                "pixels": [[0, 0], [20, 0], [20, 20], [0, 20]]},
        }).json()
        done = wait(client, headers, f"/jobs/{job['id']}")
        assert done["status"] == "done", done
        row = client.get(f"/projects/{project_id}/measurements", headers=headers).json()[0]
        assert row["value"] == pytest.approx((20 * PIXEL_SIZE_M) ** 2)
        assert row["unit"] == "m2" and row["crs_epsg"] == 32617

    def test_a_source_without_rasters_fails_with_the_reason(self, client, site, tmp_path):
        headers, _, project_id, _ = site
        source = _done_reconstruction(client, headers, project_id, tmp_path, {})
        job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
            "kind": "measurement", "options": {
                "measurement": "distance", "source_job_id": source, "pixels": [[0, 0], [5, 5]]},
        }).json()
        done = wait(client, headers, f"/jobs/{job['id']}")
        assert done["status"] == "failed"
        assert "no georeferenced orthomosaic" in done["error"]
        assert client.get(f"/projects/{project_id}/measurements", headers=headers).json() == []

    def test_malformed_options_are_refused_at_submission(self, client, site):
        headers, _, project_id, _ = site
        for options in ({"measurement": "weight"}, {"measurement": "volume", "source_job_id": 99}):
            response = client.post(f"/projects/{project_id}/jobs", headers=headers,
                                   json={"kind": "measurement", "options": options})
            assert response.status_code == 422


def _aukerman() -> Path | None:
    configured = os.environ.get("ODK_AUKERMAN_DIR", "")
    for candidate in (Path(configured) if configured else None,
                      REPO / "training" / "data" / "aukerman_subset"):
        if candidate is not None and len(list(candidate.glob("*.JPG"))) >= 8:
            return candidate
    return None


@pytest.mark.skipif(_aukerman() is None, reason="aukerman imagery not present (set ODK_AUKERMAN_DIR)")
@pytest.mark.skipif(pytest.importorskip("core.reconstruction_colmap")._HAS_PYCOLMAP is False,
                    reason="pycolmap is not installed")
def test_reconstruction_writes_into_its_own_job_folder_and_can_be_cancelled(client, site, tmp_path):
    headers, _, project_id, dataset_id = site
    for image in sorted(_aukerman().glob("*.JPG"))[:8]:
        upload(client, headers, dataset_id, image, chunk_size=4 * 1024 * 1024)

    job = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
        "kind": "reconstruction", "dataset_id": dataset_id, "engine": "colmap",
        "profile": "fast_preview"}).json()
    seen = set()
    import time

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job['id']}", headers=headers).json()
        seen.add(body["percent"])
        if body["status"] in {"done", "failed", "cancelled"}:
            break
        time.sleep(0.2)
    assert body["status"] == "done", body
    assert body["crs_epsg"] is not None
    assert not (tmp_path / "reconstruction").exists(), "output leaked into the working directory"
    assert body["artifacts"] and all(r.startswith(f"runs/job_{job['id']}/") for r in body["artifacts"])
    assert any(0 < p < 100 for p in seen), f"no intermediate progress was reported: {sorted(seen)}"
    dsm = next(r for r in body["artifacts"] if r.endswith("/dsm.tif"))
    assert client.get(f"/jobs/{job['id']}/artifacts/{dsm}", headers=headers).content[:2] in (b"II", b"MM")

    second = client.post(f"/projects/{project_id}/jobs", headers=headers, json={
        "kind": "reconstruction", "dataset_id": dataset_id, "engine": "colmap",
        "profile": "fast_preview"}).json()
    while client.get(f"/jobs/{second['id']}", headers=headers).json()["percent"] < 3:
        time.sleep(0.1)
    assert client.post(f"/jobs/{second['id']}/cancel", headers=headers).status_code == 200
    assert wait(client, headers, f"/jobs/{second['id']}", timeout=600)["status"] == "cancelled"
