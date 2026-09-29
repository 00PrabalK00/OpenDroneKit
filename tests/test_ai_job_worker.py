"""AI jobs are claimed and run by a worker, and refused up front when they cannot be.

Before this, ``POST /ai-jobs`` stored ``pending_worker`` and nothing in the codebase
ever claimed it. These tests submit through the API and wait for a real terminal
state, in both worker modes: inline (the API process) and external (the standalone
``services.worker.ai_worker`` draining the shared database).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

from hub_api_helpers import (  # noqa: E402
    configure, project_with_dataset, register, reset_engine, upload, wait, weights_installed,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_engine()


@pytest.fixture
def site(client):
    headers, org = register(client)
    project_id, dataset_id = project_with_dataset(client, headers, org)
    upload(client, headers, dataset_id)
    return headers, project_id, dataset_id


def test_a_heuristic_detection_job_runs_to_a_result(client, site):
    headers, project_id, dataset_id = site
    created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
    })
    assert created.status_code == 201, created.text
    job = wait(client, headers, f"/ai-jobs/{created.json()['id']}")
    assert job["status"] == "done", job
    assert job["result"]["model_used"] == "heuristic"
    assert job["result"]["model_key"] == ""
    assert job["result"]["summary"]["detection_count"] > 0
    results = client.get(f"/jobs/{job['id']}/artifacts/{job['artifacts'][0]}", headers=headers)
    document = json.loads(results.content)
    assert document["images"][0]["image"] == "concrete_cracked_public_domain.jpg"


def test_one_frame_of_a_dataset_can_be_selected(client, site):
    headers, project_id, dataset_id = site
    created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "segmentation", "model_key": "heuristic",
        "input_ref": f"dataset:{dataset_id}/frame:concrete_cracked_public_domain.jpg",
    }).json()
    job = wait(client, headers, f"/ai-jobs/{created['id']}")
    assert job["status"] == "done", job
    assert job["result"]["image_count"] == 1
    assert 0 < job["result"]["summary"]["mean_pixel_fraction"]["crack"] < 1


@pytest.mark.parametrize("payload, fragment", [
    ({"task": "detection", "model_key": "no_such_model"}, "not in the model registry"),
    ({"task": "classification", "model_key": "structural_multiclass_detector"}, "cannot answer"),
    ({"task": "detection", "model_key": "structural_codebrim_detector"}, "awaiting_weights"),
    ({"task": "classification", "model_key": "heuristic"}, "no heuristic"),
])
def test_requests_that_cannot_be_answered_are_refused_with_the_reason(client, site, payload, fragment):
    headers, project_id, dataset_id = site
    response = client.post(f"/projects/{project_id}/ai-jobs", headers=headers,
                           json={**payload, "input_ref": f"dataset:{dataset_id}"})
    assert response.status_code == 422
    assert fragment in response.json()["detail"]
    assert client.get(f"/projects/{project_id}/ai-jobs", headers=headers).json() == []


def test_an_input_naming_no_uploaded_image_is_refused(client, site):
    headers, project_id, dataset_id = site
    for ref, code in ((f"dataset:{dataset_id}/frame:missing.jpg", 422),
                      ("dataset:999", 404), ("not a reference", 422)):
        response = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
            "task": "detection", "model_key": "heuristic", "input_ref": ref})
        assert response.status_code == code, (ref, response.text)


def test_missing_weights_are_refused_rather_than_substituted(client, site, monkeypatch, tmp_path):
    """With the weights file absent, the request is refused; no heuristic stands in."""
    import core.models as models

    empty = tmp_path / "no_models"
    (empty).mkdir()
    shutil_copy = __import__("shutil").copy
    shutil_copy(models.registry_path(), empty / "model_registry.json")
    monkeypatch.setattr(models, "models_root", lambda: empty)
    headers, project_id, dataset_id = site
    response = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "detection", "model_key": "structural_multiclass_detector",
        "input_ref": f"dataset:{dataset_id}"})
    assert response.status_code == 422
    assert "weights are not installed" in response.json()["detail"]


def test_external_mode_waits_for_the_standalone_worker(client, site, monkeypatch):
    monkeypatch.setenv("ODK_AI_WORKER", "external")
    headers, project_id, dataset_id = site
    created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
    }).json()
    assert created["status"] == "pending_worker"
    assert client.get(f"/ai-jobs/{created['id']}", headers=headers).json()["result"] is None

    from services.worker.ai_worker import drain

    assert drain() == [(created["id"], "done")]
    assert drain() == [], "a finished job must not be claimed twice"
    job = client.get(f"/ai-jobs/{created['id']}", headers=headers).json()
    assert job["status"] == "done" and job["result"]["summary"]["detection_count"] > 0


def test_a_pending_job_can_be_cancelled_and_is_then_never_run(client, site, monkeypatch):
    monkeypatch.setenv("ODK_AI_WORKER", "external")
    headers, project_id, dataset_id = site
    created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
    }).json()
    cancelled = client.post(f"/ai-jobs/{created['id']}/cancel", headers=headers)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"

    from services.worker.ai_worker import drain

    assert drain() == []
    assert client.post(f"/ai-jobs/{created['id']}/cancel", headers=headers).status_code == 409


def test_jobs_left_pending_by_a_previous_process_resume_on_startup(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("ODK_AI_WORKER", "external")
    from services.api.main import app

    with TestClient(app) as first:
        headers, org = register(first)
        project_id, dataset_id = project_with_dataset(first, headers, org)
        upload(first, headers, dataset_id)
        created = first.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
            "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
        }).json()
        assert created["status"] == "pending_worker"

    monkeypatch.setenv("ODK_AI_WORKER", "inline")
    with TestClient(app) as second:
        assert wait(second, headers, f"/ai-jobs/{created['id']}")["status"] == "done"
    reset_engine()


@pytest.mark.skipif(not weights_installed("crack_presence_classifier"),
                    reason="crack_presence_classifier weights are not installed")
def test_the_crack_classifier_reads_its_measured_input_contract(client, site):
    headers, project_id, dataset_id = site
    created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "anomaly", "model_key": "crack_presence_classifier",
        "input_ref": f"dataset:{dataset_id}"}).json()
    job = wait(client, headers, f"/ai-jobs/{created['id']}")
    assert job["status"] == "done", job
    assert job["result"]["summary"]["predicted_counts"] == {"cracked": 1}
    assert job["result"]["model_sha256"].startswith("92a4d142")


def test_events_from_a_separate_worker_process_reach_the_api_live_stream(client, site, monkeypatch):
    """The live-event broker is the shared database, so a worker in another process
    (the compose ``ai-worker`` service, a Celery worker) reaches API WebSocket clients."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setenv("ODK_AI_WORKER", "external")
    headers, project_id, dataset_id = site
    org = client.get(f"/projects/{project_id}", headers=headers).json()["organization_id"]
    token = headers["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/organizations/{org}?token={token}") as socket:
        assert socket.receive_json()["event"] == "stream.connected"
        created = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
            "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
        }).json()
        assert created["status"] == "pending_worker"

        worker = subprocess.run(
            [sys.executable, "-m", "services.worker.ai_worker", "--once"],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ),
            capture_output=True, text=True, timeout=300,
        )
        assert worker.returncode == 0, worker.stderr[-2000:]
        for _ in range(10):
            message = socket.receive_json()
            if message["event"] == "ai.completed":
                break
    assert message["event"] == "ai.completed", message
    assert message["data"]["ai_job_id"] == created["id"] and message["data"]["status"] == "done"


def test_the_model_catalogue_lists_every_candidate_with_a_reason_when_unusable(client, site):
    headers, _, _ = site
    assert client.get("/ai-models").status_code == 401
    models = client.get("/ai-models", headers=headers).json()["models"]
    assert {"task": "detection", "model_key": "heuristic"}.items() <= next(
        m for m in models if m["model_key"] == "heuristic" and m["task"] == "detection").items()
    assert not any(m["model_key"] == "heuristic" and m["task"] == "classification" for m in models)
    from core.models import load_registry

    listed = {m["model_key"] for m in models}
    assert set(load_registry()["models"]) - listed <= {
        key for key, row in load_registry()["models"].items()
        if row.get("kind") not in {"onnx_yolo", "onnx_detection", "onnx_segmentation",
                                   "onnx_segmentation_multiclass", "onnx_classification"}}
    for model in models:
        assert model["usable"] or model["reason"], model
