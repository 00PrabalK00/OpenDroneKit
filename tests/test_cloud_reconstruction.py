"""--recon-mode cloud runs on an OpenDroneKit server, or fails saying why.

It used to run locally with a warning, then was refused outright. These drive the real
client against the real services/api app served over HTTP on a local port: imagery is
uploaded through the checksummed upload API, a job is submitted and followed, and the
products are downloaded and used as the reconstruction result.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")
uvicorn = pytest.importorskip("uvicorn")

from conftest import REPO_ROOT, survey_imagery_dir  # noqa: E402
from core.reconstruction_remote import (  # noqa: E402
    RemoteReconstructionError,
    check_cloud_request,
    parse_endpoint,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """services/api on a real socket, with its own database and storage."""
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    # A server that resolves a relative output folder must not write into the checkout.
    monkeypatch.chdir(tmp_path)

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None
    from services.api.main import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not instance.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert instance.started, "the API server did not start"

    from fastapi.testclient import TestClient

    admin = TestClient(app)
    token = admin.post("/auth/register", json={
        "email": "pilot@example.com", "password": "longenough1", "organization_name": "Cloud Co",
    }).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    organization = admin.get("/organizations", headers=headers).json()[0]["id"]
    project = admin.post(f"/organizations/{organization}/projects", headers=headers,
                         json={"name": "Remote site"}).json()["id"]
    # A missing route answers FastAPI's generic "Not Found"; a present one knows no such job.
    probe = admin.get("/jobs/999999/artifacts/x", headers=headers)
    yield {
        "endpoint": f"http://127.0.0.1:{port}/projects/{project}",
        "token": token,
        "serves_artifacts": probe.json().get("detail") != "Not Found",
    }
    instance.should_exit = True
    thread.join(timeout=10)
    db_module._engine = None
    db_module._SessionLocal = None


def test_cloud_mode_without_an_endpoint_is_refused_before_any_work():
    with pytest.raises(ValueError, match="needs --recon-cloud-endpoint"):
        check_cloud_request("cloud", "", "token")
    with pytest.raises(ValueError, match="not an OpenDroneKit server project URL"):
        check_cloud_request("cloud", "https://example.com/submit", "token")
    check_cloud_request("local", "", "")  # local runs need nothing


def test_the_endpoint_names_the_server_and_project():
    assert parse_endpoint("http://10.0.0.5:8000/projects/3/") == ("http://10.0.0.5:8000", 3)
    assert parse_endpoint("https://odk.example/api/projects/12") == ("https://odk.example/api", 12)


def test_a_bad_token_is_an_authentication_error_not_a_local_run(server, tmp_path):
    from core.reconstruction_colmap import build_reconstructor

    images = tmp_path / "images"
    images.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32, 3), np.uint8))
    engine = build_reconstructor("auto", execution_mode="cloud",
                                 cloud_endpoint=server["endpoint"], cloud_token="wrong")
    with pytest.raises(RemoteReconstructionError, match="Authenticating failed.*HTTP 401"):
        engine.reconstruct(images, output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "orthomosaic.tif").exists()


def test_a_server_side_failure_comes_back_with_the_servers_reason(server, tmp_path):
    """Noise cannot be reconstructed; the server's refusal must reach the operator."""
    import cv2
    from core.reconstruction_colmap import build_reconstructor

    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for index in range(3):
        cv2.imwrite(str(images / f"{index}.png"), (rng.random((120, 160, 3)) * 255).astype(np.uint8))
    engine = build_reconstructor("custom", execution_mode="cloud", profile="fast_preview",
                                 cloud_endpoint=server["endpoint"], cloud_token=server["token"])
    engine.poll_interval_s = 0.2
    with pytest.raises(RemoteReconstructionError, match=r"Server job \d+ failed"):
        engine.reconstruct(images, output_dir=tmp_path / "out")
    # The request is on record locally: which server, which job, which files and hashes.
    assert (tmp_path / "out" / "cloud_request.json").is_file()


def _survey_copy(root: Path, count: int = 8, width: int = 1200) -> Path:
    Image = pytest.importorskip("PIL.Image")
    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:count] if source.is_dir() else []
    if len(frames) < count:
        pytest.skip(f"Survey imagery not present at {source} (gitignored).")
    images = root / "survey"
    images.mkdir()
    for frame in frames:
        with Image.open(frame) as image:
            height = round(image.height * width / image.width)
            image.resize((width, height)).save(images / frame.name, quality=92,
                                               exif=image.info.get("exif", b""))
    return images


def test_a_survey_reconstructs_on_the_server_and_its_products_come_home(server, tmp_path):
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route (GET /jobs/{id}/artifacts/{ref}).")
    rasterio = pytest.importorskip("rasterio")
    from core.reconstruction_colmap import build_reconstructor

    images = _survey_copy(tmp_path)
    progress: list[str] = []
    engine = build_reconstructor("auto", execution_mode="cloud", profile="fast_preview",
                                 cloud_endpoint=server["endpoint"], cloud_token=server["token"])
    result = engine.reconstruct(images, output_dir=tmp_path / "out",
                                progress_callback=lambda pct, msg: progress.append(msg))

    out = (tmp_path / "out").resolve()
    assert result.execution_mode_used == "cloud"
    assert any(message.startswith("Uploading 1/") for message in progress)
    assert any(message.startswith("Server job") for message in progress)
    assert result.registered_images >= 6
    for path in (result.orthomosaic_cog_path, result.dsm_cog_path, result.digital_twin_path):
        assert path and Path(path).resolve().is_relative_to(out), path
    with rasterio.open(result.orthomosaic_cog_path) as ortho:
        assert ortho.crs.to_epsg() == 32617
    assert any("Reconstructed on http://127.0.0.1" in w for w in result.warnings)


def test_the_cli_refuses_cloud_mode_without_a_server(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "ODK_CLOUD_TOKEN"}
    env["PYTHONIOENCODING"] = "utf-8"
    refused = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--images", str(images),
         "--output", str(tmp_path / "out"), "--recon-mode", "cloud"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(REPO_ROOT),
    )
    assert refused.returncode != 0
    assert "needs --recon-cloud-endpoint" in refused.stderr
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.png"))


# -- the other ways in: the desktop Api, the processing-run engine, the CLI preflight ----

def _desktop_api(root: Path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(root / "odk.db")))
    assert api.create_project("cloud desktop", str(root / "project"))["ok"]
    return api


def _wait_job(api, job_id: str, limit_s: float = 900.0) -> dict:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        job = api.job_status(job_id)["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(1.0)
    raise AssertionError("job did not finish")


def test_the_desktop_api_no_longer_ignores_a_cloud_request(tmp_path, monkeypatch):
    """execution_mode was dropped on the floor: a cloud request ran locally."""
    monkeypatch.delenv("ODK_CLOUD_TOKEN", raising=False)
    api = _desktop_api(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for call in (api.run_reconstruction, api.run_pipeline):
        refused = call({"image_dir": str(images), "execution_mode": "cloud"})
        assert refused["ok"] is False and "needs the server project URL" in refused["error"]
        refused = call({"image_dir": str(images), "execution_mode": "cloud",
                        "cloud_endpoint": "http://127.0.0.1:9/projects/1"})
        assert refused["ok"] is False and "API token" in refused["error"]
        refused = call({"image_dir": str(images), "execution_mode": "somewhere"})
        assert refused["ok"] is False and "Unknown execution_mode" in refused["error"]
    assert not api.list_jobs().get("jobs"), "a refused request must not start a job"


def test_the_processing_run_engine_refuses_cloud_without_a_token(tmp_path, monkeypatch):
    from core.errors import AppError
    from core.reconstruction_engine import ReconstructionConfig, run_reconstruction

    monkeypatch.delenv("ODK_CLOUD_TOKEN", raising=False)
    images = tmp_path / "images"
    images.mkdir()
    config = ReconstructionConfig(image_folder=str(images), output_folder=str(tmp_path / "out"),
                                  execution_mode="cloud",
                                  cloud_endpoint="http://127.0.0.1:9/projects/1")
    with pytest.raises(AppError, match="API token"):
        run_reconstruction(config)


def test_the_cli_stops_on_a_bad_token_before_analysing_any_image(server, tmp_path):
    """A wrong token used to be found after every image had been analysed (5 min here)."""
    images = tmp_path / "images"
    images.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((64, 64, 3), np.uint8))
    env = {k: v for k, v in os.environ.items() if k != "ODK_CLOUD_TOKEN"}
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.monotonic()
    refused = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--images", str(images),
         "--output", str(tmp_path / "out"), "--recon-mode", "cloud",
         "--recon-cloud-endpoint", server["endpoint"], "--recon-cloud-token", "wrong"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(REPO_ROOT),
    )
    assert refused.returncode == 2, refused.stderr[-2000:]
    assert "Authenticating failed" in refused.stderr and "HTTP 401" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert "Analyzing image" not in refused.stderr
    assert not list((tmp_path / "out").rglob("*_mask*.png"))
    assert time.monotonic() - started < 240


def test_the_processing_run_engine_reconstructs_on_the_server(server, tmp_path):
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route.")
    rasterio = pytest.importorskip("rasterio")
    from core.reconstruction_engine import ReconstructionConfig, run_reconstruction

    images = _survey_copy(tmp_path)
    result = run_reconstruction(ReconstructionConfig(
        image_folder=str(images), output_folder=str(tmp_path / "out"), profile="fast_preview",
        execution_mode="cloud", cloud_endpoint=server["endpoint"], cloud_token=server["token"]))
    assert result.quality_metrics["execution_mode_used"] == "cloud"
    assert (tmp_path / "out" / "reconstruction_summary.json").is_file()
    with rasterio.open(result.raw["orthomosaic_cog_path"]) as ortho:
        assert ortho.crs.to_epsg() == 32617


def test_the_desktop_api_reconstructs_on_the_server_and_registers_the_layers(server, tmp_path):
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route.")
    rasterio = pytest.importorskip("rasterio")

    images = _survey_copy(tmp_path)
    api = _desktop_api(tmp_path)
    started = api.run_reconstruction({
        "image_dir": str(images), "output_dir": str(tmp_path / "recon"), "profile": "fast_preview",
        "execution_mode": "cloud", "cloud_endpoint": server["endpoint"], "cloud_token": server["token"]})
    assert started["ok"], started
    job = _wait_job(api, started["job_id"])
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["execution_mode_used"] == "cloud"
    assert any("Reconstructed on http://127.0.0.1" in w for w in payload["warnings"])
    with rasterio.open(payload["orthomosaic_cog_path"]) as ortho:
        assert ortho.crs.to_epsg() == 32617
    assert Path(payload["orthomosaic_cog_path"]).resolve().is_relative_to((tmp_path / "recon").resolve())
    names = [layer.get("name", "") for layer in api.list_layers()["layers"]]
    assert any("rtho" in name for name in names), names
