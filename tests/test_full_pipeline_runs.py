"""The full inspection pipeline, driven the way the operator drives it.

"Run Full Pipeline" called StructuralFaultPipeline.run(output_root=...), a keyword the
method does not take, so every press failed with a TypeError two seconds in -- and no
test submitted the job, so nothing noticed. The CLI printed nothing until the end of a
run that could not end at drone resolution. And a sparse-cloud DSM at several metres
per pixel produced "Cut 2453.83 m3" in the report as if a survey had measured it.

These run the real Api job and the real CLI on real geotagged frames, downscaled so the
suite stays minutes rather than hours. The imagery is gitignored; without it they skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from conftest import REPO_ROOT, survey_imagery_dir

pytest.importorskip("pycolmap", reason="COLMAP engine not installed")

FRAME_COUNT = 8
FRAME_WIDTH = 1200


@pytest.fixture(scope="module")
def small_survey(tmp_path_factory) -> Path:
    """Eight consecutive Aukerman frames at 1200 px, with their EXIF (GPS, camera) kept."""
    Image = pytest.importorskip("PIL.Image")
    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:FRAME_COUNT] if source.is_dir() else []
    if len(frames) < FRAME_COUNT:
        pytest.skip(f"Survey imagery not present at {source} (gitignored).")
    target = tmp_path_factory.mktemp("survey") / "images"
    target.mkdir()
    for frame in frames:
        with Image.open(frame) as image:
            exif = image.info.get("exif", b"")
            height = round(image.height * FRAME_WIDTH / image.width)
            image.resize((FRAME_WIDTH, height)).save(target / frame.name, quality=92, exif=exif)
    return target


def _wait(api, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = api.job_status(job_id)["job"]
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture(scope="module")
def pipeline_run(small_survey, tmp_path_factory):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    root = tmp_path_factory.mktemp("pipeline")
    api = Api(AppSession(ProjectStore(db_path=str(root / "odk.db"))))
    api._session.create_project("Pipeline", root_dir=str(root / "project"))
    assert api.set_active_dataset(str(small_survey))["ok"]

    submitted = api.run_pipeline({"engine": "colmap", "profile": "fast", "dense": False})
    assert submitted["ok"], submitted
    job = _wait(api, submitted["job_id"], timeout=900)
    return api, job


@pytest.fixture(scope="module")
def pipeline_job(pipeline_run):
    return pipeline_run[1]


def test_run_full_pipeline_completes_from_the_api(pipeline_job):
    assert pipeline_job["status"] == "done", pipeline_job.get("error") or pipeline_job.get("log", [])[-5:]
    result = pipeline_job["result"]
    assert Path(result["summary_json_path"]).exists()
    assert Path(result["report_markdown_path"]).exists()
    assert Path(result["point_cloud_path"]).exists()


def test_the_pipeline_reconstructs_with_the_engine_it_was_asked_for(pipeline_job):
    summary = json.loads(Path(pipeline_job["result"]["summary_json_path"]).read_text(encoding="utf-8"))
    recon = summary["reconstruction"]
    assert recon["engine"] == "colmap"
    assert recon["registered_images"] >= 3
    assert recon["crs_epsg"], "geotagged frames must come out georeferenced"


def test_a_sparse_surface_refuses_to_report_a_volume(pipeline_job):
    """A cut/fill figure off a sparse multi-metre DSM is interpolation, not a measurement."""
    summary = json.loads(Path(pipeline_job["result"]["summary_json_path"]).read_text(encoding="utf-8"))
    recon = summary["reconstruction"]
    if recon["dense_point_count"] > 0 and (recon["ground_sample_distance_m"] or 99) <= 0.5:
        pytest.skip("This machine produced a dense surface; the refusal does not apply.")
    volume = summary["volume_estimation"]
    if volume:
        assert volume.get("refused") is True and "cut_volume_m3" not in json.dumps(volume)
        assert "sparse" in volume["reason"] or "m/px" in volume["reason"]
    report = Path(pipeline_job["result"]["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Cut (above reference)" not in report
    assert "Surface basis:" in report


def test_the_report_names_the_mesh_for_what_it_is(pipeline_job):
    report = Path(pipeline_job["result"]["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Textured mesh OBJ" not in report, "the COLMAP mesh has no texture"


def test_the_mesh_file_names_the_engine_that_built_it(pipeline_job):
    """The COLMAP Poisson mesh shared a writer with the custom engine and said so."""
    summary = json.loads(Path(pipeline_job["result"]["summary_json_path"]).read_text(encoding="utf-8"))
    obj = summary["reconstruction"].get("mesh_obj_path")
    if not obj:
        pytest.skip("open3d produced no mesh on this machine")
    header = Path(obj).read_text(encoding="utf-8").splitlines()[0]
    assert "CustomDroneReconstructor" not in header
    assert "Colmap" in header and "untextured" in header


def test_the_pipeline_puts_its_reconstruction_on_the_map(pipeline_run):
    """A full run used to leave the ortho, DSM and camera track on disk only.

    The layers must be the files the run wrote, not whatever was registered before.
    """
    api, job = pipeline_run
    assert job["status"] == "done"
    summary = json.loads(Path(job["result"]["summary_json_path"]).read_text(encoding="utf-8"))
    recon = summary["reconstruction"]
    layers = {layer["name"]: layer for layer in api.list_layers()["layers"]}
    expected = {
        "Orthomosaic": recon.get("orthomosaic_cog_path"),
        "DSM": recon.get("dsm_cog_path"),
        "Camera positions": recon.get("camera_track_geojson_path"),
    }
    for name, path in expected.items():
        assert path and Path(path).exists(), f"the run wrote no {name}"
        assert name in layers, f"{name} was not registered; layers: {sorted(layers)}"
        assert Path(layers[name]["path"]).resolve() == Path(path).resolve()
    assert set(expected) <= set(job["result"]["layers_registered"])
    assert layers["Orthomosaic"].get("crs_epsg") == recon["crs_epsg"]


def test_cancel_stops_the_pipeline(small_survey, tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(db_path=str(tmp_path / "odk.db"))))
    api._session.create_project("Cancel", root_dir=str(tmp_path / "project"))
    api.set_active_dataset(str(small_survey))
    job_id = api.run_pipeline({"engine": "colmap", "profile": "fast", "dense": False})["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline and api.job_status(job_id)["job"]["percent"] < 36:
        time.sleep(0.1)
    assert api.cancel_job(job_id)["cancelled"] is True
    job = _wait(api, job_id, timeout=120)
    assert job["status"] == "cancelled"


def test_the_cli_reports_progress_and_refuses_cloud_mode(small_survey, tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    refused = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--images", str(small_survey),
         "--output", str(tmp_path / "cloud"), "--recon-mode", "cloud"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(REPO_ROOT),
    )
    assert refused.returncode != 0
    assert "offline-first" in refused.stderr
    assert not list((tmp_path / "cloud").glob("run_*/masks/*")), "refused before any analysis ran"

    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--images", str(small_survey),
         "--output", str(tmp_path / "out"), "--engine", "colmap", "--recon-profile", "fast_preview",
         "--no-dense", "--skip-coverage-validation"],
        capture_output=True, text=True, timeout=900, env=env, cwd=str(REPO_ROOT),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    progress_lines = [line for line in done.stderr.splitlines() if line.startswith("[")]
    assert any("Analyzing image 1/" in line for line in progress_lines)
    assert any("Reconstruction:" in line for line in progress_lines)
    assert json.loads(done.stdout[: done.stdout.rindex("}") + 1])["report_markdown_path"]
