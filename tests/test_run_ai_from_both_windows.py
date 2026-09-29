"""Run AI / Start (cockpit) and Run Full Pipeline (classic) finish a real run.

Audit: all three called StructuralFaultPipeline.run(output_root=...), a keyword it does
not take, so every press failed with a TypeError seconds in. tests/test_full_pipeline_runs
submits the Api job directly; these press the buttons, in headless Chromium against the
real Api, on eight real geotagged Aukerman frames, and wait for the job the button
started to finish.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cockpit_harness import Cockpit, make_api
from conftest import survey_imagery_dir

pytest.importorskip("pycolmap", reason="COLMAP engine not installed")

FRAME_COUNT = 8
FRAME_WIDTH = 1000


@pytest.fixture(scope="module")
def small_survey(tmp_path_factory) -> Path:
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


def _finish(api, job_id: str, timeout: float = 1200) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = api.job_status(job_id)["job"]
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.5)
    api.cancel_job(job_id)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def _check(api, cockpit, profile: str) -> None:
    started = cockpit.called("run_pipeline")
    assert started and started[-1]["ok"], started
    assert started[-1]["args"][0].get("profile") == profile, started[-1]
    job_id = api.list_jobs()["jobs"][0]["id"]
    job = _finish(api, job_id)
    assert job["status"] == "done", (job.get("error"), job.get("message"))
    assert "TypeError" not in str(job.get("error") or "")
    result = job["result"]
    assert Path(result["summary_json_path"]).is_file(), result
    assert result["reconstruction_status"] != "failed", result


def test_run_ai_in_the_cockpit_finishes_a_real_run(tmp_path, small_survey) -> None:
    api = make_api(tmp_path)
    assert api.set_active_dataset(str(small_survey))["ok"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("AI Inspection");
            await ODK.press("Run AI");
            await ODK.answer();
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) =>
              t.startsWith("Pipeline started") || t.includes("rror")), 15000) };
        """)
    assert result["toast"] and result["toast"].startswith("Pipeline started"), result
    _check(api, cockpit, "standard")


def test_run_full_pipeline_in_the_classic_window_finishes_a_real_run(tmp_path, small_survey) -> None:
    api = make_api(tmp_path)
    assert api.set_active_dataset(str(small_survey))["ok"]
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        result = cockpit.run("""
            await ODK.waitFor(() => ODK.$("#status-message").textContent.startsWith("OpenDroneKit ready")
              || ODK.$("#basemap-note"), 30000);
            ODK.$("#recon-profile").value = "fast_preview";
            ODK.$("#btn-pipeline").click();
            return { status: await ODK.waitFor(() => {
              const t = ODK.$("#status-message").textContent;
              return t.startsWith("Pipeline started") || t.includes("rror") ? t : null; }, 15000) };
        """)
    assert result["status"] == "Pipeline started.", result
    _check(api, cockpit, "fast_preview")
