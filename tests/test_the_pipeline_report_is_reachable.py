"""Run Pipeline's markdown inspection report can be found, and says only what happened.

Found by driving Run Pipeline on real Aukerman frames:

1. core/reporting.py wrote <project>/analysis/run_*/inspection_report.md and no screen
   mentioned, listed or opened it. The only way to read it was to know the folder.
2. With the reconstruction FAILED, the markdown still printed "Frames: 0", "Reconstructed
   points: 0" and an empty "Point cloud:" line, as if those were results, plus a
   "Surface basis: the surface comes from the sparse SfM cloud" line about a surface
   that was never built.
3. "Screening index trend across runs of this asset: stable" on the first run of an
   asset. There was nothing to compare with; "stable" was the default.
4. The HTML/PDF report said "No reconstruction was run." for a project whose pipeline
   ran one and it failed.

Everything goes through the real Api and the real pipeline on a real survey frame; the
two windows are the shipped pages in headless Chromium with the real Api behind them.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from conftest import survey_imagery_dir
from desktop_window import run_page


def _real_frames(target: Path, count: int) -> Path:
    """`count` real Aukerman frames, downscaled so the pipeline runs in seconds.

    EXIF (GPS, camera) is carried over, so the dataset is what a user imports.
    """
    from PIL import Image

    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:count] if source.is_dir() else []
    if len(frames) < count:
        pytest.skip(f"survey imagery not present at {source} (set ODK_SURVEY_IMAGERY)")
    target.mkdir(parents=True)
    for frame in frames:
        with Image.open(frame) as image:
            exif = image.info.get("exif", b"")
            image.thumbnail((800, 800))
            image.save(target / frame.name, exif=exif, quality=90)
    return target


def _run_pipeline(api: Api) -> dict:
    job = api.run_pipeline({"engine": "custom"})
    assert job["ok"], job
    deadline = time.time() + 600
    while time.time() < deadline:
        status = api.job_status(job["job_id"])["job"]
        if status["status"] in ("done", "failed", "cancelled"):
            assert status["status"] == "done", status.get("error")
            return status["result"]
        time.sleep(0.5)
    raise AssertionError("pipeline did not finish")


@pytest.fixture(scope="module")
def surveyed(tmp_path_factory):
    """A project with one real frame and one Run Pipeline run (reconstruction fails:
    structure from motion needs at least three images)."""
    base = tmp_path_factory.mktemp("pipeline_report")
    images = _real_frames(base / "imagery", 1)
    api = Api(AppSession(ProjectStore(base / "state.db")))
    assert api.create_project("Aukerman", str(base / "project"))["ok"]
    assert api.import_dataset(str(images), "Aukerman frame")["ok"]
    result = _run_pipeline(api)
    return api, result, base


class TestTheMarkdownSaysOnlyWhatHappened:
    def test_the_reconstruction_really_failed_so_the_case_is_real(self, surveyed) -> None:
        _, result, _ = surveyed
        assert result["reconstruction_status"] == "failed"

    def test_a_failed_reconstruction_prints_no_zero_results(self, surveyed) -> None:
        _, result, _ = surveyed
        text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Reconstruction FAILED" in text
        section = text.split("## 3D Reconstruction", 1)[1].split("\n## ", 1)[0]
        for line in ("Frames: `0`", "Reconstructed points: `0`", "Point cloud: ``",
                     "Camera poses: ``"):
            assert line not in section, line

    def test_no_surface_basis_is_given_for_a_surface_that_was_not_built(self, surveyed) -> None:
        _, result, _ = surveyed
        text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Surface basis" not in text

    def test_a_first_run_has_no_trend(self, surveyed) -> None:
        _, result, _ = surveyed
        text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert "trend across runs of this asset: `stable`" not in text
        assert "Trend: not computed" in text


def test_a_trend_needs_two_runs_a_week_apart() -> None:
    from core.health_scoring import _build_timeline_trend

    first = _build_timeline_trend([], "2026-01-01T00:00:00+00:00", 80.0, 20.0)
    assert first["trend_computed"] is False
    close = _build_timeline_trend([{"run_utc": "2026-01-01T00:00:00+00:00",
                                    "integrity_score": 80.0, "severity_index": 20.0}],
                                  "2026-01-02T00:00:00+00:00", 60.0, 40.0)
    assert close["trend_computed"] is False
    apart = _build_timeline_trend([{"run_utc": "2026-01-01T00:00:00+00:00",
                                    "integrity_score": 80.0, "severity_index": 20.0}],
                                  "2026-03-01T00:00:00+00:00", 60.0, 40.0)
    assert apart["trend_computed"] is True and apart["trend_direction"] == "degrading"

    from core.reporting import write_markdown_report
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        text = Path(write_markdown_report(Path(folder) / "r.md", {
            "health_scoring": {"severity_index": 40.0, "trend": apart}})).read_text(encoding="utf-8")
    assert "across 2 runs of this asset over 59 days: `degrading`" in text


class TestTheApiFindsTheReport:
    def test_it_is_listed(self, surveyed) -> None:
        api, result, _ = surveyed
        listed = api.pipeline_reports()
        assert listed["ok"], listed
        run = Path(result["run_dir"]).name
        assert [r["id"] for r in listed["reports"]] == [run]
        assert listed["reports"][0]["reconstruction_status"] == "failed"
        assert listed["reports"][0]["image_count"] == 1

    def test_its_text_is_the_file(self, surveyed) -> None:
        api, result, _ = surveyed
        read = api.read_pipeline_report(Path(result["run_dir"]).name)
        assert read["ok"], read
        assert read["markdown"] == Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert api.read_pipeline_report("")["markdown"] == read["markdown"]

    def test_an_unknown_run_is_refused_by_name(self, surveyed) -> None:
        api, _, _ = surveyed
        refused = api.read_pipeline_report("run_19990101_000000")
        assert refused["ok"] is False and "run_19990101_000000" in refused["error"]

    def test_a_project_with_no_run_says_so(self, tmp_path) -> None:
        api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
        assert api.create_project("Empty", str(tmp_path / "project"))["ok"]
        assert api.pipeline_reports()["reports"] == []
        refused = api.read_pipeline_report("")
        assert refused["ok"] is False and "Run Pipeline" in refused["error"]


class TestTheHtmlReportSaysTheReconstructionFailed:
    def test_failed_is_not_reported_as_never_run(self, surveyed) -> None:
        api, _, _ = surveyed
        written = api.generate_report("", "Aukerman", "standard", formats=["html"])
        assert written["ok"], written
        html = Path(written["html_path"]).read_text(encoding="utf-8")
        assert "No reconstruction was run." not in html
        assert "Reconstruction was attempted 1 time(s) and failed" in html

    def test_readiness_says_the_same(self, surveyed) -> None:
        api, _, _ = surveyed
        readiness = api.report_readiness("", "standard", ["html"])
        assert readiness["ok"] and readiness["ready"]
        notes = " ".join(readiness["notes"])
        assert "attempted and failed" in notes and "none was run" not in notes


    def test_leaving_out_the_defect_section_does_not_hide_the_failure(self, surveyed) -> None:
        """include_defects=False emptied pipeline_runs, and the failed reconstruction was
        again reported as "No reconstruction was run." """
        from dataclasses import replace

        from app import report_builder

        api, _, _ = surveyed
        config = replace(api._report_config(report_type="standard", formats=["html"]),
                         include_defects=False)
        readiness = report_builder.readiness(api._session, config)
        assert "attempted and failed" in " ".join(readiness.notes)
        result = report_builder.generate(api._session, config)
        html = Path(result.html_path).read_text(encoding="utf-8")
        assert "No reconstruction was run." not in html
        assert "Reconstruction was attempted 1 time(s) and failed" in html
        assert "reconstruction failed (" in html

def test_readiness_warns_when_an_imported_folder_is_gone(tmp_path) -> None:
    images = _real_frames(tmp_path / "imagery", 1)
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Moved", str(tmp_path / "project"))["ok"]
    assert api.import_dataset(str(images), "Frames")["ok"]
    assert not any("no longer exists" in w for w in api.report_readiness()["warnings"])
    shutil.rmtree(images)
    warnings = api.report_readiness()["warnings"]
    assert any("'Frames'" in w and "no longer exists" in w for w in warnings), warnings


WINDOW_DRIVER = r"""
const $ = (s) => document.querySelector(s);
const out = {};
$('#btn-reports').click();
await __waitFor(() => __modalOpen('Generated Reports')
  && document.querySelector('#modal-body button[data-pipeline-report]'));
out.listed = $('#modal-body').textContent;
document.querySelector('#modal-body button[data-pipeline-report]').click();
await __waitFor(() => $('#modal-title').textContent.startsWith('Inspection Report')
  && $('#modal-body pre'));
out.title = $('#modal-title').textContent;
out.text = $('#modal-body pre').textContent;
out.status = $('#status-message').textContent;
__finish(out);
"""

COCKPIT_DRIVER = r"""
const qa = (s) => Array.from(document.querySelectorAll(s));
const out = {};
// The Readiness panel fills once the shell has the open project.
await __waitFor(() => /warning|note|ready|missing/.test(
  qa('.console .line').map((l) => l.textContent).join(' ')));
// One click, as a user makes it. The shell re-reads the open project before refusing
// with "Open a project first", so no retry loop is needed here.
qa('.tbtn').find((b) => b.textContent === 'Inspection Report').click();
await __waitFor(() => document.querySelector('textarea.modal-reveal'));
out.text = document.querySelector('textarea.modal-reveal').value;
out.title = document.body.innerText.match(/Inspection report: run_\d+_\d+/)?.[0] || '';
__finish(out);
"""


class TestBothWindowsShowIt:
    def test_the_default_window_lists_and_shows_it(self, surveyed, tmp_path) -> None:
        api, result, _ = surveyed
        shown = run_page(api, WINDOW_DRIVER, tmp_path / "profile")
        assert "error" not in shown, shown
        run = Path(result["run_dir"]).name
        assert "Pipeline inspection report" in shown["listed"] and run in shown["listed"]
        assert shown["title"] == f"Inspection Report: {run}"
        on_disk = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert shown["text"].strip() == on_disk.strip()
        assert "inspection_report.md" in shown["status"]

    def test_the_cockpit_shows_it(self, surveyed, tmp_path) -> None:
        api, result, _ = surveyed
        shown = run_page(
            api, COCKPIT_DRIVER, tmp_path / "profile", page_name="workspace.html",
            before_js="localStorage.setItem('odk.workspace.last', 'reports');",
            ready_js="Array.from(document.querySelectorAll('.tbtn')).some("
                     "(b) => b.textContent === 'Inspection Report')",
        )
        assert "error" not in shown, shown
        on_disk = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert shown["text"].strip() == on_disk.strip()
        assert shown["title"] == f"Inspection report: {Path(result['run_dir']).name}"


def test_the_native_menu_reaches_it() -> None:
    """Reports > Pipeline Inspection Report... is handled by both windows."""
    pytest.importorskip("webview")
    from test_menu_coverage import _menu_messages

    actions = {action for _, action, _ in _menu_messages()}
    assert "report.pipeline" in actions
    web = Path(__file__).resolve().parents[1] / "app" / "web" / "js"
    assert "'report.pipeline'" in (web / "app.js").read_text(encoding="utf-8")
    assert 'case "report.pipeline"' in (web / "workspace" / "shell.js").read_text(encoding="utf-8")


def test_the_cockpit_opens_it_when_run_ai_finishes(tmp_path) -> None:
    """The default window opened the report when a pipeline job finished; the cockpit only
    named the menu it could be found under. Run AI is clicked in the cockpit, and the
    finished run's report opens by itself, identical to the file on disk."""
    from live_ui import LiveUI

    images = _real_frames(tmp_path / "imagery", 1)
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Cockpit run", str(tmp_path / "project"))["ok"]
    assert api.import_dataset(str(images), "Aukerman frame")["ok"]
    ui = LiveUI(api, "inspection")
    try:
        # The shell refuses actions until it has read the open project.
        ui.js("T.until(() => document.body.innerText.includes('Cockpit run'), 240000, 'the project')")
        ui.js("T.tool('Run AI')")
        ui.js("T.ok('Confirm')")
        message = ui.js("T.toast('finished|failed', 900000)")
        assert "finished" in message, message
        shown = ui.js("T.until(() => { const t = document.querySelector('textarea.modal-reveal');"
                      " return t && { text: t.value, title: (document.body.innerText"
                      r".match(/Inspection report: run_\d+_\d+/) || [''])[0] }; },"
                      " 60000, 'the inspection report')")
    finally:
        ui.close()
    reports = sorted((tmp_path / "project" / "analysis").glob("run_*/inspection_report.md"))
    assert len(reports) == 1
    assert shown["title"] == f"Inspection report: {reports[0].parent.name}"
    assert shown["text"].strip() == reports[0].read_text(encoding="utf-8").strip()


class TestTheReportsAgreeWithEachOther:
    """Found by the round-2 verifier: the HTML said no propagation analysis was run while
    the same run's markdown carried a forecast; the markdown heading said "(heuristic)"
    over a basis line naming a trained model; "Frames: 10" hid two unregistered frames."""

    def test_the_html_names_the_pipelines_forecast(self, surveyed) -> None:
        import json

        api, result, _ = surveyed
        summary = json.loads((Path(result["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        forecast = summary.get("forecast") or {}
        written = api.generate_report("", "Aukerman", "standard", formats=["html"])
        html = Path(written["html_path"]).read_text(encoding="utf-8")
        markdown = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        if forecast.get("target_image"):
            assert f"Target image: `{forecast['target_image']}`" in markdown
            assert "No crack propagation analysis was run." not in html
            assert f"growth forecast of the detected crack mask on {forecast['target_image']}" in html
            assert "not a measurement" in html
        else:
            assert "No crack propagation analysis was run." in html

    def test_the_heading_names_the_same_evidence_as_the_basis(self, surveyed) -> None:
        _, result, _ = surveyed
        text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        section = text.split("## Structural Health Screening", 1)[1].split("\n## ", 1)[0]
        heading = section.splitlines()[0]
        if "trained segmentation model" in section:
            assert "trained model" in heading and "heuristic" not in heading
        else:
            assert "(heuristic)" in heading

    def test_frames_say_how_many_were_registered(self, tmp_path) -> None:
        from core.reporting import write_markdown_report

        def frames(recon: dict) -> str:
            text = Path(write_markdown_report(tmp_path / "r.md", {"reconstruction": recon})
                        ).read_text(encoding="utf-8")
            return next(line for line in text.splitlines() if line.startswith("- Frames:"))

        assert frames({"frame_count": 10, "registered_images": 8}) == (
            "- Frames: `10` (`8` registered, `2` not in the model)")
        assert frames({"frame_count": 3, "registered_images": 3}) == "- Frames: `3` (all registered)"
        assert frames({"frame_count": 4}) == "- Frames: `4`"
