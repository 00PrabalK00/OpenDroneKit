"""A report that cannot be written says what is missing; screening is not assessment.

Found by an independent re-drive of the reports fixes:

1. Api.report_readiness spread the checklist's own "ok": False over the call's ok, so a
   refused type (mission_summary with no saved mission) came back as a failed call with
   no error. Both UIs throw on ok === false, so the default window printed
   "report_readiness: Unknown error" and kept the previous type's checklist, and the
   cockpit printed "Generate: Unknown error". Nobody was told what was missing.

2. The pipeline's markdown report turned heuristic crack ratios on an ordinary photo
   into "Condition grade Level 5 (Critical)" and "Failure probability (1 year) 7.54%",
   and printed "Factor of safety 0.000" / "Crack length 0.000000 m" from a geometric
   forecast that computes no physics at all.

Every test goes through the real Api (or the real pipeline) and, for the UIs, the
shipped pages in a headless browser with the real Api behind the bridge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from desktop_window import run_page
from test_reports_in_the_desktop_window import _geotagged_jpeg

pytest.importorskip("reportlab")
pytest.importorskip("docx")

MISSING = "mission summary needs a saved mission"


def _project_without_a_mission(base: Path) -> tuple[Api, Path]:
    images = base / "imagery"
    images.mkdir(parents=True)
    for index in range(3):
        _geotagged_jpeg(images / f"IMG_{index:04d}.jpg", 41.3038 + index * 1e-4, -81.7504, 90)
    api = Api(AppSession(ProjectStore(base / "state.db")))
    root = base / "project"
    assert api.create_project("Culvert 11", str(root))["ok"]
    assert api.import_dataset(str(images), "Deck imagery")["ok"]
    return api, root


class TestReadinessThroughTheApi:
    def test_a_refused_type_is_a_successful_call_that_names_what_is_missing(
            self, tmp_path) -> None:
        api, _ = _project_without_a_mission(tmp_path)
        readiness = api.report_readiness("", "mission_summary", ["html"])
        assert readiness["ok"] is True, readiness
        assert readiness["ready"] is False
        assert any(MISSING in item for item in readiness["missing"]), readiness

    def test_generate_for_the_same_type_refuses_with_the_same_reason(self, tmp_path) -> None:
        api, root = _project_without_a_mission(tmp_path)
        refused = api.generate_report("", "M", "mission_summary")
        assert refused["ok"] is False and MISSING in refused["error"]
        assert not (root / "reports").exists() or not any((root / "reports").iterdir())

    def test_a_ready_type_is_still_ready(self, tmp_path) -> None:
        api, _ = _project_without_a_mission(tmp_path)
        readiness = api.report_readiness("", "dataset_quality", ["html"])
        assert readiness["ok"] is True and readiness["ready"] is True
        assert readiness["missing"] == []


WINDOW_DRIVER = r"""
const $ = (s) => document.querySelector(s);
const out = {};
$('#btn-report').click();
await __waitFor(() => __modalOpen('Generate Report')
  && /Ready|Warning|Missing|Note/.test($('#rp-readiness').textContent));
out.before = $('#rp-readiness').textContent;
$('#rp-type').value = 'mission_summary';
$('#rp-type').dispatchEvent(new Event('change'));
await __waitFor(() => /Missing/.test($('#rp-readiness').textContent));
out.after = $('#rp-readiness').textContent;
out.status = $('#status-message').textContent;
__clickAction('Generate');
await __waitFor(() => /Report not generated/.test($('#status-message').textContent));
out.generate = $('#status-message').textContent;
__finish(out);
"""


COCKPIT_DRIVER = r"""
const q = (s) => document.querySelector(s);
const qa = (s) => Array.from(document.querySelectorAll(s));
const seen = [];
new MutationObserver(() => qa('.toast').forEach((t) => {
  if (!seen.includes(t.textContent)) seen.push(t.textContent);
})).observe(document.body, { childList: true, subtree: true });
const out = {};
const lines = () => qa('.console .line').map((l) => l.textContent).join(' | ');

await __waitFor(() => /warning|note|ready|missing/.test(lines()));
out.before = lines();
const field = qa('.field').find((f) => f.querySelector('label').textContent === 'Report');
const select = field.querySelector('select');
select.value = 'mission_summary';
select.dispatchEvent(new Event('change'));
await __waitFor(() => /missing/.test(lines()) && /saved mission/.test(lines()));
out.after = lines();

qa('.tbtn').find((b) => b.textContent === 'Generate').click();
await __waitFor(() => seen.some((t) => t.startsWith('Generate')));
out.generate = seen.filter((t) => t.startsWith('Generate')).join(' | ');
__finish(out);
"""


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    api, _ = _project_without_a_mission(tmp_path_factory.mktemp("refused_window"))
    result = run_page(api, WINDOW_DRIVER, tmp_path_factory.mktemp("profile_w"))
    assert "error" not in result, result
    return result


@pytest.fixture(scope="module")
def cockpit(tmp_path_factory):
    api, _ = _project_without_a_mission(tmp_path_factory.mktemp("refused_cockpit"))
    result = run_page(
        api, COCKPIT_DRIVER, tmp_path_factory.mktemp("profile_c"), page_name="workspace.html",
        before_js="localStorage.setItem('odk.workspace.last', 'reports');",
        ready_js="Array.from(document.querySelectorAll('.tbtn')).some("
                 "(b) => b.textContent === 'Generate')",
    )
    assert "error" not in result, result
    return result


class TestTheDefaultWindowSaysWhatIsMissing:
    def test_changing_to_a_refused_type_shows_the_missing_item(self, window) -> None:
        assert "Missing" not in window["before"]
        assert "Missing" in window["after"] and MISSING in window["after"]
        assert "Unknown error" not in window["status"]

    def test_generate_names_the_reason(self, window) -> None:
        assert MISSING in window["generate"]
        assert "Unknown error" not in window["generate"]


class TestTheCockpitSaysWhatIsMissing:
    def test_the_readiness_panel_follows_the_chosen_type(self, cockpit) -> None:
        assert "saved mission" not in cockpit["before"]
        assert MISSING in cockpit["after"]

    def test_generate_names_the_reason(self, cockpit) -> None:
        assert MISSING in cockpit["generate"], cockpit["generate"]
        assert "Unknown error" not in cockpit["generate"]


def _photo_with_crack_like_edges(path: Path) -> None:
    """An ordinary scene with dark thin strokes: the classical crack heuristic fires on
    it, which is exactly the case where it must not be reported as a structural grade."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    image = np.full((360, 480, 3), 150, np.uint8)
    image += rng.integers(0, 25, image.shape, dtype=np.uint8)
    for k in range(12):
        x0 = int(rng.integers(0, 480))
        cv2.line(image, (x0, 0), (int(rng.integers(0, 480)), 359), (25, 25, 25), 2)
        cv2.line(image, (0, 30 * k), (479, 30 * k + int(rng.integers(-20, 20))),
                 (30, 30, 30), 1)
    cv2.imwrite(str(path), image)


def _without_the_crack_model():
    """This case is about the classical crack heuristic, so it must be the one that runs.

    On a machine with the trained crack segmenter installed, the pipeline prefers it and
    the heuristic never fires on these strokes. The installed-weights lookup is the
    external boundary; reporting the crack models as absent is how a machine without
    them behaves, and everything downstream runs for real.
    """
    import core.detection as detection

    real = detection.model_status

    def status(key: str, *args, **kwargs):
        info = real(key, *args, **kwargs)
        if str(key).startswith("crack"):
            info = {**info, "exists": False}
        return info

    patch = pytest.MonkeyPatch()
    patch.setattr(detection, "model_status", status)
    return patch


def _run(api: Api) -> dict:
    job = api.run_pipeline({"engine": "custom"})
    assert job["ok"], job
    deadline = time.time() + 600
    while time.time() < deadline:
        status = api.job_status(job["job_id"])["job"]
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.5)
    raise AssertionError("pipeline did not finish")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("screening")
    images = base / "imagery"
    images.mkdir()
    for index in range(3):
        _photo_with_crack_like_edges(images / f"PARK_{index}.jpg")
    api = Api(AppSession(ProjectStore(base / "state.db")))
    assert api.create_project("Park", str(base / "project"))["ok"]
    assert api.import_dataset(str(images), "Park photos")["ok"]
    patch = _without_the_crack_model()
    try:
        status = _run(api)
    finally:
        patch.undo()
    assert status["status"] == "done", status.get("error")
    return status["result"]


class TestThePipelineReportDoesNotGradeAStructure:
    def test_the_heuristic_fired_so_the_case_is_real(self, run) -> None:
        summary = json.loads(Path(run["summary_json_path"]).read_text(encoding="utf-8"))
        assert max(float(i["crack_ratio"]) for i in summary["images"]) > 0.0
        assert summary["forecast"]["target_image"]

    def test_no_grade_or_failure_probability_is_printed(self, run) -> None:
        text = Path(run["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Condition grade:" not in text or "not reported" in text
        for phrase in ("Level 5", "Level 4", "(Critical)", "Failure probability (1 year)",
                       "Failure probability (6 months)", "Integrity score:"):
            assert phrase not in text, phrase
        assert "Screening severity index (heuristic)" in text
        assert "not a structural assessment" in text

    def test_a_geometric_forecast_prints_no_physics(self, run) -> None:
        text = Path(run["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Factor of safety: `0.000`" not in text
        assert "0.000000 m" not in text
        assert "no crack length, stress intensity, factor of safety" in text
        summary = json.loads(Path(run["summary_json_path"]).read_text(encoding="utf-8"))
        assert summary["forecast"]["physics_summary"] == {}

    def test_the_score_file_says_what_it_is(self, run) -> None:
        summary = json.loads(Path(run["summary_json_path"]).read_text(encoding="utf-8"))
        basis = summary["health_scoring"]["basis"]
        assert basis["defect_evidence"] == "heuristic" and basis["calibrated"] is False


def test_a_physics_forecast_is_labelled_as_model_output_from_assumed_inputs(tmp_path) -> None:
    from core.pipeline import PipelineConfig, StructuralFaultPipeline

    images = tmp_path / "imagery"
    images.mkdir()
    for index in range(3):
        _photo_with_crack_like_edges(images / f"PARK_{index}.jpg")
    config = PipelineConfig()
    config.reconstruction_engine = "custom"
    config.propagation_mode = "physics_informed"
    patch = _without_the_crack_model()
    try:
        result = StructuralFaultPipeline(config).run(image_dir=str(images),
                                                     output_dir=str(tmp_path / "out"))
    finally:
        patch.undo()
    payload = result.to_dict()
    text = Path(payload["report_markdown_path"]).read_text(encoding="utf-8")
    assert "assumed inputs, not measured on site" in text
    assert "Factor of safety: `0.000`" not in text and "0.000000 m" not in text
    summary = json.loads(Path(payload["summary_json_path"]).read_text(encoding="utf-8"))
    physics = summary["forecast"]["physics_summary"]
    if physics and float(physics.get("final_crack_length_m", 0.0)) > 0.0:
        assert "Modelled crack length" in text
    else:
        assert "Fracture-mechanics output: not computed" in text
