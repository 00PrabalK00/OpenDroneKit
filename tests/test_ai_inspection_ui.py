"""AI Inspection, clicked the way a user clicks it, against the real Api.

The audit found that the models worked when called directly and the product around them
did not: nothing reached the labelling canvas, the Findings table never showed a model
finding, a review left the table showing what it showed before, and nothing on screen
said which detectors were classical heuristics. Python tests of the Api prove the calls
work; these prove the buttons reach them and the screen shows what came back.

Headless Chromium drives the real workspace page (tests/live_ui.py); only the
pywebview transport is replaced. The pre-label test runs the installed structural
detector and skips where its gitignored weights are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from core.models import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"
IMAGE = "000_concrete.jpg"


def _installed(key: str) -> bool:
    path = resolve_model_path(key)
    return bool(path) and path.is_file()


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from live_ui import LiveUI

    root = tmp_path_factory.mktemp("ai_inspection_ui")
    api = Api(AppSession(ProjectStore(root / "odk.db")))
    assert api.create_project("ai inspection ui", str(root / "project"))["ok"]
    images = root / "images"
    images.mkdir()
    shutil.copy(FIXTURE, images / IMAGE)
    assert api.import_dataset(str(images), "photos")["ok"]
    ui = LiveUI(api, "inspection")
    try:
        yield api, ui
    finally:
        ui.close()


class TestTheDetectorsPanelNamesTheHeuristics:
    def test_metal_and_solar_are_marked_heuristic(self, session):
        """One global model count said nothing about metal and solar being rules. Both
        now answer from a model or refuse; neither is ever listed as a heuristic."""
        _, ui = session
        text = ui.js("T.until(() => [...document.querySelectorAll('.tree')]"
                     ".map((t) => t.textContent).join(' ').includes('corrosion_severity') && "
                     "[...document.querySelectorAll('.tree')].map((t) => t.textContent).join(' '),"
                     " 240000, 'the detectors list')")
        assert "structuralmodel" in text or "structuralHEURISTIC" in text
        assert "metalHEURISTIC" not in text and "solarHEURISTIC" not in text
        assert ("metalmodel" if _installed("corrosion_severity_segmentation")
                else "metalunavailable") in text


class TestTheLabellingCanvasIsReachable:
    def test_boxes_drawn_in_the_dialog_are_saved_to_the_project(self, session):
        api, ui = session
        ui.js("T.tool('Label')")
        ui.js(f"T.choose('Label which image?', {json.dumps(IMAGE)})")
        ui.js("T.fill('Classes', ['crack, spalling'])")
        ui.js("T.waitModal('Label ').then(() => T.until(() => T.modal().labeller.image, 30000, 'image'))")
        drawn = ui.js("""(() => {
            const m = T.modal(); const canvas = m.querySelector('canvas');
            T.drag(canvas, 30, 30, 200, 120);
            const pick = m.querySelector('select');
            pick.value = 'spalling'; pick.dispatchEvent(new Event('change'));
            T.drag(canvas, 250, 150, 400, 260);
            return m.labeller.boxes.map((b) => b.label); })()""")
        assert drawn == ["crack", "spalling"]
        ui.js("T.ok('Label ')")
        assert "Saved 2 box(es)" in ui.js("T.toast('Saved')")

        stored = api.list_label_boxes()
        assert stored["images"] == 1 and stored["classes"] == ["crack", "spalling"]
        for box in stored["entries"][0]["boxes"]:
            # Normalised to the image, whatever size the dialog drew it at.
            assert 0 <= box["x"] < 1 and 0 < box["width"] <= 1

    def test_build_corpus_says_why_two_boxes_are_not_a_corpus(self, session):
        _, ui = session
        ui.js("T.tool('Build Corpus')")
        ui.js("T.ok('Confirm')")
        assert "Too few boxes" in ui.js("T.toast('Too few|Corpus at|Build Corpus: ')")


@pytest.mark.skipif(not _installed("structural_multiclass_detector"),
                    reason="structural_multiclass_detector weights not on this machine")
class TestAPrelabelIsReviewedOnScreen:
    def test_prelabel_findings_appear_and_a_reject_shows_without_reopening(self, session):
        api, ui = session
        ui.js("T.tool('Prelabel')")
        ui.js("T.choose('Pre-label with which detector?', 'structural_multiclass_detector')")
        ui.js(f"T.choose('Which image?', {json.dumps(IMAGE)})")
        message = ui.js("T.toast('waiting for review|Prelabel:', 600000)")
        assert "waiting for review" in message, message

        rows = ui.js("T.until(() => T.table('Confidence'), 30000, 'findings')")
        found = api.find_annotations()["annotations"]
        assert len(rows) == len(found) > 0
        first = rows[0]
        # Class, the model's own score, the model, awaiting review, out of the report.
        assert first[2] != "—" and 0 < float(first[2]) <= 1
        assert first[3] == "structural_multiclass_detector"
        assert first[4:] == ["in_review", "no"]

        ui.js("T.clickRow('Confidence', 0)")
        ui.js("T.tool('Reject')")
        assert "dismissed" in ui.js("T.toast('Rejected')")
        # The table re-reads the project after the action; it used to keep showing
        # in_review until the workspace was reopened.
        after = ui.js("T.until(() => { const r = T.table('Confidence'); "
                      "return r && r[0][4] === 'dismissed' && r; }, 30000, 'the table to update')")
        assert after[0][5] == "no"
        details = ui.js("T.until(() => T.props().some((p) => p.startsWith('reject')) && T.props(),"
                        " 30000, 'the review history')")
        assert any(p.startswith("Review history (1)") for p in details)
        assert any(p.startswith("Modelstructural_multiclass_detector") for p in details)


@pytest.mark.skipif(not _installed("structural_multiclass_detector"),
                    reason="structural_multiclass_detector weights not on this machine")
class TestRunAISendsWhatTheAssetIs:
    def test_a_solar_asset_declared_on_screen_runs_the_solar_pass(self, session):
        """Run AI sent only engine, profile and dense, so no screen could declare a solar
        or steel asset and the solar and metal passes were unreachable. The Run AI
        Settings panel declares it. Solar refuses a photograph (no model reads one) and
        the finished-job message says so and why; metal is the corrosion model."""
        api, ui = session
        ui.js("""(() => {
            const field = (label) => [...document.querySelectorAll('.field')]
              .find((f) => f.querySelector('label').textContent === label).querySelector('select');
            for (const [label, value] of [['Structure', 'solar'], ['Metal detection', 'on']]) {
              const pick = field(label); pick.value = value;
              pick.dispatchEvent(new Event('change'));
            }
            return true; })()""")
        ui.js("T.tool('Run AI')")
        ui.js("T.ok('Confirm')")
        message = ui.js("T.toast('finished|failed', 1800000)")
        assert "finished" in message, message
        if not (_installed("solar_defect_detector") or _installed("solar_pv_multidefect_detector")):
            # A photograph: no solar model reads it, and the message says so and why.
            assert "solar refused on 1 image(s): No solar model for RGB photographs" in message
        assert "ran as a heuristic" not in message

        runs = sorted((Path(api._session.project_root()) / "analysis").glob("run_*/summary.json"))
        summary = json.loads(runs[-1].read_text(encoding="utf-8"))
        assert summary["health_scoring"]["structure_type"] == "solar"
        row = summary["images"][0]
        assert row["solar_model_used"] != "not_run"
        assert row["metal_model_used"] == (
            "onnx:corrosion_severity_segmentation.onnx"
            if _installed("corrosion_severity_segmentation") else "refused")

    def test_the_declared_solar_imagery_reaches_the_solar_pass(self, session):
        """Solar imagery = el_cell on a colour photo: the EL model is not run on it, and
        the message names that refusal -- proof the setting reaches the pipeline."""
        api, ui = session
        ui.js("""(() => {
            const field = (label) => [...document.querySelectorAll('.field')]
              .find((f) => f.querySelector('label').textContent === label).querySelector('select');
            for (const [label, value] of [['Structure', 'solar'], ['Solar imagery', 'el_cell']]) {
              const pick = field(label); pick.value = value;
              pick.dispatchEvent(new Event('change'));
            }
            return true; })()""")
        ui.js("T.tool('Run AI')")
        ui.js("T.ok('Confirm')")
        # Matched on this run's refusal, so an earlier run's toast cannot answer.
        message = ui.js("T.toast('Declared as el cell|Job [0-9a-f]+ failed', 1800000)")
        assert "finished" in message, message
        assert "solar refused on 1 image(s): Declared as el cell imagery" in message
        runs = sorted((Path(api._session.project_root()) / "analysis").glob("run_*/summary.json"))
        row = json.loads(runs[-1].read_text(encoding="utf-8"))["images"][0]
        assert row["solar_imagery"] == "el_cell" and row["solar_model_used"] == "refused"
