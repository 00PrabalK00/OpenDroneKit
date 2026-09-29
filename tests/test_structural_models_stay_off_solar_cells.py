"""The concrete structural detector does not read solar EL cells.

The re-audit ran Run AI on three PVEL-AD electroluminescence cells with the asset
declared solar and the imagery declared el_cell. The solar pass ran the EL model, as it
should -- and structural_multiclass_detector, trained on concrete, also ran on every
cell and reported Spallation and ExposedBars at 0.27-0.45. Those became findings in the
review queue, and health scoring put structural_component at 1.0 and the asset at
'Critical', one of the three being a PVEL-AD 'good' cell. An out-of-domain answer that
reads exactly like a real one is the thing this project must not produce.

The surface models (crack presence, crack segmentation, structural, metal) now stay off
an image the solar pass reads as an EL cell or IR module crop, and off a declared solar
asset unless Structural detection is switched on; the row, the health score and the
finished-job message all say they were not run, which is not "no defects".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import pytest

from core.models import resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONCRETE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"
FAULT_LABELS = {"spallation", "exposedbars", "crack", "efflorescence", "corrosionstain"}


def _installed(key: str) -> bool:
    path = resolve_model_path(key)
    return bool(path) and path.is_file()


def _el_cells(count: int = 3) -> list[Path]:
    root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
    base = Path(root) if root else REPO_ROOT / "training" / "data"
    folder = base / "prepared" / "pvel_ad_det" / "test" / "images"
    if not folder.is_dir():
        pytest.skip(f"{folder} is not on this machine (set ODK_TRAINING_DATA_DIR)")
    return sorted(folder.iterdir())[:count]


needs_models = pytest.mark.skipif(
    not (_installed("structural_multiclass_detector") and _installed("solar_cell_defect_detector")),
    reason="structural_multiclass_detector / solar_cell_defect_detector weights not on this machine")


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    assert instance.create_project("solar el", str(tmp_path / "project"))["ok"]
    return instance


def _run(api, folder: Path, **options) -> tuple[dict, dict]:
    started = api.run_pipeline({"image_dir": str(folder), "engine": "auto",
                                "profile": "fast", **options})
    assert started["ok"], started
    deadline = time.time() + 1800
    while time.time() < deadline:
        job = api.job_status(started["job_id"])["job"]
        if job.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)
    assert job["status"] == "done", job.get("error")
    result = job["result"]
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    return result, summary


def _structural_findings(api) -> list[dict]:
    found = api.find_annotations()["annotations"]
    return [f for f in found
            if "structural_multiclass_detector" in json.dumps(f)
            or str(f.get("label") or f.get("cls") or "").lower() in {"spallation", "exposedbars"}]


def _assert_el_run_is_clean(result: dict, summary: dict, api) -> None:
    for row in summary["images"]:
        assert row["structural_run"] is False, row
        assert row["structural_model_used"] == "not_run:el_cell_imagery"
        assert "structural_detections" not in row and "structural_ratio" not in row
        assert row["crack_segmentation_run"] is False
        assert row["crack_model_used"] == "not_run:el_cell_imagery"
        assert "electroluminescence" in row["surface_models_not_run_reason"]
        # The EL model still read every cell.
        assert row["solar_model_used"] == "onnx:solar_cell_defect_detector.onnx"
    assert not _structural_findings(api), _structural_findings(api)
    components = summary["health_scoring"]["components"]
    assert components["structural_component"] is None
    assert components["crack_component"] is None
    assert {"structural", "crack"} <= set(components["not_assessed"])
    assert components["solar_component"] is not None
    assert summary["health_scoring"]["basis"]["defect_evidence"] == "not_assessed"
    assert result["detectors_run"]["structural"] == ["not_run:el_cell_imagery"]
    assert result["surface_models_not_run"][0]["images"] == len(summary["images"])
    report = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
    assert "no crack or structural model read these images" in report
    assert "classical image processing" not in report


class TestTheAssetDecidesWhichModelsRun:
    def test_structural_follows_the_asset(self):
        from core.pipeline import PipelineConfig, StructuralFaultPipeline

        assert StructuralFaultPipeline(PipelineConfig())._structural_enabled()
        assert not StructuralFaultPipeline(PipelineConfig(structure_type="solar"))._structural_enabled()
        # Still reachable: a solar asset's photographs (its footings, say) on request.
        assert StructuralFaultPipeline(PipelineConfig(
            structure_type="solar", run_structural_detection=True))._structural_enabled()
        # Declaring EL cells makes the solar pass run even with the structure unset.
        assert StructuralFaultPipeline(PipelineConfig(solar_imagery="el_cell"))._solar_enabled()

    def test_health_does_not_score_what_was_not_read(self):
        from core.health_scoring import evaluate_asset_health

        rows = [{"crack_ratio": 0.0, "crack_segmentation_run": False, "structural_run": False,
                 "solar_ratio": 0.0, "solar_model_used": "onnx:x.onnx"}]
        health = evaluate_asset_health(asset_id="a", run_utc="2026-01-01T00:00:00Z",
                                       structure_type="solar", material_type="concrete",
                                       image_rows=rows, forecast_payload={},
                                       reconstruction_payload={}, coverage_payload={})
        assert health["components"]["structural_component"] is None
        assert health["components"]["crack_component"] is None
        assert health["components"]["solar_component"] == 0.0


@needs_models
class TestRunAIOnELCells:
    def test_the_structural_detector_makes_no_findings_on_el_cells(self, api, tmp_path):
        folder = tmp_path / "el"
        folder.mkdir()
        for cell in _el_cells():
            shutil.copy(cell, folder / cell.name)
        result, summary = _run(api, folder, structure_type="solar", solar_imagery="el_cell")
        _assert_el_run_is_clean(result, summary, api)

    def test_structural_on_still_runs_on_a_solar_assets_photograph(self, api, tmp_path):
        """Switching Structural detection on is honoured on photographs of a solar
        asset -- the feature is not removed, only kept off imagery it cannot read."""
        folder = tmp_path / "footing"
        folder.mkdir()
        shutil.copy(CONCRETE, folder / "footing.jpg")
        _, summary = _run(api, folder, structure_type="solar", run_structural_detection=True)
        row = summary["images"][0]
        assert row["structural_run"] is True
        assert row["structural_model_used"] == "onnx:structural_multiclass_detector.onnx"
        assert row["crack_segmentation_run"] is True
        # Off by default on the same declared-solar photograph, and said so.
        _, summary = _run(api, folder, structure_type="solar")
        row = summary["images"][0]
        assert row["structural_run"] is False
        assert row["structural_model_used"] == "not_run:solar_asset"


@needs_models
def test_run_ai_on_screen_makes_no_structural_findings_on_el_cells(tmp_path):
    """The user path: Run AI Settings says solar / el_cell, Run AI, read the message
    and the Findings table."""
    pytest.importorskip("websocket")
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from live_ui import LiveUI

    api = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    assert api.create_project("solar el ui", str(tmp_path / "project"))["ok"]
    images = tmp_path / "images"
    images.mkdir()
    for cell in _el_cells():
        shutil.copy(cell, images / cell.name)
    assert api.import_dataset(str(images), "el cells")["ok"]
    ui = LiveUI(api, "inspection")
    try:
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
        message = ui.js("T.toast('finished|failed', 1800000)")
        assert "finished" in message, message
        assert "crack/structural models not run on 3 image(s)" in message, message
        assert "electroluminescence" in message

        rows = ui.js("T.until(() => T.table('Confidence'), 30000, 'findings')")
        assert rows, "the EL model's findings should be listed"
        sources = {r[3] for r in rows}
        classes = {str(r[1]).lower() for r in rows}
        assert "structural_multiclass_detector" not in sources, rows
        assert not classes & {"spallation", "exposedbars"}, rows
        assert "solar_cell_defect_detector" in sources, rows

        runs = sorted((Path(api._session.project_root()) / "analysis").glob("run_*/summary.json"))
        summary = json.loads(runs[-1].read_text(encoding="utf-8"))
        components = summary["health_scoring"]["components"]
        assert components["structural_component"] is None
        assert "structural" in components["not_assessed"]
    finally:
        ui.close()
