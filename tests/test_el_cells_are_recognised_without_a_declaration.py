"""EL cells keep the concrete models off them even when nobody declared them.

The round-3 verifier ran three PVEL-AD electroluminescence cells through Run AI with
the UI defaults -- Structure 'generic', Solar imagery 'auto' -- and the concrete crack
and structural models read every cell (4 Spallation findings with the weights
installed; a heuristic structural 1.0 / 'Critical' without). The imagery kind was only
ever taken from a declaration, so the default path never knew these were EL cells.

It also found that a run in which no defect model read anything still printed a
severity index of 38, 'Level 3 / Fair', from the coverage penalty alone, and that the
report table printed 0.0000 crack ratio and 0 structural hits for images nothing
looked at.

Now an undeclared grey image with the signs of an EL cell (busbars, a black frame, a
near-black field, chamfered corners) keeps the surface models off, with the reason in
the row and the finished-job message; Structural detection 'on' overrides it. A run
with no assessed defect component has no index, score or grade. Not-run figures are
None in the summary and '-' in the report.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONCRETE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"
IR_CROP = Path(__file__).parent / "fixtures" / "ir_module_hot_spot.jpg"


def _data_root() -> Path:
    root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
    return Path(root) if root else REPO_ROOT / "training" / "data"


def _el_cells(count: int = 3) -> list[Path]:
    folder = _data_root() / "prepared" / "pvel_ad_det" / "test" / "images"
    if not folder.is_dir():
        pytest.skip(f"{folder} is not on this machine (set ODK_TRAINING_DATA_DIR)")
    return sorted(folder.iterdir())[:count]


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    assert instance.create_project("el default", str(tmp_path / "project"))["ok"]
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


def _copy(paths: list[Path], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy(p, folder / p.name)
    return folder


# ---------------------------------------------------------------- the signature


class TestTheSignature:
    def test_el_cells_carry_it(self):
        from core.detection import el_cell_signature

        for cell in _el_cells(10):
            sig = el_cell_signature(cv2.imread(str(cell)))
            assert sig["el_cell"], (cell.name, sig)

    def test_a_concrete_photo_and_an_ir_crop_do_not(self):
        from core.detection import el_cell_signature

        assert not el_cell_signature(cv2.imread(str(CONCRETE)))["el_cell"]
        # 24x40 px infrared module crops are grey too, and are not EL cells.
        assert not el_cell_signature(cv2.imread(str(IR_CROP)))["el_cell"]

    def test_grey_concrete_corpora_rarely_carry_it(self):
        """Grey concrete and pavement tiles are common (a quarter of crack_seg, nearly
        half of SDNET2018): grey alone must not keep the crack model off them."""
        from core.detection import _is_grayscale, el_cell_signature

        root = _data_root()
        files = (glob.glob(str(root / "prepared" / "crack_seg" / "test" / "images" / "*"))
                 + glob.glob(str(root / "prepared" / "crack_cls" / "test" / "*" / "*"))
                 + glob.glob(str(root / "sdnet2018" / "*" / "*" / "*")))
        if not files:
            pytest.skip("no concrete corpus on this machine (set ODK_TRAINING_DATA_DIR)")
        random.Random(0).shuffle(files)
        grey = flagged = 0
        for f in files[:900]:
            image = cv2.imread(f)
            if image is None or not _is_grayscale(image):
                continue
            grey += 1
            flagged += bool(el_cell_signature(image)["el_cell"])
        assert grey >= 50, grey
        assert flagged / grey < 0.02, (flagged, grey)

    def test_the_solar_pass_reads_an_undeclared_el_cell_as_one(self):
        """Structure solar with Solar imagery auto used to refuse every grey image."""
        from core.detection import detect_solar_defects, el_cell_signature

        # Only the busbar and black-frame signs pick a model; a near-black field alone
        # keeps the concrete models off but is not enough to choose the EL detector.
        strong = 0
        for cell in _el_cells(10):
            image = cv2.imread(str(cell))
            result = detect_solar_defects(image, imagery="auto")
            if el_cell_signature(image)["strong"]:
                strong += 1
                assert result.imagery == "el_cell", (cell.name, result.refused_reason)
            else:
                assert result.imagery == "auto" and result.refused, cell.name
        assert strong >= 5, strong


# ---------------------------------------------------------------- health and report


class TestNothingAssessedMeansNoScore:
    def test_no_defect_component_gives_no_index_or_grade(self):
        from core.health_scoring import evaluate_asset_health

        rows = [{"crack_ratio": None, "crack_segmentation_run": False,
                 "structural_run": False, "solar_model_used": "not_run",
                 "metal_model_used": "not_run"}]
        health = evaluate_asset_health(asset_id="a", run_utc="2026-01-01T00:00:00Z",
                                       structure_type="generic", material_type="concrete",
                                       image_rows=rows, forecast_payload={},
                                       reconstruction_payload={}, coverage_payload={})
        assert health["severity_index"] is None
        assert health["integrity_score"] is None
        assert health["structural_risk_score"] is None
        assert health["condition_level"] is None
        assert health["condition_label"] == "Not assessed"
        assert health["assessed"] is False

    def test_a_not_assessed_run_is_not_a_point_on_the_trend(self):
        from core.health_scoring import _history_entry_from_summary

        assert _history_entry_from_summary({"run_utc": "2026-01-01T00:00:00Z",
                                            "health_scoring": {"severity_index": None}}) is None

    def test_crack_averages_skip_images_nothing_looked_at(self):
        from core.health_scoring import _aggregate_image_metrics

        rows = [{"crack_ratio": 0.02, "crack_length_px": 100.0},
                {"crack_ratio": None, "crack_length_px": None,
                 "crack_segmentation_run": False}]
        metrics = _aggregate_image_metrics(rows)
        assert metrics["crack_ratio_avg"] == pytest.approx(0.02)


# ---------------------------------------------------------------- the pipeline


def _assert_defaults_kept_off(result: dict, summary: dict, api) -> None:
    for row in summary["images"]:
        assert row["surface_models_not_run"] == "not_run:el_cell_imagery_inferred", row
        assert row["structural_run"] is False
        assert "structural_detections" not in row
        assert row["crack_segmentation_run"] is False
        assert row["crack_ratio"] is None and row["crack_length_px"] is None
        assert row["imagery_inferred"]["kind"] == "el_cell"
        assert "looks like an electroluminescence" in row["surface_models_not_run_reason"]
    found = api.find_annotations()["annotations"]
    assert not [f for f in found if str(f.get("label") or "").lower()
                in {"spallation", "exposedbars", "crack", "efflorescence", "corrosionstain"}]
    health = summary["health_scoring"]
    assert health["severity_index"] is None and health["condition_level"] is None
    assert result["asset_integrity_score"] is None and result["condition_level"] is None
    assert result["health_not_assessed"]
    assert result["surface_models_not_run"][0]["images"] == len(summary["images"])
    report = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Screening severity index: not assessed" in report
    table = [ln for ln in report.splitlines()
             if any(ln.startswith(f"| {r['image']} ") for r in summary["images"])]
    assert len(table) == len(summary["images"])
    for line in table:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1] == "-" and cells[2] == "-" and cells[5] == "-", line


class TestRunAIWithTheDefaults:
    def test_el_cells_are_kept_off_the_concrete_models(self, api, tmp_path):
        folder = _copy(_el_cells(), tmp_path / "el")
        result, summary = _run(api, folder)
        _assert_defaults_kept_off(result, summary, api)

    def test_a_concrete_photo_still_gets_the_crack_model(self, api, tmp_path):
        folder = _copy([CONCRETE], tmp_path / "concrete")
        _, summary = _run(api, folder)
        row = summary["images"][0]
        assert "surface_models_not_run" not in row
        assert row["crack_segmentation_run"] is True
        assert not str(row["crack_model_used"]).startswith("not_run")
        assert summary["health_scoring"]["severity_index"] is not None

    def test_structural_detection_on_overrides_the_inference(self, api, tmp_path):
        folder = _copy(_el_cells(1), tmp_path / "el")
        _, summary = _run(api, folder, run_structural_detection=True)
        row = summary["images"][0]
        assert "surface_models_not_run" not in row
        assert row["structural_run"] is True and row["crack_segmentation_run"] is True


def test_run_ai_on_screen_with_the_defaults_keeps_el_cells_off_the_concrete_models(tmp_path):
    """The user path: import EL cells, leave Run AI Settings alone, Run AI."""
    pytest.importorskip("websocket")
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from live_ui import LiveUI

    api = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    assert api.create_project("el default ui", str(tmp_path / "project"))["ok"]
    images = _copy(_el_cells(), tmp_path / "images")
    assert api.import_dataset(str(images), "el cells")["ok"]
    ui = LiveUI(api, "inspection")
    try:
        settings = ui.js("""(() => {
            const field = (label) => [...document.querySelectorAll('.field')]
              .find((f) => f.querySelector('label').textContent === label).querySelector('select');
            return [field('Structure').value, field('Solar imagery').value]; })()""")
        assert settings == ["generic", "auto"], settings
        ui.js("T.tool('Run AI')")
        ui.js("T.ok('Confirm')")
        message = ui.js("T.toast('finished|failed', 1800000)")
        assert "finished" in message, message
        assert "crack/structural models not run on 3 image(s)" in message, message
        assert "looks like an electroluminescence" in message, message
        assert "no health score" in message, message
        rows = ui.js("T.table('Confidence') || []")
        classes = {str(r[1]).lower() for r in rows}
        sources = {r[3] for r in rows}
        assert "structural_multiclass_detector" not in sources, rows
        assert not classes & {"spallation", "exposedbars"}, rows
    finally:
        ui.close()
