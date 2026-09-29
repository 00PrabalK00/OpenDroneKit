"""Solar and metal defect detection answer from trained models, or refuse.

Both used to be classical rules dressed as detectors: the solar pass marked 963
"defects" on a farmland photo, and the metal pass marked a third of a road as metal
damage. Labelling that output `heuristic` did not make it a solar or metal detector.

Now:
- solar goes to the installed model trained on the imagery it is given --
  solar_cell_defect_detector for electroluminescence cell images,
  solar_thermal_anomaly_classifier for one infrared module per image -- and refuses,
  with the reason, for RGB photographs (no model reads them), for imagery a model was
  not trained on, and when the model is absent;
- metal is read by corrosion_severity_segmentation (every pixel graded above `good`),
  and refuses when that model is absent.

The model tests run on real inputs with ground truth where it exists: the PVEL-AD EL
test split and the Corrosion Condition State test split (both gitignored under
training/data, or where ODK_TRAINING_DATA_DIR points), and two MIT-licensed
InfraredSolarModules crops committed as fixtures. They skip where weights or data are
absent; the refusal tests never skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import pytest

from core.models import resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
PHOTO = FIXTURES / "concrete_cracked_public_domain.jpg"
IR_HOT_SPOT = FIXTURES / "ir_module_hot_spot.jpg"
IR_NO_ANOMALY = FIXTURES / "ir_module_no_anomaly.jpg"
# The prepared PVEL-AD split's class order (sorted), which is NOT the checkpoint's.
PVEL_NAMES = ["black_core", "crack", "finger", "horizontal_dislocation", "short_circuit",
              "star_crack", "thick_line", "vertical_dislocation"]


def _installed(key: str) -> bool:
    path = resolve_model_path(key)
    return bool(path) and path.is_file()


def needs(*keys: str):
    missing = [k for k in keys if not _installed(k)]
    return pytest.mark.skipif(bool(missing), reason=f"weights not on this machine: {missing}")


def _prepared(name: str) -> Path:
    root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
    base = Path(root) if root else REPO_ROOT / "training" / "data"
    folder = base / "prepared" / name / "test"
    if not (folder / "images").is_dir():
        pytest.skip(f"{folder} is not on this machine (set ODK_TRAINING_DATA_DIR)")
    return folder


def _load(path: Path) -> np.ndarray:
    from core.detection import load_image

    return load_image(path)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class TestSolarRefusesWhatNoModelReads:
    def test_an_rgb_photograph_is_refused_with_the_reason(self):
        from core.detection import detect_solar_defects

        if _installed("solar_defect_detector") or _installed("solar_pv_multidefect_detector"):
            pytest.skip("an RGB solar model is installed here")
        result = detect_solar_defects(_load(PHOTO))
        assert result.refused and result.model_used == "refused"
        assert result.imagery == "rgb_photo"
        assert "No solar model for RGB photographs" in result.refused_reason
        # Refused means not looked at: no regions, no mask, no identity.
        assert result.detections == [] and int(result.mask.sum()) == 0
        assert result.model_key == "" and result.model_sha256 == ""

    def test_there_is_no_model_free_path(self):
        from core.detection import detect_solar_defects

        result = detect_solar_defects(_load(PHOTO), use_model=False)
        assert result.refused and result.detections == []
        assert "no model-free solar detector" in result.refused_reason

    def test_the_el_model_is_not_run_on_a_photograph(self):
        """Its registry entry says it must not be applied to photographs. Before, asked
        by key, it ran on the concrete photo and its answer read like a real one."""
        from core.detection import detect_solar_defects

        for kwargs in ({"model_key": "solar_cell_defect_detector"}, {"imagery": "el_cell"}):
            result = detect_solar_defects(_load(PHOTO), **kwargs)
            assert result.refused, kwargs
            assert "colour" in result.refused_reason

    def test_a_grey_image_of_unknown_kind_asks_what_it_is(self):
        from core.detection import detect_solar_defects

        grey = cv2.cvtColor(cv2.cvtColor(_load(PHOTO), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result = detect_solar_defects(grey)
        assert result.refused and "Declare the solar imagery" in result.refused_reason

    def test_a_whole_thermal_frame_is_not_classified_as_one_module(self):
        from core.detection import detect_solar_defects

        frame = cv2.resize(_load(IR_HOT_SPOT), (640, 512))
        result = detect_solar_defects(frame, imagery="thermal_module")
        assert result.refused and "ONE module" in result.refused_reason

    def test_a_missing_model_is_a_refusal_not_a_fallback(self, monkeypatch):
        import core.detection as detection

        real = detection.model_status
        monkeypatch.setattr(detection, "model_status", lambda key: (
            {"key": key, "exists": False} if key == "solar_thermal_anomaly_classifier"
            else real(key)))
        result = detection.detect_solar_defects(_load(IR_HOT_SPOT), imagery="thermal_module")
        assert result.refused and "not installed" in result.refused_reason


class TestSolarRunsTheModelForItsImagery:
    @needs("solar_cell_defect_detector")
    def test_el_cells_are_read_by_the_el_detector_and_match_ground_truth(self):
        """PVEL-AD test split: the detector's boxes land on the annotated defects, with
        the annotated class, on at least 80% of them (23 of 24 when this was written)."""
        from core.detection import detect_solar_defects

        split = _prepared("pvel_ad_det")
        found = total = 0
        for image_path in sorted((split / "images").iterdir())[:12]:
            image = _load(image_path)
            h, w = image.shape[:2]
            result = detect_solar_defects(image, imagery="el_cell")
            assert result.model_used == "onnx:solar_cell_defect_detector.onnx"
            assert result.model_key == "solar_cell_defect_detector"
            assert len(result.model_sha256) == 64
            assert "electroluminescence" in result.scope
            assert all(0 < hit.confidence <= 1 for hit in result.detections)
            for line in (split / "labels" / f"{image_path.stem}.txt").read_text().split("\n"):
                if not line.strip():
                    continue
                cls, cx, cy, bw, bh = map(float, line.split())
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                total += 1
                found += any(_iou(box, hit.bbox_xyxy) >= 0.3 and hit.label == PVEL_NAMES[int(cls)]
                             for hit in result.detections)
        assert total >= 10
        assert found / total >= 0.8, f"{found}/{total}"

    @needs("solar_thermal_anomaly_classifier")
    def test_an_infrared_module_crop_is_classified(self):
        """Two real crops (their dataset labels: Hot-Spot, No-Anomaly). These crops are
        from the classifier's own corpus, so this proves the path, not generalisation."""
        from core.detection import detect_solar_defects

        hot = detect_solar_defects(_load(IR_HOT_SPOT), imagery="thermal_module")
        assert hot.model_used == "onnx:solar_thermal_anomaly_cls.onnx"
        assert [hit.label for hit in hot.detections] == ["Hot-Spot"]
        assert hot.detections[0].confidence > 0.9
        assert abs(sum(hot.class_probabilities.values()) - 1.0) < 1e-6
        assert "PER-MODULE" in hot.scope

        sound = detect_solar_defects(_load(IR_NO_ANOMALY), imagery="thermal_module")
        assert sound.model_used.startswith("onnx:") and not sound.refused
        assert sound.detections == [] and sound.defect_ratio == 0.0

    @needs("solar_thermal_anomaly_classifier")
    def test_a_radiometric_array_goes_to_the_thermal_classifier(self):
        from core.detection import detect_solar_defects

        crop = _load(IR_HOT_SPOT)
        celsius = 20.0 + crop[..., 0].astype(np.float32) * 0.2
        result = detect_solar_defects(crop, thermal_gray=celsius)
        assert result.imagery == "thermal_module"
        assert [hit.label for hit in result.detections] == ["Hot-Spot"]


class TestMetalIsTheCorrosionModel:
    @needs("corrosion_severity_segmentation")
    def test_defect_pixels_match_the_annotated_corrosion(self):
        """Corrosion Condition State test split: mean IoU of the defect mask (every grade
        above `good`) against the annotated corrosion is 0.55 over these ten images."""
        from core.detection import detect_metal_defects

        split = _prepared("corrosion_seg")
        ious = []
        for image_path in sorted((split / "images").iterdir())[:10]:
            result = detect_metal_defects(_load(image_path))
            assert result.model_used == "onnx:corrosion_severity_segmentation.onnx"
            assert result.model_key == "corrosion_severity_segmentation"
            assert len(result.model_sha256) == 64
            truth = cv2.imread(str(split / "masks" / image_path.name), cv2.IMREAD_UNCHANGED) > 0
            predicted = result.mask > 0
            union = (truth | predicted).sum()
            ious.append((truth & predicted).sum() / union if union else 1.0)
        assert float(np.mean(ious)) >= 0.45, ious

    @needs("corrosion_severity_segmentation")
    def test_the_grades_travel_with_the_result(self):
        from core.detection import detect_metal_defects

        split = _prepared("corrosion_seg")
        result = detect_metal_defects(_load(sorted((split / "images").iterdir())[0]))
        assert set(result.class_pixel_ratio) == {"good", "fair", "poor", "severe"}
        assert result.worst_present_grade in {"fair", "poor", "severe"}
        assert result.defect_regions >= 1
        assert result.defect_ratio == pytest.approx(
            1.0 - result.class_pixel_ratio["good"], abs=0.05)

    def test_a_missing_model_is_a_refusal(self):
        from core.detection import detect_metal_defects

        result = detect_metal_defects(_load(PHOTO), model_key="no_such_corrosion_model")
        assert result.refused and result.model_used == "refused"
        assert "not in the model registry" in result.refused_reason
        assert int(result.mask.sum()) == 0 and result.defect_regions == 0

    def test_there_is_no_model_free_path(self):
        from core.detection import detect_metal_defects

        result = detect_metal_defects(_load(PHOTO), use_model=False)
        assert result.refused and "texture" in result.refused_reason

    def test_the_texture_rule_is_only_reachable_by_its_own_name(self):
        from core.detection import metal_texture_heuristic

        assert metal_texture_heuristic(_load(PHOTO)).model_used == "heuristic"


# -- Run AI, driven through the Api the way the Run AI Settings panel drives it ---------


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    assert instance.create_project("solar metal", str(tmp_path / "project"))["ok"]
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


class TestRunAI:
    @needs("solar_cell_defect_detector")
    def test_el_imagery_declared_on_screen_runs_the_el_detector(self, api, tmp_path):
        split = _prepared("pvel_ad_det")
        folder = tmp_path / "el"
        folder.mkdir()
        shutil.copy(sorted((split / "images").iterdir())[0], folder / "cell.jpg")
        result, summary = _run(api, folder, structure_type="solar", solar_imagery="el_cell")
        row = summary["images"][0]
        assert row["solar_model_used"] == "onnx:solar_cell_defect_detector.onnx"
        assert row["solar_imagery"] == "el_cell" and row["solar_detection_count"] >= 1
        assert result["detectors_run"]["solar"] == ["onnx:solar_cell_defect_detector.onnx"]
        assert "solar" not in result["detectors_refused"]
        # Model detections become findings for review, with the model named.
        found = api.find_annotations()["annotations"]
        assert any("solar_cell_defect_detector" in json.dumps(f) for f in found)

    @needs("corrosion_severity_segmentation")
    def test_a_photo_declared_solar_and_steel_refuses_solar_and_grades_metal(self, api, tmp_path):
        if _installed("solar_defect_detector") or _installed("solar_pv_multidefect_detector"):
            pytest.skip("an RGB solar model is installed here")
        folder = tmp_path / "photo"
        folder.mkdir()
        shutil.copy(PHOTO, folder / "a.jpg")
        result, summary = _run(api, folder, structure_type="solar", material_type="steel")
        row = summary["images"][0]
        assert row["solar_model_used"] == "refused" and "solar_ratio" not in row
        assert "No solar model for RGB photographs" in row["solar_refused_reason"]
        assert row["metal_model_used"] == "onnx:corrosion_severity_segmentation.onnx"
        assert row["metal_model_key"] == "corrosion_severity_segmentation"
        assert result["detectors_refused"]["solar"]["images"] == 1
        assert "metal" not in result["detectors_refused"]
        assert result["heuristic_images_not_scored"] == {"metal": 0, "solar": 0}
        report = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Solar detection refused on 1 image(s)" in report



def _library_dataset(tmp_path: Path, *images: Path) -> tuple[Path, str]:
    """A project folder with one imported data-library dataset (the processing-run path)."""
    from core.data_library import import_image_dataset

    folder = tmp_path / "images"
    folder.mkdir()
    for index, image in enumerate(images):
        shutil.copy(image, folder / f"{index}{image.suffix}")
    root = tmp_path / "project"
    root.mkdir()
    dataset = import_image_dataset(root, "p1", folder, generate_thumbnails=False)
    return root, dataset.id


class TestProcessingRunClassicalLegHasNoScore:
    def test_classical_metal_regions_carry_no_confidence(self, tmp_path):
        """The processing run's classical leg used to score the texture rule as
        min(0.95, 0.4 + 50 * ratio). It is a rule, not a model: its regions are 0.0 with
        the ratio in the notes, and a whole-frame hit is dropped."""
        from core.defect_engine import DefectDetectionConfig, run_defect_detection

        # A flat plate with one rough patch: the texture rule fires on the patch only.
        plate = np.full((400, 600, 3), 128, dtype=np.uint8)
        plate[150:250, 250:350] = np.random.default_rng(0).integers(0, 255, (100, 100, 3))
        patch = tmp_path / "patch.png"
        cv2.imwrite(str(patch), plate)
        noise = tmp_path / "noise.png"
        cv2.imwrite(str(noise), np.random.default_rng(1).integers(0, 255, (400, 600, 3),
                                                                   dtype=np.uint8))
        root, dataset_id = _library_dataset(tmp_path, patch, noise)
        result = run_defect_detection(root, dataset_id,
                                      config=DefectDetectionConfig(mode="classical"))
        classical = [d for d in result.defects if d.source == "classical"]
        assert any(d.defect_type == "corrosion" for d in classical)
        for defect in classical:
            assert defect.confidence == 0.0
            assert defect.notes.startswith("heuristic:")
            x1, y1, x2, y2 = defect.bbox
            assert (x2 - x1) * (y2 - y1) < 0.9 * 400 * 600
        # The all-noise frame's whole-frame "hit" locates nothing and is dropped.
        assert not any(Path(d.image_path).stem == "1" for d in classical)


class TestTheSolarWorkflowStageRunsTheSolarModels:
    """The Solar Inspection workflow's `solar_defect_detection` stage used to run the
    STRUCTURAL detector and file its output as solar."""

    @needs("solar_thermal_anomaly_classifier")
    def test_thermal_module_crops_are_classified(self, tmp_path):
        from core.processing_runs import run_solar_defect_stage

        root, dataset_id = _library_dataset(tmp_path, IR_HOT_SPOT, IR_NO_ANOMALY)
        out = run_solar_defect_stage(root, dataset_id, tmp_path / "run",
                                     imagery="thermal_module")
        payload = json.loads(out.read_text(encoding="utf-8"))
        rows = {Path(r["image_path"]).stem: r for r in payload["images"]}
        assert payload["images_refused"] == 0
        assert rows["0"]["model_used"] == "onnx:solar_thermal_anomaly_cls.onnx"
        assert [d["label"] for d in rows["0"]["detections"]] == ["Hot-Spot"]
        assert rows["1"]["detections"] == [] and rows["1"]["model_key"]

    def test_a_photograph_is_refused_not_passed_through_the_structural_model(self, tmp_path):
        from core.processing_runs import run_solar_defect_stage

        if _installed("solar_defect_detector") or _installed("solar_pv_multidefect_detector"):
            pytest.skip("an RGB solar model is installed here")
        root, dataset_id = _library_dataset(tmp_path, PHOTO)
        out = run_solar_defect_stage(root, dataset_id, tmp_path / "run")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["images_refused"] == 1
        row = payload["images"][0]
        assert row["model_used"] == "refused" and row["detections"] == []
        assert "structural" not in json.dumps(payload)
        assert any("No solar model for RGB photographs" in r for r in payload["refused_reasons"])

