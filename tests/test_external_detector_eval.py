"""The shipped detectors, measured on openly licensed external data -- and the plumbing.

tools/eval_external_detectors.py runs the installed models through the app's own
functions on three external datasets (RDD2022 road damage, a Mavic 3T thermal PV set and
the Boga Cayi H20T bridge frames) and writes what it measured to docs/holdout/. These
tests keep that honest:

* the arithmetic (AP, VOC->YOLO conversion) is checked against cases worked by hand and
  against the real RDD annotations;
* the YOLO decoder's letterbox maps boxes back to source pixels, and a graph whose class
  count does not match the registry is refused rather than decoded with wrong names;
* the published evidence names the real model digests and the real dataset counts, and
  the detector rows it reports can be reproduced from the data.

Data comes from ODK_TESTDATA_DIR (the odk-testdata folder; see its INDEX.md) and
ODK_TRAINING_DATA_DIR; weights from the registry or ODK_MODEL_WEIGHTS_DIR. Each test that
needs one SKIPS with the reason when it is absent -- nothing here is a fabricated fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import eval_external_detectors as ev

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = REPO_ROOT / "docs" / "holdout"
REGISTRY = json.loads((REPO_ROOT / "models" / "model_registry.json").read_text(encoding="utf-8"))["models"]


def _testdata() -> Path:
    root = os.environ.get("ODK_TESTDATA_DIR", "").strip()
    if not root or not Path(root).is_dir():
        pytest.skip("ODK_TESTDATA_DIR is not set to the odk-testdata folder on this machine")
    return Path(root)


def _rdd() -> Path:
    root = _testdata()
    if not (root / ev.RDD_FOLDER / "China_Drone").is_dir():
        pytest.skip(f"{ev.RDD_FOLDER} is not under ODK_TESTDATA_DIR")
    return root


def _needs_weights(key: str) -> Path:
    from core.models import resolve_model_path

    path = resolve_model_path(key)
    if not path or not path.is_file():
        pytest.skip(f"{key} weights are not on this machine (set ODK_MODEL_WEIGHTS_DIR)")
    return path


class TestAveragePrecisionArithmetic:
    """Small cases worked by hand: if these drift, every published AP drifts with them."""

    def test_perfect_detections_score_one(self) -> None:
        truths = {"a": [(0, 0, 10, 10)], "b": [(5, 5, 15, 15)]}
        preds = [("a", 0.9, (0, 0, 10, 10)), ("b", 0.8, (5, 5, 15, 15))]
        out = ev.average_precision(preds, truths)
        assert out["ap"] == 1.0 and out["recall"] == 1.0 and out["precision"] == 1.0

    def test_a_higher_scored_false_positive_halves_the_first_step(self) -> None:
        # Ranked: FP (p=0/1), TP (p=1/2, r=1/2), TP (p=2/3, r=1). Interpolated precision
        # is 2/3 over the whole recall range, so AP = 2/3.
        truths = {"a": [(0, 0, 10, 10)], "b": [(0, 0, 10, 10)]}
        preds = [("a", 0.95, (50, 50, 60, 60)), ("a", 0.9, (0, 0, 10, 10)),
                 ("b", 0.8, (0, 0, 10, 10))]
        out = ev.average_precision(preds, truths)
        assert out["ap"] == pytest.approx(2 / 3, abs=1e-4)
        assert (out["true_positives"], out["false_positives"]) == (2, 1)

    def test_a_duplicate_is_a_false_positive(self) -> None:
        truths = {"a": [(0, 0, 10, 10)]}
        preds = [("a", 0.9, (0, 0, 10, 10)), ("a", 0.8, (0, 0, 10, 10))]
        out = ev.average_precision(preds, truths)
        assert (out["true_positives"], out["false_positives"]) == (1, 1)

    def test_a_prediction_on_an_ignore_region_is_not_counted(self) -> None:
        truths = {"a": [(0, 0, 10, 10)]}
        ignore = {"a": [(100, 100, 120, 120)]}
        preds = [("a", 0.99, (100, 100, 120, 120)), ("a", 0.5, (0, 0, 10, 10))]
        out = ev.average_precision(preds, truths, ignore)
        assert out["ap"] == 1.0 and out["false_positives"] == 0

    def test_no_truth_is_not_a_score(self) -> None:
        assert ev.average_precision([("a", 0.9, (0, 0, 1, 1))], {})["ap"] is None


class TestRddConversion:
    def test_every_core_box_survives_voc_to_yolo(self, tmp_path) -> None:
        samples = ev.rdd_samples(_rdd())
        assert len(samples) == 200, "the subset's PROVENANCE.md records 200 images"
        report = ev.voc_to_yolo(samples, tmp_path, "rdd")
        assert report["images"] == 200
        # Counts from PROVENANCE.md: D00 86, D10 63, D20 41, D40 35, and 46 non-core.
        assert report["boxes_written"] == {"D00": 86, "D10": 63, "D20": 41, "D40": 35}
        assert sum(report["labels_dropped_non_core"].values()) == 46
        labels = Path(report["folder"]) / "labels"
        for sample in samples[:40]:
            stem = f"{sample['subset']}_{Path(sample['image']).stem}"
            back = ev.read_yolo_labels(labels / f"{stem}.txt", sample["width"], sample["height"])
            want = [(ev.RDD_CORE.index(n), b) for n, b in sample["objects"] if n in ev.RDD_CORE]
            assert len(back) == len(want)
            for (c1, b1), (c2, b2) in zip(back, want):
                assert c1 == c2
                assert np.allclose(b1, b2, atol=0.01)


class TestYoloDecoder:
    def _labelled_image(self) -> np.ndarray:
        root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
        base = Path(root) if root else REPO_ROOT / "training" / "data"
        images = sorted((base / "prepared" / "structural_det" / "test" / "images").glob("*"))
        if not images:
            pytest.skip("prepared/structural_det/test is not on this machine (ODK_TRAINING_DATA_DIR)")
        return cv2.imread(str(images[0]))

    def test_letterboxed_boxes_land_on_the_source_pixels(self) -> None:
        """A 640x480 frame letterboxed is the same network input as that frame pasted by
        hand into a grey 640x640 canvas 80 px down, so the boxes must agree once the
        hand-made run is shifted back up by 80 px. A wrong scale or offset in the
        letterbox mapping breaks the agreement.
        """
        path = _needs_weights("structural_multiclass_detector")
        from core.detection import _run_onnx_yolo

        square = self._labelled_image()
        assert square.shape[:2] == (640, 640)
        frame = square[80:560]
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[80:560] = frame
        spec = REGISTRY["structural_multiclass_detector"]
        args = (spec["labels"], 0.25, 0.45, 640)
        boxed = _run_onnx_yolo(frame, path, *args)
        by_hand = _run_onnx_yolo(canvas, path, *args)
        if not by_hand:
            pytest.skip("no detections on the first structural test image to compare")
        assert len(boxed) == len(by_hand)
        for hit in by_hand:
            x1, y1, x2, y2 = hit.bbox_xyxy
            want = (x1, max(0, y1 - 80), x2, min(479, y2 - 80))
            assert any(m.label == hit.label and ev.iou(m.bbox_xyxy, want) > 0.95 for m in boxed), (
                f"{hit.label} at {want} is not where the letterboxed run put it")

    def test_a_label_count_the_graph_does_not_have_is_refused(self) -> None:
        path = _needs_weights("structural_multiclass_detector")
        from core.detection import _run_onnx_yolo

        # The graph emits 4 box + 6 class columns. Four labels fit neither the v8 layout
        # (4+4) nor v5 (5+4); the old decoder read the first class column as objectness
        # and named the next four anyway. (Five labels would read as a v5 layout -- 5+5 -- which the
        # shape alone cannot tell apart; the registry's label list has to be right.)
        four = REGISTRY["structural_multiclass_detector"]["labels"][:4]
        with pytest.raises(ValueError, match="fits neither"):
            _run_onnx_yolo(self._labelled_image(), path, four, 0.25, 0.45, 640)


class TestPublishedEvidence:
    """The numbers in docs/holdout belong to the registered files and the real data."""

    @pytest.mark.parametrize("name", ["external_rdd2022_detectors.json",
                                      "external_bridge_h20t_detectors.json"])
    def test_detector_evidence_names_the_registered_digests(self, name) -> None:
        data = json.loads((HOLDOUT / name).read_text(encoding="utf-8"))
        for key, row in data["models"].items():
            assert row["sha256_measured"] == REGISTRY[key]["sha256"], key
            assert row["sha256_matches_registry"] is True

    def test_road_evidence_is_the_whole_subset_and_says_there_is_no_road_model(self) -> None:
        data = json.loads((HOLDOUT / "external_rdd2022_detectors.json").read_text(encoding="utf-8"))
        assert data["dataset"]["images"] == 200
        assert data["dataset"]["core_boxes"] == {"D00": 86, "D10": 63, "D20": 41, "D40": 35}
        assert "no road-damage detector" in data["no_road_model"]
        assert not any("road" in key for key in REGISTRY), (
            "a road model is now registered: re-run the evaluation against it")
        assert "EXPLORATORY" in data["structural_detector_crack_class"]["pothole_D40_vs_Spallation"]["note"]

    def test_pv_evidence_claims_no_accuracy_it_could_not_measure(self) -> None:
        data = json.loads((HOLDOUT / "external_pv_mavic3t_thermal.json").read_text(encoding="utf-8"))
        cls = data["thermal_anomaly_classifier_on_labelled_panels"]
        assert cls["accuracy"] is None
        assert data["panel_detection"]["measured"] is False
        # The file this run loaded did not carry the registered digest (38cd... against
        # 6933...). The evidence must say so rather than borrow the registered identity.
        assert cls["model"]["sha256_registered"] == REGISTRY["solar_thermal_anomaly_classifier"]["sha256"]
        if cls["model"]["sha256_measured"] != cls["model"]["sha256_registered"]:
            assert cls["model"]["sha256_matches_registry"] is False
            assert "do NOT describe it" in cls["identity_warning"]

    def test_bridge_evidence_says_it_is_unlabelled(self) -> None:
        data = json.loads((HOLDOUT / "external_bridge_h20t_detectors.json").read_text(encoding="utf-8"))
        assert data["dataset"]["frames"] == 81
        assert "NONE" in data["dataset"]["labels"]

    def test_bridge_crack_pixels_are_not_claimed_as_cracks(self) -> None:
        """Unlabelled bridge output must be labelled unvalidated, never 'cracks found'."""
        data = json.loads((HOLDOUT / "external_bridge_h20t_detectors.json").read_text(encoding="utf-8"))
        assert data["visual_review"].startswith("UNVALIDATED")
        assert "false positive" in data["visual_review"]
        assert "NOT validated" in data["crack_segmentation"]["native_resolution_note"]
        from docs.features.registry import ALL_FEATURES
        rows = {f.id: f for f in ALL_FEATURES}
        for key in ("ai.crack", "ai.external_measurement"):
            text = rows[key].criteria + " " + rows[key].notes
            assert "only resolves cracks when tiled" not in text
            assert "false positive" in text


class TestTheRoadRowsReproduce:
    def test_detector_rows_for_the_first_images_reproduce(self, tmp_path) -> None:
        """Re-run the first two RDD images and get the published per-image detector rows."""
        root = _rdd()
        for key in ("structural_multiclass_detector", "crack_presence_classifier", "crack_segmentation"):
            _needs_weights(key)
        published = root / "_outputs" / "detectors" / "road_rdd2022" / "per_image.json"
        if not published.is_file():
            pytest.skip(f"{published} is not on this machine (run the road evaluation)")
        expected = json.loads(published.read_text(encoding="utf-8"))[:2]
        import argparse

        args = argparse.Namespace(limit=2, overlays=0, artifacts=str(tmp_path))
        ev.evaluate_road(args)
        got = json.loads((tmp_path / "per_image.json").read_text(encoding="utf-8"))
        for row_got, row_want in zip(got, expected):
            assert row_got["image"] == row_want["image"]
            assert row_got["detector_hits_at_0.25"] == row_want["detector_hits_at_0.25"]
            assert row_got["crack_presence"] == row_want["crack_presence"]


class TestSoundConcreteIsNotADefect:
    def test_background_boxes_are_kept_out_of_the_defects(self, monkeypatch) -> None:
        """CODEBRIM's "Background" class marks undamaged concrete. The detector draws
        those boxes too, and they used to be counted -- and painted -- as defects.

        At the registry's 0.25 cut-off the model rarely draws one on this split, so the
        detector is run at 0.01 here to make sure some appear; what is checked is where
        they go, not how many there are.
        """
        _needs_weights("structural_multiclass_detector")
        import core.detection as detection
        from core.detection import detect_structural_defects, non_defect_labels

        real = detection._run_onnx_yolo

        def low_threshold(*args, **kwargs):
            kwargs["score_threshold"] = 0.01
            return real(*args, **kwargs)

        monkeypatch.setattr(detection, "_run_onnx_yolo", low_threshold)

        assert non_defect_labels("structural_multiclass_detector") == {"Background"}
        root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
        base = Path(root) if root else REPO_ROOT / "training" / "data"
        images = sorted((base / "prepared" / "structural_det" / "test" / "images").glob("*"))
        if not images:
            pytest.skip("prepared/structural_det/test is not on this machine (ODK_TRAINING_DATA_DIR)")
        for path in images[:40]:
            result = detect_structural_defects(cv2.imread(str(path)))
            assert all(h.label != "Background" for h in result.detections)
            assert "Background" not in result.mask_by_class
            if result.non_defect_detections:
                assert {h.label for h in result.non_defect_detections} == {"Background"}
                assert result.to_summary()["non_defect_detections"]
                return
        pytest.skip("the detector drew no Background box on the first 40 test images")
