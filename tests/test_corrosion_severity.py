"""Corrosion severity grading: what it answers, and what it refuses to answer.

The detector that came before this was trained and rejected -- mAP50 0.254 at recall
0.257, which misses three corrosion sites in four. The replacement is a segmentation
model over an ORDINAL scale (good < fair < poor < severe), and it changes the question
from "is there corrosion" to "how bad is this pixel".

That changes what has to be tested. A severity scale can fail in a way a detector
cannot: it can decline to ever use its worst grade and still report a respectable mean.
So the tests below check the model predicts `severe` at all, that the reported grade is
an argmax over mutually exclusive classes rather than a threshold on one channel, and
that a missing model refuses rather than returning a guess -- because unlike every other
detector here there is no honest heuristic to fall back on. Colour rules can find rust;
nothing but the corpus distinguishes poor from severe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.detection import (
    CorrosionGradingRefused,
    _run_onnx_segmentation_multiclass,
    grade_corrosion_severity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
KEY = "corrosion_severity_segmentation"
PROVENANCE_PATH = MODELS_DIR / "manifests" / "model_provenance.json"


@pytest.fixture(scope="module")
def entry() -> dict:
    registry = json.loads((MODELS_DIR / "model_registry.json").read_text(encoding="utf-8"))
    installed = registry["models"].get(KEY)
    assert installed is not None, f"{KEY} is not in the model registry"
    return installed


@pytest.fixture(scope="module")
def metrics() -> dict:
    """The provenance record, which is tracked -- unlike the metrics card.

    models/metrics/*.json is generated beside the gitignored weights, so a clone has
    neither. The provenance manifest carries the same validation dict and is committed,
    which makes these figures checkable on a machine that has never seen the model.
    """
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    record = next(
        (f for f in provenance.get("files", []) if f.get("registry_key") == KEY), None
    )
    assert record is not None, f"{KEY} has no provenance record"
    return record


def weights_installed(entry: dict) -> bool:
    # Weights are gitignored, so a fresh clone has the registry and not the file.
    return bool(entry.get("path")) and (MODELS_DIR / entry["path"]).is_file()


class TestTheScaleIsOrdinalAndSaysSo:
    def test_the_labels_run_from_best_to_worst(self, entry) -> None:
        """`worst_present_grade` reads the last label, so the order is load-bearing.

        Sorted alphabetically these would be fair, good, poor, severe -- and the code
        would report 'severe' as the worst when it is, and 'poor' as better than 'good'
        when it is not. The order is part of the contract, not presentation.
        """
        assert entry["labels"] == ["good", "fair", "poor", "severe"]

    def test_the_description_names_the_adjacent_grade_error(self, entry) -> None:
        description = entry.get("description", "").lower()
        assert "ordinal" in description, (
            "a mean IoU near 0.58 on an ordinal scale does not mean what it means on "
            "independent classes, and the entry has to say which it is"
        )
        assert "one step" in description


class TestItActuallyPredictsTheWorstGrade:
    """A model that never says `severe` has failed regardless of its mean IoU.

    This is the specific failure an averaged metric hides: three easy classes carry the
    mean while the one an inspector is looking for is never emitted.
    """

    def test_severe_has_a_measured_iou_above_zero(self, metrics) -> None:
        per_class = metrics["validation_metrics"]["per_class_iou"]
        assert per_class["severe"] > 0.0

    def test_severe_recall_is_recorded_and_is_not_a_rounding_error(self, metrics) -> None:
        validation = metrics["validation_metrics"]
        confusion = np.asarray(validation["confusion_matrix"])
        severe = list(validation["per_class_iou"]).index("severe")
        recovered = int(confusion[severe, severe])
        total = int(confusion[severe].sum())
        assert total > 0, "no severe pixels in validation; the figure would mean nothing"
        assert recovered / total > 0.5, (
            f"severe recall is {recovered / total:.3f}: the model finds fewer than half "
            "the pixels of the grade the whole capability exists to flag"
        )

    def test_the_confusion_matrix_agrees_with_the_reported_support(self, metrics) -> None:
        validation = metrics["validation_metrics"]
        confusion = np.asarray(validation["confusion_matrix"])
        support = validation["per_class_support"]
        for index, label in enumerate(validation["per_class_iou"]):
            assert int(confusion[index].sum()) == int(support[label])


class TestRefusalRatherThanAGuessedGrade:
    def test_an_unregistered_key_is_refused(self) -> None:
        with pytest.raises(CorrosionGradingRefused, match="not in the model registry"):
            grade_corrosion_severity(np.zeros((8, 8, 3), np.uint8), model_key="no_such_model")

    def test_missing_weights_refuse_instead_of_falling_back(self, monkeypatch) -> None:
        """The point of the file.

        detect_cracks degrades to morphology and reports `heuristic`. There is no
        equivalent for severity, so the honest answer to "the model is not installed" is
        no answer.
        """
        from core import detection

        monkeypatch.setattr(
            detection, "model_status", lambda key: {"exists": False, "path": "gone.onnx"}
        )
        with pytest.raises(CorrosionGradingRefused, match="weights are not installed"):
            grade_corrosion_severity(np.zeros((8, 8, 3), np.uint8))

    def test_a_graph_whose_channels_disagree_with_the_labels_is_refused(
        self, monkeypatch, entry
    ) -> None:
        """Four labels against three channels would rename every grade silently."""
        from core import detection

        class ThreeChannelNet:
            def setInput(self, blob):
                self.blob = blob

            def forward(self):
                return np.zeros((1, 3, 64, 64), dtype=np.float32)

        monkeypatch.setattr(detection, "_load_onnx_net", lambda path: ThreeChannelNet())
        assert _run_onnx_segmentation_multiclass(
            np.zeros((64, 64, 3), np.uint8), Path("whatever.onnx"),
            input_size=64, class_count=4,
        ) is None


class TestTheGradeIsAnArgmaxNotAThreshold:
    def test_a_pixel_the_model_is_unsure_about_still_gets_a_grade(self, monkeypatch) -> None:
        """Every channel below 0.5 is a normal softmax output, not an absence of answer.

        Thresholding one channel of a four-way softmax at 0.5 would return "nothing" for
        any pixel where the model is merely undecided -- which for an ordinal scale is
        most of the interesting ones, since they sit between two grades.
        """
        from core import detection

        probabilities = np.zeros((1, 4, 64, 64), dtype=np.float32)
        probabilities[:, 0] = 0.30
        probabilities[:, 1] = 0.25
        probabilities[:, 2] = 0.10
        probabilities[:, 3] = 0.35  # the argmax, and still under any 0.5 cut-off

        class UnsureNet:
            def setInput(self, blob):
                self.blob = blob

            def forward(self):
                return probabilities

        monkeypatch.setattr(detection, "_load_onnx_net", lambda path: UnsureNet())
        grade = _run_onnx_segmentation_multiclass(
            np.zeros((64, 64, 3), np.uint8), Path("whatever.onnx"),
            input_size=64, class_count=4,
        )
        assert grade is not None
        assert (grade == 3).all()


class TestAgainstTheInstalledModel:
    """Runs only where the weights are; the checks above hold everywhere."""

    def test_grading_a_real_image_reports_every_class_and_its_identity(self, entry) -> None:
        if not weights_installed(entry):
            pytest.skip(f"{KEY} weights are not on this machine (gitignored)")
        import cv2

        images = sorted(
            (REPO_ROOT / "training/data/prepared/corrosion_seg/val/images").glob("*")
        )
        if not images:
            pytest.skip("the prepared corrosion corpus is not on this machine")

        result = grade_corrosion_severity(cv2.imread(str(images[0])))
        assert result.labels == entry["labels"]
        assert result.grade_index.shape[:2] == cv2.imread(str(images[0])).shape[:2]
        assert set(result.class_pixel_ratio) == set(entry["labels"])
        assert result.class_pixel_ratio  # every class named, present or not
        assert abs(sum(result.class_pixel_ratio.values()) - 1.0) < 1e-6
        # The identity of the file that answered, not of the row that claims it.
        assert result.model_key == KEY
        assert result.model_sha256 == entry["sha256"]
        assert result.model_used.startswith("onnx:")
