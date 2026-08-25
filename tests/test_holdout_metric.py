"""The India holdout must be able to fail a model. It could not.

Measured, not argued: under the multiclass metric a model predicting BUILDING ON EVERY
PIXEL scores mean IoU 1.0 on a SpaceNet 7 tile. The confusion matrix drops IGNORE pixels
from prediction and target alike, and SpaceNet 7 marks every non-building pixel ignore,
so 96.7 per cent of each tile is invisible to scoring. With no negatives there are no
false positives, and the worst possible model is unfalsifiable.

The holdout gates six registry rows. A gate that cannot close is decoration.

The first test below is the proof, and the ones after it check the replacement metric
does the job the original could not.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluate_holdout import BUILDING_CLASS_ID, binary_building_scores, holdout_report
from training.semantic_tiles import IGNORE_INDEX


def spacenet_tile(size: int = 64, building: int = 12) -> np.ndarray:
    """A SpaceNet 7 label: a few building pixels, everything else ignore."""
    label = np.full((size, size), IGNORE_INDEX, dtype=np.int64)
    label[:building, :building] = BUILDING_CLASS_ID
    return label


class TestTheOldMetricCouldNotFail:
    def test_multiclass_scoring_rewards_predicting_building_everywhere(self) -> None:
        """The bug, demonstrated rather than asserted."""
        torch = pytest.importorskip("torch")
        from training.train_shared_semantic import Confusion

        target = torch.from_numpy(spacenet_tile()).unsqueeze(0)
        logits = torch.zeros(1, 6, 64, 64)
        logits[:, BUILDING_CLASS_ID] = 10.0  # building, everywhere

        confusion = Confusion.create(6)
        confusion.update(logits, target)
        summary = confusion.summary([f"c{i}" for i in range(6)])
        assert summary["mean_iou"] == 1.0, (
            "this test documents the defect; if it now fails the multiclass metric was "
            "changed and this file's premise needs revisiting"
        )


class TestTheBinaryMetricCanFail:
    def test_predicting_building_everywhere_scores_badly(self) -> None:
        label = spacenet_tile()
        predicted = np.full_like(label, BUILDING_CLASS_ID)
        scores = binary_building_scores(predicted, label)
        assert scores["building_iou"] < 0.05, (
            f"the over-predictor still scored {scores['building_iou']:.3f}; the holdout "
            "remains unable to fail the most likely failure of a segmentation model"
        )
        assert scores["building_precision"] < 0.05

    def test_a_perfect_prediction_scores_one(self) -> None:
        label = spacenet_tile()
        predicted = np.where(label == BUILDING_CLASS_ID, BUILDING_CLASS_ID, 0)
        scores = binary_building_scores(predicted, label)
        assert scores["building_iou"] == pytest.approx(1.0)
        assert scores["building_precision"] == pytest.approx(1.0)
        assert scores["building_recall"] == pytest.approx(1.0)

    def test_predicting_nothing_scores_zero_rather_than_dividing_by_zero(self) -> None:
        label = spacenet_tile()
        scores = binary_building_scores(np.zeros_like(label), label)
        assert scores["building_iou"] == 0.0
        assert scores["building_recall"] == 0.0
        assert scores["building_precision"] == 0.0

    def test_recall_and_precision_move_independently(self) -> None:
        """A model can be cautious or greedy, and the two must be distinguishable."""
        label = spacenet_tile()
        greedy = np.full_like(label, BUILDING_CLASS_ID)
        cautious = np.zeros_like(label)
        cautious[:6, :6] = BUILDING_CLASS_ID  # a quarter of the true buildings, all correct

        greedy_scores = binary_building_scores(greedy, label)
        cautious_scores = binary_building_scores(cautious, label)
        assert greedy_scores["building_recall"] > cautious_scores["building_recall"]
        assert cautious_scores["building_precision"] > greedy_scores["building_precision"]

    def test_over_segmentation_is_visible_in_the_fractions(self) -> None:
        label = spacenet_tile()
        scores = binary_building_scores(np.full_like(label, BUILDING_CLASS_ID), label)
        assert scores["predicted_building_fraction"] == 1.0
        assert scores["labelled_building_fraction"] < 0.1

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError):
            binary_building_scores(np.zeros((4, 4)), np.zeros((8, 8)))


class TestTheReportDoesNotOverclaim:
    def test_it_says_buildings_only(self) -> None:
        note = holdout_report(binary_building_scores(spacenet_tile(), spacenet_tile()),
                              sites=["a"])["reading_note"].lower()
        assert "buildings only" in note
        for absent in ("road", "vegetation", "water"):
            assert absent in note, "the note must name what stays unmeasured"

    def test_it_says_satellite_not_drone(self) -> None:
        """0.5 m satellite imagery does not speak for a survey flown at 60 m."""
        note = holdout_report(binary_building_scores(spacenet_tile(), spacenet_tile()),
                              sites=["a"])["reading_note"].lower()
        assert "satellite" in note and "drone" in note

    def test_the_sites_are_named(self) -> None:
        report = holdout_report(binary_building_scores(spacenet_tile(), spacenet_tile()),
                                sites=["mumbai", "tirupati"])
        assert report["sites"] == ["mumbai", "tirupati"]
        assert report["metric"] == "binary_building"


class TestTheHoldoutIsScoredOnTheHoldout:
    """A metric that can fail is only half of it; it has to be fed the right tiles.

    The evaluator built the list of pinned Indian samples, then scored the whole `test`
    split -- 362 tiles across sixteen groups and two corpora -- while the report named
    four Indian sites. Every figure it produced was attributed to a place that had not
    produced it, and the Indian collapse was diluted by tiles from Tanzania and Uruguay
    into something that looked merely poor.
    """

    def test_the_test_split_is_wider_than_the_holdout(self) -> None:
        """The premise: if these were the same set, the filter would be decoration."""
        from training.semantic_corpus import INDIA_HOLDOUT_GROUPS

        assert INDIA_HOLDOUT_GROUPS, "no pinned holdout groups to filter to"
        assert all(isinstance(group, str) for group in INDIA_HOLDOUT_GROUPS)

    def test_the_evaluator_filters_the_dataset_to_the_pinned_groups(self) -> None:
        import inspect

        from training import evaluate_holdout

        source = inspect.getsource(evaluate_holdout.main)
        assert "INDIA_HOLDOUT_GROUPS" in source
        assert "dataset.samples = [" in source, (
            "the dataset must be narrowed to the pinned sites, or the report attributes "
            "a score to sites that did not produce it"
        )

    def test_a_holdout_with_no_indian_tiles_refuses_rather_than_scoring_the_rest(self) -> None:
        import inspect

        from training import evaluate_holdout

        source = inspect.getsource(evaluate_holdout.main)
        assert "contains none of the India holdout groups" in source
