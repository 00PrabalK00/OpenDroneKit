"""The metric that will decide what gets registered has to be able to fail.

The India holdout that came before this one could not. SpaceNet 7 leaves 96.7 per cent of
each tile at IGNORE_INDEX, the confusion matrix drops ignored pixels from prediction and
target alike, and with no negatives left a model predicting building on every pixel scored
a mean IoU of 1.0. That is measured, not hypothesised, and it is why six registry rows
were gated on a number that could not detect the most likely failure of a segmentation
model.

Gorakhpur is exhaustively annotated, so the same arithmetic behaves completely differently
here -- a false positive lands on a labelled pixel and is counted. These tests pin that
difference down, because the registration decision rests on it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.evaluate_gorakhpur import confusion, scores_from  # noqa: E402
from training.semantic_tiles import IGNORE_INDEX  # noqa: E402

NAMES = ["background", "building", "road", "vegetation", "water", "bare_land"]


def test_a_perfect_prediction_scores_one():
    truth = np.array([[1, 1, 3], [3, 3, 2]], dtype=np.int64)
    scores = scores_from(confusion(truth.copy(), truth, len(NAMES)), NAMES)
    assert scores["pixel_accuracy"] == 1.0
    assert scores["mean_iou_present_classes"] == 1.0


def test_predicting_one_class_everywhere_is_caught():
    """The exact failure the old holdout could not see.

    A model answering "building" for every pixel must score close to nothing here. On the
    SpaceNet 7 holdout the same prediction scored 1.0.
    """
    truth = np.array([[1, 3, 3], [3, 3, 2]], dtype=np.int64)
    guess = np.ones_like(truth)

    scores = scores_from(confusion(guess, truth, len(NAMES)), NAMES)
    building = scores["per_class"]["building"]
    assert building["recall"] == 1.0, "it did find the one real building pixel"
    # And it is punished for the five it invented, which is the whole point.
    assert building["precision"] < 0.2
    assert scores["mean_iou_present_classes"] < 0.15
    assert scores["per_class"]["vegetation"]["iou"] == 0.0


def test_a_class_absent_from_the_holdout_is_not_averaged_in():
    """Scoring a zero for a class that never occurs understates the model for free."""
    truth = np.array([[1, 1], [3, 3]], dtype=np.int64)
    scores = scores_from(confusion(truth.copy(), truth, len(NAMES)), NAMES)

    assert scores["classes_present"] == ["building", "vegetation"]
    assert scores["per_class"]["water"]["present_in_holdout"] is False
    # Four of six classes are absent. Averaging their zeros would give 0.33, not 1.0.
    assert scores["mean_iou_present_classes"] == 1.0


def test_ignored_pixels_are_not_scored():
    truth = np.array([[1, IGNORE_INDEX], [3, IGNORE_INDEX]], dtype=np.int64)
    guess = np.array([[1, 4], [3, 4]], dtype=np.int64)
    matrix = confusion(guess, truth, len(NAMES))

    assert int(matrix.sum()) == 2, "only the two labelled pixels count"
    scores = scores_from(matrix, NAMES)
    # Water was predicted twice and never scored, because both were on ignored pixels.
    assert scores["per_class"]["water"]["predicted_px"] == 0


def test_a_missed_class_reads_as_zero_not_as_absent():
    """bare_land scored 0.000 on the real holdout. That has to be distinguishable from
    a class that simply is not there, or the registry cannot tell 'broken' from 'untested'."""
    truth = np.array([[5, 5], [3, 3]], dtype=np.int64)
    guess = np.array([[3, 3], [3, 3]], dtype=np.int64)

    scores = scores_from(confusion(guess, truth, len(NAMES)), NAMES)
    bare = scores["per_class"]["bare_land"]
    assert bare["present_in_holdout"] is True, "it IS in the holdout; the model missed it"
    assert bare["iou"] == 0.0
    assert bare["labelled_px"] == 2 and bare["predicted_px"] == 0
    assert "bare_land" in scores["classes_present"]


def test_a_prediction_outside_the_schema_cannot_crash_the_matrix():
    truth = np.array([[1, 3]], dtype=np.int64)
    guess = np.array([[99, 3]], dtype=np.int64)
    matrix = confusion(guess, truth, len(NAMES))
    assert int(matrix.sum()) == 2
