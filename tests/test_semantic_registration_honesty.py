"""The semantic model predicts six classes and two of them work. That has to stay true.

This is the registration the whole v3/v4 exercise was for. The temptation it guards
against is specific and was live: v4 scores a mean IoU of 0.317 on Gorakhpur, and a mean
is exactly the number that lets a six-class claim be made on the strength of one good
class. vegetation is 0.913. water is 0.003. bare_land predicts nothing at all.

So `usable_labels` is not a comment -- these tests make it answer to the holdout report.
A class can only be listed as usable if the measurement says so, every class left out has
to be named in the description with its number, and the label order has to match the
schema the model was exported with, because a claim about class 4 is worthless if class 4
is not where the runtime thinks it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
KEY = "shared_semantic_landcover"

HOLDOUT = REPO_ROOT / "docs" / "holdout" / "shared_semantic_v4_gorakhpur.json"
MANIFEST = MODELS_DIR / "shared_semantic_dinov2_vitb14_v4.manifest.json"
EXPORT = MODELS_DIR / "shared_semantic_dinov2_vitb14_v4.export.json"

# The bar a class has to clear to be called usable. Set at the gap the measurement
# actually shows rather than at a round number: building is 0.621 and road is 0.366, and
# nothing in between was observed, so this separates what the evidence separates.
USABLE_IOU = 0.60


@pytest.fixture(scope="module")
def entry() -> dict:
    registry = json.loads((MODELS_DIR / "model_registry.json").read_text(encoding="utf-8"))
    row = registry["models"].get(KEY)
    assert row is not None, f"{KEY} is not in the model registry"
    return row


@pytest.fixture(scope="module")
def holdout() -> dict:
    assert HOLDOUT.is_file(), (
        "the Gorakhpur report is the evidence this row rests on; without it the row is "
        "an assertion"
    )
    return json.loads(HOLDOUT.read_text(encoding="utf-8"))


def test_the_row_is_installed_and_names_its_weights(entry) -> None:
    assert entry["status"] == "installed"
    assert entry["path"], "installed but names no file"
    assert entry["sha256"], "no digest, so the metrics belong to no particular file"


def test_the_digest_matches_what_the_exporter_recorded(entry) -> None:
    """Two records of the same model. If they disagree, one of them is stale."""
    if not EXPORT.is_file():
        pytest.skip("export record not on this machine")
    assert entry["sha256"] == json.loads(EXPORT.read_text(encoding="utf-8"))["sha256"]


def test_the_labels_are_in_the_exported_schema_order(entry) -> None:
    """Class 4 has to be water everywhere or the per-class claims mean nothing."""
    if not MANIFEST.is_file():
        pytest.skip("manifest not on this machine")
    schema = json.loads(MANIFEST.read_text(encoding="utf-8"))["schema"]["classes"]
    assert entry["labels"] == [c["name"] for c in schema]


def test_every_usable_label_is_one_the_model_predicts(entry) -> None:
    assert set(entry["usable_labels"]) <= set(entry["labels"])


def test_a_class_is_only_usable_if_the_holdout_says_so(entry, holdout) -> None:
    """The test that stops this row growing back into a six-class claim.

    Adding water to usable_labels without a better measurement fails here, which is the
    whole point of writing the number down next to the claim.
    """
    weak = {
        name: holdout["per_class"][name]["iou"]
        for name in entry["usable_labels"]
        if holdout["per_class"][name]["iou"] < USABLE_IOU
    }
    assert not weak, (
        f"listed as usable but measured below {USABLE_IOU} on Gorakhpur: {weak}"
    )


def test_the_classes_left_out_are_named_with_their_numbers(entry) -> None:
    """Silence about a broken class reads as an oversight. Naming it is the deliverable.

    A registry that lists two usable classes and says nothing about the other four
    invites the reader to assume the rest are merely untested. water and bare_land are
    not untested; they were measured and they failed.
    """
    description = entry["description"].lower()
    for name in set(entry["labels"]) - set(entry["usable_labels"]):
        if name == "background":
            continue
        assert name in description, f"{name} is not usable and is not mentioned"

    assert "not working" in description, "the failing classes are not called failing"
    for number in ("0.003", "0.366"):
        assert number in description, f"the description does not carry {number}"


def test_the_dead_classes_are_not_quietly_usable(entry) -> None:
    """Named separately from the measurement, because this is the decision itself."""
    for name in ("water", "bare_land", "road"):
        assert name not in entry["usable_labels"], (
            f"{name} did not work on the India holdout and must not be registered as "
            "working; see docs/holdout/shared_semantic_v4_gorakhpur.json"
        )


def test_the_mean_is_not_offered_as_the_result(entry) -> None:
    """0.317 averages a 0.913 class with two dead ones. Quoting it is the failure mode."""
    description = entry["description"]
    assert "0.317" not in description or "should not be quoted" in description, (
        "the mean IoU appears without the warning that it averages working and dead "
        "classes together"
    )


def test_bare_land_predicting_nothing_is_recorded_as_measured(holdout) -> None:
    """0.000 because it was missed, not 0.000 because it was absent.

    The report distinguishes these and the registry decision depends on the difference:
    a class that is not in the holdout has no score to give, while bare_land is in 49
    tiles and the model answered with silence.
    """
    bare = holdout["per_class"]["bare_land"]
    assert bare["present_in_holdout"] is True
    assert bare["labelled_px"] > 0
    assert bare["predicted_px"] == 0
    assert bare["iou"] == 0.0


def test_the_row_records_that_the_run_was_cut_short(entry) -> None:
    """These are epoch-35 weights. A reader comparing against a finished run needs that."""
    description = entry["description"].lower()
    assert "epoch 35" in description
    assert "13.4" in description, "the dropped-step rate is not next to the numbers"
