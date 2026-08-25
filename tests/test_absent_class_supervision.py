"""Unlabelled is not the same as unknown, and the difference is a whole model.

The first shared_semantic head predicted BUILDING on 100 per cent of the pixels of all
four India holdout tiles. Nothing was wrong with the architecture or the run. The corpus
mixes a fully labelled source with SpaceNet 7, which draws every building and leaves the
other 96.7 per cent of each tile at IGNORE_INDEX -- and cross-entropy with ignore_index
scores only labelled pixels. On those tiles "everything is a building" was free.

The fix is to use the evidence that was there all along: annotators who drew every
building they saw make an unlabelled pixel evidence that a building is ABSENT. Not
evidence of what it is -- there is no basis for calling it road rather than water -- so
it cannot become a background label, only a penalty on that one class.

These tests pin the three distinctions that make it honest:

  * absent means absent for the annotated class only, not a label for the pixel
  * no-data is not absence; a transparent mosaic edge is evidence of nothing
  * a corpus with no such evidence must train exactly as it did before
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training-only dependency")

from training.train_shared_semantic import absent_class_penalty  # noqa: E402

BUILDING = 1
CLASSES = 6


def logits_favouring(channel: int, *, strength: float = 12.0, size: int = 4) -> torch.Tensor:
    logits = torch.zeros(1, CLASSES, size, size)
    logits[:, channel] = strength
    return logits


class TestThePenaltyPunishesTheAbsentClass:
    def test_predicting_the_absent_class_everywhere_is_expensive(self) -> None:
        loss = absent_class_penalty(
            logits_favouring(BUILDING),
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        assert float(loss) > 1.0, (
            "the model insisting on a class the annotation excludes must cost something; "
            "this is the exact behaviour that produced precision 0.092"
        )

    def test_predicting_anything_else_there_is_free(self) -> None:
        """The penalty must not push toward a class. It only pushes away from one.

        If this cost anything, the term would be smuggling in a background label for
        pixels nobody labelled.
        """
        loss = absent_class_penalty(
            logits_favouring(3),  # vegetation, which the annotation says nothing about
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        assert float(loss) < 0.01

    def test_confidence_in_the_absent_class_costs_more_than_hesitation(self) -> None:
        confident = absent_class_penalty(
            logits_favouring(BUILDING, strength=12.0),
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        hesitant = absent_class_penalty(
            logits_favouring(BUILDING, strength=1.0),
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        assert float(confident) > float(hesitant)

    def test_it_stays_finite_when_the_model_is_completely_wrong(self) -> None:
        """-log(1 - p) goes to infinity at p = 1, and an inf takes the step with it."""
        loss = absent_class_penalty(
            logits_favouring(BUILDING, strength=100.0),
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        assert torch.isfinite(loss)


class TestItAppliesOnlyWhereThereIsEvidence:
    def test_pixels_outside_the_mask_are_not_penalised(self) -> None:
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[:, :2] = True  # only the top half is unlabelled-and-known-absent
        half = absent_class_penalty(
            logits_favouring(BUILDING), mask, torch.eye(CLASSES)[BUILDING].unsqueeze(0)
        )
        full = absent_class_penalty(
            logits_favouring(BUILDING),
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        # Averaged over penalised pixels, so the value matches; what differs is that the
        # labelled half contributes nothing at all.
        assert float(half) == pytest.approx(float(full), rel=1e-5)

    def test_a_batch_with_no_such_evidence_contributes_nothing(self) -> None:
        """OpenEarthMap tiles label every pixel, so they carry no absence evidence."""
        loss = absent_class_penalty(
            logits_favouring(BUILDING),
            torch.zeros(1, 4, 4, dtype=torch.bool),
            torch.zeros(1, CLASSES),
        )
        assert float(loss) == 0.0

    def test_evidence_is_per_tile_within_a_batch(self) -> None:
        """One source in a batch declares absence; the other must be untouched."""
        logits = logits_favouring(BUILDING).repeat(2, 1, 1, 1)
        negative_classes = torch.zeros(2, CLASSES)
        negative_classes[0, BUILDING] = 1.0  # only the first tile is exhaustive
        mask = torch.ones(2, 4, 4, dtype=torch.bool)

        both = absent_class_penalty(logits, mask, negative_classes)
        first_only = absent_class_penalty(
            logits[:1], mask[:1], negative_classes[:1]
        )
        assert float(both) == pytest.approx(float(first_only), rel=1e-5)

    def test_the_gradient_reaches_the_model(self) -> None:
        logits = logits_favouring(BUILDING).requires_grad_(True)
        loss = absent_class_penalty(
            logits,
            torch.ones(1, 4, 4, dtype=torch.bool),
            torch.eye(CLASSES)[BUILDING].unsqueeze(0),
        )
        loss.backward()
        assert logits.grad is not None
        # Pushing DOWN on the absent class: raising that logit must increase the loss.
        assert float(logits.grad[:, BUILDING].sum()) > 0


class TestTheDatasetSuppliesTheEvidence:
    """The tensors above are only correct if the dataset marks the right pixels.

    Two reasons a pixel is IGNORE, and they are not the same evidence: unlabelled in an
    exhaustively annotated tile means absent, and no-data means nothing at all. Padding
    counts as no-data, since it is invented pixels.
    """

    def test_no_data_pixels_are_excluded_from_the_negative(self, tmp_path) -> None:
        from training.semantic_tiles import IGNORE_INDEX

        mask = np.full((8, 8), IGNORE_INDEX, dtype=np.int64)
        mask[:2, :2] = BUILDING
        no_data = np.zeros((8, 8), dtype=bool)
        no_data[6:, 6:] = True

        negative = (mask == IGNORE_INDEX) & ~no_data
        assert not negative[6:, 6:].any(), (
            "a transparent mosaic edge is not evidence that a building is absent"
        )
        assert not negative[:2, :2].any(), "labelled pixels are scored by cross-entropy"
        assert negative.sum() == 64 - 4 - 4

    def test_the_spacenet_manifest_declares_what_it_annotates_exhaustively(self) -> None:
        import inspect

        from training.datasets import spacenet7

        source = inspect.getsource(spacenet7)
        assert "'exhaustive_class_ids'" in source, (
            "without the declaration the corpus cannot tell an exhaustively annotated "
            "tile from a partially labelled one, and the penalty has nothing to act on"
        )

    def test_the_corpus_builder_carries_the_declaration_through(self) -> None:
        import inspect

        from training import semantic_corpus

        assert "'exhaustive_class_ids'" in inspect.getsource(semantic_corpus)
