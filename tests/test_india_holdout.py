"""The India holdout must survive a corpus rebuild.

The shared semantic engine is presented as India-first. That claim is only worth
anything if Indian ground is tested rather than trained on, and the default split is a
hash of the site name, which has no idea where India is -- before this was pinned, all
four Indian tiles landed in train purely by chance.

These tests exist because the failure is silent. A corpus rebuilt without the pin looks
completely normal: right sample count, right class coverage, sensible split ratios. The
only visible difference is which side of the wall four sites ended up on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.semantic_corpus import (
    INDIA_HOLDOUT_GROUPS,
    SemanticCorpusError,
    SplitPolicy,
    _split_for_group,
)

CORPUS = Path(__file__).resolve().parents[1] / "training" / "data" / "prepared" / "shared_semantic" / "corpus.json"


class TestPinnedSplit:
    def test_pinned_groups_land_in_test(self) -> None:
        policy = SplitPolicy(pinned_test_groups=dict(INDIA_HOLDOUT_GROUPS))
        for group in INDIA_HOLDOUT_GROUPS:
            assert _split_for_group(group, policy) == "test"

    def test_the_pin_is_what_moves_them(self) -> None:
        # If the unpinned hash already sent these to test, the pin would be untested
        # decoration and this suite would pass while proving nothing.
        plain = SplitPolicy()
        moved = [g for g in INDIA_HOLDOUT_GROUPS if _split_for_group(g, plain) != "test"]
        assert moved, "Expected the default hash to place at least one India tile outside test."

    def test_unpinned_groups_are_untouched(self) -> None:
        policy = SplitPolicy(pinned_test_groups=dict(INDIA_HOLDOUT_GROUPS))
        plain = SplitPolicy()
        for group in ("openearthmap::Kagera", "openearthmap::Paris", "spacenet7::L15-0331E-1257N_1327_3160_13"):
            assert _split_for_group(group, policy) == _split_for_group(group, plain)

    def test_a_pin_without_a_reason_is_refused(self) -> None:
        with pytest.raises(SemanticCorpusError):
            SplitPolicy(pinned_test_groups={"spacenet7::somewhere": ""})

    def test_every_holdout_group_names_a_place(self) -> None:
        # The reason is what a reader sees when asking "why is this in test?", so an
        # unhelpful one defeats the point of recording it at all.
        for group, reason in INDIA_HOLDOUT_GROUPS.items():
            assert "India holdout" in reason, group
            assert "N," in reason and "E)" in reason, f"{group} should carry its coordinates."


@pytest.mark.skipif(not CORPUS.is_file(), reason="Built corpus is not present.")
class TestBuiltCorpus:
    @staticmethod
    def _corpus() -> dict:
        return json.loads(CORPUS.read_text(encoding="utf-8"))

    def test_each_pin_captured_samples(self) -> None:
        # A pin that matches no group is the quiet failure: the corpus still reports a
        # test split, it just has no India in it.
        pinned = self._corpus()["counts"]["pinned_test_samples"]
        assert set(pinned) == set(INDIA_HOLDOUT_GROUPS)
        for group, count in pinned.items():
            assert count > 0, f"{group} is pinned but matched no samples."

    def test_no_india_sample_is_trained_on(self) -> None:
        for sample in self._corpus()["samples"]:
            if sample["group"] in INDIA_HOLDOUT_GROUPS:
                assert sample["split"] == "test", sample["id"]

    def test_holdout_samples_carry_their_reason(self) -> None:
        held = [s for s in self._corpus()["samples"] if s["group"] in INDIA_HOLDOUT_GROUPS]
        assert held
        for sample in held:
            assert sample.get("pinned_test_reason"), sample["id"]

    def test_the_holdout_only_speaks_for_buildings(self) -> None:
        # SpaceNet 7 labels buildings and nothing else. Recording that here means a
        # future contributor who reads this metric as whole-schema India performance
        # has to delete an assertion that says otherwise.
        held = [s for s in self._corpus()["samples"] if s["group"] in INDIA_HOLDOUT_GROUPS]
        assert {s["source"] for s in held} == {"spacenet7"}
        for sample in held:
            assert sample.get("class_ids") == [1], (
                f"{sample['id']} claims classes beyond building; the India holdout's "
                "scope note is no longer accurate."
            )
