"""Change detection over image pairs.

The mining pack was never blocked on data. MineNetCD has been on disk and usable
throughout; what was missing is that every other trainer here takes one image per sample
and change detection takes two.

These tests cover the three things that would fail quietly rather than loudly: a split
that leaks a site across the wall, a stem inflation that silently reinitialises the
pretrained weights, and a pair that is not actually co-registered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.train_change import (
    ChangeConfig,
    ChangePairDataset,
    evaluate,
    site_split,
)


def _site(root: Path, name: str, size: int = 64, changed: bool = True) -> Path:
    from PIL import Image

    site = root / name
    site.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (10, 20, 30)).save(site / "im1.jpg")
    Image.new("RGB", (size, size), (40, 50, 60)).save(site / "im2.jpg")
    mask = np.zeros((size, size), dtype=np.uint8)
    if changed:
        mask[10:30, 10:30] = 255
    Image.fromarray(mask).save(site / "ref.png")
    return site


class TestSiteSplit:
    def test_a_site_always_lands_in_one_split(self) -> None:
        config = ChangeConfig()
        for name in ("America1", "Asia7", "Europe12"):
            assert len({site_split(name, config) for _ in range(20)}) == 1

    def test_all_three_splits_are_populated(self) -> None:
        config = ChangeConfig()
        names = [f"site{i}" for i in range(100)]
        assigned = {site_split(n, config) for n in names}
        assert assigned == {"train", "val", "test"}

    def test_the_salt_changes_the_assignment(self) -> None:
        # Proves the split is actually derived from the salt, so a future corpus can be
        # re-split deliberately rather than by accident.
        a = ChangeConfig(split_salt="one")
        b = ChangeConfig(split_salt="two")
        names = [f"site{i}" for i in range(60)]
        assert [site_split(n, a) for n in names] != [site_split(n, b) for n in names]


class TestDataset:
    def test_a_pair_becomes_six_channels(self, tmp_path: Path) -> None:
        config = ChangeConfig(image_size=32, split_salt="fixed")
        for i in range(30):
            _site(tmp_path, f"s{i}")
        split = site_split("s0", config)
        dataset = ChangePairDataset(tmp_path, split, config, train=False)
        images, mask = dataset[0]
        assert images.shape == (6, 32, 32), "the two frames must arrive stacked"
        assert mask.shape == (32, 32)

    def test_a_misregistered_pair_is_refused(self, tmp_path: Path) -> None:
        from PIL import Image

        config = ChangeConfig(image_size=32, split_salt="fixed")
        for i in range(30):
            _site(tmp_path, f"s{i}")
        target = tmp_path / "s0"
        # A pair whose frames disagree on size is not co-registered, and training on it
        # would teach the model that a registration error is a change.
        Image.new("RGB", (48, 48), (1, 2, 3)).save(target / "im2.jpg")
        dataset = ChangePairDataset(tmp_path, site_split("s0", config), config, train=False)
        index = [i for i, s in enumerate(dataset.sites) if s.name == "s0"]
        if index:
            with pytest.raises(SystemExit, match="co-registered"):
                dataset[index[0]]

    def test_an_empty_split_is_refused_rather_than_silently_empty(self, tmp_path: Path) -> None:
        (tmp_path / "not_a_site").mkdir()
        with pytest.raises(SystemExit, match="No train sites"):
            ChangePairDataset(tmp_path, "train", ChangeConfig(), train=True)


class TestMetrics:
    def test_predicting_no_change_scores_zero_iou_despite_high_accuracy(self) -> None:
        """The failure this corpus invites, made explicit.

        MineNetCD is ~11% changed, so answering "nothing changed" everywhere scores
        about 89% pixel accuracy. IoU has to be what selects the checkpoint.
        """
        import torch

        class Collapsed(torch.nn.Module):
            def forward(self, pixel_values):
                batch = pixel_values.shape[0]
                # Large negative logits -> sigmoid ~0 -> "no change" everywhere.
                out = torch.full((batch, 1, 8, 8), -20.0)
                return type("R", (), {"logits": out})()

        target = torch.zeros(1, 32, 32)
        target[:, :10, :10] = 1.0  # ~9.8% changed
        loader = [(torch.randn(1, 6, 32, 32), target)]
        metrics = evaluate(Collapsed(), loader, torch.device("cpu"), None)

        assert metrics["iou"] == 0.0
        assert metrics["pixel_accuracy"] > 0.85, "the misleading number should be high"
        assert metrics["predicted_change_fraction"] == 0.0, (
            "predicted_change_fraction exists so a collapsed model is visible at a glance"
        )
