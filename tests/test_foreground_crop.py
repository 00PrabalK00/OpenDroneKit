"""Foreground-anchored cropping, for corpora where the target is a rounding error.

The Duke PV masks are about 0.15% foreground on 4000x2250 frames. RandomResizedCrop
takes 50-100% of that and resizes to 512, so most training crops contain no panel and
the few that do shrink one to a handful of pixels. solar_module_segformer_b2 reached
IoU 0.189 for that reason, not because the task is hard.

These tests pin the two behaviours that make the crop safe rather than merely present:
a mask with no foreground must come back untouched, and a chosen window must actually
contain the pixel it was centred on even when that pixel sits against an edge.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from training.train_seg import _foreground_crop


def _blank(height: int = 400, width: int = 600):
    return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8)


class TestForegroundCrop:
    def test_an_empty_mask_is_returned_unchanged(self) -> None:
        # Background-only tiles are legitimate training data. Cropping "around" a
        # foreground that does not exist would either crash or invent a window.
        image, mask = _blank()
        out_image, out_mask = _foreground_crop(image, mask, 128)
        assert out_image.shape == image.shape
        assert out_mask.shape == mask.shape

    def test_the_crop_contains_the_foreground(self) -> None:
        image, mask = _blank()
        mask[210:214, 300:304] = 1
        random.seed(0)
        _, out_mask = _foreground_crop(image, mask, 128)
        assert out_mask.shape == (128, 128)
        assert out_mask.sum() > 0, "cropped away the only foreground in the image"

    @pytest.mark.parametrize("row,col", [(0, 0), (399, 599), (0, 599), (399, 0), (200, 0)])
    def test_foreground_against_any_edge_is_still_captured(self, row: int, col: int) -> None:
        # The window is clamped back inside the image rather than rejected, so targets
        # near a border stay representable. Dropping them would bias the model toward
        # panels that happen to sit mid-frame.
        image, mask = _blank()
        mask[row, col] = 1
        random.seed(1)
        _, out_mask = _foreground_crop(image, mask, 128)
        assert out_mask.shape == (128, 128)
        assert out_mask.sum() > 0, f"foreground at ({row},{col}) was cropped away"

    def test_an_image_smaller_than_the_window_is_untouched(self) -> None:
        image, mask = _blank(64, 64)
        mask[10, 10] = 1
        out_image, out_mask = _foreground_crop(image, mask, 128)
        assert out_image.shape[:2] == (64, 64)
        assert out_mask.shape == (64, 64)

    def test_the_crop_raises_the_foreground_fraction(self) -> None:
        # The whole point: the model should see panels at the scale they were shot.
        image, mask = _blank(2250, 4000)
        mask[1100:1160, 2000:2100] = 1
        before = mask.mean()
        random.seed(2)
        _, out_mask = _foreground_crop(image, mask, 512)
        after = out_mask.mean()
        assert after > before * 10, f"foreground fraction {before:.5f} -> {after:.5f}"


class TestConfigDefault:
    def test_the_feature_is_off_unless_asked_for(self) -> None:
        # Existing runs must not silently change sampling; crack_seg does not need this.
        from training.train_seg import SegConfig

        assert SegConfig().foreground_crop_prob == 0.0
