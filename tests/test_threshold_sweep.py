"""Pairing images with their masks, and refusing to score against the wrong file.

This exists because of a real failure. On Windows, glob returns mixed separators --
``.../val/images\\name.png`` -- so replacing "/images/" or os.sep + "images" + os.sep
matched neither, the path came back unchanged, and the sweep scored the model against
its own input images. Thresholding a light-grey pavement photograph at >127 gives about
67% "foreground", which produced a confident IoU of 0.0255 for a model whose real figure
is 0.606.

Nothing raised. The run completed and printed a table. That is the shape of error worth
testing for: not a crash, but a plausible number that is entirely wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools.threshold_sweep import find_mask


@pytest.fixture
def split(tmp_path):
    """A dataset laid out the way the prepared corpus is."""
    images = tmp_path / "val" / "images"
    masks = tmp_path / "val" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    # A bright photograph and a sparse crack mask, which is the real contrast: most of
    # the image is above 127, almost none of the mask is.
    cv2.imwrite(str(images / "frame_001.png"),
                np.full((64, 64, 3), 200, dtype=np.uint8))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:32, 5:60] = 255
    cv2.imwrite(str(masks / "frame_001.png"), mask)
    return tmp_path


class TestMaskPairing:
    def test_the_mask_is_found_next_to_the_image(self, split):
        image = split / "val" / "images" / "frame_001.png"
        found = find_mask(str(image))

        assert found is not None
        assert "masks" in found

    def test_the_image_is_never_returned_as_its_own_mask(self, split):
        """The failure that produced a confident, meaningless score."""
        image = split / "val" / "images" / "frame_001.png"
        found = find_mask(str(image))

        assert found != str(image)
        assert "images" not in str(found).replace(str(split), "")

    def test_mixed_separators_still_resolve(self, split):
        """Windows glob returns .../images\\name.png, which broke string replacement."""
        mixed = f"{split.as_posix()}/val/images" + "\\" + "frame_001.png"
        found = find_mask(mixed)

        assert found is not None
        assert "masks" in found

    def test_a_forward_slash_path_resolves(self, split):
        image = (split / "val" / "images" / "frame_001.png").as_posix()
        assert find_mask(image) is not None

    def test_the_resolved_mask_is_sparse_as_a_crack_mask_should_be(self, split):
        """Pairing correctly is only useful if the file found is really a mask."""
        found = find_mask(str(split / "val" / "images" / "frame_001.png"))
        mask = cv2.imread(found, cv2.IMREAD_GRAYSCALE)

        foreground = float((mask > 127).mean())
        assert foreground < 0.1, "a crack mask is a few percent, not most of the frame"

    def test_a_different_mask_extension_is_still_found(self, split):
        images = split / "val" / "images"
        masks = split / "val" / "masks"
        cv2.imwrite(str(images / "frame_002.png"),
                    np.full((32, 32, 3), 180, dtype=np.uint8))
        cv2.imwrite(str(masks / "frame_002.jpg"), np.zeros((32, 32), dtype=np.uint8))

        assert find_mask(str(images / "frame_002.png")) is not None

    def test_a_path_with_no_images_component_returns_nothing(self, split):
        assert find_mask(str(split / "somewhere" / "else.png")) is None

    def test_a_missing_mask_returns_none_rather_than_the_image(self, split):
        images = split / "val" / "images"
        cv2.imwrite(str(images / "unpaired.png"), np.zeros((16, 16, 3), dtype=np.uint8))

        assert find_mask(str(images / "unpaired.png")) is None


class TestForegroundGuard:
    """The second line of defence, which is what actually caught the bug."""

    def test_a_photograph_thresholds_to_far_too_much_foreground(self):
        """Why >25% foreground means the pairing is wrong, not the model."""
        photograph = np.full((64, 64), 200, dtype=np.uint8)
        assert float((photograph > 127).mean()) > 0.25

    def test_a_real_crack_mask_is_well_under_the_guard(self, split):
        mask = cv2.imread(str(split / "val" / "masks" / "frame_001.png"),
                          cv2.IMREAD_GRAYSCALE)
        assert float((mask > 127).mean()) < 0.25

    def test_the_sweep_refuses_rather_than_scoring_a_bad_pairing(self):
        """A guard that warned and continued would still publish the wrong number."""
        import inspect

        from tools import threshold_sweep

        source = inspect.getsource(threshold_sweep.sweep)
        assert "raise SystemExit" in source
        assert "0.25" in source
