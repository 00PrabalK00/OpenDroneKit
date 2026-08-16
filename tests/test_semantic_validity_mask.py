"""No-data pixels must not be trained on as confident background.

SpaceNet 7 tiles are mosaics, and a tile that does not fully cover its footprint marks
the uncovered pixels transparent in a fourth band. One tile in eight in this corpus has
such a region -- around five per cent of its area in the case that surfaced first.

The loader read bands 1, 2 and 3 and ignored the fourth. Those pixels then arrived as
dark ground, and because the building labels contain no polygons over them, the loss
scored them as correct background. The model was being taught that absence of data is a
confident negative, which is the same mistake as scoring a survey for finding nothing in
an area it never flew.

Marked ignore rather than dropped: the pixels keep their position in the tile so the
geometry is unchanged, and the loss simply does not score them.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from training.semantic_tiles import IGNORE_INDEX, read_semantic_sample  # noqa: E402


def write_image(path, *, bands: int, invalid_rows: int = 0, size: int = 16):
    data = np.full((bands, size, size), 120, dtype="uint8")
    if bands >= 4:
        data[3, :, :] = 255
        if invalid_rows:
            data[3, :invalid_rows, :] = 0
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=bands,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)
    return str(path)


def write_label(path, *, size: int = 16, value: int = 0):
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(np.full((1, size, size), value, dtype="uint8"))
    return str(path)


def sample(image, label, **kwargs) -> dict:
    base = {"image": image, "label": label, "label_format": "raster_class_ids"}
    base.update(kwargs)
    return base


class TestValidityIsHonoured:
    def test_transparent_pixels_become_ignore(self, tmp_path) -> None:
        image = write_image(tmp_path / "img.tif", bands=4, invalid_rows=4)
        label = write_label(tmp_path / "lbl.tif")
        _, mask = read_semantic_sample(sample(image, label))
        assert (mask[:4, :] == IGNORE_INDEX).all(), (
            "no-data pixels were left as background; the model learns that absence of "
            "data is a confident negative"
        )

    def test_valid_pixels_keep_their_class(self, tmp_path) -> None:
        image = write_image(tmp_path / "img2.tif", bands=4, invalid_rows=4)
        label = write_label(tmp_path / "lbl2.tif", value=1)
        _, mask = read_semantic_sample(sample(image, label))
        assert (mask[4:, :] == 1).all()

    def test_a_fully_opaque_tile_is_untouched(self, tmp_path) -> None:
        image = write_image(tmp_path / "img3.tif", bands=4, invalid_rows=0)
        label = write_label(tmp_path / "lbl3.tif", value=2)
        _, mask = read_semantic_sample(sample(image, label))
        assert not (mask == IGNORE_INDEX).any()

    def test_a_three_band_image_still_loads(self, tmp_path) -> None:
        # OpenEarthMap tiles are RGB with no alpha; they must not be affected.
        image = write_image(tmp_path / "img4.tif", bands=3)
        label = write_label(tmp_path / "lbl4.tif", value=1)
        _, mask = read_semantic_sample(sample(image, label))
        assert not (mask == IGNORE_INDEX).any()

    def test_the_image_itself_is_unchanged_in_shape(self, tmp_path) -> None:
        """Ignore marks the label, not the pixels: geometry must not shift."""
        image = write_image(tmp_path / "img5.tif", bands=4, invalid_rows=4)
        label = write_label(tmp_path / "lbl5.tif")
        array, mask = read_semantic_sample(sample(image, label))
        assert array.shape == (3, 16, 16)
        assert mask.shape == (16, 16)

    def test_ignore_survives_a_class_map_remap(self, tmp_path) -> None:
        image = write_image(tmp_path / "img6.tif", bands=4, invalid_rows=4)
        label = write_label(tmp_path / "lbl6.tif", value=3)
        _, mask = read_semantic_sample(
            sample(image, label, class_map={"3": 1})
        )
        assert (mask[:4, :] == IGNORE_INDEX).all()
        assert (mask[4:, :] == 1).all()
