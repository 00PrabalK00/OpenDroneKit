"""Tiles from different sources have to mean the same thing on the ground.

Measured over 60 samples of each source, the corpus runs 0.20-0.50 m per pixel on
OpenEarthMap and 2.91-4.77 on SpaceNet 7 -- a spread of twenty-four times -- and the tiler
cut a fixed 518-pixel window from all of it. That window is 104 m across at one end of the
range and 2.5 km at the other: the same tensor shape standing for ground areas that have
almost nothing to do with each other.

So "building" meant several incompatible things during training. On the coarse source a
building is a two-pixel blob, and a model carrying that prior into fine imagery predicts
building over far more of the frame than is there -- which is the holdout failure exactly:
0.244 of the frame predicted against a labelled 0.092.

The registry note blamed the corpus for being small and called SpaceNet 7 "0.5 m satellite
imagery". The second half is wrong by a factor of eight, measured here from the files' own
geotransforms, and it is the better explanation of the first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "training" / "data" / "prepared" / "shared_semantic" / "corpus.json"

pytest.importorskip("rasterio")
pytest.importorskip("cv2")

from training.semantic_tiles import (  # noqa: E402
    SemanticTileDataset,
    resample_to_scale,
    sample_gsd,
)


def test_resampling_leaves_labels_as_class_ids():
    """A mask is not an image: averaging class 1 with class 3 invents class 2."""
    image = np.zeros((3, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.int64)
    mask[:4] = 1
    mask[4:] = 3
    no_data = np.zeros((8, 8), dtype=bool)

    _out_image, out_mask, _out_nodata = resample_to_scale(image, mask, no_data, 4.0)
    assert out_mask.shape == (32, 32)
    assert set(np.unique(out_mask)) <= {1, 3}, "interpolation invented a class that is not a class"


def test_the_no_data_plane_survives_resampling():
    """No-data is evidence of nothing, and must not become a confident negative."""
    image = np.zeros((3, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.int64)
    no_data = np.zeros((8, 8), dtype=bool)
    no_data[:4] = True

    _i, _m, out = resample_to_scale(image, mask, no_data, 2.0)
    assert out.dtype == bool
    assert out[:8].all() and not out[8:].any()


def test_a_factor_of_one_changes_nothing():
    image = np.random.default_rng(0).random((3, 6, 6)).astype(np.float32)
    mask = np.zeros((6, 6), dtype=np.int64)
    no_data = np.zeros((6, 6), dtype=bool)
    out_image, out_mask, _ = resample_to_scale(image, mask, no_data, 1.0)
    assert np.array_equal(out_image, image)
    assert out_mask.shape == mask.shape


@pytest.mark.skipif(not CORPUS.exists(), reason="the built corpus is not on this machine")
class TestTheRealCorpus:
    """Measured against the actual files, because the assumption was the bug."""

    def test_the_two_sources_really_do_disagree_about_scale(self):
        payload = json.loads(CORPUS.read_text(encoding="utf-8"))
        by_source = {}
        for sample in payload["samples"]:
            source = sample["source"]
            if source not in by_source and Path(sample["image"]).exists():
                by_source[source] = sample_gsd(sample)
            if len(by_source) == 2:
                break

        assert set(by_source) == {"openearthmap", "spacenet7"}
        fine, coarse = by_source["openearthmap"], by_source["spacenet7"]
        # Measured over 60 samples of each: OpenEarthMap runs 0.20-0.50 m/px and varies
        # by region, SpaceNet 7 runs 2.91-4.77. The full corpus spans a factor of 24.
        assert 0.15 < fine < 0.60, f"OpenEarthMap read as {fine} m/px"
        # The registry called this 0.5 m. It is a Planet mosaic, and nothing like it.
        assert 2.5 < coarse < 5.5, f"SpaceNet 7 read as {coarse} m/px"
        assert coarse / fine > 5.0

    def test_tiles_come_out_at_one_scale_when_a_target_is_set(self):
        dataset = SemanticTileDataset(CORPUS, "train", tile_size=64, target_gsd=0.5)
        available = [s for s in dataset.samples if Path(s["image"]).exists()]
        if len(available) < 2:
            pytest.skip("corpus images are not present on this machine")
        dataset.samples = available

        by_source = {}
        for index, sample in enumerate(dataset.samples):
            if sample["source"] not in by_source:
                by_source[sample["source"]] = dataset[index]
            if len(by_source) == 2:
                break

        for source, item in by_source.items():
            image = item["image"]
            assert image.shape[-2:] == (64, 64), f"{source} produced {image.shape}"
            assert item["mask"].shape == (64, 64)


class TestAPackedCorpusKeepsItsScale:
    """JPEG carries no geotransform, which is where this fix could have died quietly.

    The packer re-encodes images to JPEG and pre-rasterises labels, which is what makes a
    14 GB corpus fit on a rented box. It also throws away the georeferencing -- so a
    trainer reading a packed corpus finds no CRS, measures no ground sample distance, and
    would fall straight back to the fixed-pixel cropping that broke the last model. The
    run would look entirely normal for hours.
    """

    def test_a_recorded_gsd_is_used_without_opening_the_file(self):
        # The path does not exist. If this returns a number, nothing touched the disk.
        assert sample_gsd({"image": "/nonexistent/packed.jpg", "gsd_m": 0.42}) == 0.42

    def test_a_sample_with_no_gsd_refuses_rather_than_guessing(self, tmp_path):
        from training.semantic_tiles import SemanticTileError

        manifest = tmp_path / "corpus.json"
        manifest.write_text(json.dumps({
            "schema": {"classes": [{"id": 0, "name": "background"}, {"id": 1, "name": "building"}]},
            "samples": [{
                "id": "packed:1", "split": "train", "source": "packed",
                "image": "images/1.jpg", "label": "labels/1.png",
                "label_format": "raster_class_ids",
            }],
        }), encoding="utf-8")

        dataset = SemanticTileDataset(manifest, "train", tile_size=32, target_gsd=0.30)
        with pytest.raises(SemanticTileError) as raised:
            dataset[0]
        # The message has to say what would fix it, or it is just a crash.
        assert "ground sample distance" in str(raised.value)
        assert "repack" in str(raised.value)

    def test_the_packer_records_it(self):
        source = (REPO_ROOT / "tools" / "pack_semantic_corpus.py").read_text(encoding="utf-8")
        assert '"gsd_m"' in source, "a packed corpus without gsd_m silently loses the fix"
        assert "no ground sample distance" in source, "an unmeasurable sample must be recorded, not packed"
