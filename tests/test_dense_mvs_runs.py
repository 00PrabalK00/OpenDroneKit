"""Dense MVS runs end to end with a COLMAP build that works on this GPU.

On the RTX 5060 dev box the installed COLMAP 4.1.1 (CUDA 13.2) cannot run patch-match
under driver 592.15 (CUDA 13.1); the official COLMAP 4.0.4 archive (CUDA 12.9) can. The
real-run test needs such a build: point ODK_DENSE_COLMAP at its colmap.exe (see
docs/INSTALLATION.md, "Dense reconstruction on RTX 50-series"). It is skipped, and says
why, when no working build is named -- the capability report covers that case.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import survey_imagery_dir
from core import reconstruction_colmap as rc


def _reconstructor(dense_cloud: bool) -> rc.ColmapReconstructor:
    engine = rc.ColmapReconstructor(profile="standard", dense=True)
    engine._cloud_is_dense = dense_cloud
    return engine


def _thin_cloud() -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.column_stack([rng.uniform(0, 400, 900), rng.uniform(0, 300, 900), rng.uniform(0, 5, 900)])


def test_a_coarse_dense_cloud_is_not_told_to_enable_dense_mvs():
    """The warning called a fused dense cloud "sparse" and advised enabling dense MVS."""
    engine = _reconstructor(dense_cloud=True)
    engine._raster_pixel_size(_thin_cloud(), georeferenced=True)
    text = " ".join(engine.warnings)
    assert "900 dense points" in text
    assert "Enable dense MVS" not in text


def test_a_sparse_cloud_still_points_at_dense_mvs():
    engine = _reconstructor(dense_cloud=False)
    engine._raster_pixel_size(_thin_cloud(), georeferenced=True)
    text = " ".join(engine.warnings)
    assert "900 sparse points" in text and "Enable dense MVS" in text


@pytest.fixture(scope="module")
def dense_run(tmp_path_factory):
    binary = os.environ.get("ODK_DENSE_COLMAP", "").strip()
    if not binary or not Path(binary).is_file():
        pytest.skip("Set ODK_DENSE_COLMAP to a COLMAP build whose patch-match runs on this GPU.")
    if not rc.colmap_available():
        pytest.skip("pycolmap is not available in this environment.")
    frames = sorted(survey_imagery_dir().glob("*.JPG"))[:10]
    if len(frames) < 10:
        pytest.skip(f"Survey imagery not present at {survey_imagery_dir()}.")
    from PIL import Image

    root = tmp_path_factory.mktemp("dense")
    images = root / "images"
    images.mkdir()
    for frame in frames:
        with Image.open(frame) as image:
            image.resize((1600, round(image.height * 1600 / image.width))).save(
                images / frame.name, quality=92, exif=image.info.get("exif", b""))
    previous = os.environ.get("ODK_COLMAP")
    os.environ["ODK_COLMAP"] = binary
    try:
        caps = rc.engine_capabilities()
        engine = rc.ColmapReconstructor(profile="standard", dense=True, use_gpu=False)
        result = engine.reconstruct(image_dir=images, output_dir=root / "out")
    finally:
        if previous is None:
            os.environ.pop("ODK_COLMAP", None)
        else:
            os.environ["ODK_COLMAP"] = previous
    return caps, result, root / "out"


def test_the_capability_report_says_dense_is_available(dense_run):
    caps, _, _ = dense_run
    assert caps["dense_stereo"] is True, caps["dense_stereo_reason"]


def test_the_products_come_from_a_fused_dense_cloud(dense_run):
    _, result, out = dense_run
    warnings = " ".join(result.warnings)
    assert "Dense MVS skipped" not in warnings and "Dense MVS failed" not in warnings, warnings
    assert result.dense_point_count > 0
    fused = out / "colmap" / "dense" / "fused.ply"
    assert fused.is_file(), "the dense cloud is COLMAP's own fusion output"
    header = fused.read_bytes()[:400].decode("ascii", "replace")
    fused_count = int(header.split("element vertex ")[1].split()[0])
    assert fused_count == result.dense_point_count
    ply = (out / "reconstruction.ply").read_bytes()[:400].decode("ascii", "replace")
    assert int(ply.split("element vertex ")[1].split()[0]) == fused_count
    assert "Enable dense MVS" not in warnings and "sparse points" not in warnings
    assert result.crs_epsg == 32617
