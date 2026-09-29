"""The dependency-free engine has to reconstruct a real survey, or say it cannot.

This is what "Auto" runs on a machine without pycolmap and what "Fallback (no
dependencies)" runs everywhere. On 15 Aukerman frames it used to solve 1 of 14 pairs,
triangulate 2 points, emit a 4-vertex mesh and no CRS, and present the frames pasted in
a strip at an assumed 35% overlap as the orthomosaic -- and return success.

These run it on real geotagged frames (downscaled, EXIF kept) with pycolmap hidden, as
auto would on a bare install, and check the properties a survey is judged by.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import survey_imagery_dir

FRAME_COUNT = 8
FRAME_WIDTH = 1200


@pytest.fixture(scope="module")
def custom_run(tmp_path_factory):
    Image = pytest.importorskip("PIL.Image")
    import core.reconstruction_colmap as rc

    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:FRAME_COUNT] if source.is_dir() else []
    if len(frames) < FRAME_COUNT:
        pytest.skip(f"Survey imagery not present at {source} (gitignored).")
    root = tmp_path_factory.mktemp("custom")
    images = root / "images"
    images.mkdir()
    for frame in frames:
        with Image.open(frame) as image:
            height = round(image.height * FRAME_WIDTH / image.width)
            image.resize((FRAME_WIDTH, height)).save(
                images / frame.name, quality=92, exif=image.info.get("exif", b""))

    saved = rc._HAS_PYCOLMAP
    rc._HAS_PYCOLMAP = False  # what `auto` sees on a machine without pycolmap
    try:
        engine = rc.build_reconstructor("auto", profile="fast_preview")
        result = engine.reconstruct(image_dir=images, output_dir=root / "out")
    finally:
        rc._HAS_PYCOLMAP = saved
    twin = json.loads(Path(result.digital_twin_path).read_text(encoding="utf-8"))
    return type(engine).__name__, result, twin, root / "out"


def test_auto_falls_back_to_the_custom_engine(custom_run):
    name, result, _, _ = custom_run
    assert name == "CustomDroneReconstructor"
    assert result.engine == "custom"


def test_the_survey_registers_into_one_model(custom_run):
    _, result, twin, _ = custom_run
    assert result.registered_images >= FRAME_COUNT - 2, twin["warnings"]
    assert result.total_points > 500, "a survey yields hundreds of points, not two"


def test_the_model_is_bundle_adjusted_to_subpixel_error(custom_run):
    _, result, _, _ = custom_run
    assert result.reprojection_error_px is not None
    assert result.reprojection_error_px < 1.5


def test_geotagged_frames_come_out_in_metres_in_utm(custom_run):
    _, result, _, _ = custom_run
    assert result.crs_epsg == 32617, "Aukerman is in UTM zone 17N"
    assert result.geo_rmse_m is not None and result.geo_rmse_m < 10.0


def test_the_orthomosaic_is_georeferenced_and_rectified_not_a_strip(custom_run):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.warp import transform_bounds

    _, result, _, _ = custom_run
    assert result.orthomosaic_cog_path
    with rasterio.open(result.orthomosaic_cog_path) as ortho:
        assert ortho.crs.to_epsg() == 32617
        west, south, east, north = transform_bounds(ortho.crs, "EPSG:4326", *ortho.bounds)
        # The Aukerman site, not an image strip in pixel units.
        assert -81.76 < west < east < -81.74 and 41.29 < south < north < 41.31
        assert result.orthomosaic_gsd_m < 1.0


def test_the_dsm_leaves_unmeasured_cells_empty(custom_run):
    """Holes were filled with the median height: terrain nobody measured."""
    rasterio = pytest.importorskip("rasterio")
    _, result, _, _ = custom_run
    with rasterio.open(result.dsm_cog_path) as dsm:
        band = dsm.read(1)
        assert (band == dsm.nodata).any()
        valid = band[band != dsm.nodata]
        # Heights above the ellipsoid in metres over a ~100 m flight, not depth from camera 1.
        assert 150.0 < float(np.median(valid)) < 350.0


def test_the_textured_mesh_uses_a_texture_aligned_with_it(custom_run):
    _, result, _, out = custom_run
    obj = Path(result.textured_mesh_obj_path).read_text(encoding="utf-8")
    mtl = Path(result.textured_mesh_mtl_path).read_text(encoding="utf-8")
    assert "map_Kd orthomosaic.png" in mtl
    assert "nan" not in obj.lower(), "no vertex may be invented over a nodata cell"


def test_the_hillshade_only_shades_measured_terrain(custom_run):
    rasterio = pytest.importorskip("rasterio")
    _, result, _, _ = custom_run
    assert result.hillshade_path, "a georeferenced run writes a hillshade"
    with rasterio.open(result.dsm_cog_path) as dsm, rasterio.open(result.hillshade_path) as shade:
        measured = dsm.read(1) != dsm.nodata
        shaded = shade.read(1) != shade.nodata
        assert shade.crs.to_epsg() == dsm.crs.to_epsg() and shade.transform == dsm.transform
        assert shaded.any()
        assert not (shaded & ~measured).any(), "no shading where the DSM has no measurement"


def test_the_camera_track_places_each_solved_camera_near_its_gps_fix(custom_run):
    _, result, _, _ = custom_run
    track = json.loads(Path(result.camera_track_geojson_path).read_text(encoding="utf-8"))
    features = track["features"]
    assert len(features) == result.registered_images
    residuals = [f["properties"]["gps_residual_m"] for f in features]
    assert max(residuals) < 15.0, residuals
