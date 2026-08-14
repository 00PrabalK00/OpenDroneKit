"""Volume and measurement correctness on a surface whose answer is known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    BLOCK_CELLS,
    BLOCK_VOLUME_M3,
    GRID,
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE_M,
)
from core.dsm_analysis import (
    NotGeoreferenced,
    estimate_volume,
    extract_measurements,
    load_surface,
)


def test_volume_against_dtm_is_exact(surface_pair):
    result = estimate_volume(surface_pair["dsm"], dtm_path=surface_pair["dtm"])
    assert result["ok"]
    assert result["crs_epsg"] == 32617

    against_dtm = next(r for r in result["references"] if r["reference"] == "dtm")
    assert against_dtm["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3, abs=1e-6)
    assert against_dtm["fill_volume_m3"] == pytest.approx(0.0, abs=1e-6)


def test_volume_reports_every_reference_surface(surface_pair):
    """The datum drives the answer, so each option must be stated, not just the best."""
    result = estimate_volume(surface_pair["dsm"], dtm_path=surface_pair["dtm"])
    references = {r["reference"] for r in result["references"]}
    assert "dtm" in references
    assert "plane" in references
    assert result["preferred"]["reference"] == "dtm"
    assert "reference surface" in result["note"]


def test_polygon_clip_halves_the_volume(surface_pair):
    start = (GRID - BLOCK_CELLS) // 2
    half_x = ORIGIN_X + (start + BLOCK_CELLS / 2) * PIXEL_SIZE_M
    polygon = [
        [ORIGIN_X, ORIGIN_Y - GRID * PIXEL_SIZE_M],
        [half_x, ORIGIN_Y - GRID * PIXEL_SIZE_M],
        [half_x, ORIGIN_Y],
        [ORIGIN_X, ORIGIN_Y],
    ]
    clipped = estimate_volume(
        surface_pair["dsm"], dtm_path=surface_pair["dtm"], polygon_xy=polygon
    )
    against_dtm = next(r for r in clipped["references"] if r["reference"] == "dtm")
    assert against_dtm["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3 / 2, abs=1.0)


def test_ungeoreferenced_raster_is_refused(tmp_path):
    """A DSM without a CRS would yield pixel-unit numbers dressed up as cubic metres.

    The custom reconstruction engine writes exactly such a file, so this guard is what
    stops it being measured.
    """
    cv2 = pytest.importorskip("cv2")
    png = tmp_path / "dsm.png"
    cv2.imwrite(str(png), (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8))

    with pytest.raises(NotGeoreferenced):
        load_surface(png)


def test_measurements_sum_defect_geometry(defect_layer, surface_pair):
    result = extract_measurements(
        defects_geojson=defect_layer,
        dsm_path=surface_pair["dsm"],
        dtm_path=surface_pair["dtm"],
    )
    totals = result["totals"]
    assert totals["defect_count"] == 3
    assert totals["total_area_m2"] == pytest.approx(0.8 + 0.05 + 0.3)
    assert totals["largest_area_m2"] == pytest.approx(0.8)
    assert set(totals["by_type"]) == {"spalling", "crack", "efflorescence"}


def test_measurements_recover_structure_height(defect_layer, surface_pair):
    result = extract_measurements(
        defects_geojson=defect_layer,
        dsm_path=surface_pair["dsm"],
        dtm_path=surface_pair["dtm"],
    )
    above_ground = result["terrain"]["canopy_or_structure_height"]
    expected_area = (BLOCK_CELLS * PIXEL_SIZE_M) ** 2
    assert above_ground["max_m"] == pytest.approx(3.0, abs=1e-6)
    assert above_ground["above_ground_area_m2"] == pytest.approx(expected_area, abs=0.5)


def test_measurements_say_so_when_there_is_no_defect_layer(surface_pair):
    """Absent data must read as absent, never as a measured zero."""
    result = extract_measurements(defects_geojson=None, dsm_path=surface_pair["dsm"])
    assert result["totals"]["defect_count"] == 0
    assert "note" in result["totals"]


def test_pixel_area_matches_the_transform(surface_pair):
    surface = load_surface(surface_pair["dsm"])
    assert surface.pixel_size_m == pytest.approx(PIXEL_SIZE_M)
    assert surface.pixel_area_m2 == pytest.approx(PIXEL_SIZE_M**2)
