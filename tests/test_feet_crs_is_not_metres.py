"""A projected CRS in feet is not a CRS in metres.

The geographic-CRS guard let projected rasters in US survey feet through. Every surface
analysis pairs horizontal spacing with elevations in metres, so on the real Aukerman DSM
reprojected to EPSG:3734 (Ohio North, US ft) a 4,628 m3 volume came out as 49,315 "m3",
a 10.1 deg mean slope as 2.6 deg, and a 100 m line was kept in the history as "328 m".

These tests reproject the known-answer surface pair to EPSG:3734 with rasterio.warp and
drive the Api the way the map does.
"""

from __future__ import annotations

import pytest

from conftest import BLOCK_CELLS, GRID, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

rasterio = pytest.importorskip("rasterio")

from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

UTM = 32617
OHIO_NORTH_US_FT = 3734


def _to_feet(source, target):
    with rasterio.open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{OHIO_NORTH_US_FT}", src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs=f"EPSG:{OHIO_NORTH_US_FT}", transform=transform,
                       width=width, height=height)
        with rasterio.open(target, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=f"EPSG:{OHIO_NORTH_US_FT}",
                      resampling=Resampling.nearest)
    return target


def _lonlat(points_xy):
    lon, lat = geo.projected_to_wgs84([p[0] for p in points_xy],
                                      [p[1] for p in points_xy], UTM)
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def _square():
    start = (GRID - BLOCK_CELLS) // 2
    west = ORIGIN_X + (start - 2) * PIXEL_SIZE_M
    east = west + 20.0
    north = ORIGIN_Y - (start - 2) * PIXEL_SIZE_M
    south = north - 20.0
    return [(west, south), (east, south), (east, north), (west, north)]


def _api(tmp_path, dsm, dtm=None):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Feet", str(tmp_path / "project"))["ok"]
    session.register_raster_layer(dsm, name="DSM")
    if dtm:
        session.register_raster_layer(dtm, name="DTM")
    return api


@pytest.fixture
def feet_api(tmp_path, surface_pair):
    dsm = _to_feet(surface_pair["dsm"], tmp_path / "dsm_ft.tif")
    dtm = _to_feet(surface_pair["dtm"], tmp_path / "dtm_ft.tif")
    return _api(tmp_path, dsm, dtm)


@pytest.mark.parametrize("kind", ["volume", "slope", "profile"])
def test_surface_analyses_refuse_a_feet_dsm_in_words(feet_api, kind):
    geometry = _square()[:2] if kind == "profile" else _square()
    result = feet_api.measure_drawn(kind, _lonlat(geometry))
    assert result["ok"] is False, result
    assert "US survey foot" in result["error"]
    assert "metres" in result["error"]
    assert feet_api.list_measurements()["measurements"] == []


def test_ponding_refuses_a_feet_dsm(feet_api):
    result = feet_api.find_ponding(vertical_accuracy_m=0.05)
    assert result["ok"] is False, result
    assert "US survey foot" in result["error"]


def test_a_line_on_a_feet_raster_is_kept_in_metres(tmp_path, surface_pair):
    line = _square()[:2]  # 20 m along a UTM easting
    metric = _api(tmp_path / "m", surface_pair["dsm"]).measure_drawn("distance", _lonlat(line))
    assert metric["ok"], metric

    feet_dsm = _to_feet(surface_pair["dsm"], tmp_path / "dsm_ft.tif")
    api = _api(tmp_path / "ft", feet_dsm)
    result = api.measure_drawn("distance", _lonlat(line))
    assert result["ok"], result
    assert result["measurement"]["native_unit"] == "US survey foot"
    assert result["measurement"]["native_value"] == pytest.approx(
        result["value_m"] / 0.30480060960121924)
    # Same ground line, same metres, whichever CRS the raster happens to be in.
    assert result["value_m"] == pytest.approx(metric["value_m"], rel=2e-3)
    assert result["value_m"] == pytest.approx(20.0, rel=5e-3)
    kept = api.list_measurements()["measurements"]
    assert len(kept) == 1
    assert kept[0]["unit"] == "m"
    assert kept[0]["value"] == pytest.approx(result["value_m"])


def test_an_area_on_a_feet_raster_is_kept_in_square_metres(tmp_path, surface_pair):
    feet_dsm = _to_feet(surface_pair["dsm"], tmp_path / "dsm_ft.tif")
    api = _api(tmp_path, feet_dsm)
    result = api.measure_drawn("area", _lonlat(_square()))
    assert result["ok"], result
    assert result["value_m"] == pytest.approx(400.0, rel=1e-2)
    kept = api.list_measurements()["measurements"][0]
    assert (kept["unit"], kept["value"]) == ("m2", pytest.approx(result["value_m"]))


def test_the_project_crs_cannot_be_set_to_feet(tmp_path):
    api = Api(AppSession(store=ProjectStore(tmp_path / "projects.db")))
    api.create_project("crs", str(tmp_path / "project"))
    refused = api.set_project_crs(OHIO_NORTH_US_FT)
    assert refused["ok"] is False
    assert "US survey foot" in refused["error"]
    assert api.get_project_crs()["epsg"] is None
    assert api.set_project_crs(UTM)["ok"] is True
