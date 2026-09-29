"""A CRS "in metres" is not always ground metres, and terrain in feet is not terrain in metres.

Two holes the geographic/feet guards left open, found by driving the Api on the real
Aukerman DSM/DTM reprojected with rasterio.warp:

* Web Mercator (EPSG:3857) is a projected CRS whose unit is the metre, so every check
  passed -- but its metre is 1/cos(latitude) ground metres. At the Aukerman site
  (41.3 N) a 100 m line measured 133 m, a ~4,600 m3 volume 8,125 m3, and both went into
  the measurement history as real. The same inflation reached the survey packs' areas.
* A terrain DEM in a State Plane feet CRS was accepted as the terrain source, cached
  for offline flight and used by coverage QA, all of which read its heights as metres:
  770 "m" of ground for 235 m.

These reproject the known-answer surface pair and drive the Api the way the app does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import BLOCK_CELLS, BLOCK_VOLUME_M3, GRID, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

rasterio = pytest.importorskip("rasterio")

from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

UTM = 32617
WEB_MERCATOR = 3857
OHIO_NORTH_US_FT = 3734


def _warp(source, target, epsg):
    with rasterio.open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{epsg}", src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs=f"EPSG:{epsg}", transform=transform, width=width, height=height)
        with rasterio.open(target, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                      resampling=Resampling.nearest)
    return target


def _api(tmp_path, *rasters):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Scale", str(tmp_path / "project"))["ok"]
    for path, name in rasters:
        session.register_raster_layer(path, name=name)
    return api


def _lonlat(points_xy):
    lon, lat = geo.projected_to_wgs84([p[0] for p in points_xy], [p[1] for p in points_xy], UTM)
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def _square():
    start = (GRID - BLOCK_CELLS) // 2
    west = ORIGIN_X + (start - 2) * PIXEL_SIZE_M
    north = ORIGIN_Y - (start - 2) * PIXEL_SIZE_M
    return [(west, north - 14.0), (west + 14.0, north - 14.0), (west + 14.0, north), (west, north)]


class TestWebMercatorIsNotGroundMetres:
    def test_the_utm_answer_is_the_control(self, tmp_path, surface_pair):
        api = _api(tmp_path, (surface_pair["dsm"], "DSM"), (surface_pair["dtm"], "DTM"))
        result = api.measure_drawn("volume", _lonlat(_square()))
        assert result["ok"], result
        assert result["preferred"]["net_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3, rel=0.05)

    @pytest.mark.parametrize("kind", ["volume", "slope", "profile", "distance", "area"])
    def test_every_measurement_refuses_a_mercator_raster(self, tmp_path, surface_pair, kind):
        dsm = _warp(surface_pair["dsm"], tmp_path / "dsm_3857.tif", WEB_MERCATOR)
        dtm = _warp(surface_pair["dtm"], tmp_path / "dtm_3857.tif", WEB_MERCATOR)
        api = _api(tmp_path, (dsm, "DSM"), (dtm, "DTM"))
        geometry = _square()[:2] if kind in ("profile", "distance") else _square()
        result = api.measure_drawn(kind, _lonlat(geometry), str(dsm))
        # Before: volume 531 "m3" for a 300 m3 block, a 14 m line as 18.6 m.
        assert result["ok"] is False, result
        assert "EPSG:3857" in result["error"] and "ground metres" in result["error"]
        assert api.list_measurements()["measurements"] == []

    def test_ponding_and_comparisons_refuse_it_too(self, tmp_path, surface_pair):
        dsm = _warp(surface_pair["dsm"], tmp_path / "dsm_3857.tif", WEB_MERCATOR)
        api = _api(tmp_path, (dsm, "DSM"))
        assert api.find_ponding(str(dsm), 0.05)["ok"] is False
        compared = api.compare_survey_volumes(str(surface_pair["dsm"]), str(dsm))
        assert compared["ok"] is False and "ground metres" in compared["error"]

    def test_a_survey_pack_refuses_a_mercator_class_map(self, tmp_path):
        from training.verification.conftest import write_raster

        classes = np.zeros((10, 10), dtype=np.uint16)
        classes[3:7, 3:7] = 1
        utm = write_raster(tmp_path / "classes.tif", classes, nodata=65535)
        mercator = _warp(utm, tmp_path / "classes_3857.tif", WEB_MERCATOR)
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"classes": [{"id": 0, "name": "background"},
                                                  {"id": 1, "name": "building"}]}),
                          encoding="utf-8")
        api = _api(tmp_path)
        assert api.india_land_gis(str(utm), str(schema))["ok"] is True
        refused = api.india_land_gis(str(mercator), str(schema))
        assert refused["ok"] is False and "EPSG:3857" in refused["error"], refused


class TestTerrainHeightsMustBeMetres:
    @pytest.fixture
    def feet_dtm(self, tmp_path, surface_pair):
        return _warp(surface_pair["dtm"], tmp_path / "dtm_ft.tif", OHIO_NORTH_US_FT)

    def test_the_terrain_source_refuses_a_feet_dem(self, tmp_path, feet_dtm, surface_pair):
        api = _api(tmp_path)
        refused = api.set_terrain_source(str(feet_dtm))
        assert refused["ok"] is False and "US survey foot" in refused["error"], refused
        assert api.set_terrain_source(str(surface_pair["dtm"]))["ok"] is True

    def test_the_offline_terrain_cache_refuses_it(self, tmp_path, feet_dtm):
        api = _api(tmp_path)
        refused = api.cache_terrain(str(feet_dtm), "site")
        assert refused["ok"] is False and "metres" in refused["error"], refused
        assert not list((tmp_path / "project").rglob("site.tif"))

    def test_a_dem_without_a_crs_is_refused_as_terrain(self, tmp_path, surface_pair):
        with rasterio.open(surface_pair["dtm"]) as src:
            profile, data = src.profile.copy(), src.read()
        profile.update(crs=None)
        bare = tmp_path / "bare.tif"
        with rasterio.open(bare, "w", **profile) as dst:
            dst.write(data)
        refused = _api(tmp_path).set_terrain_source(str(bare))
        assert refused["ok"] is False and "coordinate reference system" in refused["error"]

    def test_a_lonlat_dem_is_still_accepted(self, tmp_path, surface_pair):
        # SRTM and most national DEMs are EPSG:4326 with heights in metres.
        lonlat = _warp(surface_pair["dtm"], tmp_path / "dtm_4326.tif", 4326)
        assert _api(tmp_path).set_terrain_source(str(lonlat))["ok"] is True
