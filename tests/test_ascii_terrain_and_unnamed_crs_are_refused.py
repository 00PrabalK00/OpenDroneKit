"""Terrain guards cover every format the Import Terrain picker offers, and scale is judged by the CRS.

Found by the round-2 verifier:

* The Import Terrain picker offers ESRI ASCII grids (*.asc). check_terrain_file only
  looked at GeoTIFF-like suffixes, so a DEM in Ohio North US feet written as .asc with
  its .prj -- the same DEM refused as a GeoTIFF -- was accepted by set_terrain_source,
  and an .asc with no .prj (no CRS at all) was accepted too. The planner then read a
  georeferenced .asc's UTM x/y as mission-local metres, half a million metres off.
* The true-scale check was keyed on the EPSG code, so a Mercator written as
  +proj=merc with no code measured a ~37.6 m ground line as 50 m, and the DSM path
  refused it as having "no coordinate reference system", which was untrue.

These write the known-answer DTM in those forms and drive the Api the way the app does.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from conftest import GRID, GROUND_M, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

rasterio = pytest.importorskip("rasterio")

from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

FEET_PER_METRE = 3.28084
UNNAMED_MERCATOR = "+proj=merc +lon_0=-82 +datum=WGS84 +units=m +no_defs"


def _api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Terrain", str(tmp_path / "project"))["ok"]
    return api


def _to_ascii_grid(source, target, dst_crs=None, scale_heights=1.0):
    """Write ``source`` (optionally reprojected) as an ESRI ASCII grid with a .prj."""
    with rasterio.open(source) as src:
        crs = dst_crs or src.crs
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds)
        # AAIGrid needs square cells.
        cell = max(abs(transform.a), abs(transform.e))
        transform = rasterio.Affine(cell, 0, transform.c, 0, -cell, transform.f)
        data = np.full((height, width), -9999.0, dtype=np.float32)
        reproject(src.read(1), data, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
                  dst_nodata=-9999.0)
    data = np.where(data == -9999.0, -9999.0, data * scale_heights).astype(np.float32)
    with rasterio.open(target, "w", driver="AAIGrid", width=width, height=height, count=1,
                       dtype="float32", crs=crs, transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(data, 1)
    return target


class TestAsciiGridTerrainIsChecked:
    def test_a_feet_ascii_grid_is_refused_as_the_terrain_source(self, tmp_path, surface_pair):
        feet = _to_ascii_grid(surface_pair["dtm"], tmp_path / "dem_ft.asc",
                              dst_crs="EPSG:3734", scale_heights=FEET_PER_METRE)
        assert (tmp_path / "dem_ft.prj").exists()
        api = _api(tmp_path)
        result = api.set_terrain_source(str(feet))
        assert not result["ok"], result
        assert "feet" in result["error"].lower() or "foot" in result["error"].lower()
        assert "metres" in result["error"]
        # The offline cache and coverage QA refuse it the same way.
        assert not api.cache_terrain(str(feet))["ok"]

    def test_an_ascii_grid_without_its_prj_is_refused(self, tmp_path, surface_pair):
        grid = _to_ascii_grid(surface_pair["dtm"], tmp_path / "dem.asc")
        bare = tmp_path / "bare"
        bare.mkdir()
        shutil.copy(grid, bare / "dem.asc")
        api = _api(tmp_path)
        result = api.set_terrain_source(str(bare / "dem.asc"))
        assert not result["ok"], result
        assert "no coordinate reference system" in result["error"]
        assert ".prj" in result["error"]

    def test_an_unreadable_terrain_raster_is_refused(self, tmp_path):
        junk = tmp_path / "dem.asc"
        junk.write_text("this is not a grid\n", encoding="utf-8")
        result = _api(tmp_path).set_terrain_source(str(junk))
        assert not result["ok"]
        assert "cannot be read as a terrain raster" in result["error"]

    def test_a_metric_ascii_grid_is_accepted_and_placed_by_its_crs(self, tmp_path, surface_pair):
        from mission.planner import AssetReferenceFrame, _load_terrain_model_from_path

        grid = _to_ascii_grid(surface_pair["dtm"], tmp_path / "dem_utm.asc")
        api = _api(tmp_path)
        result = api.set_terrain_source(str(grid))
        assert result["ok"], result

        centre_x = ORIGIN_X + GRID * PIXEL_SIZE_M / 2
        centre_y = ORIGIN_Y - GRID * PIXEL_SIZE_M / 2
        lon, lat = geo.projected_to_wgs84([centre_x], [centre_y], 32617)
        frame = AssetReferenceFrame(asset_id="a", origin_lon=float(lon[0]),
                                    origin_lat=float(lat[0]))
        model = _load_terrain_model_from_path(str(grid), frame)
        assert model is not None
        samples = np.asarray(model["samples_local"], dtype=float)
        # Reprojected about the frame origin: the 40 m tile sits within ~30 m of it,
        # not at the UTM easting of 500,000 read as local metres.
        assert np.abs(samples[:, :2]).max() < 30.0
        assert np.allclose(samples[:, 2], GROUND_M)

    def test_the_planner_does_not_read_a_feet_grid_as_metres(self, tmp_path, surface_pair):
        from mission.planner import AssetReferenceFrame, _load_terrain_model_from_path

        feet = _to_ascii_grid(surface_pair["dtm"], tmp_path / "dem_ft.asc",
                              dst_crs="EPSG:3734", scale_heights=FEET_PER_METRE)
        frame = AssetReferenceFrame(asset_id="a", origin_lon=-81.0, origin_lat=41.28)
        # None surfaces in the plan as "the terrain source could not be read".
        assert _load_terrain_model_from_path(str(feet), frame) is None


class TestAMercatorWithNoEpsgCodeIsStillMercator:
    def _mercator(self, tmp_path, surface_pair):
        target = tmp_path / "merc.tif"
        with rasterio.open(surface_pair["dtm"]) as src:
            transform, width, height = calculate_default_transform(
                src.crs, UNNAMED_MERCATOR, src.width, src.height, *src.bounds)
            profile = src.profile.copy()
            profile.update(crs=UNNAMED_MERCATOR, transform=transform, width=width,
                           height=height)
            with rasterio.open(target, "w", **profile) as dst:
                reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=UNNAMED_MERCATOR,
                          resampling=Resampling.nearest)
        with rasterio.open(target) as check:
            assert check.crs is not None and check.crs.to_epsg() is None
        return target

    def test_measure_on_raster_refuses_it(self, tmp_path, surface_pair):
        merc = self._mercator(tmp_path, surface_pair)
        result = _api(tmp_path).measure_on_raster(str(merc), [[5, 5], [60, 5]], "distance")
        assert not result["ok"], result
        assert "differ from ground metres" in result["error"]
        assert "+proj=merc" in result["error"]

    def test_the_dsm_path_says_why_not_that_it_has_no_crs(self, tmp_path, surface_pair):
        from core.dsm_analysis import NotProjected, load_surface

        merc = self._mercator(tmp_path, surface_pair)
        with pytest.raises(NotProjected) as refused:
            load_surface(merc)
        assert "no EPSG code" in str(refused.value)
        assert "no coordinate reference system" not in str(refused.value)

    def test_the_utm_control_still_measures(self, tmp_path, surface_pair):
        result = _api(tmp_path).measure_on_raster(
            str(surface_pair["dtm"]), [[5, 5], [65, 5]], "distance")
        assert result["ok"], result
        assert result["measurement"]["value"] == pytest.approx(60 * PIXEL_SIZE_M, rel=1e-6)
