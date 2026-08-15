"""Planning against a terrain file that does not cover the site.

The offline cache records the extent of every DEM it holds, but the planner takes its
terrain from whatever path the operator selected, and nothing about a raster says which
site it belongs to. So the cache can be perfectly honest and the plan still degrade to
flat earth: the wrong DEM loads without complaint, and a DEM that stops halfway across
the area follows the ground until the data runs out and then flies level.

These tests go through the real Api rather than the coverage function directly, because
the point is not that the check exists -- it is that planning actually consults it.
"""

from __future__ import annotations

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np
from rasterio.transform import from_origin

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

SITE_LON, SITE_LAT = -81.7505, 41.3042


def write_dem(path, *, west=-81.76, north=41.31, size=200, step=0.0001):
    """A sloping DEM, so a terrain-following plan differs from a flat one."""
    rows = np.linspace(280.0, 340.0, size, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, step, step),
    ) as dst:
        dst.write(np.tile(rows.reshape(-1, 1), (1, size)), 1)
    return path


def site_polygon(lon=SITE_LON, lat=SITE_LAT, half=0.002):
    return [[lon - half, lat - half], [lon + half, lat - half],
            [lon + half, lat + half], [lon - half, lat + half]]


@pytest.fixture
def api(tmp_path):
    """A session on its own database, so tests cannot contend for the shared store."""
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("terrain coverage", root_dir=str(tmp_path / "project"))
    return Api(session)


def plan_over(api, terrain_path, polygon):
    assert api.set_aoi(polygon)["ok"]
    assert api.set_terrain_source(str(terrain_path))["ok"]
    result = api.plan_mission({"altitude_m": 60.0})
    assert result["ok"], result.get("error")
    return result


class TestTerrainSourceIsCheckedAgainstTheArea:
    def test_a_dem_that_covers_the_site_plans_without_a_coverage_warning(self, api, tmp_path):
        result = plan_over(api, write_dem(tmp_path / "site.tif"), site_polygon())

        # Assert the terrain was genuinely read as well, or "no coverage warning" would
        # also be true of a plan that quietly failed to load any terrain at all.
        warnings = " ".join(result.get("warnings", []))
        assert "could not be read" not in warnings
        assert "does not cover the planned area" not in warnings

    def test_a_dem_that_stops_short_warns_that_the_rest_is_flown_level(self, api, tmp_path):
        """The dangerous case: the plan is clean and the transition is invisible."""
        small = write_dem(tmp_path / "small.tif", west=-81.752, north=41.306,
                          size=40, step=0.0001)
        result = plan_over(api, small, site_polygon(half=0.004))

        warnings = " ".join(result.get("warnings", []))
        assert "does not cover the planned area" in warnings
        assert "fly level where it does not" in warnings

    def test_a_dem_for_the_wrong_site_is_named_as_such_while_planning(self, api, tmp_path):
        result = plan_over(api, write_dem(tmp_path / "ohio.tif"),
                           site_polygon(lon=151.2, lat=-33.8))

        warnings = " ".join(result.get("warnings", []))
        assert "does not cover the planned area" in warnings
        assert "nowhere near this site" in warnings


class TestChoosingTheSourceAnswersImmediately:
    def test_selecting_a_dem_reports_coverage_of_the_drawn_area(self, api, tmp_path):
        assert api.set_aoi(site_polygon())["ok"]
        response = api.set_terrain_source(str(write_dem(tmp_path / "site.tif")))

        assert response["coverage"]["covered"] is True
        assert response["coverage"]["cell_size_m"] > 0

    def test_selecting_a_dem_for_elsewhere_says_so_before_planning(self, api, tmp_path):
        assert api.set_aoi(site_polygon(lon=151.2, lat=-33.8))["ok"]
        response = api.set_terrain_source(str(write_dem(tmp_path / "ohio.tif")))

        assert response["coverage"]["covered"] is False
        assert response["coverage"]["reason"] == "elsewhere"

    def test_with_no_area_drawn_there_is_nothing_to_check_yet(self, api, tmp_path):
        response = api.set_terrain_source(str(write_dem(tmp_path / "site.tif")))

        assert response["ok"] is True
        assert response["coverage"] is None
