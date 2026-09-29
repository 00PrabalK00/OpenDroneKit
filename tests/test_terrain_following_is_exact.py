"""Terrain following must hold the requested height above the ground, not half of it.

The audit flew a grid over a DEM rising 18.7 m and measured 37.6-46.6 m clearance
against a 50 m target: the planner followed about 52% of the slope. Two causes. The DEM
was reduced to scattered samples and interpolated with a 1/d weight over every sample on
the raster, which averages the whole hill into every point. And heights were referenced
to the mean elevation of the DEM, while every exported altitude is relative to home.

AMSL mode was flat planning under another name: the terrain delta was zeroed for it and
the file was still written relative to home.

These tests plan over a real GeoTIFF through the Api, export the .plan, and check the
height above the DEM at every waypoint.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402

AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]
WEST, NORTH, STEP, SIZE = -81.7600, 41.3100, 0.0001, 200
TARGET_AGL = 50.0


def ground(lon: float, lat: float) -> float:
    """The DEM below, analytically: 280 m at the north edge rising 60 m southward, plus
    a 0.2 m/m eastward slope so the surface is not aligned with any grid axis."""
    row = (NORTH - lat) / STEP - 0.5
    col = (lon - WEST) / STEP - 0.5
    mx = 111_320.0 * np.cos(np.radians(41.30))
    return 280.0 + 60.0 * row / (SIZE - 1) + 0.2 * col * STEP * mx


@pytest.fixture
def dem(tmp_path) -> Path:
    path = tmp_path / "slope.tif"
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    lat = NORTH - (rows + 0.5) * STEP
    lon = WEST + (cols + 0.5) * STEP
    z = np.vectorize(ground)(lon, lat).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(WEST, NORTH, STEP, STEP)) as dst:
        dst.write(z, 1)
    return path


@pytest.fixture
def api(tmp_path) -> Api:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("terrain", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi(AOI)["ok"]
    return api


def exported_plan(api: Api, tmp_path: Path, options: dict) -> tuple[dict, dict]:
    planned = api.plan_mission(options)
    assert planned["ok"], planned.get("error")
    written = api.export_mission(["qgc_plan", "kml"], str(tmp_path / "out"))["written"]
    return planned, written


def nav_items(written: dict) -> tuple[list[dict], list[float]]:
    payload = json.loads(Path(written["qgc_plan"]).read_text())
    items = [i for i in payload["mission"]["items"] if i["command"] in (16, 19)]
    return items, payload["mission"]["plannedHomePosition"]


class TestAboveGroundLevel:
    @pytest.mark.parametrize("template", ["grid", "corridor", "double_grid"])
    def test_every_waypoint_is_at_the_requested_height_above_the_dem(self, api, dem, tmp_path, template):
        assert api.set_terrain_source(str(dem))["ok"]
        planned, written = exported_plan(api, tmp_path, {"template": template, "altitude_m": TARGET_AGL})
        assert not any("could not be read" in w or "does not cover" in w for w in planned["warnings"])
        items, home = nav_items(written)
        assert {i["frame"] for i in items} == {3}
        home_ground = ground(home[1], home[0])
        agl = [i["params"][6] + home_ground - ground(i["params"][5], i["params"][4]) for i in items]
        assert min(agl) == pytest.approx(TARGET_AGL, abs=0.5)
        assert max(agl) == pytest.approx(TARGET_AGL, abs=0.5)

    def test_the_relief_is_real_so_the_test_can_fail(self, api, dem, tmp_path):
        """Guard against a vacuous pass: the site must actually slope."""
        assert api.set_terrain_source(str(dem))["ok"]
        _, written = exported_plan(api, tmp_path, {"template": "grid", "altitude_m": TARGET_AGL})
        items, _ = nav_items(written)
        grounds = [ground(i["params"][5], i["params"][4]) for i in items]
        assert max(grounds) - min(grounds) > 15.0


class TestAbsoluteAltitude:
    def test_amsl_exports_absolute_altitudes_that_clear_the_ground(self, api, dem, tmp_path):
        assert api.set_terrain_source(str(dem))["ok"]
        planned, written = exported_plan(api, tmp_path, {
            "template": "grid", "altitude_m": TARGET_AGL, "terrain_follow_mode": "amsl"})
        assert planned["summary"]["altitude_frame"] == "amsl"
        items, home = nav_items(written)
        assert {i["frame"] for i in items} == {0}, "AMSL must use MAV_FRAME_GLOBAL"
        agl = [i["params"][6] - ground(i["params"][5], i["params"][4]) for i in items]
        assert min(agl) == pytest.approx(TARGET_AGL, abs=0.5)
        assert max(agl) == pytest.approx(TARGET_AGL, abs=0.5)
        assert home[2] == pytest.approx(ground(home[1], home[0]), abs=0.5)

        kml = ET.parse(written["kml"]).getroot()
        modes = {e.text for e in kml.iter("{http://www.opengis.net/kml/2.2}altitudeMode")}
        assert modes == {"absolute"}

    def test_amsl_without_terrain_says_it_cannot_be_absolute(self, api, tmp_path):
        planned, written = exported_plan(api, tmp_path, {"template": "grid", "terrain_follow_mode": "amsl"})
        assert planned["summary"]["altitude_frame"] == "relative_to_home"
        assert any("AMSL" in w for w in planned["warnings"])


class TestTerrainCsv:
    def _csv(self, tmp_path: Path, header: str) -> Path:
        path = tmp_path / "terrain.csv"
        lines = [header]
        for lat in np.arange(41.302, 41.306, 0.0002):
            for lon in np.arange(-81.755, -81.750, 0.0002):
                lines.append(f"{lon:.6f},{lat:.6f},{ground(lon, lat):.3f}")
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_an_elev_column_is_read(self, api, tmp_path):
        assert api.set_terrain_source(str(self._csv(tmp_path, "lon,lat,elev")))["ok"]
        planned, written = exported_plan(api, tmp_path, {"template": "grid", "altitude_m": TARGET_AGL})
        assert not any("could not be read" in w for w in planned["warnings"])
        items, home = nav_items(written)
        agl = [i["params"][6] + ground(home[1], home[0]) - ground(i["params"][5], i["params"][4])
               for i in items]
        assert max(abs(a - TARGET_AGL) for a in agl) < 1.0

    def test_an_unusable_header_is_named_in_the_warning(self, api, tmp_path):
        assert api.set_terrain_source(str(self._csv(tmp_path, "lon,lat,metres")))["ok"]
        planned = api.plan_mission({"template": "grid"})
        text = " ".join(planned["warnings"])
        assert "could not be read" in text and "metres" in text
