"""Planning must say out loud when it has stopped following the terrain.

The registry carried this row as implemented with the note "warning surfaced in
app/api.py; needs a test", and the gap mattered more than a missing test usually does:
the warning had already failed silently once. The resolved terrain model lives under
the recipe's `metadata`, and reading it from the recipe root returned None, so the
condition never fired and every plan came back clean regardless of what it had done.

A plan that quietly degrades to flat earth is the most dangerous artefact this program
can produce. The waypoints look identical. The altitudes are the same numbers. The
only difference is whether they mean height above the ground or height above a plane
through the launch point, and over sloping ground that difference is the aircraft.

So these tests do not check that a warning function exists. They plan real missions
through the real Api and assert on what comes back.
"""

from __future__ import annotations

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402

SITE_LON, SITE_LAT = -81.7505, 41.3042


def site_polygon(half: float = 0.002) -> list[list[float]]:
    return [
        [SITE_LON - half, SITE_LAT - half],
        [SITE_LON + half, SITE_LAT - half],
        [SITE_LON + half, SITE_LAT + half],
        [SITE_LON - half, SITE_LAT + half],
    ]


def write_sloping_dem(path, *, west=-81.76, north=41.31, size=200, step=0.0001):
    """A DEM with real relief, so following it differs from flying level."""
    rows = np.linspace(280.0, 340.0, size, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, step, step),
    ) as dst:
        dst.write(np.tile(rows.reshape(-1, 1), (1, size)), 1)
    return path


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("terrain", root_dir=str(tmp_path / "project"))
    return Api(session)


def _warnings(result: dict) -> str:
    assert result["ok"], result.get("error")
    return " ".join(result.get("warnings") or [])


class TestNoTerrainLoaded:
    """The common case, and the one an operator is most likely to miss."""

    def test_planning_without_terrain_says_the_altitudes_are_off_a_flat_plane(self, api):
        assert api.set_aoi(site_polygon())["ok"]
        text = _warnings(api.plan_mission({"altitude_m": 60.0}))
        assert "flat plane" in text.lower(), (
            "a plan with no terrain model reported nothing; the operator has no way to "
            "know the altitudes are not above ground"
        )

    def test_the_warning_names_what_to_do_about_it(self, api):
        # A warning an operator cannot act on is decoration.
        assert api.set_aoi(site_polygon())["ok"]
        text = _warnings(api.plan_mission({"altitude_m": 60.0}))
        assert "terrain" in text.lower()

    def test_the_warning_survives_into_the_plan_result_not_just_the_log(self, api):
        assert api.set_aoi(site_polygon())["ok"]
        result = api.plan_mission({"altitude_m": 60.0})
        assert isinstance(result.get("warnings"), list)
        assert result["warnings"], "warnings list came back empty"


class TestUnreadableTerrainSource:
    """Requesting terrain following and not getting it is worse than never asking.

    The operator has explicitly said the ground is not flat. Falling back to a flat
    plane silently answers a question they did not ask.
    """

    def test_an_unreadable_source_is_reported_rather_than_ignored(self, api, tmp_path):
        assert api.set_aoi(site_polygon())["ok"]
        broken = tmp_path / "not-a-dem.tif"
        broken.write_bytes(b"this is not a GeoTIFF")
        api._session.terrain_source_path = str(broken)

        result = api.plan_mission({"altitude_m": 60.0, "terrain_follow": True})
        # Asserted rather than tolerated: the current behaviour is to plan and warn, and
        # a test that accepted either outcome would keep passing if the warning vanished.
        assert result["ok"], result.get("error")
        text = " ".join(result.get("warnings") or [])
        assert "could not be read" in text, (
            "an unreadable terrain source produced a plan with no warning"
        )
        assert "flat ground" in text

    def test_a_missing_file_does_not_pass_as_terrain_following(self, api, tmp_path):
        assert api.set_aoi(site_polygon())["ok"]
        api._session.terrain_source_path = str(tmp_path / "absent.tif")

        result = api.plan_mission({"altitude_m": 60.0, "terrain_follow": True})
        assert result["ok"], result.get("error")
        text = " ".join(result.get("warnings") or [])
        assert "could not be read" in text, "a missing DEM planned silently"


class TestTerrainActuallyFollowed:
    """The other half of the contract: do not cry wolf on a good plan."""

    def test_a_covering_dem_does_not_raise_the_flat_earth_warning(self, api, tmp_path):
        dem = write_sloping_dem(tmp_path / "site.tif")
        assert api.set_aoi(site_polygon())["ok"]
        api._session.terrain_source_path = str(dem)

        result = api.plan_mission({"altitude_m": 60.0, "terrain_follow": True})
        assert result["ok"], result.get("error")
        text = " ".join(result.get("warnings") or []).lower()
        assert "relative to a flat plane" not in text, (
            "terrain was loaded and usable, but the plan still claimed flat-earth "
            "altitudes; a warning that fires on good plans trains operators to ignore it"
        )
