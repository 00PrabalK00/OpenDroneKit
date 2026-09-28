"""Measuring what was drawn on the map, on the project's own rasters, end to end.

The analysis modules were correct and unreachable: the cockpit's Volume, Cut & Fill,
Slope, Stockpile and Compare buttons passed null where a raster and a geometry belonged,
and the map gives lon/lat while every raster is in a projected CRS. These tests drive the
Api the way the canvas does -- lon/lat vertices in, a registered project layer found by
name -- against GeoTIFFs whose answers are known exactly, and check that every result is
kept in the project's measurement history.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    BLOCK_CELLS,
    BLOCK_HEIGHT_M,
    BLOCK_VOLUME_M3,
    GRID,
    GROUND_M,
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE_M,
)

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

EPSG = 32617


def _lonlat(points_xy):
    lon, lat = geo.projected_to_wgs84([p[0] for p in points_xy], [p[1] for p in points_xy], EPSG)
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def _block_square(margin_cells: float = 2.0):
    """The block's footprint plus a margin, in projected metres."""
    start = (GRID - BLOCK_CELLS) // 2
    west = ORIGIN_X + (start - margin_cells) * PIXEL_SIZE_M
    east = ORIGIN_X + (start + BLOCK_CELLS + margin_cells) * PIXEL_SIZE_M
    north = ORIGIN_Y - (start - margin_cells) * PIXEL_SIZE_M
    south = ORIGIN_Y - (start + BLOCK_CELLS + margin_cells) * PIXEL_SIZE_M
    return [(west, south), (east, south), (east, north), (west, north)]


@pytest.fixture
def api(tmp_path, surface_pair):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    created = api.create_project("Pile survey", str(tmp_path / "project"))
    assert created["ok"], created
    session.register_raster_layer(surface_pair["dsm"], name="DSM")
    session.register_raster_layer(surface_pair["dtm"], name="DTM")
    return api


class TestTheProjectsOwnSurfacesAreFound:
    def test_dsm_and_dtm_are_resolved_from_the_layer_tree(self, api, surface_pair):
        found = api.project_surfaces()
        assert found["ok"]
        assert found["dsm"] == str(surface_pair["dsm"])
        assert found["dtm"] == str(surface_pair["dtm"])

    def test_a_project_with_no_dsm_is_told_to_run_process(self, tmp_path):
        bare = Api(AppSession(store=ProjectStore(tmp_path / "bare.db")))
        bare.create_project("empty", str(tmp_path / "empty"))
        result = bare.measure_drawn("volume", _lonlat(_block_square()))
        assert result["ok"] is False
        assert "no DSM" in result["error"]


class TestDrawnGeometryIsMeasured:
    def test_a_drawn_polygon_measures_the_known_volume(self, api):
        result = api.measure_drawn("volume", _lonlat(_block_square()))
        assert result["ok"], result
        assert result["preferred"]["reference"] == "dtm"
        assert result["preferred"]["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3, abs=1e-3)
        assert result["polygon_supplied"] is True

    def test_cut_and_fill_uses_the_stated_design_level_not_the_dtm(self, api):
        level = GROUND_M + 1.0
        result = api.measure_drawn("cut_fill", _lonlat(_block_square(0)), "",
                                   {"base_elevation_m": level})
        assert result["ok"], result
        assert result["preferred"]["reference"] == "plane"
        assert result["preferred"]["reference_elevation_m"] == level
        block_area = (BLOCK_CELLS * PIXEL_SIZE_M) ** 2
        assert result["preferred"]["cut_volume_m3"] == pytest.approx(
            block_area * (BLOCK_HEIGHT_M - 1.0), rel=0.02)

    def test_cut_and_fill_without_a_level_is_refused(self, api):
        result = api.measure_drawn("cut_fill", _lonlat(_block_square()))
        assert result["ok"] is False
        assert "elevation" in result["error"]

    def test_a_stockpile_writes_its_report_package(self, api):
        result = api.measure_drawn("stockpile", _lonlat(_block_square()))
        assert result["ok"], result
        assert result["preferred"]["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3, abs=1e-3)
        from pathlib import Path

        assert Path(result["report_path"]).is_file()
        assert Path(result["selection_path"]).is_file()

    def test_distance_is_measured_on_the_raster_with_a_surface_length(self, api):
        start = (GRID - BLOCK_CELLS) // 2
        y = ORIGIN_Y - (start + BLOCK_CELLS / 2) * PIXEL_SIZE_M
        line = [(ORIGIN_X + 2.0, y), (ORIGIN_X + 38.0, y)]
        result = api.measure_drawn("distance", _lonlat(line))
        assert result["ok"], result
        assert result["measurement"]["value"] == pytest.approx(36.0, abs=0.01)
        assert result["measurement"]["epsg"] == EPSG
        # The line climbs onto the block and off it again: 3 m up, 3 m down.
        assert result["surface_length_m"] > result["measurement"]["value"]

    def test_area_and_perimeter_of_a_drawn_square(self, api):
        square = _block_square(0)
        area = api.measure_drawn("area", _lonlat(square))
        perimeter = api.measure_drawn("perimeter", _lonlat(square))
        side = BLOCK_CELLS * PIXEL_SIZE_M
        assert area["measurement"]["value"] == pytest.approx(side * side, rel=1e-4)
        assert perimeter["measurement"]["value"] == pytest.approx(4 * side, rel=1e-4)

    def test_a_profile_crosses_the_block(self, api):
        start = (GRID - BLOCK_CELLS) // 2
        y = ORIGIN_Y - (start + BLOCK_CELLS / 2) * PIXEL_SIZE_M
        line = [(ORIGIN_X + 1.0, y), (ORIGIN_X + 39.0, y)]
        result = api.measure_drawn("profile", _lonlat(line))
        assert result["ok"], result
        assert result["elevation_min_m"] == pytest.approx(GROUND_M)
        assert result["elevation_max_m"] == pytest.approx(GROUND_M + BLOCK_HEIGHT_M)
        assert result["samples_without_data"] == 0

    def test_slope_over_a_drawn_polygon(self, api):
        result = api.measure_drawn("slope", _lonlat(_block_square()))
        assert result["ok"], result
        assert result["epsg"] == EPSG


class TestMeasurementsAreKept:
    def test_every_measurement_lands_in_the_history(self, api, tmp_path):
        api.measure_drawn("volume", _lonlat(_block_square()))
        api.measure_drawn("area", _lonlat(_block_square(0)))
        history = api.list_measurements()
        assert history["ok"]
        kinds = [m["measurement_type"] for m in history["measurements"]]
        assert sorted(kinds) == ["area", "volume"]
        volume = next(m for m in history["measurements"] if m["measurement_type"] == "volume")
        assert volume["unit"] == "m3"
        assert volume["geometry"]["type"] == "Polygon"
        assert volume["details"]["preferred"]["reference"] == "dtm"

    def test_the_history_survives_a_restart(self, api, tmp_path):
        api.measure_drawn("area", _lonlat(_block_square(0)))
        reopened = Api(AppSession(store=ProjectStore(tmp_path / "projects.db")))
        assert reopened.list_measurements()["count"] == 1

    def test_a_measurement_can_be_deleted(self, api):
        taken = api.measure_drawn("area", _lonlat(_block_square(0)))
        assert api.delete_measurement(taken["measurement_id"])["ok"]
        assert api.list_measurements()["count"] == 0


def _write(path, array, *, west, north, pixel, epsg=EPSG):
    geo.write_geotiff(path, np.asarray(array, dtype=np.float32), epsg=epsg,
                      west=west, north=north, pixel_size=pixel, cog=False)
    return str(path)


class TestTwoRealSurveysCanBeCompared:
    """Independent reconstructions never share a grid. They used to be refused outright."""

    def test_surveys_on_different_grids_are_resampled_and_say_so(self, api, tmp_path):
        earlier = _write(tmp_path / "e.tif", np.full((40, 40), 50.0), west=ORIGIN_X,
                         north=ORIGIN_Y, pixel=1.0)
        later_values = np.full((50, 50), 50.0)
        # A 10 x 10 m, 2 m high dump in the later survey, on a 0.8 m grid offset by 3 m.
        later_values[10:22, 10:22] = 52.0
        later = _write(tmp_path / "l.tif", later_values, west=ORIGIN_X - 3.0,
                       north=ORIGIN_Y + 3.0, pixel=0.8)
        result = api.compare_survey_volumes(earlier, later)
        assert result["ok"], result
        assert result["resampled"] is True
        assert any("resampled" in w for w in result["warnings"])
        # 9.6 m x 9.6 m x 2 m = 184 m3 before bilinear smoothing at the edges.
        assert result["added_volume_m3"] == pytest.approx(184.3, rel=0.12)
        assert result["removed_volume_m3"] == pytest.approx(0.0, abs=1.0)

    def test_a_systematic_offset_is_reported_not_hidden(self, api, tmp_path):
        earlier = _write(tmp_path / "e.tif", np.full((30, 30), 50.0), west=ORIGIN_X,
                         north=ORIGIN_Y, pixel=1.0)
        later = _write(tmp_path / "l.tif", np.full((30, 30), 50.6), west=ORIGIN_X + 0.5,
                       north=ORIGIN_Y, pixel=1.0)
        result = api.compare_survey_volumes(earlier, later)
        assert result["ok"], result
        assert result["median_difference_m"] == pytest.approx(0.6, abs=1e-3)
        assert any("offset" in w for w in result["warnings"])

    def test_deformation_between_different_grids_is_computed(self, api, tmp_path):
        earlier = _write(tmp_path / "e.tif", np.full((40, 40), 50.0), west=ORIGIN_X,
                         north=ORIGIN_Y, pixel=1.0)
        sunk = np.full((40, 40), 50.0)
        sunk[10:25, 10:25] = 49.0
        later = _write(tmp_path / "l.tif", sunk, west=ORIGIN_X + 0.25,
                       north=ORIGIN_Y - 0.25, pixel=1.0)
        result = api.compare_surveys(earlier, later, 0.05, 0.05, 0.05)
        assert result["ok"], result
        assert result["resampled"] is True
        assert result["moved"] is True
        assert result["regions"][0]["direction"] == "subsidence"


class TestDegreesAreRefusedNotZeroed:
    """A geographic DSM used to return 0 m3, 0 change, "moved: false" and 0 ponds."""

    @pytest.fixture
    def degrees_dsm(self, tmp_path):
        values = np.full((30, 30), 100.0)
        values[10:20, 10:20] = 105.0
        values[2:5, 2:5] = 99.0
        return _write(tmp_path / "wgs84.tif", values, west=-81.76, north=41.31,
                      pixel=0.00001, epsg=4326)

    def test_volume(self, degrees_dsm):
        from core.dsm_analysis import NotProjected, estimate_volume

        with pytest.raises(NotProjected, match="geographic"):
            estimate_volume(degrees_dsm)

    def test_change(self, degrees_dsm):
        from core.change_detection import compare_surfaces
        from core.dsm_analysis import NotProjected

        with pytest.raises(NotProjected):
            compare_surfaces(degrees_dsm, degrees_dsm)

    def test_ponding_through_the_api(self, degrees_dsm):
        result = Api().find_ponding(degrees_dsm, 0.05)
        assert result["ok"] is False
        assert "geographic" in result["error"]

    def test_deformation_through_the_api(self, degrees_dsm):
        result = Api().compare_surveys(degrees_dsm, degrees_dsm, 0.05, 0.05, 0.05)
        assert result["ok"] is False
        assert "geographic" in result["error"]

    def test_ponding_on_an_in_memory_surface(self):
        from core.dsm_analysis import NotProjected, RasterSurface
        from core.ponding import find_ponding

        surface = RasterSurface(np.full((10, 10), 1.0), (1e-5, 0, -81.7, 0, -1e-5, 41.3), 4326)
        with pytest.raises(NotProjected):
            find_ponding(surface, vertical_accuracy_m=0.05)


class TestTheProjectCrsCanBeSet:
    """"Set CRS" called check_spatial_reference with no image list and failed."""

    def test_a_projected_crs_is_kept(self, api):
        result = api.set_project_crs(32617)
        assert result["ok"], result
        assert "UTM zone 17N" in result["name"]
        assert api.get_project_crs()["epsg"] == 32617

    def test_degrees_are_refused(self, api):
        result = api.set_project_crs(4326)
        assert result["ok"] is False
        assert "degrees" in result["error"]

    def test_blank_returns_to_automatic(self, api):
        api.set_project_crs(32617)
        assert api.set_project_crs(None)["epsg"] is None
        assert api.get_project_crs()["epsg"] is None
