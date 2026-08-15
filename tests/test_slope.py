"""Gradient measurement, checked against surfaces whose true slope is known.

A slope figure is easy to produce and hard to check, which is why every test here builds
a surface with an arithmetic answer -- a 1-in-4 plane really is 14.036 degrees -- rather
than asserting that the number looks reasonable. The refusals matter as much: a raster
in degrees will produce an angle from a ratio of degrees to metres, and that angle is
wrong in a way nothing downstream can detect.
"""

from __future__ import annotations

import math

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np
from rasterio.transform import from_origin

from core.dsm_analysis import NotGeoreferenced
from core.slope import NotProjected, classify_gradient, measure_slope

# A metric CRS, so a cell step really is a metre.
UTM17N = "EPSG:32617"
ORIGIN_X, ORIGIN_Y = 500_000.0, 4_600_000.0


def write_surface(path, elevation, *, step=1.0, crs=UTM17N):
    data = np.asarray(elevation, dtype="float32")
    profile = {"driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
               "count": 1, "dtype": "float32",
               "transform": from_origin(ORIGIN_X, ORIGIN_Y, step, step),
               "nodata": -9999.0}
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def ramp(rows=30, cols=30, rise_per_metre=0.25, step=1.0):
    """A plane falling east at a known gradient."""
    x = np.arange(cols, dtype="float64") * step
    return np.tile(x * rise_per_metre, (rows, 1))


class TestAKnownPlane:
    def test_a_one_in_four_plane_measures_as_one_in_four(self, tmp_path):
        """arctan(0.25) is 14.0362 degrees. Anything else is a bug, not a tolerance."""
        path = write_surface(tmp_path / "ramp.tif", ramp(rise_per_metre=0.25))
        result = measure_slope(path)

        assert result["ok"] is True
        # Tolerance is the reported precision -- results round to 3 decimals -- not slack
        # in the measurement itself.
        assert result["slope"]["mean_deg"] == pytest.approx(math.degrees(math.atan(0.25)), abs=5e-4)
        assert result["slope"]["mean_percent"] == pytest.approx(25.0, abs=5e-3)

    def test_the_fitted_plane_agrees_with_the_cell_gradient(self, tmp_path):
        path = write_surface(tmp_path / "ramp.tif", ramp(rise_per_metre=0.1))
        slope = measure_slope(path)["slope"]

        assert slope["plane_dip_deg"] == pytest.approx(slope["mean_deg"], abs=1e-6)
        assert slope["planar"] is True
        assert slope["plane_residual_m"] < 1e-6

    def test_a_flat_roof_is_flat_and_has_no_aspect(self, tmp_path):
        path = write_surface(tmp_path / "flat.tif", np.full((20, 20), 12.5))
        slope = measure_slope(path)["slope"]

        assert slope["mean_deg"] == pytest.approx(0.0, abs=1e-9)
        assert slope["aspect_deg"] is None
        assert slope["plane_aspect_deg"] is None

    def test_a_roof_pitch_is_reported_the_way_a_roofer_states_it(self, tmp_path):
        """A 4:12 roof rises 4 units per 12 run: arctan(4/12) = 18.435 degrees."""
        path = write_surface(tmp_path / "roof.tif", ramp(rise_per_metre=4.0 / 12.0))
        slope = measure_slope(path)["slope"]

        assert slope["plane_dip_deg"] == pytest.approx(18.4349, abs=1e-3)
        assert slope["roof_pitch_ratio"] == "4.0:12"


class TestAspect:
    def test_a_surface_falling_east_reports_an_easterly_aspect(self, tmp_path):
        """Elevation rising with x means the surface falls towards the west."""
        path = write_surface(tmp_path / "east.tif", ramp(rise_per_metre=0.2))
        slope = measure_slope(path)["slope"]

        assert slope["aspect_deg"] == pytest.approx(270.0, abs=0.5)

    def test_a_surface_falling_south_reports_a_southerly_aspect(self, tmp_path):
        rows = np.arange(30, dtype="float64") * 0.2
        # Row index increases southward, so elevation falling with row falls south.
        path = write_surface(tmp_path / "south.tif",
                             np.tile((rows[::-1]).reshape(-1, 1), (1, 30)))
        slope = measure_slope(path)["slope"]

        assert slope["aspect_deg"] == pytest.approx(180.0, abs=0.5)


class TestResolutionIsStated:
    def test_the_cell_size_the_gradient_was_measured_across_is_reported(self, tmp_path):
        path = write_surface(tmp_path / "fine.tif", ramp(rise_per_metre=0.25, step=0.05),
                             step=0.05)
        result = measure_slope(path)

        assert result["slope"]["cell_size_m"] == pytest.approx(0.05)
        assert any("cell size" in limit for limit in result["limits"])

    def test_the_mean_survives_a_change_of_resolution(self, tmp_path):
        """The mean is the figure worth quoting; the maximum is mostly grid spacing."""
        coarse = measure_slope(write_surface(
            tmp_path / "coarse.tif", ramp(rise_per_metre=0.3, step=1.0), step=1.0))
        fine = measure_slope(write_surface(
            tmp_path / "fine.tif", ramp(rise_per_metre=0.3, step=0.25), step=0.25))

        assert fine["slope"]["mean_deg"] == pytest.approx(coarse["slope"]["mean_deg"], abs=1e-6)


class TestNonPlanarSurfaces:
    def test_a_valley_is_not_reported_as_one_pitch(self, tmp_path):
        """Two facets meeting at a ridge average to something neither of them is."""
        left = np.tile(np.arange(15, dtype="float64") * 0.5, (30, 1))
        surface = np.concatenate([left, left[:, ::-1]], axis=1)
        result = measure_slope(write_surface(tmp_path / "valley.tif", surface))

        assert result["slope"]["planar"] is False
        assert any("not one flat facet" in limit for limit in result["limits"])

    def test_the_plane_residual_says_how_far_from_flat_it_is(self, tmp_path):
        left = np.tile(np.arange(15, dtype="float64") * 0.5, (30, 1))
        surface = np.concatenate([left, left[:, ::-1]], axis=1)
        slope = measure_slope(write_surface(tmp_path / "valley.tif", surface))["slope"]

        assert slope["plane_residual_m"] > 0.5


class TestClipping:
    def test_a_polygon_restricts_the_measurement_to_one_facet(self, tmp_path):
        left = np.tile(np.arange(15, dtype="float64") * 0.5, (30, 1))
        surface = np.concatenate([left, left[:, ::-1]], axis=1)
        path = write_surface(tmp_path / "valley.tif", surface)

        # The western facet only, which is a clean 1-in-2 plane.
        polygon = [[ORIGIN_X + 2, ORIGIN_Y - 2], [ORIGIN_X + 12, ORIGIN_Y - 2],
                   [ORIGIN_X + 12, ORIGIN_Y - 28], [ORIGIN_X + 2, ORIGIN_Y - 28]]
        clipped = measure_slope(path, polygon_xy=polygon)

        assert clipped["slope"]["planar"] is True
        assert clipped["slope"]["mean_deg"] == pytest.approx(
            math.degrees(math.atan(0.5)), abs=5e-4)


class TestRefusals:
    def test_a_geographic_raster_is_refused_rather_than_measured(self, tmp_path):
        """Degrees horizontally and metres vertically produce a plausible wrong angle."""
        path = write_surface(tmp_path / "wgs84.tif", ramp(), crs="EPSG:4326")
        with pytest.raises(NotProjected, match="geographic"):
            measure_slope(path)

    def test_a_raster_without_a_crs_is_refused(self, tmp_path):
        path = write_surface(tmp_path / "bare.tif", ramp(), crs=None)
        with pytest.raises(NotGeoreferenced):
            measure_slope(path)

    def test_too_few_cells_is_reported_rather_than_averaged(self, tmp_path):
        path = write_surface(tmp_path / "tiny.tif", ramp(rows=2, cols=2))
        result = measure_slope(path)

        assert result["ok"] is False
        assert "at least" in result["reason"]

    def test_scattered_valid_cells_cannot_be_differentiated(self, tmp_path):
        """A gradient across a hole is a gradient across an invented elevation."""
        surface = np.full((20, 20), -9999.0)
        surface[::4, ::4] = 10.0
        path = write_surface(tmp_path / "holes.tif", surface)
        result = measure_slope(path)

        assert result["ok"] is False
        assert "continuous surface" in result["reason"]


class TestThroughTheApi:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("slope", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_a_roof_is_measured_through_the_api(self, api, tmp_path):
        path = write_surface(tmp_path / "roof.tif", ramp(rise_per_metre=0.25))
        result = api.measure_slope(str(path))

        assert result["ok"] is True
        assert result["slope"]["mean_percent"] == pytest.approx(25.0, abs=5e-3)
        assert result["limits"]

    def test_a_geographic_surface_is_refused_with_the_reason(self, api, tmp_path):
        path = write_surface(tmp_path / "wgs84.tif", ramp(), crs="EPSG:4326")
        result = api.measure_slope(str(path))

        assert result["ok"] is False
        assert "geographic" in result["error"]

    def test_a_region_too_small_to_measure_fails_with_its_cell_count(self, api, tmp_path):
        path = write_surface(tmp_path / "tiny.tif", ramp(rows=2, cols=2))
        result = api.measure_slope(str(path))

        assert result["ok"] is False
        assert result["cell_count"] == 4


class TestGradientDescription:
    def test_a_gradient_is_described_without_a_verdict(self, tmp_path):
        described = classify_gradient(8.0, standard="ramp limit quoted by the client")

        assert described["one_in"] == pytest.approx(12.5)
        assert described["slope_deg"] == pytest.approx(4.574, abs=1e-3)
        assert "not encoded here" in described["note"]

    def test_a_level_run_has_no_ratio_rather_than_a_huge_one(self):
        assert classify_gradient(0.0)["one_in"] is None
