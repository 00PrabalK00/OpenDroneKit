"""Measuring on an orthomosaic, and refusing to measure on something that isn't one.

Two kinds of test here. The first checks the arithmetic against a raster whose geometry
is known exactly, so a wrong answer is a wrong answer rather than a plausible one. The
second checks the refusals, which matter more: a measurement taken from a PNG or from a
raster in degrees would come back as a confident number in the wrong units, land in a
report, and be believed.
"""

from __future__ import annotations

import math

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np
from rasterio.transform import from_origin

from core.raster_measurement import (
    NotGeoreferenced,
    NotProjected,
    measure_area,
    measure_distance,
    measure_perimeter,
    pixel_to_world,
    raster_ground_sample_distance,
)

# A UTM 17N raster at exactly 1 m per pixel makes every expected answer checkable by
# hand: n pixels is n metres.
EPSG_UTM = 32617
ORIGIN_X, ORIGIN_Y = 437000.0, 4572900.0
CELL = 1.0
SIZE = 100


def write_raster(path, *, crs=f"EPSG:{EPSG_UTM}", cell=CELL):
    transform = from_origin(ORIGIN_X, ORIGIN_Y, cell, cell)
    data = np.arange(SIZE * SIZE, dtype="float32").reshape(SIZE, SIZE)
    profile = {
        "driver": "GTiff", "height": SIZE, "width": SIZE, "count": 1,
        "dtype": "float32", "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def utm_raster(tmp_path):
    return write_raster(tmp_path / "orthomosaic.tif")


class TestDistance:
    def test_a_horizontal_line_measures_its_pixel_count_in_metres(self, utm_raster):
        """At 1 m per pixel, 40 pixels is 40 m. Anything else is a bug in the transform."""
        result = measure_distance(utm_raster, [(10, 10), (50, 10)])
        assert result.value == pytest.approx(40.0, abs=0.01)
        assert result.unit == "metre"
        assert result.epsg == EPSG_UTM

    def test_a_diagonal_uses_real_geometry_not_pixel_stepping(self, utm_raster):
        result = measure_distance(utm_raster, [(0, 0), (30, 40)])
        assert result.value == pytest.approx(50.0, abs=0.01)

    def test_a_polyline_sums_its_segments(self, utm_raster):
        result = measure_distance(utm_raster, [(0, 0), (10, 0), (10, 10)])
        assert result.value == pytest.approx(20.0, abs=0.01)

    def test_a_single_point_is_not_a_distance(self, utm_raster):
        with pytest.raises(ValueError, match="at least two points"):
            measure_distance(utm_raster, [(5, 5)])

    def test_the_world_coordinates_are_returned_for_inspection(self, utm_raster):
        result = measure_distance(utm_raster, [(0, 0), (10, 0)])
        x, y = result.vertices_world[0]
        # rasterio.xy returns cell centres, half a cell in from the origin.
        assert x == pytest.approx(ORIGIN_X + CELL / 2, abs=0.01)
        assert y == pytest.approx(ORIGIN_Y - CELL / 2, abs=0.01)


class TestArea:
    def test_a_square_measures_its_side_squared(self, utm_raster):
        result = measure_area(utm_raster, [(10, 10), (30, 10), (30, 30), (10, 30)])
        assert result.value == pytest.approx(400.0, abs=1.0)
        assert result.unit == "square metre"

    def test_a_triangle_is_half_its_bounding_rectangle(self, utm_raster):
        result = measure_area(utm_raster, [(0, 0), (20, 0), (0, 20)])
        assert result.value == pytest.approx(200.0, abs=1.0)

    def test_winding_direction_does_not_change_the_area(self, utm_raster):
        clockwise = [(10, 10), (30, 10), (30, 30), (10, 30)]
        counter = list(reversed(clockwise))
        assert measure_area(utm_raster, clockwise).value == pytest.approx(
            measure_area(utm_raster, counter).value, abs=0.01)

    def test_two_points_do_not_enclose_an_area(self, utm_raster):
        with pytest.raises(ValueError, match="at least three points"):
            measure_area(utm_raster, [(0, 0), (10, 10)])


class TestPerimeter:
    def test_a_square_perimeter_is_four_sides(self, utm_raster):
        result = measure_perimeter(utm_raster, [(10, 10), (30, 10), (30, 30), (10, 30)])
        assert result.value == pytest.approx(80.0, abs=0.1)
        assert result.kind == "perimeter"

    def test_the_closing_vertex_is_not_reported_as_a_corner(self, utm_raster):
        """The operator drew four corners, so four is what comes back."""
        result = measure_perimeter(utm_raster, [(10, 10), (30, 10), (30, 30), (10, 30)])
        assert len(result.vertices_world) == 4


class TestRefusals:
    def test_a_png_is_refused_because_it_cannot_carry_a_crs(self, tmp_path):
        """The custom engine writes PNGs; measuring one would report pixels as metres."""
        png = tmp_path / "dsm.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        with pytest.raises(NotGeoreferenced, match="cannot carry georeferencing"):
            measure_distance(png, [(0, 0), (10, 10)])

    def test_a_geotiff_without_a_crs_is_refused(self, tmp_path):
        bare = write_raster(tmp_path / "bare.tif", crs=None)
        with pytest.raises(NotGeoreferenced, match="no coordinate reference system"):
            measure_area(bare, [(0, 0), (10, 0), (10, 10)])

    def test_a_geographic_raster_is_refused_for_area(self, tmp_path):
        """A shoelace area in degrees is not square metres, and varies with latitude."""
        degrees = write_raster(tmp_path / "wgs84.tif", crs="EPSG:4326", cell=0.0001)
        with pytest.raises(NotProjected, match="geographic coordinates"):
            measure_area(degrees, [(0, 0), (10, 0), (10, 10)])

    def test_a_missing_file_is_refused_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            measure_distance(tmp_path / "absent.tif", [(0, 0), (1, 1)])


class TestHonestReporting:
    def test_the_projection_caveat_is_carried_with_the_result(self, utm_raster):
        result = measure_distance(utm_raster, [(0, 0), (10, 0)])
        assert any("scale factor" in c for c in result.caveats)

    def test_the_result_says_the_georeferencing_residual_bounds_it(self, utm_raster):
        """The arithmetic is exact; the survey it rests on is not."""
        result = measure_area(utm_raster, [(0, 0), (10, 0), (10, 10)])
        assert any("georeferencing residual" in c for c in result.caveats)

    def test_cell_size_is_reported_as_the_limit_of_what_is_observed(self, utm_raster):
        info = raster_ground_sample_distance(utm_raster)
        assert info["cell_width"] == pytest.approx(CELL)
        assert info["epsg"] == EPSG_UTM
        assert "interpolation, not observation" in info["note"]

    def test_pixel_to_world_round_trips_through_the_transform(self, utm_raster):
        world = pixel_to_world(utm_raster, [(0, 0), (99, 99)])
        assert world[0][0] < world[1][0]   # x increases eastwards
        assert world[0][1] > world[1][1]   # y decreases southwards


class TestRealOrthomosaic:
    def test_a_pipeline_orthomosaic_can_be_measured(self, tmp_path):
        """The point of all this: measure the artifact the pipeline actually produces."""
        from pathlib import Path

        candidates = [
            Path("final_toolkit_outputs/orthomosaic.tif"),
            Path("final_toolkit_outputs/e2e_verify/orthomosaic.tif"),
        ]
        ortho = next((c for c in candidates if c.exists()), None)
        if ortho is None:
            pytest.skip("No pipeline orthomosaic present in this checkout.")

        info = raster_ground_sample_distance(ortho)
        assert info["epsg"] is not None
        assert info["geographic"] is False

        result = measure_distance(ortho, [(10, 10), (60, 10)])
        assert result.value > 0
        assert "metre" in result.unit
