"""Measuring in the 3D model, checked against shapes whose answers are arithmetic.

A cube of side two encloses eight cubic metres. A 3-4-5 triangle has an area of six. If
the code disagrees with those, no tolerance argument saves it.

The refusals carry as much weight. Structure from motion recovers shape up to scale, so
an ungeoreferenced model measures in units that look like metres; and an open surface --
a building exterior with no floor -- has no interior, however confidently a signed-volume
sum will report one.
"""

from __future__ import annotations

import json
import math

import pytest

from core.model_measurement import (
    NotClosed,
    NotMetric,
    measure_area,
    measure_height,
    measure_length,
    measure_volume,
    model_scale,
    read_mesh,
)
from core.provenance import sidecar_path

CUBE_VERTICES = [
    (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
    (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
]
# Outward-facing triangles for a closed axis-aligned cube.
CUBE_FACES = [
    (1, 3, 2), (1, 4, 3),        # bottom
    (5, 6, 7), (5, 7, 8),        # top
    (1, 2, 6), (1, 6, 5),        # -y
    (2, 3, 7), (2, 7, 6),        # +x
    (3, 4, 8), (3, 8, 7),        # +y
    (4, 1, 5), (4, 5, 8),        # -x
]


def write_obj(path, vertices=CUBE_VERTICES, faces=CUBE_FACES):
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines += ["f " + " ".join(str(i) for i in face) for face in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def georeference(path, epsg=32643, engine="colmap"):
    """Write the provenance sidecar that records what made this model metric."""
    sidecar_path(path).write_text(json.dumps({
        "artifact": str(path), "engine": engine, "crs_epsg": epsg,
        "sources": [], "recorded_at": "2026-08-15T00:00:00Z",
    }), encoding="utf-8")
    return path


@pytest.fixture
def cube(tmp_path):
    return georeference(write_obj(tmp_path / "cube.obj"))


class TestScaleMustBeRecorded:
    def test_a_model_with_no_sidecar_cannot_be_measured(self, tmp_path):
        model = write_obj(tmp_path / "unscaled.obj")
        with pytest.raises(NotMetric, match="no provenance sidecar"):
            model_scale(model)

    def test_a_model_reconstructed_without_a_crs_is_refused(self, tmp_path):
        """SfM units look like metres and are not. This is the whole gate."""
        model = write_obj(tmp_path / "sfm.obj")
        sidecar_path(model).write_text(json.dumps({
            "artifact": str(model), "engine": "colmap", "crs_epsg": None,
        }), encoding="utf-8")

        with pytest.raises(NotMetric, match="six metres or"):
            model_scale(model)

    def test_an_unreadable_sidecar_is_not_read_as_unscaled(self, tmp_path):
        model = write_obj(tmp_path / "corrupt.obj")
        sidecar_path(model).write_text("{ not json", encoding="utf-8")

        with pytest.raises(NotMetric, match="not the same as the model being unscaled"):
            model_scale(model)

    def test_a_georeferenced_model_reports_where_its_scale_came_from(self, cube):
        scale = model_scale(cube)
        assert scale.epsg == 32643
        assert scale.engine == "colmap"

    def test_every_measurement_refuses_an_unscaled_model(self, tmp_path):
        model = write_obj(tmp_path / "unscaled.obj")
        with pytest.raises(NotMetric):
            measure_length(model, (0, 0, 0), (1, 0, 0))
        with pytest.raises(NotMetric):
            measure_height(model, (0, 0, 3), (0, 0, 0))
        with pytest.raises(NotMetric):
            measure_area(model, [(0, 0, 0), (1, 0, 0), (1, 1, 0)])
        with pytest.raises(NotMetric):
            measure_volume(model)


class TestLength:
    def test_a_3_4_5_triangle_measures_five(self, cube):
        result = measure_length(cube, (0, 0, 0), (3, 4, 0))
        assert result["slope_distance_m"] == pytest.approx(5.0)
        assert result["horizontal_distance_m"] == pytest.approx(5.0)
        assert result["vertical_distance_m"] == pytest.approx(0.0)

    def test_horizontal_and_vertical_parts_are_reported_separately(self, cube):
        """A cable run and the gap it spans are different numbers."""
        result = measure_length(cube, (0, 0, 0), (3, 4, 12))
        assert result["slope_distance_m"] == pytest.approx(13.0)
        assert result["horizontal_distance_m"] == pytest.approx(5.0)
        assert result["vertical_distance_m"] == pytest.approx(12.0)

    def test_the_inclination_of_the_run_is_given(self, cube):
        result = measure_length(cube, (0, 0, 0), (10, 0, 10))
        assert result["inclination_deg"] == pytest.approx(45.0)

    def test_the_crs_the_number_belongs_to_travels_with_it(self, cube):
        assert measure_length(cube, (0, 0, 0), (1, 0, 0))["crs_epsg"] == 32643


class TestHeight:
    def test_height_is_the_vertical_component_only(self, cube):
        result = measure_height(cube, (5, 9, 14.5), (0, 0, 2.5))
        assert result["height_m"] == pytest.approx(12.0)

    def test_picking_the_points_the_other_way_round_gives_the_same_height(self, cube):
        up = measure_height(cube, (0, 0, 10), (0, 0, 4))["height_m"]
        down = measure_height(cube, (0, 0, 4), (0, 0, 10))["height_m"]
        assert up == down == pytest.approx(6.0)

    def test_the_base_assumption_is_stated(self, cube):
        result = measure_height(cube, (0, 0, 10), (0, 0, 0))
        assert any("not on the ground" in limit for limit in result["limits"])


class TestArea:
    def test_a_flat_square_measures_its_side_squared(self, cube):
        result = measure_area(cube, [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)])
        assert result["surface_area_m2"] == pytest.approx(16.0)
        assert result["planimetric_area_m2"] == pytest.approx(16.0)
        assert result["slope_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_a_pitched_surface_has_more_area_than_its_footprint(self, cube):
        """A 45 degree roof covers a plan area of one and needs sqrt(2) of covering."""
        result = measure_area(cube, [(0, 0, 0), (4, 0, 0), (4, 4, 4), (0, 4, 4)])

        assert result["planimetric_area_m2"] == pytest.approx(16.0)
        assert result["surface_area_m2"] == pytest.approx(16.0 * math.sqrt(2))
        assert result["slope_deg"] == pytest.approx(45.0, abs=1e-3)

    def test_a_closed_ring_is_not_counted_twice(self, cube):
        closed = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0), (0, 0, 0)]
        assert measure_area(cube, closed)["surface_area_m2"] == pytest.approx(16.0)

    def test_an_outline_spanning_two_facets_says_so(self, cube):
        bent = [(0, 0, 0), (4, 0, 0), (4, 4, 4), (0, 4, 0)]
        result = measure_area(cube, bent)

        assert result["planar"] is False
        assert any("more than one facet" in limit for limit in result["limits"])

    def test_two_points_do_not_make_an_area(self, cube):
        with pytest.raises(ValueError, match="at least three"):
            measure_area(cube, [(0, 0, 0), (1, 1, 1)])


class TestVolume:
    def test_a_two_metre_cube_encloses_eight_cubic_metres(self, cube):
        result = measure_volume(cube)
        assert result["volume_m3"] == pytest.approx(8.0)
        assert result["surface_area_m2"] == pytest.approx(24.0)
        assert result["closed"] is True

    def test_the_volume_does_not_depend_on_where_the_origin_sits(self, tmp_path):
        """The divergence-theorem sum is origin-independent only for a closed mesh."""
        shifted = [(x + 1000, y - 500, z + 77) for x, y, z in CUBE_VERTICES]
        model = georeference(write_obj(tmp_path / "shifted.obj", shifted))
        assert measure_volume(model)["volume_m3"] == pytest.approx(8.0)

    def test_an_open_surface_is_refused_rather_than_summed(self, tmp_path):
        """A building exterior with no floor encloses nothing."""
        open_box = georeference(write_obj(tmp_path / "open.obj", CUBE_VERTICES,
                                          CUBE_FACES[:-2]))
        with pytest.raises(NotClosed, match="belong to only"):
            measure_volume(open_box)

    def test_a_point_cloud_is_sent_to_the_dsm_path_instead(self, tmp_path):
        cloud = georeference(write_obj(tmp_path / "points.obj", CUBE_VERTICES, []))
        with pytest.raises(NotClosed, match="against a DSM"):
            measure_volume(cloud)


class TestThroughTheApi:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("model measurement", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_a_volume_is_measured_through_the_api(self, api, cube):
        result = api.measure_in_model(str(cube), "volume")

        assert result["ok"] is True
        assert result["measurement"]["volume_m3"] == pytest.approx(8.0)

    def test_a_length_needs_two_points(self, api, cube):
        assert api.measure_in_model(str(cube), "length", [[0, 0, 0]])["ok"] is False

    def test_an_unscaled_model_is_refused_with_the_reason(self, api, tmp_path):
        model = write_obj(tmp_path / "unscaled.obj")
        result = api.measure_in_model(str(model), "volume")

        assert result["ok"] is False
        assert "provenance sidecar" in result["error"]

    def test_an_unknown_kind_is_named_rather_than_guessed(self, api, cube):
        result = api.measure_in_model(str(cube), "circumference")
        assert result["ok"] is False
        assert "length, height, area or volume" in result["error"]


class TestMeshReading:
    def test_a_quad_face_is_triangulated_rather_than_dropped(self, tmp_path):
        model = tmp_path / "quad.obj"
        model.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8")

        vertices, faces = read_mesh(model)
        assert len(vertices) == 4
        assert len(faces) == 2

    def test_texture_and_normal_references_do_not_confuse_the_indices(self, tmp_path):
        model = tmp_path / "textured.obj"
        model.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1/1/1 2/2/1 3/3/1\n", encoding="utf-8")

        _, faces = read_mesh(model)
        assert faces.tolist() == [[0, 1, 2]]

    def test_a_model_with_no_vertices_is_refused(self, tmp_path):
        model = tmp_path / "empty.obj"
        model.write_text("# nothing here\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no vertices"):
            read_mesh(model)
