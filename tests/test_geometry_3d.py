"""Planning a flight from an imported surface.

The reason this exists is that an outline cannot describe what it is standing around. A
grid over a building's footprint photographs its roof; a courtyard wall faces inward and
is never seen at all. A surface knows which way each face points, so a capture position
can be placed off the face along its own normal.

The shapes below are chosen so the right answer is arithmetic: a wall four metres square
with a stand-off of eight puts the aircraft exactly eight metres out along the normal,
looking straight back at the wall. The refusals matter equally -- a point cloud has no
normals to stand off along, and an unscaled mesh makes "eight metres" a number of
unknown units.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.model_measurement import NotMetric
from core.provenance import sidecar_path
from mission.geometry_3d import (
    NoUsableSurface,
    UnsupportedSurface,
    plan_from_surface,
    read_surface,
)

# A 10 m x 6 m wall in the x-z plane at y = 0, facing south (normal -y).
WALL_VERTICES = [(0, 0, 0), (10, 0, 0), (10, 0, 6), (0, 0, 6)]
WALL_FACES = [(1, 3, 2), (1, 4, 3)]


def write_obj(path, vertices=WALL_VERTICES, faces=WALL_FACES):
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines += ["f " + " ".join(str(i) for i in face) for face in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_ply(path, vertices=WALL_VERTICES, faces=WALL_FACES):
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_index",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in vertices]
    body += ["3 " + " ".join(str(i - 1) for i in face) for face in faces]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return path


def georeference(path, epsg=32643):
    sidecar_path(path).write_text(json.dumps({
        "artifact": str(path), "engine": "colmap", "crs_epsg": epsg,
    }), encoding="utf-8")
    return path


@pytest.fixture
def wall(tmp_path):
    return georeference(write_obj(tmp_path / "wall.obj"))


class TestReadingSurfaces:
    def test_an_obj_wall_reads_back_as_two_triangles(self, tmp_path):
        vertices, faces = read_surface(write_obj(tmp_path / "wall.obj"))
        assert len(vertices) == 4
        assert len(faces) == 2

    def test_an_ascii_ply_reads_the_same_geometry(self, tmp_path):
        vertices, faces = read_surface(write_ply(tmp_path / "wall.ply"))
        assert len(vertices) == 4
        assert len(faces) == 2

    def test_a_glb_written_by_the_exporter_reads_back(self, tmp_path):
        """Round trip against the toolkit's own writer rather than a hand-made file."""
        formats = pytest.importorskip("core.model_formats")
        geometry = formats.ModelGeometry(
            vertices=np.asarray(WALL_VERTICES, dtype=np.float64),
            faces=np.asarray([[i - 1 for i in face] for face in WALL_FACES], dtype=np.int64),
        )
        path = tmp_path / "wall.glb"
        formats.write_glb(path, geometry)

        vertices, faces = read_surface(path)
        assert len(vertices) == 4
        assert len(faces) == 2

    def test_ifc_is_refused_with_what_would_be_needed(self, tmp_path):
        path = tmp_path / "building.ifc"
        path.write_text("not really ifc", encoding="utf-8")

        with pytest.raises(UnsupportedSurface, match="export the geometry to OBJ"):
            read_surface(path)

    def test_las_is_refused_because_points_have_no_normals(self, tmp_path):
        path = tmp_path / "cloud.las"
        path.write_bytes(b"LASF")

        with pytest.raises(UnsupportedSurface, match="no faces"):
            read_surface(path)

    def test_a_binary_ply_is_refused_rather_than_misread(self, tmp_path):
        path = tmp_path / "binary.ply"
        path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                         b"property float x\nend_header\n\x00\x00\x00\x00")

        with pytest.raises(UnsupportedSurface, match="binary PLY"):
            read_surface(path)


class TestStandOffGeometry:
    def test_the_aircraft_stands_off_along_the_face_normal(self, wall):
        """This winding puts the outward normal at +y, so the aircraft sits north of it."""
        result = plan_from_surface(wall, standoff_m=8.0, min_separation_m=0.0)
        poses = result["poses"]

        assert poses
        for pose in poses:
            assert pose["y_m"] == pytest.approx(8.0, abs=1e-6)
            assert pose["standoff_m"] == 8.0

    def test_the_camera_looks_back_at_the_wall(self, wall):
        """A capture point that stands off correctly and looks away sees the horizon."""
        poses = plan_from_surface(wall, standoff_m=8.0, min_separation_m=0.0)["poses"]

        for pose in poses:
            # Standing north of the wall, the camera must look south: yaw 180.
            assert pose["yaw_deg"] == pytest.approx(180.0, abs=1e-6)
            # A vertical wall is looked at level, not down.
            assert pose["gimbal_pitch_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_reversing_the_winding_reverses_the_stand_off(self, tmp_path):
        """Which side of a wall the aircraft flies is decided by the surface, not a guess."""
        path = georeference(write_obj(tmp_path / "flipped.obj", WALL_VERTICES,
                                      [(1, 2, 3), (1, 3, 4)]))
        poses = plan_from_surface(path, standoff_m=5.0, min_separation_m=0.0)["poses"]

        assert all(pose["y_m"] == pytest.approx(-5.0, abs=1e-6) for pose in poses)
        assert all(pose["yaw_deg"] == pytest.approx(0.0, abs=1e-6) for pose in poses)

    def test_capture_points_are_ordered_bottom_to_top(self, tmp_path):
        tall = [(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4),
                (0, 0, 10), (4, 0, 10)]
        faces = [(1, 3, 2), (1, 4, 3), (4, 6, 5), (4, 3, 6)]
        path = georeference(write_obj(tmp_path / "tall.obj", tall, faces))

        poses = plan_from_surface(path, standoff_m=6.0, min_separation_m=0.0)["poses"]
        heights = [pose["z_m"] for pose in poses]
        assert heights == sorted(heights)


class TestFiltering:
    def test_small_faces_are_dropped_and_the_count_is_reported(self, tmp_path):
        """Reconstruction litter should not each earn a capture point."""
        vertices = list(WALL_VERTICES) + [(20, 0, 0), (20.1, 0, 0), (20.1, 0, 0.1)]
        faces = list(WALL_FACES) + [(5, 6, 7)]
        path = georeference(write_obj(tmp_path / "littered.obj", vertices, faces))

        result = plan_from_surface(path, standoff_m=6.0, min_face_area_m2=1.0,
                                   min_separation_m=0.0)
        assert result["face_count"] == 3
        assert result["faces_considered"] == 2
        assert any("not photographed by this plan" in limit for limit in result["limits"])

    def test_roofs_are_excluded_by_default_and_included_on_request(self, tmp_path):
        """A facade survey wants walls; the same surface can also be flown for roofs."""
        roof = [(0, 0, 10), (10, 0, 10), (10, 8, 10), (0, 8, 10)]
        path = georeference(write_obj(tmp_path / "roof.obj", roof, WALL_FACES))

        with pytest.raises(NoUsableSurface, match="within 60"):
            plan_from_surface(path, standoff_m=6.0)

        included = plan_from_surface(path, standoff_m=6.0, max_vertical_deg=90.0,
                                     min_separation_m=0.0)
        assert included["pose_count"] >= 1

    def test_separation_thins_duplicate_capture_points(self, wall):
        dense = plan_from_surface(wall, standoff_m=8.0, min_separation_m=0.0)
        thinned = plan_from_surface(wall, standoff_m=8.0, min_separation_m=50.0)

        assert thinned["pose_count"] < dense["pose_count"]
        assert thinned["pose_count"] >= 1

    def test_the_covered_area_is_reported_against_the_whole_surface(self, wall):
        result = plan_from_surface(wall, standoff_m=8.0, min_separation_m=0.0)
        assert result["surface_area_m2"] == pytest.approx(60.0)
        assert result["covered_area_m2"] <= result["surface_area_m2"]


class TestRefusals:
    def test_an_unscaled_surface_is_refused(self, tmp_path):
        """Eight metres of structure-from-motion units is eight of nothing."""
        path = write_obj(tmp_path / "unscaled.obj")

        with pytest.raises(NotMetric, match="fly the wrong distance"):
            plan_from_surface(path, standoff_m=8.0)

    def test_a_point_cloud_has_no_normal_to_stand_off_along(self, tmp_path):
        path = georeference(write_obj(tmp_path / "points.obj", WALL_VERTICES, []))

        with pytest.raises(NoUsableSurface, match="invent the orientation"):
            plan_from_surface(path, standoff_m=8.0)

    def test_a_surface_with_nothing_large_enough_is_refused_with_the_thresholds(self, wall):
        with pytest.raises(NoUsableSurface, match="larger than"):
            plan_from_surface(wall, standoff_m=8.0, min_face_area_m2=10_000.0)

    def test_a_negative_stand_off_is_rejected(self, wall):
        with pytest.raises(ValueError, match="Stand-off must be positive"):
            plan_from_surface(wall, standoff_m=-1.0)

    def test_planning_without_the_metric_gate_is_possible_but_explicit(self, tmp_path):
        """Escape hatch for a synthetic surface, taken deliberately rather than by default."""
        path = write_obj(tmp_path / "unscaled.obj")
        result = plan_from_surface(path, standoff_m=8.0, min_separation_m=0.0,
                                   require_metric=False)

        assert result["crs_epsg"] is None
        assert result["pose_count"] >= 1
