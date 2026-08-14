"""Back-projection of 2-D defect masks onto the reconstructed surface.

This is what turns a pixel blob into a defect with a real position and an area in
square metres. Every number it produces looks equally plausible whether the geometry
is right or wrong, so the tests use cameras and surfaces whose answers are known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.defect_projection import (
    CameraPose,
    SurfaceModel,
    _cluster_by_distance,
    _shape_metrics,
    _worst_severity,
    intersect_surface,
)


def nadir_camera(height_m: float = 100.0, focal_px: float = 1000.0) -> CameraPose:
    """A camera looking straight down from `height_m`, centred on the origin.

    World axes are east/north/up. A camera looking down has its optical axis along
    -Z, which is the rotation below.
    """
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    centre = np.array([0.0, 0.0, height_m])
    translation = -rotation @ centre
    intrinsics = np.array([
        [focal_px, 0.0, 320.0],
        [0.0, focal_px, 240.0],
        [0.0, 0.0, 1.0],
    ])
    return CameraPose(
        image="nadir.jpg", rotation=rotation, translation=translation,
        intrinsics=intrinsics, width=640, height=480,
    )


def flat_surface(elevation_m: float = 0.0, extent_m: float = 200.0, pixel_size: float = 0.5):
    cells = int(extent_m / pixel_size)
    return SurfaceModel(
        elevation=np.full((cells, cells), elevation_m, dtype=np.float64),
        west=-extent_m / 2.0,
        north=extent_m / 2.0,
        pixel_size=pixel_size,
        epsg=32617,
    )


class TestCameraGeometry:
    def test_camera_centre_is_recovered(self):
        camera = nadir_camera(height_m=100.0)
        assert camera.center == pytest.approx([0.0, 0.0, 100.0], abs=1e-9)

    def test_principal_ray_points_straight_down(self):
        camera = nadir_camera()
        ray = camera.ray_direction(320.0, 240.0)
        assert ray == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)

    def test_ray_directions_are_unit_length(self):
        camera = nadir_camera()
        for x, y in [(0, 0), (639, 479), (320, 100), (100, 240)]:
            assert np.linalg.norm(camera.ray_direction(x, y)) == pytest.approx(1.0)

    def test_projection_inverts_the_ray(self):
        """A point along a pixel's ray must project back to that same pixel."""
        camera = nadir_camera(height_m=100.0)
        for pixel in [(320.0, 240.0), (400.0, 300.0), (150.0, 120.0)]:
            ray = camera.ray_direction(*pixel)
            world = camera.center + ray * 100.0
            x, y, depth = camera.project(world)
            assert (x, y) == pytest.approx(pixel, abs=1e-6)
            assert depth > 0


class TestSurfaceSampling:
    def test_sample_returns_the_elevation_inside_the_grid(self):
        surface = flat_surface(elevation_m=12.5)
        assert surface.sample(0.0, 0.0) == pytest.approx(12.5)

    def test_sample_outside_the_grid_is_nan(self):
        surface = flat_surface(extent_m=100.0)
        assert math.isnan(surface.sample(10_000.0, 0.0))
        assert math.isnan(surface.sample(0.0, -10_000.0))

    def test_holes_read_as_nan(self):
        surface = flat_surface()
        surface.elevation[10, 10] = np.nan
        east = surface.west + (10 + 0.5) * surface.pixel_size
        north = surface.north - (10 + 0.5) * surface.pixel_size
        assert math.isnan(surface.sample(east, north))


class TestRayIntersection:
    def test_nadir_ray_lands_directly_below_the_camera(self):
        camera = nadir_camera(height_m=100.0)
        surface = flat_surface(elevation_m=0.0)

        hit = intersect_surface(camera.center, camera.ray_direction(320.0, 240.0), surface)

        assert hit is not None
        assert hit == pytest.approx([0.0, 0.0, 0.0], abs=0.05)

    def test_oblique_ray_lands_at_the_expected_offset(self):
        """An off-centre pixel maps to a ground offset of height * (dx / focal)."""
        focal, height = 1000.0, 100.0
        camera = nadir_camera(height_m=height, focal_px=focal)
        surface = flat_surface(elevation_m=0.0)

        offset_px = 100.0
        hit = intersect_surface(
            camera.center, camera.ray_direction(320.0 + offset_px, 240.0), surface
        )

        assert hit is not None
        assert hit[0] == pytest.approx(height * offset_px / focal, abs=0.1)
        assert hit[2] == pytest.approx(0.0, abs=0.05)

    def test_raised_surface_is_hit_higher_up(self):
        camera = nadir_camera(height_m=100.0)
        surface = flat_surface(elevation_m=25.0)

        hit = intersect_surface(camera.center, camera.ray_direction(320.0, 240.0), surface)

        assert hit is not None
        assert hit[2] == pytest.approx(25.0, abs=0.05)

    def test_ray_pointing_away_from_the_surface_misses(self):
        camera = nadir_camera(height_m=100.0)
        surface = flat_surface(elevation_m=0.0)

        hit = intersect_surface(camera.center, np.array([0.0, 0.0, 1.0]), surface)

        assert hit is None, "an upward ray must not report a hit"


class TestShapeMetrics:
    def test_square_area_is_measured_in_square_metres(self):
        """A 2 m x 2 m planar square is 4 m2, whatever its orientation in the grid."""
        points = np.array([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
        ])
        area, length, width = _shape_metrics(points)
        assert area == pytest.approx(4.0, rel=0.05)
        assert length == pytest.approx(2.0, rel=0.1)
        assert width == pytest.approx(2.0, rel=0.1)

    def test_elongated_shape_reports_length_above_width(self):
        points = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.5, 0.0], [0.0, 0.5, 0.0],
        ])
        _, length, width = _shape_metrics(points)
        assert length > width * 5, "a crack must not be reported as square"

    def test_degenerate_input_does_not_explode(self):
        area, length, width = _shape_metrics(np.array([[1.0, 1.0, 1.0]]))
        assert area >= 0.0 and length >= 0.0 and width >= 0.0


class TestClustering:
    def test_nearby_points_merge_into_one_defect(self):
        """One defect seen from several views must not be reported several times."""
        points = np.array([
            [0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.2, 0.2, 0.0],
        ])
        clusters = _cluster_by_distance(points, radius=1.0)
        assert len(clusters) == 1

    def test_distant_points_stay_separate(self):
        points = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        clusters = _cluster_by_distance(points, radius=1.0)
        assert len(clusters) == 2

    def test_every_point_is_assigned_exactly_once(self):
        rng = np.random.default_rng(5)
        points = rng.normal(size=(40, 3)) * 10.0
        clusters = _cluster_by_distance(points, radius=3.0)
        assigned = sorted(index for cluster in clusters for index in cluster)
        assert assigned == list(range(len(points)))

    def test_chained_points_link_into_one_cluster(self):
        """Single-link clustering must follow a chain, not require mutual proximity."""
        points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
        clusters = _cluster_by_distance(points, radius=1.5)
        assert len(clusters) == 1


class TestSeverity:
    def test_worst_severity_wins(self):
        assert _worst_severity(["low", "high", "medium"]) == "high"

    def test_unknown_values_do_not_crash(self):
        assert _worst_severity(["nonsense"]) is not None

    def test_empty_input_is_handled(self):
        assert _worst_severity([]) is not None
