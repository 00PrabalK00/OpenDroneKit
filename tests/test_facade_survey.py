"""Measuring a facade from a georeferenced cloud and planning the next flight on it.

The geometry checks use a wall whose answer is known exactly. The real-imagery check
(the FH Hagenberg facade, DJI M30T wide frames) lives in tests/test_real_recon_datasets.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.facade_survey import (
    FacadeError,
    measure_facade,
    plan_facade_mission,
    waypoint_offsets_from_wall,
)

EPSG = 32633
ORIGIN = np.array([450000.0, 5357000.0])


def wall_cloud(noise=0.02, seed=3):
    """A 30 m x 12 m wall running 20 degrees north of east, cameras 6 m south of it."""
    rng = np.random.default_rng(seed)
    angle = np.radians(20)
    along = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([np.sin(angle), -np.cos(angle)])     # towards the cameras
    s = rng.uniform(0, 30, 3000)
    z = rng.uniform(500, 512, 3000)
    xy = ORIGIN + np.outer(s, along) + np.outer(rng.normal(0, noise, 3000), normal)
    clutter = ORIGIN + rng.uniform(-5, 35, (300, 2))
    points = np.vstack([np.column_stack([xy, z]),
                        np.column_stack([clutter, rng.uniform(500, 503, 300)])])
    cams = np.column_stack([ORIGIN + np.outer(np.linspace(2, 28, 20), along) + 6.0 * normal,
                            np.full(20, 505.0)])
    return points, cams, along, normal


def test_the_wall_line_height_and_standoff_are_recovered():
    points, cams, along, normal = wall_cloud()
    wall = measure_facade(points, cams, EPSG, home_elevation=500.0)
    assert np.asarray(wall.normal_xy) == pytest.approx(normal, abs=0.01)
    assert wall.length_m == pytest.approx(30 * 0.96, abs=0.5)     # 2nd..98th percentile
    assert wall.base_z == pytest.approx(0.24, abs=0.2)
    assert wall.top_z == pytest.approx(11.76, abs=0.2)
    assert wall.height_reference == "home"
    assert wall.camera_standoff_m["median"] == pytest.approx(6.0, abs=0.05)
    assert wall.flatness_rms_m == pytest.approx(0.02, abs=0.01)
    # ~2 points per 0.5 m cell on average: Poisson occupancy 1 - e^-2.2 = 0.89.
    assert 0.8 < wall.coverage_fraction < 0.95


def test_the_planned_mission_stands_off_the_measured_wall_on_the_camera_side():
    points, cams, _along, _normal = wall_cloud()
    wall = measure_facade(points, cams, EPSG, home_elevation=500.0)
    plan, polygon = plan_facade_mission(wall)
    assert plan.template == "facade" and len(plan.waypoints) > 4
    offsets = waypoint_offsets_from_wall(plan, wall)
    assert np.all(offsets > 0), "a waypoint lies on or behind the wall"
    assert np.median(offsets) == pytest.approx(6.0, abs=0.5)


def test_a_wall_in_an_unknown_height_datum_is_not_planned_on():
    points, cams, _a, _n = wall_cloud()
    wall = measure_facade(points, cams, EPSG)          # no home elevation
    assert wall.height_reference == "anchor_datum"
    with pytest.raises(FacadeError, match="take-off"):
        plan_facade_mission(wall)


def test_too_few_points_is_refused():
    with pytest.raises(FacadeError):
        measure_facade(np.zeros((10, 3)), np.zeros((3, 3)), EPSG)
