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


def test_a_facade_is_imaged_across_the_standoff_with_the_camera_that_flies_it():
    """GSD from the 6 m stand-off, not the waypoint altitude; storage from the real camera.

    The Hagenberg plan quoted 0.34 cm (mean altitude 9.4 m) for a wall 33.6 m away that
    the M30T images at 1.22 cm, and sized its storage from a fallback camera.
    """
    points, cams, _a, _n = wall_cloud()
    wall = measure_facade(points, cams, EPSG, home_elevation=500.0)
    plan, _polygon = plan_facade_mission(wall, camera="m30t_wide")
    expected = wall.camera_standoff_m["median"] * 6.4 / 4.4 / 4000 * 100
    assert plan.estimated_gsd_cm == pytest.approx(expected, rel=1e-6)
    storage = plan.estimates()["storage"]
    assert storage["megapixels"] == pytest.approx(12.0)
    assert not any("not in the preset database" in a for a in storage["assumptions"])


def _jpeg_with_exif(path, model, focal_mm, width):
    """A small JPEG carrying the EXIF tags a camera writes: Model, FocalLength, width."""
    from PIL import Image

    image = Image.new("RGB", (64, 48))
    exif = Image.Exif()
    exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    sub[0x920A] = focal_mm
    sub[0xA002] = width
    image.save(path, exif=exif)
    return path


def test_the_camera_is_read_from_the_frames_exif(tmp_path):
    from core.facade_survey import camera_from_exif

    frames = [_jpeg_with_exif(tmp_path / f"f{i}.jpg", "M30T", 4.4, 4000) for i in range(3)]
    key, reason = camera_from_exif(frames)
    # The Mavic 3T wide camera has the same 4.4 mm / 4000 px geometry; the model decides.
    assert key == "m30t_wide", reason


def test_an_unknown_camera_is_not_guessed(tmp_path):
    from core.facade_survey import camera_from_exif

    frames = [_jpeg_with_exif(tmp_path / "f.jpg", "Canon PowerShot S110", 5.2, 4000)]
    key, reason = camera_from_exif(frames)
    assert key is None
    assert "5.2 mm" in reason


def test_the_mission_form_reaches_the_facade_from_model_planner(tmp_path):
    """Implemented-but-unreachable was the defect: the form's type routes to it, and
    with no reconstruction it says what to do rather than planning anything."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from mission.mission_types import SPECIAL_MISSION_TYPES

    assert "facade_from_model" in SPECIAL_MISSION_TYPES
    api = Api(AppSession(store=ProjectStore(tmp_path / "p.db")))
    api.create_project("facade", str(tmp_path / "project"))
    assert "facade_from_model" in api.mission_templates()["special_templates"]
    result = api.plan_mission({"template": "facade_from_model", "image_dir": str(tmp_path)})
    assert not result["ok"]
    assert "No reconstruction" in result["error"]
