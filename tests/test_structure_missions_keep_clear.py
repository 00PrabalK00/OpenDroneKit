"""A structure mission must stand off the structure, not just face it.

Each test plans through the real Api, exports the QGC .plan, and measures the exported
NAV points and the legs between them against geometry computed here from the inputs:
the drawn footprint, and for a turbine its hub, blades and nacelle. Nothing is read
back from the planner's own idea of where the structure is.

What these pin down:
- An L-shaped building: multi-facade planned one wall's passes INSIDE the building
  (outward was decided by the vertex mean, which is the inside corner), and the box and
  closed-loop circuits were projected onto a fence that touched the inside corner, so
  they were flown 0-1 m from the walls.
- A box asked for 15 m flew its long walls at 8 m (vertices were scaled away from the
  centre instead of the walls being offset).
- Multi-facade rows were spaced by the nadir footprint at survey altitude: 20 m between
  rows that each covered 7 m of wall.
- A wind turbine path flew from one blade's front-face tip straight through the blade
  to its back-face root, and two nacelle points sat in the rotor plane 6 m from a blade.
- Tower rings flew 4 m from the tower and aimed every ring at mid-height, so the top
  ring looked 79 deg down and the section beside the aircraft was never in frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

LAT0, LON0 = 52.2051, 0.1180
M_PER_DEG = 6371008.8 * math.pi / 180.0
NAV = (16, 19, 22)
# The gimbal is set with DO_GIMBAL_MANAGER_PITCHYAW (1000), pitch in param1.
CONDITION_YAW, DO_GIMBAL_PITCHYAW = 115, 1000


def lonlat(x: float, y: float) -> list[float]:
    return [LON0 + x / (M_PER_DEG * math.cos(math.radians(LAT0))), LAT0 + y / M_PER_DEG]


def local(lon: float, lat: float, origin: tuple[float, float]) -> np.ndarray:
    return np.array([(lon - origin[0]) * math.cos(math.radians(origin[1])) * M_PER_DEG,
                     (lat - origin[1]) * M_PER_DEG])


L_SHAPE = [(-30, -20), (30, -20), (30, 0), (0, 0), (0, 20), (-30, 20)]


def rotated_rectangle(width: float, depth: float, degrees: float):
    a = math.radians(degrees)
    return [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))
            for x, y in [(-width / 2, -depth / 2), (width / 2, -depth / 2),
                         (width / 2, depth / 2), (-width / 2, depth / 2)]]


def make_api(tmp_path: Path, footprint_xy) -> tuple[Api, tuple[float, float]]:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("clear", root_dir=str(tmp_path / "project"))
    api = Api(session)
    aoi = [lonlat(x, y) for x, y in footprint_xy]
    assert api.set_aoi(aoi)["ok"]
    origin = (sum(p[0] for p in aoi) / len(aoi), sum(p[1] for p in aoi) / len(aoi))
    return api, origin


def plan_export(api: Api, tmp_path: Path, options: dict) -> tuple[dict, list[dict]]:
    planned = api.plan_mission(options)
    assert planned["ok"], planned.get("error")
    exported = api.export_mission(["qgc_plan"], str(tmp_path / "out"))
    assert exported["ok"], exported.get("error")
    items = json.loads(Path(exported["written"]["qgc_plan"]).read_text())["mission"]["items"]
    return planned, items


def nav_points(items, origin) -> list[np.ndarray]:
    return [np.array([*local(i["params"][5], i["params"][4], origin), i["params"][6]])
            for i in items if i["command"] in NAV and i["params"][4]]


def footprint_in_frame(footprint_xy, origin):
    """The drawn footprint re-expressed about the export's origin (the vertex mean)."""
    ring = [local(*lonlat(x, y), origin) for x, y in footprint_xy]
    return ring


def point_in_polygon(p, ring) -> bool:
    inside = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
    return inside


def distance_to_segment(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def distance_to_walls(p, ring) -> float:
    return min(distance_to_segment(p[:2], a, b) for a, b in zip(ring, ring[1:] + ring[:1]))


def legs(points, step: float = 0.5):
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(n + 1):
            yield a + (b - a) * (k / n)


class TestConcaveBuilding:
    @pytest.mark.parametrize("template, standoff", [
        ("multi_facade", 8.0), ("box_inspection", 15.0), ("closed_loop", 8.0)])
    def test_no_point_or_leg_comes_inside_the_stand_off(self, tmp_path, template, standoff):
        api, origin = make_api(tmp_path, L_SHAPE)
        _, items = plan_export(api, tmp_path, {"template": template, "altitude_m": 50})
        ring = footprint_in_frame(L_SHAPE, origin)
        points = nav_points(items, origin)[1:]  # after takeoff
        assert points
        assert not [p for p in points if point_in_polygon(p[:2], ring)]
        closest = min(distance_to_walls(p, ring) for p in points)
        assert closest >= 0.95 * standoff, f"{template}: a point {closest:.1f} m from a wall"
        closest_leg = min(distance_to_walls(q, ring) for q in legs(points))
        # Legs between faces cut the outside corners; they stay well clear of the walls.
        assert closest_leg >= 0.5 * standoff, f"{template}: a leg {closest_leg:.1f} m from a wall"

    def test_multi_facade_covers_every_wall_from_outside(self, tmp_path):
        api, origin = make_api(tmp_path, L_SHAPE)
        _, items = plan_export(api, tmp_path, {"template": "multi_facade", "altitude_m": 50})
        ring = footprint_in_frame(L_SHAPE, origin)
        walls = list(zip(ring, ring[1:] + ring[:1]))
        per_wall = [0] * len(walls)
        for p in nav_points(items, origin)[1:]:
            d = [distance_to_segment(p[:2], a, b) for a, b in walls]
            per_wall[int(np.argmin(d))] += 1
        assert all(count > 0 for count in per_wall), per_wall

    def test_multi_facade_rows_overlap_on_the_wall(self, tmp_path):
        """Rows closer than the frame height at the stand-off, or the wall has gaps."""
        api, origin = make_api(tmp_path, L_SHAPE)
        _, items = plan_export(api, tmp_path, {"template": "multi_facade", "altitude_m": 50})
        altitudes = sorted({round(p[2], 2) for p in nav_points(items, origin)[1:]})
        gaps = np.diff(altitudes)
        # Default camera (mavic2pro, 13.2 x 8.8 mm, 10.26 mm) at 8 m: frame height 6.9 m.
        assert len(altitudes) > 3 and gaps.max() < 6.9 * 0.5


class TestBoxHoldsItsStandOff:
    def test_every_wall_is_flown_at_the_requested_distance(self, tmp_path):
        footprint = rotated_rectangle(60.0, 30.0, 35.0)
        api, origin = make_api(tmp_path, footprint)
        _, items = plan_export(api, tmp_path, {"template": "box_inspection", "altitude_m": 50})
        ring = footprint_in_frame(footprint, origin)
        distances = [distance_to_walls(p, ring) for p in nav_points(items, origin)[1:]]
        assert min(distances) >= 15.0 - 0.05
        # Beside a wall (its perpendicular foot on the wall, not past a corner) the
        # circuit is at exactly 15 m. Scaling vertices from the centre put the long
        # walls at 8 m.
        beside = []
        for p in nav_points(items, origin)[1:]:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                ab = b - a
                t = float((p[:2] - a) @ ab / (ab @ ab))
                if 0.02 < t < 0.98 and distance_to_segment(p[:2], a, b) < 20.0:
                    beside.append(distance_to_segment(p[:2], a, b))
        assert len(beside) > 100
        assert max(abs(d - 15.0) for d in beside) < 0.05


class TestTheFenceClearsTheStructure:
    def test_the_exported_inclusion_fence_is_a_true_offset(self, tmp_path):
        api, origin = make_api(tmp_path, L_SHAPE)
        planned, _ = plan_export(api, tmp_path, {"template": "closed_loop", "altitude_m": 50})
        exported = api.export_mission(["qgc_plan"], str(tmp_path / "fence"))
        plan = json.loads(Path(exported["written"]["qgc_plan"]).read_text())
        polygons = plan["geoFence"]["polygons"]
        inclusion = [p for p in polygons if p.get("inclusion")]
        assert inclusion
        fence = [local(lon_lat[1], lon_lat[0], origin) for lon_lat in inclusion[0]["polygon"]]
        ring = footprint_in_frame(L_SHAPE, origin)
        # The inside corner of the L used to be a fence vertex.
        assert min(distance_to_walls(v, ring) for v in fence) >= 8.0 + 10.0 - 0.1
        assert planned["summary"].get("adjustments", {}).get("geofence_projections", 0) >= 0


class TestTower:
    def test_rings_stand_off_and_look_level(self, tmp_path):
        api, origin = make_api(tmp_path, rotated_rectangle(6.0, 6.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "tower_mapping", "altitude_m": 50,
            "tower_bottom_altitude_m": 20, "tower_top_altitude_m": 45})
        points = nav_points(items, origin)[1:]
        radii = [float(np.linalg.norm(p[:2])) for p in points]
        # Default tower radius 2 m plus the 8 m structure stand-off.
        assert min(radii) >= 10.0 - 0.05
        assert 20.0 - 0.05 <= min(p[2] for p in points) and max(p[2] for p in points) <= 45.0 + 0.05
        pitches = {round(i["params"][0], 3) for i in items if i["command"] == DO_GIMBAL_PITCHYAW}
        assert pitches == {0.0}
        assert not [w for w in planned["warnings"] if "Tower height not given" in w]

    def test_an_unstated_tower_height_is_said(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(6.0, 6.0, 0.0))
        planned, _ = plan_export(api, tmp_path, {"template": "tower_mapping", "altitude_m": 50})
        assert [w for w in planned["warnings"] if "Tower height not given" in w]

    def test_a_base_below_the_altitude_band_is_said_not_silently_raised(self, tmp_path):
        api, origin = make_api(tmp_path, rotated_rectangle(6.0, 6.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "tower_mapping", "altitude_m": 50,
            "tower_bottom_altitude_m": 10, "tower_top_altitude_m": 60})
        lowest = min(p[2] for p in nav_points(items, origin)[1:])
        assert lowest > 10.5  # the default 20 m floor
        assert [w for w in planned["warnings"]
                if f"flown at {lowest:.1f} m, not the 10.0 m given" in w]
        # Lowering the floor flies the base that was asked for, and says nothing.
        planned, items = plan_export(api, tmp_path, {
            "template": "tower_mapping", "altitude_m": 50, "min_altitude_m": 5,
            "tower_bottom_altitude_m": 10, "tower_top_altitude_m": 60})
        assert abs(min(p[2] for p in nav_points(items, origin)[1:]) - 10.0) < 0.05
        assert not [w for w in planned["warnings"] if "not the 10.0 m given" in w]


def turbine_geometry(facing_deg, hub_height, blade_length, nacelle_length, blade_angles):
    front = np.array([math.sin(math.radians(facing_deg)), math.cos(math.radians(facing_deg))])
    across = np.array([front[1], -front[0]])
    hub = np.array([0.0, 0.0, hub_height])
    parts = {"tower": (np.array([0.0, 0.0, 0.0]), hub),
             "nacelle": (hub, hub - np.array([*(front * nacelle_length), 0.0]))}
    for k, angle in enumerate(blade_angles):
        r = math.radians(angle)
        parts[f"blade{k}"] = (hub, hub + np.array([*(across * math.cos(r) * blade_length),
                                                    math.sin(r) * blade_length]))
    return front, parts


class TestWindTurbine:
    @pytest.mark.parametrize("facing, angles", [
        (225.0, None), (30.0, "0, 120, 240"), (100.0, [60, 180, 300])])
    def test_the_path_never_comes_within_the_stand_off(self, tmp_path, facing, angles):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        options = {"template": "wind_turbine", "altitude_m": 90, "turbine_rotor_facing_deg": facing,
                   "turbine_hub_height_m": 100, "turbine_blade_length_m": 50,
                   "turbine_nacelle_length_m": 12}
        if angles is not None:
            options["turbine_blade_angles_deg"] = angles
        planned, items = plan_export(api, tmp_path, options)
        blade_angles = [90.0, 210.0, 330.0] if angles is None else (
            [float(v) for v in angles.split(",")] if isinstance(angles, str) else [float(v) for v in angles])
        front, parts = turbine_geometry(facing, 100.0, 50.0, 12.0, blade_angles)
        points = nav_points(items, origin)[1:]
        standoff = 12.0
        for q in legs(points):
            for name, (a, b) in parts.items():
                assert distance_to_segment(q, a, b) >= standoff * 0.97, (name, q.round(1))
        # The rotor plane is crossed once, not once per blade.
        sides = [np.sign(float(p[:2] @ front)) for p in points]
        assert sum(1 for s, t in zip(sides, sides[1:]) if s != t) == 1

    def test_the_small_drawn_footprint_does_not_squash_the_blades(self, tmp_path):
        """The fence holds the rotor; a fence round the tower base pulled blade points in."""
        api, origin = make_api(tmp_path, rotated_rectangle(10.0, 10.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 90, "turbine_rotor_facing_deg": 0.0})
        assert planned["summary"]["adjustments"]["geofence_projections"] == 0
        front, parts = turbine_geometry(0.0, 90.0, 55.0, 12.0, [90.0, 210.0, 330.0])
        for p in nav_points(items, origin)[1:]:
            for name, (a, b) in parts.items():
                assert distance_to_segment(p, a, b) >= 12.0 * 0.97, name

    def test_assumed_turbine_facts_are_said(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned, _ = plan_export(api, tmp_path, {"template": "wind_turbine", "altitude_m": 90})
        text = " ".join(planned["warnings"])
        assert "Rotor direction not given" in text
        assert "blade length 55 m" in text
        given, _ = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 90, "turbine_rotor_facing_deg": 10,
            "turbine_hub_height_m": 95, "turbine_blade_length_m": 50,
            "turbine_nacelle_length_m": 10, "turbine_blade_angles_deg": "90, 210, 330"})
        text = " ".join(given["warnings"])
        assert "Rotor direction not given" not in text
        assert "Turbine dimensions assumed" not in text


class TestFacadeSaysWhichWall:
    def test_the_single_wall_and_height_assumption_are_stated(self, tmp_path):
        api, _ = make_api(tmp_path, L_SHAPE)
        planned, _ = plan_export(api, tmp_path, {"template": "facade", "altitude_m": 50})
        text = " ".join(planned["warnings"])
        assert "photographs one wall: the first edge drawn (60 m, running 90 deg)" in text
        assert "Wall height not given" in text

    def test_given_wall_heights_are_flown(self, tmp_path):
        api, origin = make_api(tmp_path, L_SHAPE)
        planned, items = plan_export(api, tmp_path, {
            "template": "facade", "altitude_m": 50,
            "facade_bottom_altitude_m": 6, "facade_top_altitude_m": 24})
        altitudes = [p[2] for p in nav_points(items, origin)[1:]]
        assert min(altitudes) >= 6 - 0.05 and max(altitudes) <= 24 + 0.05
        assert "Wall height not given" not in " ".join(planned["warnings"])
