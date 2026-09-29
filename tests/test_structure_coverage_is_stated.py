"""What a structure mission cannot photograph as asked must be flown differently or said.

Measured on the exported QGC .plan against geometry built here from the inputs:

- Multi-facade on an L-shaped building dropped the points of each inner wall that fell
  within the stand-off of the other wall, and put nothing in their place: 5 m of the
  inner wall at 8 m (7.5 m at 12 m) was in no frame, and nothing said so.
- A wind turbine's single rotor-plane crossing was floored at the minimum altitude.
  With a blade pointing down, that pulled the crossing leg to 7.6 m from its tip.
  Blade stations above the maximum altitude were flattened onto it with no word.
- A dome whose lower rings fall under the minimum altitude had those poses pushed up
  onto one flat layer, their gimbal still set for the height they were planned at, so
  the camera looked past the dome; the only sign was a count in the summary table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from test_structure_missions_keep_clear import (
    CONDITION_YAW, L_SHAPE, NAV, distance_to_segment, footprint_in_frame, legs,
    local, make_api, nav_points, plan_export, rotated_rectangle, turbine_geometry,
)

DO_MOUNT_CONTROL = 205
# Default camera (mavic2pro): 13.2 mm sensor width, 10.26 mm focal length.
HALF_FOV = math.degrees(math.atan(6.6 / 10.26))


def compass(a, b) -> float:
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def off(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def shots(items, origin):
    """(position xyz, compass heading, gimbal pitch) of each NAV item a heading precedes."""
    out, heading, pitch = [], None, None
    for item in items:
        if item["command"] == DO_MOUNT_CONTROL:
            pitch = item["params"][0]
        elif item["command"] == CONDITION_YAW:
            heading = item["params"][0]
        elif item["command"] in NAV and item["params"][4] and heading is not None:
            p = item["params"]
            out.append((np.array([*local(p[5], p[4], origin), p[6]]), heading, pitch))
            heading = None
    return out


def _crosses(p, q, a, b) -> bool:
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    return (cross(a, b, p) * cross(a, b, q) < -1e-9) and (cross(p, q, a) * cross(p, q, b) < -1e-9)


class TestMultiFacadeInsideCorners:
    @pytest.mark.parametrize("standoff", [8.0, 12.0])
    def test_every_metre_of_every_wall_is_in_a_frame(self, tmp_path, standoff):
        api, origin = make_api(tmp_path, L_SHAPE)
        planned, items = plan_export(api, tmp_path, {
            "template": "multi_facade", "altitude_m": 50, "standoff_m": standoff})
        ring = footprint_in_frame(L_SHAPE, origin)
        walls = list(zip(ring, ring[1:] + ring[:1]))
        taken = shots(items, origin)[1:]
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in walls) / 2.0
        unseen = {}
        for i, (a, b) in enumerate(walls):
            length = float(np.linalg.norm(b - a))
            normal = np.array([(b - a)[1], -(b - a)[0]]) / length
            if area < 0:
                normal = -normal
            for k in range(int(length * 2) + 1):
                on_wall = a + (b - a) * (k / (length * 2))
                target = on_wall + normal * 0.05
                seen = False
                for position, heading, _ in taken:
                    p = position[:2]
                    if (p - on_wall) @ normal <= 0:
                        continue   # behind the wall
                    if off(heading, compass(p, target)) > HALF_FOV:
                        continue   # outside the frame
                    if any(_crosses(p, target, c, d) for j, (c, d) in enumerate(walls) if j != i):
                        continue   # the building is in the way
                    seen = True
                    break
                if not seen:
                    unseen.setdefault(i, []).append(round(k / 2, 1))
        assert not unseen, f"wall metres in no frame: {unseen}"

    def test_turned_shots_still_stand_off(self, tmp_path):
        api, origin = make_api(tmp_path, L_SHAPE)
        _, items = plan_export(api, tmp_path, {"template": "multi_facade", "altitude_m": 50})
        ring = footprint_in_frame(L_SHAPE, origin)
        walls = list(zip(ring, ring[1:] + ring[:1]))
        taken = shots(items, origin)[1:]
        nearest = []
        for position, heading, _ in taken:
            d = [distance_to_segment(position[:2], a, b) for a, b in walls]
            assert min(d) >= 0.95 * 8.0
            nearest.append(off(heading, compass(position[:2], _foot(position[:2], walls))))
        # Most shots are square to their wall; the inside corners add turned ones.
        turned = [e for e in nearest if e > 1.0]
        assert turned and len(turned) < 0.2 * len(nearest)


def _foot(p, walls):
    best = None
    for a, b in walls:
        ab = b - a
        t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        q = a + t * ab
        if best is None or np.linalg.norm(q - p) < best[0]:
            best = (float(np.linalg.norm(q - p)), q)
    return best[1]


class TestTurbineCrossing:
    @pytest.mark.parametrize("facing, hub, blade, angles, standoff", [
        (300.0, 70.0, 45.0, "45, 165, 285", 15.0),
        (0.0, 45.0, 45.0, None, 12.0),
    ])
    def test_a_downward_last_blade_does_not_pull_the_crossing_in(
            self, tmp_path, facing, hub, blade, angles, standoff):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        options = {"template": "wind_turbine", "altitude_m": 55, "min_altitude_m": 20,
                   "turbine_rotor_facing_deg": facing, "turbine_hub_height_m": hub,
                   "turbine_blade_length_m": blade, "turbine_nacelle_length_m": 12,
                   "standoff_m": standoff}
        if angles:
            options["turbine_blade_angles_deg"] = angles
        planned, items = plan_export(api, tmp_path, options)
        blade_angles = [float(v) for v in angles.split(",")] if angles else [90.0, 210.0, 330.0]
        front, parts = turbine_geometry(facing, hub, blade, 12.0, blade_angles)
        points = nav_points(items, origin)[1:]
        worst = min(distance_to_segment(q, a, b) for q in legs(points) for a, b in parts.values())
        assert worst >= 0.97 * standoff, f"a leg passes {worst:.1f} m from the turbine"
        assert min(p[2] for p in points) >= 20.0 - 0.05

    def test_blade_stations_above_the_ceiling_are_said(self, tmp_path):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 90, "turbine_rotor_facing_deg": 0,
            "turbine_hub_height_m": 100, "turbine_blade_length_m": 60,
            "turbine_nacelle_length_m": 12, "turbine_blade_angles_deg": "90, 210, 330"})
        text = " ".join(planned["warnings"])
        assert "Blade 1: the outer" in text and "above the 120 m maximum altitude" in text
        points = nav_points(items, origin)[1:]
        # Nothing is flattened onto the ceiling: no stack of points at exactly 120 m.
        assert sum(1 for p in points if abs(p[2] - 120.0) < 0.01) <= 2
        assert planned["summary"]["adjustments"]["altitude_clamps"] == 0


class TestDomeUnderTheFloor:
    @pytest.mark.parametrize("footprint", [rotated_rectangle(60.0, 30.0, 35.0),
                                           rotated_rectangle(20.0, 20.0, 0.0)])
    def test_low_bands_are_looked_at_from_the_floor_and_said(self, tmp_path, footprint):
        api, origin = make_api(tmp_path, footprint)
        # As the cockpit sends a blank form: empty strings for the dome inputs.
        planned, items = plan_export(api, tmp_path, {
            "template": "dome_inspection", "altitude_m": 55, "min_altitude_m": 20,
            "max_altitude_m": 120, "dome_radius_m": "", "dome_base_altitude_m": ""})
        assert planned["summary"]["adjustments"]["altitude_clamps"] == 0
        ring = footprint_in_frame(footprint, origin)
        xs = np.array([p[0] for p in ring])
        ys = np.array([p[1] for p in ring])
        radius = math.sqrt(0.5 * abs(xs @ np.roll(ys, -1) - ys @ np.roll(xs, -1)) / math.pi)
        centre = np.array([xs.mean(), ys.mean()])
        taken = shots(items, origin)[1:]
        assert taken
        for position, heading, pitch in taken:
            rho = float(np.linalg.norm(position[:2] - centre))
            z = float(position[2])
            assert z >= 20.0 - 0.05
            assert math.hypot(rho, z) - radius >= 8.0 - 0.05   # clear of the dome
            if rho < 0.5:
                continue
            assert off(heading, compass(position[:2], centre)) < 1.0
            # The camera axis, in the vertical plane through the centre, meets the dome
            # above the ground.
            d = np.array([-math.cos(math.radians(pitch)), math.sin(math.radians(pitch))])
            o = np.array([rho, z])
            b, c = o @ d, o @ o - radius * radius
            assert b * b - c >= 0.0, "the camera looks past the dome"
            hit = o + d * (-b - math.sqrt(b * b - c))
            assert hit[1] >= -0.05, "the camera looks at the ground, not the dome"
        text = " ".join(planned["warnings"])
        assert "under the 20 m minimum altitude" in text and "Lower min_altitude_m" in text

    def test_a_dome_above_the_floor_says_nothing_extra(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(60.0, 30.0, 35.0))
        planned, _ = plan_export(api, tmp_path, {
            "template": "dome_inspection", "altitude_m": 55, "dome_radius_m": 20,
            "dome_base_altitude_m": 25})
        assert "minimum altitude" not in " ".join(planned["warnings"])


class TestTypedInputsAreRefusedByName:
    def test_a_non_numeric_pylon_height_names_the_element(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned = api.plan_mission({"template": "pylon_inspection", "pylon_elements": "crossarm@abc, body@20"})
        assert not planned["ok"]
        assert "crossarm" in planned["error"] and "ValueError" not in planned["error"]

    def test_a_non_numeric_panel_position_says_what_is_wanted(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        payload = next(p["key"] for p in api.list_payloads()["payloads"] if p.get("kind") == "multispectral")
        planned = api.plan_mission({"template": "multispectral", "payload_key": payload,
                                    "calibration_panel_lonlat": "no"})
        assert not planned["ok"]
        assert "lon, lat" in planned["error"] and "ValueError" not in planned["error"]


class TestTurbineWholePath:
    """Every leg, not only the rotor crossing: a blade parked pointing down on a low hub
    left the back faces at the bottom of the rotor, and the route to the nacelle sides
    climbed straight up from there through the nacelle (0.1 m from it)."""

    @pytest.mark.parametrize("facing, hub, blade, angles, standoff", [
        (90.0, 40.0, 38.0, "270, 30, 150", 10.0),
        (0.0, 35.0, 33.0, "270, 30, 150", 12.0),
        (341.17521928452203, 39.9, 24.0, "75.3, 195.3, 315.3", 20.0),
        (152.82690809130503, 64.7, 23.0, "10.9, 130.9, 250.9", 20.0),
    ])
    def test_no_leg_comes_within_the_stand_off(self, tmp_path, facing, hub, blade, angles, standoff):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 55, "min_altitude_m": 20,
            "turbine_rotor_facing_deg": facing, "turbine_hub_height_m": hub,
            "turbine_blade_length_m": blade, "turbine_nacelle_length_m": 12,
            "turbine_blade_angles_deg": angles, "standoff_m": standoff})
        _, parts = turbine_geometry(facing, hub, blade, 12.0, [float(v) for v in angles.split(",")])
        points = nav_points(items, origin)[1:]
        worst = min(distance_to_segment(q, a, b) for q in legs(points) for a, b in parts.values())
        assert worst >= 0.97 * standoff, f"a leg passes {worst:.1f} m from the turbine"
        assert planned["summary"]["adjustments"]["altitude_clamps"] == 0

    @pytest.mark.parametrize("facing, hub, blade, angles, standoff, nacelle, floor, ceiling", [
        # A ceiling under hub + stand-off: the "over the nacelle" detour was chosen at
        # hub + stand-off and then clamped down to the ceiling, straight over the nacelle.
        (0.0, 105.0, 40.0, "270, 30, 150", 20.0, 12.0, 20.0, 120.0),
        (45.0, 110.0, 45.0, "270, 30, 150", 12.0, 12.0, 20.0, 120.0),
        (7.3, 94.0, 14.5, "38.9, 158.9, 278.9", 10.0, 12.3, 6.9, 101.9),
        (95.5, 33.2, 22.6, "161.9, 281.9, 41.9", 15.0, 15.4, 25.1, 39.9),
        (30.3, 40.8, 33.5, "183.8, 303.8, 63.8", 25.0, 17.8, 34.7, 46.1),
    ])
    def test_the_altitude_band_does_not_pull_a_detour_onto_the_nacelle(
            self, tmp_path, facing, hub, blade, angles, standoff, nacelle, floor, ceiling):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 55,
            "min_altitude_m": floor, "max_altitude_m": ceiling,
            "turbine_rotor_facing_deg": facing, "turbine_hub_height_m": hub,
            "turbine_blade_length_m": blade, "turbine_nacelle_length_m": nacelle,
            "turbine_blade_angles_deg": angles, "standoff_m": standoff})
        _, parts = turbine_geometry(facing, hub, blade, nacelle, [float(v) for v in angles.split(",")])
        points = nav_points(items, origin)[1:]
        assert max(p[2] for p in points) <= ceiling + 0.05
        worst = min(distance_to_segment(q, a, b) for q in legs(points) for a, b in parts.values())
        assert worst >= 0.97 * standoff, f"a leg passes {worst:.1f} m from the turbine"
        assert "from the nacelle, inside the" not in " ".join(planned["warnings"])

    def test_a_hub_above_the_ceiling_is_said_not_clamped(self, tmp_path):
        api, origin = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned, items = plan_export(api, tmp_path, {
            "template": "wind_turbine", "altitude_m": 55, "max_altitude_m": 90,
            "turbine_rotor_facing_deg": 0, "turbine_hub_height_m": 110,
            "turbine_blade_length_m": 50, "turbine_nacelle_length_m": 12, "standoff_m": 12})
        assert planned["summary"]["adjustments"]["altitude_clamps"] == 0
        assert "nacelle shot(s) at hub height (110 m) are outside" in " ".join(planned["warnings"])
        _, parts = turbine_geometry(0.0, 110.0, 50.0, 12.0, [90.0, 210.0, 330.0])
        points = nav_points(items, origin)[1:]
        assert max(p[2] for p in points) <= 90.0 + 0.05
        assert min(distance_to_segment(q, a, b) for q in legs(points) for a, b in parts.values()) >= 0.97 * 12

    def test_a_turbine_wholly_under_the_floor_is_refused(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned = api.plan_mission({
            "template": "wind_turbine", "altitude_m": 55, "min_altitude_m": 70,
            "turbine_rotor_facing_deg": 0, "turbine_hub_height_m": 30,
            "turbine_blade_length_m": 25, "turbine_nacelle_length_m": 12, "standoff_m": 12})
        assert not planned["ok"]
        assert "below the 70 m minimum altitude" in planned["error"]


def _dome_check(items, origin, centre, radius, base, standoff, floor, ceiling):
    points = nav_points(items, origin)[1:]

    def clearance(q):
        return math.hypot(float(np.linalg.norm(q[:2] - centre)), float(q[2]) - base) - radius
    assert min(p[2] for p in points) >= floor - 0.05
    assert max(p[2] for p in points) <= ceiling + 0.05
    worst = min(clearance(q) for q in legs(points))
    assert worst >= 0.97 * standoff, f"the path comes {worst:.1f} m from the dome"
    taken = shots(items, origin)[1:]
    assert taken
    for position, heading, pitch in taken:
        rho = float(np.linalg.norm(position[:2] - centre))
        d = np.array([-math.cos(math.radians(pitch)), math.sin(math.radians(pitch))])
        o = np.array([rho, float(position[2]) - base])
        b, c = o @ d, o @ o - radius * radius
        assert b * b - c >= 0.0, f"the camera at {rho:.1f} m / {position[2]:.1f} m looks past the dome"
        hit = o + d * (-b - math.sqrt(b * b - c))
        assert hit[1] >= -0.05, "the camera looks at the ground beside the dome"
    return taken


class TestDomeUnderTheCeiling:
    """Rings above max_altitude_m were clamped down onto (R40, max 40) or into (R40,
    max 30) the dome; and a crown inside the generic stand-off from the centre was
    pushed sideways, still looking straight down, at the ground beside the dome."""

    @pytest.mark.parametrize("ceiling", [40.0, 30.0])
    def test_high_bands_are_looked_at_from_the_ceiling_and_said(self, tmp_path, ceiling):
        footprint = rotated_rectangle(60.0, 30.0, 35.0)
        api, origin = make_api(tmp_path, footprint)
        planned, items = plan_export(api, tmp_path, {
            "template": "dome_inspection", "altitude_m": 55, "min_altitude_m": 20,
            "max_altitude_m": ceiling, "dome_radius_m": 40, "standoff_m": 8})
        assert planned["summary"]["adjustments"]["altitude_clamps"] == 0
        ring = footprint_in_frame(footprint, origin)
        centre = np.array([np.mean([p[0] for p in ring]), np.mean([p[1] for p in ring])])
        _dome_check(items, origin, centre, 40.0, 0.0, 8.0, 20.0, ceiling)
        text = " ".join(planned["warnings"])
        assert f"under the {ceiling:.0f} m maximum altitude" in text
        assert "not photographed" in text and "Raise max_altitude_m" in text

    def test_the_crown_is_looked_down_on_from_above_it(self, tmp_path):
        footprint = rotated_rectangle(20.0, 20.0, 0.0)
        api, origin = make_api(tmp_path, footprint)
        planned, items = plan_export(api, tmp_path, {
            "template": "dome_inspection", "altitude_m": 55, "min_altitude_m": 20,
            "dome_radius_m": 15, "standoff_m": 20})
        ring = footprint_in_frame(footprint, origin)
        centre = np.array([np.mean([p[0] for p in ring]), np.mean([p[1] for p in ring])])
        taken = _dome_check(items, origin, centre, 15.0, 0.0, 20.0, 20.0, 120.0)
        top = max(taken, key=lambda s: s[0][2])
        assert float(np.linalg.norm(top[0][:2] - centre)) < 0.5 and abs(top[0][2] - 35.0) < 0.1
        assert top[2] == pytest.approx(-90.0)

    def test_a_dome_above_the_ceiling_is_refused(self, tmp_path):
        api, _ = make_api(tmp_path, rotated_rectangle(20.0, 20.0, 0.0))
        planned = api.plan_mission({
            "template": "dome_inspection", "altitude_m": 45, "max_altitude_m": 50,
            "dome_radius_m": 20, "dome_base_altitude_m": 60, "standoff_m": 8})
        assert not planned["ok"]
        assert "above the 50 m maximum altitude" in planned["error"]
