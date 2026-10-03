"""Fixed-wing planning and export, without a simulator.

The flights themselves are proved against a real ArduPlane in
tests/sitl/test_plane_missions.py; these pin the geometry and the item stream those
flights depend on, so a change that breaks them fails on any machine.
"""

from __future__ import annotations

import math

import pytest

from mission import exporters
from mission.fixed_wing import (
    LEAD_IN_RADII,
    _pass_order,
    convex_hull,
    landing_pattern,
    normalise_vehicle_class,
)
from mission.planner import MissionPlanner

LAT0, LON0 = -35.363261, 149.165230


def _off(n: float, e: float) -> list[float]:
    return [LON0 + e / (111_320.0 * math.cos(math.radians(LAT0))), LAT0 + n / 110_540.0]


FIELD = [_off(n, e) for n, e in ((-60, -200), (-60, 200), (240, 200), (240, -200))]


def _plan(template: str = "grid", **kw):
    plan = MissionPlanner().generate(polygon_lonlat=FIELD, mode=template, altitude_m=80.0,
                                     speed_m_s=18.0, vehicle_class="fixed_wing", **kw)
    data = plan.to_dict()
    data["fixed_wing"] = {"landing": landing_pattern(LAT0, LON0, 0.0)}
    return data


def test_vehicle_class_names():
    assert normalise_vehicle_class("plane") == "fixed_wing"
    assert normalise_vehicle_class(None) == "multirotor"
    with pytest.raises(ValueError):
        normalise_vehicle_class("blimp")


def test_pass_order_never_turns_tighter_than_the_turn_diameter_when_it_can():
    offsets = [31.0 * i for i in range(14)]
    order = _pass_order(offsets, 120.0)
    assert sorted(order) == list(range(14))
    assert all(abs(offsets[b] - offsets[a]) >= 120.0 for a, b in zip(order, order[1:]))


def test_pass_order_falls_back_when_no_wide_order_exists():
    order = _pass_order([0.0, 31.0], 120.0)
    assert sorted(order) == [0, 1]


def test_grid_for_a_plane_has_lead_points_and_wide_turns():
    plan = _plan()
    report = plan["flight_recipe"]["metadata"]["fixed_wing"]
    assert report["tight_turns"] == 0 and report["min_turn_spacing_m"] >= 120.0
    viewpoints = exporters._viewpoints(plan)
    # Every pass is bracketed by non-capturing lead-in / lead-out points.
    assert not viewpoints[0]["trigger"] and not viewpoints[-1]["trigger"]
    leads = sum(1 for v in viewpoints if not v["trigger"])
    assert leads == 2 * report["passes"]


def test_every_pass_is_one_straight_line_from_lead_in_to_lead_out():
    # The planner pulls corner points inside the area; a plane would bank for the kink.
    vps = exporters._viewpoints(_plan())
    k = math.cos(math.radians(LAT0)) * 111_320.0
    pts = [((v["lon"] - LON0) * k, (v["lat"] - LAT0) * 110_540.0, v["trigger"]) for v in vps]
    passes, run = [], []
    for x, y, trig in pts:
        run.append((x, y))
        if not trig and len(run) > 1:  # a lead-out closes the pass
            passes.append(run)
            run = []
    for run in passes:
        (ax, ay), (bx, by) = run[0], run[-1]
        length = math.hypot(bx - ax, by - ay)
        worst = max(abs((x - ax) * (by - ay) - (y - ay) * (bx - ax)) / length for x, y in run)
        assert worst < 0.05, worst


def test_lead_in_is_five_turn_radii_before_the_first_capture():
    plan = _plan()
    vps = exporters._viewpoints(plan)
    lead, first = vps[0], vps[1]
    gap = math.hypot((lead["lon"] - first["lon"]) * 111_320.0 * math.cos(math.radians(LAT0)),
                     (lead["lat"] - first["lat"]) * 110_540.0)
    assert gap == pytest.approx(LEAD_IN_RADII * 60.0, abs=0.5)


def test_fixed_wing_items_take_off_fly_airspeed_and_land():
    plan = _plan()
    items = [i.to_dict() for i in exporters.build_mission_items(plan)]
    cmds = [i["command"] for i in items]
    assert cmds[0] == 22 and items[0]["param1"] == 15.0
    assert cmds[-3:] == [189, 31, 21]
    assert 20 not in cmds and 115 not in cmds
    speed = next(i for i in items if i["command"] == 178)
    assert speed["param1"] == 0.0 and speed["param2"] == 18.0
    loiter = items[-2]
    assert loiter["alt"] == 30.0 and loiter["param2"] == 60.0 and loiter["param4"] == 1.0
    assert math.isnan(items[-1]["param4"])
    # Lead-outs (the point after each pass) are flown to: a 1 m acceptance radius.
    waypoints = [i for i in items if i["command"] == 16]
    passes = plan["flight_recipe"]["metadata"]["fixed_wing"]["passes"]
    assert sum(1 for w in waypoints if w["param2"] == 1.0) == passes


def test_a_plane_mission_without_a_landing_is_refused():
    plan = _plan()
    plan["fixed_wing"] = {}
    with pytest.raises(ValueError, match="landing"):
        exporters.build_mission_items(plan)


def test_hover_templates_are_refused_for_a_plane():
    with pytest.raises(ValueError, match="fixed-wing"):
        MissionPlanner().generate(polygon_lonlat=FIELD, mode="orbit", altitude_m=80.0,
                                  vehicle_class="fixed_wing")


def test_fence_contains_the_turns_and_the_landing():
    plan = _plan()
    (inclusion, ring), = [r for r in exporters.fence_rings(plan) if r[0]]
    from core.preflight import _inside, _ring

    poly = _ring(ring)
    for vp in exporters._viewpoints(plan):
        assert _inside((vp["lon"], vp["lat"]), poly)
    landing = plan["fixed_wing"]["landing"]
    assert _inside((landing["loiter_lon"], landing["loiter_lat"]), poly)
    assert _inside((landing["land_lon"], landing["land_lat"]), poly)


def test_landing_pattern_sits_behind_the_touchdown_point():
    p = landing_pattern(LAT0, LON0, 0.0)  # landing northwards: approach from the south
    assert p["loiter_lat"] < LAT0 and abs(p["loiter_lon"] - LON0) < 1e-9
    assert p["approach_distance_m"] >= 300.0


def test_convex_hull():
    hull = convex_hull([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    assert len(hull) == 4


def test_multirotor_plans_are_unchanged():
    plan = MissionPlanner().generate(polygon_lonlat=FIELD, mode="grid", altitude_m=80.0,
                                     speed_m_s=8.0).to_dict()
    assert "fixed_wing" not in plan["flight_recipe"]["metadata"]
    items = [i.to_dict() for i in exporters.build_mission_items(plan)]
    assert items[0]["param1"] == 0.0 and items[-1]["command"] == 20
    assert next(i for i in items if i["command"] == 178)["param1"] == 1.0
