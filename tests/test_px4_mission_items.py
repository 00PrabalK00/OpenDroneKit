"""Mission items written for PX4 are ones PX4 accepts, and keep the plan's headings.

PX4 v1.15.4 SITL refused an orbit upload at its first CONDITION_YAW ("IGN MISSION_ITEM:
Invalid item", MAV_MISSION_UNSUPPORTED): PX4's mission has no CONDITION_YAW. It flies a
waypoint's param4 as the heading to hold there, so for PX4 the heading travels in the
waypoint and the yaw command is left out. ArduPilot's items are unchanged.
"""

from __future__ import annotations

import math

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from mission.exporters import build_mission_items

CONDITION_YAW, NAV_WAYPOINT, NAV_LOITER_TIME = 115, 16, 19
AOI = [[-73.75760, 45.46722], [-73.75745, 45.46722], [-73.75745, 45.46731], [-73.75760, 45.46731]]

# Commands PX4 v1.15.4 accepts in a mission (src/modules/mavlink/mavlink_mission.cpp,
# parse_mavlink_mission_item) that the planner can emit.
PX4_ACCEPTS = {16, 17, 19, 20, 21, 22, 112, 178, 201, 203, 206, 1000, 2000, 2001}


def _plan(tmp_path, template: str, **options) -> dict:
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("px4", str(tmp_path / "project"))
    assert api.set_aoi(AOI)["ok"]
    planned = api.plan_mission({"template": template, "altitude_m": 20.0, **options})
    assert planned["ok"], planned
    return api._session.mission_plan_dict


@pytest.mark.parametrize("template", ["orbit", "grid"])
def test_px4_items_use_only_commands_px4_accepts(tmp_path, template):
    items = build_mission_items(_plan(tmp_path, template), autopilot="px4")
    assert {int(i.command) for i in items} <= PX4_ACCEPTS
    assert [i.seq for i in items] == list(range(len(items)))


def test_an_orbit_keeps_its_headings_on_px4(tmp_path):
    plan = _plan(tmp_path, "orbit")
    ardupilot = build_mission_items(plan)
    px4 = build_mission_items(plan, autopilot="px4")
    assert any(i.command == CONDITION_YAW for i in ardupilot)
    assert not any(i.command == CONDITION_YAW for i in px4)

    def headings(items):
        return [round(i.param4, 3) for i in items if i.command == NAV_WAYPOINT]

    # Every waypoint carries the same heading on both, and on ArduPilot it is also the
    # CONDITION_YAW that precedes it -- so PX4 loses nothing by going without one.
    assert headings(px4) == headings(ardupilot)
    assert all(math.isfinite(h) for h in headings(px4))
    yaws = [round(i.param1, 3) for i in ardupilot if i.command == CONDITION_YAW]
    assert [h % 360.0 for h in headings(ardupilot)] == pytest.approx(yaws)


def test_a_hold_on_px4_is_a_waypoint_that_keeps_its_heading():
    plan = {"world_poses": [{"lat": 45.4672, "lon": -73.7576, "alt": 20.0, "yaw_deg": 90.0,
                             "camera_yaw_locked": True, "gimbal_pitch_deg": -30.0,
                             "dwell_s": 1.5, "trigger": True}]}
    px4 = build_mission_items(plan, autopilot="px4")
    holds = [i for i in px4 if i.command == NAV_WAYPOINT and i.param1 > 0]
    assert len(holds) == 1
    assert holds[0].param1 == pytest.approx(1.5) and holds[0].param4 == pytest.approx(90.0)
    assert not any(i.command in (CONDITION_YAW, NAV_LOITER_TIME) for i in px4)
    # ArduPilot keeps its LOITER_TIME (whole seconds, rounded up) and CONDITION_YAW.
    ardupilot = build_mission_items(plan)
    assert any(i.command == NAV_LOITER_TIME and i.param1 == 2.0 for i in ardupilot)
    assert any(i.command == CONDITION_YAW and i.param1 == pytest.approx(90.0) for i in ardupilot)
