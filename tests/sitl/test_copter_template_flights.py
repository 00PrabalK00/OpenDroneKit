"""Grid, orbit and facade missions flown to completion by ArduCopter in AUTO.

Planned and uploaded through the Api, started with ``vehicle_command("start")`` after
preflight, and then judged entirely from the vehicle's independent MAVLink stream:

* MISSION_ITEM_REACHED for every waypoint of the plan;
* one CAMERA_FEEDBACK per planned capture, taken where the plan put the capture
  (position and altitude), facing where the plan pointed it (yaw, for templates that
  lock the heading), with the gimbal at the planned pitch (the servo gimbal's
  GIMBAL_DEVICE_ATTITUDE_STATUS when the shutter fired);
* the mission's closing RTL: the aircraft lands at home and disarms.
"""

from __future__ import annotations

import json
import math
import os
import time

import pytest

from ._support import (
    NAV_COMMANDS,
    angle_diff_deg,
    connect_api,
    distance_m,
    evidence_dir,
    fresh_battery,
    make_api,
    plan_template,
    settle_on_ground,
    wait_telemetry,
)

pytestmark = pytest.mark.sitl

POSITION_TOL_M = 3.5   # WP_RADIUS is 2 m: the shutter fires when the waypoint is reached
ALTITUDE_TOL_M = 1.5
# A facade turns 90 deg at each building corner within one short leg; ArduCopter arrives
# up to ~12 deg short there (on the straight runs, and around an orbit, within ~1-5 deg).
YAW_TOL_DEG = 15.0
GIMBAL_TOL_DEG = 2.0


def _gimbal_pitch_deg(status: dict) -> float:
    """Pitch from GIMBAL_DEVICE_ATTITUDE_STATUS's quaternion (w, x, y, z)."""
    w, x, y, z = status["q"]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


@pytest.mark.parametrize("template", ["grid", "orbit", "facade"])
def test_template_flies_to_completion_with_its_captures(tmp_path, sitl_connection, sitl_observer,
                                                         sitl_trace, template):
    from mission import exporters

    api = make_api(tmp_path)
    home = connect_api(api, sitl_connection)
    record: dict = {"template": template}
    try:
        assert plan_template(api, template)["ok"]
        plan = api._session.mission_plan_dict
        items = [i.to_dict() for i in exporters.build_mission_items(plan)]
        viewpoints = exporters._viewpoints(plan)
        upload = api.upload_mission()
        assert upload["ok"], upload

        assert api.vehicle_command("loiter")["ok"]
        fresh_battery(sitl_observer)
        wait_telemetry(api, "a charged battery", lambda t: (t.get("battery_pct") or 0) > 90, 15.0)
        preflight = api.run_preflight()
        confirmed = api.run_preflight(acknowledged=preflight["pending_confirmation"])
        record["preflight"] = [(c["id"], c["severity"], c["message"]) for c in confirmed["checks"]]
        assert confirmed["can_start"], [c for c in record["preflight"] if c[1] != "pass"]

        mark = sitl_trace.mark()
        started = api.vehicle_command("start")
        record["start"] = started
        assert started["ok"], started
        wait_telemetry(api, "AUTO", lambda t: t["flight_mode"] == "AUTO", 20.0)

        timeout_s = float(os.environ.get("ODK_SITL_FLIGHT_TIMEOUT_S", "900"))
        landed = wait_telemetry(api, "the mission, its RTL and the landing",
                                lambda t: not t["armed"], timeout_s)
        time.sleep(1.0)

        # Vehicle sequence numbers run one ahead of the plan's: seq 0 is ArduPilot's home.
        # ArduPilot sends no MISSION_ITEM_REACHED for the takeoff, and the closing RTL
        # ends in a landing, so those two are judged by the landing at home instead.
        nav_seqs = {int(i["seq"]) + 1 for i in items if int(i["command"]) in NAV_COMMANDS
                    and int(i["command"]) not in (20, 22)}
        reached = {int(e.data["seq"]) for e in sitl_trace.messages("MISSION_ITEM_REACHED", mark)}
        feedback = [e for e in sitl_trace.messages("CAMERA_FEEDBACK", mark)]
        mounts = sitl_trace.messages("GIMBAL_DEVICE_ATTITUDE_STATUS", mark)
        planned_shots = [v for v in viewpoints if v["trigger"]]

        shots = []
        for index, (pose, entry) in enumerate(zip(planned_shots, feedback)):
            fb = entry.data
            lat, lon = fb["lat"] / 1e7, fb["lng"] / 1e7
            mount = next((m.data for m in reversed(mounts) if m.t <= entry.t), None)
            shots.append({
                "shot": index,
                "planned": [round(pose["lat"], 7), round(pose["lon"], 7), round(pose["alt"], 2),
                            round(pose["yaw_deg"] % 360.0, 1), pose["gimbal_pitch_deg"]],
                "camera_feedback": [lat, lon, round(fb["alt_rel"], 2), round(fb["yaw"] % 360.0, 1)],
                "position_error_m": round(distance_m(lat, lon, pose["lat"], pose["lon"]), 2),
                "altitude_error_m": round(abs(fb["alt_rel"] - pose["alt"]), 2),
                "yaw_error_deg": (round(angle_diff_deg(fb["yaw"], pose["yaw_deg"]), 1)
                                  if pose["yaw_locked"] else None),
                "mount_pitch_deg": None if mount is None else round(_gimbal_pitch_deg(mount), 1),
            })
        record.update({
            "items": len(items), "viewpoints": len(viewpoints),
            "planned_shots": len(planned_shots), "camera_feedback": len(feedback),
            "nav_items": len(nav_seqs), "reached": len(reached & nav_seqs),
            "missed": sorted(nav_seqs - reached),
            "landed_from_home_m": distance_m(landed["latitude"], landed["longitude"],
                                             home["home_lat"], home["home_lon"]),
            "shots": shots,
            "status_text": sitl_trace.statustext(mark)[-30:],
        })

        assert not record["missed"], f"waypoints never reached: {record['missed']}"
        assert len(feedback) == len(planned_shots), (len(feedback), len(planned_shots))
        bad_position = [s for s in shots if s["position_error_m"] > POSITION_TOL_M
                        or s["altitude_error_m"] > ALTITUDE_TOL_M]
        assert not bad_position, bad_position[:5]
        bad_yaw = [s for s in shots if s["yaw_error_deg"] is not None
                   and s["yaw_error_deg"] > YAW_TOL_DEG]
        assert not bad_yaw, bad_yaw[:5]
        assert mounts, "no GIMBAL_DEVICE_ATTITUDE_STATUS: the gimbal was never reported"
        bad_gimbal = [s for s in shots if s["mount_pitch_deg"] is None
                      or abs(s["mount_pitch_deg"] - s["planned"][4]) > GIMBAL_TOL_DEG]
        assert not bad_gimbal, bad_gimbal[:5]
        assert record["landed_from_home_m"] < 5.0, record["landed_from_home_m"]
    finally:
        (evidence_dir("flights") / f"{template}.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8")
        try:
            settle_on_ground(api)
        finally:
            api.disconnect_vehicle()


def test_distance_triggered_capture_fires_at_the_planned_spacing(tmp_path, sitl_connection,
                                                                 sitl_observer, sitl_trace):
    """A continuous-capture plan: DO_SET_CAM_TRIGG_DIST on, flown, then off.

    No built-in template asks for distance triggering yet, so the plan is given to the
    Api directly -- the same path an imported or saved continuous-capture plan takes.
    """
    from ._support import offset_lonlat

    api = make_api(tmp_path)
    home = connect_api(api, sitl_connection)
    spacing = 6.0
    route = [offset_lonlat(0.0, 0.0), offset_lonlat(0.0, 42.0), offset_lonlat(30.0, 42.0)]
    poses = [{"lon": lon, "lat": lat, "alt_m": 20.0, "yaw_deg": 0.0, "gimbal_pitch_deg": -90.0,
              "dwell_s": 0.0, "trigger": False} for lon, lat in route]
    api._session.mission_plan_dict = {
        "template": "waypoints", "altitude_m": 20.0, "cruise_speed_m_s": 5.0,
        "flight_recipe": {"world_poses": poses,
                          "capture": {"continuous_capture": True, "capture_spacing_m": spacing}},
        "safety_constraints": {"rth_altitude_m": 25.0},
    }
    record: dict = {"spacing_m": spacing}
    try:
        assert api.upload_mission()["ok"]
        assert api.vehicle_command("loiter")["ok"]
        fresh_battery(sitl_observer)
        mark = sitl_trace.mark()
        started = api.vehicle_command("start", override_preflight=True)
        assert started["ok"], started
        wait_telemetry(api, "mission and landing", lambda t: not t["armed"], 600.0)
        reached = sitl_trace.messages("MISSION_ITEM_REACHED", mark)
        shots = [(e.data["lat"] / 1e7, e.data["lng"] / 1e7) for e in sitl_trace.messages("CAMERA_FEEDBACK", mark)]
        gaps = [distance_m(a[0], a[1], b[0], b[1]) for a, b in zip(shots, shots[1:])]
        # Triggering starts on the leg after the takeoff and stops at the last waypoint.
        on_route = [s for s in shots if distance_m(s[0], s[1], home["home_lat"], home["home_lon"]) < 60.0]
        record.update(shots=len(shots), gaps_m=[round(g, 2) for g in gaps],
                      reached=[e.data["seq"] for e in reached])
        expected = int((42.0 + 30.0) / spacing)
        assert abs(len(on_route) - expected) <= 2, (len(on_route), expected, record)
        inner = gaps[1:-1]
        assert inner and all(abs(g - spacing) < 1.5 for g in inner), record
    finally:
        (evidence_dir("flights") / "distance_trigger.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8")
        try:
            settle_on_ground(api)
        finally:
            api.disconnect_vehicle()
