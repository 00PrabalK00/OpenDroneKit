"""Fixed-wing missions flown by a real ArduPlane (Plane-4.5.7 SITL), through the app.

Run in the plane image (infrastructure/docker/Dockerfile.sitl-plane, which sets
ODK_SITL_VEHICLE=ArduPlane); in any other environment these tests skip.

Everything goes through ``app.api.Api`` as the UIs drive it -- connect, plan, upload,
preflight, start, RTL, land -- and is judged from the vehicle's own MAVLink stream on an
independent observer link:

* the vehicle identifies as a fixed-wing and the app plans for one (turn-aware pass
  order, lead-in/lead-out points, a landing sequence instead of a closing RTL);
* upload/download fidelity: every item ArduPlane stores is the item the app sent;
* corridor and grid flown in AUTO from takeoff to touchdown: every waypoint reached,
  the passes flown on their lines at the planned height and airspeed, one photograph per
  planned capture and none in the turns, landing at the planned point, disarmed;
* RTL mid-mission brings the plane home to circle, and Land then flies the landing
  sequence to the ground.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from typing import Any

import pytest

from tools.sitl.launch import SITL_HOME_LAT, SITL_HOME_LON, sitl_vehicle

from ._support import (
    MavlinkTrace,
    connect_api,
    distance_m,
    evidence_dir,
    fresh_battery,
    make_api,
    offset_lonlat,
    read_vehicle_param,
    wait_telemetry,
)

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.sitl_plane,
    pytest.mark.skipif(sitl_vehicle() != "ArduPlane",
                       reason="needs the ArduPlane SITL image (ODK_SITL_VEHICLE=ArduPlane)"),
]

NAV_WAYPOINT, NAV_LAND, NAV_TAKEOFF, NAV_LOITER_TO_ALT = 16, 21, 22, 31
NAV_RTL, CONDITION_YAW, DO_CHANGE_SPEED, DO_LAND_START = 20, 115, 178, 189
DO_DIGICAM_CONTROL = 203

# The survey site: a field beside the SITL home, and a corridor strip north of it.
GRID_FIELD = [offset_lonlat(n, e) for n, e in ((-60, -200), (-60, 200), (240, 200), (240, -200))]
CORRIDOR_STRIP = [offset_lonlat(n, e) for n, e in ((200, -450), (200, 450), (225, 450), (225, -450))]
PLAN_OPTIONS: dict[str, Any] = {
    "altitude_m": 80.0, "speed_m_s": 18.0, "min_altitude_m": 30.0, "max_altitude_m": 120.0,
    # Above the survey height, so the return is seen to use the plan's altitude.
    "rth_altitude_m": 90.0,
}

# What "flown as planned" means for a plane, measured on the observer stream.
# On a capture pass, after the first capture point: distance from the pass line.
CROSS_TRACK_TOL_M = 10.0
PASS_ALTITUDE_TOL_M = 5.0
AIRSPEED_TOL_M_S = 2.0
# A photograph counts as the planned one when taken this close to its capture point.
# The shutter fires on reaching the waypoint, and ArduPlane "reaches" a straight-ahead
# waypoint when it passes it, within about one second's flight (~18 m).
CAPTURE_POSITION_TOL_M = 25.0
# A nadir photograph taken in a bank looks sideways: at 80 m, 5 degrees moves the
# footprint ~7 m. Every photo is to be taken with the wings level to within this.
PHOTO_BANK_TOL_DEG = 5.0
# Touchdown and roll-out: ArduPlane flares and rolls on along the landing heading.
TOUCHDOWN_TOL_M = 60.0
FLIGHT_TIMEOUT_S = 900.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _connect(api, uri: str) -> dict:
    tel = connect_api(api, uri)
    client = api._session._drone_client
    assert client.vehicle_class == "fixed_wing", client.vehicle_class
    return tel


def _plan(api, template: str) -> dict:
    polygon = CORRIDOR_STRIP if template == "corridor" else GRID_FIELD
    assert api.set_aoi(polygon)["ok"]
    # No vehicle_class given: the connected vehicle's HEARTBEAT decides.
    planned = api.plan_mission({"template": template, **PLAN_OPTIONS})
    assert planned["ok"], planned
    return planned


def _local(lat: float, lon: float) -> tuple[float, float]:
    """(east, north) metres from the SITL home."""
    return ((lon - SITL_HOME_LON) * 111_320.0 * math.cos(math.radians(SITL_HOME_LAT)),
            (lat - SITL_HOME_LAT) * 110_540.0)


def _cross_track_m(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = math.hypot(dx, dy) or 1.0
    return abs((p[0] - ax) * dy - (p[1] - ay) * dx) / length


def _vehicle_items(api) -> list[dict]:
    return api._session._drone_client.download_mission()


def _passes(items: list[dict], viewpoints: list[dict]) -> list[list[dict]]:
    """Capture passes: runs of consecutive capturing waypoints, as uploaded items."""
    waypoints = [i for i in items if int(i["command"]) == NAV_WAYPOINT]
    assert len(waypoints) == len(viewpoints), (len(waypoints), len(viewpoints))
    passes: list[list[dict]] = []
    run: list[dict] = []
    for item, vp in zip(waypoints, viewpoints):
        if vp["trigger"]:
            run.append(item)
        elif run:
            passes.append(run)
            run = []
    if run:
        passes.append(run)
    return passes


# How ArduPilot 4.5 stores each command (libraries/AP_Mission/AP_Mission.cpp,
# mavlink_int_to_mission_cmd / mission_cmd_to_mavlink_int), so a read-back is compared
# with what the vehicle can hold rather than with fields it has nowhere to keep:
# * the frame is stored only for commands that carry a location (stored_in_location);
#   a do-command reads back frame 0 whatever was sent;
# * NAV_WAYPOINT on ArduPlane packs param2 (acceptance radius) and param3 (pass-by) into
#   p1; there is no room left for param1 (hold), and param4 is not kept;
# * NAV_LOITER_TO_ALT keeps |param2| (radius, sign = direction) and param4 > 0 (exit
#   tangent); param1 "heading required" is not stored because ArduPlane always requires
#   it (Plane::verify_loiter_to_alt's secondary goal is verify_loiter_heading);
# * NAV_LAND keeps param1 (abort altitude) and param4's sign as the deep-stall direction,
#   which reads back as +1 or -1;
# * DO_LAND_START keeps only its location.
LOCATION_COMMANDS = {NAV_WAYPOINT, NAV_LAND, NAV_TAKEOFF, NAV_LOITER_TO_ALT, DO_LAND_START}
STORED_PARAMS = {
    NAV_WAYPOINT: ("param2", "param3"),
    NAV_TAKEOFF: ("param1",),
    NAV_LOITER_TO_ALT: ("param2", "param4"),
    NAV_LAND: ("param1", "param4"),
    DO_LAND_START: (),
}


def _as_stored(command: int, key: str, value: float) -> float:
    if command == NAV_LAND and key == "param4":
        return -1.0 if value < 0 else 1.0  # NaN compares False: no direction, +1
    if math.isnan(value):
        return 0.0
    return value


def stored_item_mismatches(sent: list[dict], stored: list[dict]) -> list[str]:
    problems = []
    if len(stored) != len(sent):
        problems.append(f"count: sent {len(sent)}, stored {len(stored)}")
    for a, b in zip(sent, stored):
        cmd = int(a["command"])
        where = f"seq {a['seq']} cmd {cmd} ({a['label']})"
        if int(b["command"]) != cmd:
            problems.append(f"{where}: stored cmd {b['command']}")
            continue
        keys = STORED_PARAMS.get(cmd, ("param1", "param2", "param3", "param4"))
        tolerances = {k: 1e-3 for k in keys}
        if cmd in LOCATION_COMMANDS:
            if int(a["frame"]) != int(b["frame"]):
                problems.append(f"{where}: frame {a['frame']} stored {b['frame']}")
            tolerances.update({"lat": 1.5e-7, "lon": 1.5e-7, "alt": 0.011})
        else:
            tolerances.update({"lat": 1e-3, "lon": 1e-3, "alt": 1e-3})  # x/y/z params
        for key, tol in tolerances.items():
            want = _as_stored(cmd, key, float(a[key]))
            if abs(want - float(b[key])) > tol:
                problems.append(f"{where}: {key} sent {a[key]} stored {b[key]}")
    return problems


def _write(name: str, record: dict) -> None:
    (evidence_dir("plane") / f"{name}.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")


def _start(api, observer) -> dict:
    fresh_battery(observer)
    wait_telemetry(api, "a charged battery", lambda t: (t.get("battery_pct") or 0) > 90, 20.0)
    preflight = api.run_preflight()
    confirmed = api.run_preflight(acknowledged=preflight["pending_confirmation"])
    checks = [(c["id"], c["severity"], c["message"]) for c in confirmed["checks"]]
    assert confirmed["can_start"], [c for c in checks if c[1] != "pass"]
    started = api.vehicle_command("start")
    assert started["ok"], started
    wait_telemetry(api, "AUTO", lambda t: t["flight_mode"] == "AUTO", 20.0)
    return {"preflight": checks, "start": started}


def _wait_landed(api, trace: MavlinkTrace, since: float, timeout_s: float) -> dict:
    """Until the plane has landed and disarmed itself (LAND_DISARMDELAY)."""
    deadline = time.monotonic() + timeout_s
    airborne = False
    while time.monotonic() < deadline:
        tel = api.telemetry()["telemetry"]
        airborne = airborne or tel["altitude_rel_m"] > 20.0
        if airborne and not tel["armed"]:
            return tel
        time.sleep(1.0)
    pytest.fail(f"The plane did not land and disarm within {timeout_s:.0f}s; "
                f"last STATUSTEXT: {trace.statustext(since)[-8:]}")


def _settle(api) -> None:
    """Leave the plane on the ground and disarmed for the next test."""
    tel = api.telemetry()["telemetry"]
    if not tel.get("armed"):
        return
    if tel.get("altitude_rel_m", 0.0) > 2.0:
        api.vehicle_command("land")
        deadline = time.monotonic() + FLIGHT_TIMEOUT_S
        while time.monotonic() < deadline and api.telemetry()["telemetry"].get("armed"):
            time.sleep(1.0)
    if api.telemetry()["telemetry"].get("armed"):
        api.vehicle_command("disarm")


def _mission_restarted_at(trace: MavlinkTrace, since: float, timeout_s: float = 30.0) -> float:
    """When the mission was set back to its first item for this start.

    Until then MISSION_CURRENT still reports where the previous mission ended -- its
    NAV_LAND, a high sequence number that can fall inside this mission's passes -- so
    anything keyed on the current item must start from here.
    """
    entry = trace.wait_for("MISSION_CURRENT", lambda d: int(d["seq"]) <= 2, since, timeout_s)
    assert entry is not None, "the mission was never set back to its start"
    return entry.t


def _flight_report(api, trace: MavlinkTrace, since: float, items_sent: list[dict],
                   plan: dict) -> dict:
    """Everything the observer saw, judged against the plan."""
    from mission import exporters

    viewpoints = exporters._viewpoints(plan)
    landing = plan["fixed_wing"]["landing"]
    # The vehicle numbers its items from 1: 0 is home. A waypoint is reached when the
    # vehicle says so, in MISSION_ITEM_REACHED or in its "Reached/Passed waypoint #N"
    # text (both are the vehicle's own report; a UDP stream can drop either one).
    reached = {int(e.data["seq"]) - 1 for e in trace.messages("MISSION_ITEM_REACHED", since)}
    for text in trace.statustext(since):
        found = re.match(r"(?:Reached|Passed) waypoint #(\d+)", text)
        if found:
            reached.add(int(found.group(1)) - 1)
    since = _mission_restarted_at(trace, since)
    current = [(e.t, int(e.data["seq"]) - 1) for e in trace.messages("MISSION_CURRENT", since)]
    positions = [(e.t, e.data) for e in trace.messages("GLOBAL_POSITION_INT", since)]
    hud = [(e.t, e.data) for e in trace.messages("VFR_HUD", since)]

    def target_at(t: float) -> int | None:
        value = None
        for ts, seq in current:
            if ts > t:
                break
            value = seq
        return value

    report: dict[str, Any] = {"passes": []}
    waypoint_seqs = [int(i["seq"]) for i in items_sent if int(i["command"]) == NAV_WAYPOINT]
    report["waypoints_not_reached"] = [s for s in waypoint_seqs if s not in reached]
    for run in _passes(items_sent, viewpoints):
        a = _local(run[0]["lat"], run[0]["lon"])
        b = _local(run[-1]["lat"], run[-1]["lon"])
        on_pass = {int(i["seq"]) for i in run[1:]}
        xt, alt_err, airspeed = [], [], []
        for t, d in positions:
            if target_at(t) in on_pass:
                p = _local(d["lat"] / 1e7, d["lon"] / 1e7)
                xt.append(_cross_track_m(p, a, b))
                alt_err.append(d["relative_alt"] / 1000.0 - float(run[0]["alt"]))
        for t, d in hud:
            if target_at(t) in on_pass:
                airspeed.append(float(d["airspeed"]))
        report["passes"].append({
            "first_seq": run[0]["seq"], "last_seq": run[-1]["seq"], "captures": len(run),
            "samples": len(xt),
            "max_cross_track_m": round(max(xt), 2) if xt else None,
            "max_abs_alt_err_m": round(max(abs(v) for v in alt_err), 2) if alt_err else None,
            "median_airspeed_m_s": round(statistics.median(airspeed), 2) if airspeed else None,
        })
    captures = [vp for vp in viewpoints if vp["trigger"]]
    feedback = [e.data for e in trace.messages("CAMERA_FEEDBACK", since)]
    photo_misses = []
    for fb in feedback:
        p = (fb["lat"] / 1e7, fb["lng"] / 1e7)
        nearest = min(distance_m(p[0], p[1], c["lat"], c["lon"]) for c in captures)
        if nearest > CAPTURE_POSITION_TOL_M:
            photo_misses.append({"lat": p[0], "lon": p[1], "nearest_capture_m": round(nearest, 1),
                                 "alt_rel_m": fb.get("alt_rel")})
    rolls = [round(float(fb.get("roll", 0.0)), 1) for fb in feedback]
    report.update({
        "planned_captures": len(captures), "photos": len(feedback),
        "photos_away_from_a_capture_point": photo_misses,
        # CAMERA_FEEDBACK's roll: the bank the aircraft was in when the shutter fired.
        "max_photo_bank_deg": max((abs(r) for r in rolls), default=None),
        "photos_banked_over_tol": [(i, r) for i, r in enumerate(rolls) if abs(r) > PHOTO_BANK_TOL_DEG],
    })
    last = positions[-1][1] if positions else None
    if last is not None:
        report["stopped_at_m_from_landing_point"] = round(distance_m(
            last["lat"] / 1e7, last["lon"] / 1e7, landing["land_lat"], landing["land_lon"]), 1)
        report["stopped_alt_rel_m"] = round(last["relative_alt"] / 1000.0, 2)
    report["statustext"] = trace.statustext(since)
    return report


def _assert_flown_as_planned(report: dict) -> None:
    assert not report["waypoints_not_reached"], report["waypoints_not_reached"]
    for p in report["passes"]:
        assert p["samples"] > 0, p
        assert p["max_cross_track_m"] <= CROSS_TRACK_TOL_M, p
        assert p["max_abs_alt_err_m"] <= PASS_ALTITUDE_TOL_M, p
        assert abs(p["median_airspeed_m_s"] - PLAN_OPTIONS["speed_m_s"]) <= AIRSPEED_TOL_M_S, p
    assert report["photos"] == report["planned_captures"], (report["photos"], report["planned_captures"])
    assert not report["photos_away_from_a_capture_point"], report["photos_away_from_a_capture_point"][:5]
    assert not report["photos_banked_over_tol"], report["photos_banked_over_tol"][:10]
    assert report["stopped_at_m_from_landing_point"] <= TOUCHDOWN_TOL_M, report
    assert report["stopped_alt_rel_m"] < 2.0, report


# ── tests ────────────────────────────────────────────────────────────────────

def test_plane_identifies_as_fixed_wing_and_refuses_hover_templates(tmp_path, sitl_connection,
                                                                    sitl_trace):
    api = make_api(tmp_path)
    _connect(api, sitl_connection)
    heartbeats = [e.data for e in sitl_trace.messages("HEARTBEAT")]
    assert heartbeats and all(int(h["type"]) == 1 for h in heartbeats), heartbeats[:3]
    assert api.set_aoi(GRID_FIELD)["ok"]
    refused = api.plan_mission({"template": "orbit", **PLAN_OPTIONS, "orbit_radius_m": 40.0})
    assert not refused["ok"] and "fixed-wing" in refused["error"], refused
    # A multirotor plan is not sent to a plane.
    copter_plan = api.plan_mission({"template": "grid", **PLAN_OPTIONS, "vehicle_class": "multirotor"})
    assert copter_plan["ok"], copter_plan
    upload = api.upload_mission()
    assert not upload["ok"] and "fixed-wing" in upload["error"], upload
    _write("identify", {"heartbeat": heartbeats[-1], "orbit": refused, "copter_upload": upload})


@pytest.mark.parametrize("template", ["corridor", "grid"])
def test_fixed_wing_plan_is_stored_exactly(tmp_path, sitl_connection, template):
    """Upload, then read back: ArduPlane holds every item as the app sent it."""
    from mission import exporters

    api = make_api(tmp_path)
    _connect(api, sitl_connection)
    planned = _plan(api, template)
    plan = api._session.mission_plan_dict
    sent = [i.to_dict() for i in exporters.build_mission_items(plan)]
    commands = [int(i["command"]) for i in sent]
    # What makes it a fixed-wing mission.
    assert commands[0] == NAV_TAKEOFF and sent[0]["param1"] == 15.0
    assert commands[-3:] == [DO_LAND_START, NAV_LOITER_TO_ALT, NAV_LAND], commands[-3:]
    assert NAV_RTL not in commands and CONDITION_YAW not in commands
    speed = [i for i in sent if int(i["command"]) == DO_CHANGE_SPEED]
    assert speed and all(i["param1"] == 0.0 for i in speed), speed  # airspeed
    fw = plan["fixed_wing"]
    assert fw["passes"]["tight_turns"] == (1 if template == "corridor" else 0), fw

    upload = api.upload_mission()
    assert upload["ok"], upload
    stored = _vehicle_items(api)
    problems = stored_item_mismatches(sent, stored)
    _write(f"fidelity-{template}", {"planned": planned.get("summary"), "fixed_wing": fw,
                                    "items": len(sent), "problems": problems,
                                    "tail_sent": sent[-4:], "tail_stored": stored[-4:]})
    assert not problems, problems[:20]


@pytest.mark.parametrize("template", ["corridor", "grid"])
def test_fixed_wing_mission_flies_auto_from_takeoff_to_landing(tmp_path, sitl_connection,
                                                               sitl_observer, sitl_trace, template):
    from mission import exporters

    api = make_api(tmp_path)
    _connect(api, sitl_connection)
    record: dict[str, Any] = {"template": template}
    try:
        _plan(api, template)
        plan = api._session.mission_plan_dict
        sent = [i.to_dict() for i in exporters.build_mission_items(plan)]
        assert api.upload_mission()["ok"]
        mark = sitl_trace.mark()
        record.update(_start(api, sitl_observer))
        record["landed"] = _wait_landed(api, sitl_trace, mark, FLIGHT_TIMEOUT_S)
        record["report"] = report = _flight_report(api, sitl_trace, mark, sent, plan)
        _write(f"flight-{template}", record)
        _assert_flown_as_planned(report)
    finally:
        _write(f"flight-{template}", record)
        _settle(api)


def _fly_into_survey(api, observer, trace, template: str, pass_index: int):
    """Plan, upload, start, and wait until the plane is on pass `pass_index`."""
    from mission import exporters

    _plan(api, template)
    plan = api._session.mission_plan_dict
    sent = [i.to_dict() for i in exporters.build_mission_items(plan)]
    upload = api.upload_mission()
    assert upload["ok"], upload
    mark = trace.mark()
    started = _start(api, observer)
    restarted = _mission_restarted_at(trace, mark)
    target = _passes(sent, exporters._viewpoints(plan))[pass_index][0]["seq"]
    assert trace.wait_for("MISSION_CURRENT", lambda d: int(d["seq"]) - 1 >= target,
                          restarted, 400.0), f"the plane never reached pass {pass_index}"
    return plan, sent, upload, started


def test_rtl_mid_mission_returns_home_at_plan_altitude_and_lands(tmp_path, sitl_connection,
                                                                 sitl_observer, sitl_trace):
    """RTL: home at the plan's return altitude, then the landing sequence (RTL_AUTOLAND 1)."""
    api = make_api(tmp_path)
    _connect(api, sitl_connection)
    record: dict[str, Any] = {}
    try:
        plan, sent, upload, started = _fly_into_survey(api, sitl_observer, sitl_trace, "grid", 2)
        record.update({"upload_return": upload["upload"].get("return"), **started})
        land_start_seq = next(int(i["seq"]) for i in sent if int(i["command"]) == DO_LAND_START)
        record["vehicle_params"] = {
            k: read_vehicle_param(sitl_observer, sitl_trace, k)
            for k in ("RTL_AUTOLAND", "RTL_ALTITUDE", "WP_LOITER_RAD")}
        assert record["vehicle_params"]["RTL_AUTOLAND"] == 1.0, record["vehicle_params"]
        assert record["vehicle_params"]["RTL_ALTITUDE"] == PLAN_OPTIONS["rth_altitude_m"]

        rtl_mark = sitl_trace.mark()
        rtl = api.vehicle_command("rtl")
        record["rtl"] = rtl
        assert rtl["ok"], rtl
        wait_telemetry(api, "RTL", lambda t: t["flight_mode"] == "RTL", 10.0)
        # Home as the vehicle holds it now: ArduPlane sets it again when it arms.
        now = api.telemetry()["telemetry"]
        home_lat, home_lon = now["home_lat"], now["home_lon"]
        # The RTL leg, from the observer's full-rate stream (polling the app at 1 Hz
        # misses most of a leg flown at 4x speed-up): from the RTL until the vehicle
        # itself switches to the landing sequence.
        handover = sitl_trace.wait_for("STATUSTEXT", lambda d: d["text"] == "Landing sequence start",
                                       rtl_mark, 300.0)
        assert handover is not None, ("RTL never handed over to the landing sequence: "
                                      f"{sitl_trace.statustext(rtl_mark)[-10:]}")
        leg = [e.data for e in sitl_trace.messages("GLOBAL_POSITION_INT", rtl_mark)
               if e.t <= handover.t]
        track = [(round(distance_m(d["lat"] / 1e7, d["lon"] / 1e7, home_lat, home_lon), 1),
                  round(d["relative_alt"] / 1000.0, 1)) for d in leg]
        name = api._session._drone_client._mode_name
        modes = [name(e.data["custom_mode"], e.data["base_mode"])
                 for e in sitl_trace.messages("HEARTBEAT", rtl_mark) if e.t < handover.t]
        modes = modes[modes.index("RTL"):] if "RTL" in modes else modes
        while modes and modes[-1] == "AUTO":
            modes.pop()  # the handover itself: AUTO is heard just before its STATUSTEXT
        record["rtl_modes_before_handover"] = sorted(set(modes))
        record["rtl_track_start_end"] = (track[:3], track[-3:])
        record["rtl_start_distance_m"] = track[0][0]
        record["rtl_distance_at_handover_m"] = track[-1][0]
        # Flew home in RTL, at the plan's return altitude, until ArduPlane itself judged
        # home reached -- its L1 controller switching from capture to circling the home
        # loiter (_WPcircle), about one loiter radius plus one L1 distance out -- and
        # started the landing sequence (RTL_AUTOLAND 1, set by the upload).
        assert set(modes) == {"RTL"}, modes
        assert track[-1][0] < 0.5 * track[0][0], record["rtl_track_start_end"]
        assert abs(track[-1][1] - PLAN_OPTIONS["rth_altitude_m"]) < 10.0, track[-3:]
        # RTL_AUTOLAND 1: home reached, the vehicle itself starts the landing sequence.
        assert api.telemetry()["telemetry"]["flight_mode"] == "AUTO"
        jumped = sitl_trace.wait_for("MISSION_CURRENT",
                                     lambda d: int(d["seq"]) - 1 > land_start_seq, rtl_mark, 10.0)
        assert jumped is not None, "RTL did not continue into the landing sequence"
        record["landed"] = stopped = _wait_landed(api, sitl_trace, rtl_mark, FLIGHT_TIMEOUT_S)
        landing = plan["fixed_wing"]["landing"]
        record["stopped_at_m_from_landing_point"] = round(distance_m(
            stopped["latitude"], stopped["longitude"], landing["land_lat"], landing["land_lon"]), 1)
        record["statustext"] = sitl_trace.statustext(rtl_mark)
        assert record["stopped_at_m_from_landing_point"] <= TOUCHDOWN_TOL_M, record
        assert stopped["altitude_rel_m"] < 2.0, stopped
    finally:
        _write("rtl", record)
        _settle(api)


def test_land_command_mid_mission_flies_the_landing_sequence(tmp_path, sitl_connection,
                                                             sitl_observer, sitl_trace):
    """Land on a plane: DO_LAND_START, not NAV_LAND (which ArduPlane answers unsupported)."""
    api = make_api(tmp_path)
    _connect(api, sitl_connection)
    record: dict[str, Any] = {}
    try:
        plan, sent, _, started = _fly_into_survey(api, sitl_observer, sitl_trace, "corridor", 1)
        record.update(started)
        land_start_seq = next(int(i["seq"]) for i in sent if int(i["command"]) == DO_LAND_START)
        land_mark = sitl_trace.mark()
        land = api.vehicle_command("land")
        record["land"] = land
        assert land["ok"], land
        acks = sitl_trace.messages("COMMAND_ACK", land_mark)
        record["command_acks"] = [a.data for a in acks]
        assert any(int(a.data["command"]) == DO_LAND_START and int(a.data["result"]) == 0
                   for a in acks), record["command_acks"]
        jumped = sitl_trace.wait_for("MISSION_CURRENT",
                                     lambda d: int(d["seq"]) - 1 > land_start_seq, land_mark, 10.0)
        assert jumped is not None, "Land did not jump to the landing sequence"
        record["landed"] = stopped = _wait_landed(api, sitl_trace, land_mark, FLIGHT_TIMEOUT_S)
        landing = plan["fixed_wing"]["landing"]
        record["stopped_at_m_from_landing_point"] = round(distance_m(
            stopped["latitude"], stopped["longitude"], landing["land_lat"], landing["land_lon"]), 1)
        record["statustext"] = sitl_trace.statustext(land_mark)
        assert record["stopped_at_m_from_landing_point"] <= TOUCHDOWN_TOL_M, record
    finally:
        _write("land-command", record)
        _settle(api)
