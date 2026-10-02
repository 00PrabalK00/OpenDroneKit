"""Every vehicle command the app offers, sent through app.api.Api to ArduCopter.

Each command is judged twice: by what the Api tells the operator (ok / the vehicle's
ack) and by what ArduPilot then does, read from the independent MAVLink trace and the
vehicle's own telemetry -- the mode it reports, whether it is moving, CAMERA_FEEDBACK
for a shutter, MISSION_ITEM_REACHED for progress. Commands the vehicle refuses must
surface as refusals with the vehicle's answer, never as "sent".

The results are written to the evidence folder as a command x result matrix.
"""

from __future__ import annotations

import json
import time

import pytest

from ._support import (
    connect_api,
    distance_m,
    evidence_dir,
    make_api,
    plan_template,
    settle_on_ground,
    wait_telemetry,
)

pytestmark = pytest.mark.sitl

FLIGHT_MODES_IN_MISSION = {"AUTO"}


class Matrix:
    """command x result, as the operator saw it and as the vehicle behaved."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[dict] = []

    def add(self, command: str, response: dict, expected: str, observed: str,
            passed: bool) -> None:
        result = response.get("result") or {}
        self.rows.append({
            "command": command,
            "api_ok": response.get("ok"),
            "ack": result.get("ack", ""),
            "message": result.get("message") or response.get("error") or response.get("message", ""),
            "expected": expected,
            "observed": observed,
            "pass": bool(passed),
        })

    def write(self) -> None:
        (evidence_dir("commands") / f"{self.name}.json").write_text(
            json.dumps(self.rows, indent=1, default=str), encoding="utf-8")


def _mode(api) -> str:
    return str(api.telemetry()["telemetry"].get("flight_mode", "")).upper()


def _still(api, seconds: float = 2.0, limit_mps: float = 0.4) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if float(api.telemetry()["telemetry"].get("speed_mps") or 0.0) > limit_mps:
            return False
        time.sleep(0.2)
    return True


def _settles(api, within_s: float = 15.0) -> bool:
    """The aircraft comes to a stop (and stays stopped for 2 s) within `within_s`."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if _still(api, 2.0):
            return True
    return False


def _clear_vehicle_mission(api) -> None:
    """Harness set-up: empty the autopilot's mission store (MISSION_CLEAR_ALL)."""
    conn = api._session._drone_client._conn
    conn.mav.mission_clear_all_send(conn.target_system, conn.target_component, 0)
    time.sleep(1.0)


@pytest.fixture
def api(tmp_path, sitl_connection):
    api = make_api(tmp_path)
    connect_api(api, sitl_connection)
    try:
        yield api
    finally:
        try:
            settle_on_ground(api)
        finally:
            api.disconnect_vehicle()


def test_refused_commands_surface_the_vehicles_refusal(api, sitl_trace):
    matrix = Matrix("refusals_on_the_ground")
    try:
        _clear_vehicle_mission(api)
        assert api.telemetry()["telemetry"]["armed"] is False

        # Start with nothing on the vehicle and the checklist not passed: the app holds it.
        held = api.vehicle_command("start")
        matrix.add("start (preflight not passed)", held, "app refuses, names checks",
                   held.get("error", ""), held["ok"] is False and held.get("preflight_blocked"))
        assert held["ok"] is False and held.get("preflight_blocked"), held

        # The pilot overrides the checklist; ArduPilot itself refuses AUTO with no mission.
        mark = sitl_trace.mark()
        forced = api.vehicle_command("start", override_preflight=True)
        acks = [e.data for e in sitl_trace.messages("COMMAND_ACK", mark)]
        texts = sitl_trace.statustext(mark)
        time.sleep(1.0)
        armed_after = bool(api.telemetry()["telemetry"]["armed"])
        matrix.add("start (override, no mission on vehicle)", forced,
                   "vehicle refuses MISSION_START; aircraft not left armed",
                   f"acks={[(a['command'], a['result']) for a in acks]} text={texts} "
                   f"armed_after={armed_after}",
                   forced["ok"] is False and not armed_after)
        assert forced["ok"] is False, forced
        assert any(a["command"] == 300 and a["result"] != 0 for a in acks), acks
        assert forced["result"]["ack"] == "denied", forced
        assert not armed_after, "a refused start left the aircraft armed on the ground"

        for command in ("pause", "resume"):
            mark = sitl_trace.mark()
            response = api.vehicle_command(command)
            acks = [(e.data["command"], e.data["result"]) for e in sitl_trace.messages("COMMAND_ACK", mark)]
            matrix.add(f"{command} (no mission running)", response, "vehicle refuses",
                       f"acks={acks} text={sitl_trace.statustext(mark)}", response["ok"] is False)
            assert response["ok"] is False and response["result"]["ack"] == "denied", response

        mark = sitl_trace.mark()
        lifted = api.vehicle_command("takeoff", altitude_m=10.0)
        acks = [(e.data["command"], e.data["result"]) for e in sitl_trace.messages("COMMAND_ACK", mark)]
        matrix.add("takeoff (disarmed)", lifted, "vehicle refuses NAV_TAKEOFF",
                   f"acks={acks}", lifted["ok"] is False)
        assert lifted["ok"] is False and lifted["result"]["ack"] == "denied", lifted
        assert api.telemetry()["telemetry"]["armed"] is False

        mark = sitl_trace.mark()
        video = api.camera_command("start_video")
        matrix.add("camera start_video (servo trigger has no video)", video,
                   "vehicle refuses", f"acks={[(e.data['command'], e.data['result']) for e in sitl_trace.messages('COMMAND_ACK', mark)]}",
                   video["ok"] is False)
        assert video["ok"] is False, video
        exposure = api.camera_command("set_exposure_setting", setting="iso", value=100)
        matrix.add("camera set_exposure_setting", exposure, "app refuses (no MAVLink path)",
                   exposure.get("error", ""), exposure["ok"] is False)
        assert exposure["ok"] is False
    finally:
        matrix.write()


def test_every_command_in_flight(api, sitl_trace):
    """arm, disarm, takeoff, start, pause, resume, photo, abort, manual hand-back,
    resume autonomy, loiter, RTL, land, set home, emergency stop."""
    matrix = Matrix("in_flight")
    try:
        planned = plan_template(api, "grid")
        assert planned["ok"], planned
        upload = api.upload_mission()
        assert upload["ok"], upload

        held = api.vehicle_command("loiter")
        matrix.add("loiter (on ground)", held, "LOITER", _mode(api), held["ok"] and _mode(api) == "LOITER")
        assert held["ok"], held

        for command, want in (("arm", True), ("disarm", False), ("arm", True)):
            mark = sitl_trace.mark()
            response = api.vehicle_command(command)
            ack = sitl_trace.wait_for("COMMAND_ACK", lambda d: d["command"] == 400, mark, 5.0)
            telemetry = wait_telemetry(api, f"armed={want}", lambda t, w=want: t["armed"] is w, 15.0)
            matrix.add(command, response, f"armed={want}",
                       f"armed={telemetry['armed']} ack={ack.data['result'] if ack else None}",
                       response["ok"] and telemetry["armed"] is want)
            assert response["ok"] and response["result"]["ack"] == "accepted", response

        lifted = api.vehicle_command("takeoff", altitude_m=12.0)
        telemetry = wait_telemetry(api, "takeoff to 12 m", lambda t: t["altitude_rel_m"] >= 10.5, 90.0)
        matrix.add("takeoff 12 m", lifted, "GUIDED climb to 12 m",
                   f"mode={telemetry['flight_mode']} alt={telemetry['altitude_rel_m']}",
                   lifted["ok"] and telemetry["altitude_rel_m"] >= 10.5)
        assert lifted["ok"], lifted

        preflight = api.run_preflight()
        confirmed = api.run_preflight(acknowledged=preflight["pending_confirmation"])
        assert confirmed["can_start"], [(c["id"], c["message"]) for c in confirmed["checks"] if not c["ok"]]
        mark = sitl_trace.mark()
        started = api.vehicle_command("start")
        wait_telemetry(api, "AUTO", lambda t: t["flight_mode"] == "AUTO", 20.0)
        reached = sitl_trace.wait_for("MISSION_ITEM_REACHED", lambda d: d["seq"] >= 3, mark, 120.0)
        matrix.add("start", started, "AUTO and waypoints reached",
                   f"mode={_mode(api)} reached={reached.data if reached else None}",
                   started["ok"] and reached is not None)
        assert started["ok"] and reached is not None, started

        # Fly into a leg, then pause: AUTO holds position without leaving the mission.
        wait_telemetry(api, "moving on a leg", lambda t: (t.get("speed_mps") or 0) > 2.0, 60.0)
        current_before = api.telemetry()["telemetry"].get("waypoint_index")
        mark = sitl_trace.mark()
        paused = api.vehicle_command("pause")
        stopped = _settles(api)
        matrix.add("pause", paused, "stops, stays in AUTO",
                   f"mode={_mode(api)} still={stopped} text={sitl_trace.statustext(mark)}",
                   paused["ok"] and stopped and _mode(api) == "AUTO")
        assert paused["ok"], paused
        assert stopped and _mode(api) == "AUTO"
        assert api.telemetry()["telemetry"].get("waypoint_index") == current_before

        mark = sitl_trace.mark()
        resumed = api.vehicle_command("resume")
        moving = wait_telemetry(api, "moving again", lambda t: (t.get("speed_mps") or 0) > 1.0, 30.0)
        next_reached = sitl_trace.wait_for("MISSION_ITEM_REACHED", since=mark, timeout_s=90.0)
        matrix.add("resume", resumed, "continues the mission",
                   f"speed={moving['speed_mps']} reached={next_reached.data if next_reached else None}",
                   resumed["ok"] and next_reached is not None)
        assert resumed["ok"] and next_reached is not None, resumed

        mark = sitl_trace.mark()
        photo = api.camera_command("take_photo")
        feedback = sitl_trace.wait_for("CAMERA_FEEDBACK", since=mark, timeout_s=5.0)
        matrix.add("camera take_photo", photo, "CAMERA_FEEDBACK",
                   f"feedback={feedback.data if feedback else None}",
                   photo["ok"] and feedback is not None)
        assert photo["ok"] and feedback is not None, photo
        for action, kwargs in (("stop_photo_sequence", {}), ("set_zoom", {"level_pct": 50.0}),
                               ("set_focus", {"level_pct": 50.0}), ("set_mode", {"mode": "photo"})):
            mark = sitl_trace.mark()
            response = api.camera_command(action, **kwargs)
            acks = [(e.data["command"], e.data["result"]) for e in sitl_trace.messages("COMMAND_ACK", mark)]
            # Recorded as the vehicle answered; a servo trigger has no zoom or focus.
            matrix.add(f"camera {action}", response, "vehicle's answer surfaced",
                       f"acks={acks}", bool(acks) or not response["ok"])

        aborted = api.vehicle_command("abort")
        stopped = _settles(api)
        matrix.add("abort", aborted, "BRAKE, stopped", f"mode={_mode(api)} still={stopped}",
                   aborted["ok"] and _mode(api) == "BRAKE" and stopped)
        assert aborted["ok"] and _mode(api) == "BRAKE", aborted

        manual = api.take_manual_control()
        state = api.control_state()["state"]
        matrix.add("take_manual_control", manual, "LOITER, pilot has control",
                   f"mode={state['mode']} pilot={state['pilot_has_control']}",
                   manual["ok"] and state["pilot_has_control"])
        assert manual["ok"] and state["pilot_has_control"] and state["mode"] == "LOITER", manual

        # Hand autonomy back: start again resumes the interrupted mission.
        mark = sitl_trace.mark()
        again = api.vehicle_command("start")
        wait_telemetry(api, "AUTO again", lambda t: t["flight_mode"] == "AUTO", 20.0)
        progressed = sitl_trace.wait_for("MISSION_ITEM_REACHED", since=mark, timeout_s=90.0)
        matrix.add("start (hand autonomy back)", again, "AUTO resumes",
                   f"mode={_mode(api)} reached={progressed.data if progressed else None}",
                   again["ok"] and progressed is not None)
        assert again["ok"] and progressed is not None, again

        # Fly on until the aircraft is well away from the launch point, so landing, a
        # new home and RTL to it are distinguishable from simply staying at home.
        original_home = api.telemetry()["telemetry"]
        home_lat, home_lon = original_home["home_lat"], original_home["home_lon"]
        wait_telemetry(api, "the grid's far side", lambda t: distance_m(
            t["latitude"], t["longitude"], home_lat, home_lon) > 25.0, 180.0)

        loiter = api.vehicle_command("loiter")
        matrix.add("loiter (in flight)", loiter, "LOITER", _mode(api), loiter["ok"] and _mode(api) == "LOITER")
        assert loiter["ok"] and _mode(api) == "LOITER"

        landing_from = api.telemetry()["telemetry"]
        land = api.vehicle_command("land")
        wait_telemetry(api, "LAND", lambda t: t["flight_mode"] == "LAND", 10.0)
        down = wait_telemetry(api, "landed and disarmed", lambda t: not t["armed"], 240.0)
        moved = distance_m(landing_from["latitude"], landing_from["longitude"],
                           down["latitude"], down["longitude"])
        matrix.add("land", land, "LAND where it was, disarm",
                   f"armed={down['armed']} drift={moved:.1f} m", land["ok"] and moved < 5.0)
        assert land["ok"] and moved < 5.0, (land, moved)
        away = distance_m(down["latitude"], down["longitude"], home_lat, home_lon)
        assert away > 20.0, f"landed only {away:.1f} m from home"

        mark = sitl_trace.mark()
        home = api.vehicle_command("set_home")
        new_home = sitl_trace.wait_for("HOME_POSITION", since=mark, timeout_s=10.0)
        shown = wait_telemetry(api, "the app showing the new home", lambda t: distance_m(
            t.get("home_lat", 0.0), t.get("home_lon", 0.0),
            down["latitude"], down["longitude"]) < 3.0, 15.0)
        moved_home = None if new_home is None else distance_m(
            new_home.data["latitude"] / 1e7, new_home.data["longitude"] / 1e7,
            down["latitude"], down["longitude"])
        matrix.add("set_home (current position)", home, "home moves to where it landed",
                   f"vehicle HOME_POSITION {moved_home} m from aircraft, {away:.1f} m from "
                   f"old home; app shows {shown['home_lat']}, {shown['home_lon']}",
                   home["ok"] and moved_home is not None and moved_home < 3.0)
        assert home["ok"] and moved_home is not None and moved_home < 3.0, (home, new_home)

        # RTL now has to come back here, not to the launch point.
        api.vehicle_command("loiter")
        assert api.vehicle_command("arm")["ok"]
        assert api.vehicle_command("takeoff", altitude_m=10.0)["ok"]
        wait_telemetry(api, "climb", lambda t: t["altitude_rel_m"] >= 8.5, 90.0)
        api._session._drone_client.goto(home_lat, home_lon, 10.0)  # fly back over the old home
        wait_telemetry(api, "over the old home", lambda t: distance_m(
            t["latitude"], t["longitude"], home_lat, home_lon) < 5.0, 120.0)
        rtl = api.vehicle_command("rtl")
        wait_telemetry(api, "RTL", lambda t: t["flight_mode"] == "RTL", 10.0)
        returned = wait_telemetry(api, "RTL landing and disarm", lambda t: not t["armed"], 300.0)
        error = distance_m(returned["latitude"], returned["longitude"],
                           down["latitude"], down["longitude"])
        matrix.add("rtl", rtl, "returns to the new home and lands",
                   f"landed {error:.1f} m from the new home", rtl["ok"] and error < 5.0)
        assert rtl["ok"] and error < 5.0, (rtl, error)

        api.vehicle_command("loiter")
        assert api.vehicle_command("arm")["ok"]
        wait_telemetry(api, "armed", lambda t: t["armed"], 10.0)
        stop = api.vehicle_command("emergency_stop")
        stopped = wait_telemetry(api, "motors cut", lambda t: not t["armed"], 10.0)
        matrix.add("emergency_stop (armed on ground)", stop, "disarmed immediately",
                   f"armed={stopped['armed']}", stop["ok"] and not stopped["armed"])
        assert stop["ok"] and not stopped["armed"], stop

        session = api.flight_session()["session"]
        logged = {c["command"] for c in session["commands"]}
        assert {"start_mission", "pause_mission", "resume_mission", "abort_mission",
                "return_to_home"} <= logged, logged
    finally:
        matrix.write()
