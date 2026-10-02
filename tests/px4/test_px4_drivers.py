"""The app's PX4 path, flown against real PX4 v1.15.4 SITL -- both drivers.

Every test runs twice: through the MAVSDK driver (``mavsdk``, PX4's own SDK) and through
the pymavlink driver (``mavlink``), each on its own PX4 UDP link. The vehicle is a fresh
PX4 per test (tools/sitl/px4_docker.py). What PX4 actually did is read from the other
link by an independent observer, never from what the app says.

    pytest -m px4 tests/px4
"""

from __future__ import annotations

import math
import time

import pytest

from ._support import (PLAN_OPTIONS, PX4_LOITER, PX4_MISSION, PX4_RTL, connect, distance_m, fields_compared,
                       item_mismatches, make_api, plan, planned_items, telemetry, wait_telemetry,
                       when_armable, write_json)
from .conftest import evidence_dir

pytestmark = pytest.mark.px4


def _evidence(request, payload) -> None:
    name = request.node.name.replace("/", "_").replace("[", "-").replace("]", "")
    write_json(evidence_dir() / f"{name}.result.json", payload)


def _request_home(observer) -> dict:
    """HOME_POSITION on the observer's link; asked for if PX4 has not streamed it."""
    found = observer.wait_for("HOME_POSITION", timeout_s=3.0)
    for _ in range(5):
        if found is not None:
            return found.data
        # MAV_CMD_REQUEST_MESSAGE (512) for HOME_POSITION (242).
        observer.conn.mav.command_long_send(1, 1, 512, 0, 242, 0, 0, 0, 0, 0, 0)
        found = observer.wait_for("HOME_POSITION", timeout_s=2.0)
    pytest.fail("PX4 sent no HOME_POSITION on the observer's link.")


def _home_in_step(api, observer, timeout_s: float = 20.0) -> tuple[dict, dict]:
    """The app's telemetry and PX4's latest HOME_POSITION, once they agree (or at timeout)."""
    deadline = time.monotonic() + timeout_s
    while True:
        home = observer.latest("HOME_POSITION")
        t = telemetry(api)
        if home and t.get("home_set"):
            horizontal = distance_m(t["home_lat"], t["home_lon"],
                                    home["latitude"] / 1e7, home["longitude"] / 1e7)
            vertical = abs(float(t["home_alt_m"]) - home["altitude"] / 1000.0)
            if (horizontal < 0.5 and vertical < 0.5) or time.monotonic() > deadline:
                return t, home
        elif time.monotonic() > deadline:
            return t, home or {}
        time.sleep(0.5)


def _driver_mismatches(sent: list[dict], held: list[dict]) -> list[str]:
    """What the app's own driver downloads, against what it sent (both in planner form)."""
    from core.mission_planner_bridge import _NON_POSITIONAL_COMMANDS

    problems = []
    if len(sent) != len(held):
        problems.append(f"count: sent {len(sent)}, driver read back {len(held)}")
    for index, (a, b) in enumerate(zip(sent, held)):
        where = f"#{index} cmd {a['command']}"
        # The driver returns planner frames (no *_INT aliases), so frames compare exactly.
        for key in ("command", "frame"):
            if key not in fields_compared(a):
                continue
            if int(a.get(key, 0)) != int(b.get(key, -1)):
                problems.append(f"{where}: {key} {a.get(key)} -> {b.get(key)}")
        for key in ("param1", "param2", "param3", "param4", "alt"):
            want, got = float(a.get(key, 0.0)), float(b.get(key, 0.0))
            if not (math.isnan(want) and math.isnan(got)) and abs(want - got) > 1e-4:
                problems.append(f"{where}: {key} {want} -> {got}")
        tol = 0.5 if int(a["command"]) in _NON_POSITIONAL_COMMANDS else 1.5e-7
        for key in ("lat", "lon"):
            if abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) > tol:
                problems.append(f"{where}: {key} {a.get(key)} -> {b.get(key)}")
    return problems


# ── 1. Connect and telemetry ─────────────────────────────────────────────────

def test_connect_reports_px4_telemetry(px4, driver, observer, tmp_path, request):
    api = make_api(tmp_path)
    try:
        t = connect(api, driver)
        _request_home(observer)
        # PX4 refines home while it sits disarmed (the trace shows HOME_POSITION moving
        # by decimetres in the first ~20 s), so the app's home is compared with the
        # home PX4 is reporting at the same moment, not with a snapshot from earlier.
        t, home = _home_in_step(api, observer)
        pos = observer.latest("GLOBAL_POSITION_INT")
        gps = observer.latest("GPS_RAW_INT")
        hb = observer.latest("HEARTBEAT")
        battery = observer.latest("BATTERY_STATUS") or {}
        sys_status = observer.latest("SYS_STATUS") or {}
        px4_remaining = battery.get("battery_remaining", sys_status.get("battery_remaining", -1))

        checks = {
            "position_error_m": distance_m(t["latitude"], t["longitude"],
                                           pos["lat"] / 1e7, pos["lon"] / 1e7),
            "home_error_m": distance_m(t["home_lat"], t["home_lon"],
                                       home["latitude"] / 1e7, home["longitude"] / 1e7),
            "home_alt_error_m": abs(float(t["home_alt_m"]) - home["altitude"] / 1000.0),
            "app_gps_fix": t["gps_fix"], "px4_gps_fix": gps["fix_type"],
            "app_battery_pct": t["battery_pct"], "px4_battery_remaining": px4_remaining,
            "app_mode": t["flight_mode"], "px4_mode": hb["mode"],
            "app_armed": t["armed"], "px4_armed": hb["armed"],
        }
        _evidence(request, {"driver": driver[0], "telemetry": t, "checks": checks})

        assert checks["position_error_m"] < 2.0, checks
        assert checks["home_error_m"] < 0.5, checks
        assert checks["home_alt_error_m"] < 0.5, checks
        assert t["home_set"] is True
        assert t["gps_fix"] >= 3 and gps["fix_type"] >= 3, checks
        assert px4_remaining >= 0, "PX4 reported no battery remaining"
        assert abs(float(t["battery_pct"]) - float(px4_remaining)) <= 2.0, checks
        assert t["armed"] is False and hb["armed"] is False
        # PX4's idle mode, in the toolkit's vocabulary on both drivers.
        expected = {"LOITER": "LOITER", "MISSION": "AUTO", "RTL": "RTL"}.get(hb["mode"], hb["mode"])
        if driver[0] == "mavsdk" and hb["mode"] == "READY":
            expected = "READY"
        assert t["flight_mode"] == expected, checks
    finally:
        api.disconnect_vehicle()


# ── 2. Mission upload and download fidelity ──────────────────────────────────

@pytest.mark.parametrize("template", ["grid", "orbit"])
def test_mission_upload_matches_what_px4_holds(px4, driver, observer, tmp_path, request,
                                               template):
    api = make_api(tmp_path)
    try:
        connect(api, driver)
        planned = plan(api, template)
        uploaded = api.upload_mission()
        assert uploaded["ok"], uploaded
        sent, fence = planned_items(api)
        assert sent, "the planner produced no items"

        stored = observer.download(0)
        stored_fence = observer.download(1) if fence else []
        client = api._session._drone_client
        read_back = client.download_mission()
        read_fence = client.download_geofence() if fence else []

        problems = {
            "mission_vs_px4": item_mismatches(sent, stored),
            "fence_vs_px4": item_mismatches(fence, stored_fence) if fence else [],
            "mission_vs_driver_download": _driver_mismatches(sent, read_back),
            "fence_vs_driver_download": _driver_mismatches(fence, read_fence) if fence else [],
        }
        t = telemetry(api)
        _evidence(request, {
            "driver": driver[0], "template": template, "upload": uploaded,
            "items_sent": len(sent), "items_px4": len(stored), "fence_sent": len(fence),
            "fence_px4": len(stored_fence), "commands": sorted({int(i["command"]) for i in sent}),
            "first_items_px4": stored[:4], "problems": problems,
            "app_waypoint_total": t.get("waypoint_total"),
            "plan_summary": {k: planned.get(k) for k in ("ok", "summary") if k in planned},
        })
        assert not any(problems.values()), problems
        # PX4 keeps no home slot: the plan's first item is PX4's seq 0.
        assert int(stored[0]["command"]) == int(sent[0]["command"])
        assert t.get("waypoint_total") == len(sent), t
    finally:
        api.disconnect_vehicle()


# ── 3. Arm, take off, land ───────────────────────────────────────────────────

def test_arm_takeoff_land(px4, driver, observer, tmp_path, request):
    api = make_api(tmp_path)
    log: dict = {"driver": driver[0]}
    try:
        connect(api, driver)
        armed, refusals = when_armable(lambda: api.vehicle_command("arm"), observer)
        log.update(arm=armed, arm_refusals=refusals)
        assert armed.get("ok"), log
        assert observer.wait_for("HEARTBEAT", lambda d: d["armed"], timeout_s=5.0), \
            "the app said armed; PX4's heartbeat never did"

        mark = observer.mark()
        took = api.vehicle_command("takeoff", {"altitude_m": 10.0})
        log["takeoff"] = took
        if not took.get("ok") and not observer.armed():
            # PX4 auto-disarms an idle armed vehicle; arm again and retry once.
            armed, more = when_armable(lambda: api.vehicle_command("arm"), observer)
            log["arm_refusals"] += more
            took = api.vehicle_command("takeoff", {"altitude_m": 10.0})
            log["takeoff_retry"] = took
        assert took.get("ok"), log
        climbed = observer.wait_for("GLOBAL_POSITION_INT",
                                    lambda d: d["relative_alt"] / 1000.0 >= 9.0,
                                    since=mark, timeout_s=45.0)
        assert climbed is not None, "PX4 never climbed to 9 m after the app's takeoff"
        log["px4_alt_reached_m"] = climbed.data["relative_alt"] / 1000.0
        modes = sorted({e.data["mode"] for e in observer.messages("HEARTBEAT", mark)})
        log["px4_modes_during_takeoff"] = modes
        assert "TAKEOFF" in modes, modes
        app_alt = wait_telemetry(api, "the app's altitude to show the climb",
                                 lambda t: float(t.get("altitude_rel_m") or 0.0) >= 8.0, 10.0)
        log["app_alt_m"] = app_alt["altitude_rel_m"]
        time.sleep(3.0)

        mark = observer.mark()
        landed = api.vehicle_command("land")
        log["land"] = landed
        assert landed.get("ok"), log
        assert observer.wait_for("HEARTBEAT", lambda d: d["mode"] == "LAND",
                                 since=mark, timeout_s=5.0), "PX4 never entered LAND"
        down = observer.wait_for("HEARTBEAT", lambda d: not d["armed"], since=mark,
                                 timeout_s=60.0)
        assert down is not None, "PX4 did not land and disarm within 60 s"
        final = observer.latest("GLOBAL_POSITION_INT")
        log["px4_final_rel_alt_m"] = final["relative_alt"] / 1000.0
        assert abs(final["relative_alt"] / 1000.0) < 1.0, log
        wait_telemetry(api, "the app to show the vehicle disarmed",
                       lambda t: t.get("armed") is False, 10.0)
    finally:
        log["statustext"] = observer.statustext()
        _evidence(request, log)
        api.disconnect_vehicle()


# ── 4. A planned mission flown: start, pause, resume, RTL ────────────────────

def _current_seq(observer, since: float = 0.0) -> int:
    found = observer.messages("MISSION_CURRENT", since)
    return int(found[-1].data["seq"]) if found else -1


@pytest.mark.parametrize("template", ["grid", "orbit"])
def test_mission_flight_pause_resume_rtl(px4, driver, observer, tmp_path, request, template):
    api = make_api(tmp_path)
    log: dict = {"driver": driver[0], "template": template}
    try:
        connect(api, driver)
        home = _request_home(observer)
        home_lat, home_lon = home["latitude"] / 1e7, home["longitude"] / 1e7
        plan(api, template)
        uploaded = api.upload_mission()
        assert uploaded["ok"], uploaded
        total = len(planned_items(api)[0])
        log["items"] = total
        # The plan's return altitude is on the vehicle, read back from PX4 itself.
        rth_planned = PLAN_OPTIONS["rth_altitude_m"]
        log["return_altitude_upload"] = uploaded["upload"].get("return_altitude")
        log["px4_RTL_RETURN_ALT"] = observer.param("RTL_RETURN_ALT")
        assert log["px4_RTL_RETURN_ALT"] == pytest.approx(rth_planned), log

        # Start: the pilot overrides the checklist's manual items (camera, weather, ...)
        # that a simulator cannot confirm; PX4's own arming checks still apply.
        mark = observer.mark()
        started, refusals = when_armable(
            lambda: api.vehicle_command("start", {"override_preflight": True}), observer)
        log.update(start=started, start_refusals=refusals)
        assert started.get("ok"), log
        flying = observer.wait_for("HEARTBEAT", lambda d: d["armed"] and d["mode"] == PX4_MISSION,
                                   since=mark, timeout_s=15.0)
        assert flying is not None, "PX4 was not armed in MISSION after Start"

        # Progress: PX4 works through the plan, and the app shows the same item.
        progressed = observer.wait_for("MISSION_CURRENT", lambda d: int(d["seq"]) >= 4,
                                       since=mark, timeout_s=120.0)
        assert progressed is not None, f"PX4 never reached item 4: {observer.statustext(mark)}"
        reached = [int(e.data["seq"]) for e in observer.messages("MISSION_ITEM_REACHED", mark)]
        log["px4_items_reached_before_pause"] = reached
        time.sleep(1.0)
        px4_seq = _current_seq(observer)
        app = telemetry(api)
        log["progress_check"] = {"px4_seq": px4_seq, "app_index": app.get("waypoint_index"),
                                 "app_total": app.get("waypoint_total")}
        assert abs(int(app.get("waypoint_index") or 0) - px4_seq) <= 1, log["progress_check"]
        assert app.get("waypoint_total") == total, log["progress_check"]
        assert app.get("flight_mode") == "AUTO" and app.get("armed") is True, app

        # Pause: PX4 holds position, mid-mission.
        mark = observer.mark()
        paused = api.vehicle_command("pause")
        log["pause"] = paused
        assert paused.get("ok"), log
        assert observer.wait_for("HEARTBEAT", lambda d: d["mode"] == PX4_LOITER, since=mark,
                                 timeout_s=5.0), "PX4 did not enter LOITER on pause"
        time.sleep(6.0)
        held_seq = _current_seq(observer)
        before = observer.latest("GLOBAL_POSITION_INT")
        time.sleep(4.0)
        after = observer.latest("GLOBAL_POSITION_INT")
        drift = distance_m(before["lat"] / 1e7, before["lon"] / 1e7,
                           after["lat"] / 1e7, after["lon"] / 1e7)
        speed = observer.latest("VFR_HUD")["groundspeed"]
        log["paused"] = {"drift_4s_m": drift, "groundspeed": speed, "seq": held_seq,
                         "seq_after": _current_seq(observer), "app_mode": telemetry(api)["flight_mode"]}
        assert drift < 2.0 and speed < 1.0, log["paused"]
        assert _current_seq(observer) == held_seq, log["paused"]
        assert observer.armed()
        assert telemetry(api)["flight_mode"] == "LOITER"

        # Resume: back to MISSION, continuing from where it stopped (not from item 0).
        mark = observer.mark()
        resumed = api.vehicle_command("resume")
        log["resume"] = resumed
        assert resumed.get("ok"), log
        assert observer.wait_for("HEARTBEAT", lambda d: d["mode"] == PX4_MISSION, since=mark,
                                 timeout_s=5.0), "PX4 did not re-enter MISSION on resume"
        onward = observer.wait_for("MISSION_CURRENT", lambda d: int(d["seq"]) > held_seq,
                                   since=mark, timeout_s=90.0)
        log["resumed"] = {"from_seq": held_seq,
                          "next_seq": onward.data["seq"] if onward else None,
                          "min_seq_after_resume": min(
                              (int(e.data["seq"]) for e in observer.messages("MISSION_CURRENT", mark)),
                              default=None)}
        assert onward is not None, log["resumed"]
        assert log["resumed"]["min_seq_after_resume"] >= held_seq, log["resumed"]

        # RTL: PX4 flies home, lands and disarms there. Home is where PX4 put it when it
        # armed (it re-sets home on arming), as it reports it now.
        home = observer.latest("HOME_POSITION") or home
        home_lat, home_lon = home["latitude"] / 1e7, home["longitude"] / 1e7
        log["home_at_rtl"] = {"lat": home_lat, "lon": home_lon}
        mark = observer.mark()
        rtl = api.vehicle_command("rtl")
        log["rtl"] = rtl
        assert rtl.get("ok"), log
        assert observer.wait_for("HEARTBEAT", lambda d: d["mode"] == PX4_RTL, since=mark,
                                 timeout_s=5.0), "PX4 did not enter RTL"
        down = observer.wait_for("HEARTBEAT", lambda d: not d["armed"], since=mark,
                                 timeout_s=150.0)
        assert down is not None, f"PX4 did not land after RTL: {observer.statustext(mark)}"
        final = observer.latest("GLOBAL_POSITION_INT")
        landed_from_home = distance_m(final["lat"] / 1e7, final["lon"] / 1e7, home_lat, home_lon)
        peak = max(e.data["relative_alt"] for e in observer.messages("GLOBAL_POSITION_INT", mark)) / 1000.0
        log["rtl_result"] = {"landed_from_home_m": landed_from_home,
                             "final_rel_alt_m": final["relative_alt"] / 1000.0,
                             "peak_rel_alt_during_rtl_m": peak,
                             "rtl_seconds": round(down.t - mark, 1)}
        assert landed_from_home < 5.0, log["rtl_result"]
        # PX4 returned at its own default (30 m here) before the plan's RTH was set.
        assert peak <= rth_planned + 1.5, log["rtl_result"]
        assert abs(final["relative_alt"] / 1000.0) < 1.0, log["rtl_result"]
        wait_telemetry(api, "the app to show the vehicle disarmed",
                       lambda t: t.get("armed") is False, 10.0)
    finally:
        log["statustext"] = observer.statustext()
        _evidence(request, log)
        api.disconnect_vehicle()
