"""Failsafes, against ArduCopter's own failsafe logic, with the app connected through
its production driver.

Each fault is injected into the *environment* -- the telemetry link cut, the simulated
battery sagging, the simulated GPS receiver switched off, the aircraft pushed across its
fence -- over the independent observer link or the link relay, never through the app
under test. Each test then checks two things:

* what the vehicle did, from the independent MAVLink trace (its STATUSTEXT, mode,
  FENCE_STATUS, BATTERY_STATUS, GPS_RAW_INT), and
* what the app told the operator, from ``Api.telemetry()`` and ``Api.flight_session()``.

Each test restores the parameters it changed and leaves the vehicle landed and
disarmed, so the next one starts from a clean aircraft.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from ._support import (
    LinkCutter,
    connect_api,
    distance_m,
    evidence_dir,
    fresh_battery,
    make_api,
    offset_lonlat,
    plan_template,
    read_vehicle_param,
    set_vehicle_param,
    settle_on_ground,
    start_mission_from_ground,
    wait_armable,
    wait_telemetry,
)

pytestmark = pytest.mark.sitl

# ArduCopter custom_mode numbers, as the vehicle reports them in HEARTBEAT.
COPTER_MODES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER",
                6: "RTL", 9: "LAND", 17: "BRAKE", 21: "SMART_RTL"}


def _mode_entry(trace, names: set[str], since: float, timeout_s: float):
    return trace.wait_for("HEARTBEAT", lambda d: COPTER_MODES.get(int(d["custom_mode"])) in names,
                          since, timeout_s)


def _modes_seen(trace, since: float) -> list[str]:
    seen: list[str] = []
    for entry in trace.messages("HEARTBEAT", since):
        name = COPTER_MODES.get(int(entry.data["custom_mode"]), str(entry.data["custom_mode"]))
        if not seen or seen[-1] != name:
            seen.append(name)
    return seen


def _write(name: str, record: dict) -> None:
    (evidence_dir("failsafes") / f"{name}.json").write_text(
        json.dumps(record, indent=1, default=str), encoding="utf-8")


def _poll(fn, predicate, timeout_s: float, interval_s: float = 0.2):
    """(value, seconds taken) once predicate(value) holds, else (last value, None)."""
    start = time.monotonic()
    value = fn()
    while not predicate(value):
        if time.monotonic() - start > timeout_s:
            return value, None
        time.sleep(interval_s)
        value = fn()
    return value, time.monotonic() - start


def _alerts(api) -> dict[str, str]:
    payload = api.telemetry()["telemetry"]
    return {a["kind"]: a["message"] for a in payload.get("alerts", [])}


def _wait_reached(trace, since: float, seq: int = 2, timeout_s: float = 120.0):
    reached = trace.wait_for("MISSION_ITEM_REACHED", lambda d: d["seq"] >= seq, since, timeout_s)
    assert reached is not None, f"the mission never reached item {seq}"
    return reached


def _land_and_settle(api, timeout_s: float = 300.0) -> dict:
    return wait_telemetry(api, "landed and disarmed", lambda t: not t["armed"], timeout_s)


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


# ── 1. Link loss mid-mission ────────────────────────────────────────────────

def test_link_loss_mid_mission_gcs_failsafe(tmp_path, sitl_connection, sitl_observer, sitl_trace):
    """Cut the app's link in AUTO: ArduPilot's GCS failsafe flies RTL; the app reports
    the link lost within its threshold, recovers when the link returns, and the flight
    log marks the gap."""
    from core.drone import TELEMETRY_STALE_AFTER_S

    cutter = LinkCutter(sitl_connection)
    api = make_api(tmp_path)
    record: dict = {}
    fs_before = read_vehicle_param(sitl_observer, sitl_trace, "FS_GCS_ENABLE")
    record["FS_GCS_ENABLE_before"] = fs_before
    try:
        connect_api(api, cutter.app_uri)
        record["param_set"] = set_vehicle_param(sitl_observer, sitl_trace, "FS_GCS_ENABLE", 1)  # RTL
        mark = sitl_trace.mark()
        start_mission_from_ground(api)
        reached = _wait_reached(sitl_trace, mark)
        # The app's heartbeat is all that keeps the GCS failsafe away: none so far.
        assert not any("gcs" in t.lower() and "failsafe" in t.lower()
                       for t in sitl_trace.statustext(mark)), sitl_trace.statustext(mark)
        record["in_auto_reached"] = reached.data
        record["session_status_before"] = api.flight_session()["session"]["state"]["status"]

        cut_mark = sitl_trace.mark()
        cut_wall = time.time()
        cutter.cut = True

        # App: link lost, reported within its threshold.
        lost, detect_s = _poll(lambda: api.telemetry()["telemetry"],
                               lambda t: t.get("link_lost") is True, TELEMETRY_STALE_AFTER_S + 5.0)
        record["app_link_lost_after_s"] = detect_s
        record["app_while_cut"] = {k: lost.get(k) for k in
                                   ("connected", "stale", "link_lost", "reason", "alerts",
                                    "telemetry_age_s", "flight_mode")}
        assert detect_s is not None, lost
        assert detect_s <= TELEMETRY_STALE_AFTER_S + 2.0, detect_s
        assert lost["connected"] is False and "Link lost" in lost["reason"]
        assert [a["kind"] for a in lost["alerts"]] == ["link_lost"]
        session, _ = _poll(lambda: api.flight_session()["session"]["state"],
                           lambda s: s["status"] == "link_lost", 5.0)
        record["session_while_cut"] = {k: session[k] for k in ("status", "warnings", "link_gaps")}
        assert session["status"] == "link_lost", session

        # Vehicle: GCS failsafe, RTL -- seen only on the independent observer link.
        failsafe_text = sitl_trace.wait_for(
            "STATUSTEXT", lambda d: "gcs" in d["text"].lower() and "failsafe" in d["text"].lower(),
            cut_mark, 20.0)
        rtl = _mode_entry(sitl_trace, {"RTL"}, cut_mark, 20.0)
        record["vehicle_failsafe_text"] = failsafe_text.data if failsafe_text else None
        record["vehicle_rtl_after_s"] = None if rtl is None else round(rtl.t - cut_mark, 2)
        record["vehicle_statustext_while_cut"] = sitl_trace.statustext(cut_mark)
        assert failsafe_text is not None, sitl_trace.statustext(cut_mark)
        assert rtl is not None, _modes_seen(sitl_trace, cut_mark)
        # The app heard none of this: everything it showed was labelled as old.
        assert api.telemetry()["telemetry"]["link_lost"] is True
        time.sleep(3.0)

        # Restore the link.
        cutter.cut = False
        outage_s = time.time() - cut_wall
        record["outage_s"] = round(outage_s, 1)
        record["relay_dropped"] = dict(cutter.dropped)
        restore_mark = sitl_trace.mark()
        back, recover_s = _poll(lambda: api.telemetry()["telemetry"],
                                lambda t: t.get("stale") is False, 10.0)
        record["app_recovered_after_s"] = recover_s
        assert recover_s is not None, back
        # Every snapshot from reconnection on: the pre-gap "AUTO" must never be shown as
        # live while the aircraft is flying RTL. The mode is unknown until the vehicle's
        # next HEARTBEAT says what it is.
        shown_modes: list[str] = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            shown = api.telemetry()["telemetry"]
            if not shown.get("stale") and (not shown_modes or shown_modes[-1] != shown["flight_mode"]):
                shown_modes.append(shown["flight_mode"])
            time.sleep(0.05)
        record["app_modes_after_restore"] = shown_modes
        assert "AUTO" not in shown_modes, shown_modes
        back = api.telemetry()["telemetry"]
        record["app_after_restore"] = {k: back.get(k) for k in
                                       ("connected", "stale", "flight_mode", "failsafe",
                                        "system_status", "alerts", "vehicle_messages")}
        assert back["connected"] is True and back["flight_mode"] in ("RTL", "LAND"), back
        # The vehicle hears the app's heartbeat again and clears its failsafe.
        cleared = sitl_trace.wait_for("STATUSTEXT", lambda d: "gcs" in d["text"].lower()
                                      and "clear" in d["text"].lower(), restore_mark, 10.0)
        record["vehicle_failsafe_cleared"] = cleared.data if cleared else None
        assert cleared is not None, sitl_trace.statustext(restore_mark)
        session, _ = _poll(lambda: api.flight_session()["session"]["state"],
                           lambda s: s["status"] != "link_lost", 5.0)
        record["session_after_restore"] = {k: session[k] for k in ("status", "link_gaps")}
        assert session["status"] != "link_lost", session
        gap = session["link_gaps"][-1]
        assert gap["duration_s"] >= outage_s - TELEMETRY_STALE_AFTER_S - 1.0, (gap, outage_s)
        assert gap["mode_after"] in ("RTL", "LAND"), gap

        landed = _land_and_settle(api)
        record["landed_from_home_m"] = round(distance_m(
            landed["latitude"], landed["longitude"], landed["home_lat"], landed["home_lon"]), 1)
        assert record["landed_from_home_m"] < 5.0, record

        # The flight log: the gap is marked, in the exported record and in the session log.
        exported = api.export_flight_log(str(tmp_path / "flightlog"), "json")
        assert exported["ok"], exported
        gaps = exported["summary"]["link_gaps"]
        record["flight_log_gaps"] = gaps
        assert any(g["duration_s"] >= outage_s - TELEMETRY_STALE_AFTER_S - 1.0 for g in gaps), gaps
        log_dir = Path(api.flight_session()["session"]["log_dir"])
        events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text().splitlines()]
        record["session_events"] = events
        assert [e["event"] for e in events][-2:] == ["link_lost", "link_restored"], events
    finally:
        cutter.cut = False
        try:
            # ArduCopter skips its GCS failsafe check entirely while FS_GCS_ENABLE is 0, so
            # a failsafe still latched when the parameter is restored stays latched and
            # refuses every later arm ("Arm: GCS failsafe on"). Let the app's heartbeat
            # clear it first.
            if any("gcs failsafe" in t.lower() for t in sitl_trace.statustext()):
                sitl_trace.wait_for("STATUSTEXT", lambda d: "gcs" in d["text"].lower()
                                    and "clear" in d["text"].lower(), timeout_s=15.0)
            set_vehicle_param(sitl_observer, sitl_trace, "FS_GCS_ENABLE", fs_before)
            settle_on_ground(api)
        finally:
            _write("link_loss", record)
            api.disconnect_vehicle()
            cutter.close()


# ── 2. Low battery ──────────────────────────────────────────────────────────

def test_low_battery_failsafe_rtl(api, sitl_observer, sitl_trace):
    """Sag the simulated battery under BATT_LOW_VOLT: ArduPilot's battery failsafe flies
    RTL, and the app shows the failsafe in the vehicle's own words."""
    record: dict = {}
    saved = {name: read_vehicle_param(sitl_observer, sitl_trace, name)
             for name in ("BATT_LOW_VOLT", "BATT_FS_LOW_ACT", "SIM_BATT_VOLTAGE")}
    record["params_before"] = saved
    try:
        mark = sitl_trace.mark()
        start_mission_from_ground(api)
        _wait_reached(sitl_trace, mark)
        before = api.telemetry()["telemetry"]
        record["app_before"] = {k: before.get(k) for k in ("battery_pct", "battery_v",
                                                          "battery_state", "battery_failsafe")}
        assert before.get("battery_failsafe") is False, before

        set_vehicle_param(sitl_observer, sitl_trace, "BATT_FS_LOW_ACT", 2)  # RTL
        set_vehicle_param(sitl_observer, sitl_trace, "BATT_LOW_VOLT", 11.0)
        sag_mark = sitl_trace.mark()
        set_vehicle_param(sitl_observer, sitl_trace, "SIM_BATT_VOLTAGE", 10.6)

        low_text = sitl_trace.wait_for("STATUSTEXT", lambda d: "battery" in d["text"].lower()
                                       and "low" in d["text"].lower(), sag_mark, 40.0)
        charge = sitl_trace.wait_for("BATTERY_STATUS", lambda d: int(d["charge_state"]) >= 2,
                                     sag_mark, 40.0)
        rtl = _mode_entry(sitl_trace, {"RTL"}, sag_mark, 40.0)
        record["vehicle_text"] = low_text.data if low_text else None
        record["vehicle_battery_status"] = charge.data if charge else None
        record["vehicle_rtl_after_sag_s"] = None if rtl is None else round(rtl.t - sag_mark, 2)
        record["vehicle_statustext"] = sitl_trace.statustext(sag_mark)
        assert low_text is not None, sitl_trace.statustext(sag_mark)
        assert charge is not None and rtl is not None, _modes_seen(sitl_trace, sag_mark)

        shown, _ = _poll(lambda: api.telemetry()["telemetry"],
                         lambda t: t.get("battery_failsafe") is True, 10.0)
        alerts = {a["kind"]: a["message"] for a in shown.get("alerts", [])}
        record["app_during"] = {k: shown.get(k) for k in
                                ("battery_pct", "battery_v", "battery_state", "battery_failsafe",
                                 "failsafe", "flight_mode", "alerts", "vehicle_messages")}
        assert shown.get("battery_failsafe") is True, shown
        assert "battery_failsafe" in alerts, alerts
        # In the vehicle's words, not just a flag.
        assert "Battery 1 is low" in alerts["battery_failsafe"], alerts
        assert any("battery" in m["text"].lower() for m in shown.get("vehicle_messages", [])), shown
        warnings, _ = _poll(lambda: api.flight_session()["session"]["state"]["warnings"],
                            lambda w: any(x.startswith("Battery failsafe") for x in w), 5.0)
        record["session_warnings"] = warnings
        # The percentage still reads healthy: the failsafe is on voltage.
        assert any(w.startswith("Battery failsafe") for w in warnings), warnings

        landed = _land_and_settle(api)
        record["landed_from_home_m"] = round(distance_m(
            landed["latitude"], landed["longitude"], landed["home_lat"], landed["home_lon"]), 1)
        assert record["landed_from_home_m"] < 5.0, record

        # On the ground with the failsafe still latched, preflight blocks on the vehicle's
        # own battery state, not on its percentage.
        battery_check = {c["id"]: c for c in api.run_preflight()["checks"]}["battery"]
        record["preflight_battery_after"] = battery_check
        assert battery_check["severity"] == "blocking" and "failsafe" in battery_check["message"], battery_check
    finally:
        for name, value in saved.items():
            set_vehicle_param(sitl_observer, sitl_trace, name, value)
        settle_on_ground(api)
        fresh_battery(sitl_observer)
        cleared, _ = _poll(lambda: api.telemetry()["telemetry"],
                           lambda t: t.get("battery_failsafe") is False, 20.0)
        record["app_after_reset"] = {k: cleared.get(k) for k in ("battery_state", "battery_failsafe",
                                                                 "battery_v", "battery_pct")}
        _write("low_battery", record)


# ── 3. GPS loss ─────────────────────────────────────────────────────────────

def test_gps_loss_in_flight(api, sitl_observer, sitl_trace):
    """Switch the simulated GPS off in AUTO: ArduPilot's EKF/GPS failsafe takes over,
    and the app stops showing a 3D fix as soon as the vehicle stops having one."""
    record: dict = {}
    try:
        mark = sitl_trace.mark()
        start_mission_from_ground(api)
        _wait_reached(sitl_trace, mark)
        assert api.telemetry()["telemetry"]["gps_fix"] >= 3

        off_mark = sitl_trace.mark()
        set_vehicle_param(sitl_observer, sitl_trace, "SIM_GPS_DISABLE", 1)
        # Vehicle: its own GPS report drops below 3D (or stops).
        lost_fix = sitl_trace.wait_for("GPS_RAW_INT", lambda d: int(d["fix_type"]) < 3, off_mark, 30.0)
        record["vehicle_gps_raw_after"] = lost_fix.data if lost_fix else None
        record["vehicle_gps_raw_count_after_off"] = len(sitl_trace.messages("GPS_RAW_INT", off_mark))
        # App: no stale "fix 3D".
        shown, app_s = _poll(lambda: api.telemetry()["telemetry"],
                             lambda t: int(t.get("gps_fix", 3)) < 3, 15.0)
        record["app_fix_lost_after_s"] = app_s
        record["app_vehicle_fix_lost_after_s"] = None if lost_fix is None else round(lost_fix.t - off_mark, 2)
        record["app_during"] = {k: shown.get(k) for k in ("gps_fix", "gps_stale", "satellites",
                                                         "flight_mode", "alerts")}
        assert app_s is not None, shown
        assert "gps_lost" in {a["kind"] for a in shown.get("alerts", [])}, shown

        # Vehicle: the failsafe it takes (EKF failsafe -> LAND on Copter 4.5 defaults).
        reaction = _mode_entry(sitl_trace, {"LAND", "ALT_HOLD", "RTL"}, off_mark, 90.0)
        record["vehicle_modes_after_gps_off"] = _modes_seen(sitl_trace, off_mark)
        record["vehicle_statustext"] = sitl_trace.statustext(off_mark)
        record["vehicle_reaction_after_s"] = None if reaction is None else round(reaction.t - off_mark, 2)
        assert reaction is not None, record
        failsafe_text = [t for t in record["vehicle_statustext"]
                         if any(w in t.lower() for w in ("ekf", "gps", "failsafe"))]
        assert failsafe_text, record["vehicle_statustext"]

        during, _ = _poll(lambda: api.telemetry()["telemetry"],
                          lambda t: t.get("flight_mode") != "AUTO", 10.0)
        alerts = {a["kind"]: a["message"] for a in during.get("alerts", [])}
        record["app_failsafe"] = {k: during.get(k) for k in ("flight_mode", "failsafe",
                                                            "system_status", "gps_fix",
                                                            "vehicle_messages")}
        record["app_alerts"] = alerts
        assert during["flight_mode"] == COPTER_MODES[int(reaction.data["custom_mode"])], during
        assert "gps_lost" in alerts, alerts
        # Every sample from here to touchdown: never a 3D fix while the receiver is off.
        deadline = time.monotonic() + 300.0
        samples = []
        while time.monotonic() < deadline:
            t = api.telemetry()["telemetry"]
            samples.append(int(t.get("gps_fix", 0)))
            if not t["armed"]:
                break
            time.sleep(0.5)
        record["app_fix_while_down"] = sorted(set(samples))
        assert max(samples) < 3, record["app_fix_while_down"]
        assert not api.telemetry()["telemetry"]["armed"], "did not land"
    finally:
        set_vehicle_param(sitl_observer, sitl_trace, "SIM_GPS_DISABLE", 0)
        _write("gps_loss", record)
        settle_on_ground(api)
    wait_telemetry(api, "GPS back", lambda t: t.get("gps_fix", 0) >= 3, 90.0)
    wait_armable(api, 180.0)


# ── 4. Geofence breach ──────────────────────────────────────────────────────

def _push_east(observer, speed_m_s: float = 5.0) -> None:
    """Harness: a GUIDED velocity demand east, as a pilot's stick or a gust would push."""
    observer.mav.set_position_target_local_ned_send(
        0, observer.target_system, observer.target_component,
        1,  # MAV_FRAME_LOCAL_NED
        0b0000111111000111,  # velocity only
        0, 0, 0, 0.0, speed_m_s, 0.0, 0, 0, 0, 0, 0)


def test_geofence_breach_triggers_fence_action(api, sitl_observer, sitl_trace):
    """Enable the uploaded fence, then send the aircraft out of it: FENCE_ACTION RTL,
    and the app reports the breach as the vehicle reported it."""
    record: dict = {}
    client = api._session._drone_client
    avoid_before = read_vehicle_param(sitl_observer, sitl_trace, "AVOID_ENABLE")
    record["AVOID_ENABLE_before"] = avoid_before
    try:
        assert plan_template(api, "grid")["ok"]
        upload = api.upload_mission()
        assert upload["ok"] and upload["upload"]["fence"]["success"], upload
        enabled = api.vehicle_command("fence_enable")
        assert enabled["ok"], enabled
        wait_telemetry(api, "fence enabled", lambda t: t.get("geofence_enabled") is True, 10.0)
        preflight = api.run_preflight()
        confirmed = api.run_preflight(acknowledged=preflight["pending_confirmation"])
        assert confirmed["can_start"], [(c["id"], c["message"]) for c in confirmed["checks"] if not c["ok"]]
        mark = sitl_trace.mark()
        assert api.vehicle_command("start")["ok"]
        wait_telemetry(api, "AUTO, airborne",
                       lambda t: t["flight_mode"] == "AUTO" and t["altitude_rel_m"] > 5.0, 90.0)
        _wait_reached(sitl_trace, mark)
        inside = api.telemetry()["telemetry"]
        assert inside.get("fence_breached") is False, inside

        # A GUIDED position target outside the enabled fence: ArduCopter refuses to fly
        # there (the destination check), so the aircraft stays inside.
        outside_lon, outside_lat = offset_lonlat(15.0, 90.0)  # 40 m east of the field
        goto_mark = sitl_trace.mark()
        record["goto_outside"] = client.goto(outside_lat, outside_lon, 10.0).to_dict()
        time.sleep(6.0)
        after_goto = api.telemetry()["telemetry"]
        record["after_goto_outside"] = {
            "mode": after_goto["flight_mode"], "fence_breached": after_goto.get("fence_breached"),
            "distance_to_target_m": round(distance_m(after_goto["latitude"], after_goto["longitude"],
                                                     outside_lat, outside_lon), 1),
            "statustext": sitl_trace.statustext(goto_mark)}
        # The refusal reaches the operator: DO_REPOSITION is acknowledged, FAILED here.
        goto_acks = [e.data for e in sitl_trace.messages("COMMAND_ACK", goto_mark)
                     if int(e.data["command"]) == 192]
        record["goto_outside_vehicle_ack"] = goto_acks
        assert goto_acks and goto_acks[0]["result"] != 0, goto_acks
        assert record["goto_outside"]["success"] is False, record["goto_outside"]
        assert record["after_goto_outside"]["distance_to_target_m"] > 30.0, record
        assert not sitl_trace.messages("FENCE_STATUS", goto_mark) or not any(
            e.data["breach_status"] for e in sitl_trace.messages("FENCE_STATUS", goto_mark)), \
            "a refused GUIDED target should not have breached"

        # Pushed out with a velocity demand (GUIDED velocity is not fence-checked ahead;
        # ArduPilot's own fence avoidance, which would brake it, is off for the push).
        set_vehicle_param(sitl_observer, sitl_trace, "AVOID_ENABLE", 0)
        push_mark = sitl_trace.mark()
        breach = None
        deadline = time.monotonic() + 60.0
        while breach is None and time.monotonic() < deadline:
            _push_east(sitl_observer)
            time.sleep(0.5)
            breach = sitl_trace.wait_for("FENCE_STATUS", lambda d: d["breach_status"] == 1,
                                         push_mark, 0.1)
        record["vehicle_breach"] = breach.data if breach else None
        assert breach is not None, _modes_seen(sitl_trace, push_mark)
        action = _mode_entry(sitl_trace, {"RTL", "LAND", "SMART_RTL", "BRAKE"}, push_mark, 20.0)
        record["vehicle_fence_action"] = None if action is None else COPTER_MODES[int(action.data["custom_mode"])]
        record["vehicle_statustext"] = sitl_trace.statustext(push_mark)
        assert record["vehicle_fence_action"] in ("RTL", "LAND"), _modes_seen(sitl_trace, push_mark)

        shown, app_s = _poll(lambda: api.telemetry()["telemetry"],
                             lambda t: t.get("fence_breached") is True, 10.0)
        alerts = {a["kind"]: a["message"] for a in shown.get("alerts", [])}
        record["app_breach_after_vehicle_s"] = app_s
        record["app_during"] = {k: shown.get(k) for k in
                                ("fence_breached", "fence_breach_count", "fence_breach_type",
                                 "flight_mode", "failsafe", "alerts", "vehicle_messages")}
        assert app_s is not None, shown
        assert "fence_breach" in alerts and shown.get("fence_breach_count", 0) >= 1, shown
        warnings, _ = _poll(lambda: api.flight_session()["session"]["state"]["warnings"],
                            lambda w: "Geofence breached" in w, 5.0)
        record["session_warnings"] = warnings
        assert "Geofence breached" in warnings, warnings

        landed = _land_and_settle(api)
        record["landed_from_home_m"] = round(distance_m(
            landed["latitude"], landed["longitude"], landed["home_lat"], landed["home_lon"]), 1)
        assert record["landed_from_home_m"] < 5.0, record
    finally:
        set_vehicle_param(sitl_observer, sitl_trace, "AVOID_ENABLE", avoid_before)
        api.vehicle_command("fence_disable")
        _write("geofence_breach", record)


# ── 5. Preflight blocks start; the override is recorded ─────────────────────

def test_preflight_blocks_start_and_records_override(api, sitl_observer, sitl_trace):
    """The vehicle reports a nearly flat battery: preflight blocks Start; the pilot's
    override starts the mission and is recorded in the audit log and the flight log."""
    record: dict = {}
    try:
        assert plan_template(api, "grid")["ok"]
        assert api.upload_mission()["ok"]
        # Environment: the vehicle's battery monitor says 10 % (MAV_CMD_BATTERY_RESET).
        sitl_observer.mav.command_long_send(sitl_observer.target_system,
                                            sitl_observer.target_component,
                                            42651, 0, 0xFF, 10.0, 0, 0, 0, 0, 0)
        low = sitl_trace.wait_for("SYS_STATUS", lambda d: 0 <= int(d["battery_remaining"]) <= 11,
                                  sitl_trace.mark(), 10.0)
        record["vehicle_sys_status"] = low.data if low else None
        assert low is not None
        wait_telemetry(api, "the app reading 10 %", lambda t: (t.get("battery_pct") or 100) <= 11, 10.0)

        preflight = api.run_preflight()
        confirmed = api.run_preflight(acknowledged=preflight["pending_confirmation"])
        battery = {c["id"]: c for c in confirmed["checks"]}["battery"]
        record["preflight_battery"] = battery
        record["preflight_can_start"] = confirmed["can_start"]
        assert battery["severity"] == "blocking" and not confirmed["can_start"], battery

        mark = sitl_trace.mark()
        held = api.vehicle_command("start")
        time.sleep(2.0)
        record["start_held"] = held
        record["acks_after_held_start"] = [e.data for e in sitl_trace.messages("COMMAND_ACK", mark)]
        assert held["ok"] is False and held.get("preflight_blocked"), held
        assert any("Battery" in b for b in held["blocking"]), held
        # Held by the app: nothing was sent to the aircraft.
        assert not any(int(e.data["command"]) in (400, 300)
                       for e in sitl_trace.messages("COMMAND_ACK", mark)), record["acks_after_held_start"]
        assert api.telemetry()["telemetry"]["armed"] is False

        mark = sitl_trace.mark()
        forced = api.vehicle_command("start", override_preflight=True)
        record["start_override"] = forced
        assert forced["ok"], forced
        flying = wait_telemetry(api, "AUTO, airborne",
                                lambda t: t["flight_mode"] == "AUTO" and t["altitude_rel_m"] > 3.0, 90.0)
        record["vehicle_acks"] = [e.data for e in sitl_trace.messages("COMMAND_ACK", mark)]
        record["app_flying"] = {k: flying.get(k) for k in ("flight_mode", "armed", "altitude_rel_m",
                                                          "battery_pct")}

        audit = [e for e in api.audit_log(50)["events"] if "preflight_override" in json.dumps(e, default=str)]
        record["audit_override"] = audit[:1]
        assert audit and "Battery" in json.dumps(audit[0], default=str), audit
        commands = api.flight_session()["session"]["commands"]
        starts = [c for c in commands if c["command"] == "start_mission"]
        record["flight_log_start"] = starts
        assert starts and starts[-1]["reason"] == "operator_preflight_override", starts

        rtl = api.vehicle_command("rtl")
        assert rtl["ok"], rtl
        _land_and_settle(api)
    finally:
        fresh_battery(sitl_observer)
        _write("preflight_override", record)
