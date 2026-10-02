"""Arming refused by ArduCopter's own pre-arm checks, surfaced through the Api.

* GPS lost (SITL's simulated receiver switched off with SIM_GPS_DISABLE) while in a
  position mode: ArduPilot refuses to arm, and the Api must report the refusal -- not
  "sent" -- while preflight blocks on the GPS check.
* The enabled geofence does not contain the aircraft: ArduPilot refuses to arm.

Fault injection goes through the independent observer link, as an environment change
(a receiver failing), never through the app under test.
"""

from __future__ import annotations

import json
import time

import pytest

from ._support import (
    connect_api,
    evidence_dir,
    make_api,
    offset_lonlat,
    settle_on_ground,
    wait_telemetry,
)

pytestmark = pytest.mark.sitl


def _set_param(observer, name: str, value: float) -> None:
    observer.mav.param_set_send(observer.target_system, observer.target_component,
                                name.encode(), float(value), 9)  # MAV_PARAM_TYPE_REAL32


@pytest.fixture
def api(tmp_path, sitl_connection):
    api = make_api(tmp_path)
    connect_api(api, sitl_connection)
    try:
        yield api
    finally:
        try:
            api.vehicle_command("fence_disable")
            settle_on_ground(api)
        finally:
            api.disconnect_vehicle()


def test_arming_without_gps_is_refused_by_the_vehicle(api, sitl_observer, sitl_trace):
    record: dict = {}
    try:
        assert api.vehicle_command("loiter")["ok"]
        _set_param(sitl_observer, "SIM_GPS_DISABLE", 1)
        lost = wait_telemetry(api, "GPS lost", lambda t: t.get("gps_fix", 3) < 3, 30.0)
        record["telemetry_without_gps"] = {k: lost.get(k) for k in ("gps_fix", "satellites", "flight_mode")}
        time.sleep(2.0)

        preflight = api.run_preflight()
        gps = {c["id"]: c for c in preflight["checks"]}["gps_status"]
        record["preflight_gps"] = gps
        assert gps["severity"] == "blocking", gps

        mark = sitl_trace.mark()
        refused = api.vehicle_command("arm")
        ack = sitl_trace.wait_for("COMMAND_ACK", lambda d: d["command"] == 400, mark, 5.0)
        time.sleep(1.0)
        record.update(arm=refused, ack=ack.data if ack else None, text=sitl_trace.statustext(mark),
                      armed_after=api.telemetry()["telemetry"]["armed"])
        assert refused["ok"] is False, refused
        assert refused["result"]["ack"] == "denied"
        assert ack is not None and ack.data["result"] != 0
        assert any(t.startswith(("PreArm", "Arm")) and "GPS" in t for t in record["text"]), record["text"]
        # The vehicle's reason reaches the operator, not just "failed".
        assert "GPS" in refused["error"], refused["error"]
        assert record["armed_after"] is False
    finally:
        _set_param(sitl_observer, "SIM_GPS_DISABLE", 0)
        (evidence_dir("refusals") / "arm_without_gps.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8")
    wait_telemetry(api, "GPS back", lambda t: t.get("gps_fix", 0) >= 3, 60.0)
    # And with the fix back, the same command is accepted: the refusal was the GPS.
    deadline = time.monotonic() + 90.0
    armed = api.vehicle_command("arm")
    while not armed["ok"] and time.monotonic() < deadline:
        time.sleep(3.0)
        armed = api.vehicle_command("arm")
    assert armed["ok"], armed
    assert api.vehicle_command("disarm")["ok"]


def test_arming_outside_the_enabled_fence_is_refused(api, sitl_trace):
    record: dict = {}
    try:
        # Survey a field 60-100 m north of the launch point: its fence (the field plus a
        # 5 m margin) does not contain the aircraft.
        far_field = [offset_lonlat(n, e) for n, e in ((60, 0), (60, 40), (100, 40), (100, 0))]
        assert api.set_aoi(far_field)["ok"]
        assert api.plan_mission({"template": "grid", "altitude_m": 30.0, "speed_m_s": 6.0,
                                 "max_altitude_m": 60.0, "rth_altitude_m": 40.0})["ok"]
        upload = api.upload_mission()
        assert upload["ok"], upload
        preflight = api.run_preflight()
        geofence = {c["id"]: c for c in preflight["checks"]}["geofence"]
        record["preflight_geofence"] = geofence
        # Preflight says so before the autopilot does.
        assert geofence["severity"] == "blocking" and "outside the geofence" in geofence["message"], geofence

        enabled = api.vehicle_command("fence_enable")
        assert enabled["ok"], enabled
        assert api.vehicle_command("loiter")["ok"]
        mark = sitl_trace.mark()
        refused = api.vehicle_command("arm")
        time.sleep(1.0)
        record.update(arm=refused, text=sitl_trace.statustext(mark),
                      armed_after=api.telemetry()["telemetry"]["armed"])
        assert refused["ok"] is False and refused["result"]["ack"] == "denied", refused
        assert any("fence" in t.lower() for t in record["text"]), record["text"]
        assert "fence" in refused["error"].lower(), refused["error"]
        assert record["armed_after"] is False
    finally:
        (evidence_dir("refusals") / "arm_outside_fence.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8")
