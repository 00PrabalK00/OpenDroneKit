"""The same lifecycle, driven through app.api.Api instead of the bridge.

test_flight_lifecycle.py proves the bridge against ArduPilot. The flight audit found the
bridge sound and the layer above it broken -- a rejected upload reported as uploaded,
commands reported as sent rather than accepted, a lost link shown as connected -- so this
drives the Api the way both UIs do: connect, preflight, upload (mission and fence),
arm/takeoff, start, manual override, RTL. Every step asserts on what the Api tells the
operator, and on the aircraft's own telemetry.

Opt-in like the rest of tests/sitl: `pytest -m sitl` with the SITL container running.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.sitl


def _wait(api, description: str, predicate, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    telemetry: dict = {}
    while time.monotonic() < deadline:
        telemetry = api.telemetry()["telemetry"]
        if predicate(telemetry):
            return telemetry
        time.sleep(0.25)
    pytest.fail(f"Timed out waiting for {description}; last telemetry: {telemetry}")


def test_connect_upload_fly_override_and_return_through_the_api(
        tmp_path, sitl_connection, sitl_mission):
    from app.api import Api, _plan_to_dict
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("sitl", str(tmp_path / "project"))
    try:
        connected = api.connect_vehicle(sitl_connection, "mavlink")
        assert connected["vehicle"]["connected"], connected
        telemetry = _wait(api, "a 3D fix and home", lambda t: t.get("gps_fix", 0) >= 3
                          and t.get("home_set"), 60.0)
        assert telemetry["stale"] is False and telemetry["is_simulated"] is False

        api._session.mission_plan = sitl_mission.plan
        api._session.mission_plan_dict = _plan_to_dict(sitl_mission.plan)
        upload = api.upload_mission()
        assert upload["ok"], upload
        assert upload["result"]["success"] is True

        preflight = api.run_preflight()
        checks = {c["id"]: c for c in preflight["checks"]}
        for check in ("drone_connection", "gps_status", "home_position", "mission_uploaded"):
            assert checks[check]["severity"] == "pass", checks[check]

        takeoff = api.vehicle_command("arm")
        assert takeoff["ok"], takeoff
        assert takeoff["result"]["ack"] in ("accepted", "in_progress")
        lifted = api.vehicle_command("takeoff", altitude_m=10.0)
        assert lifted["ok"], lifted
        _wait(api, "takeoff altitude", lambda t: t.get("altitude_rel_m", 0) >= 8.0, 60.0)

        started = api.vehicle_command("start")
        assert started["ok"], started
        _wait(api, "AUTO", lambda t: t.get("flight_mode") == "AUTO", 20.0)

        override = api.take_manual_control()
        assert override["ok"], override
        assert api.control_state()["state"]["pilot_has_control"] is True

        rtl = api.vehicle_command("rtl")
        assert rtl["ok"], rtl
        _wait(api, "RTL", lambda t: t.get("flight_mode") == "RTL", 20.0)
        _wait(api, "landing and disarm", lambda t: not t.get("armed"), 180.0)

        exported = api.export_flight_log()
        assert exported["ok"] and exported["simulated"] is False
        assert exported["summary"]["positioned_samples"] > 0
        session = api.flight_session()["session"]
        assert any(c["command"] == "return_to_home" and c["success"] for c in session["commands"])
    finally:
        api.disconnect_vehicle()
