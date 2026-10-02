"""The MAVSDK (PX4) driver, against a stand-in for the mavsdk package.

The driver was a stub even with mavsdk installed: telemetry was an empty DroneTelemetry
that nothing ever updated, upload_mission always answered "MAVSDK async upload not yet
wired", and set_flight_mode was unsupported, so manual override could not work on PX4.

mavsdk is not installed here and needs a PX4 to talk to, so tests/fake_mavsdk.py stands
in at the SDK's public API. Everything on this side of it is production code, driven
through the Api where a user would drive it. A real PX4 SITL run is what verifies the
SDK boundary itself.
"""

from __future__ import annotations

import sys
import time

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from fake_mavsdk import FakePX4, build_module

AOI = [[8.5440, 47.3970], [8.5470, 47.3970], [8.5470, 47.3985], [8.5440, 47.3985]]


def install(monkeypatch, module, *, name: str = "mavsdk") -> None:
    """Install the stand-in as the SDK, under `name`, and hide any real one.

    The driver prefers ``mavsdk_grpc`` (the gRPC API's current name) to ``mavsdk``, so
    an installed real SDK would otherwise be picked over the stand-in.
    """
    for other in ("mavsdk", "mavsdk_grpc"):
        monkeypatch.setitem(sys.modules, other, None)
        monkeypatch.setitem(sys.modules, f"{other}.mission_raw", None)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, f"{name}.mission_raw", module.mission_raw)


@pytest.fixture
def px4(monkeypatch):
    vehicle = FakePX4()
    module = build_module(vehicle)
    install(monkeypatch, module)
    return vehicle


@pytest.fixture
def api(tmp_path, px4):
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("px4", str(tmp_path / "project"))
    connected = api.connect_vehicle("udp://:14540", "mavsdk")
    assert connected["vehicle"]["connected"] is True, connected
    time.sleep(0.6)
    yield api
    api.disconnect_vehicle()


def test_telemetry_streams_into_the_session(api, px4):
    telemetry = api.telemetry()["telemetry"]
    assert telemetry["connected"] is True
    assert telemetry["latitude"] == pytest.approx(47.3977)
    assert telemetry["altitude_rel_m"] == pytest.approx(25.0)
    assert telemetry["battery_pct"] == pytest.approx(64.0)
    assert telemetry["satellites"] == 12 and telemetry["gps_fix"] == 3
    assert telemetry["hdop"] == pytest.approx(0.9)
    assert telemetry["speed_mps"] == pytest.approx(5.0)
    assert telemetry["flight_mode"] == "LOITER"      # PX4 HOLD, in the app's vocabulary
    assert telemetry["home_set"] is True


def test_a_mavsdk_1_battery_fraction_is_read_as_a_fraction(tmp_path, monkeypatch):
    vehicle = FakePX4()
    vehicle.battery_remaining = 0.64
    module = build_module(vehicle, version="1.4.9")
    install(monkeypatch, module)
    from core.drone import MAVSDKDroneClient

    client = MAVSDKDroneClient()
    client.connect("udp://:14540")
    try:
        time.sleep(0.4)
        assert client.get_telemetry().battery_pct == pytest.approx(64.0)
    finally:
        client.disconnect()


def test_the_mission_uploads_with_fence_and_without_a_home_slot(api, px4):
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 50})["ok"]
    result = api.upload_mission()
    assert result["ok"] is True, result
    stored = px4.stored[0]
    assert len(stored) == result["items"], "PX4 reserves no home slot; none may be added"
    assert stored[0].command == 22 and stored[0].seq == 0     # NAV_TAKEOFF first
    waypoint = next(item for item in stored if item.command == 16)   # NAV_WAYPOINT
    assert abs(waypoint.x) > 1e8 and abs(waypoint.y) > 1e7, "positions go as degrees x 1e7"
    assert len(px4.stored[1]) == len(AOI)


def test_the_mission_reads_back_in_the_frames_it_was_planned_in(api, px4):
    """PX4 v1.15.4 returned every GLOBAL_RELATIVE_ALT (3) item as its *_INT alias (6).

    Same meaning, different number: a read-back compared with the plan showed every
    waypoint as changed. The driver gives read-back items in the planner's frames.
    """
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 50})["ok"]
    assert api.upload_mission()["ok"] is True
    held = api._session._drone_client.download_mission()
    sent = px4.stored[0]
    assert [item["frame"] for item in held] == [item.frame for item in sent]
    assert 3 in {item["frame"] for item in held}
    assert not {5, 6, 11} & {item["frame"] for item in held}
    waypoint = next(item for item in held if item["command"] == 16)
    assert 47.39 < waypoint["lat"] < 47.40 and 8.54 < waypoint["lon"] < 8.55


def test_the_upload_puts_the_plans_return_altitude_on_px4(api, px4):
    """A mission's RTL item carries no altitude PX4 reads; PX4 returned at its own 30 m
    from a plan that said 25 m (v1.15.4 SITL). RTL_RETURN_ALT is set and read back."""
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 50, "rth_altitude_m": 70,
                             "max_altitude_m": 100})["ok"]
    result = api.upload_mission()
    assert result["ok"] is True, result
    assert px4.return_altitude_m == pytest.approx(70.0)
    assert result["upload"]["return_altitude"]["success"] is True


def test_a_return_altitude_the_vehicle_refuses_fails_the_upload(api, px4):
    px4.deny_actions.add("set_return_to_launch_altitude")
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 50})["ok"]
    result = api.upload_mission()
    assert result["ok"] is False and result.get("mission_uploaded") is True, result
    assert "return altitude" in result["error"]


def test_a_rejected_upload_is_a_failure(api, px4):
    px4.reject_upload = True
    api.set_aoi(AOI)
    api.plan_mission({"template": "grid", "altitude_m": 50})
    result = api.upload_mission()
    assert result["ok"] is False
    assert "rejected" in result["error"].lower()


def test_commands_report_acceptance_and_denial(api, px4):
    accepted = api.vehicle_command("rth")
    assert accepted["ok"] is True and accepted["result"]["ack"] == "accepted"
    px4.deny_actions.add("land")
    denied = api.vehicle_command("land")
    assert denied["ok"] is False and "COMMAND_DENIED" in denied["error"]


def test_manual_override_is_confirmed_against_the_reported_mode(api, px4):
    px4.mode = "MISSION"
    time.sleep(0.3)
    result = api.take_manual_control()
    assert result["ok"] is True
    assert px4.mode == "HOLD"


def test_manual_override_that_does_not_take_is_reported(api, px4):
    px4.mode = "MISSION"
    px4.ignore_mode_for.add("hold")
    time.sleep(0.3)
    result = api.take_manual_control("HOLD")
    assert result["ok"] is False
    assert "still in AUTO" in result["error"]


def test_a_vehicle_that_never_appears_is_not_connected(tmp_path, monkeypatch):
    vehicle = FakePX4()
    module = build_module(vehicle)

    async def never():
        import asyncio
        while True:
            await asyncio.sleep(1)
            yield None

    module_core = module.System().core
    type(module_core).connection_state = lambda self: never()
    install(monkeypatch, module)
    from core.drone import MAVSDKDroneClient

    client = MAVSDKDroneClient()
    client.CONNECT_TIMEOUT_S = 1.0
    with pytest.raises(RuntimeError, match="Failed to connect"):
        client.connect("udp://:14540")
    assert client.is_connected() is False


def test_disconnect_releases_the_sdk_on_its_own_event_loop(api, px4):
    """After a real PX4 takeoff and landing, Disconnect never returned.

    The System (and its grpc.aio channel) was dropped on the caller's thread after the
    event loop had stopped; grpc's finaliser then spun holding the GIL. mavsdk_server is
    now stopped and the System released on the loop that owns the channel.
    """
    api.disconnect_vehicle()
    stops = [c for c in px4.calls if c.startswith("stop_server:")]
    assert stops == ["stop_server:mavsdk-loop:True"], px4.calls


def test_the_sdk_waits_long_enough_for_an_acknowledgement(api, px4):
    """mavsdk_server's 0.5 s default lost a real PX4's late ACKs on a loaded machine."""
    assert "mavlink_timeout:2" in px4.calls


def test_a_resume_whose_ack_is_lost_is_confirmed_by_the_vehicles_mode(api, px4):
    """PX4 resumed the mission; the ACK arrived after MAVSDK's timeout, and the app said
    the resume had failed. The mode PX4 then reports settles it."""
    px4.unacked_actions.add("start_mission")
    result = api.vehicle_command("resume")
    assert result["ok"] is True, result
    assert result["result"]["ack"] == "none"
    assert "No acknowledgement arrived" in result["result"]["message"]


def test_an_unacknowledged_command_the_vehicle_did_not_carry_out_is_not_success(api, px4):
    px4.unacked_actions.add("return_to_launch")
    px4.ignore_mode_for.add("return_to_launch")
    from core.drone import MAVSDKDroneClient

    MAVSDKDroneClient.MODE_CONFIRM_S = 0.5
    try:
        result = api.vehicle_command("rtl")
    finally:
        MAVSDKDroneClient.MODE_CONFIRM_S = 5.0
    assert result["ok"] is False, result
    assert result["result"]["ack"] == "none"
    assert "did not acknowledge" in result["result"]["message"]


def test_a_start_refused_while_px4_checks_the_new_plan_is_tried_again(api, px4):
    """A real PX4 refused MISSION a second after accepting the upload ("Mission start
    denied! No valid mission"): it was still checking the plan. Start tries again."""
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 50})["ok"]
    assert api.upload_mission()["ok"] is True
    px4.deny_times["start_mission"] = 2
    result = api.vehicle_command("start", {"override_preflight": True})
    assert result["ok"] is True, result
    assert px4.calls.count("start_mission") == 3 and px4.armed is True


def test_start_enters_mission_then_arms(api, px4):
    """mission.start_mission only changes the mode: a real PX4 sat disarmed in MISSION."""
    result = api.vehicle_command("start", {"override_preflight": True})
    assert result["ok"] is True, result
    assert px4.calls[-2:] == ["start_mission", "arm"]
    assert px4.armed is True and px4.mode == "MISSION"


def test_a_start_whose_arm_is_refused_says_the_aircraft_is_on_the_ground(api, px4):
    px4.deny_actions.add("arm")
    result = api.vehicle_command("start", {"override_preflight": True})
    assert result["ok"] is False
    assert "refused to arm" in result["error"] and "still on the ground" in result["error"]
    assert px4.armed is False


def test_the_grpc_api_is_found_under_its_new_name(monkeypatch):
    vehicle = FakePX4()
    module = build_module(vehicle, version="3.17.4")
    install(monkeypatch, module, name="mavsdk_grpc")
    from core.drone import MAVSDKDroneClient

    client = MAVSDKDroneClient()
    client.connect("udp://:14540")
    try:
        assert client.is_connected() and client._sdk_major == 3
    finally:
        client.disconnect()


def test_the_native_mavsdk_4_binding_is_refused_by_name(monkeypatch):
    """`pip install mavsdk` now installs a different library; say so, not 'cannot connect'."""
    module = build_module(FakePX4(), version="4.0.3")
    install(monkeypatch, module)
    from core.drone import MAVSDKDroneClient

    with pytest.raises(RuntimeError, match="mavsdk-grpc"):
        MAVSDKDroneClient().connect("udp://:14540")


@pytest.mark.parametrize("uri, major, expected", [
    ("udp://:14540", 3, "udpin://0.0.0.0:14540"),
    ("udp://192.168.1.5:14540", 3, "udpin://192.168.1.5:14540"),
    ("udpin://0.0.0.0:14540", 2, "udp://:14540"),
    ("udp://:14540", 2, "udp://:14540"),
    ("serial:///dev/ttyACM0:57600", 3, "serial:///dev/ttyACM0:57600"),
])
def test_the_connection_uri_is_spelled_for_the_installed_sdk(uri, major, expected):
    from core.drone import _mavsdk_address

    assert _mavsdk_address(uri, major) == expected
