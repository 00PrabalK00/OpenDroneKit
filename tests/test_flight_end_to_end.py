"""The flight path, driven through the Api the way both UIs drive it.

An audit drove app.api.Api against a MAVLink peer and found the core bridge sound and
everything between it and the pilot broken: the default UI's "MAVLink (real autopilot)"
driver name was rejected, a rejected mission upload was reported as uploaded, a refused
manual override surfaced as a Python TypeError, a silent aircraft read "connected, AUTO,
armed" forever, fly-to-draw refused every corner, and the exported flight log stitched
a simulated San Francisco flight onto a real one for a 3,477 km "flight".

Each test here drives the production Api, session and bridge against
tests/fake_mavlink_vehicle.py -- a real UDP MAVLink 2 peer, not a mock of the bridge --
and asserts on what a pilot would be told.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("pymavlink")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from fake_mavlink_vehicle import FakeVehicle, free_udp_port  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


@pytest.fixture
def api(tmp_path):
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("flight e2e", str(tmp_path / "project"))
    yield api
    api.disconnect_vehicle()


@pytest.fixture
def vehicles():
    started: list[FakeVehicle] = []

    def start(**options) -> tuple[FakeVehicle, str]:
        port = free_udp_port()
        vehicle = FakeVehicle(port, **options).start()
        started.append(vehicle)
        return vehicle, f"udpin:127.0.0.1:{port}"

    yield start
    for vehicle in started:
        vehicle.stop()


def connect(api: Api, uri: str, driver: str = "pymavlink") -> dict:
    result = api.connect_vehicle(uri, driver)
    assert result["ok"] and result["vehicle"]["connected"], result
    # Let the link-quality window and a few telemetry messages arrive.
    time.sleep(2.4)
    return result


def plan(api: Api) -> None:
    api.set_aoi(AOI)
    planned = api.plan_mission({"template": "grid", "altitude_m": 60})
    assert planned["ok"], planned


# --------------------------------------------------------------------- connecting


class TestConnecting:
    def test_the_default_ui_driver_name_connects_a_real_link(self, api, vehicles):
        """index.html offered driver 'mavlink'; the factory rejected it by name."""
        _, uri = vehicles()
        result = api.connect_vehicle(uri, "mavlink")
        assert result["vehicle"]["connected"] is True, result
        assert result["vehicle"]["is_simulated"] is False

    def test_every_documented_pymavlink_name_is_accepted(self):
        from core.drone import PYMAVLINK_DRIVERS, create_drone_client
        from core.mission_planner_bridge import MissionPlannerDroneClient

        for name in PYMAVLINK_DRIVERS:
            assert isinstance(create_drone_client(name), MissionPlannerDroneClient), name

    def test_telemetry_reports_what_the_vehicle_sent(self, api, vehicles):
        _, uri = vehicles()
        connect(api, uri)
        telemetry = api.telemetry()["telemetry"]
        assert telemetry["connected"] is True and telemetry["stale"] is False
        assert telemetry["latitude"] == pytest.approx(41.3042)
        assert telemetry["battery_pct"] == pytest.approx(77.0)
        assert telemetry["gps_fix"] == 3 and telemetry["satellites"] == 11
        # Measured from MAVLink sequence numbers; a clean loopback delivers everything.
        assert telemetry["link_quality_pct"] == pytest.approx(100.0)


class TestUnreportedIsNotZero:
    def test_a_battery_the_vehicle_cannot_measure_is_absent_not_empty(self, api, vehicles):
        """SYS_STATUS battery_remaining = -1 was clamped to 0 %: an empty battery on
        screen for every flight controller without a current sensor."""
        _, uri = vehicles(battery_remaining=-1)
        connect(api, uri)
        telemetry = api.telemetry()["telemetry"]
        assert "battery_pct" not in telemetry
        assert telemetry["battery_v"] == pytest.approx(16.2)

    def test_preflight_treats_an_unreported_battery_as_a_manual_check(self, api, vehicles):
        _, uri = vehicles(battery_remaining=-1)
        connect(api, uri)
        checks = {c["id"]: c for c in api.run_preflight()["checks"]}
        assert checks["battery"]["severity"] == "warning"
        assert checks["battery"]["requires_manual"] is True
        assert "not reported" in checks["battery"]["message"]


class TestALostLinkIsReported:
    def test_a_silent_vehicle_is_not_shown_as_connected(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        assert api.telemetry()["telemetry"]["connected"] is True

        vehicle.go_silent()
        time.sleep(4.0)
        telemetry = api.telemetry()["telemetry"]
        assert telemetry["connected"] is False
        assert telemetry["link_lost"] is True and telemetry["stale"] is True
        assert telemetry["telemetry_age_s"] > 3.0
        assert "Link lost" in telemetry["reason"]
        # The last known position is kept, labelled, because it is where to look.
        assert telemetry["latitude"] == pytest.approx(41.3042)

    def test_stale_telemetry_is_not_recorded_as_flight(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        vehicle.go_silent()
        time.sleep(3.6)
        before = len(api._session.flight_log.samples)
        for _ in range(3):
            api.telemetry()
            time.sleep(0.4)
        assert len(api._session.flight_log.samples) == before

    def test_the_flight_session_warns_of_the_lost_link(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        vehicle.go_silent()
        time.sleep(4.5)
        state = api.flight_session()["session"]["state"]
        assert "Telemetry lost" in state["warnings"]


# ----------------------------------------------------------------------- upload


class TestMissionUpload:
    def test_an_accepted_upload_carries_the_geofence_too(self, api, vehicles):
        """The docstring promised mission, geofence and rally; only the mission went."""
        vehicle, uri = vehicles()
        connect(api, uri)
        plan(api)
        result = api.upload_mission()
        assert result["ok"] is True, result
        assert result["upload"]["mission"]["success"] is True
        assert result["upload"]["fence"]["success"] is True
        assert len(vehicle.stored[0]) == result["items"] + 1   # plus ArduPilot's home slot
        assert len(vehicle.stored[1]) == len(AOI)
        assert api.check_interrupted_flight()["recovered"] is True

    def test_a_rejected_upload_is_a_failure(self, api, vehicles):
        """ok:true over MISSION_ACK 'generic error' made both UIs say 'Mission uploaded'."""
        _, uri = vehicles(reject_upload=True)
        connect(api, uri)
        plan(api)
        result = api.upload_mission()
        assert result["ok"] is False
        assert "rejected" in result["error"].lower()
        assert api._session.mission_uploaded is False

    def test_a_rejected_upload_is_not_recorded_as_on_the_aircraft(self, api, vehicles):
        """Crash recovery would otherwise claim a mission the vehicle refused."""
        _, uri = vehicles(reject_upload=True)
        connect(api, uri)
        plan(api)
        api.upload_mission()
        assert api.check_interrupted_flight()["recovered"] is False

    def test_preflight_knows_whether_the_mission_is_on_the_aircraft(self, api, vehicles):
        _, uri = vehicles()
        connect(api, uri)
        plan(api)
        before = {c["id"]: c for c in api.run_preflight()["checks"]}
        assert before["mission_uploaded"]["severity"] == "blocking"
        api.upload_mission()
        after = {c["id"]: c for c in api.run_preflight()["checks"]}
        assert after["mission_uploaded"]["severity"] == "pass"


# --------------------------------------------------------------------- commands


class TestCommandsReportTheVehiclesAnswer:
    def test_an_accepted_command_says_so(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        result = api.vehicle_command("rth")
        assert result["ok"] is True
        assert result["result"]["ack"] == "accepted"
        time.sleep(0.5)
        assert vehicle.mode_name() == "RTL"

    def test_rtl_is_accepted_as_a_name(self, api, vehicles):
        """The cockpit's RTL button sent 'rtl'; the Api knew only 'rth'."""
        vehicle, uri = vehicles()
        connect(api, uri)
        assert api.vehicle_command("rtl")["ok"] is True
        assert (20, 0.0, 0.0) in vehicle.commands

    def test_a_denied_command_is_a_failure(self, api, vehicles):
        _, uri = vehicles(command_ack="deny")
        connect(api, uri)
        result = api.vehicle_command("land")
        assert result["ok"] is False
        assert result["result"]["ack"] == "denied"
        assert "denied" in result["error"]

    def test_silence_is_not_success(self, api, vehicles):
        """Every command used to answer 'Command sent.' whether or not anyone heard."""
        _, uri = vehicles(command_ack=None)
        connect(api, uri)
        result = api.vehicle_command("land")
        assert result["ok"] is False
        assert result["result"]["ack"] == "none"
        assert "did not acknowledge" in result["error"]

    def test_abort_holds_position_and_says_where(self, api, vehicles):
        """Abort was a bare DO_PAUSE_CONTINUE reported as 'Command sent.'."""
        vehicle, uri = vehicles()
        vehicle.custom_mode = 3  # AUTO
        connect(api, uri)
        result = api.vehicle_command("abort")
        assert result["ok"] is True
        assert "holding in BRAKE" in result["result"]["message"]
        assert vehicle.mode_name() == "BRAKE"
        assert api.flight_session()["session"]["state"]["status"] == "aborted"

    def test_capture_now_reaches_the_camera(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        result = api.camera_command("take_photo")
        assert result["ok"] is True
        assert any(command == 2000 for command, *_ in vehicle.commands)

    def test_commands_are_logged_by_the_flight_session(self, api, vehicles):
        """FlightManager was unreachable from the app: no commands.jsonl, no state."""
        _, uri = vehicles()
        connect(api, uri)
        api.vehicle_command("rth")
        session = api.flight_session()["session"]
        assert session["state"]["status"] == "rth"
        logged = session["commands"][-1]
        assert logged["command"] == "return_to_home" and logged["ack"] == "accepted"
        assert (Path(session["log_dir"]) / "commands.jsonl").is_file()
        assert (Path(session["log_dir"]) / "telemetry.jsonl").is_file()

    def test_flying_is_what_the_aircraft_reports_not_what_was_sent(self, api, vehicles):
        vehicle, uri = vehicles()
        connect(api, uri)
        plan(api)
        assert api.upload_mission()["ok"]
        vehicle.accept_modes = True
        result = api.vehicle_command("start")
        assert result["ok"] is True
        # The fake arms and enters AUTO on MISSION_START; the session follows the
        # telemetry there on its next poll.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if api.flight_session()["session"]["state"]["status"] == "flying":
                break
            time.sleep(0.2)
        assert api.flight_session()["session"]["state"]["status"] == "flying"


class TestManualOverride:
    def test_a_refused_hand_back_is_explained_not_a_type_error(self, api, vehicles):
        """fail(result['error'], **result) passed 'error' twice and raised TypeError on
        exactly the path the feature exists for."""
        vehicle, uri = vehicles(accept_modes=False)
        vehicle.custom_mode = 3  # AUTO
        connect(api, uri)
        result = api.take_manual_control("LOITER")
        assert result["ok"] is False
        assert "TypeError" not in result["error"]
        assert "still in AUTO" in result["error"]
        assert result["attempts"]

    def test_an_accepted_hand_back_is_confirmed(self, api, vehicles):
        vehicle, uri = vehicles()
        vehicle.custom_mode = 3
        connect(api, uri)
        result = api.take_manual_control()
        assert result["ok"] is True and result["mode"] == "LOITER"
        assert vehicle.mode_name() == "LOITER"


# ------------------------------------------------------------------ fly to draw


class TestFlyToDraw:
    def test_corners_are_marked_from_live_telemetry(self, api, vehicles):
        """mark_from_telemetry read 'fix_type'; the session sends 'gps_fix', so every
        corner was refused as fix type 0."""
        vehicle, uri = vehicles()
        connect(api, uri)
        for lat, lon in ((41.3030, -81.7520), (41.3030, -81.7480), (41.3060, -81.7480)):
            vehicle.lat, vehicle.lon = lat, lon
            time.sleep(0.8)
            marked = api.mark_boundary_corner("")
            assert marked["ok"] is True, marked
            assert marked["mark"]["fix_type"] == 3
        boundary = api.boundary_from_marks(True)
        assert boundary["ok"] is True
        assert boundary["area_hectares"] > 0


# -------------------------------------------------------------------- flight log


class TestFlightLogIsOneConnection:
    def test_a_mock_connection_does_not_leak_into_a_real_log(self, api, vehicles):
        """The audit's 3,477 km flight: mock San Francisco samples then real Ohio ones."""
        api.connect_vehicle("mock://", "mock")
        for _ in range(3):
            api.telemetry()
            time.sleep(0.2)
        _, uri = vehicles()
        connected = connect(api, uri)
        assert connected["flight_log"]["simulated"] is False
        # The mock log was archived, not silently dropped, and is marked simulated.
        archived = connected["flight_log"]["archived"]
        assert archived and archived["summary"]["simulated"] is True

        exported = api.export_flight_log()
        assert exported["ok"] and exported["simulated"] is False
        summary = exported["summary"]
        assert summary["driver"] == "pymavlink"
        assert summary["distance_m"] < 1000
        rows = Path(exported["files"]["csv"]).read_text(encoding="utf-8").splitlines()
        assert not any("37.7749" in row for row in rows), "mock samples leaked into the log"
        assert "simulated" in rows[0]

    def test_a_simulated_log_is_marked_everywhere(self, api):
        api.connect_vehicle("mock://", "mock")
        for _ in range(3):
            api.telemetry()
            time.sleep(0.1)
        exported = api.export_flight_log()
        assert exported["ok"] is True and exported["simulated"] is True
        assert "SIMULATED" in exported["warning"]
        for path in exported["files"].values():
            assert Path(path).name.startswith("SIMULATED_")
        payload = json.loads(Path(exported["files"]["json"]).read_text(encoding="utf-8"))
        assert payload["summary"]["simulated"] is True
        assert all(sample["simulated"] is True for sample in payload["samples"])
        assert "SIMULATED" in Path(exported["files"]["gpx"]).read_text(encoding="utf-8")


# --------------------------------------------------------------------- preflight


class TestPreflightFromTheApi:
    def test_a_camera_that_said_nothing_is_not_ready(self, api, vehicles):
        """check_camera_ready defaulted to PASS 'Camera ready.' for a vehicle with no
        camera at all."""
        _, uri = vehicles()
        connect(api, uri)
        camera = {c["id"]: c for c in api.run_preflight()["checks"]}["camera"]
        assert camera["severity"] == "warning" and camera["requires_manual"] is True

    def test_a_camera_that_reported_itself_passes(self, api, vehicles):
        _, uri = vehicles(send_camera_information=True)
        connect(api, uri)
        camera = {c["id"]: c for c in api.run_preflight()["checks"]}["camera"]
        assert camera["severity"] == "pass"

    def test_the_geofence_check_is_computed_from_the_path(self, api, vehicles):
        _, uri = vehicles()
        connect(api, uri)
        plan(api)
        geofence = {c["id"]: c for c in api.run_preflight()["checks"]}["geofence"]
        assert geofence["severity"] == "pass"
        assert "path points" in geofence["message"]

    ZONE = [[-81.7510, 41.3040], [-81.7490, 41.3040], [-81.7490, 41.3050], [-81.7510, 41.3050]]

    def test_a_no_fly_zone_drawn_after_planning_blocks(self, api, vehicles):
        """"No NFZs declared" was hardcoded. A zone drawn across the middle of the survey
        after the plan was made is crossed by that plan, and preflight has to say so."""
        _, uri = vehicles()
        connect(api, uri)
        plan(api)
        api.set_no_fly_zones([self.ZONE])
        nfz = {c["id"]: c for c in api.run_preflight()["checks"]}["no_fly_zones"]
        assert nfz["severity"] == "blocking"
        assert nfz["details"]["conflicts"][0]["conflict"] is True

    def test_a_plan_made_around_the_zone_clears_it(self, api, vehicles):
        _, uri = vehicles()
        connect(api, uri)
        api.set_no_fly_zones([self.ZONE])
        plan(api)
        nfz = {c["id"]: c for c in api.run_preflight()["checks"]}["no_fly_zones"]
        assert nfz["severity"] == "pass"
        assert "1 NFZs clear" in nfz["message"]

    def test_a_confirmation_is_kept_and_cannot_clear_a_blocker(self, api, vehicles):
        _, uri = vehicles()
        connect(api, uri)
        plan(api)
        first = api.run_preflight(["weather", "mission_uploaded"])
        checks = {c["id"]: c for c in first["checks"]}
        assert checks["weather"]["confirmed"] is True
        # Not uploaded is a fact about the aircraft, not something to acknowledge away.
        assert checks["mission_uploaded"]["confirmed"] is False
        again = {c["id"]: c for c in api.run_preflight()["checks"]}
        assert again["weather"]["confirmed"] is True
        assert "weather" not in api.run_preflight()["pending_confirmation"]


# ------------------------------------------------------------------------- fleet


class TestFleetOnAFreshDatabase:
    @pytest.fixture
    def fresh_db(self, tmp_path, monkeypatch):
        pytest.importorskip("sqlalchemy")
        import services.api.db as db_module

        monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
        monkeypatch.setattr(db_module, "_engine", None)
        monkeypatch.setattr(db_module, "_SessionLocal", None)
        yield
        db_module._engine = None          # noqa: SLF001
        db_module._SessionLocal = None    # noqa: SLF001

    def test_the_first_aircraft_on_a_new_install_is_recorded(self, api, fresh_db):
        """Organisation 1 only existed if the web service had registered someone, so
        a new desktop install failed every fleet record with a FOREIGN KEY error."""
        added = api.add_aircraft(1, "ODK-01", "Quad", "SN-1")
        assert added["ok"] is True, added
        assert api.add_battery(1, "B-1", 5000, 300)["ok"] is True
        assert api.add_pilot(1, "A Pilot", "L-1", "2030-01-01")["ok"] is True
        fleet = api.list_fleet(1)
        assert [a["name"] for a in fleet["aircraft"]] == ["ODK-01"]
        assert api.log_maintenance(added["id"], "inspection", "", "operator")["ok"] is True
        assert api.assign_mission_to_aircraft(added["id"], "Mission")["ok"] is True

    def test_an_organisation_nobody_created_is_refused(self, api, fresh_db):
        result = api.add_aircraft(99, "Ghost", "", "")
        assert result["ok"] is False
        assert "No organisation with id 99" in result["error"]
