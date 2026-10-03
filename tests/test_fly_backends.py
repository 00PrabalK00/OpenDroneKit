"""The Api the v1 Fly screen stands on (U6): every control there has a backend.

  * vehicle_messages: the autopilot's STATUSTEXT (pre-arm failures, failsafes) reaches
    the pilot. The pymavlink bridge always published it; nothing listened.
  * vehicle_mission: what is on the aircraft, so flown / remaining are drawn from the
    items the vehicle acknowledged, not from a plan edited since.
  * load_mission_version: the Fly screen's mission queue makes a saved version the one
    to upload.
  * The simulator (driver "mock") flies its uploaded route, arms, lands and triggers a
    camera, so the Fly screen's Arm / Land / Capture Now are backed on the mock driver
    too. Everything it reports stays marked simulated.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core.drone import MockDroneClient  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


@pytest.fixture
def api(tmp_path):
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("fly backends", str(tmp_path / "project"))
    yield api
    api.disconnect_vehicle()


def planned(api: Api) -> None:
    api.set_aoi(AOI)
    result = api.plan_mission({"template": "grid", "altitude_m": 40})
    assert result["ok"], result


class TestVehicleMessages:
    def test_the_simulator_s_messages_reach_the_log_marked_simulated(self, api):
        api.connect_vehicle("mock://", "mock")
        api.vehicle_command("arm")
        answer = api.vehicle_messages(0)
        assert answer["ok"] and answer["is_simulated"] is True
        texts = [m["text"] for m in answer["messages"]]
        assert "Armed." in texts
        assert all(m["is_simulated"] is True and m["source"] == "vehicle" for m in answer["messages"])
        # A poll from the last seq carries only what is new.
        assert api.vehicle_messages(answer["last_seq"])["messages"] == []
        api.vehicle_command("disarm")
        newer = api.vehicle_messages(answer["last_seq"])["messages"]
        assert [m["text"] for m in newer] == ["Disarmed."]

    def test_a_new_connection_starts_a_new_log(self, api):
        api.connect_vehicle("mock://", "mock")
        api.vehicle_command("arm")
        api.connect_vehicle("mock://again", "mock")
        texts = [m["text"] for m in api.vehicle_messages(0)["messages"]]
        assert "Armed." not in texts

    def test_an_autopilot_statustext_reaches_the_log(self, api):
        from fake_mavlink_vehicle import FakeVehicle, free_udp_port

        port = free_udp_port()
        vehicle = FakeVehicle(port).start()
        try:
            result = api.connect_vehicle(f"udpin:127.0.0.1:{port}", "pymavlink")
            assert result["ok"], result
            vehicle.say("PreArm: Compass not calibrated", severity=2)
            deadline = time.monotonic() + 10
            found = []
            while time.monotonic() < deadline and not found:
                found = [m for m in api.vehicle_messages(0)["messages"] if "Compass" in m["text"]]
                time.sleep(0.2)
            assert found, api.vehicle_messages(0)
            message = found[0]
            assert message["text"] == "PreArm: Compass not calibrated"
            assert message["severity"] == 2 and message["level"] == "critical"
            assert message["prearm"] is True and message["is_simulated"] is False
        finally:
            api.disconnect_vehicle()
            vehicle.stop()

    def test_the_station_logs_a_lost_and_a_regained_link(self, api):
        from fake_mavlink_vehicle import FakeVehicle, free_udp_port

        port = free_udp_port()
        vehicle = FakeVehicle(port).start()
        try:
            assert api.connect_vehicle(f"udpin:127.0.0.1:{port}", "pymavlink")["ok"]
            time.sleep(1.0)
            vehicle.go_silent()
            time.sleep(4.0)
            assert api.telemetry()["telemetry"]["link_lost"] is True
            vehicle.resume()
            time.sleep(1.5)
            api.telemetry()
            station = [m["text"] for m in api.vehicle_messages(0)["messages"] if m["source"] == "station"]
            assert station[0].startswith("Link lost") and station[-1] == "Link regained."
        finally:
            api.disconnect_vehicle()
            vehicle.stop()


class TestVehicleMission:
    def test_nothing_is_on_the_aircraft_before_an_upload(self, api):
        planned(api)
        answer = api.vehicle_mission()
        assert answer["on_vehicle"] is False and answer["uploaded"] == []
        assert answer["planned"]["template"] == "grid" and answer["planned"]["waypoints"] > 0

    def test_the_uploaded_items_are_what_the_vehicle_acknowledged(self, api):
        planned(api)
        api.connect_vehicle("mock://", "mock")
        upload = api.upload_mission()
        assert upload["ok"], upload
        answer = api.vehicle_mission()
        assert answer["on_vehicle"] is True and answer["item_count"] == upload["items"]
        seqs = [item["seq"] for item in answer["uploaded"]]
        assert seqs == sorted(seqs) and all(item["lat"] and item["lon"] for item in answer["uploaded"])
        # A new plan is not what is on the aircraft.
        api.plan_mission({"template": "grid", "altitude_m": 50})
        assert api.vehicle_mission()["on_vehicle"] is False


class TestLoadMissionVersion:
    def test_a_saved_version_becomes_the_mission_to_upload(self, api):
        planned(api)
        api.save_mission("first")
        waypoints = api.vehicle_mission()["planned"]["waypoints"]
        api.plan_mission({"template": "grid", "altitude_m": 80})
        loaded = api.load_mission_version(1)
        assert loaded["ok"], loaded
        assert loaded["waypoints"] == waypoints
        assert api.vehicle_mission()["planned"]["source_version"] == 1
        api.connect_vehicle("mock://", "mock")
        assert api.upload_mission()["ok"]

    def test_an_unknown_version_is_refused_with_the_ones_there_are(self, api):
        planned(api)
        api.save_mission("first")
        refused = api.load_mission_version(9)
        assert refused["ok"] is False and "Saved versions: 1" in refused["error"]


class TestTheSimulatorFlies:
    def test_it_is_placed_on_the_site_and_flies_the_route(self, api):
        planned(api)
        api.connect_vehicle("mock://", "mock")
        start = api.telemetry()["telemetry"]
        # On the project's site, not the driver's default in San Francisco.
        assert 41.30 < start["latitude"] < 41.31 and -81.76 < start["longitude"] < -81.74
        assert api.upload_mission()["ok"]
        assert api.vehicle_command("start", {"override_preflight": True})["ok"]
        time.sleep(1.2)
        flying = api.telemetry()["telemetry"]
        assert flying["armed"] is True and flying["flight_mode"] == "AUTO"
        assert flying["altitude_rel_m"] > 0 and flying["is_simulated"] is True

    def test_arm_land_and_capture_are_answered(self, api):
        api.connect_vehicle("mock://", "mock")
        assert api.vehicle_command("arm")["ok"]
        assert api.vehicle_command("land")["ok"]
        photo = api.camera_command("take_photo")
        assert photo["ok"], photo
        assert api.telemetry()["telemetry"]["armed"] in (True, False)

    def test_the_route_is_flown_in_order(self):
        client = MockDroneClient()
        client.connect("mock://")
        items = [{"seq": 0, "command": 22, "lat": 41.3, "lon": -81.75, "alt": 2.0},
                 {"seq": 1, "command": 178, "param2": 50.0},
                 {"seq": 2, "command": 16, "lat": 41.3001, "lon": -81.75, "alt": 2.0},
                 {"seq": 3, "command": 16, "lat": 41.3002, "lon": -81.75, "alt": 2.0}]
        assert client.upload_mission(items).success
        assert client.start_mission().success
        seen = []
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            tel = client.get_telemetry()
            seen.append(tel.waypoint_index)
            if client._phase in ("rth", "landing", "landed") and len(set(seen)) >= 2:
                break
            time.sleep(0.1)
        assert seen == sorted(seen) and 2 in seen
        assert math.isclose(tel.home_lat, 41.3) and math.isclose(tel.home_lon, -81.75)

    def test_disarm_is_refused_in_the_air(self):
        client = MockDroneClient()
        client.connect("mock://")
        client.arm()
        assert client.takeoff(10).success
        time.sleep(0.6)
        client.get_telemetry()
        refused = client.disarm()
        assert refused.success is False and "airborne" in refused.message
