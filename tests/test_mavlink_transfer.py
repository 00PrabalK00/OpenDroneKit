"""MAVLink mission transfer against a loopback peer.

Two defects motivate these tests, both of which passed unnoticed before:

*The upload never waited.* Items were pushed straight after MISSION_COUNT rather
than in response to MISSION_REQUEST, which ArduPilot discards, and nothing read
MISSION_ACK, so the upload reported success no matter what the vehicle stored.

*mission_type is a MAVLink 2 extension.* Under MAVLink 1 it is truncated off the
wire and reads back as 0, so a fence or rally upload silently overwrote the flight
plan.

The peer here is a socket, not a mock: a mocked link cannot exhibit either failure.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

# Must be set before pymavlink is first imported or the dialect resolves to v1.
os.environ.setdefault("MAVLINK20", "1")

pytest.importorskip("pymavlink")

from core.mission_planner_bridge import (  # noqa: E402
    MAV_MISSION_TYPE_FENCE,
    MAV_MISSION_TYPE_MISSION,
    MAV_MISSION_TYPE_RALLY,
    MissionPlannerDroneClient,
)
from mission.exporters import (  # noqa: E402
    MAV_CMD_CONDITION_YAW,
    MAV_CMD_DO_DIGICAM_CONTROL,
    MAV_CMD_DO_MOUNT_CONTROL,
    MAV_CMD_NAV_LOITER_TIME,
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    build_mission_items,
)

PLAN = {
    "altitude_m": 60.0,
    "gimbal_tilt_deg": -90.0,
    "waypoints": [],
    "flight_recipe": {
        "world_poses": [
            {
                "lon": -81.7505 + index * 0.0004, "lat": 41.3042, "alt_m": 60.0,
                "yaw_deg": 90.0, "gimbal_pitch_deg": -90.0, "dwell_s": 2.0,
                "trigger": True, "camera_yaw_locked": True,
            }
            for index in range(4)
        ],
        "capture": {"continuous_capture": False},
    },
    "safety_constraints": {
        "geofence": [
            [-81.7520, 41.3030], [-81.7480, 41.3030],
            [-81.7480, 41.3060], [-81.7520, 41.3060],
        ],
        "no_fly_polygons": [
            [[-81.7500, 41.3045], [-81.7495, 41.3045],
             [-81.7495, 41.3050], [-81.7500, 41.3050]]
        ],
        "rally_points": [{"lon": -81.7510, "lat": 41.3035, "alt_m": 50.0}],
        "rth_altitude_m": 75.0,
    },
}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LoopbackVehicle:
    """A MAVLink peer that implements the mission transfer protocol.

    Deliberately request-driven: it sends MISSION_REQUEST_INT for each sequence
    number and only acks once it has them all, so an uploader that blasts items
    without waiting will time out rather than appear to work.
    """

    def __init__(self, port: int):
        from pymavlink import mavutil

        self._mavutil = mavutil
        self.connection = mavutil.mavlink_connection(
            f"tcpin:127.0.0.1:{port}", source_system=1, source_component=1
        )
        self.stored: dict[int, list] = {0: [], 1: [], 2: []}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        try:
            self.connection.close()
        except Exception:
            pass

    def _heartbeat(self):
        self.connection.mav.heartbeat_send(
            self._mavutil.mavlink.MAV_TYPE_QUADROTOR,
            self._mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            0, 0, self._mavutil.mavlink.MAV_STATE_STANDBY,
        )

    def _serve(self):
        pending_type = 0
        expected = 0
        received: list = []
        last_beat = 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_beat > 0.25:
                last_beat = now
                try:
                    self._heartbeat()
                except Exception:
                    pass

            message = self.connection.recv_match(blocking=True, timeout=0.2)
            if message is None:
                continue
            kind = message.get_type()

            if kind == "MISSION_COUNT":
                pending_type = int(getattr(message, "mission_type", 0))
                expected = int(message.count)
                received = []
                self.connection.mav.mission_request_int_send(
                    message.get_srcSystem(), message.get_srcComponent(), 0, pending_type
                )

            elif kind == "MISSION_ITEM_INT":
                received.append(message)
                seq = int(message.seq)
                if seq + 1 < expected:
                    self.connection.mav.mission_request_int_send(
                        message.get_srcSystem(), message.get_srcComponent(),
                        seq + 1, pending_type,
                    )
                else:
                    self.stored[pending_type] = received
                    self.connection.mav.mission_ack_send(
                        message.get_srcSystem(), message.get_srcComponent(), 0, pending_type
                    )

            elif kind == "MISSION_REQUEST_LIST":
                requested = int(getattr(message, "mission_type", 0))
                self.connection.mav.mission_count_send(
                    message.get_srcSystem(), message.get_srcComponent(),
                    len(self.stored[requested]), requested,
                )

            elif kind == "MISSION_REQUEST_INT":
                requested = int(getattr(message, "mission_type", 0))
                seq = int(message.seq)
                items = self.stored[requested]
                if seq < len(items):
                    item = items[seq]
                    self.connection.mav.mission_item_int_send(
                        message.get_srcSystem(), message.get_srcComponent(),
                        item.seq, item.frame, item.command, item.current,
                        item.autocontinue, item.param1, item.param2, item.param3,
                        item.param4, item.x, item.y, item.z, requested,
                    )


@pytest.fixture
def linked_vehicle():
    port = _free_port()
    vehicle = LoopbackVehicle(port).start()
    client = MissionPlannerDroneClient()
    try:
        client.connect(f"tcp:127.0.0.1:{port}")
        yield client, vehicle
    finally:
        client.disconnect()
        vehicle.stop()


def test_link_negotiates_mavlink_2(linked_vehicle):
    """Fence and rally cannot be addressed at all under MAVLink 1."""
    from pymavlink import mavutil

    assert "mission_type" in mavutil.mavlink.MAVLink_mission_count_message.fieldnames


def test_mission_upload_is_acknowledged(linked_vehicle):
    client, vehicle = linked_vehicle
    items = [item.to_dict() for item in build_mission_items(PLAN)]

    result = client.upload_mission(items)

    assert result.success, result.message
    # One more than the plan: MAVLink reserves sequence 0 for home, and ArduPilot
    # overwrites whatever is stored there. This assertion used to read `== len(items)`
    # and passed against a mock that simply stored what it was given -- which is why
    # the plan's NAV_TAKEOFF sat in the home slot until SITL flew it.
    assert len(vehicle.stored[MAV_MISSION_TYPE_MISSION]) == len(items) + 1
    stored = vehicle.stored[MAV_MISSION_TYPE_MISSION]
    assert int(stored[1].command) == int(items[0]["command"])


def test_capture_commands_reach_the_vehicle(linked_vehicle):
    client, vehicle = linked_vehicle
    client.upload_mission([item.to_dict() for item in build_mission_items(PLAN)])

    commands = [int(m.command) for m in vehicle.stored[MAV_MISSION_TYPE_MISSION]]
    for expected in (
        MAV_CMD_DO_MOUNT_CONTROL,
        MAV_CMD_CONDITION_YAW,
        MAV_CMD_NAV_LOITER_TIME,
        MAV_CMD_DO_DIGICAM_CONTROL,
        MAV_CMD_NAV_RETURN_TO_LAUNCH,
    ):
        assert expected in commands, f"MAV_CMD {expected} did not survive upload"


def test_each_list_lands_in_its_own_slot(linked_vehicle):
    """Under MAVLink 1 the fence and rally uploads overwrote the flight plan."""
    client, vehicle = linked_vehicle

    report = client.upload_mission_plan(PLAN)

    assert report["mission"]["success"], report["mission"]["message"]
    assert report["fence"]["success"], report["fence"]["message"]
    assert report["rally"]["success"], report["rally"]["message"]

    # The report counts the plan; the vehicle also holds the reserved home item at
    # sequence 0. Fence and rally have no such reservation, so they match exactly.
    assert len(vehicle.stored[MAV_MISSION_TYPE_MISSION]) == report["mission"]["count"] + 1
    assert len(vehicle.stored[MAV_MISSION_TYPE_FENCE]) == report["fence"]["count"]
    assert len(vehicle.stored[MAV_MISSION_TYPE_RALLY]) == report["rally"]["count"]
    # The flight plan must still be there after the other two uploads.
    assert len(vehicle.stored[MAV_MISSION_TYPE_MISSION]) > 1


def test_download_round_trips_the_uploaded_mission(linked_vehicle):
    """What the vehicle reports back is the only trustworthy evidence."""
    client, _ = linked_vehicle
    sent = [item.to_dict() for item in build_mission_items(PLAN)]
    client.upload_mission(sent)

    stored = client.download_mission()

    assert len(stored) == len(sent)
    for original, returned in zip(sent, stored):
        assert original["command"] == returned["command"]


def test_positional_items_keep_their_coordinates(linked_vehicle):
    """A dwell turns a waypoint into NAV_LOITER_TIME, so match on any nav command."""
    client, _ = linked_vehicle
    # NAV_WAYPOINT, NAV_LOITER_TIME, NAV_TAKEOFF -- all carry real degrees in x/y.
    positional = {16, 19, 22}
    sent = [item.to_dict() for item in build_mission_items(PLAN)]
    client.upload_mission(sent)

    stored = client.download_mission()
    sent_positions = [i for i in sent if i["command"] in positional]
    back_positions = [i for i in stored if i["command"] in positional]

    assert sent_positions, "expected at least one positional nav command"
    assert len(sent_positions) == len(back_positions)
    for original, returned in zip(sent_positions, back_positions):
        assert returned["lat"] == pytest.approx(original["lat"], abs=1e-6)
        assert returned["lon"] == pytest.approx(original["lon"], abs=1e-6)
        # Coordinates must come back as degrees, not 1e7-scaled integers.
        assert -90.0 <= returned["lat"] <= 90.0
        assert -180.0 <= returned["lon"] <= 180.0


def test_non_positional_params_are_not_scaled_as_coordinates(linked_vehicle):
    """CONDITION_YAW carries plain numbers in x/y; scaling them by 1e7 corrupts them."""
    client, _ = linked_vehicle
    client.upload_mission([item.to_dict() for item in build_mission_items(PLAN)])

    stored = client.download_mission()
    yaw = next(i for i in stored if i["command"] == MAV_CMD_CONDITION_YAW)
    assert yaw["param1"] == pytest.approx(90.0)
    assert abs(yaw["lat"]) < 1e3, "x slot was scaled as if it were a latitude"


def test_empty_upload_is_refused(linked_vehicle):
    client, _ = linked_vehicle
    result = client.upload_mission([])
    assert not result.success
    assert "empty" in result.message.lower()
