"""A MAVLink peer that behaves like an ArduCopter at the protocol level.

This is the far side of the radio, and nothing more: a UDP socket speaking real MAVLink
2 through pymavlink. It is here so the application can be driven end to end -- the Api,
the session, the bridge and the UI above them -- against something that answers the way
an autopilot answers, including the ways an autopilot says no.

It is not an autopilot. It does not fly, it has no pre-arm checks and it models no
physics. What it does do, each behaviour switchable:

  * streams HEARTBEAT, SYS_STATUS, GPS_RAW_INT, GLOBAL_POSITION_INT and HOME_POSITION,
    or -- with stream_home=False, as ArduPilot does -- sends HOME_POSITION only when
    asked (MAV_CMD_REQUEST_MESSAGE 242 or MAV_CMD_GET_HOME_POSITION),
  * runs the request-driven mission protocol for mission, fence and rally lists, and
    can reject an upload with MISSION_ACK type 1 (generic error),
  * answers SET_MODE by changing mode, or ignores it the way an autopilot refusing a
    mode does,
  * answers COMMAND_LONG with COMMAND_ACK ACCEPTED or DENIED, or with nothing,
  * can fall silent, which is what a lost link looks like from the ground.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

COPTER_MODES = {
    "STABILIZE": 0, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5, "RTL": 6,
    "LAND": 9, "POSHOLD": 16, "BRAKE": 17,
}

MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_MISSION_START = 300
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_GET_HOME_POSITION = 410
MAV_CMD_REQUEST_MESSAGE = 512
MAVLINK_MSG_ID_HOME_POSITION = 242


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeVehicle:
    """One simulated ArduCopter endpoint sending to a ground station on `gcs_port`."""

    def __init__(
        self,
        gcs_port: int,
        *,
        accept_modes: bool = True,
        reject_upload: bool = False,
        command_ack: str | None = "accept",   # "accept" | "deny" | None (never acks)
        battery_remaining: int = 77,
        send_camera_information: bool = False,
        stream_home: bool = True,
        lat: float = 41.3042,
        lon: float = -81.7505,
        refuse_commands: frozenset[int] | set[int] = frozenset(),
    ):
        self.conn = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{gcs_port}", source_system=1, source_component=1)
        self.custom_mode = COPTER_MODES["STABILIZE"]
        self.armed = False
        self.accept_modes = accept_modes
        self.reject_upload = reject_upload
        self.command_ack = command_ack
        # Commands this vehicle answers MAV_RESULT_FAILED, as ArduCopter answers
        # MISSION_START with no usable mission.
        self.refuse_commands = set(refuse_commands)
        self.battery_remaining = battery_remaining
        self.send_camera_information = send_camera_information
        self.stream_home = stream_home
        # HOME_POSITION requests received, as (command, param1). Kept apart from
        # `commands`, which records what the operator asked the vehicle to do.
        self.home_requests: list[tuple[int, float]] = []
        # False models an autopilot before its first fix: no home to report yet.
        self.home_known = True
        self.stored: dict[int, list[Any]] = {0: [], 1: [], 2: []}
        self.commands: list[tuple[int, float, float]] = []
        self.mode_requests: list[int] = []
        self.lat, self.lon = lat, lon
        self._stop = threading.Event()
        self._silent = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "FakeVehicle":
        tx = threading.Thread(target=self._tx, daemon=True)
        tx.start()
        time.sleep(0.3)
        rx = threading.Thread(target=self._rx, daemon=True)
        rx.start()
        self._threads = [tx, rx]
        return self

    def go_silent(self) -> None:
        """Stop transmitting, as a vehicle out of radio range does."""
        self._silent.set()

    def resume(self) -> None:
        """Start transmitting again, as a vehicle back in radio range does."""
        self._silent.clear()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def mode_name(self) -> str:
        for name, number in COPTER_MODES.items():
            if number == self.custom_mode:
                return name
        return str(self.custom_mode)

    # -- transmit ----------------------------------------------------------------

    def _tx(self) -> None:
        m = self.conn.mav
        while not self._stop.is_set():
            if self._silent.is_set():
                time.sleep(0.1)
                continue
            try:
                base = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (0x80 if self.armed else 0)
                m.heartbeat_send(mavutil.mavlink.MAV_TYPE_QUADROTOR,
                                 mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, base,
                                 self.custom_mode, 3)
                present = 0b111  # gyro, accelerometer, magnetometer -- all healthy
                m.sys_status_send(present, present, present, 500, 16200, -1,
                                  self.battery_remaining, 0, 0, 0, 0, 0, 0)
                m.gps_raw_int_send(int(time.time() * 1e6), 3, int(self.lat * 1e7),
                                   int(self.lon * 1e7), 250000, 120, 65535, 0, 0, 11)
                m.global_position_int_send(0, int(self.lat * 1e7), int(self.lon * 1e7),
                                           250000, 12000, 100, 0, 0, 9000)
                if self.stream_home:
                    self._send_home()
                if self.send_camera_information:
                    m.camera_information_send(0, b"FakeCam".ljust(32, b"\0"),
                                              b"Model-1".ljust(32, b"\0"), 1, 8.8, 13.2, 8.8,
                                              5472, 3648, 0, 0b111, 0, b"")
            except OSError:
                pass
            time.sleep(0.2)

    def _send_home(self) -> None:
        self.conn.mav.home_position_send(int(self.lat * 1e7), int(self.lon * 1e7), 238000,
                                         0, 0, 0, [1, 0, 0, 0], 0, 0, 0)

    # -- receive -----------------------------------------------------------------

    def _ack(self, command: int, accepted: bool) -> None:
        if self.command_ack is None:
            return
        result = 0 if accepted and self.command_ack == "accept" else 2
        self.conn.mav.command_ack_send(command, result)

    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.3)
            except OSError:
                time.sleep(0.05)
                continue
            if msg is None or self._silent.is_set():
                continue
            kind = msg.get_type()
            if kind == "MISSION_COUNT":
                self._receive_list(int(msg.count), int(getattr(msg, "mission_type", 0)))
            elif kind == "SET_MODE":
                self.mode_requests.append(int(msg.custom_mode))
                if self.accept_modes:
                    self.custom_mode = int(msg.custom_mode)
            elif kind == "COMMAND_LONG" and (
                    int(msg.command) == MAV_CMD_GET_HOME_POSITION
                    or (int(msg.command) == MAV_CMD_REQUEST_MESSAGE
                        and int(msg.param1) == MAVLINK_MSG_ID_HOME_POSITION)):
                self.home_requests.append((int(msg.command), float(msg.param1)))
                if self.home_known:
                    self._send_home()
                self.conn.mav.command_ack_send(int(msg.command), 0)
            elif kind == "COMMAND_LONG":
                command = int(msg.command)
                self.commands.append((command, float(msg.param1), float(msg.param2)))
                accepted = self.command_ack == "accept" and command not in self.refuse_commands
                if accepted:
                    if command == MAV_CMD_COMPONENT_ARM_DISARM:
                        self.armed = bool(msg.param1)
                    elif command == MAV_CMD_MISSION_START:
                        self.custom_mode = COPTER_MODES["AUTO"]
                    elif command == MAV_CMD_NAV_RETURN_TO_LAUNCH:
                        self.custom_mode = COPTER_MODES["RTL"]
                    elif command == MAV_CMD_NAV_LAND:
                        self.custom_mode = COPTER_MODES["LAND"]
                self._ack(command, accepted)
            elif kind == "MISSION_CLEAR_ALL":
                mission_type = int(getattr(msg, "mission_type", 0))
                self.stored[mission_type] = []
                self.conn.mav.mission_ack_send(255, 0, 0, mission_type)
            elif kind == "MISSION_REQUEST_LIST":
                mission_type = int(getattr(msg, "mission_type", 0))
                self.conn.mav.mission_count_send(255, 0, len(self.stored[mission_type]), mission_type)
            elif kind == "MISSION_REQUEST_INT":
                mission_type = int(getattr(msg, "mission_type", 0))
                item = self.stored[mission_type][int(msg.seq)]
                self.conn.mav.mission_item_int_send(
                    255, 0, int(msg.seq), item.frame, item.command, 0, 1,
                    item.param1, item.param2, item.param3, item.param4,
                    item.x, item.y, item.z, mission_type)

    def _receive_list(self, count: int, mission_type: int) -> None:
        items = []
        for seq in range(count):
            self.conn.mav.mission_request_int_send(255, 0, seq, mission_type)
            deadline = time.time() + 3.0
            received = None
            while time.time() < deadline:
                message = self.conn.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=0.5)
                if message is not None and int(message.seq) == seq:
                    received = message
                    break
            if received is None:
                return
            items.append(received)
        if not self.reject_upload:
            self.stored[mission_type] = items
        self.conn.mav.mission_ack_send(255, 0, 1 if self.reject_upload else 0, mission_type)
