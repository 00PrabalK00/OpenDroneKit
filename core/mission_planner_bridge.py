"""Bridge to Mission Planner / QGroundControl GCS over MAVLink.

Acts as both a DroneClient (proxy through a forwarded MAVLink stream coming
out of Mission Planner) AND a file-level interchange (QGC WPL .waypoints
import/export, MAVLink connection autodiscovery).

This module is the one-stop integration with Mission Planner:

  * Connect to Mission Planner's "UDP/TCP output" forwarder.
  * Read live telemetry as a DroneTelemetry stream.
  * Upload mission items via MISSION_COUNT / MISSION_ITEM_INT.
  * Send flight commands: ARM, MODE, START, PAUSE, RESUME, RTL, LAND.
  * Read / write Mission Planner .waypoints files (QGC WPL 110).
  * Detect a Mission Planner instance running on this host.
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time

# `mission_type` is a MAVLink 2 extension field on every message of the mission
# transfer protocol. Under MAVLink 1 it is truncated off the wire and silently
# reads back as 0, so a fence or rally upload would be stored as the flight plan
# and overwrite it. pymavlink decides the wire version at import time from this
# variable, so it has to be set before the first `from pymavlink import ...`.
os.environ.setdefault("MAVLINK20", "1")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .drone import (
    ACK_ACCEPTED,
    ACK_DENIED,
    ACK_IN_PROGRESS,
    ACK_NONE,
    TELEMETRY_STALE_AFTER_S,
    CommandResult,
    DroneTelemetry,
)
from .errors import AppError, ERR_DRONE_NOT_CONNECTED, ERR_INVALID_INPUT


# Default endpoints used by Mission Planner forwarding presets.
DEFAULT_UDPIN_PORT = 14550        # Mission Planner default UDP target for Pixhawk telemetry
DEFAULT_TCP_PORT = 5760           # Mission Planner default TCP server when GCS is host
DEFAULT_FORWARD_UDP_PORTS = [14551, 14552, 14553, 14554, 14555, 14556]
DEFAULT_FORWARD_TCP_PORTS = [5762, 5763, 5764, 5765, 5766]

# MAV_MISSION_TYPE. One autopilot holds three independent item lists, selected by
# this field on every message of the transfer protocol. Fence and rally uploads are
# the same handshake as a mission upload with a different type.
MAV_MISSION_TYPE_MISSION = 0
MAV_MISSION_TYPE_FENCE = 1
MAV_MISSION_TYPE_RALLY = 2

# MAV_MISSION_RESULT
MAV_MISSION_ACCEPTED = 0

# MAV_RESULT, as carried by COMMAND_ACK. Only ACCEPTED and IN_PROGRESS mean the vehicle
# is doing what it was asked.
MAV_RESULT_NAMES = {
    0: "accepted",
    1: "temporarily rejected",
    2: "denied",
    3: "unsupported",
    4: "failed",
    5: "in progress",
    6: "cancelled",
}

# MAV_SYS_STATUS_SENSOR bit for the geofence in SYS_STATUS's present/enabled/health.
MAV_SYS_STATUS_GEOFENCE = 1 << 20

_NUL = chr(0)  # STATUSTEXT pads its fixed-length text with NULs

# SYS_STATUS sentinels for "this vehicle has no such sensor".
_BATTERY_REMAINING_UNKNOWN = -1
_VOLTAGE_UNKNOWN = 0xFFFF

# HOME_POSITION is not part of any ArduPilot telemetry stream. The autopilot sends it
# once, when home is set -- normally on the first GPS fix, long before a ground station
# connects -- and afterwards only when asked. A GCS that never asks never learns home,
# and the preflight "Home Position" check can never pass.
MAVLINK_MSG_ID_HOME_POSITION = 242
MAV_CMD_GET_HOME_POSITION = 410     # deprecated, still answered by older firmware
MAV_CMD_REQUEST_MESSAGE = 512
HOME_REQUEST_INTERVAL_S = 2.0

# This client is the vehicle's ground station, so it says so once a second. ArduPilot's
# GCS failsafe (FS_GCS_ENABLE) watches for HEARTBEATs from SYSID_MYGCS; a GCS that never
# sends one is never "seen", so losing its link triggers nothing and the aircraft flies
# on with nobody watching. Mission Planner and QGroundControl both send one at 1 Hz.
GCS_HEARTBEAT_INTERVAL_S = 1.0
MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8
MAV_AUTOPILOT_PX4 = 12
MAV_STATE_ACTIVE = 4

# PX4 speaks MAVLink too, but differently from ArduPilot in the places this link relied
# on: it has no home slot at mission sequence 0, encodes its mode as main/sub mode in
# custom_mode, is commanded into a mode with DO_SET_MODE, has no GUIDED, answers
# DO_PAUSE_CONTINUE with "unsupported", and takes NAV_TAKEOFF's altitude as AMSL. Each
# was found against PX4 v1.15.4 SITL; before it, this link uploaded a mission PX4 refused
# to fly ("takeoff not first waypoint item"), showed the mode as UNKNOWN, and could not
# change it.
#
# PX4 mode names (pymavlink's) in the vocabulary the rest of the toolkit speaks -- the
# same one the MAVSDK driver reports.
_PX4_MODE_NAMES = {"MISSION": "AUTO", "POSCTL": "POSHOLD", "ALTCTL": "ALT_HOLD",
                   "STABILIZED": "STABILIZE"}
# A mode asked for in that vocabulary, as PX4 calls it.
_PX4_MODE_REQUESTS = {"AUTO": "MISSION", "HOLD": "LOITER", "BRAKE": "LOITER",
                      "POSHOLD": "POSCTL", "ALT_HOLD": "ALTCTL", "STABILIZE": "STABILIZED",
                      "RETURN_TO_LAUNCH": "RTL"}

# ArduCopter modes the vehicle refuses to arm in from a MAVLink arm command.
_NOT_ARMABLE_ON_GROUND = {"RTL", "LAND", "SMART_RTL", "AUTO", "BRAKE", "AUTO_RTL",
                          "FLIP", "AUTOTUNE", "THROW"}

# How many of the vehicle's STATUSTEXT messages are kept for the operator to read.
VEHICLE_MESSAGES_KEPT = 30

_MISSION_RESULT_TEXT = {
    0: "accepted",
    1: "generic error",
    2: "unsupported frame",
    3: "unsupported command",
    4: "no space left on vehicle",
    5: "invalid parameter",
    6: "param1 invalid",
    7: "param2 invalid",
    8: "param3 invalid",
    9: "param4 invalid",
    10: "x/latitude invalid",
    11: "y/longitude invalid",
    12: "z/altitude invalid",
    13: "mission type not supported",
    14: "vehicle busy",
    15: "operation cancelled",
}

# Commands whose x/y fields carry plain numbers rather than degrees. Scaling these
# by 1e7 like a coordinate is the classic way to make an autopilot reject a mission,
# so the uploader checks membership here before converting.
_NON_POSITIONAL_COMMANDS = frozenset(
    {
        112,  # CONDITION_DELAY
        113,  # CONDITION_CHANGE_ALT
        114,  # CONDITION_DISTANCE
        115,  # CONDITION_YAW
        176,  # DO_SET_MODE
        177,  # DO_JUMP
        178,  # DO_CHANGE_SPEED
        181,  # DO_SET_RELAY
        183,  # DO_SET_SERVO
        203,  # DO_DIGICAM_CONTROL
        205,  # DO_MOUNT_CONTROL
        206,  # DO_SET_CAM_TRIGG_DIST
        530,  # SET_CAMERA_MODE
        531,  # SET_CAMERA_ZOOM
        532,  # SET_CAMERA_FOCUS
        1000,  # DO_GIMBAL_MANAGER_PITCHYAW: x is the flags bitmask
        1001,  # DO_GIMBAL_MANAGER_CONFIGURE
        2000,  # IMAGE_START_CAPTURE
        2001,  # IMAGE_STOP_CAPTURE
        2500,  # VIDEO_START_CAPTURE (was labelled gimbal-manager configure, which is 1001)
        2501,  # VIDEO_STOP_CAPTURE
        # DO_SET_ROI (201) is deliberately absent: its x/y ARE the ROI's latitude and
        # longitude. Listed here, they were truncated to whole degrees: a real
        # ArduCopter stored a -35.3631 ROI as -35.0, a point of interest ~40 km away.
    }
)


# ── Connection autodiscovery ──────────────────────────────────────────────────

def _tcp_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _udp_listener(port: int, listen_for_s: float = 0.5) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(listen_for_s)
        s.bind(("0.0.0.0", port))
        try:
            data, _ = s.recvfrom(2048)
            return bool(data)
        except socket.timeout:
            return False
        finally:
            s.close()
    except Exception:
        return False


@dataclass
class MissionPlannerEndpoint:
    transport: str           # "udp" | "tcp" | "serial"
    host: str = "127.0.0.1"
    port: int = DEFAULT_UDPIN_PORT
    device: str = ""
    baud: int = 115200
    description: str = ""

    def connection_uri(self) -> str:
        if self.transport == "udp":
            return f"udpin:{self.host}:{self.port}"
        if self.transport == "tcp":
            return f"tcp:{self.host}:{self.port}"
        if self.transport == "serial":
            return f"{self.device},{self.baud}"
        return ""


def discover_mission_planner_endpoints(host: str = "127.0.0.1") -> list[MissionPlannerEndpoint]:
    """Probe common Mission Planner forwarding ports on a host.

    Mission Planner can be configured (Ctrl-F → MAVLink → output) to forward
    its telemetry stream on UDP/TCP. This scan only looks for listeners; the
    actual MAVLink handshake is done after a user picks an endpoint.
    """
    out: list[MissionPlannerEndpoint] = []
    # TCP ports (Mission Planner often *listens* on TCP)
    for port in [DEFAULT_TCP_PORT, *DEFAULT_FORWARD_TCP_PORTS]:
        if _tcp_open(host, port):
            out.append(MissionPlannerEndpoint(
                transport="tcp",
                host=host,
                port=port,
                description="Mission Planner TCP server",
            ))
    # UDP ports — we listen briefly for the GCS HEARTBEAT
    for port in [DEFAULT_UDPIN_PORT, *DEFAULT_FORWARD_UDP_PORTS]:
        if _udp_listener(port, listen_for_s=0.4):
            out.append(MissionPlannerEndpoint(
                transport="udp",
                host="0.0.0.0",
                port=port,
                description="Mission Planner UDP output",
            ))
    return out


# ── QGC .waypoints (Mission Planner) file I/O ─────────────────────────────────

def _parse_qgc_wpl(text: str) -> list[dict[str, Any]]:
    """Parse QGroundControl WPL 110 (Mission Planner) waypoint file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("QGC WPL"):
        raise AppError(ERR_INVALID_INPUT, "Not a QGC waypoints file.",
                       recovery_action="Open File → Save WP File from Mission Planner.")
    items: list[dict[str, Any]] = []
    for raw in lines[1:]:
        parts = raw.split("\t") if "\t" in raw else raw.split()
        if len(parts) < 12:
            continue
        try:
            items.append({
                "seq": int(parts[0]),
                "current": int(parts[1]),
                "frame": int(parts[2]),
                "command": int(parts[3]),
                "param1": float(parts[4]),
                "param2": float(parts[5]),
                "param3": float(parts[6]),
                "param4": float(parts[7]),
                "lat": float(parts[8]),
                "lon": float(parts[9]),
                "alt": float(parts[10]),
                "autocontinue": int(parts[11]),
            })
        except ValueError:
            continue
    return items


def import_mission_planner_waypoints(file_path: Path | str) -> dict[str, Any]:
    """Read a Mission Planner / QGC .waypoints file.

    Returns dict with keys: `format`, `items`, and a flattened `waypoints`
    list of `[lon, lat, alt]` triplets ready for our MissionPlan model.
    """
    p = Path(file_path)
    if not p.exists():
        raise AppError(ERR_INVALID_INPUT, f"Waypoint file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    items = _parse_qgc_wpl(text)
    waypoints = [[it["lon"], it["lat"], it["alt"]] for it in items if it["command"] in (16, 22, 82)]
    return {
        "format": "QGC WPL 110",
        "source_path": str(p),
        "items": items,
        "waypoints": waypoints,
        "home": (items[0] if items and items[0].get("command") in (16, 22, 179) else None),
    }


def _with_home_slot(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend the home item MAVLink reserves at mission sequence 0, and renumber.

    ArduPilot stores its own home at sequence 0 and overwrites whatever arrives there.
    A plan uploaded without allowing for it loses its first command silently -- see
    upload_mission. The placeholder copies the first positional waypoint's coordinates
    so a vehicle that has not yet set home has something sane rather than Null Island;
    the autopilot replaces it either way.
    """
    if not items:
        return []
    first = next(
        (it for it in items if int(it.get("command", 16)) not in _NON_POSITIONAL_COMMANDS),
        items[0],
    )
    home = {
        "seq": 0,
        "command": 16,  # NAV_WAYPOINT, which is what ArduPilot reports home as
        "frame": 0,     # MAV_FRAME_GLOBAL: home altitude is AMSL, not relative
        "param1": 0.0,
        "param2": 0.0,
        "param3": 0.0,
        "param4": 0.0,
        "lat": float(first.get("lat", 0.0)),
        "lon": float(first.get("lon", 0.0)),
        "alt": 0.0,
        "autocontinue": 1,
        "current": 1,
    }
    renumbered = []
    for index, item in enumerate(items, start=1):
        copied = dict(item)
        copied["seq"] = index
        copied["current"] = 0
        renumbered.append(copied)
    return [home, *renumbered]


def _without_home_slot(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the vehicle's home item and renumber, so a download mirrors the upload.

    The caller uploaded a plan and should read back that plan. Home is protocol
    bookkeeping owned by the autopilot, not part of anyone's flight plan, and callers
    that want it have it on telemetry as home_lat/home_lon/home_alt.
    """
    if not items:
        return []
    return [dict(item, seq=index) for index, item in enumerate(items[1:])]


def export_mission_planner_waypoints(
    file_path: Path | str,
    waypoints: list[list[float]],
    default_altitude_m: float = 50.0,
    home: tuple[float, float, float] | None = None,
) -> Path:
    """Write a Mission Planner .waypoints file (QGC WPL 110)."""
    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["QGC WPL 110"]
    seq = 0
    if home is not None:
        h_lat, h_lon, h_alt = home
        lines.append(
            f"{seq}\t1\t0\t16\t0\t0\t0\t0\t{h_lat:.8f}\t{h_lon:.8f}\t{h_alt:.2f}\t1"
        )
        seq += 1
    for idx, row in enumerate(waypoints):
        lon = float(row[0])
        lat = float(row[1])
        alt = float(row[2]) if len(row) >= 3 else float(default_altitude_m)
        current = 1 if seq == 0 else 0
        lines.append(
            f"{seq}\t{current}\t3\t16\t0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{alt:.2f}\t1"
        )
        seq += 1
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── pymavlink-backed live client ──────────────────────────────────────────────

class MissionPlannerDroneClient:
    """Connects to a Mission Planner-forwarded MAVLink stream.

    Mission Planner setup:
        1. Connect Mission Planner to the autopilot as usual.
        2. Ctrl-F → MAVLink → "UDP" or "TCP" Add. Pick a port.
        3. Point OpenDroneKit at that host:port via this client.

    The same client also works against any MAVLink endpoint that pymavlink
    can speak (serial, udpin, udpout, tcp).
    """

    def __init__(self) -> None:
        self._conn = None
        self._connected = False
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._latest_telemetry = DroneTelemetry()
        self._lock = threading.RLock()
        self._uri = ""
        self._mission_uploaded_count = 0
        # The listener thread owns the socket, so an uploader cannot call recv_match
        # itself without stealing messages from telemetry. Protocol traffic is
        # forwarded here instead and consumed by whichever transfer is in flight.
        self._protocol_queue: queue.Queue = queue.Queue()
        self._transfer_lock = threading.Lock()
        # What the autopilot says while a transfer is in flight. A MISSION_ACK carries
        # only a result code; ArduPilot puts the reason ("Only 650 items are supported")
        # in a STATUSTEXT, and a rejection that drops it leaves the pilot guessing.
        self._transfer_texts: list[str] | None = None
        # The same for a command: ArduPilot answers a refused arm with MAV_RESULT_FAILED
        # and says why ("Arm: GPS 1: Bad fix") only in a STATUSTEXT.
        self._command_texts: list[str] | None = None
        self._telemetry_subscribers: list[Any] = []
        # COMMAND_ACKs, forwarded by the listener to whichever command is waiting.
        # Commands are serialised so an ack cannot be claimed by the wrong one.
        self._ack_queue: queue.Queue = queue.Queue()
        self._command_lock = threading.Lock()
        self.command_ack_timeout_s = 3.0
        # Link quality is measured from MAVLink sequence gaps -- the fraction of packets
        # that arrived -- because RADIO_STATUS exists only on SiK-style radios. A USB or
        # UDP link has none, and reporting 0 % for it said the link was dead.
        self._link_window: tuple[float, int, int] | None = None
        # monotonic time of the last HOME_POSITION request; asked again until it arrives.
        self._last_home_request = 0.0
        self._last_gcs_heartbeat = 0.0
        # MAV_AUTOPILOT of the vehicle, from its HEARTBEAT. Not every autopilot is
        # ArduPilot; PX4 differs in the ways listed at _PX4_MODE_NAMES.
        self._autopilot = 0

    def _px4(self) -> bool:
        return self._autopilot == MAV_AUTOPILOT_PX4

    # ── DroneClient protocol ─────────────────────────────────────────────────

    def connect(self, connection_uri: str = "udpin:0.0.0.0:14550") -> None:
        try:
            from pymavlink import mavutil
        except ImportError:
            raise AppError(
                "MAVLINK_MISSING",
                "pymavlink is not installed.",
                technical_message="ImportError on pymavlink. Install via `pip install pymavlink`.",
                recovery_action="Run: pip install pymavlink",
            )
        try:
            self._conn = mavutil.mavlink_connection(connection_uri, autoreconnect=True, source_system=255)
            # Wait for first HEARTBEAT to confirm link.
            hb = self._conn.wait_heartbeat(timeout=10)
            if hb is None:
                raise AppError(ERR_DRONE_NOT_CONNECTED,
                               f"No heartbeat from {connection_uri}.",
                               recovery_action="Verify Mission Planner is forwarding telemetry.")
            if "mission_type" not in mavutil.mavlink.MAVLink_mission_count_message.fieldnames:
                raise AppError(
                    ERR_DRONE_NOT_CONNECTED,
                    "MAVLink 1 link: geofence and rally upload are not supported.",
                    technical_message=(
                        "pymavlink resolved to the MAVLink 1 dialect, which truncates the "
                        "mission_type field. Set MAVLINK20=1 before pymavlink is first imported."
                    ),
                    recovery_action="Set the MAVLINK20=1 environment variable and restart.",
                )
            self._autopilot = int(getattr(hb, "autopilot", 0) or 0)
            self._connected = True
            self._uri = connection_uri
            self._send_gcs_heartbeat()
            self._request_home_position()
            self._start_listener()
        except AppError:
            raise
        except Exception as exc:
            raise AppError(ERR_DRONE_NOT_CONNECTED,
                           f"Failed to connect to Mission Planner stream at {connection_uri}.",
                           technical_message=str(exc),
                           recovery_action="Check host/port and Mission Planner output config.")

    def disconnect(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._conn is not None

    def get_telemetry(self) -> DroneTelemetry:
        with self._lock:
            return self._latest_telemetry

    # ── Listener thread ──────────────────────────────────────────────────────

    def _start_listener(self) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._listener_loop, name="mp-mavlink-listener", daemon=True
        )
        self._heartbeat_thread.start()

    def _listener_loop(self) -> None:
        last_publish = 0.0
        while not self._heartbeat_stop.is_set() and self._conn is not None:
            try:
                msg = self._conn.recv_match(blocking=True, timeout=1.0)
                if msg is not None:
                    self._absorb_message(msg)
                if time.monotonic() - self._last_gcs_heartbeat >= GCS_HEARTBEAT_INTERVAL_S:
                    self._send_gcs_heartbeat()
                if (not self._latest_telemetry.home_set
                        and time.monotonic() - self._last_home_request >= HOME_REQUEST_INTERVAL_S):
                    self._request_home_position()
                # Telemetry arrives far faster than any UI needs it, so subscribers
                # get a snapshot at a fixed rate rather than one event per message.
                now = time.monotonic()
                if now - last_publish >= 0.25:
                    last_publish = now
                    with self._lock:
                        snapshot = self._latest_telemetry
                    self._publish({"type": "telemetry", "telemetry": snapshot.to_dict()
                                   if hasattr(snapshot, "to_dict") else vars(snapshot)})
            except Exception:
                time.sleep(0.5)

    def _absorb_message(self, msg) -> None:
        t = msg.get_type()
        if t in {"MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK", "MISSION_COUNT", "MISSION_ITEM_INT"}:
            self._protocol_queue.put(msg)
        if t == "COMMAND_ACK":
            self._ack_queue.put(msg)
        if t == "STATUSTEXT":
            if int(getattr(msg, "severity", 6)) <= 4:
                text = str(getattr(msg, "text", "")).strip(_NUL).strip()
                for texts in (self._transfer_texts, self._command_texts):
                    if texts is not None:
                        texts.append(text)
            entry = {"time": time.time(), "severity": int(getattr(msg, "severity", 6)),
                     "text": str(getattr(msg, "text", "")).strip(_NUL).strip()}
            with self._lock:
                # What the autopilot said, kept so the operator can read why it did what
                # it did ("Battery 1 is low", "GCS Failsafe", "EKF variance"). These
                # were published to subscribers and then lost: the API had no way to say
                # why the aircraft had just switched itself to RTL.
                kept = list(self._latest_telemetry.raw.get("vehicle_messages", ()))
                kept.append(entry)
                self._latest_telemetry.raw["vehicle_messages"] = kept[-VEHICLE_MESSAGES_KEPT:]
            self._publish({"type": "statustext", "severity": entry["severity"],
                           "text": entry["text"]})
        with self._lock:
            tel = self._latest_telemetry
            tel.connected = True
            now = time.time()
            if now - float(tel.timestamp or 0.0) > TELEMETRY_STALE_AFTER_S:
                # The link is back after a gap. What the vehicle said before the gap is
                # not current: any message refreshes the timestamp, so without this the
                # first GPS or attitude packet after a GCS failsafe relabelled the old
                # "AUTO" as live while the aircraft was flying RTL (SITL). Each field
                # comes back when the vehicle next reports it.
                tel.flight_mode = "UNKNOWN"
                for key in ("system_status", "fence_status", "battery_charge_state"):
                    tel.raw.pop(key, None)
            tel.timestamp = now
            if t == "HEARTBEAT":
                # custom_mode is a number. Reporting it as the flight mode gives the
                # operator "3" where they need "AUTO", and makes it impossible to say
                # who is flying the aircraft, so it is resolved to its name here.
                tel.flight_mode = self._mode_name(getattr(msg, "custom_mode", None),
                                                  getattr(msg, "base_mode", 0)) or tel.flight_mode
                tel.armed = bool(getattr(msg, "base_mode", 0) & 0x80)
                # MAV_STATE. ArduPilot reports CRITICAL while any failsafe is active
                # (radio, GCS, battery, EKF, ...) and EMERGENCY when it has given up.
                tel.raw["system_status"] = int(getattr(msg, "system_status", 0) or 0)
            elif t == "GLOBAL_POSITION_INT":
                tel.latitude = float(msg.lat) / 1e7
                tel.longitude = float(msg.lon) / 1e7
                tel.altitude_rel_m = float(msg.relative_alt) / 1000.0
                tel.altitude_abs_m = float(msg.alt) / 1000.0
                tel.heading_deg = float(getattr(msg, "hdg", 0) or 0) / 100.0
                tel.speed_mps = (float(msg.vx) ** 2 + float(msg.vy) ** 2) ** 0.5 / 100.0
            elif t == "CAMERA_INFORMATION":
                # Kept so camera control can gate commands on what the payload says it
                # can do, instead of transmitting into silence.
                tel.raw["camera_information"] = {
                    "flags": int(getattr(msg, "flags", 0) or 0),
                    "vendor_name": getattr(msg, "vendor_name", b""),
                    "model_name": getattr(msg, "model_name", b""),
                }
            elif t == "SYS_STATUS":
                # -1 and UINT16_MAX are how MAVLink says "no such sensor". Clamping -1 to
                # 0 put an empty battery on screen for every flight controller without
                # a current sensor.
                voltage = int(getattr(msg, "voltage_battery", _VOLTAGE_UNKNOWN))
                tel.battery_v = None if voltage == _VOLTAGE_UNKNOWN else voltage / 1000.0
                remaining = int(getattr(msg, "battery_remaining", _BATTERY_REMAINING_UNKNOWN))
                tel.battery_pct = (None if remaining == _BATTERY_REMAINING_UNKNOWN
                                   else max(0.0, min(100.0, float(remaining))))
                # Sensor health bitmasks, kept so preflight can report compass, gyro
                # and accelerometer state instead of leaving them unchecked. Stored
                # raw: interpreting the bits belongs with the checks, not the parser.
                tel.raw["sensors_present"] = int(
                    getattr(msg, "onboard_control_sensors_present", 0) or 0)
                tel.raw["sensors_enabled"] = int(
                    getattr(msg, "onboard_control_sensors_enabled", 0) or 0)
                tel.raw["sensors_health"] = int(
                    getattr(msg, "onboard_control_sensors_health", 0) or 0)
            elif t == "FENCE_STATUS":
                # The vehicle's own word on the fence: breached or not, how often, and
                # which limit. Without it a breach was visible only as an unexplained RTL.
                tel.raw["fence_status"] = {
                    "breach_status": int(getattr(msg, "breach_status", 0) or 0),
                    "breach_count": int(getattr(msg, "breach_count", 0) or 0),
                    "breach_type": int(getattr(msg, "breach_type", 0) or 0),
                }
            elif t == "BATTERY_STATUS":
                if int(getattr(msg, "id", 0) or 0) == 0:
                    # MAV_BATTERY_CHARGE_STATE. ArduPilot sets LOW / CRITICAL when its
                    # battery failsafe triggers -- on voltage or mAh, either of which
                    # can fire long before the percentage looks alarming.
                    tel.raw["battery_charge_state"] = int(getattr(msg, "charge_state", 0) or 0)
            elif t == "GPS_RAW_INT":
                tel.raw["gps_received_at"] = time.time()
                tel.gps_fix = int(msg.fix_type)
                tel.satellites = int(getattr(msg, "satellites_visible", 0))
                # eph is HDOP x 100, and UINT16_MAX when the receiver does not know.
                eph = float(getattr(msg, "eph", 0xFFFF))
                tel.hdop = eph / 100.0 if eph < 0xFFFF else None
            elif t == "MISSION_CURRENT":
                # Reported in the plan's own numbering, not the vehicle's. ArduPilot
                # counts from its reserved home item at sequence 0, so its indices run
                # one ahead of the plan the operator uploaded. download_mission already
                # hides that slot; a waypoint_index that did not would mean the API
                # answered "which waypoint" in two different numbering schemes
                # depending on which call you asked -- and the one place that shows up
                # is an operator watching progress against the plan they drew.
                # PX4 has no home slot, so its sequence is the plan's own.
                slot = 0 if self._px4() else 1
                tel.waypoint_index = max(0, int(getattr(msg, "seq", tel.waypoint_index + slot)) - slot)
            elif t == "MISSION_COUNT":
                # Only the flight plan's count is progress, and it includes the home slot
                # the plan never had. A fence or rally download also answers MISSION_COUNT
                # and used to overwrite the waypoint total with its vertex count.
                if int(getattr(msg, "mission_type", MAV_MISSION_TYPE_MISSION) or 0) == MAV_MISSION_TYPE_MISSION:
                    tel.waypoint_total = max(0, int(getattr(msg, "count", 1))
                                             - (0 if self._px4() else 1))
            elif t == "HOME_POSITION":
                tel.home_lat = float(msg.latitude) / 1e7
                tel.home_lon = float(msg.longitude) / 1e7
                tel.home_alt = float(msg.altitude) / 1000.0
                tel.home_set = True
            elif t == "RADIO_STATUS":
                # Kept raw. "100 - remnoise" was reported as link quality, but remnoise is
                # the remote noise floor, not a fraction of anything.
                tel.raw["radio_status"] = {
                    "rssi": int(getattr(msg, "rssi", 0) or 0),
                    "remrssi": int(getattr(msg, "remrssi", 0) or 0),
                    "noise": int(getattr(msg, "noise", 0) or 0),
                    "remnoise": int(getattr(msg, "remnoise", 0) or 0),
                }
            self._update_link_quality(tel)

    def _send_gcs_heartbeat(self) -> None:
        """Tell the vehicle its ground station is here (fire and forget)."""
        self._last_gcs_heartbeat = time.monotonic()
        conn = self._conn
        if conn is None:
            return
        try:
            conn.mav.heartbeat_send(MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, 0, 0, MAV_STATE_ACTIVE)
        except Exception:
            # Retried next second; a send that keeps failing is a lost link, which the
            # telemetry age already reports.
            pass

    def _request_home_position(self) -> None:
        """Ask the vehicle for HOME_POSITION (fire and forget; the listener parses it).

        MAV_CMD_REQUEST_MESSAGE is the current way; MAV_CMD_GET_HOME_POSITION is kept
        for firmware that predates it. Until home is set the vehicle has nothing to
        send, so the listener repeats the request until a HOME_POSITION arrives.
        The ACKs are ignored here: _command_long only accepts an ack for its own
        command, so they cannot be mistaken for the answer to an operator command.
        """
        self._last_home_request = time.monotonic()
        conn = self._conn
        if conn is None:
            return
        try:
            conn.mav.command_long_send(conn.target_system, conn.target_component,
                                       MAV_CMD_REQUEST_MESSAGE, 0,
                                       MAVLINK_MSG_ID_HOME_POSITION, 0, 0, 0, 0, 0, 0)
            conn.mav.command_long_send(conn.target_system, conn.target_component,
                                       MAV_CMD_GET_HOME_POSITION, 0, 0, 0, 0, 0, 0, 0, 0)
        except Exception:
            # A failed send is retried on the next interval; the link layer reports loss.
            pass

    def _update_link_quality(self, tel: DroneTelemetry, window_s: float = 2.0) -> None:
        """Percentage of MAVLink packets that arrived, over the last couple of seconds.

        pymavlink counts received packets and sequence-number gaps on the link; their
        ratio is the packet delivery rate, which is what Mission Planner shows as link
        quality. Until a window has enough packets the field stays unreported.
        """
        conn = self._conn
        if conn is None:
            return
        count = int(getattr(conn, "mav_count", 0) or 0)
        loss = int(getattr(conn, "mav_loss", 0) or 0)
        now = time.monotonic()
        if self._link_window is None:
            self._link_window = (now, count, loss)
            return
        started, count0, loss0 = self._link_window
        if now - started < window_s:
            return
        received, lost = count - count0, loss - loss0
        if received + lost >= 5:
            tel.link_quality_pct = 100.0 * received / float(received + lost)
        self._link_window = (now, count, loss)

    # ── Mission upload / commands ────────────────────────────────────────────

    def _ensure(self):
        if not self.is_connected() or self._conn is None:
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Mission Planner link not connected.")
        return self._conn

    # ── Telemetry subscribers ────────────────────────────────────────────────

    def subscribe(self, callback) -> None:
        """Register a callable invoked with each telemetry/status event.

        The desktop shell uses this to push live telemetry into the UI without
        polling. Callbacks run on the listener thread, so they must not block.
        """
        with self._lock:
            if callback not in self._telemetry_subscribers:
                self._telemetry_subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        with self._lock:
            if callback in self._telemetry_subscribers:
                self._telemetry_subscribers.remove(callback)

    def _publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._telemetry_subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # A broken subscriber must not take the MAVLink listener down with it.
                pass

    # ── Mission transfer protocol ────────────────────────────────────────────

    def _drain_protocol_queue(self) -> None:
        while True:
            try:
                self._protocol_queue.get_nowait()
            except queue.Empty:
                return

    def _await_protocol(self, types: set[str], timeout: float):
        """Wait for one of `types` off the listener's protocol queue."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = self._protocol_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if msg.get_type() in types:
                return msg

    def _send_item(self, conn, item: dict[str, Any], seq: int, mission_type: int) -> None:
        command = int(item.get("command", 16))  # NAV_WAYPOINT
        frame = int(item.get("frame", 3))       # MAV_FRAME_GLOBAL_RELATIVE_ALT
        if command in _NON_POSITIONAL_COMMANDS:
            # These carry plain values in x/y, not degrees; MISSION_ITEM_INT still
            # transports them as int32, so they are passed through unscaled.
            x = int(float(item.get("lat", item.get("x", 0.0))))
            y = int(float(item.get("lon", item.get("y", 0.0))))
        else:
            x = int(round(float(item.get("lat", 0.0)) * 1e7))
            y = int(round(float(item.get("lon", 0.0)) * 1e7))
        conn.mav.mission_item_int_send(
            conn.target_system,
            conn.target_component,
            seq,
            frame,
            command,
            int(item.get("current", 1 if seq == 0 else 0)),
            int(item.get("autocontinue", 1)),
            float(item.get("param1", 0.0)),
            float(item.get("param2", 0.0)),
            float(item.get("param3", 0.0)),
            float(item.get("param4", 0.0)),
            x,
            y,
            float(item.get("alt", 0.0)),
            mission_type,
        )

    def _upload_list(
        self,
        items: list[dict[str, Any]],
        *,
        mission_type: int,
        name: str,
        timeout_s: float = 20.0,
    ) -> CommandResult:
        """Run the MAVLink mission transfer handshake for one item list.

        The protocol is request-driven: the GCS announces a count, then sends each
        item only when the vehicle asks for it by sequence number. The previous
        implementation pushed all items immediately after MISSION_COUNT, which
        ArduPilot discards -- and because nothing waited for MISSION_ACK, the upload
        reported success regardless of what the vehicle actually stored.
        """
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, name, exc.user_message)

        if not items:
            return CommandResult(False, name, "Nothing to upload: the item list is empty.")

        if not self._transfer_lock.acquire(timeout=timeout_s):
            return CommandResult(False, name, "Another mission transfer is already in progress.")
        self._transfer_texts = []
        try:
            self._drain_protocol_queue()
            conn.mav.mission_count_send(
                conn.target_system, conn.target_component, len(items), mission_type
            )

            sent: set[int] = set()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                msg = self._await_protocol(
                    {"MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"},
                    timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                )
                if msg is None:
                    continue

                if msg.get_type() == "MISSION_ACK":
                    result = int(getattr(msg, "type", 1))
                    if result == MAV_MISSION_ACCEPTED:
                        if mission_type == MAV_MISSION_TYPE_MISSION:
                            self._mission_uploaded_count = len(items)
                        return CommandResult(
                            True, name, f"Vehicle accepted {len(items)} items."
                        )
                    reason = _MISSION_RESULT_TEXT.get(result, f"result {result}")
                    time.sleep(0.2)  # the STATUSTEXT may trail the ACK on the link
                    said = "; ".join(dict.fromkeys(t for t in (self._transfer_texts or []) if t))
                    detail = f" The vehicle said: {said}." if said else ""
                    where = (f" at item {max(sent)} of {len(items)}"
                             if sent and result != 4 else "")
                    return CommandResult(
                        False, name,
                        f"Vehicle rejected the upload{where}: {reason}.{detail}"
                        + (f" The upload was {len(items)} items, more than the autopilot's "
                           "mission storage holds: split the mission, or give the vehicle "
                           "more storage (ArduPilot: BRD_SD_MISSION)."
                           if result == 4 else ""),
                    )

                seq = int(getattr(msg, "seq", 0))
                if not 0 <= seq < len(items):
                    return CommandResult(
                        False, name, f"Vehicle requested out-of-range item {seq}."
                    )
                self._send_item(conn, items[seq], seq, mission_type)
                sent.add(seq)
                # Each request is fresh progress, so the window extends with it and
                # only a genuinely silent link times out.
                deadline = time.monotonic() + timeout_s

            return CommandResult(
                False,
                name,
                f"Timed out after sending {len(sent)}/{len(items)} items with no MISSION_ACK.",
            )
        except Exception as exc:
            return CommandResult(False, name, str(exc))
        finally:
            self._transfer_texts = None
            self._transfer_lock.release()

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        """Upload the flight plan, including gimbal, yaw, dwell, and trigger items.

        A home slot is prepended, because MAVLink reserves mission sequence 0 for the
        home position and ArduPilot overwrites whatever is stored there with the
        vehicle's own home. Without it the first real command is swallowed: SITL showed
        a plan whose NAV_TAKEOFF (22) was sent at seq 0 and read back as NAV_WAYPOINT
        (16). Nothing errored. The upload reported success, the item count matched, and
        the aircraft simply would not have taken off -- it would have proceeded to the
        first waypoint at whatever altitude it happened to be at.

        This is the defect the SITL harness was built to find, and it was invisible to
        every test that did not talk to a real autopilot.
        """
        # PX4 keeps home out of the mission. Given ArduPilot's home slot it flies it as the
        # first item, and its feasibility check refused the plan outright: "Mission
        # rejected: takeoff not first waypoint item" (PX4 v1.15.4 SITL).
        result = self._upload_list(
            [dict(item, seq=index) for index, item in enumerate(mission_items)]
            if self._px4() else _with_home_slot(mission_items),
            mission_type=MAV_MISSION_TYPE_MISSION,
            name="upload_mission",
        )
        if result.success:
            # The vehicle acknowledged this many plan items, so progress can be shown
            # against them; MISSION_COUNT only arrives when the mission is downloaded.
            with self._lock:
                self._latest_telemetry.waypoint_total = len(mission_items)
        return result

    # Fence and rally have no home slot -- the reservation is specific to the mission
    # mission-type -- so they upload their items unchanged.

    def upload_geofence(self, fence_items: list[dict[str, Any]]) -> CommandResult:
        """Upload inclusion/exclusion fence polygons as MAV_MISSION_TYPE_FENCE."""
        return self._upload_list(
            fence_items, mission_type=MAV_MISSION_TYPE_FENCE, name="upload_geofence"
        )

    def upload_rally_points(self, rally_items: list[dict[str, Any]]) -> CommandResult:
        """Upload rally / emergency landing points as MAV_MISSION_TYPE_RALLY."""
        return self._upload_list(
            rally_items, mission_type=MAV_MISSION_TYPE_RALLY, name="upload_rally_points"
        )

    def upload_mission_plan(self, mission_plan: Any) -> dict[str, Any]:
        """Upload a planner MissionPlan in full: waypoints, then fence, then rally.

        Fence and rally are optional -- a plan with no geofence simply skips them,
        and a vehicle that does not support those mission types reports the rejection
        rather than failing the whole upload.
        """
        from mission.exporters import build_fence_items, build_mission_items, build_rally_items

        report: dict[str, Any] = {"mission": None, "fence": None, "rally": None}

        items = [item.to_dict() for item in build_mission_items(mission_plan)]
        mission_result = self.upload_mission(items)
        report["mission"] = {
            "success": mission_result.success,
            "message": mission_result.message,
            "count": len(items),
        }
        if not mission_result.success:
            return report

        fence = [item.to_dict() for item in build_fence_items(mission_plan)]
        if fence:
            fence_result = self.upload_geofence(fence)
            report["fence"] = {
                "success": fence_result.success,
                "message": fence_result.message,
                "count": len(fence),
            }

        rally = [item.to_dict() for item in build_rally_items(mission_plan)]
        if rally:
            rally_result = self.upload_rally_points(rally)
            report["rally"] = {
                "success": rally_result.success,
                "message": rally_result.message,
                "count": len(rally),
            }

        return report

    def _download_list(self, mission_type: int, timeout_s: float) -> list[dict[str, Any]]:
        """Read one of the vehicle's item lists (mission, fence or rally) back.

        What the vehicle reports is the only trustworthy evidence that it stored what
        was sent. A vehicle that refuses the request (MISSION_ACK with an error, e.g. a
        mission type it does not support) is reported as that refusal rather than as a
        silent timeout.
        """
        conn = self._ensure()
        if not self._transfer_lock.acquire(timeout=timeout_s):
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Another mission transfer is in progress.")
        try:
            self._drain_protocol_queue()
            conn.mav.mission_request_list_send(
                conn.target_system, conn.target_component, mission_type
            )
            count_msg = self._await_protocol({"MISSION_COUNT", "MISSION_ACK"}, timeout=timeout_s)
            if count_msg is None:
                raise AppError(ERR_DRONE_NOT_CONNECTED, "Vehicle did not report a mission count.")
            if count_msg.get_type() == "MISSION_ACK":
                result = int(getattr(count_msg, "type", 1))
                raise AppError(
                    ERR_DRONE_NOT_CONNECTED,
                    "Vehicle refused the download: "
                    f"{_MISSION_RESULT_TEXT.get(result, f'result {result}')}.")

            total = int(getattr(count_msg, "count", 0))
            items: list[dict[str, Any]] = []
            for seq in range(total):
                item = None
                # A lost request or reply is retried, as the protocol expects of a GCS.
                for _attempt in range(3):
                    conn.mav.mission_request_int_send(
                        conn.target_system, conn.target_component, seq, mission_type
                    )
                    deadline = time.monotonic() + min(timeout_s, 3.0)
                    while time.monotonic() < deadline:
                        candidate = self._await_protocol(
                            {"MISSION_ITEM_INT"}, timeout=max(0.05, deadline - time.monotonic()))
                        if candidate is not None and int(candidate.seq) == seq:
                            item = candidate
                            break
                    if item is not None:
                        break
                if item is None:
                    raise AppError(
                        ERR_DRONE_NOT_CONNECTED, f"Vehicle stopped responding at item {seq}."
                    )
                positional = int(item.command) not in _NON_POSITIONAL_COMMANDS
                items.append(
                    {
                        "seq": int(item.seq),
                        "command": int(item.command),
                        "frame": int(item.frame),
                        "param1": float(item.param1),
                        "param2": float(item.param2),
                        "param3": float(item.param3),
                        "param4": float(item.param4),
                        "lat": float(item.x) / 1e7 if positional else float(item.x),
                        "lon": float(item.y) / 1e7 if positional else float(item.y),
                        "alt": float(item.z),
                        "autocontinue": int(item.autocontinue),
                    }
                )
            conn.mav.mission_ack_send(
                conn.target_system,
                conn.target_component,
                MAV_MISSION_ACCEPTED,
                mission_type,
            )
            return items
        finally:
            self._transfer_lock.release()

    def download_mission(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        """Read the flight plan stored on the vehicle, without its home slot."""
        items = self._download_list(MAV_MISSION_TYPE_MISSION, timeout_s)
        return items if self._px4() else _without_home_slot(items)

    def _clear_list(self, mission_type: int, name: str, timeout_s: float = 5.0) -> CommandResult:
        """MISSION_CLEAR_ALL for one list, confirmed by the vehicle's MISSION_ACK."""
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, name, exc.user_message)
        if not self._transfer_lock.acquire(timeout=timeout_s):
            return CommandResult(False, name, "Another mission transfer is already in progress.")
        try:
            self._drain_protocol_queue()
            conn.mav.mission_clear_all_send(conn.target_system, conn.target_component,
                                            mission_type)
            ack = self._await_protocol({"MISSION_ACK"}, timeout=timeout_s)
            if ack is None:
                return CommandResult(False, name, "The vehicle did not confirm the clear.")
            result = int(getattr(ack, "type", 1))
            if result != MAV_MISSION_ACCEPTED:
                return CommandResult(False, name, "Vehicle refused to clear: "
                                     f"{_MISSION_RESULT_TEXT.get(result, f'result {result}')}.")
            return CommandResult(True, name, "Cleared on the vehicle.")
        finally:
            self._transfer_lock.release()

    def clear_geofence(self) -> CommandResult:
        """Remove every fence item from the vehicle (MAV_MISSION_TYPE_FENCE)."""
        return self._clear_list(MAV_MISSION_TYPE_FENCE, "clear_geofence")

    def clear_rally_points(self) -> CommandResult:
        """Remove every rally point from the vehicle (MAV_MISSION_TYPE_RALLY)."""
        return self._clear_list(MAV_MISSION_TYPE_RALLY, "clear_rally_points")

    def download_geofence(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        """Read the fence polygons stored on the vehicle (MAV_MISSION_TYPE_FENCE)."""
        return self._download_list(MAV_MISSION_TYPE_FENCE, timeout_s)

    def download_rally_points(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        """Read the rally points stored on the vehicle (MAV_MISSION_TYPE_RALLY)."""
        return self._download_list(MAV_MISSION_TYPE_RALLY, timeout_s)

    def _command_long(self, command: int, *params: float, name: str = "command") -> CommandResult:
        """Send a COMMAND_LONG and report what the vehicle answered.

        Transmitting is not the same as being obeyed. Every command used to come back
        "Command sent." whether the autopilot accepted it, denied it -- RTL with no home,
        arming with a failed pre-arm check -- or never heard it. The answer is now the
        vehicle's COMMAND_ACK, and silence is reported as silence.
        """
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, name, exc.user_message)
        with self._command_lock:
            while True:
                try:
                    self._ack_queue.get_nowait()
                except queue.Empty:
                    break
            self._command_texts = []
            try:
                return self._await_command_ack(conn, command, params, name)
            finally:
                self._command_texts = None

    def _command_int(self, command: int, frame: int, params: tuple[float, float, float, float],
                     x: int, y: int, z: float, name: str = "command") -> CommandResult:
        """Send a COMMAND_INT and report the vehicle's COMMAND_ACK, as _command_long does."""
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, name, exc.user_message)
        with self._command_lock:
            while True:
                try:
                    self._ack_queue.get_nowait()
                except queue.Empty:
                    break
            self._command_texts = []
            try:
                def send() -> None:
                    conn.mav.command_int_send(conn.target_system, conn.target_component, frame,
                                              command, 0, 0, *params, int(x), int(y), float(z))
                return self._await_command_ack(conn, command, (), name, send=send)
            finally:
                self._command_texts = None

    def _await_command_ack(self, conn, command: int, params: tuple, name: str,
                           send=None) -> CommandResult:
        """Send one COMMAND_LONG (or `send`) and wait for its ack. Runs under _command_lock."""
        try:
            if send is not None:
                send()
            else:
                p = list(params) + [0.0] * (7 - len(params))
                conn.mav.command_long_send(
                    conn.target_system, conn.target_component, command, 0,
                    *p[:7],
                )
        except Exception as exc:
            return CommandResult(False, name, str(exc))

        timeout = max(0.1, float(self.command_ack_timeout_s))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return CommandResult(
                    False, name,
                    f"Sent, but the vehicle did not acknowledge it within {timeout:.0f} s. "
                    "It may not have been received -- check the aircraft before "
                    "assuming it acted.",
                    ack=ACK_NONE,
                )
            try:
                ack = self._ack_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if int(getattr(ack, "command", -1)) != int(command):
                continue
            code = int(getattr(ack, "result", -1))
            label = MAV_RESULT_NAMES.get(code, f"result {code}")
            if code == 0:
                return CommandResult(True, name, "Accepted by the vehicle.", ack=ACK_ACCEPTED)
            if code == 5:
                return CommandResult(True, name,
                                     "Accepted; the vehicle reports it in progress.",
                                     ack=ACK_IN_PROGRESS)
            time.sleep(0.2)  # the reason may trail the ACK on the link
            said = "; ".join(dict.fromkeys(t for t in (self._command_texts or []) if t))
            return CommandResult(False, name, f"The vehicle answered: {label}."
                                 + (f" It said: {said}." if said else ""),
                                 ack=ACK_DENIED)

    # ── Flight commands (the set a ground station is expected to offer) ──────

    def arm(self, force: bool = False) -> CommandResult:
        """Arm the motors. `force` skips the autopilot's own arming checks.

        Forcing is offered because a legitimate ground test sometimes needs it, but it
        is never the default: the pre-arm checks exist to catch a bad compass or a
        missing GPS fix before the aircraft is in the air.
        """
        # MAV_CMD_COMPONENT_ARM_DISARM = 400. param2 = 21196 is the documented
        # "force" magic number.
        return self._command_long(400, 1.0, 21196.0 if force else 0.0, name="arm")

    def disarm(self, force: bool = False) -> CommandResult:
        return self._command_long(400, 0.0, 21196.0 if force else 0.0, name="disarm")

    def takeoff(self, altitude_m: float) -> CommandResult:
        """Climb to `altitude_m` above the home position.

        ArduPilot requires GUIDED before it will accept a takeoff command, so the mode
        is set first rather than leaving the caller to discover the ordering.
        """
        if altitude_m <= 0:
            return CommandResult(False, "takeoff", "Takeoff altitude must be positive.")
        if self._px4():
            # PX4 has no GUIDED: NAV_TAKEOFF itself puts it in TAKEOFF mode. Its param7 is
            # an altitude above mean sea level, not above home.
            with self._lock:
                home_set = self._latest_telemetry.home_set
                home_alt = float(self._latest_telemetry.home_alt)
            if not home_set:
                return CommandResult(False, "takeoff",
                                     "PX4 takes off to an altitude above sea level, and the "
                                     "vehicle has not reported its home altitude yet.")
            nan = float("nan")
            return self._command_long(22, -1.0, 0.0, 0.0, nan, nan, nan,
                                      home_alt + float(altitude_m), name="takeoff")
        mode = self.set_flight_mode("GUIDED")
        if not mode.success:
            return CommandResult(False, "takeoff", f"Could not enter GUIDED: {mode.message}")
        # MAV_CMD_NAV_TAKEOFF = 22, altitude in param7.
        return self._command_long(22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(altitude_m),
                                  name="takeoff")

    def land(self) -> CommandResult:
        """Land at the current position. MAV_CMD_NAV_LAND = 21."""
        return self._command_long(21, name="land")

    def goto(self, latitude: float, longitude: float, altitude_m: float) -> CommandResult:
        """Fly to a position in GUIDED mode, as the vehicle confirms it.

        Sent as MAV_CMD_DO_REPOSITION (a COMMAND_INT, so it is acknowledged) rather than
        a mission item, so it does not disturb the uploaded flight plan. It was a
        SET_POSITION_TARGET_GLOBAL_INT, which has no acknowledgement: ArduCopter silently
        ignores a target outside its enabled fence, and goto still answered "Flying to
        ..." while the aircraft hovered where it was (SITL, fence enabled).
        """
        mode = self.set_flight_mode("GUIDED")
        if not mode.success:
            return CommandResult(False, "goto", f"Could not enter GUIDED: {mode.message}")
        # MAV_CMD_DO_REPOSITION = 192; param1 -1 keeps the current speed, param4 NaN the
        # current yaw. Frame 6 = MAV_FRAME_GLOBAL_RELATIVE_ALT_INT.
        result = self._command_int(192, 6, (-1.0, 0.0, 0.0, float("nan")),
                                   int(round(latitude * 1e7)), int(round(longitude * 1e7)),
                                   float(altitude_m), name="goto")
        if result.success:
            result.message = f"The vehicle accepted the target {latitude:.6f}, {longitude:.6f}."
        return result

    def set_altitude(self, altitude_m: float) -> CommandResult:
        """Change altitude while holding position. MAV_CMD_CONDITION_CHANGE_ALT = 113."""
        return self._command_long(113, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(altitude_m),
                                  name="set_altitude")

    def set_speed(self, speed_m_s: float, airspeed: bool = False) -> CommandResult:
        """Change the commanded speed. MAV_CMD_DO_CHANGE_SPEED = 178."""
        if speed_m_s <= 0:
            return CommandResult(False, "set_speed", "Speed must be positive.")
        return self._command_long(178, 0.0 if airspeed else 1.0, float(speed_m_s), -1.0,
                                  name="set_speed")

    def set_home(self, latitude: float | None = None, longitude: float | None = None,
                 altitude_m: float = 0.0) -> CommandResult:
        """Set the home position, defaulting to the vehicle's current location.

        Home is where return-to-launch flies to, so getting it wrong is not a cosmetic
        problem.
        """
        # MAV_CMD_DO_SET_HOME = 179. param1 = 1 means "use current position".
        if latitude is None or longitude is None:
            result = self._command_long(179, 1.0, name="set_home")
        else:
            result = self._command_long(179, 0.0, 0.0, 0.0, 0.0,
                                        float(latitude), float(longitude), float(altitude_m),
                                        name="set_home")
        if result.success:
            # Home was only ever asked for until the first answer, so after a SET_HOME
            # the app kept showing -- and preflight kept checking -- the old one.
            self._request_home_position()
        return result

    def emergency_stop(self) -> CommandResult:
        """Cut the motors immediately.

        This drops a flying aircraft out of the sky. It is separate from `abort_mission`
        precisely so it cannot be reached by accident, and the caller is expected to
        have confirmed intent.
        """
        # MAV_CMD_COMPONENT_ARM_DISARM with the force magic number, which disarms even
        # while armed and flying.
        return self._command_long(400, 0.0, 21196.0, name="emergency_stop")

    def camera(self):
        """A camera controller wired to this vehicle's command channel.

        Capabilities come from the payload's own CAMERA_INFORMATION when it has sent
        one; otherwise they are unknown, which is different from absent.
        """
        from .camera_control import CameraCapabilities, CameraController, capabilities_from_message

        with self._lock:
            declared = (self._latest_telemetry.raw or {}).get("camera_information")

        if declared:
            class _Info:
                flags = declared["flags"]
                vendor_name = declared["vendor_name"]
                model_name = declared["model_name"]

            capabilities = capabilities_from_message(_Info())
        else:
            capabilities = CameraCapabilities()

        def send(command: int, *params: float) -> CommandResult:
            return self._command_long(command, *params, name="camera")

        return CameraController(send, capabilities)

    def _set_geofence(self, enabled: bool, timeout_s: float = 4.0) -> CommandResult:
        """MAV_CMD_DO_FENCE_ENABLE (207), confirmed against SYS_STATUS.

        Uploading a fence does not enforce it: ArduPilot ships with FENCE_ENABLE=0, so
        the polygon the app uploaded sat on the vehicle doing nothing unless someone
        enabled it from another ground station. The ack says the command was taken; the
        geofence bit of SYS_STATUS's enabled sensors says the fence is actually on.
        """
        name = "fence_enable" if enabled else "fence_disable"
        result = self._command_long(207, 1.0 if enabled else 0.0, name=name)
        if not result.success:
            return result
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                raw = dict(self._latest_telemetry.raw or {})
            if raw.get("sensors_present", 0) & MAV_SYS_STATUS_GEOFENCE:
                if bool(raw.get("sensors_enabled", 0) & MAV_SYS_STATUS_GEOFENCE) == enabled:
                    state = "enabled" if enabled else "disabled"
                    return CommandResult(True, name, f"Geofence {state}; the vehicle reports it "
                                         f"{state}.", ack=result.ack)
            time.sleep(0.1)
        return CommandResult(
            False, name,
            "The vehicle accepted the command but does not report the geofence as "
            f"{'enabled' if enabled else 'disabled'}. Check FENCE_TYPE and the uploaded fence.",
            ack=result.ack)

    def enable_geofence(self) -> CommandResult:
        return self._set_geofence(True)

    def disable_geofence(self) -> CommandResult:
        return self._set_geofence(False)

    def loiter(self) -> CommandResult:
        """Hold the current position."""
        return self.set_flight_mode("LOITER")

    def _mode_name(self, custom_mode: Any, base_mode: Any = 0) -> str:
        """Resolve a numeric custom_mode to the name a pilot recognises."""
        if custom_mode is None:
            return ""
        if self._px4():
            from pymavlink import mavutil

            name = str(mavutil.interpret_px4_mode(int(base_mode or 0), int(custom_mode))).upper()
            return _PX4_MODE_NAMES.get(name, name)
        try:
            mapping = self._conn.mode_mapping() if self._conn is not None else None
        except Exception:
            mapping = None
        if not mapping:
            return ""
        for name, identifier in mapping.items():
            if identifier == custom_mode:
                return str(name).upper()
        return ""

    def set_flight_mode(self, mode: str, *, verify: bool = True,
                        timeout_s: float = 3.0) -> CommandResult:
        """Change flight mode, and by default wait for the vehicle to confirm it.

        Sending the command is not the same as being in the mode. An autopilot rejects
        a mode it cannot enter -- LOITER without a position estimate, AUTO with no
        mission loaded -- and says nothing. Reporting success on the strength of having
        transmitted would tell a pilot they had handed control back when the aircraft is
        still flying its mission.
        """
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, "set_flight_mode", exc.user_message)
        if self._px4():
            return self._set_px4_mode(mode, verify=verify, timeout_s=timeout_s)
        try:
            wanted = mode.upper()
            mode_id = conn.mode_mapping().get(wanted)
            if mode_id is None:
                available = ", ".join(sorted(conn.mode_mapping()))
                return CommandResult(False, "set_flight_mode",
                                     f"Unknown mode {mode!r}. Available: {available}.")
            from pymavlink import mavutil
            conn.mav.set_mode_send(
                conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
            )
            if not verify:
                return CommandResult(True, "set_flight_mode",
                                     f"Mode {wanted} requested; not confirmed.")

            deadline = time.monotonic() + max(0.1, float(timeout_s))
            while time.monotonic() < deadline:
                with self._lock:
                    current = (self._latest_telemetry.flight_mode or "").upper()
                if current == wanted:
                    return CommandResult(True, "set_flight_mode", f"Mode is {wanted}.")
                time.sleep(0.1)

            with self._lock:
                current = (self._latest_telemetry.flight_mode or "unknown").upper()
            return CommandResult(
                False, "set_flight_mode",
                f"The vehicle did not enter {wanted} within {timeout_s:.0f} s; it is "
                f"still in {current}. The autopilot may be refusing the mode -- LOITER "
                "needs a position estimate, AUTO needs a loaded mission.",
            )
        except Exception as exc:
            return CommandResult(False, "set_flight_mode", str(exc))

    def _set_px4_mode(self, mode: str, *, verify: bool, timeout_s: float) -> CommandResult:
        """PX4: MAV_CMD_DO_SET_MODE with its main/sub mode, acknowledged, then confirmed.

        ArduPilot's SET_MODE with a single custom_mode number means nothing to PX4, and
        pymavlink's PX4 table holds (base, main, sub) tuples, so the ArduPilot path sent a
        tuple where a number belonged and failed.
        """
        from pymavlink import mavutil

        asked = str(mode).upper()
        px4_name = _PX4_MODE_REQUESTS.get(asked, asked)
        entry = mavutil.px4_map.get(px4_name)
        if entry is None:
            available = ", ".join(sorted(_PX4_MODE_NAMES.get(n, n) for n in mavutil.px4_map))
            return CommandResult(False, "set_flight_mode",
                                 f"Unknown mode {mode!r} on PX4. Available: {available}.")
        base_mode, main_mode, sub_mode = entry
        sent = self._command_long(176, float(base_mode), float(main_mode), float(sub_mode),
                                  name="set_flight_mode")
        if not sent.success or not verify:
            return sent
        expected = _PX4_MODE_NAMES.get(px4_name, px4_name)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                current = (self._latest_telemetry.flight_mode or "").upper()
            if current == expected:
                return CommandResult(True, "set_flight_mode", f"Mode is {expected}.",
                                     ack=sent.ack)
            time.sleep(0.1)
        with self._lock:
            current = (self._latest_telemetry.flight_mode or "unknown").upper()
        return CommandResult(False, "set_flight_mode",
                             f"The vehicle accepted {expected} but did not enter it within "
                             f"{timeout_s:.0f} s; it is still in {current}.", ack=sent.ack)

    def start_mission(self) -> CommandResult:
        """Arm if needed, then MAV_CMD_MISSION_START (300).

        A start the vehicle refuses must not leave behind the arming this command did.
        ArduCopter arms, then refuses AUTO with no usable mission ("Mode change to AUTO
        failed: init failed"); the start used to report the refusal and leave the motors
        armed on the ground, spinning under a pilot who had just been told nothing
        happened. Motors this command armed are disarmed again on refusal.
        """
        with self._lock:
            was_armed = bool(self._latest_telemetry.armed)
            mode = (self._latest_telemetry.flight_mode or "").upper()
        if not was_armed and mode in _NOT_ARMABLE_ON_GROUND:
            # An aircraft that landed itself (an RTL or LAND failsafe, a finished
            # mission) sits in that mode, and ArduCopter will not arm in it: Start
            # answered "Arm: RTL mode not armable" after every failsafe landing (SITL).
            # A position-holding mode is armable and MISSION_START takes it to AUTO.
            held = self.set_flight_mode("LOITER")
            if not held.success:
                return CommandResult(False, "start_mission",
                                     f"The vehicle is in {mode}, which it will not arm in, "
                                     f"and refused LOITER: {held.message}")
        arm = self._command_long(400, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, name="arm")
        if not arm.success:
            return arm
        started = self._command_long(300, 0.0, 0.0, name="start_mission")
        if started.success or was_armed:
            return started
        disarm = self._command_long(400, 0.0, 0.0, name="disarm")
        started.message += (" The motors this command armed were disarmed again."
                            if disarm.success else
                            f" The motors this command armed are STILL ARMED: disarm "
                            f"refused ({disarm.message}).")
        return started

    def pause_mission(self) -> CommandResult:
        if self._px4():
            # PX4 answers DO_PAUSE_CONTINUE "unsupported"; a mission is paused by holding
            # (AUTO.LOITER) and resumed by re-entering MISSION, which continues from the
            # current item. This is what QGroundControl and MAVSDK do.
            return self._renamed(self.set_flight_mode("LOITER"), "pause_mission")
        # MAV_CMD_DO_PAUSE_CONTINUE = 193, p1=0 to pause
        return self._command_long(193, 0.0, name="pause_mission")

    def resume_mission(self) -> CommandResult:
        if self._px4():
            return self._renamed(self.set_flight_mode("AUTO"), "resume_mission")
        return self._command_long(193, 1.0, name="resume_mission")

    @staticmethod
    def _renamed(result: CommandResult, name: str) -> CommandResult:
        result.command = name
        return result

    def return_to_home(self) -> CommandResult:
        # MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
        return self._command_long(20, name="return_to_home")

    def abort_mission(self) -> CommandResult:
        """Stop the mission and hold position, confirmed against the vehicle's mode.

        DO_PAUSE_CONTINUE alone was the whole abort, and an autopilot that does not
        support pausing (older ArduPilot, most planes) acknowledges nothing useful. A
        position-holding mode is what abort means to a pilot, so that is tried first and
        confirmed; the pause command is the fallback.
        """
        for mode in ("BRAKE", "LOITER"):
            try:
                available = self._conn.mode_mapping() if self._conn is not None else {}
            except Exception:  # noqa: BLE001
                available = {}
            if mode not in (available or {}):
                continue
            held = self.set_flight_mode(mode)
            if held.success:
                return CommandResult(True, "abort_mission",
                                     f"Mission aborted; the vehicle is holding in {mode}.",
                                     ack=held.ack or ACK_ACCEPTED)
        paused = self._command_long(193, 0.0, name="abort_mission")
        if paused.success:
            paused.message = "Mission paused (DO_PAUSE_CONTINUE); no hold mode was confirmed."
        return paused


# ── Spec-named helpers ────────────────────────────────────────────────────────

def connect_mission_planner(
    transport: str = "udp",
    host: str = "127.0.0.1",
    port: int = DEFAULT_UDPIN_PORT,
) -> MissionPlannerDroneClient:
    """Build and connect a Mission Planner-backed drone client."""
    transport = transport.lower().strip()
    if transport == "udp":
        uri = f"udpin:{host}:{port}"
    elif transport == "tcp":
        uri = f"tcp:{host}:{port}"
    elif transport == "serial":
        uri = f"{host},{port}"
    else:
        raise AppError(ERR_INVALID_INPUT, f"Unsupported transport: {transport!r}",
                       recovery_action="Use 'udp', 'tcp' or 'serial'.")
    client = MissionPlannerDroneClient()
    client.connect(uri)
    return client


def mission_plan_to_mavlink_items(
    mission_plan: dict[str, Any],
    default_altitude_m: float = 50.0,
) -> list[dict[str, Any]]:
    """Convert our MissionPlan dict into MAVLink mission items for upload.

    Accepts either:
      * a `waypoints` list of `[lon, lat, alt]`, or
      * a GeoJSON-style `features` list with Point geometries.
    """
    items: list[dict[str, Any]] = []
    waypoints = mission_plan.get("waypoints") or []
    if not waypoints:
        for feat in (mission_plan.get("geojson", {}).get("features", []) or []):
            geom = feat.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [0, 0, default_altitude_m])
                if len(coords) >= 2:
                    waypoints.append(coords)
    for row in waypoints:
        lon = float(row[0])
        lat = float(row[1])
        alt = float(row[2]) if len(row) >= 3 else default_altitude_m
        items.append({"lat": lat, "lon": lon, "alt": alt, "command": 16, "frame": 3})
    return items
