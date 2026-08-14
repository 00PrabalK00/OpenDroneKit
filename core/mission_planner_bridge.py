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

from .drone import CommandResult, DroneTelemetry
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
        201,  # DO_SET_ROI
        203,  # DO_DIGICAM_CONTROL
        205,  # DO_MOUNT_CONTROL
        206,  # DO_SET_CAM_TRIGG_DIST
        2500,  # DO_GIMBAL_MANAGER_CONFIGURE
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
        self._telemetry_subscribers: list[Any] = []

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
            self._connected = True
            self._uri = connection_uri
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
        if t == "STATUSTEXT":
            self._publish({"type": "statustext", "severity": int(getattr(msg, "severity", 6)),
                           "text": str(getattr(msg, "text", "")).strip("\x00")})
        with self._lock:
            tel = self._latest_telemetry
            tel.connected = True
            tel.timestamp = time.time()
            if t == "HEARTBEAT":
                # custom_mode is a number. Reporting it as the flight mode gives the
                # operator "3" where they need "AUTO", and makes it impossible to say
                # who is flying the aircraft, so it is resolved to its name here.
                tel.flight_mode = self._mode_name(getattr(msg, "custom_mode", None)) or tel.flight_mode
                tel.armed = bool(getattr(msg, "base_mode", 0) & 0x80)
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
                tel.battery_v = float(msg.voltage_battery) / 1000.0
                tel.battery_pct = max(0.0, float(msg.battery_remaining))
                # Sensor health bitmasks, kept so preflight can report compass, gyro
                # and accelerometer state instead of leaving them unchecked. Stored
                # raw: interpreting the bits belongs with the checks, not the parser.
                tel.raw["sensors_present"] = int(
                    getattr(msg, "onboard_control_sensors_present", 0) or 0)
                tel.raw["sensors_enabled"] = int(
                    getattr(msg, "onboard_control_sensors_enabled", 0) or 0)
                tel.raw["sensors_health"] = int(
                    getattr(msg, "onboard_control_sensors_health", 0) or 0)
            elif t == "GPS_RAW_INT":
                tel.gps_fix = int(msg.fix_type)
                tel.satellites = int(getattr(msg, "satellites_visible", 0))
                eph = float(getattr(msg, "eph", 9999.0))
                tel.hdop = eph / 100.0 if eph < 9999.0 else tel.hdop
            elif t == "MISSION_CURRENT":
                tel.waypoint_index = int(getattr(msg, "seq", tel.waypoint_index))
            elif t == "MISSION_COUNT":
                tel.waypoint_total = int(getattr(msg, "count", tel.waypoint_total))
            elif t == "HOME_POSITION":
                tel.home_lat = float(msg.latitude) / 1e7
                tel.home_lon = float(msg.longitude) / 1e7
                tel.home_alt = float(msg.altitude) / 1000.0
                tel.home_set = True
            elif t == "RADIO_STATUS":
                tel.link_quality_pct = max(0.0, min(100.0, 100.0 - float(getattr(msg, "remnoise", 0))))

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
                    return CommandResult(
                        False, name, f"Vehicle rejected the upload: {reason}."
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
            self._transfer_lock.release()

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        """Upload the flight plan, including gimbal, yaw, dwell, and trigger items."""
        return self._upload_list(
            mission_items, mission_type=MAV_MISSION_TYPE_MISSION, name="upload_mission"
        )

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

    def download_mission(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        """Read the mission currently stored on the vehicle.

        Used to verify an upload round-trips: what the vehicle reports back is the
        only trustworthy evidence that it stored what was sent.
        """
        conn = self._ensure()
        if not self._transfer_lock.acquire(timeout=timeout_s):
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Another mission transfer is in progress.")
        try:
            self._drain_protocol_queue()
            conn.mav.mission_request_list_send(
                conn.target_system, conn.target_component, MAV_MISSION_TYPE_MISSION
            )
            count_msg = self._await_protocol({"MISSION_COUNT"}, timeout=timeout_s)
            if count_msg is None:
                raise AppError(ERR_DRONE_NOT_CONNECTED, "Vehicle did not report a mission count.")

            total = int(getattr(count_msg, "count", 0))
            items: list[dict[str, Any]] = []
            for seq in range(total):
                conn.mav.mission_request_int_send(
                    conn.target_system, conn.target_component, seq, MAV_MISSION_TYPE_MISSION
                )
                item = self._await_protocol({"MISSION_ITEM_INT"}, timeout=timeout_s)
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
                MAV_MISSION_TYPE_MISSION,
            )
            return items
        finally:
            self._transfer_lock.release()

    def _command_long(self, command: int, *params: float, name: str = "command") -> CommandResult:
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, name, exc.user_message)
        try:
            p = list(params) + [0.0] * (7 - len(params))
            conn.mav.command_long_send(
                conn.target_system, conn.target_component, command, 0,
                *p[:7],
            )
            return CommandResult(True, name, "Command sent.")
        except Exception as exc:
            return CommandResult(False, name, str(exc))

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
        """Fly to a position in GUIDED mode.

        Sent as SET_POSITION_TARGET_GLOBAL_INT rather than a mission item, so it does
        not disturb the uploaded flight plan.
        """
        try:
            conn = self._ensure()
            from pymavlink import mavutil
        except (AppError, ImportError) as exc:
            return CommandResult(False, "goto", str(exc))

        mode = self.set_flight_mode("GUIDED")
        if not mode.success:
            return CommandResult(False, "goto", f"Could not enter GUIDED: {mode.message}")

        try:
            conn.mav.set_position_target_global_int_send(
                0, conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                # Type mask: use position only, ignore velocity, acceleration and yaw.
                0b0000111111111000,
                int(latitude * 1e7), int(longitude * 1e7), float(altitude_m),
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            return CommandResult(True, "goto", f"Flying to {latitude:.6f}, {longitude:.6f}.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, "goto", str(exc))

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
            return self._command_long(179, 1.0, name="set_home")
        return self._command_long(179, 0.0, 0.0, 0.0, 0.0,
                                  float(latitude), float(longitude), float(altitude_m),
                                  name="set_home")

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

    def loiter(self) -> CommandResult:
        """Hold the current position."""
        return self.set_flight_mode("LOITER")

    def _mode_name(self, custom_mode: Any) -> str:
        """Resolve a numeric custom_mode to the name a pilot recognises."""
        if custom_mode is None:
            return ""
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

    def start_mission(self) -> CommandResult:
        # MAV_CMD_MISSION_START = 300
        arm = self._command_long(400, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, name="arm")
        if not arm.success:
            return arm
        return self._command_long(300, 0.0, 0.0, name="start_mission")

    def pause_mission(self) -> CommandResult:
        # MAV_CMD_DO_PAUSE_CONTINUE = 193, p1=0 to pause
        return self._command_long(193, 0.0, name="pause_mission")

    def resume_mission(self) -> CommandResult:
        return self._command_long(193, 1.0, name="resume_mission")

    def return_to_home(self) -> CommandResult:
        # MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
        return self._command_long(20, name="return_to_home")

    def abort_mission(self) -> CommandResult:
        # Hold mode via DO_PAUSE_CONTINUE; on ArduPilot we also disarm if low altitude
        return self._command_long(193, 0.0, name="abort_mission")


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
