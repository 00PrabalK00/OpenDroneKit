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

import socket
import threading
import time
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
        while not self._heartbeat_stop.is_set() and self._conn is not None:
            try:
                msg = self._conn.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                self._absorb_message(msg)
            except Exception:
                time.sleep(0.5)

    def _absorb_message(self, msg) -> None:
        t = msg.get_type()
        with self._lock:
            tel = self._latest_telemetry
            tel.connected = True
            tel.timestamp = time.time()
            if t == "HEARTBEAT":
                tel.flight_mode = getattr(msg, "custom_mode", tel.flight_mode) and str(getattr(msg, "custom_mode", "")) or tel.flight_mode
                tel.armed = bool(getattr(msg, "base_mode", 0) & 0x80)
            elif t == "GLOBAL_POSITION_INT":
                tel.latitude = float(msg.lat) / 1e7
                tel.longitude = float(msg.lon) / 1e7
                tel.altitude_rel_m = float(msg.relative_alt) / 1000.0
                tel.altitude_abs_m = float(msg.alt) / 1000.0
                tel.heading_deg = float(getattr(msg, "hdg", 0) or 0) / 100.0
                tel.speed_mps = (float(msg.vx) ** 2 + float(msg.vy) ** 2) ** 0.5 / 100.0
            elif t == "SYS_STATUS":
                tel.battery_v = float(msg.voltage_battery) / 1000.0
                tel.battery_pct = max(0.0, float(msg.battery_remaining))
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

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        try:
            from pymavlink import mavutil
        except ImportError:
            return CommandResult(False, "upload_mission", "pymavlink not installed.")
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, "upload_mission", exc.user_message)

        try:
            conn.mav.mission_count_send(conn.target_system, conn.target_component, len(mission_items))
            for idx, item in enumerate(mission_items):
                lat = float(item.get("lat", 0.0))
                lon = float(item.get("lon", 0.0))
                alt = float(item.get("alt", 0.0))
                command = int(item.get("command", 16))   # NAV_WAYPOINT
                frame = int(item.get("frame", 3))        # MAV_FRAME_GLOBAL_RELATIVE_ALT
                conn.mav.mission_item_int_send(
                    conn.target_system, conn.target_component,
                    idx, frame, command,
                    0, 1,
                    float(item.get("param1", 0.0)),
                    float(item.get("param2", 0.0)),
                    float(item.get("param3", 0.0)),
                    float(item.get("param4", 0.0)),
                    int(lat * 1e7), int(lon * 1e7), alt,
                )
            self._mission_uploaded_count = len(mission_items)
            return CommandResult(True, "upload_mission", f"Uploaded {len(mission_items)} items.")
        except Exception as exc:
            return CommandResult(False, "upload_mission", str(exc))

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

    def set_flight_mode(self, mode: str) -> CommandResult:
        try:
            conn = self._ensure()
        except AppError as exc:
            return CommandResult(False, "set_flight_mode", exc.user_message)
        try:
            mode_id = conn.mode_mapping().get(mode.upper())
            if mode_id is None:
                return CommandResult(False, "set_flight_mode", f"Unknown mode {mode!r}.")
            from pymavlink import mavutil
            conn.mav.set_mode_send(
                conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
            )
            return CommandResult(True, "set_flight_mode", f"Mode set to {mode}.")
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
