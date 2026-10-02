"""Drone connection abstraction — Protocol + Mock implementation."""

from __future__ import annotations

import asyncio
import gc
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

# ── Data models ───────────────────────────────────────────────────────────────


# Telemetry older than this is a lost link. ArduPilot streams several messages a
# second, so three seconds of silence is a lost link, not a slow one. The session's
# display, the flight session's warning and the preflight checks all use this one value.
TELEMETRY_STALE_AFTER_S = 3.0

# MAV_STATE, from HEARTBEAT.system_status. ArduPilot reports CRITICAL while a failsafe
# is active and EMERGENCY when it has lost control.
MAV_STATE_CRITICAL = 5
MAV_STATE_EMERGENCY = 6
MAV_STATE_NAMES = {0: "uninit", 1: "boot", 2: "calibrating", 3: "standby", 4: "active",
                   5: "critical", 6: "emergency", 7: "poweroff", 8: "flight_termination"}
# MAV_BATTERY_CHARGE_STATE, from BATTERY_STATUS.charge_state.
BATTERY_CHARGE_STATE_NAMES = {1: "ok", 2: "low", 3: "critical", 4: "emergency",
                              5: "failed", 6: "unhealthy", 7: "charging"}
BATTERY_FAILSAFE_STATES = {2, 3, 4, 5}
# FENCE_BREACH, from FENCE_STATUS.breach_type.
FENCE_BREACH_NAMES = {0: "none", 1: "minimum altitude", 2: "maximum altitude",
                      3: "boundary"}

@dataclass
class DroneTelemetry:
    connected: bool = False
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_rel_m: float = 0.0
    altitude_abs_m: float = 0.0
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    # None means "the vehicle has not reported this", which is a different answer from
    # zero. A USB or UDP link carries no RADIO_STATUS, and a flight controller with no
    # current sensor sends battery_remaining = -1; showing either as 0 % puts a number
    # on screen that nothing measured.
    battery_pct: float | None = None
    battery_v: float | None = None
    gps_fix: int = 0          # 0=none, 1=no fix, 2=2D, 3=3D, 4=RTK float, 5=RTK fixed
    satellites: int = 0
    hdop: float | None = None
    home_lat: float = 0.0
    home_lon: float = 0.0
    home_alt: float = 0.0
    home_set: bool = False
    waypoint_index: int = 0
    waypoint_total: int = 0
    distance_to_next_m: float = 0.0
    link_quality_pct: float | None = None
    timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def gps_ok(self) -> bool:
        return self.gps_fix >= 3 and self.satellites >= 6

    @property
    def battery_ok(self) -> bool:
        return self.battery_pct is not None and self.battery_pct >= 25.0

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the vehicle last said anything."""
        return max(0.0, (time.time() if now is None else now) - float(self.timestamp or 0.0))

    def _vehicle_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """What the vehicle says about its own failsafes, fence and GPS, when it said it.

        Each field is the autopilot's own report (HEARTBEAT system_status, BATTERY_STATUS
        charge_state, FENCE_STATUS, STATUSTEXT), never an inference from the numbers: a
        battery failsafe fires on voltage while the percentage still reads 70 %.
        """
        out: dict[str, Any] = {}
        if "system_status" in raw:
            status = int(raw["system_status"])
            out["system_status"] = MAV_STATE_NAMES.get(status, str(status))
            out["failsafe"] = status in (MAV_STATE_CRITICAL, MAV_STATE_EMERGENCY)
        if "battery_charge_state" in raw:
            state = int(raw["battery_charge_state"])
            if state:  # 0 is MAV_BATTERY_CHARGE_STATE_UNDEFINED: not reported
                out["battery_state"] = BATTERY_CHARGE_STATE_NAMES.get(state, str(state))
                out["battery_failsafe"] = state in BATTERY_FAILSAFE_STATES
        fence = raw.get("fence_status")
        if fence:
            out["fence_breached"] = bool(fence.get("breach_status"))
            out["fence_breach_count"] = int(fence.get("breach_count", 0))
            if fence.get("breach_status"):
                out["fence_breach_type"] = FENCE_BREACH_NAMES.get(
                    int(fence.get("breach_type", 0)), str(fence.get("breach_type")))
        gps_at = raw.get("gps_received_at")
        if gps_at is not None and float(self.timestamp or 0.0) - float(gps_at) > TELEMETRY_STALE_AFTER_S:
            # The rest of the link is alive but the GPS report has stopped. The last
            # fix is not a fix now: a stale "3D" over a dead receiver is the one value
            # that must not stay on screen.
            out["gps_fix"] = 0
            out["gps_stale"] = True
        messages = raw.get("vehicle_messages") or []
        if messages:
            out["vehicle_messages"] = [
                {"severity": m["severity"], "text": m["text"],
                 "age_s": round(max(0.0, time.time() - float(m["time"])), 1)}
                for m in messages[-10:]]
        return out

    def to_dict(self) -> dict[str, Any]:
        """The telemetry, omitting anything the vehicle never reported.

        An omitted key is the honest encoding of "not reported": every consumer already
        treats a missing key as unknown, and none of them can tell a defaulted 0.0 from
        a measured one.
        """
        def rounded(value: float | None, digits: int) -> float | None:
            return None if value is None else round(float(value), digits)

        payload = {
            "connected": self.connected,
            "armed": self.armed,
            "flight_mode": self.flight_mode,
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "altitude_rel_m": round(self.altitude_rel_m, 2),
            "altitude_abs_m": round(self.altitude_abs_m, 2),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "battery_pct": rounded(self.battery_pct, 1),
            "battery_v": rounded(self.battery_v, 2),
            "gps_fix": self.gps_fix,
            "satellites": self.satellites,
            "hdop": rounded(self.hdop, 2),
            "home_set": self.home_set,
            "waypoint_index": self.waypoint_index,
            "waypoint_total": self.waypoint_total,
            "link_quality_pct": rounded(self.link_quality_pct, 1),
            "timestamp": self.timestamp,
        }
        present = int((self.raw or {}).get("sensors_present", 0) or 0)
        if present & (1 << 20):  # MAV_SYS_STATUS_GEOFENCE: the vehicle has a fence
            # Whether the uploaded fence is actually enforced, as the vehicle reports it.
            payload["geofence_enabled"] = bool(
                int((self.raw or {}).get("sensors_enabled", 0) or 0) & (1 << 20))
        payload.update(self._vehicle_state(raw=self.raw or {}))
        if self.home_set:
            # Where RTL will fly to. Reported only once the vehicle has said: a home of
            # 0.0, 0.0 is Null Island, not "unknown". It was never exported at all, so
            # neither UI could show it -- or that SET_HOME had moved it.
            payload["home_lat"] = round(self.home_lat, 7)
            payload["home_lon"] = round(self.home_lon, 7)
            payload["home_alt_m"] = round(self.home_alt, 2)
        return {key: value for key, value in payload.items() if value is not None}


# How a vehicle answered a command. "sent" means the driver has no acknowledgement
# channel at all; "none" means one exists and the vehicle stayed silent.
ACK_ACCEPTED = "accepted"
ACK_IN_PROGRESS = "in_progress"
ACK_DENIED = "denied"
ACK_NONE = "none"
ACK_SENT = "sent"


@dataclass
class CommandResult:
    success: bool
    command: str
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    ack: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {"success": self.success, "command": self.command, "message": self.message}
        if self.ack:
            payload["ack"] = self.ack
        return payload


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class DroneClient(Protocol):
    def connect(self, connection_uri: str) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def get_telemetry(self) -> DroneTelemetry: ...
    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult: ...
    def start_mission(self) -> CommandResult: ...
    def pause_mission(self) -> CommandResult: ...
    def resume_mission(self) -> CommandResult: ...
    def return_to_home(self) -> CommandResult: ...
    def abort_mission(self) -> CommandResult: ...
    def set_flight_mode(self, mode: str) -> CommandResult: ...


# ── Mock client ───────────────────────────────────────────────────────────────

class MockDroneClient:
    """
    Simulated drone for development and UI testing.
    Provides realistic telemetry drift and flight state transitions.
    """

    def __init__(self):
        self._connected = False
        self._armed = False
        self._flight_mode = "STABILIZE"
        self._mission_running = False
        self._mission_paused = False
        self._waypoint_index = 0
        self._waypoints: list[dict[str, Any]] = []
        self._lat = 37.7749
        self._lon = -122.4194
        self._alt = 0.0
        self._heading = 0.0
        self._speed = 0.0
        self._battery = 87.5
        self._battery_v = 22.1
        self._link = 95.0
        self._t0 = time.time()
        self._log: list[dict[str, Any]] = []

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, connection_uri: str = "mock://") -> None:
        self._connected = True
        self._flight_mode = "LOITER"
        self._log_event("connected", f"Mock drone connected to {connection_uri}")

    def disconnect(self) -> None:
        self._connected = False
        self._armed = False
        self._mission_running = False
        self._log_event("disconnected", "Mock drone disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    # ── Telemetry ────────────────────────────────────────────────────────────

    def get_telemetry(self) -> DroneTelemetry:
        if not self._connected:
            return DroneTelemetry(connected=False)

        t = time.time() - self._t0
        # Battery drains slowly
        self._battery = max(0.0, self._battery - 0.002)
        # Simulate small position drift when flying
        if self._mission_running and not self._mission_paused:
            self._lat += math.sin(t * 0.05) * 0.000005
            self._lon += math.cos(t * 0.05) * 0.000005
            self._alt = max(0.0, 50.0 + math.sin(t * 0.1) * 2.0)
            self._speed = 5.0 + math.sin(t * 0.2) * 0.5
            self._heading = (self._heading + 0.1 * t) % 360.0
            # Advance waypoints
            if self._waypoints and int(t * 0.5) > self._waypoint_index:
                self._waypoint_index = min(int(t * 0.5), max(0, len(self._waypoints) - 1))
        else:
            self._speed = 0.0

        return DroneTelemetry(
            connected=True,
            armed=self._armed,
            flight_mode=self._flight_mode,
            latitude=self._lat,
            longitude=self._lon,
            altitude_rel_m=self._alt,
            altitude_abs_m=self._alt + 15.0,
            heading_deg=self._heading,
            speed_mps=self._speed,
            battery_pct=self._battery,
            battery_v=self._battery_v,
            gps_fix=5 if self._connected else 0,
            satellites=14,
            hdop=0.8,
            home_lat=self._lat,
            home_lon=self._lon,
            home_alt=15.0,
            home_set=True,
            waypoint_index=self._waypoint_index,
            waypoint_total=len(self._waypoints),
            distance_to_next_m=max(0.0, 20.0 - (t % 20.0) * 1.5),
            link_quality_pct=min(100.0, self._link + random.uniform(-2.0, 2.0)),
        )

    # ── Mission control ───────────────────────────────────────────────────────

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "upload_mission", "Not connected.")
        self._waypoints = mission_items
        self._waypoint_index = 0
        self._log_event("mission_uploaded", f"Uploaded {len(mission_items)} waypoints.")
        return CommandResult(True, "upload_mission", f"Uploaded {len(mission_items)} items.")

    def start_mission(self) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "start_mission", "Not connected.")
        if not self._waypoints:
            return CommandResult(False, "start_mission", "No mission uploaded.")
        self._armed = True
        self._mission_running = True
        self._mission_paused = False
        self._flight_mode = "AUTO"
        self._waypoint_index = 0
        self._t0 = time.time()
        self._log_event("mission_started", "Mission started.")
        return CommandResult(True, "start_mission", "Mission started.")

    def pause_mission(self) -> CommandResult:
        if not self._mission_running:
            return CommandResult(False, "pause_mission", "No active mission.")
        self._mission_paused = True
        self._flight_mode = "LOITER"
        self._log_event("mission_paused", f"Paused at waypoint {self._waypoint_index}.")
        return CommandResult(True, "pause_mission", "Mission paused.")

    def resume_mission(self) -> CommandResult:
        if not self._mission_running or not self._mission_paused:
            return CommandResult(False, "resume_mission", "Mission not paused.")
        self._mission_paused = False
        self._flight_mode = "AUTO"
        self._log_event("mission_resumed", "Mission resumed.")
        return CommandResult(True, "resume_mission", "Mission resumed.")

    def return_to_home(self) -> CommandResult:
        self._mission_running = False
        self._mission_paused = False
        self._flight_mode = "RTL"
        self._log_event("rth_triggered", "Return to home triggered.")
        return CommandResult(True, "return_to_home", "RTH triggered.")

    def abort_mission(self) -> CommandResult:
        self._mission_running = False
        self._mission_paused = False
        self._armed = False
        self._flight_mode = "LOITER"
        self._speed = 0.0
        self._log_event("mission_aborted", "Mission aborted — LOITER mode.")
        return CommandResult(True, "abort_mission", "Mission aborted.")

    def set_flight_mode(self, mode: str) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "set_flight_mode", "Not connected.")
        self._flight_mode = mode.upper()
        return CommandResult(True, "set_flight_mode", f"Mode set to {self._flight_mode}.")

    def get_command_log(self) -> list[dict[str, Any]]:
        return list(self._log)

    def _log_event(self, event: str, detail: str) -> None:
        self._log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "detail": detail,
        })
        if len(self._log) > 500:
            self._log = self._log[-500:]


# ── MAVSDK client (optional, requires mavsdk installed) ───────────────────────

# MAVSDK flight-mode names, mapped onto the calls that enter them. MAVSDK has no generic
# "set mode" -- each mode is reached through the action or mission plugin -- so a mode
# with no entry here is refused by name rather than approximated.
_MAVSDK_MODE_ENTRY = {
    "HOLD": ("action", "hold"),
    "LOITER": ("action", "hold"),
    "POSHOLD": ("action", "hold"),
    "BRAKE": ("action", "hold"),
    "RTL": ("action", "return_to_launch"),
    "RETURN_TO_LAUNCH": ("action", "return_to_launch"),
    "LAND": ("action", "land"),
    "MISSION": ("mission", "start_mission"),
    "AUTO": ("mission", "start_mission"),
}

# What each MAVSDK FlightMode is called by the rest of this toolkit, which speaks
# ArduPilot's vocabulary (flight_control.classify_mode reads these names).
_MAVSDK_MODE_NAMES = {
    "HOLD": "LOITER",
    "MISSION": "AUTO",
    "RETURN_TO_LAUNCH": "RTL",
    "POSCTL": "POSHOLD",
    "ALTCTL": "ALT_HOLD",
    "STABILIZED": "STABILIZE",
}


def _load_mavsdk():
    """The gRPC MAVSDK-Python API and its version, under whichever name it is installed.

    The API this driver speaks (System, mavsdk_server over gRPC, mission_raw) was
    published as ``mavsdk`` up to 3.x. From 3.17 it is also ``mavsdk-grpc``, imported as
    ``mavsdk_grpc``; and ``mavsdk`` 4.x is a different library -- a native binding with
    its own API -- so ``pip install mavsdk`` now installs something this driver cannot
    drive. That is refused by name rather than failing later inside a call.
    """
    import importlib

    for name, dist in (("mavsdk_grpc", "mavsdk-grpc"), ("mavsdk", "mavsdk")):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        version = str(getattr(module, "__version__", "") or "")
        if not version:
            try:
                from importlib.metadata import version as dist_version
                version = dist_version(dist)
            except Exception:  # noqa: BLE001 - an unpackaged checkout has no metadata
                version = ""
        major = _major_version(version)
        if name == "mavsdk" and major is not None and major >= 4:
            raise RuntimeError(
                f"mavsdk {version} is the native MAVSDK binding, which has a different API "
                "from the one this driver uses. Install the gRPC API with "
                "`pip install mavsdk-grpc` (or `pip install \"mavsdk<4\"`).")
        raw = importlib.import_module(f"{name}.mission_raw")
        return module, raw, (major if major is not None else 2)
    raise RuntimeError(
        "MAVSDK not installed. Install it with: pip install mavsdk-grpc\n"
        "Or use MockDroneClient for offline testing.")


def _major_version(version: str) -> int | None:
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return None


def _mavsdk_address(uri: str, sdk_major: int) -> str:
    """The connection URI in the spelling this MAVSDK accepts.

    MAVSDK 3 deprecated ``udp://`` (a listening socket) for ``udpin://``, and 2.x does not
    know ``udpin://``; the app and its users say ``udp://:14540`` either way.
    """
    text = str(uri or "").strip()
    if sdk_major >= 3 and text.startswith("udp://"):
        host_port = text[len("udp://"):]
        host, _, port = host_port.rpartition(":")
        return f"udpin://{host or '0.0.0.0'}:{port}"
    if sdk_major < 3 and text.startswith("udpin://"):
        host_port = text[len("udpin://"):]
        host, _, port = host_port.rpartition(":")
        return f"udp://{'' if host in ('', '0.0.0.0') else host}:{port}"
    return text


def _free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mavsdk_battery_percent(remaining: float, sdk_major: int) -> float:
    """MAVSDK 1.x reports remaining charge as 0..1; 2.x reports 0..100."""
    value = float(remaining)
    return value * 100.0 if sdk_major < 2 else value


class MAVSDKDroneClient:
    """
    Real drone client using MAVSDK-Python (PX4).
    Requires `pip install mavsdk`; the package ships its own mavsdk_server.

    MAVSDK is asyncio-only, and its telemetry arrives as async streams that have to be
    consumed continuously. So the client owns one event loop on a background thread:
    the telemetry subscriptions run there for the life of the link, and every command is
    submitted to that loop and waited on. The previous version drove a loop only while a
    command ran, so its telemetry was an empty DroneTelemetry that never changed, and its
    upload always answered "not yet wired".

    A missing mavsdk raises on connect rather than degrading to a no-op: a client that
    reported itself connected while doing nothing would let a pilot believe a mission
    had been uploaded.
    """

    COMMAND_TIMEOUT_S = 15.0
    CONNECT_TIMEOUT_S = 10.0
    # Starting mavsdk_server is separate from hearing a vehicle and can be much slower:
    # its first start on Windows took 24 s (a 49 MB executable, scanned on launch), and
    # the 10 s the whole connect used to get reported "no vehicle answered" about a PX4
    # that was streaming the whole time.
    SERVER_START_TIMEOUT_S = 60.0
    # How long mavsdk_server waits for each MAVLink acknowledgement (its default is
    # 0.5 s). On a loaded ground station a real PX4's COMMAND_ACKs arrived later than
    # that: mavsdk_server logged "Received ack for not-existing command" and a resume
    # PX4 had carried out was reported as failed. A telemetry radio is slower still.
    MAVLINK_TIMEOUT_S = 2.0
    # How long a mode command whose acknowledgement never came is checked against the
    # mode the vehicle then reports.
    MODE_CONFIRM_S = 5.0
    # How long after an upload a refused Start is tried again. PX4 checks a mission
    # (and checks it again when the fence or a parameter changes) before it will enter
    # MISSION; a Start sent in that window was refused ("Mission start denied! No valid
    # mission") on a loaded machine, a second after PX4 had accepted the upload.
    PX4_MISSION_SETTLE_S = 10.0
    # The flight stack mission items are written for (mission.exporters).
    autopilot = "px4"

    def __init__(self):
        self._drone = None
        self._connected = False
        self._telemetry = DroneTelemetry()
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tasks: list[Any] = []
        self._sdk_major = 2
        self._mission_raw: Any = None
        # Items in the flight plan the vehicle holds, as this link uploaded or read it.
        self._mission_total: int | None = None
        # monotonic time of the last upload the vehicle accepted (mission, fence, rally).
        self._last_upload = float("-inf")

    # -- event loop ---------------------------------------------------------

    def _start_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        self._thread = threading.Thread(target=run, name="mavsdk-loop", daemon=True)
        self._thread.start()
        ready.wait(5.0)
        self._loop = loop
        return loop

    def _run(self, coro, timeout: float | None = None):
        loop = self._loop
        if loop is None or not loop.is_running():
            coro.close()
            raise RuntimeError("MAVSDK event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout or self.COMMAND_TIMEOUT_S)

    def _stop_loop(self) -> None:
        loop = self._loop
        # The System is released on its own loop, below; nothing else may hold it.
        holder = [self._drone] if self._drone is not None else []
        self._drone = None
        if loop is None:
            return

        async def shutdown() -> None:
            # Cancelled and awaited, so no stream is left half-closed on a stopped loop.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Every gRPC object -- the stream tasks, the System and its grpc.aio channel --
            # is released here, on the loop that owns it, and mavsdk_server is killed.
            # Released on another thread after the loop had stopped (`self._tasks = []`,
            # `self._drone = None`), grpc.aio's finalisers spun at 100% CPU holding the
            # GIL and Disconnect never returned: against a real PX4, every disconnect
            # after a flight, and some after only a connect.
            del tasks
            self._tasks = []
            while holder:
                system = holder.pop()
                stop_server = getattr(system, "_stop_mavsdk_server", None)
                if callable(stop_server):
                    stop_server()
                del system
            # Objects in reference cycles (a cancelled stream's frame and its call) are
            # finalised now, on this thread, not by a later collection elsewhere.
            gc.collect()
            await asyncio.sleep(0)

        if loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=10.0)
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not loop.is_running():
            loop.close()
        self._thread = None
        self._loop = None

    # -- connection ---------------------------------------------------------

    def connect(self, connection_uri: str = "udp://:14540") -> None:
        sdk, self._mission_raw, self._sdk_major = _load_mavsdk()
        System = sdk.System

        self._start_loop()
        try:
            # mavsdk_server listens on a free local port, not the library's fixed
            # 50051: anything else already on 50051 (a Windows endpoint agent here)
            # answered the SDK's gRPC with "unknown service mavsdk.rpc.core.CoreService"
            # and the link never came up.
            self._drone = System(port=_free_tcp_port())
            self._run(self._drone.connect(
                system_address=_mavsdk_address(connection_uri, self._sdk_major)),
                timeout=self.SERVER_START_TIMEOUT_S)
            # connect() returns once mavsdk_server is listening, not once a vehicle has
            # been heard. Without this a link with nothing on the other end would be
            # reported as connected.
            self._run(self._await_vehicle(), timeout=self.CONNECT_TIMEOUT_S)
            set_timeout = getattr(getattr(self._drone, "core", None), "set_mavlink_timeout", None)
            if set_timeout is not None:
                self._run(set_timeout(self.MAVLINK_TIMEOUT_S))
        except Exception as exc:  # noqa: BLE001
            self._stop_loop()
            reason = str(exc) or ("no vehicle answered within "
                                  f"{self.CONNECT_TIMEOUT_S:.0f} s")
            raise RuntimeError(f"Failed to connect to drone at {connection_uri}: {reason}") from exc

        self._connected = True
        with self._lock:
            self._telemetry = DroneTelemetry(connected=True)
            self._mission_total = None
        self._start_telemetry()

    async def _await_vehicle(self) -> None:
        async for state in self._drone.core.connection_state():
            if getattr(state, "is_connected", False):
                return

    def disconnect(self) -> None:
        self._connected = False
        self._stop_loop()
        with self._lock:
            self._telemetry.connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -- telemetry ----------------------------------------------------------

    _STREAMS = (
        ("position", "_on_position"),
        ("battery", "_on_battery"),
        ("gps_info", "_on_gps_info"),
        ("raw_gps", "_on_raw_gps"),
        ("armed", "_on_armed"),
        ("flight_mode", "_on_flight_mode"),
        ("heading", "_on_heading"),
        ("velocity_ned", "_on_velocity"),
        ("home", "_on_home"),
    )

    def _start_telemetry(self) -> None:
        loop = self._loop
        if loop is None:
            return
        drone = self._drone

        def spawn() -> None:
            for stream, handler in self._STREAMS:
                opener = getattr(getattr(drone, "telemetry", None), stream, None)
                if opener is not None:
                    self._tasks.append(loop.create_task(
                        self._follow(opener, getattr(self, handler))))
            # The plan goes up through mission_raw, so its progress is mission_raw's. The
            # Mission plugin only tracks missions it uploaded itself: on a real PX4 its
            # stream stayed at item 0 for a whole flown survey.
            progress = (getattr(getattr(drone, "mission_raw", None), "mission_progress", None)
                        or getattr(getattr(drone, "mission", None), "mission_progress", None))
            if progress is not None:
                self._tasks.append(loop.create_task(self._follow(progress, self._on_progress)))

        loop.call_soon_threadsafe(spawn)

    async def _follow(self, open_stream, handler) -> None:
        """Consume one telemetry stream until the link closes.

        A stream the vehicle does not provide ends or raises; that field then stays
        unreported, which is the truth, rather than taking the other streams down.
        """
        try:
            async for value in open_stream():
                with self._lock:
                    handler(self._telemetry, value)
                    self._telemetry.connected = True
                    self._telemetry.timestamp = time.time()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one missing stream is not a lost link
            return

    @staticmethod
    def _on_position(tel: DroneTelemetry, p: Any) -> None:
        tel.latitude = float(p.latitude_deg)
        tel.longitude = float(p.longitude_deg)
        tel.altitude_abs_m = float(p.absolute_altitude_m)
        tel.altitude_rel_m = float(p.relative_altitude_m)

    def _on_battery(self, tel: DroneTelemetry, b: Any) -> None:
        remaining = getattr(b, "remaining_percent", None)
        if remaining is not None and not math.isnan(float(remaining)):
            tel.battery_pct = max(0.0, min(100.0, _mavsdk_battery_percent(remaining, self._sdk_major)))
        voltage = getattr(b, "voltage_v", None)
        if voltage is not None and not math.isnan(float(voltage)):
            tel.battery_v = float(voltage)

    @staticmethod
    def _on_gps_info(tel: DroneTelemetry, g: Any) -> None:
        tel.satellites = int(getattr(g, "num_satellites", 0) or 0)
        fix = getattr(g, "fix_type", 0)
        tel.gps_fix = int(getattr(fix, "value", fix) or 0)

    @staticmethod
    def _on_raw_gps(tel: DroneTelemetry, r: Any) -> None:
        hdop = getattr(r, "hdop", None)
        if hdop is not None and not math.isnan(float(hdop)):
            tel.hdop = float(hdop)

    @staticmethod
    def _on_armed(tel: DroneTelemetry, armed: Any) -> None:
        tel.armed = bool(armed)

    @staticmethod
    def _on_flight_mode(tel: DroneTelemetry, mode: Any) -> None:
        name = str(getattr(mode, "name", mode)).upper().split(".")[-1]
        tel.flight_mode = _MAVSDK_MODE_NAMES.get(name, name)

    @staticmethod
    def _on_heading(tel: DroneTelemetry, h: Any) -> None:
        tel.heading_deg = float(getattr(h, "heading_deg", 0.0) or 0.0)

    @staticmethod
    def _on_velocity(tel: DroneTelemetry, v: Any) -> None:
        tel.speed_mps = math.hypot(float(v.north_m_s), float(v.east_m_s))

    @staticmethod
    def _on_home(tel: DroneTelemetry, p: Any) -> None:
        tel.home_lat = float(p.latitude_deg)
        tel.home_lon = float(p.longitude_deg)
        tel.home_alt = float(p.absolute_altitude_m)
        tel.home_set = True

    def _on_progress(self, tel: DroneTelemetry, progress: Any) -> None:
        tel.waypoint_index = int(getattr(progress, "current", 0) or 0)
        # mission_raw's total follows the last list transferred, whatever its type:
        # after the fence (4 vertices) was read back from a real PX4, a 273-item survey
        # showed as "item 0 of 4". The flight plan's own count is used once known.
        total = self._mission_total
        tel.waypoint_total = int(total if total is not None
                                 else (getattr(progress, "total", 0) or 0))

    def get_telemetry(self) -> DroneTelemetry:
        with self._lock:
            return self._telemetry

    # -- mission upload -----------------------------------------------------

    def _raw_items(self, items: list[dict[str, Any]], mission_type: int) -> list[Any]:
        """Planner items as MAVSDK MissionRaw items.

        mission_raw carries MAVLink items verbatim, so gimbal, yaw, dwell and trigger
        commands survive -- the high-level Mission plugin would drop every one of them.
        PX4 does not reserve sequence 0 for home the way ArduPilot does, so no home slot
        is added here.
        """
        MissionItem = self._mission_raw.MissionItem

        # The same scaling rule as the pymavlink bridge: positional commands carry
        # degrees scaled to int32, the rest carry plain values in x/y.
        from .mission_planner_bridge import _NON_POSITIONAL_COMMANDS

        raw = []
        for seq, item in enumerate(items):
            command = int(item.get("command", 16))
            if command in _NON_POSITIONAL_COMMANDS:
                x = int(float(item.get("lat", item.get("x", 0.0))))
                y = int(float(item.get("lon", item.get("y", 0.0))))
            else:
                x = int(round(float(item.get("lat", 0.0)) * 1e7))
                y = int(round(float(item.get("lon", 0.0)) * 1e7))
            raw.append(MissionItem(
                seq, int(item.get("frame", 3)), command, 1 if seq == 0 else 0,
                int(item.get("autocontinue", 1)),
                float(item.get("param1", 0.0)), float(item.get("param2", 0.0)),
                float(item.get("param3", 0.0)), float(item.get("param4", 0.0)),
                x, y, float(item.get("alt", 0.0)), mission_type,
            ))
        return raw

    def _upload(self, name: str, items: list[dict[str, Any]], mission_type: int) -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, name, "Not connected.")
        if not items:
            return CommandResult(False, name, "Nothing to upload: the list is empty.")
        uploader = getattr(getattr(self._drone, "mission_raw", None), name, None)
        if uploader is None:
            return CommandResult(False, name, f"This MAVSDK version has no mission_raw.{name}.")
        try:
            raw = self._raw_items(items, mission_type)
            self._run(uploader(raw))
        except Exception as exc:  # noqa: BLE001 - MissionRawError carries the reason
            return CommandResult(False, name, f"Vehicle rejected the upload: {exc}", ack=ACK_DENIED)
        # mission_raw.upload_* resolves only on the vehicle's MISSION_ACK.
        self._last_upload = time.monotonic()
        if mission_type == 0:
            # Progress is shown against the plan from the moment the vehicle holds it;
            # MAVSDK's mission_progress only speaks once the mission is running.
            with self._lock:
                self._mission_total = len(items)
                self._telemetry.waypoint_total = len(items)
                self._telemetry.waypoint_index = 0
        return CommandResult(True, name,
                             f"Uploaded {len(items)} items; the vehicle acknowledged.",
                             ack=ACK_ACCEPTED)

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_mission", mission_items, 0)

    def upload_geofence(self, fence_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_geofence", fence_items, 1)

    def upload_rally_points(self, rally_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_rally_points", rally_items, 2)

    def clear_geofence(self) -> CommandResult:
        """Remove the vehicle's fence, so it matches a plan that has none."""
        return self._call("clear_geofence", "geofence", "clear_geofence",
                          done="Cleared on the vehicle.")

    # -- mission download ---------------------------------------------------

    def _download(self, method: str, timeout_s: float) -> list[dict[str, Any]]:
        """Read one item list back from the vehicle, in the form the planner wrote it.

        The same dicts the pymavlink link returns, so a plan can be compared with what
        the aircraft holds whichever driver is connected.
        """
        if not self._connected or self._drone is None:
            raise RuntimeError("Not connected.")
        downloader = getattr(getattr(self._drone, "mission_raw", None), method, None)
        if downloader is None:
            raise RuntimeError(f"This MAVSDK version has no mission_raw.{method}.")
        from .mission_planner_bridge import _NON_POSITIONAL_COMMANDS, _planner_frame

        items = []
        for raw in self._run(downloader(), timeout=timeout_s):
            command = int(raw.command)
            positional = command not in _NON_POSITIONAL_COMMANDS
            items.append({
                "seq": int(raw.seq), "command": command, "frame": _planner_frame(raw.frame),
                "param1": float(raw.param1), "param2": float(raw.param2),
                "param3": float(raw.param3), "param4": float(raw.param4),
                "lat": float(raw.x) / 1e7 if positional else float(raw.x),
                "lon": float(raw.y) / 1e7 if positional else float(raw.y),
                "alt": float(raw.z), "autocontinue": int(raw.autocontinue),
            })
        return items

    def download_mission(self, timeout_s: float = 30.0) -> list[dict[str, Any]]:
        """The flight plan stored on the vehicle. PX4 keeps no home slot to strip."""
        items = self._download("download_mission", timeout_s)
        with self._lock:
            self._mission_total = len(items)
            self._telemetry.waypoint_total = len(items)
        return items

    def download_geofence(self, timeout_s: float = 30.0) -> list[dict[str, Any]]:
        return self._download("download_geofence", timeout_s)

    def download_rally_points(self, timeout_s: float = 30.0) -> list[dict[str, Any]]:
        # MAVSDK spells this one without the underscore.
        return self._download("download_rallypoints", timeout_s)

    # -- commands -----------------------------------------------------------

    def _call(self, name: str, plugin: str, method: str, *args: Any,
              done: str = "Accepted by the vehicle.") -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, name, "Not connected.")
        target = getattr(getattr(self._drone, plugin, None), method, None)
        if target is None:
            return CommandResult(False, name, f"MAVSDK has no {plugin}.{method}.")
        try:
            self._run(target(*args))
        except Exception as exc:  # noqa: BLE001 - ActionError / MissionError carry the reason
            text = str(exc) or type(exc).__name__
            if text.upper().startswith("TIMEOUT") or isinstance(exc, TimeoutError):
                # Silence is not a refusal: the vehicle may well have acted.
                return CommandResult(
                    False, name, "Sent, but the vehicle did not acknowledge it in time. It "
                    f"may still have acted -- check the aircraft. ({text})", ack=ACK_NONE)
            return CommandResult(False, name, text, ack=ACK_DENIED)
        # MAVSDK's action and mission calls resolve on the vehicle's COMMAND_ACK, so a
        # call that returned is an acknowledged one, not merely a transmitted one.
        return CommandResult(True, name, done, ack=ACK_ACCEPTED)

    def arm(self, force: bool = False) -> CommandResult:
        return self._call("arm", "action", "arm", done="Armed.")

    def disarm(self, force: bool = False) -> CommandResult:
        return self._call("disarm", "action", "disarm", done="Disarmed.")

    def takeoff(self, altitude_m: float) -> CommandResult:
        if altitude_m <= 0:
            return CommandResult(False, "takeoff", "Takeoff altitude must be positive.")
        setter = self._call("takeoff", "action", "set_takeoff_altitude", float(altitude_m))
        if not setter.success:
            return setter
        return self._call("takeoff", "action", "takeoff", done=f"Taking off to {altitude_m:g} m.")

    def land(self) -> CommandResult:
        return self._mode_command("land", "action", "land", "LAND", "Landing.")

    def set_return_altitude(self, altitude_m: float) -> CommandResult:
        """PX4's RTL_RETURN_ALT, set and read back, so RTL climbs to what the plan says."""
        name = "set_return_altitude"
        result = self._call(name, "action", "set_return_to_launch_altitude",
                            float(altitude_m))
        if not result.success:
            return result
        try:
            held = float(self._run(self._drone.action.get_return_to_launch_altitude()))
        except Exception as exc:  # noqa: BLE001 - ActionError carries the reason
            return CommandResult(False, name, f"Set, but it could not be read back: {exc}")
        if abs(held - float(altitude_m)) > 0.01:
            return CommandResult(False, name, f"Asked for {altitude_m:g} m; the vehicle "
                                 f"holds {held:g} m.", ack=ACK_DENIED)
        return CommandResult(True, name, f"Return altitude {held:g} m on the vehicle.",
                             ack=ACK_ACCEPTED)

    def emergency_stop(self) -> CommandResult:
        return self._call("emergency_stop", "action", "kill", done="Motors killed.")

    def start_mission(self) -> CommandResult:
        """Enter MISSION and arm: PX4 takes off into its mission once both hold.

        mission.start_mission only switches the mode. This used to be the whole start, so
        on a real PX4 (v1.15 SITL) Start Mission answered "Mission started." while the
        aircraft sat disarmed on the ground in MISSION mode and never left it. The mode is
        entered first, so a vehicle that refuses it (no valid mission) is never armed; the
        arm that follows is what launches it.
        """
        with self._lock:
            was_armed = bool(self._telemetry.armed)
        started = self._mode_command("start_mission", "mission", "start_mission", "AUTO",
                                     "Mission started.")
        # PX4 refuses MISSION while it is still checking a just-uploaded plan; MAVSDK
        # gives no reason with the denial, so a refusal inside the settle window is
        # tried again until the window closes.
        settle_until = self._last_upload + self.PX4_MISSION_SETTLE_S
        while (not started.success and started.ack == ACK_DENIED
               and time.monotonic() < settle_until):
            time.sleep(1.0)
            started = self._mode_command("start_mission", "mission", "start_mission",
                                         "AUTO", "Mission started.")
        if not started.success or was_armed:
            return started
        armed = self._call("start_mission", "action", "arm",
                           done="Mission started; armed and taking off.")
        if not armed.success:
            armed.message = (f"The vehicle entered MISSION but refused to arm, so it is "
                             f"still on the ground: {armed.message}")
        return armed

    def _mode_command(self, name: str, plugin: str, method: str, expected: str,
                      done: str) -> CommandResult:
        """A command that puts the vehicle in a mode; an unacknowledged one is checked.

        When no acknowledgement arrives, the vehicle's own reported mode decides: against
        a real PX4 a resume whose ack came late was reported as failed while PX4 flew on
        with the mission.
        """
        result = self._call(name, plugin, method, done=done)
        if result.success or result.ack != ACK_NONE:
            return result
        deadline = time.monotonic() + self.MODE_CONFIRM_S
        while time.monotonic() < deadline:
            with self._lock:
                current = (self._telemetry.flight_mode or "").upper()
            if current == expected:
                return CommandResult(True, name, f"{done} No acknowledgement arrived; the "
                                     f"vehicle reports {expected}.", ack=ACK_NONE)
            time.sleep(0.1)
        return result

    def pause_mission(self) -> CommandResult:
        return self._mode_command("pause_mission", "mission", "pause_mission", "LOITER",
                                  "Mission paused.")

    def resume_mission(self) -> CommandResult:
        return self._mode_command("resume_mission", "mission", "start_mission", "AUTO",
                                  "Mission resumed.")

    def return_to_home(self) -> CommandResult:
        return self._mode_command("return_to_home", "action", "return_to_launch", "RTL",
                                  "RTH triggered.")

    def abort_mission(self) -> CommandResult:
        return self._mode_command("abort_mission", "action", "hold", "LOITER",
                                  "Mission aborted — hold mode.")

    def loiter(self) -> CommandResult:
        return self.set_flight_mode("HOLD")

    def set_flight_mode(self, mode: str, *, verify: bool = True,
                        timeout_s: float = 3.0) -> CommandResult:
        """Enter a mode through the MAVSDK call that reaches it, and confirm it.

        Confirmed against the vehicle's reported mode, as the pymavlink bridge does:
        manual override depends on this answer being about the aircraft rather than
        about the request.
        """
        wanted = str(mode).upper()
        entry = _MAVSDK_MODE_ENTRY.get(wanted)
        if entry is None:
            return CommandResult(
                False, "set_flight_mode",
                f"MAVSDK cannot enter {wanted} directly. Available: "
                f"{', '.join(sorted(_MAVSDK_MODE_ENTRY))}.")
        result = self._call("set_flight_mode", *entry, done=f"Mode {wanted} requested.")
        if not result.success or not verify:
            return result

        expected = {"HOLD": "LOITER", "POSHOLD": "LOITER", "BRAKE": "LOITER",
                    "MISSION": "AUTO", "RETURN_TO_LAUNCH": "RTL"}.get(wanted, wanted)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                current = (self._telemetry.flight_mode or "").upper()
            if current == expected:
                return CommandResult(True, "set_flight_mode", f"Mode is {expected}.",
                                     ack=ACK_ACCEPTED)
            time.sleep(0.1)
        with self._lock:
            current = (self._telemetry.flight_mode or "unknown").upper()
        return CommandResult(
            False, "set_flight_mode",
            f"The vehicle accepted the request but did not enter {expected} within "
            f"{timeout_s:.0f} s; it is still in {current}.", ack=ACK_ACCEPTED)


# ── No vehicle at all ─────────────────────────────────────────────────────────

class NoDroneClient:
    """The absence of a vehicle, stated rather than simulated.

    Used where a caller did not name a driver. Defaulting to MockDroneClient there gave
    SDK users a simulated aircraft -- RTK fix, fourteen satellites, a charged battery --
    that they had not asked for, and a preflight that passed against it.
    """

    def connect(self, connection_uri: str = "") -> None:
        raise RuntimeError(
            "No drone driver was chosen. Create one with create_drone_client('pymavlink') "
            "for a real autopilot, or create_drone_client('mock') for a simulation.")

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return False

    def get_telemetry(self) -> DroneTelemetry:
        return DroneTelemetry(connected=False)

    def _refuse(self, command: str) -> CommandResult:
        return CommandResult(False, command, "No drone is connected.")

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        return self._refuse("upload_mission")

    def start_mission(self) -> CommandResult:
        return self._refuse("start_mission")

    def pause_mission(self) -> CommandResult:
        return self._refuse("pause_mission")

    def resume_mission(self) -> CommandResult:
        return self._refuse("resume_mission")

    def return_to_home(self) -> CommandResult:
        return self._refuse("return_to_home")

    def abort_mission(self) -> CommandResult:
        return self._refuse("abort_mission")

    def set_flight_mode(self, mode: str) -> CommandResult:
        return self._refuse("set_flight_mode")


# ── Factory ───────────────────────────────────────────────────────────────────

# Every name a user or a UI might reasonably call the pymavlink link by. The default UI
# offers "MAVLink (real autopilot)" with the value "mavlink", which was missing here, so
# the one option for flying a real aircraft failed with "Unknown drone driver".
PYMAVLINK_DRIVERS = ("pymavlink", "mavlink", "mission_planner", "mp", "ardupilot", "qgc")
SIMULATED_DRIVERS = frozenset({"mock", "sitl"})


def create_drone_client(driver: str = "mock"):
    """
    Create a drone client.
    Supported drivers: 'mock', 'mavsdk' (alias 'px4'), and the pymavlink link under any
    of 'pymavlink', 'mavlink', 'mission_planner', 'mp', 'ardupilot', 'qgc'.
    """
    driver = str(driver or "mock").lower().strip()
    if driver == "mock":
        return MockDroneClient()
    if driver in ("mavsdk", "px4"):
        return MAVSDKDroneClient()
    if driver in PYMAVLINK_DRIVERS:
        from .mission_planner_bridge import MissionPlannerDroneClient
        return MissionPlannerDroneClient()
    raise ValueError(
        f"Unknown drone driver: {driver!r}. Supported: 'mock', 'mavsdk', 'pymavlink' "
        "(also accepted: 'mavlink', 'mission_planner')."
    )
