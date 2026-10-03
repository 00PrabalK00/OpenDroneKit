"""Drone connection abstraction — Protocol + Mock implementation."""

from __future__ import annotations

import asyncio
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

class _Publisher:
    """Subscribers for a driver's own messages (STATUSTEXT and the like).

    The pymavlink bridge has always published STATUSTEXT this way; the other drivers
    share the same three methods so the session can listen to whichever is connected.
    Callbacks run on the driver's thread and must not block.
    """

    def _subscribers(self) -> list[Any]:
        found = self.__dict__.get("_event_subscribers")
        if found is None:
            found = self.__dict__["_event_subscribers"] = []
        return found

    def subscribe(self, callback) -> None:
        subscribers = self._subscribers()
        if callback not in subscribers:
            subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        subscribers = self._subscribers()
        if callback in subscribers:
            subscribers.remove(callback)

    def _publish(self, event: dict[str, Any]) -> None:
        for callback in list(self._subscribers()):
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - a broken subscriber must not stop the driver
                pass


# MAVLink MAV_SEVERITY values, for the messages the simulator says about itself.
SEVERITY_CRITICAL = 2
SEVERITY_WARNING = 4
SEVERITY_INFO = 6

# The simulator's flight model: plain kinematics, labelled simulated everywhere it shows.
_SIM_CRUISE_MPS = 5.0
_SIM_CLIMB_MPS = 3.0
_SIM_DESCENT_MPS = 1.5
_SIM_ARRIVAL_M = 1.0
_METRES_PER_DEG_LAT = 111_320.0
# MAV_CMD_DO_CHANGE_SPEED: param2 is the speed the plan asks for.
_MAV_CMD_DO_CHANGE_SPEED = 178
# MAV_CMD_NAV_* items that are positions (waypoint, loiters, land, takeoff, spline); a
# DO_ command reuses x/y for parameters, so its "latitude" is not a place.
_NAV_POSITIONAL = frozenset({16, 17, 18, 19, 21, 22, 31, 82})


class MockDroneClient(_Publisher):
    """
    Simulated drone for development and UI testing.

    A small kinematic simulator: once a mission is uploaded and started it flies the
    uploaded items in order at the plan's speed, returns home and lands, so the Fly
    screen can be rehearsed end to end. Nothing it reports is a measurement: the
    session marks every value from this driver `is_simulated`, and the interface tags it.
    """

    def __init__(self):
        self._connected = False
        self._armed = False
        self._flight_mode = "STABILIZE"
        self._mission_running = False
        self._mission_paused = False
        self._waypoint_index = 0
        self._waypoints: list[dict[str, Any]] = []
        self._route: list[dict[str, float]] = []
        self._target = 0
        self._phase = "landed"          # landed | takeoff | mission | hold | rth | landing
        self._takeoff_alt = 0.0
        self._cruise = _SIM_CRUISE_MPS
        self._lat = 37.7749
        self._lon = -122.4194
        self._home = (self._lat, self._lon)
        self._alt = 0.0
        self._heading = 0.0
        self._speed = 0.0
        self._battery = 87.5
        self._battery_v = 22.1
        self._link = 95.0
        self._photos = 0
        self._t0 = time.time()
        self._last_tick = time.time()
        self._log: list[dict[str, Any]] = []

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, connection_uri: str = "mock://") -> None:
        self._connected = True
        self._flight_mode = "LOITER"
        self._last_tick = time.time()
        self._log_event("connected", f"Mock drone connected to {connection_uri}")

    def disconnect(self) -> None:
        self._connected = False
        self._armed = False
        self._mission_running = False
        self._log_event("disconnected", "Mock drone disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    def place_at(self, latitude: float, longitude: float) -> bool:
        """Put the simulated aircraft on the ground at a position (it is a simulator).

        Refused while it is flying: a real aircraft cannot be moved by a ground station.
        """
        if self._phase != "landed" or self._armed:
            return False
        self._lat, self._lon = float(latitude), float(longitude)
        self._home = (self._lat, self._lon)
        self._alt = 0.0
        self._log_event("placed", f"Simulated aircraft placed at {self._lat:.6f}, {self._lon:.6f}.")
        return True

    # ── Simulation ───────────────────────────────────────────────────────────

    def _offset_m(self, lat: float, lon: float) -> tuple[float, float]:
        north = (lat - self._lat) * _METRES_PER_DEG_LAT
        east = (lon - self._lon) * _METRES_PER_DEG_LAT * math.cos(math.radians(self._lat))
        return east, north

    def _move_towards(self, lat: float, lon: float, alt: float, dt: float) -> bool:
        """Advance toward a point at cruise speed; True on arrival."""
        east, north = self._offset_m(lat, lon)
        distance = math.hypot(east, north)
        step = self._cruise * dt
        if distance > 0.01:
            self._heading = math.degrees(math.atan2(east, north)) % 360.0
        arrived = distance <= max(step, _SIM_ARRIVAL_M)
        if arrived:
            self._lat, self._lon = lat, lon
            self._speed = min(self._cruise, distance / dt) if dt > 0 else 0.0
        else:
            k = step / distance
            self._lat += (north * k) / _METRES_PER_DEG_LAT
            self._lon += (east * k) / (_METRES_PER_DEG_LAT * math.cos(math.radians(self._lat)))
            self._speed = self._cruise
        climb = _SIM_CLIMB_MPS * dt
        self._alt += max(-climb, min(climb, alt - self._alt))
        return arrived and abs(alt - self._alt) < _SIM_ARRIVAL_M

    def _advance(self) -> None:
        now = time.time()
        dt = max(0.0, min(5.0, now - self._last_tick))
        self._last_tick = now
        self._battery = max(0.0, self._battery - (0.05 if self._armed else 0.002) * dt)
        self._speed = 0.0
        if dt <= 0:
            return
        phase = self._phase
        if phase == "mission" and not self._mission_paused:
            if not self._route:
                # A mission with no positions (items without coordinates) still steps.
                t = now - self._t0
                if self._waypoints and int(t * 0.5) > self._waypoint_index:
                    self._waypoint_index = min(int(t * 0.5), max(0, len(self._waypoints) - 1))
                return
            point = self._route[self._target]
            if self._move_towards(point["lat"], point["lon"], point["alt"], dt):
                self._target += 1
                if self._target >= len(self._route):
                    self._mission_running = False
                    self._phase = "rth"
                    self._flight_mode = "RTL"
                    self._log_event("mission_complete", "Mission complete, returning to launch.")
                else:
                    self._waypoint_index = int(self._route[self._target]["seq"])
        elif phase == "takeoff":
            if self._move_towards(self._lat, self._lon, self._takeoff_alt, dt):
                self._phase = "hold"
                self._log_event("takeoff_complete", f"Reached {self._takeoff_alt:g} m.")
        elif phase == "rth":
            home_lat, home_lon = self._home
            if self._move_towards(home_lat, home_lon, self._alt, dt):
                self._phase = "landing"
                self._flight_mode = "LAND"
                self._log_event("landing", "Over home, landing.")
        elif phase == "landing":
            self._alt = max(0.0, self._alt - _SIM_DESCENT_MPS * dt)
            if self._alt <= 0.0:
                self._phase = "landed"
                self._armed = False
                self._mission_running = False
                self._log_event("landed", "Landed. Disarmed.")

    def get_telemetry(self) -> DroneTelemetry:
        if not self._connected:
            return DroneTelemetry(connected=False)
        self._advance()
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
            home_lat=self._home[0],
            home_lon=self._home[1],
            home_alt=15.0,
            home_set=True,
            waypoint_index=self._waypoint_index,
            waypoint_total=len(self._waypoints),
            distance_to_next_m=self._distance_to_next(),
            link_quality_pct=min(100.0, self._link + random.uniform(-2.0, 2.0)),
            raw={"simulated_photos": self._photos},
        )

    def _distance_to_next(self) -> float:
        if self._phase == "mission" and self._route and self._target < len(self._route):
            point = self._route[self._target]
            east, north = self._offset_m(point["lat"], point["lon"])
            return math.hypot(east, north)
        return 0.0

    # ── Mission control ───────────────────────────────────────────────────────

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "upload_mission", "Not connected.")
        self._waypoints = mission_items
        self._waypoint_index = 0
        self._target = 0
        self._cruise = _SIM_CRUISE_MPS
        route = []
        for item in mission_items:
            lat = float(item.get("lat", item.get("latitude", 0.0)) or 0.0)
            lon = float(item.get("lon", item.get("longitude", 0.0)) or 0.0)
            if int(item.get("command", 0) or 0) == _MAV_CMD_DO_CHANGE_SPEED:
                speed = float(item.get("param2", 0.0) or 0.0)
                if speed > 0:
                    self._cruise = speed
            if int(item.get("command", 16) or 0) in _NAV_POSITIONAL and (lat or lon):
                route.append({"seq": int(item.get("seq", len(route))), "lat": lat, "lon": lon,
                              "alt": float(item.get("alt", item.get("altitude", 0.0)) or 0.0)})
        self._route = route
        # A simulator can be anywhere, so on the ground it starts at the mission's first
        # position rather than flying in from wherever it was created.
        if route and self._phase == "landed" and not self._armed:
            self._lat, self._lon = route[0]["lat"], route[0]["lon"]
            self._home = (self._lat, self._lon)
        self._log_event("mission_uploaded", f"Uploaded {len(mission_items)} waypoints.")
        return CommandResult(True, "upload_mission", f"Uploaded {len(mission_items)} items.",
                             ack=ACK_ACCEPTED)

    def upload_geofence(self, fence_items: list[dict[str, Any]]) -> CommandResult:
        """Kept, not enforced: the simulator flies its route and checks no fence."""
        if not self._connected:
            return CommandResult(False, "upload_geofence", "Not connected.")
        self._fence = list(fence_items)
        self._log_event("fence_uploaded", f"Geofence received ({len(fence_items)} items).")
        return CommandResult(True, "upload_geofence", f"Uploaded {len(fence_items)} fence items.",
                             ack=ACK_ACCEPTED)

    def upload_rally_points(self, rally_items: list[dict[str, Any]]) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "upload_rally_points", "Not connected.")
        self._rally = list(rally_items)
        self._log_event("rally_uploaded", f"Rally points received ({len(rally_items)}).")
        return CommandResult(True, "upload_rally_points", f"Uploaded {len(rally_items)} rally points.",
                             ack=ACK_ACCEPTED)

    def start_mission(self) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "start_mission", "Not connected.")
        if not self._waypoints:
            self._log_event("refused", "Start refused: no mission uploaded.", SEVERITY_WARNING)
            return CommandResult(False, "start_mission", "No mission uploaded.", ack=ACK_DENIED)
        self._advance()
        self._armed = True
        self._mission_running = True
        self._mission_paused = False
        self._flight_mode = "AUTO"
        self._phase = "mission"
        self._target = 0
        self._waypoint_index = int(self._route[0]["seq"]) if self._route else 0
        self._t0 = time.time()
        self._log_event("mission_started", "Mission started.")
        return CommandResult(True, "start_mission", "Mission started.", ack=ACK_ACCEPTED)

    def pause_mission(self) -> CommandResult:
        if not self._mission_running:
            return CommandResult(False, "pause_mission", "No active mission.", ack=ACK_DENIED)
        self._advance()
        self._mission_paused = True
        self._flight_mode = "LOITER"
        self._log_event("mission_paused", f"Paused at waypoint {self._waypoint_index}.")
        return CommandResult(True, "pause_mission", "Mission paused.", ack=ACK_ACCEPTED)

    def resume_mission(self) -> CommandResult:
        if not self._mission_running or not self._mission_paused:
            return CommandResult(False, "resume_mission", "Mission not paused.", ack=ACK_DENIED)
        self._advance()
        self._mission_paused = False
        self._flight_mode = "AUTO"
        self._log_event("mission_resumed", "Mission resumed.")
        return CommandResult(True, "resume_mission", "Mission resumed.", ack=ACK_ACCEPTED)

    def return_to_home(self) -> CommandResult:
        self._advance()
        self._mission_running = False
        self._mission_paused = False
        self._flight_mode = "RTL"
        self._phase = "rth" if self._alt > 0.0 else "landed"
        self._log_event("rth_triggered", "Return to home triggered.")
        return CommandResult(True, "return_to_home", "RTH triggered.", ack=ACK_ACCEPTED)

    def abort_mission(self) -> CommandResult:
        self._advance()
        self._mission_running = False
        self._mission_paused = False
        self._flight_mode = "LAND"
        self._phase = "landing" if self._alt > 0.0 else "landed"
        if self._phase == "landed":
            self._armed = False
        self._log_event("mission_aborted", "Mission aborted. Landing in place.", SEVERITY_CRITICAL)
        return CommandResult(True, "abort_mission", "Mission aborted. Landing in place.",
                             ack=ACK_ACCEPTED)

    def arm(self, force: bool = False) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "arm", "Not connected.")
        self._advance()
        self._armed = True
        self._log_event("armed", "Armed.")
        return CommandResult(True, "arm", "Armed.", ack=ACK_ACCEPTED)

    def disarm(self, force: bool = False) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "disarm", "Not connected.")
        self._advance()
        if self._alt > 0.5 and not force:
            self._log_event("refused", "Disarm refused: the aircraft is airborne.", SEVERITY_WARNING)
            return CommandResult(False, "disarm", "Refused: the aircraft is airborne.", ack=ACK_DENIED)
        self._armed = False
        self._mission_running = False
        self._log_event("disarmed", "Disarmed.")
        return CommandResult(True, "disarm", "Disarmed.", ack=ACK_ACCEPTED)

    def takeoff(self, altitude_m: float) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "takeoff", "Not connected.")
        if not self._armed:
            self._log_event("refused", "Takeoff refused: not armed.", SEVERITY_WARNING)
            return CommandResult(False, "takeoff", "Refused: arm first.", ack=ACK_DENIED)
        self._advance()
        self._takeoff_alt = float(altitude_m)
        self._phase = "takeoff"
        self._flight_mode = "GUIDED"
        self._log_event("takeoff", f"Taking off to {altitude_m:g} m.")
        return CommandResult(True, "takeoff", f"Taking off to {altitude_m:g} m.", ack=ACK_ACCEPTED)

    def land(self) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "land", "Not connected.")
        self._advance()
        self._mission_running = False
        self._mission_paused = False
        self._flight_mode = "LAND"
        self._phase = "landing" if self._alt > 0.0 else "landed"
        self._log_event("land", "Landing.")
        return CommandResult(True, "land", "Landing.", ack=ACK_ACCEPTED)

    def loiter(self) -> CommandResult:
        return self.set_flight_mode("LOITER")

    def emergency_stop(self) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "emergency_stop", "Not connected.")
        self._advance()
        self._armed = False
        self._mission_running = False
        self._phase = "landed"
        self._alt = 0.0
        self._log_event("emergency_stop", "Motors stopped.", SEVERITY_CRITICAL)
        return CommandResult(True, "emergency_stop", "Motors stopped.", ack=ACK_ACCEPTED)

    def set_flight_mode(self, mode: str) -> CommandResult:
        if not self._connected:
            return CommandResult(False, "set_flight_mode", "Not connected.")
        self._advance()
        self._flight_mode = mode.upper()
        if self._flight_mode in ("RTL", "RETURN_TO_LAUNCH"):
            self._phase = "rth" if self._alt > 0.0 else "landed"
            self._mission_running = False
        elif self._flight_mode == "LAND":
            self._phase = "landing" if self._alt > 0.0 else "landed"
            self._mission_running = False
        elif self._flight_mode == "AUTO" and self._mission_running:
            self._mission_paused = False
        elif self._mission_running:
            # A pilot mode: the simulator holds where it is.
            self._mission_paused = True
        elif self._phase == "takeoff":
            self._phase = "hold"
        self._log_event("mode", f"Mode {self._flight_mode}.")
        return CommandResult(True, "set_flight_mode", f"Mode set to {self._flight_mode}.",
                             ack=ACK_ACCEPTED)

    def camera(self):
        """A simulated payload: it counts the photographs it is told to take."""
        from .camera_control import CameraCapabilities, CameraController

        def send(command: int, *params: float) -> CommandResult:
            count = int(params[2]) if len(params) > 2 and params[2] else 1
            self._photos += max(1, count)
            self._log_event("camera", f"Camera triggered ({self._photos} so far).")
            return CommandResult(True, "camera", "The simulated camera took the photo.",
                                 ack=ACK_ACCEPTED)

        return CameraController(send, CameraCapabilities())

    def get_command_log(self) -> list[dict[str, Any]]:
        return list(self._log)

    def _log_event(self, event: str, detail: str, severity: int = SEVERITY_INFO) -> None:
        self._log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "detail": detail,
        })
        if len(self._log) > 500:
            self._log = self._log[-500:]
        self._publish({"type": "statustext", "severity": int(severity), "text": detail})


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


def _mavsdk_battery_percent(remaining: float, sdk_major: int) -> float:
    """MAVSDK 1.x reports remaining charge as 0..1; 2.x reports 0..100."""
    value = float(remaining)
    return value * 100.0 if sdk_major < 2 else value


class MAVSDKDroneClient(_Publisher):
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

    def __init__(self):
        self._drone = None
        self._connected = False
        self._telemetry = DroneTelemetry()
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tasks: list[Any] = []
        self._sdk_major = 2

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
        if loop is None:
            return

        async def shutdown() -> None:
            # Cancelled and awaited, so no stream is left half-closed on a stopped loop.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=3.0)
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not loop.is_running():
            loop.close()
        self._tasks = []
        self._thread = None
        self._loop = None

    # -- connection ---------------------------------------------------------

    def connect(self, connection_uri: str = "udp://:14540") -> None:
        try:
            import mavsdk
            from mavsdk import System
        except ImportError:
            raise RuntimeError(
                "MAVSDK not installed. Install with: pip install mavsdk\n"
                "Or use MockDroneClient for offline testing."
            )
        try:
            self._sdk_major = int(str(getattr(mavsdk, "__version__", "2")).split(".")[0] or 2)
        except ValueError:
            self._sdk_major = 2

        self._start_loop()
        try:
            self._drone = System()
            self._run(self._drone.connect(system_address=connection_uri),
                      timeout=self.CONNECT_TIMEOUT_S)
            # connect() returns once mavsdk_server is listening, not once a vehicle has
            # been heard. Without this a link with nothing on the other end would be
            # reported as connected.
            self._run(self._await_vehicle(), timeout=self.CONNECT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            self._drone = None
            self._stop_loop()
            reason = str(exc) or ("no vehicle answered within "
                                  f"{self.CONNECT_TIMEOUT_S:.0f} s")
            raise RuntimeError(f"Failed to connect to drone at {connection_uri}: {reason}") from exc

        self._connected = True
        with self._lock:
            self._telemetry = DroneTelemetry(connected=True)
        self._start_telemetry()

    async def _await_vehicle(self) -> None:
        async for state in self._drone.core.connection_state():
            if getattr(state, "is_connected", False):
                return

    def disconnect(self) -> None:
        self._connected = False
        self._stop_loop()
        self._drone = None
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
            progress = getattr(getattr(drone, "mission", None), "mission_progress", None)
            if progress is not None:
                self._tasks.append(loop.create_task(self._follow(progress, self._on_progress)))
            # STATUSTEXT (pre-arm failures, mode changes, failsafes) as MAVSDK sees it.
            status = getattr(getattr(drone, "telemetry", None), "status_text", None)
            if status is not None:
                self._tasks.append(loop.create_task(self._follow_status(status)))

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

    # MAVSDK StatusTextType names, as MAVLink MAV_SEVERITY numbers.
    _STATUS_SEVERITY = {"EMERGENCY": 0, "ALERT": 1, "CRITICAL": 2, "ERROR": 3,
                        "WARNING": 4, "NOTICE": 5, "INFO": 6, "DEBUG": 7}

    async def _follow_status(self, open_stream) -> None:
        """Publish each status text the vehicle sends, for the session's message log."""
        try:
            async for value in open_stream():
                kind = getattr(getattr(value, "type", None), "name", str(getattr(value, "type", "")))
                self._publish({"type": "statustext",
                               "severity": self._STATUS_SEVERITY.get(str(kind).upper(), 6),
                               "text": str(getattr(value, "text", "")).strip()})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a vehicle without the stream simply sends none
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

    @staticmethod
    def _on_progress(tel: DroneTelemetry, progress: Any) -> None:
        tel.waypoint_index = int(getattr(progress, "current", 0) or 0)
        tel.waypoint_total = int(getattr(progress, "total", 0) or 0)

    def get_telemetry(self) -> DroneTelemetry:
        with self._lock:
            return self._telemetry

    # -- mission upload -----------------------------------------------------

    @staticmethod
    def _raw_items(items: list[dict[str, Any]], mission_type: int) -> list[Any]:
        """Planner items as MAVSDK MissionRaw items.

        mission_raw carries MAVLink items verbatim, so gimbal, yaw, dwell and trigger
        commands survive -- the high-level Mission plugin would drop every one of them.
        PX4 does not reserve sequence 0 for home the way ArduPilot does, so no home slot
        is added here.
        """
        from mavsdk.mission_raw import MissionItem

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
        return CommandResult(True, name,
                             f"Uploaded {len(items)} items; the vehicle acknowledged.",
                             ack=ACK_ACCEPTED)

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_mission", mission_items, 0)

    def upload_geofence(self, fence_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_geofence", fence_items, 1)

    def upload_rally_points(self, rally_items: list[dict[str, Any]]) -> CommandResult:
        return self._upload("upload_rally_points", rally_items, 2)

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
            return CommandResult(False, name, str(exc) or type(exc).__name__, ack=ACK_DENIED)
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
        return self._call("land", "action", "land", done="Landing.")

    def emergency_stop(self) -> CommandResult:
        return self._call("emergency_stop", "action", "kill", done="Motors killed.")

    def start_mission(self) -> CommandResult:
        return self._call("start_mission", "mission", "start_mission", done="Mission started.")

    def pause_mission(self) -> CommandResult:
        return self._call("pause_mission", "mission", "pause_mission", done="Mission paused.")

    def resume_mission(self) -> CommandResult:
        return self._call("resume_mission", "mission", "start_mission", done="Mission resumed.")

    def return_to_home(self) -> CommandResult:
        return self._call("return_to_home", "action", "return_to_launch", done="RTH triggered.")

    def abort_mission(self) -> CommandResult:
        return self._call("abort_mission", "action", "hold", done="Mission aborted — hold mode.")

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
