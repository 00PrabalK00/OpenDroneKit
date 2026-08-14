"""Drone connection abstraction — Protocol + Mock implementation."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

# ── Data models ───────────────────────────────────────────────────────────────

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
    battery_pct: float = 0.0
    battery_v: float = 0.0
    gps_fix: int = 0          # 0=none, 1=no fix, 2=2D, 3=3D, 4=RTK float, 5=RTK fixed
    satellites: int = 0
    hdop: float = 99.9
    home_lat: float = 0.0
    home_lon: float = 0.0
    home_alt: float = 0.0
    home_set: bool = False
    waypoint_index: int = 0
    waypoint_total: int = 0
    distance_to_next_m: float = 0.0
    link_quality_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def gps_ok(self) -> bool:
        return self.gps_fix >= 3 and self.satellites >= 6

    @property
    def battery_ok(self) -> bool:
        return self.battery_pct >= 25.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "armed": self.armed,
            "flight_mode": self.flight_mode,
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "altitude_rel_m": round(self.altitude_rel_m, 2),
            "altitude_abs_m": round(self.altitude_abs_m, 2),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "battery_pct": round(self.battery_pct, 1),
            "battery_v": round(self.battery_v, 2),
            "gps_fix": self.gps_fix,
            "satellites": self.satellites,
            "hdop": round(self.hdop, 2),
            "home_set": self.home_set,
            "waypoint_index": self.waypoint_index,
            "waypoint_total": self.waypoint_total,
            "link_quality_pct": round(self.link_quality_pct, 1),
        }


@dataclass
class CommandResult:
    success: bool
    command: str
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "command": self.command, "message": self.message}


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

class MAVSDKDroneClient:
    """
    Real drone client using MAVSDK-Python.
    Requires `pip install mavsdk` and a running MAVSDK server.

    A missing mavsdk raises on connect rather than degrading to a no-op: a client that
    reported itself connected while doing nothing would let a pilot believe a mission
    had been uploaded.
    """

    def __init__(self):
        self._drone = None
        self._connected = False
        self._telemetry = DroneTelemetry()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _run(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    def connect(self, connection_uri: str = "udp://:14540") -> None:
        try:
            from mavsdk import System
            self._drone = System()
            self._run(self._drone.connect(system_address=connection_uri))
            self._connected = True
        except ImportError:
            raise RuntimeError(
                "MAVSDK not installed. Install with: pip install mavsdk\n"
                "Or use MockDroneClient for offline testing."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to drone at {connection_uri}: {exc}") from exc

    def disconnect(self) -> None:
        self._connected = False
        self._drone = None

    def is_connected(self) -> bool:
        return self._connected

    def get_telemetry(self) -> DroneTelemetry:
        return self._telemetry

    def upload_mission(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        return CommandResult(False, "upload_mission", "MAVSDK async upload not yet wired — use run_upload().")

    def start_mission(self) -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, "start_mission", "Not connected.")
        try:
            self._run(self._drone.mission.start_mission())
            return CommandResult(True, "start_mission", "Mission started.")
        except Exception as exc:
            return CommandResult(False, "start_mission", str(exc))

    def pause_mission(self) -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, "pause_mission", "Not connected.")
        try:
            self._run(self._drone.mission.pause_mission())
            return CommandResult(True, "pause_mission", "Mission paused.")
        except Exception as exc:
            return CommandResult(False, "pause_mission", str(exc))

    def resume_mission(self) -> CommandResult:
        return self.start_mission()

    def return_to_home(self) -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, "return_to_home", "Not connected.")
        try:
            self._run(self._drone.action.return_to_launch())
            return CommandResult(True, "return_to_home", "RTH triggered.")
        except Exception as exc:
            return CommandResult(False, "return_to_home", str(exc))

    def abort_mission(self) -> CommandResult:
        if not self._connected or self._drone is None:
            return CommandResult(False, "abort_mission", "Not connected.")
        try:
            self._run(self._drone.action.hold())
            return CommandResult(True, "abort_mission", "Mission aborted — hold mode.")
        except Exception as exc:
            return CommandResult(False, "abort_mission", str(exc))

    def set_flight_mode(self, mode: str) -> CommandResult:
        return CommandResult(False, "set_flight_mode", "Direct mode set not supported via MAVSDK high-level API.")


# ── Factory ───────────────────────────────────────────────────────────────────

def create_drone_client(driver: str = "mock"):
    """
    Create a drone client.
    Supported drivers: 'mock', 'mavsdk', 'mission_planner' (alias 'mp'), 'pymavlink'.
    """
    driver = str(driver or "mock").lower().strip()
    if driver == "mock":
        return MockDroneClient()
    if driver in ("mavsdk", "px4"):
        return MAVSDKDroneClient()
    if driver in ("mission_planner", "mp", "pymavlink", "ardupilot", "qgc"):
        from .mission_planner_bridge import MissionPlannerDroneClient
        return MissionPlannerDroneClient()
    raise ValueError(
        f"Unknown drone driver: {driver!r}. Supported: 'mock', 'mavsdk', 'mission_planner', 'pymavlink'."
    )
