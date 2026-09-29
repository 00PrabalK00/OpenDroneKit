"""A stand-in for the `mavsdk` package, at its public API.

MAVSDK is not installed in this environment and needs a PX4 on the other end, so the
MAVSDK driver is exercised against this: a module with the same shape -- ``System``
with ``core``, ``telemetry``, ``action``, ``mission`` and ``mission_raw`` plugins,
async methods and async-generator telemetry streams -- whose vehicle holds a little
state (armed, flight mode, the stored mission) and can refuse.

Only the SDK boundary is faked. Everything on this side of it -- the driver's event
loop thread, its telemetry consumers, its item conversion, its mode confirmation, and
the Api and session above it -- is the production code.
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from typing import Any


class ActionError(Exception):
    pass


class MissionRawError(Exception):
    pass


@dataclass
class MissionItem:
    seq: int
    frame: int
    command: int
    current: int
    autocontinue: int
    param1: float
    param2: float
    param3: float
    param4: float
    x: int
    y: int
    z: float
    mission_type: int


class _Mode:
    def __init__(self, name: str):
        self.name = name


class FakePX4:
    """The vehicle's side: what the fake SDK reports and what it has been told."""

    def __init__(self) -> None:
        self.mode = "HOLD"
        self.armed = False
        self.stored: dict[int, list[MissionItem]] = {0: [], 1: [], 2: []}
        self.reject_upload = False
        self.deny_actions: set[str] = set()
        self.ignore_mode_for: set[str] = set()
        self.calls: list[str] = []
        self.battery_remaining = 64.0  # MAVSDK 2.x: percent


def build_module(vehicle: FakePX4, version: str = "2.8.0") -> types.ModuleType:
    """A module object to install as sys.modules['mavsdk'] (and 'mavsdk.mission_raw')."""

    async def stream(make):
        while True:
            yield make()
            await asyncio.sleep(0.05)

    class Telemetry:
        def position(self):
            return stream(lambda: types.SimpleNamespace(
                latitude_deg=47.3977, longitude_deg=8.5456,
                absolute_altitude_m=488.0, relative_altitude_m=25.0))

        def battery(self):
            return stream(lambda: types.SimpleNamespace(
                remaining_percent=vehicle.battery_remaining, voltage_v=15.4))

        def gps_info(self):
            return stream(lambda: types.SimpleNamespace(
                num_satellites=12, fix_type=types.SimpleNamespace(value=3)))

        def raw_gps(self):
            return stream(lambda: types.SimpleNamespace(hdop=0.9))

        def armed(self):
            return stream(lambda: vehicle.armed)

        def flight_mode(self):
            return stream(lambda: _Mode(vehicle.mode))

        def heading(self):
            return stream(lambda: types.SimpleNamespace(heading_deg=45.0))

        def velocity_ned(self):
            return stream(lambda: types.SimpleNamespace(north_m_s=3.0, east_m_s=4.0, down_m_s=0.0))

        def home(self):
            return stream(lambda: types.SimpleNamespace(
                latitude_deg=47.3977, longitude_deg=8.5456, absolute_altitude_m=463.0))

    class Core:
        def connection_state(self):
            return stream(lambda: types.SimpleNamespace(is_connected=True))

    def action_call(name: str, effect=None):
        async def run(*args: Any) -> None:
            vehicle.calls.append(name)
            if name in vehicle.deny_actions:
                raise ActionError(f"COMMAND_DENIED: {name}")
            if effect is not None and name not in vehicle.ignore_mode_for:
                effect(*args)
        return run

    def set_mode(mode: str):
        return lambda *args: setattr(vehicle, "mode", mode)

    class Action:
        arm = staticmethod(action_call("arm", lambda: setattr(vehicle, "armed", True)))
        disarm = staticmethod(action_call("disarm", lambda: setattr(vehicle, "armed", False)))
        hold = staticmethod(action_call("hold", set_mode("HOLD")))
        return_to_launch = staticmethod(action_call("return_to_launch", set_mode("RETURN_TO_LAUNCH")))
        land = staticmethod(action_call("land", set_mode("LAND")))
        kill = staticmethod(action_call("kill", lambda: setattr(vehicle, "armed", False)))
        set_takeoff_altitude = staticmethod(action_call("set_takeoff_altitude"))
        takeoff = staticmethod(action_call("takeoff", set_mode("TAKEOFF")))

    class Mission:
        start_mission = staticmethod(action_call("start_mission", set_mode("MISSION")))
        pause_mission = staticmethod(action_call("pause_mission", set_mode("HOLD")))

        def mission_progress(self):
            return stream(lambda: types.SimpleNamespace(current=0, total=len(vehicle.stored[0])))

    def upload(kind: int):
        async def run(items):
            vehicle.calls.append(f"upload:{kind}")
            if vehicle.reject_upload:
                raise MissionRawError("MissionRawResult.Result.ERROR: vehicle rejected")
            vehicle.stored[kind] = list(items)
        return run

    class MissionRaw:
        upload_mission = staticmethod(upload(0))
        upload_geofence = staticmethod(upload(1))
        upload_rally_points = staticmethod(upload(2))

    class System:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.core = Core()
            self.telemetry = Telemetry()
            self.action = Action()
            self.mission = Mission()
            self.mission_raw = MissionRaw()

        async def connect(self, system_address: str = "") -> None:
            vehicle.calls.append(f"connect:{system_address}")

    module = types.ModuleType("mavsdk")
    module.__version__ = version
    module.System = System
    mission_raw = types.ModuleType("mavsdk.mission_raw")
    mission_raw.MissionItem = MissionItem
    mission_raw.MissionRawError = MissionRawError
    module.mission_raw = mission_raw
    return module
