"""Shared helpers for the ArduCopter SITL campaign tests.

Everything here drives or observes a real ArduPilot process. There is no simulated
vehicle in this module: the app is exercised through ``app.api.Api`` exactly as the UIs
use it, and the evidence is read from an independent MAVLink stream (MAVProxy's second
``--out``), so a test cannot pass on the strength of what the app *says* happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

import pytest

from tools.sitl.launch import SITL_HOME_LAT, SITL_HOME_LON

# Nav commands that carry a viewpoint, and the camera commands that fire a shutter.
NAV_WAYPOINT = 16
NAV_LOITER_TIME = 19
NAV_RETURN_TO_LAUNCH = 20
NAV_LAND = 21
NAV_TAKEOFF = 22
CONDITION_YAW = 115
DO_CHANGE_SPEED = 178
DO_DIGICAM_CONTROL = 203
DO_MOUNT_CONTROL = 205
DO_SET_CAM_TRIGG_DIST = 206
DO_GIMBAL_MANAGER_PITCHYAW = 1000
IMAGE_START_CAPTURE = 2000
NAV_COMMANDS = {NAV_WAYPOINT, NAV_LOITER_TIME, NAV_RETURN_TO_LAUNCH, NAV_LAND, NAV_TAKEOFF}
VIEWPOINT_COMMANDS = {NAV_WAYPOINT, NAV_LOITER_TIME}

# MISSION_ITEM_INT carries lat/lon as int32 degE7, so 1e-7 deg is the wire resolution.
LATLON_TOL_DEG = 1.5e-7
# z is a float32 on the wire; ArduPilot stores altitude as int32 centimetres.
ALT_TOL_M = 0.011
FLOAT32_TOL = 1e-3


def evidence_dir(name: str) -> Path:
    """Where a test writes its MAVLink trace: ODK_SITL_EVIDENCE_DIR when set."""
    root = Path(os.environ.get("ODK_SITL_EVIDENCE_DIR", "") or "sitl-evidence")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def offset_lonlat(north_m: float, east_m: float,
                  lat0: float = SITL_HOME_LAT, lon0: float = SITL_HOME_LON) -> list[float]:
    """A [lon, lat] that many metres north/east of the SITL home."""
    return [lon0 + east_m / (111_320.0 * math.cos(math.radians(lat0))),
            lat0 + north_m / 110_540.0]


def distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_m = 6_371_000.0
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi, d_lambda = math.radians(lat_b - lat_a), math.radians(lon_b - lon_a)
    value = (math.sin(d_phi / 2.0) ** 2
             + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2.0) ** 2)
    return radius_m * 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def angle_diff_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


# An Aukerman-like site beside the SITL home: an open field with a building in it. The
# field is what area templates (grid, corridor, ...) cover; the building is what the
# structure templates (facade, orbit, tower, ...) inspect. The launch point sits inside
# the field, as it does at Aukerman, so the inclusion fence contains the aircraft.
FIELD_POLYGON = [offset_lonlat(n, e) for n, e in ((-10, -10), (-10, 50), (40, 50), (40, -10))]
BUILDING_POLYGON = [offset_lonlat(n, e) for n, e in ((12, 18), (12, 30), (22, 30), (22, 18))]
ROUTE_LINE = [offset_lonlat(n, e) for n, e in ((0, 0), (0, 40), (30, 40))]


# ── Waiting on the aircraft ──────────────────────────────────────────────────

def wait_telemetry(api, description: str, predicate: Callable[[dict], bool],
                   timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    telemetry: dict = {}
    while time.monotonic() < deadline:
        telemetry = api.telemetry()["telemetry"]
        if predicate(telemetry):
            return telemetry
        time.sleep(0.2)
    pytest.fail(f"Timed out after {timeout_s:.0f}s waiting for {description}; "
                f"last telemetry: {telemetry}")


def make_api(tmp_path: Path, name: str = "sitl"):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    created = api.create_project(name, str(tmp_path / "project"))
    assert created.get("ok", True), created
    return api


def connect_api(api, uri: str) -> dict:
    connected = api.connect_vehicle(uri, "mavlink")
    assert connected["vehicle"]["connected"], connected
    return wait_telemetry(api, "a 3D fix and a home position",
                          lambda t: t.get("gps_fix", 0) >= 3 and t.get("home_set"), 60.0)


def settle_on_ground(api, timeout_s: float = 240.0) -> None:
    """Leave the shared SITL landed and disarmed for whatever runs next."""
    telemetry = api.telemetry()["telemetry"]
    if telemetry.get("armed"):
        if telemetry.get("altitude_rel_m", 0.0) > 1.0:
            api.vehicle_command("land")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and api.telemetry()["telemetry"].get("armed"):
            time.sleep(0.5)
        if api.telemetry()["telemetry"].get("armed"):
            api.vehicle_command("disarm")


# ── Independent MAVLink trace ────────────────────────────────────────────────

@dataclass
class TraceEntry:
    t: float
    type: str
    data: dict[str, Any]


class MavlinkTrace:
    """Every message on the independent observer stream, timestamped.

    Runs on its own UDP stream, so it sees what ArduPilot sent regardless of what the
    app under test made of it. Written out as JSON lines for the evidence folder.
    """

    KEEP = {
        "HEARTBEAT", "COMMAND_ACK", "STATUSTEXT", "MISSION_ITEM_REACHED", "MISSION_CURRENT",
        "MISSION_ACK", "MISSION_COUNT", "CAMERA_FEEDBACK", "CAMERA_TRIGGER", "MOUNT_STATUS",
        "GIMBAL_DEVICE_ATTITUDE_STATUS", "FENCE_STATUS", "HOME_POSITION", "SYS_STATUS",
        "GLOBAL_POSITION_INT", "VFR_HUD", "ATTITUDE",
    }
    # Kept in memory for assertions but thinned in the written trace.
    HIGH_RATE = {"GLOBAL_POSITION_INT", "VFR_HUD", "ATTITUDE", "SYS_STATUS", "HEARTBEAT"}

    def __init__(self, connection) -> None:
        self._conn = connection
        self._entries: list[TraceEntry] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sitl-trace", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._conn.recv_match(blocking=True, timeout=0.5)
            except Exception:  # noqa: BLE001 - a closed socket ends the trace
                if self._stop.is_set():
                    return
                time.sleep(0.1)
                continue
            if message is None:
                continue
            kind = message.get_type()
            if kind not in self.KEEP:
                continue
            data = message.to_dict()
            data.pop("mavpackettype", None)
            if kind == "STATUSTEXT":
                data["text"] = str(data.get("text", "")).strip("\x00")
            with self._lock:
                self._entries.append(TraceEntry(time.monotonic(), kind, data))

    def mark(self) -> float:
        return time.monotonic()

    def messages(self, kind: str | Iterable[str], since: float = 0.0) -> list[TraceEntry]:
        kinds = {kind} if isinstance(kind, str) else set(kind)
        with self._lock:
            return [e for e in self._entries if e.type in kinds and e.t >= since]

    def wait_for(self, kind: str, predicate: Callable[[dict], bool] = lambda d: True,
                 since: float = 0.0, timeout_s: float = 10.0) -> TraceEntry | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for entry in self.messages(kind, since):
                if predicate(entry.data):
                    return entry
            time.sleep(0.1)
        return None

    def statustext(self, since: float = 0.0) -> list[str]:
        return [e.data.get("text", "") for e in self.messages("STATUSTEXT", since)]

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def write(self, path: Path, *, start: float | None = None) -> Path:
        """JSON lines, with high-rate position/attitude thinned to 1 Hz."""
        with self._lock:
            entries = list(self._entries)
        origin = start if start is not None else (entries[0].t if entries else 0.0)
        last_kept: dict[str, float] = {}
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                if entry.type in self.HIGH_RATE:
                    if entry.t - last_kept.get(entry.type, -10.0) < 1.0:
                        continue
                    last_kept[entry.type] = entry.t
                handle.write(json.dumps({"t": round(entry.t - origin, 3), "type": entry.type,
                                         **entry.data}, default=str) + "\n")
        return path


# ── What ArduPilot will actually do with a mission ──────────────────────────

@dataclass
class VehicleViewpoint:
    """One viewpoint as ArduPilot will fly it, reconstructed from the stored items."""

    seq: int
    command: int
    lat: float
    lon: float
    alt: float
    frame: int
    dwell_s: float
    gimbal: tuple[float, float, float] | None
    yaw_deg: float | None
    yaw_command_lost: bool
    trigger: bool
    speed_m_s: float | None
    trigger_distance_m: float


@dataclass
class VehicleMission:
    takeoff: dict[str, Any] | None
    viewpoints: list[VehicleViewpoint] = field(default_factory=list)
    ends_with: int | None = None
    triggers_before_first_viewpoint: int = 0


def reconstruct(items: list[dict[str, Any]]) -> VehicleMission:
    """Apply ArduPilot's mission semantics (AP_Mission, Copter 4.5) to a stored item list.

    * The do-commands between nav item k-1 and nav item k run when k-1 completes, so a
      DO_MOUNT_CONTROL / CONDITION_YAW placed before a waypoint is in effect on the way
      to it, and a DO_DIGICAM_CONTROL placed after a waypoint fires on arrival at it.
    * The first do-command of each gap starts before the next nav item; the rest start
      after it. Copter's wp_start resets an auto-yaw to the default (look-ahead), so a
      CONDITION_YAW that is the *first* do-command of its gap is undone immediately --
      reported as `yaw_command_lost`.
    * Mount, speed and trigger-distance are persistent state; yaw is per waypoint.
    """
    mission = VehicleMission(takeoff=None)
    gimbal: tuple[float, float, float] | None = None
    speed: float | None = None
    trig_dist = 0.0
    pending_yaw: float | None = None
    yaw_first_in_gap = False
    gap_has_do = False
    for item in items:
        command = int(item["command"])
        if command in NAV_COMMANDS:
            if command == NAV_TAKEOFF:
                mission.takeoff = dict(item)
            elif command in VIEWPOINT_COMMANDS:
                mission.viewpoints.append(VehicleViewpoint(
                    seq=int(item["seq"]), command=command,
                    lat=float(item["lat"]), lon=float(item["lon"]), alt=float(item["alt"]),
                    frame=int(item["frame"]), dwell_s=float(item["param1"]),
                    gimbal=gimbal, yaw_deg=None if yaw_first_in_gap else pending_yaw,
                    yaw_command_lost=yaw_first_in_gap,
                    trigger=False, speed_m_s=speed, trigger_distance_m=trig_dist,
                ))
            else:
                mission.ends_with = command
            pending_yaw, yaw_first_in_gap, gap_has_do = None, False, False
            continue
        if command == CONDITION_YAW:
            pending_yaw = float(item["param1"]) % 360.0
            yaw_first_in_gap = not gap_has_do
        elif command == DO_MOUNT_CONTROL:
            gimbal = (float(item["param1"]), float(item["param2"]), float(item["param3"]))
        elif command == DO_GIMBAL_MANAGER_PITCHYAW:
            gimbal = (float(item["param1"]), 0.0, float(item["param2"]))
        elif command == DO_CHANGE_SPEED:
            speed = float(item["param2"])
        elif command == DO_SET_CAM_TRIGG_DIST:
            trig_dist = float(item["param1"])
        elif command in (DO_DIGICAM_CONTROL, IMAGE_START_CAPTURE):
            fires = (command == IMAGE_START_CAPTURE
                     or int(round(float(item.get("lat", 0.0)))) == 1)
            if fires:
                if mission.viewpoints:
                    mission.viewpoints[-1].trigger = True
                else:
                    mission.triggers_before_first_viewpoint += 1
        gap_has_do = True
    return mission


def _stored(command: int, field_name: str, value: float) -> float:
    """What ArduPilot stores for a field it narrows: whole seconds for hold times."""
    if field_name == "param1" and command in (NAV_WAYPOINT, NAV_LOITER_TIME):
        return float(int(value))  # uint16 seconds in AP_Mission::Mission_Command.p1
    return value


def item_mismatches(sent: list[dict[str, Any]], received: list[dict[str, Any]]) -> list[str]:
    """Field-by-field differences between what the app sent and what ArduPilot stored.

    Fields ArduPilot does not store for a command (NAV_WAYPOINT's param4 yaw on Copter,
    DO_MOUNT_CONTROL's mount-mode z) are not compared; the plan-level check covers the
    behaviour they were meant to carry.
    """
    problems: list[str] = []
    if len(sent) != len(received):
        problems.append(f"count: sent {len(sent)}, vehicle holds {len(received)}")
    for a, b in zip(sent, received):
        cmd = int(a["command"])
        where = f"seq {a['seq']} cmd {cmd} ({a.get('label', '')})"
        if int(b["command"]) != cmd:
            problems.append(f"{where}: vehicle holds command {b['command']}")
            continue
        compare: dict[str, float] = {}
        if cmd in (NAV_WAYPOINT, NAV_LOITER_TIME, NAV_TAKEOFF, NAV_LAND):
            if int(b["frame"]) != int(a["frame"]):
                problems.append(f"{where}: frame {a['frame']} -> {b['frame']}")
            for key, tol in (("lat", LATLON_TOL_DEG), ("lon", LATLON_TOL_DEG), ("alt", ALT_TOL_M)):
                if abs(float(a[key]) - float(b[key])) > tol:
                    problems.append(f"{where}: {key} {a[key]} -> {b[key]}")
            if cmd in (NAV_WAYPOINT, NAV_LOITER_TIME):
                compare["param1"] = float(a["param1"])
        elif cmd == CONDITION_YAW:
            compare = {k: float(a[k]) for k in ("param1", "param2", "param3", "param4")}
        elif cmd == DO_MOUNT_CONTROL:
            compare = {k: float(a[k]) for k in ("param1", "param2", "param3")}
        elif cmd == DO_CHANGE_SPEED:
            compare = {k: float(a[k]) for k in ("param1", "param2", "param3")}
        elif cmd == DO_SET_CAM_TRIGG_DIST:
            compare = {"param1": float(a["param1"]), "param3": float(a["param3"])}
        elif cmd == DO_DIGICAM_CONTROL:
            compare = {k: float(a[k]) for k in ("param1", "param2", "param3", "param4")}
            compare["lat"] = float(a["lat"])
        elif cmd in (DO_GIMBAL_MANAGER_PITCHYAW, IMAGE_START_CAPTURE):
            compare = {k: float(a[k]) for k in ("param1", "param2", "param3", "param4")}
        for key, value in compare.items():
            if abs(float(b[key]) - value) > FLOAT32_TOL:
                stored = _stored(cmd, key, value)
                note = (f" (ArduPilot keeps whole seconds; {value} became {stored})"
                        if stored != value else "")
                problems.append(f"{where}: {key} sent {value} -> vehicle {b[key]}{note}")
    return problems


def plan_mismatches(plan: dict[str, Any], received: list[dict[str, Any]]) -> list[str]:
    """Whether the vehicle will fly the *plan*: every viewpoint's position, altitude,
    gimbal, yaw, dwell and shutter, from ArduPilot's own copy of the mission."""
    from mission import exporters

    planned = exporters._viewpoints(plan)
    vehicle = reconstruct(received)
    problems: list[str] = []
    if len(planned) != len(vehicle.viewpoints):
        problems.append(f"viewpoints: plan has {len(planned)}, vehicle will fly "
                        f"{len(vehicle.viewpoints)}")
    cruise = exporters._cruise_speed(plan, 0.0)
    spacing = exporters._capture_spacing(plan)
    _, home_elev = exporters._altitude_frame(plan)
    for index, (pose, vp) in enumerate(zip(planned, vehicle.viewpoints)):
        where = f"viewpoint {index} (seq {vp.seq})"
        if abs(pose["lat"] - vp.lat) > LATLON_TOL_DEG or abs(pose["lon"] - vp.lon) > LATLON_TOL_DEG:
            problems.append(f"{where}: position {pose['lat']},{pose['lon']} -> {vp.lat},{vp.lon}")
        if abs(pose["alt"] + home_elev - vp.alt) > ALT_TOL_M:
            problems.append(f"{where}: altitude {pose['alt']} -> {vp.alt}")
        if vp.gimbal is None or abs(vp.gimbal[0] - pose["gimbal_pitch_deg"]) > FLOAT32_TOL:
            problems.append(f"{where}: gimbal pitch {pose['gimbal_pitch_deg']} -> "
                            f"{None if vp.gimbal is None else vp.gimbal[0]}")
        if pose["yaw_locked"]:
            if vp.yaw_deg is None or angle_diff_deg(vp.yaw_deg, pose["yaw_deg"] % 360.0) > 0.01:
                lost = " (CONDITION_YAW undone by wp_start)" if vp.yaw_command_lost else ""
                problems.append(f"{where}: yaw {pose['yaw_deg'] % 360.0:.2f} -> {vp.yaw_deg}{lost}")
        elif vp.yaw_deg is not None:
            problems.append(f"{where}: plan leaves yaw free, vehicle forced to {vp.yaw_deg}")
        planned_dwell = float(pose["dwell_s"])
        if planned_dwell > 0.0:
            if vp.command != NAV_LOITER_TIME or vp.dwell_s < planned_dwell - 1e-6:
                problems.append(f"{where}: dwell {planned_dwell}s planned, vehicle holds "
                                f"{vp.dwell_s}s (cmd {vp.command})")
        elif vp.dwell_s > 0.0:
            problems.append(f"{where}: no dwell planned, vehicle holds {vp.dwell_s}s")
        want_trigger = bool(pose["trigger"]) and spacing <= 0.0
        if want_trigger != vp.trigger:
            problems.append(f"{where}: shutter {'expected' if want_trigger else 'not expected'}, "
                            f"vehicle {'fires' if vp.trigger else 'does not fire'}")
        if cruise > 0.0 and (vp.speed_m_s is None or abs(vp.speed_m_s - cruise) > FLOAT32_TOL):
            problems.append(f"{where}: speed {cruise} -> {vp.speed_m_s}")
        if spacing > 0.0 and abs(vp.trigger_distance_m - spacing) > FLOAT32_TOL:
            problems.append(f"{where}: trigger distance {spacing} -> {vp.trigger_distance_m}")
    if vehicle.takeoff is None:
        problems.append("the vehicle's mission has no NAV_TAKEOFF")
    return problems


# ── Planning every template the app offers ───────────────────────────────────

STRUCTURE_TEMPLATES = {
    "facade", "facade_mapping", "multi_facade", "closed_loop", "tower_mapping", "orbit",
    "wind_turbine", "bubble_360", "dome_inspection", "box_inspection", "panorama",
    "roof_inspection",
}
ROUTE_TEMPLATES = {"waypoints", "linear_inspection", "lateral_capture"}

BASE_OPTIONS: dict[str, Any] = {
    "altitude_m": 30.0, "speed_m_s": 6.0, "min_altitude_m": 15.0, "max_altitude_m": 60.0,
    "rth_altitude_m": 40.0,
}
TEMPLATE_OPTIONS: dict[str, dict[str, Any]] = {
    "facade": {"facade_top_altitude_m": 25.0, "facade_bottom_altitude_m": 18.0},
    "facade_mapping": {"facade_top_altitude_m": 25.0, "facade_bottom_altitude_m": 18.0},
    "multi_facade": {"facade_top_altitude_m": 25.0, "facade_bottom_altitude_m": 18.0},
    "tower_mapping": {"tower_top_altitude_m": 30.0, "tower_bottom_altitude_m": 18.0,
                      "tower_object_radius_m": 3.0},
    "wind_turbine": {"turbine_hub_height_m": 40.0, "turbine_blade_length_m": 15.0},
    "dome_inspection": {"dome_radius_m": 8.0},
    "box_inspection": {"box_altitude_levels_m": [18.0, 26.0]},
    "orbit": {"orbit_radius_m": 15.0},
    "thermal": {"thermal_camera": "zenmuse_h20t_thermal", "target_gsd_cm": 5.0},
    "multispectral": {"payload_key": "micasense_rededge_mx", "altitude_m": 40.0,
                      "calibration_panel_lonlat": offset_lonlat(0.0, -5.0)},
    "pylon_inspection": {"pylon_elements": "crossarm@28, insulator@24:8",
                         "standoff_m": 10.0},
}


def all_templates() -> list[str]:
    """The real list: the planner's templates plus the special mission types."""
    from mission.mission_types import SPECIAL_MISSION_TYPES
    from mission.planner import available_templates

    return [*available_templates(), *SPECIAL_MISSION_TYPES]


def plan_template(api, template: str) -> dict[str, Any]:
    """Plan `template` over the Aukerman-like site through the Api, as the UI does."""
    if template == "linked_mission":
        # A linked mission joins saved missions: survey the field, inspect the building.
        ids = []
        for part, polygon in (("grid", FIELD_POLYGON), ("orbit", BUILDING_POLYGON)):
            assert api.set_aoi(polygon)["ok"]
            planned = api.plan_mission({"template": part, **BASE_OPTIONS,
                                        **TEMPLATE_OPTIONS.get(part, {})})
            assert planned["ok"], planned
            saved = api.save_mission(name=f"linked-{part}")
            assert saved["ok"], saved
            ids.append(int(saved["version"]["id"]))
        return api.plan_linked_mission(ids, speed_m_s=6.0)
    polygon = BUILDING_POLYGON if template in STRUCTURE_TEMPLATES or template == "pylon_inspection" else FIELD_POLYGON
    assert api.set_aoi(polygon)["ok"]
    options = {"template": template, **BASE_OPTIONS, **TEMPLATE_OPTIONS.get(template, {})}
    if template in ROUTE_TEMPLATES:
        options["path_lonlat"] = ROUTE_LINE
    return api.plan_mission(options)
