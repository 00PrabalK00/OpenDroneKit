"""Shared helpers for the PX4 SITL tests.

The app is driven through ``app.api.Api`` exactly as the UIs drive it. What the vehicle
did is read from an independent MAVLink stream -- PX4's other UDP link, which the app is
not using -- so a test cannot pass on the strength of what the app says happened.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

os.environ.setdefault("MAVLINK20", "1")

# Item fields compared between what the app sent and what PX4 stores. Floats travel as
# float32; positions as degrees x 1e7.
FLOAT32_TOL = 1e-4
LATLON_TOL_DEG = 1.5e-7

# PX4 navigation modes as pymavlink names them (mode_string_v10).
PX4_MISSION, PX4_LOITER, PX4_RTL = "MISSION", "LOITER", "RTL"

# STATUSTEXT fragments PX4 uses when its own health checks refuse arming. SIH's
# estimator briefly flags "horizontal/vertical velocity unstable" now and then; the
# refusal is the vehicle's, and is asserted to have left the motors disarmed.
NOT_ARMABLE = ("Arming denied", "refused to arm", "health", "Preflight Fail")


@dataclass
class Entry:
    t: float
    type: str
    data: dict[str, Any]


class Observer:
    """Every message PX4 sends on one UDP link, timestamped, plus a mission downloader."""

    KEEP = {"HEARTBEAT", "COMMAND_ACK", "STATUSTEXT", "MISSION_CURRENT", "MISSION_ITEM_REACHED",
            "MISSION_ACK", "MISSION_COUNT", "HOME_POSITION", "GLOBAL_POSITION_INT", "VFR_HUD",
            "BATTERY_STATUS", "GPS_RAW_INT", "SYS_STATUS", "EXTENDED_SYS_STATE"}
    HIGH_RATE = {"GLOBAL_POSITION_INT", "VFR_HUD", "HEARTBEAT", "BATTERY_STATUS", "GPS_RAW_INT",
                 "SYS_STATUS", "EXTENDED_SYS_STATE", "HOME_POSITION"}
    PROTOCOL = {"MISSION_COUNT", "MISSION_ITEM_INT", "MISSION_ACK"}

    def __init__(self, port: int) -> None:
        from pymavlink import mavutil

        self.port = port
        self.conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}", source_system=253,
                                               source_component=190)
        self._entries: list[Entry] = []
        self._lock = threading.Lock()
        self._protocol: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.origin = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="px4-observer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from pymavlink import mavutil

        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.5)
            except Exception:  # noqa: BLE001 - a closed socket ends the trace
                if self._stop.is_set():
                    return
                time.sleep(0.1)
                continue
            if msg is None:
                continue
            kind = msg.get_type()
            if kind in self.PROTOCOL:
                self._protocol.put(msg)
            if kind not in self.KEEP:
                continue
            data = msg.to_dict()
            data.pop("mavpackettype", None)
            if kind == "HEARTBEAT":
                if int(data.get("autopilot", 0)) != 12:   # not PX4: a GCS or the app
                    continue
                data["mode"] = mavutil.mode_string_v10(msg)
                data["armed"] = bool(int(data.get("base_mode", 0)) & 0x80)
            elif kind == "STATUSTEXT":
                data["text"] = str(data.get("text", "")).strip("\x00").strip()
            with self._lock:
                self._entries.append(Entry(time.monotonic(), kind, data))

    # -- reading ------------------------------------------------------------

    def wait_vehicle(self, timeout_s: float = 30.0) -> None:
        if self.wait_for("HEARTBEAT", timeout_s=timeout_s) is None:
            pytest.fail(f"The independent observer on UDP {self.port} heard no PX4 heartbeat.")

    def mark(self) -> float:
        return time.monotonic()

    def messages(self, kind: str, since: float = 0.0) -> list[Entry]:
        with self._lock:
            return [e for e in self._entries if e.type == kind and e.t >= since]

    def latest(self, kind: str) -> dict[str, Any] | None:
        found = self.messages(kind)
        return found[-1].data if found else None

    def wait_for(self, kind: str, predicate: Callable[[dict], bool] = lambda d: True,
                 since: float = 0.0, timeout_s: float = 10.0) -> Entry | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for entry in self.messages(kind, since):
                if predicate(entry.data):
                    return entry
            time.sleep(0.1)
        return None

    def mode(self) -> str:
        hb = self.latest("HEARTBEAT")
        return str(hb["mode"]) if hb else ""

    def armed(self) -> bool:
        hb = self.latest("HEARTBEAT")
        return bool(hb and hb["armed"])

    def statustext(self, since: float = 0.0) -> list[str]:
        return [e.data["text"] for e in self.messages("STATUSTEXT", since)]

    # -- mission download (MAVLink mission protocol, on this link) -----------

    def download(self, mission_type: int = 0, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        """MISSION_REQUEST_LIST / MISSION_REQUEST_INT, independently of the app."""
        from pymavlink import mavutil

        mav = self.conn.mav
        while not self._protocol.empty():
            self._protocol.get_nowait()
        for _ in range(3):
            mav.mission_request_list_send(1, 1, mission_type)
            count = self._await_protocol("MISSION_COUNT", 3.0)
            if count is not None:
                break
        else:
            pytest.fail(f"PX4 did not answer MISSION_REQUEST_LIST (type {mission_type}).")
        items = []
        for seq in range(int(count.count)):
            for _ in range(5):
                mav.mission_request_int_send(1, 1, seq, mission_type)
                got = self._await_protocol("MISSION_ITEM_INT", 2.0,
                                           lambda m, s=seq: int(m.seq) == s)
                if got is not None:
                    break
            else:
                pytest.fail(f"PX4 stopped answering at item {seq} of {count.count}.")
            items.append({k: getattr(got, k) for k in (
                "seq", "frame", "command", "current", "autocontinue", "param1", "param2",
                "param3", "param4", "x", "y", "z", "mission_type")})
        mav.mission_ack_send(1, 1, mavutil.mavlink.MAV_MISSION_ACCEPTED, mission_type)
        return items

    def _await_protocol(self, kind: str, timeout_s: float, predicate=lambda m: True):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                msg = self._protocol.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                return None
            if msg.get_type() == kind and predicate(msg):
                return msg
        return None

    # -- evidence -------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def write(self, path: Path) -> Path:
        """JSON lines; high-rate messages thinned to 1 Hz, mode changes always kept."""
        with self._lock:
            entries = list(self._entries)
        last: dict[str, float] = {}
        last_mode = None
        with path.open("w", encoding="utf-8") as handle:
            for e in entries:
                if e.type in self.HIGH_RATE:
                    changed = e.type == "HEARTBEAT" and (e.data["mode"], e.data["armed"]) != last_mode
                    if not changed and e.t - last.get(e.type, -10.0) < 1.0:
                        continue
                    last[e.type] = e.t
                    if e.type == "HEARTBEAT":
                        last_mode = (e.data["mode"], e.data["armed"])
                handle.write(json.dumps({"t": round(e.t - self.origin, 3), "type": e.type,
                                         **e.data}, default=str) + "\n")
        return path


# ── The app ────────────────────────────────────────────────────────────────

def make_api(tmp_path: Path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    created = api.create_project("px4", str(tmp_path / "project"))
    assert created.get("ok", True), created
    return api


def telemetry(api) -> dict[str, Any]:
    return api.telemetry()["telemetry"]


def wait_telemetry(api, description: str, predicate: Callable[[dict], bool],
                   timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = telemetry(api)
        if predicate(last):
            return last
        time.sleep(0.2)
    pytest.fail(f"Timed out after {timeout_s:.0f}s waiting for {description}; last: {last}")


def connect(api, driver: tuple[str, str, int]) -> dict:
    name, uri, _ = driver
    connected = api.connect_vehicle(uri, name)
    assert connected["vehicle"]["connected"], connected
    return wait_telemetry(
        api, "position, a 3D fix, a home position and a flight mode",
        lambda t: t.get("gps_fix", 0) >= 3 and t.get("home_set") and t.get("latitude")
        and t.get("flight_mode") not in (None, "", "UNKNOWN") and t.get("battery_pct") is not None,
        60.0)


def offset_lonlat(lat0: float, lon0: float, north_m: float, east_m: float) -> list[float]:
    return [lon0 + east_m / (111_320.0 * math.cos(math.radians(lat0))),
            lat0 + north_m / 110_540.0]


def distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi, d_lambda = math.radians(lat_b - lat_a), math.radians(lon_b - lon_a)
    value = (math.sin(d_phi / 2) ** 2
             + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2)
    return 6_371_000.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


PLAN_OPTIONS = {"altitude_m": 20.0, "speed_m_s": 6.0, "min_altitude_m": 10.0,
                "max_altitude_m": 60.0, "rth_altitude_m": 25.0}


def plan(api, template: str) -> dict[str, Any]:
    """Plan a grid over a field around home, or an orbit of a building in it."""
    t = telemetry(api)
    lat0, lon0 = float(t["home_lat"]), float(t["home_lon"])
    if template == "orbit":
        polygon = [offset_lonlat(lat0, lon0, n, e) for n, e in ((12, 18), (12, 30), (22, 30), (22, 18))]
        options = {"orbit_radius_m": 15.0}
    else:
        polygon = [offset_lonlat(lat0, lon0, n, e) for n, e in ((-10, -10), (-10, 50), (40, 50), (40, -10))]
        options = {}
    assert api.set_aoi(polygon)["ok"]
    planned = api.plan_mission({"template": template, **PLAN_OPTIONS, **options})
    assert planned["ok"], planned
    return planned


def planned_items(api) -> tuple[list[dict], list[dict]]:
    """The mission and fence exactly as Api.upload_mission builds them."""
    from mission import exporters

    plan_dict = api._session.mission_plan_dict
    autopilot = getattr(api._session._drone_client, "autopilot", "ardupilot")
    assert autopilot == "px4", f"the app took this PX4 for {autopilot!r}"
    return ([i.to_dict() for i in exporters.build_mission_items(plan_dict, autopilot=autopilot)],
            [i.to_dict() for i in exporters.build_fence_items(plan_dict)])


# Fence vertices (MAV_CMD_NAV_FENCE_*, 5001-5004). PX4 v1.15.4 keeps a vertex's command,
# vertex count / radius, position and altitude, but not its autocontinue flag, and
# rebuilds its frame on download from an "altitude is relative" flag that fence points
# do not carry -- so a vertex sent in GLOBAL_RELATIVE_ALT comes back as GLOBAL_INT
# (src/modules/mavlink/mavlink_mission.cpp, format_mavlink_mission_item). Neither is
# used by PX4's geofence (a polygon's vertical limit is GF_MAX_VER_DIST), so for these
# commands the two fields are not compared; everything PX4 does keep is.
FENCE_COMMANDS = frozenset({5001, 5002, 5003, 5004})


def fields_compared(item: dict[str, Any]) -> tuple[str, ...]:
    if int(item["command"]) in FENCE_COMMANDS:
        return ("command",)
    return ("command", "frame", "autocontinue")


def item_mismatches(sent: list[dict[str, Any]], stored: list[dict[str, Any]]) -> list[str]:
    """Every field of every item, sent (planner dicts) against stored (raw MAVLink)."""
    from core.mission_planner_bridge import _NON_POSITIONAL_COMMANDS, _planner_frame

    problems = []
    if len(sent) != len(stored):
        problems.append(f"count: sent {len(sent)}, PX4 holds {len(stored)}")
    for a, b in zip(sent, stored):
        where = f"seq {b['seq']} cmd {a['command']} ({a.get('label', '')})"
        for key in fields_compared(a):
            want = int(a.get(key, 1 if key == "autocontinue" else 0))
            got = int(b[key])
            if key == "frame":
                want, got = _planner_frame(want), _planner_frame(got)
            if want != got:
                problems.append(f"{where}: {key} {a.get(key)} -> {b[key]}")
        for key in ("param1", "param2", "param3", "param4"):
            want, got = float(a.get(key, 0.0)), float(b[key])
            if not (math.isnan(want) and math.isnan(got)) and abs(want - got) > FLOAT32_TOL:
                problems.append(f"{where}: {key} {want} -> {got}")
        if int(a["command"]) in _NON_POSITIONAL_COMMANDS:
            pairs = ((int(float(a.get("lat", 0.0))), int(b["x"]), "x"),
                     (int(float(a.get("lon", 0.0))), int(b["y"]), "y"))
            for want, got, key in pairs:
                if want != got:
                    problems.append(f"{where}: {key} {want} -> {got}")
        else:
            for key, raw in (("lat", "x"), ("lon", "y")):
                if abs(float(a.get(key, 0.0)) - int(b[raw]) / 1e7) > LATLON_TOL_DEG:
                    problems.append(f"{where}: {key} {a.get(key)} -> {int(b[raw]) / 1e7}")
        if abs(float(a.get("alt", 0.0)) - float(b["z"])) > FLOAT32_TOL:
            problems.append(f"{where}: alt {a.get('alt')} -> {b['z']}")
    return problems


def when_armable(attempt: Callable[[], dict], observer: Observer, tries: int = 8,
                 wait_s: float = 5.0) -> tuple[dict, list[str]]:
    """Run a command that arms, again while PX4's own health checks refuse arming.

    Each refusal is checked to have left the vehicle disarmed (on the independent
    stream) and is returned, so the evidence shows every refusal PX4 gave.
    """
    refusals: list[str] = []
    result: dict = {}
    for _ in range(tries):
        mark = observer.mark()
        result = attempt()
        if result.get("ok"):
            return result, refusals
        text = str(result.get("error", ""))
        said = observer.statustext(mark)
        if not any(s in text or any(s in line for line in said) for s in NOT_ARMABLE):
            return result, refusals
        time.sleep(1.0)
        assert not observer.armed(), f"PX4 refused ({text}) but is armed"
        refusals.append(f"{text} | PX4: {'; '.join(said)}")
        time.sleep(wait_s)
    return result, refusals


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
