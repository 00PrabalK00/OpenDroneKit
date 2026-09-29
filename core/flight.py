"""Live flight manager — orchestrates active mission execution, telemetry logging, command gateway."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .drone import TELEMETRY_STALE_AFTER_S, CommandResult, DroneClient, DroneTelemetry
from .errors import AppError, ERR_DRONE_NOT_CONNECTED, ERR_PREFLIGHT_FAILED
from .events import (
    DRONE_CONNECTED,
    DRONE_DISCONNECTED,
    FLIGHT_ABORTED,
    FLIGHT_COMMAND,
    FLIGHT_PAUSED,
    FLIGHT_RESUMED,
    FLIGHT_RTH,
    FLIGHT_STARTED,
    TELEMETRY_UPDATED,
    publish_event,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FlightState:
    session_id: str
    # idle | starting | flying | paused | rth | aborted | landed | ended | link_lost
    status: str = "idle"
    current_waypoint: int = 0
    waypoint_total: int = 0
    progress_pct: float = 0.0
    battery_pct: float | None = None
    battery_v: float | None = None
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_rel_m: float = 0.0
    speed_mps: float = 0.0
    flight_mode: str = "UNKNOWN"
    warnings: list[str] = field(default_factory=list)
    last_update: str = field(default_factory=_now_iso)
    # The last telemetry read that failed, and how many have. Swallowing these silently
    # made a dead link look like a quiet aircraft.
    last_error: str = ""
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlightSession:
    id: str
    project_id: str
    mission_id: str
    drone_id: str
    log_dir: str
    state: FlightState
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None
    end_reason: str = ""
    command_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "mission_id": self.mission_id,
            "drone_id": self.drone_id,
            "log_dir": self.log_dir,
            "state": self.state.to_dict(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "command_count": self.command_count,
        }


# ── Logging ───────────────────────────────────────────────────────────────────

def _session_dir(project_root: Path, session_id: str) -> Path:
    p = Path(project_root) / "flight_logs" / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def record_flight_log(session: FlightSession, telemetry: DroneTelemetry) -> None:
    """Append telemetry sample to flight log JSONL."""
    log_path = Path(session.log_dir) / "telemetry.jsonl"
    record = {
        "ts": time.time(),
        "ts_iso": _now_iso(),
        **telemetry.to_dict(),
    }
    _append_jsonl(log_path, record)


# Modes in which the autopilot is flying the uploaded mission.
_MISSION_MODES = {"AUTO", "MISSION"}


def record_command(session: FlightSession, command: str, result: CommandResult, reason: str = "") -> None:
    """Append command + result to command log JSONL."""
    log_path = Path(session.log_dir) / "commands.jsonl"
    record = {
        "ts": time.time(),
        "ts_iso": _now_iso(),
        "command": command,
        "success": bool(result.success),
        "message": result.message,
        "ack": getattr(result, "ack", ""),
        "reason": reason,
    }
    _append_jsonl(log_path, record)
    session.command_count += 1


# ── Flight Manager ────────────────────────────────────────────────────────────

class FlightManager:
    """Singleton-style coordinator for a live flight session."""

    def __init__(self) -> None:
        self._session: FlightSession | None = None
        self._drone: DroneClient | None = None
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()
        self._poll_interval_s = 1.0
        self._lock = threading.RLock()
        self._telemetry_callbacks: list[Callable[[DroneTelemetry], None]] = []

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start_session(
        self,
        project_root: Path | str,
        project_id: str,
        mission_id: str,
        drone_id: str,
        drone: DroneClient,
    ) -> FlightSession:
        if not drone.is_connected():
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Drone not connected.",
                           recovery_action="Connect drone first.")
        with self._lock:
            session_id = str(uuid.uuid4())
            log_dir = _session_dir(Path(project_root), session_id)
            session = FlightSession(
                id=session_id,
                project_id=project_id,
                mission_id=mission_id,
                drone_id=drone_id,
                log_dir=str(log_dir),
                state=FlightState(session_id=session_id, status="idle"),
            )
            self._session = session
            self._drone = drone
            self._start_polling()
            publish_event(DRONE_CONNECTED, {"drone_id": drone_id, "session_id": session_id})
            return session

    def end_session(self, reason: str = "completed") -> FlightSession | None:
        with self._lock:
            session = self._session
            if session is None:
                return None
            self._stop_polling()
            session.ended_at = _now_iso()
            session.end_reason = reason
            # A disconnect is not a landing and not an abort; saying either would put a
            # claim about the aircraft in the record that nothing observed.
            if reason == "completed":
                session.state.status = "landed"
            elif reason.startswith("abort"):
                session.state.status = "aborted"
            else:
                session.state.status = "ended"
            # Persist session summary
            summary_path = Path(session.log_dir) / "session.json"
            summary_path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
            publish_event(DRONE_DISCONNECTED, {"session_id": session.id, "reason": reason})
            self._session = None
            return session

    def active_session(self) -> FlightSession | None:
        return self._session

    # ── Telemetry polling ────────────────────────────────────────────────────

    def set_poll_interval(self, seconds: float) -> None:
        self._poll_interval_s = max(0.1, float(seconds))

    def add_telemetry_listener(self, cb: Callable[[DroneTelemetry], None]) -> None:
        if cb not in self._telemetry_callbacks:
            self._telemetry_callbacks.append(cb)

    def remove_telemetry_listener(self, cb: Callable[[DroneTelemetry], None]) -> None:
        if cb in self._telemetry_callbacks:
            self._telemetry_callbacks.remove(cb)

    def _start_polling(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="flight-telemetry", daemon=True)
        self._poll_thread.start()

    def _stop_polling(self) -> None:
        self._poll_stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        last_recorded: float | None = None
        while not self._poll_stop.is_set():
            drone, session = self._drone, self._session
            if drone is None or session is None:
                break
            try:
                telem = drone.get_telemetry()
            except Exception as exc:  # noqa: BLE001
                # Recorded, not swallowed: a telemetry read that keeps failing is a lost
                # link, and the state has to say so rather than freeze at the last value.
                session.state.last_error = f"Telemetry read failed: {exc}"
                session.state.error_count += 1
                if "Telemetry read failed" not in session.state.warnings:
                    session.state.warnings = [*session.state.warnings, "Telemetry read failed"]
                self._poll_stop.wait(self._poll_interval_s)
                continue
            self._apply_telemetry(telem)
            stamp = getattr(telem, "timestamp", None)
            # The same snapshot read again is not a new observation.
            if stamp is None or stamp != last_recorded:
                last_recorded = stamp
                if not self._is_stale(telem):
                    record_flight_log(session, telem)
                    publish_event(TELEMETRY_UPDATED, telem.to_dict())
                    for cb in list(self._telemetry_callbacks):
                        try:
                            cb(telem)
                        except Exception:  # noqa: BLE001 - one listener cannot stop the feed
                            pass
            self._poll_stop.wait(self._poll_interval_s)

    @staticmethod
    def _is_stale(telem: DroneTelemetry) -> bool:
        age = telem.age_s() if hasattr(telem, "age_s") else 0.0
        return age > TELEMETRY_STALE_AFTER_S

    def _apply_telemetry(self, telem: DroneTelemetry) -> None:
        if not self._session:
            return
        state = self._session.state
        state.current_waypoint = telem.waypoint_index
        state.waypoint_total = telem.waypoint_total
        if telem.waypoint_total > 0:
            state.progress_pct = 100.0 * telem.waypoint_index / telem.waypoint_total
        state.battery_pct = telem.battery_pct
        state.battery_v = telem.battery_v
        state.latitude = telem.latitude
        state.longitude = telem.longitude
        state.altitude_rel_m = telem.altitude_rel_m
        state.speed_mps = telem.speed_mps
        state.flight_mode = telem.flight_mode
        state.last_update = _now_iso()

        # "flying" is what the aircraft reports, not what was requested: a start command
        # the vehicle accepted still has to put it armed and in a mission mode.
        mode = str(telem.flight_mode or "").upper()
        if state.status == "starting" and telem.armed and mode in _MISSION_MODES:
            state.status = "flying"

        # Warning sweep. A value the vehicle never reported raises no warning either way:
        # "Link quality low" for a USB link with no radio status was a false alarm.
        warnings: list[str] = []
        if self._is_stale(telem):
            warnings.append("Telemetry lost")
        if telem.battery_pct is not None and telem.battery_pct < 20.0:
            warnings.append("Battery low")
        if telem.gps_fix < 3:
            warnings.append("GPS fix degraded")
        if telem.link_quality_pct is not None and telem.link_quality_pct < 50.0:
            warnings.append("Link quality low")
        state.warnings = warnings

    # ── Flight commands (gateway) ─────────────────────────────────────────────

    def _require_session(self) -> tuple[FlightSession, DroneClient]:
        if self._session is None or self._drone is None:
            raise AppError(ERR_DRONE_NOT_CONNECTED, "No active flight session.")
        return self._session, self._drone

    def _execute(self, command_name: str, reason: str, action: Callable[[], CommandResult]) -> CommandResult:
        session, _ = self._require_session()
        try:
            result = action()
        except Exception as exc:
            result = CommandResult(False, command_name, str(exc))
        record_command(session, command_name, result, reason=reason)
        publish_event(FLIGHT_COMMAND, {
            "session_id": session.id,
            "command": command_name,
            "success": result.success,
            "message": result.message,
            "reason": reason,
        })
        return result

    def execute(self, command_name: str, action: Callable[[], CommandResult],
                reason: str = "operator") -> CommandResult:
        """Run any vehicle command through the session, so it is logged.

        The named methods below also move the flight state; this is for the commands
        that do not (arm, land, set home, emergency stop).
        """
        return self._execute(command_name, reason, action)

    def start_mission(self, reason: str = "operator_start") -> CommandResult:
        """Ask the vehicle to start the mission.

        The status becomes "starting", and only becomes "flying" once telemetry shows
        the aircraft armed and in a mission mode. It used to read "flying" the moment
        the command was transmitted -- before the vehicle had said anything.
        """
        session, drone = self._require_session()
        previous = session.state.status
        session.state.status = "starting"
        result = self._execute("start_mission", reason, drone.start_mission)
        if result.success:
            publish_event(FLIGHT_STARTED, {"session_id": session.id})
        else:
            session.state.status = previous
        return result

    def pause_flight(self, reason: str = "operator_pause") -> CommandResult:
        session, drone = self._require_session()
        result = self._execute("pause_mission", reason, drone.pause_mission)
        if result.success:
            session.state.status = "paused"
            publish_event(FLIGHT_PAUSED, {"session_id": session.id})
        return result

    def resume_flight(self, reason: str = "operator_resume") -> CommandResult:
        session, drone = self._require_session()
        result = self._execute("resume_mission", reason, drone.resume_mission)
        if result.success:
            session.state.status = "flying"
            publish_event(FLIGHT_RESUMED, {"session_id": session.id})
        return result

    def trigger_rth(self, reason: str = "operator_rth") -> CommandResult:
        session, drone = self._require_session()
        result = self._execute("return_to_home", reason, drone.return_to_home)
        if result.success:
            session.state.status = "rth"
            publish_event(FLIGHT_RTH, {"session_id": session.id, "reason": reason})
        return result

    def abort_flight(self, reason: str = "operator_abort") -> CommandResult:
        session, drone = self._require_session()
        result = self._execute("abort_mission", reason, drone.abort_mission)
        if result.success:
            session.state.status = "aborted"
            publish_event(FLIGHT_ABORTED, {"session_id": session.id, "reason": reason})
            # The session stays open. An aborted aircraft is still in the air, and the
            # telemetry after an abort is the part of the log an investigation wants.
        return result

    def upload_mission(self, mission_items: list[dict[str, Any]], reason: str = "operator_upload") -> CommandResult:
        session, drone = self._require_session()
        return self._execute("upload_mission", reason, lambda: drone.upload_mission(mission_items))

    def status(self) -> dict[str, Any] | None:
        """The session, its state and its recent commands, for a UI to show."""
        session = self._session
        if session is None:
            return None
        return {**session.to_dict(), "commands": self.get_command_log()[-50:]}

    def update_flight_state(self) -> FlightState | None:
        return self._session.state if self._session else None

    def get_command_log(self) -> list[dict[str, Any]]:
        if not self._session:
            return []
        log_path = Path(self._session.log_dir) / "commands.jsonl"
        if not log_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: FlightManager | None = None


def get_flight_manager() -> FlightManager:
    global _manager
    if _manager is None:
        _manager = FlightManager()
    return _manager


# Spec-named module functions
def start_mission_execution(
    project_root: Path | str,
    project_id: str,
    mission_id: str,
    drone_id: str,
    drone: DroneClient,
    preflight_can_start: bool,
) -> FlightSession:
    if not preflight_can_start:
        raise AppError(ERR_PREFLIGHT_FAILED, "Preflight checks not passed.",
                       recovery_action="Fix blocking issues in the Preflight page.")
    mgr = get_flight_manager()
    session = mgr.start_session(project_root, project_id, mission_id, drone_id, drone)
    mgr.start_mission(reason="start_mission_execution")
    return session


def pause_flight(session_id: str) -> CommandResult:
    mgr = get_flight_manager()
    if not mgr.active_session() or mgr.active_session().id != session_id:  # type: ignore[union-attr]
        raise AppError(ERR_DRONE_NOT_CONNECTED, f"No active session: {session_id}")
    return mgr.pause_flight()


def resume_flight(session_id: str) -> CommandResult:
    mgr = get_flight_manager()
    if not mgr.active_session() or mgr.active_session().id != session_id:  # type: ignore[union-attr]
        raise AppError(ERR_DRONE_NOT_CONNECTED, f"No active session: {session_id}")
    return mgr.resume_flight()


def trigger_rth(session_id: str, reason: str = "operator") -> CommandResult:
    mgr = get_flight_manager()
    if not mgr.active_session() or mgr.active_session().id != session_id:  # type: ignore[union-attr]
        raise AppError(ERR_DRONE_NOT_CONNECTED, f"No active session: {session_id}")
    return mgr.trigger_rth(reason=reason)


def abort_flight(session_id: str, reason: str = "operator") -> CommandResult:
    mgr = get_flight_manager()
    if not mgr.active_session() or mgr.active_session().id != session_id:  # type: ignore[union-attr]
        raise AppError(ERR_DRONE_NOT_CONNECTED, f"No active session: {session_id}")
    return mgr.abort_flight(reason=reason)


def update_flight_state(session_id: str, telemetry: DroneTelemetry) -> FlightState | None:
    mgr = get_flight_manager()
    if not mgr.active_session() or mgr.active_session().id != session_id:  # type: ignore[union-attr]
        return None
    mgr._apply_telemetry(telemetry)  # internal but exposed via spec
    return mgr.active_session().state  # type: ignore[union-attr]
