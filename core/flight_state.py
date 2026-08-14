"""Surviving a crash of the ground station without lying about the aircraft.

If the laptop dies mid-mission, the aircraft does not. It carries on flying the mission
it was given, and the person who restarts the software has no idea what state anything
is in. That gap is where the dangerous behaviour lives: an application that reopens,
sees an incomplete mission and quietly re-uploads or restarts it can send an airborne
aircraft back to the start of its route.

So this records what was happening, and on restart says what it knows and what it does
not. It never resumes anything on its own. Every recovery path ends in a question for
the operator, because the software genuinely cannot tell from its own files whether the
aircraft is flying, landed, or in a tree.

State is written after each transition rather than periodically. A crash lands between
writes either way, but transition-triggered writes mean the last record is a thing that
actually happened rather than a sample of a moment nobody chose.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_FILENAME = "flight_state.json"

# Phases a mission passes through. Ordered, but not assumed: a crash can happen between
# any two of them, and the file may record any one of them as the last thing seen.
PHASE_IDLE = "idle"
PHASE_PLANNED = "planned"
PHASE_UPLOADED = "uploaded"
PHASE_ARMED = "armed"
PHASE_FLYING = "flying"
PHASE_PAUSED = "paused"
PHASE_COMPLETED = "completed"
PHASE_ABORTED = "aborted"

# Phases where the aircraft may still be in the air when the software comes back.
AIRBORNE_PHASES = {PHASE_ARMED, PHASE_FLYING, PHASE_PAUSED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FlightState:
    """What the ground station last saw."""

    phase: str = PHASE_IDLE
    mission_name: str = ""
    mission_id: str = ""
    waypoint_index: int = 0
    waypoint_total: int = 0
    completed_captures: list[int] = field(default_factory=list)
    last_latitude: float | None = None
    last_longitude: float | None = None
    last_altitude_rel_m: float | None = None
    last_battery_pct: float | None = None
    vehicle_driver: str = ""
    clean_shutdown: bool = False
    updated_at: str = field(default_factory=_now)

    @property
    def possibly_airborne(self) -> bool:
        """Whether the aircraft may still have been flying when contact was lost."""
        return self.phase in AIRBORNE_PHASES and not self.clean_shutdown

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def state_path(project_root: str | Path) -> Path:
    return Path(project_root) / STATE_FILENAME


def save_state(project_root: str | Path, state: FlightState) -> Path:
    """Write the state atomically, so a crash mid-write cannot corrupt it.

    A half-written state file is worse than none: it would either fail to parse, losing
    the record entirely, or parse into something that never happened.
    """
    target = state_path(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _now()

    handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state.to_dict(), stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def load_state(project_root: str | Path) -> FlightState | None:
    """Read the last recorded state, or None when there is none to read."""
    path = state_path(project_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt file means we do not know what happened, which is different from
        # nothing having happened. The caller is told via recover().
        return None

    known = set(FlightState.__dataclass_fields__)
    return FlightState(**{k: v for k, v in payload.items() if k in known})


def record_transition(project_root: str | Path, phase: str, *,
                      telemetry: Any = None, **fields: Any) -> FlightState:
    """Record that the mission has entered a new phase."""
    state = load_state(project_root) or FlightState()
    state.phase = phase
    state.clean_shutdown = phase in (PHASE_IDLE, PHASE_COMPLETED, PHASE_ABORTED)

    for key, value in fields.items():
        if key in FlightState.__dataclass_fields__:
            setattr(state, key, value)

    if telemetry is not None:
        get = (telemetry.get if isinstance(telemetry, dict)
               else lambda k, d=None: getattr(telemetry, k, d))
        state.last_latitude = get("latitude", state.last_latitude)
        state.last_longitude = get("longitude", state.last_longitude)
        state.last_altitude_rel_m = get("altitude_rel_m", state.last_altitude_rel_m)
        state.last_battery_pct = get("battery_pct", state.last_battery_pct)
        state.waypoint_index = int(get("waypoint_index", state.waypoint_index) or 0)
        state.waypoint_total = int(get("waypoint_total", state.waypoint_total) or 0)

    save_state(project_root, state)
    return state


def clear_state(project_root: str | Path) -> None:
    """Forget the recorded flight, once the operator has resolved it."""
    state_path(project_root).unlink(missing_ok=True)


def recover(project_root: str | Path) -> dict[str, Any]:
    """What to tell the operator when the software restarts.

    This deliberately offers information and choices, never an action. The software
    cannot tell from a file on disk whether the aircraft is airborne, and an application
    that guessed and re-uploaded a mission could send a flying aircraft back to the
    beginning of its route.
    """
    path = state_path(project_root)
    if path.exists():
        state = load_state(project_root)
        if state is None:
            return {
                "recovered": False,
                "corrupt": True,
                "requires_operator": True,
                "summary": (
                    "A flight state file exists but could not be read, so what was "
                    "happening is unknown. Establish the aircraft's status directly "
                    "before doing anything else."
                ),
                "options": ["discard_state"],
            }
    else:
        return {
            "recovered": False,
            "requires_operator": False,
            "summary": "No interrupted flight was recorded.",
            "options": [],
        }

    if state.clean_shutdown:
        return {
            "recovered": True,
            "state": state.to_dict(),
            "requires_operator": False,
            "possibly_airborne": False,
            "summary": (
                f"The last mission ended in {state.phase}. Nothing was interrupted."
            ),
            "options": ["discard_state"],
        }

    location = ""
    if state.last_latitude is not None and state.last_longitude is not None:
        location = (f" Last seen at {state.last_latitude:.6f}, "
                    f"{state.last_longitude:.6f}")
        if state.last_altitude_rel_m is not None:
            location += f" at {state.last_altitude_rel_m:.0f} m"
        location += "."

    progress = ""
    if state.waypoint_total:
        progress = (f" It was at waypoint {state.waypoint_index} of "
                    f"{state.waypoint_total}.")

    battery = ""
    if state.last_battery_pct is not None:
        battery = f" Battery was {state.last_battery_pct:.0f}% when contact was lost."

    return {
        "recovered": True,
        "state": state.to_dict(),
        "requires_operator": True,
        "possibly_airborne": state.possibly_airborne,
        "summary": (
            f"The ground station stopped while the mission {state.mission_name or ''} "
            f"was {state.phase}.{progress}{location}{battery} "
            "The aircraft may still be airborne: this software cannot tell from its own "
            "records. Establish its status before reconnecting, and do not assume the "
            "mission stopped when the software did."
        ).replace("  ", " "),
        "options": ["reconnect_and_observe", "resume_remaining", "discard_state"],
        "note": (
            "Nothing has been resumed. Re-uploading a mission to an aircraft that is "
            "still flying one would send it back to the start of its route, so that "
            "decision is left to you."
        ),
    }
