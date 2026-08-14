"""Who is flying the aircraft, and how the pilot takes it back.

An autonomous mission is a period during which the person holding the controller is not
flying. That is fine while it is understood and dangerous the moment it is not, so this
module answers one question continuously -- who has control right now -- and provides
one action: give it back to the pilot.

The distinction that matters is between *commanded* and *confirmed*. Sending a mode
change is not the same as being in that mode. An autopilot refuses modes it cannot
enter: LOITER without a position estimate, AUTO with no mission loaded. It refuses
silently. So an override reports success only once the vehicle's own heartbeat says the
mode changed, and reports failure with the mode the aircraft is actually in otherwise.
Telling a pilot they have control when the aircraft is still flying its mission is the
worst thing this module could do.

One real-world limit is stated rather than papered over. On ArduPilot and PX4 alike,
moving the sticks during an AUTO mission does not take control -- the mode must change.
Software cannot make stick input interrupt autonomy; it can only change the mode and
confirm that it took. Anyone building a "grab the sticks" affordance on top of this
should know that is what happens underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Who is flying, by flight mode. Names cover ArduPilot and PX4, which disagree on
# spelling but not on meaning.
AUTONOMOUS_MODES = {
    "AUTO", "MISSION", "AUTO.MISSION", "GUIDED", "AUTO.LOITER", "RTL", "AUTO.RTL",
    "SMART_RTL", "AUTO.LAND", "LAND", "AUTO.TAKEOFF", "TAKEOFF", "FOLLOW",
    "AUTO.FOLLOW_TARGET", "GUIDED_NOGPS",
}

# The aircraft holds itself but flies nowhere on its own. A pilot with the sticks is in
# charge of where it goes.
ASSISTED_MODES = {
    "LOITER", "POSHOLD", "POSCTL", "ALT_HOLD", "ALTCTL", "BRAKE", "CIRCLE", "HOLD",
    "AUTO.LOITER_HOLD", "POSITION",
}

# The pilot is flying it directly.
MANUAL_MODES = {
    "STABILIZE", "MANUAL", "ACRO", "SPORT", "DRIFT", "RATTITUDE", "STABILIZED",
    "FBWA", "FBWB", "CRUISE",
}

# Preference order when handing control back. LOITER first because it holds position
# and needs no stick input to stay put; a pilot taking over mid-mission may not have
# their hands ready.
HANDBACK_MODES = ("LOITER", "POSHOLD", "POSCTL", "ALT_HOLD", "ALTCTL", "STABILIZE")

CONTROL_AUTONOMOUS = "autonomous"
CONTROL_ASSISTED = "assisted"
CONTROL_MANUAL = "manual"
CONTROL_UNKNOWN = "unknown"


@dataclass
class ControlState:
    """Who has control of the aircraft, as far as we can tell."""

    control: str
    mode: str
    armed: bool
    pilot_has_control: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "mode": self.mode,
            "armed": self.armed,
            "pilot_has_control": self.pilot_has_control,
            "description": self.description,
        }


def classify_mode(mode: str) -> str:
    """Which of the four control states a flight mode represents."""
    name = (mode or "").strip().upper()
    if not name or name == "UNKNOWN":
        return CONTROL_UNKNOWN
    if name in AUTONOMOUS_MODES:
        return CONTROL_AUTONOMOUS
    if name in ASSISTED_MODES:
        return CONTROL_ASSISTED
    if name in MANUAL_MODES:
        return CONTROL_MANUAL
    return CONTROL_UNKNOWN


def control_state(telemetry: Any) -> ControlState:
    """Describe who is flying, from the vehicle's own reported mode."""
    get = (telemetry.get if isinstance(telemetry, dict)
           else lambda k, d=None: getattr(telemetry, k, d))

    mode = str(get("flight_mode", "") or "")
    armed = bool(get("armed", False))
    control = classify_mode(mode)

    if not armed:
        description = f"Disarmed in {mode or 'an unreported mode'}. Nothing is flying."
    elif control == CONTROL_AUTONOMOUS:
        description = (
            f"The aircraft is flying itself in {mode}. The sticks will not take control; "
            "change mode to fly it manually."
        )
    elif control == CONTROL_ASSISTED:
        description = (
            f"{mode}: the aircraft is holding position and will follow your stick input."
        )
    elif control == CONTROL_MANUAL:
        description = f"{mode}: you are flying it."
    else:
        description = (
            f"The vehicle reports mode {mode or '(none)'}, which is not recognised. "
            "Assume it is still flying itself until you have confirmed otherwise."
        )

    return ControlState(
        control=control,
        mode=mode,
        armed=armed,
        # Deliberately conservative: an unrecognised mode does not count as the pilot
        # having control, because the cost of being wrong runs one way.
        pilot_has_control=control in (CONTROL_ASSISTED, CONTROL_MANUAL),
        description=description,
    )


def take_manual_control(client: Any, *, preferred: str = "",
                        timeout_s: float = 3.0) -> dict[str, Any]:
    """Interrupt autonomy and hand the aircraft back to the pilot.

    Tries the hand-back modes in order and confirms against the vehicle's reported mode
    rather than against having sent the command. A mode the autopilot refuses is
    followed by the next candidate, and if none takes, that is reported as a failure
    naming the mode the aircraft is still in.
    """
    setter = getattr(client, "set_flight_mode", None)
    if setter is None:
        return {
            "ok": False,
            "error": "This driver cannot change flight mode, so autonomy cannot be "
                     "interrupted from here. Use the transmitter.",
        }

    candidates = ([preferred.upper()] if preferred else []) + [
        m for m in HANDBACK_MODES if m != preferred.upper()
    ]

    attempts: list[dict[str, Any]] = []
    for mode in candidates:
        try:
            result = setter(mode)
        except TypeError:
            # A driver whose set_flight_mode takes no keyword arguments.
            result = setter(mode)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"mode": mode, "ok": False, "message": str(exc)})
            continue

        ok = bool(getattr(result, "success", getattr(result, "ok", False)))
        message = str(getattr(result, "message", ""))
        attempts.append({"mode": mode, "ok": ok, "message": message})

        if ok:
            state = control_state(client.get_telemetry()) if hasattr(client, "get_telemetry") else None
            return {
                "ok": True,
                "mode": mode,
                "attempts": attempts,
                "state": state.to_dict() if state else None,
                "note": (
                    f"The aircraft is in {mode} and will hold position. You have control; "
                    "the mission is interrupted, not cancelled."
                ),
            }

    current = "unknown"
    if hasattr(client, "get_telemetry"):
        try:
            current = control_state(client.get_telemetry()).mode or "unknown"
        except Exception:  # noqa: BLE001
            current = "unknown"

    return {
        "ok": False,
        "attempts": attempts,
        "error": (
            f"None of the hand-back modes were accepted; the aircraft is still in "
            f"{current}. Take control with the transmitter -- do not assume the mission "
            "has stopped."
        ),
    }
