"""Payload profiles: what is bolted under the aircraft, and what it can be told to do.

A camera profile answers "how big is a pixel on the ground". This answers a different
question: what commands does this payload understand, and what does the mission have to
do differently because it is fitted. A LiDAR needs a slower, lower, overlapping pass and
starts a continuous scan rather than triggering per waypoint. A magnetometer samples on
a timer and is ruined by the airframe's own field unless it hangs below on a tether. A
multispectral head captures a band set and needs a calibration panel shot before and
after the flight. Fly all three with the same "take a photo at each waypoint" plan and
each one produces a file that looks fine and is useless.

Two rules run through this module.

The first is that an unknown payload is reported as unknown, exactly as an unknown
camera is. Resolving it to "generic RGB" would emit a shutter trigger to a LiDAR.

The second is that a command is refused unless the payload declares it. The refusal
matters more than the command list does: a plan containing `start_scan` for a payload
that has no such mode exports cleanly, uploads cleanly, and produces an empty card.

Physical figures -- mass, power draw -- are left as None unless the number is genuinely
known, because an endurance estimate computed from an invented payload mass is worse
than one that admits it does not know the mass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The payload kinds the mission engine knows how to plan differently for. "custom" is
# the honest escape hatch: it carries whatever commands the operator declares, and the
# planner treats it as unknown rather than guessing which of the others it resembles.
KINDS = ("rgb", "thermal", "multispectral", "lidar", "magnetometer", "custom")

# Commands the mission engine can emit. A payload declares which of these it accepts;
# anything else is refused rather than passed through to the aircraft.
COMMANDS = (
    "trigger",            # take one frame at this point
    "start_interval",     # begin distance- or time-triggered capture
    "stop_interval",
    "start_recording",    # continuous video or radiometric sequence
    "stop_recording",
    "start_scan",         # LiDAR or sampling payloads that stream rather than snap
    "stop_scan",
    "calibrate",          # reflectance panel, bias run, boresight
    "set_gimbal",
)


class UnknownPayload(KeyError):
    """The payload is not described, and guessing its commands would fly a wrong plan."""


class PayloadCommandRefused(ValueError):
    """A command was asked for that this payload does not declare."""


@dataclass
class PayloadProfile:
    """One payload, its commands, and what it forces the mission to do differently."""

    key: str
    name: str
    kind: str
    commands: tuple[str, ...]
    # Physical figures are optional on purpose: None means "not known", which the
    # estimator can report, rather than a plausible number nobody measured.
    mass_g: float | None = None
    power_w: float | None = None
    mount: str = ""
    # Multispectral band centres in nanometres, published per instrument.
    bands_nm: tuple[float, ...] = ()
    # Capture at a point, or streaming for the length of the pass. This single flag is
    # what makes a LiDAR plan differ from a photogrammetry one.
    continuous: bool = False
    requires_calibration: bool = False
    source: str = "manufacturer"
    notes: str = ""

    def __post_init__(self) -> None:
        self.commands = tuple(dict.fromkeys(self.commands))
        self.bands_nm = tuple(float(b) for b in self.bands_nm)
        self.validate()

    def validate(self) -> None:
        if not self.key.strip() or not self.name.strip():
            raise ValueError("A payload needs a key and a name.")
        if self.kind not in KINDS:
            raise ValueError(
                f"{self.key}: unknown payload kind {self.kind!r}. "
                f"Use one of {', '.join(KINDS)}, or 'custom' if none of them fit."
            )
        unknown = [c for c in self.commands if c not in COMMANDS]
        if unknown:
            raise ValueError(
                f"{self.key}: undefined command(s) {', '.join(unknown)}. A command the "
                "engine cannot emit would be silently dropped on export."
            )
        if not self.commands:
            raise ValueError(f"{self.key}: a payload that accepts no command cannot be flown.")
        for label, value in (("mass_g", self.mass_g), ("power_w", self.power_w)):
            if value is not None and (value <= 0 or value != value):
                raise ValueError(f"{self.key}: {label} must be positive when it is stated.")
        if self.kind == "multispectral" and not self.bands_nm:
            raise ValueError(
                f"{self.key}: a multispectral payload must state its band centres; "
                "indices computed from unknown bands mean nothing."
            )
        if self.continuous and not ({"start_scan", "start_recording"} & set(self.commands)):
            raise ValueError(
                f"{self.key}: a continuous payload must declare how a run is started."
            )

    def supports(self, command: str) -> bool:
        return command in self.commands

    def require(self, command: str) -> str:
        """Return the command, or refuse it by name rather than dropping it quietly."""
        if command not in COMMANDS:
            raise PayloadCommandRefused(
                f"{command!r} is not a mission-engine command. Known commands: "
                f"{', '.join(COMMANDS)}."
            )
        if not self.supports(command):
            raise PayloadCommandRefused(
                f"{self.name} does not accept {command!r}. It accepts: "
                f"{', '.join(self.commands)}."
            )
        return command

    def capture_command(self) -> str:
        """What this payload is told at a capture point.

        A streaming payload is started once for the pass; a framing one is triggered at
        each point. Emitting the wrong one of these is the failure this database exists
        to prevent.
        """
        return "start_scan" if self.continuous and self.supports("start_scan") else (
            "start_recording" if self.continuous else "trigger")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "kind": self.kind,
            "commands": list(self.commands),
            "mass_g": self.mass_g, "power_w": self.power_w,
            "mass_known": self.mass_g is not None,
            "mount": self.mount, "bands_nm": list(self.bands_nm),
            "band_count": len(self.bands_nm),
            "continuous": self.continuous,
            "requires_calibration": self.requires_calibration,
            "capture_command": self.capture_command(),
            "source": self.source, "notes": self.notes,
        }


FRAMING = ("trigger", "start_interval", "stop_interval", "set_gimbal")
STREAMING = ("start_recording", "stop_recording", "set_gimbal")

# Published payloads in common survey and inspection use. Band centres are the
# manufacturer's stated figures. Mass and power are deliberately absent: they vary with
# mount and cabling, and an endurance penalty computed from a guessed mass would look
# exactly like one computed from a measured mass.
BUILT_IN: dict[str, PayloadProfile] = {
    p.key: p for p in [
        PayloadProfile("rgb_generic", "Generic RGB camera", "rgb", FRAMING,
                       notes="Frames on command; the default photogrammetry payload."),
        PayloadProfile("zenmuse_p1", "DJI Zenmuse P1", "rgb",
                       FRAMING + ("start_recording", "stop_recording"),
                       mount="DJI Skyport",
                       notes="Full-frame survey payload with a mechanical shutter."),
        PayloadProfile("zenmuse_h20t", "DJI Zenmuse H20T", "thermal",
                       FRAMING + ("start_recording", "stop_recording"),
                       mount="DJI Skyport",
                       notes="Radiometric thermal alongside RGB and zoom; thermal frames "
                             "carry per-pixel temperature, so they are captured, not filmed."),
        PayloadProfile("flir_vue_pro_r", "FLIR Vue Pro R", "thermal", FRAMING,
                       notes="Radiometric fixed payload common on custom airframes."),
        PayloadProfile("micasense_rededge_mx", "MicaSense RedEdge-MX", "multispectral",
                       FRAMING + ("calibrate",),
                       bands_nm=(475.0, 560.0, 668.0, 717.0, 840.0),
                       requires_calibration=True,
                       notes="Blue, green, red, red edge and NIR. The reflectance panel "
                             "must be shot before and after the flight or the indices "
                             "are not comparable between surveys."),
        PayloadProfile("mavic3m_multispectral", "DJI Mavic 3 Multispectral", "multispectral",
                       FRAMING + ("calibrate",),
                       bands_nm=(560.0, 650.0, 730.0, 860.0),
                       requires_calibration=True,
                       notes="Green, red, red edge and NIR, with a sunlight sensor for "
                             "irradiance compensation."),
        PayloadProfile("zenmuse_l2", "DJI Zenmuse L2", "lidar",
                       ("start_scan", "stop_scan", "set_gimbal"),
                       mount="DJI Skyport", continuous=True,
                       notes="Streams for the length of the pass. Needs slower, lower, "
                             "overlapping lines than a photogrammetry grid, and a "
                             "figure-of-eight before and after for IMU convergence."),
        PayloadProfile("magnetometer_towed", "Towed magnetometer", "magnetometer",
                       ("start_scan", "stop_scan"), continuous=True,
                       notes="Samples continuously on a tether below the airframe. Flown "
                             "on the aircraft body the readings are dominated by its own "
                             "motors, which looks like signal rather than like an error."),
    ]
}


def _store_path() -> Path:
    """Operator payloads live outside the repository so they survive an upgrade."""
    return Path.home() / ".opendronekit" / "payloads.json"


def load_user_profiles(path: Path | None = None) -> dict[str, PayloadProfile]:
    store = path or _store_path()
    if not store.exists():
        return {}
    try:
        payload = json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    profiles: dict[str, PayloadProfile] = {}
    for key, entry in payload.items():
        try:
            profiles[key] = PayloadProfile(
                key=key, name=entry.get("name", key), kind=str(entry["kind"]),
                commands=tuple(entry.get("commands") or ()),
                mass_g=None if entry.get("mass_g") is None else float(entry["mass_g"]),
                power_w=None if entry.get("power_w") is None else float(entry["power_w"]),
                mount=str(entry.get("mount", "")),
                bands_nm=tuple(entry.get("bands_nm") or ()),
                continuous=bool(entry.get("continuous", False)),
                requires_calibration=bool(entry.get("requires_calibration", False)),
                source="user", notes=str(entry.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError):
            # One malformed entry must not hide the rest of an operator's payloads.
            continue
    return profiles


def save_user_profile(profile: PayloadProfile, path: Path | None = None) -> Path:
    """Add or replace an operator-defined payload."""
    if profile.key in BUILT_IN:
        raise ValueError(
            f"{profile.key!r} is a built-in payload. Choose another key rather than "
            "shadowing published manufacturer specifications."
        )
    profile.validate()

    store = path or _store_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if store.exists():
        try:
            existing = json.loads(store.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            existing = {}

    entry = profile.to_dict()
    entry["source"] = "user"
    existing[profile.key] = entry
    store.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return store


def delete_user_profile(key: str, path: Path | None = None) -> bool:
    store = path or _store_path()
    if not store.exists():
        return False
    try:
        existing = json.loads(store.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return False
    if key not in existing:
        return False
    del existing[key]
    store.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return True


def all_profiles(path: Path | None = None) -> dict[str, PayloadProfile]:
    """Built-in payloads with the operator's own layered on top."""
    return {**BUILT_IN, **load_user_profiles(path)}


def get_payload(key: str, path: Path | None = None) -> PayloadProfile:
    """The payload, or a refusal naming what is available.

    Deliberately no default. A mission planned for an unnamed payload would emit shutter
    triggers regardless of what is actually fitted.
    """
    profiles = all_profiles(path)
    resolved = str(key or "").strip().lower()
    if resolved not in profiles:
        raise UnknownPayload(
            f"Payload {key!r} is not described. Known payloads: "
            f"{', '.join(sorted(profiles))}. Describe it before planning with it, "
            "rather than flying a plan written for a different instrument."
        )
    return profiles[resolved]


def list_payloads(path: Path | None = None) -> list[dict[str, Any]]:
    return [p.to_dict() for p in sorted(all_profiles(path).values(), key=lambda p: p.name)]


def payload_plan_notes(profile: PayloadProfile) -> list[str]:
    """What this payload changes about the mission, in the operator's words.

    Returned as plain statements rather than applied silently, because each one costs
    flight time and the operator is the one who decides to spend it.
    """
    notes: list[str] = []
    if profile.continuous:
        notes.append(
            f"{profile.name} records continuously across each pass rather than framing "
            "at capture points, so the plan starts and stops a run instead of triggering "
            "a shutter."
        )
    if profile.requires_calibration:
        notes.append(
            f"{profile.name} needs a calibration capture before and after the flight. "
            "Without both, values from this survey cannot be compared with another one."
        )
    if profile.kind == "lidar":
        notes.append(
            "LiDAR wants slower, lower and more overlapped lines than a photogrammetry "
            "grid, and an IMU convergence manoeuvre at each end of the flight."
        )
    if profile.kind == "magnetometer":
        notes.append(
            "Magnetometer readings taken on the airframe are dominated by its own "
            "motors. Fly it on a tether below the aircraft, and record the height."
        )
    if profile.mass_g is None:
        notes.append(
            f"The mass of {profile.name} is not recorded, so the endurance penalty for "
            "carrying it is not included in any estimate. Enter it to have it counted."
        )
    return notes
