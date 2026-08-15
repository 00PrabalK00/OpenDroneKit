"""RTK and PPK inputs: camera event marks, base station observations, and time.

Post-processed kinematic positioning is what turns a survey from "about a metre" into
"a few centimetres" without laying control. It works by pairing every camera event with
the base station's observations at that instant, so the whole method rests on two files
agreeing about when things happened. When they do not, nothing fails: the solution
quietly falls back to a float or a single-point fix, the images still carry coordinates,
and the deliverable is out by a metre in a way no one can see by looking at it.

So this module does three things and refuses to pretend to do a fourth.

It reads the camera event marks a DJI RTK aircraft writes beside the images -- the
``.MRK`` file, one line per shutter event, with GPS time, the correction applied, the
standard deviations and the solution flag.

It reads the header of a RINEX observation file to find where the base stood and which
window it observed.

It aligns the two in time, and reports exactly which events are covered, which are not,
and where the gaps are.

What it does not do is compute the PPK solution. Double-differencing carrier phase is
the job of a GNSS processor, and a module that quietly returned uncorrected coordinates
labelled "PPK" would produce the precise failure this whole file exists to prevent. The
accuracy claim is gated on the solution flag the aircraft actually recorded: a float
solution is reported as a float, never as centimetre-level RTK.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

# GPS time began at midnight UTC on 6 January 1980 and does not observe leap seconds, so
# converting an event time to UTC needs the current offset. It has been 18 s since the
# leap second of 31 December 2016; it is a parameter rather than a constant because the
# day it changes, every survey processed with a hard-coded 18 is out by a second, which
# at 15 m/s is 15 metres of position.
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
DEFAULT_LEAP_SECONDS = 18

# DJI writes corrections in millimetres and standard deviations in metres.
MM_TO_M = 0.001

# Solution flags, as written in the MRK line. Only "fixed" supports a centimetre claim.
SOLUTION_FLAGS = {0: "single", 1: "float", 16: "fixed", 34: "fixed", 50: "fixed"}


class RtkError(ValueError):
    """The RTK/PPK inputs cannot support the accuracy that would be claimed from them."""


@dataclass
class CameraEvent:
    """One shutter event as the aircraft recorded it."""

    sequence: int
    gps_week: int
    gps_seconds: float
    latitude_deg: float
    longitude_deg: float
    height_m: float
    north_correction_m: float
    east_correction_m: float
    vertical_correction_m: float
    std_north_m: float
    std_east_m: float
    std_vertical_m: float
    solution: str

    @property
    def utc(self) -> datetime:
        return gps_to_utc(self.gps_week, self.gps_seconds)

    @property
    def horizontal_std_m(self) -> float:
        return math.hypot(self.std_north_m, self.std_east_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "utc": self.utc.isoformat(),
            "gps_week": self.gps_week, "gps_seconds": round(self.gps_seconds, 6),
            "latitude_deg": self.latitude_deg, "longitude_deg": self.longitude_deg,
            "height_m": self.height_m,
            "correction_m": {
                "north": round(self.north_correction_m, 4),
                "east": round(self.east_correction_m, 4),
                "vertical": round(self.vertical_correction_m, 4),
            },
            "std_m": {
                "north": self.std_north_m, "east": self.std_east_m,
                "vertical": self.std_vertical_m,
                "horizontal": round(self.horizontal_std_m, 4),
            },
            "solution": self.solution,
        }


@dataclass
class BaseStation:
    """What a RINEX observation file says about the base and its session."""

    path: str
    marker_name: str
    approx_xyz_m: tuple[float, float, float]
    first_obs: datetime
    last_obs: datetime | None
    interval_s: float | None
    receiver: str = ""
    antenna: str = ""

    @property
    def duration_s(self) -> float | None:
        if self.last_obs is None:
            return None
        return (self.last_obs - self.first_obs).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "marker_name": self.marker_name,
            "approx_xyz_m": list(self.approx_xyz_m),
            "first_obs": self.first_obs.isoformat(),
            "last_obs": None if self.last_obs is None else self.last_obs.isoformat(),
            "interval_s": self.interval_s,
            "duration_s": self.duration_s,
            "receiver": self.receiver, "antenna": self.antenna,
        }


def gps_to_utc(week: int, seconds_of_week: float,
               leap_seconds: int = DEFAULT_LEAP_SECONDS) -> datetime:
    """Convert GPS week and second-of-week to UTC.

    GPS time ignores leap seconds, so the offset must be subtracted explicitly. Getting
    it wrong shifts every camera event by a whole second, which at survey speeds is
    metres of along-track error -- and the resulting positions still look ordinary.
    """
    if week < 0 or not math.isfinite(seconds_of_week):
        raise RtkError("A GPS timestamp needs a non-negative week and a finite second.")
    return GPS_EPOCH + timedelta(weeks=week, seconds=seconds_of_week - leap_seconds)


def read_camera_events(path: str | Path) -> list[CameraEvent]:
    """Read a DJI ``.MRK`` camera event file.

    Each line is one shutter event: sequence, second-of-week, week, the north/east/
    vertical corrections in millimetres, the recorded position, its standard deviations
    and the solution flag.
    """
    source = Path(path)
    if not source.exists():
        raise RtkError(f"Camera event file not found: {source}")

    events: list[CameraEvent] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8",
                                                       errors="ignore").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise RtkError(f"{source.name} line {line_number}: not a camera event record.")
        tagged: dict[str, float] = {}
        for token in fields[3:]:
            if "," in token:
                value, _, tag = token.rpartition(",")
                try:
                    tagged[tag.strip()] = float(value)
                except ValueError:
                    continue
        try:
            sequence = int(fields[0])
            seconds = float(fields[1])
            week = int(fields[2])
        except ValueError as exc:
            raise RtkError(
                f"{source.name} line {line_number}: sequence, second-of-week and GPS "
                "week must be numeric."
            ) from exc

        flag = int(tagged.get("Q", tagged.get("Flag", 0)))
        events.append(CameraEvent(
            sequence=sequence, gps_week=week, gps_seconds=seconds,
            latitude_deg=float(tagged.get("Lat", 0.0)),
            longitude_deg=float(tagged.get("Lon", 0.0)),
            height_m=float(tagged.get("Ellh", 0.0)),
            north_correction_m=float(tagged.get("N", 0.0)) * MM_TO_M,
            east_correction_m=float(tagged.get("E", 0.0)) * MM_TO_M,
            vertical_correction_m=float(tagged.get("V", 0.0)) * MM_TO_M,
            std_north_m=float(tagged.get("Ns", 0.0)),
            std_east_m=float(tagged.get("Es", 0.0)),
            std_vertical_m=float(tagged.get("Vs", 0.0)),
            solution=SOLUTION_FLAGS.get(flag, f"unknown({flag})"),
        ))

    if not events:
        raise RtkError(
            f"{source.name} contains no camera events. Without shutter timestamps there "
            "is nothing to align the base station observations against."
        )
    return events


def _rinex_epoch(fields: Sequence[str]) -> datetime:
    year, month, day, hour, minute = (int(float(v)) for v in fields[:5])
    second = float(fields[5])
    whole = int(second)
    micro = int(round((second - whole) * 1_000_000))
    return datetime(year, month, day, hour, minute, whole, micro, tzinfo=timezone.utc)


def read_base_station(path: str | Path) -> BaseStation:
    """Read the header of a RINEX observation file.

    Only the header is parsed. The observations themselves are the GNSS processor's
    business; what is needed here is where the base stood and when it was recording, so
    that coverage can be checked before a survey is trusted.
    """
    source = Path(path)
    if not source.exists():
        raise RtkError(f"RINEX observation file not found: {source}")

    marker = ""
    receiver = ""
    antenna = ""
    approx: tuple[float, float, float] | None = None
    first: datetime | None = None
    last: datetime | None = None
    interval: float | None = None
    saw_header_end = False

    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            label = raw[60:].strip()
            body = raw[:60]
            if label == "MARKER NAME":
                marker = body.strip()
            elif label == "REC # / TYPE / VERS":
                receiver = " ".join(body.split()[1:3]) if len(body.split()) >= 3 else body.strip()
            elif label == "ANT # / TYPE":
                antenna = " ".join(body.split()[1:]) if body.split() else ""
            elif label == "APPROX POSITION XYZ":
                values = [float(v) for v in body.split()[:3]]
                if len(values) == 3:
                    approx = (values[0], values[1], values[2])
            elif label == "TIME OF FIRST OBS":
                first = _rinex_epoch(body.split())
            elif label == "TIME OF LAST OBS":
                last = _rinex_epoch(body.split())
            elif label == "INTERVAL":
                interval = float(body.split()[0])
            elif label == "END OF HEADER":
                saw_header_end = True
                break

    if not saw_header_end:
        raise RtkError(
            f"{source.name} has no END OF HEADER line, so it is not a readable RINEX "
            "observation file."
        )
    if first is None:
        raise RtkError(
            f"{source.name} declares no TIME OF FIRST OBS, so the observation window is "
            "unknown and no camera event can be shown to be covered by it."
        )
    if approx is None:
        raise RtkError(
            f"{source.name} declares no APPROX POSITION XYZ, so where the base stood is "
            "unknown. Every corrected position would be offset by the base's own error, "
            "identically and invisibly, across the whole survey."
        )
    return BaseStation(path=str(source), marker_name=marker, approx_xyz_m=approx,
                       first_obs=first, last_obs=last, interval_s=interval,
                       receiver=receiver, antenna=antenna)


def align_events_to_base(events: Sequence[CameraEvent], base: BaseStation, *,
                         leap_seconds: int = DEFAULT_LEAP_SECONDS,
                         margin_s: float = 0.0) -> dict[str, Any]:
    """Check every camera event falls inside the base station's observation window.

    An event outside the window cannot be corrected. Reporting that plainly is the
    point: a processor handed partial coverage will still produce a solution for the
    covered frames and a degraded one for the rest, and the mixture is invisible in the
    final product.
    """
    if not events:
        raise RtkError("There are no camera events to align.")
    if base.last_obs is None:
        raise RtkError(
            f"{Path(base.path).name} declares no TIME OF LAST OBS, so the end of the "
            "observation session is unknown and coverage cannot be demonstrated. Treat "
            "the session as unverified rather than assuming it ran long enough."
        )

    start = base.first_obs - timedelta(seconds=margin_s)
    end = base.last_obs + timedelta(seconds=margin_s)
    times = [gps_to_utc(e.gps_week, e.gps_seconds, leap_seconds) for e in events]
    covered = [e for e, t in zip(events, times) if start <= t <= end]
    before = [e for e, t in zip(events, times) if t < start]
    after = [e for e, t in zip(events, times) if t > end]

    first_event, last_event = min(times), max(times)
    solutions: dict[str, int] = {}
    for event in events:
        solutions[event.solution] = solutions.get(event.solution, 0) + 1

    return {
        "event_count": len(events),
        "covered_count": len(covered),
        "coverage_fraction": round(len(covered) / len(events), 6),
        "uncovered_before": [e.sequence for e in before],
        "uncovered_after": [e.sequence for e in after],
        "flight_window": {"first": first_event.isoformat(), "last": last_event.isoformat()},
        "base_window": {"first": base.first_obs.isoformat(),
                        "last": base.last_obs.isoformat()},
        "lead_in_s": round((first_event - base.first_obs).total_seconds(), 3),
        "lead_out_s": round((base.last_obs - last_event).total_seconds(), 3),
        "leap_seconds": leap_seconds,
        "solution_counts": solutions,
    }


def positioning_report(events_path: str | Path, rinex_path: str | Path, *,
                       leap_seconds: int = DEFAULT_LEAP_SECONDS,
                       margin_s: float = 0.0,
                       required_coverage: float = 1.0) -> dict[str, Any]:
    """Whether these inputs can support a PPK solution, and what they cannot support.

    Returns a description, never a corrected coordinate. The accuracy statement is
    derived from the solution flags the aircraft recorded, so a survey flown on a float
    solution is described as a float solution however much the client wanted RTK.
    """
    events = read_camera_events(events_path)
    base = read_base_station(rinex_path)
    alignment = align_events_to_base(events, base, leap_seconds=leap_seconds,
                                     margin_s=margin_s)

    fixed = alignment["solution_counts"].get("fixed", 0)
    fixed_fraction = fixed / len(events)
    horizontal = sorted(e.horizontal_std_m for e in events)
    median_horizontal = horizontal[len(horizontal) // 2] if horizontal else float("nan")

    blocking: list[str] = []
    if alignment["coverage_fraction"] < required_coverage:
        missing = len(events) - alignment["covered_count"]
        blocking.append(
            f"{missing} of {len(events)} camera events fall outside the base station's "
            "observation window, so those images cannot be corrected. Processing anyway "
            "produces a survey that is centimetre-accurate in places and metre-accurate "
            "in others, with nothing in the output marking which is which."
        )
    if base.duration_s is not None and base.duration_s <= 0:
        blocking.append("The base station session has no duration.")

    caveats: list[str] = [
        "This is a check of the inputs, not a PPK solution. Corrected positions come "
        "from a GNSS processor; nothing here rewrites a coordinate.",
        f"GPS-to-UTC conversion assumed {leap_seconds} leap seconds. If that is wrong "
        "for the survey date, every event time is out by the difference.",
    ]
    if fixed_fraction < 1.0:
        caveats.append(
            f"{len(events) - fixed} of {len(events)} events were not a fixed solution "
            "when recorded, so no centimetre-level claim can be made for them. They are "
            f"reported as: {', '.join(f'{k} x{v}' for k, v in sorted(alignment['solution_counts'].items()))}."
        )
    if base.marker_name == "":
        caveats.append(
            "The RINEX file names no marker, so which physical station these "
            "observations came from rests on the filename alone."
        )

    return {
        "ok": not blocking,
        "usable_for_ppk": not blocking,
        "blocking": blocking,
        "events": len(events),
        "fixed_fraction": round(fixed_fraction, 6),
        "median_horizontal_std_m": (None if not horizontal else round(median_horizontal, 4)),
        "alignment": alignment,
        "base_station": base.to_dict(),
        "accuracy_statement": _accuracy_statement(fixed_fraction, median_horizontal),
        "caveats": caveats,
    }


def _accuracy_statement(fixed_fraction: float, median_horizontal_std_m: float) -> str:
    """What may honestly be said about this survey's positioning."""
    if fixed_fraction >= 1.0:
        return (
            "Every camera event was recorded with a fixed solution. Subject to a "
            f"successful PPK run, horizontal precision at the camera is around "
            f"{median_horizontal_std_m * 100:.1f} cm as reported by the aircraft; this "
            "is precision at the antenna, not the accuracy of the final deliverable, "
            "which also depends on the base position and the lever arm."
        )
    if fixed_fraction <= 0.0:
        return (
            "No camera event was recorded with a fixed solution, so this flight does "
            "not support an RTK or PPK accuracy claim at all. Its positions are "
            "autonomous or float quality -- metre level -- and should be described that "
            "way to the client, or the survey should be controlled with GCPs."
        )
    return (
        f"{fixed_fraction:.0%} of camera events held a fixed solution. A single accuracy "
        "figure for the survey would be wrong for the rest, so quote the fixed and "
        "non-fixed portions separately, or control the flight with GCPs."
    )
