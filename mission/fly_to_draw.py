"""Defining a boundary by flying to its corners.

Tracing a survey boundary on a map works when the boundary is visible from above and
someone knows where it is. A field edge under tree canopy, a stockpile whose toe moved
since the imagery was taken, a compound whose fence line is not the line on the
cadastral map -- none of those can be drawn accurately from a tablet, and the operator is
standing next to them.

So the aircraft marks them. Fly to a corner, mark it, fly to the next. The boundary comes
out of the flight rather than out of a basemap that may be years old.

What makes this honest rather than convenient is what it refuses to accept. A mark taken
without a position fix has no coordinates, and recording it as "wherever the aircraft
thought it was" would put a corner in the wrong place with nothing to show for it. Two
marks in the same spot describe an edge of zero length. A ring whose edges cross itself
has no inside, so its area is meaningless even though the arithmetic returns a number.
Each of those is refused with the reason, because the operator can walk the aircraft back
and re-mark while they are still on site -- which is the one window in which any of this
is fixable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

# Two marks closer than this describe the same corner. Held low because a genuinely small
# site is legitimate; this only catches a double press.
MIN_MARK_SEPARATION_M = 1.0

# GPS fix types below this are a position the receiver itself does not trust.
MIN_FIX_TYPE = 3


class BoundaryRefused(ValueError):
    """The marked positions cannot describe a boundary that means anything."""


@dataclass
class BoundaryMark:
    """One corner, as the aircraft recorded it."""

    longitude: float
    latitude: float
    altitude_m: float = 0.0
    fix_type: int = 0
    satellites: int = 0
    hdop: float = 0.0
    marked_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "longitude": round(self.longitude, 8), "latitude": round(self.latitude, 8),
            "altitude_m": round(self.altitude_m, 3),
            "fix_type": self.fix_type, "satellites": self.satellites,
            "hdop": self.hdop, "marked_utc": self.marked_utc, "note": self.note,
        }


def mark_from_telemetry(telemetry: dict[str, Any], note: str = "") -> BoundaryMark:
    """Take a corner from the live telemetry, or refuse and say why.

    The refusal is the point. A corner recorded from an unfixed receiver looks identical
    in the output to one recorded from a good fix.
    """
    if not telemetry.get("connected"):
        raise BoundaryRefused(
            "No vehicle is connected, so there is no position to mark. "
            + str(telemetry.get("reason", "")).strip()
        )

    longitude = telemetry.get("longitude", telemetry.get("lon"))
    latitude = telemetry.get("latitude", telemetry.get("lat"))
    if longitude is None or latitude is None:
        raise BoundaryRefused(
            "The telemetry carries no position, so this mark would have no coordinates."
        )
    longitude, latitude = float(longitude), float(latitude)
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        raise BoundaryRefused("The reported position is not a finite coordinate.")
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise BoundaryRefused(
            f"The reported position ({longitude}, {latitude}) is not on Earth."
        )

    fix_type = int(telemetry.get("fix_type", telemetry.get("gps_fix_type", 0)) or 0)
    if fix_type < MIN_FIX_TYPE:
        raise BoundaryRefused(
            f"The receiver reports fix type {fix_type}, which is not a 3D fix. A corner "
            "marked without one is a position the receiver itself does not trust, and it "
            "would look exactly like a good corner in the finished boundary. Wait for a "
            "fix and mark again."
        )

    return BoundaryMark(
        longitude=longitude, latitude=latitude,
        altitude_m=float(telemetry.get("altitude_m", telemetry.get("alt_m", 0.0)) or 0.0),
        fix_type=fix_type,
        satellites=int(telemetry.get("satellites", telemetry.get("satellites_visible", 0)) or 0),
        hdop=float(telemetry.get("hdop", telemetry.get("eph", 0.0)) or 0.0),
        note=note,
    )


def _metres_between(a: BoundaryMark, b: BoundaryMark) -> float:
    """Great-circle distance, good enough at the scale of a survey boundary."""
    radius = 6_371_000.0
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orientation(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    d3, d4 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _self_intersects(ring: Sequence[Sequence[float]]) -> bool:
    count = len(ring)
    for i in range(count):
        a, b = ring[i], ring[(i + 1) % count]
        for j in range(i + 1, count):
            # Skip the neighbouring and wrapping edges, which share a vertex by design.
            if j == i or (j + 1) % count == i or j == (i + 1) % count:
                continue
            if _segments_cross(a, b, ring[j], ring[(j + 1) % count]):
                return True
    return False


def boundary_from_marks(marks: Sequence[BoundaryMark]) -> dict[str, Any]:
    """Turn marked corners into an area of interest, or refuse with the reason."""
    if len(marks) < 3:
        raise BoundaryRefused(
            f"{len(marks)} corner(s) marked. An area needs at least three, since two "
            "points describe a line and a line has no inside to survey."
        )

    for index in range(1, len(marks)):
        gap = _metres_between(marks[index - 1], marks[index])
        if gap < MIN_MARK_SEPARATION_M:
            raise BoundaryRefused(
                f"Corners {index} and {index + 1} are {gap:.2f} m apart, which is closer "
                "than one aircraft length. That is a double press rather than an edge; "
                "remove one and mark the real corner."
            )

    ring = [[mark.longitude, mark.latitude] for mark in marks]
    if _self_intersects(ring):
        raise BoundaryRefused(
            "The marked corners cross over themselves, so the outline has no inside. Its "
            "area would still compute to a number, which is why this is refused rather "
            "than reported. Re-mark the corners in the order they are flown around the "
            "site."
        )

    from core import geo

    area_m2 = float(geo.polygon_area_m2(ring))
    perimeter_m = sum(_metres_between(marks[i - 1], marks[i]) for i in range(1, len(marks)))
    perimeter_m += _metres_between(marks[-1], marks[0])

    worst_fix = min(mark.fix_type for mark in marks)
    worst_sats = min(mark.satellites for mark in marks)
    limits = [
        "The boundary is where the aircraft was when each corner was marked, which "
        "includes the receiver's own error at that moment. It is not a surveyed "
        "boundary and should not be used as one for a legal edge.",
        f"The weakest corner was marked on fix type {worst_fix} with {worst_sats} "
        "satellites.",
    ]
    if any(mark.hdop and mark.hdop > 2.0 for mark in marks):
        limits.append(
            "At least one corner was marked with an HDOP above 2, so its horizontal "
            "position is several metres uncertain."
        )

    return {
        "polygon": ring,
        "marks": [mark.to_dict() for mark in marks],
        "corner_count": len(marks),
        "area_m2": round(area_m2, 2),
        "area_hectares": round(area_m2 / 10_000.0, 4),
        "perimeter_m": round(perimeter_m, 2),
        "source": "flown_marks",
        "limits": limits,
    }
