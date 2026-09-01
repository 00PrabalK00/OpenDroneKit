"""The things on a site that are not part of the flight, and matter anyway.

A mission plan says where the aircraft goes. It says nothing about where the pilot stands,
where the van can park, which corner of the yard has the overhead line, or where the
client asked you not to fly over. Those live in somebody's head on the day, and in nobody's
head six months later when the site is reflown by a different crew.

A marker is a named point or area with a kind, so the plan carries its own briefing. Kinds
are a fixed list rather than free text: "hazard" and "Hazard" and "danger" are one thing to
a pilot and three to a filter, and the whole value here is that a hazard cannot be missed
because somebody typed it differently.

Deliberately separate from geofences and no-fly zones. Those CONSTRAIN the aircraft and are
enforced by the planner. A marker is information for the crew -- drawing them from the same
store would eventually let an annotation stop a flight, or let a hazard note be silently
treated as a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence
import uuid


class MarkerRefused(ValueError):
    """A marker that cannot be placed, or would mean nothing where it is."""


#: What a marker can be. Fixed, because the point of the list is that a pilot scanning it
#: sees every hazard, and free text guarantees they will not.
MARKER_KINDS: dict[str, str] = {
    "takeoff": "Where the aircraft launches and lands",
    "observer": "Where a visual observer stands",
    "pilot": "Where the remote pilot stands",
    "hazard": "Overhead line, mast, crane, anything to keep clear of",
    "access": "Gate, track or parking the crew needs",
    "restricted": "Ground the client or landowner asked to be left alone",
    "landmark": "A reference point for orientation on the day",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SiteMarker:
    """One named thing on the ground."""

    name: str
    kind: str
    #: [lon, lat]. A point marker has one; an area has three or more.
    points: list[list[float]] = field(default_factory=list)
    note: str = ""
    #: Metres. Only meaningful on a point, and the reason a hazard is more than a dot:
    #: "crane, 40 m radius" is actionable and "crane" is not.
    radius_m: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_utc: str = field(default_factory=_now)

    @property
    def is_area(self) -> bool:
        return len(self.points) >= 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "points": [list(p) for p in self.points],
            "note": self.note,
            "radius_m": float(self.radius_m),
            "is_area": self.is_area,
            "created_utc": self.created_utc,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SiteMarker":
        return SiteMarker(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "landmark")),
            points=[[float(c) for c in p] for p in (raw.get("points") or [])],
            note=str(raw.get("note", "")),
            radius_m=float(raw.get("radius_m", 0.0)),
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            created_utc=str(raw.get("created_utc") or _now()),
        )


def validate(marker: SiteMarker) -> None:
    if not marker.name.strip():
        raise MarkerRefused("A marker needs a name; an unnamed hazard is a dot on a map.")
    if marker.kind not in MARKER_KINDS:
        raise MarkerRefused(
            f"{marker.kind!r} is not a marker kind. Use one of: "
            + ", ".join(sorted(MARKER_KINDS))
        )
    if not marker.points:
        raise MarkerRefused("A marker needs a position.")
    for point in marker.points:
        if len(point) < 2:
            raise MarkerRefused("Positions are [lon, lat].")
        lon, lat = float(point[0]), float(point[1])
        if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
            raise MarkerRefused(f"({lon}, {lat}) is not on Earth.")
    if len(marker.points) == 2:
        raise MarkerRefused(
            "Two points is neither a position nor an area. Give one point, or three or "
            "more to enclose ground."
        )
    if marker.radius_m < 0.0:
        raise MarkerRefused("A radius cannot be negative.")
    if marker.radius_m and marker.is_area:
        raise MarkerRefused(
            "A radius applies to a point marker. This one already encloses ground."
        )


def _metres_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle distance, good enough over a site."""
    lon1, lat1, lon2, lat2 = (math.radians(float(v)) for v in (a[0], a[1], b[0], b[1]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def hazards_near(markers: Sequence[SiteMarker], waypoints: Sequence[Sequence[float]],
                 clearance_m: float = 30.0) -> list[dict[str, Any]]:
    """Which hazard markers the planned flight passes close to.

    This does NOT stop a flight -- markers are information, not constraints, and a planner
    that silently rerouted around a note somebody typed would be worse than one that says
    nothing. It reports, so the briefing names the crane before the day rather than after.

    A hazard's own radius counts: a 40 m crane at 25 m from the line is closer than the
    coordinates alone suggest.
    """
    findings: list[dict[str, Any]] = []
    for marker in markers:
        if marker.kind != "hazard" or not marker.points:
            continue
        centre = marker.points[0]
        nearest = min(
            (_metres_between(centre, wp) for wp in waypoints if len(wp) >= 2),
            default=float("inf"),
        )
        effective = nearest - float(marker.radius_m)
        if effective <= clearance_m:
            findings.append({
                "marker_id": marker.id,
                "name": marker.name,
                "distance_m": nearest,
                "clearance_m": effective,
                "note": marker.note,
            })
    return sorted(findings, key=lambda f: f["clearance_m"])


class MarkerStore:
    """Site markers for one project."""

    FILENAME = "site_markers.json"

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / self.FILENAME

    def load(self) -> list[SiteMarker]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [SiteMarker.from_dict(entry) for entry in raw.get("markers", [])]

    def save(self, markers: Sequence[SiteMarker]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"markers": [m.to_dict() for m in markers]}, indent=2),
            encoding="utf-8",
        )

    def add(self, marker: SiteMarker) -> list[SiteMarker]:
        validate(marker)
        markers = self.load()
        # Names are how a marker is referred to on the radio, so two hazards called "the
        # crane" is a briefing problem rather than a storage one.
        if any(m.name.strip().lower() == marker.name.strip().lower() for m in markers):
            raise MarkerRefused(f"There is already a marker called {marker.name!r}.")
        markers.append(marker)
        self.save(markers)
        return markers

    def remove(self, marker_id: str) -> list[SiteMarker]:
        markers = self.load()
        remaining = [m for m in markers if m.id != marker_id]
        if len(remaining) == len(markers):
            raise MarkerRefused(f"No marker with id {marker_id!r}.")
        self.save(remaining)
        return remaining

    def to_geojson(self) -> dict[str, Any]:
        """For drawing on the map, and for handing the briefing to somebody else."""
        features = []
        for marker in self.load():
            geometry = (
                {"type": "Polygon", "coordinates": [[*marker.points, marker.points[0]]]}
                if marker.is_area
                else {"type": "Point", "coordinates": marker.points[0]}
            )
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "id": marker.id, "name": marker.name, "kind": marker.kind,
                    "note": marker.note, "radius_m": marker.radius_m,
                },
            })
        return {"type": "FeatureCollection", "features": features}
