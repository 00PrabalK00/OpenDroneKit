"""Recording a flight, and exporting it in the formats an investigation needs.

A telemetry stream is transient. The moment worth having it is afterwards -- when a
survey came out wrong, when an aircraft behaved oddly, when a client asks what altitude
something was flown at, or when an authority asks where the aircraft was at a given
time. None of those questions can be answered from a live feed that was never written
down.

Four formats, because each is read by different software. CSV opens in a spreadsheet and
is what most operators actually look at. JSON keeps the full record including fields the
other formats have no column for. GPX is what mapping tools and flight-log analysers
import. KML is what gets dropped into Google Earth to show someone where the aircraft
went.

Two things are deliberate. A sample with no GPS fix is written to CSV and JSON but
omitted from the GPX and KML tracks, because a track is a claim about position and
0,0 off the coast of Africa is not one. And the exporters never interpolate: a gap in
the recording stays a gap, since a smooth line through missing data is a picture of a
flight that was not observed.
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# GPS fix types below this do not give a usable position. 0 is no GPS, 1 is a receiver
# with no fix; 2 is 2D and 3 is 3D.
MIN_USABLE_FIX = 2

SUPPORTED_FORMATS = ("csv", "json", "gpx", "kml")

# Columns written to CSV, in the order an operator reads them.
CSV_FIELDS = (
    "timestamp_utc", "elapsed_s", "latitude", "longitude", "altitude_rel_m",
    "altitude_abs_m", "heading_deg", "speed_mps", "battery_pct", "battery_v",
    "gps_fix", "satellites", "hdop", "flight_mode", "armed",
    "waypoint_index", "waypoint_total", "link_quality_pct",
)


@dataclass
class FlightSample:
    """One telemetry snapshot, as recorded."""

    timestamp: float
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_rel_m: float = 0.0
    altitude_abs_m: float = 0.0
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    battery_pct: float = 0.0
    battery_v: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    hdop: float = 99.9
    flight_mode: str = "UNKNOWN"
    armed: bool = False
    waypoint_index: int = 0
    waypoint_total: int = 0
    link_quality_pct: float = 0.0

    @property
    def has_position(self) -> bool:
        """Whether this sample carries a position anyone should plot."""
        if self.gps_fix < MIN_USABLE_FIX:
            return False
        # A receiver reporting a fix at exactly the null island coordinate is reporting
        # its uninitialised state, not a location off Ghana.
        return not (abs(self.latitude) < 1e-9 and abs(self.longitude) < 1e-9)

    @classmethod
    def from_telemetry(cls, telemetry: Any) -> "FlightSample":
        get = (telemetry.get if isinstance(telemetry, dict)
               else lambda k, d=None: getattr(telemetry, k, d))
        return cls(
            timestamp=float(get("timestamp", 0.0) or 0.0),
            latitude=float(get("latitude", 0.0) or 0.0),
            longitude=float(get("longitude", 0.0) or 0.0),
            altitude_rel_m=float(get("altitude_rel_m", 0.0) or 0.0),
            altitude_abs_m=float(get("altitude_abs_m", 0.0) or 0.0),
            heading_deg=float(get("heading_deg", 0.0) or 0.0),
            speed_mps=float(get("speed_mps", 0.0) or 0.0),
            battery_pct=float(get("battery_pct", 0.0) or 0.0),
            battery_v=float(get("battery_v", 0.0) or 0.0),
            gps_fix=int(get("gps_fix", 0) or 0),
            satellites=int(get("satellites", 0) or 0),
            hdop=float(get("hdop", 99.9) or 99.9),
            flight_mode=str(get("flight_mode", "UNKNOWN") or "UNKNOWN"),
            armed=bool(get("armed", False)),
            waypoint_index=int(get("waypoint_index", 0) or 0),
            waypoint_total=int(get("waypoint_total", 0) or 0),
            link_quality_pct=float(get("link_quality_pct", 0.0) or 0.0),
        )

    def to_dict(self, start: float = 0.0) -> dict[str, Any]:
        return {
            "timestamp_utc": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc).isoformat(),
            "elapsed_s": round(self.timestamp - start, 2) if start else 0.0,
            "latitude": self.latitude, "longitude": self.longitude,
            "altitude_rel_m": self.altitude_rel_m, "altitude_abs_m": self.altitude_abs_m,
            "heading_deg": self.heading_deg, "speed_mps": self.speed_mps,
            "battery_pct": self.battery_pct, "battery_v": self.battery_v,
            "gps_fix": self.gps_fix, "satellites": self.satellites, "hdop": self.hdop,
            "flight_mode": self.flight_mode, "armed": self.armed,
            "waypoint_index": self.waypoint_index,
            "waypoint_total": self.waypoint_total,
            "link_quality_pct": self.link_quality_pct,
        }


@dataclass
class FlightLog:
    """A recorded flight."""

    samples: list[FlightSample] = field(default_factory=list)
    aircraft: str = ""
    pilot: str = ""
    mission_name: str = ""

    def record(self, telemetry: Any) -> FlightSample:
        sample = FlightSample.from_telemetry(telemetry)
        self.samples.append(sample)
        return sample

    @property
    def start_time(self) -> float:
        return self.samples[0].timestamp if self.samples else 0.0

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    def positioned_samples(self) -> list[FlightSample]:
        return [s for s in self.samples if s.has_position]

    def distance_m(self) -> float:
        """Ground distance actually flown, over samples that had a fix."""
        located = self.positioned_samples()
        total = 0.0
        for previous, current in zip(located, located[1:]):
            total += _haversine_m(previous.latitude, previous.longitude,
                                  current.latitude, current.longitude)
        return total

    def summary(self) -> dict[str, Any]:
        located = self.positioned_samples()
        altitudes = [s.altitude_rel_m for s in located]
        batteries = [s.battery_pct for s in self.samples if s.battery_pct > 0]

        return {
            "sample_count": len(self.samples),
            "positioned_samples": len(located),
            "samples_without_fix": len(self.samples) - len(located),
            "duration_s": round(self.duration_s, 1),
            "distance_m": round(self.distance_m(), 1),
            "max_altitude_rel_m": round(max(altitudes), 1) if altitudes else None,
            "battery_start_pct": batteries[0] if batteries else None,
            "battery_end_pct": batteries[-1] if batteries else None,
            "aircraft": self.aircraft,
            "pilot": self.pilot,
            "mission_name": self.mission_name,
            "note": (
                "Distance and altitude are computed only from samples with a usable GPS "
                "fix. Gaps in the recording are left as gaps; nothing is interpolated."
            ),
        }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Arguments are (lat, lon), not (lon, lat)."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def write_csv(log: FlightLog, path: str | Path) -> Path:
    """Every sample, including those with no fix, which a spreadsheet can filter."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = log.start_time

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for sample in log.samples:
            row = sample.to_dict(start)
            writer.writerow({key: row[key] for key in CSV_FIELDS})
    return target


def write_json(log: FlightLog, path: str | Path) -> Path:
    """The full record, including the summary and the fields other formats drop."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = log.start_time

    payload = {
        "summary": log.summary(),
        "samples": [sample.to_dict(start) for sample in log.samples],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def write_gpx(log: FlightLog, path: str | Path) -> Path:
    """A track for mapping tools. Samples without a fix are omitted, not zeroed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    gpx = ET.Element("gpx", {
        "version": "1.1", "creator": "OpenDroneKit",
        "xmlns": "http://www.topografix.com/GPX/1/1",
    })
    metadata = ET.SubElement(gpx, "metadata")
    ET.SubElement(metadata, "name").text = log.mission_name or "Flight"

    track = ET.SubElement(gpx, "trk")
    ET.SubElement(track, "name").text = log.mission_name or "Flight"
    segment = ET.SubElement(track, "trkseg")

    for sample in log.positioned_samples():
        point = ET.SubElement(segment, "trkpt", {
            "lat": f"{sample.latitude:.8f}", "lon": f"{sample.longitude:.8f}"})
        ET.SubElement(point, "ele").text = f"{sample.altitude_abs_m:.2f}"
        ET.SubElement(point, "time").text = datetime.fromtimestamp(
            sample.timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ET.SubElement(point, "sat").text = str(sample.satellites)
        ET.SubElement(point, "hdop").text = f"{sample.hdop:.2f}"

    ET.ElementTree(gpx).write(target, encoding="utf-8", xml_declaration=True)
    return target


def write_kml(log: FlightLog, path: str | Path) -> Path:
    """A track for Google Earth, drawn at its real altitude rather than draped."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    kml = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = log.mission_name or "Flight"

    placemark = ET.SubElement(document, "Placemark")
    ET.SubElement(placemark, "name").text = log.mission_name or "Flight track"
    line = ET.SubElement(placemark, "LineString")
    ET.SubElement(line, "extrude").text = "0"
    # absolute rather than clampToGround: the altitude flown is part of the record.
    ET.SubElement(line, "altitudeMode").text = "absolute"

    coordinates = " ".join(
        f"{s.longitude:.8f},{s.latitude:.8f},{s.altitude_abs_m:.2f}"
        for s in log.positioned_samples()
    )
    ET.SubElement(line, "coordinates").text = coordinates

    ET.ElementTree(kml).write(target, encoding="utf-8", xml_declaration=True)
    return target


_WRITERS = {"csv": write_csv, "json": write_json, "gpx": write_gpx, "kml": write_kml}


def export(log: FlightLog, path: str | Path, output_format: str = "csv") -> Path:
    """Write a flight log in one of the supported formats."""
    key = output_format.lower().lstrip(".")
    writer = _WRITERS.get(key)
    if writer is None:
        raise ValueError(
            f"Unsupported flight log format {output_format!r}. "
            f"Use one of: {', '.join(SUPPORTED_FORMATS)}."
        )
    return writer(log, path)


def export_all(log: FlightLog, directory: str | Path,
               stem: str = "flight") -> dict[str, str]:
    """Write every format at once, which is what an archive wants."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    return {
        fmt: str(export(log, root / f"{stem}.{fmt}", fmt))
        for fmt in SUPPORTED_FORMATS
    }
