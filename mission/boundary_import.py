"""Reading an area of interest out of the files surveyors actually have.

A boundary rarely starts life in this tool. It arrives as a KML from a client, a
shapefile export turned GeoJSON, a GPX track walked around the site, or a CSV of
corner coordinates typed out of a title deed. Making the operator redraw it by hand is
both slow and a source of error, since a hand-traced boundary is not the boundary.

Every reader here returns the same thing: a list of ``[longitude, latitude]`` pairs in
WGS84, ready for the planner.

Two failure modes get explicit attention, because both produce a plausible wrong answer
rather than an error:

*Coordinate order.* KML writes ``lon,lat,alt`` while GPX attributes are ``lat`` and
``lon``, and a CSV may be either. Silently swapping them puts an Indian survey in
Somalia -- still on the map, still plausible-looking. Ordering is therefore explicit
per format, and results are range-checked before being returned.

*Ambiguous CSV columns.* Rather than guess which column is which, a CSV without usable
headers is refused with a message saying what was found.
"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Iterable

# Anything outside these is not a WGS84 coordinate, and the usual cause is swapped
# axes rather than a genuinely odd location.
MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0

# A polygon needs three distinct corners; below that it encloses no area.
MIN_POLYGON_POINTS = 3

SUPPORTED_SUFFIXES = {".kml", ".kmz", ".geojson", ".json", ".gpx", ".csv", ".txt"}


class BoundaryImportError(ValueError):
    """The file could not be read as a boundary, with a reason an operator can act on."""


def _validate(points: list[list[float]], source: str) -> list[list[float]]:
    """Check coordinates are in range and the ring encloses something."""
    if not points:
        raise BoundaryImportError(f"No coordinates found in {source}.")

    for lon, lat in points:
        if not -MAX_LONGITUDE <= lon <= MAX_LONGITUDE or not -MAX_LATITUDE <= lat <= MAX_LATITUDE:
            raise BoundaryImportError(
                f"{source} contains the out-of-range coordinate ({lon}, {lat}). "
                "Longitude must be within +/-180 and latitude within +/-90; a value "
                "outside that usually means the axes are swapped in the source file."
            )

    # A closing point repeating the first adds no corner.
    deduped: list[list[float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()

    if len(deduped) < MIN_POLYGON_POINTS:
        raise BoundaryImportError(
            f"{source} yielded {len(deduped)} distinct point(s); a boundary needs at "
            f"least {MIN_POLYGON_POINTS} to enclose an area."
        )
    return deduped


def _parse_kml_coordinates(text: str) -> list[list[float]]:
    """KML coordinate tuples are ``lon,lat[,alt]`` separated by whitespace."""
    points: list[list[float]] = []
    for chunk in text.replace("\n", " ").split():
        parts = chunk.split(",")
        if len(parts) < 2:
            continue
        try:
            points.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return points


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_kml(path: str | Path) -> list[list[float]]:
    """Read the first polygon (or line) from a KML document."""
    source = Path(path)
    try:
        root = ET.fromstring(source.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError as exc:
        raise BoundaryImportError(f"{source.name} is not valid KML: {exc}") from exc

    # Prefer a polygon's outer ring; fall back to any coordinates in the document, which
    # covers LineString boundaries and hand-made files with unusual nesting.
    outer = [element for element in root.iter()
             if _strip_ns(element.tag) == "outerBoundaryIs"]
    for boundary in outer:
        for element in boundary.iter():
            if _strip_ns(element.tag) == "coordinates" and element.text:
                points = _parse_kml_coordinates(element.text)
                if points:
                    return _validate(points, source.name)

    for element in root.iter():
        if _strip_ns(element.tag) == "coordinates" and element.text:
            points = _parse_kml_coordinates(element.text)
            if points:
                return _validate(points, source.name)

    raise BoundaryImportError(f"No <coordinates> element found in {source.name}.")


def read_kmz(path: str | Path) -> list[list[float]]:
    """A KMZ is a zipped KML; read the document inside it."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".kml")]
            if not names:
                raise BoundaryImportError(f"{source.name} contains no .kml file.")
            # doc.kml is the convention; otherwise take the first.
            chosen = next((n for n in names if n.lower().endswith("doc.kml")), names[0])
            text = archive.read(chosen).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise BoundaryImportError(f"{source.name} is not a readable KMZ: {exc}") from exc

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise BoundaryImportError(f"KML inside {source.name} is not valid: {exc}") from exc

    outer = [e for e in root.iter() if _strip_ns(e.tag) == "outerBoundaryIs"]
    for boundary in outer:
        for element in boundary.iter():
            if _strip_ns(element.tag) == "coordinates" and element.text:
                points = _parse_kml_coordinates(element.text)
                if points:
                    return _validate(points, source.name)

    for element in root.iter():
        if _strip_ns(element.tag) == "coordinates" and element.text:
            points = _parse_kml_coordinates(element.text)
            if points:
                return _validate(points, source.name)

    raise BoundaryImportError(f"No <coordinates> element found inside {source.name}.")


def read_geojson(path: str | Path) -> list[list[float]]:
    """Read the first polygon from a GeoJSON file. GeoJSON is lon, lat by specification."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BoundaryImportError(f"{source.name} is not valid JSON: {exc}") from exc

    def geometries(node: Any) -> Iterable[dict[str, Any]]:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind == "FeatureCollection":
            for feature in node.get("features") or []:
                yield from geometries(feature)
        elif kind == "Feature":
            yield from geometries(node.get("geometry") or {})
        elif kind in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
            yield node

    for geometry in geometries(payload):
        coordinates = geometry.get("coordinates") or []
        kind = geometry.get("type")
        ring: Any = None
        if kind == "Polygon":
            ring = coordinates[0] if coordinates else None
        elif kind == "MultiPolygon":
            ring = coordinates[0][0] if coordinates and coordinates[0] else None
        elif kind == "LineString":
            ring = coordinates
        elif kind == "MultiLineString":
            ring = coordinates[0] if coordinates else None
        if ring:
            points = [[float(c[0]), float(c[1])] for c in ring if len(c) >= 2]
            if points:
                return _validate(points, source.name)

    raise BoundaryImportError(f"No polygon or line geometry found in {source.name}.")


def read_gpx(path: str | Path) -> list[list[float]]:
    """Read a walked boundary from a GPX track, route or waypoint list.

    GPX carries position as ``lat``/``lon`` attributes, the reverse of KML's ordering.
    """
    source = Path(path)
    try:
        root = ET.fromstring(source.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError as exc:
        raise BoundaryImportError(f"{source.name} is not valid GPX: {exc}") from exc

    # Track points first: a walked perimeter is the usual case. Then route points, then
    # standalone waypoints.
    for wanted in ("trkpt", "rtept", "wpt"):
        points: list[list[float]] = []
        for element in root.iter():
            if _strip_ns(element.tag) != wanted:
                continue
            lat, lon = element.get("lat"), element.get("lon")
            if lat is None or lon is None:
                continue
            try:
                points.append([float(lon), float(lat)])
            except ValueError:
                continue
        if points:
            return _validate(points, source.name)

    raise BoundaryImportError(f"No track, route or waypoint entries found in {source.name}.")


# Header spellings seen in the wild. Matched case-insensitively against the whole cell.
_LAT_HEADERS = {"lat", "latitude", "y", "northing_deg", "lat_deg"}
_LON_HEADERS = {"lon", "long", "lng", "longitude", "x", "easting_deg", "lon_deg"}


def read_csv(path: str | Path) -> list[list[float]]:
    """Read corner coordinates from a CSV.

    Column order is decided from the header rather than assumed, because a file of bare
    number pairs is genuinely ambiguous: 28, 77 is Delhi read one way and the Somali
    coast read the other. A file without a usable header is refused.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    rows = [row for row in csv.reader(text.splitlines()) if row and any(c.strip() for c in row)]
    if not rows:
        raise BoundaryImportError(f"{source.name} is empty.")

    header = [cell.strip().lower() for cell in rows[0]]
    lat_index = next((i for i, name in enumerate(header) if name in _LAT_HEADERS), None)
    lon_index = next((i for i, name in enumerate(header) if name in _LON_HEADERS), None)

    if lat_index is None or lon_index is None:
        found = ", ".join(repr(cell) for cell in rows[0][:8]) or "nothing"
        raise BoundaryImportError(
            f"{source.name} has no latitude/longitude header, so which column is which "
            f"cannot be determined. Found: {found}. Add a header row using lat/latitude "
            "and lon/longitude; guessing would risk placing the survey in the wrong "
            "hemisphere."
        )

    points: list[list[float]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if max(lat_index, lon_index) >= len(row):
            continue
        try:
            lat = float(row[lat_index].strip())
            lon = float(row[lon_index].strip())
        except ValueError:
            # A blank or non-numeric row is skipped; a whole file of them fails the
            # emptiness check below with a clearer message than a parse error here.
            continue
        points.append([lon, lat])

    if not points:
        raise BoundaryImportError(
            f"{source.name} has a usable header but no rows with numeric coordinates."
        )
    return _validate(points, source.name)


_READERS = {
    ".kml": read_kml,
    ".kmz": read_kmz,
    ".geojson": read_geojson,
    ".json": read_geojson,
    ".gpx": read_gpx,
    ".csv": read_csv,
    ".txt": read_csv,
}


def read_boundary(path: str | Path) -> list[list[float]]:
    """Read an area of interest from any supported file, chosen by extension."""
    source = Path(path)
    if not source.exists():
        raise BoundaryImportError(f"{source} does not exist.")

    suffix = source.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise BoundaryImportError(
            f"Cannot read a boundary from {suffix or 'a file with no extension'}. "
            f"Supported: {supported}."
        )
    return reader(source)


def describe_boundary(points: list[list[float]]) -> dict[str, Any]:
    """Summarise an imported boundary so it can be checked before planning against it."""
    from core import geo

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]

    # Shoelace in metres via a local equirectangular approximation, which is accurate
    # enough for a sanity check at survey scale.
    import math

    lat0 = sum(lats) / len(lats)
    scale_x = 111_320.0 * math.cos(math.radians(lat0))
    scale_y = 110_540.0
    xy = [(lon * scale_x, lat * scale_y) for lon, lat in points]
    area2 = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        area2 += x1 * y2 - x2 * y1
    area_m2 = abs(area2) / 2.0

    return {
        "point_count": len(points),
        "centroid": [sum(lons) / len(lons), lat0],
        "bounds": [min(lons), min(lats), max(lons), max(lats)],
        "area_m2": round(area_m2, 1),
        "area_hectares": round(area_m2 / 10_000, 3),
        "utm_epsg": geo.auto_utm_epsg(lat0, sum(lons) / len(lons)),
    }
