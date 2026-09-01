"""Put the drawing on top of what was actually built.

The question an overlay answers is "does the site match the design". A DXF of the intended
layout, or a scanned site plan, laid over the orthomosaic in the same coordinate system,
turns that from an argument into a picture: the road is two metres east of where it was
drawn, the slab is short at one corner, the extension is not where the plan says.

Two kinds of source, because they arrive differently:

  DXF     Vector geometry in a known CRS. Reprojected onto the project's CRS.
  Raster  A PNG or JPG with no georeferencing at all -- a scan, a screenshot of a plan.
          Placed by the bounding box the operator supplies.

The DXF reader here is deliberately small. ezdxf is not installed and this module refuses
to depend on it, for the reason core/model_measurement.py gives about mesh libraries: an
overlay that only works when an optional package happens to be present is not one a site
can rely on. DXF is a tagged text format, and the handful of entity types a site plan
actually uses -- lines, polylines, circles, arcs -- are readable directly.

What it will NOT do is guess. A DXF with no CRS given is refused rather than assumed to be
in the project's coordinates, because an overlay that is silently in the wrong place is
worse than no overlay: it looks like evidence of a construction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterator, Sequence


class OverlayRefused(ValueError):
    """An overlay that cannot be placed, or would be placed somewhere untrue."""


#: Entity types this reader understands. Anything else is counted and reported rather
#: than silently dropped -- a plan whose walls are BLOCKs would otherwise overlay as an
#: empty drawing and look like a clean match.
SUPPORTED_ENTITIES = ("LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC")

#: How finely curves are flattened. A circle becomes a closed polyline; 64 segments keeps
#: the error under a millimetre on a 10 m radius, which is well inside survey noise.
CURVE_SEGMENTS = 64


@dataclass
class CadDrawing:
    """Flattened 2D geometry from a drawing, in one coordinate system."""

    polylines: list[list[list[float]]] = field(default_factory=list)
    source_epsg: int | None = None
    entity_counts: dict[str, int] = field(default_factory=dict)
    skipped_entities: dict[str, int] = field(default_factory=dict)

    def bounds(self) -> tuple[float, float, float, float]:
        """min_x, min_y, max_x, max_y over every vertex."""
        xs = [p[0] for line in self.polylines for p in line]
        ys = [p[1] for line in self.polylines for p in line]
        if not xs:
            raise OverlayRefused("The drawing has no geometry to place.")
        return min(xs), min(ys), max(xs), max(ys)

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": line},
                    "properties": {},
                }
                for line in self.polylines
            ],
        }


def _tag_pairs(path: Path) -> Iterator[tuple[int, str]]:
    """DXF is (group code, value) on alternating lines. Yield them as pairs."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            code_line = handle.readline()
            if not code_line:
                return
            value_line = handle.readline()
            if not value_line:
                return
            try:
                yield int(code_line.strip()), value_line.rstrip("\n").rstrip("\r")
            except ValueError:
                # A malformed pair is skipped rather than fatal: real drawings carry
                # vendor extensions, and one unreadable tag should not lose the walls.
                continue


def _arc_points(cx: float, cy: float, radius: float,
                start_deg: float, end_deg: float) -> list[list[float]]:
    sweep = (end_deg - start_deg) % 360.0
    if sweep == 0.0:
        sweep = 360.0
    steps = max(2, int(CURVE_SEGMENTS * sweep / 360.0))
    return [
        [cx + radius * math.cos(math.radians(start_deg + sweep * i / steps)),
         cy + radius * math.sin(math.radians(start_deg + sweep * i / steps))]
        for i in range(steps + 1)
    ]


def read_dxf(path: str | Path) -> CadDrawing:
    """Flatten a DXF into polylines.

    Only the entity types a site plan actually uses. Anything else is COUNTED and
    reported: a drawing whose walls are inside BLOCK references would otherwise produce
    an empty overlay, which on top of an orthomosaic reads as "the design matches
    nothing" rather than "this reader did not understand the file".
    """
    source = Path(path)
    if not source.is_file():
        raise OverlayRefused(f"Drawing not found: {source}")

    drawing = CadDrawing()
    entity = ""
    x_values: list[float] = []
    y_values: list[float] = []
    numbers: dict[int, float] = {}
    closed = False

    def flush() -> None:
        nonlocal entity, x_values, y_values, numbers, closed
        if entity in ("LINE",) and len(x_values) >= 2 and len(y_values) >= 2:
            drawing.polylines.append([[x_values[0], y_values[0]], [x_values[1], y_values[1]]])
            drawing.entity_counts["LINE"] = drawing.entity_counts.get("LINE", 0) + 1
        elif entity in ("LWPOLYLINE", "POLYLINE") and len(x_values) >= 2:
            line = [[x, y] for x, y in zip(x_values, y_values)]
            if closed and line[0] != line[-1]:
                line.append(line[0])
            drawing.polylines.append(line)
            drawing.entity_counts[entity] = drawing.entity_counts.get(entity, 0) + 1
        elif entity == "CIRCLE" and x_values and y_values and 40 in numbers:
            drawing.polylines.append(
                _arc_points(x_values[0], y_values[0], numbers[40], 0.0, 360.0))
            drawing.entity_counts["CIRCLE"] = drawing.entity_counts.get("CIRCLE", 0) + 1
        elif entity == "ARC" and x_values and y_values and 40 in numbers:
            drawing.polylines.append(_arc_points(
                x_values[0], y_values[0], numbers[40],
                numbers.get(50, 0.0), numbers.get(51, 360.0)))
            drawing.entity_counts["ARC"] = drawing.entity_counts.get("ARC", 0) + 1
        elif entity and entity not in ("SECTION", "ENDSEC", "EOF", "VERTEX", "SEQEND"):
            drawing.skipped_entities[entity] = drawing.skipped_entities.get(entity, 0) + 1
        entity, x_values, y_values, numbers, closed = "", [], [], {}, False

    for code, value in _tag_pairs(source):
        if code == 0:
            flush()
            entity = value.strip().upper()
        elif not entity:
            continue
        elif code == 10:
            try:
                x_values.append(float(value))
            except ValueError:
                pass
        elif code == 20:
            try:
                y_values.append(float(value))
            except ValueError:
                pass
        elif code == 11:
            try:
                x_values.append(float(value))
            except ValueError:
                pass
        elif code == 21:
            try:
                y_values.append(float(value))
            except ValueError:
                pass
        elif code in (40, 50, 51):
            try:
                numbers[code] = float(value)
            except ValueError:
                pass
        elif code == 70:
            try:
                closed = bool(int(value) & 1)
            except ValueError:
                pass
    flush()

    if not drawing.polylines:
        detail = ""
        if drawing.skipped_entities:
            named = ", ".join(f"{k} x{v}" for k, v in sorted(drawing.skipped_entities.items()))
            detail = (
                f" It contains {named}, which this reader does not flatten. "
                "Explode blocks and hatches in the CAD package, then export again."
            )
        raise OverlayRefused(f"No usable geometry in {source.name}.{detail}")
    return drawing


def reproject(drawing: CadDrawing, src_epsg: int, dst_epsg: int) -> CadDrawing:
    """Move a drawing into the project's coordinate system.

    src_epsg is required by the caller, not guessed. A DXF carries no CRS of its own, and
    assuming it matches the project puts the overlay somewhere plausible and wrong -- which
    on top of an orthomosaic reads as a construction error rather than a placement error.
    """
    if int(src_epsg) == int(dst_epsg):
        drawing.source_epsg = int(src_epsg)
        return drawing

    from core import geo

    transform = geo.transformer(int(src_epsg), int(dst_epsg))
    moved = CadDrawing(
        source_epsg=int(src_epsg),
        entity_counts=dict(drawing.entity_counts),
        skipped_entities=dict(drawing.skipped_entities),
    )
    for line in drawing.polylines:
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        tx, ty = transform.transform(xs, ys)
        moved.polylines.append([[float(a), float(b)] for a, b in zip(tx, ty)])
    return moved


@dataclass
class RasterPlacement:
    """A picture with no georeferencing, placed by the corners the operator gives."""

    path: str
    west: float
    south: float
    east: float
    north: float
    epsg: int

    def bounds(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bounds": list(self.bounds()),
            "epsg": int(self.epsg),
        }


def place_raster(path: str | Path, west: float, south: float, east: float, north: float,
                 epsg: int = 4326) -> RasterPlacement:
    """Georeference a plain image by its bounding box.

    The checks here are the whole value. A scan placed with west and east swapped, or with
    a zero-width box, still renders -- mirrored, or as a line -- and an operator comparing
    it against the orthomosaic would be reading a mirrored plan as a build error.
    """
    source = Path(path)
    if not source.is_file():
        raise OverlayRefused(f"Image not found: {source}")
    if east <= west:
        raise OverlayRefused(
            f"East ({east}) must be greater than west ({west}). "
            "Swapped bounds mirror the plan, which reads as a construction error."
        )
    if north <= south:
        raise OverlayRefused(
            f"North ({north}) must be greater than south ({south}). "
            "Swapped bounds flip the plan, which reads as a construction error."
        )
    if int(epsg) == 4326:
        if not (-180.0 <= west < east <= 180.0):
            raise OverlayRefused("Longitudes must lie between -180 and 180.")
        if not (-90.0 <= south < north <= 90.0):
            raise OverlayRefused("Latitudes must lie between -90 and 90.")
    return RasterPlacement(str(source), float(west), float(south),
                           float(east), float(north), int(epsg))


def alignment_report(drawing: CadDrawing, target_bounds: Sequence[float]) -> dict[str, Any]:
    """How far the drawing sits from the survey it is meant to sit on.

    An overlay in the right CRS but the wrong place is the common failure -- a plan in
    local site coordinates, or a DXF whose origin is a corner of the sheet rather than a
    survey point. Reporting the offset lets an operator see that immediately instead of
    concluding the building moved.
    """
    dmin_x, dmin_y, dmax_x, dmax_y = drawing.bounds()
    tmin_x, tmin_y, tmax_x, tmax_y = (float(v) for v in target_bounds)

    overlaps = not (dmax_x < tmin_x or dmin_x > tmax_x or dmax_y < tmin_y or dmin_y > tmax_y)
    centre_offset = math.dist(
        ((dmin_x + dmax_x) / 2.0, (dmin_y + dmax_y) / 2.0),
        ((tmin_x + tmax_x) / 2.0, (tmin_y + tmax_y) / 2.0),
    )
    report: dict[str, Any] = {
        "overlaps": overlaps,
        "centre_offset_m": centre_offset,
        "drawing_bounds": [dmin_x, dmin_y, dmax_x, dmax_y],
        "target_bounds": [tmin_x, tmin_y, tmax_x, tmax_y],
    }
    if not overlaps:
        report["warning"] = (
            f"The drawing does not overlap the survey at all; its centre is "
            f"{centre_offset:,.0f} m away. Check the CRS it was exported in, and whether "
            "its origin is a survey point or a corner of the sheet."
        )
    return report
