"""Measuring distance, area and perimeter directly on a georeferenced raster.

The existing measurement helpers work in pixels and take a scale the caller supplies.
That is right for a close-range photograph, where the scale comes from a ruler in the
frame, but wrong for an orthomosaic: the raster already knows its own scale, and asking
the operator to retype it invites a wrong number that nothing can catch.

So this reads the geometry from the file. Given an orthomosaic or DSM produced by the
pipeline, a click in pixel space becomes a position in the raster's CRS, and lengths
come out in that CRS's units without anyone declaring a scale.

The rule this module exists to enforce: **a raster with no CRS cannot be measured in
metres.** The custom reconstruction engine writes PNGs, and a PNG carries no
georeferencing at all. Measuring one and reporting square metres would produce a number
that looks like an area and is actually a pixel count -- the kind of output that ends up
in a report and is believed. Every entry point here refuses that case by name.

Areas are computed with the shoelace formula in projected coordinates. For a UTM
orthomosaic covering a survey site this is exact to the projection itself; the residual
error is UTM's own scale distortion, which is under 0.1% near the central meridian and
is reported alongside the result rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# UTM's scale factor at the central meridian. The projection shrinks distances slightly
# in the middle of a zone and stretches them at the edges; quoting it lets a surveyor
# decide whether it matters for their tolerance.
UTM_CENTRAL_SCALE_FACTOR = 0.9996


class NotGeoreferenced(ValueError):
    """The raster carries no CRS, so nothing on it can be measured in real units."""


class NotProjected(ValueError):
    """The raster is in degrees, where a shoelace area would not be in square metres."""


@dataclass
class RasterMeasurement:
    """One measurement taken on a georeferenced raster."""

    kind: str
    value: float
    unit: str
    vertices_world: list[tuple[float, float]] = field(default_factory=list)
    epsg: int | None = None
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": round(self.value, 3),
            "unit": self.unit,
            "epsg": self.epsg,
            "vertices_world": [[round(x, 3), round(y, 3)] for x, y in self.vertices_world],
            "caveats": self.caveats,
        }


def _open(path: str | Path):
    """Open a raster, refusing anything that cannot carry a CRS."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "rasterio is required to measure on a raster. Install it with "
            "`pip install rasterio`."
        ) from exc

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Raster not found: {source}")

    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        # Named explicitly, because this is the case that would otherwise yield a
        # confident number in the wrong units.
        raise NotGeoreferenced(
            f"{source.name} is an image format that cannot carry georeferencing. "
            "Measurements on it would be in pixels, not metres. Use the GeoTIFF the "
            "COLMAP engine produces (dsm.tif, orthomosaic.tif)."
        )

    raster = rasterio.open(source)
    if raster.crs is None:
        raster.close()
        raise NotGeoreferenced(
            f"{source.name} has no coordinate reference system, so a distance on it "
            "has no real-world length. Reproject it or reprocess with a georeferenced "
            "engine."
        )
    return raster


def _require_projected(raster, path: Path) -> int | None:
    """Refuse a geographic CRS, where planar area is not an area in square metres."""
    crs = raster.crs
    if crs.is_geographic:
        raise NotProjected(
            f"{path.name} is in geographic coordinates (degrees). A shoelace area in "
            "degrees is not square metres and varies with latitude. Reproject to a "
            "projected CRS such as the survey's UTM zone first."
        )
    return crs.to_epsg()


def pixel_to_world(path: str | Path,
                   pixels: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convert pixel coordinates to the raster's CRS."""
    raster = _open(path)
    try:
        return [tuple(raster.xy(row, col)) for col, row in pixels]
    finally:
        raster.close()


def measure_distance(path: str | Path,
                     pixels: Sequence[tuple[float, float]]) -> RasterMeasurement:
    """Length of a polyline drawn on the raster, in the raster's units."""
    if len(pixels) < 2:
        raise ValueError("A distance needs at least two points.")

    raster = _open(path)
    try:
        epsg = _require_projected(raster, Path(path))
        world = [tuple(raster.xy(row, col)) for col, row in pixels]
        unit = raster.crs.linear_units or "metre"
    finally:
        raster.close()

    total = 0.0
    for (x1, y1), (x2, y2) in zip(world, world[1:]):
        total += math.hypot(x2 - x1, y2 - y1)

    return RasterMeasurement(
        kind="distance", value=total, unit=unit, vertices_world=world, epsg=epsg,
        caveats=_projection_caveats(epsg),
    )


def measure_area(path: str | Path,
                 pixels: Sequence[tuple[float, float]]) -> RasterMeasurement:
    """Area of a polygon drawn on the raster, by shoelace in projected coordinates."""
    if len(pixels) < 3:
        raise ValueError("An area needs at least three points.")

    raster = _open(path)
    try:
        epsg = _require_projected(raster, Path(path))
        world = [tuple(raster.xy(row, col)) for col, row in pixels]
        unit = raster.crs.linear_units or "metre"
    finally:
        raster.close()

    doubled = 0.0
    count = len(world)
    for index in range(count):
        x1, y1 = world[index]
        x2, y2 = world[(index + 1) % count]
        doubled += x1 * y2 - x2 * y1

    return RasterMeasurement(
        kind="area", value=abs(doubled) / 2.0, unit=f"square {unit}",
        vertices_world=world, epsg=epsg, caveats=_projection_caveats(epsg),
    )


def measure_perimeter(path: str | Path,
                      pixels: Sequence[tuple[float, float]]) -> RasterMeasurement:
    """Perimeter of a closed polygon drawn on the raster."""
    if len(pixels) < 3:
        raise ValueError("A perimeter needs at least three points.")

    closed = list(pixels) + [pixels[0]]
    result = measure_distance(path, closed)
    result.kind = "perimeter"
    # The repeated closing vertex is an artifact of the calculation, not a corner the
    # operator drew.
    result.vertices_world = result.vertices_world[:-1]
    return result


def _projection_caveats(epsg: int | None) -> list[str]:
    caveats = [
        "Measured in the raster's projected CRS; accuracy is limited by the "
        "reconstruction's georeferencing residual, not by this arithmetic.",
    ]
    if epsg and (32600 < epsg < 32661 or 32700 < epsg < 32761):
        caveats.append(
            f"UTM applies a scale factor of {UTM_CENTRAL_SCALE_FACTOR} at the zone's "
            "central meridian, so ground distances differ by up to about 0.1%."
        )
    return caveats


def raster_ground_sample_distance(path: str | Path) -> dict[str, Any]:
    """Report the raster's cell size, which sets the finest measurement worth trusting."""
    raster = _open(path)
    try:
        epsg = raster.crs.to_epsg()
        x_size, y_size = abs(raster.transform.a), abs(raster.transform.e)
        unit = raster.crs.linear_units or "metre"
        geographic = raster.crs.is_geographic
    finally:
        raster.close()

    return {
        "cell_width": round(x_size, 4),
        "cell_height": round(y_size, 4),
        "unit": unit,
        "epsg": epsg,
        "geographic": geographic,
        "note": (
            "A measurement finer than one cell is interpolation, not observation. "
            f"This raster resolves about {max(x_size, y_size):.3f} {unit} per cell."
        ),
    }
