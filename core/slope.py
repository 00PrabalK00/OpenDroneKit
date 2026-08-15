"""Gradient from an elevation raster: roof pitch, pavement fall, ramp compliance.

Slope is the measurement clients quote back at you. A roof is specified as a pitch, a
car park as a fall, a ramp as a maximum gradient someone has to comply with, and each of
those is a number that gets written into a report and argued about later.

Two things make it easy to produce a confident wrong answer.

The first is the grid. Slope is computed between neighbouring cells, so a 5 cm DSM and
a 50 cm DSM of the same roof do not give the same maximum -- the fine grid resolves a
tile edge, the coarse one averages it away. The mean is fairly stable; the maximum is
mostly a statement about resolution. So the cell size is reported alongside every
result, and the maximum is labelled for what it is.

The second is the CRS. In a geographic CRS the horizontal units are degrees and the
vertical ones metres, so a gradient computed from them is a ratio of unlike quantities.
It still evaluates. It produces a plausible-looking angle. It is meaningless, so a
geographic raster is refused rather than measured.

Where a surface really is one plane -- a roof facet, a ramp -- fitting that plane and
reporting the residual says more than a distribution does: the residual is what tells
you whether "the pitch" was a single pitch at all, or an average across a valley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dsm_analysis import NotGeoreferenced, RasterSurface, _polygon_mask, load_surface

# Below this many cells there is not enough surface to describe a gradient: three cells
# can be fitted exactly by a plane, which would report a residual of zero and mean it.
MIN_CELLS = 12

# A plane fit whose residual exceeds this fraction of the height range is not describing
# one facet. Reported rather than enforced, since a slightly bowed slab is still usefully
# summarised by its mean plane.
PLANAR_RESIDUAL_WARN = 0.15


class NotProjected(ValueError):
    """The raster's horizontal units are not metres, so a gradient is not an angle."""


@dataclass
class SlopeResult:
    """Gradient over one region, with what the numbers depend on stated."""

    cell_count: int
    cell_size_m: float
    mean_deg: float
    median_deg: float
    max_deg: float
    p95_deg: float
    mean_percent: float
    aspect_deg: float | None
    plane_dip_deg: float
    plane_aspect_deg: float | None
    plane_residual_m: float
    height_range_m: float
    planar: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_count": self.cell_count,
            "cell_size_m": round(self.cell_size_m, 4),
            "mean_deg": round(self.mean_deg, 3),
            "median_deg": round(self.median_deg, 3),
            "max_deg": round(self.max_deg, 3),
            "p95_deg": round(self.p95_deg, 3),
            "mean_percent": round(self.mean_percent, 2),
            "aspect_deg": None if self.aspect_deg is None else round(self.aspect_deg, 1),
            "plane_dip_deg": round(self.plane_dip_deg, 3),
            "plane_aspect_deg": (None if self.plane_aspect_deg is None
                                 else round(self.plane_aspect_deg, 1)),
            "plane_residual_m": round(self.plane_residual_m, 4),
            "height_range_m": round(self.height_range_m, 4),
            "planar": self.planar,
            "roof_pitch_ratio": self.roof_pitch_ratio(),
        }

    def roof_pitch_ratio(self) -> str:
        """The dip as a rise-per-12 ratio, which is how a roof is usually specified."""
        rise = math.tan(math.radians(self.plane_dip_deg)) * 12.0
        return f"{rise:.1f}:12"


def _require_projected(surface: RasterSurface) -> None:
    if surface.epsg is None:
        raise NotGeoreferenced(
            f"{surface.path or 'The raster'} has no coordinate reference system, so a "
            "gradient measured on it would not be an angle."
        )
    try:
        from rasterio.crs import CRS

        crs = CRS.from_epsg(int(surface.epsg))
    except Exception as exc:  # noqa: BLE001 - any CRS failure is a refusal
        raise NotProjected(
            f"EPSG:{surface.epsg} could not be resolved, so the horizontal units of "
            "this raster are unknown and no gradient can be computed from it."
        ) from exc
    if not crs.is_projected:
        raise NotProjected(
            f"EPSG:{surface.epsg} is geographic: horizontal distances are in degrees "
            "while elevations are in metres, so their ratio is not a slope. Reproject "
            "the surface to a metric CRS -- a UTM zone -- before measuring gradient."
        )


def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Least-squares z = ax + by + c, returned as (a, b, c)."""
    design = np.column_stack([x, y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, z, rcond=None)
    return float(solution[0]), float(solution[1]), float(solution[2])


def _aspect_deg(dz_dx: float, dz_dy: float) -> float | None:
    """Compass bearing the surface falls towards, or None on a level surface."""
    if abs(dz_dx) < 1e-12 and abs(dz_dy) < 1e-12:
        return None
    # Downhill direction is the negative gradient; bearing is clockwise from north.
    bearing = math.degrees(math.atan2(-dz_dx, -dz_dy))
    return bearing % 360.0


def measure_slope(
    surface_path: str | Path,
    *,
    polygon_xy: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Gradient statistics over a raster, optionally clipped to a polygon.

    Returns both a per-cell distribution and a single fitted plane. They answer
    different questions: the distribution describes a surface that varies, the plane
    describes one that is meant not to, and the residual between them says which of the
    two this actually is.
    """
    surface = load_surface(surface_path)
    _require_projected(surface)

    region = surface.valid_mask
    if polygon_xy:
        region &= _polygon_mask(surface, polygon_xy)

    cell_count = int(region.sum())
    if cell_count < MIN_CELLS:
        return {
            "ok": False,
            "reason": (
                f"Only {cell_count} valid cells inside the region; at least {MIN_CELLS} "
                "are needed before a gradient describes a surface rather than noise."
            ),
            "surface_path": str(surface_path),
            "cell_count": cell_count,
        }

    elevation = surface.elevation
    step_x = abs(float(surface.transform[0]))
    step_y = abs(float(surface.transform[4]))
    if step_x <= 0 or step_y <= 0:
        raise NotProjected("The raster transform has no ground spacing to measure across.")

    # Central differences over the whole grid, then restricted to the region. Cells whose
    # neighbours are nodata drop out: a gradient computed across a hole in the surface is
    # a gradient across an invented elevation.
    filled = np.where(np.isfinite(elevation), elevation, np.nan)
    dz_dy, dz_dx = np.gradient(filled, step_y, step_x)
    usable = region & np.isfinite(dz_dx) & np.isfinite(dz_dy)
    if int(usable.sum()) < MIN_CELLS:
        return {
            "ok": False,
            "reason": (
                "The region has too few cells with valid neighbours to differentiate. "
                "Gradient needs a continuous surface, not scattered points."
            ),
            "surface_path": str(surface_path),
            "cell_count": int(usable.sum()),
        }

    # Rows increase southward in a north-up raster, so the y derivative is negated to
    # make positive dz_dy mean "rising to the north".
    gradient_north = -dz_dy[usable]
    gradient_east = dz_dx[usable]
    magnitude = np.hypot(gradient_east, gradient_north)
    degrees = np.degrees(np.arctan(magnitude))

    rows, cols = np.nonzero(usable)
    world_x, world_y = surface.xy_of(rows, cols)
    heights = elevation[usable]
    plane_x = np.asarray(world_x, dtype=np.float64)
    plane_y = np.asarray(world_y, dtype=np.float64)
    a, b, c = _fit_plane(plane_x, plane_y, heights)
    residual = float(np.sqrt(np.mean(np.square(heights - (a * plane_x + b * plane_y + c)))))
    height_range = float(np.max(heights) - np.min(heights))
    dip = math.degrees(math.atan(math.hypot(a, b)))

    result = SlopeResult(
        cell_count=int(usable.sum()),
        cell_size_m=(step_x + step_y) / 2.0,
        mean_deg=float(np.mean(degrees)),
        median_deg=float(np.median(degrees)),
        max_deg=float(np.max(degrees)),
        p95_deg=float(np.percentile(degrees, 95)),
        mean_percent=float(np.mean(magnitude) * 100.0),
        aspect_deg=_aspect_deg(float(np.mean(gradient_east)), float(np.mean(gradient_north))),
        plane_dip_deg=dip,
        plane_aspect_deg=_aspect_deg(a, b),
        plane_residual_m=residual,
        height_range_m=height_range,
        planar=bool(height_range > 0 and residual <= PLANAR_RESIDUAL_WARN * height_range),
    )

    limits = [
        f"Gradient is computed between cells {result.cell_size_m:.3f} m apart. The mean "
        "is stable across resolutions; the maximum largely reflects this cell size.",
        "Slope is measured on the surface as reconstructed, which on a roof includes "
        "tiles, vents and anything sitting on it.",
    ]
    if not result.planar:
        limits.append(
            f"The fitted plane leaves {residual:.3f} m RMS residual against a "
            f"{height_range:.3f} m height range, so this region is not one flat facet. "
            "A single pitch quoted for it would average across more than one surface."
        )
    return {
        "ok": True,
        "surface_path": str(surface_path),
        "epsg": surface.epsg,
        "slope": result.to_dict(),
        "limits": limits,
    }


def classify_gradient(slope_percent: float, *, standard: str = "") -> dict[str, Any]:
    """Describe a gradient against a stated threshold, without asserting compliance.

    Deliberately not a pass/fail verdict. Ramp and pavement limits differ by
    jurisdiction, by whether the run is a landing or a slope, and by tolerances this
    module has no knowledge of, so the honest output is the measured number set beside
    the threshold the caller named, leaving the judgement where it belongs.
    """
    if not math.isfinite(slope_percent):
        raise ValueError("A gradient must be finite to be described.")
    ratio = math.inf if slope_percent == 0 else 100.0 / abs(slope_percent)
    return {
        "slope_percent": round(float(slope_percent), 3),
        "slope_deg": round(math.degrees(math.atan(abs(slope_percent) / 100.0)), 3),
        "one_in": None if not math.isfinite(ratio) else round(ratio, 2),
        "compared_with": standard,
        "note": (
            "Measured gradient only. Whether it complies with a standard depends on the "
            "standard's own definitions of run, landing and tolerance, which are not "
            "encoded here."
        ),
    }
