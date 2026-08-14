"""Quantitative analysis over the georeferenced DSM/DTM pair.

These are the measurements a survey is actually commissioned for -- how much material
is in a stockpile, how far apart two points are, how large a defect is -- and they
only became computable once reconstruction started emitting real metric rasters in a
known CRS. Before that the pipeline carried three stages that wrote a note saying the
work happened somewhere else.

Everything here works in the raster's projected CRS, so lengths are metres and areas
are square metres without further conversion. Volumes are reported against several
reference surfaces because the choice of datum is the dominant source of disagreement
between two people measuring the same pile, and stating it explicitly is the only way
the number means anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import geo


@dataclass
class RasterSurface:
    """An elevation raster plus the georeferencing needed to measure on it."""

    elevation: np.ndarray          # 2-D, NaN where there is no data
    transform: tuple[float, ...]   # affine, GDAL order
    epsg: int | None
    path: str = ""

    @property
    def pixel_area_m2(self) -> float:
        """Ground area of one cell. The affine's scale terms are metres in a projected CRS."""
        return abs(float(self.transform[0]) * float(self.transform[4]))

    @property
    def pixel_size_m(self) -> float:
        return (abs(float(self.transform[0])) + abs(float(self.transform[4]))) / 2.0

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.elevation)

    def xy_of(self, row: np.ndarray | int, col: np.ndarray | int):
        """Map pixel indices to projected coordinates at cell centres."""
        a, b, c, d, e, f = (float(v) for v in self.transform[:6])
        x = c + (np.asarray(col) + 0.5) * a + (np.asarray(row) + 0.5) * b
        y = f + (np.asarray(col) + 0.5) * d + (np.asarray(row) + 0.5) * e
        return x, y


def load_surface(path: str | Path) -> RasterSurface:
    """Read an elevation GeoTIFF, normalising nodata to NaN."""
    data, meta = geo.read_geotiff(path)
    elevation = np.asarray(data[0], dtype=np.float64)

    nodata = meta.get("nodata")
    if nodata is not None and np.isfinite(nodata):
        elevation = np.where(np.isclose(elevation, float(nodata)), np.nan, elevation)
    # Some writers emit the float32 sentinel instead of declaring nodata.
    elevation = np.where(elevation < -1e30, np.nan, elevation)

    return RasterSurface(
        elevation=elevation,
        transform=tuple(meta["transform"]),
        epsg=meta.get("epsg"),
        path=str(path),
    )


# --------------------------------------------------------------------------
# volume
# --------------------------------------------------------------------------


@dataclass
class VolumeResult:
    """Cut/fill volumes against one reference surface."""

    reference: str
    reference_elevation_m: float | None
    cut_volume_m3: float       # material above the reference
    fill_volume_m3: float      # void below the reference
    net_volume_m3: float
    area_m2: float
    cell_count: int
    mean_height_m: float
    max_height_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "reference_elevation_m": self.reference_elevation_m,
            "cut_volume_m3": round(self.cut_volume_m3, 3),
            "fill_volume_m3": round(self.fill_volume_m3, 3),
            "net_volume_m3": round(self.net_volume_m3, 3),
            "area_m2": round(self.area_m2, 3),
            "cell_count": self.cell_count,
            "mean_height_m": round(self.mean_height_m, 4),
            "max_height_m": round(self.max_height_m, 4),
        }


def _polygon_mask(surface: RasterSurface, polygon_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """Rasterise a polygon by even-odd crossing test.

    Implemented directly rather than via rasterio.features so volume measurement
    does not acquire a hard GDAL dependency on top of the read.
    """
    rows, cols = surface.elevation.shape
    row_index, col_index = np.mgrid[0:rows, 0:cols]
    x, y = surface.xy_of(row_index, col_index)

    vertices = [v for v in polygon_xy if len(v) >= 2]
    if len(vertices) >= 3 and np.allclose(vertices[0][:2], vertices[-1][:2]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return np.zeros_like(surface.elevation, dtype=bool)

    inside = np.zeros(x.shape, dtype=bool)
    count = len(vertices)
    for i in range(count):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % count][0]), float(vertices[(i + 1) % count][1])
        if y1 == y2:
            continue
        straddles = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= straddles & (x < crossing_x)
    return inside


def estimate_volume(
    dsm_path: str | Path,
    *,
    dtm_path: str | Path | None = None,
    polygon_xy: Sequence[Sequence[float]] | None = None,
    base_elevation_m: float | None = None,
) -> dict[str, Any]:
    """Compute stockpile / excavation volumes from the reconstruction rasters.

    Three references are reported, because a volume figure is meaningless without
    the datum it was measured against:

    ``dtm``
        Against the filtered bare-earth surface. The right answer for a pile sitting
        on uneven ground, and only available when a DTM was produced.
    ``plane``
        Against a fixed elevation, either supplied or taken as the boundary minimum.
    ``lowest_point``
        Against the minimum elevation in the region. Always an over-estimate; useful
        only as an upper bound.
    """
    surface = load_surface(dsm_path)
    region = surface.valid_mask
    if polygon_xy:
        region &= _polygon_mask(surface, polygon_xy)

    cell_count = int(region.sum())
    if cell_count == 0:
        return {
            "ok": False,
            "reason": "No valid DSM cells inside the requested region.",
            "dsm_path": str(dsm_path),
        }

    elevation = surface.elevation
    cell_area = surface.pixel_area_m2
    results: list[VolumeResult] = []

    def accumulate(reference_name: str, reference: np.ndarray, reference_elevation: float | None):
        difference = np.where(region, elevation - reference, np.nan)
        finite = np.isfinite(difference)
        if not finite.any():
            return
        positive = np.where(finite & (difference > 0), difference, 0.0)
        negative = np.where(finite & (difference < 0), -difference, 0.0)
        results.append(
            VolumeResult(
                reference=reference_name,
                reference_elevation_m=reference_elevation,
                cut_volume_m3=float(positive.sum() * cell_area),
                fill_volume_m3=float(negative.sum() * cell_area),
                net_volume_m3=float((positive.sum() - negative.sum()) * cell_area),
                area_m2=float(int(finite.sum()) * cell_area),
                cell_count=int(finite.sum()),
                mean_height_m=float(np.nanmean(difference[finite])),
                max_height_m=float(np.nanmax(difference[finite])),
            )
        )

    if dtm_path and Path(dtm_path).exists():
        ground = load_surface(dtm_path)
        if ground.elevation.shape == elevation.shape:
            accumulate("dtm", ground.elevation, None)
        else:
            results.append(
                VolumeResult(
                    reference="dtm_unavailable", reference_elevation_m=None,
                    cut_volume_m3=0.0, fill_volume_m3=0.0, net_volume_m3=0.0,
                    area_m2=0.0, cell_count=0, mean_height_m=0.0, max_height_m=0.0,
                )
            )

    region_values = elevation[region]
    minimum = float(np.nanmin(region_values))
    plane_elevation = float(base_elevation_m) if base_elevation_m is not None else minimum
    accumulate("plane", np.full_like(elevation, plane_elevation), plane_elevation)
    if base_elevation_m is not None:
        accumulate("lowest_point", np.full_like(elevation, minimum), minimum)

    preferred = next((r for r in results if r.reference == "dtm"), None)
    if preferred is None:
        preferred = next((r for r in results if r.reference == "plane"), None)

    return {
        "ok": True,
        "dsm_path": str(dsm_path),
        "dtm_path": str(dtm_path) if dtm_path else None,
        "crs_epsg": surface.epsg,
        "pixel_size_m": round(surface.pixel_size_m, 4),
        "cell_area_m2": round(cell_area, 6),
        "region_cells": cell_count,
        "region_area_m2": round(cell_count * cell_area, 3),
        "elevation_min_m": round(minimum, 3),
        "elevation_max_m": round(float(np.nanmax(region_values)), 3),
        "polygon_supplied": bool(polygon_xy),
        "references": [r.to_dict() for r in results],
        "preferred": preferred.to_dict() if preferred else None,
        "note": (
            "Volumes depend entirely on the reference surface. The 'dtm' figure is the "
            "defensible one; 'plane' and 'lowest_point' assume flat ground and will "
            "over-estimate on a slope."
        ),
    }


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


@dataclass
class MeasurementSet:
    """Derived measurements over a run's defect layer and terrain."""

    defect_measurements: list[dict[str, Any]] = field(default_factory=list)
    terrain: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_measurements": self.defect_measurements,
            "terrain": self.terrain,
            "totals": self.totals,
        }


def extract_measurements(
    *,
    defects_geojson: str | Path | None = None,
    dsm_path: str | Path | None = None,
    dtm_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarise the metric quantities a run produced.

    Reads the georeferenced defect layer written by `core.defect_projection` (whose
    areas are already in m^2 because they were measured on the reconstructed surface)
    and adds terrain statistics from the DSM/DTM pair.
    """
    measurements = MeasurementSet()

    if defects_geojson and Path(defects_geojson).exists():
        payload = json.loads(Path(defects_geojson).read_text(encoding="utf-8"))
        areas: list[float] = []
        lengths: list[float] = []
        by_type: dict[str, dict[str, float]] = {}

        for feature in payload.get("features", []) or []:
            properties = feature.get("properties", {}) or {}
            area = float(properties.get("area_m2", 0.0) or 0.0)
            length = float(properties.get("length_m", 0.0) or 0.0)
            defect_type = str(properties.get("defect_type", "unknown"))

            areas.append(area)
            lengths.append(length)
            bucket = by_type.setdefault(defect_type, {"count": 0, "area_m2": 0.0, "length_m": 0.0})
            bucket["count"] += 1
            bucket["area_m2"] += area
            bucket["length_m"] += length

            measurements.defect_measurements.append(
                {
                    "defect_id": properties.get("defect_id"),
                    "defect_type": defect_type,
                    "area_m2": round(area, 5),
                    "length_m": round(length, 4),
                    "width_m": round(float(properties.get("width_m", 0.0) or 0.0), 4),
                    "severity": properties.get("severity"),
                    "confidence": properties.get("confidence"),
                    "observation_count": properties.get("observation_count"),
                }
            )

        measurements.totals = {
            "defect_count": len(areas),
            "total_area_m2": round(float(sum(areas)), 4),
            "total_length_m": round(float(sum(lengths)), 3),
            "largest_area_m2": round(max(areas), 4) if areas else 0.0,
            "by_type": {
                name: {
                    "count": int(values["count"]),
                    "area_m2": round(values["area_m2"], 4),
                    "length_m": round(values["length_m"], 3),
                }
                for name, values in sorted(by_type.items())
            },
        }
    else:
        measurements.totals = {
            "defect_count": 0,
            "note": "No georeferenced defect layer was available for this run.",
        }

    if dsm_path and Path(dsm_path).exists():
        surface = load_surface(dsm_path)
        valid = surface.valid_mask
        values = surface.elevation[valid]
        terrain: dict[str, Any] = {
            "crs_epsg": surface.epsg,
            "pixel_size_m": round(surface.pixel_size_m, 4),
            "covered_area_m2": round(float(valid.sum()) * surface.pixel_area_m2, 2),
            "coverage_fraction": round(float(valid.mean()), 4),
        }
        if values.size:
            terrain.update(
                {
                    "elevation_min_m": round(float(values.min()), 3),
                    "elevation_max_m": round(float(values.max()), 3),
                    "elevation_mean_m": round(float(values.mean()), 3),
                    "relief_m": round(float(values.max() - values.min()), 3),
                }
            )

        if dtm_path and Path(dtm_path).exists():
            ground = load_surface(dtm_path)
            if ground.elevation.shape == surface.elevation.shape:
                height = surface.elevation - ground.elevation
                finite = np.isfinite(height)
                if finite.any():
                    above = height[finite]
                    terrain["canopy_or_structure_height"] = {
                        "mean_m": round(float(above.mean()), 3),
                        "max_m": round(float(above.max()), 3),
                        # Anything more than 0.5 m above bare earth is a structure or
                        # canopy rather than ground-filter noise.
                        "above_ground_area_m2": round(
                            float((above > 0.5).sum()) * surface.pixel_area_m2, 2
                        ),
                    }
        measurements.terrain = terrain
    else:
        measurements.terrain = {"note": "No DSM was available for this run."}

    return measurements.to_dict()
