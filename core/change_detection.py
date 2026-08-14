"""Comparing two surveys of the same asset.

Repeat inspection is the point of a digital twin: the value is not one model but the
difference between this one and the last. Three questions get answered here.

*What moved.* An elevation difference raster between two DSMs, plus the volume added
and removed.

*What is new, gone, or changed.* Defects are matched between surveys by position and
type. Matching is deliberately conservative: a defect that moved further than the
match radius is reported as one resolved and one new, not as the same defect having
grown, because claiming continuity that was not established would fabricate a history.

*What cannot be compared.* Two surveys in different coordinate systems, or at
different resolutions, are not silently resampled into agreement. That is reported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import geo
from .dsm_analysis import RasterSurface, _polygon_mask, load_surface

# Beyond this, two findings are treated as different defects rather than one that
# moved. Chosen to exceed typical georeferencing error (~1 m on the verified
# Aukerman run) without spanning separate features.
DEFAULT_MATCH_RADIUS_M = 2.0

# Below this, an area difference is noise from segmentation rather than real growth.
AREA_CHANGE_TOLERANCE = 0.10


@dataclass
class SurfaceChange:
    """The difference between two elevation surfaces."""

    added_volume_m3: float
    removed_volume_m3: float
    net_volume_m3: float
    max_rise_m: float
    max_fall_m: float
    changed_area_m2: float
    compared_cells: int
    crs_epsg: int | None
    pixel_size_m: float
    difference_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_volume_m3": round(self.added_volume_m3, 3),
            "removed_volume_m3": round(self.removed_volume_m3, 3),
            "net_volume_m3": round(self.net_volume_m3, 3),
            "max_rise_m": round(self.max_rise_m, 3),
            "max_fall_m": round(self.max_fall_m, 3),
            "changed_area_m2": round(self.changed_area_m2, 3),
            "compared_cells": self.compared_cells,
            "crs_epsg": self.crs_epsg,
            "pixel_size_m": round(self.pixel_size_m, 4),
            "difference_path": self.difference_path,
        }


class IncomparableSurveys(ValueError):
    """Raised when two surveys cannot be compared without inventing agreement."""


def _require_comparable(earlier: RasterSurface, later: RasterSurface) -> None:
    if earlier.epsg != later.epsg:
        raise IncomparableSurveys(
            f"Surveys are in different coordinate systems (EPSG:{earlier.epsg} and "
            f"EPSG:{later.epsg}). Reproject one before comparing; silently resampling "
            "would produce a difference map that looks valid and is not."
        )
    if earlier.elevation.shape != later.elevation.shape:
        raise IncomparableSurveys(
            f"Surveys have different raster shapes ({earlier.elevation.shape} and "
            f"{later.elevation.shape}). They must be gridded identically to be differenced."
        )
    if not math.isclose(earlier.pixel_size_m, later.pixel_size_m, rel_tol=1e-6):
        raise IncomparableSurveys(
            f"Surveys have different resolutions ({earlier.pixel_size_m} m and "
            f"{later.pixel_size_m} m per pixel)."
        )
    if not np.allclose(
        np.asarray(earlier.transform[:6], dtype=float),
        np.asarray(later.transform[:6], dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise IncomparableSurveys(
            "Surveys use different grid origins or orientations. Matching CRS, shape, "
            "and resolution are not enough when the cells cover different ground."
        )


def _write_difference_raster(
    reference_path: str | Path,
    output_path: str | Path,
    difference: np.ndarray,
) -> str:
    """Write a single-band float raster preserving the later DSM's exact grid."""

    geo.require("rasterio")
    import rasterio

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=float("nan"),
        compress="DEFLATE",
    )
    with rasterio.open(out, "w", **profile) as target:
        target.write(np.asarray(difference, dtype=np.float32), 1)
    return str(out)


def compare_surfaces(
    earlier_dsm: str | Path,
    later_dsm: str | Path,
    *,
    output_path: str | Path | None = None,
    change_threshold_m: float = 0.05,
    roi_polygon_xy: Sequence[Sequence[float]] | None = None,
) -> SurfaceChange:
    """Difference two DSMs, reporting volume added and removed.

    `change_threshold_m` suppresses reconstruction noise from the changed-area figure;
    volumes are computed over every finite cell, so a genuine broad shallow change is
    not discarded.
    """
    earlier = load_surface(earlier_dsm)
    later = load_surface(later_dsm)
    _require_comparable(earlier, later)

    difference = later.elevation - earlier.elevation
    valid = np.isfinite(difference)
    if roi_polygon_xy is not None:
        valid &= _polygon_mask(later, roi_polygon_xy)
    if not valid.any():
        if roi_polygon_xy is not None:
            raise IncomparableSurveys(
                "The two surveys share no comparable cell inside the selected ROI."
            )
        raise IncomparableSurveys("The two surveys share no cell with valid elevation data.")

    cell_area = later.pixel_area_m2
    rise = np.where(valid & (difference > 0), difference, 0.0)
    fall = np.where(valid & (difference < 0), -difference, 0.0)
    changed = valid & (np.abs(difference) >= change_threshold_m)

    written = ""
    if output_path is not None:
        # NaN where a cell could not be compared, so the raster never implies
        # "no change" where the truth is "no data".
        raster = np.where(valid, difference, np.nan).astype(np.float32)
        written = _write_difference_raster(later.path, output_path, raster)

    return SurfaceChange(
        added_volume_m3=float(rise.sum() * cell_area),
        removed_volume_m3=float(fall.sum() * cell_area),
        net_volume_m3=float((rise.sum() - fall.sum()) * cell_area),
        max_rise_m=max(0.0, float(np.nanmax(np.where(valid, difference, np.nan)))),
        max_fall_m=max(0.0, float(-np.nanmin(np.where(valid, difference, np.nan)))),
        changed_area_m2=float(int(changed.sum()) * cell_area),
        compared_cells=int(valid.sum()),
        crs_epsg=later.epsg,
        pixel_size_m=later.pixel_size_m,
        difference_path=written,
    )


# ---------------------------------------------------------------------------
# defect comparison
# ---------------------------------------------------------------------------


@dataclass
class DefectChange:
    """How the defect population changed between two surveys."""

    new: list[dict[str, Any]] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    grown: list[dict[str, Any]] = field(default_factory=list)
    shrunk: list[dict[str, Any]] = field(default_factory=list)
    unchanged: list[dict[str, Any]] = field(default_factory=list)
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M

    def to_dict(self) -> dict[str, Any]:
        return {
            "new": self.new, "resolved": self.resolved, "grown": self.grown,
            "shrunk": self.shrunk, "unchanged": self.unchanged,
            "counts": {
                "new": len(self.new), "resolved": len(self.resolved),
                "grown": len(self.grown), "shrunk": len(self.shrunk),
                "unchanged": len(self.unchanged),
            },
            "match_radius_m": self.match_radius_m,
            "method": (
                "Defects are matched by type within the match radius. A defect that "
                "moved further than the radius is reported as one resolved and one new, "
                "because continuity that was not established must not be asserted."
            ),
        }


def _load_features(source: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    """Read a defect layer into flat records with position and extent."""
    if isinstance(source, dict):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    for feature in payload.get("features", []) or []:
        properties = dict(feature.get("properties", {}) or {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")

        longitude = properties.get("longitude")
        latitude = properties.get("latitude")
        if geometry.get("type") == "Point" and isinstance(coordinates, list):
            longitude, latitude = coordinates[0], coordinates[1]
        elif geometry.get("type") == "Polygon" and coordinates:
            ring = coordinates[0]
            if ring:
                longitude = sum(point[0] for point in ring) / len(ring)
                latitude = sum(point[1] for point in ring) / len(ring)

        if longitude is None or latitude is None:
            continue
        records.append({
            "defect_id": properties.get("defect_id"),
            "defect_type": str(properties.get("defect_type", "unknown")).lower(),
            "longitude": float(longitude),
            "latitude": float(latitude),
            "area_m2": float(properties.get("area_m2") or 0.0),
            "length_m": float(properties.get("length_m") or 0.0),
            "severity": properties.get("severity"),
        })
    return records


def compare_defects(
    earlier: str | Path | dict[str, Any],
    later: str | Path | dict[str, Any],
    *,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    area_tolerance: float = AREA_CHANGE_TOLERANCE,
) -> DefectChange:
    """Match defects between two surveys and classify what changed.

    Matching is greedy nearest-first within the radius, and only between defects of
    the same type: a crack becoming a spall is two findings, not one that changed.
    """
    before = _load_features(earlier)
    after = _load_features(later)
    change = DefectChange(match_radius_m=match_radius_m)

    candidates: list[tuple[float, int, int]] = []
    for before_index, old in enumerate(before):
        for after_index, new in enumerate(after):
            if old["defect_type"] != new["defect_type"]:
                continue
            distance = geo.haversine_m(
                old["latitude"], old["longitude"], new["latitude"], new["longitude"]
            )
            if distance <= match_radius_m:
                candidates.append((distance, before_index, after_index))

    candidates.sort()
    matched_before: set[int] = set()
    matched_after: set[int] = set()

    for distance, before_index, after_index in candidates:
        if before_index in matched_before or after_index in matched_after:
            continue
        matched_before.add(before_index)
        matched_after.add(after_index)

        old, new = before[before_index], after[after_index]
        record = {
            "defect_type": old["defect_type"],
            "earlier_id": old["defect_id"], "later_id": new["defect_id"],
            "earlier_area_m2": round(old["area_m2"], 5),
            "later_area_m2": round(new["area_m2"], 5),
            "area_delta_m2": round(new["area_m2"] - old["area_m2"], 5),
            "moved_m": round(distance, 3),
            "longitude": new["longitude"], "latitude": new["latitude"],
        }

        if old["area_m2"] <= 0.0 and new["area_m2"] <= 0.0:
            # Neither was measured, so growth cannot be claimed either way.
            record["note"] = "Neither survey measured an extent; change is undetermined."
            change.unchanged.append(record)
            continue

        relative = (new["area_m2"] - old["area_m2"]) / max(old["area_m2"], 1e-9)
        if relative > area_tolerance:
            change.grown.append(record)
        elif relative < -area_tolerance:
            change.shrunk.append(record)
        else:
            change.unchanged.append(record)

    change.resolved = [
        {**old, "note": "Present in the earlier survey, absent in the later one."}
        for index, old in enumerate(before) if index not in matched_before
    ]
    change.new = [
        {**new, "note": "Absent in the earlier survey, present in the later one."}
        for index, new in enumerate(after) if index not in matched_after
    ]
    return change


def compare_surveys(
    *,
    earlier_dsm: str | Path | None = None,
    later_dsm: str | Path | None = None,
    earlier_defects: str | Path | None = None,
    later_defects: str | Path | None = None,
    output_path: str | Path | None = None,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
) -> dict[str, Any]:
    """Compare two surveys, reporting what could not be compared as well as what could."""
    result: dict[str, Any] = {"surface": None, "defects": None, "warnings": []}

    if earlier_dsm and later_dsm:
        try:
            result["surface"] = compare_surfaces(
                earlier_dsm, later_dsm, output_path=output_path
            ).to_dict()
        except (IncomparableSurveys, Exception) as exc:  # noqa: BLE001
            result["warnings"].append(f"Surface comparison unavailable: {exc}")
    else:
        result["warnings"].append(
            "No elevation surfaces supplied, so no volume or geometry change was computed."
        )

    if earlier_defects and later_defects:
        result["defects"] = compare_defects(
            earlier_defects, later_defects, match_radius_m=match_radius_m
        ).to_dict()
    else:
        result["warnings"].append(
            "No defect layers supplied, so no defect progression was computed."
        )

    return result
