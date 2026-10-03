"""Tilt and azimuth of planar surfaces (PV arrays, roof facets) measured from a DSM.

A plane ``z = a·x + b·y + c`` is fitted by least squares to the DSM cells of one
surface. From it:

* tilt (inclination) = atan(sqrt(a² + b²)), in degrees from horizontal;
* azimuth = the direction the surface faces (its downslope direction), in degrees
  clockwise from **true** north. The downslope vector is drawn in the DSM's projected
  CRS and both ends are converted to WGS84, so grid convergence is accounted for;
* fit RMSE (m) and the largest absolute residual, so a reader can see whether the cells
  really form a plane. A fit whose RMSE exceeds ``max_rmse_m`` is returned with
  ``accepted: False`` and the reason; it is never silently reported as a measurement.

Surfaces come either from operator outlines (one polygon per array or facet, in the
DSM's CRS or as lon/lat) or, with no outline, from RANSAC segmentation of the cells that
stand ``min_height_m`` above a DTM. Segmented facets are proposals: the report labels
them "segmented automatically" so an inspector can reject a facet that is not a roof.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class PlaneFitError(ValueError):
    """The surface cannot be measured (too few cells, no CRS, nothing above ground)."""


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Least-squares plane through points; returns slope terms, tilt and residuals."""
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    if x.size < 3:
        raise PlaneFitError("A plane needs at least three points.")
    x0, y0 = float(x.mean()), float(y.mean())
    design = np.column_stack([x - x0, y - y0, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ np.array([a, b, c])
    return {"a": float(a), "b": float(b), "c": float(c), "x0": x0, "y0": y0,
            "tilt_deg": math.degrees(math.atan(math.hypot(a, b))),
            "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_residual_m": float(np.max(np.abs(residual))), "points": int(x.size)}


def _true_azimuth(crs: Any, x0: float, y0: float, a: float, b: float) -> float | None:
    """Downslope azimuth from true north, via a short vector reprojected to WGS84."""
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return None
    from pyproj import Geod, Transformer

    norm = math.hypot(a, b)
    dx, dy = -a / norm * 10.0, -b / norm * 10.0
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = to_wgs.transform(x0, y0)
    lon1, lat1 = to_wgs.transform(x0 + dx, y0 + dy)
    azimuth, _, _ = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
    return float(azimuth % 360.0)


def compass(azimuth: float | None) -> str:
    if azimuth is None:
        return "flat"
    names = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
             "W", "WNW", "NW", "NNW")
    return names[int((azimuth % 360) / 22.5 + 0.5) % 16]


def _describe(fit: dict[str, float], crs: Any, cell_area: float, *, label: str,
              origin: str, max_rmse_m: float) -> dict[str, Any]:
    azimuth = _true_azimuth(crs, fit["x0"], fit["y0"], fit["a"], fit["b"]) \
        if fit["tilt_deg"] >= 0.5 else None
    accepted = fit["rmse_m"] <= max_rmse_m
    return {
        "label": label, "origin": origin,
        "tilt_deg": round(fit["tilt_deg"], 1),
        "azimuth_deg": round(azimuth, 1) if azimuth is not None else None,
        "facing": compass(azimuth) if azimuth is not None else "flat (tilt < 0.5°)",
        "rmse_m": round(fit["rmse_m"], 3), "max_abs_residual_m": round(fit["max_abs_residual_m"], 3),
        "cells": fit["points"], "area_m2": round(fit["points"] * cell_area, 1),
        "centre_xy": [round(fit["x0"], 2), round(fit["y0"], 2)],
        "accepted": accepted,
        "reason": "" if accepted else (f"fit RMSE {fit['rmse_m']:.2f} m exceeds "
                                       f"{max_rmse_m:g} m: the cells are not one plane"),
    }


def _cells(dsm_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, float]:
    import rasterio

    with rasterio.open(dsm_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise PlaneFitError(f"{dsm_path.name} has no projected CRS, so slopes in metres "
                                "cannot be measured.")
        band = src.read(1, masked=True).astype(float)
        transform, crs = src.transform, src.crs
    rows, cols = np.indices(band.shape)
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    cell_area = abs(transform.a * transform.e - transform.b * transform.d)
    return band, xs, ys, transform, crs, cell_area


def measure_outlined_surfaces(dsm_path: str | Path, outlines: Sequence[dict[str, Any]], *,
                              max_rmse_m: float = 0.15, min_cells: int = 6) -> dict[str, Any]:
    """One plane per operator outline: ``{"label", "polygon": [[x, y], ...], "crs"?}``.

    ``crs`` defaults to the DSM's; ``"EPSG:4326"`` takes lon/lat vertices.
    """
    from rasterio.features import geometry_mask
    from pyproj import Transformer

    dsm = Path(dsm_path)
    band, xs, ys, transform, crs, cell_area = _cells(dsm)
    out = []
    for index, outline in enumerate(outlines):
        ring = [tuple(map(float, p)) for p in outline["polygon"]]
        if outline.get("crs") and str(outline["crs"]) != str(crs):
            to_dsm = Transformer.from_crs(outline["crs"], crs, always_xy=True)
            ring = [to_dsm.transform(x, y) for x, y in ring]
        inside = ~geometry_mask([{"type": "Polygon", "coordinates": [ring + ring[:1]]}],
                                out_shape=band.shape, transform=transform)
        use = inside & ~np.ma.getmaskarray(band)
        label = str(outline.get("label") or f"surface {index + 1}")
        if int(use.sum()) < min_cells:
            out.append({"label": label, "origin": "operator outline", "accepted": False,
                        "reason": f"only {int(use.sum())} DSM cells inside the outline "
                                  f"(need {min_cells})"})
            continue
        fit = fit_plane(xs[use], ys[use], band.data[use])
        out.append(_describe(fit, crs, cell_area, label=label, origin="operator outline",
                             max_rmse_m=max_rmse_m))
    return {"dsm": str(dsm), "crs": str(crs), "cell_size_m": round(math.sqrt(cell_area), 3),
            "surfaces": out, "method": _method(max_rmse_m)}


def segment_roof_facets(dsm_path: str | Path, dtm_path: str | Path, *,
                        min_height_m: float = 2.5, inlier_m: float = 0.3,
                        max_rmse_m: float = 0.3, min_cells: int = 12, max_facets: int = 8,
                        seed: int = 0) -> dict[str, Any]:
    """RANSAC planes in the cells standing ``min_height_m`` above the DTM (proposals)."""
    import rasterio

    dsm, dtm = Path(dsm_path), Path(dtm_path)
    band, xs, ys, transform, crs, cell_area = _cells(dsm)
    with rasterio.open(dtm) as src:
        ground = src.read(1, masked=True).astype(float)
        if ground.shape != band.shape or src.transform != transform:
            raise PlaneFitError("The DTM is not on the DSM's grid.")
    valid = ~np.ma.getmaskarray(band) & ~np.ma.getmaskarray(ground)
    above = valid & ((band.data - ground.data) >= min_height_m)
    remaining = np.flatnonzero(above.ravel())
    X, Y, Z = xs.ravel(), ys.ravel(), band.data.ravel()
    rng = np.random.default_rng(seed)
    facets: list[dict[str, Any]] = []
    while remaining.size >= min_cells and len(facets) < max_facets:
        best: np.ndarray | None = None
        for _ in range(300):
            pick = rng.choice(remaining, 3, replace=False)
            p = np.column_stack([X[pick], Y[pick], Z[pick]])
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(normal) < 1e-9 or abs(normal[2]) < 1e-6:
                continue
            normal /= np.linalg.norm(normal)
            dist = np.abs((np.column_stack([X[remaining], Y[remaining], Z[remaining]]) - p[0])
                          @ normal)
            inl = remaining[dist <= inlier_m]
            if best is None or inl.size > best.size:
                best = inl
        if best is None or best.size < min_cells:
            break
        # A facet is one contiguous patch: cells that merely share the plane elsewhere on
        # the site are left for later rounds instead of being averaged into it.
        from scipy import ndimage

        grid = np.zeros(band.shape, dtype=bool)
        grid.ravel()[best] = True
        labels, count = ndimage.label(grid)
        if count > 1:
            sizes = ndimage.sum(grid, labels, index=range(1, count + 1))
            keep = labels.ravel()[best] == (int(np.argmax(sizes)) + 1)
            patch = best[keep]
        else:
            patch = best
        if patch.size < min_cells:
            remaining = np.setdiff1d(remaining, best)
            continue
        best = patch
        fit = fit_plane(X[best], Y[best], Z[best])
        facets.append(_describe(fit, crs, cell_area, label=f"facet {len(facets) + 1}",
                                origin="segmented automatically (RANSAC)",
                                max_rmse_m=max_rmse_m))
        remaining = np.setdiff1d(remaining, best)
    if not facets:
        raise PlaneFitError(f"No planar facet of {min_cells}+ cells stands {min_height_m:g} m "
                            "above the DTM.")
    return {"dsm": str(dsm), "dtm": str(dtm), "crs": str(crs),
            "cell_size_m": round(math.sqrt(cell_area), 3), "surfaces": facets,
            "unassigned_cells": int(remaining.size), "method": _method(max_rmse_m) +
            f" Facets proposed by RANSAC (inlier band ±{inlier_m:g} m) among cells "
            f"≥ {min_height_m:g} m above the DTM."}


def _method(max_rmse_m: float) -> str:
    return ("Least-squares plane z = a·x + b·y + c through the DSM cells; tilt = "
            "atan(√(a²+b²)); azimuth = downslope direction from true north (vector "
            f"reprojected to WGS84); fits with RMSE > {max_rmse_m:g} m are rejected.")
