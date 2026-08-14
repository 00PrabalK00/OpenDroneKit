"""Real-file fixtures for the scoped India-pack verification suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin


EPSG = 32643
WEST = 500_000.0
NORTH = 2_000_000.0
TRANSFORM = from_origin(WEST, NORTH, 1.0, 1.0)


def write_raster(
    path: Path,
    array: np.ndarray,
    *,
    nodata: int | float | None = None,
    tags: dict[str, str] | None = None,
    epsg: int | None = EPSG,
) -> Path:
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "transform": TRANSFORM,
    }
    if epsg is not None:
        profile["crs"] = f"EPSG:{epsg}"
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as target:
        target.write(data)
        if tags:
            target.update_tags(**tags)
    return path


def polygon(col0: float, row0: float, col1: float, row1: float) -> dict:
    west = WEST + col0
    east = WEST + col1
    north = NORTH - row0
    south = NORTH - row1
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, north],
            [east, north],
            [east, south],
            [west, south],
            [west, north],
        ]],
    }


def point(col: float, row: float) -> dict:
    return {
        "type": "Point",
        "coordinates": [WEST + col, NORTH - row],
    }


def line(points: Sequence[tuple[float, float]]) -> dict:
    return {
        "type": "LineString",
        "coordinates": [[WEST + col, NORTH - row] for col, row in points],
    }


def write_geojson(path: Path, features: Sequence[dict], *, epsg: int = EPSG) -> Path:
    payload = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        },
        "features": list(features),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def feature(geometry: dict, **properties: Any) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties}
