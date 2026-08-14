"""Shared geospatial invariants for the India-first mission packs.

The helpers in this module are intentionally strict.  A raster without a CRS is
an image, not a survey measurement, and a GeoJSON file without an explicit CRS is
interpreted as WGS84 only because that is the GeoJSON standard.  Metric outputs
are emitted only from projected coordinate systems with a known linear unit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import geo
from .semantic_engine import CLASS_NODATA, SemanticSchema


class IndiaPackRefused(ValueError):
    """Raised when the supplied evidence cannot support the requested result."""


@dataclass(frozen=True)
class IndiaPackResult:
    """A small common contract for client-facing India-pack artifacts."""

    summary_path: str
    artifact_paths: tuple[str, ...]
    status: str = "complete"


@dataclass(frozen=True)
class RasterEvidence:
    path: str
    data: np.ndarray
    valid: np.ndarray
    crs: Any
    epsg: int
    transform: Any
    width: int
    height: int
    nodata: int | float | None
    tags: dict[str, str]
    linear_unit_to_m: float | None
    pixel_area_m2: float | None


def canonical_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.casefold())).strip("_")


def read_raster_evidence(
    path: str | Path,
    *,
    band: int = 1,
    require_projected: bool,
) -> RasterEvidence:
    """Read one real raster band and preserve the grid needed by downstream outputs."""

    geo.require("rasterio")
    import rasterio

    source_path = Path(path)
    if not source_path.is_file():
        raise IndiaPackRefused(f"Raster does not exist: {source_path}")
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise IndiaPackRefused(
                f"{source_path.name} has no CRS; it cannot support a geolocated result."
            )
        if require_projected and not source.crs.is_projected:
            raise IndiaPackRefused(
                f"{source_path.name} is not in a projected CRS; metric areas and lengths "
                "would be invalid. Reproject the source before analysis."
            )
        epsg = source.crs.to_epsg()
        if epsg is None:
            raise IndiaPackRefused(
                f"{source_path.name} has no EPSG identifier, so interoperable GIS output "
                "cannot be written."
            )
        if band < 1 or band > source.count:
            raise IndiaPackRefused(
                f"Band {band} is unavailable in {source_path.name} ({source.count} bands)."
            )
        data = source.read(band)
        valid = source.read_masks(band) > 0
        determinant = abs(
            float(
                source.transform.a * source.transform.e
                - source.transform.b * source.transform.d
            )
        )
        if determinant <= 0:
            raise IndiaPackRefused(f"{source_path.name} has a degenerate affine transform.")
        factor: float | None = None
        area: float | None = None
        if source.crs.is_projected:
            try:
                factor = float(source.crs.linear_units_factor[1])
            except Exception as exc:  # pragma: no cover - unusual CRS backend failure
                raise IndiaPackRefused(
                    f"{source_path.name} does not declare a usable linear unit."
                ) from exc
            if not math.isfinite(factor) or factor <= 0:
                raise IndiaPackRefused(
                    f"{source_path.name} does not declare a usable linear unit."
                )
            area = determinant * factor * factor
        return RasterEvidence(
            path=str(source_path),
            data=np.asarray(data),
            valid=np.asarray(valid, dtype=bool),
            crs=source.crs,
            epsg=int(epsg),
            transform=source.transform,
            width=int(source.width),
            height=int(source.height),
            nodata=source.nodata,
            tags={str(k): str(v) for k, v in source.tags().items()},
            linear_unit_to_m=factor,
            pixel_area_m2=area,
        )


def assert_aligned(first: RasterEvidence, second: RasterEvidence) -> None:
    """Refuse silent resampling between measurement products."""

    if first.crs != second.crs:
        raise IndiaPackRefused("Raster CRS values do not match.")
    if (first.height, first.width) != (second.height, second.width):
        raise IndiaPackRefused("Raster dimensions do not match.")
    if not np.allclose(
        tuple(first.transform)[:6], tuple(second.transform)[:6], rtol=0.0, atol=1e-9
    ):
        raise IndiaPackRefused("Raster transforms do not match; explicit registration is required.")


def write_aligned_raster(
    path: str | Path,
    array: np.ndarray,
    reference: RasterEvidence,
    *,
    dtype: str,
    nodata: int | float,
    tags: Mapping[str, Any] | None = None,
    categorical: bool,
) -> str:
    """Write an array on exactly the source survey grid."""

    geo.require("rasterio")
    import rasterio

    data = np.asarray(array)
    if data.shape != (reference.height, reference.width):
        raise ValueError("Output raster shape must match its reference grid.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "driver": "COG",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": dtype,
        "crs": reference.crs,
        "transform": reference.transform,
        "nodata": nodata,
        "compress": "DEFLATE",
    }
    try:
        with rasterio.open(output, "w", **profile) as target:
            target.write(data.astype(dtype, copy=False), 1)
            if tags:
                target.update_tags(**{str(k): str(v) for k, v in tags.items()})
    except Exception:
        output.unlink(missing_ok=True)
        profile.update(
            {"driver": "GTiff", "tiled": True, "blockxsize": 256, "blockysize": 256}
        )
        with rasterio.open(output, "w", **profile) as target:
            target.write(data.astype(dtype, copy=False), 1)
            if tags:
                target.update_tags(**{str(k): str(v) for k, v in tags.items()})
            factors = [n for n in (2, 4, 8, 16) if min(data.shape) // n >= 1]
            if factors:
                method = (
                    rasterio.enums.Resampling.nearest
                    if categorical
                    else rasterio.enums.Resampling.average
                )
                target.build_overviews(factors, method)
    return str(output)


def load_class_names(
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
) -> dict[int, str]:
    """Load class ids from a schema object, mapping, or semantic manifest JSON."""

    if isinstance(schema, SemanticSchema):
        return {int(item.id): canonical_name(item.name) for item in schema.classes}
    if isinstance(schema, Mapping):
        return {int(key): canonical_name(str(value)) for key, value in schema.items()}
    path = Path(schema)
    if not path.is_file():
        raise IndiaPackRefused(f"Semantic schema or manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_payload = payload.get("schema", payload)
    classes = schema_payload.get("classes") if isinstance(schema_payload, dict) else None
    if not isinstance(classes, list):
        raise IndiaPackRefused(f"{path.name} does not contain a semantic class schema.")
    return {
        int(item["id"]): canonical_name(str(item["name"]))
        for item in classes
        if isinstance(item, dict) and "id" in item and "name" in item
    }


def ids_for_names(
    class_names: Mapping[int, str], names: Iterable[str]
) -> tuple[set[int], set[str]]:
    requested = {canonical_name(name) for name in names}
    ids = {int(class_id) for class_id, name in class_names.items() if name in requested}
    found = {class_names[class_id] for class_id in ids}
    return ids, requested - found


def _epsg_from_geojson(payload: Mapping[str, Any]) -> int:
    properties = payload.get("properties", {})
    if isinstance(properties, Mapping) and properties.get("epsg") is not None:
        return int(properties["epsg"])
    crs = payload.get("crs")
    if isinstance(crs, Mapping):
        crs_properties = crs.get("properties", {})
        name = str(crs_properties.get("name", "")) if isinstance(crs_properties, Mapping) else ""
        match = re.search(r"EPSG(?::|::)(\d+)", name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    # RFC 7946 fixes unqualified GeoJSON coordinates to WGS84 longitude/latitude.
    return 4326


def read_geojson_features(
    path: str | Path,
    *,
    target_crs: Any | None = None,
    allowed_geometry_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Read real GeoJSON features and optionally transform them to a target CRS."""

    geo.require("rasterio")
    from rasterio.crs import CRS
    from rasterio.warp import transform_geom

    source_path = Path(path)
    if not source_path.is_file():
        raise IndiaPackRefused(f"GeoJSON does not exist: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise IndiaPackRefused(f"{source_path.name} is not a GeoJSON FeatureCollection.")
    source_epsg = _epsg_from_geojson(payload)
    source_crs = CRS.from_epsg(source_epsg)
    destination = CRS.from_user_input(target_crs) if target_crs is not None else source_crs
    features: list[dict[str, Any]] = []
    for feature in payload["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise IndiaPackRefused(f"{source_path.name} contains a malformed feature.")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type"):
            raise IndiaPackRefused(f"{source_path.name} contains a feature without geometry.")
        if allowed_geometry_types and geometry["type"] not in allowed_geometry_types:
            raise IndiaPackRefused(
                f"{source_path.name} contains {geometry['type']}; expected one of "
                f"{sorted(allowed_geometry_types)}."
            )
        converted = dict(feature)
        if source_crs != destination:
            converted["geometry"] = transform_geom(source_crs, destination, geometry, precision=9)
        converted["properties"] = dict(feature.get("properties") or {})
        features.append(converted)
    destination_epsg = destination.to_epsg()
    if destination_epsg is None:
        raise IndiaPackRefused("The target vector CRS has no EPSG identifier.")
    return features, int(destination_epsg)


def rasterize_features(
    features: Sequence[Mapping[str, Any]],
    reference: RasterEvidence,
    *,
    all_touched: bool = False,
) -> np.ndarray:
    geo.require("rasterio")
    from rasterio.features import rasterize

    if not features:
        return np.zeros((reference.height, reference.width), dtype=bool)
    shapes = [(dict(feature["geometry"]), 1) for feature in features]
    return rasterize(
        shapes,
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def geometry_area_units2(geometry: Mapping[str, Any]) -> float:
    def ring_area(ring: Sequence[Sequence[float]]) -> float:
        if len(ring) < 3:
            return 0.0
        return abs(
            sum(
                float(ring[index][0]) * float(ring[(index + 1) % len(ring)][1])
                - float(ring[(index + 1) % len(ring)][0]) * float(ring[index][1])
                for index in range(len(ring))
            )
        ) / 2.0

    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return max(0.0, ring_area(coordinates[0]) - sum(ring_area(r) for r in coordinates[1:]))
    if geometry.get("type") == "MultiPolygon":
        return sum(
            max(0.0, ring_area(poly[0]) - sum(ring_area(r) for r in poly[1:]))
            for poly in coordinates
        )
    return 0.0


def polygonize_mask(
    mask: np.ndarray,
    reference: RasterEvidence,
    *,
    properties: Mapping[str, Any],
    min_area_m2: float = 0.0,
) -> list[dict[str, Any]]:
    """Polygonize a measured mask, including exact projected area metadata."""

    if reference.linear_unit_to_m is None:
        raise IndiaPackRefused("Polygon area requires a projected raster source.")
    geo.require("rasterio")
    from rasterio.features import shapes

    selected = np.asarray(mask, dtype=bool) & reference.valid
    features: list[dict[str, Any]] = []
    for geometry, value in shapes(
        selected.astype(np.uint8),
        mask=selected,
        transform=reference.transform,
        connectivity=8,
    ):
        if int(value) != 1:
            continue
        area_m2 = geometry_area_units2(geometry) * reference.linear_unit_to_m**2
        if area_m2 + 1e-12 < min_area_m2:
            continue
        feature_properties = dict(properties)
        feature_properties["area_m2"] = round(float(area_m2), 6)
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": feature_properties}
        )
    return features


def geometry_length_units(geometry: Mapping[str, Any]) -> float:
    def line_length(line: Sequence[Sequence[float]]) -> float:
        return sum(
            math.hypot(
                float(line[index][0]) - float(line[index - 1][0]),
                float(line[index][1]) - float(line[index - 1][1]),
            )
            for index in range(1, len(line))
        )

    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if kind == "LineString":
        return line_length(coordinates)
    if kind == "MultiLineString":
        return sum(line_length(line) for line in coordinates)
    return 0.0


def feature_anchor_xy(feature: Mapping[str, Any]) -> tuple[float, float]:
    """Return a defensible point used only for map association, never geometry metrics."""

    geometry = feature.get("geometry", {})
    if geometry.get("type") == "Point":
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) >= 2:
            return float(coordinates[0]), float(coordinates[1])

    points: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(isinstance(item, (int, float)) for item in value[:2])
        ):
            points.append((float(value[0]), float(value[1])))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(geometry.get("coordinates", []))
    if not points:
        raise IndiaPackRefused("A feature has no usable coordinate.")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def write_summary(path: str | Path, payload: Mapping[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return str(output)


def semantic_valid_mask(evidence: RasterEvidence) -> np.ndarray:
    return evidence.valid & (evidence.data.astype(np.int64, copy=False) != CLASS_NODATA)
