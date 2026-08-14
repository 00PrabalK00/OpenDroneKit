"""Mapped road, corridor-asset and solar-module intelligence packages."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import geo
from .india_geospatial import (
    IndiaPackRefused,
    IndiaPackResult,
    canonical_name,
    feature_anchor_xy,
    geometry_length_units,
    ids_for_names,
    load_class_names,
    polygonize_mask,
    rasterize_features,
    read_geojson_features,
    read_raster_evidence,
    semantic_valid_mask,
    write_aligned_raster,
    write_summary,
)
from .semantic_engine import SemanticSchema


ROAD_ALIASES = ("road", "paved_road", "unpaved_road", "road_surface")
ROAD_DEFECTS = {"pothole", "crack", "waterlogging", "debris"}
SEVERITIES = {"minor", "moderate", "severe", "critical"}


def _validate_finding(feature: Mapping[str, Any], *, class_key: str, allowed: set[str]) -> dict:
    properties = dict(feature.get("properties") or {})
    finding_class = canonical_name(str(properties.get(class_key) or ""))
    if finding_class not in allowed:
        raise IndiaPackRefused(
            f"Finding class {finding_class!r} is not in the validated pack schema {sorted(allowed)}."
        )
    evidence_source = canonical_name(str(properties.get("evidence_source") or ""))
    if evidence_source not in {"validated_model", "manual_review"}:
        raise IndiaPackRefused(
            "Each finding must state evidence_source=validated_model or manual_review."
        )
    if evidence_source == "validated_model":
        if not str(properties.get("model_key") or "").strip() or not str(
            properties.get("model_version") or ""
        ).strip():
            raise IndiaPackRefused(
                "A validated-model finding must include model_key and model_version."
            )
        confidence = properties.get("confidence")
        if confidence is None or not 0 <= float(confidence) <= 1:
            raise IndiaPackRefused("A validated-model finding needs confidence in [0, 1].")
    properties[class_key] = finding_class
    properties["evidence_source"] = evidence_source
    return properties


def _road_edge_features(road_polygons: Sequence[dict]) -> list[dict]:
    edges: list[dict] = []
    for index, feature in enumerate(road_polygons, start=1):
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon_index, polygon in enumerate(polygons, start=1):
            for ring_index, ring in enumerate(polygon, start=1):
                edges.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": ring},
                        "properties": {
                            "edge_id": f"road-{index}-{polygon_index}-{ring_index}",
                            "edge_kind": "outer" if ring_index == 1 else "inner",
                            "source": "semantic_road_polygon_boundary",
                        },
                    }
                )
    return edges


def create_road_condition_package(
    semantic_class_raster: str | Path,
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
    road_centerline_geojson: str | Path,
    defect_findings_geojson: str | Path,
    output_dir: str | Path,
    *,
    min_road_polygon_area_m2: float = 1.0,
) -> IndiaPackResult:
    """Combine road semantics, explicit surveyed centreline and validated findings."""

    road = read_raster_evidence(
        semantic_class_raster, band=1, require_projected=True
    )
    names = load_class_names(schema)
    road_ids, _ = ids_for_names(names, ROAD_ALIASES)
    if not road_ids:
        raise IndiaPackRefused("The semantic schema has no road-surface class.")
    valid = semantic_valid_mask(road)
    road_mask = valid & np.isin(road.data, list(road_ids))
    if not np.any(road_mask):
        raise IndiaPackRefused("No road pixels are present in the semantic class raster.")
    centreline, _ = read_geojson_features(
        road_centerline_geojson,
        target_crs=road.crs,
        allowed_geometry_types={"LineString", "MultiLineString"},
    )
    if not centreline:
        raise IndiaPackRefused("A measured surveyed centreline is required for road distance.")
    findings, _ = read_geojson_features(
        defect_findings_geojson,
        target_crs=road.crs,
        allowed_geometry_types={"Point", "Polygon", "MultiPolygon"},
    )

    road_polygons = polygonize_mask(
        road_mask,
        road,
        properties={"layer": "road_surface", "source": "semantic_class_raster"},
        min_area_m2=min_road_polygon_area_m2,
    )
    edges = _road_edge_features(road_polygons)
    normalized_findings: list[dict] = []
    from rasterio.transform import rowcol

    for finding in findings:
        properties = _validate_finding(
            finding, class_key="defect_type", allowed=ROAD_DEFECTS
        )
        severity = canonical_name(str(properties.get("severity") or ""))
        if severity not in SEVERITIES:
            raise IndiaPackRefused(
                f"Road finding severity must be one of {sorted(SEVERITIES)}."
            )
        x, y = feature_anchor_xy(finding)
        row, column = rowcol(road.transform, x, y)
        on_road = (
            0 <= row < road.height
            and 0 <= column < road.width
            and bool(road_mask[row, column])
        )
        properties["severity"] = severity
        properties["on_semantic_road_surface"] = on_road
        properties["requires_review"] = not on_road
        normalized_findings.append({**finding, "properties": properties})

    factor = float(road.linear_unit_to_m)
    surveyed_distance_m = sum(
        geometry_length_units(feature["geometry"]) * factor for feature in centreline
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    roads_path = geo.write_geojson(
        output / "road_surface_and_edges.geojson",
        [*road_polygons, *edges],
        epsg=road.epsg,
        properties={"source_raster": road.path, "units": {"area": "m2"}},
    )
    centerline_path = geo.write_geojson(
        output / "surveyed_road_centerline.geojson",
        centreline,
        epsg=road.epsg,
        properties={
            "source": str(road_centerline_geojson),
            "surveyed_distance_m": round(surveyed_distance_m, 6),
        },
    )
    findings_path = geo.write_geojson(
        output / "road_findings.geojson",
        normalized_findings,
        epsg=road.epsg,
        properties={"allowed_defect_classes": sorted(ROAD_DEFECTS)},
    )
    by_type = Counter(item["properties"]["defect_type"] for item in normalized_findings)
    by_severity = Counter(item["properties"]["severity"] for item in normalized_findings)
    summary_path = write_summary(
        output / "road_condition.json",
        {
            "status": "complete",
            "analysis": "mapped_road_condition",
            "source": {
                "semantic_class_raster": road.path,
                "centerline": str(road_centerline_geojson),
                "defect_findings": str(defect_findings_geojson),
                "epsg": road.epsg,
            },
            "units": {"road_area": "m2", "surveyed_distance": "m"},
            "road_surface_area_m2": round(
                float(road_mask.sum()) * float(road.pixel_area_m2), 6
            ),
            "surveyed_distance_m": round(surveyed_distance_m, 6),
            "finding_count": len(normalized_findings),
            "findings_by_type": dict(sorted(by_type.items())),
            "findings_by_severity": dict(sorted(by_severity.items())),
            "off_road_review_count": sum(
                not item["properties"]["on_semantic_road_surface"]
                for item in normalized_findings
            ),
            "artifacts": {
                "road_surface_and_edges": str(roads_path),
                "surveyed_centerline": str(centerline_path),
                "findings": str(findings_path),
            },
        },
    )
    return IndiaPackResult(
        summary_path, (str(roads_path), str(centerline_path), str(findings_path))
    )


POWER_CLASSES = {
    "tower",
    "pole",
    "crossarm",
    "insulator",
    "conductor",
    "transformer",
    "vegetation",
}
RAIL_CLASSES = {
    "track",
    "rail",
    "bridge",
    "overhead_equipment",
    "signal",
    "obstacle",
    "vegetation",
}


def _create_corridor_package(
    asset_kind: str,
    detections_geojson: str | Path,
    corridor_centerline_geojson: str | Path,
    output_dir: str | Path,
    *,
    capture_geometry: str,
    validation_scope: str,
    allowed_classes: set[str],
    allowed_capture_geometries: set[str],
) -> IndiaPackResult:
    if canonical_name(capture_geometry) not in allowed_capture_geometries:
        raise IndiaPackRefused(
            f"{asset_kind.title()} component inspection requires capture geometry in "
            f"{sorted(allowed_capture_geometries)}; received {capture_geometry!r}."
        )
    if not validation_scope.strip():
        raise IndiaPackRefused("A detection validation scope is required.")
    centerline, epsg = read_geojson_features(
        corridor_centerline_geojson,
        allowed_geometry_types={"LineString", "MultiLineString"},
    )
    if not centerline:
        raise IndiaPackRefused("A corridor centreline is required.")
    from rasterio.crs import CRS

    target_crs = CRS.from_epsg(epsg)
    if not target_crs.is_projected:
        raise IndiaPackRefused(
            "Corridor distance requires a projected GeoJSON CRS; WGS84 degrees are not metres."
        )
    try:
        factor = float(target_crs.linear_units_factor[1])
    except Exception as exc:  # pragma: no cover
        raise IndiaPackRefused("Corridor CRS has no usable linear unit.") from exc
    detections, _ = read_geojson_features(
        detections_geojson,
        target_crs=target_crs,
        allowed_geometry_types={"Point", "Polygon", "MultiPolygon", "LineString"},
    )
    normalized: list[dict] = []
    for feature in detections:
        properties = _validate_finding(
            feature, class_key="asset_class", allowed=allowed_classes
        )
        properties["validation_scope"] = validation_scope
        finding = canonical_name(str(properties.get("finding") or "normal"))
        finding_validated = bool(properties.get("finding_validated", finding == "normal"))
        properties["finding"] = finding
        properties["finding_status"] = (
            "validated" if finding_validated else "review_candidate"
        )
        normalized.append({**feature, "properties": properties})

    distance_m = sum(
        geometry_length_units(feature["geometry"]) * factor for feature in centerline
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    centerline_path = geo.write_geojson(
        output / f"{asset_kind}_corridor.geojson",
        centerline,
        epsg=epsg,
        properties={"surveyed_distance_m": round(distance_m, 6)},
    )
    findings_path = geo.write_geojson(
        output / f"{asset_kind}_assets.geojson",
        normalized,
        epsg=epsg,
        properties={
            "capture_geometry": capture_geometry,
            "validation_scope": validation_scope,
        },
    )
    by_class = Counter(item["properties"]["asset_class"] for item in normalized)
    by_finding = Counter(item["properties"]["finding"] for item in normalized)
    summary_path = write_summary(
        output / f"{asset_kind}_inspection.json",
        {
            "status": "complete",
            "analysis": f"mapped_{asset_kind}_corridor_inspection",
            "source": {
                "detections": str(detections_geojson),
                "centerline": str(corridor_centerline_geojson),
                "epsg": epsg,
            },
            "capture_geometry": capture_geometry,
            "validation_scope": validation_scope,
            "surveyed_distance_m": round(distance_m, 6),
            "asset_count": len(normalized),
            "assets_by_class": dict(sorted(by_class.items())),
            "findings_by_type": dict(sorted(by_finding.items())),
            "review_candidate_count": sum(
                item["properties"]["finding_status"] == "review_candidate"
                for item in normalized
            ),
            "units": {"surveyed_distance": "m"},
            "artifacts": {
                "corridor": str(centerline_path),
                "assets_and_findings": str(findings_path),
            },
        },
    )
    return IndiaPackResult(summary_path, (str(centerline_path), str(findings_path)))


def create_power_inspection_package(
    detections_geojson: str | Path,
    corridor_centerline_geojson: str | Path,
    output_dir: str | Path,
    *,
    capture_geometry: str,
    validation_scope: str,
) -> IndiaPackResult:
    return _create_corridor_package(
        "power",
        detections_geojson,
        corridor_centerline_geojson,
        output_dir,
        capture_geometry=capture_geometry,
        validation_scope=validation_scope,
        allowed_classes=POWER_CLASSES,
        allowed_capture_geometries={"close_range_oblique", "close_range_orbit"},
    )


def create_rail_inspection_package(
    detections_geojson: str | Path,
    corridor_centerline_geojson: str | Path,
    output_dir: str | Path,
    *,
    capture_geometry: str,
    validation_scope: str,
) -> IndiaPackResult:
    return _create_corridor_package(
        "rail",
        detections_geojson,
        corridor_centerline_geojson,
        output_dir,
        capture_geometry=capture_geometry,
        validation_scope=validation_scope,
        allowed_classes=RAIL_CLASSES,
        allowed_capture_geometries={"corridor_nadir", "corridor_nadir_oblique"},
    )


def create_solar_module_inventory(
    module_instance_raster: str | Path,
    output_dir: str | Path,
    *,
    validation_scope: str,
    approved_layout_geojson: str | Path | None = None,
    findings_geojson: str | Path | None = None,
    array_join_distance_m: float = 1.0,
    min_module_area_m2: float = 0.01,
) -> IndiaPackResult:
    """Inventory geolocated module instances and compare an optional approved layout."""

    if not validation_scope.strip():
        raise IndiaPackRefused("Solar module inventory requires a validation scope.")
    if array_join_distance_m < 0 or min_module_area_m2 <= 0:
        raise ValueError("Array distance must be non-negative and module area positive.")
    source = read_raster_evidence(
        module_instance_raster, band=1, require_projected=True
    )
    if canonical_name(source.tags.get("ODK_INSTANCE_KIND", "")) != "solar_module":
        raise IndiaPackRefused(
            "The raster is not tagged ODK_INSTANCE_KIND=solar_module."
        )
    if not source.tags.get("ODK_MODEL_KEY") or not source.tags.get("ODK_MODEL_VERSION"):
        raise IndiaPackRefused(
            "Solar instance raster must preserve model key and version provenance."
        )
    labels = np.where(source.valid & (source.data > 0), source.data, 0).astype(np.int64)
    module_ids = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    if not module_ids:
        raise IndiaPackRefused("The solar instance raster contains no module instances.")

    from scipy import ndimage
    from rasterio.transform import xy

    factor = float(source.linear_unit_to_m)
    pixel_m = min(
        float(np.hypot(source.transform.a, source.transform.d) * factor),
        float(np.hypot(source.transform.b, source.transform.e) * factor),
    )
    iterations = int(np.ceil(array_join_distance_m / max(pixel_m, 1e-12)))
    joined = labels > 0
    if iterations > 0:
        joined = ndimage.binary_dilation(joined, iterations=iterations)
    arrays, _ = ndimage.label(joined, structure=np.ones((3, 3), dtype=np.uint8))

    modules: list[dict] = []
    retained = np.zeros(labels.shape, dtype=np.uint32)
    retained_ids: set[int] = set()
    for module_id in module_ids:
        cells = labels == module_id
        area_m2 = float(cells.sum()) * float(source.pixel_area_m2)
        if area_m2 < min_module_area_m2:
            continue
        rows, columns = np.nonzero(cells)
        center_row = float(rows.mean())
        center_column = float(columns.mean())
        x, y = xy(source.transform, center_row, center_column, offset="center")
        row_index = int(np.clip(round(center_row), 0, source.height - 1))
        column_index = int(np.clip(round(center_column), 0, source.width - 1))
        array_id = int(arrays[row_index, column_index])
        retained[cells] = module_id
        retained_ids.add(module_id)
        modules.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": {
                    "module_id": module_id,
                    "array_id": array_id,
                    "area_m2": round(area_m2, 6),
                    "status": "observed",
                    "model_key": source.tags["ODK_MODEL_KEY"],
                    "model_version": source.tags["ODK_MODEL_VERSION"],
                    "validation_scope": validation_scope,
                },
            }
        )

    array_features: list[dict] = []
    for array_id in sorted(int(value) for value in np.unique(arrays) if int(value) > 0):
        array_features.extend(
            polygonize_mask(
                arrays == array_id,
                source,
                properties={"array_id": array_id, "layer": "module_array_group"},
                min_area_m2=0.0,
            )
        )

    missing_features: list[dict] = []
    layout_status: dict[str, Any]
    if approved_layout_geojson is None:
        layout_status = {
            "status": "unavailable",
            "missing_module_count": None,
            "reason": "No approved module layout was supplied; missing modules were not inferred."
        }
    else:
        approved, _ = read_geojson_features(
            approved_layout_geojson,
            target_crs=source.crs,
            allowed_geometry_types={"Polygon", "MultiPolygon"},
        )
        for index, feature in enumerate(approved, start=1):
            expected = rasterize_features([feature], source, all_touched=False)
            if not np.any(expected):
                continue
            if not np.any(expected & (retained > 0)):
                properties = dict(feature.get("properties") or {})
                properties.update(
                    {
                        "expected_module_id": str(
                            properties.get("module_id") or f"expected-{index}"
                        ),
                        "status": "not_observed",
                        "finding": "missing_module_review",
                        "requires_review": True,
                    }
                )
                missing_features.append({**feature, "properties": properties})
        layout_status = {
            "status": "available",
            "approved_module_count": len(approved),
            "missing_module_review_count": len(missing_features),
        }

    normalized_findings: list[dict] = []
    if findings_geojson is not None:
        findings, _ = read_geojson_features(
            findings_geojson,
            target_crs=source.crs,
            allowed_geometry_types={"Point", "Polygon", "MultiPolygon"},
        )
        for finding in findings:
            properties = _validate_finding(
                finding,
                class_key="finding",
                allowed={"damaged_module", "obstructed_module", "vegetation_obstruction"},
            )
            module_id = int(properties.get("module_id", 0))
            if module_id not in retained_ids:
                properties["association_status"] = "review_candidate"
                properties["association_reason"] = "module_id is absent from observed inventory"
            else:
                properties["association_status"] = "associated"
            normalized_findings.append({**finding, "properties": properties})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raster_path = write_aligned_raster(
        output / "solar_module_instances.tif",
        retained,
        source,
        dtype="uint32",
        nodata=0,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "validated_solar_module_inventory",
            "ODK_INSTANCE_KIND": "solar_module",
            "ODK_MODEL_KEY": source.tags["ODK_MODEL_KEY"],
            "ODK_MODEL_VERSION": source.tags["ODK_MODEL_VERSION"],
            "ODK_VALIDATION_SCOPE": validation_scope,
        },
    )
    inventory_path = geo.write_geojson(
        output / "solar_module_inventory.geojson",
        [*modules, *array_features],
        epsg=source.epsg,
        properties={"validation_scope": validation_scope},
    )
    missing_path = geo.write_geojson(
        output / "solar_missing_module_review.geojson",
        missing_features,
        epsg=source.epsg,
        properties={"layout_comparison": layout_status},
    )
    findings_path = geo.write_geojson(
        output / "solar_module_findings.geojson",
        normalized_findings,
        epsg=source.epsg,
        properties={
            "warning": "Only supplied validated findings are named; no defect was inferred from geometry."
        },
    )
    summary_path = write_summary(
        output / "solar_inventory.json",
        {
            "status": "complete",
            "analysis": "validated_solar_module_instance_inventory",
            "source": {
                "instance_raster": source.path,
                "epsg": source.epsg,
                "model_key": source.tags["ODK_MODEL_KEY"],
                "model_version": source.tags["ODK_MODEL_VERSION"],
            },
            "validation_scope": validation_scope,
            "module_count": len(modules),
            "array_count": len({item["properties"]["array_id"] for item in modules}),
            "approved_layout": layout_status,
            "finding_count": len(normalized_findings),
            "findings_by_type": dict(
                sorted(Counter(item["properties"]["finding"] for item in normalized_findings).items())
            ),
            "units": {"module_area": "m2", "count": "instances"},
            "artifacts": {
                "instance_raster": raster_path,
                "inventory": str(inventory_path),
                "missing_module_review": str(missing_path),
                "findings": str(findings_path),
            },
        },
    )
    return IndiaPackResult(
        summary_path,
        (raster_path, str(inventory_path), str(missing_path), str(findings_path)),
    )
