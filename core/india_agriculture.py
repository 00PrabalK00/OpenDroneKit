"""Honest agriculture indices, canopy metrics, stress zones and instance counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import geo
from .india_geospatial import (
    IndiaPackRefused,
    IndiaPackResult,
    assert_aligned,
    canonical_name,
    ids_for_names,
    load_class_names,
    polygonize_mask,
    read_raster_evidence,
    semantic_valid_mask,
    write_aligned_raster,
    write_summary,
)
from .semantic_engine import CLASS_NODATA, SemanticSchema


@dataclass(frozen=True)
class ReflectanceBandCalibration:
    """Explicit conversion from stored samples to unitless surface reflectance."""

    band: int
    scale: float
    offset: float
    source: str
    valid_min: float = 0.0
    valid_max: float = 1.0
    quantity: str = "surface_reflectance"

    def __post_init__(self) -> None:
        if self.band < 1:
            raise ValueError("band must use one-based raster indexing.")
        if not np.isfinite(self.scale) or self.scale == 0:
            raise ValueError("reflectance scale must be finite and non-zero.")
        if not np.isfinite(self.offset):
            raise ValueError("reflectance offset must be finite.")
        if not self.source.strip():
            raise ValueError("calibration source is required.")
        if canonical_name(self.quantity) != "surface_reflectance":
            raise ValueError(
                "Vegetation indices require surface-reflectance calibration, not raw DN values."
            )
        if not self.valid_min < self.valid_max:
            raise ValueError("valid_min must be smaller than valid_max.")


INDEX_BANDS: dict[str, tuple[str, str]] = {
    "NDVI": ("nir", "red"),
    "NDRE": ("nir", "red_edge"),
    "GNDVI": ("nir", "green"),
}


def compute_vegetation_indices(
    multispectral_raster: str | Path,
    calibrations: Mapping[str, ReflectanceBandCalibration],
    output_dir: str | Path,
) -> IndiaPackResult:
    """Compute only indices supported by explicitly calibrated reflectance bands."""

    normalised = {canonical_name(name): value for name, value in calibrations.items()}
    # Band one establishes georeferencing even when every requested index is unavailable.
    reference = read_raster_evidence(multispectral_raster, band=1, require_projected=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cache: dict[str, tuple[np.ndarray, np.ndarray, ReflectanceBandCalibration]] = {}
    for name, calibration in normalised.items():
        try:
            evidence = read_raster_evidence(
                multispectral_raster,
                band=calibration.band,
                require_projected=False,
            )
            assert_aligned(reference, evidence)
        except IndiaPackRefused:
            continue
        reflectance = evidence.data.astype(np.float32) * float(calibration.scale)
        reflectance += float(calibration.offset)
        valid = (
            evidence.valid
            & np.isfinite(reflectance)
            & (reflectance >= calibration.valid_min)
            & (reflectance <= calibration.valid_max)
        )
        cache[name] = (reflectance, valid, calibration)

    results: dict[str, dict] = {}
    artifacts: list[str] = []
    for index_name, (positive_name, negative_name) in INDEX_BANDS.items():
        missing = [name for name in (positive_name, negative_name) if name not in cache]
        if missing:
            results[index_name] = {
                "status": "unavailable",
                "path": None,
                "reason": (
                    f"Cannot compute {index_name}: calibrated surface-reflectance "
                    f"band(s) missing or unreadable: {', '.join(missing)}. No proxy was substituted."
                ),
            }
            continue
        positive, positive_valid, positive_cal = cache[positive_name]
        negative, negative_valid, negative_cal = cache[negative_name]
        denominator = positive + negative
        valid = positive_valid & negative_valid & (np.abs(denominator) > 1e-8)
        index = np.full(reference.data.shape, np.nan, dtype=np.float32)
        index[valid] = (positive[valid] - negative[valid]) / denominator[valid]
        index[valid] = np.clip(index[valid], -1.0, 1.0)
        path = write_aligned_raster(
            output / f"{index_name.casefold()}.tif",
            index,
            reference,
            dtype="float32",
            nodata=float("nan"),
            categorical=False,
            tags={
                "ODK_ANALYSIS": "calibrated_vegetation_index",
                "ODK_INDEX": index_name,
                "ODK_CALIBRATED_REFLECTANCE": "true",
                "ODK_POSITIVE_BAND": positive_name,
                "ODK_NEGATIVE_BAND": negative_name,
                "ODK_POSITIVE_CALIBRATION_SOURCE": positive_cal.source,
                "ODK_NEGATIVE_CALIBRATION_SOURCE": negative_cal.source,
            },
        )
        artifacts.append(path)
        values = index[valid]
        results[index_name] = {
            "status": "available",
            "path": path,
            "valid_pixel_count": int(valid.sum()),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "formula": f"({positive_name} - {negative_name}) / ({positive_name} + {negative_name})",
            "calibrations": {
                positive_name: asdict(positive_cal),
                negative_name: asdict(negative_cal),
            },
        }

    available = sum(item["status"] == "available" for item in results.values())
    status = "complete" if available == len(INDEX_BANDS) else (
        "partial" if available else "unavailable"
    )
    summary_path = write_summary(
        output / "vegetation_indices.json",
        {
            "status": status,
            "analysis": "calibrated_spectral_indices",
            "source": {"raster": reference.path, "epsg": reference.epsg},
            "units": "unitless_ratio",
            "indices": results,
            "warning": (
                "These are deterministic spectral indices, not disease diagnoses or AI predictions."
            ),
        },
    )
    return IndiaPackResult(summary_path, tuple(artifacts), status=status)


CANOPY_ALIASES = ("crop", "crop_canopy", "vegetation", "tree", "trees", "orchard")
UNWANTED_ALIASES = ("weed", "weeds", "unwanted_vegetation")
SOIL_ALIASES = ("soil", "bare_soil", "bare_land", "bareland")
WATER_ALIASES = ("water", "water_body", "waterlogged")


def analyse_canopy_cover(
    semantic_class_raster: str | Path,
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
    output_dir: str | Path,
    *,
    min_region_area_m2: float = 0.1,
) -> IndiaPackResult:
    """Measure canopy and potential bare/missing regions from explicit semantic classes."""

    if min_region_area_m2 < 0:
        raise ValueError("min_region_area_m2 cannot be negative.")
    raster = read_raster_evidence(
        semantic_class_raster, band=1, require_projected=True
    )
    names = load_class_names(schema)
    canopy_ids, _ = ids_for_names(names, CANOPY_ALIASES)
    unwanted_ids, _ = ids_for_names(names, UNWANTED_ALIASES)
    soil_ids, _ = ids_for_names(names, SOIL_ALIASES)
    water_ids, _ = ids_for_names(names, WATER_ALIASES)
    if not canopy_ids:
        raise IndiaPackRefused("The semantic schema has no crop/canopy/vegetation class.")

    valid = semantic_valid_mask(raster)
    canopy = valid & np.isin(raster.data, list(canopy_ids))
    unwanted = valid & np.isin(raster.data, list(unwanted_ids)) if unwanted_ids else np.zeros_like(valid)
    soil = valid & np.isin(raster.data, list(soil_ids)) if soil_ids else np.zeros_like(valid)
    water = valid & np.isin(raster.data, list(water_ids)) if water_ids else np.zeros_like(valid)
    field = canopy | unwanted | soil
    if not np.any(field):
        raise IndiaPackRefused("No crop/canopy or soil pixels are present in the semantic layer.")

    encoded = np.full(raster.data.shape, 255, dtype=np.uint8)
    encoded[valid] = 0
    encoded[soil] = 3
    encoded[water] = 4
    encoded[unwanted] = 2
    encoded[canopy] = 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    class_path = write_aligned_raster(
        output / "canopy_classes.tif",
        encoded,
        raster,
        dtype="uint8",
        nodata=255,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "semantic_canopy_cover",
            "ODK_CLASS_0": "other_valid_surface",
            "ODK_CLASS_1": "canopy",
            "ODK_CLASS_2": "unwanted_vegetation",
            "ODK_CLASS_3": "bare_soil_potential_missing_crop",
            "ODK_CLASS_4": "water",
        },
    )
    region_features: list[dict] = []
    for category, mask in (
        ("canopy", canopy),
        ("unwanted_vegetation", unwanted),
        ("bare_soil_potential_missing_crop", soil),
        ("water", water),
    ):
        region_features.extend(
            polygonize_mask(
                mask,
                raster,
                properties={"category": category, "source": "semantic_class_raster"},
                min_area_m2=min_region_area_m2,
            )
        )
    regions_path = geo.write_geojson(
        output / "canopy_regions.geojson",
        region_features,
        epsg=raster.epsg,
        properties={
            "warning": (
                "Bare soil is a potential gap only; expected planting geometry is required "
                "to call a plant missing."
            )
        },
    )
    pixel_area = float(raster.pixel_area_m2)
    canopy_percent = 100.0 * float(canopy.sum()) / float(field.sum())
    summary_path = write_summary(
        output / "canopy_cover.json",
        {
            "status": "complete",
            "analysis": "semantic_canopy_cover",
            "source": {"semantic_class_raster": raster.path, "epsg": raster.epsg},
            "units": {"area": "m2", "coverage": "percent"},
            "field_analysis_area_m2": round(float(field.sum()) * pixel_area, 6),
            "canopy_area_m2": round(float(canopy.sum()) * pixel_area, 6),
            "canopy_cover_percent": round(canopy_percent, 6),
            "unwanted_vegetation": (
                {
                    "status": "available",
                    "area_m2": round(float(unwanted.sum()) * pixel_area, 6),
                }
                if unwanted_ids
                else {
                    "status": "unavailable",
                    "area_m2": None,
                    "reason": "The semantic schema has no unwanted-vegetation class.",
                }
            ),
            "bare_or_potential_missing_area": (
                {
                    "status": "available",
                    "area_m2": round(float(soil.sum()) * pixel_area, 6),
                    "interpretation": "bare soil; not a confirmed missing plant",
                }
                if soil_ids
                else {
                    "status": "unavailable",
                    "area_m2": None,
                    "reason": "The semantic schema has no soil/bare-land class.",
                }
            ),
            "water": (
                {"status": "available", "area_m2": round(float(water.sum()) * pixel_area, 6)}
                if water_ids
                else {
                    "status": "unavailable",
                    "area_m2": None,
                    "reason": "The semantic schema has no water class.",
                }
            ),
            "artifacts": {"class_raster": class_path, "regions": str(regions_path)},
        },
    )
    return IndiaPackResult(summary_path, (class_path, str(regions_path)))


def create_stress_zones(
    vegetation_index_raster: str | Path,
    output_dir: str | Path,
    *,
    severe_below: float,
    moderate_below: float,
    validation_scope: str,
    canopy_mask_raster: str | Path | None = None,
    min_region_area_m2: float = 0.1,
) -> IndiaPackResult:
    """Turn a calibrated vegetation index into explicitly scoped stress zones."""

    if not -1.0 <= severe_below < moderate_below <= 1.0:
        raise ValueError("Stress thresholds must satisfy -1 <= severe < moderate <= 1.")
    if not validation_scope.strip():
        raise IndiaPackRefused(
            "A crop/sensor validation scope is required before index thresholds can be called stress."
        )
    index = read_raster_evidence(
        vegetation_index_raster, band=1, require_projected=True
    )
    if index.tags.get("ODK_CALIBRATED_REFLECTANCE", "").casefold() != "true":
        raise IndiaPackRefused(
            "The source is not tagged as an index computed from calibrated reflectance."
        )
    index_name = index.tags.get("ODK_INDEX", "").upper()
    if index_name not in INDEX_BANDS:
        raise IndiaPackRefused("The source does not identify NDVI, NDRE, or GNDVI.")
    valid = index.valid & np.isfinite(index.data)
    if canopy_mask_raster is not None:
        canopy = read_raster_evidence(canopy_mask_raster, band=1, require_projected=True)
        assert_aligned(index, canopy)
        # A binary mask uses >0; the canopy product produced above uses class 1.
        valid &= canopy.valid & (canopy.data > 0)
    if not np.any(valid):
        raise IndiaPackRefused("No valid canopy pixels remain for stress zoning.")

    severe = valid & (index.data < severe_below)
    moderate = valid & (index.data >= severe_below) & (index.data < moderate_below)
    reference = valid & (index.data >= moderate_below)
    encoded = np.full(index.data.shape, 255, dtype=np.uint8)
    encoded[valid] = 3
    encoded[moderate] = 2
    encoded[severe] = 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raster_path = write_aligned_raster(
        output / "stress_zones.tif",
        encoded,
        index,
        dtype="uint8",
        nodata=255,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "validated_index_stress_zoning",
            "ODK_INDEX": index_name,
            "ODK_VALIDATION_SCOPE": validation_scope,
            "ODK_CLASS_1": "severe_stress",
            "ODK_CLASS_2": "moderate_stress",
            "ODK_CLASS_3": "reference_or_healthy_range",
        },
    )
    features: list[dict] = []
    for zone, mask in (
        ("severe_stress", severe),
        ("moderate_stress", moderate),
        ("reference_or_healthy_range", reference),
    ):
        features.extend(
            polygonize_mask(
                mask,
                index,
                properties={
                    "zone": zone,
                    "index": index_name,
                    "validation_scope": validation_scope,
                    "diagnosed_cause": None,
                },
                min_area_m2=min_region_area_m2,
            )
        )
    regions_path = geo.write_geojson(
        output / "stress_regions.geojson",
        features,
        epsg=index.epsg,
        properties={
            "index": index_name,
            "validation_scope": validation_scope,
            "warning": "Zones do not diagnose disease, nutrient deficiency, or irrigation cause.",
        },
    )
    area = float(index.pixel_area_m2)
    summary_path = write_summary(
        output / "stress_zones.json",
        {
            "status": "complete",
            "analysis": "validated_index_threshold_zoning",
            "source": {"index_raster": index.path, "index": index_name, "epsg": index.epsg},
            "validation_scope": validation_scope,
            "thresholds": {"severe_below": severe_below, "moderate_below": moderate_below},
            "units": {"area": "m2", "index": "unitless_ratio"},
            "zones": {
                "severe_stress": {"area_m2": round(float(severe.sum()) * area, 6)},
                "moderate_stress": {"area_m2": round(float(moderate.sum()) * area, 6)},
                "reference_or_healthy_range": {"area_m2": round(float(reference.sum()) * area, 6)},
            },
            "interpretation": (
                "Threshold zones are valid only within the stated crop/sensor scope and do "
                "not identify a biological cause."
            ),
            "artifacts": {"class_raster": raster_path, "regions": str(regions_path)},
        },
    )
    return IndiaPackResult(summary_path, (raster_path, str(regions_path)))


def count_plant_instances(
    instance_raster: str | Path,
    output_dir: str | Path,
    *,
    validation_scope: str,
    instance_encoding: str = "binary",
    min_instance_area_m2: float = 0.01,
    max_instance_area_m2: float | None = None,
    health_class_raster: str | Path | None = None,
    health_schema: SemanticSchema | Mapping[int | str, str] | str | Path | None = None,
    health_validation_scope: str = "",
) -> IndiaPackResult:
    """Count geolocated plant/tree instances from a validated instance mask."""

    if not validation_scope.strip():
        raise IndiaPackRefused("Plant/tree counting requires a stated validation scope.")
    if instance_encoding not in {"binary", "label_ids"}:
        raise ValueError("instance_encoding must be binary or label_ids.")
    if min_instance_area_m2 <= 0:
        raise ValueError("min_instance_area_m2 must be positive.")
    if max_instance_area_m2 is not None and max_instance_area_m2 < min_instance_area_m2:
        raise ValueError("max_instance_area_m2 cannot be below the minimum.")
    source = read_raster_evidence(instance_raster, band=1, require_projected=True)
    kind = canonical_name(source.tags.get("ODK_INSTANCE_KIND", ""))
    if kind not in {"plant", "tree", "plant_or_tree"}:
        raise IndiaPackRefused(
            "The raster is not tagged ODK_INSTANCE_KIND=plant/tree/plant_or_tree."
        )
    positive = source.valid & (source.data > 0)
    if instance_encoding == "binary":
        from scipy import ndimage

        labels, _ = ndimage.label(positive, structure=np.ones((3, 3), dtype=np.uint8))
    else:
        labels = np.where(positive, source.data, 0).astype(np.int64)

    health: object = None
    health_names: dict[int, str] = {}
    if health_class_raster is not None:
        if health_schema is None or not health_validation_scope.strip():
            raise IndiaPackRefused(
                "Health categories require both a schema and a stated validation scope."
            )
        health = read_raster_evidence(health_class_raster, band=1, require_projected=True)
        assert_aligned(source, health)
        health_names = load_class_names(health_schema)

    retained = np.zeros(labels.shape, dtype=np.uint32)
    instances: list[dict] = []
    next_id = 1
    import rasterio.transform

    for raw_id in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        cells = labels == raw_id
        area_m2 = float(cells.sum()) * float(source.pixel_area_m2)
        if area_m2 < min_instance_area_m2:
            continue
        if max_instance_area_m2 is not None and area_m2 > max_instance_area_m2:
            continue
        rows, columns = np.nonzero(cells)
        x, y = rasterio.transform.xy(
            source.transform,
            float(rows.mean()),
            float(columns.mean()),
            offset="center",
        )
        properties: dict[str, object] = {
            "instance_id": next_id,
            "source_instance_id": raw_id,
            "instance_kind": kind,
            "area_m2": round(area_m2, 6),
            "validation_scope": validation_scope,
        }
        if health is None:
            properties["health_category"] = None
            properties["health_status"] = "unavailable"
            properties["health_reason"] = "No validated health-class raster was supplied."
        else:
            health_evidence = health
            values = health_evidence.data[cells & health_evidence.valid]
            values = values[np.isin(values, list(health_names))]
            if values.size:
                ids, counts = np.unique(values.astype(np.int64), return_counts=True)
                health_id = int(ids[int(np.argmax(counts))])
                properties["health_category"] = health_names[health_id]
                properties["health_status"] = "available"
                properties["health_validation_scope"] = health_validation_scope
            else:
                properties["health_category"] = None
                properties["health_status"] = "unavailable"
                properties["health_reason"] = "No valid health-class pixels overlap this instance."
        retained[cells] = next_id
        instances.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": properties,
            }
        )
        next_id += 1

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raster_path = write_aligned_raster(
        output / "plant_instances.tif",
        retained,
        source,
        dtype="uint32",
        nodata=0,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "validated_plant_tree_instance_count",
            "ODK_INSTANCE_KIND": kind,
            "ODK_VALIDATION_SCOPE": validation_scope,
        },
    )
    points_path = geo.write_geojson(
        output / "plant_instances.geojson",
        instances,
        epsg=source.epsg,
        properties={
            "validation_scope": validation_scope,
            "missing_plants": {
                "status": "unavailable",
                "reason": "No expected planting layout was supplied."
            },
        },
    )
    summary_path = write_summary(
        output / "plant_count.json",
        {
            "status": "complete",
            "analysis": "validated_instance_mask_count",
            "source": {"instance_raster": source.path, "epsg": source.epsg},
            "validation_scope": validation_scope,
            "count": len(instances),
            "units": {"area": "m2", "count": "instances"},
            "filters": {
                "minimum_instance_area_m2": min_instance_area_m2,
                "maximum_instance_area_m2": max_instance_area_m2,
            },
            "missing_plants": {
                "status": "unavailable",
                "reason": (
                    "An observed instance mask can count plants but cannot know which plants "
                    "are missing without an expected planting layout."
                ),
            },
            "health_categories": (
                {"status": "available", "validation_scope": health_validation_scope}
                if health is not None
                else {
                    "status": "unavailable",
                    "reason": "No validated health-class raster was supplied."
                }
            ),
            "artifacts": {"instance_raster": raster_path, "points": str(points_path)},
        },
    )
    return IndiaPackResult(summary_path, (raster_path, str(points_path)))
