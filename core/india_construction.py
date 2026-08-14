"""Construction semantic schema and approved-design progress evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from . import geo
from .india_geospatial import (
    IndiaPackRefused,
    IndiaPackResult,
    canonical_name,
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
from .semantic_engine import (
    SemanticClass,
    SemanticInferenceConfig,
    SemanticInferencePackage,
    SemanticModelMetadata,
    SemanticPredictor,
    SemanticSchema,
    run_semantic_inference,
)


CONSTRUCTION_SCHEMA = SemanticSchema(
    id="india-construction-site",
    version="1.0.0",
    classes=(
        SemanticClass(0, "background", (0, 0, 0), background=True),
        SemanticClass(1, "building", (210, 80, 80)),
        SemanticClass(2, "unfinished_building", (255, 145, 0)),
        SemanticClass(3, "road", (120, 120, 120)),
        SemanticClass(4, "bare_soil", (156, 112, 72)),
        SemanticClass(5, "vegetation", (60, 150, 70)),
        SemanticClass(6, "water", (50, 110, 220)),
        SemanticClass(7, "concrete", (190, 190, 190)),
        SemanticClass(8, "excavation", (115, 75, 45)),
        SemanticClass(9, "stockpile", (230, 195, 90)),
        SemanticClass(10, "construction_material", (170, 80, 190)),
        SemanticClass(11, "equipment", (245, 220, 30)),
    ),
)


def run_construction_segmentation(
    orthomosaic_path: str | Path,
    output_dir: str | Path,
    *,
    model: SemanticModelMetadata,
    predictor: SemanticPredictor | Callable[[np.ndarray], np.ndarray],
    config: SemanticInferenceConfig | None = None,
) -> SemanticInferencePackage:
    """Run a task-trained construction head through the shared semantic engine.

    The shared engine performs the task-training, validation-metric and checkpoint
    hash gates.  No generic foundation output is silently presented as a construction
    map.
    """

    if (model.schema_id, model.schema_version) != (
        CONSTRUCTION_SCHEMA.id,
        CONSTRUCTION_SCHEMA.version,
    ):
        raise IndiaPackRefused(
            "The selected model was not trained for the versioned India construction schema."
        )
    return run_semantic_inference(
        orthomosaic_path,
        output_dir,
        schema=CONSTRUCTION_SCHEMA,
        model=model,
        predictor=predictor,
        config=config,
    )


def measure_approved_design_progress(
    semantic_class_raster: str | Path,
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
    approved_design_geojson: str | Path,
    output_dir: str | Path,
    *,
    complete_at_percent: float = 95.0,
    min_region_area_m2: float = 0.25,
) -> IndiaPackResult:
    """Measure observed semantic coverage inside explicit approved design elements.

    The result is deliberately called *observed surface evidence*.  It is not a
    contractual percentage complete, which would also require schedule, quantities,
    hidden work and sign-off evidence.
    """

    if not 0 < complete_at_percent <= 100:
        raise ValueError("complete_at_percent must be in (0, 100].")
    if min_region_area_m2 < 0:
        raise ValueError("min_region_area_m2 cannot be negative.")
    raster = read_raster_evidence(
        semantic_class_raster, band=1, require_projected=True
    )
    class_names = load_class_names(schema)
    design_features, _ = read_geojson_features(
        approved_design_geojson,
        target_crs=raster.crs,
        allowed_geometry_types={"Polygon", "MultiPolygon"},
    )
    if not design_features:
        raise IndiaPackRefused("The approved design contains no polygon elements.")
    valid = semantic_valid_mask(raster)
    design_union = rasterize_features(design_features, raster, all_touched=False)
    if not np.any(design_union):
        raise IndiaPackRefused("The approved design does not overlap the semantic survey grid.")

    progress_features: list[dict] = []
    missing_mask = np.zeros_like(valid)
    expected_ids_union: set[int] = set()
    elements: list[dict] = []
    weighted_observed_cells = 0
    weighted_design_cells = 0
    for index, feature in enumerate(design_features, start=1):
        properties = dict(feature.get("properties") or {})
        element_id = str(properties.get("element_id") or properties.get("id") or f"element-{index}")
        expected_name = canonical_name(str(properties.get("expected_class") or ""))
        output_properties = dict(properties)
        output_properties["element_id"] = element_id
        if not expected_name:
            output_properties.update(
                {
                    "evidence_status": "unavailable",
                    "observed_surface_coverage_percent": None,
                    "reason": "Approved element has no expected_class property.",
                }
            )
            progress_features.append({**feature, "properties": output_properties})
            elements.append(output_properties)
            continue
        expected_ids, _ = ids_for_names(class_names, (expected_name,))
        if not expected_ids:
            output_properties.update(
                {
                    "evidence_status": "unavailable",
                    "observed_surface_coverage_percent": None,
                    "reason": f"Semantic schema has no {expected_name!r} class.",
                }
            )
            progress_features.append({**feature, "properties": output_properties})
            elements.append(output_properties)
            continue
        expected_ids_union.update(expected_ids)
        element_mask = rasterize_features([feature], raster, all_touched=False) & valid
        design_cells = int(element_mask.sum())
        if design_cells == 0:
            output_properties.update(
                {
                    "evidence_status": "unavailable",
                    "observed_surface_coverage_percent": None,
                    "reason": "Approved element has no valid covered survey cells.",
                }
            )
            progress_features.append({**feature, "properties": output_properties})
            elements.append(output_properties)
            continue
        observed = element_mask & np.isin(raster.data, list(expected_ids))
        observed_cells = int(observed.sum())
        coverage = 100.0 * observed_cells / design_cells
        missing_mask |= element_mask & ~observed
        weighted_observed_cells += observed_cells
        weighted_design_cells += design_cells
        state = "observed" if coverage >= complete_at_percent else (
            "partially_observed" if coverage > 0 else "not_observed"
        )
        output_properties.update(
            {
                "expected_class": expected_name,
                "evidence_status": "available",
                "observed_surface_state": state,
                "observed_surface_coverage_percent": round(coverage, 6),
                "design_area_m2": round(design_cells * float(raster.pixel_area_m2), 6),
                "observed_area_m2": round(observed_cells * float(raster.pixel_area_m2), 6),
                "contractual_completion_percent": None,
                "requires_review": True,
            }
        )
        progress_features.append({**feature, "properties": output_properties})
        elements.append(output_properties)

    observed_relevant = (
        valid & np.isin(raster.data, list(expected_ids_union))
        if expected_ids_union
        else np.zeros_like(valid)
    )
    deviation_mask = observed_relevant & ~design_union
    review_features = polygonize_mask(
        missing_mask,
        raster,
        properties={
            "finding": "approved_surface_not_observed",
            "requires_review": True,
        },
        min_area_m2=min_region_area_m2,
    )
    review_features.extend(
        polygonize_mask(
            deviation_mask,
            raster,
            properties={
                "finding": "observed_relevant_surface_outside_approved_design",
                "requires_review": True,
            },
            min_area_m2=min_region_area_m2,
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = geo.write_geojson(
        output / "approved_design_progress.geojson",
        progress_features,
        epsg=raster.epsg,
        properties={
            "source_design": str(approved_design_geojson),
            "metric": "observed_surface_coverage_percent",
            "contractual_completion": "not_measured",
        },
    )
    review_path = geo.write_geojson(
        output / "approved_design_review_regions.geojson",
        review_features,
        epsg=raster.epsg,
        properties={
            "warning": "Semantic mismatch regions require imagery/design registration review."
        },
    )
    encoded = np.full(raster.data.shape, 255, dtype=np.uint8)
    encoded[valid] = 0
    encoded[missing_mask] = 1
    encoded[deviation_mask] = 2
    raster_path = write_aligned_raster(
        output / "approved_design_review.tif",
        encoded,
        raster,
        dtype="uint8",
        nodata=255,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "approved_design_semantic_evidence",
            "ODK_CLASS_1": "approved_surface_not_observed",
            "ODK_CLASS_2": "observed_surface_outside_approved_design",
        },
    )
    overall = (
        100.0 * weighted_observed_cells / weighted_design_cells
        if weighted_design_cells
        else None
    )
    summary_path = write_summary(
        output / "approved_design_progress.json",
        {
            "status": "complete",
            "analysis": "approved_design_observed_surface_evidence",
            "source": {
                "semantic_class_raster": raster.path,
                "approved_design": str(approved_design_geojson),
                "epsg": raster.epsg,
            },
            "units": {"area": "m2", "coverage": "percent"},
            "observed_surface_coverage_percent": (
                round(overall, 6) if overall is not None else None
            ),
            "contractual_completion_percent": {
                "status": "unavailable",
                "value": None,
                "reason": (
                    "Visible semantic coverage does not measure schedule, quantities, hidden "
                    "work, quality acceptance, or contractual completion."
                ),
            },
            "elements": elements,
            "review_areas": {
                "approved_not_observed_m2": round(
                    float(missing_mask.sum()) * float(raster.pixel_area_m2), 6
                ),
                "observed_outside_design_m2": round(
                    float(deviation_mask.sum()) * float(raster.pixel_area_m2), 6
                ),
            },
            "artifacts": {
                "elements": str(progress_path),
                "review_regions": str(review_path),
                "review_raster": raster_path,
            },
        },
    )
    return IndiaPackResult(
        summary_path, (str(progress_path), str(review_path), raster_path)
    )
