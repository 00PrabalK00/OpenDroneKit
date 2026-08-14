"""Land-survey GIS extraction and cadastral encroachment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import geo
from .india_geospatial import (
    IndiaPackRefused,
    IndiaPackResult,
    assert_aligned,
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
from .semantic_engine import CLASS_NODATA, SemanticSchema


LAND_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "building": ("building", "buildings", "building_footprint"),
    "road_path": ("road", "paved_road", "unpaved_road", "path"),
    "water": ("water", "water_body", "pond", "river", "canal"),
    "vegetation": ("vegetation", "tree", "trees", "crop", "grass", "forest", "scrub"),
}


def _category_ids(class_names: Mapping[int, str], aliases: Sequence[str]) -> set[int]:
    ids, _ = ids_for_names(class_names, aliases)
    return ids


def extract_land_gis(
    semantic_class_raster: str | Path,
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
    output_dir: str | Path,
    *,
    min_polygon_area_m2: float = 1.0,
) -> IndiaPackResult:
    """Extract observed land-cover vectors from a georeferenced semantic layer.

    The survey extent is also emitted, but it is explicitly not represented as a
    cadastral parcel.  Legal boundaries must come from an authoritative import.
    """

    if min_polygon_area_m2 < 0:
        raise ValueError("min_polygon_area_m2 cannot be negative.")
    raster = read_raster_evidence(
        semantic_class_raster, band=1, require_projected=True
    )
    class_names = load_class_names(schema)
    valid = semantic_valid_mask(raster)
    if not np.any(valid):
        raise IndiaPackRefused("The semantic class raster contains no valid survey pixels.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features: list[dict] = []
    categories: dict[str, dict] = {}
    for category, aliases in LAND_CLASS_ALIASES.items():
        class_ids = _category_ids(class_names, aliases)
        if not class_ids:
            categories[category] = {
                "status": "unavailable",
                "reason": f"The semantic schema has no supported {category} class.",
                "area_m2": None,
                "polygon_count": 0,
            }
            continue
        mask = valid & np.isin(raster.data, list(class_ids))
        category_features = polygonize_mask(
            mask,
            raster,
            properties={
                "layer": category,
                "source": "semantic_class_raster",
                "source_class_ids": sorted(class_ids),
            },
            min_area_m2=min_polygon_area_m2,
        )
        features.extend(category_features)
        categories[category] = {
            "status": "available",
            "area_m2": round(float(mask.sum()) * float(raster.pixel_area_m2), 6),
            "polygon_count": len(category_features),
            "source_class_ids": sorted(class_ids),
        }

    extent_features = polygonize_mask(
        valid,
        raster,
        properties={
            "layer": "survey_extent",
            "legal_boundary": False,
            "warning": "Observed raster coverage; not a cadastral parcel boundary.",
        },
        min_area_m2=0.0,
    )
    vector_path = geo.write_geojson(
        output / "land_gis.geojson",
        [*features, *extent_features],
        epsg=raster.epsg,
        properties={
            "source_raster": raster.path,
            "units": {"area": "m2"},
            "parcel_boundary": {
                "status": "unavailable",
                "reason": (
                    "A semantic land-cover layer cannot establish a legal parcel boundary; "
                    "import an authoritative cadastral boundary for encroachment analysis."
                ),
            },
        },
    )
    summary_path = write_summary(
        output / "land_gis.json",
        {
            "status": "complete",
            "analysis": "observed_land_cover_gis_extraction",
            "source": {"semantic_class_raster": raster.path, "epsg": raster.epsg},
            "units": {"area": "m2"},
            "categories": categories,
            "survey_extent_area_m2": round(
                float(valid.sum()) * float(raster.pixel_area_m2), 6
            ),
            "parcel_boundary": {
                "status": "unavailable",
                "reason": (
                    "Semantic imagery shows observed surfaces, not legal ownership. "
                    "No cadastral parcel was inferred."
                ),
            },
            "artifacts": {"vectors": str(vector_path)},
        },
    )
    return IndiaPackResult(summary_path, (str(vector_path),))


def detect_cadastral_encroachment(
    semantic_class_raster: str | Path,
    schema: SemanticSchema | Mapping[int | str, str] | str | Path,
    cadastral_boundary_geojson: str | Path,
    output_dir: str | Path,
    *,
    previous_semantic_class_raster: str | Path | None = None,
    boundary_tolerance_m: float = 0.0,
    min_polygon_area_m2: float = 0.25,
) -> IndiaPackResult:
    """Map observed buildings/roads outside an imported cadastral boundary."""

    if boundary_tolerance_m < 0:
        raise ValueError("boundary_tolerance_m cannot be negative.")
    if min_polygon_area_m2 < 0:
        raise ValueError("min_polygon_area_m2 cannot be negative.")
    current = read_raster_evidence(
        semantic_class_raster, band=1, require_projected=True
    )
    class_names = load_class_names(schema)
    boundary_features, _ = read_geojson_features(
        cadastral_boundary_geojson,
        target_crs=current.crs,
        allowed_geometry_types={"Polygon", "MultiPolygon"},
    )
    if not boundary_features:
        raise IndiaPackRefused("The cadastral boundary contains no polygon features.")
    inside = rasterize_features(boundary_features, current, all_touched=False)
    if not np.any(inside):
        raise IndiaPackRefused(
            "The cadastral boundary does not overlap the semantic survey grid."
        )

    allowed = inside.copy()
    if boundary_tolerance_m > 0:
        from scipy import ndimage

        factor = float(current.linear_unit_to_m)
        row_spacing_m = float(
            np.hypot(current.transform.b, current.transform.e) * factor
        )
        column_spacing_m = float(
            np.hypot(current.transform.a, current.transform.d) * factor
        )
        distance_outside_m = ndimage.distance_transform_edt(
            ~inside, sampling=(row_spacing_m, column_spacing_m)
        )
        allowed |= distance_outside_m <= boundary_tolerance_m

    building_ids = _category_ids(class_names, LAND_CLASS_ALIASES["building"])
    road_ids = _category_ids(class_names, LAND_CLASS_ALIASES["road_path"])
    if not building_ids and not road_ids:
        raise IndiaPackRefused(
            "The semantic schema has neither a building nor a road/path class."
        )
    valid = semantic_valid_mask(current)
    masks = {
        "building_encroachment": valid & np.isin(current.data, list(building_ids)) & ~allowed,
        "road_path_encroachment": valid & np.isin(current.data, list(road_ids)) & ~allowed,
    }
    all_encroachment = masks["building_encroachment"] | masks["road_path_encroachment"]
    vector_features: list[dict] = []
    totals: dict[str, dict] = {}
    for kind, mask in masks.items():
        kind_features = polygonize_mask(
            mask,
            current,
            properties={
                "finding": kind,
                "interpretation": "observed semantic surface outside imported boundary",
                "requires_human_review": True,
            },
            min_area_m2=min_polygon_area_m2,
        )
        vector_features.extend(kind_features)
        totals[kind] = {
            "area_m2": round(float(mask.sum()) * float(current.pixel_area_m2), 6),
            "polygon_count": len(kind_features),
        }

    change_features: list[dict] = []
    change_summary: dict[str, object]
    if previous_semantic_class_raster is None:
        change_summary = {
            "status": "unavailable",
            "reason": (
                "No earlier aligned semantic survey was supplied; new buildings, new roads, "
                "and structure expansion were not inferred from a single date."
            ),
        }
    else:
        previous = read_raster_evidence(
            previous_semantic_class_raster, band=1, require_projected=True
        )
        assert_aligned(current, previous)
        previous_valid = semantic_valid_mask(previous)
        previous_building = previous_valid & np.isin(previous.data, list(building_ids))
        previous_road = previous_valid & np.isin(previous.data, list(road_ids))
        new_masks = {
            "new_building_or_structure_expansion": (
                valid & np.isin(current.data, list(building_ids)) & ~previous_building
            ),
            "new_road_path": valid & np.isin(current.data, list(road_ids)) & ~previous_road,
        }
        change_totals: dict[str, dict] = {}
        for kind, mask in new_masks.items():
            items = polygonize_mask(
                mask,
                current,
                properties={
                    "change": kind,
                    "interpretation": "semantic class present now and absent in earlier aligned survey",
                    "requires_human_review": True,
                },
                min_area_m2=min_polygon_area_m2,
            )
            change_features.extend(items)
            change_totals[kind] = {
                "area_m2": round(float(mask.sum()) * float(current.pixel_area_m2), 6),
                "polygon_count": len(items),
            }
        change_summary = {
            "status": "available",
            "previous_semantic_class_raster": previous.path,
            "totals": change_totals,
        }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    boundary_path = geo.write_geojson(
        output / "cadastral_boundary.geojson",
        boundary_features,
        epsg=current.epsg,
        properties={
            "source": str(cadastral_boundary_geojson),
            "boundary_tolerance_m": boundary_tolerance_m,
        },
    )
    encroachment_path = geo.write_geojson(
        output / "encroachment_findings.geojson",
        vector_features,
        epsg=current.epsg,
        properties={
            "units": {"area": "m2"},
            "warning": (
                "A semantic overlap is a review finding, not a legal determination of encroachment."
            ),
        },
    )
    change_path = geo.write_geojson(
        output / "land_change_findings.geojson",
        change_features,
        epsg=current.epsg,
        properties={"comparison": change_summary},
    )
    encoded = np.full(current.data.shape, CLASS_NODATA, dtype=np.uint16)
    encoded[current.valid] = 0
    encoded[masks["building_encroachment"]] = 1
    encoded[masks["road_path_encroachment"]] = 2
    raster_path = write_aligned_raster(
        output / "encroachment_classes.tif",
        encoded,
        current,
        dtype="uint16",
        nodata=CLASS_NODATA,
        categorical=True,
        tags={
            "ODK_ANALYSIS": "cadastral_encroachment_review",
            "ODK_CLASS_0": "none",
            "ODK_CLASS_1": "building_encroachment",
            "ODK_CLASS_2": "road_path_encroachment",
            "ODK_BOUNDARY_TOLERANCE_M": boundary_tolerance_m,
        },
    )
    summary_path = write_summary(
        output / "encroachment.json",
        {
            "status": "complete",
            "analysis": "semantic_overlap_against_imported_cadastral_boundary",
            "source": {
                "semantic_class_raster": current.path,
                "cadastral_boundary": str(cadastral_boundary_geojson),
                "epsg": current.epsg,
            },
            "units": {"area": "m2", "boundary_tolerance": "m"},
            "boundary_tolerance_m": boundary_tolerance_m,
            "totals": totals,
            "total_encroachment_review_area_m2": round(
                float(all_encroachment.sum()) * float(current.pixel_area_m2), 6
            ),
            "temporal_change": change_summary,
            "interpretation": (
                "Observed semantic building/road pixels outside the imported boundary are "
                "review findings only. Survey accuracy, cadastral authority, registration, "
                "and applicable law must be checked before any legal conclusion."
            ),
            "artifacts": {
                "boundary": str(boundary_path),
                "findings": str(encroachment_path),
                "temporal_change": str(change_path),
                "class_raster": str(raster_path),
            },
        },
    )
    return IndiaPackResult(
        summary_path,
        (str(boundary_path), str(encroachment_path), str(change_path), str(raster_path)),
    )
