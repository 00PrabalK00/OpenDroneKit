from __future__ import annotations

import json

import numpy as np

from core.india_construction import CONSTRUCTION_SCHEMA, measure_approved_design_progress
from training.verification.conftest import feature, polygon, write_geojson, write_raster


def test_construction_schema_covers_the_registry_contract():
    assert {item.name for item in CONSTRUCTION_SCHEMA.classes} == {
        "background",
        "building",
        "unfinished_building",
        "road",
        "bare_soil",
        "vegetation",
        "water",
        "concrete",
        "excavation",
        "stockpile",
        "construction_material",
        "equipment",
    }


def test_approved_design_progress_measures_observed_surface_not_contract_completion(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint8)
    classes[2:6, 2:4] = 1
    classes[7:9, 7:9] = 1
    raster = write_raster(tmp_path / "construction.tif", classes, nodata=255)
    design = write_geojson(
        tmp_path / "approved.geojson",
        [feature(polygon(2, 2, 6, 6), element_id="B1", expected_class="building")],
    )
    result = measure_approved_design_progress(
        raster,
        {0: "background", 1: "building"},
        design,
        tmp_path / "progress",
        min_region_area_m2=0,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert summary["observed_surface_coverage_percent"] == 50.0
    assert summary["contractual_completion_percent"]["status"] == "unavailable"
    assert summary["review_areas"]["approved_not_observed_m2"] == 8.0
    assert summary["review_areas"]["observed_outside_design_m2"] == 4.0
