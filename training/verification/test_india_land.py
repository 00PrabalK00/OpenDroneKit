from __future__ import annotations

import json

import numpy as np

from core.india_geospatial import IndiaPackRefused
from core.india_land import detect_cadastral_encroachment, extract_land_gis
from training.verification.conftest import feature, polygon, write_geojson, write_raster


SCHEMA = {0: "background", 1: "building", 2: "road", 3: "vegetation", 4: "water"}


def test_land_gis_extracts_real_georeferenced_semantic_classes(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint16)
    classes[1:3, 1:3] = 1
    classes[7:9, :] = 2
    classes[3:7, 3:7] = 3
    classes[0, 8:10] = 4
    source = write_raster(tmp_path / "semantic.tif", classes, nodata=65535)

    result = extract_land_gis(source, SCHEMA, tmp_path / "land", min_polygon_area_m2=0)
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    vectors = json.loads(open(result.artifact_paths[0], encoding="utf-8").read())

    assert summary["categories"]["building"]["area_m2"] == 4.0
    assert summary["categories"]["road_path"]["area_m2"] == 20.0
    assert summary["parcel_boundary"]["status"] == "unavailable"
    assert {item["properties"]["layer"] for item in vectors["features"]} >= {
        "building", "road_path", "vegetation", "water", "survey_extent"
    }


def test_encroachment_uses_imported_boundary_and_aligned_previous_survey(tmp_path):
    previous = np.zeros((10, 10), dtype=np.uint16)
    previous[2:4, 2:4] = 1
    current = previous.copy()
    current[2:4, 7:9] = 1
    current[8, 7:10] = 2
    earlier_path = write_raster(tmp_path / "earlier.tif", previous, nodata=65535)
    current_path = write_raster(tmp_path / "current.tif", current, nodata=65535)
    boundary = write_geojson(
        tmp_path / "parcel.geojson", [feature(polygon(0, 0, 6, 10), parcel_id="P-1")]
    )

    result = detect_cadastral_encroachment(
        current_path,
        SCHEMA,
        boundary,
        tmp_path / "encroachment",
        previous_semantic_class_raster=earlier_path,
        min_polygon_area_m2=0,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())

    assert summary["totals"]["building_encroachment"]["area_m2"] == 4.0
    assert summary["totals"]["road_path_encroachment"]["area_m2"] == 3.0
    assert summary["temporal_change"]["status"] == "available"
    assert summary["temporal_change"]["totals"]["new_building_or_structure_expansion"]["area_m2"] == 4.0
    assert "before any legal conclusion" in summary["interpretation"]


def test_metric_land_analysis_refuses_unreferenced_raster(tmp_path):
    source = write_raster(
        tmp_path / "unreferenced.tif", np.ones((3, 3), dtype=np.uint8), epsg=None
    )
    try:
        extract_land_gis(source, {1: "building"}, tmp_path / "out")
    except IndiaPackRefused as exc:
        assert "no CRS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Unreferenced metric analysis was not refused")
