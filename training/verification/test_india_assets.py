from __future__ import annotations

import json

import numpy as np

from core.india_assets import (
    create_power_inspection_package,
    create_rail_inspection_package,
    create_road_condition_package,
    create_solar_module_inventory,
)
from core.india_geospatial import IndiaPackRefused
from training.verification.conftest import feature, line, point, polygon, write_geojson, write_raster


def _validated_properties(**values):
    return {
        "evidence_source": "validated_model",
        "model_key": "fixture-model",
        "model_version": "1.0",
        "confidence": 0.92,
        **values,
    }


def test_road_condition_uses_explicit_centerline_for_metric_distance(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint8)
    classes[4:7, :] = 1
    raster = write_raster(tmp_path / "road.tif", classes, nodata=255)
    centerline = write_geojson(
        tmp_path / "centerline.geojson", [feature(line(((0.5, 5.5), (9.5, 5.5))), road_id="R1")]
    )
    findings = write_geojson(
        tmp_path / "road_findings.geojson",
        [
            feature(point(2.5, 5.5), **_validated_properties(defect_type="pothole", severity="severe")),
            feature(point(6.5, 5.5), **_validated_properties(defect_type="crack", severity="minor")),
        ],
    )
    result = create_road_condition_package(
        raster, {0: "background", 1: "road"}, centerline, findings, tmp_path / "roads"
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert summary["surveyed_distance_m"] == 9.0
    assert summary["road_surface_area_m2"] == 30.0
    assert summary["findings_by_type"] == {"crack": 1, "pothole": 1}
    assert summary["off_road_review_count"] == 0


def test_power_and_rail_enforce_capture_geometry_and_map_real_vectors(tmp_path):
    centerline = write_geojson(
        tmp_path / "corridor.geojson", [feature(line(((0, 5), (10, 5))), corridor="C1")]
    )
    power = write_geojson(
        tmp_path / "power.geojson",
        [feature(point(2, 5), **_validated_properties(asset_class="insulator", finding="corrosion", finding_validated=True))],
    )
    rail = write_geojson(
        tmp_path / "rail.geojson",
        [feature(point(4, 5), **_validated_properties(asset_class="track", finding="obstacle", finding_validated=False))],
    )
    power_result = create_power_inspection_package(
        power,
        centerline,
        tmp_path / "power_pack",
        capture_geometry="close_range_oblique",
        validation_scope="insulator model, 2cm GSD, oblique captures",
    )
    rail_result = create_rail_inspection_package(
        rail,
        centerline,
        tmp_path / "rail_pack",
        capture_geometry="corridor_nadir_oblique",
        validation_scope="UAV-RSOD rail head, 3cm GSD",
    )
    power_summary = json.loads(open(power_result.summary_path, encoding="utf-8").read())
    rail_summary = json.loads(open(rail_result.summary_path, encoding="utf-8").read())
    assert power_summary["surveyed_distance_m"] == 10.0
    assert power_summary["assets_by_class"] == {"insulator": 1}
    assert rail_summary["review_candidate_count"] == 1

    try:
        create_power_inspection_package(
            power,
            centerline,
            tmp_path / "bad_capture",
            capture_geometry="high_altitude_nadir",
            validation_scope="insulator model",
        )
    except IndiaPackRefused as exc:
        assert "capture geometry" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Inappropriate component capture geometry was accepted")


def test_solar_inventory_counts_modules_and_only_calls_layout_gaps_missing(tmp_path):
    labels = np.zeros((12, 12), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[1:3, 4:6] = 2
    labels[1:3, 7:9] = 3
    source = write_raster(
        tmp_path / "modules.tif",
        labels,
        nodata=0,
        tags={
            "ODK_INSTANCE_KIND": "solar_module",
            "ODK_MODEL_KEY": "solar-yolo11l-seg",
            "ODK_MODEL_VERSION": "candidate-1",
        },
    )
    layout = write_geojson(
        tmp_path / "layout.geojson",
        [
            feature(polygon(1, 1, 3, 3), module_id="M1"),
            feature(polygon(4, 1, 6, 3), module_id="M2"),
            feature(polygon(7, 1, 9, 3), module_id="M3"),
            feature(polygon(10, 1, 12, 3), module_id="M4"),
        ],
    )
    findings = write_geojson(
        tmp_path / "solar_findings.geojson",
        [feature(point(5, 2), **_validated_properties(finding="damaged_module", module_id=2))],
    )
    result = create_solar_module_inventory(
        source,
        tmp_path / "solar",
        validation_scope="Duke UAV module masks, 2cm GSD",
        approved_layout_geojson=layout,
        findings_geojson=findings,
        array_join_distance_m=1,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert summary["module_count"] == 3
    assert summary["approved_layout"]["missing_module_review_count"] == 1
    assert summary["findings_by_type"] == {"damaged_module": 1}
