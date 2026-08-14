from __future__ import annotations

import json

import numpy as np
import rasterio

from core.india_agriculture import (
    ReflectanceBandCalibration,
    analyse_canopy_cover,
    compute_vegetation_indices,
    count_plant_instances,
    create_stress_zones,
)
from training.verification.conftest import write_raster


def _calibrations(include_red_edge=True):
    values = {
        "red": ReflectanceBandCalibration(1, 0.0001, 0, "camera-panel-v1"),
        "green": ReflectanceBandCalibration(2, 0.0001, 0, "camera-panel-v1"),
        "nir": ReflectanceBandCalibration(4, 0.0001, 0, "camera-panel-v1"),
    }
    if include_red_edge:
        values["red_edge"] = ReflectanceBandCalibration(3, 0.0001, 0, "camera-panel-v1")
    return values


def test_indices_use_calibrated_bands_and_mark_missing_band_unavailable(tmp_path):
    red = np.full((4, 4), 2000, dtype=np.uint16)
    green = np.full((4, 4), 3000, dtype=np.uint16)
    red_edge = np.full((4, 4), 4000, dtype=np.uint16)
    nir = np.full((4, 4), 6000, dtype=np.uint16)
    source = write_raster(tmp_path / "multispectral.tif", np.stack([red, green, red_edge, nir]))

    complete = compute_vegetation_indices(source, _calibrations(), tmp_path / "indices")
    summary = json.loads(open(complete.summary_path, encoding="utf-8").read())
    with rasterio.open(summary["indices"]["NDVI"]["path"]) as dataset:
        ndvi = dataset.read(1)
    assert np.allclose(ndvi, 0.5)
    assert summary["indices"]["NDRE"]["status"] == "available"
    assert summary["indices"]["GNDVI"]["status"] == "available"

    partial = compute_vegetation_indices(
        source, _calibrations(include_red_edge=False), tmp_path / "partial"
    )
    partial_summary = json.loads(open(partial.summary_path, encoding="utf-8").read())
    assert partial_summary["indices"]["NDRE"]["status"] == "unavailable"
    assert "No proxy was substituted" in partial_summary["indices"]["NDRE"]["reason"]


def test_canopy_cover_uses_real_semantic_raster(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint8)
    classes[:5, :] = 1
    classes[5:8, :] = 2
    classes[8, :] = 3
    classes[9, :] = 4
    source = write_raster(tmp_path / "crop_classes.tif", classes, nodata=255)
    result = analyse_canopy_cover(
        source,
        {0: "background", 1: "crop", 2: "soil", 3: "unwanted vegetation", 4: "water"},
        tmp_path / "canopy",
        min_region_area_m2=0,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert summary["canopy_area_m2"] == 50.0
    assert summary["canopy_cover_percent"] == 55.555556
    assert summary["bare_or_potential_missing_area"]["interpretation"] == "bare soil; not a confirmed missing plant"


def test_stress_zones_require_and_preserve_crop_sensor_scope(tmp_path):
    red = np.full((3, 3), 2000, dtype=np.uint16)
    green = np.full((3, 3), 3000, dtype=np.uint16)
    red_edge = np.full((3, 3), 3000, dtype=np.uint16)
    nir = np.array(
        [[2000, 4000, 8000], [2000, 4000, 8000], [2000, 4000, 8000]], dtype=np.uint16
    )
    source = write_raster(tmp_path / "stress_source.tif", np.stack([red, green, red_edge, nir]))
    indices = compute_vegetation_indices(source, _calibrations(), tmp_path / "indices")
    index_summary = json.loads(open(indices.summary_path, encoding="utf-8").read())
    result = create_stress_zones(
        index_summary["indices"]["NDVI"]["path"],
        tmp_path / "stress",
        severe_below=0.2,
        moderate_below=0.5,
        validation_scope="wheat, sensor-X, vegetative stage, 20m AGL",
        min_region_area_m2=0,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert summary["zones"]["severe_stress"]["area_m2"] == 3.0
    assert summary["zones"]["moderate_stress"]["area_m2"] == 3.0
    assert summary["zones"]["reference_or_healthy_range"]["area_m2"] == 3.0
    assert "do not identify a biological cause" in summary["interpretation"]


def test_plant_count_counts_connected_instances_without_inventing_missing_or_health(tmp_path):
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    mask[8:10, 2:4] = 1
    source = write_raster(
        tmp_path / "plants.tif", mask, nodata=0, tags={"ODK_INSTANCE_KIND": "plant"}
    )
    result = count_plant_instances(
        source,
        tmp_path / "count",
        validation_scope="mango crowns, orchard-A, 3cm GSD",
        min_instance_area_m2=1.0,
    )
    summary = json.loads(open(result.summary_path, encoding="utf-8").read())
    points = json.loads(open(result.artifact_paths[1], encoding="utf-8").read())
    assert summary["count"] == 3
    assert summary["missing_plants"]["status"] == "unavailable"
    assert summary["health_categories"]["status"] == "unavailable"
    assert all(item["geometry"]["type"] == "Point" for item in points["features"])
