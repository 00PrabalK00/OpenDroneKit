"""Shared anomaly intelligence over real georeferenced feature rasters."""

from __future__ import annotations

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from core.india_anomaly import detect_anomaly_candidates, fit_validated_baseline
from core.india_geospatial import IndiaPackRefused


def write_features(path, data, *, crs="EPSG:32643", schema="surface-v1"):
    values = np.asarray(data, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[2], height=values.shape[1],
        count=values.shape[0], dtype="float32", crs=crs,
        transform=from_origin(500000, 2000000, 1, 1), nodata=-9999.0,
    ) as target:
        target.write(values)
        if schema:
            target.update_tags(ODK_FEATURE_SCHEMA=schema)
    return path


class TestValidatedBaseline:
    def test_real_feature_rasters_produce_only_mapped_deviation_candidates(self, tmp_path):
        rows, columns = np.indices((20, 20))
        normal = 10.0 + ((rows + columns) % 4)
        first = write_features(tmp_path / "normal-a.tif", normal)
        second = write_features(tmp_path / "normal-b.tif", np.roll(normal, 1, axis=1))
        baseline_path = tmp_path / "baseline.json"
        baseline = fit_validated_baseline(
            [first, second], baseline_path,
            validated_by="two-person field review",
            validation_scope="concrete deck, dry season, camera profile A",
        )

        observed = normal.copy()
        observed[6:10, 8:12] = 30.0
        target = write_features(tmp_path / "observed.tif", observed)
        package = detect_anomaly_candidates(
            target, baseline_path, tmp_path / "result",
            threshold_sigma=5.0, minimum_area_m2=4.0,
        )

        candidates = json.loads(open(package.candidates_path, encoding="utf-8").read())
        assert candidates["crs"]["properties"]["name"] == "EPSG:32643"
        assert len(candidates["features"]) == 1
        finding = candidates["features"][0]["properties"]
        assert finding["label"] == "deviation_candidate"
        assert finding["interpretation"] == "review_required_not_a_named_defect"
        assert finding["area_m2"] == pytest.approx(16.0)
        assert finding["max_score_sigma"] > 5.0

        with rasterio.open(package.score_raster_path) as score:
            assert score.crs.to_epsg() == 32643
            assert score.tags()["ODK_BASELINE_ID"] == baseline.baseline_id
            assert score.tags()["ODK_NAMED_DEFECT_CLASS"] == "none"
            assert float(score.read(1)[7, 9]) > 5.0

        summary = json.loads(open(package.summary_path, encoding="utf-8").read())
        assert summary["named_defect_class"] is None
        assert summary["candidate_area_m2"] == pytest.approx(16.0)
        assert summary["baseline"]["validated_by"] == "two-person field review"
        assert any("not confirmed faults" in item for item in summary["limits"])

    def test_baseline_without_validation_scope_is_refused(self, tmp_path):
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        paths = [write_features(tmp_path / f"normal-{index}.tif", values + index)
                 for index in range(2)]
        with pytest.raises(IndiaPackRefused, match="validated_by and its validation_scope"):
            fit_validated_baseline(paths, tmp_path / "baseline.json",
                                   validated_by="reviewer", validation_scope="")


class TestGeospatialRefusals:
    def test_geographic_rasters_are_not_reported_in_square_metres(self, tmp_path):
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        paths = [write_features(tmp_path / f"degrees-{index}.tif", values + index,
                                crs="EPSG:4326") for index in range(2)]
        with pytest.raises(IndiaPackRefused, match="projected CRS"):
            fit_validated_baseline(paths, tmp_path / "baseline.json",
                                   validated_by="reviewer", validation_scope="test")

    def test_untagged_feature_bands_are_refused(self, tmp_path):
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        paths = [write_features(tmp_path / f"unknown-{index}.tif", values + index,
                                schema="") for index in range(2)]
        with pytest.raises(IndiaPackRefused, match="ODK_FEATURE_SCHEMA"):
            fit_validated_baseline(paths, tmp_path / "baseline.json",
                                   validated_by="reviewer", validation_scope="test")

