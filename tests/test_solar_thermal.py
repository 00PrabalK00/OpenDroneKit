"""Measured RGB/thermal registration and module association evidence.

The fixtures are files on disk: a readable RGB frame, radiometric raw counts
with camera calibration, and projected module polygons. No detector, raster,
registration, or temperature loader is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.india_geospatial import IndiaPackRefused
from core.india_thermal import (
    ThermalRegistration,
    associate_module_temperatures,
    score_rgb_thermal_registration,
)
from core.thermal import ABSOLUTE_ZERO_C, Calibration, kelvin_to_raw


METADATA = {
    "PlanckR1": 17096.453,
    "PlanckR2": 0.046642166,
    "PlanckB": 1428.0,
    "PlanckF": 1.0,
    "PlanckO": -342.0,
    "Emissivity": 1.0,
    "ReflectedApparentTemperature": 20.0,
    "AtmosphericTransmission": 1.0,
}


def _raw_counts(celsius: np.ndarray) -> np.ndarray:
    calibration = Calibration.from_metadata(METADATA)
    return kelvin_to_raw(celsius - ABSOLUTE_ZERO_C, calibration)


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("rasterio")

    rgb = root / "rgb.png"
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:, :, 1] = np.arange(100, dtype=np.uint8)[None, :]
    assert cv2.imwrite(str(rgb), frame)

    temperatures = np.full((40, 50), 25.0, dtype=np.float64)
    temperatures[2:8, 22:30] = 30.0
    temperatures[12:18, 2:10] = 33.0
    temperatures[12:18, 12:20] = 33.0
    temperatures[14, 14] = 45.0
    temperatures[12:18, 22:30] = 33.0
    thermal = root / "thermal-radiometric.json"
    thermal.write_text(json.dumps({
        "raw": _raw_counts(temperatures).tolist(),
        "metadata": METADATA,
        "georeferencing": {
            "epsg": 32643,
            "transform": [1.0, 0.0, 500000.0, 0.0, -1.0, 2000000.0],
        },
    }), encoding="utf-8")

    points = root / "observed-tie-points.json"
    points.write_text(json.dumps({"points": [
        {"rgb": [4, 4], "thermal": [2, 2]},
        {"rgb": [94, 4], "thermal": [47, 2]},
        {"rgb": [94, 74], "thermal": [47, 37]},
        {"rgb": [4, 74], "thermal": [2, 37]},
        {"rgb": [50, 40], "thermal": [25, 20]},
        {"rgb": [30, 60], "thermal": [15, 30]},
    ]}), encoding="utf-8")
    return rgb, thermal, points


def _write_modules(path: Path) -> Path:
    rectangles = [
        ("A-1", "A", 2, 2, 10, 8),
        ("A-2", "A", 12, 2, 20, 8),
        ("A-3", "A", 22, 2, 30, 8),
        ("B-1", "B", 2, 12, 10, 18),
        ("B-2", "B", 12, 12, 20, 18),
        ("B-3", "B", 22, 12, 30, 18),
    ]
    features = []
    for module_id, string_id, col0, row0, col1, row1 in rectangles:
        rgb_polygon = [
            [2 * col0, 2 * row0], [2 * col1, 2 * row0],
            [2 * col1, 2 * row1], [2 * col0, 2 * row1],
        ]
        world_polygon = [[
            [500000 + col0, 2000000 - row0],
            [500000 + col1, 2000000 - row0],
            [500000 + col1, 2000000 - row1],
            [500000 + col0, 2000000 - row1],
            [500000 + col0, 2000000 - row0],
        ]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": world_polygon},
            "properties": {
                "module_id": module_id,
                "string_id": string_id,
                "rgb_polygon_px": rgb_polygon,
            },
        })
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32643"}},
        "features": features,
    }), encoding="utf-8")
    return path


class TestRgbThermalRegistration:
    def test_redundant_operator_tie_points_report_components_not_confidence(self, tmp_path):
        rgb, thermal, points = _write_inputs(tmp_path)
        registration = score_rgb_thermal_registration(
            rgb, thermal, points, validated_by="survey-team",
        )

        assert registration.method == "homography_ransac_operator_tie_point_self_consistency"
        assert registration.tie_point_count == 6
        assert registration.inlier_count == 6
        assert registration.inlier_fraction == pytest.approx(1.0)
        assert registration.residual_px < 0.001
        assert registration.coverage_fraction > 0.7
        assert registration.ransac_reproj_threshold_px == pytest.approx(3.0)
        assert registration.acceptance_rmse_px == pytest.approx(1.5)
        assert registration.evidence_kind == (
            "operator_tie_point_self_consistency_not_image_feature_matching"
        )
        assert "quality_score" not in registration.to_dict()
        assert registration.rgb_to_thermal_homography is not None
        assert registration.rgb_sha256 and registration.thermal_sha256

    def test_spatially_clustered_points_do_not_pass_quality_gate(self, tmp_path):
        rgb, thermal, _ = _write_inputs(tmp_path)
        clustered = [
            {"rgb": [2, 2], "thermal": [1, 1]},
            {"rgb": [10, 2], "thermal": [5, 1]},
            {"rgb": [10, 10], "thermal": [5, 5]},
            {"rgb": [2, 10], "thermal": [1, 5]},
            {"rgb": [6, 2], "thermal": [3, 1]},
            {"rgb": [6, 10], "thermal": [3, 5]},
        ]
        with pytest.raises(IndiaPackRefused, match="spatial coverage"):
            score_rgb_thermal_registration(
                rgb, thermal, clustered, validated_by="survey-team",
                minimum_coverage_fraction=0.2,
            )

    def test_four_points_are_refused_as_exactly_determined(self, tmp_path):
        rgb, thermal, points = _write_inputs(tmp_path)
        rows = json.loads(points.read_text(encoding="utf-8"))["points"][:4]
        with pytest.raises(IndiaPackRefused, match="exactly determine"):
            score_rgb_thermal_registration(
                rgb, thermal, rows, validated_by="survey-team",
            )

    def test_ransac_selection_and_rmse_acceptance_are_separate_gates(self, tmp_path):
        rgb, thermal, _ = _write_inputs(tmp_path)
        noisy = [
            {"rgb": [4, 4], "thermal": [2.4, 1.7]},
            {"rgb": [94, 4], "thermal": [46.7, 2.5]},
            {"rgb": [94, 74], "thermal": [47.5, 36.6]},
            {"rgb": [4, 74], "thermal": [1.6, 37.4]},
            {"rgb": [50, 40], "thermal": [25.5, 19.6]},
            {"rgb": [30, 60], "thermal": [14.6, 30.5]},
            {"rgb": [70, 20], "thermal": [35.4, 9.6]},
            {"rgb": [20, 30], "thermal": [9.6, 15.5]},
        ]
        with pytest.raises(IndiaPackRefused, match="RMSE"):
            score_rgb_thermal_registration(
                rgb, thermal, noisy, validated_by="survey-team",
                maximum_rmse_px=0.1, ransac_reproj_threshold_px=2.0,
            )


class TestModuleTemperatureAssociation:
    def test_module_hotspots_have_radiometric_temperature_backed_severity(self, tmp_path):
        rgb, thermal, points = _write_inputs(tmp_path)
        registration = score_rgb_thermal_registration(
            rgb, thermal, points, validated_by="survey-team",
        )
        modules = _write_modules(tmp_path / "modules.geojson")

        result = associate_module_temperatures(
            thermal, registration, modules, tmp_path / "result",
            warning_delta_c=4.0, critical_delta_c=8.0, hot_cell_delta_c=6.0,
            threshold_basis="project-configured review convention THERMAL-QA-01",
        )

        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        associated = json.loads(
            (tmp_path / "result" / "module_temperatures.geojson").read_text(encoding="utf-8")
        )
        candidates = json.loads(
            (tmp_path / "result" / "thermal_review_candidates.geojson").read_text(
                encoding="utf-8"
            )
        )
        by_id = {row["properties"]["module_id"]: row["properties"]
                 for row in associated["features"]}

        assert summary["module_count"] == 6
        assert summary["module_hotspot_count"] == 4
        assert summary["counts"]["hot_cell_candidate"] == 1
        assert summary["counts"]["module_deviation_candidate"] == 1
        assert summary["counts"]["string_deviation_candidate"] == 3
        assert summary["severity_threshold_basis"].endswith("THERMAL-QA-01")
        assert summary["severity_classification"] == (
            "configured_convention_not_equipment_diagnosis"
        )
        assert len(candidates["features"]) == 4
        assert by_id["B-2"]["max_c"] == pytest.approx(45.0, abs=0.1)
        assert by_id["B-2"]["measured_severity_delta_c"] == pytest.approx(12.0, abs=0.1)
        assert by_id["B-2"]["severity"] == "critical"
        assert by_id["A-3"]["severity"] == "warning"
        assert by_id["A-3"]["module_delta_c"] == pytest.approx(5.0, abs=0.1)
        assert all(row["properties"]["area_m2"] == pytest.approx(48.0)
                   for row in associated["features"])
        assert all(row["properties"]["review_state"] == "review_required"
                   for row in candidates["features"])
        assert all(kind.endswith("_candidate")
                   for row in candidates["features"]
                   for kind in row["properties"]["finding_types"])
        forbidden = {"crack", "diode_failure", "pid", "electrical_fault"}
        assert not forbidden.intersection(
            kind for row in candidates["features"]
            for kind in row["properties"]["finding_types"]
        )


class TestSolarThermalRefusals:
    def test_rendered_palette_pair_is_not_registered_as_temperature(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        rgb = tmp_path / "rgb.png"
        palette = tmp_path / "thermal-palette.jpg"
        assert cv2.imwrite(str(rgb), np.zeros((20, 20, 3), dtype=np.uint8))
        assert cv2.imwrite(str(palette), np.zeros((10, 10, 3), dtype=np.uint8))
        points = [
            {"rgb": [1, 1], "thermal": [1, 1]},
            {"rgb": [18, 1], "thermal": [8, 1]},
            {"rgb": [18, 18], "thermal": [8, 8]},
            {"rgb": [1, 18], "thermal": [1, 8]},
        ]

        with pytest.raises(IndiaPackRefused, match="radiometric counts"):
            score_rgb_thermal_registration(
                rgb, palette, points, validated_by="survey-team",
            )

    def test_module_association_refuses_an_unregistered_pair(self, tmp_path):
        _, thermal, _ = _write_inputs(tmp_path)
        modules = _write_modules(tmp_path / "modules.geojson")
        descriptive_only = ThermalRegistration(
            method="operator_note", rgb_width=100, rgb_height=80,
            residual_px=0.5, validated_by="operator",
        )

        with pytest.raises(IndiaPackRefused, match="scored RGB-to-thermal"):
            associate_module_temperatures(
                thermal, descriptive_only, modules, tmp_path / "result",
                warning_delta_c=4, critical_delta_c=8, hot_cell_delta_c=6,
                threshold_basis="test convention",
            )

    def test_module_association_refuses_an_unscored_matrix(self, tmp_path):
        _, thermal, _ = _write_inputs(tmp_path)
        modules = _write_modules(tmp_path / "modules.geojson")
        unscored = ThermalRegistration(
            method="unchecked_matrix", rgb_width=100, rgb_height=80,
            thermal_width=50, thermal_height=40, residual_px=0.0,
            validated_by="operator",
            rgb_to_thermal_homography=(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        )

        with pytest.raises(IndiaPackRefused, match="scored RGB-to-thermal"):
            associate_module_temperatures(
                thermal, unscored, modules, tmp_path / "result",
                warning_delta_c=4, critical_delta_c=8, hot_cell_delta_c=6,
                threshold_basis="test convention",
            )
