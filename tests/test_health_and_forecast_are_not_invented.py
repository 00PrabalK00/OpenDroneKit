"""Health scoring and the crack forecast report only what was measured.

Found by driving them on real inputs:

* evaluate_asset_health with ZERO defects on a concrete bridge reported a 2.95% one-year
  and 8.6% three-year failure probability. They came from a hard-coded baseline hazard
  table and hand-weighted multipliers, and the run report printed them as findings.
* the crack "forecast" extends each tip by a fixed 4 px per step with no time base, and
  its growth fed the health score, so the asset looked worse the further it was
  extrapolated. Its physics mode also produced a "failure probability" from an
  uncalibrated curve.
* with no critical-point file, criticality was set to the constants 0.35 / 0.50.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.crack_engine import CrackPropagationConfig, run_crack_propagation
from core.health_scoring import evaluate_asset_health
from core.propagation import ILLUSTRATIVE_NOTE, CrackPropagationForecaster
from core.reporting import write_markdown_report

cv2 = pytest.importorskip("cv2")


def _numbers_under_probability_keys(node, path=""):
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if "probab" in key.lower() and isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append(here)
            found += _numbers_under_probability_keys(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _numbers_under_probability_keys(value, f"{path}[{index}]")
    return found


def _health(**overrides):
    arguments = dict(
        asset_id="bridge-1", run_utc="2026-09-28T00:00:00Z", structure_type="bridge",
        material_type="concrete", image_rows=[{"crack_ratio": 0.0, "quality_ok": True}],
        forecast_payload={}, reconstruction_payload={}, coverage_payload={},
    )
    arguments.update(overrides)
    return evaluate_asset_health(**arguments)


class TestNoFailureProbabilityIsInvented:
    def test_a_sound_asset_has_no_failure_probability(self):
        health = _health()
        assert _numbers_under_probability_keys(health) == []
        assert health["probabilistic_failure"]["available"] is False
        assert "calibrated" in health["probabilistic_failure"]["reason"]

    def test_the_index_says_what_it_is(self):
        assert "Uncalibrated" in _health()["index_basis"]

    def test_the_report_does_not_print_a_percentage(self, tmp_path):
        target = tmp_path / "report.md"
        write_markdown_report(target, {"health_scoring": _health(), "images": []})
        text = target.read_text(encoding="utf-8")
        assert "Failure probability (1 year)" not in text
        assert "Failure probability: not computed" in text


class TestTheForecastIsNotScored:
    def _forecast(self, final_ratio: float):
        return {
            "basis": "illustrative_extrapolation",
            "metrics": [{"crack_ratio": 0.001}, {"crack_ratio": final_ratio}],
        }

    def test_extrapolated_growth_does_not_move_the_score(self):
        calm = _health(image_rows=[{"crack_ratio": 0.01}], forecast_payload=self._forecast(0.001))
        wild = _health(image_rows=[{"crack_ratio": 0.01}], forecast_payload=self._forecast(0.5))
        assert calm["integrity_score"] == wild["integrity_score"]
        assert calm["structural_risk_score"] == wild["structural_risk_score"]
        assert wild["components"]["growth_component"] is None
        assert "growth" in wild["components"]["not_assessed"]

    def test_growth_measured_between_surveys_is_scored(self):
        measured = {"basis": "measured_between_surveys",
                    "metrics": [{"crack_ratio": 0.001}, {"crack_ratio": 0.2}]}
        scored = _health(image_rows=[{"crack_ratio": 0.01}], forecast_payload=measured)
        assert scored["components"]["growth_component"] == pytest.approx(1.0)

    def test_criticality_is_not_invented_without_scores(self):
        health = _health(reconstruction_payload={"total_points": 1000, "crack_points": 40})
        assert health["components"]["criticality_component"] is None
        assert "criticality" in health["components"]["not_assessed"]


class TestTheForecastSaysWhatItIs:
    @pytest.fixture
    def crack(self):
        mask = np.zeros((300, 300), np.uint8)
        cv2.line(mask, (50, 150), (250, 150), 255, 2)
        return mask

    def test_physics_mode_gives_no_failure_probability(self, crack):
        _frames, metrics = CrackPropagationForecaster().forecast(crack, steps=3, mode="physics")
        assert all(m.failure_probability_horizon is None for m in metrics)
        assert metrics[-1].factor_of_safety > 0

    def test_the_standalone_engine_labels_its_output(self, tmp_path, crack):
        image = tmp_path / "wall.png"
        mask = tmp_path / "wall_mask.png"
        cv2.imwrite(str(image), np.full((300, 300, 3), 128, np.uint8))
        cv2.imwrite(str(mask), crack)
        result = run_crack_propagation(
            CrackPropagationConfig(image_path=str(image), mask_path=str(mask), steps=3),
            output_dir=tmp_path / "out",
        )
        assert ILLUSTRATIVE_NOTE in result.summary
        assert "Failure probability" not in result.summary
        assert all("failure_probability_horizon" not in p for p in result.critical_points)

    def test_the_report_calls_its_risk_level_illustrative(self, tmp_path, crack):
        from core.report_engine import _render_section

        image = tmp_path / "wall.png"
        mask = tmp_path / "wall_mask.png"
        cv2.imwrite(str(image), np.full((300, 300, 3), 128, np.uint8))
        cv2.imwrite(str(mask), crack)
        result = run_crack_propagation(
            CrackPropagationConfig(image_path=str(image), mask_path=str(mask), steps=3),
            output_dir=tmp_path / "out",
        )
        html_text = _render_section("crack_propagation", {"crack_runs": [result.to_dict()]})
        # The level comes from default stresses (35/12 MPa) and an assumed pixel size.
        assert "Illustrative risk level" in html_text
        assert "not measured" in html_text
        assert ">Risk level:" not in html_text
