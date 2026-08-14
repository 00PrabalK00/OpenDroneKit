"""The ONNX parity gate: what it must let through, and what it must not.

The gate exists to stop an export that loads but computes something different from the
model whose metrics we publish. It was originally a flat absolute tolerance, which
turned out to measure the wrong thing for a detector: box coordinates run to the image
size, so an absolute 1e-3 there is a demand for agreement to one part in 600,000, while
the same 1e-3 applied to a 0..1 class probability is exactly right.

The tolerance is therefore scaled to each value's magnitude. These tests pin both
halves of that: the coordinate-scale noise a correct fp32 export really does produce
must pass, and a graph that genuinely computes something else must still fail. The
second half is the one that matters -- a tolerance loosened until everything passes
would be worse than no gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.export_onnx import PARITY_ATOL, PARITY_RTOL, parity_violation


class TestAgreementPasses:
    def test_identical_outputs_are_a_perfect_match(self):
        values = np.random.default_rng(0).normal(size=(1, 10, 8400)).astype(np.float32)
        assert parity_violation(values, values) == 0.0

    def test_fp32_reassociation_noise_on_pixel_coordinates_passes(self):
        """The real case: YOLO11x at 640 px differed by 0.0011 on a 633 px coordinate.

        That is a box shifted by a thousandth of a pixel -- the same box.
        """
        coordinates = np.full((1, 4, 8400), 633.2, dtype=np.float64)
        exported = coordinates + 0.0011
        assert parity_violation(coordinates, exported) <= 1.0

    def test_probabilities_are_still_held_to_the_absolute_tolerance(self):
        """A class score is where 1e-3 is the right scale, and it stays enforced."""
        scores = np.full((1, 6, 8400), 0.5, dtype=np.float64)
        just_inside = scores + PARITY_ATOL * 0.9
        just_outside = scores + PARITY_ATOL * 3.0

        assert parity_violation(scores, just_inside) <= 1.0
        assert parity_violation(scores, just_outside) > 1.0


class TestDisagreementFails:
    def test_a_graph_computing_something_else_entirely_fails(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=(1, 10, 100))
        unrelated = rng.normal(size=(1, 10, 100))
        assert parity_violation(reference, unrelated) > 1.0

    def test_a_box_off_by_a_whole_pixel_fails(self):
        """A thousandth of a pixel is noise; a whole pixel is a different box."""
        coordinates = np.full((1, 4, 8400), 320.0, dtype=np.float64)
        assert parity_violation(coordinates, coordinates + 1.0) > 1.0

    def test_a_single_bad_value_among_correct_ones_fails(self):
        """The check is a maximum, not an average: one wrong output is enough."""
        reference = np.full((1, 10, 1000), 0.5)
        candidate = reference.copy()
        candidate[0, 3, 500] = 0.9
        assert parity_violation(reference, candidate) > 1.0

    def test_a_sign_flip_fails(self):
        values = np.full((1, 4, 100), 42.0)
        assert parity_violation(values, -values) > 1.0

    def test_an_all_zero_export_fails_against_a_real_output(self):
        """The failure mode where a graph loads and returns nothing useful."""
        reference = np.full((1, 6, 1000), 0.7)
        assert parity_violation(reference, np.zeros_like(reference)) > 1.0


class TestScaling:
    def test_the_allowance_grows_with_magnitude(self):
        """The whole point: 1e-3 means something different at 0.5 than at 633."""
        small = np.array([[0.5]])
        large = np.array([[633.0]])
        offset = 0.01

        assert parity_violation(small, small + offset) > 1.0
        assert parity_violation(large, large + offset) <= 1.0

    def test_the_relative_term_is_what_admits_large_values(self):
        value = np.array([[1000.0]])
        # Just inside the relative allowance, ignoring the atol floor.
        assert parity_violation(value, value + PARITY_RTOL * 1000.0 * 0.9) <= 1.0
        assert parity_violation(value, value + PARITY_RTOL * 1000.0 * 20) > 1.0

    def test_the_violation_is_reported_as_a_multiple_of_tolerance(self):
        """A number that says how far out it is, not merely that it is out."""
        values = np.zeros((1, 4))
        doubled = values + PARITY_ATOL * 2.0
        assert parity_violation(values, doubled) == pytest.approx(2.0, rel=1e-6)


class TestRealExport:
    def test_the_installed_crack_model_recorded_a_passing_parity_check(self):
        """The model actually shipped must carry evidence it matched its checkpoint."""
        import json
        from pathlib import Path

        manifest = Path("models/manifests/model_provenance.json")
        if not manifest.exists():
            pytest.skip("No provenance manifest in this checkout.")

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else payload.get("models", [])
        checked = [e for e in entries if isinstance(e, dict) and "parity_ok" in e]
        if not checked:
            pytest.skip("No provenance entry records a parity result yet.")
        assert all(entry["parity_ok"] for entry in checked)
