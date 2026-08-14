"""Regression tests for fabrications that were removed.

Each of these once shipped: synthetic geometry presented as measurement, an outbound
request that was ignored, a heuristic reported as a model. They are cheap to
reintroduce by accident, so each has a test that fails loudly if it returns.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.reconstruction import RECONSTRUCTION_PROFILES, CustomDroneReconstructor


class TestNoSyntheticDensification:
    """`_densify_point_cloud` cloned points with Gaussian jitter and called it MVS."""

    @pytest.mark.parametrize("profile", sorted(RECONSTRUCTION_PROFILES))
    def test_point_cloud_is_returned_unchanged(self, profile):
        reconstructor = CustomDroneReconstructor(profile=profile)
        points = np.random.default_rng(0).random((500, 3)).astype(np.float32)
        colors = np.zeros((500, 3), dtype=np.uint8)

        out_points, out_colors = reconstructor._densify_point_cloud(points, colors)

        assert len(out_points) == len(points), "point count must not be inflated"
        assert np.array_equal(out_points, points)
        assert np.array_equal(out_colors, colors)

    def test_no_gaussian_noise_in_the_module(self):
        import core.reconstruction as module

        source = inspect.getsource(module)
        assert "rng.normal" not in source, "synthetic point jitter has returned"

    @pytest.mark.parametrize("profile", sorted(RECONSTRUCTION_PROFILES))
    def test_profiles_carry_no_densification_multiplier(self, profile):
        assert "mvs_multiplier" not in RECONSTRUCTION_PROFILES[profile]
        assert "mvs_jitter_scale" not in RECONSTRUCTION_PROFILES[profile]


class TestNoSilentNetworkCall:
    """Cloud mode POSTed the imagery path off-machine, then discarded the reply."""

    def test_cloud_request_sends_nothing(self, tmp_path):
        reconstructor = CustomDroneReconstructor(
            profile="fast_preview",
            execution_mode="cloud",
            cloud_endpoint="http://example.invalid/submit",
        )
        request_path, response_path, warnings = reconstructor._request_cloud_run(
            image_dir="some/dir", output_dir=tmp_path, image_count=12
        )

        assert response_path == "", "a response implies a request was actually sent"
        assert any("no data left this machine" in w for w in warnings)
        assert any("was not contacted" in w for w in warnings)

        import json
        from pathlib import Path

        recorded = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert recorded["status"] == "not_submitted"

    def test_no_http_client_in_the_cloud_path(self):
        source = inspect.getsource(CustomDroneReconstructor._request_cloud_run)
        assert "urlopen" not in source
        assert "urllib" not in source

    def test_module_does_not_import_an_http_client(self):
        import core.reconstruction as module

        source = inspect.getsource(module)
        assert "import urllib" not in source, "the reconstruction engine is offline-only"


class TestEngineLabelsItselfAccurately:
    def test_custom_engine_does_not_claim_mvs(self):
        """The label must be the emitted value, not merely mentioned in a comment."""
        import re

        import core.reconstruction as module

        source = inspect.getsource(module)
        assignments = re.findall(r'"pipeline"\s*:\s*"([^"]+)"', source)
        assert assignments, "expected the digital twin to declare a pipeline label"
        assert "sfm_mvs_custom" not in assignments, (
            "this engine performs no multi-view stereo, so it must not claim MVS"
        )
        assert "sfm_sparse_custom" in assignments


class TestOrthomosaicReportsGuessedOffsets:
    """A failed correlation substituted 35% of frame width and said nothing."""

    def test_uncorrelatable_frames_are_counted(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        rng = np.random.default_rng(0)
        paths = []
        for index in range(4):
            noise = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
            target = tmp_path / f"{index}.png"
            cv2.imwrite(str(target), noise)
            paths.append(target)

        reconstructor = CustomDroneReconstructor(profile="fast_preview")
        reconstructor._build_orthomosaic(paths, tmp_path / "mosaic.png")

        assert reconstructor._mosaic_guessed_offsets > 0, (
            "pure noise cannot correlate, so substituted offsets must be recorded"
        )


class TestDetectionReportsWhatItActuallyUsed:
    def test_heuristic_path_is_labelled_heuristic(self):
        cv2 = pytest.importorskip("cv2")
        from core.detection import detect_cracks

        image = np.full((256, 256, 3), 200, dtype=np.uint8)
        cv2.line(image, (20, 40), (230, 180), (40, 40, 40), 2)

        result = detect_cracks(image, use_model=False)
        assert result.model_used == "heuristic", "the classical path must never claim a model"

    def test_model_used_is_carried_on_the_result(self):
        from core.detection import CrackDetectionResult

        assert "model_used" in CrackDetectionResult.__dataclass_fields__


class TestFindingNothingIsAnAnswer:
    """A trained detector reporting a clean surface must not be overruled.

    The structural path used to treat "the model returned no detections" as though the
    model had failed, and fall through to the classical heuristic. On a sound wall that
    turns a correct empty result into a set of invented findings, attributed to a run
    that says a model was available. Silence from a model that ran is data.
    """

    def _clean_surface(self):
        cv2 = pytest.importorskip("cv2")
        # Flat, featureless: nothing for either the model or the heuristic to latch on
        # to legitimately.
        return np.full((640, 640, 3), 175, dtype=np.uint8), cv2

    def test_an_empty_model_result_is_not_replaced_by_the_heuristic(self):
        from core.detection import detect_structural_defects
        from core.models import model_status

        image, _ = self._clean_surface()
        if not model_status("structural_multiclass_detector").get("exists"):
            pytest.skip("No structural model installed in this checkout.")

        result = detect_structural_defects(image)
        assert result.model_used.startswith("onnx:"), (
            "a model that ran and found nothing must still be credited with running, "
            f"got {result.model_used!r}"
        )

    def test_the_heuristic_is_still_labelled_when_the_model_is_switched_off(self):
        from core.detection import detect_structural_defects

        image, _ = self._clean_surface()
        result = detect_structural_defects(image, use_model=False)
        assert result.model_used == "heuristic"

    def test_a_real_defect_image_produces_model_detections(self):
        """The other half: silence must mean silence, not a broken inference path."""
        import glob

        cv2 = pytest.importorskip("cv2")
        from core.detection import detect_structural_defects
        from core.models import model_status

        if not model_status("structural_multiclass_detector").get("exists"):
            pytest.skip("No structural model installed in this checkout.")

        paths = sorted(glob.glob("training/data/prepared/structural_det/test/images/*.jpg"))
        if not paths:
            pytest.skip("Prepared CODEBRIM test images are not present in this checkout.")

        found = 0
        for path in paths[:12]:
            result = detect_structural_defects(cv2.imread(path))
            assert result.model_used.startswith("onnx:")
            found += len(result.detections)
        assert found > 0, "the installed detector found nothing across 12 defect images"
