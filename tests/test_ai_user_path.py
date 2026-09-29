"""The path a user takes to the models, driven through the real Api.

An audit ran every detector directly and found the models mostly worked. The product
around them did not: Run AI failed with a TypeError before touching an image, a
mixed-size image set crashed coverage validation, a reconstruction failure threw away a
finished detection run, detections never became findings, solar and metal reported
invented confidences, six installed models had no caller, and the model manager read a
path relative to wherever the app happened to be launched.

Each class here pins one of those, through the same call the UI makes, on a real
photograph (tests/fixtures/concrete_cracked_public_domain.jpg) and the real installed
weights. Tests that need weights skip where the gitignored weights are absent; set
ODK_MODEL_WEIGHTS_DIR to a folder holding them to run them from another checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from core.models import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"


def _installed(key: str) -> bool:
    path = resolve_model_path(key)
    return bool(path) and path.is_file()


def needs(*keys: str):
    missing = [k for k in keys if not _installed(k)]
    return pytest.mark.skipif(bool(missing), reason=f"weights not on this machine: {missing}")


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    created = instance.create_project("ai path", str(tmp_path / "project"))
    assert created["ok"], created
    return instance


@pytest.fixture
def mixed_size_dataset(tmp_path):
    """Two frames of different size cut from one real photograph."""
    image = cv2.imread(str(FIXTURE))
    assert image is not None
    folder = tmp_path / "images"
    folder.mkdir()
    cv2.imwrite(str(folder / "a_full.jpg"), image)
    h, w = image.shape[:2]
    cv2.imwrite(str(folder / "b_crop.jpg"), image[: h * 2 // 3, : w // 2])
    return folder


def _wait(api, job_id, timeout=900):
    started = time.time()
    while time.time() - started < timeout:
        job = api.job_status(job_id)["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(1)
    raise AssertionError("job did not finish")


class TestTheModelManagerFindsTheRegistryFromAnyDirectory:
    def test_launched_elsewhere_it_still_sees_the_registry(self, api, tmp_path, monkeypatch):
        """Api.model_status read Path('models/model_registry.json'): 0 of 0 from a shortcut."""
        monkeypatch.chdir(tmp_path)
        status = api.model_status()
        assert status["ok"]
        assert status["total"] >= 15

    def test_it_says_which_detectors_are_heuristic(self, api):
        status = api.model_status()
        by_name = {d["detector"]: d for d in status["detectors"]}
        # No metal model exists anywhere, so this must read heuristic on every machine.
        assert by_name["metal"]["backend"] == "heuristic"
        assert "metal" in status["heuristic_detectors"]
        # The solar routing key has no weights; the row says so and says why.
        assert by_name["solar"]["backend"] == ("model" if _installed("solar_defect_detector")
                                               else "heuristic")
        assert by_name["solar"]["status_note"]


class TestHeuristicsCarryNoScore:
    """Area-derived 'confidence' of 0.35..0.95 read as a model score and meant nothing."""

    def test_solar_heuristic_regions_have_no_confidence(self):
        from core.detection import detect_solar_defects, load_image

        result = detect_solar_defects(load_image(FIXTURE), use_model=False)
        assert result.model_used == "heuristic"
        assert result.detections, "the heuristic should still report its regions"
        assert all(hit.confidence is None for hit in result.detections)
        assert all(d["confidence"] is None for d in result.to_summary()["detections"])
        assert result.model_key == "" and result.model_sha256 == ""

    def test_structural_heuristic_regions_have_no_confidence(self):
        from core.detection import detect_structural_defects, load_image

        result = detect_structural_defects(load_image(FIXTURE), use_model=False)
        assert result.model_used == "heuristic"
        assert all(hit.confidence is None for hit in result.detections)

    def test_metal_says_it_is_a_heuristic(self):
        from core.detection import detect_metal_defects, load_image

        result = detect_metal_defects(load_image(FIXTURE))
        assert result.model_used == "heuristic"
        assert result.model_key == "" and result.model_sha256 == ""


class TestSolarReachesItsInstalledModel:
    @needs("solar_cell_defect_detector")
    def test_an_onnx_detection_kind_runs_the_model(self):
        """kind 'onnx_detection' was rejected, so the EL model fell back to the heuristic."""
        from core.detection import detect_solar_defects, load_image

        result = detect_solar_defects(load_image(FIXTURE), model_key="solar_cell_defect_detector")
        assert result.model_used == "onnx:solar_cell_defect_detector.onnx"
        # Identity was computed and then dropped on the floor; now it is on the result.
        assert result.model_key == "solar_cell_defect_detector"
        assert len(result.model_sha256) == 64
        # A concrete wall is not an EL cell: whatever the model says, its answer stands,
        # and nothing it returns is a confidence-free heuristic region.
        assert all(hit.confidence is not None for hit in result.detections)


class TestMixedSizeImagesDoNotCrashCoverage:
    def test_phase_shift_between_different_sizes_is_no_measurement(self):
        from core.coverage_validation import _phase_shift

        shift, response = _phase_shift(np.zeros((40, 60), np.uint8), np.zeros((50, 60), np.uint8))
        assert response == 0.0 and not shift.any()

    def test_validation_runs_on_a_mixed_set(self, mixed_size_dataset, tmp_path):
        from core.coverage_validation import CoverageValidator

        images = sorted(mixed_size_dataset.iterdir())
        payload = CoverageValidator().validate(image_paths=images, output_dir=tmp_path / "cov")
        assert len(payload["images"]) == 2


class TestAReconstructionFailureKeepsTheDetectionRun:
    def test_summary_and_report_are_written(self, mixed_size_dataset, tmp_path):
        """COLMAP 'could not register any images' used to abort after detection."""
        from core.pipeline import PipelineConfig, StructuralFaultPipeline

        class Refuses:
            def reconstruct(self, **_):
                raise RuntimeError("COLMAP could not register any images.")

        pipeline = StructuralFaultPipeline(PipelineConfig(
            crack_presence_gate=False, run_structural_models=False))
        pipeline.reconstructor = Refuses()
        # The crack model is slow on CPU and irrelevant to this property.
        import core.pipeline as module
        original = module.detect_cracks
        module.detect_cracks = lambda image, **kw: original(image, use_model=False, **kw)
        try:
            result = pipeline.run(mixed_size_dataset, tmp_path / "out")
        finally:
            module.detect_cracks = original
        summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))
        assert summary["reconstruction"]["status"] == "failed"
        assert "could not register" in summary["reconstruction"]["error"]
        assert len(summary["images"]) == 2
        report = Path(result.report_markdown_path).read_text(encoding="utf-8")
        assert "Reconstruction FAILED" in report


class TestThePipelineDoesNotRunDetectorsOnTheWrongAsset:
    def test_solar_and_metal_follow_the_asset(self):
        from core.pipeline import PipelineConfig, StructuralFaultPipeline

        concrete = StructuralFaultPipeline(PipelineConfig())
        assert not concrete._solar_enabled() and not concrete._metal_enabled()
        solar = StructuralFaultPipeline(PipelineConfig(structure_type="solar"))
        assert solar._solar_enabled()
        steel = StructuralFaultPipeline(PipelineConfig(material_type="steel"))
        assert steel._metal_enabled()
        forced = StructuralFaultPipeline(PipelineConfig(run_metal_detection=True))
        assert forced._metal_enabled()

    def test_the_forecast_says_it_is_a_simulation(self):
        from core.health_scoring import _forecast_growth_metric
        from core.pipeline import PipelineConfig, _forecast_assumptions

        stated = _forecast_assumptions(PipelineConfig())
        assert stated["basis"] == "what_if_simulation"
        assert stated["assumed_inputs"]["sigma_nominal_mpa"] == 35.0
        # And a simulated scenario does not move the structure's health score.
        growth, meta = _forecast_growth_metric({**stated, "metrics": [
            {"crack_ratio": 0.0}, {"crack_ratio": 0.5}]})
        assert growth == 0.0 and meta["excluded_what_if_simulation"] == 1.0


@needs("structural_multiclass_detector", "crack_segmentation", "crack_presence_classifier")
class TestRunAIFromTheApi:
    """The Start / Run AI button: Api.run_pipeline, then the Findings panel's call."""

    # One run for the class: the pipeline takes minutes on CPU, and each test reads a
    # different part of the same result. The review test mutates findings, so it runs last.
    @pytest.fixture(scope="class")
    def finished(self, tmp_path_factory):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        root = tmp_path_factory.mktemp("run_ai")
        api = Api(AppSession(ProjectStore(root / "odk.db")))
        assert api.create_project("ai path", str(root / "project"))["ok"]
        image = cv2.imread(str(FIXTURE))
        folder = root / "images"
        folder.mkdir()
        cv2.imwrite(str(folder / "a_full.jpg"), image)
        h, w = image.shape[:2]
        cv2.imwrite(str(folder / "b_crop.jpg"), image[: h * 2 // 3, : w // 2])
        assert api.set_active_dataset(str(folder))["ok"]
        started = api.run_pipeline({"engine": "auto", "profile": "fast"})
        assert started["ok"], started
        job = _wait(api, started["job_id"])
        assert job["status"] == "done", job.get("error")
        return api, job["result"]

    def test_the_run_completes_and_writes_its_summary(self, finished):
        _, result = finished
        summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
        assert [row["image"] for row in summary["images"]] == ["a_full.jpg", "b_crop.jpg"]
        for row in summary["images"]:
            assert row["metal_model_used"] == "not_run"
            assert row["solar_model_used"] == "not_run"
            assert row["crack_width_measured"] is False
            assert "crack_max_width_px" not in row
            assert row["crack_presence"]["model_key"] == "crack_presence_classifier"
        assert summary["forecast"]["basis"] == "what_if_simulation"
        assert result["reconstruction_status"] in ("ok", "failed")

    def test_model_detections_become_reviewable_findings(self, finished):
        api, result = finished
        found = api.find_annotations()["annotations"]
        assert result["findings_created"] == len(found) > 0
        for finding in found:
            assert finding["created_by"] == "model"
            assert finding["model_key"] == "structural_multiclass_detector"
            assert len(finding["model_sha256"]) == 64
            assert 0.0 < finding["confidence"] <= 1.0
            # Awaiting review, and out of the report until someone accepts it.
            assert finding["status"] == "in_review"
            assert finding["include_in_report"] is False

    def test_reviewing_a_model_finding_keeps_the_claim(self, finished):
        api, _ = finished
        finding = api.find_annotations()["annotations"][0]
        rejected = api.review_finding(finding["id"], "reject", "inspector", "shadow, not a crack")
        assert rejected["ok"]
        assert rejected["status"] == "dismissed" and rejected["include_in_report"] is False
        assert rejected["confidence"] == finding["confidence"]
        assert rejected["review_history"][-1]["comment"] == "shadow, not a crack"
        accepted = api.review_finding(finding["id"], "accept", "lead")
        assert accepted["include_in_report"] is True
        assert [h["reviewer"] for h in accepted["review_history"]] == ["inspector", "lead"]


class TestEveryInstalledModelIsReachable:
    @pytest.mark.parametrize("key", [
        "crack_presence_classifier", "solar_cell_defect_detector",
        "solar_thermal_anomaly_classifier", "rail_corridor_segmentation",
        "rail_obstacle_detector", "corrosion_severity_segmentation",
        "structural_multiclass_detector", "crack_segmentation",
    ])
    def test_run_model_answers_with_provenance(self, api, key):
        if not _installed(key):
            pytest.skip(f"{key} weights not on this machine")
        result = api.run_model(key, str(FIXTURE))
        assert result["ok"], result.get("error")
        assert result["model_key"] == key
        assert len(result["model_sha256"]) == 64
        assert result["model_used"].startswith("onnx:")
        assert result["scope"], "the registry's scope note must travel with the result"
        assert Path(result["overlay_path"]).is_file() or result["kind"] == "onnx_classification"

    @needs("crack_presence_classifier")
    def test_the_crack_classifier_flags_a_cracked_photograph(self, api):
        result = api.run_model("crack_presence_classifier", str(FIXTURE))
        assert result["output"]["any_crack"] is True
        assert result["output"]["flagged_tile_count"] >= 1

    def test_a_model_with_no_weights_is_refused_with_its_reason(self, api):
        if _installed("solar_defect_detector"):
            pytest.skip("the solar routing model is installed here")
        result = api.run_model("solar_defect_detector", str(FIXTURE))
        assert not result["ok"]
        assert "Routing target" in result["error"]

    @needs("shared_semantic_landcover")
    def test_the_landcover_model_is_sent_to_its_own_run(self, api):
        result = api.run_model("shared_semantic_landcover", str(FIXTURE))
        assert not result["ok"] and "orthomosaic" in result["error"]
        assert not api.run_landcover()["ok"], "no orthomosaic yet, so it must refuse"

    @needs("corrosion_severity_segmentation")
    def test_corrosion_grading_has_an_api(self, api):
        result = api.grade_corrosion(str(FIXTURE))
        assert result["ok"]
        assert result["output"]["labels"] == ["good", "fair", "poor", "severe"]
        assert result["model_key"] == "corrosion_severity_segmentation"


class TestPrelabelFromTheDesktop:
    @needs("structural_multiclass_detector")
    def test_a_detector_prelabel_becomes_findings(self, api):
        result = api.prelabel_image(str(FIXTURE))
        assert result["ok"], result.get("error")
        found = api.find_annotations()["annotations"]
        assert len(found) == result["count"]
        assert all(f["model_key"] == "structural_multiclass_detector" for f in found)
        assert all(isinstance(f["confidence"], float) for f in found)

    def test_a_model_without_per_region_scores_is_refused(self, api):
        result = api.prelabel_image(str(FIXTURE), "crack_segmentation")
        assert not result["ok"]
        assert "confidence" in result["error"]

    def test_heuristic_output_can_never_become_a_finding(self, tmp_path):
        from app import desktop_ops

        with pytest.raises(ValueError, match="model identity"):
            desktop_ops.findings_from_detections(
                tmp_path, "1", "a.jpg", model_used="heuristic", model_key="",
                model_sha256="", detections=[{"label": "hotspot", "confidence": None,
                                              "bbox_xyxy": [0, 0, 5, 5]}])


class TestClassifierPreprocessingIsRecorded:
    def test_each_classifier_says_how_to_feed_it(self):
        from core.models import load_registry

        rows = load_registry()["models"]
        for key, row in rows.items():
            if row.get("kind") == "onnx_classification":
                assert {"normalisation", "output"} <= set(row.get("preprocessing") or {}), key


@needs("shared_semantic_landcover")
class TestTheSemanticEngineReportsTheDeviceThatRan:
    def test_a_cuda_request_that_fell_back_says_cpu(self):
        """ONNX Runtime silently falls back to CPU when cuBLAS is missing."""
        from core.semantic_engine import ONNXSemanticPredictor

        predictor = ONNXSemanticPredictor(resolve_model_path("shared_semantic_landcover"),
                                          device="cuda")
        providers = predictor._session.get_providers()
        expected = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        assert predictor.device == expected
