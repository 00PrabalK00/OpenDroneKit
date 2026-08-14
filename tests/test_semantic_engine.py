'''Shared semantic inference is tiled, georeferenced and provenance-safe.'''

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
from rasterio.transform import from_origin

from core.processing_runs import (
    STATUS_COMPLETED,
    create_processing_run,
    run_pipeline_stage,
    validate_pipeline_inputs,
)
from core.semantic_engine import (
    CLASS_NODATA,
    SemanticClass,
    SemanticInferenceConfig,
    SemanticInferenceRefused,
    SemanticModelMetadata,
    SemanticSchema,
    load_semantic_manifest,
    run_semantic_inference,
)
from core.workflows import get_workflow_template, validate_workflow_readiness


def schema() -> SemanticSchema:
    return SemanticSchema(
        id='test_landcover',
        version='1.0.0',
        classes=(
            SemanticClass(0, 'background', (0, 0, 0), background=True),
            SemanticClass(4, 'building', (220, 60, 60)),
        ),
    )


def model(*, task_trained: bool = True) -> SemanticModelMetadata:
    return SemanticModelMetadata(
        key='test_semantic',
        version='2026.08.14',
        architecture='DINOv2 ViT-B/14 + OpenDroneKit UPerNet',
        checkpoint_sha256='1' * 64,
        schema_id='test_landcover',
        schema_version='1.0.0',
        task_trained=task_trained,
        training_origin='synthetic test fixture',
        validation_metrics={'mean_iou': 0.75},
    )


def write_orthomosaic(path: Path, *, width: int = 73, height: int = 61) -> Path:
    image = np.zeros((3, height, width), dtype=np.uint8)
    image[0, :, width // 2:] = 255
    image[1, :, :width // 2] = 60
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=width,
        height=height,
        count=3,
        dtype='uint8',
        crs='EPSG:32643',
        transform=from_origin(500_000, 3_000_000, 0.5, 0.5),
    ) as target:
        target.write(image)
    return path


def write_manifest(path: Path, *, task_trained: bool = True) -> Path:
    payload = {
        'manifest_schema_version': 1,
        'schema': schema().to_dict(),
        'model': model(task_trained=task_trained).to_dict(),
        'inference': {
            'tile_size': 32,
            'overlap': 8,
            'device': 'cpu',
            'allow_cpu': True,
            'max_cpu_pixels': 20_000,
            'min_polygon_area_m2': 0.1,
        },
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


class ThresholdPredictor:
    device = 'cpu'

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        building = (tile[0] - 0.5) * 12.0
        return np.stack((-building, building))


class EdgeBiasedPredictor:
    device = 'cpu'

    def predict(self, tile: np.ndarray) -> np.ndarray:
        height, width = tile.shape[1:]
        building = np.full((height, width), 4.0, dtype=np.float32)
        building[:4] = -4.0
        building[-4:] = -4.0
        building[:, :4] = -4.0
        building[:, -4:] = -4.0
        return np.stack((-building, building))


class WrongCheckpointPredictor(ThresholdPredictor):
    checkpoint_sha256 = 'f' * 64


def cpu_config(**updates) -> SemanticInferenceConfig:
    values = {
        'tile_size': 32,
        'overlap': 8,
        'device': 'cpu',
        'allow_cpu': True,
        'max_cpu_pixels': 20_000,
        'min_polygon_area_m2': 0.1,
    }
    values.update(updates)
    return SemanticInferenceConfig(**values)


class TestSemanticContracts:
    def test_schema_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match='ids must be unique'):
            SemanticSchema(
                'bad',
                '1',
                (
                    SemanticClass(0, 'background', (0, 0, 0)),
                    SemanticClass(0, 'water', (0, 0, 200)),
                ),
            )

    def test_manifest_round_trips_schema_and_model(self, tmp_path):
        manifest = write_manifest(tmp_path / 'semantic.json')
        loaded_schema, loaded_model, inference = load_semantic_manifest(manifest)
        assert loaded_schema == schema()
        assert loaded_model == model()
        assert inference['tile_size'] == 32

    def test_foundation_checkpoint_is_refused(self, tmp_path):
        with pytest.raises(SemanticInferenceRefused, match='foundation initializer'):
            run_semantic_inference(
                tmp_path / 'missing.tif',
                tmp_path / 'out',
                schema=schema(),
                model=model(task_trained=False),
                predictor=ThresholdPredictor(),
                config=cpu_config(),
            )

    def test_cpu_requires_explicit_opt_in(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        with pytest.raises(SemanticInferenceRefused, match='disabled by default'):
            run_semantic_inference(
                ortho,
                tmp_path / 'out',
                schema=schema(),
                model=model(),
                predictor=ThresholdPredictor(),
                config=cpu_config(allow_cpu=False),
            )

    def test_predictor_checkpoint_must_match_manifest(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        with pytest.raises(SemanticInferenceRefused, match='SHA-256'):
            run_semantic_inference(
                ortho,
                tmp_path / 'out',
                schema=schema(),
                model=model(),
                predictor=WrongCheckpointPredictor(),
                config=cpu_config(),
            )

    def test_cpu_pixel_limit_is_enforced(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        with pytest.raises(SemanticInferenceRefused, match='above the CPU limit'):
            run_semantic_inference(
                ortho,
                tmp_path / 'out',
                schema=schema(),
                model=model(),
                predictor=ThresholdPredictor(),
                config=cpu_config(max_cpu_pixels=100),
            )


class TestTiledSemanticPackage:
    def test_outputs_preserve_georeferencing_classes_and_provenance(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        predictor = ThresholdPredictor()
        package = run_semantic_inference(
            ortho,
            tmp_path / 'semantic',
            schema=schema(),
            model=model(),
            predictor=predictor,
            config=cpu_config(),
        )

        assert predictor.calls > 1
        assert all(Path(path).is_file() for path in package.artifact_paths())
        assert not list((tmp_path / 'semantic').glob('.semantic_*.dat'))

        with rasterio.open(ortho) as source, rasterio.open(package.class_map_path) as classes:
            values = classes.read(1)
            assert classes.crs == source.crs
            assert classes.transform == source.transform
            assert classes.nodata == CLASS_NODATA
            assert classes.tags()['ODK_MODEL_KEY'] == 'test_semantic'
            assert np.all(values[:, :36] == 0)
            assert np.all(values[:, 36:] == 4)

        with rasterio.open(package.confidence_path) as confidence:
            assert confidence.crs.to_epsg() == 32643
            assert float(confidence.read(1).min()) > 0.99

        polygons = json.loads(Path(package.polygons_path).read_text(encoding='utf-8'))
        assert polygons['crs']['properties']['name'].endswith('32643')
        assert len(polygons['features']) == 1
        feature = polygons['features'][0]
        assert feature['properties']['class_name'] == 'building'
        assert feature['properties']['model_version'] == '2026.08.14'
        assert feature['properties']['area_m2'] == pytest.approx(37 * 61 * 0.25)

        manifest = json.loads(Path(package.manifest_path).read_text(encoding='utf-8'))
        by_name = {item['class_name']: item for item in manifest['classes']}
        assert manifest['model']['task_trained'] is True
        assert by_name['building']['pixel_count'] == 37 * 61
        assert manifest['polygon_count'] == 1

    def test_overlap_blending_suppresses_internal_tile_edge_bias(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif', width=80, height=64)
        package = run_semantic_inference(
            ortho,
            tmp_path / 'semantic',
            schema=schema(),
            model=model(),
            predictor=EdgeBiasedPredictor(),
            config=cpu_config(overlap=16),
        )
        with rasterio.open(package.class_map_path) as classes:
            values = classes.read(1)
        assert np.all(values[8:-8, 8:-8] == 4)


class TestSemanticWorkflow:
    def test_workflow_names_all_required_inputs(self, tmp_path):
        template = get_workflow_template('semantic_mapping')
        assert template.processing_stages == ['semantic_segmentation']
        assert template.required_inputs == [
            'orthomosaic', 'semantic_model', 'semantic_model_manifest'
        ]
        missing = validate_workflow_readiness({'root_dir': str(tmp_path)}, 'semantic_mapping')
        assert missing.ready is False
        assert len(missing.missing_required) == 3

    def test_pipeline_readiness_rejects_untrained_manifest(self, tmp_path):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        onnx = tmp_path / 'model.onnx'
        onnx.write_bytes(b'placeholder')
        manifest = write_manifest(tmp_path / 'manifest.json', task_trained=False)
        run = create_processing_run(
            tmp_path,
            'project',
            '',
            'semantic_mapping',
            config={
                'orthomosaic_path': str(ortho),
                'semantic_model_path': str(onnx),
                'semantic_model_manifest_path': str(manifest),
            },
        )
        readiness = validate_pipeline_inputs(tmp_path, run.id)
        assert readiness.ok is False
        assert any('untrained foundation initializer' in issue for issue in readiness.issues)

    def test_processing_stage_emits_the_four_semantic_artifacts(self, tmp_path, monkeypatch):
        ortho = write_orthomosaic(tmp_path / 'ortho.tif')
        onnx = tmp_path / 'model.onnx'
        onnx.write_bytes(b'placeholder')
        manifest = write_manifest(tmp_path / 'manifest.json')

        import core.semantic_engine as semantic_module

        monkeypatch.setattr(
            semantic_module,
            'ONNXSemanticPredictor',
            lambda *args, **kwargs: ThresholdPredictor(),
        )
        run = create_processing_run(
            tmp_path,
            'project',
            '',
            'semantic_mapping',
            config={
                'orthomosaic_path': str(ortho),
                'semantic_model_path': str(onnx),
                'semantic_model_manifest_path': str(manifest),
            },
        )
        assert validate_pipeline_inputs(tmp_path, run.id).ok is True
        result = run_pipeline_stage(tmp_path, run.id, 'semantic_segmentation')
        assert result.status == STATUS_COMPLETED
        assert len(result.artifacts) == 4
        assert all(Path(path).is_file() for path in result.artifacts)


def test_upernet_decoder_restores_requested_output_size():
    torch = pytest.importorskip('torch')
    from training.shared_semantic_model import UPerNetDecoder

    decoder = UPerNetDecoder([16, 32, 64, 128], num_classes=3, channels=32).eval()
    features = [
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
    ]
    with torch.no_grad():
        logits = decoder(features, output_size=(64, 64))
    assert tuple(logits.shape) == (1, 3, 64, 64)
