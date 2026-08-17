'''Shared semantic exports create the exact honest runtime manifest contract.'''

from __future__ import annotations

import json

import pytest

from core.semantic_engine import load_semantic_manifest

# The training modules import torch at module scope, and torch is not a runtime
# dependency of this project -- the runtime loads ONNX through cv2.dnn. Without this the
# import raises during collection and takes the WHOLE suite down with exit code 2, which
# is how CI stayed red while every individual test was fine. CI installs the CPU wheel so
# these still run there; the skip is for machines that only ever run the shipped code.
pytest.importorskip("torch", reason="training-only dependency; the runtime uses cv2.dnn")

from training.export_shared_semantic import build_runtime_manifest, sha256_file  # noqa: E402


def test_runtime_manifest_uses_onnx_hash_and_numeric_metrics(tmp_path):
    onnx = tmp_path / 'model.onnx'
    onnx.write_bytes(b'full trained encoder and decoder')
    schema = {
        'id': 'shared',
        'version': '1.0.0',
        'classes': [
            {'id': 0, 'name': 'background', 'color_rgb': [0, 0, 0], 'background': True},
            {'id': 1, 'name': 'building', 'color_rgb': [255, 0, 0]},
        ],
    }
    manifest = build_runtime_manifest(
        onnx_path=onnx,
        model_key='shared_semantic',
        model_version='1.0.0',
        schema=schema,
        validation_metrics={
            'mean_iou': 0.71,
            'pixel_accuracy': 0.92,
            'per_class_iou': {'building': 0.65},
        },
        inference={'tile_size': 518, 'overlap': 126},
        training_origin='site-separated fixture',
    )
    assert manifest['model']['checkpoint_sha256'] == sha256_file(onnx)
    assert manifest['model']['task_trained'] is True
    assert manifest['model']['validation_metrics'] == {
        'mean_iou': 0.71,
        'pixel_accuracy': 0.92,
    }
    assert manifest['inference']['mean'] == [0.485, 0.456, 0.406]

    path = tmp_path / 'model.manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    loaded_schema, loaded_model, inference = load_semantic_manifest(path)
    assert loaded_schema.id == 'shared'
    assert loaded_model.checkpoint_sha256 == sha256_file(onnx)
    assert inference['tile_size'] == 518


def test_runtime_manifest_refuses_missing_numeric_evidence(tmp_path):
    onnx = tmp_path / 'model.onnx'
    onnx.write_bytes(b'model')
    with pytest.raises(ValueError, match='numeric validation metrics'):
        build_runtime_manifest(
            onnx_path=onnx,
            model_key='shared',
            model_version='1',
            schema={'id': 's', 'version': '1', 'classes': [1, 2]},
            validation_metrics={'per_class': {}},
            inference={},
            training_origin='fixture',
        )
