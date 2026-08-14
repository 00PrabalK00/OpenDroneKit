'''Export a trained shared DINOv2/UPerNet checkpoint and runtime manifest.'''

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.shared_semantic_model import build_dinov2_vitb14_upernet


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_manifest(
    *,
    onnx_path: str | Path,
    model_key: str,
    model_version: str,
    schema: dict[str, Any],
    validation_metrics: dict[str, Any],
    inference: dict[str, Any],
    training_origin: str,
    architecture: str = 'DINOv2 ViT-B/14 + OpenDroneKit UPerNet',
) -> dict[str, Any]:
    numeric_metrics = {
        str(key): float(value)
        for key, value in validation_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not numeric_metrics:
        raise ValueError('Runtime manifest requires numeric validation metrics.')
    return {
        'manifest_schema_version': 1,
        'schema': schema,
        'model': {
            'key': model_key,
            'version': model_version,
            'architecture': architecture,
            'checkpoint_sha256': sha256_file(onnx_path),
            'schema_id': str(schema['id']),
            'schema_version': str(schema['version']),
            'task_trained': True,
            'training_origin': training_origin,
            'validation_metrics': numeric_metrics,
        },
        'inference': {
            'tile_size': int(inference.get('tile_size', 518)),
            'overlap': int(inference.get('overlap', 126)),
            'device': str(inference.get('device', 'cuda')),
            'allow_cpu': bool(inference.get('allow_cpu', False)),
            'max_cpu_pixels': int(inference.get('max_cpu_pixels', 4_000_000)),
            'min_polygon_area_m2': float(inference.get('min_polygon_area_m2', 1.0)),
            'polygonize_background': bool(inference.get('polygonize_background', False)),
            'input_bands': list(inference.get('input_bands', (1, 2, 3))),
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    }


def export_shared_semantic(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    model_key: str,
    model_version: str,
    dinov2_source: str | None = None,
    source: str | None = None,
    opset: int = 17,
    parity_tolerance: float = 1e-4,
) -> dict[str, Any]:
    import onnxruntime as ort

    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) or {}
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    schema = dict(checkpoint.get('schema') or {})
    if schema != dict(config.get('schema') or {}):
        raise ValueError('Checkpoint and export config semantic schemas do not match.')
    classes = list(schema.get('classes') or [])
    architecture = dict(config.get('architecture') or {})
    resolved_dinov2_source = str(
        dinov2_source or architecture.get('encoder_source') or 'facebookresearch/dinov2'
    )
    resolved_source_type = str(
        source or architecture.get('encoder_source_type') or 'github'
    )
    model = build_dinov2_vitb14_upernet(
        architecture['encoder_checkpoint'],
        len(classes),
        dinov2_source=resolved_dinov2_source,
        source=resolved_source_type,
    )
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    tile_size = int(config.get('inference', {}).get('tile_size', 518))
    example = torch.rand(1, 3, tile_size, tile_size, dtype=torch.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch_output = model(example).cpu().numpy()
    torch.onnx.export(
        model,
        example,
        output,
        input_names=['images'],
        output_names=['logits'],
        opset_version=opset,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    onnx_output = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]
    max_abs_diff = float(np.abs(torch_output - onnx_output).max())
    if max_abs_diff > parity_tolerance:
        raise RuntimeError(
            f'ONNX parity failed: max absolute difference {max_abs_diff} exceeds '
            f'{parity_tolerance}.'
        )

    validation = dict(checkpoint.get('validation_metrics') or {})
    training_origin = (
        f'OpenDroneKit shared semantic training; corpus_sha256='
        f'{checkpoint.get("corpus_sha256", "unknown")}'
    )
    manifest = build_runtime_manifest(
        onnx_path=output,
        model_key=model_key,
        model_version=model_version,
        schema=schema,
        validation_metrics=validation,
        inference=dict(config.get('inference') or {}),
        training_origin=training_origin,
    )
    manifest_path = output.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = {
        'kind': 'onnx_shared_semantic',
        'onnx_path': str(output),
        'manifest_path': str(manifest_path),
        'sha256': manifest['model']['checkpoint_sha256'],
        'bytes': output.stat().st_size,
        'opset': opset,
        'input_shape': list(example.shape),
        'output_shape': list(onnx_output.shape),
        'max_abs_diff': max_abs_diff,
        'parity_tolerance': parity_tolerance,
    }
    output.with_suffix('.export.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export trained shared semantic ONNX.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--dinov2-source')
    parser.add_argument('--source', choices=('github', 'local'))
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--parity-tolerance', type=float, default=1e-4)
    args = parser.parse_args(argv)
    report = export_shared_semantic(
        args.config,
        args.checkpoint,
        args.output,
        model_key=args.model_key,
        model_version=args.model_version,
        dinov2_source=args.dinov2_source,
        source=args.source,
        opset=args.opset,
        parity_tolerance=args.parity_tolerance,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
