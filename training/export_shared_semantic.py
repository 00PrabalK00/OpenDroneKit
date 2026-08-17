'''Export a trained shared DINOv2/UPerNet checkpoint and runtime manifest.'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from core.semantic_engine import onnx_model_files, sha256_onnx_model
from training.shared_semantic_model import build_dinov2_vitb14_upernet


def sha256_file(path: str | Path) -> str:
    '''Digest a single file.

    Kept for callers that mean one file. An ONNX model is NOT one file once the exporter
    spills weights to external data, so the manifest uses sha256_onnx_model instead --
    see core.semantic_engine.
    '''
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parity(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    '''Check an exported graph against the torch model it came from.

    Parity is decided on the LABEL, not on the logit. Nothing downstream consumes a
    logit -- the semantic engine takes an argmax and polygonises the label raster -- so
    the question that decides whether an export is usable is "does any pixel change
    class", and the answer has to be no.

    A logits tolerance alone gets this wrong in both directions. Too tight and it rejects
    a correct export: a ViT-B/14 accumulates through twelve attention blocks in fp32, and
    ONNX Runtime fusing that arithmetic differently is expected rather than a defect --
    the first shared_semantic export failed a 1e-4 gate at 5.3e-4 while every one of
    1,073,296 pixels agreed on its class. Too loose and it passes an export whose labels
    have quietly moved, which is the failure that actually reaches an operator. So both
    are checked, and the logit bound is kept as a drift alarm rather than as the thing
    being proven.

    Raises RuntimeError on any label disagreement, or on a logit gap wide enough that
    today's agreement is luck.
    '''
    if not pairs:
        raise ValueError('Parity cannot be verified without at least one probe.')
    max_abs_diff = 0.0
    disagreeing = 0
    compared = 0
    for reference, candidate in pairs:
        if reference.shape != candidate.shape:
            raise RuntimeError(
                f'ONNX parity failed: the exported graph returns {candidate.shape} '
                f'where the model returns {reference.shape}.'
            )
        max_abs_diff = max(max_abs_diff, float(np.abs(reference - candidate).max()))
        differ = reference.argmax(axis=1) != candidate.argmax(axis=1)
        disagreeing += int(differ.sum())
        compared += int(differ.size)
    if disagreeing:
        raise RuntimeError(
            f'ONNX parity failed: {disagreeing} of {compared} pixels are assigned a '
            'different class by the exported graph. The runtime would produce a '
            'different map from the model that was measured.'
        )
    if max_abs_diff > tolerance:
        raise RuntimeError(
            f'ONNX parity failed: every pixel agrees on its class, but the largest '
            f'logit difference {max_abs_diff} exceeds {tolerance}. Labels match today '
            'by a margin this small only by luck.'
        )
    return {
        'max_abs_diff': max_abs_diff,
        'pixels_compared': compared,
        'label_disagreements': disagreeing,
        'probes': len(pairs),
    }


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
    # NaN and infinity are not measurements, and they are not JSON either.
    #
    # The shared_semantic checkpoint carried validation loss = NaN beside a perfectly
    # real mean_iou of 0.6128 -- the IoU comes from the confusion matrix and survives a
    # tile whose loss is undefined, typically one where every pixel is masked out. Two
    # things went wrong when that reached the manifest. A reader sees "loss: NaN" listed
    # among the validation metrics as though it were one, and `json.dump` writes the
    # bare token NaN, which Python re-reads happily and every other JSON parser rejects.
    #
    # Non-finite values are therefore dropped from the metrics and named separately, so
    # the fact that a loss came back undefined is visible rather than quietly erased.
    candidates = {
        str(key): float(value)
        for key, value in validation_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    numeric_metrics = {k: v for k, v in candidates.items() if math.isfinite(v)}
    non_finite = sorted(k for k in candidates if not math.isfinite(candidates[k]))
    if not numeric_metrics:
        raise ValueError('Runtime manifest requires numeric validation metrics.')
    return {
        'manifest_schema_version': 1,
        'schema': schema,
        'model': {
            'key': model_key,
            'version': model_version,
            'architecture': architecture,
            # The whole model, sidecars included. The runtime recomputes this the
            # same way and refuses a mismatch, so a graph-only digest here would
            # both under-identify the weights and fail to load.
            'checkpoint_sha256': sha256_onnx_model(onnx_path),
            'schema_id': str(schema['id']),
            'schema_version': str(schema['version']),
            'task_trained': True,
            'training_origin': training_origin,
            'validation_metrics': numeric_metrics,
            # Named, not silently dropped: a reader deciding whether to trust this model
            # should see that a metric came back undefined.
            'non_finite_metrics': non_finite,
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
    parity_tolerance: float = 1e-3,
    parity_inputs: int = 4,
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

    # Clear a previous export before writing this one.
    #
    # A large model writes its weights to a sidecar, and re-exporting over the top of one
    # does not reliably replace it -- the third export of shared_semantic over its own
    # leftovers died inside onnx's version converter, having succeeded twice from clean.
    # The quieter danger is the one that does NOT fail: a stale .onnx.data left beside a
    # newly written graph is hashed into the model digest and can be loaded, so an
    # abandoned export would masquerade as the current model.
    for stale in onnx_model_files(output):
        stale.unlink(missing_ok=True)

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
    input_name = session.get_inputs()[0].name

    # One example input is not enough to see a decision flip, because most pixels sit
    # nowhere near a boundary between two classes. Several random probes are run and ANY
    # label disagreement anywhere fails the export -- see verify_parity.
    generator = torch.Generator().manual_seed(0)
    probes = [example] + [
        torch.rand(1, 3, tile_size, tile_size, generator=generator, dtype=torch.float32)
        for _ in range(max(0, parity_inputs - 1))
    ]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for probe in probes:
        with torch.no_grad():
            reference = model(probe).cpu().numpy()
        pairs.append((reference, session.run(None, {input_name: probe.numpy()})[0]))
    parity = verify_parity(pairs, tolerance=parity_tolerance)
    max_abs_diff = parity['max_abs_diff']
    onnx_output = pairs[0][1]

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
    # allow_nan=False so a non-finite value can never be written as the bare token NaN,
    # which Python re-reads happily and every other JSON parser rejects. Metrics are
    # already filtered above; this is the backstop that makes the file portable.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8'
    )
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
        # Named explicitly so the report shows whether weights spilled to a sidecar.
        # A 1.1 MB "model" beside a 378 MB .onnx.data is not a packaging detail: both
        # files have to travel together or the model does not load at all.
        'files': [str(f.name) for f in onnx_model_files(output)],
        'total_bytes': sum(f.stat().st_size for f in onnx_model_files(output)),
        'parity_inputs': parity['probes'],
        'parity_pixels_compared': parity['pixels_compared'],
        'parity_label_disagreements': parity['label_disagreements'],
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
    parser.add_argument('--parity-tolerance', type=float, default=1e-3)
    parser.add_argument('--parity-inputs', type=int, default=4,
                        help='Random probes compared for label agreement (min 1).')
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
        parity_inputs=args.parity_inputs,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
