'''Index the official SpaceNet 7 archive for shared semantic training.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import yaml


DATE_PATTERN = re.compile(r'global_monthly_(\d{4})_(\d{2})_')
LABEL_DIRS = ('labels_match', 'labels')


class SpaceNet7IndexError(ValueError):
    pass


def _load_schema(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    schema = dict(payload.get('schema') or {})
    if not schema.get('id') or not schema.get('version') or not schema.get('classes'):
        raise SpaceNet7IndexError('Semantic config has no versioned schema.')
    return schema


def _capture_date(image: Path) -> str:
    match = DATE_PATTERN.search(image.name)
    if not match:
        raise SpaceNet7IndexError(f'Cannot recover capture month from {image.name}')
    return f'{match.group(1)}-{match.group(2)}-01'


def _label_for(image: Path) -> Path | None:
    site = image.parent.parent
    expected_name = f'{image.stem}_Buildings.geojson'
    for directory in LABEL_DIRS:
        candidate = site / directory / expected_name
        if candidate.is_file():
            return candidate
    for directory in LABEL_DIRS:
        root = site / directory
        if root.is_dir():
            candidates = sorted(root.glob(f'{image.stem}*_Buildings.geojson'))
            if candidates:
                return candidates[0]
    return None


def index_spacenet7(
    dataset_root: str | Path,
    output_manifest: str | Path,
    *,
    schema_config: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root)
    image_paths = sorted(root.rglob('images_masked/*.tif'))
    if not image_paths:
        raise SpaceNet7IndexError(f'No SpaceNet 7 images_masked GeoTIFFs found under {root}')
    schema = _load_schema(Path(schema_config))
    class_names = {str(item.get('name')): int(item['id']) for item in schema['classes']}
    if 'building' not in class_names:
        raise SpaceNet7IndexError('Shared semantic schema has no building class.')

    records: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for image in image_paths:
        label = _label_for(image)
        if label is None:
            missing_labels.append(str(image))
            continue
        site_id = image.parent.parent.name
        udm = image.parent.parent / 'UDM_masks' / image.name
        record = {
            'id': f'spacenet7:{site_id}:{image.stem}',
            'source': 'spacenet7',
            'site_id': site_id,
            'capture_date': _capture_date(image),
            'license': 'CC BY-SA 4.0',
            'image': str(image.resolve()),
            'label': str(label.resolve()),
            'label_format': 'geojson_polygons',
            'class_id': class_names['building'],
            'class_ids': [class_names['building']],
            'background_id': 255,
            # SpaceNet 7's annotators drew every building they saw across the whole
            # tile, so a pixel outside every polygon is evidence there is no building
            # there -- even though it says nothing about what there IS. Without this
            # the loss never penalises a false building on the 96.7 per cent of each
            # tile that carries no label, and "everything is a building" becomes a
            # free answer. It is not hypothetical: the head trained without it
            # predicted building on 100 per cent of the India holdout.
            'exhaustive_class_ids': [class_names['building']],
        }
        if udm.is_file():
            record['udm_mask'] = str(udm.resolve())
        records.append(record)

    if strict and missing_labels:
        raise SpaceNet7IndexError(
            f'{len(missing_labels)} SpaceNet image(s) have no matching building label; '
            f'first: {missing_labels[0]}'
        )
    if not records:
        raise SpaceNet7IndexError('No labelled SpaceNet 7 samples were indexed.')
    payload = {
        'manifest_schema_version': 1,
        'schema': schema,
        'source': {
            'id': 'spacenet7',
            'license': 'CC BY-SA 4.0',
            'dataset_root': str(root.resolve()),
            'image_count': len(image_paths),
            'labelled_count': len(records),
            'missing_label_count': len(missing_labels),
            'label_preference': list(LABEL_DIRS),
        },
        'samples': records,
    }
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Index extracted SpaceNet 7 semantic samples.')
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('output_manifest', type=Path)
    parser.add_argument(
        '--schema-config',
        type=Path,
        default=Path('training/configs/shared_semantic_dinov2_vitb14.yaml'),
    )
    parser.add_argument('--allow-missing-labels', action='store_true')
    args = parser.parse_args(argv)
    result = index_spacenet7(
        args.dataset_root,
        args.output_manifest,
        schema_config=args.schema_config,
        strict=not args.allow_missing_labels,
    )
    print(json.dumps(result['source'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
