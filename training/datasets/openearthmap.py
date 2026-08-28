'''Index only commercially compatible OpenEarthMap regions.

OpenEarthMap inherits the licence of each source region. The project additionally
states that labels over public-domain or unspecified imagery use CC BY-NC-SA 4.0,
so those regions are deliberately absent from this allowlist.
'''

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml


ATTRIBUTION_URL = 'https://open-earth-map.org/attribution.html'
CAPTURE_DATE_UNKNOWN_REASON = (
    'OpenEarthMap does not publish per-image acquisition dates in the release.'
)
IMAGE_SUFFIXES = frozenset({'.tif', '.tiff', '.png'})

# Exact production-compatible rows from the official attribution table.
CC_BY_4_REGIONS = (
    'Christchurch',
    'Chisinau',
    'Ngaoundere',
    'Kinshasa',
    'Pointenoire',
    'Accra',
    'Monrovia',
    'Niamey',
    'Mahe',
    'Dar es salaam',
    'Zanzibar',
    'Buenos aires',
    'Rosario',
    'Melbourne',
    "Cox's bazar",
    'Dhaka',
    'Santiago',
    'Bogota',
    'Svaneti',
    'Western',
    'Al qurnah',
    'Dowa',
    'Ulaanbaatar',
    'Maputo',
    'Baybay',
    'San tome',
    'Chiangmai',
    'Lohur',
    'Kagera',
    'Tonga',
    'Soriano',
)
CC_BY_SA_4_REGIONS = (
    'Rio',
    'Shanghai',
    'Paris',
    'Rotterdam',
    'Khartoum',
    'Vegas',
)
# The remaining regions carry CC BY-NC-SA 4.0: OpenEarthMap applies that licence to
# labels drawn over public-domain or unspecified imagery. They are excluded by default,
# and admitting them is a deliberate act behind --include-noncommercial rather than a
# quiet widening -- a model trained on them is arguably encumbered by the same terms, and
# that has to be a decision someone took rather than one that happened.
CC_BY_NC_SA_4_REGIONS = (
    'aachen', 'abancay', 'austin', 'bielefeld', 'chicago', 'chiclayo', 'chincha',
    'dolnoslaskie', 'dortmund', 'duesseldorf', 'ica', 'kampala', 'kitsap', 'koeln',
    'kujawsko', 'kyoto', 'lambayeque', 'lima', 'lodzkie', 'lubuskie', 'malopolskie',
    'mazowieckie', 'muenster', 'pisco', 'piura', 'podkarpackie', 'podlaskie',
    'pomorskie', 'sechura', 'slaskie', 'swietokrzyskie', 'tokyo', 'tyrolw', 'vienna',
    'viru', 'warminsko', 'wielkopolskie', 'zachodniopomorskie',
)

SAFE_REGION_LICENSES = {
    **{region: 'CC BY 4.0' for region in CC_BY_4_REGIONS},
    **{region: 'CC BY-SA 4.0' for region in CC_BY_SA_4_REGIONS},
}
NONCOMMERCIAL_REGION_LICENSES = {
    region: 'CC BY-NC-SA 4.0' for region in CC_BY_NC_SA_4_REGIONS
}


def region_licenses(include_noncommercial: bool = False) -> dict[str, str]:
    """Which regions may be indexed, and under what terms."""
    if include_noncommercial:
        return {**SAFE_REGION_LICENSES, **NONCOMMERCIAL_REGION_LICENSES}
    return dict(SAFE_REGION_LICENSES)

OEM_CLASS_NAMES = {
    0: 'unknown',
    1: 'Bareland',
    2: 'Grass',
    3: 'Pavement',
    4: 'Road',
    5: 'Tree',
    6: 'Water',
    7: 'Cropland',
    8: 'buildings',
}
# Target IDs are resolved from the shared schema at runtime. Pavement stays ignored
# because the source class includes non-road developed surfaces.
OEM_TO_SHARED_NAMES = {
    0: 'background',
    1: 'bare_land',
    2: 'vegetation',
    3: None,
    4: 'road',
    5: 'vegetation',
    6: 'water',
    7: 'vegetation',
    8: 'building',
}


class OpenEarthMapIndexError(ValueError):
    pass


def _normalise_region(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', value.casefold())


SAFE_REGION_ALIASES = {
    _normalise_region(region): region for region in SAFE_REGION_LICENSES
}


def _load_schema(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    schema = dict(payload.get('schema') or {})
    if not schema.get('id') or not schema.get('version') or not schema.get('classes'):
        raise OpenEarthMapIndexError('Semantic config has no versioned schema.')
    names = {str(item.get('name')) for item in schema['classes']}
    required = {name for name in OEM_TO_SHARED_NAMES.values() if name is not None}
    missing = sorted(required - names)
    if missing:
        raise OpenEarthMapIndexError(
            f'Shared semantic schema is missing OpenEarthMap targets: {missing}'
        )
    return schema


def _region_for(image: Path, root: Path, aliases: dict[str, str] | None = None) -> str | None:
    aliases = SAFE_REGION_ALIASES if aliases is None else aliases
    relative = image.relative_to(root)
    values = [part for part in relative.parts[:-1] if part.casefold() != 'images']
    values.append(image.stem)
    for value in reversed(values[:-1]):
        region = aliases.get(_normalise_region(value))
        if region:
            return region
    stem = _normalise_region(values[-1])
    for alias in sorted(aliases, key=len, reverse=True):
        remainder = stem[len(alias):] if stem.startswith(alias) else ''
        if stem == alias or (remainder and remainder[0].isdigit()):
            return aliases[alias]
    return None


def _region_hint(image: Path) -> str:
    stem = image.stem
    prefix = re.split(r'[_\-\s]+', stem, maxsplit=1)[0]
    return prefix or stem


def _label_for(image: Path) -> Path | None:
    parts = list(image.parts)
    image_indices = [
        index for index, part in enumerate(parts) if part.casefold() == 'images'
    ]
    if not image_indices:
        return None
    parts[image_indices[-1]] = 'labels'
    direct = Path(*parts)
    if direct.is_file():
        return direct
    parent = direct.parent
    if parent.is_dir():
        for suffix in IMAGE_SUFFIXES:
            candidate = parent / f'{direct.stem}{suffix}'
            if candidate.is_file():
                return candidate
    return None


def _labelled_split_names(root: Path) -> set[str] | None:
    split_paths = [root / 'train.txt', root / 'val.txt']
    existing = [path for path in split_paths if path.is_file()]
    if not existing:
        return None
    if len(existing) != len(split_paths):
        raise OpenEarthMapIndexError(
            'OpenEarthMap needs both train.txt and val.txt when split lists are present.'
        )
    names: set[str] = set()
    for path in split_paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            name = line.strip()
            if name:
                names.add(name)
    if not names:
        raise OpenEarthMapIndexError('OpenEarthMap train/validation lists are empty.')
    return names


def _source_class_counts(label: Path) -> Counter[int]:
    import rasterio

    with rasterio.open(label) as source:
        if source.count != 1:
            raise OpenEarthMapIndexError(
                f'OpenEarthMap label must have one band: {label}'
            )
        values, counts = np.unique(source.read(1), return_counts=True)
    result = Counter({
        int(value): int(count) for value, count in zip(values, counts)
    })
    invalid = sorted(set(result) - set(OEM_CLASS_NAMES))
    if invalid:
        raise OpenEarthMapIndexError(
            f'OpenEarthMap label {label} has unknown class IDs: {invalid}'
        )
    return result


def index_openearthmap(
    dataset_root: str | Path,
    output_manifest: str | Path,
    *,
    schema_config: str | Path,
    strict: bool = True,
    include_noncommercial: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root)
    licenses = region_licenses(include_noncommercial)
    aliases = {_normalise_region(region): region for region in licenses}
    all_image_paths = sorted(
        path
        for path in root.rglob('*')
        if path.is_file()
        and path.suffix.casefold() in IMAGE_SUFFIXES
        and any(part.casefold() == 'images' for part in path.parts)
    )
    if not all_image_paths:
        raise OpenEarthMapIndexError(
            f'No OpenEarthMap image files under an images directory in {root}'
        )
    labelled_names = _labelled_split_names(root)
    image_paths = (
        all_image_paths
        if labelled_names is None
        else [path for path in all_image_paths if path.name in labelled_names]
    )
    if not image_paths:
        raise OpenEarthMapIndexError(
            'No extracted OpenEarthMap images match the train/validation lists.'
        )
    schema = _load_schema(Path(schema_config))
    target_ids = {
        str(item['name']): int(item['id']) for item in schema['classes']
    }
    class_map = {
        str(source_id): (
            255 if target_name is None else target_ids[target_name]
        )
        for source_id, target_name in OEM_TO_SHARED_NAMES.items()
    }
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    missing_labels: list[str] = []
    accepted_by_license: Counter[str] = Counter()
    source_pixel_counts: Counter[int] = Counter()
    shared_pixel_counts: Counter[int] = Counter()

    for image in image_paths:
        region = _region_for(image, root, aliases)
        if region is None:
            exclusions.append({
                'image': str(image.resolve()),
                'region_hint': _region_hint(image),
                'reason': 'region_not_in_production_allowlist',
            })
            continue
        label = _label_for(image)
        if label is None:
            missing_labels.append(str(image.resolve()))
            continue
        class_counts = _source_class_counts(label)
        source_ids = set(class_counts)
        mapped_ids = sorted({
            int(class_map[str(source_id)])
            for source_id in source_ids
            if int(class_map[str(source_id)]) != 255
        })
        if not mapped_ids:
            exclusions.append({
                'image': str(image.resolve()),
                'region_hint': region,
                'reason': 'no_mapped_shared_classes',
            })
            continue
        licence = licenses[region]
        accepted_by_license[licence] += 1
        source_pixel_counts.update(class_counts)
        mapped_pixel_counts: Counter[int] = Counter()
        for source_id, count in class_counts.items():
            target_id = int(class_map[str(source_id)])
            if target_id != 255:
                shared_pixel_counts[target_id] += count
                mapped_pixel_counts[target_id] += count
        relative_id = image.relative_to(root).with_suffix('').as_posix()
        records.append({
            'id': f'openearthmap:{relative_id}',
            'source': 'openearthmap',
            'site_id': region,
            'capture_date': None,
            'capture_date_unknown_reason': CAPTURE_DATE_UNKNOWN_REASON,
            'license': licence,
            'image': str(image.resolve()),
            'label': str(label.resolve()),
            'label_format': 'raster_class_ids',
            'class_ids': mapped_ids,
            'class_map': class_map,
            'class_pixel_counts': {
                str(class_id): mapped_pixel_counts[class_id]
                for class_id in sorted(mapped_pixel_counts)
            },
        })

    if strict and missing_labels:
        raise OpenEarthMapIndexError(
            f'{len(missing_labels)} allowlisted OpenEarthMap image(s) have no matching '
            f'label; first: {missing_labels[0]}'
        )
    if not records:
        raise OpenEarthMapIndexError(
            'No labelled, production-compatible OpenEarthMap samples were indexed.'
        )
    exclusion_counts = Counter(item['reason'] for item in exclusions)
    payload = {
        'manifest_schema_version': 1,
        'schema': schema,
        'source': {
            'id': 'openearthmap',
            'license': (
                'Per-region allowlist: CC BY 4.0, CC BY-SA 4.0 or CC BY-NC-SA 4.0'
                if include_noncommercial
                else 'Per-region allowlist: CC BY 4.0 or CC BY-SA 4.0'
            ),
            'includes_noncommercial': bool(include_noncommercial),
            'attribution_url': ATTRIBUTION_URL,
            'dataset_root': str(root.resolve()),
            'image_count': len(all_image_paths),
            'labelled_candidate_count': len(image_paths),
            'upstream_unlabelled_test_count': len(all_image_paths) - len(image_paths),
            'labelled_count': len(records),
            'missing_label_count': len(missing_labels),
            'production_region_count': len(licenses),
            'accepted_by_license': dict(sorted(accepted_by_license.items())),
            'excluded_file_count': len(exclusions),
            'excluded_by_reason': dict(sorted(exclusion_counts.items())),
            'source_class_names': OEM_CLASS_NAMES,
            'class_map': class_map,
            'source_pixel_counts': {
                str(class_id): source_pixel_counts[class_id]
                for class_id in sorted(source_pixel_counts)
            },
            'shared_pixel_counts': {
                str(class_id): shared_pixel_counts[class_id]
                for class_id in sorted(shared_pixel_counts)
            },
            'capture_date_status': 'unknown_with_reason',
        },
        'excluded': exclusions,
        'samples': records,
    }
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Index the commercial-compatible OpenEarthMap region subset.'
    )
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('output_manifest', type=Path)
    parser.add_argument(
        '--schema-config',
        type=Path,
        default=Path('training/configs/shared_semantic_dinov2_vitb14.yaml'),
    )
    parser.add_argument('--allow-missing-labels', action='store_true')
    parser.add_argument(
        '--include-noncommercial',
        action='store_true',
        help=(
            'Also index the CC BY-NC-SA 4.0 regions. A model trained on them is arguably '
            'encumbered by the same terms, so this is deliberate rather than default.'
        ),
    )
    args = parser.parse_args(argv)
    result = index_openearthmap(
        args.dataset_root,
        args.output_manifest,
        schema_config=args.schema_config,
        strict=not args.allow_missing_labels,
        include_noncommercial=args.include_noncommercial,
    )
    print(json.dumps(result['source'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
