'''Build a licence-filtered, leakage-safe shared semantic corpus manifest.

Input is a small JSON document containing ``schema`` and ``samples``. Every sample
must identify its source, site and capture date. All dates from one source/site stay
in one split, preventing neighboring tiles or repeat flights from leaking into a
holdout. Files are referenced, not copied, so a 25 GB source is never duplicated.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PRODUCTION_LICENSE_ALLOWLIST = frozenset({
    'Apache-2.0',
    'CC BY 4.0',
    'CC BY-SA 4.0',
    'MIT',
    'Public Domain',
})


class SemanticCorpusError(ValueError):
    pass


@dataclass(frozen=True)
class SplitPolicy:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    salt: str = 'opendronekit-shared-semantic-v1'
    # Groups that must land in test whatever the hash says, each with the reason it
    # was pinned. A random split is the right default, but it cannot guarantee that a
    # specific region is held out, and "India-first" is a claim about a specific
    # region. Pinning is recorded in the corpus so the holdout is auditable rather
    # than an accident of the salt.
    pinned_test_groups: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values):
            raise SemanticCorpusError('Split ratios cannot be negative.')
        if abs(sum(values) - 1.0) > 1e-9:
            raise SemanticCorpusError('Split ratios must sum to 1.0.')
        if not self.salt:
            raise SemanticCorpusError('Split salt cannot be empty.')
        for group, reason in dict(self.pinned_test_groups).items():
            if not str(reason).strip():
                raise SemanticCorpusError(
                    f'Pinned test group {group!r} needs a reason it was pinned.'
                )


# The India holdout for the shared semantic engine.
#
# No open dataset carries India land-cover labels at the 0.25-0.5 m resolution this
# engine works at, so these four SpaceNet 7 tiles are what an India holdout can honestly
# be built from today. Their coordinates were decoded from the slippy-map tile numbers
# embedded in the site id, not assumed from the name.
#
# Read the resulting metric narrowly. SpaceNet 7 labels buildings and nothing else, so
# these tiles measure one of the schema's six classes over Indian ground. They say
# nothing about road, vegetation, water or bare_land transfer, and any report that
# quotes them must say so.
INDIA_HOLDOUT_GROUPS: dict[str, str] = {
    'spacenet7::L15-1438E-1134N_5753_3655_13': 'India holdout: Mumbai (19.021N, 72.817E).',
    'spacenet7::L15-1439E-1134N_5759_3655_13': 'India holdout: Kalyan/Thane (19.021N, 73.081E).',
    'spacenet7::L15-1479E-1101N_5916_3785_13': 'India holdout: Tirupati (13.539N, 79.980E).',
    'spacenet7::L15-1481E-1119N_5927_3715_13': 'India holdout: Vijayawada (16.510N, 80.464E).',
    # The four above are Planet mosaics at 2.91-4.77 m per pixel, labelled for buildings
    # and nothing else. They can say nothing about road, vegetation, water or bare land,
    # and they test a satellite scale no drone flies at -- so a model can pass them and
    # still be useless for the product, or fail them and be fine.
    #
    # Gorakhpur is the holdout that answers the question actually being asked: Indian
    # ground at 0.448 m per pixel, exhaustively labelled for all six classes. Its imagery
    # comes from xBD rather than the OpenEarthMap download, which is why it sat unusable
    # until training/datasets/xbd_imagery.py went and got it.
    # Lowercase because that is the region key the indexer produces for the xBD-sourced
    # regions, and a pin that does not match a group silently pins nothing -- the holdout
    # would land in training and every number after it would be measured on data the
    # model had seen. tests/test_india_holdout.py asserts this group exists.
    'openearthmap::gorakhpur': (
        'India holdout: Gorakhpur, Uttar Pradesh (26.76N, 83.37E). 0.448 m/px, all six '
        'classes labelled -- the only India evidence that speaks for more than buildings.'
    ),
}


def _normalise_date(
    value: Any,
    sample_id: str,
    unknown_reason: Any = '',
) -> str | None:
    text = str(value or '').strip()
    reason = str(unknown_reason or '').strip()
    if not text:
        if reason:
            return None
        raise SemanticCorpusError(
            f'Sample {sample_id!r} needs capture_date or capture_date_unknown_reason.'
        )
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise SemanticCorpusError(
            f'Sample {sample_id!r} needs capture_date in YYYY-MM-DD format.'
        ) from exc


def _split_for_group(group: str, policy: SplitPolicy) -> str:
    if group in policy.pinned_test_groups:
        return 'test'
    digest = hashlib.sha256(f'{policy.salt}\0{group}'.encode('utf-8')).digest()
    position = int.from_bytes(digest[:8], 'big') / float(2 ** 64)
    if position < policy.train:
        return 'train'
    if position < policy.train + policy.validation:
        return 'validation'
    return 'test'


def build_semantic_corpus(
    source_manifest: str | Path | Sequence[str | Path],
    output_manifest: str | Path,
    *,
    allowed_licenses: Sequence[str] = tuple(sorted(PRODUCTION_LICENSE_ALLOWLIST)),
    policy: SplitPolicy | None = None,
    require_files: bool = True,
) -> dict[str, Any]:
    source_paths = (
        [Path(source_manifest)]
        if isinstance(source_manifest, (str, Path))
        else [Path(value) for value in source_manifest]
    )
    if not source_paths:
        raise SemanticCorpusError('At least one source manifest is required.')
    schema: dict[str, Any] | None = None
    samples: list[tuple[dict[str, Any], Path]] = []
    for source_path in source_paths:
        payload = json.loads(source_path.read_text(encoding='utf-8'))
        if int(payload.get('manifest_schema_version', 0)) != 1:
            raise SemanticCorpusError(
                f'Unsupported source manifest schema version in {source_path}.'
            )
        candidate_schema = dict(payload.get('schema') or {})
        if (
            not candidate_schema.get('id')
            or not candidate_schema.get('version')
            or not candidate_schema.get('classes')
        ):
            raise SemanticCorpusError(
                f'Source manifest {source_path} needs a versioned semantic schema.'
            )
        if schema is None:
            schema = candidate_schema
        elif candidate_schema != schema:
            raise SemanticCorpusError(
                f'Source manifest {source_path} uses a different semantic schema.'
            )
        samples.extend(
            (dict(raw), source_path.parent)
            for raw in list(payload.get('samples') or [])
        )
    if not samples:
        raise SemanticCorpusError('Source manifests contain no samples.')
    assert schema is not None

    split_policy = policy or SplitPolicy()
    allowed = {str(value).strip() for value in allowed_licenses if str(value).strip()}
    if not allowed:
        raise SemanticCorpusError('At least one allowed licence is required.')
    seen_ids: set[str] = set()
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for sample, base in samples:
        sample_id = str(sample.get('id') or '').strip()
        if not sample_id:
            raise SemanticCorpusError('Every semantic sample needs a stable id.')
        if sample_id in seen_ids:
            raise SemanticCorpusError(f'Duplicate semantic sample id: {sample_id}')
        seen_ids.add(sample_id)
        source = str(sample.get('source') or '').strip()
        site_id = str(sample.get('site_id') or '').strip()
        licence = str(sample.get('license') or '').strip()
        if not source or not site_id or not licence:
            raise SemanticCorpusError(
                f'Sample {sample_id!r} needs source, site_id and license.'
            )
        unknown_date_reason = str(
            sample.get('capture_date_unknown_reason') or ''
        ).strip()
        capture_date = _normalise_date(
            sample.get('capture_date'),
            sample_id,
            unknown_date_reason,
        )
        if licence not in allowed:
            excluded.append({
                'id': sample_id,
                'source': source,
                'site_id': site_id,
                'license': licence,
                'reason': 'license_not_allowed',
            })
            continue

        image_text = str(sample.get('image') or '').strip()
        label_text = str(sample.get('label') or '').strip()
        if not image_text or not label_text:
            raise SemanticCorpusError(f'Sample {sample_id!r} needs image and label paths.')
        image = Path(image_text)
        label = Path(label_text)
        resolved_image = image if image.is_absolute() else (base / image).resolve()
        resolved_label = label if label.is_absolute() else (base / label).resolve()
        if require_files:
            if not resolved_image.is_file():
                raise SemanticCorpusError(f'Sample {sample_id!r} image does not exist: {resolved_image}')
            if not resolved_label.is_file():
                raise SemanticCorpusError(f'Sample {sample_id!r} label does not exist: {resolved_label}')

        group = f'{source}::{site_id}'
        accepted.append({
            'id': sample_id,
            'source': source,
            'site_id': site_id,
            'capture_date': capture_date,
            'license': licence,
            'image': str(resolved_image),
            'label': str(resolved_label),
            'group': group,
            'split': _split_for_group(group, split_policy),
            # Carried on the sample so a reader of one record can tell a deliberate
            # holdout from a hash outcome without consulting the policy.
            **(
                {'pinned_test_reason': split_policy.pinned_test_groups[group]}
                if group in split_policy.pinned_test_groups
                else {}
            ),
            **{
                key: sample[key]
                for key in (
                    'label_format',
                    'class_id',
                    'class_ids',
                    'exhaustive_class_ids',
                    'class_map',
                    'class_pixel_counts',
                    'background_id',
                    'udm_mask',
                    'capture_date_unknown_reason',
                )
                if key in sample
            },
        })

    if not accepted:
        raise SemanticCorpusError('No samples remain after licence filtering.')
    accepted.sort(
        key=lambda item: (
            item['split'],
            item['source'],
            item['site_id'],
            item['capture_date'] or '',
            item['id'],
        )
    )
    split_counts = {
        split: sum(item['split'] == split for item in accepted)
        for split in ('train', 'validation', 'test')
    }
    site_counts = {
        split: len({item['group'] for item in accepted if item['split'] == split})
        for split in ('train', 'validation', 'test')
    }
    schema_classes = list(schema.get('classes') or [])
    schema_ids = [int(item['id']) for item in schema_classes]
    background_ids = {
        int(item['id'])
        for item in schema_classes
        if bool(item.get('background', False)) or str(item.get('name', '')).casefold() == 'background'
    }
    declared_counts = {class_id: 0 for class_id in schema_ids}
    declared_pixel_counts = {class_id: 0 for class_id in schema_ids}
    for item in accepted:
        declared = {int(value) for value in item.get('class_ids', [])}
        if 'class_id' in item:
            declared.add(int(item['class_id']))
        if declared and int(item.get('background_id', 0)) != 255:
            declared.update(background_ids)
        for class_id in declared:
            if class_id in declared_counts:
                declared_counts[class_id] += 1
        for class_id, count in dict(item.get('class_pixel_counts') or {}).items():
            parsed_id = int(class_id)
            parsed_count = int(count)
            if parsed_count < 0:
                raise SemanticCorpusError(
                    f'Sample {item["id"]!r} has a negative class pixel count.'
                )
            if parsed_id in declared_pixel_counts:
                declared_pixel_counts[parsed_id] += parsed_count
    result = {
        'manifest_schema_version': 1,
        'schema': schema,
        'source_manifests': [str(path.resolve()) for path in source_paths],
        'split_policy': {
            **asdict(split_policy),
            'group_by': ['source', 'site_id'],
            'capture_date_policy': (
                'ISO date required unless capture_date_unknown_reason is recorded.'
            ),
            'rule': 'All dates and tiles from one source/site remain in one split.',
        },
        'allowed_licenses': sorted(allowed),
        'counts': {
            'accepted_samples': len(accepted),
            'excluded_samples': len(excluded),
            'unknown_capture_date_samples': sum(
                item['capture_date'] is None for item in accepted
            ),
            'samples_by_split': split_counts,
            'sites_by_split': site_counts,
            # A pinned group that matched nothing is the quiet failure here: the corpus
            # would look correct while the holdout it claims is empty. Counting what
            # each pin actually captured makes that visible.
            'pinned_test_samples': {
                group: sum(item['group'] == group for item in accepted)
                for group in sorted(split_policy.pinned_test_groups)
            },
            'declared_class_sample_counts': {
                str(class_id): declared_counts[class_id] for class_id in schema_ids
            },
            'declared_class_pixel_counts': {
                str(class_id): declared_pixel_counts[class_id]
                for class_id in schema_ids
            },
            'uncovered_class_ids': [
                class_id for class_id in schema_ids if declared_counts[class_id] == 0
            ],
        },
        'excluded': excluded,
        'samples': accepted,
    }
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build a leakage-safe semantic corpus manifest.')
    parser.add_argument('source_manifest', type=Path)
    parser.add_argument('output_manifest', type=Path)
    parser.add_argument(
        '--add-source',
        action='append',
        type=Path,
        default=[],
        help='Additional source manifest using the exact same semantic schema.',
    )
    parser.add_argument('--allow-license', action='append', dest='licenses')
    parser.add_argument('--no-require-files', action='store_true')
    parser.add_argument('--salt', default=SplitPolicy.salt)
    parser.add_argument(
        '--no-india-holdout',
        action='store_true',
        help=(
            'Do not pin the India tiles into test. The engine is presented as '
            'India-first, so holding them out is the default and switching it off '
            'should be a deliberate act.'
        ),
    )
    args = parser.parse_args(argv)
    result = build_semantic_corpus(
        [args.source_manifest, *args.add_source],
        args.output_manifest,
        allowed_licenses=args.licenses or tuple(sorted(PRODUCTION_LICENSE_ALLOWLIST)),
        policy=SplitPolicy(
            salt=args.salt,
            pinned_test_groups={} if args.no_india_holdout else dict(INDIA_HOLDOUT_GROUPS),
        ),
        require_files=not args.no_require_files,
    )
    print(json.dumps(result['counts'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
