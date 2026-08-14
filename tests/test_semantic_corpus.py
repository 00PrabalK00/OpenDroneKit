'''Semantic training manifests prevent licence and spatial leakage.'''

from __future__ import annotations

import json

import pytest

from training.semantic_corpus import (
    SemanticCorpusError,
    SplitPolicy,
    build_semantic_corpus,
)


def write_source(tmp_path, samples):
    payload = {
        'manifest_schema_version': 1,
        'schema': {
            'id': 'india_shared_landcover',
            'version': '0.1.0-draft',
            'classes': [
                {'id': 0, 'name': 'background'},
                {'id': 1, 'name': 'building'},
            ],
        },
        'samples': samples,
    }
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def sample(tmp_path, index, site, capture_date, license='CC BY 4.0'):
    image = tmp_path / f'image_{index}.tif'
    label = tmp_path / f'label_{index}.tif'
    image.write_bytes(b'image')
    label.write_bytes(b'label')
    return {
        'id': f'sample-{index}',
        'source': 'source-a',
        'site_id': site,
        'capture_date': capture_date,
        'license': license,
        'image': image.name,
        'label': label.name,
    }


def test_all_dates_and_tiles_from_a_site_stay_in_one_split(tmp_path):
    samples = []
    index = 0
    for site in [f'site-{value}' for value in range(30)]:
        for capture_date in ('2025-01-01', '2026-01-01'):
            samples.append(sample(tmp_path, index, site, capture_date))
            index += 1
    result = build_semantic_corpus(
        write_source(tmp_path, samples),
        tmp_path / 'corpus.json',
    )
    assignments = {}
    for item in result['samples']:
        assignments.setdefault(item['group'], set()).add(item['split'])
    assert assignments
    assert all(len(splits) == 1 for splits in assignments.values())
    assert sum(result['counts']['samples_by_split'].values()) == len(samples)
    assert set(result['counts']['samples_by_split']) == {'train', 'validation', 'test'}


def test_split_is_deterministic_when_input_order_changes(tmp_path):
    samples = [sample(tmp_path, i, f'site-{i}', '2026-03-01') for i in range(8)]
    first = build_semantic_corpus(
        write_source(tmp_path, samples), tmp_path / 'first.json', policy=SplitPolicy(salt='fixed')
    )
    second = build_semantic_corpus(
        write_source(tmp_path, list(reversed(samples))),
        tmp_path / 'second.json',
        policy=SplitPolicy(salt='fixed'),
    )
    assert first['samples'] == second['samples']


def test_noncommercial_sample_is_excluded_and_audited(tmp_path):
    samples = [
        sample(tmp_path, 1, 'commercial', '2026-01-01'),
        sample(tmp_path, 2, 'research-only', '2026-01-02', license='CC BY-NC 3.0'),
    ]
    result = build_semantic_corpus(write_source(tmp_path, samples), tmp_path / 'corpus.json')
    assert [item['id'] for item in result['samples']] == ['sample-1']
    assert result['excluded'] == [{
        'id': 'sample-2',
        'source': 'source-a',
        'site_id': 'research-only',
        'license': 'CC BY-NC 3.0',
        'reason': 'license_not_allowed',
    }]


def test_capture_date_is_required_for_holdout_audit(tmp_path):
    bad = sample(tmp_path, 1, 'site', '2026-01-01')
    bad['capture_date'] = ''
    with pytest.raises(SemanticCorpusError, match='capture_date'):
        build_semantic_corpus(write_source(tmp_path, [bad]), tmp_path / 'corpus.json')


def test_unknown_capture_date_needs_an_explicit_audit_reason(tmp_path):
    record = sample(tmp_path, 1, 'site', '2026-01-01')
    record['capture_date'] = None
    record['capture_date_unknown_reason'] = 'Upstream release has no acquisition metadata.'
    record['class_map'] = {'1': 5}
    result = build_semantic_corpus(
        write_source(tmp_path, [record]),
        tmp_path / 'corpus.json',
    )
    assert result['counts']['unknown_capture_date_samples'] == 1
    assert result['samples'][0]['capture_date'] is None
    assert result['samples'][0]['class_map'] == {'1': 5}


def test_multiple_sources_resolve_files_relative_to_each_manifest(tmp_path):
    left = tmp_path / 'left'
    right = tmp_path / 'right'
    left.mkdir()
    right.mkdir()
    first = sample(left, 1, 'site-a', '2026-01-01')
    second = sample(right, 2, 'site-b', '2026-01-02')
    first_path = write_source(left, [first])
    second_path = write_source(right, [second])
    result = build_semantic_corpus(
        [first_path, second_path],
        tmp_path / 'corpus.json',
    )
    assert result['counts']['accepted_samples'] == 2
    assert {item['id'] for item in result['samples']} == {'sample-1', 'sample-2'}
    assert len(result['source_manifests']) == 2


def test_missing_files_are_refused(tmp_path):
    record = sample(tmp_path, 1, 'site', '2026-01-01')
    record['image'] = 'missing.tif'
    with pytest.raises(SemanticCorpusError, match='image does not exist'):
        build_semantic_corpus(write_source(tmp_path, [record]), tmp_path / 'corpus.json')


def test_empty_image_or_label_path_is_refused(tmp_path):
    record = sample(tmp_path, 1, 'site', '2026-01-01')
    record['label'] = ''
    with pytest.raises(SemanticCorpusError, match='image and label paths'):
        build_semantic_corpus(write_source(tmp_path, [record]), tmp_path / 'corpus.json')


def test_duplicate_sample_ids_are_refused(tmp_path):
    first = sample(tmp_path, 1, 'site-a', '2026-01-01')
    second = sample(tmp_path, 2, 'site-b', '2026-01-01')
    second['id'] = first['id']
    with pytest.raises(SemanticCorpusError, match='Duplicate'):
        build_semantic_corpus(write_source(tmp_path, [first, second]), tmp_path / 'corpus.json')
