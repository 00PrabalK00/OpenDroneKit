'''SpaceNet 7 indexing pairs monthly imagery and official building vectors.'''

from __future__ import annotations

import json

import pytest

from training.datasets.spacenet7 import SpaceNet7IndexError, index_spacenet7
from training.semantic_corpus import build_semantic_corpus


def write_config(tmp_path):
    path = tmp_path / 'semantic.yaml'
    path.write_text(
        '''
schema:
  id: shared
  version: 1.0.0
  classes:
    - {id: 0, name: background}
    - {id: 3, name: building}
'''.strip(),
        encoding='utf-8',
    )
    return path


def add_month(root, site, year, month, *, label_dir='labels_match', with_label=True):
    site_root = root / 'train' / site
    image_dir = site_root / 'images_masked'
    image_dir.mkdir(parents=True, exist_ok=True)
    stem = f'global_monthly_{year}_{month:02d}_mosaic_{site}'
    image = image_dir / f'{stem}.tif'
    image.write_bytes(b'image')
    if with_label:
        labels = site_root / label_dir
        labels.mkdir(parents=True, exist_ok=True)
        (labels / f'{stem}_Buildings.geojson').write_text(
            json.dumps({'type': 'FeatureCollection', 'features': []}), encoding='utf-8'
        )
    return image


def test_index_pairs_months_and_preserves_site_identity(tmp_path):
    root = tmp_path / 'spacenet7'
    add_month(root, 'AOI_A', 2025, 1)
    add_month(root, 'AOI_A', 2025, 2, label_dir='labels')
    add_month(root, 'AOI_B', 2026, 3)
    source_path = tmp_path / 'source.json'
    result = index_spacenet7(
        root,
        source_path,
        schema_config=write_config(tmp_path),
    )
    assert result['source']['image_count'] == 3
    assert result['source']['labelled_count'] == 3
    assert {item['capture_date'] for item in result['samples']} == {
        '2025-01-01', '2025-02-01', '2026-03-01'
    }
    assert all(item['class_id'] == 3 for item in result['samples'])
    assert all(item['background_id'] == 255 for item in result['samples'])
    assert all(item['label_format'] == 'geojson_polygons' for item in result['samples'])

    corpus = build_semantic_corpus(source_path, tmp_path / 'corpus.json')
    a_splits = {item['split'] for item in corpus['samples'] if item['site_id'] == 'AOI_A'}
    assert len(a_splits) == 1
    assert all(item['class_id'] == 3 for item in corpus['samples'])
    assert corpus['counts']['uncovered_class_ids'] == [0]
    assert corpus['counts']['declared_class_sample_counts']['0'] == 0
    assert corpus['counts']['declared_class_sample_counts']['3'] == 3


def test_missing_labels_are_refused_by_default(tmp_path):
    root = tmp_path / 'spacenet7'
    add_month(root, 'AOI_A', 2025, 1, with_label=False)
    with pytest.raises(SpaceNet7IndexError, match='no matching building label'):
        index_spacenet7(
            root,
            tmp_path / 'source.json',
            schema_config=write_config(tmp_path),
        )


def test_month_must_be_recoverable_from_official_filename(tmp_path):
    root = tmp_path / 'spacenet7'
    site = root / 'train' / 'AOI_A'
    (site / 'images_masked').mkdir(parents=True)
    image = site / 'images_masked' / 'unexpected.tif'
    image.write_bytes(b'image')
    (site / 'labels_match').mkdir()
    (site / 'labels_match' / 'unexpected_Buildings.geojson').write_text('{}', encoding='utf-8')
    with pytest.raises(SpaceNet7IndexError, match='capture month'):
        index_spacenet7(
            root,
            tmp_path / 'source.json',
            schema_config=write_config(tmp_path),
        )
