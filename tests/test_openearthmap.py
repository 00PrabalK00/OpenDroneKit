'''OpenEarthMap indexing enforces the per-region commercial licence gate.'''

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
from rasterio.transform import from_origin

from training.datasets.openearthmap import (
    CAPTURE_DATE_UNKNOWN_REASON,
    OpenEarthMapIndexError,
    index_openearthmap,
)
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
    - {id: 1, name: building}
    - {id: 2, name: road}
    - {id: 3, name: vegetation}
    - {id: 4, name: water}
    - {id: 5, name: bare_land}
'''.strip(),
        encoding='utf-8',
    )
    return path


def add_pair(root, stem, values=None, *, with_label=True):
    image_dir = root / 'train' / 'images'
    label_dir = root / 'train' / 'labels'
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f'{stem}.tif'
    image = np.zeros((3, 2, 5), dtype=np.uint8)
    with rasterio.open(
        image_path,
        'w',
        driver='GTiff',
        width=5,
        height=2,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_origin(70, 20, 0.01, 0.01),
    ) as target:
        target.write(image)
    if with_label:
        label_path = label_dir / f'{stem}.tif'
        mask = np.asarray(
            values if values is not None else [[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]],
            dtype=np.uint8,
        )
        with rasterio.open(
            label_path,
            'w',
            driver='GTiff',
            width=5,
            height=2,
            count=1,
            dtype='uint8',
            crs='EPSG:4326',
            transform=from_origin(70, 20, 0.01, 0.01),
        ) as target:
            target.write(mask, 1)
    return image_path


def test_only_explicitly_commercial_regions_are_indexed(tmp_path):
    root = tmp_path / 'OpenEarthMap'
    add_pair(root, 'christchurch_001')
    add_pair(root, 'aachen_001')
    add_pair(root, 'tyrol_001')
    output = tmp_path / 'source.json'
    result = index_openearthmap(
        root,
        output,
        schema_config=write_config(tmp_path),
    )
    assert result['source']['image_count'] == 3
    assert result['source']['labelled_count'] == 1
    assert result['source']['excluded_file_count'] == 2
    sample = result['samples'][0]
    assert sample['site_id'] == 'Christchurch'
    assert sample['license'] == 'CC BY 4.0'
    assert sample['capture_date'] is None
    assert sample['capture_date_unknown_reason'] == CAPTURE_DATE_UNKNOWN_REASON
    assert sample['class_ids'] == [0, 1, 2, 3, 4, 5]
    assert sample['class_map']['3'] == 255
    assert result['source']['source_pixel_counts']['3'] == 1
    assert result['source']['shared_pixel_counts']['3'] == 3
    assert sample['class_pixel_counts']['3'] == 3

    corpus = build_semantic_corpus(output, tmp_path / 'corpus.json')
    assert corpus['counts']['uncovered_class_ids'] == []
    assert corpus['counts']['unknown_capture_date_samples'] == 1
    assert corpus['counts']['declared_class_pixel_counts']['3'] == 3


def test_allowlisted_image_without_label_is_refused(tmp_path):
    root = tmp_path / 'OpenEarthMap'
    add_pair(root, 'christchurch_001', with_label=False)
    with pytest.raises(OpenEarthMapIndexError, match='no matching label'):
        index_openearthmap(
            root,
            tmp_path / 'source.json',
            schema_config=write_config(tmp_path),
        )


def test_non_allowlisted_only_archive_is_refused(tmp_path):
    root = tmp_path / 'OpenEarthMap'
    add_pair(root, 'aachen_001')
    with pytest.raises(OpenEarthMapIndexError, match='production-compatible'):
        index_openearthmap(
            root,
            tmp_path / 'source.json',
            schema_config=write_config(tmp_path),
        )


def test_official_unlabelled_test_images_are_not_missing_labels(tmp_path):
    root = tmp_path / 'OpenEarthMap'
    add_pair(root, 'christchurch_001')
    add_pair(root, 'christchurch_002', with_label=False)
    (root / 'train.txt').write_text('christchurch_001.tif\n', encoding='utf-8')
    with pytest.raises(OpenEarthMapIndexError, match='both train.txt and val.txt'):
        index_openearthmap(
            root,
            tmp_path / 'source.json',
            schema_config=write_config(tmp_path),
        )
    (root / 'val.txt').write_text('christchurch_001.tif\n', encoding='utf-8')
    result = index_openearthmap(
        root,
        tmp_path / 'source.json',
        schema_config=write_config(tmp_path),
    )
    assert result['source']['image_count'] == 2
    assert result['source']['labelled_candidate_count'] == 1
    assert result['source']['upstream_unlabelled_test_count'] == 1
    assert result['source']['missing_label_count'] == 0
