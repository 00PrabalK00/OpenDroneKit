'''Shared semantic training tiles preserve labels and schema channel mapping.'''

from __future__ import annotations

import json

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('torch')
from rasterio.transform import from_origin

from training.semantic_tiles import (
    IGNORE_INDEX,
    SemanticTileDataset,
    SemanticTileError,
    rasterize_geojson_polygons,
    read_semantic_sample,
)


def write_image(tmp_path, width=32, height=24):
    path = tmp_path / 'image.tif'
    pixels = np.zeros((3, height, width), dtype=np.uint8)
    pixels[0] = 255
    with rasterio.open(
        path, 'w', driver='GTiff', width=width, height=height, count=3,
        dtype='uint8', crs='EPSG:4326', transform=from_origin(70, 20, 0.01, 0.01),
    ) as target:
        target.write(pixels)
    return path


def write_vector(tmp_path):
    path = tmp_path / 'labels.geojson'
    path.write_text(json.dumps({
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[70.16, 20], [70.32, 20], [70.32, 19.76], [70.16, 19.76], [70.16, 20]]],
            },
            'properties': {},
        }],
    }), encoding='utf-8')
    return path


def write_corpus(tmp_path, image, label):
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps({
        'schema': {
            'id': 'shared',
            'version': '1',
            'classes': [
                {'id': 0, 'name': 'background'},
                {'id': 4, 'name': 'building'},
            ],
        },
        'samples': [{
            'id': 'one',
            'source': 'fixture',
            'site_id': 'site',
            'capture_date': '2026-01-01',
            'license': 'CC BY 4.0',
            'image': str(image),
            'label': str(label),
            'label_format': 'geojson_polygons',
            'class_id': 4,
            'split': 'train',
        }],
    }), encoding='utf-8')
    return path


def test_geojson_polygons_are_rasterized_in_the_image_grid(tmp_path):
    image = write_image(tmp_path)
    label = write_vector(tmp_path)
    mask = rasterize_geojson_polygons(image, label, class_id=4)
    assert mask.shape == (24, 32)
    assert np.all(mask[:, :16] == 0)
    assert np.all(mask[:, 16:] == 4)


def test_sample_reader_normalizes_rgb_and_keeps_schema_ids(tmp_path):
    image = write_image(tmp_path)
    label = write_vector(tmp_path)
    pixels, mask = read_semantic_sample({
        'image': str(image),
        'label': str(label),
        'label_format': 'geojson_polygons',
        'class_id': 4,
    })
    assert pixels.dtype == np.float32
    assert pixels.min() == 0.0
    assert pixels.max() == 1.0
    assert set(np.unique(mask)) == {0, 4}


def test_partial_vector_labels_can_ignore_everything_outside_polygons(tmp_path):
    image = write_image(tmp_path)
    label = write_vector(tmp_path)
    _, mask = read_semantic_sample({
        'image': str(image),
        'label': str(label),
        'label_format': 'geojson_polygons',
        'class_id': 4,
        'background_id': IGNORE_INDEX,
    })
    assert set(np.unique(mask)) == {4, IGNORE_INDEX}


def test_raster_class_map_merges_source_classes_and_ignores_pavement(tmp_path):
    image = write_image(tmp_path, width=4, height=2)
    label = tmp_path / 'mask.tif'
    source_ids = np.array([[0, 1, 2, 3], [4, 5, 6, 8]], dtype=np.uint8)
    with rasterio.open(
        label,
        'w',
        driver='GTiff',
        width=4,
        height=2,
        count=1,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_origin(70, 20, 0.01, 0.01),
    ) as target:
        target.write(source_ids, 1)
    _, mask = read_semantic_sample({
        'image': str(image),
        'label': str(label),
        'label_format': 'raster_class_ids',
        'class_map': {'0': 0, '1': 5, '2': 3, '4': 2, '5': 3, '6': 4, '8': 1},
    })
    assert mask.tolist() == [[0, 5, 3, IGNORE_INDEX], [2, 3, 4, 1]]


def test_dataset_maps_noncontiguous_schema_ids_to_model_channels(tmp_path):
    image = write_image(tmp_path)
    label = write_vector(tmp_path)
    dataset = SemanticTileDataset(
        write_corpus(tmp_path, image, label), 'train', tile_size=32
    )
    item = dataset[0]
    assert tuple(item['image'].shape) == (3, 32, 32)
    assert tuple(item['mask'].shape) == (32, 32)
    assert set(item['mask'].unique().tolist()) == {0, 1, IGNORE_INDEX}
    assert item['sample_id'] == 'one'


def test_empty_split_is_refused(tmp_path):
    image = write_image(tmp_path)
    label = write_vector(tmp_path)
    with pytest.raises(SemanticTileError, match='no samples'):
        SemanticTileDataset(write_corpus(tmp_path, image, label), 'validation', tile_size=32)
