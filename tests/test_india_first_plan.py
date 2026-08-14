'''The India-first backlog is executable, ordered and provenance-safe.'''

import json
from pathlib import Path

from training.datasets.registry import DATASETS, GROUPS
from training.india_first_plan import WORKSTREAMS, validate_plan
from training.pretrained import ASSETS


def test_the_user_selected_order_is_locked():
    assert validate_plan() == []
    assert [item.order for item in WORKSTREAMS] == list(range(1, 10))
    assert WORKSTREAMS[0].id == 'selected_roi_change'
    assert WORKSTREAMS[-1].id == 'power_rail'


def test_completed_pack_engines_and_withheld_training_are_recorded_honestly():
    assert WORKSTREAMS[0].state == 'complete'
    assert WORKSTREAMS[1].state == 'started'
    assert WORKSTREAMS[2].state == 'started'
    assert WORKSTREAMS[3].state == 'complete'
    assert WORKSTREAMS[4].state == 'started'
    assert all(item.state == 'complete' for item in WORKSTREAMS[5:])


def test_fetchable_dataset_references_exist():
    missing = {
        dataset_id
        for item in WORKSTREAMS
        for dataset_id in item.dataset_ids
        if dataset_id not in DATASETS
    }
    assert missing == set()
    assert set(GROUPS['india_first']).issubset(DATASETS)
    assert DATASETS['spacenet7'].expected_md5 == '6eda13b9c28f6f5cdf00a7e8e218c1b1'
    assert DATASETS['openearthmap_mixed'].expected_md5 == '64155d1dc9d3b68536063f79878e1a67'
    assert 'sample-level allowlist' in DATASETS['openearthmap_mixed'].license


def test_pretrained_assets_are_https_and_do_not_overclaim():
    assert {'dinov2_vitb14', 'yolo11x', 'yolo11l_seg', 'weedsgalore_deeplabv3plus'} <= set(ASSETS)
    for asset in ASSETS.values():
        assert asset.url.startswith('https://')
        assert asset.license
        assert asset.redistribution
    assert ASSETS['dinov2_vitb14'].task_trained is False
    assert ASSETS['yolo11x'].task_trained is False
    assert ASSETS['yolo11l_seg'].task_trained is False
    assert ASSETS['weedsgalore_deeplabv3plus'].task_trained is True


def test_committed_pretrained_provenance_matches_asset_registry():
    manifest_path = Path(__file__).parents[1] / 'models' / 'manifests' / 'model_provenance.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = {record['registry_key']: record for record in manifest['files']}
    key_map = {
        'dinov2_vitb14': 'foundation_dinov2_vitb14',
        'yolo11x': 'foundation_yolo11x',
        'yolo11l_seg': 'foundation_yolo11l_seg',
        'weedsgalore_deeplabv3plus': 'upstream_weedsgalore_deeplabv3plus',
    }
    for asset_id, registry_key in key_map.items():
        asset = ASSETS[asset_id]
        record = records[registry_key]
        assert record['source_url'] == asset.url
        assert record['license'] == asset.license
        assert record['task_trained'] is asset.task_trained
        assert len(record['sha256']) == 64
        assert record['bytes'] > 0
