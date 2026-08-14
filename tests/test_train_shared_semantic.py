'''Metric and normalization helpers used by shared semantic training.'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from training.semantic_tiles import IGNORE_INDEX
from training.train_shared_semantic import (
    Confusion,
    _atomic_torch_save,
    _normalise,
    class_weights_from_corpus,
    require_declared_class_coverage,
)


def test_confusion_reports_per_class_iou_and_ignores_padding():
    logits = torch.tensor([[
        [[4.0, -2.0], [-2.0, 4.0]],
        [[-2.0, 4.0], [4.0, -2.0]],
    ]])
    target = torch.tensor([[[0, 1], [1, IGNORE_INDEX]]])
    confusion = Confusion.create(2)
    confusion.update(logits, target)
    summary = confusion.summary(['background', 'building'])
    assert summary['mean_iou'] == 1.0
    assert summary['pixel_accuracy'] == 1.0
    assert summary['confusion_matrix'] == [[1, 0], [0, 2]]


def test_imagenet_normalization_is_channel_specific():
    images = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    normalised = _normalise(images)
    assert tuple(normalised.shape) == (1, 3, 2, 2)
    assert np.isclose(float(normalised[0, 0, 0, 0]), (1.0 - 0.485) / 0.229)
    assert not torch.equal(normalised[:, 0], normalised[:, 1])


def test_training_refuses_a_schema_class_without_declared_labels():
    with pytest.raises(ValueError, match=r'class id\(s\): 2, 4'):
        require_declared_class_coverage({'counts': {'uncovered_class_ids': [2, 4]}})


def test_training_accepts_complete_declared_class_coverage():
    require_declared_class_coverage({'counts': {'uncovered_class_ids': []}})


def test_log_inverse_weights_raise_the_rare_class_and_normalize():
    weights = class_weights_from_corpus(
        {'counts': {'declared_class_pixel_counts': {'0': 900, '1': 100}}},
        [0, 1],
    )
    assert np.isclose(np.mean(weights), 1.0)
    assert weights[1] > weights[0]


def test_balanced_loss_refuses_missing_pixel_evidence():
    with pytest.raises(ValueError, match=r'class id\(s\): 1'):
        class_weights_from_corpus(
            {'counts': {'declared_class_pixel_counts': {'0': 100, '1': 0}}},
            [0, 1],
        )


def test_checkpoint_publish_is_atomic(tmp_path):
    destination = tmp_path / 'last.pt'
    _atomic_torch_save({'epoch': 3}, destination)
    assert torch.load(destination, weights_only=False) == {'epoch': 3}
    assert not destination.with_name('last.pt.tmp').exists()
