"""The pipeline has to get through full-resolution drone frames.

At 4896x3672 the CLI analysed 2 of 20 frames in ten minutes and never reached
reconstruction: _filter_connected_components compared the whole label image against
each component in turn (201 s per frame), and the crack forecast sorted the distance
from every skeleton tip to every skeleton pixel (hours on a frame whose heuristic mask
covers half the image). Both are now single passes; these pin that they return the
same answer the slow versions did, fast enough on a frame-sized input.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _reference_filter(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == idx] = 255
    return cleaned


def test_component_filter_matches_the_per_label_version():
    from core.detection import _filter_connected_components

    rng = np.random.default_rng(3)
    mask = (rng.random((300, 400)) > 0.7).astype(np.uint8) * 255
    for min_area in (1, 5, 40):
        assert np.array_equal(_filter_connected_components(mask, min_area), _reference_filter(mask, min_area))


def test_component_filter_handles_a_drone_frame_worth_of_blobs_quickly():
    from core.detection import _filter_connected_components

    rng = np.random.default_rng(4)
    mask = (rng.random((3672, 4896)) > 0.8).astype(np.uint8) * 255
    started = time.perf_counter()
    out = _filter_connected_components(mask, 30)
    assert time.perf_counter() - started < 5.0
    assert out.dtype == np.uint8 and out.shape == mask.shape


def _reference_tip_directions(skeleton, tips):
    points = np.argwhere(skeleton == 1)
    result = []
    for tip in tips:
        d2 = np.sum((points - tip) ** 2, axis=1)
        d2[d2 == 0] = np.iinfo(d2.dtype).max
        nearest = int(np.min(d2))
        # Every neighbour at the minimum distance is an equally valid answer.
        options = [(tip - p).astype(np.float32) for p in points[d2 == nearest]]
        result.append((tip, [o / np.linalg.norm(o) for o in options]))
    return result


def test_tip_directions_point_away_from_the_nearest_skeleton_pixel():
    from core.propagation import CrackPropagationForecaster as Forecaster

    mask = np.zeros((80, 80), np.uint8)
    cv2.line(mask, (10, 10), (60, 30), 255, 3)
    cv2.line(mask, (20, 70), (70, 60), 255, 3)
    skeleton = Forecaster._skeleton(mask)
    tips = Forecaster._tip_points(skeleton)
    fast = Forecaster._tip_directions(skeleton, tips)
    reference = _reference_tip_directions(skeleton, tips)

    assert len(fast) == len(reference) > 0
    for (tip, direction), (ref_tip, options) in zip(fast, reference):
        assert np.array_equal(tip, ref_tip)
        assert any(np.allclose(direction, option, atol=1e-6) for option in options)


def test_the_forecast_finishes_on_a_frame_sized_mask():
    from core.propagation import CrackPropagationForecaster as Forecaster

    rng = np.random.default_rng(5)
    noisy = (rng.random((1836, 2448)) > 0.6).astype(np.uint8) * 255
    started = time.perf_counter()
    sequence, metrics = Forecaster().forecast(noisy, steps=2)
    assert time.perf_counter() - started < 120.0
    assert len(sequence) == len(metrics) == 3
