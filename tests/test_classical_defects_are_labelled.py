"""What the classical detector hands a report, and the GPS the data library reads.

Driving the engine on two ordinary park aerials produced "corrosion, confidence 0.95,
critical" with a bounding box equal to the whole frame. The confidence was
min(0.95, 0.4 + 50 * pixel_ratio) -- a threshold hit rescaled to look like a
probability. The same report said GPS "no" for 24 of 24 geotagged images because the
GPS block was read from the wrong EXIF directory.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.data_library import extract_image_metadata
from core.defect_engine import DefectDetectionConfig, _bbox_from_mask, detect_defects_classical
from test_reports_in_the_desktop_window import _geotagged_jpeg


def test_gps_is_read_from_a_geotagged_photo(tmp_path) -> None:
    photo = tmp_path / "geo.jpg"
    _geotagged_jpeg(photo, 41.3038, -81.7504, 100)
    meta = extract_image_metadata(photo)
    assert meta["gps_lat"] == pytest.approx(41.3038, abs=1e-6)
    assert meta["gps_lon"] == pytest.approx(-81.7504, abs=1e-6)
    assert meta["camera_make"] == "TestCam"


class TestClassicalOutput:
    @pytest.fixture
    def line_image(self, tmp_path):
        image = np.full((400, 600, 3), 180, np.uint8)
        cv2.line(image, (100, 200), (300, 230), (20, 20, 20), 3)
        path = tmp_path / "line.png"
        cv2.imwrite(str(path), image)
        return path

    def test_no_pseudo_confidence_is_reported(self, line_image) -> None:
        found = detect_defects_classical(line_image, DefectDetectionConfig())
        assert found, "the dark line should be picked up"
        for defect in found:
            assert defect.confidence == 0.0
            assert defect.notes.startswith("heuristic")
            assert defect.source == "classical"

    def test_a_whole_frame_hit_is_not_reported_as_a_located_defect(self, tmp_path) -> None:
        from core.detection import load_image, metal_texture_heuristic

        noise = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
        path = tmp_path / "noise.png"
        cv2.imwrite(str(path), noise)
        # Precondition: the underlying detector does fire, across the whole frame.
        metal = metal_texture_heuristic(load_image(path), min_area_px=DefectDetectionConfig().min_area_px)
        box = _bbox_from_mask(metal.mask)
        assert metal.defect_pixels > 0 and (box[2] - box[0]) * (box[3] - box[1]) >= 0.9 * 400 * 600
        assert detect_defects_classical(path, DefectDetectionConfig()) == []
