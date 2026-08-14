"""The check that decides whether it is safe to leave the site.

Everything here is about the asymmetry between finding a problem now and finding it
later. A missed capture point is a ten-minute re-fly with the aircraft still out of its
case, and a wasted trip once it is packed. So the tests check that genuinely
unrecoverable problems block, that ordinary imperfection does not, and above all that
anything which could not be checked is reported as unchecked rather than quietly
counted as a pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.site_verification import (
    BLUR_TOLERANCE_PCT,
    MINIMUM_COVERAGE_PCT,
    assess_quality,
    find_unreadable,
    list_images,
    verify_site,
)

BASE_LON, BASE_LAT = -81.7505, 41.3042


def sharp_image(seed: int = 0) -> np.ndarray:
    """A frame with plenty of high-frequency detail, so it reads as in focus."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def blurred_image(seed: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(sharp_image(seed), (31, 31), 12.0)


def write_images(folder, count: int, maker=sharp_image, start: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(start, start + count):
        path = folder / f"DSC{i:05d}.JPG"
        cv2.imwrite(str(path), maker(i))
        paths.append(path)
    return paths


def plan_with(count: int, spacing_deg: float = 0.0005) -> dict:
    return {
        "template": "grid", "camera": "mavic2pro", "altitude_m": 60.0,
        "flight_recipe": {"world_poses": [
            {"lon": BASE_LON + i * spacing_deg, "lat": BASE_LAT, "alt_m": 60.0,
             "trigger": True}
            for i in range(count)
        ]},
    }


def fake_matching(monkeypatch, folder, plan, captured_indices):
    """Stand in for EXIF reading, which test JPEGs written by cv2 do not carry."""
    from core.capture_matching import CapturedImage

    poses = plan["flight_recipe"]["world_poses"]
    images = [
        CapturedImage(path=str(folder / f"DSC{i:05d}.JPG"),
                      longitude=poses[i]["lon"], latitude=poses[i]["lat"],
                      altitude_m=60.0)
        for i in captured_indices
    ]
    monkeypatch.setattr("core.capture_matching.images_from_folder", lambda f: images)


class TestUnreadableFiles:
    def test_a_truncated_file_is_caught_by_decoding_not_by_size(self, tmp_path):
        """A truncated JPEG is the right sort of size and only fails when read."""
        write_images(tmp_path, 2)
        broken = tmp_path / "DSC00099.JPG"
        broken.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4000)

        found = find_unreadable(list_images(tmp_path))
        assert [Path(f["path"]).name for f in found] == ["DSC00099.JPG"]

    def test_an_empty_file_is_reported_as_empty(self, tmp_path):
        write_images(tmp_path, 1)
        (tmp_path / "DSC00042.JPG").write_bytes(b"")

        found = find_unreadable(list_images(tmp_path))
        assert any("empty" in f["reason"] for f in found)

    def test_readable_images_produce_no_findings(self, tmp_path):
        write_images(tmp_path, 3)
        assert find_unreadable(list_images(tmp_path)) == []


class TestQuality:
    def test_sharp_images_are_not_flagged(self, tmp_path):
        write_images(tmp_path, 6, sharp_image)
        quality = assess_quality(list_images(tmp_path))
        assert quality["flagged_count"] == 0

    def test_blurred_images_are_flagged(self, tmp_path):
        write_images(tmp_path, 6, blurred_image)
        quality = assess_quality(list_images(tmp_path))
        assert quality["flagged_count"] == 6
        assert "blur" in quality["flagged"][0]["flags"]

    def test_sampling_is_declared_so_a_rate_is_not_over_read(self, tmp_path):
        """A rate measured over 4 frames must not read as a claim about 20."""
        write_images(tmp_path, 20, sharp_image)
        quality = assess_quality(list_images(tmp_path), sample_limit=4)

        assert quality["sampled"] is True
        assert quality["checked"] == 4
        assert quality["population"] == 20

    def test_a_small_set_is_checked_in_full(self, tmp_path):
        write_images(tmp_path, 5)
        quality = assess_quality(list_images(tmp_path), sample_limit=200)
        assert quality["sampled"] is False
        assert quality["checked"] == 5


class TestVerdict:
    def test_a_clean_survey_is_safe_to_leave(self, tmp_path, monkeypatch):
        plan = plan_with(8)
        write_images(tmp_path, 8)
        fake_matching(monkeypatch, tmp_path, plan, range(8))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert verdict["ok"] is True
        assert verdict["blocking"] == []
        assert "Safe to leave" in verdict["summary"]

    def test_missed_capture_points_block_departure(self, tmp_path, monkeypatch):
        """The whole reason this check exists."""
        plan = plan_with(10)
        write_images(tmp_path, 6)
        fake_matching(monkeypatch, tmp_path, plan, range(6))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert verdict["ok"] is False
        assert any("Re-fly points" in b for b in verdict["blocking"])

    def test_the_blocking_message_names_the_points_to_re_fly(self, tmp_path, monkeypatch):
        plan = plan_with(10)
        write_images(tmp_path, 5)
        fake_matching(monkeypatch, tmp_path, plan, range(5))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert "5" in " ".join(verdict["blocking"])

    def test_widespread_blur_blocks_departure(self, tmp_path, monkeypatch):
        plan = plan_with(8)
        write_images(tmp_path, 8, blurred_image)
        fake_matching(monkeypatch, tmp_path, plan, range(8))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert verdict["ok"] is False
        assert any("systematic problem" in b for b in verdict["blocking"])

    def test_a_few_soft_frames_are_a_warning_not_a_blocker(self, tmp_path, monkeypatch):
        """Ordinary imperfection must not strand a pilot on site."""
        plan = plan_with(40)
        write_images(tmp_path, 39, sharp_image)
        write_images(tmp_path, 1, blurred_image, start=39)
        fake_matching(monkeypatch, tmp_path, plan, range(40))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert verdict["ok"] is True
        assert verdict["details"]["quality"]["flagged_pct"] <= BLUR_TOLERANCE_PCT

    def test_unreadable_files_block_because_the_card_is_still_here(self, tmp_path, monkeypatch):
        plan = plan_with(4)
        write_images(tmp_path, 4)
        (tmp_path / "DSC00050.JPG").write_bytes(b"\xff\xd8" + b"\x00" * 2000)
        fake_matching(monkeypatch, tmp_path, plan, range(4))

        verdict = verify_site(tmp_path, plan).to_dict()
        assert verdict["ok"] is False
        assert any("Re-copy" in b for b in verdict["blocking"])

    def test_an_empty_folder_blocks_and_says_to_check_the_copy(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        verdict = verify_site(tmp_path).to_dict()
        assert verdict["ok"] is False
        assert any("no images at all" in b for b in verdict["blocking"])

    def test_a_missing_folder_is_refused_clearly(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            verify_site(tmp_path / "never_existed")


class TestUnchecked:
    def test_without_a_plan_coverage_is_reported_as_unchecked(self, tmp_path):
        """The dangerous alternative is reporting full coverage against no plan."""
        write_images(tmp_path, 5)
        verdict = verify_site(tmp_path).to_dict()

        assert any("coverage against the plan was not checked" in u
                   for u in verdict["unchecked"])
        assert verdict["details"].get("coverage") is None

    def test_a_plan_without_capture_points_is_unchecked_not_passed(self, tmp_path):
        write_images(tmp_path, 3)
        verdict = verify_site(tmp_path, {"flight_recipe": {"world_poses": []}}).to_dict()
        assert any("no capture points" in u for u in verdict["unchecked"])

    def test_the_note_says_unchecked_is_not_passed(self, tmp_path):
        write_images(tmp_path, 2)
        verdict = verify_site(tmp_path).to_dict()
        assert "rather than counted as passed" in verdict["note"]

    def test_the_verdict_is_advisory_and_says_so(self, tmp_path):
        write_images(tmp_path, 2)
        verdict = verify_site(tmp_path).to_dict()
        assert "the operator decides" in verdict["note"]


class TestUngeotagged:
    def test_images_without_gps_are_warned_about(self, tmp_path, monkeypatch):
        from core.capture_matching import CapturedImage

        plan = plan_with(4)
        write_images(tmp_path, 4)
        poses = plan["flight_recipe"]["world_poses"]
        images = [
            CapturedImage(path=str(tmp_path / f"DSC{i:05d}.JPG"),
                          longitude=poses[i]["lon"], latitude=poses[i]["lat"])
            for i in range(4)
        ]
        images.append(CapturedImage(path=str(tmp_path / "DSC00099.JPG")))
        monkeypatch.setattr("core.capture_matching.images_from_folder", lambda f: images)

        verdict = verify_site(tmp_path, plan).to_dict()
        assert any("no GPS position" in w for w in verdict["warnings"])
