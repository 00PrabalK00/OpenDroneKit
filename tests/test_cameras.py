"""Camera profiles, and the refusal to invent one.

Sensor geometry decides the GSD of every survey flown with it. A wrong profile does not
fail: the mission flies, the photographs are taken, and the deliverable is simply at a
different resolution than the one the client specified. So these tests check the
arithmetic against figures that can be worked out by hand, and check that an unknown
camera is reported as unknown rather than resolved to a plausible default.
"""

from __future__ import annotations

import json

import pytest

from mission.cameras import (
    BUILT_IN,
    CameraProfile,
    UnknownCamera,
    all_profiles,
    delete_user_profile,
    describe,
    load_user_profiles,
    require,
    resolve,
    save_user_profile,
)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "cameras.json"


class TestGeometry:
    def test_gsd_matches_the_hand_calculation(self):
        """P4RTK at 100 m: 13.2/8.8 * 100 / 5472 m per pixel, about 2.7 cm."""
        profile = BUILT_IN["phantom4rtk"]
        assert profile.gsd_cm(100.0) == pytest.approx(2.74, abs=0.02)

    def test_gsd_scales_linearly_with_altitude(self):
        profile = BUILT_IN["mavic2pro"]
        assert profile.gsd_cm(120.0) == pytest.approx(profile.gsd_cm(60.0) * 2, rel=1e-9)

    def test_a_longer_lens_gives_a_finer_gsd_at_the_same_height(self):
        """The Zenmuse P1's 35 mm lens is why it is the survey payload."""
        assert BUILT_IN["zenmuse_p1"].gsd_cm(100.0) < BUILT_IN["mavic2pro"].gsd_cm(100.0)

    def test_footprint_matches_the_similar_triangles(self):
        profile = BUILT_IN["phantom4rtk"]
        width, height = profile.footprint_m(88.0)
        # At 88 m with an 8.8 mm lens the scale is exactly 10 000:1.
        assert width == pytest.approx(132.0, abs=0.5)
        assert height == pytest.approx(88.0, abs=0.5)

    def test_altitude_for_a_target_gsd_inverts_the_gsd_calculation(self):
        """Surveys are specified as a GSD, so this is the direction operators use."""
        profile = BUILT_IN["mavic3e"]
        altitude = profile.altitude_for_gsd_m(2.0)
        assert profile.gsd_cm(altitude) == pytest.approx(2.0, rel=1e-6)

    def test_field_of_view_is_derived_from_the_sensor_and_lens(self):
        profile = BUILT_IN["mavic2pro"]
        assert 60.0 < profile.horizontal_fov_deg < 80.0
        assert profile.vertical_fov_deg < profile.horizontal_fov_deg

    def test_pixel_pitch_is_reported_in_micrometres(self):
        profile = BUILT_IN["phantom4rtk"]
        assert profile.pixel_pitch_um == pytest.approx(2.41, abs=0.05)

    def test_a_zero_altitude_is_refused_rather_than_dividing_by_zero(self):
        with pytest.raises(ValueError, match="Altitude must be positive"):
            BUILT_IN["mavic2pro"].gsd_cm(0.0)


class TestValidation:
    def test_an_impossible_focal_length_is_refused(self):
        with pytest.raises(ValueError, match="focal length"):
            CameraProfile("bad", "Bad", 13.2, 8.8, 0.001, 4000, 3000)

    def test_an_impossible_sensor_size_is_refused(self):
        with pytest.raises(ValueError, match="sensor width"):
            CameraProfile("bad", "Bad", 900.0, 8.8, 10.0, 4000, 3000)

    def test_zero_resolution_is_refused(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            CameraProfile("bad", "Bad", 13.2, 8.8, 10.0, 0, 3000)


class TestResolution:
    def test_a_known_camera_resolves_and_says_so(self):
        profile, known = resolve("phantom4rtk")
        assert known is True
        assert profile.key == "phantom4rtk"

    def test_lookup_tolerates_spacing_and_case(self):
        assert resolve("Phantom4RTK")[1] is True
        assert resolve("mavic 2 pro".replace(" ", ""))[1] is True

    def test_an_unknown_camera_is_reported_as_unknown(self):
        """The placeholder is still returned so callers work, but the flag is the point."""
        profile, known = resolve("some-camera-nobody-has-described")
        assert known is False
        assert profile.key == "custom"

    def test_require_refuses_to_substitute_a_placeholder(self):
        with pytest.raises(UnknownCamera, match="not in the database"):
            require("no-such-camera")

    def test_require_lists_what_is_available(self):
        with pytest.raises(UnknownCamera, match="phantom4rtk"):
            require("no-such-camera")


class TestDescribe:
    def test_a_description_carries_the_working_numbers(self):
        payload = describe("mavic3e", altitude_m=80.0)
        assert payload["known"] is True
        assert payload["gsd_cm"] > 0
        assert len(payload["footprint_m"]) == 2
        assert "warning" not in payload

    def test_an_unknown_camera_carries_a_warning_not_a_silent_default(self):
        payload = describe("mystery-cam", altitude_m=80.0)
        assert payload["known"] is False
        assert "do not describe your equipment" in payload["warning"]

    def test_the_generic_profile_declares_itself_indicative(self):
        assert "indicative" in BUILT_IN["custom"].notes


class TestThermal:
    def test_thermal_cameras_are_flagged(self):
        assert BUILT_IN["mavic3t_thermal"].thermal is True
        assert BUILT_IN["mavic2pro"].thermal is False

    def test_a_thermal_sensor_has_a_much_coarser_gsd(self):
        """640x512 over the same ground is why thermal is flown lower."""
        thermal = BUILT_IN["mavic3t_thermal"].gsd_cm(60.0)
        rgb = BUILT_IN["mavic3t_wide"].gsd_cm(60.0)
        assert thermal > rgb * 3


class TestUserProfiles:
    def test_a_user_camera_is_saved_and_read_back(self, store):
        profile = CameraProfile("survey_cam_a", "Our fixed-wing payload",
                                23.5, 15.6, 16.0, 6000, 4000, source="user")
        save_user_profile(profile, store)

        loaded = load_user_profiles(store)
        assert "survey_cam_a" in loaded
        assert loaded["survey_cam_a"].focal_mm == 16.0
        assert loaded["survey_cam_a"].source == "user"

    def test_user_cameras_resolve_alongside_built_in_ones(self, store):
        save_user_profile(
            CameraProfile("survey_cam_a", "Ours", 23.5, 15.6, 16.0, 6000, 4000), store)
        profile, known = resolve("survey_cam_a", store)
        assert known is True
        assert profile.name == "Ours"

    def test_a_user_camera_cannot_shadow_a_published_one(self, store):
        """Overriding manufacturer geometry would be undetectable later."""
        with pytest.raises(ValueError, match="built-in camera"):
            save_user_profile(
                CameraProfile("phantom4rtk", "Mine", 13.2, 8.8, 20.0, 5472, 3648), store)

    def test_an_invalid_user_camera_is_refused_at_save_time(self, store):
        with pytest.raises(ValueError):
            save_user_profile(
                CameraProfile.__new__(CameraProfile).__class__(
                    "bad_cam", "Bad", 13.2, 8.8, 900.0, 4000, 3000), store)

    def test_a_user_camera_can_be_deleted(self, store):
        save_user_profile(
            CameraProfile("temp_cam", "Temp", 13.2, 8.8, 10.0, 4000, 3000), store)
        assert delete_user_profile("temp_cam", store) is True
        assert "temp_cam" not in load_user_profiles(store)

    def test_deleting_something_absent_is_false_not_an_error(self, store):
        assert delete_user_profile("never_existed", store) is False

    def test_one_malformed_entry_does_not_hide_the_rest_of_the_fleet(self, store):
        store.write_text(json.dumps({
            "good_cam": {"name": "Good", "sensor_w_mm": 13.2, "sensor_h_mm": 8.8,
                         "focal_mm": 10.0, "image_w_px": 4000, "image_h_px": 3000},
            "broken_cam": {"name": "Broken"},
        }), encoding="utf-8")

        loaded = load_user_profiles(store)
        assert "good_cam" in loaded
        assert "broken_cam" not in loaded

    def test_a_corrupt_store_reads_as_empty_rather_than_crashing(self, store):
        store.write_text("{ not json", encoding="utf-8")
        assert load_user_profiles(store) == {}

    def test_all_profiles_includes_both_sources(self, store):
        save_user_profile(
            CameraProfile("mine", "Mine", 13.2, 8.8, 10.0, 4000, 3000), store)
        profiles = all_profiles(store)
        assert "phantom4rtk" in profiles and "mine" in profiles


class TestAgreementWithThePlanner:
    def test_profiles_agree_with_the_planner_presets_they_replace(self):
        """The planner's own GSD must not disagree with the camera database."""
        from mission.planner import CAMERA_PRESETS, _estimate_gsd_cm

        for key in ("mavic2pro", "phantom4rtk", "custom"):
            preset = CAMERA_PRESETS[key]
            profile = BUILT_IN[key]
            assert profile.sensor_w_mm == pytest.approx(preset["sensor_w_mm"])
            assert profile.focal_mm == pytest.approx(preset["focal_mm"])
            assert profile.image_w_px == preset["image_w_px"]
            assert profile.gsd_cm(60.0) == pytest.approx(_estimate_gsd_cm(60.0, key),
                                                         rel=1e-9)
