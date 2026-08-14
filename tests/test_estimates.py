"""Pre-flight estimates: batteries, storage and what they refuse to guess.

The cases worth catching are the ones that change what an operator packs. A mission
that cannot be flown on one battery must say so rather than round down, and a camera
the database does not know must be reported as unknown rather than sized from a
plausible default presented as fact.
"""

from __future__ import annotations

import pytest

from mission.estimates import (
    DEFAULT_RESERVE_PCT,
    JPEG_BYTES_PER_PIXEL,
    MIN_SAFE_RESERVE_PCT,
    AircraftProfile,
    estimate_batteries,
    estimate_mission,
    estimate_storage,
)


class TestStorage:
    def test_storage_scales_with_the_number_of_images(self):
        one = estimate_storage(1, "mavic2pro")
        many = estimate_storage(250, "mavic2pro")
        assert many.total_bytes == one.bytes_per_image * 250

    def test_a_known_camera_is_sized_from_its_real_sensor(self):
        estimate = estimate_storage(100, "mavic2pro")
        # 5472 x 3648 is just under 20 MP.
        assert estimate.megapixels == pytest.approx(19.96, abs=0.1)
        assert estimate.known_camera is True
        expected = int(round(5472 * 3648 * JPEG_BYTES_PER_PIXEL))
        assert estimate.bytes_per_image == expected

    def test_an_unknown_camera_is_flagged_rather_than_quietly_defaulted(self):
        """An operator told a guess can pack margin; one shown a fact cannot."""
        estimate = estimate_storage(100, "some-camera-we-have-never-seen")
        assert estimate.known_camera is False
        assert any("not in the preset database" in note for note in estimate.assumptions)

    def test_raw_is_substantially_larger_than_jpeg(self):
        jpeg = estimate_storage(100, "phantom4rtk", image_format="jpeg")
        raw = estimate_storage(100, "phantom4rtk", image_format="raw")
        assert raw.total_bytes > jpeg.total_bytes * 3

    def test_the_compression_assumption_is_stated_not_hidden(self):
        payload = estimate_storage(10, "mavic2pro").to_dict()
        assert any("bytes/pixel" in note for note in payload["assumptions"])

    def test_an_unsupported_format_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported image format"):
            estimate_storage(10, "mavic2pro", image_format="tiff")

    def test_a_negative_image_count_is_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            estimate_storage(-1)

    def test_zero_images_is_zero_bytes_not_an_error(self):
        assert estimate_storage(0, "mavic2pro").total_bytes == 0

    def test_a_realistic_survey_lands_in_a_sane_range(self):
        """600 frames from a 20 MP camera is tens of gigabytes, not hundreds."""
        estimate = estimate_storage(600, "mavic2pro")
        assert 3.0 < estimate.total_gb < 30.0


class TestBatteries:
    def test_a_short_flight_needs_one_battery(self):
        estimate = estimate_batteries(10.0, AircraftProfile(endurance_min=25.0))
        assert estimate.batteries_required == 1
        assert estimate.swaps_required == 0
        assert estimate.fits_in_one_flight is True

    def test_the_reserve_is_held_back_not_flown(self):
        aircraft = AircraftProfile(endurance_min=20.0, reserve_pct=25.0)
        assert aircraft.usable_endurance_min == pytest.approx(15.0)
        # 16 minutes exceeds the usable 15 even though it is under the raw 20.
        estimate = estimate_batteries(16.0, aircraft)
        assert estimate.fits_in_one_flight is False

    def test_a_long_mission_reports_the_swaps_it_needs(self):
        aircraft = AircraftProfile(endurance_min=20.0, reserve_pct=25.0)  # 15 usable
        estimate = estimate_batteries(40.0, aircraft)
        assert estimate.batteries_required == 3
        assert estimate.swaps_required == 2

    def test_a_mission_that_cannot_be_flown_in_one_sortie_says_so(self):
        """Rounding this down would strand the aircraft mid-survey."""
        estimate = estimate_batteries(45.0, AircraftProfile(endurance_min=25.0))
        assert estimate.fits_in_one_flight is False
        assert any("single sortie" in warning for warning in estimate.warnings)

    def test_too_few_batteries_on_the_profile_is_flagged(self):
        aircraft = AircraftProfile(endurance_min=20.0, reserve_pct=25.0, batteries_owned=2)
        estimate = estimate_batteries(40.0, aircraft)
        assert any("batteries are needed" in warning for warning in estimate.warnings)

    def test_an_unsafe_reserve_is_flagged(self):
        estimate = estimate_batteries(5.0, AircraftProfile(endurance_min=25.0, reserve_pct=5.0))
        assert any(str(int(MIN_SAFE_RESERVE_PCT)) in warning for warning in estimate.warnings)

    def test_a_reserve_leaving_no_usable_endurance_is_refused(self):
        with pytest.raises(ValueError, match="no usable endurance"):
            estimate_batteries(5.0, AircraftProfile(endurance_min=20.0, reserve_pct=100.0))

    def test_zero_endurance_is_refused_rather_than_dividing_by_zero(self):
        with pytest.raises(ValueError, match="endurance must be positive"):
            estimate_batteries(5.0, AircraftProfile(endurance_min=0.0))

    def test_the_endurance_assumption_is_stated(self):
        payload = estimate_batteries(5.0).to_dict()
        assert any("still air" in note for note in payload["assumptions"])
        assert payload["reserve_pct"] == DEFAULT_RESERVE_PCT


class TestWholeMission:
    def test_a_real_plan_produces_estimates_for_every_field(self):
        from mission.planner import MissionPlanner

        aoi = [[-81.7510, 41.3035], [-81.7490, 41.3035],
               [-81.7490, 41.3050], [-81.7510, 41.3050]]
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=aoi, altitude_m=60.0)

        estimate = estimate_mission(plan)
        assert estimate["image_count"] > 0
        assert estimate["storage"]["total_gb"] >= 0.0
        assert estimate["battery"]["batteries_required"] >= 1
        assert estimate["distance_m"] > 0

    def test_estimates_work_from_the_dict_form_too(self):
        """A plan that has been through storage or an API must still estimate."""
        from mission.planner import MissionPlanner

        aoi = [[-81.7510, 41.3035], [-81.7490, 41.3035],
               [-81.7490, 41.3050], [-81.7510, 41.3050]]
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=aoi, altitude_m=60.0)

        assert estimate_mission(plan.to_dict())["image_count"] == \
            estimate_mission(plan)["image_count"]

    def test_transit_waypoints_are_not_counted_as_photographs(self):
        plan = {
            "estimated_time_min": 10.0, "camera": "mavic2pro", "path_distance_m": 500.0,
            "flight_recipe": {"world_poses": [
                {"lon": 0.0, "lat": 0.0, "alt_m": 60.0, "trigger": True},
                {"lon": 0.001, "lat": 0.0, "alt_m": 60.0, "trigger": False},
            ]},
        }
        assert estimate_mission(plan)["image_count"] == 1

    def test_the_result_says_these_are_estimates(self):
        plan = {"estimated_time_min": 5.0, "camera": "mavic2pro", "waypoints": [[0, 0, 60]]}
        assert "not measurements" in estimate_mission(plan)["note"]
