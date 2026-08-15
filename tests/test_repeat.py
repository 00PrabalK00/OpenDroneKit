"""Repeat surveys, and whether two of them can honestly be compared.

The failure worth designing against is quiet. Fly the same site again with a different
camera at the same altitude and both surveys are internally valid, both produce clean
orthomosaics, and the difference between them is partly an artefact of the lens. Nothing
errors. So the tests below check that a camera change moves the altitude to hold ground
resolution rather than holding altitude and losing the comparison, and that every repeat
states what it changed.
"""

from __future__ import annotations

import pytest

from mission.repeat import (
    GSD_COMPARABILITY_TOLERANCE,
    REPEAT_EXACT,
    REPEAT_MODIFIED_BOUNDARY,
    REPEAT_UPDATED_TERRAIN,
    altitude_for_matching_gsd,
    compare_specifications,
    repeat_mission,
)

BASE_LON, BASE_LAT = -81.7505, 41.3042


def original(camera="mavic2pro", altitude=60.0, count=20, **extra) -> dict:
    summary = {
        "template": "grid", "camera": camera, "altitude_m": altitude,
        "front_overlap_pct": 75.0, "side_overlap_pct": 65.0,
        "waypoints": [[BASE_LON + i * 0.0004, BASE_LAT, altitude] for i in range(count)],
    }
    summary.update(extra)
    return summary


class TestExactRepeat:
    def test_an_exact_repeat_reuses_the_stored_waypoints(self):
        plan = original()
        repeat = repeat_mission(plan, mode=REPEAT_EXACT)

        assert len(repeat.waypoints) == 20
        assert repeat.altitude_m == pytest.approx(60.0)

    def test_an_unchanged_repeat_says_so_rather_than_listing_nothing(self):
        repeat = repeat_mission(original(), mode=REPEAT_EXACT)
        assert any("re-flies the original exactly" in c for c in repeat.changes)

    def test_an_exact_repeat_is_comparable(self):
        assert repeat_mission(original(), mode=REPEAT_EXACT).comparability == "comparable"

    def test_a_mission_with_no_waypoints_cannot_be_repeated_exactly(self):
        with pytest.raises(ValueError, match="nothing to repeat"):
            repeat_mission(original(count=0), mode=REPEAT_EXACT)

    def test_an_unknown_repeat_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown repeat mode"):
            repeat_mission(original(), mode="vibes")


class TestDifferentAircraft:
    def test_a_camera_change_holds_resolution_not_altitude(self):
        """The whole point: preserving altitude would preserve the flight and lose the
        comparison."""
        repeat = repeat_mission(original(camera="mavic2pro", altitude=60.0),
                                camera="zenmuse_p1")

        assert repeat.altitude_m != pytest.approx(60.0)
        assert repeat.achieved_gsd_cm == pytest.approx(repeat.target_gsd_cm, rel=1e-6)

    def test_a_longer_lens_flies_higher_for_the_same_resolution(self):
        repeat = repeat_mission(original(camera="mavic2pro", altitude=60.0),
                                camera="zenmuse_p1")
        assert repeat.altitude_m > 60.0

    def test_the_altitude_change_is_reported_with_the_resolution_it_holds(self):
        repeat = repeat_mission(original(camera="mavic2pro"), camera="phantom4rtk")
        change = " ".join(repeat.changes)

        assert "mavic2pro -> phantom4rtk" in change
        assert "cm/px" in change

    def test_the_stored_waypoints_take_the_new_altitude(self):
        repeat = repeat_mission(original(camera="mavic2pro", altitude=60.0),
                                camera="zenmuse_p1")
        assert all(w[2] == pytest.approx(repeat.altitude_m) for w in repeat.waypoints)

    def test_the_helper_returns_both_altitude_and_the_gsd_it_matches(self):
        altitude, gsd = altitude_for_matching_gsd("mavic2pro", 60.0, "phantom4rtk")
        assert altitude > 0 and gsd > 0

    def test_an_unknown_replacement_camera_is_flagged_as_untrustworthy(self):
        repeat = repeat_mission(original(), camera="some-camera-nobody-owns")
        assert any("placeholder sensor geometry" in w for w in repeat.warnings)

    def test_an_unknown_original_camera_undermines_the_comparison(self):
        repeat = repeat_mission(original(camera="mystery-cam"), mode=REPEAT_EXACT)
        assert any("not in the database" in w for w in repeat.warnings)


class TestUpdatedTerrain:
    def test_updated_terrain_requires_a_terrain_source(self):
        """Without one it is an exact repeat wearing a different name."""
        with pytest.raises(ValueError, match="needs a terrain source"):
            repeat_mission(original(), mode=REPEAT_UPDATED_TERRAIN)

    def test_updated_terrain_warns_that_elevation_change_is_partly_the_model(self):
        repeat = repeat_mission(original(), mode=REPEAT_UPDATED_TERRAIN,
                                terrain_source="site_2026.tif")

        assert repeat.comparability == "comparable_with_caveat"
        assert any("new terrain model" in w for w in repeat.warnings)

    def test_the_terrain_source_is_named_in_the_changes(self):
        repeat = repeat_mission(original(), mode=REPEAT_UPDATED_TERRAIN,
                                terrain_source="site_2026.tif")
        assert any("site_2026.tif" in c for c in repeat.changes)


class TestModifiedBoundary:
    def test_a_modified_boundary_needs_a_boundary(self):
        with pytest.raises(ValueError, match="at least three points"):
            repeat_mission(original(), mode=REPEAT_MODIFIED_BOUNDARY)

    def test_a_modified_boundary_reports_partial_overlap(self):
        repeat = repeat_mission(
            original(), mode=REPEAT_MODIFIED_BOUNDARY,
            boundary=[[-81.751, 41.303], [-81.749, 41.303], [-81.749, 41.305]])

        assert repeat.comparability == "partial_overlap"
        assert any("new coverage, not new damage" in w for w in repeat.warnings)

    def test_the_old_waypoints_are_not_carried_into_a_new_area(self):
        """They describe the old boundary and would fly outside the new one."""
        repeat = repeat_mission(
            original(), mode=REPEAT_MODIFIED_BOUNDARY,
            boundary=[[-81.751, 41.303], [-81.749, 41.303], [-81.749, 41.305]])
        assert repeat.waypoints == []


class TestComparability:
    def test_two_identical_specifications_are_comparable(self):
        result = compare_specifications(original(), original())

        assert result["comparable"] is True
        assert result["differences"] == []
        assert "can be read as change" in result["note"]

    def test_a_camera_change_at_the_same_altitude_is_not_comparable(self):
        """Both surveys are valid; differencing them measures the lens as well."""
        result = compare_specifications(
            original(camera="mavic2pro", altitude=60.0),
            original(camera="zenmuse_p1", altitude=60.0))

        assert result["comparable"] is False
        assert any("camera" in d for d in result["differences"])
        assert "change in specification" in result["note"]

    def test_the_resolution_difference_is_quantified(self):
        result = compare_specifications(
            original(camera="mavic2pro", altitude=60.0),
            original(camera="mavic2pro", altitude=120.0))

        assert result["gsd_difference_pct"] == pytest.approx(100.0, abs=1.0)
        assert result["comparable"] is False

    def test_a_small_altitude_change_stays_comparable(self):
        """Tolerance exists so a metre of drift does not invalidate a repeat survey."""
        result = compare_specifications(
            original(altitude=60.0), original(altitude=61.0))
        assert result["comparable"] is True

    def test_an_overlap_change_is_reported_even_when_resolution_matches(self):
        result = compare_specifications(
            original(front_overlap_pct=75.0), original(front_overlap_pct=85.0))
        assert any("front overlap" in d for d in result["differences"])

    def test_the_tolerance_is_the_documented_one(self):
        """A change just inside it passes, just outside it does not."""
        base = 60.0
        inside = base * (1 + GSD_COMPARABILITY_TOLERANCE * 0.5)
        outside = base * (1 + GSD_COMPARABILITY_TOLERANCE * 3)

        assert compare_specifications(original(altitude=base),
                                      original(altitude=inside))["comparable"] is True
        assert compare_specifications(original(altitude=base),
                                      original(altitude=outside))["comparable"] is False


class TestAgainstARealPlan:
    def test_a_planner_mission_can_be_repeated_with_a_different_aircraft(self):
        from mission.planner import MissionPlanner

        aoi = [[-81.7510, 41.3035], [-81.7490, 41.3035],
               [-81.7490, 41.3050], [-81.7510, 41.3050]]
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=aoi,
                                         altitude_m=60.0, camera="mavic2pro")

        repeat = repeat_mission(plan.to_dict(), camera="phantom4rtk")

        assert repeat.waypoints
        assert repeat.achieved_gsd_cm == pytest.approx(repeat.target_gsd_cm, rel=1e-6)
        assert repeat.comparability == "comparable"
