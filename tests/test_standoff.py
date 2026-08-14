"""Stand-off policies and the clearance guarantee.

The case that matters is the conflict: an inspection ordered at a resolution that would
require flying closer to the building than the minimum clearance allows. Silently
honouring the resolution flies the aircraft too close; silently honouring the clearance
delivers the wrong survey while reporting success. Both are refusals to make the
operator's decision visible, and the tests below pin which one happens under which
setting.

The other half is verification. A policy is an intention; waypoints are what gets flown,
and only measuring the compiled points against the geometry shows whether the intention
survived compilation.
"""

from __future__ import annotations

import pytest

from mission.standoff import (
    ABSOLUTE_MINIMUM_CLEARANCE_M,
    StandoffConflict,
    distance_to_surface_m,
    resolve_standoff,
    standoff_for_gsd,
    verify_clearance,
)

# A wall running 100 m east along y = 0.
WALL = [(0.0, 0.0), (100.0, 0.0)]


class TestFixedPolicy:
    def test_a_fixed_stand_off_is_used_as_given(self):
        decision = resolve_standoff(policy="fixed", fixed_distance_m=12.0)
        assert decision.distance_m == pytest.approx(12.0)
        assert decision.clamped is False

    def test_a_fixed_stand_off_inside_the_clearance_is_raised(self):
        decision = resolve_standoff(policy="fixed", fixed_distance_m=3.0,
                                    minimum_clearance_m=6.0)
        assert decision.distance_m == pytest.approx(6.0)
        assert decision.clamped is True

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="Unknown stand-off policy"):
            resolve_standoff(policy="whatever-feels-right")


class TestPerSurfacePolicy:
    def test_each_surface_gets_its_own_distance(self):
        table = {"north": 6.0, "south": 15.0}
        north = resolve_standoff("north", policy="per_surface", per_surface_m=table)
        south = resolve_standoff("south", policy="per_surface", per_surface_m=table)

        assert north.distance_m == pytest.approx(6.0)
        assert south.distance_m == pytest.approx(15.0)

    def test_a_surface_with_no_entry_is_refused_rather_than_defaulted(self):
        """Falling back to a default would fly one face at the wrong distance."""
        with pytest.raises(KeyError, match="No stand-off defined"):
            resolve_standoff("east", policy="per_surface",
                             per_surface_m={"north": 6.0})

    def test_the_error_says_which_surfaces_are_defined(self):
        with pytest.raises(KeyError, match="north"):
            resolve_standoff("east", policy="per_surface", per_surface_m={"north": 6.0})


class TestAdaptivePolicy:
    def test_stand_off_is_derived_from_the_requested_resolution(self):
        """A finer GSD must put the camera closer, not merely change a label."""
        fine = resolve_standoff(policy="adaptive", camera="mavic2pro",
                                target_gsd_cm=0.2, minimum_clearance_m=2.0)
        coarse = resolve_standoff(policy="adaptive", camera="mavic2pro",
                                  target_gsd_cm=1.0, minimum_clearance_m=2.0)
        assert fine.distance_m < coarse.distance_m

    def test_the_achieved_resolution_matches_what_was_asked_when_it_fits(self):
        decision = resolve_standoff(policy="adaptive", camera="mavic2pro",
                                    target_gsd_cm=1.0, minimum_clearance_m=2.0)
        assert decision.achieved_gsd_cm == pytest.approx(1.0, rel=1e-3)
        assert decision.clamped is False

    def test_adaptive_without_a_target_gsd_is_refused(self):
        with pytest.raises(ValueError, match="target_gsd_cm must be supplied"):
            resolve_standoff(policy="adaptive", camera="mavic2pro")

    def test_a_longer_lens_reaches_the_same_resolution_from_further_away(self):
        """Which is the honest answer to a GSD that will not fit inside the clearance."""
        wide = resolve_standoff(policy="adaptive", camera="mavic2pro",
                                target_gsd_cm=0.5, minimum_clearance_m=2.0)
        long_lens = resolve_standoff(policy="adaptive", camera="zenmuse_p1",
                                     target_gsd_cm=0.5, minimum_clearance_m=2.0)
        assert long_lens.distance_m > wide.distance_m


class TestTheConflict:
    def test_an_impossible_resolution_clamps_and_says_what_was_lost(self):
        decision = resolve_standoff(policy="adaptive", camera="mavic2pro",
                                    target_gsd_cm=0.05, minimum_clearance_m=10.0)

        assert decision.clamped is True
        assert decision.distance_m == pytest.approx(10.0)
        # The requested figure is preserved next to what is actually achievable.
        assert decision.requested_gsd_cm == pytest.approx(0.05)
        assert decision.achieved_gsd_cm > decision.requested_gsd_cm
        assert any("not achievable" in w for w in decision.warnings)

    def test_a_survey_specified_to_a_tolerance_can_refuse_to_compromise(self):
        """When the resolution is contractual, silently delivering less is worse."""
        with pytest.raises(StandoffConflict, match="inside the"):
            resolve_standoff(policy="adaptive", camera="mavic2pro",
                             target_gsd_cm=0.05, minimum_clearance_m=10.0,
                             allow_gsd_compromise=False)

    def test_the_conflict_message_offers_the_real_options(self):
        with pytest.raises(StandoffConflict, match="longer lens"):
            resolve_standoff(policy="adaptive", camera="mavic2pro",
                             target_gsd_cm=0.05, minimum_clearance_m=10.0,
                             allow_gsd_compromise=False)

    def test_an_unknown_camera_makes_an_adaptive_stand_off_untrustworthy(self):
        decision = resolve_standoff(policy="adaptive", camera="not-a-real-camera",
                                    target_gsd_cm=1.0, minimum_clearance_m=2.0)
        assert any("placeholder sensor geometry" in w for w in decision.warnings)


class TestTheFloor:
    def test_clearance_never_goes_below_the_absolute_floor(self):
        decision = resolve_standoff(policy="fixed", fixed_distance_m=0.5,
                                    minimum_clearance_m=0.5)
        assert decision.distance_m >= ABSOLUTE_MINIMUM_CLEARANCE_M

    def test_asking_for_less_than_the_floor_is_reported(self):
        decision = resolve_standoff(policy="fixed", fixed_distance_m=1.0,
                                    minimum_clearance_m=1.0)
        assert any("below the" in w for w in decision.warnings)


class TestGeometry:
    def test_distance_to_a_wall_is_perpendicular_where_that_is_nearest(self):
        assert distance_to_surface_m((50.0, 8.0), WALL) == pytest.approx(8.0)

    def test_a_point_beyond_the_end_measures_to_the_end_not_the_infinite_line(self):
        """Otherwise a point past the corner reads as safely far from the wall."""
        # 30 m east of the wall's end, level with it: the true distance is 30.
        assert distance_to_surface_m((130.0, 0.0), WALL) == pytest.approx(30.0)

    def test_a_diagonal_approach_is_measured_correctly(self):
        assert distance_to_surface_m((103.0, 4.0), WALL) == pytest.approx(5.0)

    def test_a_surface_needs_at_least_two_points(self):
        with pytest.raises(ValueError, match="at least two points"):
            distance_to_surface_m((0.0, 0.0), [(1.0, 1.0)])


class TestVerification:
    def test_a_compliant_plan_passes_and_says_how_close_it_came(self):
        points = [(10.0, 8.0), (50.0, 8.0), (90.0, 8.0)]
        report = verify_clearance(points, {"north": WALL}, minimum_clearance_m=6.0)

        assert report["ok"] is True
        assert report["closest_approach_m"] == pytest.approx(8.0)
        assert report["violation_count"] == 0

    def test_every_offending_point_is_named_not_just_the_first(self):
        """A single pass or fail leaves the operator hunting for which waypoints."""
        points = [(10.0, 8.0), (50.0, 3.0), (90.0, 2.5)]
        report = verify_clearance(points, {"north": WALL}, minimum_clearance_m=6.0)

        assert report["ok"] is False
        assert report["violation_count"] == 2
        assert {v["point_index"] for v in report["violations"]} == {1, 2}

    def test_the_worst_shortfall_is_quantified(self):
        points = [(50.0, 3.0)]
        report = verify_clearance(points, {"north": WALL}, minimum_clearance_m=6.0)
        assert report["worst_shortfall_m"] == pytest.approx(3.0)

    def test_a_point_too_close_to_any_surface_fails_even_if_far_from_others(self):
        surfaces = {
            "north": WALL,
            "east": [(100.0, 0.0), (100.0, 100.0)],
        }
        # Comfortably clear of the north wall, but 1 m from the east one.
        report = verify_clearance([(99.0, 50.0)], surfaces, minimum_clearance_m=6.0)
        assert report["ok"] is False
        assert report["violations"][0]["surface_id"] == "east"

    def test_the_summary_tells_the_operator_what_to_do(self):
        report = verify_clearance([(50.0, 1.0)], {"north": WALL}, minimum_clearance_m=6.0)
        assert "before flying" in report["summary"]

    def test_an_empty_plan_is_vacuously_clear_but_reports_no_approach(self):
        report = verify_clearance([], {"north": WALL})
        assert report["ok"] is True
        assert report["closest_approach_m"] is None

    def test_verification_enforces_the_floor_too(self):
        report = verify_clearance([(50.0, 1.5)], {"north": WALL}, minimum_clearance_m=0.1)
        assert report["minimum_clearance_m"] >= ABSOLUTE_MINIMUM_CLEARANCE_M
        assert report["ok"] is False


class TestAgainstARealPlan:
    """Measuring compiled missions, which is where the intention is actually tested.

    These caught two real defects. ``facade_inspection`` was missing from the template
    alias table and fell through to the "grid" default, producing a nadir lawnmower
    sweep across the top of the building instead of a stand-off pass along its wall.
    And because the drawn polygon became the geofence, every capture point placed at
    stand-off fell outside the fence and was projected back onto the structure -- 465
    projections on a 40-waypoint mission that still reported an 8 m stand-off it had
    not achieved.
    """

    FOOTPRINT = [[-81.7510, 41.3040], [-81.7505, 41.3040],
                 [-81.7505, 41.3044], [-81.7510, 41.3044]]

    def _footprint_ring_m(self):
        import math

        lat0 = 41.3042
        scale_x = 111_320.0 * math.cos(math.radians(lat0))
        scale_y = 110_540.0
        ring = [(lon * scale_x, lat * scale_y) for lon, lat in self.FOOTPRINT]
        return ring + [ring[0]]

    def _to_m(self, waypoint):
        import math

        lat0 = 41.3042
        return (waypoint[0] * 111_320.0 * math.cos(math.radians(lat0)),
                waypoint[1] * 110_540.0)

    @pytest.mark.parametrize("mode", ["facade_inspection", "facade_mapping",
                                      "multi_facade", "closed_loop"])
    def test_a_structure_mission_keeps_its_stand_off_from_the_structure(self, mode):
        from mission.planner import MissionPlanner

        plan = MissionPlanner().generate(
            mode=mode, polygon_lonlat=self.FOOTPRINT, altitude_m=40.0)
        assert len(plan.waypoints) > 0

        ring = self._footprint_ring_m()
        distances = [distance_to_surface_m(self._to_m(w), ring) for w in plan.waypoints]

        declared = float(plan.to_dict().get("facade_standoff_m") or 8.0)
        # Within a metre of the declared stand-off, allowing for the local projection.
        assert min(distances) > declared - 1.0, (
            f"{mode} declares {declared} m stand-off but flies within "
            f"{min(distances):.2f} m of the structure"
        )

    def test_facade_inspection_is_not_silently_a_nadir_grid(self):
        """It was, because the alias table had no entry for it."""
        from mission.planner import MissionPlanner

        plan = MissionPlanner().generate(
            mode="facade_inspection", polygon_lonlat=self.FOOTPRINT, altitude_m=40.0)
        assert plan.to_dict()["flight_recipe"]["template"] == "facade"

    def test_every_named_mission_type_compiles_to_its_own_template(self):
        """A mode that quietly resolves to grid delivers a survey nobody ordered."""
        from mission.planner import MissionPlanner

        expected = {
            "double_grid": "double_grid",
            "roof_inspection": "roof_inspection",
            "facade_mapping": "facade_mapping",
            "facade_inspection": "facade",
            "multi_facade": "multi_facade",
            "closed_loop": "closed_loop",
            "linear_inspection": "linear_inspection",
            "linear_mapping": "corridor",
            "tower_mapping": "tower_mapping",
            "orbit": "orbit",
            "panorama": "panorama",
            "solar_inspection": "solar_inspection",
            "corridor": "corridor",
            "wind_turbine": "wind_turbine",
            "waypoints": "waypoints",
        }

        for mode, template in expected.items():
            plan = MissionPlanner().generate(
                mode=mode, polygon_lonlat=self.FOOTPRINT, altitude_m=40.0)
            actual = plan.to_dict()["flight_recipe"]["template"]
            assert actual == template, (
                f"{mode!r} compiled to {actual!r}, not {template!r}"
            )

    def test_the_geofence_no_longer_projects_capture_points_onto_the_wall(self):
        from mission.planner import MissionPlanner

        plan = MissionPlanner().generate(
            mode="facade_inspection", polygon_lonlat=self.FOOTPRINT, altitude_m=40.0)
        adjustments = plan.to_dict()["safety_adjustments"]
        assert adjustments["geofence_projections"] == 0, (
            "capture points are still being pulled back inside a geofence drawn around "
            "the structure being inspected"
        )

    def test_a_compiled_facade_mission_passes_its_own_clearance_check(self):
        """End to end: the plan the compiler produced, measured by the verifier."""
        from mission.planner import MissionPlanner

        plan = MissionPlanner().generate(
            mode="facade_inspection", polygon_lonlat=self.FOOTPRINT, altitude_m=40.0)

        points = [self._to_m(w) for w in plan.waypoints]
        report = verify_clearance(points, {"building": self._footprint_ring_m()},
                                  minimum_clearance_m=5.0)
        assert report["ok"] is True, report["summary"]
