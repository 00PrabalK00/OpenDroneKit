"""Multi-facade, closed-loop and wind-turbine capture patterns.

These assert the geometry rather than the waypoint count. A pattern that produces the
right number of points in the wrong places would still fly, still look plausible on a
map, and still fail to reconstruct the asset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mission.planner import MissionPlanner

AOI = [
    [-81.7510, 41.3035],
    [-81.7490, 41.3035],
    [-81.7490, 41.3050],
    [-81.7510, 41.3050],
]

# A 40 m x 25 m rectangular footprint in local metres.
RECTANGLE = [[0.0, 0.0], [40.0, 0.0], [40.0, 25.0], [0.0, 25.0]]


def planner() -> MissionPlanner:
    return MissionPlanner()


class TestMultiFacade:
    def test_every_face_gets_its_own_pass(self):
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 8.0,
            "top_altitude_m": 20.0, "bottom_altitude_m": 5.0,
        })
        faces = {pose.primitive for pose in poses}
        assert len(faces) == 4, f"expected one pass per edge, got {faces}"

    def test_capture_points_stand_off_outside_the_building(self):
        """Inside the footprint the aircraft would be flying through the building."""
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 8.0,
            "top_altitude_m": 20.0, "bottom_altitude_m": 5.0,
        })
        for pose in poses:
            inside_x = 0.0 < pose.x_m < 40.0
            inside_y = 0.0 < pose.y_m < 25.0
            assert not (inside_x and inside_y), \
                f"a capture point fell inside the footprint at ({pose.x_m}, {pose.y_m})"

    def test_stand_off_distance_is_respected(self):
        standoff = 8.0
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": standoff,
            "top_altitude_m": 20.0, "bottom_altitude_m": 5.0,
        })
        # Distance from the point to the rectangle itself, not to an edge's infinite
        # line: a corner point lies on the perpendicular wall's line at distance zero
        # while still being a correct stand-off from the wall it is inspecting.
        for pose in poses:
            dx = max(0.0, 0.0 - pose.x_m, pose.x_m - 40.0)
            dy = max(0.0, 0.0 - pose.y_m, pose.y_m - 25.0)
            assert math.hypot(dx, dy) == pytest.approx(standoff, abs=0.01)

    def test_altitudes_span_the_requested_band(self):
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 6.0,
            "top_altitude_m": 22.0, "bottom_altitude_m": 4.0, "vertical_spacing_m": 3.0,
        })
        altitudes = [pose.alt_m for pose in poses]
        assert min(altitudes) == pytest.approx(4.0)
        assert max(altitudes) == pytest.approx(22.0)

    def test_per_face_overrides_are_applied(self):
        """One face taller than the others is the case this exists for."""
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 8.0,
            "top_altitude_m": 20.0, "bottom_altitude_m": 5.0,
            "face_overrides": {"0": {"top_altitude_m": 45.0, "standoff_m": 15.0}},
        })
        face_zero = [p for p in poses if p.primitive == "facade_face_0"]
        others = [p for p in poses if p.primitive != "facade_face_0"]
        assert max(p.alt_m for p in face_zero) == pytest.approx(45.0)
        assert max(p.alt_m for p in others) == pytest.approx(20.0)
        # The overridden stand-off applies too.
        assert min(abs(p.y_m) for p in face_zero) == pytest.approx(15.0, abs=0.01)

    def test_the_camera_faces_the_wall(self):
        """A facade pass whose camera points outward photographs the horizon."""
        poses = planner()._compile_multi_facade_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 8.0,
            "top_altitude_m": 12.0, "bottom_altitude_m": 6.0,
        })
        # The south wall (y = 0) is approached from y < 0, so the camera must look
        # north, which is yaw 0.
        south = [p for p in poses if p.y_m < 0]
        assert south
        for pose in south:
            assert pose.yaw_deg == pytest.approx(0.0, abs=1.0)
            assert pose.camera_yaw_locked

    def test_a_degenerate_polygon_yields_nothing_rather_than_garbage(self):
        assert planner()._compile_multi_facade_primitive({"polygon_local": [[0, 0], [1, 1]]}) == []


class TestClosedLoop:
    def test_points_lie_on_a_circle_of_the_requested_radius(self):
        poses = planner()._compile_closed_loop_primitive({
            "center_local": [0.0, 0.0], "object_radius_m": 5.0,
            "radii_m": [30.0], "altitude_levels_m": [40.0], "points_per_loop": 24,
        })
        assert len(poses) == 24
        for pose in poses:
            radius = math.hypot(pose.x_m, pose.y_m)
            assert radius == pytest.approx(30.0, abs=0.01)

    def test_the_camera_always_faces_the_structure(self):
        """Otherwise the loop is a perimeter flight, not a reconstruction capture."""
        poses = planner()._compile_closed_loop_primitive({
            "center_local": [0.0, 0.0], "radii_m": [25.0],
            "altitude_levels_m": [30.0], "points_per_loop": 16,
        })
        for pose in poses:
            # Bearing from the aircraft to the centre.
            expected = math.degrees(math.atan2(-pose.x_m, -pose.y_m))
            # Compare circularly: 0 and 360 are the same bearing.
            difference = abs((pose.yaw_deg - expected + 180.0) % 360.0 - 180.0)
            assert difference < 1.0, f"yaw {pose.yaw_deg} does not face the centre"

    def test_multiple_rings_and_altitudes_multiply(self):
        poses = planner()._compile_closed_loop_primitive({
            "center_local": [0.0, 0.0], "radii_m": [20.0, 35.0],
            "altitude_levels_m": [25.0, 45.0], "points_per_loop": 12,
        })
        assert len(poses) == 2 * 2 * 12
        assert {round(math.hypot(p.x_m, p.y_m)) for p in poses} == {20, 35}
        assert {p.alt_m for p in poses} == {25.0, 45.0}

    def test_direction_is_honoured(self):
        """Clockwise and counter-clockwise must actually differ."""
        common = {"center_local": [0.0, 0.0], "radii_m": [20.0],
                  "altitude_levels_m": [30.0], "points_per_loop": 8}
        clockwise = planner()._compile_closed_loop_primitive({**common, "clockwise": True})
        counter = planner()._compile_closed_loop_primitive({**common, "clockwise": False})
        assert [round(p.y_m, 3) for p in clockwise] != [round(p.y_m, 3) for p in counter]

    def test_a_polygon_sizes_the_loop_around_the_structure(self):
        poses = planner()._compile_closed_loop_primitive({
            "polygon_local": RECTANGLE, "standoff_m": 10.0,
            "altitude_levels_m": [30.0], "points_per_loop": 16,
        })
        centre = np.asarray(RECTANGLE).mean(axis=0)
        extent = float(np.linalg.norm(np.asarray(RECTANGLE) - centre, axis=1).max())
        for pose in poses:
            radius = math.hypot(pose.x_m - centre[0], pose.y_m - centre[1])
            assert radius == pytest.approx(extent + 10.0, abs=0.01)

    def test_no_input_yields_nothing(self):
        assert planner()._compile_closed_loop_primitive({}) == []


class TestWindTurbine:
    PARAMS = {
        "center_local": [0.0, 0.0], "hub_height_m": 90.0, "blade_length_m": 55.0,
        "standoff_m": 12.0, "tower_levels": 6, "blade_stations": 8,
    }

    def test_tower_nacelle_and_blades_are_all_captured(self):
        poses = planner()._compile_wind_turbine_primitive(self.PARAMS)
        parts = {pose.primitive.split("_")[1] for pose in poses}
        assert {"tower", "nacelle", "blade"} <= parts

    def test_the_tower_is_climbed_to_hub_height(self):
        poses = planner()._compile_wind_turbine_primitive(self.PARAMS)
        tower = [p for p in poses if p.primitive == "turbine_tower"]
        assert len(tower) == 6
        assert max(p.alt_m for p in tower) == pytest.approx(90.0)

    def test_the_nacelle_is_captured_from_four_aspects(self):
        poses = planner()._compile_wind_turbine_primitive(self.PARAMS)
        nacelle = [p for p in poses if p.primitive == "turbine_nacelle"]
        assert len(nacelle) == 4
        assert all(p.alt_m == pytest.approx(90.0) for p in nacelle)
        # Four distinct bearings around the hub.
        assert len({round(p.yaw_deg) % 360 for p in nacelle}) == 4

    def test_each_blade_is_captured_from_both_faces(self):
        """A crack on the trailing edge is invisible from the leading side."""
        poses = planner()._compile_wind_turbine_primitive(self.PARAMS)
        blades = [p for p in poses if p.primitive.startswith("turbine_blade")]
        suffixes = {p.primitive[-1] for p in blades}
        assert suffixes == {"a", "b"}
        for pose in blades:
            assert abs(pose.y_m) == pytest.approx(12.0, abs=0.01)

    def test_blade_stations_reach_toward_the_tip(self):
        poses = planner()._compile_wind_turbine_primitive(
            {**self.PARAMS, "blade_angles_deg": [90.0]})
        upward = [p for p in poses if p.primitive.startswith("turbine_blade")]
        # A blade pointing straight up reaches hub height plus blade length.
        assert max(p.alt_m for p in upward) == pytest.approx(145.0, abs=0.1)

    def test_points_below_the_minimum_altitude_are_skipped(self):
        """A blade tip near the ground cannot be inspected from the air."""
        poses = planner()._compile_wind_turbine_primitive({
            **self.PARAMS, "blade_angles_deg": [270.0], "min_altitude_m": 20.0,
        })
        blades = [p for p in poses if p.primitive.startswith("turbine_blade")]
        assert blades, "the whole blade was discarded"
        assert min(p.alt_m for p in blades) >= 20.0

    def test_parked_rotor_angles_are_honoured(self):
        one_blade = planner()._compile_wind_turbine_primitive(
            {**self.PARAMS, "blade_angles_deg": [90.0]})
        three_blades = planner()._compile_wind_turbine_primitive(
            {**self.PARAMS, "blade_angles_deg": [90.0, 210.0, 330.0]})
        assert len(three_blades) > len(one_blade)


class TestTemplateIntegration:
    @pytest.mark.parametrize("template", ["multi_facade", "closed_loop", "wind_turbine"])
    def test_each_template_plans_over_a_drawn_area(self, template):
        plan = MissionPlanner().generate(mode=template, polygon_lonlat=AOI, altitude_m=60.0)
        assert len(plan.waypoints) > 0
        assert plan.path_distance_m > 0

    @pytest.mark.parametrize(
        "alias,expected",
        [("turbine", "wind_turbine"), ("closed_loop_facade", "closed_loop"),
         ("multi_facade_inspection", "multi_facade")],
    )
    def test_aliases_resolve(self, alias, expected):
        from mission.planner import _normalize_template

        assert _normalize_template(alias) == expected

    def test_waypoints_carry_valid_coordinates(self):
        for template in ("multi_facade", "closed_loop", "wind_turbine"):
            plan = MissionPlanner().generate(mode=template, polygon_lonlat=AOI, altitude_m=60.0)
            for lon, lat, alt in ((w[0], w[1], w[2]) for w in plan.waypoints):
                assert -180.0 <= lon <= 180.0
                assert -90.0 <= lat <= 90.0
                assert math.isfinite(alt)
