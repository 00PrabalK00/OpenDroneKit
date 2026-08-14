"""Mission planning constraints.

These are the safety claims: a waypoint must not sit inside a no-fly polygon, must
not leave the geofence, and must respect the altitude band. A planner that quietly
ignores a constraint is worse than one that refuses the mission.
"""

from __future__ import annotations

import math

import pytest

from mission.planner import MissionPlanner

AOI = [
    [-81.7510, 41.3035],
    [-81.7490, 41.3035],
    [-81.7490, 41.3050],
    [-81.7510, 41.3050],
]


def _point_in_ring(lon: float, lat: float, ring) -> bool:
    """Even-odd crossing test, independent of the planner's own implementation."""
    vertices = [v for v in ring if len(v) >= 2]
    if vertices and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    inside = False
    count = len(vertices)
    for index in range(count):
        x1, y1 = float(vertices[index][0]), float(vertices[index][1])
        x2, y2 = float(vertices[(index + 1) % count][0]), float(vertices[(index + 1) % count][1])
        if (y1 > lat) != (y2 > lat):
            crossing = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < crossing:
                inside = not inside
    return inside


class TestPlanGeometry:
    def test_a_plan_over_a_drawn_polygon_produces_waypoints(self):
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        assert len(plan.waypoints) > 0
        assert plan.path_distance_m > 0.0

    def test_waypoints_carry_three_ordinates(self):
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        for waypoint in plan.waypoints:
            assert len(waypoint) >= 3
            lon, lat, alt = waypoint[0], waypoint[1], waypoint[2]
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0
            assert math.isfinite(alt)

    def test_waypoints_land_near_the_requested_area(self):
        """A plan must cover the polygon it was given, not some default elsewhere."""
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        lons = [w[0] for w in plan.waypoints]
        lats = [w[1] for w in plan.waypoints]
        assert min(lons) > -81.76 and max(lons) < -81.74
        assert min(lats) > 41.30 and max(lats) < 41.31


class TestNoFlyZones:
    def test_no_waypoint_sits_inside_an_exclusion_polygon(self):
        no_fly = [
            [-81.7504, 41.3040],
            [-81.7496, 41.3040],
            [-81.7496, 41.3046],
            [-81.7504, 41.3046],
        ]
        plan = MissionPlanner().generate(
            mode="grid",
            polygon_lonlat=AOI,
            altitude_m=60.0,
            constraints={"geofence": AOI, "no_fly_polygons": [no_fly]},
        )
        offenders = [
            (w[0], w[1]) for w in plan.waypoints if _point_in_ring(w[0], w[1], no_fly)
        ]
        assert not offenders, f"{len(offenders)} waypoints fell inside the no-fly zone"

    def test_a_no_fly_zone_changes_the_route(self):
        base = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        no_fly = [
            [-81.7504, 41.3040], [-81.7496, 41.3040],
            [-81.7496, 41.3046], [-81.7504, 41.3046],
        ]
        constrained = MissionPlanner().generate(
            mode="grid",
            polygon_lonlat=AOI,
            altitude_m=60.0,
            constraints={"geofence": AOI, "no_fly_polygons": [no_fly]},
        )
        assert constrained.waypoints != base.waypoints


class TestAltitudeBand:
    def test_altitude_is_clamped_into_the_configured_band(self):
        plan = MissionPlanner().generate(
            mode="grid",
            polygon_lonlat=AOI,
            altitude_m=300.0,
            constraints={"geofence": AOI, "min_altitude_m": 20.0, "max_altitude_m": 120.0},
        )
        altitudes = [w[2] for w in plan.waypoints]
        assert max(altitudes) <= 120.0 + 1e-6, "a waypoint exceeded the altitude ceiling"

    def test_low_request_is_raised_to_the_floor(self):
        plan = MissionPlanner().generate(
            mode="grid",
            polygon_lonlat=AOI,
            altitude_m=2.0,
            constraints={"geofence": AOI, "min_altitude_m": 25.0, "max_altitude_m": 120.0},
        )
        altitudes = [w[2] for w in plan.waypoints]
        assert min(altitudes) >= 25.0 - 1e-6, "a waypoint fell below the altitude floor"


class TestTemplates:
    @pytest.mark.parametrize(
        "template",
        [
            "grid", "double_grid", "corridor", "facade", "tower_mapping",
            "solar_inspection", "orbit", "panorama", "bubble_360", "waypoints",
            "roof_inspection", "linear_inspection", "lateral_capture",
            "magnetic_mapping", "smart_adaptive",
        ],
    )
    def test_every_template_produces_a_plan(self, template):
        plan = MissionPlanner().generate(
            mode=template, polygon_lonlat=AOI, altitude_m=60.0
        )
        assert len(plan.waypoints) > 0, f"{template} produced no waypoints"


class TestSmartAdaptive:
    def test_without_interest_input_it_matches_a_plain_grid(self):
        """It must not imply adaptivity it did not perform."""
        grid = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        adaptive = MissionPlanner().generate(
            mode="smart_adaptive", polygon_lonlat=AOI, altitude_m=60.0
        )
        assert len(adaptive.waypoints) == len(grid.waypoints)

    def test_prior_defects_densify_the_plan(self):
        plain = MissionPlanner().generate(
            mode="smart_adaptive", polygon_lonlat=AOI, altitude_m=60.0
        )
        adaptive = MissionPlanner().generate(
            mode="smart_adaptive",
            polygon_lonlat=AOI,
            altitude_m=60.0,
            adaptive_prior_defects_lonlat=[
                [-81.7500, 41.3042], [-81.7499, 41.3043], [-81.7495, 41.3038],
            ],
        )
        assert len(adaptive.waypoints) > len(plain.waypoints)

    def test_detail_passes_fly_lower_than_the_base_pass(self):
        adaptive = MissionPlanner().generate(
            mode="smart_adaptive",
            polygon_lonlat=AOI,
            altitude_m=60.0,
            adaptive_prior_defects_lonlat=[[-81.7500, 41.3042]],
        )
        altitudes = {round(w[2], 1) for w in adaptive.waypoints}
        assert len(altitudes) > 1, "a closer look must actually fly closer"
        assert min(altitudes) < 60.0


class TestPhotogrammetry:
    def test_higher_flight_gives_coarser_ground_sample(self):
        low = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=30.0)
        high = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=90.0)
        assert high.estimated_gsd_cm > low.estimated_gsd_cm

    def test_more_overlap_costs_more_waypoints(self):
        sparse = MissionPlanner().generate(
            mode="grid", polygon_lonlat=AOI, altitude_m=60.0,
            front_overlap_pct=60.0, side_overlap_pct=50.0,
        )
        dense = MissionPlanner().generate(
            mode="grid", polygon_lonlat=AOI, altitude_m=60.0,
            front_overlap_pct=90.0, side_overlap_pct=85.0,
        )
        assert len(dense.waypoints) > len(sparse.waypoints)

    def test_duration_estimate_is_positive_and_finite(self):
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=AOI, altitude_m=60.0)
        assert plan.estimated_time_min > 0.0
        assert math.isfinite(plan.estimated_time_min)
