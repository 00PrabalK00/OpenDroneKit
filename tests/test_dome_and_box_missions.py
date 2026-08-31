"""Two mission types whose geometry was wrong or absent.

`dome` aliased to `tower_mapping`. A tower is a cylinder, so every ring flies at the same
radius. Point that at a silo cap or a stadium roof and the upper rings stand off further
and further from a surface curving away underneath them: the stand-off the operator asked
for is only correct at the widest ring, ground sample drifts all the way up, and the crown
is photographed from the side rather than looked at. The plan was generated, flown and
delivered without anything reporting a problem.

`box` did not exist. A crosshatch over a rectangular building photographs the roof and
misses the vertical edges and the sides of rooftop plant.

These tests are about the geometry rather than the wiring: whether the path is the right
shape for the thing being inspected.
"""

from __future__ import annotations

import math

import pytest

from mission.planner import MissionPlanner

DOME = {
    "center_local": [0.0, 0.0],
    "object_radius_m": 20.0,
    "standoff_m": 10.0,
    "rings": 5,
    "points_per_orbit": 12,
}
FOOTPRINT = [[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]]
BOX = {
    "polygon_local": FOOTPRINT,
    "standoff_m": 15.0,
    "altitude_levels_m": [30.0, 45.0],
    "capture_spacing_m": 10.0,
}


@pytest.fixture
def planner() -> MissionPlanner:
    return MissionPlanner()


class TestTheDomeFollowsItsCurve:
    def test_the_standoff_is_the_same_on_every_ring(self, planner) -> None:
        """The defect the tower alias caused, stated as a number.

        Distance from the dome surface must be the requested stand-off at the crown as
        well as at the equator. Under tower_mapping the top ring sat 20 m from a surface
        the operator asked to be 10 m from.
        """
        poses = planner._compile_dome_inspection_primitive(DOME)
        radius = DOME["object_radius_m"]
        by_ring: dict[str, list] = {}
        for pose in poses:
            by_ring.setdefault(pose.primitive, []).append(pose)

        for name, ring in by_ring.items():
            pose = ring[0]
            # Altitude is floored at 1 m for safety, which perturbs the equator ring only.
            if pose.alt_m <= 1.0:
                continue
            distance = math.hypot(math.hypot(pose.x_m, pose.y_m), pose.alt_m) - radius
            assert distance == pytest.approx(DOME["standoff_m"], abs=0.05), name

    def test_the_gimbal_follows_the_surface_normal(self, planner) -> None:
        """Horizontal at the equator, straight down at the crown."""
        poses = planner._compile_dome_inspection_primitive(DOME)
        pitches = {p.primitive: p.gimbal_pitch_deg for p in poses}
        assert pitches["dome_inspection_ring1"] == pytest.approx(0.0, abs=0.5)
        assert pitches["dome_inspection_ring5"] == pytest.approx(-90.0, abs=0.5)

        ordered = [pitches[f"dome_inspection_ring{i}"] for i in range(1, 6)]
        assert ordered == sorted(ordered, reverse=True), "pitch must fall monotonically"

    def test_the_rings_climb(self, planner) -> None:
        poses = planner._compile_dome_inspection_primitive(DOME)
        heights = []
        for i in range(1, 6):
            ring = [p for p in poses if p.primitive == f"dome_inspection_ring{i}"]
            heights.append(ring[0].alt_m)
        assert heights == sorted(heights)

    def test_the_crown_is_one_shot_not_a_ring(self, planner) -> None:
        """At the top the flight radius collapses to zero. Flying a ring there repeats
        the same photograph a dozen times from the same place."""
        poses = planner._compile_dome_inspection_primitive(DOME)
        crown = [p for p in poses if p.primitive == "dome_inspection_ring5"]
        assert len(crown) == 1
        assert crown[0].x_m == pytest.approx(0.0, abs=1e-6)
        assert crown[0].y_m == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("requested", ["dome", "dome_inspection", "silo"])
    def test_a_dome_no_longer_resolves_to_a_tower(self, requested) -> None:
        """The alias WAS the bug: all the geometry above is unreachable without this.

        Asserted through the resolver rather than the source text, so it tests what a
        caller actually gets.
        """
        from mission.planner import _normalize_template

        assert _normalize_template(requested) == "dome_inspection"

    @pytest.mark.parametrize("requested", ["box", "box_inspection"])
    def test_a_box_resolves_to_its_own_primitive(self, requested) -> None:
        from mission.planner import _normalize_template

        assert _normalize_template(requested) == "box_inspection"

    def test_a_tower_is_still_a_tower(self) -> None:
        """Separating the dome must not move the cylinder it was borrowing."""
        from mission.planner import _normalize_template

        for requested in ("tower", "tower_mapping", "cell_tower", "stack"):
            assert _normalize_template(requested) == "tower_mapping"


class TestTheBoxCircuitLooksInward:
    def test_every_pose_points_at_the_building_centre(self, planner) -> None:
        """The whole reason the mission type exists: obliques of the vertical edges and
        rooftop plant, which a nadir crosshatch never sees."""
        poses = planner._compile_box_inspection_primitive(BOX)
        assert poses
        for pose in poses:
            wanted = math.degrees(math.atan2(-pose.y_m, -pose.x_m))
            error = abs((wanted - pose.yaw_deg + 180.0) % 360.0 - 180.0)
            assert error == pytest.approx(0.0, abs=0.01)

    def test_it_follows_the_footprint_rather_than_a_circle(self, planner) -> None:
        """A circle around a 40x20 building holds the corners at stand-off and leaves the
        middle of each long wall much further away. The offset footprint does not."""
        poses = [p for p in planner._compile_box_inspection_primitive(BOX)
                 if p.primitive.endswith("level1")]
        spread = {round(math.hypot(p.x_m, p.y_m), 1) for p in poses}
        assert len(spread) > 1, "a single radius means this is still a circle"

    def test_each_altitude_level_is_flown(self, planner) -> None:
        poses = planner._compile_box_inspection_primitive(BOX)
        levels = {p.primitive for p in poses}
        assert levels == {"box_inspection_level1", "box_inspection_level2"}
        for level, altitude in (("level1", 30.0), ("level2", 45.0)):
            got = {p.alt_m for p in poses if p.primitive.endswith(level)}
            assert got == {altitude}

    def test_the_camera_is_oblique_not_nadir(self, planner) -> None:
        """Straight down would photograph the roof, which is the thing a crosshatch
        already does well."""
        poses = planner._compile_box_inspection_primitive(BOX)
        for pose in poses:
            assert -80.0 < pose.gimbal_pitch_deg < 0.0

    def test_an_empty_footprint_produces_nothing_rather_than_guessing(self, planner) -> None:
        assert planner._compile_box_inspection_primitive({"polygon_local": []}) == []
