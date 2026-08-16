"""Facade passes around irregular footprints.

The failure this guards against is silent. A naive offset path around a concave footprint
folds back through the building at each reflex corner: the mission uploads cleanly, the
aircraft flies it, and the operator discovers the problem at the wall.

So the central test is not that an L-shaped building produces a plan. It is that no
planned segment lies inside the building.
"""

from __future__ import annotations

import pytest

from mission.footprints import (
    FootprintRefused,
    analyse_footprint,
    facade_segments,
    point_in_polygon,
    signed_area,
)

# 20 x 20 m square.
SQUARE = [[0, 0], [20, 0], [20, 20], [0, 20]]

# L-shape: a 20x20 square with a 10x10 bite out of the north-east corner. The vertex at
# (10, 10) is reflex.
L_SHAPE = [[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]]


class TestAnalysis:
    def test_a_square_is_convex(self) -> None:
        result = analyse_footprint(SQUARE)
        assert result.is_convex
        assert result.reflex_count == 0
        assert result.area_m2 == pytest.approx(400.0)

    def test_an_l_shape_has_exactly_one_reflex_corner(self) -> None:
        result = analyse_footprint(L_SHAPE)
        assert not result.is_convex
        assert result.reflex_count == 1
        assert result.area_m2 == pytest.approx(300.0)

    def test_the_reflex_corner_is_named_not_smoothed(self) -> None:
        # The caller needs to know before planning, not after.
        payload = analyse_footprint(L_SHAPE).to_dict()
        assert len(payload["reflex_corners"]) == 1
        assert payload["reflex_corners"][0]["xy"] == [10.0, 10.0]
        assert payload["reflex_corners"][0]["interior_angle_deg"] > 180.0

    def test_winding_does_not_change_the_answer(self) -> None:
        # A footprint digitised clockwise must analyse the same as counter-clockwise.
        assert analyse_footprint(list(reversed(L_SHAPE))).reflex_count == 1

    def test_the_note_explains_the_consequence(self) -> None:
        assert "THROUGH the building" in analyse_footprint(L_SHAPE).note


class TestRefusals:
    def test_too_few_corners_is_refused(self) -> None:
        with pytest.raises(FootprintRefused, match="at least 3"):
            analyse_footprint([[0, 0], [1, 1]])

    def test_a_degenerate_footprint_is_refused(self) -> None:
        with pytest.raises(FootprintRefused, match="too small"):
            analyse_footprint([[0, 0], [0.5, 0], [0.5, 0.5]])

    def test_degrees_mistaken_for_metres_is_caught(self) -> None:
        # A footprint in degrees encloses a tiny "area" in those units. Better to refuse
        # than to plan a mission a few centimetres across.
        with pytest.raises(FootprintRefused, match="metres, not degrees"):
            analyse_footprint([[77.5, 12.9], [77.5001, 12.9], [77.5001, 12.9001]])

    def test_a_negative_standoff_is_refused(self) -> None:
        with pytest.raises(FootprintRefused, match="positive"):
            facade_segments(SQUARE, standoff_m=-5)

    def test_a_footprint_with_no_usable_wall_is_refused(self) -> None:
        # Every wall shorter than the minimum: there is no pass to fly, so the planner
        # says so rather than returning an empty list a caller might treat as success.
        with pytest.raises(FootprintRefused, match="No facade pass survives"):
            facade_segments(SQUARE, standoff_m=5.0, min_wall_m=100.0)


class TestSegments:
    def test_a_square_gives_one_pass_per_wall(self) -> None:
        segments = facade_segments(SQUARE, standoff_m=5.0)
        assert len(segments) == 4

    def test_no_segment_lies_inside_the_building(self) -> None:
        """The whole point of the module.

        On an L-shaped footprint a naive offset ring puts track through the structure at
        the reflex corner. Every planned point must be outside.
        """
        polygon = [(float(x), float(y)) for x, y in L_SHAPE]
        for segment in facade_segments(L_SHAPE, standoff_m=4.0):
            for point in (segment.start, segment.end):
                assert not point_in_polygon(point, polygon), (
                    f"wall {segment.wall_index} plans a pass at {point}, inside the building"
                )

    def test_segments_sit_outside_at_the_standoff(self) -> None:
        segments = facade_segments(SQUARE, standoff_m=5.0)
        polygon = [(float(x), float(y)) for x, y in SQUARE]
        for segment in segments:
            assert not point_in_polygon(segment.start, polygon)
            assert not point_in_polygon(segment.end, polygon)

    def test_short_walls_are_skipped(self) -> None:
        # A 0.5 m jog is not a facade; the neighbouring walls cover it.
        notched = [[0, 0], [20, 0], [20, 10], [19.5, 10], [19.5, 20], [0, 20]]
        for segment in facade_segments(notched, standoff_m=3.0):
            assert segment.length_m >= 1.0

    def test_clockwise_input_still_plans_outside(self) -> None:
        polygon = [(float(x), float(y)) for x, y in L_SHAPE]
        for segment in facade_segments(list(reversed(L_SHAPE)), standoff_m=4.0):
            assert not point_in_polygon(segment.start, polygon)
            assert not point_in_polygon(segment.end, polygon)


class TestGeometryHelpers:
    def test_signed_area_sign_follows_winding(self) -> None:
        assert signed_area([(0, 0), (10, 0), (10, 10), (0, 10)]) > 0
        assert signed_area([(0, 10), (10, 10), (10, 0), (0, 0)]) < 0

    def test_point_in_polygon_on_an_l_shape(self) -> None:
        polygon = [(float(x), float(y)) for x, y in L_SHAPE]
        assert point_in_polygon((5.0, 5.0), polygon)
        # Inside the bite taken out of the corner: outside the building.
        assert not point_in_polygon((15.0, 15.0), polygon)
