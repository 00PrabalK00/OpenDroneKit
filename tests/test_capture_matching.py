"""Matching captured images back to planned capture points.

The case worth catching is a missed capture point, because it is cheap to fix while
the pilot is on site and expensive once the aircraft is packed away. So the tests
check that a gap is reported as a gap, that one photograph never satisfies two
capture points, and that an image beyond the match radius is not claimed as a match.
"""

from __future__ import annotations

import pytest

from core.capture_matching import (
    CapturedImage,
    PlannedCapture,
    match_captures,
    planned_captures_from_plan,
    pose_priors_for_reconstruction,
)

# Roughly 11 m apart in latitude at this latitude, comfortably inside a 15 m radius.
BASE_LON, BASE_LAT = -81.7505, 41.3042


def planned(count: int, spacing_deg: float = 0.0005) -> list[PlannedCapture]:
    return [
        PlannedCapture(index=i, longitude=BASE_LON + i * spacing_deg,
                       latitude=BASE_LAT, altitude_m=60.0)
        for i in range(count)
    ]


def image(index: int, lon: float, lat: float, alt: float | None = 60.0) -> CapturedImage:
    return CapturedImage(path=f"DSC{index:05d}.JPG", longitude=lon, latitude=lat,
                         altitude_m=alt)


class TestMatching:
    def test_a_perfectly_flown_plan_matches_every_point(self):
        points = planned(5)
        images = [image(i, p.longitude, p.latitude) for i, p in enumerate(points)]

        report = match_captures(points, images)
        assert len(report.matches) == 5
        assert report.missed == []
        assert report.unplanned == []
        assert report.coverage_pct == pytest.approx(100.0)

    def test_a_skipped_capture_point_is_reported_by_index(self):
        """This is the failure that ruins a reconstruction days later."""
        points = planned(5)
        images = [image(i, p.longitude, p.latitude)
                  for i, p in enumerate(points) if i != 2]

        report = match_captures(points, images)
        assert len(report.matches) == 4
        assert [capture.index for capture in report.missed] == [2]

        payload = report.to_dict()
        assert payload["coverage_pct"] == pytest.approx(80.0)
        assert any("re-fly" in note.lower() for note in payload["recommendations"])

    def test_an_extra_image_is_listed_not_folded_into_the_plan(self):
        """A manual capture is not an error, but it is not a planned point either."""
        points = planned(3)
        images = [image(i, p.longitude, p.latitude) for i, p in enumerate(points)]
        images.append(image(99, BASE_LON + 0.01, BASE_LAT + 0.01))

        report = match_captures(points, images)
        assert len(report.matches) == 3
        assert len(report.unplanned) == 1
        assert report.unplanned[0].path.endswith("DSC00099.JPG")

    def test_one_image_never_satisfies_two_capture_points(self):
        """Otherwise a gap would be hidden behind a double-counted photograph."""
        points = [
            PlannedCapture(index=0, longitude=BASE_LON, latitude=BASE_LAT, altitude_m=60.0),
            PlannedCapture(index=1, longitude=BASE_LON + 0.00002, latitude=BASE_LAT,
                           altitude_m=60.0),
        ]
        images = [image(0, BASE_LON, BASE_LAT)]

        report = match_captures(points, images)
        assert len(report.matches) == 1
        assert len(report.missed) == 1

    def test_an_image_beyond_the_radius_is_not_matched(self):
        points = planned(1)
        # ~110 m north, far outside the 15 m radius.
        images = [image(0, BASE_LON, BASE_LAT + 0.001)]

        report = match_captures(points, images, match_radius_m=15.0)
        assert report.matches == []
        assert len(report.missed) == 1
        assert len(report.unplanned) == 1

    def test_the_nearest_image_wins_when_several_are_in_range(self):
        points = [PlannedCapture(index=0, longitude=BASE_LON, latitude=BASE_LAT,
                                 altitude_m=60.0)]
        far = image(1, BASE_LON + 0.0001, BASE_LAT)
        near = image(2, BASE_LON + 0.00001, BASE_LAT)

        report = match_captures(points, [far, near])
        assert report.matches[0].image_path.endswith("DSC00002.JPG")


class TestGeotagging:
    def test_an_image_without_a_position_is_set_aside_not_guessed(self):
        """Guessing from filename order would fabricate a correspondence."""
        points = planned(2)
        images = [
            image(0, points[0].longitude, points[0].latitude),
            CapturedImage(path="DSC00001.JPG"),  # no GPS
        ]

        report = match_captures(points, images)
        assert len(report.ungeotagged) == 1
        assert len(report.matches) == 1
        assert len(report.missed) == 1

    def test_ungeotagged_images_produce_a_recommendation(self):
        points = planned(1)
        report = match_captures(points, [CapturedImage(path="DSC00001.JPG")])
        payload = report.to_dict()
        assert any("no GPS position" in note for note in payload["recommendations"])


class TestDeviation:
    def test_deviation_statistics_describe_how_closely_the_plan_was_flown(self):
        points = planned(3)
        # Each image offset slightly north of its planned point.
        images = [image(i, p.longitude, p.latitude + 0.00003)
                  for i, p in enumerate(points)]

        report = match_captures(points, images)
        stats = report.deviation_stats()
        assert stats["mean_m"] > 0
        assert stats["max_m"] < 15.0

    def test_a_loosely_flown_plan_is_flagged_while_still_matching(self):
        """The plan was flown, but not closely: worth saying so."""
        points = planned(2)
        # ~11 m north: inside the 15 m radius, outside the 8 m warning.
        images = [image(i, p.longitude, p.latitude + 0.0001)
                  for i, p in enumerate(points)]

        report = match_captures(points, images)
        assert len(report.matches) == 2
        payload = report.to_dict()
        assert payload["deviation"]["over_warning"] == 2
        assert any("loosely" in note for note in payload["recommendations"])

    def test_altitude_difference_is_reported_when_known(self):
        points = planned(1)
        images = [image(0, points[0].longitude, points[0].latitude, alt=68.0)]

        report = match_captures(points, images)
        assert report.matches[0].altitude_difference_m == pytest.approx(8.0)

    def test_a_clean_flight_produces_no_recommendations(self):
        points = planned(3)
        images = [image(i, p.longitude, p.latitude) for i, p in enumerate(points)]
        assert match_captures(points, images).to_dict()["recommendations"] == []


class TestPlanExtraction:
    def test_capture_points_come_from_the_flight_recipe(self):
        plan = {"flight_recipe": {"world_poses": [
            {"lon": -81.75, "lat": 41.30, "alt_m": 60.0, "yaw_deg": 90.0, "trigger": True},
            {"lon": -81.74, "lat": 41.30, "alt_m": 60.0, "yaw_deg": 90.0, "trigger": True},
        ]}}
        captures = planned_captures_from_plan(plan)
        assert len(captures) == 2
        assert captures[0].yaw_deg == pytest.approx(90.0)

    def test_a_transit_waypoint_is_not_counted_as_a_capture_point(self):
        """Otherwise every turn would be reported as a missed photograph."""
        plan = {"flight_recipe": {"world_poses": [
            {"lon": -81.75, "lat": 41.30, "alt_m": 60.0, "trigger": True},
            {"lon": -81.745, "lat": 41.30, "alt_m": 60.0, "trigger": False},
        ]}}
        assert len(planned_captures_from_plan(plan)) == 1

    def test_bare_waypoints_are_used_when_there_is_no_recipe(self):
        plan = {"waypoints": [[-81.75, 41.30, 60.0], [-81.74, 41.30, 60.0]]}
        captures = planned_captures_from_plan(plan)
        assert len(captures) == 2
        assert captures[0].altitude_m == pytest.approx(60.0)

    def test_a_real_plan_from_the_planner_yields_capture_points(self):
        from mission.planner import MissionPlanner

        aoi = [[-81.7510, 41.3035], [-81.7490, 41.3035],
               [-81.7490, 41.3050], [-81.7510, 41.3050]]
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=aoi, altitude_m=60.0)
        captures = planned_captures_from_plan(plan.to_dict())
        assert len(captures) > 0
        for capture in captures:
            assert -180.0 <= capture.longitude <= 180.0
            assert -90.0 <= capture.latitude <= 90.0


class TestReconstructionPriors:
    def test_priors_are_supplied_only_for_matched_images(self):
        """An unmatched image must not be given a position it was not observed at."""
        points = planned(3)
        images = [image(i, p.longitude, p.latitude) for i, p in enumerate(points[:2])]

        report = match_captures(points, images)
        priors = pose_priors_for_reconstruction(report, points)

        assert priors["count"] == 2
        assert set(priors["priors"]) == {"DSC00000.JPG", "DSC00001.JPG"}
        assert "a position they were not observed at" in priors["note"]

    def test_a_prior_carries_the_planned_pose_and_the_observed_offset(self):
        points = [PlannedCapture(index=0, longitude=BASE_LON, latitude=BASE_LAT,
                                 altitude_m=60.0, yaw_deg=45.0, gimbal_pitch_deg=-90.0)]
        report = match_captures(points, [image(0, BASE_LON, BASE_LAT)])
        prior = pose_priors_for_reconstruction(report, points)["priors"]["DSC00000.JPG"]

        assert prior["yaw_deg"] == pytest.approx(45.0)
        assert prior["gimbal_pitch_deg"] == pytest.approx(-90.0)
        assert prior["planned_index"] == 0
        assert prior["observed_offset_m"] >= 0.0


class TestReportShape:
    def test_the_matching_method_is_stated(self):
        report = match_captures(planned(1), [])
        assert "one-to-one" in report.to_dict()["method"]

    def test_totals_are_reported_even_when_nothing_matched(self):
        report = match_captures(planned(4), [])
        payload = report.to_dict()
        assert payload["planned_total"] == 4
        assert payload["image_total"] == 0
        assert payload["matched"] == 0
        assert payload["coverage_pct"] == pytest.approx(0.0)
