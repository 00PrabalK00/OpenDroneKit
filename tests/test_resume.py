"""Resuming a survey after a battery swap.

One property matters above the rest: the resumed sortie must not skip a capture point.
Re-photographing one costs a little battery; missing one leaves a hole in the overlap
that surfaces days later when reconstruction fails, with the aircraft long packed away.
So every ambiguous case below is required to resolve towards re-flying, and the tests
check that direction explicitly rather than just checking the counts add up.
"""

from __future__ import annotations

import pytest

from core.capture_matching import CapturedImage, PlannedCapture
from mission.estimates import AircraftProfile
from mission.resume import (
    FlightSegment,
    ResumeState,
    plan_battery_segments,
    resume_captures,
    resume_plan,
    state_from_images,
    state_from_segments,
)

BASE_LON, BASE_LAT = -81.7505, 41.3042


def plan_with(count: int, spacing_deg: float = 0.0005) -> dict:
    return {
        "template": "grid", "camera": "mavic2pro", "altitude_m": 60.0,
        "front_overlap_pct": 75.0, "side_overlap_pct": 65.0,
        "estimated_time_min": 30.0, "path_distance_m": 4000.0,
        "flight_recipe": {"world_poses": [
            {"lon": BASE_LON + i * spacing_deg, "lat": BASE_LAT, "alt_m": 60.0,
             "yaw_deg": 90.0, "gimbal_pitch_deg": -90.0, "trigger": True}
            for i in range(count)
        ]},
    }


def write_images(folder, plan, indices, offset_deg: float = 0.0):
    """Write geotagged stand-ins for the images a partial flight would have produced."""
    poses = plan["flight_recipe"]["world_poses"]
    images = []
    for i in indices:
        images.append(CapturedImage(
            path=str(folder / f"DSC{i:05d}.JPG"),
            longitude=poses[i]["lon"] + offset_deg,
            latitude=poses[i]["lat"],
            altitude_m=60.0,
        ))
    return images


class TestStateFromSegments:
    def test_progress_is_the_union_of_completed_segments(self):
        state = state_from_segments(10, [
            FlightSegment(index=1, completed_indices=[0, 1, 2, 3]),
            FlightSegment(index=2, completed_indices=[4, 5]),
        ])
        assert state.progress_pct == pytest.approx(60.0)
        assert state.remaining_indices == [6, 7, 8, 9]

    def test_a_point_recorded_twice_is_counted_once(self):
        state = state_from_segments(5, [
            FlightSegment(index=1, completed_indices=[0, 1, 2]),
            FlightSegment(index=2, completed_indices=[2, 3]),
        ])
        assert state.remaining_indices == [4]

    def test_a_finished_mission_reports_complete(self):
        state = state_from_segments(3, [FlightSegment(index=1, completed_indices=[0, 1, 2])])
        assert state.is_complete is True
        assert state.remaining_indices == []

    def test_nothing_flown_yet_leaves_everything_remaining(self):
        state = state_from_segments(4, [])
        assert state.remaining_indices == [0, 1, 2, 3]
        assert state.progress_pct == 0.0


class TestStateFromImages:
    def test_captured_points_are_recognised_from_the_imagery(self, tmp_path, monkeypatch):
        plan = plan_with(10)
        images = write_images(tmp_path, plan, [0, 1, 2, 3, 4])
        monkeypatch.setattr("core.capture_matching.images_from_folder",
                            lambda folder: images)

        state = state_from_images(plan, tmp_path)
        assert sorted(state.completed_indices) == [0, 1, 2, 3, 4]
        assert state.remaining_indices == [5, 6, 7, 8, 9]

    def test_an_image_beyond_the_match_radius_does_not_count_as_done(
            self, tmp_path, monkeypatch):
        """Erring here would leave a gap; erring the other way costs one photograph."""
        plan = plan_with(4)
        # ~110 m north of its planned point, far outside the 15 m radius.
        far = write_images(tmp_path, plan, [0])
        far[0].latitude += 0.001
        monkeypatch.setattr("core.capture_matching.images_from_folder", lambda folder: far)

        state = state_from_images(plan, tmp_path)
        assert 0 in state.remaining_indices

    def test_an_ungeotagged_image_is_re_flown_and_the_camera_is_flagged(
            self, tmp_path, monkeypatch):
        plan = plan_with(4)
        images = [CapturedImage(path=str(tmp_path / "DSC00000.JPG"))]
        monkeypatch.setattr("core.capture_matching.images_from_folder", lambda folder: images)

        state = state_from_images(plan, tmp_path)
        assert state.remaining_indices == [0, 1, 2, 3]
        assert any("geotagging" in w for w in state.to_dict()["warnings"])

    def test_a_plan_with_no_capture_points_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to resume"):
            state_from_images({"flight_recipe": {"world_poses": []}}, tmp_path)

    def test_the_method_states_which_way_ambiguity_is_resolved(self):
        payload = state_from_segments(3, []).to_dict()
        assert "re-flown" in payload["method"]


class TestResumePlan:
    def test_the_resume_mission_covers_exactly_what_is_left(self):
        plan = plan_with(10)
        state = state_from_segments(10, [FlightSegment(1, completed_indices=list(range(6)))])

        resumed = resume_plan(plan, state)
        assert resumed["capture_count"] == 4
        assert resumed["capture_indices"] == [6, 7, 8, 9]

    def test_no_capture_point_is_flown_twice(self):
        """The whole point of resuming rather than repeating."""
        plan = plan_with(8)
        state = state_from_segments(8, [FlightSegment(1, completed_indices=[0, 1, 2, 3])])

        resumed = resume_plan(plan, state)
        assert set(resumed["capture_indices"]).isdisjoint({0, 1, 2, 3})

    def test_no_capture_point_is_dropped_between_the_two_sorties(self):
        plan = plan_with(9)
        first = list(range(4))
        state = state_from_segments(9, [FlightSegment(1, completed_indices=first)])

        resumed = resume_plan(plan, state)
        covered = set(first) | set(resumed["capture_indices"])
        assert covered == set(range(9)), "a point flown by neither sortie is a coverage hole"

    def test_camera_altitude_and_overlap_carry_over(self):
        """A second sortie at different settings does not reconstruct with the first."""
        plan = plan_with(6)
        state = state_from_segments(6, [FlightSegment(1, completed_indices=[0, 1])])

        resumed = resume_plan(plan, state)
        assert resumed["camera"] == "mavic2pro"
        assert resumed["altitude_m"] == 60.0
        assert resumed["front_overlap_pct"] == 75.0

    def test_a_finished_mission_produces_nothing_to_fly(self):
        plan = plan_with(3)
        state = state_from_segments(3, [FlightSegment(1, completed_indices=[0, 1, 2])])

        resumed = resume_plan(plan, state)
        assert resumed["complete"] is True
        assert resumed["waypoints"] == []

    def test_each_remaining_pose_keeps_its_original_index(self):
        """So a third sortie can be resumed against the same original plan."""
        plan = plan_with(6)
        state = state_from_segments(6, [FlightSegment(1, completed_indices=[0, 1])])

        resumed = resume_plan(plan, state)
        assert [p["original_index"] for p in resumed["poses"]] == [2, 3, 4, 5]

    def test_the_remaining_poses_keep_their_yaw_and_gimbal(self):
        plan = plan_with(4)
        state = state_from_segments(4, [FlightSegment(1, completed_indices=[0])])

        pose = resume_plan(plan, state)["poses"][0]
        assert pose["yaw_deg"] == pytest.approx(90.0)
        assert pose["gimbal_pitch_deg"] == pytest.approx(-90.0)

    def test_resume_captures_returns_them_in_order(self):
        plan = plan_with(6)
        state = state_from_segments(6, [FlightSegment(1, completed_indices=[0, 3])])

        remaining = resume_captures(plan, state)
        assert [c.index for c in remaining] == [1, 2, 4, 5]


class TestBatterySegments:
    def test_a_short_mission_is_not_split(self):
        plan = plan_with(20)
        plan["estimated_time_min"] = 8.0

        split = plan_battery_segments(plan, AircraftProfile(endurance_min=30.0))
        assert split["segments"] == 1
        assert split["splits"] == []

    def test_a_long_mission_is_split_to_fit_the_battery(self):
        plan = plan_with(60)
        plan["estimated_time_min"] = 45.0

        split = plan_battery_segments(plan, AircraftProfile(endurance_min=20.0))
        assert split["segments"] >= 3

    def test_the_splits_cover_every_capture_point_exactly_once(self):
        """A split that loses a point is worse than not splitting at all."""
        plan = plan_with(50)
        plan["estimated_time_min"] = 40.0

        split = plan_battery_segments(plan, AircraftProfile(endurance_min=20.0))
        covered: list[int] = []
        for segment in split["splits"]:
            covered.extend(segment["capture_indices"])

        assert sorted(covered) == list(range(50))
        assert len(covered) == len(set(covered)), "a capture point appears in two sorties"

    def test_the_split_tells_the_operator_not_to_trust_it_over_the_aircraft(self):
        plan = plan_with(40)
        plan["estimated_time_min"] = 40.0

        split = plan_battery_segments(plan, AircraftProfile(endurance_min=15.0))
        assert "battery warning" in split["note"]

    def test_a_plan_with_no_captures_is_refused(self):
        with pytest.raises(ValueError, match="no capture points"):
            plan_battery_segments({"flight_recipe": {"world_poses": []},
                                   "estimated_time_min": 10.0})


class TestAgainstARealPlan:
    def test_a_planner_mission_can_be_split_and_resumed(self):
        from mission.planner import MissionPlanner

        aoi = [[-81.7515, 41.3030], [-81.7480, 41.3030],
               [-81.7480, 41.3060], [-81.7515, 41.3060]]
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=aoi, altitude_m=60.0)
        payload = plan.to_dict()

        split = plan_battery_segments(payload, AircraftProfile(endurance_min=12.0))
        assert split["segments"] >= 1

        first = split["splits"][0]["capture_indices"] if split["splits"] else []
        total = len(payload["waypoints"])
        state = state_from_segments(total, [FlightSegment(1, completed_indices=list(range(len(first))))])

        resumed = resume_plan(payload, state)
        assert resumed["capture_count"] == total - len(first)
