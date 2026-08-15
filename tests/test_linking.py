"""Per-segment completion for a linked mission.

A linked sortie is several surveys flown as one flight. Knowing it is 80 per cent
complete is close to useless: it does not distinguish "every survey is nearly done" from
"three surveys are finished and the fourth was never started", and those call for
opposite actions on site.

The tests below therefore assert attribution, not totals. A segment is complete only when
every one of its own capture points matched an image, and the refusals fire whenever the
attribution cannot be trusted -- an unstamped pose, a declared segment count that
disagrees with the poses, a plan that was never linked at all. Reporting a segment as
finished when points belonging to it were never counted is the failure being designed
against.
"""

from __future__ import annotations

import pytest

from core.capture_matching import CapturedImage
from mission.linking import (
    NotLinked,
    linked_progress,
    resume_linked_mission,
    segments_from_plan,
)

BASE_LON, BASE_LAT = -81.7505, 41.3042


def linked_plan(segment_sizes=(3, 3, 3), *, recipe_ids=None, stamp=True,
                declared_count=None) -> dict:
    """A compiled linked mission: poses stamped with the segment that produced them."""
    poses = []
    index = 0
    for segment_number, size in enumerate(segment_sizes, start=1):
        for _ in range(size):
            primitive = f"linked_seg{segment_number}:grid" if stamp else "grid"
            poses.append({
                "lon": BASE_LON + index * 0.0005, "lat": BASE_LAT, "alt_m": 60.0,
                "yaw_deg": 90.0, "gimbal_pitch_deg": -90.0, "trigger": True,
                "primitive": primitive,
            })
            index += 1

    ids = list(recipe_ids or [f"fr-seg-{n}" for n in range(1, len(segment_sizes) + 1)])
    return {
        "template": "grid", "camera": "mavic2pro", "altitude_m": 60.0,
        "front_overlap_pct": 75.0, "side_overlap_pct": 65.0,
        "estimated_time_min": 40.0, "path_distance_m": 6000.0,
        "flight_recipe": {
            "world_poses": poses,
            "metadata": {
                "linked_segment_count": (len(segment_sizes) if declared_count is None
                                         else declared_count),
                "linked_segment_recipe_ids": ids,
            },
        },
    }


def flown(monkeypatch, plan, indices, offset_deg: float = 0.0):
    """Stand in for the images a partial sortie would have left on the card.

    The offset is applied north rather than east on purpose: capture points here are
    spaced along a line of latitude, so an eastward offset would simply land the image on
    a different capture point and be matched to that one instead of missing altogether.
    """
    poses = plan["flight_recipe"]["world_poses"]
    images = [
        CapturedImage(path=f"DSC{i:05d}.JPG",
                      longitude=poses[i]["lon"],
                      latitude=poses[i]["lat"] + offset_deg, altitude_m=60.0)
        for i in indices
    ]
    monkeypatch.setattr("core.capture_matching.images_from_folder", lambda folder: images)


class TestAttribution:
    def test_capture_points_are_split_by_the_segment_that_produced_them(self):
        segments = segments_from_plan(linked_plan((2, 3, 4)))

        assert [s.index for s in segments] == [1, 2, 3]
        assert [s.planned_total for s in segments] == [2, 3, 4]

    def test_each_segment_carries_the_recipe_it_came_from(self):
        segments = segments_from_plan(linked_plan((2, 2), recipe_ids=["roof-a", "roof-b"]))
        assert [s.recipe_id for s in segments] == ["roof-a", "roof-b"]

    def test_transit_waypoints_are_not_counted_as_capture_points(self):
        """A transit between surveys is not a missed photograph."""
        plan = linked_plan((2, 2))
        plan["flight_recipe"]["world_poses"].insert(2, {
            "lon": BASE_LON, "lat": BASE_LAT, "alt_m": 80.0, "yaw_deg": 0.0,
            "gimbal_pitch_deg": 0.0, "trigger": False,
            "primitive": "linked_transition",
        })

        segments = segments_from_plan(plan)
        assert sum(s.planned_total for s in segments) == 4


class TestProgress:
    def test_a_finished_segment_is_reported_complete_and_the_rest_are_not(
            self, tmp_path, monkeypatch):
        plan = linked_plan((3, 3, 3))
        flown(monkeypatch, plan, [0, 1, 2, 3])

        report = linked_progress(plan, tmp_path)
        assert report["complete_segments"] == [1]
        assert report["partial_segments"] == [2]
        assert report["not_started_segments"] == [3]

    def test_a_segment_missing_one_point_is_partial_not_complete(
            self, tmp_path, monkeypatch):
        """The distinction the whole feature exists for."""
        plan = linked_plan((3, 3))
        flown(monkeypatch, plan, [0, 1, 3, 4, 5])

        report = linked_progress(plan, tmp_path)
        assert report["complete_segments"] == [2]
        assert report["partial_segments"] == [1]
        assert report["segments"][0]["remaining_indices"] == [2]

    def test_the_next_action_names_the_part_flown_segment_first(
            self, tmp_path, monkeypatch):
        plan = linked_plan((3, 3, 3))
        flown(monkeypatch, plan, [0, 1, 2, 3])

        assert "Segment 2 is part flown" in linked_progress(plan, tmp_path)["next_action"]

    def test_with_nothing_part_flown_the_next_untouched_segment_is_named(
            self, tmp_path, monkeypatch):
        plan = linked_plan((2, 2, 2))
        flown(monkeypatch, plan, [0, 1, 2, 3])

        action = linked_progress(plan, tmp_path)["next_action"]
        assert "Segment 3 has not been started" in action

    def test_a_finished_sortie_says_nothing_remains(self, tmp_path, monkeypatch):
        plan = linked_plan((2, 2))
        flown(monkeypatch, plan, [0, 1, 2, 3])

        report = linked_progress(plan, tmp_path)
        assert report["complete_segments"] == [1, 2]
        assert "nothing remains" in report["next_action"]

    def test_the_overall_figure_is_kept_alongside_the_per_segment_view(
            self, tmp_path, monkeypatch):
        plan = linked_plan((5, 5))
        flown(monkeypatch, plan, list(range(8)))

        report = linked_progress(plan, tmp_path)
        assert report["overall"]["progress_pct"] == pytest.approx(80.0)
        # 80 per cent overall, but one whole survey is untouched by three points.
        assert report["partial_segments"] == [2]

    def test_an_image_beyond_the_match_radius_leaves_its_segment_incomplete(
            self, tmp_path, monkeypatch):
        """Erring towards re-flying, exactly as the single-mission resume does."""
        plan = linked_plan((2, 2))
        flown(monkeypatch, plan, [0, 1, 2, 3], offset_deg=0.001)

        report = linked_progress(plan, tmp_path)
        assert report["complete_segments"] == []


class TestResumingASortie:
    def test_completed_segments_are_not_re_flown(self, tmp_path, monkeypatch):
        plan = linked_plan((3, 3, 3))
        flown(monkeypatch, plan, [0, 1, 2, 3])

        report = resume_linked_mission(plan, tmp_path)
        assert report["skipped_segments"] == [1]
        assert report["skipped_captures"] == 3

    def test_the_resume_plan_covers_only_what_is_left(self, tmp_path, monkeypatch):
        plan = linked_plan((3, 3))
        flown(monkeypatch, plan, [0, 1, 2])

        report = resume_linked_mission(plan, tmp_path)
        assert report["resume_plan"]["complete"] is False
        assert report["overall"]["remaining"] == 3

    def test_what_is_skipped_is_stated_rather_than_implied(self, tmp_path, monkeypatch):
        plan = linked_plan((2, 2))
        flown(monkeypatch, plan, [0, 1])

        assert "not re-flown" in resume_linked_mission(plan, tmp_path)["note"]


class TestRefusals:
    def test_a_single_mission_plan_is_refused_rather_than_called_one_segment(
            self, tmp_path):
        plan = linked_plan((3,))
        plan["flight_recipe"]["metadata"] = {}

        with pytest.raises(NotLinked, match="not compiled as a linked mission"):
            segments_from_plan(plan)

    def test_unstamped_poses_are_refused_rather_than_partly_attributed(self):
        """A segment reported complete while its points went uncounted is the bug."""
        with pytest.raises(NotLinked, match="carry no segment stamp"):
            segments_from_plan(linked_plan((2, 2), stamp=False))

    def test_a_declared_count_that_disagrees_with_the_poses_is_refused(self):
        with pytest.raises(NotLinked, match="not trustworthy"):
            segments_from_plan(linked_plan((2, 2), declared_count=3))

    def test_a_linked_plan_with_no_capture_points_is_refused(self):
        plan = linked_plan((2, 2))
        for pose in plan["flight_recipe"]["world_poses"]:
            pose["trigger"] = False

        with pytest.raises(NotLinked, match="nothing to attribute"):
            segments_from_plan(plan)
