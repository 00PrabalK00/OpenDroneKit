"""Which segment of a linked mission actually completed.

A linked mission is several surveys flown as one sortie: three roof inspections and the
transits between them, compiled into a single plan. That much already works. What did
not was knowing where it got to.

Without per-segment tracking the only answer available after a battery swap is a
percentage of the whole sortie, and a percentage cannot say the thing an operator needs:
segment two is finished, segment three is half done, segment four was never started. So
the choice is to re-fly everything after the last confirmed image, which throws away
completed surveys, or to guess, which leaves a hole in one of them.

Attribution comes from the plan itself. The compiler already stamps every pose with the
segment that produced it -- ``linked_seg2:grid`` -- so the mapping from capture point to
survey is recorded at compile time by the code that did the linking. Reading it back is
honest in a way a separate progress counter is not: a counter can disagree with the
flight, while the stamp is the flight.

Completion still comes from the images, through the same matcher the single-mission
resume uses. A segment counts as complete only when every one of its capture points
matched an image. Everything ambiguous is re-flown, for the reason stated in
``mission.resume``: a duplicate photograph costs battery, a missing one costs the survey.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import capture_matching

from . import resume

SEGMENT_PREFIX = re.compile(r"^linked_seg(\d+):")


class NotLinked(ValueError):
    """This plan is not a linked mission, so it has no segments to report on."""


@dataclass
class SegmentProgress:
    """One survey inside the sortie, and how much of it is in the bag."""

    index: int
    recipe_id: str
    planned_indices: list[int] = field(default_factory=list)
    completed_indices: list[int] = field(default_factory=list)

    @property
    def planned_total(self) -> int:
        return len(self.planned_indices)

    @property
    def remaining_indices(self) -> list[int]:
        done = set(self.completed_indices)
        return [i for i in self.planned_indices if i not in done]

    @property
    def is_complete(self) -> bool:
        return bool(self.planned_indices) and not self.remaining_indices

    @property
    def is_started(self) -> bool:
        return bool(self.completed_indices)

    @property
    def progress_pct(self) -> float:
        if not self.planned_indices:
            return 0.0
        return 100.0 * len(set(self.completed_indices)) / len(self.planned_indices)

    @property
    def state(self) -> str:
        if self.is_complete:
            return "complete"
        return "partial" if self.is_started else "not_started"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "recipe_id": self.recipe_id,
            "state": self.state,
            "planned_total": self.planned_total,
            "completed": len(set(self.completed_indices)),
            "remaining": len(self.remaining_indices),
            "remaining_indices": self.remaining_indices,
            "progress_pct": round(self.progress_pct, 1),
        }


def _plan_dict(plan: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    return plan.to_dict() if hasattr(plan, "to_dict") else {}


def linked_metadata(plan: dict[str, Any] | Any) -> dict[str, Any]:
    """The linking metadata, or a refusal saying this is a single mission."""
    payload = _plan_dict(plan)
    recipe = payload.get("flight_recipe") or {}
    metadata = dict(recipe.get("metadata") or {})
    count = int(metadata.get("linked_segment_count", 0) or 0)
    if count <= 0:
        raise NotLinked(
            "This plan was not compiled as a linked mission, so it has one segment by "
            "definition. Use the ordinary resume report for it."
        )
    return metadata


def segments_from_plan(plan: dict[str, Any] | Any) -> list[SegmentProgress]:
    """Split the planned capture points into the surveys that produced them."""
    metadata = linked_metadata(plan)
    recipe_ids = [str(v) for v in (metadata.get("linked_segment_recipe_ids") or [])]
    declared = int(metadata.get("linked_segment_count", 0) or 0)

    captures = capture_matching.planned_captures_from_plan(plan)
    if not captures:
        raise NotLinked(
            "The linked plan carries no capture points, so there is nothing to attribute "
            "to a segment."
        )

    buckets: dict[int, list[int]] = {}
    unstamped = 0
    for capture in captures:
        match = SEGMENT_PREFIX.match(capture.primitive or "")
        if match is None:
            unstamped += 1
            continue
        buckets.setdefault(int(match.group(1)), []).append(capture.index)

    if unstamped:
        raise NotLinked(
            f"{unstamped} of {len(captures)} capture points carry no segment stamp, so "
            "they cannot be attributed to a survey. Reporting per-segment progress from "
            "a partial attribution would say a segment is finished when points belonging "
            "to it were never counted."
        )
    if len(buckets) != declared:
        raise NotLinked(
            f"The plan declares {declared} segments but its capture points name "
            f"{len(buckets)}. The attribution is not trustworthy, so no per-segment "
            "progress is reported."
        )

    segments: list[SegmentProgress] = []
    for number in sorted(buckets):
        recipe_id = recipe_ids[number - 1] if 0 < number <= len(recipe_ids) else ""
        segments.append(SegmentProgress(index=number, recipe_id=recipe_id,
                                        planned_indices=sorted(buckets[number])))
    return segments


def linked_progress(plan: dict[str, Any] | Any,
                    image_folder: str | Path,
                    *,
                    match_radius_m: float = capture_matching.DEFAULT_MATCH_RADIUS_M
                    ) -> dict[str, Any]:
    """Per-segment completion for a linked mission, worked out from the images.

    The overall figure is kept alongside, because a sortie that is 80 per cent complete
    with one survey untouched is a different job from one that is 80 per cent complete
    across the board, and only the per-segment view distinguishes them.
    """
    segments = segments_from_plan(plan)
    state = resume.state_from_images(plan, image_folder, match_radius_m=match_radius_m)
    completed = set(state.completed_indices)

    for segment in segments:
        segment.completed_indices = [i for i in segment.planned_indices if i in completed]

    complete = [s for s in segments if s.is_complete]
    partial = [s for s in segments if s.state == "partial"]
    untouched = [s for s in segments if s.state == "not_started"]

    if partial:
        next_action = (
            f"Segment {partial[0].index} is part flown: "
            f"{len(partial[0].remaining_indices)} capture point(s) left. Finish it before "
            "starting a segment that has not been begun."
        )
    elif untouched:
        next_action = (
            f"Segment {untouched[0].index} has not been started. Every earlier segment "
            "is complete, so the aircraft can go straight to it."
        )
    else:
        next_action = "Every segment is complete; nothing remains to fly."

    return {
        "linked": True,
        "segment_count": len(segments),
        "segments": [s.to_dict() for s in segments],
        "complete_segments": [s.index for s in complete],
        "partial_segments": [s.index for s in partial],
        "not_started_segments": [s.index for s in untouched],
        "overall": state.to_dict(),
        "next_action": next_action,
        "method": (
            "Segment attribution is read from the primitive stamp the compiler wrote at "
            "link time, not from a progress counter that could disagree with the flight. "
            "A segment counts as complete only when every one of its capture points "
            "matched an image."
        ),
    }


def resume_linked_mission(plan: dict[str, Any] | Any,
                          image_folder: str | Path,
                          *,
                          match_radius_m: float = capture_matching.DEFAULT_MATCH_RADIUS_M
                          ) -> dict[str, Any]:
    """A resume plan for the linked sortie, with what it skips stated per segment."""
    progress = linked_progress(plan, image_folder, match_radius_m=match_radius_m)
    state = resume.state_from_images(plan, image_folder, match_radius_m=match_radius_m)
    resumed = resume.resume_plan(plan, state)

    skipped = [s for s in progress["segments"] if s["state"] == "complete"]
    return {
        **progress,
        "resume_plan": resumed,
        "skipped_segments": [s["index"] for s in skipped],
        "skipped_captures": sum(s["planned_total"] for s in skipped),
        "note": (
            f"{len(skipped)} completed segment(s) are not re-flown. Any segment with a "
            "single unmatched capture point is flown again in full from its remaining "
            "points, because a partial survey is not a survey."
        ),
    }
