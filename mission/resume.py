"""Resuming a survey after a battery swap, without re-flying what is already done.

Any site large enough to matter needs more than one battery. The mission estimator will
say so before the flight, but saying so is not much help on its own: the operator still
has to work out where the aircraft got to and fly the rest, usually from memory and a
half-charged tablet.

This works it out from evidence instead. The images already on the card are the record
of what was captured, so a resume mission is the planned capture points that produced
no image. That reuses the matching in ``core.capture_matching`` rather than trusting a
separate progress counter, which is the right source because a counter can be wrong
about a photograph that was never written to disk.

The rule everywhere here is to err towards re-flying. A capture point is treated as done
only when an image actually matched it. Anything ambiguous -- an image beyond the match
radius, an image with no GPS, a point the aircraft passed but did not trigger over -- is
put back into the resume mission. Re-photographing a point costs a little battery. Not
photographing it leaves a hole in the overlap that surfaces days later when
reconstruction fails, and by then the aircraft is packed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from core import capture_matching


@dataclass
class FlightSegment:
    """One battery's worth of flying, recorded when it ends."""

    index: int
    completed_indices: list[int] = field(default_factory=list)
    battery_id: str = ""
    ended_reason: str = "battery"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "completed": len(self.completed_indices),
            "completed_indices": self.completed_indices,
            "battery_id": self.battery_id,
            "ended_reason": self.ended_reason,
            "note": self.note,
        }


@dataclass
class ResumeState:
    """What has been flown so far, and what is left."""

    planned_total: int
    completed_indices: list[int] = field(default_factory=list)
    segments: list[FlightSegment] = field(default_factory=list)
    ungeotagged_images: list[str] = field(default_factory=list)

    @property
    def remaining_indices(self) -> list[int]:
        done = set(self.completed_indices)
        return [i for i in range(self.planned_total) if i not in done]

    @property
    def is_complete(self) -> bool:
        return not self.remaining_indices

    @property
    def progress_pct(self) -> float:
        if not self.planned_total:
            return 0.0
        return 100.0 * len(set(self.completed_indices)) / self.planned_total

    def to_dict(self) -> dict[str, Any]:
        warnings: list[str] = []
        if self.ungeotagged_images:
            warnings.append(
                f"{len(self.ungeotagged_images)} image(s) carry no GPS position, so the "
                "points they may have covered are being re-flown. Check the camera's "
                "geotagging settings before the next sortie."
            )

        return {
            "planned_total": self.planned_total,
            "completed": len(set(self.completed_indices)),
            "remaining": len(self.remaining_indices),
            "remaining_indices": self.remaining_indices,
            "progress_pct": round(self.progress_pct, 1),
            "complete": self.is_complete,
            "segments": [s.to_dict() for s in self.segments],
            "warnings": warnings,
            "method": (
                "A capture point counts as flown only when an image matched it. "
                "Anything ambiguous is re-flown, because a duplicate photograph costs "
                "battery while a missing one costs the survey."
            ),
        }


def state_from_images(plan: dict[str, Any] | Any,
                      image_folder: str | Path,
                      *,
                      match_radius_m: float = capture_matching.DEFAULT_MATCH_RADIUS_M
                      ) -> ResumeState:
    """Work out what is left to fly from the images already captured."""
    planned = capture_matching.planned_captures_from_plan(plan)
    if not planned:
        raise ValueError(
            "The plan contains no capture points, so there is nothing to resume."
        )

    images = capture_matching.images_from_folder(image_folder)
    report = capture_matching.match_captures(planned, images,
                                             match_radius_m=match_radius_m)

    # planned_captures_from_plan preserves each point's original index, which is what
    # the resume mission must refer back to.
    position_of = {capture.index: position for position, capture in enumerate(planned)}
    completed = sorted(
        position_of[match.planned_index]
        for match in report.matches
        if match.planned_index in position_of
    )

    return ResumeState(
        planned_total=len(planned),
        completed_indices=completed,
        ungeotagged_images=[image.path for image in report.ungeotagged],
    )


def state_from_segments(planned_total: int,
                        segments: Iterable[FlightSegment]) -> ResumeState:
    """Assemble progress from recorded segments, for when the imagery is not to hand."""
    segments = list(segments)
    completed: list[int] = []
    for segment in segments:
        completed.extend(segment.completed_indices)

    return ResumeState(
        planned_total=planned_total,
        completed_indices=sorted(set(completed)),
        segments=segments,
    )


def resume_captures(plan: dict[str, Any] | Any,
                    state: ResumeState) -> list[capture_matching.PlannedCapture]:
    """The capture points still to be flown, in their original order."""
    planned = capture_matching.planned_captures_from_plan(plan)
    remaining = set(state.remaining_indices)
    return [capture for position, capture in enumerate(planned) if position in remaining]


def resume_plan(plan: dict[str, Any] | Any, state: ResumeState) -> dict[str, Any]:
    """Build a plan covering only what is left.

    The result keeps the original plan's camera, altitude and overlap so the resumed
    imagery matches what came before it -- a second sortie flown at a different altitude
    would produce a survey that does not reconstruct as one.
    """
    payload = plan if isinstance(plan, dict) else plan.to_dict()
    remaining = resume_captures(payload, state)

    if not remaining:
        return {
            "complete": True,
            "waypoints": [],
            "capture_count": 0,
            "note": "Every planned capture point already has an image. Nothing to re-fly.",
        }

    waypoints = [[c.longitude, c.latitude, c.altitude_m] for c in remaining]

    return {
        "complete": False,
        "resumed_from": payload.get("template") or payload.get("mode") or "mission",
        "camera": payload.get("camera"),
        "altitude_m": payload.get("altitude_m"),
        "front_overlap_pct": payload.get("front_overlap_pct"),
        "side_overlap_pct": payload.get("side_overlap_pct"),
        "waypoints": waypoints,
        "capture_count": len(remaining),
        "capture_indices": [c.index for c in remaining],
        "poses": [
            {
                "lon": c.longitude, "lat": c.latitude, "alt_m": c.altitude_m,
                "yaw_deg": c.yaw_deg, "gimbal_pitch_deg": c.gimbal_pitch_deg,
                "primitive": c.primitive, "trigger": True,
                "original_index": c.index,
            }
            for c in remaining
        ],
        "progress": state.to_dict(),
        "note": (
            f"{len(remaining)} of {state.planned_total} capture points remain. Camera, "
            "altitude and overlap are carried over from the original plan: a resumed "
            "sortie flown to different settings would not reconstruct with the imagery "
            "already captured."
        ),
    }


def plan_battery_segments(plan: dict[str, Any] | Any,
                          aircraft=None) -> dict[str, Any]:
    """Split a mission into sorties that each fit inside one battery.

    Done before the flight rather than after it, so the operator knows where the swaps
    will fall and can position themselves accordingly, rather than discovering the
    aircraft is low over the far end of the site.
    """
    from .estimates import AircraftProfile, estimate_mission

    payload = plan if isinstance(plan, dict) else plan.to_dict()
    aircraft = aircraft or AircraftProfile()

    estimate = estimate_mission(payload, aircraft=aircraft)
    captures = capture_matching.planned_captures_from_plan(payload)
    total = len(captures)

    if not total:
        raise ValueError("The plan contains no capture points to split.")

    required = int(estimate["battery"]["batteries_required"])
    if required <= 1:
        return {
            "segments": 1,
            "splits": [],
            "note": (
                "The mission fits inside one battery, so no swap is planned. The "
                "reserve is already held back from the endurance used here."
            ),
            "estimate": estimate,
        }

    # Split by capture count, which tracks flying time closely enough for a survey
    # flown at constant speed and spacing. Where it does not -- a mission mixing
    # transit legs with dense inspection -- the operator sees the estimate alongside.
    per_segment = -(-total // required)   # ceiling division
    splits: list[dict[str, Any]] = []
    for segment_index in range(required):
        start = segment_index * per_segment
        end = min(total, start + per_segment)
        if start >= end:
            break
        splits.append({
            "segment": segment_index + 1,
            "capture_indices": [captures[i].index for i in range(start, end)],
            "first_capture": captures[start].index,
            "last_capture": captures[end - 1].index,
            "capture_count": end - start,
        })

    return {
        "segments": len(splits),
        "splits": splits,
        "estimate": estimate,
        "note": (
            f"Split into {len(splits)} sorties of about {per_segment} capture points "
            "each, sized so every sortie fits inside one battery with its reserve "
            "intact. Segment boundaries are estimates from capture count; fly to the "
            "aircraft's own battery warning, not to this number."
        ),
    }
