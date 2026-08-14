"""How close the aircraft may fly to a structure, and proving it never flew closer.

Inspection wants the camera near the surface: resolution improves linearly as stand-off
shrinks, and a hairline crack invisible at fifteen metres is obvious at four. Safety
wants the opposite. Every inspection mission is that trade, and the number chosen is
usually typed once and never checked against the geometry that came out.

Three policies are supported. *Fixed* uses one distance for the whole job. *Per surface*
lets each face carry its own, because a blank wall and a balconied elevation do not
deserve the same clearance. *Adaptive* works backwards from the resolution the survey
was ordered at, which is how the requirement actually arrives.

The guarantee is the part worth having. Adaptive stand-off is clamped so it can never
fall below the minimum clearance, and when the ordered GSD would require flying closer
than that, the conflict is reported rather than resolved. Quietly honouring the GSD
would fly the aircraft too close to a building; quietly honouring the clearance would
deliver a survey at the wrong resolution while reporting success. Neither is ours to
choose silently -- the operator decides, having been told.

``verify_clearance`` then checks the compiled mission against the geometry, because a
policy is an intention and waypoints are what gets flown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Below this, rotor wash and GPS error make a fixed-wing or multirotor stand-off
# meaningless regardless of what the policy asks for.
ABSOLUTE_MINIMUM_CLEARANCE_M = 2.0

DEFAULT_MINIMUM_CLEARANCE_M = 5.0

POLICIES = ("fixed", "per_surface", "adaptive")


class StandoffConflict(ValueError):
    """The requested resolution and the required clearance cannot both be met."""


@dataclass
class StandoffDecision:
    """The stand-off chosen for one surface, and why."""

    surface_id: str
    distance_m: float
    policy: str
    requested_gsd_cm: float | None = None
    achieved_gsd_cm: float | None = None
    clamped: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "distance_m": round(self.distance_m, 2),
            "policy": self.policy,
            "requested_gsd_cm": self.requested_gsd_cm,
            "achieved_gsd_cm": (round(self.achieved_gsd_cm, 3)
                                if self.achieved_gsd_cm is not None else None),
            "clamped": self.clamped,
            "warnings": self.warnings,
        }


@dataclass
class ClearanceViolation:
    """One capture point closer to a surface than the mission allows."""

    point_index: int
    surface_id: str
    distance_m: float
    required_m: float

    @property
    def shortfall_m(self) -> float:
        return self.required_m - self.distance_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "surface_id": self.surface_id,
            "distance_m": round(self.distance_m, 2),
            "required_m": round(self.required_m, 2),
            "shortfall_m": round(self.shortfall_m, 2),
        }


def standoff_for_gsd(camera: str, gsd_cm: float) -> float:
    """Distance at which a camera resolves the requested ground sample distance.

    The same similar-triangles relation as altitude for a nadir survey: for a facade the
    stand-off plays the role the altitude plays over ground.
    """
    from .cameras import resolve

    if gsd_cm <= 0:
        raise ValueError("A target GSD must be positive.")

    profile, known = resolve(camera)
    distance = profile.altitude_for_gsd_m(gsd_cm)
    if not known:
        # Returned anyway so callers can proceed, but the caller is told.
        return distance
    return distance


def resolve_standoff(
    surface_id: str = "surface",
    *,
    policy: str = "fixed",
    fixed_distance_m: float = 8.0,
    per_surface_m: dict[str, float] | None = None,
    camera: str = "custom",
    target_gsd_cm: float | None = None,
    minimum_clearance_m: float = DEFAULT_MINIMUM_CLEARANCE_M,
    allow_gsd_compromise: bool = True,
) -> StandoffDecision:
    """Choose the stand-off for one surface under the given policy.

    ``allow_gsd_compromise`` decides what happens when the ordered resolution would put
    the aircraft inside the minimum clearance. True clamps to the clearance and records
    the resolution actually achievable; False raises, so a survey specified to a
    tolerance fails loudly rather than silently delivering something coarser.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown stand-off policy {policy!r}. Use one of: "
                         f"{', '.join(POLICIES)}.")

    minimum = max(float(minimum_clearance_m), ABSOLUTE_MINIMUM_CLEARANCE_M)
    warnings: list[str] = []

    if minimum_clearance_m < ABSOLUTE_MINIMUM_CLEARANCE_M:
        warnings.append(
            f"Requested minimum clearance {minimum_clearance_m} m is below the "
            f"{ABSOLUTE_MINIMUM_CLEARANCE_M} m floor; {minimum} m used instead."
        )

    from .cameras import resolve as resolve_camera

    profile, camera_known = resolve_camera(camera)
    if not camera_known and (policy == "adaptive" or target_gsd_cm is not None):
        warnings.append(
            f"Camera {camera!r} is not in the database, so this stand-off is derived "
            "from placeholder sensor geometry and will not deliver the resolution "
            "requested. Add the real camera before relying on it."
        )

    if policy == "fixed":
        distance = float(fixed_distance_m)
    elif policy == "per_surface":
        table = per_surface_m or {}
        if surface_id not in table:
            raise KeyError(
                f"No stand-off defined for surface {surface_id!r}. Per-surface policy "
                f"requires an entry for every surface; have: "
                f"{', '.join(sorted(table)) or 'none'}."
            )
        distance = float(table[surface_id])
    else:  # adaptive
        if target_gsd_cm is None:
            raise ValueError(
                "Adaptive stand-off is derived from a target GSD, so target_gsd_cm "
                "must be supplied."
            )
        distance = standoff_for_gsd(camera, float(target_gsd_cm))

    requested_gsd = float(target_gsd_cm) if target_gsd_cm is not None else None
    clamped = False

    if distance < minimum:
        if not allow_gsd_compromise:
            raise StandoffConflict(
                f"Surface {surface_id!r} needs a stand-off of {distance:.1f} m to reach "
                f"{requested_gsd} cm/px, which is inside the {minimum:.1f} m minimum "
                "clearance. Either accept a coarser resolution, choose a longer lens, "
                "or lower the clearance deliberately."
            )
        warnings.append(
            f"Stand-off raised from {distance:.1f} m to the {minimum:.1f} m minimum "
            "clearance. The requested resolution is not achievable with this camera at "
            "a safe distance."
        )
        distance = minimum
        clamped = True

    achieved = profile.gsd_cm(distance) if distance > 0 else None

    return StandoffDecision(
        surface_id=surface_id, distance_m=distance, policy=policy,
        requested_gsd_cm=requested_gsd, achieved_gsd_cm=achieved,
        clamped=clamped, warnings=warnings,
    )


def _point_to_segment_m(point: Sequence[float],
                        start: Sequence[float],
                        end: Sequence[float]) -> float:
    """Shortest distance from a point to a line segment, in the input's units."""
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])

    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)

    # Projection parameter, clamped to the segment so the nearest point is not off
    # the end of the wall.
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_surface_m(point: Sequence[float],
                          surface: Sequence[Sequence[float]]) -> float:
    """Shortest distance from a point to a polyline or polygon boundary."""
    if len(surface) < 2:
        raise ValueError("A surface needs at least two points.")

    return min(
        _point_to_segment_m(point, surface[i], surface[i + 1])
        for i in range(len(surface) - 1)
    )


def verify_clearance(capture_points: Iterable[Sequence[float]],
                     surfaces: dict[str, Sequence[Sequence[float]]],
                     minimum_clearance_m: float = DEFAULT_MINIMUM_CLEARANCE_M
                     ) -> dict[str, Any]:
    """Check compiled capture points against the surfaces they inspect.

    A policy states an intention; waypoints are what the aircraft flies. This measures
    the second against the first, in projected metres, and names every point that comes
    closer than allowed rather than returning a single pass or fail.
    """
    points = [tuple(p) for p in capture_points]
    minimum = max(float(minimum_clearance_m), ABSOLUTE_MINIMUM_CLEARANCE_M)

    violations: list[ClearanceViolation] = []
    closest_overall = math.inf

    for index, point in enumerate(points):
        for surface_id, surface in surfaces.items():
            distance = distance_to_surface_m(point, surface)
            closest_overall = min(closest_overall, distance)
            if distance < minimum:
                violations.append(ClearanceViolation(
                    point_index=index, surface_id=surface_id,
                    distance_m=distance, required_m=minimum,
                ))

    worst = max((v.shortfall_m for v in violations), default=0.0)

    return {
        "ok": not violations,
        "point_count": len(points),
        "minimum_clearance_m": minimum,
        "closest_approach_m": (round(closest_overall, 2)
                               if closest_overall != math.inf else None),
        "violations": [v.to_dict() for v in violations],
        "violation_count": len(violations),
        "worst_shortfall_m": round(worst, 2),
        "summary": (
            f"All {len(points)} capture points stay at least {minimum} m from every "
            "surface."
            if not violations else
            f"{len(violations)} capture point(s) come closer than {minimum} m, the "
            f"worst by {worst:.1f} m. Fix the plan before flying: these are the points "
            "that would put the aircraft into the structure."
        ),
    }
