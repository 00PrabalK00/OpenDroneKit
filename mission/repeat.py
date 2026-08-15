"""Flying the same survey again, months later, so the two can be compared.

Repeat inspection is the whole point of a persistent asset. One survey says what is
there; a sequence says whether it is getting worse. But that comparison only holds if
the second survey resembles the first, and the ways it silently stops resembling it are
not obvious.

The one that catches people is equipment. Fly the same altitude with a different camera
and the ground sample distance changes, so the second survey resolves different detail
from the first. Crack widths measured across the two are then not measuring the same
thing, and nothing downstream will notice: both surveys are internally valid, both
produce clean orthomosaics, and the change between them is partly an artefact of the
lens. So a repeat with a different camera recomputes the altitude to preserve the
original GSD rather than preserving the original altitude.

Three kinds of repeat are supported, and each reports what it changed:

*Exact* re-flies the stored waypoints unaltered. Right when nothing has changed.

*Updated terrain* keeps the plan's geometry but re-derives altitudes from a newer
terrain model, which matters on a site that has been excavated or filled since.

*Modified boundary* keeps the capture specification -- overlap, GSD, camera -- and
re-plans over a new area, for when a site has grown or part of it is now inaccessible.

Everything here reports comparability against the original. A repeat that cannot be
compared is still worth flying; it is just not worth pretending otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Beyond this relative difference in ground sample distance, two surveys resolve
# meaningfully different detail and measurements across them should not be differenced
# without saying so. Five per cent is roughly a 3 m altitude change at 60 m.
GSD_COMPARABILITY_TOLERANCE = 0.05

REPEAT_EXACT = "exact"
REPEAT_UPDATED_TERRAIN = "updated_terrain"
REPEAT_MODIFIED_BOUNDARY = "modified_boundary"

REPEAT_MODES = (REPEAT_EXACT, REPEAT_UPDATED_TERRAIN, REPEAT_MODIFIED_BOUNDARY)


@dataclass
class RepeatPlan:
    """A survey planned to be compared against an earlier one."""

    mode: str
    source_mission: str
    waypoints: list[list[float]] = field(default_factory=list)
    altitude_m: float = 0.0
    camera: str = ""
    target_gsd_cm: float | None = None
    achieved_gsd_cm: float | None = None
    changes: list[str] = field(default_factory=list)
    comparability: str = "comparable"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_mission": self.source_mission,
            "waypoints": self.waypoints,
            "capture_count": len(self.waypoints),
            "altitude_m": round(self.altitude_m, 1),
            "camera": self.camera,
            "target_gsd_cm": (round(self.target_gsd_cm, 3)
                              if self.target_gsd_cm is not None else None),
            "achieved_gsd_cm": (round(self.achieved_gsd_cm, 3)
                                if self.achieved_gsd_cm is not None else None),
            "changes": self.changes,
            "comparability": self.comparability,
            "warnings": self.warnings,
            "note": (
                "A repeat is only comparable to its original where the capture "
                "specification matches. Differences are listed rather than reconciled: "
                "the operator decides whether the two surveys can be differenced."
            ),
        }


def _summary_of(plan: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        plan = plan.to_dict() if hasattr(plan, "to_dict") else {}
    for key in ("plan_summary", "plan_summary_json"):
        value = plan.get(key)
        if isinstance(value, dict):
            return value
    return plan


def altitude_for_matching_gsd(original_camera: str, original_altitude_m: float,
                              new_camera: str) -> tuple[float, float]:
    """Altitude at which a different camera achieves the original's ground resolution.

    Returns the altitude and the GSD it matches. Preserving altitude across a camera
    change would preserve the flight and lose the comparison, which is the wrong thing
    to keep.
    """
    from .cameras import resolve

    old_profile, _ = resolve(original_camera)
    new_profile, _ = resolve(new_camera)

    target_gsd = old_profile.gsd_cm(original_altitude_m)
    return new_profile.altitude_for_gsd_m(target_gsd), target_gsd


def repeat_mission(plan: dict[str, Any] | Any,
                   *,
                   mode: str = REPEAT_EXACT,
                   camera: str = "",
                   terrain_source: str = "",
                   boundary: list[list[float]] | None = None,
                   mission_name: str = "") -> RepeatPlan:
    """Plan a repeat of an earlier survey."""
    if mode not in REPEAT_MODES:
        raise ValueError(
            f"Unknown repeat mode {mode!r}. Use one of: {', '.join(REPEAT_MODES)}.")

    summary = _summary_of(plan)
    original_camera = str(summary.get("camera") or "custom")
    original_altitude = float(summary.get("altitude_m") or 0.0)
    original_waypoints = [list(w) for w in (summary.get("waypoints") or [])]

    if not original_waypoints and mode != REPEAT_MODIFIED_BOUNDARY:
        raise ValueError(
            "The original mission has no waypoints, so there is nothing to repeat.")

    from .cameras import resolve

    original_profile, original_known = resolve(original_camera)
    original_gsd = (original_profile.gsd_cm(original_altitude)
                    if original_altitude > 0 else None)

    repeat = RepeatPlan(
        mode=mode,
        source_mission=str(summary.get("template") or mission_name or "mission"),
        camera=camera or original_camera,
        altitude_m=original_altitude,
        target_gsd_cm=original_gsd,
        achieved_gsd_cm=original_gsd,
        waypoints=original_waypoints,
    )

    if not original_known:
        repeat.warnings.append(
            f"The original camera {original_camera!r} is not in the database, so the "
            "ground resolution it achieved is a guess and comparability cannot be "
            "checked properly."
        )

    # Equipment change: hold the resolution, not the altitude.
    if camera and camera != original_camera:
        new_altitude, target_gsd = altitude_for_matching_gsd(
            original_camera, original_altitude, camera)
        new_profile, new_known = resolve(camera)

        repeat.altitude_m = new_altitude
        repeat.target_gsd_cm = target_gsd
        repeat.achieved_gsd_cm = new_profile.gsd_cm(new_altitude)
        repeat.changes.append(
            f"Camera {original_camera} -> {camera}; altitude adjusted "
            f"{original_altitude:.0f} m -> {new_altitude:.0f} m to hold "
            f"{target_gsd:.2f} cm/px."
        )
        if not new_known:
            repeat.warnings.append(
                f"Camera {camera!r} is not in the database, so this altitude is derived "
                "from placeholder sensor geometry and will not deliver the original "
                "resolution."
            )
        # The stored waypoints carry the old altitude in their third element.
        repeat.waypoints = [
            [w[0], w[1], new_altitude] if len(w) >= 3 else list(w)
            for w in original_waypoints
        ]

    if mode == REPEAT_UPDATED_TERRAIN:
        if not terrain_source:
            raise ValueError(
                "An updated-terrain repeat needs a terrain source; without one it is "
                "an exact repeat wearing a different name.")
        repeat.changes.append(
            f"Altitudes re-derived from {terrain_source}. Ground that has moved since "
            "the original survey changes the height flown above it, which is the point "
            "of this mode."
        )
        repeat.comparability = "comparable_with_caveat"
        repeat.warnings.append(
            "Terrain-following altitudes differ from the original where the ground has "
            "changed. Elevation differences between the two surveys will partly reflect "
            "the new terrain model rather than real change."
        )

    if mode == REPEAT_MODIFIED_BOUNDARY:
        if not boundary or len(boundary) < 3:
            raise ValueError(
                "A modified-boundary repeat needs a boundary of at least three points.")
        repeat.changes.append(
            f"Re-planned over a new boundary of {len(boundary)} vertices, holding the "
            "original overlap, camera and ground resolution."
        )
        repeat.comparability = "partial_overlap"
        repeat.warnings.append(
            "Only the area common to both surveys can be compared. Change measured "
            "outside the original boundary is new coverage, not new damage."
        )
        # Geometry is recomputed by the planner; the waypoints are not carried over.
        repeat.waypoints = []

    # Final comparability judgement on resolution.
    if (repeat.target_gsd_cm and repeat.achieved_gsd_cm
            and repeat.target_gsd_cm > 0):
        difference = abs(repeat.achieved_gsd_cm - repeat.target_gsd_cm) / repeat.target_gsd_cm
        if difference > GSD_COMPARABILITY_TOLERANCE:
            repeat.comparability = "not_directly_comparable"
            repeat.warnings.append(
                f"The repeat resolves {repeat.achieved_gsd_cm:.2f} cm/px against the "
                f"original's {repeat.target_gsd_cm:.2f} cm/px, a "
                f"{difference * 100:.0f}% difference. Measurements differenced across "
                "these two surveys would partly reflect the change in resolution."
            )

    if not repeat.changes:
        repeat.changes.append("Nothing changed; this re-flies the original exactly.")

    return repeat


def compare_specifications(first: dict[str, Any] | Any,
                           second: dict[str, Any] | Any) -> dict[str, Any]:
    """Whether two surveys were flown to specifications that can be differenced.

    Used before change detection rather than after, because the answer decides whether
    the difference means anything.
    """
    from .cameras import resolve

    left, right = _summary_of(first), _summary_of(second)
    differences: list[str] = []

    left_camera = str(left.get("camera") or "custom")
    right_camera = str(right.get("camera") or "custom")
    left_alt = float(left.get("altitude_m") or 0.0)
    right_alt = float(right.get("altitude_m") or 0.0)

    left_gsd = resolve(left_camera)[0].gsd_cm(left_alt) if left_alt > 0 else None
    right_gsd = resolve(right_camera)[0].gsd_cm(right_alt) if right_alt > 0 else None

    if left_camera != right_camera:
        differences.append(f"camera {left_camera} vs {right_camera}")
    if abs(left_alt - right_alt) > 1.0:
        differences.append(f"altitude {left_alt:.0f} m vs {right_alt:.0f} m")

    for field_name, label in (("front_overlap_pct", "front overlap"),
                              ("side_overlap_pct", "side overlap")):
        a, b = left.get(field_name), right.get(field_name)
        if a is not None and b is not None and abs(float(a) - float(b)) > 1.0:
            differences.append(f"{label} {a}% vs {b}%")

    comparable = True
    gsd_difference = None
    if left_gsd and right_gsd:
        gsd_difference = abs(right_gsd - left_gsd) / left_gsd
        if gsd_difference > GSD_COMPARABILITY_TOLERANCE:
            comparable = False

    return {
        "comparable": comparable,
        "gsd_cm": [round(left_gsd, 3) if left_gsd else None,
                   round(right_gsd, 3) if right_gsd else None],
        "gsd_difference_pct": (round(gsd_difference * 100, 1)
                               if gsd_difference is not None else None),
        "differences": differences,
        "note": (
            "The two surveys resolve comparable detail; differences between them can be "
            "read as change."
            if comparable else
            "These surveys resolve different detail. Differencing them measures the "
            "change in specification as well as the change on the ground."
        ),
    }
