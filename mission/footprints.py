"""Facade passes around irregular building footprints.

A rectangular building is four walls and a naive planner handles it by walking the
polygon edges at a standoff. An L-shaped or otherwise concave footprint is where that
breaks, and it breaks quietly: at a concave ("reflex") corner the offset path folds back
through the building, so the planned track passes through the structure it is meant to
photograph. Nothing errors -- the mission uploads, the aircraft flies it, and the
operator finds out at the wall.

So this module does two things the planner cannot do by walking edges:

It finds reflex vertices, where the interior angle exceeds 180 degrees, and reports them
rather than smoothing them away. A footprint with reflex corners needs a different
strategy, and the caller should know that before planning rather than after.

It builds the standoff path segment by segment and drops any part that ends up inside
the footprint, which is what makes an L-shaped plan safe. The inner corner is covered by
the two segments that meet there rather than by an arc through the building.

    from mission.footprints import analyse_footprint, facade_segments
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

Point = tuple[float, float]


class FootprintRefused(ValueError):
    """The footprint cannot be planned around as given."""


# Below this, consecutive vertices are duplicates from a digitising click rather than a
# real corner, and the angle between them is meaningless.
MIN_EDGE_M = 0.25

# A wall shorter than this cannot hold a useful facade pass at any sane standoff.
MIN_WALL_M = 1.0


@dataclass
class Corner:
    index: int
    xy: Point
    interior_angle_deg: float

    @property
    def is_reflex(self) -> bool:
        return self.interior_angle_deg > 180.0


@dataclass
class FootprintAnalysis:
    corners: list[Corner] = field(default_factory=list)
    area_m2: float = 0.0
    is_convex: bool = True
    reflex_count: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "corner_count": len(self.corners),
            "reflex_count": self.reflex_count,
            "is_convex": self.is_convex,
            "area_m2": round(self.area_m2, 2),
            "reflex_corners": [
                {"index": c.index, "xy": [round(v, 3) for v in c.xy],
                 "interior_angle_deg": round(c.interior_angle_deg, 1)}
                for c in self.corners if c.is_reflex
            ],
            "note": self.note,
        }


def _clean(polygon: Sequence[Sequence[float]]) -> list[Point]:
    """Drop the closing repeat and any duplicate vertices."""
    points: list[Point] = [(float(p[0]), float(p[1])) for p in polygon]
    if len(points) > 1 and math.dist(points[0], points[-1]) < 1e-9:
        points = points[:-1]
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or math.dist(cleaned[-1], point) >= MIN_EDGE_M:
            cleaned.append(point)
    if len(cleaned) > 2 and math.dist(cleaned[0], cleaned[-1]) < MIN_EDGE_M:
        cleaned.pop()
    return cleaned


def signed_area(points: Sequence[Point]) -> float:
    """Shoelace. Positive is counter-clockwise."""
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting. Used to reject standoff segments that fall inside the building."""
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing > x:
                inside = not inside
    return inside


def analyse_footprint(polygon: Sequence[Sequence[float]]) -> FootprintAnalysis:
    """Corner angles and convexity, with reflex corners named rather than smoothed."""
    points = _clean(polygon)
    if len(points) < 3:
        raise FootprintRefused(
            f"A footprint needs at least 3 distinct corners; got {len(points)} after "
            f"removing vertices closer than {MIN_EDGE_M} m. If the outline looked "
            "correct on a map, check the units are metres, not degrees: a building in "
            "degrees has vertices millimetres apart and collapses to a point here."
        )

    area = signed_area(points)
    if abs(area) < 1.0:
        raise FootprintRefused(
            f"The footprint encloses {abs(area):.3f} m2, which is too small to plan a "
            "facade mission around. Check the units are metres, not degrees."
        )
    # Work counter-clockwise so the interior-angle test has a consistent sense.
    if area < 0:
        points = list(reversed(points))
        area = -area

    corners: list[Corner] = []
    count = len(points)
    for i in range(count):
        previous, current, following = points[i - 1], points[i], points[(i + 1) % count]
        incoming = math.atan2(current[1] - previous[1], current[0] - previous[0])
        outgoing = math.atan2(following[1] - current[1], following[0] - current[0])
        turn = math.degrees(outgoing - incoming)
        while turn <= -180.0:
            turn += 360.0
        while turn > 180.0:
            turn -= 360.0
        corners.append(Corner(index=i, xy=current, interior_angle_deg=180.0 - turn))

    reflex = sum(1 for c in corners if c.is_reflex)
    convex = reflex == 0
    return FootprintAnalysis(
        corners=corners,
        area_m2=area,
        is_convex=convex,
        reflex_count=reflex,
        note=(
            "Convex footprint: a standoff path around the edges stays outside the "
            "building."
            if convex else
            f"{reflex} reflex corner(s). A naive offset path folds back THROUGH the "
            "building at each of them, so segments falling inside the footprint are "
            "dropped and the inner corner is covered by the two walls that meet there."
        ),
    )


@dataclass
class FacadeSegment:
    wall_index: int
    start: Point
    end: Point
    length_m: float
    heading_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_index": self.wall_index,
            "start": [round(v, 3) for v in self.start],
            "end": [round(v, 3) for v in self.end],
            "length_m": round(self.length_m, 2),
            "heading_deg": round(self.heading_deg, 1),
        }


def facade_segments(
    polygon: Sequence[Sequence[float]],
    *,
    standoff_m: float,
    min_wall_m: float = MIN_WALL_M,
) -> list[FacadeSegment]:
    """One outward-offset pass per wall, with anything inside the building removed.

    Segments are built per wall rather than as a single offset ring. A ring has to
    resolve what happens at each corner, and at a reflex corner every resolution is
    wrong: an arc cuts the building, a mitre shoots off to infinity as the angle
    approaches 180. Per-wall segments simply stop at the corner, and the adjacent wall's
    segment picks up the other face.
    """
    if standoff_m <= 0:
        raise FootprintRefused("Standoff must be a positive distance in metres.")

    points = _clean(polygon)
    if len(points) < 3:
        raise FootprintRefused("A footprint needs at least 3 distinct corners.")
    if signed_area(points) < 0:
        points = list(reversed(points))

    segments: list[FacadeSegment] = []
    count = len(points)
    for index in range(count):
        start, end = points[index], points[(index + 1) % count]
        length = math.dist(start, end)
        if length < min_wall_m:
            # Too short to fly a useful pass along; the neighbouring walls cover it.
            continue
        dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        # Counter-clockwise winding puts the exterior to the right of travel.
        nx, ny = dy, -dx
        offset_start = (start[0] + nx * standoff_m, start[1] + ny * standoff_m)
        offset_end = (end[0] + nx * standoff_m, end[1] + ny * standoff_m)

        # The check that makes concave footprints safe. At a reflex corner the offset
        # of a short wall can land inside the building; flying it would put the aircraft
        # through the structure.
        midpoint = ((offset_start[0] + offset_end[0]) / 2.0,
                    (offset_start[1] + offset_end[1]) / 2.0)
        if (point_in_polygon(offset_start, points)
                or point_in_polygon(offset_end, points)
                or point_in_polygon(midpoint, points)):
            continue

        segments.append(FacadeSegment(
            wall_index=index,
            start=offset_start,
            end=offset_end,
            length_m=length,
            heading_deg=(math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0,
        ))

    if not segments:
        raise FootprintRefused(
            f"No facade pass survives a {standoff_m:g} m standoff on this footprint: "
            "every offset segment falls inside the building. The standoff is larger "
            "than the geometry allows."
        )
    return segments


def courtyard_segments(
    outer: Sequence[Sequence[float]],
    courtyards: Sequence[Sequence[Sequence[float]]],
    *,
    standoff_m: float,
    min_wall_m: float = MIN_WALL_M,
) -> list[FacadeSegment]:
    """Facade passes for a building with interior courtyards.

    A courtyard inverts the problem. On the outer wall the aircraft stands off
    *outward*, away from the building. Inside a courtyard the building surrounds the
    aircraft, so the standoff goes *inward*, toward the courtyard's centre -- offsetting
    the same direction as the outer wall would fly straight into the masonry.

    That inversion is why a courtyard cannot be handled by feeding the inner ring to the
    outer-wall planner. The two rings need opposite offsets, and a planner that treats
    them alike produces a mission that looks complete and flies into a wall.

    Courtyard passes are also checked against the outer footprint: a standoff larger than
    the courtyard is wide puts the aircraft inside the building on the far side.
    """
    segments = facade_segments(outer, standoff_m=standoff_m, min_wall_m=min_wall_m)

    outer_points = _clean(outer)
    if signed_area(outer_points) < 0:
        outer_points = list(reversed(outer_points))

    for court_index, courtyard in enumerate(courtyards):
        points = _clean(courtyard)
        if len(points) < 3:
            raise FootprintRefused(
                f"Courtyard {court_index} has fewer than 3 distinct corners."
            )
        # Reversed relative to the outer ring, so the same right-of-travel rule puts the
        # offset into the open courtyard rather than into the building.
        if signed_area(points) > 0:
            points = list(reversed(points))

        found = 0
        count = len(points)
        for index in range(count):
            start, end = points[index], points[(index + 1) % count]
            length = math.dist(start, end)
            if length < min_wall_m:
                continue
            dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
            nx, ny = dy, -dx
            # Inset along the wall as well as away from it. Offsetting only outward puts
            # the segment ends exactly on the corner, where they belong to both walls and
            # sit on the courtyard boundary; the aircraft turns before the corner in any
            # case, so pulling the ends in matches how the pass is actually flown.
            inset = min(standoff_m, length / 3.0)
            offset_start = (start[0] + nx * standoff_m + dx * inset,
                            start[1] + ny * standoff_m + dy * inset)
            offset_end = (end[0] + nx * standoff_m - dx * inset,
                          end[1] + ny * standoff_m - dy * inset)
            midpoint = ((offset_start[0] + offset_end[0]) / 2.0,
                        (offset_start[1] + offset_end[1]) / 2.0)

            # Must be inside the courtyard void: within the outer building outline but
            # not inside the courtyard walls themselves.
            inside_courtyard = all(
                point_in_polygon(p, points) for p in (offset_start, offset_end, midpoint)
            )
            within_building = all(
                point_in_polygon(p, outer_points)
                for p in (offset_start, offset_end, midpoint)
            )
            if not (inside_courtyard and within_building):
                continue

            segments.append(FacadeSegment(
                wall_index=1000 * (court_index + 1) + index,
                start=offset_start,
                end=offset_end,
                length_m=length,
                heading_deg=(math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0,
            ))
            found += 1

        if found == 0:
            raise FootprintRefused(
                f"No pass fits inside courtyard {court_index} at a {standoff_m:g} m "
                "standoff: the courtyard is narrower than twice the standoff, so the "
                "aircraft would be in the far wall. Reduce the standoff or fly it as a "
                "vertical descent rather than a facade sweep."
            )
    return segments


@dataclass
class OcclusionReport:
    """Whether a single standoff sweep can actually see the whole facade."""

    overhang_depth_m: float
    standoff_m: float
    occluded: bool
    recommended_extra_passes: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overhang_depth_m": round(self.overhang_depth_m, 2),
            "standoff_m": round(self.standoff_m, 2),
            "occluded": self.occluded,
            "recommended_extra_passes": self.recommended_extra_passes,
            "note": self.note,
        }


def assess_occlusion(
    overhang_depth_m: float,
    *,
    standoff_m: float,
    camera_fov_deg: float = 78.0,
) -> OcclusionReport:
    """Whether a balcony or overhang hides facade a nadir-ish sweep will never see.

    A single vertical sweep at a fixed standoff photographs the outermost surface. Under
    a balcony, a recessed window reveal, or a projecting cornice, the wall behind it is
    simply not in view -- and the resulting model shows a smooth surface there rather
    than a hole, so the gap does not announce itself.

    Reported as extra oblique passes needed, not silently planned around, because the
    caller may reasonably decide the recess does not matter for their inspection.
    """
    if standoff_m <= 0:
        raise FootprintRefused("Standoff must be a positive distance in metres.")
    if overhang_depth_m < 0:
        raise FootprintRefused("Overhang depth cannot be negative.")

    # How far under a projection the camera can see from a level sweep, given half the
    # field of view. Deeper than this and the surface behind is hidden.
    half_fov = math.radians(max(1.0, min(camera_fov_deg, 179.0)) / 2.0)
    visible_depth = standoff_m * math.tan(half_fov)
    occluded = overhang_depth_m > visible_depth

    if not occluded:
        note = (
            f"An overhang of {overhang_depth_m:.2f} m is within the {visible_depth:.2f} m "
            f"a {camera_fov_deg:g} degree lens sees under a projection at {standoff_m:g} m "
            "standoff. One sweep covers it."
        )
        extra = 0
    else:
        extra = max(1, int(math.ceil(overhang_depth_m / max(visible_depth, 0.1))))
        note = (
            f"An overhang of {overhang_depth_m:.2f} m exceeds the {visible_depth:.2f} m "
            f"visible under a projection at {standoff_m:g} m standoff. The wall behind it "
            f"will not be photographed, and the reconstruction will show a smooth "
            f"surface there rather than a gap -- the omission does not announce itself. "
            f"Add about {extra} oblique pass(es) angled up under the projection, or "
            "reduce the standoff."
        )
    return OcclusionReport(
        overhang_depth_m=float(overhang_depth_m),
        standoff_m=float(standoff_m),
        occluded=occluded,
        recommended_extra_passes=extra,
        note=note,
    )
