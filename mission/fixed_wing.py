"""What a fixed-wing aircraft needs from a mission that a multirotor plan assumes away.

A multirotor can stop, turn on the spot and land where it is. A plane can do none of
these, and the survey templates were written for multirotors: a grid flew its lines
side by side at the camera's line spacing (tens of metres), which a plane with a 60 m
turn radius cannot turn between, and every mission ended in RETURN_TO_LAUNCH, which on
ArduPlane circles home forever instead of landing.

Checked against a real ArduPlane (Plane-4.5.7 SITL, tests/sitl/test_plane_missions.py).
This module supplies the three things a plane needs:

* Pass ordering: the straight passes of a grid or corridor are flown in an order in
  which consecutive passes are at least one turn diameter apart, so every turn between
  them is a turn the aircraft can make (a "skip" or racetrack pattern).
* Lead-in and lead-out: each pass is extended at both ends (five turn radii before,
  one after, flown to) with points that do not trigger the camera, so the aircraft is
  out of its turn, level and on the line for the first photograph and still level for
  the last.
* A landing sequence: DO_LAND_START, a descending loiter on the approach centreline,
  and NAV_LAND at the landing point -- the pattern QGroundControl and Mission Planner
  generate for ArduPlane. The same DO_LAND_START is what an ArduPlane RTL or a GCS
  "land" jumps to.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

import numpy as np

MULTIROTOR = "multirotor"
FIXED_WING = "fixed_wing"
VEHICLE_CLASSES = (MULTIROTOR, FIXED_WING)

# ArduPlane's WP_LOITER_RAD default. A plane at a 22 m/s cruise and a 45 degree bank
# turns on a ~50 m radius; the default leaves margin for wind.
DEFAULT_TURN_RADIUS_M = 60.0
# How far before a pass its lead-in point sits, in turn radii. ArduPlane comes out of
# the turn between passes up to ~65 m off the next line and its L1 controller takes a
# few hundred metres to settle on it. At one radius the first photographs of every pass
# were 30-68 m from where they were planned; at three they were on position (within
# 6.5 m) but the first six of each pass were taken in a 5-12 degree bank, tilting a
# nadir footprint 7-17 m sideways at 80 m (Plane-4.5.7 SITL, 60 m radius). Five radii
# (300 m) is where the bank had settled below 2 degrees in those flights.
LEAD_IN_RADII = 5.0
# The lead-out: one radius past the last capture, flown to rather than cut. ArduPlane
# accepts a waypoint before a large turn up to WP_RADIUS (90 m) early, so it began the
# turn before the last capture point and shot it in a 12-23 degree bank; the lead-out
# carries an explicit 1 m acceptance radius (NAV_WAYPOINT param2) so the turn starts
# after it. The fence is widened by a turn radius to contain that turn.
LEAD_OUT_RADII = 1.0
LEAD_OUT_ACCEPTANCE_M = 1.0

# Final approach: the descending loiter ends this high, this far from touchdown.
DEFAULT_APPROACH_ALTITUDE_M = 30.0
MIN_APPROACH_DISTANCE_M = 300.0

# Templates a plane can fly as planned: straight passes over an area or along a route.
# The rest hover, hold a heading at a point, or circle a structure inside a turn radius
# (facade, orbit, tower, panorama, ...), which a fixed-wing cannot do.
FIXED_WING_TEMPLATES = frozenset({"grid", "double_grid", "corridor", "waypoints"})

# MAV_TYPE values from a HEARTBEAT that are fixed-wing aircraft without a hover mode.
MAV_TYPE_FIXED_WING = 1


def normalise_vehicle_class(value: Any) -> str:
    text = str(value or MULTIROTOR).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"plane", "fixedwing", "fixed_wing", "airplane", "aeroplane"}:
        return FIXED_WING
    if text in {"multirotor", "copter", "multicopter", "quad", "quadcopter", "rotorcraft"}:
        return MULTIROTOR
    raise ValueError(f"Unknown vehicle class {value!r}; use 'multirotor' or 'fixed_wing'.")


def vehicle_class_from_mav_type(mav_type: int) -> str:
    return FIXED_WING if int(mav_type) == MAV_TYPE_FIXED_WING else MULTIROTOR


# ── Pass ordering and lead-in/out (planner, local metres) ────────────────────

def _passes(poses: Sequence[Any]) -> list[list[Any]]:
    """Split poses into straight passes: runs with one primitive and one heading."""
    passes: list[list[Any]] = []
    for pose in poses:
        if passes:
            last = passes[-1][-1]
            same_heading = abs((float(pose.yaw_deg) - float(last.yaw_deg) + 180.0) % 360.0 - 180.0) < 1.0
            if pose.primitive == last.primitive and same_heading:
                passes[-1].append(pose)
                continue
        passes.append([pose])
    return passes


def _xy(pose: Any) -> np.ndarray:
    return np.array([float(pose.x_m), float(pose.y_m)], dtype=np.float64)


def _straighten(run: list[Any]) -> tuple[list[Any], float]:
    """Put every point of a pass on one straight line; returns the run and the largest move.

    The planner pulls a pass's corner point a few metres inside the area to keep it in
    the fence (a grid row ending on a field corner came back 6 m short and 4.5 m to the
    side). A multirotor flies that kink; a plane banks for it, and ArduPlane shot the
    last photo of such passes in a 12-23 degree bank (Plane-4.5.7 SITL). The line is the
    median direction of the pass's legs through the median offset of its points, so
    one displaced point cannot tilt it.
    """
    if len(run) < 3:
        return run, 0.0
    pts = np.array([_xy(p) for p in run])
    legs = np.diff(pts, axis=0)
    norms = np.linalg.norm(legs, axis=1)
    units = legs[norms > 1e-6] / norms[norms > 1e-6, None]
    if len(units) == 0:
        return run, 0.0
    units = np.array([u if float(u @ units[0]) >= 0.0 else -u for u in units])
    direction = np.median(units, axis=0)
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    normal = np.array([-direction[1], direction[0]])
    offsets = pts @ normal
    line = float(np.median(offsets))
    moved = offsets - line
    straight = [replace(p, x_m=float(xy[0]), y_m=float(xy[1]))
                for p, xy in zip(run, pts - moved[:, None] * normal)]
    return straight, float(np.max(np.abs(moved)))


def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(np.degrees(np.arctan2(v[1], v[0]))) if float(np.linalg.norm(v)) > 1e-9 else 0.0


def _pass_order(offsets: Sequence[float], min_gap: float, budget: int = 20_000) -> list[int]:
    """An order visiting every pass where each step moves at least `min_gap` sideways.

    Depth-first, preferring the nearest pass that is far enough (the shortest transit),
    starting from the planner's first pass and then from the other edge. A greedy walk
    strands passes -- 14 lines at 31 m with a 120 m minimum left two tight turns -- where
    an order without any exists (0, 4, 8, 12, 1, 5, ...). When the search budget runs out
    or no such order exists, the greedy order is used and its tight turns are counted.
    """
    n = len(offsets)
    if n <= 1:
        return list(range(n))
    ranked = sorted(range(n), key=lambda i: offsets[i])
    starts = [0] + [i for i in (ranked[0], ranked[-1]) if i != 0]
    expansions = 0

    def search(path: list[int], left: set[int]) -> list[int] | None:
        nonlocal expansions
        if not left:
            return path
        expansions += 1
        if expansions > budget:
            return None
        here = offsets[path[-1]]
        for nxt in sorted((i for i in left if abs(offsets[i] - here) >= min_gap - 1e-6),
                          key=lambda i: (abs(offsets[i] - here), i)):
            found = search(path + [nxt], left - {nxt})
            if found is not None:
                return found
        return None

    import sys
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, n + 100))
    try:
        for start in starts:
            expansions = 0
            found = search([start], set(range(n)) - {start})
            if found is not None:
                return found
    finally:
        sys.setrecursionlimit(limit)

    order, left = [0], set(range(1, n))
    while left:
        here = offsets[order[-1]]
        wide = [i for i in left if abs(offsets[i] - here) >= min_gap - 1e-6]
        nxt = min(wide or left, key=lambda i: (abs(offsets[i] - here), i))
        order.append(nxt)
        left.remove(nxt)
    return order


def arrange_passes(poses: Sequence[Any], turn_radius_m: float) -> tuple[list[Any], dict[str, Any]]:
    """Reorder and extend straight passes so a plane with `turn_radius_m` can fly them.

    `poses` are the planner's local capture poses (x/y metres, yaw in the planner's
    convention). Passes of one primitive are parallel; within each primitive they are
    flown in an order where each next pass is at least 2 * turn radius to the side of
    the one before, choosing the nearest such pass. When no remaining pass is that far
    away the nearest one is taken and counted in ``tight_turns``: the aircraft flies a
    loop to make it, which is legal but slower, and the report says so.
    """
    radius = float(max(0.0, turn_radius_m))
    report: dict[str, Any] = {"turn_radius_m": radius, "passes": 0, "tight_turns": 0,
                              "min_turn_spacing_m": None, "passes_straightened": 0,
                              "max_straighten_m": 0.0}
    if radius <= 0.0 or not poses:
        return list(poses), report
    groups: list[list[list[Any]]] = []
    for run in _passes(poses):
        run, moved = _straighten(run)
        if moved > 0.05:
            report["passes_straightened"] += 1
            report["max_straighten_m"] = round(max(report["max_straighten_m"], moved), 2)
        if groups and groups[-1][0][0].primitive == run[0].primitive:
            groups[-1].append(run)
        else:
            groups.append([run])

    out: list[Any] = []
    spacings: list[float] = []
    exit_xy: np.ndarray | None = None
    for runs in groups:
        long_runs = [r for r in runs if len(r) >= 2]
        if not long_runs:
            out.extend(p for r in runs for p in r)
            continue
        axis = _xy(long_runs[0][-1]) - _xy(long_runs[0][0])
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        normal = np.array([-axis[1], axis[0]])
        offsets = [float(np.mean([_xy(p) @ normal for p in r])) for r in runs]
        order = _pass_order(offsets, 2.0 * radius)
        for a, b in zip(order, order[1:]):
            gap = abs(offsets[b] - offsets[a])
            spacings.append(gap)
            if gap < 2.0 * radius - 1e-6:
                report["tight_turns"] += 1
        for index in order:
            run = runs[index]
            if exit_xy is not None and len(run) >= 2:
                if float(np.linalg.norm(_xy(run[-1]) - exit_xy)) < float(np.linalg.norm(_xy(run[0]) - exit_xy)):
                    run = list(reversed(run))
            first, last = _xy(run[0]), _xy(run[-1])
            direction = last - first
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                out.extend(run)
                exit_xy = last
                continue
            direction /= length
            yaw = _bearing(first, last)
            run = [replace(p, yaw_deg=yaw) for p in run]
            lead_in = first - direction * LEAD_IN_RADII * radius
            lead_out = last + direction * LEAD_OUT_RADII * radius
            out.append(replace(run[0], x_m=float(lead_in[0]), y_m=float(lead_in[1]),
                               trigger=False, dwell_s=0.0, primitive=f"{run[0].primitive}"))
            out.extend(run)
            out.append(replace(run[-1], x_m=float(lead_out[0]), y_m=float(lead_out[1]),
                               trigger=False, dwell_s=0.0))
            report["passes"] += 1
            exit_xy = lead_out
    if spacings:
        report["min_turn_spacing_m"] = round(min(spacings), 2)
    return out, report


# ── Landing sequence (world coordinates) ─────────────────────────────────────

def _offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    return (lat + north_m / 110_540.0,
            lon + east_m / (111_320.0 * math.cos(math.radians(lat))))


def landing_pattern(land_lat: float, land_lon: float, heading_deg: float, *,
                    turn_radius_m: float = DEFAULT_TURN_RADIUS_M,
                    approach_altitude_m: float = DEFAULT_APPROACH_ALTITUDE_M) -> dict[str, Any]:
    """Where a plane descends and lines up to touch down at a point on `heading_deg`.

    The descending loiter is centred on the extended centreline, behind the touchdown
    point, at a distance that gives a shallow final glide (about 1:10) and room to leave
    the circle on the centreline.
    """
    radius = float(max(20.0, turn_radius_m))
    altitude = float(max(10.0, approach_altitude_m))
    distance = float(max(MIN_APPROACH_DISTANCE_M, 10.0 * altitude, 3.0 * radius))
    heading = float(heading_deg) % 360.0
    back = math.radians(heading + 180.0)
    loiter_lat, loiter_lon = _offset(land_lat, land_lon,
                                     distance * math.cos(back), distance * math.sin(back))
    return {
        "land_lat": float(land_lat), "land_lon": float(land_lon),
        "heading_deg": round(heading, 2),
        "loiter_lat": loiter_lat, "loiter_lon": loiter_lon,
        "loiter_radius_m": radius, "approach_altitude_m": altitude,
        "approach_distance_m": distance,
    }


def landing_outline(pattern: dict[str, Any], samples: int = 16) -> list[list[float]]:
    """[lon, lat] points covering the landing pattern: touchdown and the loiter circle."""
    points = [[pattern["land_lon"], pattern["land_lat"]]]
    for k in range(samples):
        angle = 2.0 * math.pi * k / samples
        lat, lon = _offset(pattern["loiter_lat"], pattern["loiter_lon"],
                           pattern["loiter_radius_m"] * math.cos(angle),
                           pattern["loiter_radius_m"] * math.sin(angle))
        points.append([lon, lat])
    return points


def convex_hull(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Counter-clockwise hull of [x, y] points (Andrew's monotone chain)."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) < 3:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]
