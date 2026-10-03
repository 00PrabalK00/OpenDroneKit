"""What the Plan screen shows about the mission on screen, read from the plan itself.

`plan_mission` answers with a summary once, at planning time; a screen that is opened
later, re-rendered, or switched to Field mode had nothing to read that summary back
from, and the waypoint inspector, the altitude profile and the photo footprints had no
source at all. This module turns the session's plan into that view, from the plan's own
numbers:

  * the per-waypoint table (position, altitude, yaw, gimbal, trigger, dwell), with each
    waypoint's distance along the route and the straight line (leg) it sits on;
  * the camera's ground footprint at each nadir capture point, from the camera profile
    and the height above ground -- a planned footprint, labelled as one;
  * the ground elevation under each waypoint when a terrain model is loaded, and
    nothing (None, with the reason) when it is not;
  * the safety adjustments and terrain model the planner reported.

Nothing is estimated that the plan does not say. A figure this cannot derive is None.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

# Two consecutive moves whose bearings differ by more than this start a new leg. A grid
# turns through ~90-180 degrees at the end of each line; an orbit through a few degrees
# per point, so an orbit stays one leg and a grid has one per line.
LEG_TURN_DEG = 25.0

# A gimbal within this of straight down is a nadir photograph, whose footprint is the
# camera's rectangle on the ground. An oblique photograph covers a trapezoid this does
# not model, so it gets no footprint rather than the wrong one.
NADIR_TOLERANCE_DEG = 10.0


def _bearing_deg(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Initial great-circle bearing from a to b (lon/lat degrees), or None if equal."""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    if abs(x) < 1e-15 and abs(y) < 1e-15:
        return None
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _ground_m(a: Sequence[float], b: Sequence[float]) -> float:
    """Ground distance in metres (haversine, WGS84 mean radius)."""
    r = 6371008.8
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _turn(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _offset(lon: float, lat: float, east_m: float, north_m: float) -> list[float]:
    """A point east/north metres from lon/lat (local tangent plane; footprint scale)."""
    per_lat = 111_320.0
    per_lon = per_lat * math.cos(math.radians(lat)) or 1e-9
    return [lon + east_m / per_lon, lat + north_m / per_lat]


def footprint_ring(lon: float, lat: float, heading_deg: float, along_m: float,
                   across_m: float) -> list[list[float]]:
    """The closed lon/lat ring of an along x across rectangle centred on a point."""
    h = math.radians(heading_deg)
    ux, uy = math.sin(h), math.cos(h)          # along the heading (east, north)
    vx, vy = math.cos(h), -math.sin(h)         # across it, to the right
    corners = []
    for sa, sc in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        e = sa * ux * along_m / 2 + sc * vx * across_m / 2
        n = sa * uy * along_m / 2 + sc * vy * across_m / 2
        corners.append(_offset(lon, lat, e, n))
    corners.append(list(corners[0]))
    return corners


def waypoint_features(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The plan's waypoint features, ordered by index."""
    features = ((plan.get("geojson") or {}).get("features") or [])
    points = [f for f in features
              if (f.get("properties") or {}).get("role") == "waypoint"
              and (f.get("geometry") or {}).get("type") == "Point"]
    return sorted(points, key=lambda f: int((f.get("properties") or {}).get("index", 0) or 0))


def overview(plan: dict[str, Any], *,
             footprint_at: Callable[[float], tuple[float, float] | None] | None = None,
             ground_at: Callable[[float, float], float | None] | None = None,
             home_ground_m: float | None = None) -> dict[str, Any]:
    """The Plan screen's view of `plan` (the session's mission_plan_dict).

    footprint_at(height_m) -> (across_m, along_m) for the plan's camera, or None.
    ground_at(lon, lat) -> ground elevation in metres from the loaded terrain, or None.
    home_ground_m: the ground elevation at the take-off point, when terrain is loaded.
    """
    features = waypoint_features(plan)
    if features:
        coords = [list(map(float, f["geometry"]["coordinates"])) for f in features]
    else:
        coords = [list(map(float, p)) for p in (plan.get("waypoints") or []) if len(p) >= 2]
        features = [{"properties": {"index": i}} for i in range(len(coords))]

    frame = str(plan.get("altitude_frame") or "relative_to_home")
    default_alt = plan.get("altitude_m")
    rows: list[dict[str, Any]] = []
    leg = 0
    previous_bearing: float | None = None
    along = 0.0
    for i, (point, feature) in enumerate(zip(coords, features)):
        props = feature.get("properties") or {}
        if i:
            step = _ground_m(coords[i - 1], point)
            along += step
            bearing = _bearing_deg(coords[i - 1], point)
            if bearing is not None:
                if previous_bearing is not None and _turn(bearing, previous_bearing) > LEG_TURN_DEG:
                    leg += 1
                previous_bearing = bearing
        alt = point[2] if len(point) > 2 else default_alt
        ground = ground_at(point[0], point[1]) if ground_at else None
        rows.append({
            "index": int(props.get("index", i)),
            "lon": point[0],
            "lat": point[1],
            "alt_m": None if alt is None else float(alt),
            "yaw_deg": props.get("yaw_deg"),
            "gimbal_deg": props.get("gimbal_pitch_deg", plan.get("gimbal_tilt_deg")),
            "trigger": bool(props.get("trigger", True)),
            "dwell_s": float(props.get("dwell_s", 0.0) or 0.0),
            "primitive": props.get("primitive") or "",
            "leg": leg,
            "along_m": round(along, 2),
            "ground_m": ground,
        })

    # Legs that carry a photograph: the survey lines, not the turns between them.
    leg_triggers: dict[int, int] = {}
    for row in rows:
        if row["trigger"]:
            leg_triggers[row["leg"]] = leg_triggers.get(row["leg"], 0) + 1
    lines = sorted(k for k, n in leg_triggers.items() if n >= 2)
    line_of = {k: n + 1 for n, k in enumerate(lines)}
    for row in rows:
        row["line"] = line_of.get(row["leg"])

    # Footprints: nadir photographs only, at the height above the ground under them.
    footprints = []
    footprint_basis = None
    if footprint_at is not None and rows:
        cache: dict[float, tuple[float, float] | None] = {}
        for i, row in enumerate(rows):
            gimbal = row["gimbal_deg"]
            if not row["trigger"] or gimbal is None or abs(float(gimbal) + 90.0) > NADIR_TOLERANCE_DEG:
                continue
            if row["alt_m"] is None:
                continue
            height = row["alt_m"]
            if frame == "relative_to_home" and row["ground_m"] is not None and home_ground_m is not None:
                height = row["alt_m"] + home_ground_m - row["ground_m"]
                footprint_basis = "height above the terrain model under each point"
            elif frame == "amsl" and row["ground_m"] is not None:
                height = row["alt_m"] - row["ground_m"]
                footprint_basis = "height above the terrain model under each point"
            if height <= 0:
                continue
            key = round(height, 1)
            if key not in cache:
                cache[key] = footprint_at(key)
            size = cache[key]
            if not size:
                continue
            across, along_m = size
            heading = row["yaw_deg"]
            if heading is None:
                nxt = rows[i + 1] if i + 1 < len(rows) else None
                prv = rows[i - 1] if i else None
                a, b = (row, nxt) if nxt else (prv, row)
                heading = _bearing_deg([a["lon"], a["lat"]], [b["lon"], b["lat"]]) if a and b else 0.0
            row["footprint_m"] = [round(across, 1), round(along_m, 1)]
            footprints.append({
                "type": "Feature",
                "properties": {"index": row["index"], "role": "footprint",
                               "across_m": round(across, 1), "along_m": round(along_m, 1)},
                "geometry": {"type": "Polygon",
                             "coordinates": [footprint_ring(row["lon"], row["lat"],
                                                            float(heading or 0.0), along_m, across)]},
            })
        if footprints and footprint_basis is None:
            footprint_basis = ("height above the take-off point: no terrain model is loaded, "
                               "so the ground is taken as flat")

    # The compass bearing of the survey lines, from the longest straight leg (0-180:
    # a line is flown both ways). The planner's own flight_direction_deg is an angle
    # from east, counter-clockwise, which is not what an operator reads off a map.
    leg_ends: dict[int, tuple[dict, dict]] = {}
    for row in rows:
        first, _last = leg_ends.get(row["leg"], (row, row))
        leg_ends[row["leg"]] = (first, row)
    line_bearing = None
    longest = 0.0
    for first, last in leg_ends.values():
        length = last["along_m"] - first["along_m"]
        bearing = _bearing_deg([first["lon"], first["lat"]], [last["lon"], last["lat"]])
        if bearing is not None and length > longest:
            longest, line_bearing = length, round(bearing % 180.0, 1)

    safety = plan.get("safety_constraints") or {}
    terrain_type = plan.get("terrain_model_type") or "flat"
    return {
        "template": plan.get("template") or plan.get("mode"),
        "camera": plan.get("camera"),
        "altitude_m": plan.get("altitude_m"),
        "altitude_frame": frame,
        "front_overlap_pct": plan.get("front_overlap_pct"),
        "side_overlap_pct": plan.get("side_overlap_pct"),
        "speed_m_s": plan.get("speed_m_s"),
        "cruise_speed_m_s": plan.get("cruise_speed_m_s"),
        "gimbal_tilt_deg": plan.get("gimbal_tilt_deg"),
        "flight_direction_deg": plan.get("flight_direction_deg"),
        "line_bearing_deg": line_bearing,
        "line_spacing_m": plan.get("line_spacing_m") or None,
        "capture_spacing_m": plan.get("capture_spacing_m") or None,
        "distance_m": plan.get("path_distance_m"),
        "duration_min": plan.get("estimated_time_min"),
        "gsd_cm": plan.get("estimated_gsd_cm"),
        "waypoint_count": len(rows),
        "trigger_count": sum(1 for r in rows if r["trigger"]),
        "line_count": len(lines),
        "leg_count": (rows[-1]["leg"] + 1) if rows else 0,
        "route_length_m": rows[-1]["along_m"] if rows else 0.0,
        "rth_altitude_m": safety.get("rth_altitude_m"),
        "min_altitude_m": safety.get("min_altitude_m"),
        "max_altitude_m": safety.get("max_altitude_m"),
        "geofence_vertices": max(0, len(safety.get("geofence") or []) - 1),
        "no_fly_count": len(safety.get("no_fly_polygons") or []),
        "safety_adjustments": dict(plan.get("safety_adjustments") or {}),
        "terrain": {
            "follow": bool(plan.get("terrain_follow_enabled")),
            "mode": plan.get("terrain_follow_mode"),
            "model_type": terrain_type,
            "model_source": plan.get("terrain_model_source") or "none",
            "source_path": plan.get("terrain_source_path") or "",
            "sampled": any(r["ground_m"] is not None for r in rows),
            "home_ground_m": home_ground_m,
        },
        "warnings": list(plan.get("coverage_notes") or []),
        "waypoints": rows,
        "footprints": {"type": "FeatureCollection", "features": footprints},
        "footprint_basis": footprint_basis,
    }


def camera_hull(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """The convex hull (open ring, counter-clockwise) of lon/lat points."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) < 3:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]
