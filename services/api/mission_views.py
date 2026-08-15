"""Read-only mission preview and playback views derived from persisted compiled plans."""

from __future__ import annotations

import json
import math
from typing import Any

from pyproj import CRS, Geod

from .models import Mission


WGS84 = Geod(ellps="WGS84")


def load_compiled_plan(mission: Mission) -> dict[str, Any]:
    try:
        plan = json.loads(mission.plan_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The persisted compiled plan is not valid JSON.") from exc
    if not isinstance(plan, dict) or not plan.get("waypoints"):
        raise ValueError("The persisted mission has no compiled waypoint path.")
    return plan


def _aoi_area(mission: Mission) -> dict[str, Any]:
    try:
        geometry = json.loads(mission.aoi_geojson or "null")
        ring = geometry["coordinates"][0]
        crs = CRS.from_epsg(mission.crs_epsg)
    except (TypeError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "unavailable", "reason": f"AOI or CRS is unusable: {exc}"}
    if len(ring) < 4:
        return {"status": "unavailable", "reason": "AOI has no closed polygon ring."}
    if crs.is_geographic:
        area, _ = WGS84.polygon_area_perimeter(
            [float(point[0]) for point in ring], [float(point[1]) for point in ring]
        )
        area_m2 = abs(float(area))
        method = "WGS84_geodesic"
    elif crs.is_projected:
        area_m2 = abs(sum(
            float(ring[index][0]) * float(ring[index + 1][1])
            - float(ring[index + 1][0]) * float(ring[index][1])
            for index in range(len(ring) - 1)
        ) / 2.0)
        method = "projected_crs_shoelace"
    else:
        return {"status": "unavailable", "reason": "AOI CRS is neither geographic nor projected."}
    return {
        "status": "measured_from_georeferenced_aoi", "area_m2": area_m2,
        "crs_epsg": mission.crs_epsg, "method": method,
    }


def _safety(plan: dict[str, Any]) -> dict[str, Any]:
    constraints = plan.get("safety_constraints") or (
        (plan.get("flight_recipe") or {}).get("constraints") or {}
    )
    return {
        "geofence": constraints.get("geofence") or [],
        "no_fly_polygons": constraints.get("no_fly_polygons") or [],
        "min_altitude_m": constraints.get("min_altitude_m"),
        "max_altitude_m": constraints.get("max_altitude_m"),
        "rth_altitude_m": constraints.get("rth_altitude_m"),
        "standoff_m": constraints.get("standoff_m"),
    }


def mission_preview(mission: Mission) -> dict[str, Any]:
    plan = load_compiled_plan(mission)
    waypoints = [[float(value) for value in point[:3]] for point in plan["waypoints"]]
    altitudes = [point[2] for point in waypoints if len(point) >= 3]
    inputs = plan.get("operator_inputs") or {}
    aircraft = inputs.get("aircraft") or {}
    return {
        "mission_id": mission.id, "name": mission.name, "version": mission.version,
        "compiled_plan": {
            "source": plan.get("source") or "unrecorded",
            "recipe_id": (plan.get("flight_recipe") or {}).get("recipe_id"),
        },
        "path": {"type": "LineString", "coordinates": waypoints},
        "area": _aoi_area(mission),
        "altitude": {
            "minimum_m": min(altitudes) if altitudes else None,
            "maximum_m": max(altitudes) if altitudes else None,
            "reference": "compiled_waypoint_altitude",
        },
        "distance_m": float(plan.get("path_distance_m", mission.distance_m)),
        "duration_min": float(plan.get("estimated_time_min", mission.duration_min)),
        "drone": (
            {"status": "operator_declared", **aircraft}
            if aircraft.get("model") else
            {"status": "unavailable", "reason": "No aircraft model was recorded with the mission."}
        ),
        "safety_areas": _safety(plan),
    }


def _distance_m(left: list[float], right: list[float]) -> float:
    _, _, distance = WGS84.inv(left[0], left[1], right[0], right[1])
    vertical = float(right[2] - left[2]) if len(left) >= 3 and len(right) >= 3 else 0.0
    return math.hypot(float(distance), vertical)


def mission_simulation(mission: Mission) -> dict[str, Any]:
    """Build timeline frames from commands and capture poses in the stored plan."""
    plan = load_compiled_plan(mission)
    commands = [row for row in (plan.get("autopilot_commands") or [])
                if row.get("command") == "NAV_WAYPOINT"]
    path = (
        [[float(row["lon"]), float(row["lat"]), float(row["alt_m"])] for row in commands]
        if commands else
        [[float(value) for value in point[:3]] for point in plan["waypoints"]]
    )
    poses = ((plan.get("repeat_anchor") or {}).get("capture_poses_local") or [])
    default_gimbal = float(plan.get("gimbal_tilt_deg", -90.0))
    capture_enabled = bool((plan.get("flight_recipe") or {}).get("metadata", {}).get(
        "waypoint_capture_enabled", True
    ))
    inputs = plan.get("operator_inputs") or {}
    battery_input = inputs.get("battery") or {}
    battery_available = (
        battery_input.get("start_pct") is not None
        and float(battery_input.get("usable_minutes", 0.0)) > 0
    )
    elapsed = 0.0
    timeline: list[dict[str, Any]] = []
    for index, coordinate in enumerate(path):
        command = commands[index] if index < len(commands) else {}
        pose = poses[index] if index < len(poses) else {}
        if index:
            speed = float(command.get("speed_m_s", 0.0) or 0.0)
            if speed <= 0:
                raise ValueError(f"Compiled waypoint {index + 1} has no positive speed.")
            elapsed += _distance_m(path[index - 1], coordinate) / speed
        elapsed += float(pose.get("dwell_s", 0.0) or 0.0)
        battery_pct = None
        if battery_available:
            consumed = elapsed / (float(battery_input["usable_minutes"]) * 60.0) * 100.0
            battery_pct = max(0.0, float(battery_input["start_pct"]) - consumed)
        timeline.append({
            "index": index, "time_s": round(elapsed, 3), "position": coordinate,
            "yaw_deg": float(command.get("yaw_deg", pose.get("yaw_deg", 0.0)) or 0.0),
            "gimbal_pitch_deg": float(pose.get("gimbal_pitch_deg", default_gimbal)),
            "capture": capture_enabled and index < len(poses),
            "battery_pct": round(battery_pct, 3) if battery_pct is not None else None,
        })

    terrain_type = str(plan.get("terrain_model_type", "flat") or "flat")
    terrain_source = str(plan.get("terrain_model_source", "none") or "none")
    terrain_enabled = bool(plan.get("terrain_follow_enabled", False))
    unavailable_sources = {"", "none", "disabled", "missing_terrain_source"}
    terrain_available = terrain_enabled and not (
        terrain_type == "flat" or terrain_source in unavailable_sources
    )
    terrain = (
        {
            "status": "available", "model_type": terrain_type,
            "source": terrain_source, "samples": plan.get("terrain_profile") or [],
            "note": "Only terrain samples embedded in the compiled plan are rendered.",
        } if terrain_available else {
            "status": "unavailable", "model_type": terrain_type,
            "source": terrain_source, "samples": [],
            "reason": (
                "The compiled plan has no surveyed terrain model. Playback does not "
                "draw a flat surface as though terrain had been measured."
            ),
        }
    )
    battery = (
        {
            "status": "estimated", "basis": "operator_declared_usable_minutes",
            "start_pct": float(battery_input["start_pct"]),
            "usable_minutes": float(battery_input["usable_minutes"]),
        } if battery_available else {
            "status": "unavailable",
            "reason": "No battery start percentage and usable flight time were recorded.",
        }
    )
    return {
        "type": "odk-mission-simulation", "mission_id": mission.id,
        "source": "persisted_compiled_plan", "plan_source": plan.get("source"),
        "timeline": timeline,
        "capture_points": [row["position"] for row in timeline if row["capture"]],
        "safety_areas": _safety(plan), "terrain": terrain, "battery": battery,
        "drone": (inputs.get("aircraft") or {
            "status": "unavailable", "reason": "No aircraft model was recorded."
        }),
    }
