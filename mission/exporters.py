"""Mission exporters for real ground-control software and autopilots.

The planner already computes yaw, gimbal pitch, dwell time, and camera triggers for
every viewpoint. The original QGC WPL writer discarded all of it and emitted bare
NAV_WAYPOINT rows, which is why exported missions flew the route but captured nothing
useful. These exporters carry the full command stream into each target format.

Formats: MAVLink mission items, QGroundControl `.plan`, QGC WPL 110, DJI WPML `.kmz`,
Litchi CSV, KML, and GeoJSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET
import zipfile

# MAV_CMD values used by the exporters.
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LOITER_TIME = 19
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_CONDITION_YAW = 115
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_DIGICAM_CONTROL = 203
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_CMD_NAV_FENCE_RETURN_POINT = 5000
MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION = 5001
MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION = 5002
MAV_CMD_NAV_RALLY_POINT = 5100

MAV_FRAME_GLOBAL = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_MISSION = 2
MAV_MOUNT_MODE_MAVLINK_TARGETING = 2


@dataclass
class MissionItem:
    """One MAVLink mission item, in the shape both the uploader and writers consume."""

    seq: int
    command: int
    frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    autocontinue: int = 1
    current: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "command": self.command,
            "frame": self.frame,
            "param1": self.param1,
            "param2": self.param2,
            "param3": self.param3,
            "param4": self.param4,
            "lat": self.latitude,
            "lon": self.longitude,
            "alt": self.altitude,
            "autocontinue": self.autocontinue,
            "current": self.current,
            "label": self.label,
        }


def _plan_dict(mission: Any) -> dict[str, Any]:
    """Accept a MissionPlan dataclass or an already-serialized dict."""
    if isinstance(mission, dict):
        return mission
    for attr in ("to_dict", "asdict"):
        fn = getattr(mission, attr, None)
        if callable(fn):
            return fn()
    return {key: getattr(mission, key) for key in dir(mission) if not key.startswith("_")}


def _pose_from_mapping(pose: dict[str, Any], default_alt: float, default_gimbal: float) -> dict[str, Any]:
    return {
        "lon": float(pose.get("lon", pose.get("longitude", 0.0))),
        "lat": float(pose.get("lat", pose.get("latitude", 0.0))),
        "alt": float(pose.get("alt_m", pose.get("alt", pose.get("altitude_m", default_alt)))),
        "yaw_deg": float(pose.get("yaw_deg") if pose.get("yaw_deg") is not None else 0.0),
        "gimbal_pitch_deg": float(
            pose.get("gimbal_pitch_deg") if pose.get("gimbal_pitch_deg") is not None else default_gimbal
        ),
        "dwell_s": float(pose.get("dwell_s", 0.0) or 0.0),
        "trigger": bool(pose.get("trigger", True)),
        "speed_mps": float(pose.get("speed_mps", 0.0) or 0.0),
        "yaw_locked": bool(pose.get("camera_yaw_locked", False)),
    }


def _geojson_poses(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-waypoint poses from the plan's GeoJSON, ordered by waypoint index.

    The planner writes yaw, gimbal pitch, dwell, trigger and yaw lock onto every
    waypoint feature. The flight recipe does NOT carry the compiled world poses (it is
    the recipe, not its compilation), so reading only the recipe meant every export
    fell back to bare nadir waypoints: a tower orbit pointed its camera at the ground.
    """
    geojson = plan.get("geojson") or {}
    features = geojson.get("features") if isinstance(geojson, dict) else None
    rows: list[tuple[int, dict[str, Any]]] = []
    for position, feature in enumerate(features or []):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        geometry = feature.get("geometry") or {}
        if props.get("role") != "waypoint" or geometry.get("type") != "Point":
            continue
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        pose = dict(props)
        pose["lon"], pose["lat"] = float(coords[0]), float(coords[1])
        if len(coords) >= 3:
            pose["alt_m"] = float(coords[2])
        index = props.get("index")
        rows.append((int(index) if isinstance(index, int) else 10**9 + position, pose))
    rows.sort(key=lambda row: row[0])
    return [pose for _, pose in rows]


def _viewpoints(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover per-viewpoint pose detail, falling back to bare waypoints.

    Order of preference: explicit world poses (recipe or plan), then the pose
    properties the planner writes onto each GeoJSON waypoint, then bare geometry.
    Only the last loses the capture behaviour, and it is only used when the plan
    genuinely carries nothing else.
    """
    recipe = plan.get("flight_recipe") or {}
    poses = (
        plan.get("world_poses")
        or recipe.get("world_poses")
        or recipe.get("poses")
        or _geojson_poses(plan)
    )
    default_alt = float(plan.get("altitude_m", 50.0) or 50.0)
    gimbal = plan.get("gimbal_tilt_deg", -90.0)
    default_gimbal = float(-90.0 if gimbal is None else gimbal)

    viewpoints: list[dict[str, Any]] = []
    if poses:
        return [_pose_from_mapping(pose, default_alt, default_gimbal) for pose in poses]

    for row in plan.get("waypoints") or []:
        viewpoints.append(
            {
                "lon": float(row[0]),
                "lat": float(row[1]),
                "alt": float(row[2]) if len(row) >= 3 else default_alt,
                "yaw_deg": 0.0,
                "gimbal_pitch_deg": default_gimbal,
                "dwell_s": 0.0,
                "trigger": True,
                "speed_mps": 0.0,
                "yaw_locked": False,
            }
        )
    return viewpoints


def _cruise_speed(plan: dict[str, Any], fallback: float) -> float:
    """The speed the planner planned for, or `fallback` when the plan carries none.

    `cruise_speed_m_s` is what the planner flew its estimate at (after any wind or
    profile cap); `wind_adjusted_speed_m_s` is the same figure on plans saved before
    that field existed.
    """
    for key in ("cruise_speed_m_s", "cruise_speed_mps", "wind_adjusted_speed_m_s"):
        value = plan.get(key)
        try:
            speed = float(value)
        except (TypeError, ValueError):
            continue
        if speed > 0.0:
            return speed
    return float(fallback)


def _altitude_frame(plan: dict[str, Any]) -> tuple[str, float]:
    """("amsl", home ground elevation) or ("relative", 0.0).

    Waypoint altitudes in a plan are always relative to the home/takeoff ground. An
    AMSL plan additionally knows the ground elevation at home from its terrain model,
    and exporters that can carry absolute altitude add it back and say so.
    """
    frame = str(plan.get("altitude_frame", "") or "").lower()
    home = plan.get("home_elevation_amsl_m")
    if frame == "amsl" and home is not None:
        try:
            return "amsl", float(home)
        except (TypeError, ValueError):
            pass
    return "relative", 0.0


def _capture_spacing(plan: dict[str, Any]) -> float:
    """Distance-triggered capture spacing in metres, 0 when triggering per waypoint."""
    recipe = plan.get("flight_recipe") or {}
    capture = recipe.get("capture") or {}
    if not bool(capture.get("continuous_capture", False)):
        return 0.0
    return float(capture.get("capture_spacing_m", 0.0) or 0.0)


def build_mission_items(mission: Any, *, include_rth: bool = True) -> list[MissionItem]:
    """Expand a mission plan into a full MAVLink item sequence.

    Emits gimbal, yaw, dwell, and camera commands around each waypoint so the vehicle
    reproduces the planned capture, not merely the planned path.
    """
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    if not viewpoints:
        raise ValueError("Mission plan contains no waypoints to export.")

    constraints = plan.get("safety_constraints") or {}
    spacing = _capture_spacing(plan)
    altitude_frame, home_elevation = _altitude_frame(plan)
    # Positional items carry absolute altitude (frame 0) on an AMSL plan; otherwise
    # altitude relative to home (frame 3), which is what every plan stores internally.
    nav_frame = MAV_FRAME_GLOBAL if altitude_frame == "amsl" else MAV_FRAME_GLOBAL_RELATIVE_ALT
    items: list[MissionItem] = []
    seq = 0

    def add(command: int, **kwargs: Any) -> None:
        nonlocal seq
        if kwargs.get("frame", nav_frame) != MAV_FRAME_MISSION:
            kwargs.setdefault("frame", nav_frame)
            if nav_frame == MAV_FRAME_GLOBAL and command != MAV_CMD_NAV_LAND:
                kwargs["altitude"] = float(kwargs.get("altitude", 0.0)) + home_elevation
        items.append(MissionItem(seq=seq, command=command, current=1 if seq == 0 else 0, **kwargs))
        seq += 1

    first = viewpoints[0]
    add(
        MAV_CMD_NAV_TAKEOFF,
        param1=0.0,
        latitude=first["lat"],
        longitude=first["lon"],
        altitude=first["alt"],
        label="takeoff",
    )

    speed = _cruise_speed(plan, 0.0)
    if speed > 0.0:
        # DO_CHANGE_SPEED is what actually reaches the autopilot; the .plan cruiseSpeed
        # field is only QGC's own estimate and never leaves the ground station.
        add(
            MAV_CMD_DO_CHANGE_SPEED,
            frame=MAV_FRAME_MISSION,
            param1=1.0,
            param2=speed,
            param3=-1.0,
            label=f"speed {speed:.1f} m/s",
        )

    if spacing > 0.0:
        add(
            MAV_CMD_DO_SET_CAM_TRIGG_DIST,
            frame=MAV_FRAME_MISSION,
            param1=spacing,
            param3=1.0,
            label=f"trigger every {spacing:.1f} m",
        )

    for index, pose in enumerate(viewpoints):
        gimbal = pose["gimbal_pitch_deg"]
        # DO_MOUNT_CONTROL: params 1-3 are pitch/roll/yaw and param7 (the z slot) is
        # the mount mode, which is why the mode is written to `altitude` here. Mount yaw
        # is relative to the airframe, and CONDITION_YAW below turns the airframe to the
        # heading, so the mount stays centred; an absolute heading here would be added
        # on top by any gimbal with a yaw axis.
        add(
            MAV_CMD_DO_MOUNT_CONTROL,
            frame=MAV_FRAME_MISSION,
            param1=gimbal,
            param2=0.0,
            param3=0.0,
            altitude=float(MAV_MOUNT_MODE_MAVLINK_TARGETING),
            label=f"gimbal {gimbal:.1f} deg",
        )

        if pose["yaw_locked"]:
            add(
                MAV_CMD_CONDITION_YAW,
                frame=MAV_FRAME_MISSION,
                param1=pose["yaw_deg"] % 360.0,
                param2=25.0,
                param3=0.0,
                param4=0.0,
                label=f"yaw {pose['yaw_deg']:.1f} deg",
            )

        dwell = pose["dwell_s"]
        if dwell > 0.0:
            add(
                MAV_CMD_NAV_LOITER_TIME,
                param1=dwell,
                latitude=pose["lat"],
                longitude=pose["lon"],
                altitude=pose["alt"],
                label=f"hold {dwell:.1f} s",
            )
        else:
            add(
                MAV_CMD_NAV_WAYPOINT,
                param1=0.0,
                param4=pose["yaw_deg"] if pose["yaw_locked"] else float("nan"),
                latitude=pose["lat"],
                longitude=pose["lon"],
                altitude=pose["alt"],
                label=f"waypoint {index + 1}",
            )

        if spacing <= 0.0 and pose["trigger"]:
            # DO_DIGICAM_CONTROL takes the shutter command in param5, which is the
            # x/latitude slot of a mission item.
            add(
                MAV_CMD_DO_DIGICAM_CONTROL,
                frame=MAV_FRAME_MISSION,
                latitude=1.0,
                label="capture",
            )

    if spacing > 0.0:
        add(MAV_CMD_DO_SET_CAM_TRIGG_DIST, frame=MAV_FRAME_MISSION, param1=0.0, label="stop trigger")

    if include_rth:
        action = str(constraints.get("rth_action", "return_home") or "return_home").lower()
        rth_alt = float(constraints.get("rth_altitude_m", viewpoints[-1]["alt"]) or viewpoints[-1]["alt"])
        if action in {"land", "land_in_place"}:
            add(
                MAV_CMD_NAV_LAND,
                frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
                latitude=viewpoints[-1]["lat"],
                longitude=viewpoints[-1]["lon"],
                altitude=0.0,
                label="land",
            )
        else:
            add(MAV_CMD_NAV_RETURN_TO_LAUNCH, frame=MAV_FRAME_MISSION, altitude=rth_alt, label="return to launch")

    return items


# Distance the exported inclusion fence sits outside the drawn area. ArduPilot breaches
# a polygon fence when the vehicle comes within FENCE_MARGIN (2 m by default) of it, and
# a mapping grid deliberately flies its outer lines on the area's edge; a fence equal to
# the area would therefore trip on the first outer pass.
INCLUSION_FENCE_MARGIN_M = 5.0


def _offset_ring_outward(vertices: Sequence[Sequence[float]], margin_m: float) -> list[list[float]]:
    """Offset a lon/lat ring outward by `margin_m`, with mitred corners.

    Worked in a local equirectangular frame, which is exact to well under a centimetre
    at the scale of a survey site.
    """
    if margin_m <= 0.0 or len(vertices) < 3:
        return [[float(v[0]), float(v[1])] for v in vertices]
    lon0 = sum(float(v[0]) for v in vertices) / len(vertices)
    lat0 = sum(float(v[1]) for v in vertices) / len(vertices)
    mx = 111_320.0 * math.cos(math.radians(lat0))
    my = 110_540.0
    pts = [((float(v[0]) - lon0) * mx, (float(v[1]) - lat0) * my) for v in vertices]
    n = len(pts)
    area2 = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    # Outward normal of edge (dx, dy) is (dy, -dx) for a counter-clockwise ring.
    sign = 1.0 if area2 > 0 else -1.0
    out: list[list[float]] = []
    for i in range(n):
        prev_pt, here, next_pt = pts[i - 1], pts[i], pts[(i + 1) % n]
        normals = []
        for start, end in ((prev_pt, here), (here, next_pt)):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy) or 1.0
            normals.append((sign * dy / length, -sign * dx / length))
        bx, by = normals[0][0] + normals[1][0], normals[0][1] + normals[1][1]
        blen = math.hypot(bx, by)
        if blen < 1e-9:
            bx, by, scale = normals[1][0], normals[1][1], margin_m
        else:
            bx, by = bx / blen, by / blen
            cos_half = max(0.25, bx * normals[1][0] + by * normals[1][1])  # mitre limit 4x
            scale = margin_m / cos_half
        out.append([lon0 + (here[0] + bx * scale) / mx, lat0 + (here[1] + by * scale) / my])
    return out


def fence_rings(
    mission: Any, *, inclusion_margin_m: float = INCLUSION_FENCE_MARGIN_M
) -> list[tuple[bool, list[list[float]]]]:
    """Every fence polygon as (is_inclusion, [[lon, lat], ...]), one entry per ring.

    Rings are kept separate because a writer that concatenates them (as the .plan
    writer once did) turns two no-fly rectangles into one self-intersecting polygon.
    """
    plan = _plan_dict(mission)
    constraints = plan.get("safety_constraints") or {}

    def clean(ring: Sequence[Sequence[float]]) -> list[list[float]]:
        vertices = [[float(v[0]), float(v[1])] for v in ring if len(v) >= 2]
        if len(vertices) >= 3 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        return vertices

    rings: list[tuple[bool, list[list[float]]]] = []
    # The planner accepts several spellings on input (`parse_constraints` reads
    # no_fly_polygons, no_fly_zones, and obstacles), so the exporter accepts the
    # same ones rather than silently emitting no fence for a plan that has one.
    geofence = clean(
        constraints.get("geofence")
        or constraints.get("geofence_polygon")
        or plan.get("polygon")
        or []
    )
    if len(geofence) >= 3:
        rings.append((True, _offset_ring_outward(geofence, inclusion_margin_m)))
    exclusions = (
        constraints.get("no_fly_polygons")
        or constraints.get("no_fly_zones")
        or constraints.get("obstacles")
        or []
    )
    for polygon in exclusions:
        ring = clean(polygon)
        if len(ring) >= 3:
            rings.append((False, ring))
    return rings


def build_fence_items(mission: Any) -> list[MissionItem]:
    """Geofence inclusion and no-fly exclusion polygons as MAVLink fence items.

    param1 of every vertex is its own ring's vertex count, which is how an autopilot
    tells where one polygon ends and the next begins.
    """
    items: list[MissionItem] = []
    for inclusion, ring in fence_rings(mission):
        command = (MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION if inclusion
                   else MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION)
        for vertex in ring:
            items.append(
                MissionItem(
                    seq=len(items),
                    command=command,
                    frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    param1=float(len(ring)),
                    latitude=float(vertex[1]),
                    longitude=float(vertex[0]),
                    label="fence" if inclusion else "no-fly",
                )
            )
    return items


def build_rally_items(mission: Any) -> list[MissionItem]:
    """Rally points from the mission's emergency landing zones, when present."""
    plan = _plan_dict(mission)
    constraints = plan.get("safety_constraints") or {}
    points = constraints.get("rally_points") or constraints.get("emergency_landing_points") or []
    items: list[MissionItem] = []
    for index, point in enumerate(points):
        if isinstance(point, dict):
            lon, lat = float(point.get("lon", 0.0)), float(point.get("lat", 0.0))
            alt = float(point.get("alt_m", constraints.get("rth_altitude_m", 50.0)) or 50.0)
        elif len(point) >= 2:
            lon, lat = float(point[0]), float(point[1])
            alt = float(point[2]) if len(point) >= 3 else 50.0
        else:
            continue
        items.append(
            MissionItem(
                seq=index,
                command=MAV_CMD_NAV_RALLY_POINT,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                label=f"rally {index + 1}",
            )
        )
    return items


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def export_qgc_wpl(path: str | Path, mission: Any) -> str:
    """QGC WPL 110 text mission, carrying every command rather than positions only.

    Row 0 is the home position. ArduPilot and Mission Planner reserve sequence 0 for
    home and overwrite whatever is loaded there, so a file whose first row is the
    takeoff loses its takeoff on load -- the same reservation the live uploader
    allows for in core/mission_planner_bridge.py (_with_home_slot).
    """
    plan = _plan_dict(mission)
    items = build_mission_items(plan)
    _, home_elevation = _altitude_frame(plan)
    first_positional = next((i for i in items if i.frame != MAV_FRAME_MISSION), items[0])
    home = MissionItem(
        seq=0,
        command=MAV_CMD_NAV_WAYPOINT,
        frame=MAV_FRAME_GLOBAL,
        latitude=first_positional.latitude,
        longitude=first_positional.longitude,
        altitude=home_elevation,
        current=1,
        label="home",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def clean(value: float) -> float:
        return 0.0 if isinstance(value, float) and math.isnan(value) else float(value)

    lines = ["QGC WPL 110"]
    for seq, item in enumerate([home, *items]):
        lines.append(
            "\t".join(
                [
                    str(seq),
                    "1" if seq == 0 else "0",
                    str(item.frame),
                    str(item.command),
                    f"{clean(item.param1):.8f}",
                    f"{clean(item.param2):.8f}",
                    f"{clean(item.param3):.8f}",
                    f"{clean(item.param4):.8f}",
                    f"{item.latitude:.8f}",
                    f"{item.longitude:.8f}",
                    f"{item.altitude:.6f}",
                    str(item.autocontinue),
                ]
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def export_qgc_plan(path: str | Path, mission: Any, *, vehicle_type: int = 2, firmware_type: int = 12) -> str:
    """QGroundControl `.plan` JSON, including geofence and rally points.

    Defaults describe an ArduPilot multirotor (firmware 12, vehicle 2).
    """
    plan = _plan_dict(mission)
    items = build_mission_items(mission)
    rally_items = build_rally_items(mission)
    constraints = plan.get("safety_constraints") or {}
    altitude_frame, home_elevation = _altitude_frame(plan)
    # QGC AltitudeMode: 1 relative to home, 2 AMSL.
    altitude_mode = 2 if altitude_frame == "amsl" else 1

    simple_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        params: list[Any] = [item.param1, item.param2, item.param3, item.param4]
        params.append(item.latitude if item.frame != MAV_FRAME_MISSION else 0.0)
        params.append(item.longitude if item.frame != MAV_FRAME_MISSION else 0.0)
        params.append(item.altitude if item.frame != MAV_FRAME_MISSION else 0.0)
        params = [None if isinstance(p, float) and math.isnan(p) else p for p in params]
        simple_items.append(
            {
                "AMSLAltAboveTerrain": None,
                "Altitude": item.altitude,
                "AltitudeMode": altitude_mode,
                "autoContinue": bool(item.autocontinue),
                "command": item.command,
                "doJumpId": index + 1,
                "frame": item.frame,
                "params": params,
                "type": "SimpleItem",
            }
        )

    first = items[0]
    # One polygon per ring: concatenating every no-fly zone into a single exclusion
    # polygon made a self-intersecting shape that fenced off the wrong area.
    polygons: list[dict[str, Any]] = [
        {"inclusion": inclusion, "polygon": [[v[1], v[0]] for v in ring], "version": 1}
        for inclusion, ring in fence_rings(plan)
    ]
    speed = _cruise_speed(plan, 0.0)

    payload = {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": polygons, "version": 2},
        "groundStation": "OpenDroneKit",
        "mission": {
            # QGC uses hoverSpeed for multirotors and cruiseSpeed for fixed wing; both
            # carry the planned speed so the ground station's estimate matches ours.
            "cruiseSpeed": speed if speed > 0.0 else 12.0,
            "firmwareType": int(firmware_type),
            "globalPlanAltitudeMode": altitude_mode,
            "hoverSpeed": speed if speed > 0.0 else 5.0,
            "items": simple_items,
            "plannedHomePosition": [
                first.latitude,
                first.longitude,
                home_elevation if altitude_frame == "amsl"
                else float(constraints.get("home_altitude_m", 0.0) or 0.0),
            ],
            "vehicleType": int(vehicle_type),
            "version": 2,
        },
        "rallyPoints": {"points": [[i.latitude, i.longitude, i.altitude] for i in rally_items], "version": 2},
        "version": 1,
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return str(target)


# Litchi refuses a mission of more than 99 waypoints.
LITCHI_MAX_WAYPOINTS = 99
LITCHI_ACTION_NONE = -1
LITCHI_ACTION_STAY_MS = 0
LITCHI_ACTION_TAKE_PHOTO = 1
LITCHI_ACTION_TILT_CAMERA_DEG = 5


def export_litchi_csv(path: str | Path, mission: Any) -> str | list[str]:
    """Litchi Mission Hub CSV, with per-waypoint gimbal pitch, heading, and capture.

    Litchi accepts at most 99 waypoints per mission, so a longer plan is written as
    consecutive parts (`<stem>_part1of3.csv`, ...), each flyable on its own and each
    starting where the previous one ended. A single-part plan is written to `path` and
    its path returned; a split plan returns every part's path in flight order.
    """
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    if not viewpoints:
        raise ValueError("Mission plan contains no waypoints to export.")
    speed = _cruise_speed(plan, 5.0)

    header = [
        "latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)", "rotationdir",
        "gimbalmode", "gimbalpitchangle",
    ]
    for index in range(1, 16):
        header.extend([f"actiontype{index}", f"actionparam{index}"])
    header.extend([
        "altitudemode", "speed(m/s)", "poi_latitude", "poi_longitude", "poi_altitude(m)",
        "poi_altitudemode", "photo_timeinterval", "photo_distinterval",
    ])

    spacing = _capture_spacing(plan)
    rows: list[str] = []
    for pose in viewpoints:
        actions: list[str] = []
        # Litchi's action types (litchiutilities.com/docs/litchiCsv.php, and what Mission
        # Hub itself writes): -1 none, 0 stay for (ms), 1 take photo, 2 start recording,
        # 3 stop recording, 4 rotate aircraft (deg), 5 tilt camera (deg). This used to
        # write 1 for the hover and 5 for the photo, so every structure mission tilted
        # the camera to 0 deg at each point and took no photograph, and a dwell was a
        # photo instead of a hover. The hover comes first so the shot is taken steady.
        if pose["dwell_s"] > 0.0:
            actions.extend([str(LITCHI_ACTION_STAY_MS), str(int(round(pose["dwell_s"] * 1000)))])
        if spacing <= 0.0 and pose["trigger"]:
            actions.extend([str(LITCHI_ACTION_TAKE_PHOTO), "0"])
        while len(actions) < 30:
            actions.extend(["-1", "0"])
        actions = actions[:30]

        row = [
            f"{pose['lat']:.8f}", f"{pose['lon']:.8f}", f"{pose['alt']:.2f}",
            f"{pose['yaw_deg'] % 360.0:.1f}", "0.2", "0",
            "2", f"{pose['gimbal_pitch_deg']:.1f}",
        ]
        row.extend(actions)
        # altitudemode 0: relative to the takeoff point, which is what the plan's
        # altitudes are (terrain following is already folded into them). Mode 1 would
        # make Litchi add its own terrain offset on top.
        row.extend([
            "0", f"{min(15.0, pose['speed_mps'] or speed):.2f}", "0", "0", "0", "0",
            # -1 is Litchi's "not used" for the interval columns, as Mission Hub writes.
            "-1", f"{spacing:.2f}" if spacing > 0 else "-1",
        ])
        rows.append(",".join(row))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i:i + LITCHI_MAX_WAYPOINTS] for i in range(0, len(rows), LITCHI_MAX_WAYPOINTS)]
    if len(chunks) == 1:
        target.write_text("\n".join([",".join(header), *rows]) + "\n", encoding="utf-8")
        return str(target)
    written: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        part = target.with_name(f"{target.stem}_part{index}of{len(chunks)}{target.suffix}")
        part.write_text("\n".join([",".join(header), *chunk]) + "\n", encoding="utf-8")
        written.append(str(part))
    return written


def export_kml(path: str | Path, mission: Any, name: str = "OpenDroneKit Mission") -> str:
    """KML with the flight path, numbered waypoints, geofence, and no-fly zones."""
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    constraints = plan.get("safety_constraints") or {}
    altitude_frame, home_elevation = _altitude_frame(plan)
    if altitude_frame == "amsl":
        altitude_mode, offset = "absolute", home_elevation
    else:
        # KML has no "relative to takeoff"; relativeToGround is the nearest and matches
        # it exactly over flat ground.
        altitude_mode, offset = "relativeToGround", 0.0

    kml = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = name

    for style_id, color, width in (("route", "ff00aaff", "3"), ("fence", "ff00ff00", "2"), ("nofly", "ff0000ff", "2")):
        style = ET.SubElement(document, "Style", {"id": style_id})
        line = ET.SubElement(style, "LineStyle")
        ET.SubElement(line, "color").text = color
        ET.SubElement(line, "width").text = width
        poly = ET.SubElement(style, "PolyStyle")
        ET.SubElement(poly, "fill").text = "0"

    if viewpoints:
        placemark = ET.SubElement(document, "Placemark")
        ET.SubElement(placemark, "name").text = "Flight path"
        ET.SubElement(placemark, "styleUrl").text = "#route"
        line = ET.SubElement(placemark, "LineString")
        ET.SubElement(line, "altitudeMode").text = altitude_mode
        ET.SubElement(line, "tessellate").text = "1"
        ET.SubElement(line, "coordinates").text = " ".join(
            f"{p['lon']:.8f},{p['lat']:.8f},{p['alt'] + offset:.2f}" for p in viewpoints
        )

    waypoint_folder = ET.SubElement(document, "Folder")
    ET.SubElement(waypoint_folder, "name").text = "Waypoints"
    for index, pose in enumerate(viewpoints, start=1):
        placemark = ET.SubElement(waypoint_folder, "Placemark")
        ET.SubElement(placemark, "name").text = str(index)
        ET.SubElement(placemark, "description").text = (
            f"altitude: {pose['alt']:.1f} m\nheading: {pose['yaw_deg']:.1f} deg\n"
            f"gimbal: {pose['gimbal_pitch_deg']:.1f} deg\ndwell: {pose['dwell_s']:.1f} s\n"
            f"capture: {'yes' if pose['trigger'] else 'no'}"
        )
        point = ET.SubElement(placemark, "Point")
        ET.SubElement(point, "altitudeMode").text = altitude_mode
        ET.SubElement(point, "coordinates").text = f"{pose['lon']:.8f},{pose['lat']:.8f},{pose['alt'] + offset:.2f}"

    def add_ring(ring: Sequence[Sequence[float]], label: str, style_id: str) -> None:
        vertices = [v for v in ring if len(v) >= 2]
        if len(vertices) < 3:
            return
        if vertices[0] != vertices[-1]:
            vertices = list(vertices) + [vertices[0]]
        placemark = ET.SubElement(document, "Placemark")
        ET.SubElement(placemark, "name").text = label
        ET.SubElement(placemark, "styleUrl").text = f"#{style_id}"
        polygon = ET.SubElement(placemark, "Polygon")
        outer = ET.SubElement(polygon, "outerBoundaryIs")
        linear = ET.SubElement(outer, "LinearRing")
        ET.SubElement(linear, "coordinates").text = " ".join(
            f"{float(v[0]):.8f},{float(v[1]):.8f},0" for v in vertices
        )

    zone = 0
    for inclusion, ring in fence_rings(plan):
        if inclusion:
            add_ring(ring, f"Geofence (area + {INCLUSION_FENCE_MARGIN_M:.0f} m margin)", "fence")
        else:
            zone += 1
            add_ring(ring, f"No-fly zone {zone}", "nofly")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(kml).write(target, encoding="utf-8", xml_declaration=True)
    return str(target)


def export_dji_wpml(
    path: str | Path,
    mission: Any,
    *,
    drone_enum: int = 77,
    drone_sub_enum: int = 0,
    payload_enum: int = 66,
) -> str:
    """DJI WPML `.kmz` for Pilot 2 / DJI Fly waypoint missions.

    A WPML archive is a zip holding `wpmz/template.kml` (the human-editable plan) and
    `wpmz/waylines.wpml` (the executable route). Both carry the DJI wpml namespace.
    Default enums describe a Mavic 3 Enterprise (drone 77 / sub 0, camera payload 66,
    per DJI's WPML droneEnumValue and payloadEnumValue tables). The previous default of
    68 is not an assigned drone enum.
    """
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    if not viewpoints:
        raise ValueError("Mission plan contains no waypoints to export.")
    spacing = _capture_spacing(plan)
    constraints = plan.get("safety_constraints") or {}
    speed = max(1.0, min(15.0, _cruise_speed(plan, 8.0)))
    altitude_frame, home_elevation = _altitude_frame(plan)
    # EGM96 is DJI's absolute-height mode. It is only used when the plan's terrain model
    # gives the home ground elevation; the DEM's vertical datum is assumed orthometric.
    height_mode = "EGM96" if altitude_frame == "amsl" else "relativeToStartPoint"
    height_offset = home_elevation if altitude_frame == "amsl" else 0.0
    rth_alt = float(constraints.get("rth_altitude_m", 80.0) or 80.0)

    kml_ns = "http://www.opengis.net/kml/2.2"
    wpml_ns = "http://www.dji.com/wpmz/1.0.2"
    ET.register_namespace("", kml_ns)
    ET.register_namespace("wpml", wpml_ns)

    def wpml(tag: str) -> str:
        return f"{{{wpml_ns}}}{tag}"

    def build(is_template: bool) -> ET.Element:
        root = ET.Element(f"{{{kml_ns}}}kml")
        document = ET.SubElement(root, f"{{{kml_ns}}}Document")

        if is_template:
            ET.SubElement(document, wpml("author")).text = "OpenDroneKit"
            ET.SubElement(document, wpml("createTime")).text = "0"
            ET.SubElement(document, wpml("updateTime")).text = "0"

        config = ET.SubElement(document, wpml("missionConfig"))
        ET.SubElement(config, wpml("flyToWaylineMode")).text = "safely"
        ET.SubElement(config, wpml("finishAction")).text = (
            "goHome" if str(constraints.get("rth_action", "return_home")).lower() not in {"land", "land_in_place"} else "autoLand"
        )
        ET.SubElement(config, wpml("exitOnRCLost")).text = "executeLostAction"
        ET.SubElement(config, wpml("executeRCLostAction")).text = "goBack"
        ET.SubElement(config, wpml("takeOffSecurityHeight")).text = f"{min(rth_alt, 100.0):.1f}"
        ET.SubElement(config, wpml("globalTransitionalSpeed")).text = f"{speed:.1f}"
        drone_info = ET.SubElement(config, wpml("droneInfo"))
        ET.SubElement(drone_info, wpml("droneEnumValue")).text = str(int(drone_enum))
        ET.SubElement(drone_info, wpml("droneSubEnumValue")).text = str(int(drone_sub_enum))
        payload_info = ET.SubElement(config, wpml("payloadInfo"))
        ET.SubElement(payload_info, wpml("payloadEnumValue")).text = str(int(payload_enum))
        ET.SubElement(payload_info, wpml("payloadPositionIndex")).text = "0"

        folder = ET.SubElement(document, f"{{{kml_ns}}}Folder")
        ET.SubElement(folder, wpml("templateType")).text = "waypoint"
        ET.SubElement(folder, wpml("templateId")).text = "0"
        if not is_template:
            ET.SubElement(folder, wpml("waylineId")).text = "0"
        coordinate = ET.SubElement(folder, wpml("waylineCoordinateSysParam"))
        ET.SubElement(coordinate, wpml("coordinateMode")).text = "WGS84"
        ET.SubElement(coordinate, wpml("heightMode")).text = height_mode
        ET.SubElement(folder, wpml("autoFlightSpeed")).text = f"{speed:.1f}"

        for index, pose in enumerate(viewpoints):
            placemark = ET.SubElement(folder, f"{{{kml_ns}}}Placemark")
            point = ET.SubElement(placemark, f"{{{kml_ns}}}Point")
            ET.SubElement(point, f"{{{kml_ns}}}coordinates").text = f"{pose['lon']:.8f},{pose['lat']:.8f}"
            ET.SubElement(placemark, wpml("index")).text = str(index)
            ET.SubElement(placemark, wpml("executeHeight")).text = f"{pose['alt'] + height_offset:.2f}"
            ET.SubElement(placemark, wpml("waypointSpeed")).text = f"{min(15.0, pose['speed_mps'] or speed):.1f}"

            heading = ET.SubElement(placemark, wpml("waypointHeadingParam"))
            ET.SubElement(heading, wpml("waypointHeadingMode")).text = (
                "smoothTransition" if pose["yaw_locked"] else "followWayline"
            )
            ET.SubElement(heading, wpml("waypointHeadingAngle")).text = f"{pose['yaw_deg'] % 360.0:.1f}"
            ET.SubElement(heading, wpml("waypointPoiPoint")).text = "0.000000,0.000000,0.000000"
            ET.SubElement(heading, wpml("waypointHeadingPathMode")).text = "followBadArc"

            turn = ET.SubElement(placemark, wpml("waypointTurnParam"))
            ET.SubElement(turn, wpml("waypointTurnMode")).text = (
                "toPointAndStopWithDiscontinuityCurvature" if pose["dwell_s"] > 0 else "toPointAndPassWithContinuityCurvature"
            )
            ET.SubElement(turn, wpml("waypointTurnDampingDist")).text = "0.2"
            ET.SubElement(placemark, wpml("useStraightLine")).text = "1"

            # Every waypoint gets an action group because the gimbal angle is always
            # commanded; dwell and capture actions are added on top when planned.
            group = ET.SubElement(placemark, wpml("actionGroup"))
            ET.SubElement(group, wpml("actionGroupId")).text = str(index)
            ET.SubElement(group, wpml("actionGroupStartIndex")).text = str(index)
            ET.SubElement(group, wpml("actionGroupEndIndex")).text = str(index)
            ET.SubElement(group, wpml("actionGroupMode")).text = "sequence"
            trigger = ET.SubElement(group, wpml("actionTrigger"))
            ET.SubElement(trigger, wpml("actionTriggerType")).text = "reachPoint"

            actions: list[tuple[str, dict[str, str]]] = [
                (
                    "gimbalRotate",
                    {
                        "gimbalHeadingYawBase": "aircraft",
                        "gimbalRotateMode": "absoluteAngle",
                        "gimbalPitchRotateEnable": "1",
                        "gimbalPitchRotateAngle": f"{pose['gimbal_pitch_deg']:.1f}",
                        "gimbalRollRotateEnable": "0",
                        "gimbalRollRotateAngle": "0",
                        "gimbalYawRotateEnable": "0",
                        "gimbalYawRotateAngle": "0",
                        "gimbalRotateTimeEnable": "0",
                        "gimbalRotateTime": "0",
                        "payloadPositionIndex": "0",
                    },
                )
            ]
            if pose["dwell_s"] > 0.0:
                actions.append(("hover", {"hoverTime": f"{pose['dwell_s']:.1f}"}))
            if spacing <= 0.0 and pose["trigger"]:
                actions.append(("takePhoto", {"payloadPositionIndex": "0", "fileSuffix": f"wp{index + 1}"}))

            for action_id, (func, params) in enumerate(actions):
                action = ET.SubElement(group, wpml("action"))
                ET.SubElement(action, wpml("actionId")).text = str(action_id)
                ET.SubElement(action, wpml("actionActuatorFunc")).text = func
                param_node = ET.SubElement(action, wpml("actionActuatorFuncParam"))
                for key, value in params.items():
                    ET.SubElement(param_node, wpml(key)).text = value

        if spacing > 0.0:
            interval = ET.SubElement(folder, wpml("waylineActionGroup"))
            ET.SubElement(interval, wpml("actionGroupId")).text = "9000"
            ET.SubElement(interval, wpml("actionGroupMode")).text = "sequence"
            trigger = ET.SubElement(interval, wpml("actionTrigger"))
            ET.SubElement(trigger, wpml("actionTriggerType")).text = "multipleDistance"
            ET.SubElement(trigger, wpml("actionTriggerParam")).text = f"{spacing:.2f}"

        return root

    target = Path(path)
    if target.suffix.lower() != ".kmz":
        target = target.with_suffix(".kmz")
    target.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, is_template in (("wpmz/template.kml", True), ("wpmz/waylines.wpml", False)):
            archive.writestr(member, ET.tostring(build(is_template), encoding="utf-8", xml_declaration=True))
    return str(target)


def export_geojson(path: str | Path, mission: Any) -> str:
    """The plan as GeoJSON: area, no-fly zones, flight path, and every waypoint with
    its yaw, gimbal pitch, dwell and trigger as properties.

    Uses the planner's own GeoJSON when the plan carries it, and otherwise builds the
    same structure from the viewpoints so a plan loaded from elsewhere still exports.
    """
    plan = _plan_dict(mission)
    geojson = plan.get("geojson")
    if not (isinstance(geojson, dict) and geojson.get("features")):
        viewpoints = _viewpoints(plan)
        if not viewpoints:
            raise ValueError("Mission plan contains no waypoints to export.")
        features: list[dict[str, Any]] = []
        for inclusion, ring in fence_rings(plan, inclusion_margin_m=0.0):
            features.append({
                "type": "Feature",
                "properties": {"role": "survey_polygon" if inclusion else "no_fly_polygon"},
                "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
            })
        path_coords = [[p["lon"], p["lat"], p["alt"]] for p in viewpoints]
        features.append({"type": "Feature", "properties": {"role": "flight_path"},
                         "geometry": {"type": "LineString", "coordinates": path_coords}})
        for index, pose in enumerate(viewpoints):
            features.append({
                "type": "Feature",
                "properties": {
                    "role": "waypoint", "index": index, "yaw_deg": pose["yaw_deg"],
                    "gimbal_pitch_deg": pose["gimbal_pitch_deg"], "dwell_s": pose["dwell_s"],
                    "trigger": pose["trigger"], "camera_yaw_locked": pose["yaw_locked"],
                },
                "geometry": {"type": "Point", "coordinates": path_coords[index]},
            })
        geojson = {"type": "FeatureCollection", "features": features}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return str(target)


EXPORTERS = {
    "qgc_plan": (export_qgc_plan, ".plan"),
    "qgc_wpl": (export_qgc_wpl, ".waypoints"),
    "dji_wpml": (export_dji_wpml, ".kmz"),
    "litchi": (export_litchi_csv, ".csv"),
    "kml": (export_kml, ".kml"),
    "geojson": (export_geojson, ".geojson"),
}


def export_all(directory: str | Path, mission: Any, stem: str = "mission") -> dict[str, str]:
    """Write every supported format, reporting per-format failures instead of aborting."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, (writer, suffix) in EXPORTERS.items():
        try:
            record_written(written, name, writer(root / f"{stem}{suffix}", mission))
        except Exception as exc:  # noqa: BLE001
            written[name] = f"failed: {type(exc).__name__}: {exc}"
    return written


def record_written(written: dict[str, str], name: str, result: str | list[str]) -> None:
    """Record a writer's output, naming each part when a format had to be split."""
    if isinstance(result, (list, tuple)):
        total = len(result)
        for index, part in enumerate(result, start=1):
            written[name if index == 1 else f"{name} (part {index} of {total})"] = str(part)
    else:
        written[name] = str(result)
