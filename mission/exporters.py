"""Mission exporters for real ground-control software and autopilots.

The planner already computes yaw, gimbal pitch, dwell time, and camera triggers for
every viewpoint. The original QGC WPL writer discarded all of it and emitted bare
NAV_WAYPOINT rows, which is why exported missions flew the route but captured nothing
useful. These exporters carry the full command stream into each target format.

Formats: MAVLink mission items, QGroundControl `.plan`, QGC WPL 110, DJI WPML `.kmz`,
Litchi CSV, and KML.
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
MAV_CMD_DO_DIGICAM_CONTROL = 203
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_CMD_NAV_FENCE_RETURN_POINT = 5000
MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION = 5001
MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION = 5002
MAV_CMD_NAV_RALLY_POINT = 5100

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


def _viewpoints(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover per-viewpoint pose detail, falling back to bare waypoints.

    The flight recipe carries yaw/gimbal/dwell/trigger; `waypoints` is only geometry.
    Preferring the recipe is what keeps the capture behaviour in the export.
    """
    recipe = plan.get("flight_recipe") or {}
    poses = recipe.get("world_poses") or recipe.get("poses") or []
    default_alt = float(plan.get("altitude_m", 50.0) or 50.0)
    default_gimbal = float(plan.get("gimbal_tilt_deg", -90.0) or -90.0)

    viewpoints: list[dict[str, Any]] = []
    if poses:
        for pose in poses:
            viewpoints.append(
                {
                    "lon": float(pose.get("lon", pose.get("longitude", 0.0))),
                    "lat": float(pose.get("lat", pose.get("latitude", 0.0))),
                    "alt": float(pose.get("alt_m", pose.get("alt", pose.get("altitude_m", default_alt)))),
                    "yaw_deg": float(pose.get("yaw_deg", 0.0)),
                    "gimbal_pitch_deg": float(pose.get("gimbal_pitch_deg", default_gimbal)),
                    "dwell_s": float(pose.get("dwell_s", 0.0)),
                    "trigger": bool(pose.get("trigger", True)),
                    "speed_mps": float(pose.get("speed_mps", 0.0) or 0.0),
                    "yaw_locked": bool(pose.get("camera_yaw_locked", False)),
                }
            )
        return viewpoints

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
    items: list[MissionItem] = []
    seq = 0

    def add(command: int, **kwargs: Any) -> None:
        nonlocal seq
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
        # the mount mode, which is why the mode is written to `altitude` here.
        add(
            MAV_CMD_DO_MOUNT_CONTROL,
            frame=MAV_FRAME_MISSION,
            param1=gimbal,
            param2=0.0,
            param3=pose["yaw_deg"] if pose["yaw_locked"] else 0.0,
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
                latitude=viewpoints[-1]["lat"],
                longitude=viewpoints[-1]["lon"],
                altitude=0.0,
                label="land",
            )
        else:
            add(MAV_CMD_NAV_RETURN_TO_LAUNCH, frame=MAV_FRAME_MISSION, altitude=rth_alt, label="return to launch")

    return items


def build_fence_items(mission: Any) -> list[MissionItem]:
    """Geofence inclusion and no-fly exclusion polygons as MAVLink fence items."""
    plan = _plan_dict(mission)
    constraints = plan.get("safety_constraints") or {}
    items: list[MissionItem] = []
    seq = 0

    def add_ring(ring: Sequence[Sequence[float]], command: int) -> None:
        nonlocal seq
        vertices = [v for v in ring if len(v) >= 2]
        if len(vertices) >= 3 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        if len(vertices) < 3:
            return
        for vertex in vertices:
            items.append(
                MissionItem(
                    seq=seq,
                    command=command,
                    frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    param1=float(len(vertices)),
                    latitude=float(vertex[1]),
                    longitude=float(vertex[0]),
                    label="fence",
                )
            )
            seq += 1

    geofence = constraints.get("geofence") or plan.get("polygon") or []
    if geofence:
        add_ring(geofence, MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION)
    for polygon in constraints.get("no_fly_polygons") or []:
        add_ring(polygon, MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION)
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
    """QGC WPL 110 text mission, carrying every command rather than positions only."""
    items = build_mission_items(mission)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = ["QGC WPL 110"]
    for item in items:
        def clean(value: float) -> float:
            return 0.0 if isinstance(value, float) and math.isnan(value) else float(value)

        lines.append(
            "\t".join(
                [
                    str(item.seq),
                    str(item.current),
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
    fence_items = build_fence_items(mission)
    rally_items = build_rally_items(mission)
    constraints = plan.get("safety_constraints") or {}

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
                "AltitudeMode": 1,
                "autoContinue": bool(item.autocontinue),
                "command": item.command,
                "doJumpId": index + 1,
                "frame": item.frame,
                "params": params,
                "type": "SimpleItem",
            }
        )

    first = items[0]
    polygons: list[dict[str, Any]] = []
    inclusion = [i for i in fence_items if i.command == MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION]
    exclusion = [i for i in fence_items if i.command == MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION]
    if inclusion:
        polygons.append({"inclusion": True, "polygon": [[i.latitude, i.longitude] for i in inclusion], "version": 1})
    if exclusion:
        polygons.append({"inclusion": False, "polygon": [[i.latitude, i.longitude] for i in exclusion], "version": 1})

    payload = {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": polygons, "version": 2},
        "groundStation": "OpenDroneKit",
        "mission": {
            "cruiseSpeed": float(plan.get("cruise_speed_mps", 12.0) or 12.0),
            "firmwareType": int(firmware_type),
            "globalPlanAltitudeMode": 1,
            "hoverSpeed": float(plan.get("hover_speed_mps", 5.0) or 5.0),
            "items": simple_items,
            "plannedHomePosition": [first.latitude, first.longitude, float(constraints.get("home_altitude_m", 0.0) or 0.0)],
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


def export_litchi_csv(path: str | Path, mission: Any) -> str:
    """Litchi Mission Hub CSV, with per-waypoint gimbal pitch, heading, and capture."""
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    if not viewpoints:
        raise ValueError("Mission plan contains no waypoints to export.")

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
    rows = [",".join(header)]
    for pose in viewpoints:
        actions: list[str] = []
        # Litchi action type 1 is "stay for", 5 is "take photo".
        if pose["dwell_s"] > 0.0:
            actions.extend(["1", str(int(pose["dwell_s"] * 1000))])
        if spacing <= 0.0 and pose["trigger"]:
            actions.extend(["5", "0"])
        while len(actions) < 30:
            actions.extend(["-1", "0"])
        actions = actions[:30]

        row = [
            f"{pose['lat']:.8f}", f"{pose['lon']:.8f}", f"{pose['alt']:.2f}",
            f"{pose['yaw_deg'] % 360.0:.1f}", "0.2", "0",
            "2", f"{pose['gimbal_pitch_deg']:.1f}",
        ]
        row.extend(actions)
        row.extend([
            "1", f"{pose['speed_mps'] or 5.0:.2f}", "0", "0", "0", "0",
            "0", f"{spacing:.2f}" if spacing > 0 else "0",
        ])
        rows.append(",".join(row))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(target)


def export_kml(path: str | Path, mission: Any, name: str = "OpenDroneKit Mission") -> str:
    """KML with the flight path, numbered waypoints, geofence, and no-fly zones."""
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    constraints = plan.get("safety_constraints") or {}

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
        ET.SubElement(line, "altitudeMode").text = "relativeToGround"
        ET.SubElement(line, "tessellate").text = "1"
        ET.SubElement(line, "coordinates").text = " ".join(
            f"{p['lon']:.8f},{p['lat']:.8f},{p['alt']:.2f}" for p in viewpoints
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
        ET.SubElement(point, "altitudeMode").text = "relativeToGround"
        ET.SubElement(point, "coordinates").text = f"{pose['lon']:.8f},{pose['lat']:.8f},{pose['alt']:.2f}"

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

    geofence = constraints.get("geofence") or plan.get("polygon") or []
    if geofence:
        add_ring(geofence, "Geofence", "fence")
    for index, polygon in enumerate(constraints.get("no_fly_polygons") or [], start=1):
        add_ring(polygon, f"No-fly zone {index}", "nofly")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(kml).write(target, encoding="utf-8", xml_declaration=True)
    return str(target)


def export_dji_wpml(path: str | Path, mission: Any, *, drone_enum: int = 68, payload_enum: int = 66) -> str:
    """DJI WPML `.kmz` for Pilot 2 / DJI Fly waypoint missions.

    A WPML archive is a zip holding `wpmz/template.kml` (the human-editable plan) and
    `wpmz/waylines.wpml` (the executable route). Both carry the DJI wpml namespace.
    Default enums describe a Mavic 3 Enterprise class airframe.
    """
    plan = _plan_dict(mission)
    viewpoints = _viewpoints(plan)
    if not viewpoints:
        raise ValueError("Mission plan contains no waypoints to export.")
    spacing = _capture_spacing(plan)
    constraints = plan.get("safety_constraints") or {}
    speed = max(1.0, float(plan.get("cruise_speed_mps", 8.0) or 8.0))
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
        ET.SubElement(drone_info, wpml("droneSubEnumValue")).text = "0"
        payload_info = ET.SubElement(config, wpml("payloadInfo"))
        ET.SubElement(payload_info, wpml("payloadEnumValue")).text = str(int(payload_enum))
        ET.SubElement(payload_info, wpml("payloadPositionIndex")).text = "0"

        folder = ET.SubElement(document, f"{{{kml_ns}}}Folder")
        ET.SubElement(folder, wpml("templateType")).text = "waypoint"
        ET.SubElement(folder, wpml("templateId")).text = "0"
        if not is_template:
            ET.SubElement(folder, wpml("waylineId")).text = "0"
            ET.SubElement(folder, wpml("autoFlightSpeed")).text = f"{speed:.1f}"
        coordinate = ET.SubElement(folder, wpml("waylineCoordinateSysParam"))
        ET.SubElement(coordinate, wpml("coordinateMode")).text = "WGS84"
        ET.SubElement(coordinate, wpml("heightMode")).text = "relativeToStartPoint"
        ET.SubElement(folder, wpml("autoFlightSpeed")).text = f"{speed:.1f}"

        for index, pose in enumerate(viewpoints):
            placemark = ET.SubElement(folder, f"{{{kml_ns}}}Placemark")
            point = ET.SubElement(placemark, f"{{{kml_ns}}}Point")
            ET.SubElement(point, f"{{{kml_ns}}}coordinates").text = f"{pose['lon']:.8f},{pose['lat']:.8f}"
            ET.SubElement(placemark, wpml("index")).text = str(index)
            ET.SubElement(placemark, wpml("executeHeight")).text = f"{pose['alt']:.2f}"
            ET.SubElement(placemark, wpml("waypointSpeed")).text = f"{pose['speed_mps'] or speed:.1f}"

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


EXPORTERS = {
    "qgc_plan": (export_qgc_plan, ".plan"),
    "qgc_wpl": (export_qgc_wpl, ".waypoints"),
    "dji_wpml": (export_dji_wpml, ".kmz"),
    "litchi": (export_litchi_csv, ".csv"),
    "kml": (export_kml, ".kml"),
}


def export_all(directory: str | Path, mission: Any, stem: str = "mission") -> dict[str, str]:
    """Write every supported format, reporting per-format failures instead of aborting."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, (writer, suffix) in EXPORTERS.items():
        try:
            written[name] = writer(root / f"{stem}{suffix}", mission)
        except Exception as exc:  # noqa: BLE001
            written[name] = f"failed: {exc}"
    return written
