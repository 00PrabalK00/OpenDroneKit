"""Each mission export checked against the published spec for its format.

Three exporter defects were reported against the files OpenDroneKit writes:

* Litchi CSV: a photo written as action 5,0 (tilt camera) instead of 1,0 (take photo).
* QGroundControl .plan: firmwareType 12 (MAV_AUTOPILOT_PX4) on missions built for
  ArduPilot, whose MAV_AUTOPILOT value is 3 (MAV_AUTOPILOT_ARDUPILOTMEGA).
* DJI WPML: waylines.wpml without the required wpml:executeHeightMode.

Specs used, and what each test below holds the file to:

* Litchi CSV columns and action codes: https://www.litchiutilities.com/docs/litchiCsv.php
  (-1 none, 0 stay for ms, 1 take photo, 2 start recording, 3 stop recording,
  4 rotate aircraft, 5 tilt camera). Litchi's own help (https://flylitchi.com/help)
  caps "Stay for" at 32 seconds.
* QGC plan format: https://docs.qgroundcontrol.com/master/en/qgc-dev-guide/file_formats/plan.html
  (firmwareType is a MAV_AUTOPILOT, vehicleType a MAV_TYPE:
  https://mavlink.io/en/messages/minimal.html#MAV_AUTOPILOT). QGC publishes no JSON
  schema; the loader is the schema, so `qgc_plan_problems` below mirrors the key checks
  in QGC's PlanMasterController, MissionController (_loadJsonMissionFileV2),
  MissionItem, SimpleMissionItem, GeoFenceController and RallyPointController.
* DJI WPML: https://developer.dji.com/doc/cloud-api-tutorial/en/api-reference/dji-wpml/
  (source: github.com/dji-sdk/Cloud-API-Doc, docs/en/60.api-reference/00.dji-wpml/
  20.template-kml.md, 30.waylines-wpml.md, 40.common-element.md).
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from mission import exporters
from mission.exporters import (
    export_dji_wpml,
    export_litchi_csv,
    export_qgc_plan,
    export_qgc_wpl,
)

KML = "{http://www.opengis.net/kml/2.2}"
WPML = "{http://www.dji.com/wpmz/1.0.2}"

MAV_AUTOPILOT_ARDUPILOTMEGA = 3
MAV_AUTOPILOT_PX4 = 12
MAV_TYPE_FIXED_WING = 1
MAV_TYPE_QUADROTOR = 2


def make_plan(*, dwell_s: float = 0.0, yaws=(90.0, 200.0, 300.0, 10.0), **extra) -> dict:
    plan = {
        "altitude_m": 60.0,
        "gimbal_tilt_deg": -90.0,
        "waypoints": [],
        "flight_recipe": {
            "world_poses": [
                {
                    "lon": -81.7505 + index * 0.0004,
                    "lat": 41.3042,
                    "alt_m": 60.0 + index,
                    "yaw_deg": yaw,
                    "gimbal_pitch_deg": -45.0,
                    "dwell_s": dwell_s,
                    "trigger": True,
                    "camera_yaw_locked": True,
                }
                for index, yaw in enumerate(yaws)
            ],
            "capture": {"continuous_capture": False},
        },
        "safety_constraints": {
            "geofence": [
                [-81.7520, 41.3030], [-81.7480, 41.3030],
                [-81.7480, 41.3060], [-81.7520, 41.3060],
            ],
            "rally_points": [{"lon": -81.7510, "lat": 41.3035, "alt_m": 50.0}],
            "rth_altitude_m": 75.0,
        },
    }
    plan.update(extra)
    return plan


# ---------------------------------------------------------------------------
# Litchi CSV
# ---------------------------------------------------------------------------

LITCHI_HEADER_PREFIX = [
    "latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)", "rotationdir",
    "gimbalmode", "gimbalpitchangle",
]
LITCHI_HEADER_SUFFIX = [
    "altitudemode", "speed(m/s)", "poi_latitude", "poi_longitude", "poi_altitude(m)",
    "poi_altitudemode", "photo_timeinterval", "photo_distinterval",
]
NONE, STAY_MS, TAKE_PHOTO, TILT_CAMERA = "-1", "0", "1", "5"
LITCHI_ACTION_CODES = {"-1", "0", "1", "2", "3", "4", "5"}


def litchi_rows(tmp_path: Path, plan: dict) -> tuple[list[str], list[dict]]:
    path = export_litchi_csv(tmp_path / "m.csv", plan)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def litchi_actions(row: dict) -> list[tuple[str, str]]:
    pairs = [(row[f"actiontype{i}"], row[f"actionparam{i}"]) for i in range(1, 16)]
    return [pair for pair in pairs if pair[0] != NONE]


class TestLitchiCsv:
    def test_the_columns_are_litchis(self, tmp_path):
        header, _ = litchi_rows(tmp_path, make_plan())
        actions = [f"action{kind}{i}" for i in range(1, 16) for kind in ("type", "param")]
        assert header == LITCHI_HEADER_PREFIX + actions + LITCHI_HEADER_SUFFIX

    def test_a_photo_is_action_1_never_5(self, tmp_path):
        """The reported bug: 5,0 is "tilt camera to 0 deg", not a photograph."""
        _, rows = litchi_rows(tmp_path, make_plan())
        for row in rows:
            assert litchi_actions(row) == [(TAKE_PHOTO, "0")]
            assert all(code in LITCHI_ACTION_CODES for code, _ in litchi_actions(row))
            assert (TILT_CAMERA, "0") not in litchi_actions(row)

    def test_a_dwell_is_a_stay_in_milliseconds_before_the_photo(self, tmp_path):
        _, rows = litchi_rows(tmp_path, make_plan(dwell_s=2.5))
        for row in rows:
            assert litchi_actions(row) == [(STAY_MS, "2500"), (TAKE_PHOTO, "0")]

    def test_a_dwell_past_litchis_32_s_stay_is_split_not_truncated(self, tmp_path):
        """Litchi caps one "Stay for" at 32 s; a 70 s hold is three stays, then the photo."""
        _, rows = litchi_rows(tmp_path, make_plan(dwell_s=70.0))
        for row in rows:
            actions = litchi_actions(row)
            stays = [int(param) for code, param in actions if code == STAY_MS]
            assert all(0 < ms <= 32_000 for ms in stays), stays
            assert sum(stays) == 70_000
            assert actions[-1] == (TAKE_PHOTO, "0")


# ---------------------------------------------------------------------------
# QGroundControl .plan
# ---------------------------------------------------------------------------


def qgc_plan_problems(payload: dict) -> list[str]:
    """What QGroundControl's plan loader would reject, key for key."""
    problems: list[str] = []

    def need(obj: dict, key: str, kind, where: str, required: bool = True) -> None:
        if key not in obj:
            if required:
                problems.append(f"{where}: missing {key}")
            return
        value = obj[key]
        ok = (value is None) if kind is None else (
            isinstance(value, kind) and not (kind in ((int, float),) and isinstance(value, bool)))
        if not ok:
            problems.append(f"{where}: {key} is {type(value).__name__}")

    number = (int, float)
    # PlanMasterController::loadFromJson
    if payload.get("fileType") != "Plan":
        problems.append("fileType is not Plan")
    if payload.get("version") != 1:
        problems.append("plan version is not 1")
    for key in ("mission", "geoFence", "rallyPoints"):
        need(payload, key, dict, "plan")
    mission = payload.get("mission") or {}
    # MissionController::_loadJsonMissionFileV2
    if mission.get("version") != 2:
        problems.append("mission version is not 2")
    need(mission, "plannedHomePosition", list, "mission")
    need(mission, "items", list, "mission")
    need(mission, "firmwareType", number, "mission")
    for key in ("vehicleType", "cruiseSpeed", "hoverSpeed", "globalPlanAltitudeMode"):
        need(mission, key, number, "mission", required=False)
    home = mission.get("plannedHomePosition") or []
    if len(home) != 3 or not all(isinstance(v, number) for v in home):
        problems.append("plannedHomePosition is not [lat, lon, alt]")
    for index, item in enumerate(mission.get("items") or []):
        where = f"item {index}"
        # MissionItem::load (V2) and SimpleMissionItem::load
        need(item, "type", str, where)
        if item.get("type") != "SimpleItem":
            problems.append(f"{where}: type is not SimpleItem")
        need(item, "frame", number, where)
        need(item, "command", number, where)
        need(item, "params", list, where)
        need(item, "autoContinue", bool, where)
        need(item, "doJumpId", number, where, required=False)
        need(item, "AltitudeMode", number, where)
        need(item, "Altitude", number, where)
        need(item, "AMSLAltAboveTerrain", None, where)
        params = item.get("params") or []
        if len(params) != 7 or not all(p is None or isinstance(p, number) for p in params):
            problems.append(f"{where}: params is not 7 numbers")
    fence = payload.get("geoFence") or {}
    # GeoFenceController::load
    if fence.get("version") != 2:
        problems.append("geoFence version is not 2")
    need(fence, "circles", list, "geoFence")
    need(fence, "polygons", list, "geoFence")
    for index, polygon in enumerate(fence.get("polygons") or []):
        need(polygon, "inclusion", bool, f"polygon {index}")
        need(polygon, "polygon", list, f"polygon {index}")
        need(polygon, "version", number, f"polygon {index}")
    rally = payload.get("rallyPoints") or {}
    # RallyPointController::load
    if rally.get("version") != 2:
        problems.append("rallyPoints version is not 2")
    need(rally, "points", list, "rallyPoints")
    return problems


def write_plan(tmp_path: Path, plan: dict, **kwargs) -> dict:
    return json.loads(Path(export_qgc_plan(tmp_path / "m.plan", plan, **kwargs)).read_text())


class TestQgcPlan:
    def test_an_ardupilot_mission_says_ardupilot(self, tmp_path):
        """The reported bug: 12 is MAV_AUTOPILOT_PX4. ODK flies ArduPilot (3) by default."""
        mission = write_plan(tmp_path, make_plan())["mission"]
        assert mission["firmwareType"] == MAV_AUTOPILOT_ARDUPILOTMEGA
        assert mission["vehicleType"] == MAV_TYPE_QUADROTOR

    @pytest.mark.parametrize("key, value, expected", [
        ("autopilot", "px4", MAV_AUTOPILOT_PX4),
        ("autopilot", "PX4", MAV_AUTOPILOT_PX4),
        ("autopilot", "ardupilot", MAV_AUTOPILOT_ARDUPILOTMEGA),
        ("firmware", "arducopter", MAV_AUTOPILOT_ARDUPILOTMEGA),
        ("autopilot", 12, MAV_AUTOPILOT_PX4),
    ])
    def test_the_firmware_follows_the_missions_autopilot(self, tmp_path, key, value, expected):
        assert write_plan(tmp_path, make_plan(**{key: value}))["mission"]["firmwareType"] == expected

    @pytest.mark.parametrize("airframe, expected", [
        ("quadrotor", MAV_TYPE_QUADROTOR), ("multirotor", MAV_TYPE_QUADROTOR),
        ("hexarotor", 13), ("octorotor", 14), ("fixed_wing", MAV_TYPE_FIXED_WING),
        ("plane", MAV_TYPE_FIXED_WING),
    ])
    def test_the_vehicle_type_follows_the_airframe(self, tmp_path, airframe, expected):
        assert write_plan(tmp_path, make_plan(airframe=airframe))["mission"]["vehicleType"] == expected

    def test_an_explicit_argument_still_wins(self, tmp_path):
        mission = write_plan(tmp_path, make_plan(), firmware_type=MAV_AUTOPILOT_PX4, vehicle_type=1)["mission"]
        assert (mission["firmwareType"], mission["vehicleType"]) == (MAV_AUTOPILOT_PX4, 1)

    def test_an_unknown_autopilot_is_refused_with_a_reason(self, tmp_path):
        with pytest.raises(ValueError, match="autopilot"):
            write_plan(tmp_path, make_plan(autopilot="betaflight"))
        with pytest.raises(ValueError, match="airframe"):
            write_plan(tmp_path, make_plan(airframe="blimp"))

    def test_the_file_passes_qgroundcontrols_loader_checks(self, tmp_path):
        assert qgc_plan_problems(write_plan(tmp_path, make_plan(dwell_s=2.0))) == []

    def test_the_plan_and_the_waypoints_file_carry_the_same_commands(self, tmp_path):
        """Round trip the WPL through pymavlink's own loader and compare with the .plan."""
        mavwp = pytest.importorskip("pymavlink.mavwp")
        plan = make_plan(dwell_s=2.0)
        items = write_plan(tmp_path, plan)["mission"]["items"]
        loader = mavwp.MAVWPLoader()
        loader.load(export_qgc_wpl(tmp_path / "m.waypoints", plan))
        loaded = [loader.wp(i) for i in range(loader.count())]
        assert [wp.command for wp in loaded[1:]] == [item["command"] for item in items]
        for wp, item in zip(loaded[1:], items):
            assert wp.frame == item["frame"]
            if item["frame"] != 2:
                assert wp.x == pytest.approx(item["params"][4], abs=1e-7)
                assert wp.y == pytest.approx(item["params"][5], abs=1e-7)
                assert wp.z == pytest.approx(item["params"][6], abs=1e-4)

    def test_the_api_exports_an_ardupilot_plan(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("qgc", root_dir=str(tmp_path / "project"))
        api = Api(session)
        assert api.set_aoi([[-81.7535, 41.3035], [-81.7521, 41.3035],
                            [-81.7521, 41.3044], [-81.7535, 41.3044]])["ok"]
        assert api.plan_mission({"template": "grid"})["ok"]
        written = api.export_mission(["qgc_plan"], str(tmp_path / "out"))["written"]
        payload = json.loads(Path(written["qgc_plan"]).read_text())
        assert payload["mission"]["firmwareType"] == MAV_AUTOPILOT_ARDUPILOTMEGA
        assert qgc_plan_problems(payload) == []


# ---------------------------------------------------------------------------
# DJI WPML
# ---------------------------------------------------------------------------


def read_kmz(path: str) -> tuple[ET.Element, ET.Element]:
    with zipfile.ZipFile(path) as archive:
        return (ET.fromstring(archive.read("wpmz/template.kml")),
                ET.fromstring(archive.read("wpmz/waylines.wpml")))


def children(element: ET.Element) -> set[str]:
    return {child.tag.replace(WPML, "wpml:").replace(KML, "") for child in element}


def text(element: ET.Element, tag: str) -> str:
    found = element.find(f"{WPML}{tag}")
    assert found is not None, f"missing wpml:{tag}"
    return found.text or ""


# Required elements per the DJI WPML reference (see module docstring).
MISSION_CONFIG_REQUIRED = {
    "wpml:flyToWaylineMode", "wpml:finishAction", "wpml:exitOnRCLost",
    "wpml:takeOffSecurityHeight", "wpml:globalTransitionalSpeed", "wpml:droneInfo",
    "wpml:payloadInfo",
}
TEMPLATE_FOLDER_REQUIRED = {
    "wpml:templateType", "wpml:templateId", "wpml:autoFlightSpeed",
    "wpml:waylineCoordinateSysParam", "wpml:globalWaypointTurnMode", "wpml:gimbalPitchMode",
    "wpml:globalHeight",
}
TEMPLATE_PLACEMARK_REQUIRED = {
    "Point", "wpml:index", "wpml:useGlobalHeight", "wpml:ellipsoidHeight", "wpml:height",
    "wpml:useGlobalSpeed", "wpml:waypointSpeed", "wpml:useGlobalHeadingParam",
    "wpml:waypointHeadingParam", "wpml:useGlobalTurnParam", "wpml:waypointTurnParam",
    "wpml:gimbalPitchAngle",
}
WAYLINES_FOLDER_REQUIRED = {
    "wpml:templateId", "wpml:waylineId", "wpml:autoFlightSpeed", "wpml:executeHeightMode",
}
WAYLINES_PLACEMARK_REQUIRED = {
    "Point", "wpml:index", "wpml:executeHeight", "wpml:waypointSpeed",
    "wpml:waypointHeadingParam", "wpml:waypointTurnParam",
}
CONTINUITY_TURNS = {"toPointAndStopWithContinuityCurvature", "toPointAndPassWithContinuityCurvature"}
EXECUTE_HEIGHT_MODES = {"WGS84", "relativeToStartPoint", "realTimeFollowSurface"}
TEMPLATE_HEIGHT_MODES = {"EGM96", "relativeToStartPoint", "aboveGroundLevel", "realTimeFollowSurface"}


def assert_common_elements(root: ET.Element) -> None:
    """Common-element rules: heading, turn, action group, gimbal and photo params."""
    for heading in root.iter(f"{WPML}waypointHeadingParam"):
        mode = text(heading, "waypointHeadingMode")
        assert mode in {"followWayline", "manually", "fixed", "smoothTransition", "towardPOI"}
        assert text(heading, "waypointHeadingPathMode") in {"clockwise", "counterClockwise", "followBadArc"}
        if mode == "smoothTransition":
            assert -180.0 <= float(text(heading, "waypointHeadingAngle")) <= 180.0
    for turn in root.iter(f"{WPML}waypointTurnParam"):
        if text(turn, "waypointTurnMode") != "toPointAndStopWithDiscontinuityCurvature":
            assert float(text(turn, "waypointTurnDampingDist")) > 0.0
    for action in root.iter(f"{WPML}action"):
        func = text(action, "actionActuatorFunc")
        params = action.find(f"{WPML}actionActuatorFuncParam")
        if func == "gimbalRotate":
            assert children(params) >= {
                "wpml:payloadPositionIndex", "wpml:gimbalHeadingYawBase", "wpml:gimbalRotateMode",
                "wpml:gimbalPitchRotateEnable", "wpml:gimbalPitchRotateAngle",
                "wpml:gimbalRollRotateEnable", "wpml:gimbalRollRotateAngle",
                "wpml:gimbalYawRotateEnable", "wpml:gimbalYawRotateAngle",
                "wpml:gimbalRotateTimeEnable", "wpml:gimbalRotateTime"}
            assert text(params, "gimbalHeadingYawBase") == "north"
        elif func == "takePhoto":
            assert children(params) >= {"wpml:payloadPositionIndex", "wpml:fileSuffix",
                                         "wpml:payloadLensIndex", "wpml:useGlobalPayloadLensIndex"}
        elif func == "hover":
            assert float(text(params, "hoverTime")) > 0.0
    for group in root.iter(f"{WPML}actionGroup"):
        assert children(group) >= {"wpml:actionGroupId", "wpml:actionGroupStartIndex",
                                   "wpml:actionGroupEndIndex", "wpml:actionGroupMode",
                                   "wpml:actionTrigger"}


class TestDjiWpml:
    def test_waylines_declares_its_execute_height_mode(self, tmp_path):
        """The reported bug: wpml:executeHeightMode is required in waylines.wpml."""
        template, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", make_plan()))
        folder = waylines.find(f"{KML}Document/{KML}Folder")
        assert text(folder, "executeHeightMode") == "relativeToStartPoint"
        coordinate = template.find(f"{KML}Document/{KML}Folder/{WPML}waylineCoordinateSysParam")
        assert text(coordinate, "heightMode") == "relativeToStartPoint"

    def test_template_kml_has_every_required_element(self, tmp_path):
        template, _ = read_kmz(export_dji_wpml(tmp_path / "m.kmz", make_plan(dwell_s=2.0)))
        document = template.find(f"{KML}Document")
        assert children(document.find(f"{WPML}missionConfig")) >= MISSION_CONFIG_REQUIRED
        folder = document.find(f"{KML}Folder")
        assert children(folder) >= TEMPLATE_FOLDER_REQUIRED
        assert text(folder, "templateType") == "waypoint"
        assert text(folder, "gimbalPitchMode") == "usePointSetting"
        coordinate = folder.find(f"{WPML}waylineCoordinateSysParam")
        assert text(coordinate, "coordinateMode") == "WGS84"
        assert text(coordinate, "heightMode") in TEMPLATE_HEIGHT_MODES
        if text(folder, "globalWaypointTurnMode") in CONTINUITY_TURNS:
            assert "wpml:globalUseStraightLine" in children(folder)
        placemarks = folder.findall(f"{KML}Placemark")
        assert [int(text(p, "index")) for p in placemarks] == list(range(len(placemarks)))
        for placemark in placemarks:
            present = children(placemark)
            assert present >= TEMPLATE_PLACEMARK_REQUIRED, TEMPLATE_PLACEMARK_REQUIRED - present
            # executeHeight belongs to waylines.wpml only.
            assert "wpml:executeHeight" not in present
            assert text(placemark, "useGlobalHeight") == "0"
            assert float(text(placemark, "gimbalPitchAngle")) == -45.0
        assert_common_elements(template)

    def test_waylines_wpml_has_every_required_element(self, tmp_path):
        template, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", make_plan(dwell_s=2.0)))
        document = waylines.find(f"{KML}Document")
        config = document.find(f"{WPML}missionConfig")
        assert children(config) >= MISSION_CONFIG_REQUIRED | {"wpml:globalRTHHeight"}
        assert float(text(config, "globalRTHHeight")) == 75.0
        if text(config, "exitOnRCLost") == "executeLostAction":
            assert text(config, "executeRCLostAction") in {"goBack", "landing", "hover"}
        folder = document.find(f"{KML}Folder")
        assert children(folder) >= WAYLINES_FOLDER_REQUIRED
        assert text(folder, "executeHeightMode") in EXECUTE_HEIGHT_MODES
        # templateId ties the wayline to its template.
        assert text(folder, "templateId") == text(template.find(f"{KML}Document/{KML}Folder"), "templateId")
        placemarks = folder.findall(f"{KML}Placemark")
        assert [int(text(p, "index")) for p in placemarks] == list(range(len(placemarks)))
        for placemark, pose_alt in zip(placemarks, (60.0, 61.0, 62.0, 63.0)):
            present = children(placemark)
            assert present >= WAYLINES_PLACEMARK_REQUIRED, WAYLINES_PLACEMARK_REQUIRED - present
            assert float(text(placemark, "executeHeight")) == pose_alt
            turn = text(placemark.find(f"{WPML}waypointTurnParam"), "waypointTurnMode")
            if turn in CONTINUITY_TURNS:
                assert "wpml:useStraightLine" in present
        assert_common_elements(waylines)

    def test_headings_are_in_djis_range(self, tmp_path):
        """waypointHeadingAngle is [-180, 180]; 200 deg is written as -160."""
        _, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", make_plan()))
        angles = [float(e.text) for e in waylines.iter(f"{WPML}waypointHeadingAngle")]
        assert angles == [90.0, -160.0, -60.0, 10.0]

    def test_template_and_waylines_heights_describe_the_same_point(self, tmp_path):
        template, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", make_plan()))
        t_marks = template.findall(f"{KML}Document/{KML}Folder/{KML}Placemark")
        w_marks = waylines.findall(f"{KML}Document/{KML}Folder/{KML}Placemark")
        for t, w in zip(t_marks, w_marks):
            # relativeToStartPoint in both: the same number on both reference planes.
            assert float(text(t, "height")) == float(text(w, "executeHeight"))
            assert float(text(t, "ellipsoidHeight")) == float(text(t, "height"))

    def test_an_amsl_plan_without_a_geoid_model_stays_relative(self, tmp_path, monkeypatch):
        """No geoid grid, no ellipsoid height: the export keeps the plan's own frame
        rather than writing an orthometric number where DJI expects an ellipsoidal one."""
        monkeypatch.setattr(exporters, "_geoid_undulation", lambda lon, lat: None)
        plan = make_plan(altitude_frame="amsl", home_elevation_amsl_m=200.0)
        template, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", plan))
        folder = waylines.find(f"{KML}Document/{KML}Folder")
        assert text(folder, "executeHeightMode") == "relativeToStartPoint"
        coordinate = template.find(f"{KML}Document/{KML}Folder/{WPML}waylineCoordinateSysParam")
        assert text(coordinate, "heightMode") == "relativeToStartPoint"
        assert float(text(folder.find(f"{KML}Placemark"), "executeHeight")) == 60.0

    def test_an_amsl_plan_with_a_geoid_model_is_absolute(self, tmp_path, monkeypatch):
        """With a geoid undulation N: template EGM96 height H, ellipsoid H + N; the
        wayline flies WGS84 ellipsoid heights. N here is a test value, not a measurement."""
        monkeypatch.setattr(exporters, "_geoid_undulation", lambda lon, lat: -33.5)
        plan = make_plan(altitude_frame="amsl", home_elevation_amsl_m=200.0)
        template, waylines = read_kmz(export_dji_wpml(tmp_path / "m.kmz", plan))
        t_folder = template.find(f"{KML}Document/{KML}Folder")
        w_folder = waylines.find(f"{KML}Document/{KML}Folder")
        assert text(t_folder.find(f"{WPML}waylineCoordinateSysParam"), "heightMode") == "EGM96"
        assert text(w_folder, "executeHeightMode") == "WGS84"
        t_first, w_first = t_folder.find(f"{KML}Placemark"), w_folder.find(f"{KML}Placemark")
        assert float(text(t_first, "height")) == 260.0
        assert float(text(t_first, "ellipsoidHeight")) == 226.5
        assert float(text(w_first, "executeHeight")) == 226.5

    def test_the_geoid_lookup_refuses_rather_than_guesses(self):
        """Without PROJ's EGM96 grid, pyproj silently returns the input height. The
        lookup must say "unknown" (None) then, never an undulation of 0."""
        value = exporters._geoid_undulation(-81.75, 41.30)
        assert value is None or (math.isfinite(value) and -110.0 < value < 90.0 and value != 0.0)


@pytest.mark.parametrize("template", ["grid", "orbit", "tower_mapping", "facade"])
def test_planned_missions_export_spec_complete_wpml(tmp_path, template):
    """Every element check above, on files planned and exported through the real Api."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("wpml", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi([[-81.7535, 41.3035], [-81.7521, 41.3035],
                        [-81.7521, 41.3044], [-81.7535, 41.3044]])["ok"]
    assert api.plan_mission({"template": template})["ok"]
    written = api.export_mission(["dji_wpml"], str(tmp_path / "out"))["written"]
    template_root, waylines = read_kmz(written["dji_wpml"])
    t_folder = template_root.find(f"{KML}Document/{KML}Folder")
    w_folder = waylines.find(f"{KML}Document/{KML}Folder")
    assert children(t_folder) >= TEMPLATE_FOLDER_REQUIRED
    assert children(w_folder) >= WAYLINES_FOLDER_REQUIRED
    assert text(w_folder, "executeHeightMode") in EXECUTE_HEIGHT_MODES
    for placemark in t_folder.findall(f"{KML}Placemark"):
        assert children(placemark) >= TEMPLATE_PLACEMARK_REQUIRED
    for placemark in w_folder.findall(f"{KML}Placemark"):
        assert children(placemark) >= WAYLINES_PLACEMARK_REQUIRED
    assert_common_elements(template_root)
    assert_common_elements(waylines)
