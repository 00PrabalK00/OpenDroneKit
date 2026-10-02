"""What the planner decides must be what the exported file tells the aircraft.

An end-to-end audit drove every template through the Api the UI uses and parsed every
file it wrote. The plans were right and the files were not:

- every export read `flight_recipe.world_poses`, which no plan carries, so a tower orbit,
  a facade and a panorama all exported as nadir waypoints with no yaw;
- operator speed and dwell never reached any file (.plan cruise 12, WPML 8, Litchi 5);
- the .waypoints file put TAKEOFF in the home slot, where ArduPilot overwrites it;
- two no-fly zones became one self-intersecting exclusion polygon;
- Litchi got 4,879 rows although it accepts 99;
- GeoJSON was not an export format at all.

These tests plan through the real Api, export through the real Api, and read the files
back. Nothing here injects poses by hand -- that is how the old tests passed.
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]
NO_FLY = [
    [[-81.7531, 41.30375], [-81.7527, 41.30375], [-81.7527, 41.30405], [-81.7531, 41.30405]],
    [[-81.7525, 41.3041], [-81.7523, 41.3041], [-81.7523, 41.3043], [-81.7525, 41.3043]],
]
WPML = "{http://www.dji.com/wpmz/1.0.2}"

CONDITION_YAW, DO_CHANGE_SPEED, DO_MOUNT_CONTROL = 115, 178, 205
NAV_WAYPOINT, NAV_LOITER_TIME, NAV_TAKEOFF = 16, 19, 22


@pytest.fixture
def api(tmp_path) -> Api:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("exports", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi(AOI)["ok"]
    return api


def plan_and_export(api: Api, tmp_path: Path, options: dict) -> tuple[dict, dict]:
    planned = api.plan_mission(options)
    assert planned["ok"], planned.get("error")
    exported = api.export_mission(None, str(tmp_path / "out" / options.get("template", "grid")))
    assert exported["ok"], exported.get("error")
    failed = {k: v for k, v in exported["written"].items() if str(v).startswith("failed")}
    assert not failed, failed
    return planned, exported["written"]


def waypoint_props(planned: dict) -> list[dict]:
    return [f["properties"] for f in planned["geojson"]["features"]
            if f["properties"].get("role") == "waypoint"]


def plan_items(written: dict) -> list[dict]:
    return json.loads(Path(written["qgc_plan"]).read_text())["mission"]["items"]


class TestTheCameraPoseReachesTheFile:
    """The headline defect: structure missions exported as nadir photographs."""

    @pytest.mark.parametrize("template", ["tower_mapping", "orbit", "panorama", "facade",
                                          "box_inspection", "wind_turbine", "bubble_360",
                                          "closed_loop", "multi_facade", "dome_inspection",
                                          "facade_mapping"])
    def test_every_yaw_locked_pose_gets_its_yaw_and_pitch(self, api, tmp_path, template):
        planned, written = plan_and_export(api, tmp_path, {"template": template, "altitude_m": 50})
        props = waypoint_props(planned)
        locked = [p for p in props if p.get("camera_yaw_locked")]
        assert locked, f"{template} planned no yaw-locked poses"

        items = plan_items(written)
        yaws = [i for i in items if i["command"] == CONDITION_YAW]
        assert len(yaws) == len(locked), "a planned heading was dropped on export"
        assert [round(i["params"][0], 1) for i in yaws] == [
            round(p["yaw_deg"] % 360.0, 1) for p in locked]

        pitches = [round(i["params"][0], 1) for i in items if i["command"] == DO_MOUNT_CONTROL]
        assert pitches == [round(p["gimbal_pitch_deg"], 1) for p in props]
        assert set(pitches) != {-90.0}, f"{template} exported as nadir photographs"

    def test_litchi_carries_the_heading_and_pitch(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "orbit", "altitude_m": 50})
        rows = list(csv.DictReader(open(written["litchi"])))
        props = waypoint_props(planned)
        assert [float(r["heading(deg)"]) for r in rows] == [
            round(p["yaw_deg"] % 360.0, 1) for p in props]
        assert {float(r["gimbalpitchangle"]) for r in rows} == {-35.0}

    def test_dji_wpml_carries_the_pitch_and_heading(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "tower_mapping", "altitude_m": 50})
        with zipfile.ZipFile(written["dji_wpml"]) as archive:
            root = ET.fromstring(archive.read("wpmz/waylines.wpml"))
        pitches = {float(e.text) for e in root.iter(f"{WPML}gimbalPitchRotateAngle")}
        modes = {e.text for e in root.iter(f"{WPML}waypointHeadingMode")}
        assert pitches - {-90.0}, "every DJI waypoint was nadir"
        assert "smoothTransition" in modes

    def test_panorama_looks_at_the_horizon_by_default(self, api, tmp_path):
        """Nine identical nadir photographs is not a panorama."""
        planned, written = plan_and_export(api, tmp_path, {"template": "panorama", "altitude_m": 50})
        pitches = {p["gimbal_pitch_deg"] for p in waypoint_props(planned)}
        yaws = {round(p["yaw_deg"]) for p in waypoint_props(planned)}
        assert pitches == {0.0}
        assert len(yaws) == len(waypoint_props(planned)), "every shot must face a new heading"

    def test_an_explicit_gimbal_still_wins(self, api, tmp_path):
        planned, _ = plan_and_export(api, tmp_path, {"template": "orbit", "gimbal_tilt_deg": -60})
        assert {p["gimbal_pitch_deg"] for p in waypoint_props(planned)} == {-60.0}


class TestSpeedAndDwellReachTheFile:
    def test_the_planned_speed_is_commanded_in_every_format(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "grid", "speed_m_s": 3})
        payload = json.loads(Path(written["qgc_plan"]).read_text())
        assert payload["mission"]["hoverSpeed"] == pytest.approx(3.0)
        assert payload["mission"]["cruiseSpeed"] == pytest.approx(3.0)
        speed_items = [i for i in payload["mission"]["items"] if i["command"] == DO_CHANGE_SPEED]
        assert len(speed_items) == 1 and speed_items[0]["params"][1] == pytest.approx(3.0)

        rows = list(csv.DictReader(open(written["litchi"])))
        assert {r["speed(m/s)"] for r in rows} == {"3.00"}

        with zipfile.ZipFile(written["dji_wpml"]) as archive:
            root = ET.fromstring(archive.read("wpmz/waylines.wpml"))
        auto = [e.text for e in root.iter(f"{WPML}autoFlightSpeed")]
        assert auto == ["3.0"], "autoFlightSpeed must be written once, at the planned speed"

    def test_a_requested_dwell_becomes_a_hold_at_every_capture(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "grid", "dwell_s": 2})
        captures = [p for p in waypoint_props(planned) if p["trigger"]]
        loiters = [i for i in plan_items(written) if i["command"] == NAV_LOITER_TIME]
        assert len(loiters) == len(captures)
        assert {i["params"][0] for i in loiters} == {2.0}
        rows = list(csv.DictReader(open(written["litchi"])))
        assert all(r["actiontype1"] == "1" and r["actionparam1"] == "2000" for r in rows[:5])

    def test_no_dwell_is_invented_when_none_was_asked_for(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "grid"})
        assert not [i for i in plan_items(written) if i["command"] == NAV_LOITER_TIME]

    def test_roof_inspection_stops_at_each_point_by_default(self, api, tmp_path):
        """Roof inspection is the stop-and-shoot grid; its dwell used to be dropped."""
        planned, written = plan_and_export(api, tmp_path, {"template": "roof_inspection"})
        loiters = [i for i in plan_items(written) if i["command"] == NAV_LOITER_TIME]
        # The planner's 1.5 s hold, rounded UP to the whole seconds ArduPilot stores: a
        # real ArduCopter truncated 1.5 to 1 (tests/sitl/test_copter_mission_templates.py).
        assert loiters and {i["params"][0] for i in loiters} == {2.0}


def _local(lat: float, lon: float) -> tuple[float, float]:
    return (lon + 81.7528) * 111_320.0 * math.cos(math.radians(41.304)), (lat - 41.304) * 110_540.0


def _inside(p, ring) -> bool:
    x, y = p
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


class TestNoFlyZonesInTheExportedPath:
    """The audit found double_grid's hop between its two passes cut through a zone.

    Checked on the file the aircraft would fly, not the planner's own geometry: every
    exported NAV point outside every zone, and no leg between two of them crossing one.
    """

    @pytest.mark.parametrize("template", ["grid", "double_grid", "corridor", "magnetic_mapping",
                                          "solar_inspection", "orbit", "waypoints"])
    def test_no_exported_leg_enters_a_no_fly_zone(self, api, tmp_path, template):
        assert api.set_no_fly_zones(NO_FLY)["ok"]
        _, written = plan_and_export(api, tmp_path, {"template": template, "altitude_m": 50})
        nav = [_local(i["params"][4], i["params"][5]) for i in plan_items(written)
               if i["command"] in (NAV_WAYPOINT, NAV_LOITER_TIME)]
        assert len(nav) >= 4
        zones = [[_local(lat, lon) for lon, lat in zone] for zone in NO_FLY]
        for zone in zones:
            assert not [p for p in nav if _inside(p, zone)], f"{template}: waypoint inside a zone"
            edges = list(zip(zone, zone[1:] + zone[:1]))
            crossing = [(a, b) for a, b in zip(nav, nav[1:])
                        if any(_cross(a, b, c, d) for c, d in edges)]
            assert not crossing, f"{template}: {len(crossing)} legs cross a no-fly zone"


class TestFileStructure:
    def test_the_waypoints_file_leaves_sequence_zero_for_home(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "grid"})
        rows = [line.split("\t") for line in Path(written["qgc_wpl"]).read_text().splitlines()[1:]]
        assert rows[0][1] == "1" and rows[0][2] == "0" and rows[0][3] == str(NAV_WAYPOINT)
        assert rows[1][3] == str(NAV_TAKEOFF), "takeoff must not sit in the home slot"
        assert [int(r[0]) for r in rows] == list(range(len(rows)))

    def test_each_no_fly_zone_is_its_own_exclusion_polygon(self, api, tmp_path):
        api.set_no_fly_zones(NO_FLY)
        _, written = plan_and_export(api, tmp_path, {"template": "grid"})
        polygons = json.loads(Path(written["qgc_plan"]).read_text())["geoFence"]["polygons"]
        exclusions = [p["polygon"] for p in polygons if not p["inclusion"]]
        assert exclusions == [[[v[1], v[0]] for v in zone] for zone in NO_FLY]

    def test_the_inclusion_fence_leaves_room_around_the_outer_passes(self, api, tmp_path):
        """ArduPilot breaches within FENCE_MARGIN of the fence; outer passes sat on it."""
        _, written = plan_and_export(api, tmp_path, {"template": "grid"})
        payload = json.loads(Path(written["qgc_plan"]).read_text())
        fence = next(p["polygon"] for p in payload["geoFence"]["polygons"] if p["inclusion"])
        mx = 111_320.0 * math.cos(math.radians(41.304))
        edges = list(zip(fence, fence[1:] + fence[:1]))

        def clearance(lat, lon):
            best = float("inf")
            for (la1, lo1), (la2, lo2) in edges:
                ax, ay, bx, by = lo1 * mx, la1 * 110_540, lo2 * mx, la2 * 110_540
                px, py = lon * mx, lat * 110_540
                t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay))
                                 / ((bx - ax) ** 2 + (by - ay) ** 2)))
                best = min(best, math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay)))
            return best

        nav = [i for i in payload["mission"]["items"] if i["command"] == NAV_WAYPOINT]
        assert min(clearance(i["params"][4], i["params"][5]) for i in nav) >= 4.0

    def test_litchi_is_split_at_its_99_waypoint_limit(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "facade"})
        parts = [v for k, v in written.items() if k.startswith("litchi")]
        assert len(parts) > 1
        counts = [len(list(csv.DictReader(open(p)))) for p in parts]
        assert max(counts) <= 99
        assert sum(counts) == planned["summary"]["waypoints"]

    def test_geojson_is_an_export_format(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "orbit"})
        exported = json.loads(Path(written["geojson"]).read_text())
        assert exported == planned["geojson"]
        only = api.export_mission(["geojson"], str(tmp_path / "gj"))
        assert only["ok"] and Path(only["written"]["geojson"]).exists()

    def test_dji_names_a_real_airframe(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "grid"})
        with zipfile.ZipFile(written["dji_wpml"]) as archive:
            root = ET.fromstring(archive.read("wpmz/template.kml"))
        assert next(root.iter(f"{WPML}droneEnumValue")).text == "77"


class TestFacadeGeometry:
    def test_the_reported_distance_is_the_distance_flown(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "facade"})
        nav = [i["params"][4:7] for i in plan_items(written) if i["command"] == NAV_WAYPOINT]
        mx = 111_320.0 * math.cos(math.radians(41.304))
        flown = sum(math.dist((a[1] * mx, a[0] * 110_540, a[2]), (b[1] * mx, b[0] * 110_540, b[2]))
                    for a, b in zip(nav, nav[1:]))
        assert planned["summary"]["distance_m"] == pytest.approx(flown, rel=0.02)

    def test_a_facade_is_flown_at_a_real_standoff_and_facing_the_wall(self, api, tmp_path):
        planned, _ = plan_and_export(api, tmp_path, {"template": "facade", "altitude_m": 50})
        props = waypoint_props(planned)
        assert planned["summary"]["waypoints"] < 2000, "0.5 m stand-off made metre-high rows"
        assert all(p["camera_yaw_locked"] for p in props)
        assert {p["gimbal_pitch_deg"] for p in props} == {0.0}

    def test_vertical_moves_are_not_treated_as_crossing_a_no_fly_zone(self, api, tmp_path):
        api.set_no_fly_zones(NO_FLY)
        planned, _ = plan_and_export(api, tmp_path, {"template": "facade"})
        assert planned["summary"]["adjustments"]["obstacle_detours"] == 0
        assert not [p for p in waypoint_props(planned) if p.get("primitive") == "obstacle_detour"]


class TestTemplatesAreThemselves:
    def test_dome_and_box_are_different_missions(self, api, tmp_path):
        dome, _ = plan_and_export(api, tmp_path, {"template": "dome_inspection", "dome_radius_m": 15,
                                                  "dome_base_altitude_m": 20})
        box, _ = plan_and_export(api, tmp_path, {"template": "box_inspection"})
        dome_prims = {p["primitive"] for p in waypoint_props(dome)}
        assert all(name.startswith("dome_inspection_ring") for name in dome_prims)
        assert all(p["primitive"].startswith("box_inspection_level") for p in waypoint_props(box))
        assert dome["summary"]["adjustments"]["geofence_projections"] == 0

    def test_the_dome_camera_follows_its_surface(self, api, tmp_path):
        dome, _ = plan_and_export(api, tmp_path, {"template": "dome_inspection", "dome_radius_m": 15,
                                                  "dome_base_altitude_m": 20})
        by_ring: dict[str, float] = {}
        for p in waypoint_props(dome):
            by_ring.setdefault(p["primitive"], p["gimbal_pitch_deg"])
        ordered = [by_ring[k] for k in sorted(by_ring, key=lambda n: int(n.rsplit("ring", 1)[1]))]
        assert ordered[0] == pytest.approx(0.0) and ordered[-1] == pytest.approx(-90.0)
        assert ordered == sorted(ordered, reverse=True)

    def test_smart_adaptive_adds_detail_over_an_interest_region(self, api, tmp_path):
        region = [[-81.7524, 41.3036], [-81.7522, 41.3036], [-81.7522, 41.3038], [-81.7524, 41.3038]]
        plain = api.plan_mission({"template": "smart_adaptive"})
        assert any("uniform grid" in w for w in plain["warnings"])
        adaptive = api.plan_mission({"template": "smart_adaptive", "interest_regions": [region]})
        names = {p["primitive"] for p in waypoint_props(adaptive)}
        assert "smart_adaptive_detail_0" in names
        assert adaptive["summary"]["waypoints"] > plain["summary"]["waypoints"]

    def test_a_route_line_is_the_line_flown(self, api, tmp_path):
        route = [[-81.7534, 41.3036], [-81.7528, 41.3040], [-81.7522, 41.3043]]
        planned = api.plan_mission({"template": "linear_inspection", "path_lonlat": route})
        assert planned["ok"]
        first = next(f["geometry"]["coordinates"] for f in planned["geojson"]["features"]
                     if f["properties"].get("role") == "waypoint")
        assert first[0] == pytest.approx(route[0][0], abs=2e-5)
        assert first[1] == pytest.approx(route[0][1], abs=2e-5)
        assert not any("No route line" in w for w in planned["warnings"])


class TestLinkedMissions:
    def test_an_area_alone_is_refused_rather_than_mixed_into_a_fake_sortie(self, api):
        result = api.plan_mission({"template": "linked_mission"})
        assert not result["ok"]
        assert "saved missions" in result["error"]

    def test_two_saved_missions_link_into_one_exportable_sortie(self, api, tmp_path):
        ids = []
        for template in ("grid", "orbit"):
            assert api.plan_mission({"template": template, "altitude_m": 50})["ok"]
            ids.append(api.save_mission(template)["version"]["id"])
        linked = api.plan_mission({"template": "linked_mission", "linked_version_ids": ids})
        assert linked["ok"], linked.get("error")
        assert linked["linked_segment_count"] == 2
        assert linked["summary"]["template"] == "linked_mission"
        assert api.export_mission(["qgc_plan"], str(tmp_path / "linked"))["ok"]

    def test_a_failed_transition_dry_run_is_a_visible_warning(self, api, tmp_path):
        ids = []
        for template in ("grid", "orbit"):
            assert api.plan_mission({"template": template, "altitude_m": 50})["ok"]
            ids.append(api.save_mission(template)["version"]["id"])
        # A zone covering the whole site guarantees the transition crosses it.
        api.set_no_fly_zones([[[-81.7540, 41.3030], [-81.7515, 41.3030],
                               [-81.7515, 41.3050], [-81.7540, 41.3050]]])
        linked = api.plan_linked_mission(ids)
        assert linked["ok"]
        assert linked["linked_dry_run_ok"] is False
        assert any("dry run failed" in w for w in linked["warnings"])


class TestSpecialMissionTypesBecomeTheMission:
    def test_a_thermal_survey_can_be_exported(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {
            "template": "thermal", "thermal_camera": "mavic3t_thermal", "target_gsd_cm": 5})
        assert planned["summary"]["template"] == "thermal"
        assert planned["summary"]["gsd_cm"] == pytest.approx(5.0, abs=0.05)

    def test_a_thermal_survey_without_its_gsd_is_refused(self, api):
        result = api.plan_mission({"template": "thermal", "thermal_camera": "mavic3t_thermal"})
        assert not result["ok"] and "target_gsd_cm" in result["error"]

    def test_a_multispectral_survey_can_be_exported(self, api, tmp_path):
        planned, _ = plan_and_export(api, tmp_path, {
            "template": "multispectral", "payload_key": "mavic3m_multispectral",
            "calibration_panel_lonlat": "-81.7533, 41.3036"})
        assert planned["summary"]["template"] == "multispectral"

    def test_a_pylon_inspection_can_be_exported_with_one_level_per_element(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {
            "template": "pylon_inspection", "pylon_elements": "crossarm@32, insulator@28"})
        heights = {round(p) for p in
                   (i["params"][6] for i in plan_items(written)
                    if i["command"] in (NAV_WAYPOINT, NAV_LOITER_TIME))}
        assert {28, 32} <= heights
        pitches = {p["gimbal_pitch_deg"] for p in waypoint_props(planned)}
        assert {0.0, -20.0} <= pitches

    def test_the_direct_endpoints_also_set_the_mission(self, api, tmp_path):
        assert api.plan_pylon_mission([-81.7528, 41.30395], [{"name": "crossarm", "height_m": 30}])["ok"]
        assert api.export_mission(["qgc_plan"], str(tmp_path / "pylon"))["ok"]


class TestMissionManager:
    def test_weather_is_reported_unchecked_rather_than_accepted(self, tmp_path):
        from core.mission_manager import MissionManager

        manager = MissionManager(project_root=tmp_path)
        manager.generate_plan({"polygon_lonlat": AOI, "altitude_m": 50})
        weather = next(c for c in manager.validate_plan() if c.id == "weather")
        assert not weather.ok and weather.level == "warning"
        assert "not checked" in weather.message.lower()

    def test_the_saved_waypoints_file_carries_takeoff_and_capture(self, tmp_path):
        from core.mission_manager import MissionManager

        manager = MissionManager(project_root=tmp_path)
        manager.generate_plan({"polygon_lonlat": AOI, "altitude_m": 50})
        saved = manager.save_mission(output_dir=tmp_path / "saved")
        rows = [line.split("\t") for line in Path(saved.qgc_wpl_path).read_text().splitlines()[1:]]
        commands = [int(r[3]) for r in rows]
        assert commands[0] == NAV_WAYPOINT and commands[1] == NAV_TAKEOFF
        assert 203 in commands, "no camera trigger in the saved mission"
