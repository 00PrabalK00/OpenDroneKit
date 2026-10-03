"""The Plan workspace's backend (v1 interface, U5), against the real Api.

Controls on the approved Plan artboard that the Api could not back before this unit:
the waypoint inspector, altitude profile and photo footprints (mission_overview), the
area of interest from the camera positions and where an area came from (aoi_from_images,
set_aoi `source`), putting a saved version back on screen (load_mission_version), the
grouped template list (mission_templates `groups`), the line heading (plan_mission
`flight_direction_deg`) and the size of each exported file (export_mission `bytes`,
list_mission_exports). Each figure is checked against the plan or the files themselves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from mission.overview import camera_hull, footprint_ring, overview

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    instance = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    assert instance.create_project("Plan backend", str(tmp_path / "project"))["ok"]
    return instance


def _metres(a, b) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((b[0] - a[0]) * 111_320 * math.cos(lat), (b[1] - a[1]) * 111_320)


class TestMissionOverview:
    def test_with_no_plan_it_says_so_and_still_describes_the_area(self, api) -> None:
        empty = api.mission_overview()
        assert empty["ok"] and empty["plan"] is None and empty["aoi"]["set"] is False
        assert api.set_aoi(AOI)["ok"]
        area = api.mission_overview()["aoi"]
        assert area["set"] and area["vertices"] == 4 and area["source"] == "drawn on the map"
        assert area["area_m2"] == pytest.approx(api.set_aoi(AOI)["area_m2"])

    def test_the_overview_is_the_plan_the_planner_made(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        planned = api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})
        assert planned["ok"]
        view = api.mission_overview()["plan"]
        summary = planned["summary"]
        assert view["waypoint_count"] == summary["waypoints"]
        assert view["distance_m"] == pytest.approx(summary["distance_m"], abs=0.1)
        assert view["duration_min"] == pytest.approx(summary["duration_min"], abs=0.1)
        assert view["safety_adjustments"] == summary["adjustments"]
        # Every waypoint row is the plan's own waypoint, in order.
        plan = api._session.mission_plan_dict
        for row, point in zip(view["waypoints"], plan["waypoints"]):
            assert [row["lon"], row["lat"]] == pytest.approx(point[:2])
            assert row["alt_m"] == pytest.approx(point[2])
        # Distance along the route is the sum of the steps between waypoints.
        rows = view["waypoints"]
        steps = sum(_metres([a["lon"], a["lat"]], [b["lon"], b["lat"]]) for a, b in zip(rows, rows[1:]))
        assert rows[-1]["along_m"] == pytest.approx(steps, rel=0.01)
        # A grid has one photographed line per survey line, and the line spacing the
        # planner chose separates them.
        assert view["line_count"] >= 3 and view["trigger_count"] == sum(r["trigger"] for r in rows)
        assert view["line_spacing_m"] == pytest.approx(plan["line_spacing_m"])

    def test_footprints_are_the_cameras_rectangle_at_the_planned_height(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        view = api.mission_overview()["plan"]
        camera = api.describe_camera("mavic2pro", 55)["camera"]
        feature = view["footprints"]["features"][0]
        assert feature["properties"]["across_m"] == pytest.approx(camera["footprint_m"][0], abs=0.1)
        assert feature["properties"]["along_m"] == pytest.approx(camera["footprint_m"][1], abs=0.1)
        ring = feature["geometry"]["coordinates"][0]
        sides = sorted(_metres(a, b) for a, b in zip(ring, ring[1:]))
        assert sides[0] == pytest.approx(camera["footprint_m"][1], rel=0.01)
        assert sides[-1] == pytest.approx(camera["footprint_m"][0], rel=0.01)
        assert "flat" in view["footprint_basis"]

    def test_an_unknown_camera_draws_no_footprint_and_says_why(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "no_such_camera"})["ok"]
        view = api.mission_overview()["plan"]
        assert view["footprints"]["features"] == []
        assert "not a known camera" in view["footprint_basis"]

    def test_the_line_heading_reaches_the_planner(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        assert api.plan_mission({"template": "grid", "flight_direction_deg": 30})["ok"]
        assert api.mission_overview()["plan"]["flight_direction_deg"] == pytest.approx(30.0)
        assert api.plan_mission({"template": "grid", "flight_direction_deg": 120})["ok"]
        assert api.mission_overview()["plan"]["flight_direction_deg"] == pytest.approx(120.0)


class TestTheAreaOfInterest:
    def test_the_hull_of_the_camera_positions(self, api, tmp_path) -> None:
        cameras = [[-81.751 + 0.001 * (i % 5), 41.303 + 0.0008 * (i // 5)] for i in range(20)]
        track = tmp_path / "camera_track.geojson"
        track.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": c}}
            for c in cameras]}), encoding="utf-8")
        assert api.aoi_from_images()["ok"] is False  # nothing to outline yet
        api._session.register_vector_layer(track, name="Camera positions")
        result = api.aoi_from_images()
        assert result["ok"] and result["cameras"] == 20 and result["vertices"] == 4
        assert sorted(map(tuple, result["polygon"])) == sorted(map(tuple, camera_hull(cameras)))
        assert api.mission_overview()["aoi"]["source"] == "hull · 20 cameras"

    def test_the_source_follows_the_last_change(self, api, tmp_path) -> None:
        assert api.set_aoi(AOI, "edited")["ok"]
        assert api.mission_overview()["aoi"]["source"] == "edited"
        boundary = tmp_path / "site.geojson"
        boundary.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [[*AOI, AOI[0]]]}}), encoding="utf-8")
        assert api.import_boundary(str(boundary))["ok"]
        assert api.mission_overview()["aoi"]["source"] == "file · site.geojson"
        assert api.set_aoi([])["ok"]
        assert api.mission_overview()["aoi"] == {"set": False, "vertices": 0, "area_m2": None,
                                                  "source": "", "polygon": []}


class TestVersionsAndExports:
    def test_a_saved_version_comes_back_on_screen(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        first = api.plan_mission({"template": "grid", "altitude_m": 55, "flight_direction_deg": 30})
        assert api.save_mission("site_a")["ok"]
        assert api.mission_overview()["saved"] == {"name": "site_a", "version_num": 1}
        assert api.plan_mission({"template": "grid", "altitude_m": 80})["ok"]
        assert api.mission_overview()["saved"] is None  # an unsaved change is not v1
        loaded = api.load_mission_version(1, "site_a")
        assert loaded["ok"] and loaded["matches_saved"] is True, loaded
        assert loaded["summary"]["waypoints"] == first["summary"]["waypoints"]
        overview_now = api.mission_overview()
        assert overview_now["saved"] == {"name": "site_a", "version_num": 1}
        assert overview_now["plan"]["altitude_m"] == pytest.approx(55.0)
        assert overview_now["plan"]["flight_direction_deg"] == pytest.approx(30.0)
        assert overview_now["aoi"]["source"] == "saved · site_a v1"
        assert api.load_mission_version(9, "site_a")["ok"] is False

    def test_exports_report_the_bytes_on_disk(self, api) -> None:
        assert api.set_aoi(AOI)["ok"]
        assert api.plan_mission({"template": "grid"})["ok"]
        exported = api.export_mission(["qgc_plan", "dji_wpml", "geojson"])
        assert exported["ok"]
        for key, path in exported["written"].items():
            assert exported["bytes"][key] == Path(path).stat().st_size
        listed = api.list_mission_exports()
        files = listed["exports"][0]["files"]
        assert {f["format"] for f in files} == {"qgc_plan", "dji_wpml", "geojson"}
        assert all(f["bytes"] == Path(f["path"]).stat().st_size for f in files)

    def test_templates_are_grouped_and_none_is_lost(self, api) -> None:
        listed = api.mission_templates()
        grouped = [name for names in listed["groups"].values() for name in names]
        assert sorted(grouped) == sorted([*listed["templates"], *listed["special_templates"]])
        assert len(grouped) == len(set(grouped))
        assert "grid" in listed["groups"]["area"] and "orbit" in listed["groups"]["structure"]


def test_footprint_ring_is_a_closed_rectangle_turned_to_the_heading() -> None:
    ring = footprint_ring(-81.75, 41.30, 90.0, 40.0, 60.0)
    assert ring[0] == ring[-1] and len(ring) == 5
    # Heading east: the 40 m side runs east-west.
    east_west = _metres(ring[0], ring[3])
    assert east_west == pytest.approx(40.0, rel=0.01)


def test_overview_without_waypoint_features_reads_the_waypoint_list() -> None:
    plan = {"waypoints": [[-81.75, 41.30, 50], [-81.749, 41.30, 50], [-81.748, 41.30, 50]],
            "altitude_m": 50, "gimbal_tilt_deg": -90}
    view = overview(plan)
    assert view["waypoint_count"] == 3 and view["line_count"] == 1
    assert view["footprints"]["features"] == []  # no camera given: no footprint invented
