"""An exported heading must point the camera at the thing being inspected.

The planner works in a local frame where yaw is a math angle (0 along +x, counter-
clockwise). Every export format -- QGC CONDITION_YAW, the Litchi heading column, DJI
waypointHeadingAngle -- means a compass heading (0 north, clockwise). The poses crossed
that boundary unconverted, so an orbit camera faced its tower at two points a ring and
faced directly away at two more, a facade was photographed edge-on, and turbine blades
were shot from the side facing away from them. An earlier test only checked that a yaw
command existed.

These tests plan and export through the real Api, then read the headings back out of
the .plan file and compare them with the bearing from each exported position to the
subject, computed independently of the planner.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]
CONDITION_YAW, DO_MOUNT_CONTROL, NAV_WAYPOINT, NAV_LOITER_TIME = 115, 205, 16, 19

LON0 = sum(p[0] for p in AOI) / len(AOI)
LAT0 = sum(p[1] for p in AOI) / len(AOI)
EARTH_M_PER_DEG = 6371008.8 * math.pi / 180.0


def east_north(lon: float, lat: float) -> tuple[float, float]:
    return ((lon - LON0) * math.cos(math.radians(LAT0)) * EARTH_M_PER_DEG,
            (lat - LAT0) * EARTH_M_PER_DEG)


def compass_bearing(a, b) -> float:
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def angle_between(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def nearest_point_on_aoi(p):
    ring = [east_north(*v) for v in AOI]
    best = None
    for a, b in zip(ring, ring[1:] + ring[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        q = (a[0] + t * dx, a[1] + t * dy)
        if best is None or math.dist(p, q) < best[0]:
            best = (math.dist(p, q), q)
    return best[1]


@pytest.fixture
def api(tmp_path) -> Api:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("headings", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi(AOI)["ok"]
    return api


def plan_and_export(api: Api, tmp_path: Path, options: dict) -> tuple[dict, dict]:
    planned = api.plan_mission(options)
    assert planned["ok"], planned.get("error")
    exported = api.export_mission(None, str(tmp_path / "out" / options["template"]))
    assert exported["ok"], exported.get("error")
    failed = {k: v for k, v in exported["written"].items() if str(v).startswith("failed")}
    assert not failed, failed
    return planned, exported["written"]


def headed_positions(written: dict) -> list[tuple[float, tuple[float, float], float]]:
    """(heading, east/north position, mount yaw) for every NAV item a heading precedes."""
    items = json.loads(Path(written["qgc_plan"]).read_text())["mission"]["items"]
    out, heading, mount_yaw = [], None, None
    for item in items:
        if item["command"] == DO_MOUNT_CONTROL:
            mount_yaw = item["params"][2]
        elif item["command"] == CONDITION_YAW:
            heading = item["params"][0]
        elif item["command"] in (NAV_WAYPOINT, NAV_LOITER_TIME) and heading is not None:
            out.append((heading, east_north(item["params"][5], item["params"][4]), mount_yaw))
            heading = None
    return out


def errors_to(points, subject_of) -> list[float]:
    errs = []
    for heading, position, _ in points:
        subject = subject_of(position)
        if math.dist(position, subject) < 1.0:
            continue  # directly over the subject (a dome crown): no bearing to face
        errs.append(angle_between(heading, compass_bearing(position, subject)))
    return errs


class TestRingsFaceTheirCentre:
    @pytest.mark.parametrize("template", ["orbit", "tower_mapping", "dome_inspection",
                                          "box_inspection", "closed_loop"])
    def test_every_exported_heading_points_at_the_structure(self, api, tmp_path, template):
        _, written = plan_and_export(api, tmp_path, {"template": template, "altitude_m": 50})
        points = headed_positions(written)
        assert len(points) >= 12
        errs = errors_to(points, lambda _p: (0.0, 0.0))
        # The planner centres these on the footprint; a few degrees of slack covers the
        # orbit and loop centres it derives slightly differently.
        assert max(errs) < 6.0, f"{template}: camera off its subject by up to {max(errs):.0f} deg"

    @pytest.mark.parametrize("template", ["closed_loop", "tower_mapping", "box_inspection"])
    def test_a_footprint_drawn_at_an_angle_is_still_faced(self, api, tmp_path, template):
        """The local frame turns with the footprint's first edge; the heading must not."""
        tilted = [[-81.7530, 41.3035], [-81.7521, 41.3039], [-81.7526, 41.3045], [-81.7535, 41.3041]]
        assert api.set_aoi(tilted)["ok"]
        _, written = plan_and_export(api, tmp_path, {"template": template, "altitude_m": 50})
        subject = east_north(sum(p[0] for p in tilted) / 4, sum(p[1] for p in tilted) / 4)
        errs = errors_to(headed_positions(written), lambda _p: subject)
        assert errs and max(errs) < 6.0, f"{template}: off by up to {max(errs):.0f} deg"

    def test_an_orbit_round_an_explicit_centre_faces_that_centre(self, api, tmp_path):
        centre = [-81.7530, 41.3038]
        _, written = plan_and_export(api, tmp_path, {
            "template": "orbit", "altitude_m": 50, "orbit_center_lonlat": centre,
            "orbit_radius_m": 30})
        subject = east_north(*centre)
        errs = errors_to(headed_positions(written), lambda _p: subject)
        assert errs and max(errs) < 1.0

    def test_a_ring_pushed_inside_the_fence_is_re_aimed(self, api, tmp_path):
        """An orbit wider than the site is projected onto the fence; the camera follows."""
        planned, written = plan_and_export(api, tmp_path, {
            "template": "orbit", "altitude_m": 50, "orbit_radius_m": 90})
        assert planned["summary"]["adjustments"]["geofence_projections"] > 0
        errs = errors_to(headed_positions(written), lambda _p: (0.0, 0.0))
        assert errs and max(errs) < 6.0, f"off by up to {max(errs):.0f} deg after projection"

    def test_the_mount_is_not_given_an_absolute_heading(self, api, tmp_path):
        """DO_MOUNT_CONTROL yaw is body-relative; the vehicle does the turning."""
        _, written = plan_and_export(api, tmp_path, {"template": "orbit", "altitude_m": 50})
        assert {m for _, _, m in headed_positions(written)} == {0.0}

    def test_litchi_headings_agree_with_the_plan(self, api, tmp_path):
        _, written = plan_and_export(api, tmp_path, {"template": "tower_mapping", "altitude_m": 50})
        rows = []
        for key, path in written.items():
            if key.startswith("litchi"):
                rows.extend(csv.DictReader(open(path)))
        assert rows
        errs = [angle_between(float(r["heading(deg)"]),
                              compass_bearing(east_north(float(r["longitude"]), float(r["latitude"])),
                                              (0.0, 0.0)))
                for r in rows]
        assert max(errs) < 1.0


class TestWallsAreLookedAt:
    @pytest.mark.parametrize("template", ["facade", "multi_facade", "facade_mapping"])
    def test_the_camera_faces_the_nearest_wall(self, api, tmp_path, template):
        _, written = plan_and_export(api, tmp_path, {"template": template, "altitude_m": 50})
        errs = errors_to(headed_positions(written), nearest_point_on_aoi)
        assert errs and max(errs) < 1.0, f"{template}: wall seen at up to {max(errs):.0f} deg"

    def test_a_lateral_capture_looks_at_its_line(self, api, tmp_path):
        route = [[-81.7534, 41.3036], [-81.7522, 41.3036]]
        _, written = plan_and_export(api, tmp_path, {
            "template": "lateral_capture", "altitude_m": 50, "path_lonlat": route})
        a, b = east_north(*route[0]), east_north(*route[1])

        def foot(p):
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
            return (a[0] + t * dx, a[1] + t * dy)

        errs = errors_to(headed_positions(written), foot)
        assert errs and max(errs) < 1.0


class TestPylonAndTurbine:
    def test_a_pylon_orbit_faces_the_pylon(self, api, tmp_path):
        centre = [-81.7528, 41.30395]
        _, written = plan_and_export(api, tmp_path, {
            "template": "pylon_inspection", "pylon_center_lonlat": centre,
            "pylon_elements": "crossarm@32, insulator@28"})
        subject = east_north(*centre)
        errs = errors_to(headed_positions(written), lambda _p: subject)
        assert errs and max(errs) < 1.0

    def test_each_blade_face_is_shot_towards_the_blade(self, api, tmp_path):
        planned, written = plan_and_export(api, tmp_path, {"template": "wind_turbine", "altitude_m": 90})
        headings = [h for h, _, _ in headed_positions(written)]
        props = [(f["properties"], east_north(*f["geometry"]["coordinates"][:2]))
                 for f in planned["geojson"]["features"] if f["properties"].get("role") == "waypoint"]
        locked = [(p["primitive"], pos) for p, pos in props if p.get("camera_yaw_locked")]
        assert len(locked) == len(headings)
        exported = list(zip((name for name, _ in locked), (pos for _, pos in locked), headings))

        nacelle = [pos for name, pos, _ in exported if name == "turbine_nacelle"]
        hub = (sum(p[0] for p in nacelle) / len(nacelle), sum(p[1] for p in nacelle) / len(nacelle))
        for name, pos, heading in exported:
            if name in ("turbine_tower", "turbine_nacelle"):
                assert angle_between(heading, compass_bearing(pos, hub)) < 1.0, name

        blades = {}
        for name, pos, heading in exported:
            if name.startswith("turbine_blade_"):
                blades.setdefault(name, []).append((pos, heading))
        assert blades
        for name, stations in blades.items():
            if not name.endswith("a"):
                continue
            opposite = blades[name[:-1] + "b"]
            for (pa, ha), (pb, hb) in zip(stations, opposite):
                # The blade lies midway between its two faces: each face looks across at
                # the other, through the blade.
                assert angle_between(ha, compass_bearing(pa, pb)) < 1.0, name
                assert angle_between(hb, compass_bearing(pb, pa)) < 1.0, name

    def test_a_turbine_is_not_approached_to_two_metres_by_default(self, api, tmp_path):
        planned, _ = plan_and_export(api, tmp_path, {"template": "wind_turbine", "altitude_m": 90})
        by_name = {}
        for f in planned["geojson"]["features"]:
            if f["properties"].get("role") == "waypoint":
                by_name.setdefault(f["properties"]["primitive"], []).append(
                    east_north(*f["geometry"]["coordinates"][:2]))
        a, b = by_name["turbine_blade_0a"][0], by_name["turbine_blade_0b"][0]
        assert math.dist(a, b) / 2.0 >= 8.0


class TestLinkedSegmentsKeepTheirHeadings:
    def test_an_orbit_linked_after_a_grid_still_faces_its_centre(self, api, tmp_path):
        ids = []
        for template in ("grid", "orbit"):
            assert api.plan_mission({"template": template, "altitude_m": 50})["ok"]
            ids.append(api.save_mission(template)["version"]["id"])
        linked = api.plan_mission({"template": "linked_mission", "linked_version_ids": ids})
        assert linked["ok"], linked.get("error")
        written = api.export_mission(["qgc_plan"], str(tmp_path / "linked"))["written"]
        orbit_errs = errors_to(headed_positions(written), lambda _p: (0.0, 0.0))
        assert orbit_errs and max(orbit_errs) < 6.0
