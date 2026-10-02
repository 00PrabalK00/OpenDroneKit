"""Item types, malformed and oversized missions, against a real ArduCopter.

* Every mission item type the exporters can emit -- and the camera / gimbal-manager
  items the task list names but no template uses yet -- is uploaded through the app's
  MAVLink driver and read back; coordinates and parameters must survive.
* A malformed mission (a waypoint at latitude 95) and an oversized one (more items than
  the autopilot can store) are refused by ArduPilot, and the Api must say so: no "ok",
  no "mission uploaded" in the session, no passing preflight check -- with the
  vehicle's own reason in the message.
* A stock ArduCopter (650-item storage, as on a Pixhawk's FRAM) refuses the templates
  that exceed it, and the app reports which limit was hit.
"""

from __future__ import annotations

import json

import pytest

from tools.sitl.launch import ArduPilotSITL, SITLStartupError, SITLUnavailable

from ._support import (
    LATLON_TOL_DEG,
    connect_api,
    evidence_dir,
    make_api,
    offset_lonlat,
    plan_template,
)

pytestmark = pytest.mark.sitl


def _write(name: str, payload) -> None:
    (evidence_dir("integrity") / f"{name}.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")


def _plan_dict(poses: list[dict]) -> dict:
    return {
        "template": "waypoints",
        "altitude_m": 25.0,
        "world_poses": poses,
        "safety_constraints": {"rth_altitude_m": 30.0},
    }


def _pose(north: float, east: float, **extra) -> dict:
    lon, lat = offset_lonlat(north, east)
    return {"lon": lon, "lat": lat, "alt_m": 25.0, "yaw_deg": 0.0, "gimbal_pitch_deg": -90.0,
            "dwell_s": 0.0, "trigger": True, **extra}


def test_every_item_type_round_trips(sitl_client):
    roi_lon, roi_lat = offset_lonlat(15.0, 25.0)
    wp_lon, wp_lat = offset_lonlat(5.0, 5.0)
    items = [
        {"command": 22, "frame": 3, "lat": wp_lat, "lon": wp_lon, "alt": 20.0},
        {"command": 178, "frame": 2, "param1": 1.0, "param2": 4.5, "param3": -1.0},
        {"command": 205, "frame": 2, "param1": -37.5, "param2": 0.0, "param3": 0.0, "alt": 2.0},
        {"command": 1000, "frame": 2, "param1": -22.0, "param2": 15.0, "param3": 0.0,
         "param4": 0.0, "lat": 0.0, "lon": 0.0, "alt": 0.0},
        {"command": 115, "frame": 2, "param1": 271.5, "param2": 25.0, "param3": 1.0, "param4": 0.0},
        {"command": 201, "frame": 3, "lat": roi_lat, "lon": roi_lon, "alt": 5.0},
        {"command": 16, "frame": 3, "param1": 3.0, "lat": wp_lat, "lon": wp_lon, "alt": 22.5},
        {"command": 19, "frame": 3, "param1": 4.0, "lat": wp_lat, "lon": wp_lon, "alt": 23.0},
        {"command": 206, "frame": 2, "param1": 6.25, "param3": 1.0},
        {"command": 203, "frame": 2, "lat": 1.0},
        {"command": 2000, "frame": 2, "param1": 0.0, "param2": 2.0, "param3": 5.0, "param4": 0.0},
        {"command": 2001, "frame": 2, "param1": 0.0},
        {"command": 206, "frame": 2, "param1": 0.0},
        {"command": 20, "frame": 2},
    ]
    upload = sitl_client.upload_mission(items)
    assert upload.success, upload.message
    back = sitl_client.download_mission(timeout_s=30.0)
    _write("item_types", {"sent": items, "vehicle": back})

    assert [b["command"] for b in back] == [a["command"] for a in items]
    by_cmd = {(i, a["command"]): (a, b) for i, (a, b) in enumerate(zip(items, back))}
    for (_, command), (a, b) in by_cmd.items():
        for key in ("param1", "param2", "param3", "param4"):
            if key in a:
                assert b[key] == pytest.approx(a[key], abs=1e-3), (command, key, a, b)
        if command in (16, 19, 22, 201):
            assert b["lat"] == pytest.approx(a["lat"], abs=LATLON_TOL_DEG), (command, a, b)
            assert b["lon"] == pytest.approx(a["lon"], abs=LATLON_TOL_DEG), (command, a, b)
            assert b["alt"] == pytest.approx(a["alt"], abs=0.011), (command, a, b)
        if command == 203:
            assert b["lat"] == 1.0  # shutter command (param5)


def test_malformed_mission_is_refused_and_reported(tmp_path, sitl_connection):
    api = make_api(tmp_path)
    connect_api(api, sitl_connection)
    client = api._session._drone_client
    try:
        good = _plan_dict([_pose(0, 0), _pose(10, 0), _pose(10, 10)])
        api._session.mission_plan_dict = good
        first = api.upload_mission()
        assert first["ok"], first
        before = client.download_mission(timeout_s=30.0)

        bad_pose = _pose(20, 10)
        bad_pose["lat"] = 95.0  # not a latitude
        api._session.mission_plan_dict = _plan_dict([_pose(0, 0), _pose(10, 0), bad_pose])
        refused = api.upload_mission()
        after = client.download_mission(timeout_s=30.0)
        preflight = api.run_preflight()
        checks = {c["id"]: c for c in preflight.get("checks", [])}
        _write("malformed", {"refused": refused, "vehicle_before": before, "vehicle_after": after,
                             "preflight_mission_uploaded": checks.get("mission_uploaded")})

        assert refused["ok"] is False, refused
        assert "rejected" in refused["error"].lower(), refused["error"]
        assert refused["upload"]["mission"]["success"] is False
        assert api._session.mission_uploaded is False
        assert checks["mission_uploaded"]["severity"] != "pass", checks["mission_uploaded"]
        # The rejected plan's coordinates never became the vehicle's mission.
        assert all(abs(item["lat"]) <= 90.0 for item in after)
    finally:
        api.disconnect_vehicle()


def test_oversized_mission_is_refused_and_reported(tmp_path, sitl_connection):
    api = make_api(tmp_path)
    connect_api(api, sitl_connection)
    client = api._session._drone_client
    try:
        api._session.mission_plan_dict = _plan_dict([_pose(0, 0), _pose(10, 0)])
        assert api.upload_mission()["ok"]
        before = client.download_mission(timeout_s=30.0)

        # 8000 viewpoints x (mount + waypoint + shutter) = 24,000 items: more than this
        # vehicle's SD-card mission storage, and far more than a stock one.
        poses = [_pose((i // 100) * 0.5, (i % 100) * 0.5) for i in range(8000)]
        api._session.mission_plan_dict = _plan_dict(poses)
        refused = api.upload_mission()
        after = client.download_mission(timeout_s=30.0)
        _write("oversized", {"refused": refused, "vehicle_before_count": len(before),
                             "vehicle_after_count": len(after)})

        assert refused["ok"] is False, refused
        assert "no space" in refused["error"], refused["error"]
        assert "items are supported" in refused["error"], refused["error"]
        assert api._session.mission_uploaded is False
        # ArduPilot refused at MISSION_COUNT, so the previous mission is untouched.
        assert after == before
    finally:
        api.disconnect_vehicle()


@pytest.fixture(scope="module")
def stock_vehicle():
    """A second, stock ArduCopter: ArduPilot's own defaults, 650-item mission storage."""
    launcher = ArduPilotSITL(instance=3, param_files=[])
    try:
        launcher.start()
    except SITLUnavailable as exc:
        pytest.skip(f"ArduPilot SITL is unavailable: {exc}")
    except SITLStartupError as exc:
        pytest.fail(f"The stock SITL failed to start:\n{exc}", pytrace=False)
    try:
        yield launcher
    finally:
        launcher.stop()


@pytest.mark.parametrize("template,fits", [
    ("grid", True), ("tower_mapping", True), ("double_grid", False), ("dome_inspection", False),
])
def test_stock_vehicle_storage_limit_is_reported(tmp_path, stock_vehicle, template, fits):
    from mission import exporters

    api = make_api(tmp_path)
    connect_api(api, stock_vehicle.connection_string)
    try:
        assert plan_template(api, template)["ok"]
        count = len(exporters.build_mission_items(api._session.mission_plan_dict))
        result = api.upload_mission()
        _write(f"stock_{template}", {"items": count, "result": result})
        if fits:
            assert result["ok"], result
            assert len(api._session._drone_client.download_mission(timeout_s=30.0)) == count
        else:
            assert result["ok"] is False, result
            assert "no space" in result["error"] and "650 items" in result["error"], result["error"]
            assert api._session.mission_uploaded is False
    finally:
        api.disconnect_vehicle()
