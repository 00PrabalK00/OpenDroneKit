"""Geofence and rally points: uploaded through the Api, read back, enabled, enforced.

The fence is uploaded beside the mission by ``Api.upload_mission`` (inclusion polygon
plus a no-fly exclusion), the rally points the operator placed with
``Api.set_rally_points``. Both are downloaded back from ArduCopter and compared vertex
by vertex. ``vehicle_command("fence_enable")`` must turn the fence on as ArduPilot
reports it (SYS_STATUS geofence bit, FENCE_STATUS), and the fence must then actually
be enforced: an aircraft sent outside it breaches and ArduPilot takes it home.
"""

from __future__ import annotations

import json

import pytest

from ._support import (
    FIELD_POLYGON,
    LATLON_TOL_DEG,
    connect_api,
    distance_m,
    evidence_dir,
    make_api,
    offset_lonlat,
    plan_template,
    settle_on_ground,
    wait_telemetry,
)

pytestmark = pytest.mark.sitl

MAV_SYS_STATUS_GEOFENCE = 1 << 20
NO_FLY = [offset_lonlat(n, e) for n, e in ((30, 40), (30, 48), (38, 48), (38, 40))]
RALLY = [offset_lonlat(-5, 45) + [25.0], offset_lonlat(35, -5)]


def test_fence_and_rally_round_trip_enable_and_enforce(tmp_path, sitl_connection, sitl_trace):
    from mission import exporters

    api = make_api(tmp_path)
    connect_api(api, sitl_connection)
    client = api._session._drone_client
    record: dict = {}
    try:
        assert api.set_no_fly_zones([NO_FLY])["ok"]
        rally_set = api.set_rally_points(RALLY)
        assert rally_set["ok"] and rally_set["count"] == 2, rally_set
        assert plan_template(api, "grid")["ok"]
        plan = api._session.mission_plan_dict
        fence_sent = [i.to_dict() for i in exporters.build_fence_items(plan)]
        assert any(i["command"] == 5002 for i in fence_sent), "the no-fly zone is not in the fence"

        upload = api.upload_mission()
        record["upload"] = upload.get("upload")
        assert upload["ok"], upload
        assert upload["upload"]["fence"]["success"] and upload["upload"]["rally"]["success"], upload

        fence_back = client.download_geofence(timeout_s=30.0)
        rally_back = client.download_rally_points(timeout_s=30.0)
        record.update(fence_sent=fence_sent, fence_vehicle=fence_back, rally_vehicle=rally_back)
        assert len(fence_back) == len(fence_sent)
        for sent, held in zip(fence_sent, fence_back):
            assert held["command"] == sent["command"]
            assert held["param1"] == sent["param1"]
            assert held["lat"] == pytest.approx(sent["lat"], abs=LATLON_TOL_DEG)
            assert held["lon"] == pytest.approx(sent["lon"], abs=LATLON_TOL_DEG)
        assert [r["command"] for r in rally_back] == [5100, 5100]
        expected_alts = [25.0, float(plan["safety_constraints"]["rth_altitude_m"])]
        for point, held, alt in zip(RALLY, rally_back, expected_alts):
            assert held["lon"] == pytest.approx(point[0], abs=LATLON_TOL_DEG)
            assert held["lat"] == pytest.approx(point[1], abs=LATLON_TOL_DEG)
            assert held["alt"] == pytest.approx(alt, abs=0.011)
        # The mission itself was not overwritten by the fence or rally transfers.
        assert len(client.download_mission(timeout_s=30.0)) == len(exporters.build_mission_items(plan))

        mark = sitl_trace.mark()
        enabled = api.vehicle_command("fence_enable")
        record["fence_enable"] = enabled
        assert enabled["ok"], enabled
        telemetry = wait_telemetry(api, "geofence enabled", lambda t: t.get("geofence_enabled") is True, 10.0)
        status = sitl_trace.wait_for("SYS_STATUS", lambda d: bool(
            d["onboard_control_sensors_enabled"] & MAV_SYS_STATUS_GEOFENCE), mark, 10.0)
        fence_status = sitl_trace.wait_for("FENCE_STATUS", since=mark, timeout_s=10.0)
        record["fence_status_after_enable"] = fence_status.data if fence_status else None
        assert status is not None and fence_status is not None
        assert fence_status.data["breach_status"] == 0

        # Enforced, not just stored. ArduCopter will not accept a GUIDED target outside
        # an enabled fence, so the aircraft is flown out with the fence off and the fence
        # is enabled out there: ArduPilot must declare the breach and bring it home.
        assert api.vehicle_command("fence_disable")["ok"]
        api.vehicle_command("loiter")
        assert api.vehicle_command("arm")["ok"]
        assert api.vehicle_command("takeoff", altitude_m=10.0)["ok"]
        wait_telemetry(api, "climb", lambda t: t["altitude_rel_m"] >= 8.5, 90.0)
        outside_lon, outside_lat = offset_lonlat(15.0, 90.0)  # 40 m east of the field
        assert client.goto(outside_lat, outside_lon, 10.0).success
        wait_telemetry(api, "outside the fence", lambda t: distance_m(
            t["latitude"], t["longitude"], outside_lat, outside_lon) < 4.0, 120.0)
        mark = sitl_trace.mark()
        enabled_outside = api.vehicle_command("fence_enable")
        record["fence_enable_outside"] = enabled_outside
        breach = sitl_trace.wait_for("FENCE_STATUS", lambda d: d["breach_status"] == 1, mark, 30.0)
        wait_telemetry(api, "fence action RTL", lambda t: t["flight_mode"] in ("RTL", "LAND"), 30.0)
        record["breach"] = breach.data if breach else None
        record["breach_text"] = sitl_trace.statustext(mark)
        assert breach is not None, "the aircraft was outside the fence and ArduPilot reported no breach"
        landed = wait_telemetry(api, "fence RTL landing", lambda t: not t["armed"], 300.0)
        # ArduPilot's RTL goes to the nearest rally point when that is closer than home,
        # so landing on the uploaded rally point is the evidence the rally list is used.
        # From 40 m east of the field the south-east rally point is the nearest.
        record["landed_from_home_m"] = distance_m(
            landed["latitude"], landed["longitude"], telemetry["home_lat"], telemetry["home_lon"])
        record["landed_from_rally_1_m"] = distance_m(
            landed["latitude"], landed["longitude"], RALLY[0][1], RALLY[0][0])
        assert record["landed_from_rally_1_m"] < 5.0, record

        disabled = api.vehicle_command("fence_disable")
        assert disabled["ok"], disabled
        wait_telemetry(api, "geofence disabled", lambda t: t.get("geofence_enabled") is False, 10.0)
    finally:
        (evidence_dir("fence_rally") / "fence_rally.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8")
        try:
            api.vehicle_command("fence_disable")
            settle_on_ground(api)
        finally:
            api.disconnect_vehicle()
