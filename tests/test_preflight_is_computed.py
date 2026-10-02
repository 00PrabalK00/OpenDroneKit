"""Preflight passes are earned, not defaulted.

Three passes were handed out without looking at anything: check_camera_ready defaulted to
"Camera ready." for a vehicle with no camera; MissionManager.run_preflight passed
geofence={"mission_inside": True} and no_fly_zones=[] for every plan; and MissionManager
defaulted its drone to the simulator, so an SDK user who never chose a vehicle got RTK
fix, fourteen satellites and a charged battery -- and a preflight that passed on them.
"""

from __future__ import annotations

from core.drone import MockDroneClient, NoDroneClient
from core.flight import FlightManager
from core.mission_manager import MissionManager
from core.preflight import (
    SEVERITY_BLOCK,
    SEVERITY_PASS,
    SEVERITY_WARN,
    assess_geofence,
    assess_no_fly_zones,
    check_camera_ready,
    check_geofence,
    check_no_fly_zones,
)

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


class TestNothingPassesByDefault:
    def test_an_unreported_camera_is_a_manual_check(self):
        result = check_camera_ready()
        assert result.severity == SEVERITY_WARN and result.requires_manual

    def test_a_fence_nobody_compared_is_not_a_pass(self):
        result = check_geofence(None, {"polygon": AOI})
        assert result.severity == SEVERITY_WARN and result.requires_manual

    def test_unchecked_no_fly_zones_are_not_a_pass(self):
        assert check_no_fly_zones(None, None).severity == SEVERITY_WARN
        assert check_no_fly_zones(None, [{"zone": 1}]).severity == SEVERITY_WARN


class TestTheGeometryIsComputed:
    def test_points_outside_the_fence_are_counted(self):
        inside = [(-81.75, 41.304), (-81.749, 41.305)]
        outside = inside + [(-81.70, 41.304)]
        assert assess_geofence(inside, AOI)["mission_inside"] is True
        verdict = assess_geofence(outside, AOI)
        assert verdict["mission_inside"] is False and verdict["points_outside"] == 1
        assert check_geofence(None, verdict).severity == SEVERITY_BLOCK

    def test_a_launch_point_outside_the_uploaded_fence_blocks(self):
        """ArduCopter refused to arm outside the enabled fence while preflight passed."""
        from core.preflight import plan_airspace

        plan = {"world_poses": [{"lon": -81.750, "lat": 41.304, "alt_m": 40.0},
                                {"lon": -81.749, "lat": 41.305, "alt_m": 40.0}],
                "safety_constraints": {"geofence": AOI}}
        inside, _ = plan_airspace(plan, launch_lonlat=(-81.7500, 41.3045))
        assert inside["launch_inside"] is True
        assert check_geofence(None, inside).severity == SEVERITY_PASS
        # Within the uploaded fence's 5 m margin still counts as inside: that is the fence.
        edge, _ = plan_airspace(plan, launch_lonlat=(-81.75203, 41.3045))
        assert edge["launch_inside"] is True
        outside, _ = plan_airspace(plan, launch_lonlat=(-81.7600, 41.3045))
        assert outside["launch_inside"] is False and outside["mission_inside"] is True
        result = check_geofence(None, outside)
        assert result.severity == SEVERITY_BLOCK and "outside the geofence" in result.message

    def test_a_leg_crossing_a_zone_conflicts_even_with_no_point_inside(self):
        zone = [[-81.7505, 41.3035], [-81.7495, 41.3035], [-81.7495, 41.3045], [-81.7505, 41.3045]]
        leg = [(-81.7515, 41.3040), (-81.7485, 41.3040)]
        assert assess_no_fly_zones(leg, [zone])[0]["conflict"] is True
        clear = [(-81.7515, 41.3055), (-81.7485, 41.3055)]
        assert assess_no_fly_zones(clear, [zone])[0]["conflict"] is False


class TestMissionManager:
    def test_no_driver_means_no_drone(self, tmp_path):
        manager = MissionManager(project_root=tmp_path)
        assert isinstance(manager.drone, NoDroneClient)
        assert manager.drone.is_connected() is False
        assert manager.drone.start_mission().success is False

    def test_preflight_computes_the_fence_from_the_plan(self, tmp_path):
        manager = MissionManager(project_root=tmp_path, drone=MockDroneClient())
        manager.drone.connect("mock://")
        plan = manager.generate_plan({"polygon_lonlat": AOI, "altitude_m": 60.0})
        report = manager.run_preflight(plan)
        geofence = next(c for c in report.checks if c.id == "geofence")
        # Whatever the answer, it is about this plan's path: it names the points it
        # checked, which the hardcoded {"mission_inside": True} never could.
        assert "path point" in geofence.message
        assert geofence.severity in (SEVERITY_PASS, SEVERITY_BLOCK)


class TestFlightManagerState:
    def test_flying_follows_the_aircraft_not_the_request(self, tmp_path):
        drone = MockDroneClient()
        drone.connect("mock://")
        drone.upload_mission([{"seq": 0}])
        manager = FlightManager()
        manager.start_session(tmp_path, "p", "m", "mock", drone)
        try:
            result = manager.start_mission()
            assert result.success
            assert manager.active_session().state.status == "starting"
            manager._apply_telemetry(drone.get_telemetry())  # noqa: SLF001
            assert manager.active_session().state.status == "flying"
        finally:
            manager.end_session("completed")

    def test_an_abort_keeps_the_session_logging(self, tmp_path):
        drone = MockDroneClient()
        drone.connect("mock://")
        manager = FlightManager()
        manager.start_session(tmp_path, "p", "m", "mock", drone)
        try:
            assert manager.abort_flight().success
            assert manager.active_session() is not None
            assert manager.active_session().state.status == "aborted"
        finally:
            manager.end_session("completed")
