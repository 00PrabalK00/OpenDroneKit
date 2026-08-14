"""Mission export: the capture behaviour must survive the round trip.

The original exporter emitted bare NAV_WAYPOINT rows, so a mission flew the route and
captured nothing. These tests assert the gimbal, yaw, dwell, and trigger commands the
planner computes are actually present in what gets written and uploaded.
"""

from __future__ import annotations

import zipfile

import pytest

from mission.exporters import (
    EXPORTERS,
    MAV_CMD_CONDITION_YAW,
    MAV_CMD_DO_DIGICAM_CONTROL,
    MAV_CMD_DO_MOUNT_CONTROL,
    MAV_CMD_NAV_LOITER_TIME,
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION,
    MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
    build_fence_items,
    build_mission_items,
    build_rally_items,
)

PLAN = {
    "altitude_m": 60.0,
    "gimbal_tilt_deg": -90.0,
    "waypoints": [],
    "flight_recipe": {
        "world_poses": [
            {
                "lon": -81.7505 + index * 0.0004,
                "lat": 41.3042,
                "alt_m": 60.0,
                "yaw_deg": 90.0,
                "gimbal_pitch_deg": -90.0,
                "dwell_s": 2.0,
                "trigger": True,
                "camera_yaw_locked": True,
            }
            for index in range(4)
        ],
        "capture": {"continuous_capture": False},
    },
    "safety_constraints": {
        "geofence": [
            [-81.7520, 41.3030], [-81.7480, 41.3030],
            [-81.7480, 41.3060], [-81.7520, 41.3060],
        ],
        "no_fly_polygons": [
            [[-81.7500, 41.3045], [-81.7495, 41.3045],
             [-81.7495, 41.3050], [-81.7500, 41.3050]]
        ],
        "rally_points": [{"lon": -81.7510, "lat": 41.3035, "alt_m": 50.0}],
        "rth_altitude_m": 75.0,
    },
}


def test_capture_commands_survive_export():
    commands = [item.command for item in build_mission_items(PLAN)]
    for expected in (
        MAV_CMD_DO_MOUNT_CONTROL,
        MAV_CMD_CONDITION_YAW,
        MAV_CMD_NAV_LOITER_TIME,
        MAV_CMD_DO_DIGICAM_CONTROL,
        MAV_CMD_NAV_RETURN_TO_LAUNCH,
    ):
        assert expected in commands, f"MAV_CMD {expected} was dropped by the exporter"


def test_gimbal_pitch_and_mount_mode_land_in_the_right_slots():
    """DO_MOUNT_CONTROL carries pitch in param1 and the mount mode in the z slot."""
    mount = next(i for i in build_mission_items(PLAN) if i.command == MAV_CMD_DO_MOUNT_CONTROL)
    assert mount.param1 == pytest.approx(-90.0)
    assert mount.altitude == pytest.approx(2.0)  # MAV_MOUNT_MODE_MAVLINK_TARGETING


def test_fence_covers_inclusion_and_exclusion():
    commands = [item.command for item in build_fence_items(PLAN)]
    assert MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION in commands
    assert MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION in commands
    assert len(commands) == 8  # four vertices each


@pytest.mark.parametrize(
    "geofence_key,exclusion_key",
    [
        ("geofence", "no_fly_polygons"),
        ("geofence_polygon", "no_fly_zones"),
        ("geofence", "obstacles"),
    ],
)
def test_fence_accepts_the_spellings_the_planner_accepts(geofence_key, exclusion_key):
    """parse_constraints takes several spellings on input; silently exporting no
    fence for a plan that has one is worse than refusing it."""
    constraints = dict(PLAN["safety_constraints"])
    constraints.pop("geofence", None)
    constraints.pop("no_fly_polygons", None)
    constraints[geofence_key] = PLAN["safety_constraints"]["geofence"]
    constraints[exclusion_key] = PLAN["safety_constraints"]["no_fly_polygons"]

    plan = dict(PLAN, safety_constraints=constraints)
    assert len(build_fence_items(plan)) == 8


def test_rally_points_are_emitted():
    assert len(build_rally_items(PLAN)) == 1


def test_every_registered_format_writes_a_file(tmp_path):
    for name, (writer, suffix) in EXPORTERS.items():
        target = tmp_path / f"mission{suffix}"
        writer(target, PLAN)
        assert target.exists(), f"{name} wrote nothing"
        assert target.stat().st_size > 0, f"{name} wrote an empty file"


def test_dji_kmz_has_the_required_wpml_members(tmp_path):
    writer, suffix = EXPORTERS["dji_wpml"]
    target = tmp_path / f"mission{suffix}"
    writer(target, PLAN)
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
    assert "wpmz/template.kml" in names
    assert "wpmz/waylines.wpml" in names


def test_empty_plan_is_refused_rather_than_exported_blank():
    with pytest.raises(ValueError):
        build_mission_items({"waypoints": [], "flight_recipe": {}})
