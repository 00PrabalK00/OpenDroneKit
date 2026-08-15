"""Arm-to-RTL lifecycle exercised against real ArduCopter state and telemetry."""

from __future__ import annotations

import math
import os
import time

import pytest

from tools.sitl.launch import SITL_HOME_LAT, SITL_HOME_LON


pytestmark = pytest.mark.sitl


def _wait_for(client, description: str, predicate, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    last = client.get_telemetry()
    while time.monotonic() < deadline:
        last = client.get_telemetry()
        if predicate(last):
            return last
        time.sleep(0.2)
    state = last.to_dict() if hasattr(last, "to_dict") else vars(last)
    pytest.fail(f"Timed out waiting for {description}; last telemetry: {state}")


def _distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_m = 6_371_000.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def test_arm_takeoff_waypoints_and_rtl(sitl_client, sitl_observer, sitl_mission):
    upload = sitl_client.upload_mission(list(sitl_mission.items))
    assert upload.success, upload.message

    arm = sitl_client.arm()
    assert arm.success, arm.message
    _wait_for(sitl_client, "armed state", lambda telemetry: telemetry.armed, 25.0)

    takeoff = sitl_client.takeoff(10.0)
    assert takeoff.success, takeoff.message
    _wait_for(
        sitl_client,
        "takeoff altitude",
        lambda telemetry: telemetry.armed and telemetry.altitude_rel_m >= 8.0,
        45.0,
    )

    while sitl_observer.recv_match(blocking=False) is not None:
        pass
    start = sitl_client.start_mission()
    assert start.success, start.message

    # MISSION_CURRENT is what OpenDroneKit currently exposes. The independent
    # stream supplies ArduPilot's stronger MISSION_ITEM_REACHED evidence.
    expected_reached = {
        int(item["seq"]) for item in sitl_mission.items if int(item["command"]) == 16
    }
    reached: set[int] = set()
    current_sequences: set[int] = set()
    flight_timeout_s = float(os.environ.get("ODK_SITL_FLIGHT_TIMEOUT_S", "120"))
    deadline = time.monotonic() + flight_timeout_s
    while time.monotonic() < deadline and not expected_reached.issubset(reached):
        message = sitl_observer.recv_match(
            type=["MISSION_ITEM_REACHED"], blocking=True, timeout=1.0
        )
        if message is not None:
            reached.add(int(message.seq))
        current_sequences.add(int(sitl_client.get_telemetry().waypoint_index))

    assert expected_reached.issubset(reached), (
        f"ArduPilot reached {sorted(reached)}, expected waypoint sequences "
        f"{sorted(expected_reached)}; MISSION_CURRENT observed {sorted(current_sequences)}"
    )
    assert current_sequences.intersection(expected_reached), (
        "ArduPilot emitted reached events, but OpenDroneKit telemetry never exposed "
        "a corresponding MISSION_CURRENT sequence."
    )

    rtl = sitl_client.return_to_home()
    assert rtl.success, rtl.message
    _wait_for(
        sitl_client,
        "RTL flight mode",
        lambda telemetry: telemetry.flight_mode.upper() == "RTL",
        15.0,
    )
    landed = _wait_for(
        sitl_client,
        "RTL landing and disarm",
        lambda telemetry: not telemetry.armed and telemetry.altitude_rel_m <= 1.0,
        flight_timeout_s,
    )
    assert _distance_m(
        landed.latitude, landed.longitude, SITL_HOME_LAT, SITL_HOME_LON
    ) <= 5.0
