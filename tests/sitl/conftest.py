"""Opt-in fixtures for tests that talk to a real ArduPilot SITL process."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import pytest

from tools.sitl.launch import ArduPilotSITL, SITLStartupError, SITLUnavailable


@dataclass(frozen=True)
class PlannedSITLMission:
    plan: Any
    items: tuple[dict[str, Any], ...]


def pytest_configure(config) -> None:
    # The root configuration uses --strict-markers. Register locally so this
    # self-contained harness does not need to edit the shared pytest.ini.
    config.addinivalue_line(
        "markers",
        "sitl: requires an explicitly selected, real ArduPilot SITL integration run",
    )


def _sitl_was_explicitly_selected(config) -> bool:
    # A broad expression such as `-m "not slow"` must not unexpectedly launch an
    # aircraft simulator. The documented opt-in is the exact `-m sitl` expression.
    return str(config.getoption("markexpr") or "").strip() == "sitl"


def pytest_collection_modifyitems(config, items) -> None:
    if _sitl_was_explicitly_selected(config):
        return
    skip = pytest.mark.skip(reason="SITL tests are opt-in; run pytest -m sitl")
    for item in items:
        if item.get_closest_marker("sitl") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def sitl_session(pytestconfig):
    if not _sitl_was_explicitly_selected(pytestconfig):
        pytest.skip("SITL tests are opt-in; run pytest -m sitl")

    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if os.environ.get("ODK_SITL_INSTANCE") is None and worker.startswith("gw"):
        try:
            instance = int(worker[2:])
        except ValueError:
            instance = 0
    else:
        instance = None
    launcher = ArduPilotSITL(instance=instance)
    try:
        launcher.start()
    except SITLUnavailable as exc:
        pytest.skip(f"ArduPilot SITL is unavailable: {exc}")
    except SITLStartupError as exc:
        pytest.fail(f"ArduPilot SITL was found but failed to start:\n{exc}", pytrace=False)
    try:
        yield launcher
    finally:
        launcher.stop()


@pytest.fixture(scope="session")
def sitl_connection(sitl_session) -> str:
    """MAVLink URI for the production OpenDroneKit client."""

    return sitl_session.connection_string


@pytest.fixture(scope="session")
def sitl_observer_connection(sitl_session) -> str:
    """Independent MAVLink URI used only to observe autopilot evidence."""

    return sitl_session.observer_connection_string


@pytest.fixture
def sitl_client(sitl_connection):
    from core.mission_planner_bridge import MissionPlannerDroneClient

    client = MissionPlannerDroneClient()
    client.connect(sitl_connection)
    try:
        yield client
    finally:
        client.disconnect()


@pytest.fixture
def sitl_observer(sitl_observer_connection):
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink import mavutil

    observer = mavutil.mavlink_connection(
        sitl_observer_connection, autoreconnect=True, source_system=253
    )
    heartbeat = observer.wait_heartbeat(timeout=15)
    if heartbeat is None:
        observer.close()
        pytest.fail("The independent SITL telemetry observer received no heartbeat.")
    try:
        yield observer
    finally:
        observer.close()


@pytest.fixture
def sitl_mission() -> PlannedSITLMission:
    from mission.exporters import build_mission_items
    from mission.planner import MissionPlanner
    from tools.sitl.launch import SITL_HOME_LAT, SITL_HOME_LON

    path = [
        [SITL_HOME_LON, SITL_HOME_LAT],
        [SITL_HOME_LON + 0.00018, SITL_HOME_LAT],
        [SITL_HOME_LON + 0.00018, SITL_HOME_LAT + 0.00014],
        [SITL_HOME_LON, SITL_HOME_LAT + 0.00014],
    ]
    plan = MissionPlanner().generate(
        polygon_lonlat=None,
        mode="waypoints",
        waypoint_path_lonlat=path,
        altitude_m=10.0,
        speed_m_s=8.0,
        inspection_dwell_s=0.0,
        waypoint_heading_mode="tangent",
        waypoint_capture_enabled=True,
    )
    items = tuple(item.to_dict() for item in build_mission_items(plan, include_rth=False))
    return PlannedSITLMission(plan=plan, items=items)
