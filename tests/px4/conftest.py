"""Opt-in fixtures for tests that fly the app against a real PX4 autopilot.

PX4 v1.15.4 SITL runs in Docker (tools/sitl/px4_docker.py); nothing here simulates the
vehicle. Each test gets a freshly started PX4, so no test inherits another's mission,
mode, battery or failsafe state.

Opt-in, exactly like the ArduPilot SITL suite: ``pytest -m px4 tests/px4``. Without
Docker or the image the tests are skipped with the reason; they are never faked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.sitl.px4_docker import (API_PORT, GCS_PORT, PX4SITL, PX4StartupError,
                                   PX4Unavailable)

from ._support import Observer


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "px4: requires an explicitly selected run against real PX4 SITL in Docker")


def _selected(config) -> bool:
    # A broad expression such as `-m "not slow"` must not start an autopilot.
    return str(config.getoption("markexpr") or "").strip() == "px4"


def pytest_collection_modifyitems(config, items) -> None:
    if _selected(config):
        return
    skip = pytest.mark.skip(reason="PX4 SITL tests are opt-in; run pytest -m px4")
    for item in items:
        if item.get_closest_marker("px4") is not None:
            item.add_marker(skip)


def evidence_dir() -> Path:
    path = Path(os.environ.get("ODK_PX4_EVIDENCE_DIR", "") or "px4-evidence")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def px4(request):
    """A fresh PX4, ready for takeoff; its console log is kept as evidence."""
    if not _selected(request.config):
        pytest.skip("PX4 SITL tests are opt-in; run pytest -m px4")
    sitl = PX4SITL()
    try:
        sitl.start()
    except PX4Unavailable as exc:
        pytest.skip(f"PX4 SITL is unavailable: {exc}")
    except PX4StartupError as exc:
        pytest.fail(f"PX4 SITL was found but failed to start:\n{exc}", pytrace=False)
    try:
        yield sitl
    finally:
        name = request.node.name.replace("/", "_").replace("[", "-").replace("]", "")
        (evidence_dir() / f"{name}.px4.log").write_text(sitl.logs(), encoding="utf-8")
        sitl.stop()


# Which UDP port each driver listens on, and so which one the independent observer
# takes: PX4 sends the same vehicle to both.
DRIVERS = {
    "mavsdk": ("mavsdk", f"udp://:{API_PORT}", GCS_PORT),
    "pymavlink": ("mavlink", f"udpin:0.0.0.0:{GCS_PORT}", API_PORT),
}


@pytest.fixture(params=sorted(DRIVERS))
def driver(request):
    name = request.param
    if name == "mavsdk":
        from core.drone import _load_mavsdk

        try:
            _load_mavsdk()
        except RuntimeError as exc:
            pytest.skip(f"MAVSDK-Python (gRPC API) is not installed here: {exc}")
    return DRIVERS[name]


@pytest.fixture
def observer(px4, driver, request):
    """An independent MAVLink stream from the same PX4, on the port the app is not using."""
    obs = Observer(driver[2])
    try:
        obs.wait_vehicle()
        yield obs
    finally:
        obs.stop()
        name = request.node.name.replace("/", "_").replace("[", "-").replace("]", "")
        obs.write(evidence_dir() / f"{name}.mavlink.jsonl")
