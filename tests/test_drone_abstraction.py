"""The drone abstraction, and the boundary it exists to hold.

The value of an abstraction layer is not that it exists but that nothing routes around
it. One `from pymavlink import mavutil` in a mission module is invisible in review,
costs nothing that day, and quietly makes the mission engine unusable with any aircraft
that does not speak MAVLink. That is the kind of decay only a test catches, so the
architectural check below reads the source and fails when a vendor SDK is imported
outside the adapter that owns it.

The rest is about honesty at the boundary. A driver that cannot do something must say
so, because a pilot pressing a button needs to know whether the aircraft heard it. A
simulated driver must never be mistakable for a real one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.drone import (
    CommandResult,
    DroneClient,
    DroneTelemetry,
    MockDroneClient,
    create_drone_client,
)

REPO = Path(__file__).resolve().parents[1]

# Vendor and protocol SDKs that must stay behind an adapter.
VENDOR_MODULES = {"mavsdk", "pymavlink", "dji", "olympe", "djitellopy"}

# Where each SDK is allowed to appear, relative to the repository root. Anything else
# importing them is a layering violation, whatever it is trying to do.
ADAPTER_FILES = {
    "mavsdk": {"core/drone.py"},
    "pymavlink": {
        "core/mission_planner_bridge.py",
        # A capability probe, not a dependency: the UI asks whether the library is
        # present so it can grey out MAVLink options rather than offering them and
        # failing. Listed explicitly so it stays visible.
        "app/api.py",
    },
}

SEARCH_ROOTS = ("core", "mission", "app", "services", "sdk")


def python_files():
    for root in SEARCH_ROOTS:
        directory = REPO / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def imported_vendor_modules(path: Path) -> set[str]:
    """Top-level vendor packages imported anywhere in a file, including inside functions."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - would fail elsewhere first
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in VENDOR_MODULES:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in VENDOR_MODULES:
                found.add(root)
    return found


class TestLayering:
    def test_no_vendor_sdk_is_imported_outside_its_adapter(self):
        """One stray import makes the engine unusable with other aircraft."""
        violations: list[str] = []

        for path in python_files():
            relative = path.relative_to(REPO).as_posix()
            for module in imported_vendor_modules(path):
                allowed = ADAPTER_FILES.get(module, set())
                if relative not in allowed:
                    violations.append(f"{relative} imports {module}")

        assert not violations, (
            "vendor SDKs must stay behind the drone abstraction: "
            + "; ".join(sorted(violations))
        )

    def test_the_mission_engine_depends_on_no_drone_sdk_at_all(self):
        """Mission geometry must be computable with no aircraft in the room."""
        for path in (REPO / "mission").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            assert not imported_vendor_modules(path), (
                f"{path.relative_to(REPO)} imports a drone SDK; mission planning must "
                "not depend on one"
            )

    def test_the_adapters_named_here_actually_exist(self):
        """Otherwise this test passes by describing files nobody has."""
        for module, files in ADAPTER_FILES.items():
            for relative in files:
                assert (REPO / relative).exists(), f"{relative} (adapter for {module})"


class TestProtocolConformance:
    def test_the_mock_satisfies_the_protocol(self):
        assert isinstance(MockDroneClient(), DroneClient)

    def test_every_driver_the_factory_builds_satisfies_the_protocol(self):
        """A driver missing a method fails at the worst moment otherwise."""
        for driver in ("mock", "mavsdk", "mission_planner"):
            client = create_drone_client(driver)
            assert isinstance(client, DroneClient), f"{driver} does not satisfy DroneClient"

    def test_the_protocol_covers_the_commands_a_ground_station_needs(self):
        required = {
            "connect", "disconnect", "is_connected", "get_telemetry",
            "upload_mission", "start_mission", "pause_mission", "resume_mission",
            "return_to_home", "abort_mission", "set_flight_mode",
        }
        missing = sorted(
            name for name in required if not callable(getattr(DroneClient, name, None))
        )
        assert not missing, f"DroneClient is missing ground-station commands: {missing}"


class TestFactory:
    def test_aliases_reach_the_same_driver(self):
        from core.mission_planner_bridge import MissionPlannerDroneClient

        for alias in ("mission_planner", "mp", "pymavlink", "ardupilot"):
            assert isinstance(create_drone_client(alias), MissionPlannerDroneClient)

    def test_an_unknown_driver_is_refused_and_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="Unknown drone driver"):
            create_drone_client("holographic-quadcopter")

        with pytest.raises(ValueError, match="mock"):
            create_drone_client("holographic-quadcopter")

    def test_the_default_driver_is_the_simulated_one(self):
        """Defaulting to a real link would try to fly something on import."""
        assert isinstance(create_drone_client(), MockDroneClient)


class TestHonestyAtTheBoundary:
    def test_an_unconnected_client_refuses_commands_rather_than_reporting_success(self):
        client = create_drone_client("mavsdk")
        result = client.start_mission()
        assert result.success is False
        assert "not connected" in result.message.lower()

    def test_a_command_a_driver_cannot_perform_says_so(self):
        """MAVSDK's upload is not wired; reporting success would be worse than refusing."""
        client = create_drone_client("mavsdk")
        result = client.upload_mission([])
        assert result.success is False
        assert result.message

    def test_a_missing_sdk_raises_instead_of_pretending_to_connect(self):
        """A no-op client would let a pilot believe a mission had been uploaded."""
        try:
            import mavsdk  # noqa: F401
        except ImportError:
            client = create_drone_client("mavsdk")
            with pytest.raises(RuntimeError, match="MAVSDK not installed"):
                client.connect("udp://:14540")
        else:
            pytest.skip("mavsdk is installed, so the missing-SDK path cannot be exercised.")

    def test_the_mock_reports_telemetry_without_claiming_a_vehicle(self):
        client = MockDroneClient()
        assert client.is_connected() is False
        assert isinstance(client.get_telemetry(), DroneTelemetry)

    def test_a_command_result_carries_the_command_it_answers(self):
        """So a UI cannot attribute a failure to the wrong button."""
        result = create_drone_client("mavsdk").start_mission()
        assert isinstance(result, CommandResult)
        assert result.command == "start_mission"


class TestMockBehaviour:
    def test_the_mock_connects_and_disconnects(self):
        client = MockDroneClient()
        client.connect("mock://vehicle")
        assert client.is_connected() is True
        client.disconnect()
        assert client.is_connected() is False

    def test_the_mock_accepts_a_mission_and_reports_it(self):
        client = MockDroneClient()
        client.connect("mock://vehicle")
        result = client.upload_mission([{"seq": 0}, {"seq": 1}])
        assert result.success is True
