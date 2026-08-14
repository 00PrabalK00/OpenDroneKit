"""Mission and drone adapters that delegate to the existing core implementation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from core.drone import CommandResult, DroneClient, DroneTelemetry
from core.mission_manager import MissionPlanRequest, generate_mission


def plan_mission(
    request: MissionPlanRequest | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Generate through ``core.mission_manager``; the SDK owns no second planner."""

    return generate_mission(request, **overrides)


class DroneSession:
    """Stable context-managed facade over any registered ``DroneClient`` adapter."""

    def __init__(self, adapter: DroneClient, connection_uri: str) -> None:
        if not isinstance(adapter, DroneClient):
            raise TypeError("adapter does not satisfy the OpenDroneKit DroneClient protocol.")
        if not connection_uri.strip():
            raise ValueError("connection_uri cannot be empty.")
        self.adapter = adapter
        self.connection_uri = connection_uri

    def __enter__(self) -> "DroneSession":
        self.adapter.connect(self.connection_uri)
        return self

    def __exit__(self, *_: object) -> None:
        self.adapter.disconnect()

    def telemetry(self) -> DroneTelemetry:
        return self.adapter.get_telemetry()

    def telemetry_dict(self) -> dict[str, Any]:
        return asdict(self.telemetry())

    def upload(self, mission_items: list[dict[str, Any]]) -> CommandResult:
        if not mission_items:
            raise ValueError("mission_items cannot be empty.")
        return self.adapter.upload_mission(mission_items)
