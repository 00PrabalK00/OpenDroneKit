"""Stable developer-facing OpenDroneKit SDK surface."""

from .client import ApiError, JobResult, OpenDroneKitClient
from .mission import DroneSession, plan_mission
from .plugins import (
    PLUGIN_API_VERSION,
    PluginKind,
    PluginRegistry,
    PluginSpec,
    registry,
)

from core.drone import CommandResult, DroneClient, DroneTelemetry
from core.mission_manager import MissionPlanRequest

__all__ = [
    "ApiError",
    "CommandResult",
    "DroneClient",
    "DroneSession",
    "DroneTelemetry",
    "JobResult",
    "MissionPlanRequest",
    "OpenDroneKitClient",
    "PLUGIN_API_VERSION",
    "PluginKind",
    "PluginRegistry",
    "PluginSpec",
    "plan_mission",
    "registry",
]
