"""High-level mission management facade.

MissionManager gives UI and scripts one stable surface for the mission flow:

1. generate a MissionPlan from operator settings
2. validate and persist mission artifacts
3. run preflight checks
4. connect/upload/start/pause/resume/RTH/abort through the drone client
5. optionally run generation in a background worker
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import uuid
from typing import Any

from mission import (
    MissionPlan,
    MissionPlanner,
    export_flight_recipe,
    export_geojson,
    export_qgc_wpl,
    load_flight_recipe,
)

from .drone import CommandResult, DroneClient, DroneTelemetry, create_drone_client
from .errors import AppError, ERR_DRONE_NOT_CONNECTED, ERR_INVALID_INPUT, ERR_MISSION_INVALID
from .events import MISSION_CREATED, MISSION_EXPORTED, MISSION_UPDATED, MISSION_VALIDATED, publish_event
from .flight import FlightManager, FlightSession, get_flight_manager
from .preflight import PreflightReport, run_preflight
from .settings import DroneProfile
from .workers import ProgressCallback, WorkerHandle, WorkerPool, get_worker_pool


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: str, fallback: str = "mission") -> str:
    raw = str(value or "").strip() or fallback
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")
    return out or fallback


def _mission_id(plan: MissionPlan | None = None, mission_name: str = "") -> str:
    if plan is not None and isinstance(plan.flight_recipe, dict):
        rid = str(plan.flight_recipe.get("recipe_id", "") or "").strip()
        if rid:
            return rid
    return _safe_name(mission_name, "mission") + "_" + uuid.uuid4().hex[:8]


@dataclass
class MissionPlanRequest:
    """Common mission inputs accepted by MissionManager.generate_plan."""

    mission_name: str = "Mission"
    polygon_lonlat: list[list[float]] | None = None
    altitude_m: float = 60.0
    front_overlap_pct: float = 80.0
    side_overlap_pct: float = 70.0
    speed_m_s: float = 5.0
    mode: str = "grid"
    camera: str = "mavic2pro"
    drone_id: str = "mock-drone"
    asset_type: str = "custom"
    inspection_type: str = "visual_survey"
    emergency_landing_zone: str = ""
    weather_safe: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_generate_kwargs(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = dict(payload.pop("extra") or {})
        for non_planner_key in (
            "mission_name",
            "drone_id",
            "asset_type",
            "inspection_type",
            "emergency_landing_zone",
            "weather_safe",
        ):
            payload.pop(non_planner_key, None)
        payload.update(extra)
        return payload


@dataclass
class MissionValidationCheck:
    id: str
    label: str
    ok: bool
    level: str = "ok"  # ok | warning | error
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissionSaveResult:
    mission_id: str
    mission_name: str
    output_dir: str
    mission_json_path: str
    recipe_path: str
    geojson_path: str
    qgc_wpl_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MissionManager:
    """Facade over mission planning, validation, persistence, preflight, and flight."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        project_id: str = "",
        planner: MissionPlanner | None = None,
        drone: DroneClient | None = None,
        flight_manager: FlightManager | None = None,
        worker_pool: WorkerPool | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root is not None else Path("final_toolkit_outputs") / "missions"
        self.project_id = str(project_id or "default_project")
        self.planner = planner or MissionPlanner()
        self.drone: DroneClient = drone or create_drone_client("mock")
        self.flight_manager = flight_manager or get_flight_manager()
        self.worker_pool = worker_pool or get_worker_pool()
        self.current_plan: MissionPlan | None = None
        self.current_save: MissionSaveResult | None = None
        self.current_preflight: PreflightReport | None = None
        self.current_flight: FlightSession | None = None

    # Planning -----------------------------------------------------------------

    def generate_plan(self, request: MissionPlanRequest | dict[str, Any] | None = None, **overrides: Any) -> MissionPlan:
        payload = self._normalize_request(request, overrides)
        self._validate_generate_payload(payload)
        plan = self.planner.generate(**payload)
        self.current_plan = plan
        publish_event(MISSION_CREATED, {"project_id": self.project_id, "summary": self.summarize_plan(plan)})
        return plan

    def generate_plan_async(
        self,
        request: MissionPlanRequest | dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        **overrides: Any,
    ) -> WorkerHandle[MissionPlan]:
        def _run() -> MissionPlan:
            return self.generate_plan(request, **overrides)

        return self.worker_pool.submit(
            "generate_mission_plan",
            _run,
            use_context=False,
            progress_callback=progress_callback,
            metadata={"project_id": self.project_id},
        )

    def _normalize_request(
        self,
        request: MissionPlanRequest | dict[str, Any] | None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        if request is None:
            payload: dict[str, Any] = {}
        elif isinstance(request, MissionPlanRequest):
            payload = request.to_generate_kwargs()
        elif isinstance(request, dict):
            req = dict(request)
            extra = dict(req.pop("extra", {}) or {})
            for non_planner_key in (
                "mission_name",
                "drone_id",
                "asset_type",
                "inspection_type",
                "emergency_landing_zone",
                "weather_safe",
            ):
                req.pop(non_planner_key, None)
            req.update(extra)
            payload = req
        else:
            raise AppError(ERR_INVALID_INPUT, "Mission request must be a dict or MissionPlanRequest.")
        payload.update(overrides)
        allowed = set(inspect.signature(self.planner.generate).parameters.keys())
        return {k: v for k, v in payload.items() if k in allowed}

    @staticmethod
    def _validate_generate_payload(payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", "grid") or "grid")
        polygon = payload.get("polygon_lonlat")
        line = payload.get("linear_path_lonlat") or payload.get("lateral_target_path_lonlat") or payload.get("waypoint_path_lonlat")
        center = payload.get("orbit_center_lonlat") or payload.get("panorama_center_lonlat") or payload.get("bubble_center_lonlat") or payload.get("tower_center_lonlat")
        if mode in {"grid", "double_grid", "roof_inspection", "facade", "facade_mapping", "solar_inspection", "magnetic_mapping"}:
            if not polygon:
                raise AppError(ERR_MISSION_INVALID, "Mission polygon is required.", recovery_action="Draw or import a survey polygon.")
        if mode in {"linear_inspection", "lateral_capture", "waypoints"} and not line and not polygon:
            raise AppError(ERR_MISSION_INVALID, "Mission line is required.", recovery_action="Draw or import an inspection line.")
        if mode in {"orbit", "panorama", "bubble_360", "tower_mapping"} and not center and not polygon:
            raise AppError(ERR_MISSION_INVALID, "Mission center point is required.", recovery_action="Place a center point.")

    # Validation / summaries ----------------------------------------------------

    def validate_plan(
        self,
        plan: MissionPlan | None = None,
        *,
        weather_safe: bool = True,
        emergency_landing_zone: str = "",
        no_fly_conflict_count: int | None = None,
    ) -> list[MissionValidationCheck]:
        plan = plan or self.current_plan
        if plan is None:
            return [
                MissionValidationCheck(
                    "mission_exists",
                    "Mission generated",
                    False,
                    "error",
                    "Generate a mission before validation.",
                )
            ]
        coverage = plan.expected_coverage or {}
        required_battery_pct = min(100.0, max(10.0, float(plan.estimated_time_min) / 22.0 * 100.0))
        no_fly_count = (
            int(no_fly_conflict_count)
            if no_fly_conflict_count is not None
            else int((plan.safety_adjustments or {}).get("no_fly_projections", 0))
        )
        checks = [
            MissionValidationCheck("path_complete", "Flight path complete", len(plan.waypoints) > 0, message=f"{len(plan.waypoints)} waypoint(s)."),
            MissionValidationCheck(
                "no_fly_conflict",
                "No-fly zone conflict",
                no_fly_count == 0,
                "warning" if no_fly_count else "ok",
                f"{no_fly_count} conflict(s)." if no_fly_count else "Clear.",
            ),
            MissionValidationCheck("battery_sufficient", "Battery sufficient", required_battery_pct <= 85.0, "ok" if required_battery_pct <= 85.0 else "warning", f"Estimated battery required {required_battery_pct:.0f}%."),
            MissionValidationCheck("camera_profile", "Camera profile selected", bool(plan.camera), message=str(plan.camera)),
            MissionValidationCheck("return_path", "Return path available", bool(plan.safety_constraints), message="RTH constraints available."),
            MissionValidationCheck("emergency_lz", "Emergency landing zone assigned", bool(emergency_landing_zone), "ok" if emergency_landing_zone else "warning", emergency_landing_zone or "No emergency landing zone selected."),
            MissionValidationCheck("weather", "Weather safe", bool(weather_safe), "ok" if weather_safe else "warning", "Weather accepted." if weather_safe else "Weather requires operator review."),
            MissionValidationCheck(
                "coverage",
                "Coverage target",
                bool(coverage.get("meets_target", True)),
                "ok" if coverage.get("meets_target", True) else "warning",
                f"{float(coverage.get('achieved_coverage_pct', 0.0)):.1f}% expected coverage.",
            ),
        ]
        publish_event(MISSION_VALIDATED, {"project_id": self.project_id, "checks": [c.to_dict() for c in checks]})
        return checks

    @staticmethod
    def summarize_plan(plan: MissionPlan) -> dict[str, Any]:
        coverage = plan.expected_coverage or {}
        return {
            "template": plan.template,
            "source": plan.source,
            "waypoints": len(plan.waypoints),
            "autopilot_commands": len(plan.autopilot_commands),
            "distance_m": float(plan.path_distance_m),
            "estimated_time_min": float(plan.estimated_time_min),
            "estimated_gsd_cm": float(plan.estimated_gsd_cm),
            "coverage_pct": float(coverage.get("achieved_coverage_pct", 0.0)),
            "repeat_enabled": bool(plan.repeat_enabled),
        }

    # Persistence / import / export -------------------------------------------

    def save_mission(
        self,
        plan: MissionPlan | None = None,
        mission_name: str = "",
        output_dir: str | Path | None = None,
        note: str = "",
    ) -> MissionSaveResult:
        plan = plan or self.current_plan
        if plan is None:
            raise AppError(ERR_MISSION_INVALID, "No mission plan to save.", recovery_action="Generate a mission first.")
        name = _safe_name(mission_name or plan.template or "mission")
        mid = _mission_id(plan, name)
        root = Path(output_dir) if output_dir is not None else self.project_root / "missions" / name
        root.mkdir(parents=True, exist_ok=True)

        summary = self.summarize_plan(plan)
        summary.update({"mission_id": mid, "mission_name": name, "saved_at": _now_iso(), "note": str(note or "")})
        mission_json = root / f"{name}.mission.json"
        recipe_json = root / f"{name}.flight_recipe.json"
        geojson = root / f"{name}.geojson"
        qgc = root / f"{name}.waypoints"
        summary_json = root / f"{name}.summary.json"

        mission_json.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        export_flight_recipe(recipe_json, plan)
        export_geojson(geojson, plan)
        export_qgc_wpl(qgc, plan)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result = MissionSaveResult(
            mission_id=mid,
            mission_name=name,
            output_dir=str(root),
            mission_json_path=str(mission_json),
            recipe_path=str(recipe_json),
            geojson_path=str(geojson),
            qgc_wpl_path=str(qgc),
            summary_path=str(summary_json),
        )
        self.current_save = result
        publish_event(MISSION_UPDATED, {"project_id": self.project_id, **result.to_dict()})
        publish_event(MISSION_EXPORTED, {"project_id": self.project_id, **result.to_dict()})
        return result

    def load_recipe(self, path: str | Path):
        return load_flight_recipe(path)

    def export_plan(self, plan: MissionPlan | None = None, output_dir: str | Path | None = None) -> MissionSaveResult:
        return self.save_mission(plan=plan, output_dir=output_dir)

    # Drone / preflight / flight ------------------------------------------------

    def connect_drone(self, driver: str = "mock", connection_uri: str = "mock://") -> DroneClient:
        self.drone = create_drone_client(driver)
        self.drone.connect(connection_uri)
        return self.drone

    def disconnect_drone(self) -> None:
        self.drone.disconnect()

    def telemetry(self) -> DroneTelemetry:
        return self.drone.get_telemetry()

    def mission_items(self, plan: MissionPlan | None = None) -> list[dict[str, Any]]:
        plan = plan or self.current_plan
        if plan is None:
            raise AppError(ERR_MISSION_INVALID, "No mission plan available.")
        items = [dict(item) for item in (plan.autopilot_commands or []) if isinstance(item, dict)]
        if items:
            return items
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(plan.waypoints):
            if len(row) < 2:
                continue
            out.append(
                {
                    "seq": idx,
                    "command": 16,
                    "frame": 3,
                    "lon": float(row[0]),
                    "lat": float(row[1]),
                    "alt": float(row[2]) if len(row) >= 3 else float(plan.altitude_m),
                }
            )
        return out

    def upload_mission(self, plan: MissionPlan | None = None) -> CommandResult:
        if not self.drone.is_connected():
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Drone is not connected.", recovery_action="Connect a drone before upload.")
        return self.drone.upload_mission(self.mission_items(plan))

    def run_preflight(
        self,
        plan: MissionPlan | None = None,
        *,
        drone_profile: DroneProfile | None = None,
        drone_id: str = "mock-drone",
        mission_uploaded: bool = False,
        emergency_landing_zone: str = "",
        weather_acknowledged: bool = False,
        free_storage_mb: float | None = None,
    ) -> PreflightReport:
        plan = plan or self.current_plan
        if plan is None:
            mission_summary: dict[str, Any] | None = None
            mission_id = ""
        else:
            validation = self.validate_plan(
                plan,
                emergency_landing_zone=emergency_landing_zone,
                weather_safe=weather_acknowledged,
            )
            mission_summary = self.summarize_plan(plan)
            mission_summary["issues"] = [
                f"{c.level}: {c.label} - {c.message}" for c in validation if not c.ok and c.level == "error"
            ]
            mission_id = _mission_id(plan, plan.template)
        report = run_preflight(
            project_id=self.project_id,
            mission_id=mission_id,
            drone_id=str(drone_id or "drone"),
            drone=self.drone,
            drone_profile=drone_profile or DroneProfile(),
            mission_summary=mission_summary,
            mission_uploaded=mission_uploaded,
            geofence={"mission_inside": True},
            no_fly_zones=[],
            weather_acknowledged=weather_acknowledged,
            free_storage_mb=free_storage_mb,
        )
        self.current_preflight = report
        return report

    def start_flight(
        self,
        plan: MissionPlan | None = None,
        *,
        drone_id: str = "mock-drone",
        require_preflight: bool = True,
    ) -> FlightSession:
        plan = plan or self.current_plan
        if plan is None:
            raise AppError(ERR_MISSION_INVALID, "No mission plan available.")
        if require_preflight and (self.current_preflight is None or not self.current_preflight.can_start):
            raise AppError(ERR_MISSION_INVALID, "Preflight checks have not passed.", recovery_action="Run preflight and resolve blockers.")
        if not self.drone.is_connected():
            raise AppError(ERR_DRONE_NOT_CONNECTED, "Drone is not connected.")
        mission_id = _mission_id(plan, plan.template)
        session = self.flight_manager.start_session(
            project_root=self.project_root,
            project_id=self.project_id,
            mission_id=mission_id,
            drone_id=str(drone_id or "drone"),
            drone=self.drone,
        )
        self.current_flight = session
        self.flight_manager.start_mission()
        return session

    def pause_flight(self, reason: str = "operator_pause") -> CommandResult:
        return self.flight_manager.pause_flight(reason=reason)

    def resume_flight(self, reason: str = "operator_resume") -> CommandResult:
        return self.flight_manager.resume_flight(reason=reason)

    def return_to_home(self, reason: str = "operator_rth") -> CommandResult:
        return self.flight_manager.trigger_rth(reason=reason)

    def abort_flight(self, reason: str = "operator_abort") -> CommandResult:
        return self.flight_manager.abort_flight(reason=reason)


_manager: MissionManager | None = None


def get_mission_manager(project_root: str | Path | None = None, project_id: str = "") -> MissionManager:
    global _manager
    if _manager is None:
        _manager = MissionManager(project_root=project_root, project_id=project_id)
    return _manager


def generate_mission(request: MissionPlanRequest | dict[str, Any] | None = None, **overrides: Any) -> MissionPlan:
    return get_mission_manager().generate_plan(request, **overrides)


def save_mission(plan: MissionPlan, mission_name: str = "", output_dir: str | Path | None = None) -> MissionSaveResult:
    return get_mission_manager().save_mission(plan=plan, mission_name=mission_name, output_dir=output_dir)


__all__ = [
    "MissionPlanRequest",
    "MissionValidationCheck",
    "MissionSaveResult",
    "MissionManager",
    "get_mission_manager",
    "generate_mission",
    "save_mission",
]
