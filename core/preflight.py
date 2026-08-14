"""Preflight checks — verify drone, GPS, battery, RTH, mission and operator confirmations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .drone import DroneClient, DroneTelemetry
from .errors import AppError, ERR_PREFLIGHT_FAILED
from .settings import DroneProfile, validate_safety_settings


SEVERITY_PASS = "pass"
SEVERITY_WARN = "warning"
SEVERITY_BLOCK = "blocking"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    id: str
    name: str
    severity: str            # pass | warning | blocking
    message: str
    fix_action: str | None = None
    requires_manual: bool = False
    confirmed: bool = False
    operator_note: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    @property
    def ok(self) -> bool:
        if self.severity == SEVERITY_BLOCK and not self.confirmed:
            return False
        if self.requires_manual and not self.confirmed:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


@dataclass
class PreflightReport:
    id: str
    project_id: str
    mission_id: str
    drone_id: str
    checks: list[CheckResult] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    @property
    def blocking_issues(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == SEVERITY_BLOCK and not c.confirmed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == SEVERITY_WARN]

    @property
    def passed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == SEVERITY_PASS]

    @property
    def can_start(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "mission_id": self.mission_id,
            "drone_id": self.drone_id,
            "checks": [c.to_dict() for c in self.checks],
            "blocking_count": len(self.blocking_issues),
            "warning_count": len(self.warnings),
            "can_start": self.can_start,
            "created_at": self.created_at,
        }


# ── Individual checks ─────────────────────────────────────────────────────────

def check_drone_connection(drone: DroneClient) -> CheckResult:
    connected = False
    try:
        connected = bool(drone.is_connected())
    except Exception as exc:
        return CheckResult(
            id="drone_connection",
            name="Drone Connection",
            severity=SEVERITY_BLOCK,
            message=f"Drone client error: {exc}",
            fix_action="Reconnect drone or restart the app.",
        )
    if connected:
        return CheckResult(
            id="drone_connection",
            name="Drone Connection",
            severity=SEVERITY_PASS,
            message="Drone connected and reachable.",
        )
    return CheckResult(
        id="drone_connection",
        name="Drone Connection",
        severity=SEVERITY_BLOCK,
        message="Drone not connected.",
        fix_action="Connect drone before flight.",
    )


def check_gps_status(telemetry: DroneTelemetry) -> CheckResult:
    if telemetry.gps_fix >= 5:
        return CheckResult("gps_status", "GPS / RTK Lock", SEVERITY_PASS,
                           f"RTK fix, {telemetry.satellites} sats, HDOP {telemetry.hdop:.1f}.")
    if telemetry.gps_fix >= 3 and telemetry.satellites >= 8:
        return CheckResult("gps_status", "GPS Lock", SEVERITY_PASS,
                           f"3D fix, {telemetry.satellites} sats, HDOP {telemetry.hdop:.1f}.")
    if telemetry.gps_fix >= 3:
        return CheckResult("gps_status", "GPS Lock", SEVERITY_WARN,
                           f"Marginal GPS: {telemetry.satellites} sats, HDOP {telemetry.hdop:.1f}.",
                           fix_action="Wait for more satellites or move to open sky.")
    return CheckResult("gps_status", "GPS Lock", SEVERITY_BLOCK,
                       f"No GPS fix (fix_type={telemetry.gps_fix}, sats={telemetry.satellites}).",
                       fix_action="Wait for GPS fix before takeoff.")


def check_home_position(telemetry: DroneTelemetry) -> CheckResult:
    if telemetry.home_set:
        return CheckResult("home_position", "Home Position", SEVERITY_PASS,
                           f"Home set ({telemetry.home_lat:.5f}, {telemetry.home_lon:.5f}).")
    return CheckResult("home_position", "Home Position", SEVERITY_BLOCK,
                       "Home position not set.",
                       fix_action="Arm drone or set home manually.")


def check_battery_status(
    telemetry: DroneTelemetry,
    warn_pct: float = 30.0,
    critical_pct: float = 15.0,
    mission_reserve_pct: float = 30.0,
) -> CheckResult:
    pct = float(telemetry.battery_pct)
    if pct < critical_pct:
        return CheckResult("battery", "Battery", SEVERITY_BLOCK,
                           f"Battery critically low: {pct:.0f}%.",
                           fix_action="Replace battery before flight.")
    if pct < mission_reserve_pct:
        return CheckResult("battery", "Battery", SEVERITY_WARN,
                           f"Battery {pct:.0f}% below reserve threshold ({mission_reserve_pct:.0f}%).",
                           fix_action="Consider charging or shortening mission.")
    return CheckResult("battery", "Battery", SEVERITY_PASS,
                       f"Battery {pct:.0f}% ({telemetry.battery_v:.1f} V).")


def check_storage_ready(free_mb_required: float = 500.0, free_mb_available: float | None = None) -> CheckResult:
    if free_mb_available is None:
        return CheckResult("storage", "Onboard Storage", SEVERITY_WARN,
                           "Storage status unknown.", requires_manual=True,
                           fix_action="Verify SD card before takeoff.")
    if free_mb_available < free_mb_required:
        return CheckResult("storage", "Onboard Storage", SEVERITY_BLOCK,
                           f"Only {free_mb_available:.0f} MB free; need {free_mb_required:.0f} MB.",
                           fix_action="Format / replace SD card.")
    return CheckResult("storage", "Onboard Storage", SEVERITY_PASS,
                       f"{free_mb_available:.0f} MB free.")


def check_camera_ready(payload_ok: bool = True, notes: str = "") -> CheckResult:
    if payload_ok:
        return CheckResult("camera", "Camera / Payload", SEVERITY_PASS, "Camera ready." + (f" {notes}" if notes else ""))
    return CheckResult("camera", "Camera / Payload", SEVERITY_BLOCK,
                       "Camera not detected.", fix_action="Check gimbal and SD card.")


def check_rth_altitude(profile: DroneProfile) -> CheckResult:
    msgs = validate_safety_settings(profile)
    rth_issues = [m for m in msgs if m.field in ("rth_altitude_m", "max_altitude_m", "min_altitude_m")]
    if rth_issues:
        first = rth_issues[0]
        return CheckResult("rth_altitude", "RTH Altitude", SEVERITY_BLOCK,
                           first.message, fix_action=first.fix_action or "Adjust drone profile.")
    return CheckResult("rth_altitude", "RTH Altitude", SEVERITY_PASS,
                       f"RTH {profile.rth_altitude_m:.0f} m within bounds.")


def check_mission_valid(mission_summary: dict[str, Any] | None) -> CheckResult:
    if not mission_summary:
        return CheckResult("mission_valid", "Mission Valid", SEVERITY_BLOCK,
                           "No mission selected.",
                           fix_action="Open Mission Planner and load a mission.")
    issues = list(mission_summary.get("issues") or [])
    if any(str(i).lower().startswith("error") for i in issues):
        return CheckResult("mission_valid", "Mission Valid", SEVERITY_BLOCK,
                           f"Mission has {len(issues)} validation error(s).",
                           fix_action="Fix mission in planner.", details={"issues": issues})
    if issues:
        return CheckResult("mission_valid", "Mission Valid", SEVERITY_WARN,
                           f"Mission validated with {len(issues)} warning(s).",
                           details={"issues": issues})
    return CheckResult("mission_valid", "Mission Valid", SEVERITY_PASS, "Mission validated.")


def check_mission_uploaded(uploaded: bool, drone_connected: bool) -> CheckResult:
    if not drone_connected:
        return CheckResult("mission_uploaded", "Mission Uploaded", SEVERITY_WARN,
                           "Drone not connected — upload pending.",
                           fix_action="Connect drone, then upload mission.")
    if uploaded:
        return CheckResult("mission_uploaded", "Mission Uploaded", SEVERITY_PASS, "Mission uploaded to drone.")
    return CheckResult("mission_uploaded", "Mission Uploaded", SEVERITY_BLOCK,
                       "Mission not uploaded yet.",
                       fix_action="Press Upload Mission.")


def check_geofence(mission_summary: dict[str, Any] | None, geofence: dict[str, Any] | None) -> CheckResult:
    if not geofence:
        return CheckResult("geofence", "Geofence", SEVERITY_WARN,
                           "No geofence configured.", requires_manual=True,
                           fix_action="Confirm flight area manually.")
    inside = bool(geofence.get("mission_inside", True))
    if inside:
        return CheckResult("geofence", "Geofence", SEVERITY_PASS, "Mission inside geofence.")
    return CheckResult("geofence", "Geofence", SEVERITY_BLOCK,
                       "Mission exits geofence.",
                       fix_action="Adjust path or expand geofence.")


def check_no_fly_zones(mission_summary: dict[str, Any] | None, no_fly_zones: list[dict[str, Any]] | None) -> CheckResult:
    if not no_fly_zones:
        return CheckResult("no_fly_zones", "No-Fly Zones", SEVERITY_PASS, "No NFZs declared.")
    conflicts = [z for z in no_fly_zones if z.get("conflict")]
    if conflicts:
        return CheckResult("no_fly_zones", "No-Fly Zones", SEVERITY_BLOCK,
                           f"{len(conflicts)} no-fly zone conflict(s).",
                           fix_action="Reroute mission to clear NFZs.",
                           details={"conflicts": conflicts})
    return CheckResult("no_fly_zones", "No-Fly Zones", SEVERITY_PASS, f"{len(no_fly_zones)} NFZs clear.")


def check_weather_acknowledged(acknowledged: bool = False) -> CheckResult:
    if acknowledged:
        return CheckResult("weather", "Weather Acknowledged", SEVERITY_PASS, "Operator acknowledged weather.", confirmed=True)
    return CheckResult("weather", "Weather Acknowledged", SEVERITY_WARN,
                       "Confirm wind, precipitation and visibility.", requires_manual=True,
                       fix_action="Acknowledge weather check.")


def check_obstacle_avoidance(profile: DroneProfile) -> CheckResult:
    mode = (profile.obstacle_avoidance or "off").lower()
    if mode in ("medium", "high"):
        return CheckResult("obstacle_avoidance", "Obstacle Avoidance", SEVERITY_PASS, f"{mode} OA enabled.")
    if mode == "low":
        return CheckResult("obstacle_avoidance", "Obstacle Avoidance", SEVERITY_WARN,
                           "Low OA — verify clear flight path.",
                           requires_manual=True)
    return CheckResult("obstacle_avoidance", "Obstacle Avoidance", SEVERITY_WARN,
                       "OA disabled — verify visual line of sight.",
                       requires_manual=True)


def check_operator_confirmation() -> CheckResult:
    return CheckResult("operator_confirm", "Operator Confirmation", SEVERITY_WARN,
                       "Operator must confirm flight readiness.",
                       requires_manual=True,
                       fix_action="Acknowledge operator confirmation.")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_preflight(
    project_id: str,
    mission_id: str,
    drone_id: str,
    drone: DroneClient,
    drone_profile: DroneProfile,
    mission_summary: dict[str, Any] | None = None,
    mission_uploaded: bool = False,
    geofence: dict[str, Any] | None = None,
    no_fly_zones: list[dict[str, Any]] | None = None,
    weather_acknowledged: bool = False,
    free_storage_mb: float | None = None,
    gimbal_ok: bool | None = None,
) -> PreflightReport:
    telemetry: DroneTelemetry
    try:
        telemetry = drone.get_telemetry()
    except Exception:
        telemetry = DroneTelemetry()

    checks: list[CheckResult] = []
    checks.append(check_drone_connection(drone))
    if telemetry.connected:
        checks.append(check_gps_status(telemetry))
        checks.append(check_home_position(telemetry))
        checks.append(check_battery_status(telemetry,
                                           warn_pct=drone_profile.battery_warn_pct,
                                           critical_pct=drone_profile.battery_critical_pct))
        # A bad compass arms perfectly happily and is the classic cause of a fly-away.
        checks.append(check_compass(telemetry))
        checks.append(check_imu(telemetry))
    checks.append(check_camera_ready())
    checks.append(check_gimbal(gimbal_ok))
    checks.append(check_storage_ready(free_mb_available=free_storage_mb))
    checks.append(check_mission_valid(mission_summary))
    checks.append(check_mission_uploaded(mission_uploaded, drone.is_connected()))
    checks.append(check_rth_altitude(drone_profile))
    checks.append(check_geofence(mission_summary, geofence))
    checks.append(check_no_fly_zones(mission_summary, no_fly_zones))
    checks.append(check_weather_acknowledged(weather_acknowledged))
    checks.append(check_obstacle_avoidance(drone_profile))
    checks.append(check_operator_confirmation())

    return PreflightReport(
        id=str(uuid.uuid4()),
        project_id=project_id,
        mission_id=mission_id,
        drone_id=drone_id,
        checks=checks,
    )


def confirm_manual_check(report: PreflightReport, check_id: str, operator_note: str = "") -> CheckResult:
    """Save manual confirmation for a check that requires human review."""
    for c in report.checks:
        if c.id == check_id:
            c.confirmed = True
            c.operator_note = operator_note
            c.timestamp = _now_iso()
            return c
    raise AppError(ERR_PREFLIGHT_FAILED, f"Check not found: {check_id}")


def can_start_mission(report: PreflightReport) -> bool:
    """Return True only when all blocking checks pass / are confirmed."""
    return report.can_start


# ── Sensor health ─────────────────────────────────────────────────────────────

# MAV_SYS_STATUS_SENSOR bit positions, from the MAVLink common message set. The
# autopilot reports three parallel bitmasks: which sensors exist, which are enabled,
# and which are healthy.
SENSOR_BITS = {
    "gyro": 1 << 0,        # MAV_SYS_STATUS_SENSOR_3D_GYRO
    "accelerometer": 1 << 1,  # MAV_SYS_STATUS_SENSOR_3D_ACCEL
    "compass": 1 << 2,     # MAV_SYS_STATUS_SENSOR_3D_MAG
}


def _sensor_state(telemetry: DroneTelemetry, sensor: str) -> str:
    """Report a sensor as healthy, unhealthy, absent, or unknown.

    Unknown is a distinct answer and must stay that way. A vehicle that never sent
    SYS_STATUS has not told us its compass is fine, and recording that silence as a
    pass would put a tick against a sensor nobody asked about.
    """
    raw = getattr(telemetry, "raw", None) or {}
    if "sensors_health" not in raw:
        return "unknown"

    bit = SENSOR_BITS[sensor]
    present = int(raw.get("sensors_present", 0)) & bit
    healthy = int(raw.get("sensors_health", 0)) & bit

    if not present:
        return "absent"
    return "healthy" if healthy else "unhealthy"


def check_compass(telemetry: DroneTelemetry) -> CheckResult:
    """A bad compass is the classic cause of a fly-away, and it arms perfectly happily."""
    state = _sensor_state(telemetry, "compass")
    if state == "healthy":
        return CheckResult("compass", "Compass", SEVERITY_PASS,
                           "Magnetometer reporting healthy.")
    if state == "unhealthy":
        return CheckResult("compass", "Compass", SEVERITY_BLOCK,
                           "The autopilot reports the magnetometer as unhealthy.",
                           fix_action="Calibrate the compass and move away from metal "
                                      "structures or vehicles before retrying.")
    if state == "absent":
        return CheckResult("compass", "Compass", SEVERITY_WARN,
                           "No magnetometer is present on this vehicle.",
                           fix_action="Confirm the airframe is intended to fly without "
                                      "one before proceeding.")
    return CheckResult("compass", "Compass", SEVERITY_WARN,
                       "The vehicle has not reported compass health.",
                       fix_action="Check it manually. An unreported sensor is not a "
                                  "healthy sensor.")


def check_imu(telemetry: DroneTelemetry) -> CheckResult:
    """Gyro and accelerometer together: an aircraft cannot stabilise without both."""
    gyro = _sensor_state(telemetry, "gyro")
    accel = _sensor_state(telemetry, "accelerometer")

    if "unhealthy" in (gyro, accel):
        faulty = ", ".join(n for n, s in (("gyroscope", gyro), ("accelerometer", accel))
                           if s == "unhealthy")
        return CheckResult("imu", "IMU", SEVERITY_BLOCK,
                           f"The autopilot reports {faulty} as unhealthy.",
                           fix_action="Calibrate the IMU with the aircraft level and "
                                      "still, and let it reach a stable temperature.")
    if gyro == "healthy" and accel == "healthy":
        return CheckResult("imu", "IMU", SEVERITY_PASS,
                           "Gyroscope and accelerometer reporting healthy.")
    if "unknown" in (gyro, accel):
        return CheckResult("imu", "IMU", SEVERITY_WARN,
                           "The vehicle has not reported IMU health.",
                           fix_action="Check it manually before arming.")
    return CheckResult("imu", "IMU", SEVERITY_WARN,
                       "The IMU is not reported as present on this vehicle.",
                       fix_action="Confirm the autopilot is configured correctly.")


def check_gimbal(gimbal_ok: bool | None = None, notes: str = "") -> CheckResult:
    """Gimbal readiness.

    ``None`` means nothing reported it, which is a warning rather than a pass: a survey
    flown with a stuck gimbal produces imagery pointing somewhere other than planned,
    and nothing downstream will notice until the reconstruction comes out wrong.
    """
    if gimbal_ok is True:
        return CheckResult("gimbal", "Gimbal", SEVERITY_PASS,
                           notes or "Gimbal reports ready.")
    if gimbal_ok is False:
        return CheckResult("gimbal", "Gimbal", SEVERITY_BLOCK,
                           notes or "The gimbal is not ready.",
                           fix_action="Power-cycle the payload and confirm the gimbal "
                                      "moves freely and is not caught on its lock.")
    return CheckResult("gimbal", "Gimbal", SEVERITY_WARN,
                       "Gimbal status was not reported.",
                       fix_action="Confirm by eye that the gimbal is unlocked and level.",
                       requires_manual=True)
