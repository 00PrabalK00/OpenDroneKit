"""Layered configuration — application defaults + user settings + project overrides."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .errors import AppError, ERR_INVALID_INPUT
from .validation import ValidationMessage, SEVERITY_ERROR, SEVERITY_WARNING


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class AppSettings:
    theme: str = "dark"
    default_units: str = "metric"        # "metric" | "imperial"
    workspace_root: str = ""
    offline_mode: bool = True
    log_level: str = "INFO"
    auto_save: bool = True
    auto_save_interval_s: int = 60
    show_developer_tools: bool = False
    map_tile_url: str = ""               # local mbtiles path or http URL
    use_gpu: bool = False
    device: str = "cpu"                  # "cpu" | "cuda" | "mps"
    max_workers: int = 4
    model_registry_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DroneProfile:
    name: str = "Generic Quadcopter"
    model: str = "generic"
    payload: str = "rgb_camera"
    max_altitude_m: float = 120.0
    min_altitude_m: float = 2.0
    rth_altitude_m: float = 50.0
    cruise_speed_mps: float = 5.0
    max_speed_mps: float = 15.0
    battery_capacity_mah: int = 5000
    battery_warn_pct: float = 30.0
    battery_critical_pct: float = 15.0
    obstacle_avoidance: str = "off"      # "off" | "low" | "medium" | "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingSettings:
    use_gpu: bool = False
    device: str = "cpu"
    model_registry_path: str = ""
    max_workers: int = 4
    cache_dir: str = ""
    keep_intermediate: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Storage paths ─────────────────────────────────────────────────────────────

def _toolkit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_workspace() -> Path:
    return _toolkit_root().parent / "opendrone_workspace"


def user_config_dir() -> Path:
    """Cross-platform per-user config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    target = base / "OpenDroneKit"
    target.mkdir(parents=True, exist_ok=True)
    return target


def settings_file() -> Path:
    return user_config_dir() / "settings.json"


def drone_profiles_file() -> Path:
    return user_config_dir() / "drone_profiles.json"


def processing_settings_file() -> Path:
    return user_config_dir() / "processing.json"


# ── Public API ────────────────────────────────────────────────────────────────

def load_app_settings() -> AppSettings:
    """Load user settings or return defaults. Missing fields filled from defaults."""
    defaults = AppSettings(workspace_root=str(_default_workspace()))
    sf = settings_file()
    if not sf.exists():
        return defaults
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    merged = defaults.to_dict()
    if isinstance(data, dict):
        for k, v in data.items():
            if k in merged:
                merged[k] = v
    return AppSettings(**merged)


def save_app_settings(settings: AppSettings) -> None:
    """Validate and save settings."""
    msgs = validate_app_settings(settings)
    blocking = [m for m in msgs if m.severity == SEVERITY_ERROR]
    if blocking:
        raise AppError(
            code=ERR_INVALID_INPUT,
            user_message="Settings contain invalid values.",
            technical_message="; ".join(m.message for m in blocking),
            recovery_action="Fix highlighted fields and try again.",
        )
    sf = settings_file()
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def validate_app_settings(settings: AppSettings) -> list[ValidationMessage]:
    msgs: list[ValidationMessage] = []
    if settings.theme not in ("dark", "light"):
        msgs.append(ValidationMessage("theme", SEVERITY_WARNING, f"Unknown theme: {settings.theme}", "Use 'dark' or 'light'."))
    if settings.default_units not in ("metric", "imperial"):
        msgs.append(ValidationMessage("default_units", SEVERITY_ERROR, "Units must be 'metric' or 'imperial'."))
    if settings.max_workers < 1 or settings.max_workers > 32:
        msgs.append(ValidationMessage("max_workers", SEVERITY_ERROR, "max_workers must be 1..32."))
    if settings.device not in ("cpu", "cuda", "mps"):
        msgs.append(ValidationMessage("device", SEVERITY_WARNING, f"Unknown device: {settings.device}", "Use cpu, cuda, or mps."))
    return msgs


# ── Drone profiles ────────────────────────────────────────────────────────────

def load_drone_profiles() -> list[DroneProfile]:
    f = drone_profiles_file()
    if not f.exists():
        return [DroneProfile()]
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return [DroneProfile()]
    out: list[DroneProfile] = []
    for d in (data if isinstance(data, list) else []):
        try:
            allowed = {k: v for k, v in d.items() if k in DroneProfile.__dataclass_fields__}
            out.append(DroneProfile(**allowed))
        except Exception:
            pass
    return out or [DroneProfile()]


def save_drone_profiles(profiles: list[DroneProfile]) -> None:
    f = drone_profiles_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([p.to_dict() for p in profiles], indent=2), encoding="utf-8")


def validate_safety_settings(drone_profile: DroneProfile) -> list[ValidationMessage]:
    """Check altitude, RTH altitude and speed limits."""
    msgs: list[ValidationMessage] = []
    if drone_profile.min_altitude_m < 0:
        msgs.append(ValidationMessage("min_altitude_m", SEVERITY_ERROR, "Minimum altitude cannot be negative."))
    if drone_profile.max_altitude_m <= drone_profile.min_altitude_m:
        msgs.append(ValidationMessage("max_altitude_m", SEVERITY_ERROR, "Max altitude must exceed min altitude."))
    if drone_profile.rth_altitude_m > drone_profile.max_altitude_m:
        msgs.append(ValidationMessage(
            "rth_altitude_m", SEVERITY_ERROR,
            "RTH altitude exceeds max altitude.",
            fix_action="Lower RTH altitude or raise max altitude.",
        ))
    if drone_profile.rth_altitude_m < drone_profile.min_altitude_m:
        msgs.append(ValidationMessage("rth_altitude_m", SEVERITY_ERROR, "RTH altitude below minimum altitude."))
    if drone_profile.cruise_speed_mps <= 0:
        msgs.append(ValidationMessage("cruise_speed_mps", SEVERITY_ERROR, "Cruise speed must be > 0."))
    if drone_profile.cruise_speed_mps > drone_profile.max_speed_mps:
        msgs.append(ValidationMessage(
            "cruise_speed_mps", SEVERITY_WARNING,
            "Cruise speed exceeds max speed.",
            fix_action="Reduce cruise speed or raise max speed.",
        ))
    if drone_profile.battery_warn_pct < drone_profile.battery_critical_pct:
        msgs.append(ValidationMessage(
            "battery_warn_pct", SEVERITY_ERROR, "Warn threshold must exceed critical threshold."
        ))
    return msgs


# ── Processing settings ───────────────────────────────────────────────────────

def load_processing_settings() -> ProcessingSettings:
    f = processing_settings_file()
    defaults = ProcessingSettings()
    if not f.exists():
        return defaults
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    merged = defaults.to_dict()
    if isinstance(data, dict):
        for k, v in data.items():
            if k in merged:
                merged[k] = v
    return ProcessingSettings(**merged)


def save_processing_settings(settings: ProcessingSettings) -> None:
    f = processing_settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


# ── Model registry path config ────────────────────────────────────────────────

def configure_model_path(model_key: str, path: Path | str) -> dict[str, Any]:
    """Register / update model path in the registry JSON."""
    from .models import load_registry, registry_path, models_root

    if not model_key or not str(model_key).strip():
        raise AppError(ERR_INVALID_INPUT, "Model key is required.")
    p = Path(path)
    if not p.exists():
        raise AppError(
            ERR_INVALID_INPUT,
            f"Model file not found: {p}",
            recovery_action="Choose a valid model file.",
        )
    reg = load_registry()
    models_dict = dict(reg.get("models", {}))
    entry = dict(models_dict.get(model_key, {}))
    try:
        rel = str(p.resolve().relative_to(models_root().resolve()))
    except Exception:
        rel = str(p.resolve())
    entry.setdefault("kind", "onnx_yolo")
    entry.setdefault("labels", [])
    entry["path"] = rel
    models_dict[model_key] = entry
    reg["models"] = models_dict
    registry_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return entry


def get_active_settings() -> dict[str, Any]:
    """Return everything in one bundle (debug/diagnostics)."""
    return {
        "app": load_app_settings().to_dict(),
        "drone_profiles": [p.to_dict() for p in load_drone_profiles()],
        "processing": load_processing_settings().to_dict(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
            "machine": platform.machine(),
        },
    }
