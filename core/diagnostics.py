"""System diagnostics — environment, models, folders, optional tools."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import load_registry, models_root, model_status, registry_path
from .settings import (
    drone_profiles_file,
    processing_settings_file,
    settings_file,
    user_config_dir,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""
    fix_action: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticReport:
    timestamp: str
    overall_ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_ok": self.overall_ok,
            "checks": [c.to_dict() for c in self.checks],
            "environment": self.environment,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_OPTIONAL_PACKAGES = [
    ("numpy", True),
    ("cv2", True),
    ("PIL", True),
    ("torch", False),
    ("onnxruntime", False),
    ("open3d", False),
    ("trimesh", False),
    ("pyproj", False),
    ("shapely", False),
    ("rasterio", False),
    ("mavsdk", False),
    ("pymavlink", False),
    ("imagehash", False),
    ("jinja2", False),
    ("weasyprint", False),
    ("PySide6", False),
]


def _check_package(name: str, required: bool) -> CheckResult:
    try:
        importlib.import_module(name)
        return CheckResult(name=f"pkg:{name}", ok=True, message="installed")
    except Exception as exc:
        return CheckResult(
            name=f"pkg:{name}",
            ok=not required,
            message=f"missing ({exc.__class__.__name__})" if required else "optional — not installed",
            fix_action=f"pip install {name}" if required else None,
        )


def _check_folder(path: Path, label: str) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Write probe
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return CheckResult(name=f"folder:{label}", ok=True, message=str(path))
    except Exception as exc:
        return CheckResult(
            name=f"folder:{label}",
            ok=False,
            message=f"{path} not writable: {exc}",
            fix_action="Check folder permissions.",
        )


def _check_models() -> CheckResult:
    try:
        reg = load_registry()
        models = reg.get("models", {})
        if not models:
            return CheckResult(name="model_registry", ok=False, message="No models registered.")
        missing = []
        for key in models:
            status = model_status(key)
            if not status.get("exists"):
                missing.append(key)
        ok = len(missing) == 0
        msg = f"{len(models) - len(missing)}/{len(models)} models present"
        return CheckResult(
            name="model_registry",
            ok=ok,
            message=msg,
            fix_action=("Configure model paths in Developer Tools." if not ok else None),
            details={"missing": missing, "total": len(models)},
        )
    except Exception as exc:
        return CheckResult(name="model_registry", ok=False, message=f"Registry error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def run_system_diagnostics() -> DiagnosticReport:
    """Check Python environment, model files, processing scripts, writable folders, optional tools."""
    checks: list[CheckResult] = []

    # Python version
    if sys.version_info < (3, 10):
        checks.append(CheckResult("python_version", False, f"Python {sys.version.split()[0]} too old; need >= 3.10."))
    else:
        checks.append(CheckResult("python_version", True, f"Python {sys.version.split()[0]}"))

    # Packages
    for name, required in _OPTIONAL_PACKAGES:
        checks.append(_check_package(name, required))

    # Folders
    checks.append(_check_folder(user_config_dir(), "user_config"))
    checks.append(_check_folder(models_root(), "models_root"))
    tmp_root = Path(tempfile.gettempdir()) / "opendronekit_tmp"
    checks.append(_check_folder(tmp_root, "tmp"))

    # Model registry
    checks.append(_check_models())

    # Settings files presence
    for label, path in (
        ("settings_file", settings_file()),
        ("drone_profiles", drone_profiles_file()),
        ("processing_settings", processing_settings_file()),
        ("model_registry_file", registry_path()),
    ):
        exists = Path(path).exists()
        checks.append(CheckResult(
            name=label, ok=True, message=("present" if exists else "default (not yet saved)"),
            details={"path": str(path), "exists": exists},
        ))

    env = {
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version,
        "machine": platform.machine(),
        "executable": sys.executable,
    }

    overall_ok = all(c.ok for c in checks)
    return DiagnosticReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_ok=overall_ok,
        checks=checks,
        environment=env,
    )


def export_diagnostic_bundle(output_path: Path | str) -> Path:
    """Bundle diagnostics report + settings into a single JSON file for support."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_system_diagnostics().to_dict()
    bundle = {
        "diagnostic_report": report,
        "settings_paths": {
            "settings": str(settings_file()),
            "drone_profiles": str(drone_profiles_file()),
            "processing": str(processing_settings_file()),
            "model_registry": str(registry_path()),
        },
        "config_files": {},
    }
    for label, path in bundle["settings_paths"].items():
        p = Path(path)
        if p.exists():
            try:
                bundle["config_files"][label] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                bundle["config_files"][label] = {"_error": "could not parse"}
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return out
