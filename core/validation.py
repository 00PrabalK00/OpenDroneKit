"""Common validation helpers — return ValidationMessage or None."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import (
    ValidationMessage,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
)


def validate_path_exists(
    path: Path | str | None,
    expected_type: str = "any",
    field_name: str = "path",
) -> ValidationMessage | None:
    """Check file or folder exists. expected_type: 'file' | 'folder' | 'any'."""
    if path is None or str(path).strip() == "":
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} is required.",
            fix_action="Select a file or folder.",
        )
    p = Path(path)
    if not p.exists():
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} not found: {p}",
            fix_action="Choose a valid path.",
        )
    if expected_type == "file" and not p.is_file():
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} must be a file, got folder.",
            fix_action="Select a file.",
        )
    if expected_type == "folder" and not p.is_dir():
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} must be a folder, got file.",
            fix_action="Select a folder.",
        )
    return None


def validate_numeric_range(
    field_name: str,
    value: float | int | None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ValidationMessage | None:
    if value is None:
        return ValidationMessage(field=field_name, severity=SEVERITY_ERROR, message=f"{field_name} is required.")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ValidationMessage(field=field_name, severity=SEVERITY_ERROR, message=f"{field_name} must be numeric.")
    if min_value is not None and v < float(min_value):
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} must be >= {min_value} (got {v}).",
        )
    if max_value is not None and v > float(max_value):
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} must be <= {max_value} (got {v}).",
        )
    return None


def validate_required_string(field_name: str, value: Any) -> ValidationMessage | None:
    if value is None or str(value).strip() == "":
        return ValidationMessage(field=field_name, severity=SEVERITY_ERROR, message=f"{field_name} is required.")
    return None


def validate_output_writable(path: Path | str, field_name: str = "output_path") -> ValidationMessage | None:
    """Check folder can be created and is writable."""
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"Cannot create {field_name}: {exc}",
            fix_action="Choose a different output folder.",
        )
    if not os.access(str(p), os.W_OK):
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} is not writable: {p}",
            fix_action="Check folder permissions.",
        )
    return None


def validate_units(value: float, unit: str, allowed_units: list[str], field_name: str = "unit") -> ValidationMessage | None:
    if unit not in allowed_units:
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"Unit {unit!r} not allowed. Allowed: {', '.join(allowed_units)}.",
        )
    return None


def validate_choice(
    field_name: str,
    value: Any,
    allowed: list[Any],
) -> ValidationMessage | None:
    if value not in allowed:
        return ValidationMessage(
            field=field_name,
            severity=SEVERITY_ERROR,
            message=f"{field_name} must be one of {allowed} (got {value!r}).",
        )
    return None


def collect(*messages: ValidationMessage | None) -> list[ValidationMessage]:
    """Drop Nones, keep the order."""
    return [m for m in messages if m is not None]


def has_blocking(messages: list[ValidationMessage]) -> bool:
    return any(m.blocking for m in messages)
