"""Structured application errors with user/technical separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"


class AppError(Exception):
    """Standard structured error raised by core modules.

    `user_message` is operator-facing; `technical_message` is developer-facing.
    `recovery_action` is a short suggestion shown next to the error.
    """

    def __init__(
        self,
        code: str,
        user_message: str,
        technical_message: str = "",
        severity: str = SEVERITY_ERROR,
        recovery_action: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(technical_message or user_message)
        self.code = str(code)
        self.user_message = str(user_message)
        self.technical_message = str(technical_message or user_message)
        self.severity = str(severity)
        self.recovery_action = recovery_action
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "user_message": self.user_message,
            "technical_message": self.technical_message,
            "severity": self.severity,
            "recovery_action": self.recovery_action,
            "context": self.context,
        }


@dataclass
class ValidationMessage:
    field: str | None
    severity: str
    message: str
    fix_action: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity in (SEVERITY_ERROR, SEVERITY_CRITICAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "severity": self.severity,
            "message": self.message,
            "fix_action": self.fix_action,
            "context": self.context,
        }


# Common error codes (string constants)
ERR_PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
ERR_DATASET_MISSING = "DATASET_MISSING"
ERR_MODEL_MISSING = "MODEL_MISSING"
ERR_DRONE_NOT_CONNECTED = "DRONE_NOT_CONNECTED"
ERR_PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
ERR_MISSION_INVALID = "MISSION_INVALID"
ERR_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERR_OUTPUT_NOT_WRITABLE = "OUTPUT_NOT_WRITABLE"
ERR_REPORT_NOT_READY = "REPORT_NOT_READY"
ERR_PIPELINE_INPUTS = "PIPELINE_INPUTS_MISSING"
ERR_INVALID_INPUT = "INVALID_INPUT"
