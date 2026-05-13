"""Thread-safe app-wide event bus. Qt-optional — works headless."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable


# ── Event type constants ──────────────────────────────────────────────────────

PROJECT_CREATED = "PROJECT_CREATED"
PROJECT_CHANGED = "PROJECT_CHANGED"
PROJECT_ARCHIVED = "PROJECT_ARCHIVED"
WORKFLOW_ASSIGNED = "WORKFLOW_ASSIGNED"
MISSION_CREATED = "MISSION_CREATED"
MISSION_UPDATED = "MISSION_UPDATED"
MISSION_VALIDATED = "MISSION_VALIDATED"
MISSION_EXPORTED = "MISSION_EXPORTED"
PREFLIGHT_UPDATED = "PREFLIGHT_UPDATED"
DRONE_CONNECTED = "DRONE_CONNECTED"
DRONE_DISCONNECTED = "DRONE_DISCONNECTED"
TELEMETRY_UPDATED = "TELEMETRY_UPDATED"
FLIGHT_STARTED = "FLIGHT_STARTED"
FLIGHT_PAUSED = "FLIGHT_PAUSED"
FLIGHT_RESUMED = "FLIGHT_RESUMED"
FLIGHT_RTH = "FLIGHT_RTH"
FLIGHT_ABORTED = "FLIGHT_ABORTED"
FLIGHT_COMMAND = "FLIGHT_COMMAND"
DATASET_IMPORTED = "DATASET_IMPORTED"
DATASET_VALIDATED = "DATASET_VALIDATED"
DATASET_TAGGED = "DATASET_TAGGED"
PROCESSING_STARTED = "PROCESSING_STARTED"
PROCESSING_PROGRESS = "PROCESSING_PROGRESS"
PROCESSING_STAGE_COMPLETED = "PROCESSING_STAGE_COMPLETED"
PROCESSING_COMPLETED = "PROCESSING_COMPLETED"
PROCESSING_FAILED = "PROCESSING_FAILED"
PROCESSING_CANCELLED = "PROCESSING_CANCELLED"
DEFECTS_DETECTED = "DEFECTS_DETECTED"
CRACK_FORECAST_COMPLETED = "CRACK_FORECAST_COMPLETED"
RECONSTRUCTION_COMPLETED = "RECONSTRUCTION_COMPLETED"
MEASUREMENT_ADDED = "MEASUREMENT_ADDED"
ANNOTATION_ADDED = "ANNOTATION_ADDED"
REPORT_GENERATED = "REPORT_GENERATED"
SETTINGS_CHANGED = "SETTINGS_CHANGED"
DIAGNOSTICS_UPDATED = "DIAGNOSTICS_UPDATED"
APP_ERROR = "APP_ERROR"


# ── Event bus ─────────────────────────────────────────────────────────────────

class EventBus:
    """Simple publish/subscribe. Callbacks are called synchronously."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback not in self._subs[event_type]:
                self._subs[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback in self._subs.get(event_type, []):
                self._subs[event_type].remove(callback)

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        payload = dict(payload or {})
        with self._lock:
            callbacks = list(self._subs.get(event_type, []))
            wildcard = list(self._subs.get("*", []))
        for cb in callbacks:
            try:
                cb(payload)
            except Exception:
                pass
        for cb in wildcard:
            try:
                cb({"event_type": event_type, **payload})
            except Exception:
                pass

    def clear(self, event_type: str | None = None) -> None:
        with self._lock:
            if event_type is None:
                self._subs.clear()
            else:
                self._subs.pop(event_type, None)


# ── Module-level singleton ────────────────────────────────────────────────────

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def publish_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
    get_event_bus().publish(event_type, payload or {})


def subscribe_event(event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
    get_event_bus().subscribe(event_type, callback)


def unsubscribe_event(event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
    get_event_bus().unsubscribe(event_type, callback)
