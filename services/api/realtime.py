"""Database-backed realtime broker shared by all API workers.

Every worker already shares the deployment database. Persisting a bounded event stream
there gives WebSocket workers a genuine cross-process rendezvous without introducing a
second optional service that silently degrades to process-local delivery. The broker is
deliberately modest: it provides ordered, bounded fan-out, not acknowledgements,
exactly-once delivery, or offline replay for browsers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any

from sqlalchemy import delete, select

from .db import get_session_factory
from .models import LiveEvent


DEFAULT_RETENTION_PER_ORGANIZATION = 10_000


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class BrokerEvent:
    id: int
    organization_id: int
    event: str
    data: dict[str, Any]
    created_at: datetime

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event": self.event,
            "data": self.data,
            "at": _utc(self.created_at).isoformat(),
        }


class DatabaseEventBroker:
    """An ordered broker whose rendezvous is the deployment's shared database."""

    def __init__(self, retention_per_organization: int | None = None) -> None:
        configured = int(os.environ.get(
            "ODK_LIVE_EVENT_RETENTION", DEFAULT_RETENTION_PER_ORGANIZATION
        ))
        self.retention = int(retention_per_organization or configured)
        if self.retention < 100:
            raise ValueError("ODK_LIVE_EVENT_RETENTION must be at least 100 events.")

    def latest_id(self, organization_id: int) -> int:
        session = get_session_factory()()
        try:
            row = session.scalar(
                select(LiveEvent.id)
                .where(LiveEvent.organization_id == organization_id)
                .order_by(LiveEvent.id.desc()).limit(1)
            )
            return int(row or 0)
        finally:
            session.close()

    def publish(self, organization_id: int, event: str, payload: dict[str, Any]) -> int:
        """Commit one event so every worker can observe it, returning its broker id."""
        if organization_id < 1:
            raise ValueError("organization_id must be positive.")
        if not event.strip():
            raise ValueError("event is required.")
        encoded = json.dumps(payload, allow_nan=False, default=str)
        session = get_session_factory()()
        try:
            row = LiveEvent(
                organization_id=organization_id, event=event.strip(), payload_json=encoded,
            )
            session.add(row)
            session.flush()
            event_id = int(row.id)

            # Retention is per tenant. This is a bounded live-event buffer, not an
            # ever-growing audit log (state-changing requests have AuditEntry for that).
            first_to_keep = session.scalar(
                select(LiveEvent.id)
                .where(LiveEvent.organization_id == organization_id)
                .order_by(LiveEvent.id.desc()).offset(self.retention - 1).limit(1)
            )
            if first_to_keep is not None:
                session.execute(delete(LiveEvent).where(
                    LiveEvent.organization_id == organization_id,
                    LiveEvent.id < int(first_to_keep),
                ))
            session.commit()
            return event_id
        finally:
            session.close()

    def read_after(
        self, organization_id: int, after_id: int, limit: int = 250,
    ) -> list[BrokerEvent]:
        session = get_session_factory()()
        try:
            rows = session.scalars(
                select(LiveEvent).where(
                    LiveEvent.organization_id == organization_id,
                    LiveEvent.id > after_id,
                ).order_by(LiveEvent.id).limit(max(1, min(int(limit), 1000)))
            )
            events: list[BrokerEvent] = []
            for row in rows:
                try:
                    data = json.loads(row.payload_json)
                except (TypeError, json.JSONDecodeError):
                    data = {"unavailable": True, "reason": "Stored live-event payload is invalid."}
                events.append(BrokerEvent(
                    id=row.id, organization_id=row.organization_id, event=row.event,
                    data=data, created_at=row.created_at,
                ))
            return events
        finally:
            session.close()

    def describe(self) -> dict[str, Any]:
        return {
            "transport": "websocket",
            "broker": "shared_database",
            "scope": "one organization per connection",
            "multi_worker": True,
            "retention_per_organization": self.retention,
            "limitation": (
                "Workers sharing ODK_DATABASE_URL observe the same ordered event buffer. "
                "WebSocket delivery remains best-effort: there are no client acknowledgements "
                "or exactly-once guarantees, and a disconnected browser is not replayed events "
                "from before it reconnects. Refresh the source resource when certainty matters."
            ),
        }


broker = DatabaseEventBroker()
