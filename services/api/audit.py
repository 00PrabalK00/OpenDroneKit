"""Audit trail.

Every state-changing request records who did what. The write shares the caller's
transaction deliberately: an action and its audit entry commit together, so the log
cannot end up describing something that was rolled back.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AuditEntry


def record(
    db: Session,
    *,
    action: str,
    user_id: int | None = None,
    organization_id: int | None = None,
    resource: str = "",
    detail: dict[str, Any] | str | None = None,
) -> AuditEntry:
    """Add an audit entry to the current transaction. The caller commits."""
    if isinstance(detail, dict):
        try:
            detail_text = json.dumps(detail, default=str)
        except (TypeError, ValueError):
            detail_text = str(detail)
    else:
        detail_text = str(detail or "")

    entry = AuditEntry(
        organization_id=organization_id,
        user_id=user_id,
        action=action,
        resource=resource,
        detail=detail_text,
    )
    db.add(entry)
    return entry


def recent(
    db: Session, *, organization_id: int | None = None, limit: int = 200
) -> list[AuditEntry]:
    statement = select(AuditEntry).order_by(AuditEntry.created_at.desc()).limit(limit)
    if organization_id is not None:
        statement = statement.where(AuditEntry.organization_id == organization_id)
    return list(db.scalars(statement))
