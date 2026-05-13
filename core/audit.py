"""Audit trail helpers — writes JSONL events, lists and exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project import AuditEvent, get_manager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit_event(
    project_id: str,
    event_type: str,
    summary: str,
    details: dict[str, Any] | None = None,
    actor: str = "user",
) -> None:
    """Append audit event for project. Silent on failure."""
    mgr = get_manager()
    try:
        meta = mgr.load_project(project_id)
    except Exception:
        return
    root = Path(str(meta.get("root_dir", "")))
    if not root.exists():
        return
    event = AuditEvent(
        event_type=str(event_type),
        detail=str(summary),
        actor=str(actor),
        metadata=dict(details or {}),
    )
    log_path = root / "audit_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict()) + "\n")


def list_audit_events(project_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return most recent audit events (newest first)."""
    return get_manager().get_audit_log(project_id, limit=limit)


def export_audit_log(project_id: str, output_path: Path | str, output_format: str = "jsonl") -> Path:
    """Export audit log to JSONL or CSV. Returns output path."""
    mgr = get_manager()
    meta = mgr.load_project(project_id)
    root = Path(str(meta.get("root_dir", "")))
    src = root / "audit_log.jsonl"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        out.write_text("", encoding="utf-8")
        return out

    if str(output_format).lower() == "csv":
        events: list[dict[str, Any]] = []
        for line in src.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "event_type", "actor", "detail", "metadata"]
            )
            writer.writeheader()
            for ev in events:
                writer.writerow({
                    "timestamp": ev.get("timestamp", ""),
                    "event_type": ev.get("event_type", ""),
                    "actor": ev.get("actor", ""),
                    "detail": ev.get("detail", ""),
                    "metadata": json.dumps(ev.get("metadata", {}), ensure_ascii=False),
                })
        return out

    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out
