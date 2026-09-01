"""Telling the operator when something they walked away from has finished.

A reconstruction is eight minutes on this machine and considerably longer on a real
survey. An AI pass over four hundred images is not instant either. Nobody watches a
progress bar for that long, so they switch to another workspace, or another application,
and the result lands silently in a panel they are not looking at.

The failure that follows is specific: they come back, see a finished job, and have no idea
whether it finished two seconds ago or forty minutes ago -- so they re-run it, or they act
on stale output. A notification is the record that says which.

Deliberately in-process and local. This build has no server to push from and no account to
push to; a notification here is a row in the project, read by the shell. Pretending
otherwise would mean a "notifications" feature that silently does nothing when the
operator is away from the machine, which is exactly when it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Callable, Sequence
import uuid

#: Beyond this the file is trimmed oldest-first. A notification log is a convenience, not
#: an audit trail -- core/audit is the record that must not lose entries.
MAX_NOTIFICATIONS = 200

LEVELS = ("info", "success", "warning", "error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Notification:
    """One thing worth telling the operator about."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    level: str = "info"
    title: str = ""
    detail: str = ""
    #: What it is about, so the shell can take the operator there: a job id, an export
    #: path, a finding id. Without this a notification is a dead end.
    subject_kind: str = ""
    subject_id: str = ""
    created_utc: str = field(default_factory=_now)
    read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "detail": self.detail,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "created_utc": self.created_utc,
            "read": bool(self.read),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Notification":
        return Notification(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            level=str(raw.get("level", "info")),
            title=str(raw.get("title", "")),
            detail=str(raw.get("detail", "")),
            subject_kind=str(raw.get("subject_kind", "")),
            subject_id=str(raw.get("subject_id", "")),
            created_utc=str(raw.get("created_utc") or _now()),
            read=bool(raw.get("read", False)),
        )


class NotificationCentre:
    """The notifications for one project, on disk beside it.

    Locked, because jobs finish on worker threads while the shell is reading the list.
    Without it a reconstruction finishing mid-render drops the notification it just wrote.
    """

    FILENAME = "notifications.json"

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / self.FILENAME
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Notification], None]] = []

    # -- reading ---------------------------------------------------------------

    def load(self) -> list[Notification]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [Notification.from_dict(entry) for entry in raw.get("notifications", [])]

    def unread(self) -> list[Notification]:
        return [n for n in self.load() if not n.read]

    def unread_count(self) -> int:
        return len(self.unread())

    # -- writing ---------------------------------------------------------------

    def _save(self, items: Sequence[Notification]) -> None:
        # Newest first, so trimming drops the oldest and the shell reads the top of the
        # file for what matters.
        ordered = sorted(items, key=lambda n: n.created_utc, reverse=True)[:MAX_NOTIFICATIONS]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"notifications": [n.to_dict() for n in ordered]}, indent=2),
            encoding="utf-8",
        )

    def notify(self, title: str, detail: str = "", level: str = "info",
               subject_kind: str = "", subject_id: str = "") -> Notification:
        if level not in LEVELS:
            level = "info"
        note = Notification(
            level=level, title=str(title), detail=str(detail),
            subject_kind=str(subject_kind), subject_id=str(subject_id),
        )
        with self._lock:
            items = self.load()
            items.append(note)
            self._save(items)
        for listener in list(self._listeners):
            # A listener that raises must not lose the notification that was already
            # written, nor stop the others from hearing about it.
            try:
                listener(note)
            except Exception:  # noqa: BLE001
                pass
        return note

    def mark_read(self, notification_id: str) -> bool:
        with self._lock:
            items = self.load()
            for note in items:
                if note.id == notification_id:
                    note.read = True
                    self._save(items)
                    return True
            return False

    def mark_all_read(self) -> int:
        with self._lock:
            items = self.load()
            changed = [n for n in items if not n.read]
            for note in changed:
                note.read = True
            if changed:
                self._save(items)
            return len(changed)

    def clear(self) -> int:
        with self._lock:
            count = len(self.load())
            self._save([])
            return count

    # -- live updates ----------------------------------------------------------

    def subscribe(self, listener: Callable[[Notification], None]) -> None:
        """Called on every new notification, for a shell that wants a live badge."""
        self._listeners.append(listener)


def describe_job(job: dict[str, Any]) -> tuple[str, str, str]:
    """Turn a finished job into something worth reading.

    Returns (level, title, detail). The detail carries the failure reason when there is
    one, because "Reconstruction failed" on its own sends the operator to go and find out
    why, which is the thing the notification was supposed to save them.
    """
    name = str(job.get("name") or job.get("kind") or "Job")
    status = str(job.get("status", ""))

    if status == "done":
        return "success", f"{name} finished", str(job.get("message") or "")
    if status == "failed":
        return "error", f"{name} failed", str(job.get("error") or job.get("message") or "")
    if status == "cancelled":
        return "warning", f"{name} cancelled", str(job.get("message") or "")
    return "info", f"{name} {status}".strip(), str(job.get("message") or "")
