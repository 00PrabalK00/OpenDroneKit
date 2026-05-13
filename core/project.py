"""Project management — SQLite index + JSON metadata + audit log."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _toolkit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _toolkit_root().parent


def _default_workspace() -> Path:
    return _workspace_root() / "opendrone_workspace"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ProjectReadiness:
    has_mission: bool = False
    has_dataset: bool = False
    has_analysis: bool = False
    has_report: bool = False
    folder_ok: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.folder_ok and not self.issues

    @property
    def summary(self) -> str:
        parts = []
        if self.has_mission:
            parts.append("mission")
        if self.has_dataset:
            parts.append("dataset")
        if self.has_analysis:
            parts.append("analysis")
        if self.has_report:
            parts.append("report")
        return ", ".join(parts) if parts else "no data yet"


@dataclass
class ProjectSummary:
    id: str
    name: str
    description: str
    root_dir: str
    created_at: str
    updated_at: str
    sync_status: str
    mission_count: int
    dataset_count: int
    report_count: int
    workflow_id: str
    readiness: ProjectReadiness = field(default_factory=ProjectReadiness)
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["readiness"] = asdict(self.readiness)
        return d


@dataclass
class AuditEvent:
    event_type: str
    detail: str
    actor: str = "user"
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "actor": self.actor,
            "detail": self.detail,
            "metadata": self.metadata,
        }


# ── Project Manager ───────────────────────────────────────────────────────────

class ProjectManager:
    """Manages project lifecycle — create, load, list, validate."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        root_dir TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        sync_status TEXT DEFAULT 'local',
        workflow_id TEXT DEFAULT '',
        archived INTEGER DEFAULT 0
    );
    """

    def __init__(self, workspace: Path | None = None):
        self._workspace = Path(workspace) if workspace else _default_workspace()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._db_path = self._workspace / "project_index.db"
        self._db: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(self._SCHEMA)
        self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        root_dir: Path | None = None,
        description: str = "",
        workflow_id: str = "",
    ) -> dict[str, Any]:
        """Create project folder, metadata file, SQLite entry, initial audit event."""
        name = name.strip()
        if not name:
            raise ValueError("Project name must not be empty.")

        pid = str(uuid.uuid4())
        now = _now_iso()

        if root_dir is None:
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
            root_dir = self._workspace / safe_name.replace(" ", "_")

        root = Path(root_dir).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Project folder already contains files: {root}")
        self._create_folder_structure(root)

        meta = {
            "id": pid,
            "name": name,
            "description": description,
            "root_dir": str(root),
            "created_at": now,
            "updated_at": now,
            "sync_status": "local",
            "workflow_id": workflow_id,
            "archived": False,
        }
        (root / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        assert self._db is not None
        self._db.execute(
            "INSERT INTO projects (id,name,description,root_dir,created_at,updated_at,sync_status,workflow_id,archived) "
            "VALUES (?,?,?,?,?,?,?,?,0)",
            (pid, name, description, str(root), now, now, "local", workflow_id),
        )
        self._db.commit()

        self._write_audit(root, AuditEvent("project_created", f"Project '{name}' created.", metadata={"id": pid}))
        return meta

    def load_project(self, project_id: str) -> dict[str, Any]:
        """Load project from SQLite + project.json; raise if missing."""
        assert self._db is not None
        row = self._db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        meta = dict(row)
        root = Path(meta["root_dir"])
        pjson = root / "project.json"
        if pjson.exists():
            try:
                disk = json.loads(pjson.read_text(encoding="utf-8"))
                meta.update(disk)
            except Exception:
                pass
        meta["archived"] = bool(meta.get("archived", 0))
        return meta

    def update_project(self, project_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update editable fields; preserve created_at; write audit."""
        meta = self.load_project(project_id)
        readonly = {"id", "created_at", "root_dir"}
        for k, v in patch.items():
            if k not in readonly:
                meta[k] = v
        meta["updated_at"] = _now_iso()

        assert self._db is not None
        self._db.execute(
            "UPDATE projects SET name=?,description=?,updated_at=?,sync_status=?,workflow_id=? WHERE id=?",
            (meta["name"], meta.get("description", ""), meta["updated_at"],
             meta.get("sync_status", "local"), meta.get("workflow_id", ""), project_id),
        )
        self._db.commit()

        root = Path(meta["root_dir"])
        (root / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._write_audit(root, AuditEvent("project_updated", f"Project updated: {list(patch.keys())}"))
        return meta

    def archive_project(self, project_id: str) -> None:
        """Mark archived; do not delete files."""
        assert self._db is not None
        self._db.execute("UPDATE projects SET archived=1 WHERE id=?", (project_id,))
        self._db.commit()
        try:
            meta = self.load_project(project_id)
            root = Path(meta["root_dir"])
            self._write_audit(root, AuditEvent("project_archived", "Project archived."))
        except Exception:
            pass

    def list_projects(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Return all projects with readiness summaries."""
        assert self._db is not None
        if include_archived:
            rows = self._db.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM projects WHERE archived=0 ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            meta = dict(row)
            meta["archived"] = bool(meta.get("archived", 0))
            root = Path(meta["root_dir"])
            pjson = root / "project.json"
            if pjson.exists():
                try:
                    disk = json.loads(pjson.read_text(encoding="utf-8"))
                    meta.update(disk)
                except Exception:
                    pass
            readiness = self._check_readiness(root)
            meta["readiness"] = asdict(readiness)
            meta["mission_count"] = self._count_files(root / "missions", ".json")
            meta["dataset_count"] = self._count_dirs(root / "datasets")
            meta["report_count"] = self._count_files(root / "reports", ".html") + self._count_files(root / "reports", ".pdf")
            result.append(meta)
        return result

    def validate_project(self, project_id: str) -> ProjectReadiness:
        meta = self.load_project(project_id)
        root = Path(meta["root_dir"])
        return self._check_readiness(root)

    def get_audit_log(self, project_id: str, limit: int = 100) -> list[dict[str, Any]]:
        try:
            meta = self.load_project(project_id)
            root = Path(meta["root_dir"])
            log_path = root / "audit_log.jsonl"
            if not log_path.exists():
                return []
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            events = []
            for line in reversed(lines[-limit * 2:]):
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
            return events[:limit]
        except Exception:
            return []

    def write_project_audit(self, project_id: str, event_type: str, detail: str, metadata: dict | None = None) -> None:
        try:
            meta = self.load_project(project_id)
            root = Path(meta["root_dir"])
            self._write_audit(root, AuditEvent(event_type, detail, metadata=metadata or {}))
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _write_audit(root: Path, event: AuditEvent) -> None:
        log_path = root / "audit_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    @staticmethod
    def _create_folder_structure(root: Path) -> None:
        for sub in (
            "missions/mission_versions",
            "datasets",
            "processing",
            "analysis/defects",
            "analysis/crack_growth",
            "analysis/reconstruction",
            "analysis/measurements",
            "analysis/annotations",
            "reports",
            "cache/map_tiles",
            "cache/temp",
        ):
            (root / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_readiness(root: Path) -> ProjectReadiness:
        r = ProjectReadiness()
        r.folder_ok = root.exists() and root.is_dir()
        if not r.folder_ok:
            r.issues.append(f"Project folder missing: {root}")
            return r
        r.has_mission = any((root / "missions").glob("*.json")) if (root / "missions").exists() else False
        r.has_dataset = any((root / "datasets").iterdir()) if (root / "datasets").exists() else False
        r.has_analysis = (
            any((root / "analysis").rglob("*.json")) if (root / "analysis").exists() else False
        )
        r.has_report = (
            any((root / "reports").rglob("*.html")) or any((root / "reports").rglob("*.pdf"))
        ) if (root / "reports").exists() else False
        return r

    @staticmethod
    def _count_files(folder: Path, ext: str) -> int:
        if not folder.exists():
            return 0
        return sum(1 for f in folder.rglob(f"*{ext}") if f.is_file())

    @staticmethod
    def _count_dirs(folder: Path) -> int:
        if not folder.exists():
            return 0
        return sum(1 for p in folder.iterdir() if p.is_dir())


# ── Module-level singleton ─────────────────────────────────────────────────────

_manager: ProjectManager | None = None


def get_manager(workspace: Path | None = None) -> ProjectManager:
    global _manager
    if _manager is None:
        _manager = ProjectManager(workspace)
    return _manager
