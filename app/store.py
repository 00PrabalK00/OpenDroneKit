"""Offline-first project storage and mission version history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class StorePaths:
    db_path: Path

    @staticmethod
    def default() -> "StorePaths":
        root = Path("final_toolkit_outputs") / "app_state"
        root.mkdir(parents=True, exist_ok=True)
        return StorePaths(db_path=root / "projects.db")


class ProjectStore:
    """Local SQLite store for projects, versions, and audit events."""

    def __init__(self, db_path: str | Path | None = None):
        self.paths = StorePaths.default() if db_path is None else StorePaths(Path(db_path))
        self.paths.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The desktop shell dispatches API calls on webview worker threads and runs long
        # jobs on their own threads, so a thread-bound connection would fail. Python's
        # sqlite3 serializes individual statements itself; `_lock` additionally guards
        # the read-then-write sequences that would otherwise race.
        self._conn = sqlite3.connect(str(self.paths.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                root_dir TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                offline_enabled INTEGER NOT NULL DEFAULT 1,
                sync_status TEXT NOT NULL DEFAULT 'offline',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mission_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                mission_name TEXT NOT NULL,
                template TEXT NOT NULL,
                version_num INTEGER NOT NULL,
                parent_version_id INTEGER,
                flight_recipe_json TEXT NOT NULL,
                plan_summary_json TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mission_unique_version
            ON mission_versions(project_id, mission_name, version_num);

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                captured_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                report_type TEXT NOT NULL,
                content_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def set_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self._conn.execute(
            """
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(key), payload),
        )
        self._conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (str(key),)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(str(row["value"]))
        except Exception:
            return default

    def create_project(self, name: str, root_dir: str | Path | None = None, description: str = "") -> dict[str, Any]:
        now = _utc_now()
        name_clean = str(name).strip()
        if not name_clean:
            raise ValueError("Project name is required.")
        if root_dir is None:
            safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name_clean).strip("_")
            root = Path("final_toolkit_outputs") / "projects" / (safe_name or "project")
        else:
            root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO projects(name, root_dir, description, offline_enabled, sync_status, created_at, updated_at)
            VALUES (?, ?, ?, 1, 'offline', ?, ?)
            """,
            (name_clean, str(root), str(description or ""), now, now),
        )
        self._conn.commit()
        project_id = int(cur.lastrowid)
        self.append_audit_event(project_id, "project_created", {"name": name_clean, "root_dir": str(root)})
        self.set_active_project(project_id)
        out = self.get_project(project_id)
        if out is None:
            raise RuntimeError("Failed to create project.")
        return out

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT p.*,
                   (SELECT COUNT(1) FROM mission_versions mv WHERE mv.project_id = p.id) AS mission_versions_count,
                   (SELECT COUNT(1) FROM datasets d WHERE d.project_id = p.id) AS datasets_count,
                   (SELECT COUNT(1) FROM reports r WHERE r.project_id = p.id) AS reports_count
            FROM projects p
            ORDER BY p.updated_at DESC, p.id DESC
            """
        ).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT p.*,
                   (SELECT COUNT(1) FROM mission_versions mv WHERE mv.project_id = p.id) AS mission_versions_count,
                   (SELECT COUNT(1) FROM datasets d WHERE d.project_id = p.id) AS datasets_count,
                   (SELECT COUNT(1) FROM reports r WHERE r.project_id = p.id) AS reports_count
            FROM projects p
            WHERE p.id = ?
            """,
            (int(project_id),),
        ).fetchone()
        return self._row_to_dict(row)

    def set_active_project(self, project_id: int) -> None:
        self.set_setting("active_project_id", int(project_id))

    def get_active_project_id(self) -> int | None:
        value = self.get_setting("active_project_id", None)
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def get_active_project(self) -> dict[str, Any] | None:
        pid = self.get_active_project_id()
        if pid is None:
            return None
        return self.get_project(pid)

    def mark_sync_status(self, project_id: int, status: str) -> None:
        now = _utc_now()
        self._conn.execute(
            "UPDATE projects SET sync_status = ?, updated_at = ? WHERE id = ?",
            (str(status), now, int(project_id)),
        )
        self._conn.commit()
        self.append_audit_event(project_id, "sync_status_changed", {"status": str(status)})

    def append_audit_event(self, project_id: int, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            """
            INSERT INTO audit_events(project_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(project_id), str(event_type), json.dumps(payload or {}, ensure_ascii=True), _utc_now()),
        )
        self._conn.commit()

    def list_audit_events(self, project_id: int, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM audit_events
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(project_id), int(max(1, limit))),
        ).fetchall()
        return [self._row_to_dict(r) or {} for r in rows]

    def _next_mission_version_num(self, project_id: int, mission_name: str) -> int:
        row = self._conn.execute(
            """
            SELECT MAX(version_num) AS vmax
            FROM mission_versions
            WHERE project_id = ? AND mission_name = ?
            """,
            (int(project_id), str(mission_name)),
        ).fetchone()
        vmax = 0 if row is None or row["vmax"] is None else int(row["vmax"])
        return vmax + 1

    def save_mission_version(
        self,
        project_id: int,
        mission_name: str,
        template: str,
        flight_recipe: dict[str, Any],
        plan_summary: dict[str, Any],
        note: str = "",
        parent_version_id: int | None = None,
    ) -> dict[str, Any]:
        project_id_i = int(project_id)
        mission_name_s = str(mission_name).strip() or "mission"
        template_s = str(template).strip() or "grid"
        # Reading the current maximum and inserting the next one must be atomic, or two
        # concurrent saves both claim the same version number.
        with self._lock:
            version_num = self._next_mission_version_num(project_id_i, mission_name_s)
            now = _utc_now()
            cur = self._conn.cursor()
            cur.execute(
                """
            INSERT INTO mission_versions(
                project_id, mission_name, template, version_num, parent_version_id,
                flight_recipe_json, plan_summary_json, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project_id_i,
                    mission_name_s,
                    template_s,
                    version_num,
                    int(parent_version_id) if parent_version_id is not None else None,
                    json.dumps(flight_recipe or {}, ensure_ascii=True),
                    json.dumps(plan_summary or {}, ensure_ascii=True),
                    str(note or ""),
                    now,
                ),
            )
            mission_version_id = int(cur.lastrowid)
            self._conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id_i),
            )
            self._conn.commit()
        self.append_audit_event(
            project_id_i,
            "mission_version_saved",
            {
                "mission_version_id": mission_version_id,
                "mission_name": mission_name_s,
                "template": template_s,
                "version_num": version_num,
                "note": str(note or ""),
            },
        )
        row = self._conn.execute("SELECT * FROM mission_versions WHERE id = ?", (mission_version_id,)).fetchone()
        out = self._row_to_dict(row)
        if out is None:
            raise RuntimeError("Failed to save mission version.")
        return out

    def list_mission_versions(
        self,
        project_id: int,
        mission_name: str | None = None,
        limit: int = 400,
    ) -> list[dict[str, Any]]:
        if mission_name and str(mission_name).strip():
            rows = self._conn.execute(
                """
                SELECT * FROM mission_versions
                WHERE project_id = ? AND mission_name = ?
                ORDER BY version_num DESC, id DESC
                LIMIT ?
                """,
                (int(project_id), str(mission_name), int(max(1, limit))),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM mission_versions
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(project_id), int(max(1, limit))),
            ).fetchall()
        return [self._row_to_dict(r) or {} for r in rows]

    def save_dataset_entry(
        self,
        project_id: int,
        name: str,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO datasets(project_id, name, path, metadata_json, captured_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(project_id),
                str(name).strip() or Path(path).name,
                str(path),
                json.dumps(metadata or {}, ensure_ascii=True),
                str(captured_at or now),
                now,
            ),
        )
        self._conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (now, int(project_id)),
        )
        self._conn.commit()
        dataset_id = int(cur.lastrowid)
        self.append_audit_event(
            int(project_id),
            "dataset_imported",
            {"dataset_id": dataset_id, "name": str(name), "path": str(path)},
        )
        row = self._conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        out = self._row_to_dict(row)
        if out is None:
            raise RuntimeError("Failed to save dataset.")
        return out

    def list_datasets(self, project_id: int, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM datasets
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(project_id), int(max(1, limit))),
        ).fetchall()
        return [self._row_to_dict(r) or {} for r in rows]

    def save_report(
        self,
        project_id: int,
        title: str,
        report_type: str,
        content_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO reports(project_id, title, report_type, content_path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(project_id),
                str(title).strip() or "Inspection Report",
                str(report_type).strip() or "standard",
                str(content_path),
                json.dumps(metadata or {}, ensure_ascii=True),
                now,
            ),
        )
        self._conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (now, int(project_id)),
        )
        self._conn.commit()
        report_id = int(cur.lastrowid)
        self.append_audit_event(
            int(project_id),
            "report_generated",
            {
                "report_id": report_id,
                "title": str(title),
                "report_type": str(report_type),
                "content_path": str(content_path),
            },
        )
        row = self._conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        out = self._row_to_dict(row)
        if out is None:
            raise RuntimeError("Failed to save report.")
        return out

    def list_reports(self, project_id: int, limit: int = 300) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM reports
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(project_id), int(max(1, limit))),
        ).fetchall()
        return [self._row_to_dict(r) or {} for r in rows]
