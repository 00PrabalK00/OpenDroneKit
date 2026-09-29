"""Application state lives in one absolute place, whatever folder the app starts in.

The desktop project store and the fleet/share/webhook database were both resolved
relative to the working directory, and the report engine wrote beside the repository.
Starting the app from a shortcut or another terminal opened a fresh empty store there,
so the user's projects appeared to have been deleted.

These run the real Api in separate processes started from different folders, which is
the only honest way to test "the working directory no longer matters".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, cwd: Path, data_dir: Path) -> dict:
    env = dict(os.environ)
    env["ODK_DATA_DIR"] = str(data_dir)
    env.pop("ODK_DATABASE_URL", None)
    env.pop("ODK_STORAGE_PATH", None)
    script = "import sys, json\nsys.path.insert(0, %r)\n" % str(REPO_ROOT) + textwrap.dedent(code)
    done = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=str(cwd), env=env,
        capture_output=True, text=True, timeout=240,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_projects_survive_starting_from_another_folder(tmp_path: Path) -> None:
    data = tmp_path / "data"
    first, second = tmp_path / "shortcut_cwd", tmp_path / "terminal_cwd"
    first.mkdir()
    second.mkdir()

    created = _run(
        """
        from app.api import Api
        api = Api()
        out = api.create_project("Bridge survey", "")
        print(json.dumps(out))
        """,
        first, data,
    )
    assert created.get("ok") is not False, created

    listed = _run(
        """
        from app.api import Api
        print(json.dumps(Api().list_projects()))
        """,
        second, data,
    )
    names = [p["name"] for p in listed["projects"]]
    assert "Bridge survey" in names

    root = next(p["root_dir"] for p in listed["projects"] if p["name"] == "Bridge survey")
    assert Path(root).is_absolute()
    assert Path(root).resolve().is_relative_to(data.resolve())

    # Nothing was written relative to either working directory.
    for cwd in (first, second):
        assert list(cwd.iterdir()) == [], f"state leaked into {cwd}: {list(cwd.iterdir())}"
    assert (data / "app_state" / "projects.db").is_file()


def test_fleet_database_and_report_workspace_share_the_root(tmp_path: Path) -> None:
    pytest.importorskip("sqlalchemy")
    data = tmp_path / "data"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()

    out = _run(
        """
        from app.api import Api
        from core.project import get_manager
        from services.api.db import database_url
        from services.api.paths import storage_root
        fleet = Api().list_fleet()
        print(json.dumps({
            "fleet_ok": fleet.get("ok", True) is not False,
            "fleet": fleet,
            "db": database_url(),
            "workspace": str(get_manager()._workspace),
            "storage": str(storage_root()),
        }))
        """,
        cwd, data,
    )
    assert out["fleet_ok"], out["fleet"]
    assert out["db"].startswith("sqlite:///")
    db_file = Path(out["db"][len("sqlite:///"):])
    assert db_file.is_absolute() and db_file.resolve().is_relative_to(data.resolve())
    assert db_file.is_file()
    assert Path(out["workspace"]).resolve().is_relative_to(data.resolve())
    assert Path(out["storage"]).resolve().is_relative_to(data.resolve())
    assert list(cwd.iterdir()) == []


def test_default_root_is_the_toolkit_folder_not_the_cwd(monkeypatch, tmp_path) -> None:
    """Unconfigured, state goes where the launcher has always put it -- next to the
    toolkit -- so existing projects are still found after this change."""
    from core import paths

    monkeypatch.delenv("ODK_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.data_root() == (REPO_ROOT / "final_toolkit_outputs").resolve()


def test_legacy_relative_project_folder_resolves_against_the_toolkit(tmp_path) -> None:
    """Rows written by earlier versions hold a cwd-relative folder."""
    from app.store import ProjectStore

    store = ProjectStore(tmp_path / "projects.db")
    try:
        with store._lock:  # noqa: SLF001 - writing a legacy row the old code produced
            store._conn.execute(
                "INSERT INTO projects(name, root_dir, created_at, updated_at) VALUES (?,?,?,?)",
                ("Legacy", "final_toolkit_outputs\\projects\\Legacy", "2026-01-01", "2026-01-01"),
            )
            store._conn.commit()
        legacy = next(p for p in store.list_projects() if p["name"] == "Legacy")
        assert Path(legacy["root_dir"]) == (REPO_ROOT / "final_toolkit_outputs" / "projects" / "Legacy").resolve()
    finally:
        store.close()
