"""tools/e2e_workflow.py no longer writes its test rows into the user's own data.

Every run added an "E2E aircraft", a battery and a maintenance record to the real fleet
database -- the main checkout's Fleet screen listed five identical "E2E aircraft". The
tool now runs in a throwaway folder unless --into-my-data is passed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(monkeypatch, argv):
    seen: dict[str, str] = {}

    class RecordingApi:
        def __init__(self, _session):
            for key in ("ODK_DATA_DIR", "ODK_HUB_HOME", "ODK_DATABASE_URL"):
                seen[key] = os.environ.get(key, "")

        def __getattr__(self, name):
            return lambda *a, **k: {"ok": False, "error": "recorded only"}

    monkeypatch.setitem(sys.modules, "app.api", types.SimpleNamespace(Api=RecordingApi))
    monkeypatch.setitem(sys.modules, "app.session",
                        types.SimpleNamespace(AppSession=lambda: None))
    for key in ("ODK_DATA_DIR", "ODK_HUB_HOME", "ODK_DATABASE_URL"):
        monkeypatch.setenv(key, f"REAL-{key}")
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import e2e_workflow
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    monkeypatch.setattr(e2e_workflow, "results", [])
    e2e_workflow.main(argv)
    return seen


def test_a_run_writes_into_a_throwaway_folder(monkeypatch):
    seen = _run(monkeypatch, [])
    import tempfile

    temp = Path(tempfile.gettempdir()).resolve()
    for key in ("ODK_DATA_DIR", "ODK_HUB_HOME"):
        assert Path(seen[key]).resolve().is_relative_to(temp), seen
    assert seen["ODK_DATABASE_URL"].startswith("sqlite:///") and "odk_e2e_" in seen["ODK_DATABASE_URL"]


def test_into_my_data_is_the_only_way_to_touch_real_data(monkeypatch):
    seen = _run(monkeypatch, ["--into-my-data"])
    assert seen == {k: f"REAL-{k}" for k in ("ODK_DATA_DIR", "ODK_HUB_HOME", "ODK_DATABASE_URL")}
