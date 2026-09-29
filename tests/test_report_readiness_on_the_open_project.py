"""A report of a project that holds nothing is refused, and says why.

Api.report_readiness() answered {ok: true, missing: []} for a fresh project with no
images, findings or missions: the engine judged its own empty mirror of the project and
its own "ok" was spread over the call's. The cockpit printed "Nothing missing" and
Generate wrote a report of empty sections -- indistinguishable from a survey that found
nothing. The engine's warnings were dropped before they reached the UI.

These drive the real Api in a fresh process with its own data root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, data_dir: Path) -> dict:
    env = dict(os.environ)
    env["ODK_DATA_DIR"] = str(data_dir)
    env.pop("ODK_DATABASE_URL", None)
    env.pop("ODK_STORAGE_PATH", None)
    script = "import sys, json\nsys.path.insert(0, %r)\n" % str(REPO_ROOT) + textwrap.dedent(code)
    done = subprocess.run([sys.executable, "-B", "-c", script], cwd=str(data_dir.parent),
                          env=env, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_an_empty_project_is_not_ready_and_generate_refuses(tmp_path: Path) -> None:
    out = _run(
        """
        from pathlib import Path
        from app.api import Api
        api = Api()
        api.create_project("Empty culvert", "", "")
        readiness = api.report_readiness()
        generated = api.generate_report("", "Culvert report", "standard", "")
        root = Path(api._session.project_root())
        written = [str(p) for p in root.rglob("report.html")]
        print(json.dumps({"readiness": readiness, "generated": generated, "written": written}))
        """,
        tmp_path / "data",
    )
    readiness = out["readiness"]
    assert readiness["ok"] is True, readiness          # the check itself ran
    assert readiness["ready"] is False, readiness
    assert any(item.startswith("content:") for item in readiness["missing"]), readiness
    assert isinstance(readiness["warnings"], list)
    generated = out["generated"]
    assert generated["ok"] is False
    assert "nothing to be about" in generated["error"]
    assert out["written"] == []


def test_a_project_with_a_finding_is_ready_and_its_warnings_are_shown(tmp_path: Path) -> None:
    out = _run(
        """
        from pathlib import Path
        from app.api import Api
        from core.annotations import create_annotation
        api = Api()
        project = api.create_project("Culvert with a crack", "", "")["project"]
        root = Path(api._session.project_root())
        create_annotation(root, str(project["id"]), "image", "IMG_0001.JPG", "point",
                          {"type": "Point", "coordinates": [10, 20]}, "crack", "high", "open")
        readiness = api.report_readiness()
        exported = api.export_report("pdf", "Culvert report")
        print(json.dumps({"readiness": readiness, "exported": exported}))
        """,
        tmp_path / "data",
    )
    readiness = out["readiness"]
    assert readiness["ok"] is True and readiness["ready"] is True, readiness
    assert not any(item.startswith("content:") for item in readiness["missing"])
    # The engine's warnings reach the caller instead of being dropped, including that
    # its own report will not carry this project's findings (export_report does).
    assert any("No dataset imported" in w for w in readiness["warnings"]), readiness
    assert any("does not read this project's recorded findings" in w
               for w in readiness["warnings"]), readiness
    exported = out["exported"]
    assert exported["ok"] is True, exported
    assert Path(exported["path"]).is_file()


def test_the_cockpit_reads_ready_not_ok() -> None:
    actions = (REPO_ROOT / "app" / "web" / "js" / "workspace" / "actions.js").read_text(encoding="utf-8")
    assert 'call("report_readiness");\n      if (!readiness.ready)' in actions
    panel = (REPO_ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js").read_text(encoding="utf-8")
    assert "provenance together" not in panel
    assert "r.warnings" in panel
