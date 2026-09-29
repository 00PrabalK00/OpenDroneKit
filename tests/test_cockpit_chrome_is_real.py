"""The cockpit's frame -- top bar, status bar, command palette -- shows the real state.

Left over after the first chrome sweep:

* the status bar ended in a literal "v1.0" while the package is 0.1.0;
* the top bar carried a fixed "local-first" label with an idle dot in every state,
  never saying where this install actually keeps its projects;
* the search box promised "projects, missions, findings", and the palette listed only
  projects -- a saved mission or a recorded finding typed into it matched nothing.

Each is driven here through the shipped workspace.html in a headless browser with the
real Api behind the bridge, on a project that has one saved mission and one finding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api import Api, app_version
from app.session import AppSession
from app.store import ProjectStore
from core.annotations import create_annotation
from desktop_window import run_page

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_JS = REPO_ROOT / "app" / "web" / "js" / "workspace" / "shell.js"

DRIVER = r"""
const q = (s) => document.querySelector(s);
const qa = (s) => Array.from(document.querySelectorAll(s));
const out = {};
await __waitFor(() => /^v\d/.test(q('.statusbar').textContent.split('·').pop().trim())
  || qa('.statusbar .mono').some((n) => /^v\d/.test(n.textContent)));
await __waitFor(() => /under /.test(q('.nav-storage').parentElement.title));
out.status = q('.statusbar').textContent;
out.nav = q('.nav-right').textContent;
out.storageTitle = q('.nav-storage').parentElement.title;
out.version = qa('.statusbar .mono').map((n) => n.textContent).find((t) => /^v/.test(t));

document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }));
await __waitFor(() => qa('.palette-item').length > 0);
out.palette = qa('.palette-item').map((r) => r.textContent);

const input = q('.palette input');
input.value = 'efflorescence';
input.dispatchEvent(new Event('input'));
await __waitFor(() => qa('.palette-item').length === 1);
qa('.palette-item')[0].click();
await __waitFor(() => /efflorescence/.test(
  (q('[data-panel-id="ai.finding"]') || {}).textContent || ''));
out.workspace = q('.nav-item.active').textContent;
out.details = q('[data-panel-id="ai.finding"]').textContent;
__finish(out);
"""


@pytest.fixture
def driven(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("ODK_DATA_DIR", str(data))
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    root = tmp_path / "project"
    project = api.create_project("Pier 7 chrome", str(root))["project"]
    create_annotation(root, str(project["id"]), "image", "IMG_0001.JPG", "point",
                      {"type": "Point", "coordinates": [10, 20]}, "efflorescence", "high", "open")
    assert api.set_aoi([[-81.7504, 41.3038], [-81.7494, 41.3038], [-81.7494, 41.3046],
                        [-81.7504, 41.3046]])["ok"]
    assert api.plan_mission({"mode": "grid", "altitude_m": 60})["ok"]
    assert api.save_mission("Deck grid")["ok"]
    result = run_page(
        api, DRIVER, tmp_path / "profile", page_name="workspace.html",
        ready_js="document.querySelectorAll('.tbtn').length > 0",
    )
    assert "error" not in result, result
    result["data_root"] = str(api.get_state()["data_paths"]["data_root"])
    return result


def test_the_frame_reports_the_real_version_and_storage(driven) -> None:
    assert app_version() == "0.1.0"  # the package's own number, not a literal
    assert driven["version"] == f"v{app_version()}"
    assert "v1.0" not in driven["status"]
    assert driven["data_root"] in driven["storageTitle"]
    assert "local-first" not in driven["nav"]
    assert "Pier 7 chrome" in driven["status"]


def test_the_palette_finds_the_projects_missions_and_findings(driven) -> None:
    rows = driven["palette"]
    assert any(r.startswith("project") and "Pier 7 chrome" in r for r in rows), rows
    assert any(r.startswith("mission") and "Deck grid" in r for r in rows), rows
    assert any(r.startswith("finding") and "efflorescence" in r for r in rows), rows


def test_choosing_a_finding_opens_it_where_it_can_be_reviewed(driven) -> None:
    assert driven["workspace"].strip() == "AI Inspection"
    assert "efflorescence" in driven["details"] and "high" in driven["details"]


def test_no_literal_status_is_left_in_the_frame() -> None:
    source = SHELL_JS.read_text(encoding="utf-8")
    for literal in ('text: "v1.0"', 'text: " local-first"'):
        assert literal not in source, literal


HOME_DRIVER = r"""
const panel = () => document.querySelector('[data-panel-id="home.system"]');
await __waitFor(() => panel() && /Storage/.test(panel().textContent)
  && /Version/.test(panel().textContent));
__finish({ system: panel().textContent });
"""


def test_the_home_system_status_names_the_real_storage_and_version(tmp_path, monkeypatch) -> None:
    """System Status showed a fixed "Storage local-first" chip. It now shows the data
    root get_state reports and the package version."""
    monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "data"))
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Pier 8 status", str(tmp_path / "project"))["ok"]
    result = run_page(
        api, HOME_DRIVER, tmp_path / "profile", page_name="workspace.html",
        before_js="localStorage.setItem('odk.workspace.last', 'home');",
        ready_js="document.querySelectorAll('.tbtn').length > 0",
    )
    assert "error" not in result, result
    data_root = str(api.get_state()["data_paths"]["data_root"])
    assert data_root in result["system"], result
    assert f"v{app_version()}" in result["system"]
    assert "local-first" not in result["system"]
