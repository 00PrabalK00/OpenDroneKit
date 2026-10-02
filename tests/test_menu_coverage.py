"""Every native menu item does something: in odk.html (the v1 shell) and in both old UIs.

app/shell.py builds the OS menu bar and dispatches each item into the page as
`window.odk.onMenu({action, payload})`. The classic UI handled all of them; the cockpit
let seven fall through to "Menu action not handled here" (Connect Vehicle, Arm, Start,
Pause, Resume, Return to Home, Abort) and sent others to the wrong verb -- Model Manager
started the full reconstruction pipeline, Mission History ran Simulate.

The action ids are collected by pressing every real menu item against a stand-in for
the webview window (the one external boundary), so the list cannot drift from the menu.
The v1 shell (js/odk/menu.js) is checked twice: its MENU table names every id, and every
item is pressed in headless Edge/Chromium against the real Api (tests/ui_live_bridge.py),
with no "not handled" and a visible effect for each. The ids added for v1 (V1_ONLY) are
answered by odk.html only; the classic and cockpit pages are retired at the switchover.
The cockpit WebView2 test is opt-in (ODK_GUI_TESTS=1) because it needs a desktop session.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "app" / "web" / "js"

pytest.importorskip("webview")

# Menu ids added with the v1 shell (REBUILD_PLAN 2.7): odk.html handles them; the classic
# and cockpit pages answer "not handled here" until they are removed (U15).
V1_ONLY = {"view.field_mode", "view.settings", "view.palette", "view.toggle_panels"}


class _RecordingWindow:
    """Stands in for the pywebview window: records what the menu asks the page to run."""

    def __init__(self) -> None:
        self.scripts: list[str] = []

    def evaluate_js(self, script: str) -> None:
        self.scripts.append(script)


def _menu_messages() -> list[tuple[str, str, dict]]:
    from app.shell import Shell

    shell = Shell()
    shell.window = _RecordingWindow()
    out = []
    for menu in shell.build_menu():
        for item in menu.items:
            function = getattr(item, "function", None)
            if function is None or getattr(function, "__name__", "") == "_quit":
                continue
            before = len(shell.window.scripts)
            function()
            if len(shell.window.scripts) == before:
                continue
            script = shell.window.scripts[-1]
            message = json.loads(script[script.index("onMenu(") + len("onMenu("):-1])
            out.append((item.title, message["action"], message["payload"]))
    return out


@pytest.fixture(scope="module")
def messages():
    return _menu_messages()


def test_the_menu_is_what_we_think(messages) -> None:
    actions = {action for _, action, _ in messages}
    assert {"fly.connect", "fly.command", "analysis.models", "mission.history",
            "view.cache_tiles", "tools.audit", "view.panel"} | V1_ONLY <= actions
    assert len(messages) >= 66  # 62 before v1, plus the four V1_ONLY ids
    commands = {p["command"] for _, a, p in messages if a == "fly.command"}
    assert commands == {"arm", "start", "pause", "resume", "rth", "abort"}


def test_the_classic_ui_handles_every_item(messages) -> None:
    source = (WEB / "app.js").read_text(encoding="utf-8")
    handled = set(re.findall(r"'([a-z_]+\.[a-z_]+)':", source))
    missing = sorted({a for _, a, _ in messages} - handled - V1_ONLY)
    assert not missing, f"classic UI has no handler for {missing}"


def test_the_cockpit_handles_every_item(messages) -> None:
    source = (WEB / "workspace" / "shell.js").read_text(encoding="utf-8")
    bridge = source.split("  installMenuBridge() {", 1)[1].split("\n  buildNav() {", 1)[0]
    mapped = set(re.findall(r'"([a-z_]+\.[a-z_]+)":\s*"', bridge.split("const panelWorkspace")[0]))
    cases = set(re.findall(r'case "([a-z_]+\.[a-z_]+)":', bridge))
    to_action = dict(re.findall(r'"([a-z_]+\.[a-z_]+)":\s*"([^"]+)"',
                                bridge.split("const toAction = {", 1)[1].split("};", 1)[0]))
    missing = sorted({a for _, a, _ in messages} - (set(to_action) | cases) - V1_ONLY)
    assert not missing, f"cockpit falls through to 'not handled' for {missing}"
    assert mapped  # sanity: the parse found the maps

    # And the verbs are the right ones.
    assert to_action["analysis.models"] == "Verify Models"
    assert to_action["mission.history"] == "History"
    assert "tools.capabilities" not in to_action  # it is not Preflight
    # Through the helper that holds Start to the preflight checklist (and asks).
    assert "this.sendVehicleCommand(command)" in bridge


def test_the_cockpit_vehicle_verbs_are_ones_the_api_accepts(messages) -> None:
    api = (REPO_ROOT / "app" / "api.py").read_text(encoding="utf-8")
    block = api.split("def vehicle_command", 1)[1].split("method_name = mapping.get", 1)[0]
    accepted = set(re.findall(r'"([a-z_]+)":\s*"', block))
    commands = {p["command"] for _, a, p in messages if a == "fly.command"}
    assert commands <= accepted


def test_clearing_the_area_of_interest_clears_it() -> None:
    """Both UIs clear with set_aoi([]); that used to be refused as a 0-vertex polygon,
    so the session kept the old area while the map showed none."""
    from app.api import Api
    from app.session import AppSession

    api = Api(AppSession())
    assert api.set_aoi([[0, 0], [0.001, 0], [0.001, 0.001]])["ok"]
    cleared = api.set_aoi([])
    assert cleared["ok"] and cleared["vertices"] == 0
    assert api.get_state()["aoi"] == []
    assert api.set_aoi([[0, 0], [1, 1]])["ok"] is False


@pytest.mark.skipif(os.environ.get("ODK_GUI_TESTS") != "1" or sys.platform != "win32",
                    reason="drives a real WebView2 window; set ODK_GUI_TESTS=1 on a desktop")
def test_every_item_is_handled_in_the_real_cockpit(tmp_path, messages) -> None:

    import webview

    from app.shell import WEB_ROOT, Shell

    shell = Shell()
    url = shell._serve(WEB_ROOT, "workspace.html", runtime=shell.runtime)
    window = webview.create_window("menu probe", url, js_api=shell.api, hidden=True)
    shell.window = window
    shell.api.bind_window(window)
    results: dict = {}
    # Opening a folder would launch Explorer on the tester's desktop.
    skip = {"project.reveal", "recon.reveal"}

    def drive() -> None:
        try:
            for _ in range(80):
                if window.evaluate_js("document.querySelector('.chip.ok') ? 1 : 0"):
                    break
                time.sleep(0.25)
            items = {a.title: a.function for m in shell.build_menu() for a in m.items
                     if getattr(a, "function", None) is not None}
            for title, action, _payload in messages:
                if action in skip:
                    continue
                window.evaluate_js("document.querySelectorAll('.toast').forEach(t => t.remove())")
                items[title]()
                time.sleep(0.6)
                results[title] = window.evaluate_js(
                    "JSON.stringify([...document.querySelectorAll('.toast')].map(t => t.textContent))")
                window.evaluate_js(
                    "document.querySelectorAll('.modal-backdrop,.palette-backdrop').forEach(n => n.remove())")
        finally:
            window.destroy()

    webview.start(drive)
    assert results, "the probe never ran"
    unhandled = {t: r for t, r in results.items() if "not handled" in (r or "")}
    assert not unhandled, unhandled


# -- the v1 shell (odk.html) -------------------------------------------------------

V1_MENU = WEB / "odk" / "menu.js"


def _v1_table() -> dict[str, dict[str, str]]:
    source = V1_MENU.read_text(encoding="utf-8")
    body = source.split("export const MENU = {", 1)[1].split("\n};", 1)[0]
    table = {}
    for mid, spec in re.findall(r'"([a-z_]+\.[a-z_]+)":\s*\{([^}]*)\}', body):
        table[mid] = dict(re.findall(r'(open|action|handler):\s*"([^"]+)"', spec))
    return table


def test_the_v1_shell_maps_every_item(messages) -> None:
    table = _v1_table()
    missing = sorted({a for _, a, _ in messages} - set(table))
    assert not missing, f"odk.html has no MENU entry for {missing}"
    source = V1_MENU.read_text(encoding="utf-8")
    handlers = set(re.findall(r"^  (?:async )?([a-zA-Z]+)\(shell", source, re.M))
    for mid, spec in table.items():
        assert ("action" in spec) != ("handler" in spec), (mid, spec)
        if "handler" in spec:
            assert spec["handler"] in handlers, (mid, spec)
    # The verbs are the right ones (the cockpit got these wrong once).
    assert table["analysis.models"]["action"] == "Verify Models"
    assert table["mission.history"]["action"] == "History"
    assert "action" not in table["tools.capabilities"]
    # The View menu's panels go to the screens they now live on (REBUILD_PLAN 2.7).
    assert 'PANEL_SCREEN = { layers: "analyse", mission: "plan", telemetry: "fly", jobs: "process" }' in source
    # Through the helper that holds Start to the preflight checklist.
    assert "shell.sendVehicleCommand(command)" in source


LOOK = """window.__toasts = [];
new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
  if (n.classList && n.classList.contains('odk-toast')) window.__toasts.push(n.textContent);
}))).observe(document.body, { childList: true, subtree: true });
window.__look = () => JSON.stringify({
  toasts: window.__toasts.length,
  dialog: !!document.querySelector('.modal-backdrop, [data-palette], [data-shortcuts]'),
  mode: document.documentElement.dataset.mode, screen: odkShell.workspaceId,
  left: document.querySelector('.odk-frame').dataset.left || '',
  inspector: document.querySelector('.odk-frame').dataset.inspector || '' });
odkShell.confirmAll = true; true"""

RESET = """document.querySelectorAll('.modal-backdrop,[data-palette],[data-shortcuts]').forEach((n) => n.remove());
(document.documentElement.dataset.mode !== 'desk' ? odkShell.setMode('desk') : Promise.resolve()).then(() => true)"""


def test_every_item_is_handled_in_odk_html(tmp_path, messages, monkeypatch) -> None:
    """Press every menu item in odk.html against the real Api; each must do something."""
    pytest.importorskip("websocket")
    pytest.importorskip("requests")
    sys.path.insert(0, str(Path(__file__).parent))
    from ui_live_bridge import Browser, chromium_path, serve

    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    monkeypatch.chdir(tmp_path)
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("menu", str(tmp_path / "project"))
    api.set_aoi([[-81.752, 41.303], [-81.748, 41.303], [-81.748, 41.306], [-81.752, 41.306]])
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    api.set_geocoding_provider("offline")
    server, base = serve(api)
    browser = Browser(executable)
    # Opening a folder would launch Explorer on the tester's desktop.
    skip = {"project.reveal", "recon.reveal"}
    results: dict[str, dict] = {}
    try:
        browser.open(f"{base}/odk.html?mode=desk", settle_s=0.2)
        browser.wait_for("window.__odkReady === true || window.__odkBootError", 30)
        assert browser.js("window.__odkBootError || null") is None
        browser.wait_for("odkShell.connection === 'connected'", 20)
        assert set(browser.js("window.odk.menuIds()")) >= {a for _, a, _ in messages}
        browser.js(LOOK)
        for title, action, payload in messages:
            if action in skip:
                continue
            before = browser.js("__look()")
            seen = browser.js("__toasts.length")
            browser.js(f"window.odk.onMenu({json.dumps({'action': action, 'payload': payload})}); true")
            after = before
            deadline = time.time() + 20
            while time.time() < deadline:
                time.sleep(0.25)
                after = browser.js("__look()")
                busy = browser.js("[...document.querySelectorAll('.odk-toast')].some((t) => t.textContent.endsWith('…'))")
                if after != before and not busy:
                    break
            results[title] = {"changed": after != before, "toasts": browser.js(f"__toasts.slice({seen})")}
            browser.js(RESET)  # close whatever the item opened; the next starts from the frame
            time.sleep(0.2)
    finally:
        browser.close()
        server.shutdown()
    assert len(results) == len(messages) - len(skip)
    unhandled = {t: r for t, r in results.items() if any("not handled" in x for x in r["toasts"])}
    assert not unhandled, unhandled
    silent = sorted(t for t, r in results.items() if not r["changed"])
    assert not silent, f"menu items with no visible effect in odk.html: {silent}"
