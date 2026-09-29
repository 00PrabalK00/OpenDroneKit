"""Every native menu item does something in both UIs.

app/shell.py builds the OS menu bar and dispatches each item into the page as
`window.odk.onMenu({action, payload})`. The classic UI handled all of them; the cockpit
let seven fall through to "Menu action not handled here" (Connect Vehicle, Arm, Start,
Pause, Resume, Return to Home, Abort) and sent others to the wrong verb -- Model Manager
started the full reconstruction pipeline, Mission History ran Simulate.

The action ids are collected by pressing every real menu item against a stand-in for
the webview window (the one external boundary), so the list cannot drift from the menu.
The last test drives the real cockpit in a hidden WebView2 window and is opt-in
(ODK_GUI_TESTS=1) because it needs a desktop session.
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
            "view.cache_tiles", "tools.audit", "view.panel"} <= actions
    commands = {p["command"] for _, a, p in messages if a == "fly.command"}
    assert commands == {"arm", "start", "pause", "resume", "rth", "abort"}


def test_the_classic_ui_handles_every_item(messages) -> None:
    source = (WEB / "app.js").read_text(encoding="utf-8")
    handled = set(re.findall(r"'([a-z_]+\.[a-z_]+)':", source))
    missing = sorted({a for _, a, _ in messages} - handled)
    assert not missing, f"classic UI has no handler for {missing}"


def test_the_cockpit_handles_every_item(messages) -> None:
    source = (WEB / "workspace" / "shell.js").read_text(encoding="utf-8")
    bridge = source.split("  installMenuBridge() {", 1)[1].split("\n  buildNav() {", 1)[0]
    mapped = set(re.findall(r'"([a-z_]+\.[a-z_]+)":\s*"', bridge.split("const panelWorkspace")[0]))
    cases = set(re.findall(r'case "([a-z_]+\.[a-z_]+)":', bridge))
    to_action = dict(re.findall(r'"([a-z_]+\.[a-z_]+)":\s*"([^"]+)"',
                                bridge.split("const toAction = {", 1)[1].split("};", 1)[0]))
    missing = sorted({a for _, a, _ in messages} - (set(to_action) | cases))
    assert not missing, f"cockpit falls through to 'not handled' for {missing}"
    assert mapped  # sanity: the parse found the maps

    # And the verbs are the right ones.
    assert to_action["analysis.models"] == "Verify Models"
    assert to_action["mission.history"] == "History"
    assert "tools.capabilities" not in to_action  # it is not Preflight
    assert 'call("vehicle_command", command)' in bridge


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
