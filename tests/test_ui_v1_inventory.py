"""The v1 interface drops nothing: every menu item, cockpit verb and Api method has a home.

docs/ui/inventory.json is the ledger the rebuild works from (REBUILD_PLAN section 1 as
data). The classic UI, the cockpit and the native menu between them reach every feature
the application has; the new interface replaces both, and the likeliest way to lose a
feature is to rebuild from one of them and forget the other. So the ledger is checked
against the sources themselves, not against a list someone typed:

  * every native menu item, read from app/shell.py, including each payload variant;
  * every key of ACTIONS in js/workspace/actions.js (the cockpit's verb catalogue);
  * every public method of app.api.Api.

Each must name a row, and each row must say where the control now lives.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "app" / "web"
INVENTORY = REPO / "docs" / "ui" / "inventory.json"


@pytest.fixture(scope="module")
def inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def menu_items() -> list[tuple[str, str]]:
    """(menu id, title) for every native menu item that dispatches into the page.

    Read from the source rather than by building the menu, so the test needs neither
    pywebview nor a window. test_menu_coverage.py presses the real items; the count
    check below keeps this parser honest against the number of dispatching items.
    """
    source = (REPO / "app" / "shell.py").read_text(encoding="utf-8")
    items = []
    for title, action, payload in re.findall(
            r'MenuAction\("([^"]+)",\s*act\("([^"]+)"(?:,\s*(\{[^}]*\}))?\)\)', source):
        data = json.loads(payload) if payload else {}
        suffix = "{" + ",".join(f"{k}={v}" for k, v in data.items()) + "}" if data else ""
        items.append((action + suffix, title))
    assert len(items) == source.count('act("'), "a menu item this parser cannot read"
    return items


def action_keys() -> list[str]:
    source = (WEB / "js" / "workspace" / "actions.js").read_text(encoding="utf-8")
    body = source.split("export const ACTIONS = {")[1].split("\nexport const UNWIRED")[0]
    return [a or b for a, b in re.findall(r'^  "([^"]+)":|^  (\w+):', body, re.M)]


def api_methods() -> list[str]:
    tree = ast.parse((REPO / "app" / "api.py").read_text(encoding="utf-8"))
    api = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Api")
    return [f.name for f in api.body
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and not f.name.startswith("_")]


class TestNothingIsDropped:
    def test_every_native_menu_item_has_a_row(self, inventory) -> None:
        items = menu_items()
        assert len(items) >= 60  # 62 when this was written; the menu only grows
        rows = {r["id"] for r in inventory["rows"]}
        missing = [mid for mid, _ in items if not inventory["menu"].get(mid, {}).get("rows")]
        assert not missing, f"menu items with no inventory row: {missing}"
        dangling = {mid: [r for r in e["rows"] if r not in rows] for mid, e in inventory["menu"].items()}
        assert not any(dangling.values()), f"menu index names rows that do not exist: {dangling}"
        for mid, title in items:
            assert any(mid in r["menu"] for r in inventory["rows"]), (mid, title)

    def test_every_cockpit_action_has_a_row(self, inventory) -> None:
        keys = action_keys()
        assert len(keys) >= 120
        missing = [k for k in keys if not inventory["actions"].get(k)]
        assert not missing, f"ACTIONS keys with no inventory row: {missing}"
        for key in keys:
            assert any(key in r["actions"] for r in inventory["rows"]), key

    def test_every_api_method_has_a_row_or_is_declared_internal(self, inventory) -> None:
        internal = set(inventory["internal_api"])
        missing = [m for m in api_methods()
                   if m not in internal and not any(m in r["api"] for r in inventory["rows"])]
        assert not missing, f"Api methods with no inventory row: {missing}"

    def test_the_ledger_names_only_things_that_exist(self, inventory) -> None:
        keys, methods = set(action_keys()), set(api_methods())
        menu = {mid for mid, _ in menu_items()}
        for row in inventory["rows"]:
            assert set(row["actions"]) <= keys, (row["id"], set(row["actions"]) - keys)
            assert set(row["api"]) <= methods, (row["id"], set(row["api"]) - methods)
            assert set(row["menu"]) <= menu, (row["id"], set(row["menu"]) - menu)

    def test_methods_no_ui_reached_are_given_a_home(self, inventory) -> None:
        for method in inventory["unreached_before_v1"]:
            homes = [r for r in inventory["rows"] if method in r["star_methods"]]
            assert homes, f"{method} was reached by neither UI and still has no home"
            assert all(r["new_home"].strip() for r in homes)


class TestEveryRowHasAHome:
    def test_rows_are_unique_and_placed(self, inventory) -> None:
        ids = [r["id"] for r in inventory["rows"]]
        assert len(ids) == len(set(ids)), "duplicate row ids"
        kinds = set(inventory["placement_kinds"])
        for row in inventory["rows"]:
            assert row["workspace"] in inventory["workspaces"], row["id"]
            assert row["new_home"].strip(), f"{row['id']} has no new home"
            assert row["placement"] in kinds, row["id"]
            assert row["id"].startswith(row["workspace"] + "."), row["id"]

    def test_the_markdown_is_generated_from_the_json(self) -> None:
        result = subprocess.run([sys.executable, str(REPO / "tools" / "ui_inventory.py"), "--check"],
                                capture_output=True, text=True, cwd=REPO)
        assert result.returncode == 0, result.stderr


class TestTheContractIsInPlace:
    WORKSPACES = ["home", "plan", "fly", "process", "analyse", "review", "fleet", "report", "settings"]
    FIELD = ["plan", "fly", "process", "analyse", "report"]

    def test_every_screen_stub_exists(self) -> None:
        screens = WEB / "js" / "odk" / "screens"
        for ws in self.WORKSPACES:
            for part in ("surface", "model", "desk", "actions"):
                assert (screens / ws / f"{part}.js").is_file(), f"{ws}/{part}.js"
        for ws in self.FIELD:
            assert (screens / "field" / f"{ws}.js").is_file(), f"field/{ws}.js"

    def test_screens_render_no_literal_numbers(self) -> None:
        """Honesty rule 1, at its U00 strength: the stubs carry no digit at all, so a
        figure pasted from an artboard cannot hide in a stub. Workspace units replace
        this with the live()-aware check (REBUILD_PLAN 3.3, test_ui_honesty)."""
        screens = WEB / "js" / "odk" / "screens"
        offenders = {}
        for path in screens.rglob("*.js"):
            text = path.read_text(encoding="utf-8")
            if "live(" in text:
                continue  # owned by the workspace unit's live()-aware guard
            digits = re.findall(r"\d", text)
            if digits:
                offenders[str(path.relative_to(WEB))] = len(digits)
        assert not offenders, f"literal digits in screen modules: {offenders}"

    def test_every_surface_declares_its_screen(self) -> None:
        for ws in self.WORKSPACES:
            text = (WEB / "js" / "odk" / "screens" / ws / "surface.js").read_text(encoding="utf-8")
            assert f'export const SCREEN = "{ws}";' in text
            assert "export const PLACEMENTS = [" in text

    def test_odk_opens_only_when_asked_for(self, monkeypatch) -> None:
        from app.shell import DEFAULT_UI, UI_PAGES, resolve_ui

        assert UI_PAGES["odk"] == "odk.html"
        assert (WEB / "odk.html").is_file()
        assert DEFAULT_UI == "classic"
        monkeypatch.delenv("ODK_UI", raising=False)
        assert resolve_ui() == "classic"
        monkeypatch.setenv("ODK_UI", "ODK")
        assert UI_PAGES[resolve_ui()] == "odk.html"
        monkeypatch.delenv("ODK_UI", raising=False)
        assert UI_PAGES[resolve_ui("odk")] == "odk.html"

    def test_the_alias_table_matches_the_contract(self) -> None:
        text = (WEB / "js" / "odk" / "screens" / "index.js").read_text(encoding="utf-8")
        body = text.split("export const WORKSPACE_ALIASES = {")[1].split("};")[0]
        aliases = dict(re.findall(r"(\w+): \"(\w+)\"", body))
        assert aliases == {
            "planning": "plan", "flight": "fly", "processing": "process", "verification": "process",
            "measurements": "analyse", "twin": "analyse", "thermal": "analyse", "inspection": "review",
            "reports": "report", "projects": "home", "developers": "settings",
        }
        doc = (REPO / "docs" / "ui" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        for old in aliases:
            assert f"`{old}`" in doc, f"{old} missing from the ARCHITECTURE alias table"
