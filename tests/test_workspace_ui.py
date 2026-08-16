"""The workspace UI is executed, not eyeballed.

A screenshot proves a layout renders once, on one machine, at one window size. It proves
nothing about the fourteen workspaces nobody screenshotted, and a panel that throws on
render disappears silently -- the dock keeps going and the operator sees a gap where the
telemetry should be.

So the JavaScript is actually run: every workspace mounted through the real dock, every
panel rendered, under a stub DOM. Node is used because it is already required for nothing
else here and the alternative -- trusting that the modules parse -- is not verification.

Skipped when Node is absent rather than silently passing, because a test that cannot run
must not report success.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "app" / "web"
CHECK = WEB / "js" / "workspace" / "__tests__" / "render_check.mjs"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


class TestTheFilesAreThere:
    def test_the_entry_page_exists(self) -> None:
        assert (WEB / "workspace.html").is_file()

    @pytest.mark.parametrize("name", ["dock.js", "primitives.js", "workspaces.js", "shell.js"])
    def test_each_module_exists(self, name) -> None:
        assert (WEB / "js" / "workspace" / name).is_file()

    def test_the_stylesheet_exists(self) -> None:
        assert (WEB / "css" / "workspace.css").is_file()

    def test_the_entry_page_references_what_it_needs(self) -> None:
        html = (WEB / "workspace.html").read_text(encoding="utf-8")
        assert "css/workspace.css" in html
        assert "js/workspace/shell.js" in html
        assert 'type="module"' in html, "ES modules need the module type or nothing loads"


@node
class TestEveryWorkspaceRenders:
    @pytest.fixture(scope="class")
    def result(self):
        return subprocess.run(
            ["node", str(CHECK)],
            cwd=CHECK.parent, capture_output=True, text=True, timeout=120,
        )

    def test_the_check_passes(self, result) -> None:
        assert result.returncode == 0, (
            f"workspace render check failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_every_workspace_is_present(self, result) -> None:
        # Home, Projects, Planning, Flight, Verification, Processing, Twin, Inspection,
        # Thermal, Measurements, Fleet, Reports, Developers, Settings.
        assert "workspaces: 14" in result.stdout

    def test_panels_actually_rendered(self, result) -> None:
        """A workspace that mounts with no panels is a blank cockpit."""
        line = [l for l in result.stdout.splitlines() if l.startswith("panels rendered:")]
        assert line, result.stdout
        assert int(line[0].split(":")[1]) > 60

    def test_nothing_threw(self, result) -> None:
        assert "all workspaces render" in result.stdout


class TestDesignRules:
    """The few style rules that carry meaning rather than taste."""

    @pytest.fixture(scope="class")
    def css(self) -> str:
        return (WEB / "css" / "workspace.css").read_text(encoding="utf-8")

    def test_the_semantic_colours_are_defined(self, css) -> None:
        for token in ("--accent:", "--ok:", "--warn:", "--error:", "--thermal:"):
            assert token in css

    def test_corner_radius_stays_low(self, css) -> None:
        # The spec asks for 2-6px. A large radius reads as a consumer card, and the
        # density of this interface depends on panels looking like panels.
        line = [l for l in css.splitlines() if "--radius:" in l][0]
        value = int("".join(c for c in line.split(":")[1] if c.isdigit()))
        assert 2 <= value <= 6

    def test_rows_are_dense(self, css) -> None:
        line = [l for l in css.splitlines() if "--row:" in l][0]
        assert int("".join(c for c in line.split(":")[1] if c.isdigit())) <= 26

    def test_safety_critical_actions_are_styled_apart(self, css) -> None:
        """Abort must never look like Save."""
        assert ".tbtn.danger" in css

    def test_the_shell_marks_unconnected_data(self) -> None:
        """The frame says plainly that nothing shown has been measured."""
        shell = (WEB / "js" / "workspace" / "shell.js").read_text(encoding="utf-8")
        assert "sample data" in shell
