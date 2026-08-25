"""The interface the documentation describes is the one the application opens.

docs/UI_GUIDE.md opens with "app/web/workspace.html is the operations interface", the
README points at it, and a test asserted the file existed. Nothing loaded it. app/shell.py
opened index.html, so the cockpit was a UI no user could reach and the screen they did
reach was documented nowhere -- two bugs that hid each other, because each artefact was
internally consistent.

These tests tie the three together: what the shell opens, what the docs claim, and
whether the thing being opened can actually talk to the application.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "app" / "web"
WORKSPACE_JS = WEB / "js" / "workspace"


@pytest.fixture(scope="module")
def shell_py() -> str:
    return (REPO_ROOT / "app" / "shell.py").read_text(encoding="utf-8")


class TestTheAppOpensTheDocumentedInterface:
    def test_the_desktop_shell_opens_the_cockpit(self, shell_py) -> None:
        assert '"workspace.html"' in shell_py

    def test_the_classic_shell_is_still_reachable(self, shell_py) -> None:
        """Kept behind ODK_UI=classic rather than deleted.

        index.html is the more completely wired of the two while the cockpit's
        workspaces are connected to the Api one at a time, and removing a working screen
        before its replacement is finished is how a rewrite loses capability quietly.
        """
        assert 'ODK_UI' in shell_py
        assert '"index.html"' in shell_py

    def test_the_documentation_names_what_actually_opens(self) -> None:
        guide = (REPO_ROOT / "docs" / "UI_GUIDE.md").read_text(encoding="utf-8")
        assert "workspace.html" in guide


class TestTheCockpitCanReachTheApplication:
    @pytest.fixture(scope="class")
    def api_js(self) -> str:
        return (WORKSPACE_JS / "api.js").read_text(encoding="utf-8")

    def test_it_speaks_to_the_same_bridge_as_the_classic_shell(self, api_js) -> None:
        assert "window.pywebview" in api_js

    def test_a_failed_result_is_an_error_not_data(self, api_js) -> None:
        """The Api answers {ok: false, error: ...} rather than raising, so a caller that
        only checks for an exception will render a refusal as though it were a result."""
        assert "ok === false" in api_js

    def test_it_waits_for_the_bridge_instead_of_racing_it(self, api_js) -> None:
        """pywebview fires pywebviewready after load; reading the API before that and
        concluding there is no bridge is how a connected app shows itself as offline."""
        assert "pywebviewready" in api_js

    def test_it_distinguishes_absent_from_unwired(self, api_js) -> None:
        """A panel with no API behind it must say so.

        'No data' and 'this was never connected' look identical on screen and mean
        completely different things -- one is an answer, the other is missing software.
        """
        assert "unwired" in api_js
        assert "export function available" in api_js


class TestTheShellDistinguishesItsThreeStates:
    @pytest.fixture(scope="class")
    def shell_js(self) -> str:
        return (WORKSPACE_JS / "shell.js").read_text(encoding="utf-8")

    def test_it_asks_the_bridge_what_is_really_there(self, shell_js) -> None:
        assert "list_projects" in shell_js

    def test_the_demo_banner_shows_whenever_content_is_synthetic(self, shell_js) -> None:
        """Connected is the only state in which what is on screen was measured.

        Keying this on demo mode alone was wrong: with no bridge the panels still render
        the structural sample, so hiding the banner outside demo mode left synthetic
        content on screen with nothing saying so -- worse than the clipped chip it
        replaced.
        """
        assert 'this.banner.classList.toggle("hidden"' in shell_js
        assert 'const synthetic = this.mode !== "connected";' in shell_js

    def test_hiding_the_banner_does_not_break_the_grid(self) -> None:
        """The shell's rows are positional, so display: none on the banner shifts every
        following child up one -- the workspace took the 34px toolbar row and the status
        bar took the canvas row. It collapses to zero height instead of leaving the grid.
        """
        css = (WEB / "css" / "workspace.css").read_text(encoding="utf-8")
        assert ".demo-banner.hidden" in css
        assert "display: block !important" in css

    def test_connecting_replaces_demo_content(self, shell_js) -> None:
        assert 'this.mode = "connected"' in shell_js

    def test_an_explicit_demo_survives_a_real_connection(self, shell_js) -> None:
        """Someone demonstrating the product on a machine with real projects still wants
        the demo they asked for."""
        assert 'if (this.mode === "demo") return;' in shell_js
