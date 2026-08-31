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


class TestTheAppOpensSomethingThatWorks:
    def test_the_desktop_shell_opens_the_wired_ui(self, shell_py) -> None:
        """index.html, because it is the only one of the two whose buttons do anything.

        The cockpit was briefly made the default on the strength of its layout. Every
        toolbar button in it routes to runAction(), which writes "not wired to the API
        yet" into the status bar -- in text small enough that the app simply appears
        broken. A better-looking UI that cannot plan a mission is a downgrade.
        """
        assert 'else "index.html"' in shell_py

    def test_the_cockpit_is_reachable_for_development(self, shell_py) -> None:
        assert 'ODK_UI' in shell_py
        assert '"workspace.html"' in shell_py

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
        """The banner must follow what is on screen, not what the bridge reports.

        This test used to assert the literal line `this.mode !== "connected"`, on the
        reasoning that connected is the only state in which what you are looking at was
        measured. That reasoning was wrong, and pinning the source text kept it wrong.

        Measured on the real window, connected, with a 77-image project open: Processing
        showed "DEMO site 1/2/3" and "1,842 images", did NOT show the project's actual 77,
        and the EXAMPLE DATA banner was collapsed to height 0. Panels fall back to the
        structural sample whenever they have nothing wired, and they do that while
        connected too -- so the app presented invented figures with its own disclaimer
        suppressed BECAUSE it believed it was connected.

        The condition must therefore consult the rendered content as well as the mode.
        """
        assert 'this.banner.classList.toggle("hidden"' in shell_js
        assert "showingSampleContent" in shell_js, (
            "the banner condition ignores what is actually rendered"
        )
        assert 'const synthetic = this.mode !== "connected" || this.showingSampleContent();' in shell_js

        # Being disconnected must still raise it: panels render the sample then too.
        assert 'this.mode !== "connected"' in shell_js

    def test_the_sample_sentinels_come_from_the_demo_constants(self) -> None:
        """The detector must not carry its own copy of what the sample looks like.

        A second hand-maintained list would drift the moment a panel changed which demo
        constant it renders, and the failure would be silent in the direction that
        matters: no banner over invented content.
        """
        demo_js = (WORKSPACE_JS / "demo.js").read_text(encoding="utf-8")
        assert "export const SAMPLE_SENTINELS" in demo_js
        sentinels = demo_js.split("export const SAMPLE_SENTINELS")[1]
        assert "DEMO.ORG" in sentinels, "sentinels must be derived from the demo constants"
        assert "DEMO.SITES" in sentinels

    def test_the_banner_is_re_evaluated_when_the_workspace_changes(self, shell_js) -> None:
        """Which panels are sampled differs between workspaces.

        Deciding once at startup would leave the banner correct on whichever workspace
        happened to open first and wrong on most of the others.
        """
        opened = shell_js.split("  open(id) {")[1].split("\n  buildToolbar(")[0]
        assert "applyMode()" in opened

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
