"""Every button in the cockpit either does something or says why it cannot.

The cockpit shipped with one handler for all seventy-five toolbar actions across
fourteen workspaces:

    runAction(action) {
      this.selectionLabel.textContent = `action: ${action} (not wired to the API yet)`;
    }

That is 11px text at the bottom of a 900px window, so the application did not look partly
finished -- it looked broken. Clicking anything appeared to do nothing at all.

These tests hold three properties:

  * an action that reaches the Api names a method that exists,
  * an action that cannot reach the Api says which capability is missing,
  * an action that starts real work or moves an aircraft asks first.

The second matters as much as the first. "Nothing happened" and "this feature does not
exist yet" look identical to a user and mean completely different things.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_JS = REPO_ROOT / "app" / "web" / "js" / "workspace"


@pytest.fixture(scope="module")
def actions_js() -> str:
    return (WORKSPACE_JS / "actions.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def shell_js() -> str:
    return (WORKSPACE_JS / "shell.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def api_methods() -> set[str]:
    source = (REPO_ROOT / "app" / "api.py").read_text(encoding="utf-8")
    return set(re.findall(r"^    def ([a-z_]+)\(", source, re.MULTILINE))


@pytest.fixture(scope="module")
def toolbar_actions() -> set[str]:
    source = (WORKSPACE_JS / "workspaces.js").read_text(encoding="utf-8")
    found: set[str] = set()
    for block in re.findall(r"toolbar:\s*\[(.*?)\]", source, re.DOTALL):
        found.update(name for name in re.findall(r'"([^"]+)"', block) if name != "|")
    return found


class TestTheStubIsGone:
    def test_no_action_reports_itself_as_unwired_wholesale(self, shell_js) -> None:
        assert "not wired to the API yet" not in shell_js

    def test_actions_run_against_the_bridge(self, shell_js) -> None:
        assert "await entry.run(this.actionContext())" in shell_js


class TestEveryCalledMethodExists:
    def test_the_actions_only_call_real_api_methods(self, actions_js, api_methods) -> None:
        """A typo here is a button that throws instead of working, and the throw would
        be swallowed by the click handler."""
        called = set(re.findall(r'call\("([a-z_]+)"', actions_js))
        assert called, "no Api calls found; the table is not wired to anything"
        missing = sorted(called - api_methods)
        assert not missing, f"actions.js calls methods that do not exist: {missing}"

    def test_the_shell_only_calls_real_api_methods(self, shell_js, api_methods) -> None:
        called = set(re.findall(r'(?:call|tryCall)\("([a-z_]+)"', shell_js))
        missing = sorted(called - api_methods)
        assert not missing, f"shell.js calls methods that do not exist: {missing}"


class TestEveryButtonIsAccountedFor:
    def test_no_toolbar_action_falls_through_silently(
        self, actions_js, toolbar_actions
    ) -> None:
        """Every label on a button is either handled or explicitly declared missing.

        A label in neither list produces "no handler", which is honest but means nobody
        decided what that button is for.
        """
        handled = set(re.findall(r'^  "?([A-Za-z0-9 &]+?)"?:\s*\{', actions_js, re.MULTILINE))
        unwired_block = actions_js[actions_js.index("export const UNWIRED"):]
        declared = set(re.findall(r'^  "?([A-Za-z0-9 &]+?)"?:\s*"', unwired_block, re.MULTILINE))
        unaccounted = sorted(toolbar_actions - handled - declared)
        assert not unaccounted, (
            f"{len(unaccounted)} toolbar actions have no handler and no explanation: "
            f"{unaccounted}"
        )

    def test_the_unwired_ones_say_what_is_missing(self, actions_js) -> None:
        """Not 'coming soon'. Which capability, and where it would live."""
        unwired_block = actions_js[actions_js.index("export const UNWIRED"):]
        reasons = re.findall(r':\s*"([^"]{20,})"', unwired_block)
        assert len(reasons) >= 10
        assert any("service" in reason for reason in reasons)


class TestReconstructionIsReachable:
    """The thing the user asked for by name: COLMAP, from a button."""

    def test_process_starts_a_reconstruction(self, actions_js) -> None:
        assert 'call("run_reconstruction"' in actions_js

    def test_start_runs_the_full_pipeline(self, actions_js) -> None:
        assert 'call("run_pipeline"' in actions_js

    def test_a_long_run_asks_before_starting(self, actions_js) -> None:
        """A reconstruction can run for hours. Starting one by misclick is not a
        recoverable mistake on a laptop in a site office."""
        process = actions_js[actions_js.index('  Process: {'):]
        assert "confirm:" in process[:400]

    def test_the_job_is_followed_to_completion(self, shell_js) -> None:
        """A job id in a toast that never updates is the same as no feedback."""
        assert "watchJob" in shell_js
        assert 'tryCall("job_status"' in shell_js


class TestDangerousActionsAskFirst:
    @pytest.mark.parametrize("action", ["Abort", "RTL", "Land", "Upload", "Manual Override"])
    def test_it_confirms(self, actions_js, action) -> None:
        """A misclick on Abort must not be the same gesture as a misclick on Pan."""
        block = actions_js[actions_js.index(f'  {action}:' if f'  {action}:' in actions_js
                                            else f'  "{action}":'):]
        assert "confirm:" in block[:400], f"{action} runs without asking"


class TestPrerequisitesAreCheckedBeforeTheCall:
    def test_a_missing_project_is_refused_locally(self, actions_js) -> None:
        """The same refusal arriving from Python three seconds later reads as a crash."""
        assert "Open a project first." in actions_js

    def test_a_missing_dataset_is_refused_locally(self, actions_js) -> None:
        assert "Select a dataset first." in actions_js

    def test_being_disconnected_is_its_own_answer(self, actions_js) -> None:
        assert "Not connected to the application." in actions_js


class TestFailuresAreShownNotSwallowed:
    def test_an_api_refusal_reaches_the_user(self, shell_js) -> None:
        """A refusal from the Api is the most useful thing this application produces,
        and it used to vanish into a rejected promise inside a click handler."""
        assert "catch (error)" in shell_js
        assert 'this.toast(`${action}: ${error.message || error}`, "error")' in shell_js

    def test_messages_appear_over_the_canvas(self, shell_js) -> None:
        assert 'el("div", { class: "toasts" })' in shell_js


class TestEveryModuleActuallyParses:
    """The entry point is the one module nothing was checking.

    render_check.mjs mounts workspaces through the dock, so it imports dock.js,
    primitives.js and workspaces.js -- but never shell.js, which is what the page
    actually loads. A syntax error there passed every test and produced a blank window
    with no message in any log, because a module that fails to parse simply never runs.

    An escaped newline written literally into a template string was enough to do it.
    """

    @pytest.mark.parametrize(
        "module",
        ["api.js", "actions.js", "demo.js", "demo-data.js", "dock.js",
         "primitives.js", "shell.js", "workspaces.js", "mapview.js"],
    )
    def test_it_imports(self, module: str) -> None:
        import shutil
        import subprocess

        if shutil.which("node") is None:
            pytest.skip("node is not installed")
        path = (WORKSPACE_JS / module).as_posix()
        result = subprocess.run(
            ["node", "--input-type=module", "-e",
             f"import('file:///{path}').then(()=>process.exit(0))"
             f".catch(e=>{{console.error(e.message);process.exit(1)}})"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"{module} does not load: {result.stderr.strip()}"
