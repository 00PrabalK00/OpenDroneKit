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

    def test_nothing_is_declared_unavailable_any_more(self, actions_js) -> None:
        """UNWIRED held twenty-three entries: fleet, sharing, webhooks, reports, review,
        plugins. Every one was implemented and carried a verified registry row -- the
        only thing missing was a path from the button to the code, because those
        capabilities lived behind the web service.

        The map is kept and empty on purpose. The next capability that genuinely does
        not exist should be declared here rather than failing silently.
        """
        unwired_block = actions_js[actions_js.index("export const UNWIRED"):]
        entries = re.findall(r':\s*"([^"]{20,})"', unwired_block[:400])
        assert not entries, f"still declared unavailable: {entries}"

    def test_the_previously_unavailable_buttons_call_the_api(self, actions_js) -> None:
        """The specific ones the user asked about."""
        for method in (
            "add_aircraft", "add_battery", "add_pilot", "log_maintenance",
            "create_share_link", "add_webhook", "generate_report",
            "review_finding", "list_plugins",
        ):
            assert f'call("{method}"' in actions_js, f"{method} is not called by any button"


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


class TestEveryButtonIsActuallyClicked:
    """Executed, not read.

    Every other test in this file greps actions.js for a method name. That would pass
    while the shell threw on the first click, and it DID pass while a syntax error in
    shell.js left the window blank -- text in a file is not evidence that a button works.

    click_every_button.mjs mounts the real Shell against a stub DOM and a fake pywebview
    bridge, then clicks all seventy-five toolbar actions across all fourteen workspaces
    and records what each one did: called the Api, changed the view, armed a tool, or
    declared itself unavailable. A button that throws, or that does nothing at all
    without saying so, fails the run.
    """

    @pytest.fixture(scope="class")
    def clicked(self):
        import shutil
        import subprocess

        if shutil.which("node") is None:
            pytest.skip("node is not installed")
        script = WORKSPACE_JS / "__tests__" / "click_every_button.mjs"
        return subprocess.run(
            ["node", str(script)], cwd=script.parent,
            capture_output=True, text=True, timeout=180,
        )

    def test_no_button_throws(self, clicked) -> None:
        assert "THREW" not in clicked.stdout, clicked.stdout
        assert clicked.returncode == 0, f"{clicked.stdout}\n{clicked.stderr}"

    def test_no_button_is_silent(self, clicked) -> None:
        """Doing nothing and saying nothing is the failure the user actually hit."""
        assert "SILENT" not in clicked.stdout, clicked.stdout

    def test_every_button_responded(self, clicked) -> None:
        assert "every button responded" in clicked.stdout, clicked.stdout

    def test_every_button_reaches_the_application(self, clicked) -> None:
        """Was 66 of 93 with 27 declared unavailable. The 27 are now wired to the same
        database the web service uses, so all of them do real work."""
        ok_line = [l for l in clicked.stdout.splitlines() if l.startswith("ok: ")]
        clicked_line = [l for l in clicked.stdout.splitlines() if l.startswith("buttons clicked: ")]
        assert ok_line and clicked_line, clicked.stdout
        assert int(ok_line[0].split(":")[1]) == int(clicked_line[0].split(":")[1])


class TestTheCanvasActuallyChanges:
    """The failure the user reported: 'the canvas is just not changing no matter what
    I click'.

    It was true twice over. The view buttons recorded a mode on the shell and set a data
    attribute nobody read, so the canvas never repainted. And once repainting was added,
    the prerequisite gate refused every action with "Not connected to the application"
    before the local handler ran -- so in a browser, and in the demo, no view button did
    anything at all.

    A view mode, a canvas tool and a workspace jump are client decisions that need no
    bridge. They are handled before the gate now, and this holds that order.
    """

    def test_local_actions_run_before_the_connection_gate(self, shell_js) -> None:
        local = shell_js.index("if (entry.view)")
        gate = shell_js.index("const blocked = prerequisite(")
        assert local < gate, (
            "the connection gate runs first, which disables every view button when "
            "there is no bridge"
        )

    def test_switching_view_repaints_the_canvas(self, shell_js) -> None:
        assert "setView(entry.view)" in shell_js
        assert "this.dock.render(WORKSPACE_BY_ID[this.workspaceId])" in shell_js

    def test_the_canvas_reads_the_current_view(self) -> None:
        primitives = (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")
        assert "currentView()" in primitives

    def test_a_view_with_no_product_says_so(self) -> None:
        """'Nothing here' and 'this has not been produced yet' look identical on a dark
        canvas, and only one of them tells the user what to do next."""
        primitives = (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")
        assert "No ${view} product yet" in primitives
        assert "run Process" in primitives

    def test_the_placeholder_is_removed_by_class_not_tag(self) -> None:
        """querySelectorAll("placeholder") is a TAG selector and matches nothing in a
        real browser, so the stale placeholder stayed under the new content. The stub DOM
        in the click harness strips the dot, which is why only a real browser caught it.
        """
        primitives = (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")
        assert 'querySelectorAll("placeholder")' not in primitives
        assert 'querySelectorAll(".placeholder")' in primitives

    def test_a_workspace_switch_returns_to_the_map(self, shell_js) -> None:
        """Otherwise a thermal view follows you into the next workspace and there is no
        button anywhere that puts the map back."""
        assert 'setView("map")' in shell_js


class TestThePlannedPathIsReal:
    """'in the demo there should be planned drone paths'.

    mapview.js has exported showMission since it was written and nothing ever called it,
    so every canvas was an empty basemap: the path was computed and thrown away.
    """

    def test_the_canvas_draws_the_mission(self) -> None:
        primitives = (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")
        assert "showMission(instance" in primitives

    def test_the_mission_was_planned_by_the_planner(self) -> None:
        """Not a zigzag drawn to look like a flight path."""
        builder = (REPO_ROOT / "tools" / "build_demo_data.py").read_text(encoding="utf-8")
        assert "MissionPlanner().generate(" in builder

    def test_the_footprint_comes_from_the_imagery(self) -> None:
        """The polygon is the survey's own GPS extent, so the plan is for a real place."""
        builder = (REPO_ROOT / "tools" / "build_demo_data.py").read_text(encoding="utf-8")
        assert "GPS EXIF" in builder

    def test_the_demo_carries_a_planned_path(self) -> None:
        import json

        text = (WORKSPACE_JS / "demo-data.js").read_text(encoding="utf-8")
        payload = json.loads(text[text.index("{"):text.rindex("}") + 1])
        mission = payload.get("mission") or {}
        assert mission.get("waypoint_count", 0) > 20, "no planned path in the demo data"
        assert len(mission.get("line") or []) > 20
        assert mission.get("distance_m", 0) > 0
        assert mission.get("gsd_cm", 0) > 0


class TestThePanelsDoSomething:
    """"The side panel is useless -- I see things there but none of them do anything."

    They were publishing correctly and nothing was listening usefully. Every tree and
    table already called selection.select(), and the only subscriber wrote an 11px label
    into the status bar, so clicking a job, a model or a finding looked exactly like
    clicking nothing. Publishing without a consequence is a button with no handler, one
    layer further in.
    """

    def test_selection_has_a_handler_not_just_a_label(self, shell_js) -> None:
        assert "onSelection(kind, value)" in shell_js
        assert 'selection.on("*", (kind, value) => this.onSelection(kind, value))' in shell_js

    def test_selecting_a_job_makes_cancel_act_on_it(self, shell_js) -> None:
        """Cancel used to need a job id the user could not see anywhere."""
        assert "this.selectedJobId = value.job || value.id" in shell_js

    def test_selecting_a_finding_makes_review_act_on_it(self, shell_js, actions_js) -> None:
        assert "this.selectedFindingId" in shell_js
        assert "ctx.selectedFinding()" in actions_js

    def test_selecting_an_aircraft_makes_maintenance_act_on_it(self, shell_js, actions_js) -> None:
        assert "this.selectedFleetId" in shell_js
        assert "ctx.selectedFleetId()" in actions_js

    def test_selecting_a_model_reports_what_it_measured(self, shell_js) -> None:
        """The registry knows the metric and the digest; the panel should say them."""
        assert 'case "model"' in shell_js
        assert "headline" in shell_js

    def test_selecting_a_project_opens_it(self, shell_js) -> None:
        """It used to answer that Open would act on it, which describes a button.

        A sentence about what some other control would do is not a consequence. Picking
        a project in the list is the operator saying which one they want.
        """
        assert "openProject(" in shell_js
        assert 'call("set_active_project"' in shell_js or "set_active_project" in shell_js
        # A different project has different datasets and layers, so the canvas must not
        # keep showing the previous one's picture.
        assert "async openProject(projectId, name)" in shell_js

    def test_selecting_a_layer_draws_it(self, shell_js) -> None:
        """The clearest case of a panel printing text back at the operator.

        A layer is the reconstruction's own output. The application holds the orthomosaic
        and used to answer by repeating its name.
        """
        assert "showLayer(" in shell_js
        assert "raster_preview" in shell_js
        # Vectors have no picture. Reporting the feature count and geometry types is a
        # real answer to "what is this"; repeating the name is not.
        assert "read_vector_layer" in shell_js
        assert "features.length" in shell_js

    def test_a_layer_that_cannot_be_drawn_says_why(self, shell_js) -> None:
        section = shell_js[shell_js.index("async showLayer("):]
        section = section[: section.index("\n  /**", 10)]
        assert "lastError.get" in section, (
            "falling back to the layer name is how this looked like it worked before"
        )


class TestEditingAFieldReachesThePlanner:
    """fields() has always accepted an onChange and no caller passed one.

    Editing an altitude updated the input element and nothing else, so Plan ran on the
    defaults -- the worst kind of broken, because the screen agreed with the user and the
    output did not.
    """

    def test_every_field_group_reports_its_edits(self) -> None:
        workspaces = (WORKSPACE_JS / "workspaces.js").read_text(encoding="utf-8")
        assert "const settingChanged" in workspaces
        assert workspaces.count("settingChanged)") >= 7

    def test_the_shell_keeps_what_was_typed(self, shell_js) -> None:
        assert 'case "setting"' in shell_js
        assert "this.settings[value.key] = value.value" in shell_js

    def test_the_planner_is_given_those_settings(self, shell_js) -> None:
        assert "missionOptions: () => {" in shell_js
        assert "this.settings" in shell_js

    def test_numbers_are_sent_as_numbers(self, shell_js) -> None:
        """A string altitude reaches Python and fails there, which reads as a planner
        bug rather than a form that never converted its input."""
        assert "Number.isFinite(asNumber)" in shell_js
