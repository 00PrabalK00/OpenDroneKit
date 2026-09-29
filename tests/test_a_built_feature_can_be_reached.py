"""Reachable by a person, not reachable from Python.

Seventy-two of a hundred and forty-nine Api methods are never called from anywhere in the
interface. Most of that number is fine: helpers, alternate entry points, methods the REST
service uses, things one call wraps another to provide.

What was not fine is that the list included most of what this project had recently built
and recorded as done -- annotation tags, site markers, hazard clearance, CAD overlays,
saved views, model clipping, thermal scaling. Each had an Api method, a core module and
passing tests. None had a control anywhere that reached it.

The sharpest part of the finding is a test file written during this same audit, called
`test_special_missions_are_reachable.py`. What it checks is that `api.plan_pylon_mission(...)`
answers. That is reachable from Python. It is not reachable by a person, which is the
sense its name claims and the only sense that matters to somebody using the software.

So this file separates the two questions and keeps them separate:

    can the Api do it?          the feature's own tests answer that
    can a person get there?     a toolbar button must resolve to an action that calls it

A capability with neither is missing. A capability with the first only is what this
codebase keeps mistaking for finished.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "app" / "web" / "js" / "workspace"


@pytest.fixture(scope="module")
def actions() -> str:
    return (JS / "actions.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workspaces() -> str:
    return (JS / "workspaces.js").read_text(encoding="utf-8")


def toolbar_labels(workspaces: str) -> set[str]:
    """Every label a toolbar offers.

    Excluding `|` from the character class breaks quote pairing: the separator "|" is
    skipped, the quotes either side of it pair with the wrong neighbours, and the
    captured "labels" come out as the commas and newlines BETWEEN the real ones. Match
    every quoted string, then drop the separators.
    """
    labels: set[str] = set()
    for row in re.findall(r"toolbar: \[(.*?)\],", workspaces, re.S):
        for label in re.findall(r'"([^"]*)"', row):
            if label.strip() and label != "|":
                labels.add(label)
    return labels


def action_body(actions: str, button: str) -> str:
    """The source of one action. Keys with a space are quoted; single words are not."""
    anchor = f'"{button}": {{' if " " in button else f"\n  {button}: {{"
    assert anchor in actions, f"no action entry for {button!r}"
    return actions.split(anchor)[1][:2400]


def action_names(actions: str) -> set[str]:
    body = actions.split("export const ACTIONS = {")[1]
    return set(re.findall(r'^  "([^"]+)":', body, re.M)) | set(
        re.findall(r"^  (\w+): \{", body, re.M))


class TestEveryButtonResolves:
    def test_no_action_name_is_defined_twice(self, actions) -> None:
        """ACTIONS is an object literal, so a repeated key silently REPLACES the first.

        Adding a mission-version diff called `Compare` overwrote the survey-comparison
        verb that Thermal's Compare button and the "Compare Dates" alias both resolve
        to. Two working buttons would have started doing something else, with no error
        raised anywhere and nothing on screen to say so.

        This is the same collision class as the Processing screen's Pause resolving to
        the aircraft's -- reintroduced, during the audit of it, by me.
        """
        body = actions.split("export const ACTIONS = {")[1]
        names = [a or b for a, b in re.findall(r'^  "([^"]+)":|^  (\w+):', body, re.M)]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"action names defined more than once: {duplicates}"

    def test_no_toolbar_offers_a_button_with_no_action(self, actions, workspaces) -> None:
        """A label with no entry falls through to "not wired", which looks like a bug
        in the application rather than a feature that does not exist."""
        missing = sorted(toolbar_labels(workspaces) - action_names(actions))
        assert not missing, f"toolbar labels with no action: {missing}"


class TestTheFeaturesBuiltHereAreReachable:
    """One case per capability that was recorded as built and could not be pressed.

    Each names the Api method and the button that has to arrive at it. If a button is
    renamed or removed, this fails and says which capability went dark.
    """

    REACHABLE = [
        ("tag_annotations", "Tag"),
        ("add_site_marker", "Add Marker"),
        ("marker_kinds", "Add Marker"),
        ("check_hazards", "Check Hazards"),
        ("import_cad_overlay", "Import CAD"),
        ("mark_all_notifications_read", "Clear Alerts"),
        # Thirteen rows the registry marked verified while nothing in the interface
        # could reach them. Each named its Api method in its own note, and each had
        # tests that passed at the Api and proved nothing about the software being
        # usable.
        ("repeat_mission", "Repeat Survey"),
        ("import_boundary", "Import Boundary"),
        ("list_cameras", "Cameras"),
        ("describe_camera", "Cameras"),
        ("list_payloads", "Payloads"),
        ("describe_payload", "Payloads"),
        ("cache_terrain", "Cache Terrain"),
        ("linked_mission_progress", "Linked Progress"),
        ("export_flight_log", "Export Log"),
        ("check_ppk_inputs", "Check PPK"),
        ("size_reconstruction_job", "Size Job"),
        ("mission_version_history", "History"),
        ("restore_mission_version", "History"),
        # Fly to draw: the pilot flies to each corner of the site and presses a button.
        # It is the one case where a boundary comes from where the aircraft actually is
        # rather than from a map drawn at a desk, which is the point of it for sites
        # whose extent is on no map. It needed a connected aircraft and nothing else.
        ("mark_boundary_corner", "Mark Corner"),
        ("boundary_from_marks", "Close Boundary"),
        ("clear_boundary_marks", "Clear Corners"),
        # The Digital Twin's 3D view (viewer3d.js) gave these a camera and a surface to
        # pick on; tests/test_cockpit_3d_viewer.py drives them. Measure in 3D also
        # offers the volume of the closed model, which measure_in_model computes.
        ("save_view", "Save View"),
        ("open_view", "Open View"),
        ("list_views", "Open View"),
        ("measure_in_model", "Measure in 3D"),
        ("import_gcps", "Add GCPs"),
        ("mark_gcp", "Mark GCP"),
        ("gcp_accuracy_report", "GCP Report"),
        ("diff_mission_versions", "Compare Versions"),
        # Declared "needs the 3D viewer" until the Digital Twin had one (viewer3d.js).
        # They act on the model on screen; tests/test_cockpit_3d_viewer.py drives them.
        ("save_view", "Save View"),
        ("open_view", "Save View"),
        ("set_default_view", "Save View"),
        ("set_clip_visible", "Clip"),
        ("remove_clip", "Clip"),
        ("export_clipped_model", "Clip"),
        # Declared "no thermal product is selectable" until it took a radiometric frame
        # from disk; tests/test_cockpit_thermal_range.py drives it.
        ("scale_thermal", "Temperature Range"),
    ]

    @pytest.mark.parametrize("method,button", REACHABLE, ids=[m for m, _ in REACHABLE])
    def test_the_button_exists_and_calls_the_method(
        self, actions, workspaces, method, button
    ) -> None:
        assert button in toolbar_labels(workspaces), f"no toolbar offers {button!r}"
        assert button in action_names(actions), f"{button!r} has no action"
        assert f'"{method}"' in action_body(actions, button), (
            f"{button!r} does not call {method}()"
        )


def test_measure_in_3d_reaches_the_model_measurement() -> None:
    """Measure in 3D opens the model with the measure tool armed; the viewer's picks go
    to measure_on_model, which the Api implements."""
    viewer = (ROOT / "app" / "web" / "js" / "workspace" / "viewer3d.js").read_text(encoding="utf-8")
    api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    assert 'call("measure_on_model"' in viewer
    assert "def measure_on_model(" in api
    # Clip's "New cut" arms the viewer, whose picked point becomes the plane.
    assert 'call("add_plane_clip"' in viewer
    assert "def add_plane_clip(" in api


class TestTheContextHelpersExist:
    """Two of the first-draft buttons called ctx.cameraPose() and ctx.lastClickedPoint().

    Neither is on the shell's action context, so both would have thrown at the moment of
    use. An action may only use helpers the shell provides.
    """

    def test_every_ctx_helper_used_is_provided(self, actions) -> None:
        shell = (JS / "shell.js").read_text(encoding="utf-8")
        # "actionContext()" appears at its call site before its definition, so splitting
        # on the first occurrence returned 800 characters of an unrelated method and
        # reported every helper as missing -- including ones plainly in use.
        # The whole method, to its closing brace: a fixed 6000-character window stopped
        # reading half way through once the context grew.
        block = shell.split("actionContext() {")[1].split("\n  }\n")[0]
        provided = set(re.findall(r"^      (\w+):", block, re.M))
        assert "prompt" in provided, "the action context could not be parsed"
        used = set(re.findall(r"ctx\.(\w+)\(", actions))
        missing = sorted(used - provided)
        assert not missing, f"actions call context helpers the shell does not provide: {missing}"
