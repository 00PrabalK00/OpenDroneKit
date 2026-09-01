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


class TestWhatCannotBeReachedSaysWhy:
    """Three capabilities need the interactive 3D viewer, which is not built.

    They are declared rather than wired. A button that always answers "no camera pose
    yet" would be the same dead control this audit exists to remove -- but silently
    dropping them would lose the fact that the capability exists and is tested.
    """

    DECLARED = ["Save View", "Clip", "Temperature Range"]

    @pytest.mark.parametrize("button", DECLARED)
    def test_it_names_the_missing_capability(self, actions, button) -> None:
        body = action_body(actions, button)
        assert "skipped" in body, f"{button!r} does not decline honestly"
        assert "implemented" in body, (
            f"{button!r} does not say the capability exists, only that it did nothing"
        )

    @pytest.mark.parametrize("button", DECLARED)
    def test_the_api_really_does_implement_it(self, button) -> None:
        """The declaration has to be true in both directions: the button says the
        capability is implemented, so it had better be."""
        api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
        method = {"Save View": "save_view", "Clip": "add_plane_clip",
                  "Temperature Range": "scale_thermal"}[button]
        assert f"def {method}(" in api


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
        block = shell.split("actionContext() {")[1]
        provided = set(re.findall(r"^      (\w+):", block[:6000], re.M))
        assert "prompt" in provided, "the action context could not be parsed"
        used = set(re.findall(r"ctx\.(\w+)\(", actions))
        missing = sorted(used - provided)
        assert not missing, f"actions call context helpers the shell does not provide: {missing}"
