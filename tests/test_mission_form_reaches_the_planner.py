"""Pressing Plan must produce the mission the operator set up.

The mission form's field keys are short -- type, alt, fwd, side, standoff. plan_mission
reads template, altitude_m, front_overlap_pct, side_overlap_pct, standoff_m. Not one of
them matched, so opts.get(...) found nothing every time and fell back to its defaults.

Measured before the fix, through the real Api:

    {"type": "Facade inspection", "alt": 40, "standoff": 12}  ->  template=grid, alt=55.0
    {"template": "facade", "altitude_m": 40}                  ->  template=facade, alt=40.0

An operator setting up a facade inspection at 40 m got a nadir grid at 55 m, with nothing
on screen to say so. That is the worst shape a bug can take in a planner: not a crash, not
a refusal, but a plausible mission that is not the one that was asked for.

These tests cover both halves -- the translation table in the shell, and the API contract
it translates into -- because either drifting alone reproduces the bug.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from app.api import Api
from app.session import AppSession

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL = REPO_ROOT / "app" / "web" / "js" / "workspace" / "shell.js"
WORKSPACES = REPO_ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"

AOI = [[-81.7530, 41.3035], [-81.7500, 41.3035], [-81.7500, 41.3055],
       [-81.7530, 41.3055], [-81.7530, 41.3035]]


@pytest.fixture(scope="module")
def shell_js() -> str:
    return SHELL.read_text(encoding="utf-8")


def _map(source: str, name: str) -> dict[str, str]:
    body = source.split(f"const {name} = {{")[1].split("};")[0]
    return {
        m.group(1) or m.group(2): m.group(3)
        for m in re.finditer(r'(?:"([^"]+)"|(\w+)):\s*"([^"]+)"', body)
    }


class TestTheFormKeysAreTranslated:
    def test_every_form_field_maps_to_something_the_planner_reads(self, shell_js) -> None:
        """The bug itself: nine fields, none of which the API had ever heard of."""
        mapping = _map(shell_js, "MISSION_OPTION_KEYS")
        api_source = (REPO_ROOT / "app" / "api.py").read_text(encoding="utf-8")
        plan_body = api_source.split("def plan_mission")[1].split("plan = MissionPlanner")[0]
        read = set(re.findall(r'opts\.get\("([a-z_0-9]+)"', plan_body))

        unread = {form: api for form, api in mapping.items() if api not in read}
        assert not unread, f"these translate to names plan_mission never reads: {unread}"

    def test_the_fields_the_form_actually_shows_are_covered(self, shell_js) -> None:
        """A field on screen that the planner ignores is worse than one that is absent:
        the operator sets it, watches it change, and it does nothing."""
        panel = WORKSPACES.read_text(encoding="utf-8")
        block = panel.split('{ title: "Mission", render: () => fields([')[1].split("])")[0]
        shown = set(re.findall(r'key:\s*"(\w+)"', block))
        mapping = _map(shell_js, "MISSION_OPTION_KEYS")
        # `type` goes through the template table. The rest must be either mapped or
        # named in MISSION_FIELDS_NOT_PLANNED -- a field that is silently inert is the
        # bug; a field that is declared inert is a known gap.
        inert = set(re.findall(r'"(\w+)"',
                               shell_js.split("MISSION_FIELDS_NOT_PLANNED = new Set([")[1]
                               .split("]")[0]))
        unmapped = shown - set(mapping) - inert - {"type"}
        assert not unmapped, f"fields shown, not sent, and not declared inert: {sorted(unmapped)}"

    def test_altitude_is_mapped(self, shell_js) -> None:
        assert _map(shell_js, "MISSION_OPTION_KEYS")["alt"] == "altitude_m"


class TestTheMissionTypeLabelsResolve:
    def test_every_label_in_the_list_has_a_template(self, shell_js) -> None:
        """The list is written for a pilot; the planner takes ids. A label with no entry
        is sent through verbatim, and _normalize_template falls back to grid."""
        panel = WORKSPACES.read_text(encoding="utf-8")
        block = panel.split("const missionTypes = [")[1].split("];")[0]
        labels = set(re.findall(r'"([^"]+)"', block))
        mapped = set(_map(shell_js, "MISSION_TEMPLATES"))
        assert not labels - mapped, f"labels with no template: {sorted(labels - mapped)}"

    def test_every_template_it_maps_to_is_one_the_planner_accepts(self, shell_js) -> None:
        from mission.planner import available_templates

        known = set(available_templates())
        targets = set(_map(shell_js, "MISSION_TEMPLATES").values())
        assert not targets - known, f"unknown templates: {sorted(targets - known)}"

    def test_facade_inspection_is_not_a_grid(self, shell_js) -> None:
        """The specific case that was measured wrong."""
        assert _map(shell_js, "MISSION_TEMPLATES")["Facade inspection"] == "facade"


class TestThePlannerHonoursWhatItIsGiven:
    @pytest.fixture
    def api(self) -> Api:
        api = Api(AppSession())
        api.set_aoi(AOI)
        return api

    def test_the_template_asked_for_is_the_one_planned(self, api) -> None:
        for requested in ("facade", "roof_inspection", "double_grid", "orbit"):
            api.plan_mission({"template": requested, "altitude_m": 40, "standoff_m": 12})
            recipe = api._session.mission_plan_dict.get("flight_recipe") or {}
            assert recipe.get("template") == requested, (
                f"asked for {requested}, planned {recipe.get('template')}"
            )

    def test_the_altitude_asked_for_is_the_one_planned(self, api) -> None:
        """55.0 was the default that appeared when altitude_m never arrived."""
        api.plan_mission({"template": "grid", "altitude_m": 42})
        assert api._session.mission_plan_dict["altitude_m"] == pytest.approx(42.0)

    def test_an_unrecognised_option_does_not_silently_become_a_default(self, api) -> None:
        """Documents the behaviour the translation exists to avoid: a key the planner
        does not read is not an error, it is a default. That is why the mapping has to be
        right rather than approximately right."""
        api.plan_mission({"template": "grid", "alt": 42})
        assert api._session.mission_plan_dict["altitude_m"] != pytest.approx(42.0)


class TestTheProcessingPanelReachesTheJob:
    """Same failure, second panel.

    reconstructionOptions() returned a hardcoded {engine: "auto", profile: "standard"},
    so the Processing Settings panel was read by nobody. An operator selecting the
    high-accuracy profile, or turning dense off to get a result before lunch, got a
    standard run with dense at its default either way.
    """

    def test_the_panel_settings_are_read(self, shell_js) -> None:
        body = shell_js.split("reconstructionOptions: () => {")[1].split("\n      },")[0]
        assert "raw.profile" in body, "the profile selection never leaves the browser"
        assert "raw.dense" in body, "the dense selection never leaves the browser"

    def test_it_is_no_longer_a_fixed_object(self, shell_js) -> None:
        assert 'reconstructionOptions: () => ({ engine: "auto", profile: "standard" })' \
            not in shell_js

    def test_dense_no_becomes_false_not_a_truthy_string(self, shell_js) -> None:
        """The trap: "no" is truthy in JavaScript. Passing the select's value straight
        through would turn dense ON when the operator asked for it off."""
        body = shell_js.split("reconstructionOptions: () => {")[1].split("\n      },")[0]
        assert 'toLowerCase() === "yes"' in body

    @pytest.mark.parametrize("label,expected", [
        ("fast", "fast_preview"),
        ("standard", "standard"),
        ("high", "inspection_high_accuracy"),
    ])
    def test_every_profile_the_panel_offers_resolves(self, label, expected) -> None:
        """The panel is labelled for an operator and the engine takes ids. A label with
        no alias falls back to standard, which is this bug wearing a different hat."""
        from core.reconstruction import _normalize_profile

        assert _normalize_profile(label) == expected

    def test_the_options_the_job_reads_are_the_ones_sent(self, shell_js) -> None:
        api_source = (REPO_ROOT / "app" / "api.py").read_text(encoding="utf-8")
        body = api_source.split("def run_reconstruction")[1][:1500]
        read = set(re.findall(r'opts\.get\("([a-z_0-9]+)"', body))
        sent = set(re.findall(r'options\.(\w+) =',
                              shell_js.split("reconstructionOptions: () => {")[1]
                              .split("\n      },")[0]))
        assert sent <= read | {"engine", "profile", "dense"}, (
            f"sending options run_reconstruction does not read: {sorted(sent - read)}"
        )

    def test_the_unused_panel_fields_are_declared(self, shell_js) -> None:
        """imgsize and mesh have no job option. Naming them makes the gap visible rather
        than leaving two controls that quietly do nothing."""
        assert "PROCESSING_FIELDS_NOT_USED" in shell_js
        declared = set(re.findall(r'"(\w+)"',
                                  shell_js.split("PROCESSING_FIELDS_NOT_USED = new Set([")[1]
                                  .split("]")[0]))
        assert {"imgsize", "mesh"} <= declared


class TestTheReportPanelReachesTheEngine:
    """Third panel, same fault.

    Generate Report called generate_report("", title, "standard", "") with the type and
    author hardcoded, so Report, Format, Organisation and Logo were read by nobody. The
    panel also offered four templates -- Inspection, Survey, Progress, Thermal -- that the
    engine has never built, so picking one got the standard report regardless.
    """

    @pytest.fixture(scope="class")
    def actions_js(self) -> str:
        return (REPO_ROOT / "app" / "web" / "js" / "workspace" / "actions.js").read_text(
            encoding="utf-8")

    def test_every_report_type_the_panel_offers_is_one_the_engine_builds(self) -> None:
        """Offering a choice that does not exist is worse than offering none, because the
        operator believes they made one."""
        panel = WORKSPACES.read_text(encoding="utf-8")
        block = panel.split('{ key: "tpl"')[1].split("},")[0]
        offered = set(re.findall(r'"(\w+)"', block.split("options:")[1]))

        engine = (REPO_ROOT / "core" / "report_engine.py").read_text(encoding="utf-8")
        declared = engine.split("report_type: str = \"standard\"")[1].split("\n")[0]
        known = set(re.findall(r"(\w+)", declared)) - {"standard"} | {"standard"}
        assert offered <= known, f"panel offers types the engine cannot build: {offered - known}"

    def test_the_action_no_longer_hardcodes_the_type(self, actions_js) -> None:
        assert 'call("generate_report", "", title, "standard", "")' not in actions_js

    def test_it_reads_the_panel(self, actions_js) -> None:
        body = actions_js.split('"Generate Report": {')[1].split("\n  },")[0]
        assert "reportOptions" in body

    def test_a_word_request_does_not_go_to_the_pdf_engine(self, actions_js) -> None:
        """The report engine renders HTML then PDF. Sending it a Word request would
        produce a PDF named correctly -- wrong in the one way nobody checks."""
        body = actions_js.split('"Generate Report": {')[1].split("\n  },")[0]
        assert 'export_report' in body
        assert 'docx' in body

    def test_the_shell_exposes_the_panel_settings(self, shell_js) -> None:
        assert "reportOptions:" in shell_js
        body = shell_js.split("reportOptions: () => {")[1].split("\n      },")[0]
        for field in ("tpl", "fmt", "org"):
            assert f"raw.{field}" in body, f"{field} is still discarded"


class TestAButtonMeansTheSameThingEverywhere:
    """One label cannot carry two verbs.

    ACTIONS is keyed by the button's label, so a workspace that reuses a word gets the
    other workspace's behaviour. Three found:

      Settings / Save      resolved to the MISSION verb and prompted "Mission name"
      Processing / Pause   resolves to the VEHICLE verb and prompts "Pause the aircraft?"
                           -- a flight command from the processing screen, with no
                           pause_job in the Api behind it in the other direction either
      AI Inspection / Validate  resolves to verify_site, the mission-planning meaning,
                           rather than moving the selected finding's status

    The vehicle one is the reason this class exists rather than being three separate
    fixes: a control that commands an aircraft must not be reachable from a screen about
    something else.
    """

    @pytest.fixture(scope="class")
    def actions_js(self) -> str:
        return (REPO_ROOT / "app" / "web" / "js" / "workspace" / "actions.js").read_text(
            encoding="utf-8")

    @pytest.fixture(scope="class")
    def toolbars(self) -> dict[str, list[str]]:
        panel = WORKSPACES.read_text(encoding="utf-8")
        found = {}
        for match in re.finditer(
            r'id: "(\w+)",\s*\n\s*title: "([^"]+)",\s*\n(?:\s*/\*.*?\*/\s*\n)?\s*toolbar: \[([^\]]+)\]',
            panel, re.S,
        ):
            labels = [x.strip().strip('"') for x in match.group(3).split(",")]
            found[match.group(2)] = [x for x in labels if x and x != "|"]
        return found

    def _vehicle_actions(self, actions_js: str) -> set[str]:
        return set(re.findall(r'^  "?([\w ]+?)"?: \{ vehicle:', actions_js, re.M))

    def test_only_the_flight_screen_can_command_the_aircraft(self, toolbars, actions_js) -> None:
        """The one that matters. A vehicle command reachable from Processing is a flight
        command one misclick from someone watching a reconstruction."""
        vehicle = self._vehicle_actions(actions_js)
        assert vehicle, "no vehicle actions found; the check would pass vacuously"

        offenders = {
            screen: sorted(set(labels) & vehicle)
            for screen, labels in toolbars.items()
            if screen != "Flight" and set(labels) & vehicle
        }
        assert not offenders, f"aircraft commands on non-flight screens: {offenders}"

    def test_processing_no_longer_offers_pause(self, toolbars) -> None:
        assert "Pause" not in toolbars["Processing"]

    def test_ai_inspection_moves_a_finding_rather_than_checking_the_site(
        self, toolbars, actions_js
    ) -> None:
        assert "Validate" not in toolbars["AI Inspection"]
        assert "Accept" in toolbars["AI Inspection"]
        body = actions_js.split("  Accept: {")[1].split("\n  },")[0]
        assert "review_finding" in body

    def test_validate_still_belongs_to_mission_planning(self, toolbars) -> None:
        """Removing the collision must not remove the verb from where it does mean
        something."""
        assert "Validate" in toolbars["Mission Planning"]

    def test_every_toolbar_label_resolves_to_an_action(self, toolbars, actions_js) -> None:
        """A label with no entry is a button that does nothing at all."""
        declared = set(re.findall(r'^  "?([\w &\u2019-]+?)"?: \{', actions_js, re.M))
        missing = {
            screen: sorted(set(labels) - declared)
            for screen, labels in toolbars.items()
            if set(labels) - declared
        }
        assert not missing, f"toolbar labels with no action: {missing}"
