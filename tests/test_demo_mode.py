"""Demo mode must be impossible to mistake for a measurement.

This is the most dangerous feature in the project: everything it produces has the shape
of a survey result, and the whole codebase otherwise exists to avoid emitting exactly
that without evidence behind it.

So these tests are not about whether demo mode produces plausible output. They are about
whether it can ever be mistaken for real, and whether the rest of the system will refuse
it when it tries to reach somewhere only measurements belong.
"""

from __future__ import annotations

import pytest

from core.demo_mode import (
    DEMO_LAT,
    DEMO_LON,
    DEMO_MARKER,
    DemoDataRefused,
    demo_project,
    is_demo_artifact,
    refuse_if_demo,
    stamp,
)


class TestMarking:
    def test_every_artefact_is_marked(self) -> None:
        project = demo_project()
        assert project[DEMO_MARKER] is True
        assert "SYNTHETIC" in project["provenance"]

    def test_the_mark_survives_being_pulled_apart(self) -> None:
        # The real risk is not someone reading the whole report. It is someone lifting
        # one finding out of it and passing that along.
        project = demo_project()
        for finding in project["findings"]:
            assert finding[DEMO_MARKER] is True, "a lifted finding must still declare itself"
            assert "SYNTHETIC" in finding["provenance"]

    def test_nested_structures_are_marked_too(self) -> None:
        marked = stamp({"outer": {"inner": {"value": 1}}})
        assert marked["outer"][DEMO_MARKER] is True
        assert marked["outer"]["inner"][DEMO_MARKER] is True

    def test_lists_of_dicts_are_marked(self) -> None:
        marked = stamp({"items": [{"a": 1}, {"b": 2}]})
        assert all(item[DEMO_MARKER] is True for item in marked["items"])


class TestImplausibility:
    def test_the_site_is_null_island(self) -> None:
        # A reader who misses the flag entirely still cannot mistake 0,0 for a survey.
        project = demo_project()
        assert project["site"]["lat"] == DEMO_LAT == 0.0
        assert project["site"]["lon"] == DEMO_LON == 0.0

    def test_the_timestamp_is_the_epoch_not_now(self) -> None:
        project = demo_project()
        assert project["captured_at"].startswith("1970-01-01")

    def test_output_is_identical_between_runs(self) -> None:
        # Deterministic output cannot be mistaken for a survey that happened today, and
        # makes demo screenshots reproducible.
        assert demo_project() == demo_project()

    def test_the_report_says_nothing_was_measured(self) -> None:
        project = demo_project()
        assert "Nothing was measured" in project["capability_note"]


class TestDetection:
    def test_a_demo_artefact_is_detected(self) -> None:
        assert is_demo_artifact(demo_project())

    def test_a_real_artefact_is_not(self) -> None:
        assert not is_demo_artifact({"class": "crack", "confidence": 0.87})

    def test_one_synthetic_fragment_taints_the_whole(self) -> None:
        # Deliberately conservative. A report that is "mostly real" is not one anyone
        # can act on, so any synthetic fragment makes the whole structure synthetic.
        mixed = {"real": {"value": 1}, "borrowed": stamp({"value": 2})}
        assert is_demo_artifact(mixed)

    def test_detection_reaches_into_lists(self) -> None:
        assert is_demo_artifact({"findings": [{"ok": 1}, stamp({"bad": 2})]})


class TestRefusal:
    def test_publishing_demo_data_is_refused(self) -> None:
        with pytest.raises(DemoDataRefused, match="not a measurement"):
            refuse_if_demo(demo_project(), action="publish this report")

    def test_real_data_passes_the_guard(self) -> None:
        refuse_if_demo({"class": "crack", "confidence": 0.87}, action="publish")

    def test_the_refusal_names_the_action(self) -> None:
        with pytest.raises(DemoDataRefused, match="register a model"):
            refuse_if_demo(demo_project(), action="register a model")
