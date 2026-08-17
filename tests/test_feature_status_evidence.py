"""Status is computed from passing tests, so what counts as "passing" decides everything.

Two things were wrong with how that was read, and both inflated the answer.

A junit case was treated as broken only when it carried <failure> or <error>, which put
every SKIPPED test into the passed set. This repository skips a great deal on purpose --
gitignored weights, PostGIS that is not running, SITL that needs a container -- and each
of those skips was crediting a row it had proved nothing about.

The other side of the same problem: fl.sitl's tests can only pass inside a container, so
once skips stop counting there is no machine on which that row can be earned. The fix is
not to trust a claim but to merge the container's own junit report.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "features"))

from feature_status import classify, evaluate, read_report  # noqa: E402
from registry import F  # noqa: E402


def _case(xml: str):
    return ET.fromstring(xml)


class TestASkipIsNotAPass:
    def test_a_plain_case_passed(self) -> None:
        assert classify(_case('<testcase name="t"/>')) == "passed"

    def test_a_failure_failed(self) -> None:
        assert classify(_case('<testcase name="t"><failure/></testcase>')) == "failed"

    def test_an_error_failed(self) -> None:
        assert classify(_case('<testcase name="t"><error/></testcase>')) == "failed"

    def test_a_skip_is_its_own_answer(self) -> None:
        """The defect. This used to read as "passed" and credit the feature."""
        assert classify(_case('<testcase name="t"><skipped/></testcase>')) == "skipped"


class TestSkippedEvidenceCannotVerifyARow:
    @staticmethod
    def _feature(claimed: str) -> Feature:
        return F("fl.sitl", "SITL", "core", "Flight", "criteria", claimed,
                 ["tests/sitl/test_flight_lifecycle.py"])

    def test_a_row_whose_tests_all_skipped_is_not_verified(self) -> None:
        """A skip puts the node in neither set, so the selector matches nothing.

        Nothing ran, so nothing is proved, and the row falls back to the claim.
        """
        status, reason = evaluate(self._feature("implemented"), set(), set())
        assert status == "implemented"
        assert "matched nothing" in reason

    def test_the_same_row_is_verified_when_the_tests_really_ran(self) -> None:
        passed = {"tests/sitl/test_flight_lifecycle.py::TestArm::test_arms"}
        status, _ = evaluate(self._feature("implemented"), passed, set())
        assert status == "verified"

    def test_a_failure_beats_a_pass_of_the_same_selector(self) -> None:
        # Merged evidence must never be able to hide a local failure.
        passed = {"tests/sitl/test_flight_lifecycle.py::TestArm::test_arms"}
        failed = {"tests/sitl/test_flight_lifecycle.py::TestArm::test_rtl"}
        status, _ = evaluate(self._feature("implemented"), passed, failed)
        assert status == "in_progress"


class TestEvidenceFromAnotherMachine:
    """fl.sitl cannot be earned where ArduPilot is not; the container's report is it."""

    @staticmethod
    def _report(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "sitl-report.xml"
        path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>",
                        encoding="utf-8")
        return path

    def test_a_passing_case_is_read_as_a_node_id(self, tmp_path) -> None:
        report = self._report(
            tmp_path,
            '<testcase classname="tests.sitl.test_mission_upload.TestUpload" name="test_home"/>',
        )
        passed, failed = read_report(report)
        assert passed == {"tests/sitl/test_mission_upload.py::TestUpload::test_home"}
        assert not failed

    def test_a_failing_case_is_carried_across_too(self, tmp_path) -> None:
        """An outside report can promote a row only by passing, never by being quiet."""
        report = self._report(
            tmp_path,
            '<testcase classname="tests.sitl.test_mission_upload.TestUpload" '
            'name="test_home"><failure/></testcase>',
        )
        passed, failed = read_report(report)
        assert failed == {"tests/sitl/test_mission_upload.py::TestUpload::test_home"}
        assert not passed

    def test_a_skip_in_the_merged_report_still_counts_for_nothing(self, tmp_path) -> None:
        """The container skipping is the exact case CI must not read as success."""
        report = self._report(
            tmp_path,
            '<testcase classname="tests.sitl.test_mission_upload.TestUpload" '
            'name="test_home"><skipped/></testcase>',
        )
        passed, failed = read_report(report)
        assert not passed and not failed

    def test_merged_evidence_verifies_the_row_that_needs_it(self, tmp_path) -> None:
        """End to end: the container's report is what moves fl.sitl."""
        report = self._report(
            tmp_path,
            '<testcase classname="tests.sitl.test_flight_lifecycle.TestArm" name="test_arms"/>',
        )
        passed, failed = read_report(report)
        feature = F("fl.sitl", "SITL", "core", "Flight", "criteria", "implemented",
                    ["tests/sitl/test_flight_lifecycle.py"])
        assert evaluate(feature, passed, failed)[0] == "verified"


class TestNodeIdsSurviveNestedDirectories:
    """A selector can only match a node id that is spelled the way pytest spells it.

    junit reports a dotted classname with nothing marking where directories end and the
    module begins. Taking a fixed two segments works for tests/test_x.py and turns
    tests/sitl/test_x.py into `tests/sitl.py::test_x::...` -- a path that exists nowhere,
    so the selector matched nothing and the feature could never be earned. tests/sitl is
    the only nested suite here, and it is the one feature that most needed to be.
    """

    @staticmethod
    def _node(classname: str, name: str) -> str:
        from feature_status import node_id

        return node_id(_case(f'<testcase classname="{classname}" name="{name}"/>'))

    def test_a_top_level_test_is_unchanged(self) -> None:
        assert self._node("tests.test_geo.TestUmeyama", "test_fits") == (
            "tests/test_geo.py::TestUmeyama::test_fits"
        )

    def test_a_nested_test_keeps_its_directory(self) -> None:
        assert self._node("tests.sitl.test_mission_upload.TestUpload", "test_home") == (
            "tests/sitl/test_mission_upload.py::TestUpload::test_home"
        )

    def test_a_module_level_function_has_no_class_segment(self) -> None:
        assert self._node("tests.sitl.test_mission_upload", "test_home") == (
            "tests/sitl/test_mission_upload.py::test_home"
        )

    def test_the_nested_id_is_what_the_registry_selector_matches(self) -> None:
        """The end of the chain: this is why fl.sitl can now be earned."""
        from feature_status import _matches

        node = self._node("tests.sitl.test_flight_lifecycle.TestArm", "test_arms")
        assert _matches("tests/sitl/test_flight_lifecycle.py", {node})
