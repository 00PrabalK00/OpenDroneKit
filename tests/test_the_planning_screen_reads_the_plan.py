"""Estimates, segments, and a Validation panel that reported checks it never ran.

The planning screen showed six fixed estimates -- 14.2 ha, 3.1 km, 18 min, 214 images,
1 battery, 12.4 GB -- whether or not a mission had been planned. "Batteries 1" is the
one that sends someone to site with a single pack.

It showed four mission segments with capture counts for a mission that did not exist.

And the Validation panel reported three results:

    geofence contains every capture point                          ok
    no terrain model loaded: altitudes are above a flat plane      warn
    payload understands the planned trigger command                ok

Nothing had been validated. "Geofence contains every capture point" is a safety check
reported as **passed** without having run -- the preflight checklist's defect, on the
planning screen.

The middle line makes it worse rather than better. It was true by accident: no terrain
model usually was loaded. So two thirds of the panel was correct, which is exactly what
makes the remaining third credible.

Validation is an action with a cost and should not run on render. So the panel now
reports what the plan really contains -- the estimator's own warnings, and the terrain
state either way -- and names the button that runs the rest.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"


def strip_comments(js: str) -> str:
    out, i = [], 0
    while i < len(js):
        if js.startswith("/*", i):
            end = js.find("*/", i + 2)
            i = len(js) if end == -1 else end + 2
            continue
        out.append(js[i])
        i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def planning() -> str:
    source = WORKSPACES.read_text(encoding="utf-8")
    block = source.split("const planning = {")[1].split("\nconst flight")[0]
    return strip_comments(block)


FABRICATED = [
    ("14.2 ha", "an area for an unplanned mission"),
    ("12.4 GB", "a storage figure for an unplanned mission"),
    ("Nadir grid", "a mission segment that does not exist"),
    ("Oblique ring", "a mission segment that does not exist"),
    ("geofence contains every capture point", "a safety check reported as passed"),
    ("payload understands the planned trigger command", "a check that never ran"),
]


@pytest.mark.parametrize("needle,why", FABRICATED, ids=[n for n, _ in FABRICATED])
def test_the_planning_screen_invents_nothing(planning, needle, why) -> None:
    assert needle not in planning, f"{needle} is still rendered -- {why}"


class TestValidationReportsOnlyWhatItKnows:
    def test_nothing_is_reported_as_passed_that_was_not_checked(self, planning) -> None:
        """The only "ok" the panel may emit is one it derived from a real call."""
        block = planning.split('title: "Validation"')[1][:2600]
        for match in re.finditer(r'level: "ok"', block):
            window = block[max(0, match.start() - 400):match.start()]
            assert "terrain" in window.lower(), (
                "an ok result appears that is not derived from a call"
            )

    def test_it_says_which_control_runs_the_rest(self, planning) -> None:
        """Removing the false checks must not leave the operator thinking there are
        none. The checks exist; they run on Validate."""
        block = planning.split('title: "Validation"')[1][:2600]
        assert "Validate" in block

    def test_terrain_is_stated_either_way(self, planning) -> None:
        """Whether a terrain model is loaded changes what a planned altitude means, so
        it is reported when present as well as when missing."""
        block = planning.split('title: "Validation"')[1][:2600]
        assert "covered" in block
        assert "flat plane" in block

    def test_the_estimators_own_warnings_are_shown(self, planning) -> None:
        block = planning.split('title: "Validation"')[1][:2600]
        assert "warnings" in block


class TestEstimatesComeFromTheEstimator:
    def test_it_reads_mission_estimates(self, planning) -> None:
        block = planning.split('title: "Estimates"')[1][:1800]
        assert 'calls: ["mission_estimates"]' in block

    def test_more_than_one_battery_is_flagged(self, planning) -> None:
        """It changes what has to be in the car."""
        block = planning.split('title: "Estimates"')[1][:1800]
        assert "fits_in_one_flight" in block

    def test_an_unknown_camera_marks_the_storage_figure(self, planning) -> None:
        """Storage for an unrecognised camera is a guess about file size. The operator
        should know which kind of number they are looking at."""
        block = planning.split('title: "Estimates"')[1][:1800]
        assert "known_camera" in block

    def test_the_fields_exist(self, planning) -> None:
        estimates = (ROOT / "mission" / "estimates.py").read_text(encoding="utf-8")
        block = planning.split('title: "Estimates"')[1][:1800]
        for field in set(re.findall(r"\b(?:est|battery|storage)\.(\w+)", block)):
            assert f'"{field}"' in estimates, f"estimates do not include {field}"


class TestSegmentsKeepTheirCaveat:
    def test_it_reads_the_splitter(self, planning) -> None:
        block = planning.split('title: "Mission Segments"')[1][:1800]
        assert 'calls: ["plan_battery_segments"]' in block

    def test_the_note_is_rendered(self, planning) -> None:
        """plan_battery_segments() returns a note saying the boundaries are estimates
        from capture count and to fly to the aircraft's own battery warning. Dropping it
        leaves a table of numbers that look surveyed."""
        block = planning.split('title: "Mission Segments"')[1][:1800]
        assert "b.note" in block

    def test_the_split_fields_exist(self, planning) -> None:
        resume = (ROOT / "mission" / "resume.py").read_text(encoding="utf-8")
        block = planning.split('title: "Mission Segments"')[1][:1800]
        for field in set(re.findall(r"\bseg\.(\w+)", block)):
            assert f'"{field}"' in resume, f"plan_battery_segments() does not return {field}"
