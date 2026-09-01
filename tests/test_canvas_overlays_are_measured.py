"""No number may be printed on the canvas that did not come from the application.

The cockpit's fabricated sample data was swept three times: once for the panels, once
for four blocks the first sweep's string list missed, and once for the settings and
mission forms. All three searched panel bodies. None of them looked at canvas overlays,
and six overlays were still printing figures written into the source:

    mission map   214 captures · 3.1 km · 18 min · GSD 1.8 cm
    flight        62 / 214 captures · AUTO · RTK fixed
    flight        Segment 1 of 4 — nadir grid pass A
    verification  1836 matched · 3 out of tolerance · 3 missing
    thermal       Fused · 7 anomalies
    thermal       Ambient 34.2 °C · palette ironbow
    measurement   Vertical accuracy ±0.05 m — nothing below 0.10 m is reported

An overlay sits in the corner of the view, in the position a readout occupies on every
instrument the operator has ever used, which makes it read as a measurement of what is
on screen more strongly than a table does.

The last one is the reason this file exists rather than six more banned strings. There
is no fixed vertical accuracy and no fixed reporting floor: detection_floor() in
core/deformation.py derives it per comparison from each survey's own accuracy and the
registration residual between them. A number stating what the instrument can resolve is
the number a reader cites to argue that a movement was real.

So this tests the shape rather than the instances: an overlay carrying a digit must be a
live() that asked for it.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_JS = REPO_ROOT / "app" / "web" / "js" / "workspace"


@pytest.fixture(scope="module")
def workspaces() -> str:
    return (WORKSPACE_JS / "workspaces.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def primitives() -> str:
    return (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")


def overlays(source: str) -> list[tuple[int, str]]:
    """Every `{ at: ..., html: ... }` overlay, with its line number."""
    found = []
    for match in re.finditer(r'\{ at: "\w+", html: (`[^`]*`|"[^"]*")', source):
        found.append((source[: match.start()].count("\n") + 1, match.group(1)))
    return found


# A year, a version or a CSS length is not a measurement. Everything else is.
ALLOWED = re.compile(r"(EPSG:\d+|\bv\d+\b|\d+px|\d+%\s*;)")


def test_no_fixed_overlay_prints_a_measurement(workspaces) -> None:
    offenders = []
    for line, text in overlays(workspaces):
        stripped = ALLOWED.sub("", text)
        if re.search(r"\d", stripped):
            offenders.append(f"workspaces.js:{line} {text[:80]}")
    assert not offenders, (
        "a canvas overlay prints a number that no API call produced:\n  "
        + "\n  ".join(offenders)
    )


def test_the_canvas_can_carry_a_live_overlay(primitives) -> None:
    """Removing the constants is only half of it; there has to be a way to ask.

    Before this, `overlays` accepted an html string and nothing else, so a panel that
    wanted a real figure in the corner had no option that was not a constant. That is
    the same shape as the panels before live(): the honest path did not exist.
    """
    body = primitives.split("export function canvas")[1].split("\n}")[0]
    assert "overlay.node" in body, "canvas overlays cannot carry an element"


def test_the_readout_helper_declares_an_empty_state(workspaces) -> None:
    """An overlay with nothing to show must say so, not disappear.

    A vanished readout looks identical to a readout that has not loaded, and both look
    like the instrument is working.
    """
    helper = workspaces.split("function readout(")[1].split("\n}")[0]
    assert "empty" in helper
    assert "live({" in helper


class TestTheFieldsTheOverlaysAskFor:
    """Each overlay must read fields its API really returns.

    Writing these six found three invented names in the first draft -- `capture_count`
    (it is `image_count`), and `checked` / `out_of_tolerance` on the GCP report (it is
    `used` / `point_count` / `outlier_count`). That is the same failure as the mission
    form, which sent `gsd` and `angle` to a planner that reads neither: a plausible name
    renders as blank or undefined, never as an error.
    """

    def test_mission_estimates_fields_exist(self, workspaces) -> None:
        from mission.estimates import estimate_mission  # noqa: F401

        source = (REPO_ROOT / "mission" / "estimates.py").read_text(encoding="utf-8")
        block = workspaces.split('calls: ["mission_estimates"]')[1][:900]
        for field in re.findall(r"est\.(\w+)", block):
            assert f'"{field}"' in source, f"estimate_mission() does not return {field}"

    def test_telemetry_fields_exist(self, workspaces) -> None:
        source = (REPO_ROOT / "core" / "drone.py").read_text(encoding="utf-8")
        declared = source.split("class DroneTelemetry")[1].split("@property")[0]
        block = workspaces.split('calls: ["telemetry"]')[1][:1400]
        for field in set(re.findall(r"tm\.(\w+)", block)):
            assert f"{field}:" in declared, f"DroneTelemetry has no {field}"

    def test_gcp_report_fields_exist(self, workspaces) -> None:
        source = (REPO_ROOT / "core" / "gcp.py").read_text(encoding="utf-8")
        block = workspaces.split('calls: ["gcp_accuracy_report"]')[1][:1200]
        for field in set(re.findall(r"\br\.(\w+)", block)):
            assert f'"{field}"' in source, f"accuracy_report() does not return {field}"

    def test_an_unmeasurable_survey_is_not_given_an_accuracy(self, workspaces) -> None:
        """accuracy_report() returns rmse_m: None with a warning saying not to quote one.

        Rendering that as 0.000 m would turn "this survey has no measured accuracy" into
        the best possible result.
        """
        block = workspaces.split('calls: ["gcp_accuracy_report"]')[1][:1200]
        assert "rmse_m === null" in block or "rmse_m == null" in block
