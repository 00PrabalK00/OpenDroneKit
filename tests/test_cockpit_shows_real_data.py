"""The cockpit must show the project, or say it has nothing. Never a plausible number.

Driving the real window with a 77-image project open found Processing reporting
"1,842 images", three worker machines with RTX 4090s, a job queue of "DEMO site 1/2/3",
a 3,140 m2 metal-deck roof with four open findings, and a survey timeline of three dates.
None of it came from anywhere. It was structural sample content that the panels rendered
because they had no other source, and it was indistinguishable from measurement.

These tests pin the two halves of the fix: the fabricated constants are gone, and the
panels that replaced them read from the application.
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


# Every one of these appeared on screen over a real project. They are specific enough
# that a match is a fabricated measurement rather than a coincidence.
FABRICATED = [
    ("1,842", "an invented image count"),
    ("1842", "an invented image count"),
    ("412,880", "an invented pair count"),
    ("RTX 4090", "worker machines that do not exist"),
    ("3,140", "the area of a roof that was never surveyed"),
    ("1,240 m", "a stockpile volume that was never measured"),
    ("redis 7.4", "a broker this local-first build does not run"),
    ("98.4%", "an invented coverage figure"),
    ("asset/roof-block-a", "an asset id that belongs to no project"),
    ("23.2591 N", "a centroid for an asset that does not exist"),
]


@pytest.mark.parametrize("needle,why", FABRICATED, ids=[n for n, _ in FABRICATED])
def test_no_fabricated_measurement_is_rendered(workspaces, needle, why) -> None:
    """A comment may DISCUSS the old value; no code may still render it."""
    code = "\n".join(
        line for line in workspaces.splitlines()
        if not line.lstrip().startswith(("*", "//", "/*"))
    )
    assert needle not in code, f"{needle} is still rendered -- {why}"


def test_the_panels_that_replaced_them_read_from_the_application(workspaces) -> None:
    """Removing the constants is only half the fix; the panels have to ask."""
    assert 'from "./live.js"' in workspaces
    # A representative spread rather than an exhaustive list, so adding a panel does not
    # fail this, but gutting the wiring does.
    for call in ("list_jobs", "list_layers", "list_datasets", "audit_log", "verify_models"):
        assert f'"{call}"' in workspaces, f"no panel reads {call}"


def test_every_live_panel_declares_what_empty_looks_like(workspaces) -> None:
    """The whole point: with no data the panel says so rather than inventing some.

    A `live()` without an `empty` message would fall through to a blank box, which is the
    state that made sample content look like the better option in the first place.
    """
    blocks = re.findall(r"live\(\{(.*?)\n    \}\)", workspaces, re.S)
    assert blocks, "no live() panels found"
    missing = [b[:70].replace("\n", " ") for b in blocks if "empty:" not in b]
    assert not missing, f"live() panels with no empty state: {missing}"


class TestTheCanvasShowsThisProject:
    def test_it_reads_the_project_layers_rather_than_the_bundled_example(self, primitives) -> None:
        """It rendered DATA.products, so every project showed the same picture -- and the
        Digital Twin, which has no entry there, showed nothing at all while a real
        orthomosaic, DSM and DTM sat on disk."""
        assert "projectProduct" in primitives
        assert "raster_preview" in primitives
        body = primitives.split("async function projectProduct")[1]
        assert "list_layers" in body

    def test_rgb_resolves_to_the_orthomosaic(self, primitives) -> None:
        """The toolbar button is labelled RGB and means the orthomosaic. Leaving it out
        of the map made the view most likely to have a product report having none."""
        table = primitives.split("const VIEW_LAYERS")[1].split("};")[0]
        assert "rgb:" in table
        assert "orthomosaic" in table.split("rgb:")[1].split("\n")[0]

    def test_a_missing_product_is_not_filled_in_from_elsewhere(self, primitives) -> None:
        """Showing another project's orthomosaic under this project's title would be
        worse than the empty canvas it replaced."""
        body = primitives.split("async function projectProduct")[1].split("\n}")[0]
        assert "DATA" not in body, "the resolver can still fall back to the example data"
