"""The EXAMPLE DATA banner must not be a list of strings.

The banner is the last thing standing between invented content and someone who believes
it. It decided by searching the rendered text for sentinels: the demo organisation's name
and three demo site names. That catches a panel whose sample content happens to name a
demo site, and misses every panel that invents something else.

Fifty do. A Processing Queue of "#4471 / Feature matching / w-02". A Fleet readout of
"Aircraft 6, Available 4". Flight telemetry, alerts, mission progress, battery estimates,
maintenance records, thermal anomaly summaries, measurement history. None contains a
sentinel, so all of them rendered in connected mode with the banner **hidden** -- on the
flight and fleet screens, where being wrong about state is worst.

That is the same failure the banner exists to fix, one level up: the disclosure mechanism
was itself a list of the fabrications someone had already thought of.

The fix is structural. Anything that draws rows marks itself `data-rows`; `live()` marks
its subtree `data-live`; content is synthetic exactly when rows exist that no API call
produced. A panel converted to `live()` leaves the synthetic count automatically, and a
new panel rendering a literal joins it the moment it is written -- with no list.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

WORKSPACE_JS = Path(__file__).resolve().parents[1] / "app" / "web" / "js" / "workspace"


def code_of(name: str) -> str:
    """A module's source with comment lines removed.

    Comments in these files quote the strings they are about, at length. A guard that
    searches raw source passes on the prose describing the defect -- that happened once
    already in this suite and the test was silently vacuous.
    """
    text = (WORKSPACE_JS / name).read_text(encoding="utf-8")
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("*", "//", "/*"))
    )


@pytest.fixture(scope="module")
def shell() -> str:
    return code_of("shell.js")


@pytest.fixture(scope="module")
def primitives() -> str:
    return code_of("primitives.js")


@pytest.fixture(scope="module")
def live_js() -> str:
    return code_of("live.js")


class TestTheDetectorIsStructural:
    def test_it_looks_for_unanswered_rows(self, shell) -> None:
        body = shell.split("showingSampleContent()")[1].split("\n  }")[0]
        assert "data-rows" in body, "the banner still decides by text alone"
        assert "data-live" in body, "nothing distinguishes answered rows from invented ones"

    def test_a_sentinel_list_is_no_longer_the_only_evidence(self, shell) -> None:
        """Keeping the sentinels is fine; relying on them alone is what failed."""
        body = shell.split("showingSampleContent()")[1].split("\n  }")[0]
        before_sentinels = body.split("SAMPLE_SENTINELS")[0]
        assert "data-rows" in before_sentinels, (
            "the structural check must run before the string check, or a panel with no "
            "sentinel still returns false"
        )

    def test_rows_outside_a_live_subtree_count_as_synthetic(self, shell) -> None:
        body = shell.split("showingSampleContent()")[1].split("\n  }")[0]
        assert "closest" in body, "nothing checks whether the rows are inside a live()"
        assert "return true" in body


class TestEveryRowPrimitiveMarksItself:
    """If one of them does not, its panels are invisible to the banner again."""

    @pytest.mark.parametrize("primitive", ["table", "properties", "tree", "readouts"])
    def test_the_primitive_is_marked(self, primitives, primitive) -> None:
        assert primitives.count('"data-rows"') >= 4
        # The mark has to be on the element the primitive returns, not merely present
        # somewhere in the file.
        section = primitives.split(f"export function {primitive}(")[1].split("\nexport ")[0]
        assert "data-rows" in section, f"{primitive}() draws rows without marking them"

    def test_live_marks_what_it_produced(self, live_js) -> None:
        assert "data-live" in live_js


class TestTheMechanismCannotBeSatisfiedByAComment:
    def test_the_guard_strips_comments(self) -> None:
        """This file's own helper, checked against itself.

        code_of() must remove comment lines, or every assertion here can be satisfied by
        the explanatory comments in the modules under test -- which quote "data-rows" and
        "data-live" verbatim.
        """
        stripped = code_of("primitives.js")
        raw = (WORKSPACE_JS / "primitives.js").read_text(encoding="utf-8")
        assert len(stripped) < len(raw)
        assert " * The EXAMPLE DATA banner used to decide" not in stripped
