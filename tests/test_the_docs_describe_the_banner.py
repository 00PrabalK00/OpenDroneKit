"""What README and the UI guide say about the EXAMPLE DATA banner is what the cockpit does.

Both documents once said the banner runs across the top "until the application is
connected" / "in every state except connected". The shipped cockpit does not do that:
with no bridge the panels say "Not connected" and the banner is hidden; it appears in
demo mode (?demo=1) or while a panel is still rendering the bundled sample. A reader who
opened the cockpit without a bridge and saw no banner would conclude the docs, or the
app, were lying.

The test reads the two passages, then loads the shipped page in a browser in the two
states the passages describe and checks the banner against them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cockpit_harness import Cockpit, make_api

ROOT = Path(__file__).resolve().parents[1]


def _readme_row() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Operations UI"))
    return row


def _guide_section() -> str:
    text = (ROOT / "docs" / "UI_GUIDE.md").read_text(encoding="utf-8")
    match = re.search(r"## What is real on screen, and what is not\n(.*?)\n## ", text, re.S)
    assert match, "UI_GUIDE lost its section on what is real on screen"
    return " ".join(match.group(1).split())


FALSE_CLAIMS = (
    "until the application is connected",
    "every state except",
    "until it can talk to the application",
)


@pytest.mark.parametrize("name,passage", [("README", _readme_row), ("UI_GUIDE", _guide_section)])
def test_the_docs_do_not_claim_the_banner_shows_whenever_disconnected(name, passage) -> None:
    text = passage().lower()
    for claim in FALSE_CLAIMS:
        assert claim not in text, f"{name} still says the banner shows {claim!r}"
    assert "?demo=1" in text, f"{name} does not say when the banner does appear"
    assert "not connected" in text, f"{name} does not say what a bridgeless cockpit shows"


def test_the_banner_behaves_as_the_docs_say(tmp_path) -> None:
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "p1", bridge=False) as cockpit:
        nobridge = cockpit.run("""
            await ODK.sleep(4000);
            return { banner: !ODK.$(".demo-banner").classList.contains("hidden"),
                     tree: ODK.panel("Project Explorer") };
        """)
    assert not nobridge["banner"], "docs say no banner without a bridge; the page shows one"
    assert nobridge["tree"].startswith("Not connected")

    with Cockpit(api, tmp_path / "p2") as cockpit:
        demo = cockpit.run("""
            await ODK.sleep(2000);
            return { banner: !ODK.$(".demo-banner").classList.contains("hidden"),
                     text: ODK.$(".demo-banner").innerText };
        """, query="?demo=1")
    assert demo["banner"], "docs say ?demo=1 raises the banner; the page does not"
    assert "EXAMPLE DATA" in demo["text"]
