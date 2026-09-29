"""docs/UI_GUIDE.md describes the cockpit's frame the way the shipped cockpit behaves.

The guide's closing section, "What it is not, yet", said the status bar read "sample data
-- not connected to a project" permanently until a project was attached, that the
numbers on screen were illustrative, that toolbar actions were not wired to the API, and
that the canvases were placeholders with a real map "the next step". Every one of those
had stopped being true: the chip reports the bridge's real state, actions.js calls the
Api, MapLibre and a WebGL viewer are mounted. A reader believing the guide would think a
connected cockpit was a mock-up; one believing the cockpit would think the guide stale.

This test pins the rewritten section to the page: it reads the section, then opens the
shipped workspace.html with no bridge, in demo mode and against a real Api, and checks
the chip says what the guide says in each state, and that the canvases the guide still
calls data-less frames exist and show no numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from cockpit_harness import Cockpit, make_api

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "UI_GUIDE.md"

STALE = (
    "sample data — not connected to a project",
    "sample data - not connected to a project",
    "toolbar actions are not wired",
    "reports that it is not wired",
    "panel-mounted placeholders",
    "is the next step",
    "illustrative structure",
    "what it is not, yet",
)

# The chip text the guide promises for each state.
CHIP = {
    "connected": "connected — ",
    "demo": "demo mode",
    "nobridge": "not connected to a project",
}

LABELLED_FRAMES = (
    ("Flight", "Telemetry Charts"),
    ("Flight", "Camera"),
    ("Thermal", "Thermal Histogram"),
)


def _section() -> str:
    text = GUIDE.read_text(encoding="utf-8")
    match = re.search(r"## What the frame says, and what is still missing\n(.*?)\n## ", text, re.S)
    assert match, "UI_GUIDE lost its section on what the frame says"
    return " ".join(match.group(1).split())


def test_the_guide_no_longer_calls_the_cockpit_a_mock_up() -> None:
    whole = " ".join(GUIDE.read_text(encoding="utf-8").split()).lower()
    for claim in STALE:
        assert claim not in whole, f"UI_GUIDE still says {claim!r}"


def test_the_guide_names_every_chip_state_and_every_empty_frame() -> None:
    section = _section()
    for text in CHIP.values():
        assert text.strip() in section, f"the guide does not say the chip reads {text!r}"
    for _, title in LABELLED_FRAMES:
        assert title in section
    assert "API console" in section
    for source in ("actions.js", "mapview.js", "viewer3d.js"):
        assert source in section
        assert (ROOT / "app" / "web" / "js" / "workspace" / source).is_file()


CHIP_JS = "(ODK.$('.statusbar .chip') || {}).textContent || ''"


def test_the_chip_says_what_the_guide_says_in_every_state(tmp_path) -> None:
    api = make_api(tmp_path)

    with Cockpit(api, tmp_path / "nb", bridge=False) as cockpit:
        nobridge = cockpit.run(f"""
            await ODK.waitFor(() => ODK.$('.statusbar .chip'));
            await ODK.sleep(3000);
            return {{ chip: {CHIP_JS}, tree: ODK.panel("Project Explorer") }};
        """)
    assert nobridge["chip"] == CHIP["nobridge"]
    assert (nobridge["tree"] or "").startswith("Not connected")

    with Cockpit(api, tmp_path / "demo") as cockpit:
        demo = cockpit.run(f"""
            await ODK.waitFor(() => /demo/.test({CHIP_JS}));
            return {{ chip: {CHIP_JS} }};
        """, query="?demo=1")
    assert demo["chip"] == CHIP["demo"]

    with Cockpit(api, tmp_path / "live") as cockpit:
        live = cockpit.run(f"""
            await ODK.waitFor(() => /^connected/.test({CHIP_JS}));
            await ODK.waitFor(() => /^v[0-9]/.test(ODK.$('.statusbar .mono[title]').textContent));
            return {{ chip: {CHIP_JS}, version: ODK.$('.statusbar .mono[title]').textContent }};
        """)
    projects = len(api.list_projects()["projects"])
    assert live["chip"] == f"connected — {projects} project(s)", live
    version = re.search(r'__version__ = "([^"]+)"', (ROOT / "__init__.py").read_text(encoding="utf-8")).group(1)
    assert live["version"] == f"v{version}", live

def test_the_frames_the_guide_calls_empty_exist_and_show_no_numbers(tmp_path) -> None:
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "p") as cockpit:
        seen = cockpit.run("""
            await ODK.waitFor(() => /^connected/.test((ODK.$('.statusbar .chip') || {}).textContent || ''));
            const out = {};
            const header = (t) => ODK.$$('.panel').find((p) =>
              ((p.querySelector('.panel-header') || {}).textContent || '').includes(t));
            for (const [ws, title] of [["Flight", "Telemetry Charts"], ["Flight", "Camera"],
                                       ["Thermal", "Thermal Histogram"]]) {
              await ODK.nav(ws);
              const p = header(title);
              out[title] = p ? p.querySelector('.panel-body').textContent : null;
            }
            await ODK.nav("Developers");
            const c = ODK.$('.workspace .canvas');
            out["API console"] = c ? c.textContent : null;
            return out;
        """)
    for title, text in seen.items():
        assert text is not None, f"the guide names {title!r} but the cockpit has no such frame"
        assert not re.search(r"[0-9]", text), f"{title} shows numbers the guide says it cannot have: {text!r}"
