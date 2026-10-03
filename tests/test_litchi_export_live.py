"""The Litchi CSV does what the plan says, and the menu item writes a Litchi CSV.

Two defects, both on the way a user gets a Litchi mission out of the application:

* The action codes were Litchi's wrong numbers. exporters.py wrote 1 for "stay" and 5
  for "take photo"; Litchi reads 0 = stay for (ms), 1 = take photo, 5 = tilt camera
  (deg). Every structure mission therefore tilted the camera to 0 deg at each point
  and took no photograph, and a 2 s dwell became a photo instead of a hover.
* Mission > Export Litchi CSV in the cockpit went through the generic Export verb,
  which dropped the menu's {format: "litchi"} and wrote all six formats.

The first class reads the files export_mission writes; the second drives the cockpit
in headless Edge/Chromium against the production Api (tests/ui_live_bridge.py).
"""

from __future__ import annotations

import csv
import time

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]

# litchiutilities.com/docs/litchiCsv.php
STAY_MS, TAKE_PHOTO, TILT_CAMERA = "0", "1", "5"


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    assert api.create_project("litchi", str(tmp_path / "project"))["ok"]
    assert api.set_aoi(AOI)["ok"]
    return api


def litchi_rows(written) -> list[dict]:
    paths = [p for k, p in written.items() if k.startswith("litchi")]
    assert paths and not any(str(p).startswith("failed") for p in paths), written
    rows: list[dict] = []
    for path in sorted(paths):
        rows.extend(csv.DictReader(open(path, encoding="utf-8")))
    return rows


def actions(row) -> list[tuple[str, str]]:
    pairs = [(row[f"actiontype{i}"], row[f"actionparam{i}"]) for i in range(1, 16)]
    return [p for p in pairs if p[0] != "-1"]


class TestTheActionCodesAreLitchis:
    @pytest.mark.parametrize("template", ["tower_mapping", "orbit", "box_inspection"])
    def test_a_structure_mission_takes_a_photo_at_each_capture(self, api, template):
        planned = api.plan_mission({"template": template})
        assert planned["ok"], planned
        exported = api.export_mission(["litchi"])
        rows = litchi_rows(exported["written"])
        photos = [r for r in rows if (TAKE_PHOTO, "0") in actions(r)]
        assert photos, "no row takes a photograph"
        # No row tilts the camera through an action: pitch is the gimbal column.
        assert not any(a == TILT_CAMERA for r in rows for a, _ in actions(r))
        captures = [f for f in planned["geojson"]["features"]
                    if f["properties"].get("role") == "waypoint" and f["properties"].get("trigger")]
        assert len(photos) == len(captures)

    def test_a_dwell_is_a_hover_before_the_photo(self, api):
        assert api.plan_mission({"template": "orbit", "dwell_s": 2})["ok"]
        rows = litchi_rows(api.export_mission(["litchi"])["written"])
        captures = [r for r in rows if (TAKE_PHOTO, "0") in actions(r)]
        assert captures
        for row in captures:
            assert actions(row)[:2] == [(STAY_MS, "2000"), (TAKE_PHOTO, "0")]

    def test_the_unused_interval_columns_say_not_used(self, api):
        assert api.plan_mission({"template": "orbit"})["ok"]
        rows = litchi_rows(api.export_mission(["litchi"])["written"])
        assert {r["photo_timeinterval"] for r in rows} == {"-1"}
        assert {r["photo_distinterval"] for r in rows} == {"-1"}

    def test_a_grid_photographs_every_capture_point(self, api):
        planned = api.plan_mission({"template": "grid"})
        assert planned["ok"]
        rows = litchi_rows(api.export_mission(["litchi"])["written"])
        captures = [f for f in planned["geojson"]["features"]
                    if f["properties"].get("role") == "waypoint" and f["properties"].get("trigger")]
        assert captures
        # The planner flies a grid with distance-triggered capture, so Litchi shoots by
        # photo_distinterval "from this waypoint to the next" (flylitchi.com/help): every
        # capture point is either photographed there or inside an interval segment.
        assert len(rows) == len(captures)
        interval = [float(r["photo_distinterval"]) > 0.0 for r in rows]
        assert any(interval)
        for i, row in enumerate(rows):
            assert (TAKE_PHOTO, "0") in actions(row) or interval[i] or (i > 0 and interval[i - 1])
        assert not any(a == TILT_CAMERA for r in rows for a, _ in actions(r))


HELPERS = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((mutations) => mutations.forEach((m) =>
    m.addedNodes.forEach((n) => {
      if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent);
    })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
true
"""


@pytest.fixture
def browser():
    pytest.importorskip("websocket")
    pytest.importorskip("requests")
    from ui_live_bridge import Browser, chromium_path

    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    driver = Browser(executable)
    yield driver
    driver.close()


class TestTheMenuItemExportsLitchi:
    def test_mission_menu_export_litchi_writes_only_the_litchi_csv(self, api, browser, tmp_path):
        from ui_live_bridge import serve

        assert api.plan_mission({"template": "tower_mapping"})["ok"]
        server, base = serve(api)
        try:
            browser.open(f"{base}/workspace.html", settle_s=3)
            browser.js(HELPERS)
            missions = tmp_path / "project" / "missions"

            # What app/shell.py's "Export Litchi CSV" item sends.
            browser.js("window.odk.onMenu({action: 'mission.export', payload: {format: 'litchi'}}); true")
            toast = str(browser.wait_for(
                "__toasts.find((t) => t.includes('Exported') || t.includes('Export:'))", timeout_s=60))
            written = sorted(missions.glob("*"))
            assert written and {p.suffix for p in written} == {".csv"}, [p.name for p in written]
            # The one-format menu item names each file it wrote (Exported litchi: a, b, ...).
            assert toast.startswith("Exported litchi:"), toast
            assert all(str(p) in toast for p in written), (toast, written)
            rows = []
            for path in written:
                rows.extend(csv.DictReader(open(path, encoding="utf-8")))
            assert any((TAKE_PHOTO, "0") in actions(r) for r in rows)
            assert not any(a == TILT_CAMERA for r in rows for a, _ in actions(r))

            # Export All Formats still writes every format.
            browser.js("window.__toasts.length = 0; true")
            time.sleep(1.1)   # a new timestamped stem
            browser.js("window.odk.onMenu({action: 'mission.export_all', payload: {}}); true")
            browser.wait_for("__toasts.find((t) => t.includes('Exported'))", timeout_s=60)
            suffixes = {p.suffix for p in missions.glob("*")}
            assert {".plan", ".waypoints", ".kmz", ".csv", ".kml", ".geojson"} <= suffixes, suffixes
        finally:
            server.shutdown()
