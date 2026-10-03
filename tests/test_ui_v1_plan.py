"""The v1 Plan workspace (U5), in a real browser, against the real Api.

odk.html is served with the production Api behind it (tests/ui_v1_map_harness.MapServer:
the page, the Api bridge and the raster tile route; only pywebview's transport is
replaced) and driven in headless Edge/Chromium through the frozen shell contract.

REBUILD_PLAN 3.4, Plan: draw the area -> Generate a grid -> the figures are the Api's
(mission_overview / mission_estimates) -> Save version -> Open it again -> Export
.plan and .kmz files that exist -> the camera picker describes the camera. Plus what
this unit built: every placement in screens/plan/surface.js renders its data-action,
the waypoint card and footprints come from the plan, the vertex editor and undo go
through set_aoi, the line heading reaches the planner, the two-way GSD is the Api's,
Field Plan re-plans from its stepper, and an empty project shows absence, not figures.

With the full-resolution Aukerman survey on this machine (ODK_AUKERMAN_FULLRES) the
project carries its orthomosaic and camera positions, the area is the hull of the 77
cameras, and screenshots of desk (1440x900) and Field (1180x820) are written to
$ODK_U5_SHOTS for comparison with SysPlan / SysFieldPlan.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("websocket")
pytest.importorskip("requests")

from ui_live_bridge import WEB_ROOT  # noqa: E402
from ui_v1_map_harness import MapBrowser, MapServer, chromium_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUKERMAN = Path(os.environ.get("ODK_AUKERMAN_FULLRES", REPO / "demo_project" / "aukerman_fullres"))
if (AUKERMAN / "reconstruction").is_dir():
    AUKERMAN = AUKERMAN / "reconstruction"
HAS_SURVEY = (AUKERMAN / "orthomosaic.tif").is_file()
SHOTS = Path(os.environ.get("ODK_U5_SHOTS") or Path(tempfile.gettempdir()) / "odk-u5-shots")
# A small site used when the survey is not on this machine.
SITE = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]

pytestmark = pytest.mark.skipif(chromium_path() is None, reason="No Chromium or Edge to drive.")

WATCH = """
window.__toasts = window.__toasts || [];
window.__errors = window.__errors || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
    if (n.classList && n.classList.contains('odk-toast')) window.__toasts.push(n.textContent);
  })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
  window.addEventListener('error', (e) => window.__errors.push(String(e.message)));
  window.addEventListener('unhandledrejection', (e) => window.__errors.push(String(e.reason && e.reason.stack || e.reason)));
}
true
"""


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    root = tmp_path_factory.mktemp("u5-plan")
    cwd = os.getcwd()
    os.chdir(root)
    previous_storage = os.environ.get("ODK_STORAGE_PATH")
    os.environ["ODK_STORAGE_PATH"] = str(root / "storage")
    api = Api(AppSession(ProjectStore(root / "projects.db")))
    assert api.create_project("Aukerman Park", str(root / "project"))["ok"]
    session = api._session
    if HAS_SURVEY:
        session.register_raster_layer(AUKERMAN / "orthomosaic.tif", name="Ortho 2016-06-29")
        session.register_vector_layer(AUKERMAN / "camera_track.geojson", name="Camera positions")
        assert api.set_project_crs(32617)["ok"]
    page = (WEB_ROOT / "odk.html").read_bytes().replace(
        b"<head>", b'<head><script src="/__odk_bridge.js"></script>', 1)
    server = MapServer(api, page)
    browser = MapBrowser(chromium_path())
    try:
        yield {"api": api, "server": server, "browser": browser, "root": root}
    finally:
        browser.close()
        server.close()
        os.chdir(cwd)
        if previous_storage is None:
            os.environ.pop("ODK_STORAGE_PATH", None)
        else:
            os.environ["ODK_STORAGE_PATH"] = previous_storage


def boot(stack, mode: str = "desk", size: tuple[int, int] = (1440, 900)) -> MapBrowser:
    browser = stack["browser"]
    browser.viewport(*size)
    base = stack["server"].base
    # Leave the previous page first, so its tile requests stop competing with this load,
    # and set the workspace from a page of the same origin that runs no script.
    browser.open("about:blank", settle_s=0.5)
    browser.open(f"{base}/css/odk/tokens.css", settle_s=0.1)
    browser.js(f"localStorage.setItem('odk.workspace','plan'); localStorage.setItem('odk.mode','{mode}'); true")
    for attempt in range(2):
        browser.open(f"{base}/__stage.html?mode={mode}", settle_s=0.2)
        browser.wait_for("window.__odkReady === true || window.__odkBootError", 180)
        error = browser.js("window.__odkBootError || null")
        # A loopback fetch the OS refused (a shared machine out of sockets) is a load
        # failure, not the page's: one reload, then it counts.
        if error and "Failed to fetch" in error and attempt == 0:
            browser.open("about:blank", settle_s=2.0)
            continue
        break
    assert browser.js("window.__odkBootError || null") is None, browser.js("window.__odkBootError")
    browser.wait_for("odkShell.connection === 'connected'", 60)
    browser.js(WATCH)
    browser.wait_for("document.querySelector('.odk-frame').dataset.rendered === 'plan'", 60)
    settle(browser)
    return browser


def settle(browser: MapBrowser, timeout_s: float = 60) -> None:
    """Every live panel on the Plan screen has answered."""
    browser.wait_for("![...document.querySelectorAll('.odk-frame .live')].some((n) => n.querySelector(':scope > .live-wait'))",
                     timeout_s)


def refresh(browser: MapBrowser) -> None:
    browser.js("odkShell.refreshState().then(() => true)")
    time.sleep(0.3)
    settle(browser)


def run(browser: MapBrowser, key: str, payload: dict | None = None):
    return browser.js(f"odkShell.runAction({json.dumps(key)}, {json.dumps(payload)})", timeout_s=180)


def answer_modal(browser: MapBrowser, values: dict[str, str] | None = None, choose: str | None = None) -> None:
    """Answer the open modal: fill inputs in order, or click the row whose text contains `choose`."""
    browser.wait_for("!!document.querySelector('.modal-backdrop .modal')", 20)
    if choose is not None:
        clicked = browser.js(f"""(() => {{
          const row = [...document.querySelectorAll('.modal .modal-row')].find((r) => r.textContent.includes({json.dumps(choose)}));
          if (row) row.click();
          return !!row; }})()""")
        assert clicked, browser.js("[...document.querySelectorAll('.modal .modal-row')].map((r) => r.textContent)")
        return
    browser.js(f"""(() => {{
      const inputs = [...document.querySelectorAll('.modal .modal-field input, .modal .modal-field select')];
      {json.dumps(list((values or {}).values()))}.forEach((v, i) => {{ if (inputs[i]) inputs[i].value = v; }});
      document.querySelector('.modal .modal-actions .tbtn.primary').click();
      return true; }})()""")


def overview(stack) -> dict:
    return stack["api"].mission_overview()


def set_site(stack) -> None:
    api = stack["api"]
    if HAS_SURVEY:
        assert api.aoi_from_images()["ok"]
    else:
        assert api.set_aoi(SITE)["ok"]


def text(browser: MapBrowser, selector: str) -> str:
    return browser.js(f"(document.querySelector({json.dumps(selector)}) || {{}}).textContent || ''")


def drag(browser: MapBrowser, start: tuple[float, float], end: tuple[float, float], steps: int = 8) -> None:
    browser._send("Input.dispatchMouseEvent", type="mouseMoved", x=start[0], y=start[1])
    browser._send("Input.dispatchMouseEvent", type="mousePressed", x=start[0], y=start[1], button="left", clickCount=1)
    for i in range(1, steps + 1):
        x = start[0] + (end[0] - start[0]) * i / steps
        y = start[1] + (end[1] - start[1]) * i / steps
        browser._send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="left", buttons=1)
        time.sleep(0.03)
    browser._send("Input.dispatchMouseEvent", type="mouseReleased", x=end[0], y=end[1], button="left", clickCount=1)


def map_point(browser: MapBrowser, lon: float, lat: float) -> tuple[float, float]:
    """Viewport pixel of a lon/lat on the Plan map."""
    return tuple(browser.js(f"""(() => {{
      const map = odkShell.actionContext().map();
      const p = map.project([{lon}, {lat}]);
      const r = map.getContainer().getBoundingClientRect();
      return [r.left + p.x, r.top + p.y]; }})()"""))


def wait_map(browser: MapBrowser) -> None:
    browser.wait_for("(() => { const m = odkShell.actionContext().map(); return !!(m && m._odkReady && m.getLayer('odk-p-mission-pt')); })()", 120)
    browser.js("new Promise((r) => { const m = odkShell.actionContext().map(); if (m.loaded()) r(true); else m.once('idle', () => r(true)); setTimeout(() => r(true), 15000); })",
               timeout_s=60)
    # The view is fitted to the project with an animation: wait until it has stopped.
    still = 0
    deadline = time.monotonic() + 30
    while still < 4 and time.monotonic() < deadline:
        moving = browser.js("(() => { const m = odkShell.actionContext().map(); return m.isMoving() || m.isZooming() || m.isEasing(); })()")
        still = 0 if moving else still + 1
        time.sleep(0.25)


# ======================================================================= the flow

class TestThePlanFlow:
    def test_an_empty_project_shows_absence_not_figures(self, stack) -> None:
        api = stack["api"]
        api.set_aoi([])
        api._session.mission_plan_dict = {}
        browser = boot(stack)
        strip = text(browser, ".odk-plan-strip")
        assert "no plan" in strip and not any(ch.isdigit() for ch in strip), strip
        assert "NOT SET" in text(browser, ".odk-plan-aoi")
        values = browser.js("[...document.querySelectorAll('.odk-plan-inspector input[data-field]')].map((n) => n.value)")
        assert values and all(v == "" for v in values), values
        assert browser.js("document.querySelector('.odk-plan-inspector [data-action=\"Plan\"]').disabled") is True
        assert text(browser, ".odk-plan-name") == "no mission planned"

    def test_draw_the_area_then_generate_and_the_figures_are_the_apis(self, stack) -> None:
        api = stack["api"]
        api.set_aoi([])
        api._session.mission_plan_dict = {}
        browser = boot(stack)
        ring = SITE
        # Draw the area with the Draw area tool, through draw.js, as clicks would.
        assert run(browser, "Draw Area").get("tool") == "polygon"
        for lon, lat in ring:
            browser.js(f"odkShell.drawing.addVertex([{lon}, {lat}]); true")
        browser.js("odkShell.drawing.finishDrawing(); true")
        browser.wait_for("odkShell.state && (odkShell.state.aoi || []).length >= 3", 30)
        settle(browser)
        assert api.mission_overview()["aoi"]["vertices"] == len(ring)
        aoi_text = text(browser, ".odk-plan-aoi")
        assert "SET" in aoi_text and f"{len(ring)}" in aoi_text and "drawn on the map" in aoi_text

        browser.js("document.querySelector('[data-template=\"grid\"]').click(); true")
        settle(browser)
        result = browser.js("odkShell.runAction('Plan')", timeout_s=180)
        assert "Planned" in (result.get("message") or ""), result
        refresh(browser)
        view = overview(stack)["plan"]
        strip = text(browser, ".odk-plan-strip")
        assert f"{view['waypoint_count']:,}" in strip
        assert f"{round(view['distance_m']):,}" in strip
        assert f"{view['duration_min']:.1f}" in strip
        assert f"{view['line_count']} lines" in strip
        # The Sorties tab: batteries and photos are mission_estimates' own.
        browser.js("document.querySelector('[data-control=\"panel-tab:segments\"]').click(); true")
        settle(browser)
        estimates = api.mission_estimates()["estimates"]
        sorties = text(browser, ".odk-plan-insp-body")
        assert f"{estimates['image_count']:,}" in sorties
        assert str(estimates["battery"]["batteries_required"]) in sorties
        browser.js("document.querySelector('[data-control=\"panel-tab:plan\"]').click(); true")
        assert browser.js("window.__errors") == []

    def test_save_a_version_change_the_plan_and_open_the_version_again(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        browser.js("(window.__saved = null, odkShell.runAction('Save').then((r) => (window.__saved = r)), true)")
        answer_modal(browser, {"name": "aukerman_grid"})
        browser.wait_for("!!window.__saved", 60)
        refresh(browser)
        assert text(browser, ".odk-plan-name") == "aukerman_grid · v1"
        saved_waypoints = overview(stack)["plan"]["waypoint_count"]

        # A different altitude, planned: the name says the plan on screen is not v1.
        browser.js("(() => { const i = document.querySelector('input[data-field=\"altitude_m\"]'); i.value = '80';"
                   " i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); return true; })()")
        assert "Planned" in (run(browser, "Plan").get("message") or "")
        refresh(browser)
        assert overview(stack)["plan"]["altitude_m"] == pytest.approx(80.0)
        assert text(browser, ".odk-plan-name") == "unsaved plan"

        browser.js("(window.__loaded = null, odkShell.runAction('Load Version').then((r) => (window.__loaded = r)), true)")
        answer_modal(browser, choose="aukerman_grid v1")
        browser.wait_for("!!window.__loaded", 120)
        loaded = browser.js("window.__loaded")
        assert "as saved" in loaded.get("message", ""), loaded
        refresh(browser)
        now = overview(stack)
        assert now["plan"]["altitude_m"] == pytest.approx(55.0)
        assert now["plan"]["waypoint_count"] == saved_waypoints
        assert text(browser, ".odk-plan-name") == "aukerman_grid · v1"
        assert browser.js("document.querySelector('input[data-field=\"altitude_m\"]').value") == "55.0"

    def test_export_writes_the_ticked_formats_and_shows_their_sizes(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        # Tick exactly QGroundControl and DJI WPML.
        browser.js("""(() => {
          for (const row of document.querySelectorAll('[data-control^="export:"]')) {
            const box = row.querySelector('input');
            const want = ['export:qgc_plan', 'export:dji_wpml'].includes(row.dataset.control);
            if (box.checked !== want) box.click();
          }
          return true; })()""")
        button = "document.querySelector('[data-action=\"Export Mission Files\"]')"
        assert browser.js(f"{button}.textContent") == "Export 2 formats"
        browser.js(f"{button}.click(); true")
        browser.wait_for("window.__toasts.some((t) => t.includes('Exported'))", 120)
        refresh(browser)
        listed = api.list_mission_exports()["exports"][0]["files"]
        written = {f["format"]: f for f in listed}
        assert set(written) == {"qgc_plan", "dji_wpml"}
        for f in written.values():
            assert Path(f["path"]).is_file() and Path(f["path"]).stat().st_size == f["bytes"] > 0
        sizes = browser.js("""Object.fromEntries([...document.querySelectorAll('[data-control^="export:"]')]
          .map((r) => [r.dataset.control.slice(7), r.querySelector('.odk-plan-export-size').textContent]))""")
        for key, f in written.items():
            assert sizes[key] == f"{f['bytes'] / 1024:.1f} KB", (key, sizes[key], f["bytes"])
        assert sizes["kml"] == "—"

    def test_the_camera_picker_describes_the_camera_at_the_planned_altitude(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        browser.js("document.querySelector('[data-control=\"camera-picker\"]').click(); true")
        browser.wait_for("!document.querySelector('[data-plan-menu=\"Camera and payload\"]').hidden", 10)
        names = browser.js("[...document.querySelectorAll('[data-plan-menu=\"Camera and payload\"] [data-control^=\"camera:\"]')].map((n) => n.dataset.control.slice(7))")
        assert sorted(names) == sorted(c["key"] for c in api.list_cameras()["cameras"])
        browser.js("document.querySelector('[data-plan-menu=\"Camera and payload\"] [data-action=\"Describe Camera\"]').click(); true")
        browser.wait_for("!!document.querySelector('.modal table.props')", 30)
        table = browser.js("Object.fromEntries([...document.querySelectorAll('.modal table.props tr')].map((r) => [r.cells[0].textContent, r.cells[1].textContent]))")
        expected = api.describe_camera("mavic2pro", 55)["camera"]
        assert table["GSD"] == f"{expected['gsd_cm']:.2f} cm/px"
        assert table["Footprint"] == f"{expected['footprint_m'][0]} × {expected['footprint_m'][1]} m"
        browser.js("document.querySelector('.modal .modal-actions .tbtn.primary').click(); true")


# ================================================================= map behaviour

class TestTheMap:
    def test_the_waypoint_card_is_the_plans_waypoint(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        wait_map(browser)
        rows = overview(stack)["plan"]["waypoints"]
        row = rows[len(rows) // 2]
        x, y = map_point(browser, row["lon"], row["lat"])
        browser._send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        browser.wait_for("!document.querySelector('.odk-plan-card').hidden", 20)
        card = text(browser, ".odk-plan-card")
        assert f"WP {row['index']:03d}" in card or "WP " in card
        shown = browser.js("document.querySelector('.odk-plan-card-id').textContent")
        index = int(shown.split()[-1])
        hit = next(r for r in rows if r["index"] == index)
        assert f"{abs(hit['lat']):.5f}° N" in card and f"{abs(hit['lon']):.5f}° W" in card
        assert f"{hit['alt_m']:.1f} m rel. home" in card
        assert f"{hit['footprint_m'][0]:.1f} × {hit['footprint_m'][1]:.1f} m" in card
        # START / END labels name the first and last waypoint.
        tags = browser.js("[...document.querySelectorAll('.odk-plan-endtag')].map((n) => n.textContent)")
        assert tags == [f"START · WP {rows[0]['index']:03d}", f"END · WP {rows[-1]['index']:03d}"]
        # The footprints on the map are the Api's.
        drawn = browser.js("odkShell.actionContext().map().getSource('odk-plan-footprints')._data.features.length")
        assert drawn == len(overview(stack)["plan"]["footprints"]["features"]) > 0

    def test_vertex_editing_and_undo_go_through_set_aoi(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        before = [list(p) for p in api._session.aoi_polygon]
        browser = boot(stack)
        wait_map(browser)
        assert "drag a corner" in (run(browser, "Edit Vertices").get("message") or "")
        assert browser.js("document.querySelector('.odk-plan-tools [data-action=\"Edit Vertices\"]').getAttribute('aria-pressed')") == "true"
        corner = before[0]
        start = map_point(browser, *corner)
        drag(browser, start, (start[0] - 40, start[1] - 30))
        deadline = time.monotonic() + 30
        while api._session.aoi_polygon[0] == before[0] and time.monotonic() < deadline:
            time.sleep(0.2)
        moved = api._session.aoi_polygon[0]
        assert moved != before[0], "the dragged corner did not reach set_aoi"
        assert moved[0] < corner[0] and moved[1] > corner[1]  # west and north, as dragged
        assert api.mission_overview()["aoi"]["source"].startswith("edited")
        refresh(browser)
        assert "Changed since this plan" in text(browser, "[data-plan-status]")
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); true")
        # Undo puts the previous area back, through set_aoi.
        assert browser.js("document.querySelector('[data-action=\"Undo Area Edit\"]').disabled") is False
        result = run(browser, "Undo Area Edit")
        assert "restored" in (result.get("message") or ""), result
        assert [list(p) for p in api._session.aoi_polygon] == before

    def test_rotate_flight_lines_replans_along_the_drawn_heading(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        wait_map(browser)
        ring = api._session.aoi_polygon
        lon0 = sum(p[0] for p in ring) / len(ring)
        lat0 = sum(p[1] for p in ring) / len(ring)
        # A line heading 40 degrees east of north, 60 m long.
        d = 60.0
        dlat = d * math.cos(math.radians(40)) / 111_320
        dlon = d * math.sin(math.radians(40)) / (111_320 * math.cos(math.radians(lat0)))
        assert "Rotate flight lines" in (run(browser, "Rotate Flight Lines").get("message") or "")
        browser.js(f"odkShell.drawing.addVertex([{lon0}, {lat0}]); odkShell.drawing.addVertex([{lon0 + dlon}, {lat0 + dlat}]);"
                   " odkShell.drawing.finishDrawing(); true")
        deadline = time.monotonic() + 120
        bearing = None
        while time.monotonic() < deadline:
            plan = api.mission_overview()["plan"]
            bearing = plan and plan["line_bearing_deg"]
            if bearing is not None and abs(((bearing - 40) + 90) % 180 - 90) < 3:
                break
            time.sleep(0.5)
        assert bearing is not None and abs(((bearing - 40) + 90) % 180 - 90) < 3, bearing

    def test_the_two_way_gsd_is_the_apis(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        set_value = ("(k, v) => { const i = document.querySelector(`input[data-field=\"${k}\"]`); i.value = v;"
                     " i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); return true; }")
        browser.js(f"({set_value})('gsd_cm', '2.0')")
        solved = api.altitude_for_gsd("mavic2pro", 2.0)["altitude_m"]
        browser.wait_for(f"document.querySelector('input[data-field=\"altitude_m\"]').value === '{solved}'", 20)
        browser.js(f"({set_value})('altitude_m', '70')")
        gsd = api.describe_camera("mavic2pro", 70)["camera"]["gsd_cm"]
        browser.wait_for(f"document.querySelector('input[data-field=\"gsd_cm\"]').value === '{gsd:.2f}'", 20)
        # Generate sends the solved altitude.
        assert "Planned" in (run(browser, "Plan").get("message") or "")
        assert api.mission_overview()["plan"]["altitude_m"] == pytest.approx(70.0)


# ============================================================ every placement

class TestEveryPlacementRenders:
    def test_every_verb_in_surface_js_is_rendered_where_it_says(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        assert api.add_site_marker("Mast", "hazard", [api._session.aoi_polygon[0]], "", 5.0)["ok"]
        api.export_mission(["qgc_plan"])
        browser = boot(stack)
        wait_map(browser)
        surface = browser.js("import('./js/odk/screens/plan/surface.js').then((m) => ({ p: m.PLACEMENTS, c: m.CONTROLS }))")
        registry = browser.js("import('./js/odk/registry.js').then((m) => Object.keys(m.ACTIONS))")
        missing = []
        regions = {"left": "[data-region=left]", "main": "[data-region=main]", "inspector": "[data-region=inspector]"}
        tabs = {"sorties": "segments", "site": "site"}
        for entry in [p for p in surface["p"] if "desk" in p.get("modes", ["desk"])]:
            assert entry["action"] in registry, entry
            part = entry["where"].split(".")
            region = regions[part[1]]
            if part[1] == "inspector":
                tab = tabs.get(part[2], "plan")
                browser.js(f"document.querySelector('[data-control=\"panel-tab:{tab}\"]').click(); true")
                settle(browser)
            if entry.get("via"):
                anchor = browser.js(f"!!document.querySelector('{region} [data-control=\"{entry['via']}\"]')")
                if not anchor:
                    missing.append((entry["action"], entry["where"], "anchor " + entry["via"]))
                    continue
                browser.js(f"document.querySelector('{region} [data-control=\"{entry['via']}\"]').click(); true")
                time.sleep(0.3)
                found = browser.js(f"!!document.querySelector('.odk-popover:not([hidden]) [data-action=\"{entry['action']}\"]')")
                browser.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); true")
            else:
                found = browser.js(f"!!document.querySelector('{region} [data-action=\"{entry['action']}\"]')")
            if not found:
                missing.append((entry["action"], entry["where"]))
        browser.js("document.querySelector('[data-control=\"panel-tab:plan\"]').click(); true")
        assert not missing, missing
        # Every Plan row of the inventory has a home on this screen.
        inventory = json.loads((REPO / "docs" / "ui" / "inventory.json").read_text(encoding="utf-8"))
        rows = {r["id"] for r in inventory["rows"] if r["workspace"] == "plan"}
        placed = {p.get("inventory") for p in surface["p"]} | {c.get("inventory") for c in surface["c"]}
        assert rows <= placed, sorted(rows - placed)
        # Every button on the screen is marked (data-action or data-control).
        unmarked = browser.js("""[...document.querySelectorAll('.odk-frame button, [data-plan-menu] button')]
            .filter((b) => !b.hasAttribute('data-action') && !b.hasAttribute('data-control'))
            .map((b) => b.outerHTML.slice(0, 120))""")
        assert unmarked == []

    def test_controls_render_where_they_say(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack)
        controls = browser.js("import('./js/odk/screens/plan/surface.js').then((m) => m.CONTROLS)")
        for entry in controls:
            if "desk" not in entry.get("modes", []):
                continue
            if entry.get("template"):
                browser.js(f"document.querySelector('[data-template=\"{entry['template']}\"]') ? "
                           f"document.querySelector('[data-template=\"{entry['template']}\"]').click() : "
                           "document.querySelector('[data-control=\"templates:more\"]').click(); true")
            if "field" in entry:
                selector = f"[data-region=inspector] input[data-field=\"{entry['field']}\"]"
                if entry["field"] == "rth_altitude_m":
                    browser.js("document.querySelector('[data-control=\"advanced\"]').click(); true")
            else:
                control = entry["control"]
                selector = (f"[data-control^=\"{control[:-1]}\"]" if control.endswith("*")
                            else f"[data-control=\"{control}\"]")
            browser.wait_for(f"!!document.querySelector({json.dumps(selector)})", 30)
        # Back to the plan's own type.
        browser.js("document.querySelector('[data-template=\"grid\"]').click(); true")


# ======================================================================= Field

class TestFieldPlan:
    def test_field_shows_the_plan_and_the_stepper_replans(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro"})["ok"]
        browser = boot(stack, "field", (1180, 820))
        view = overview(stack)["plan"]
        band = text(browser, ".odk-planf-band")
        assert f"{view['waypoint_count']:,}" in band and f"{view['gsd_cm']:.2f}" in band
        assert f"{round(view['distance_m']):,}" in band
        assert text(browser, ".odk-planf-alt-value") == "55"
        # Every target on the Field side panel is at least 64 px.
        small = browser.js("""[...document.querySelectorAll('.odk-planf-panel button, .odk-planf-edit')]
          .map((b) => [b.dataset.control || b.dataset.action, b.getBoundingClientRect().height])
          .filter(([, h]) => h < 64)""")
        assert small == [], small
        browser.js("document.querySelector('[data-control=\"altitude:up\"]').click(); true")
        deadline = time.monotonic() + 120
        while api.mission_overview()["plan"]["altitude_m"] != pytest.approx(60.0) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert api.mission_overview()["plan"]["altitude_m"] == pytest.approx(60.0)
        refresh(browser)
        assert text(browser, ".odk-planf-alt-value") == "60"
        # Send to aircraft hands the plan to Fly.
        browser.js("document.querySelector('.odk-planf-panel [data-action=\"Send to Fly\"]').click(); true")
        browser.wait_for("document.querySelector('.odk-frame').dataset.rendered === 'fly'", 60)
        assert browser.js("!!(window.odkPlanHandover && window.odkPlanHandover.uploadArmed)") is True
        browser.js("odkShell.setMode('desk').then(() => true)")


# ================================================================== screenshots

class TestScreenshots:
    def test_desk_and_field_against_the_artboards(self, stack) -> None:
        api = stack["api"]
        set_site(stack)
        assert api.plan_mission({"template": "grid", "altitude_m": 55, "camera": "mavic2pro", "line_bearing_deg": 75})["ok"]
        SHOTS.mkdir(parents=True, exist_ok=True)
        browser = boot(stack)
        wait_map(browser)
        rows = overview(stack)["plan"]["waypoints"]
        row = rows[len(rows) // 2]
        x, y = map_point(browser, row["lon"], row["lat"])
        browser._send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        time.sleep(0.6)
        browser.screenshot(SHOTS / "plan-desk-1440x900.png")
        browser.viewport(1100, 700)
        time.sleep(1.0)
        browser.screenshot(SHOTS / "plan-desk-1100x700.png")
        browser = boot(stack, "field", (1180, 820))
        wait_map(browser)
        browser.screenshot(SHOTS / "plan-field-1180x820.png")
        browser.js("odkShell.setMode('desk').then(() => true)")
        for name in ("plan-desk-1440x900.png", "plan-field-1180x820.png"):
            assert (SHOTS / name).stat().st_size > 50_000
