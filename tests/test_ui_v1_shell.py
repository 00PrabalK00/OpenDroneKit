"""The v1 shell, in a real browser, against the real Api (U2).

odk.html is served by tests/ui_live_bridge.py with the production Api behind it -- only
pywebview's transport is replaced -- and driven in headless Edge/Chromium. The checks are
the frozen shell contract (docs/ui/ARCHITECTURE.md) and the frame's claim to show only
what the Api reported:

  * new Shell(root): open(id) for every new and every old cockpit id, runAction(key),
    actionContext(), toast(), confirmAll, refreshState(), setMode();
  * the footer reads the project CRS, the jobs and the aircraft from the Api, and says
    "local · no network" only while no request has left the machine;
  * the command palette lists every ACTIONS key and searches places through
    search_places (the offline provider: nothing leaves the machine in a test);
  * Field mode: 72 px header, 88 px tab bar, 64 px targets, persisted;
  * panel collapse persists across a reload;
  * screenshots of the desk and Field shells at 1440x900 and 1180x820 are written to
    $ODK_UI_SHOTS (default: <tmp>/odk-ui-shots) for comparison with the artboards.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websocket")
pytest.importorskip("requests")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from ui_live_bridge import Browser, chromium_path, serve  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]
NEW_IDS = ["home", "plan", "fly", "process", "analyse", "review", "fleet", "report", "settings"]
OLD_IDS = {"planning": "plan", "flight": "fly", "processing": "process", "verification": "process",
           "measurements": "analyse", "twin": "analyse", "thermal": "analyse", "inspection": "review",
           "reports": "report", "projects": "home", "developers": "settings"}
SHOTS = Path(os.environ.get("ODK_UI_SHOTS") or Path(tempfile.gettempdir()) / "odk-ui-shots")

WATCH = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
    if (n.classList && n.classList.contains('odk-toast')) window.__toasts.push(n.textContent);
  })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
true
"""


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    tmp = tmp_path_factory.mktemp("shell")
    cwd = os.getcwd()
    os.chdir(tmp)
    api = Api(AppSession(ProjectStore(tmp / "projects.db")))
    api.create_project("Aukerman Park", str(tmp / "project"))
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    api.set_geocoding_provider("offline")  # a test never geocodes over the network
    server, base = serve(api)
    browser = Browser(executable)
    try:
        browser.set_viewport(1440, 900)
        yield api, browser, base
    finally:
        browser.close()
        server.shutdown()
        api.disconnect_vehicle()
        os.chdir(cwd)


def boot(browser: Browser, base: str, query: str = "mode=desk") -> None:
    browser.open(f"{base}/odk.html?{query}", settle_s=0.2)
    browser.wait_for("window.__odkReady === true || window.__odkBootError", 30)
    assert browser.js("window.__odkBootError || null") is None, browser.js("window.__odkBootError")
    # Connected: the bridge answered and the footer says where things are.
    browser.wait_for("odkShell.connection === 'connected' && !!odkShell.state", 20)
    browser.js(WATCH)


def footer(browser: Browser) -> dict:
    return browser.js("""Object.fromEntries([...document.querySelectorAll('[data-footer]')]
        .map((n) => [n.dataset.footer, n.hidden ? null : n.textContent.trim()]))""")


class TestTheShellContract:
    def test_the_members_are_there(self, stack) -> None:
        api, browser, base = stack
        boot(browser, base)
        members = browser.js("""(() => {
          const s = odkShell;
          const ctx = s.actionContext();
          return {
            fns: ['open', 'runAction', 'actionContext', 'toast', 'refreshState', 'setMode']
              .filter((m) => typeof s[m] !== 'function'),
            confirmAll: typeof s.confirmAll,
            drawing: ['addVertex', 'finishDrawing', 'canDraw'].filter((m) => typeof s.drawing[m] !== 'function'),
            ctx: ['prompt', 'confirm', 'choose', 'toast', 'refresh', 'open', 'resetLayout', 'map', 'selection',
                  'missionOptions', 'reconstructionOptions', 'reportOptions', 'requireSelection', 'pickPoint',
                  'pickImagePixel', 'showImage', 'workspaceId', 'openWorkspace', 'organizationId', 'projectId']
              .filter((m) => !(m in ctx)),
            menu: typeof (window.odk && window.odk.onMenu),
          };
        })()""")
        assert members == {"fns": [], "confirmAll": "boolean", "drawing": [], "ctx": [], "menu": "function"}

    def test_open_accepts_every_new_and_old_id(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        for wid in NEW_IDS + list(OLD_IDS):
            want = OLD_IDS.get(wid, wid)
            opened = browser.js(f"odkShell.open({wid!r})")
            assert opened is True, wid
            state = browser.js("""({ rendered: document.querySelector('.odk-frame').dataset.rendered,
                current: odkShell.workspaceId,
                tab: (document.querySelector('[data-control^="tab:"][aria-current=page]') || {}).dataset?.control || null,
                gear: document.querySelector('[data-control="open:settings"]').getAttribute('aria-current') })""")
            assert state["rendered"] == want and state["current"] == want, (wid, state)
            if want == "settings":
                assert state["gear"] == "page" and state["tab"] is None
            else:
                assert state["tab"] == f"tab:{want}", (wid, state)
        # An old id is what actions that tell two old screens apart are told.
        assert browser.js("odkShell.open('verification').then(() => odkShell.actionContext().workspaceId())") \
            == "verification"
        assert browser.js("odkShell.open('nowhere')") is False

    def test_run_action_resolves_and_never_throws(self, stack) -> None:
        api, browser, base = stack
        boot(browser, base)
        assert browser.js("odkShell.runAction('No Such Verb')") == {"error": "unknown action No Such Verb"}
        view = browser.js("odkShell.runAction('Map')")
        assert view.get("view") == "map"
        # A real verb through the real Api, answered with what verify_site said.
        result = browser.js("odkShell.runAction('Validate')")
        expected = api.verify_site()
        said = result.get("message") or result.get("error") or ""
        assert said and said in {expected.get("summary"), expected.get("error"), "Site checked."}, (result, expected)
        node = browser.js("(() => { const n = odkShell.toast('Contract: a toast', 'warn'); "
                          "return [n.className, n.textContent]; })()")
        assert "odk-toast--warn" in node[0] and "Contract" in node[1]
        # confirmAll answers ctx.confirm() without a dialog (tests only).
        assert browser.js("odkShell.confirmAll = true, odkShell.actionContext().confirm('Sure?')") is True
        assert browser.js("!!document.querySelector('.modal-backdrop')") is False
        browser.js("odkShell.confirmAll = false; true")

    def test_every_button_in_the_frame_is_marked(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        unmarked = browser.js("""[...document.querySelectorAll('.odk-frame button, .odk-frame a[href]')]
            .filter((b) => !b.hasAttribute('data-action') && !b.hasAttribute('data-control'))
            .map((b) => b.outerHTML.slice(0, 100))""")
        assert unmarked == []


class TestTheFooterTellsTheTruth:
    def test_project_crs_network_and_jobs_come_from_the_api(self, stack) -> None:
        api, browser, base = stack
        boot(browser, base)
        f = footer(browser)
        assert f["net"] == "local · no network"
        assert f["crs"] == "CRS auto · UTM from imagery"
        assert f["cursor"] == "cursor · no map on this screen"
        name = api.get_state()["project"]["name"]
        assert browser.js("document.querySelector('.odk-header-project-name').textContent") == name
        # "· OFFLINE" beside the project exactly while the network state says no network.
        assert browser.js("""(() => { const n = document.querySelector('.odk-header-project-net');
          return n.hidden ? null : n.textContent; })()""") == "· OFFLINE"

        assert api.set_project_crs(32617)["ok"]
        browser.js("odkShell.refreshState().then(() => true)")
        assert footer(browser)["crs"] == "EPSG:32617 · UTM 17N"

        # A job the Api ran shows up in the footer, numbered among the jobs it knows.
        job = api._jobs.submit("u2_probe", lambda progress, should_cancel: {"ok": True})
        deadline = time.monotonic() + 10
        while api.job_status(job)["job"]["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.1)
        total = len(api.list_jobs()["jobs"])
        browser.js("odkShell.refreshState().then(() => true)")
        assert footer(browser)["jobs"] == f"job {total} of {total} · complete"

    def test_the_vehicle_segment_follows_telemetry(self, stack) -> None:
        api, browser, base = stack
        boot(browser, base)
        browser.js("odkShell.open('fly')")
        assert footer(browser)["vehicle"] == "no vehicle connected"
        assert api.connect_vehicle("mock://", "mock")["ok"]
        try:
            browser.js("odkShell.refreshState().then(() => true)")
            telemetry = api.telemetry()["telemetry"]
            text = footer(browser)["vehicle"]
            assert text.startswith(f"driver mock · {telemetry['flight_mode']}"), text
            # Tagged because the Api said so (is_simulated), not because the UI guessed.
            assert telemetry["is_simulated"] is True and text.endswith("SIMULATED")
        finally:
            api.disconnect_vehicle()
            browser.js("odkShell.refreshState().then(() => true)")

    def test_a_request_off_the_machine_is_said(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("odkShell.noteNetworkUse('test probe'); true")
        assert footer(browser)["net"] == "local · network in use (1)"
        # Many hosts: the label keeps its length; the hosts go in the tooltip and the log.
        hosts = [f"tiles{i}.example-provider-with-a-long-name.test" for i in range(6)]
        browser.js(f"{json.dumps(hosts)}.forEach((h) => odkShell.noteNetworkUse(h)); true")
        seen = browser.js("""(() => { const n = document.querySelector('[data-footer=net]');
          const t = n.querySelector('.odk-net-text');
          return { text: n.textContent.trim(), title: n.title, tone: n.dataset.tone,
                   dot: Boolean(n.querySelector('.odk-dot--warn')), clipped: t.scrollWidth > t.clientWidth + 1,
                   log: odkShell.log.map((e) => e.text),
                   offline: document.querySelector('.odk-header-project-net').hidden }; })()""")
        assert seen["text"] == "local · network in use (7)" and seen["tone"] == "warn" and seen["dot"], seen
        assert not seen["clipped"], seen
        assert all(h in seen["title"] for h in ["test probe", *hosts]), seen["title"]
        assert any(hosts[-1] in line and "7 this session" in line for line in seen["log"]), seen["log"]
        # No longer offline, so the project label stops saying so.
        assert seen["offline"] is True
        browser.js("odkShell.setMode('field')")
        assert browser.js("document.querySelector('.odk-field-net').textContent.trim()") == "Network in use · 7"
        browser.js("odkShell.setMode('desk')")

    def test_elev_reads_the_dsm_under_the_cursor_or_says_there_is_none(self, stack, tmp_path) -> None:
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        from core import geo

        api, browser, base = stack
        boot(browser, base)
        lon, lat = -81.7500, 41.3045
        browser.js("odkShell.open('analyse')")
        browser.js(f"""(async () => {{
          const f = await import('./js/odk/map/furniture.js');
          const m = await f.mountProjectMap(document.querySelector('[data-region=main]'), {{ centre: [{lon}, {lat}], zoom: 17 }});
          window.__m = m;
          return true;
        }})()""", timeout_s=60)
        browser.wait_for("odkShell.boundMap === window.__m.map", 15)
        move = f"window.__m.map.fire('mousemove', {{ lngLat: {{ lng: {lon}, lat: {lat} }} }}); true"
        elev = "(() => { const n = document.querySelector('[data-footer=elev]'); return n.hidden ? null : [n.textContent, n.title]; })()"

        # No DSM in this project: a dash and the reason, never a number.
        browser.js(move)
        browser.wait_for(f"({elev}) && ({elev})[1].includes('No DSM')", 15)
        assert browser.js(elev)[0] == "ELEV —"
        order = browser.js("[...document.querySelectorAll('[data-footer]')].map((n) => n.dataset.footer)")
        assert order.index("cursor") + 1 == order.index("elev") and order.index("elev") + 1 == order.index("crs"), order

        # A DSM: its own pixel under the cursor.
        east, north = geo.wgs84_to_projected([lon], [lat], 32617)
        z = (230.0 + np.arange(40 * 40, dtype="float32").reshape(40, 40) * 0.013).astype("float32")
        z[:10, :10] = -9999.0
        path = tmp_path / "dsm.tif"
        transform = from_origin(float(east[0]) - 20.0, float(north[0]) + 20.0, 1.0, 1.0)
        with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=1, dtype="float32",
                           crs="EPSG:32617", transform=transform, nodata=-9999.0) as dst:
            dst.write(z, 1)
        layer = api.add_layer_from_file(str(path))["layer"]["id"]
        try:
            with rasterio.open(path) as src:
                row, col = src.index(float(east[0]), float(north[0]))
                expected = float(src.read(1)[row, col])
            browser.js("odkShell.refreshState().then(() => true)")
            browser.js(move)
            browser.wait_for(f"({elev}) && ({elev})[0] !== 'ELEV —'", 15)
            text, title = browser.js(elev)
            assert text == f"ELEV {expected:.2f} m", (text, expected)
            assert "EPSG:32617" in title and "GeoTIFF" in title, title
            # A nodata pixel: a dash, not the nodata value.
            nx, ny = rasterio.transform.xy(transform, 2, 2)
            nlon, nlat = geo.projected_to_wgs84([nx], [ny], 32617)
            browser.js(f"window.__m.map.fire('mousemove', {{ lngLat: {{ lng: {float(nlon[0])}, lat: {float(nlat[0])} }} }}); true")
            browser.wait_for(f"({elev}) && ({elev})[0] === 'ELEV —'", 15)
            assert "no data" in browser.js(elev)[1]
        finally:
            api.remove_layer(layer)


class TestThePalette:
    def test_every_action_is_in_it_and_it_runs(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'k', ctrlKey: true})); true")
        browser.wait_for("document.querySelector('[data-palette]') !== null")
        listed = browser.js("""import('./js/odk/registry.js').then((r) => {
          const shown = new Set([...document.querySelectorAll('[data-palette] [data-action]')].map((n) => n.dataset.action));
          return { total: r.actionKeys().length, missing: r.actionKeys().filter((k) => !shown.has(k)) };
        })""")
        assert listed["total"] > 100 and listed["missing"] == [], listed
        groups = browser.js("[...document.querySelectorAll('.odk-palette-group')].map((g) => g.textContent)")
        assert groups[:2] == ["Workspaces", "Projects"] and "Actions" in groups and "Shortcuts" in groups

        # Typing filters; Enter runs the highlighted row (a workspace).
        browser.js("""(() => { const i = document.querySelector('.odk-palette-input');
          i.value = 'Fleet'; i.dispatchEvent(new Event('input')); return true; })()""")
        first = browser.js("document.querySelector('.odk-palette-row[aria-selected=true]').dataset.control")
        assert first == "palette:open:fleet"
        browser.js("document.querySelector('.odk-palette-input').dispatchEvent("
                   "new KeyboardEvent('keydown', {key: 'Enter', bubbles: true})); true")
        browser.wait_for("odkShell.workspaceId === 'fleet' && !document.querySelector('[data-palette]')")

    def test_places_come_from_search_places(self, stack) -> None:
        api, browser, base = stack
        boot(browser, base)
        query = "41.3045, -81.7500"
        expected = api.search_places(query)["results"][0]["name"]
        browser.js(f"odkShell.openPalette({query!r}).then(() => true)")
        browser.js("document.querySelector('[data-control=\"palette:search-places\"]').click(); true")
        browser.wait_for("document.querySelector('[data-control=\"palette:place\"]') !== null", 15)
        assert browser.js("document.querySelector('[data-control=\"palette:place\"] .odk-palette-label')"
                          ".textContent") == expected
        # The offline provider answered: the footer still says no network.
        assert footer(browser)["net"] == "local · no network"
        browser.js("document.querySelector('[data-control=\"palette:place\"]').click(); true")
        assert expected in browser.wait_for("__toasts.find((t) => t.includes('°'))")

    def test_slash_opens_search_and_question_mark_the_shortcuts(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: '/'})); true")
        browser.wait_for("document.activeElement && document.activeElement.classList.contains('odk-palette-input')")
        browser.js("odkShell.palette.close(); document.dispatchEvent(new KeyboardEvent('keydown', {key: '?'})); true")
        browser.wait_for("document.querySelector('[data-shortcuts]') !== null")
        browser.js("document.querySelector('[data-control=close-shortcuts]').click(); true")
        # Digits switch workspace.
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: '3'})); true")
        browser.wait_for("odkShell.workspaceId === 'fly'")


class TestFieldMode:
    def test_the_field_shell(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("odkShell.open('plan')")
        browser.js("odkShell.setMode('field')")
        shape = browser.js("""(() => {
          const h = (s) => getComputedStyle(document.querySelector(s)).height;
          const small = [...document.querySelectorAll('.odk-frame button, .odk-frame a[href]')]
            .filter((n) => n.offsetParent !== null)
            .filter((n) => { const r = n.getBoundingClientRect(); return r.width < 64 || r.height < 64; })
            .map((n) => n.dataset.control || n.dataset.action);
          return { mode: document.documentElement.dataset.mode, stored: localStorage.getItem('odk.mode'),
                   frame: document.querySelector('.odk-frame').dataset.frame,
                   header: h('.odk-frame-header'), tabbar: h('.odk-field-tabbar'),
                   tabs: [...document.querySelectorAll('.odk-field-tab')].map((t) => t.textContent.trim()),
                   active: document.querySelector('.odk-field-tab[aria-current=page]').textContent.trim(),
                   net: document.querySelector('.odk-field-net').textContent.trim(),
                   footer: getComputedStyle(document.querySelector('.odk-frame-header')).backgroundColor,
                   small };
        })()""")
        assert shape == {"mode": "field", "stored": "field", "frame": "field", "header": "72px", "tabbar": "88px",
                         "tabs": ["Plan", "Fly", "Process", "Analyse", "Report"], "active": "Plan",
                         "net": "Local · no network", "footer": "rgb(10, 10, 10)", "small": []}
        # The palette stays available in Field.
        browser.js("odkShell.openPalette().then(() => true)")
        assert browser.js("parseFloat(getComputedStyle(document.querySelector('.odk-palette-row')).height)") >= 64
        browser.js("odkShell.palette.close(); true")
        # A screen with no Field renderer opens in Desk.
        browser.js("odkShell.open('fleet')")
        assert browser.js("[document.documentElement.dataset.mode, odkShell.workspaceId]") == ["desk", "fleet"]

    def test_the_menu_toggles_it(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("window.odk.onMenu({action: 'view.field_mode'})")
        assert browser.js("document.documentElement.dataset.mode") == "field"
        browser.js("window.odk.onMenu({action: 'view.field_mode'})")
        assert browser.js("document.documentElement.dataset.mode") == "desk"


class TestPanels:
    def test_collapse_persists(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        browser.js("localStorage.removeItem('odk.panel.left'); localStorage.removeItem('odk.panel.inspector'); true")
        boot(browser, base)
        browser.js("document.querySelector('[data-control=\"collapse:left\"]').click(); true")
        assert browser.js("document.querySelector('.odk-frame').dataset.left") == "collapsed"
        boot(browser, base)
        state = browser.js("""({ left: document.querySelector('.odk-frame').dataset.left || null,
            width: getComputedStyle(document.querySelector('.odk-frame-left')).width,
            expanded: document.querySelector('[data-control="collapse:left"]').getAttribute('aria-expanded') })""")
        assert state == {"left": "collapsed", "width": "0px", "expanded": "false"}
        # Ctrl+B: both panels, together.
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'b', ctrlKey: true})); true")
        assert browser.js("document.querySelector('.odk-frame').dataset.inspector") == "collapsed"
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'b', ctrlKey: true})); true")
        assert browser.js("[document.querySelector('.odk-frame').dataset.left || null, "
                          "document.querySelector('.odk-frame').dataset.inspector || null]") == [None, None]


class TestSelectionIsATone:
    """Selected rows are a fill, not a coloured stripe down one side (craft floor)."""

    CSS = Path(__file__).resolve().parents[1] / "app" / "web" / "css" / "odk"

    def test_no_stylesheet_draws_a_side_stripe(self) -> None:
        stripe = re.compile(r"box-shadow:\s*inset\s+-?[1-9]\d*px\s+0\s+0|border-(left|right):\s*[2-9]\d*px")
        found = [(css.name, n, line.strip()) for css in self.CSS.glob("*.css")
                 for n, line in enumerate(css.read_text(encoding="utf-8").splitlines(), 1) if stripe.search(line)]
        assert found == [], found

    def test_a_selected_row_is_a_fill_in_both_modes(self, stack) -> None:
        _, browser, base = stack
        boot(browser, base)
        row = """(async () => {
          await odkShell.openPalette();
          const r = document.querySelector('.odk-palette-row[aria-selected=true]');
          const cs = getComputedStyle(r);
          const out = { shadow: cs.boxShadow, bg: cs.backgroundColor, color: cs.color };
          odkShell.palette.close();
          return out;
        })()"""
        desk = browser.js(row)
        assert desk == {"shadow": "none", "bg": "rgb(24, 32, 43)", "color": "rgb(230, 234, 240)"}, desk
        browser.js("odkShell.setMode('field')")
        field = browser.js(row)
        assert field == {"shadow": "none", "bg": "rgb(0, 71, 179)", "color": "rgb(255, 255, 255)"}, field
        browser.js("odkShell.setMode('desk')")


class TestTheDemoBanner:
    def test_demo_mode_shows_the_banner(self, stack) -> None:
        _, browser, base = stack
        browser.open(f"{base}/odk.html?mode=desk&demo=1", settle_s=0.2)
        browser.wait_for("window.__odkReady === true || window.__odkBootError", 30)
        browser.wait_for("!document.querySelector('.odk-banner').hidden", 10)
        assert "EXAMPLE DATA" in browser.js("document.querySelector('.odk-banner').textContent")
        assert browser.js("document.querySelector('[data-footer=net]').textContent") == "example data · no network"


class TestScreenshots:
    """Desk and Field shells at the artboard sizes, for the owner's comparison."""

    def test_capture_the_shells(self, stack) -> None:
        _, browser, base = stack
        for width, height in ((1440, 900), (1180, 820)):
            browser.set_viewport(width, height)
            boot(browser, base)
            browser.js("odkShell.open('home')")
            browser.js("document.querySelectorAll('.odk-toast').forEach((t) => t.remove()); true")
            assert browser.screenshot(SHOTS / f"u2-desk-{width}x{height}.png").stat().st_size > 10_000
            # The header fits: nothing in it is pushed past the window edge.
            overflow = browser.js("""[...document.querySelectorAll('.odk-frame-header > *')]
                .filter((n) => n.getBoundingClientRect().right > window.innerWidth + 0.5).map((n) => n.className)""")
            assert overflow == [], overflow
            browser.js("odkShell.setMode('field')")
            browser.js("document.querySelectorAll('.odk-toast').forEach((t) => t.remove()); true")
            assert browser.screenshot(SHOTS / f"u2-field-{width}x{height}.png").stat().st_size > 10_000
            browser.js("odkShell.setMode('desk')")
        browser.set_viewport(1440, 900)
        (SHOTS / "u2-shots.json").write_text(json.dumps(sorted(p.name for p in SHOTS.glob("u2-*.png"))),
                                             encoding="utf-8")
