"""U4 Home, in a real browser, against the real Api (REBUILD_PLAN section 3.4).

odk.html is served with the production Api behind it -- only pywebview's transport and
the two native dialogs (pick_folder, open_path) are replaced -- plus the raster-tile
routes the desktop shell serves, so the Home map draws the project's own orthomosaic.
Driven in headless Edge/Chromium. What is checked:

  * every figure on Home is the Api's: the PHOTOS / SURVEY DATE tiles equal
    dataset_summary() (77 photos, 77 geotagged, the EXIF capture window), the
    reconstruction table equals reconstruction_summary(), the exports list equals
    list_exports(), STORAGE equals project_storage(), the project list equals
    home_projects();
  * every placement in screens/home/surface.js renders data-action="<key>" inside its
    [data-where] region, and every button on the screen is marked;
  * the acceptance path: New project -> Import the Aukerman photos -> the PHOTOS tile
    reads 77 from the Api -> Share -> Revoke, each through the screen's own buttons;
  * notifications: a finished job is announced, Mark read and Clear reach the Api;
  * screenshots at desk 1440x900 and at the Field window size 1180x820 go to
    $ODK_U4_SHOTS (Home has no Field screen: opened in Field mode it switches to Desk).

The Aukerman photographs come from ODK_TESTDATA_DIR (odm_aukerman/images) and the
reconstruction from ODK_AUKERMAN_FULLRES; the test skips when either is absent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("websocket")
pytest.importorskip("requests")

from ui_live_bridge import BRIDGE_JS, Browser, chromium_path  # noqa: E402
from ui_v1_map_harness import MapServer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "app" / "web"
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", REPO / "testdata"))
IMAGES = TESTDATA / "odm_aukerman" / "images"
RECON = Path(os.environ.get("ODK_AUKERMAN_FULLRES", REPO / "demo_project" / "aukerman_fullres"))
if (RECON / "reconstruction").is_dir():
    RECON = RECON / "reconstruction"
SHOTS = Path(os.environ.get("ODK_U4_SHOTS") or Path(tempfile.gettempdir()) / "odk-u4-home")

needs_data = pytest.mark.skipif(
    not IMAGES.is_dir() or not (RECON / "orthomosaic.tif").is_file(),
    reason="The Aukerman photographs (ODK_TESTDATA_DIR) or reconstruction (ODK_AUKERMAN_FULLRES) are absent.")
needs_browser = pytest.mark.skipif(chromium_path() is None, reason="No Chromium or Edge to drive.")
pytestmark = [needs_data, needs_browser]

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


def _api_class():
    from app.api import Api

    class DialogApi(Api):
        """The production Api with its two native dialogs answered by the test."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.folders: list[str] = []
            self.opened: list[str] = []

        def pick_folder(self):
            return {"ok": True, "path": self.folders.pop(0) if self.folders else ""}

        def open_path(self, path):
            if not Path(path).exists():
                return {"ok": False, "error": f"Path not found: {path}"}
            self.opened.append(str(path))
            return {"ok": True, "opened": str(path)}

    return DialogApi


def _page() -> bytes:
    html = (WEB / "odk.html").read_bytes()
    return html.replace(b"<head>", b'<head><script src="/__odk_bridge.js"></script>', 1)


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    from app.session import AppSession
    from app.store import ProjectStore

    root = tmp_path_factory.mktemp("u4-home")
    cwd = os.getcwd()
    os.chdir(root)
    os.environ.setdefault("ODK_STORAGE_PATH", str(root / "storage"))
    api = _api_class()(AppSession(ProjectStore(root / "projects.db")))
    assert api.create_project("Aukerman Park survey", str(root / "aukerman"))["ok"]
    # The reconstruction the engine wrote, where run_reconstruction writes it.
    recon = root / "aukerman" / "reconstruction"
    recon.mkdir(parents=True)
    for item in RECON.iterdir():
        if item.is_file():
            shutil.copy2(item, recon / item.name)
    api._session.restore_layers()
    assert api.import_dataset(str(IMAGES))["ok"]
    assert api.set_active_dataset(str(IMAGES))["ok"]
    # A boundary and a saved, exported grid mission over the middle of the site.
    centre = api.dataset_summary()["active"]["center_lonlat"]
    lon0, lat0 = centre
    d_lon, d_lat = 0.0012, 0.0006
    ring = [[lon0 - d_lon, lat0 - d_lat], [lon0 + d_lon, lat0 - d_lat], [lon0 + d_lon, lat0 + d_lat],
            [lon0 - d_lon, lat0 + d_lat]]
    assert api.set_aoi(ring)["ok"]
    assert api.plan_mission({"template": "grid", "altitude_m": 55})["ok"]
    assert api.save_mission("Grid")["ok"]
    assert api.export_mission()["ok"]
    assert api.set_project_crs(32617)["ok"]
    report = api.generate_report("", "Aukerman Park survey - inspection report", "standard", formats=["html", "pdf"])
    assert report["ok"], report
    api.set_geocoding_provider("offline")

    server = MapServer(api, _page())
    browser = Browser(chromium_path())
    browser.socket.settimeout(300)
    try:
        browser.set_viewport(1440, 900)
        yield {"api": api, "browser": browser, "base": server.base, "root": root, "server": server}
    finally:
        browser.close()
        server.close()
        os.chdir(cwd)


def boot(h, query: str = "mode=desk") -> Browser:
    browser = h["browser"]
    browser.open(f"{h['base']}/__stage.html?{query}", settle_s=0.2)
    browser.wait_for("window.__odkReady === true || window.__odkBootError", 60)
    assert browser.js("window.__odkBootError || null") is None, browser.js("window.__odkBootError")
    browser.wait_for("odkShell.connection === 'connected' && !!odkShell.state", 30)
    browser.js(WATCH)
    browser.js("odkShell.open('home').then(() => true)")
    settle(browser)
    return browser


def settle(browser: Browser, timeout_s: float = 90) -> None:
    """Every live panel on Home has answered (no "Reading…" left)."""
    browser.wait_for("""document.querySelector('.odk-frame').dataset.rendered === 'home'
        && document.querySelectorAll('.odk-home-left, .odk-home-main, .odk-home-inspector').length === 3
        && ![...document.querySelectorAll('.odk-frame .live-wait')].length""", timeout_s)


def text(browser: Browser, selector: str) -> str:
    return browser.js(f"(document.querySelector({selector!r}) || {{}}).textContent || ''").strip()


def click(browser: Browser, selector: str) -> None:
    assert browser.js(f"(() => {{ const b = document.querySelector({selector!r}); if (!b) return false; b.click(); return true; }})()"), selector


def answer_prompt(browser: Browser, value: str | None = None, timeout_s: float = 20) -> None:
    """Fill the modal's field (when given) and press its primary button."""
    browser.wait_for("!!document.querySelector('.modal-backdrop .modal')", timeout_s)
    if value is not None:
        browser.js(f"""(() => {{ const i = document.querySelector('.modal input, .modal textarea');
            i.value = {json.dumps(value)}; i.dispatchEvent(new Event('input')); return true; }})()""")
    browser.js("document.querySelector('.modal .modal-actions .primary').click(), true")
    browser.wait_for("!document.querySelector('.modal-backdrop')", timeout_s)


def refresh(browser: Browser) -> None:
    browser.js("odkShell.refreshState().then(() => true)")
    settle(browser)


# --------------------------------------------------------------------------- tests

class TestHomeReadsTheProject:
    def test_tiles_are_the_apis_figures(self, home) -> None:
        api, browser = home["api"], boot(home)
        summary = api.dataset_summary()["active"]
        assert summary["image_count"] == 77 and summary["geotagged_count"] == 77
        photos = text(browser, "[data-where='home.tiles.photos']")
        assert re.search(r"(?<!\d)77\s*/ 77 geotagged", photos), photos
        assert summary["camera"] in photos
        capture = summary["capture"]
        assert capture["dated"] == 77 and capture["first"].startswith("2016-06-29")
        survey = browser.js("""[...document.querySelectorAll('.odk-home-tile')]
            .find((t) => t.textContent.includes('Survey date')).textContent""")
        assert capture["first"][:10] in survey
        assert f"{capture['first'][11:16]}–{capture['last'][11:16]}" in survey, survey
        # The mission tile is the saved version's own summary.
        saved = json.loads(api.list_mission_versions()["versions"][0]["plan_summary_json"])
        mission = browser.js("""[...document.querySelectorAll('.odk-home-tile')]
            .find((t) => /mission/i.test(t.textContent)).textContent""")
        assert f"{saved['waypoints']:,}" in mission and "wp" in mission
        # The boundary is measured by the Api, not in the page.
        browser.wait_for("""[...document.querySelectorAll('.odk-home-tile')]
            .some((t) => t.textContent.includes('vertices'))""", 20)
        area = api.measure_geometry(api.get_state()["aoi"], "area")
        boundary = browser.js("""[...document.querySelectorAll('.odk-home-tile')]
            .find((t) => t.textContent.includes('Site boundary')).textContent""")
        assert f"{round(area['area_m2']):,}" in boundary, (boundary, area)
        findings = api.find_annotations()["count"]
        assert text(browser, ".odk-home-tile:last-child .odk-home-tile-value") == str(findings)

    def test_header_and_project_list(self, home) -> None:
        api, browser = home["api"], boot(home)
        project = api.get_project()["project"]
        assert text(browser, ".odk-home-title h1") == project["name"]
        place = text(browser, ".odk-home-place")
        assert "EPSG:32617" in place and "UTM 17N" in place and "° N" in place
        listed = api.home_projects()["projects"]
        rows = browser.js("[...document.querySelectorAll('.odk-home-project')].map((r) => r.dataset.control)")
        assert sorted(rows) == sorted(f"project:{p['id']}" for p in listed)
        mine = next(p for p in listed if p["active"])
        row = text(browser, f"[data-control='project:{mine['id']}']")
        assert "77 photos" in row and "2016-06-29" in row and "Processed" in row
        assert text(browser, ".odk-home-count") == f"{len(listed)} local"

    def test_reconstruction_card_is_reconstruction_summary(self, home) -> None:
        api, browser = home["api"], boot(home)
        run = api.reconstruction_summary()["runs"][0]
        table = text(browser, "[data-where='home.reconstruction']")
        assert f"{run['registered_images']} / {run['total_images']}" in table
        assert f"{run['sparse_points']:,}" in table
        assert f"{run['reprojection_error_px']:.3f} px" in table
        assert f"{run['georef_rmse_m']:.2f} m" in table
        assert f"{run['ortho_gsd_m']:.3f} / {run['dsm_gsd_m']:.2f} m" in table
        if run["runtime_s"] is not None:
            assert f"{run['runtime_s']:,.1f} s" in table
        else:
            assert "not reported" in table
        # The map card's title is the orthomosaic's own pixel size, and the map drew it.
        assert f"{run['ortho_gsd_m']:.3f} m/px" in text(browser, "[data-where='home.map']")
        browser.wait_for("""(() => { const m = document.querySelector('.odk-home-map');
            return !!(m && m.querySelector('canvas') && m.querySelector('[data-furniture]')); })()""", 120)

    def test_storage_and_exports_are_the_apis(self, home) -> None:
        api, browser = home["api"], boot(home)
        storage = api.project_storage()
        used = storage["project_bytes"] + storage["datasets_outside_bytes"]
        assert used > 0
        status = text(browser, ".odk-home-status")
        from app.home_data import FORMAT_LABELS  # noqa: F401 - the labels the list uses

        def si(n: float) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while n >= 1000 and i < len(units) - 1:
                n /= 1000
                i += 1
            return f"{n:.{0 if i == 0 else 2 if n < 10 else 1 if n < 100 else 0}f} {units[i]}"

        assert si(used) in status, (status, used)
        exports = api.list_exports()
        assert exports["count"] >= 6  # the six mission formats written in the fixture
        shown = browser.js("[...document.querySelectorAll('.odk-home-export')].map((b) => [b.title, b.textContent])")
        assert [path for path, _ in shown] == [f["path"] for f in exports["files"][:len(shown)]]
        for (path, label), f in zip(shown, exports["files"]):
            assert f"{f['bytes']:,} B" in label, (label, f)
        click(browser, ".odk-home-export")
        browser.wait_for(f"window.__toasts.some((t) => t.includes('Opened'))", 20)
        assert api.opened and api.opened[-1] == exports["files"][0]["path"]

    def test_every_placement_renders_where_it_says(self, home) -> None:
        browser = boot(home)
        placements = browser.js("""import('./js/odk/screens/home/surface.js').then((m) => m.PLACEMENTS)""")
        assert placements
        # The overflow menu and the dataset popover are drawn on demand; open them.
        click(browser, "[data-control='project-more']")
        click(browser, "[data-control='datasets']")
        browser.js("document.querySelector('[data-control=\"panel-tab:notifications\"]').click(), true")
        missing = browser.js("""((ps) => ps.filter((p) => {
            const scope = [...document.querySelectorAll(`[data-where="${p.where}"]`)];
            return !scope.some((s) => s.querySelector(`[data-action="${p.action}"]`)
              || s.matches(`[data-action="${p.action}"]`));
          }).map((p) => p.action + ' @ ' + p.where))""" + f"({json.dumps(placements)})")
        # A notification row exists only when there is a notification; that row is
        # exercised in TestNotifications.
        missing = [m for m in missing if "home.notifications.row" not in m]
        assert missing == [], missing
        unmarked = browser.js("""[...document.querySelectorAll('.odk-frame button, .odk-popover button')]
            .filter((b) => !b.hasAttribute('data-action') && !b.hasAttribute('data-control'))
            .map((b) => b.outerHTML.slice(0, 120))""")
        assert unmarked == [], unmarked
        assert browser.js("window.__errors") == []


class TestTheAcceptancePath:
    def test_create_import_share_revoke(self, home) -> None:
        api, browser = home["api"], boot(home)
        before = {p["id"] for p in api.home_projects()["projects"]}
        folder = home["root"] / "second"
        folder.mkdir()
        api.folders.append(str(folder))
        click(browser, "[data-where='home.projects.footer'] [data-action='New Project']")
        answer_prompt(browser, "Second site")
        browser.wait_for("window.__toasts.some((t) => t.includes('Second site'))", 30)
        created = [p for p in api.home_projects()["projects"] if p["id"] not in before]
        assert len(created) == 1 and created[0]["active"] and created[0]["name"] == "Second site"
        browser.wait_for("(document.querySelector('.odk-home-title h1') || {}).textContent === 'Second site'", 60)
        # The header and the tiles are separate live panels re-read on the same refresh;
        # the title can land first, so wait for the tile rather than read it once.
        browser.wait_for(r"""/none|no dataset/.test(
            (document.querySelector("[data-where='home.tiles.photos']") || {}).textContent || '')""", 60)

        # Import the Aukerman photographs through the header's own button.
        api.folders.append(str(IMAGES))
        click(browser, "[data-where='home.header'] [data-action='Import']")
        browser.wait_for("window.__toasts.some((t) => /Imported 77 image/.test(t))", 120)
        assert api.dataset_summary()["active"]["image_count"] == 77
        # The panels re-read on the refresh the import asked for; the tile follows.
        browser.wait_for(r"""/(^|\D)77\s*\/ 77 geotagged/.test(
            (document.querySelector("[data-where='home.tiles.photos']") || {}).textContent || '')""", 60)

        # Share, then Revoke, from the overflow menu.
        browser.js("odkShell.confirmAll = true; true")
        try:
            click(browser, "[data-control='project-more']")
            click(browser, ".odk-popover:not([hidden]) [data-action='Share']")
            answer_prompt(browser, "Client review")
            # The token is shown once. The first link of a session cold-loads the REST
            # service's models and database; on a loaded machine that took 84 s here.
            answer_prompt(browser, timeout_s=240)
            browser.wait_for("window.__toasts.some((t) => /Link .* issued/.test(t))", 60)
            pid = api.get_project()["project"]["id"]
            links = api.list_share_links(pid)["links"]
            assert len(links) == 1 and not links[0]["revoked"]
            click(browser, "[data-control='project-more']")
            click(browser, ".odk-popover:not([hidden]) [data-action='Revoke Links']")
            browser.wait_for("window.__toasts.some((t) => t.includes('Revoked 1 link'))", 30)
            assert all(link["revoked"] for link in api.list_share_links(pid)["links"])
        finally:
            browser.js("odkShell.confirmAll = false; true")
        # Back to the Aukerman project for the other tests, through its list row.
        first = min(before)
        click(browser, f"[data-control='project:{first}']")
        browser.wait_for(f"odkShell.project && odkShell.project.id === {first}", 30)
        settle(browser)

    def test_the_datasets_popover_selects_through_the_api(self, home) -> None:
        api, browser = home["api"], boot(home)
        click(browser, "[data-control='datasets']")
        rows = browser.js("[...document.querySelectorAll('.odk-popover:not([hidden]) .odk-home-dataset')].map((r) => r.title)")
        assert rows == [d["path"] for d in api.dataset_summary()["datasets"]]
        active = browser.js("""(() => { const b = document.querySelector(
            '.odk-popover:not([hidden]) .odk-home-dataset [data-action="Select Dataset"]');
            return [b.textContent, b.disabled]; })()""")
        assert active == ["In use", True]
        # One popover per panel: a re-read replaces it rather than stacking another.
        refresh(browser)
        browser.js("odkShell.open('home').then(() => true)")
        settle(browser)
        assert browser.js("document.querySelectorAll('.odk-home-datasets').length") == 1


class TestNotifications:
    def test_a_finished_job_is_announced_and_can_be_cleared(self, home) -> None:
        api, browser = home["api"], boot(home)
        job = api._jobs.submit("u4_probe", lambda progress, should_cancel: {"ok": True})
        deadline = time.monotonic() + 15
        while api.job_status(job)["job"]["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.1)
        deadline = time.monotonic() + 10
        while not api.list_notifications(True)["notifications"] and time.monotonic() < deadline:
            time.sleep(0.1)
        unread = api.list_notifications(True)["notifications"]
        assert unread, "the finished job was not announced"
        refresh(browser)
        browser.js("document.querySelector('[data-control=\"panel-tab:notifications\"]').click(), true")
        browser.wait_for(f"(document.querySelector('[data-control=\"panel-tab:notifications\"] .odk-num') || {{}})"
                         f".textContent === '{len(unread)}'", 30)
        assert browser.js("!!document.querySelector('[data-control=\"tab:home\"] .odk-home-unread')")
        click(browser, "[data-where='home.notifications.row'] [data-action='Mark Notification Read']")
        browser.wait_for("window.__toasts.some((t) => t.includes('Marked read'))", 20)
        assert api.list_notifications(True)["notifications"] == unread[1:] or \
            len(api.list_notifications(True)["notifications"]) == len(unread) - 1
        browser.js("odkShell.confirmAll = true; true")
        try:
            click(browser, "[data-where='home.notifications'] [data-action='Clear Notifications']")
            browser.wait_for("window.__toasts.some((t) => t.includes('Removed'))", 20)
        finally:
            browser.js("odkShell.confirmAll = false; true")
        assert api.list_notifications()["notifications"] == []
        browser.wait_for("!document.querySelector('[data-control=\"tab:home\"] .odk-home-unread')", 30)
        assert not browser.js("!!document.querySelector('[data-control=\"tab:home\"] .odk-home-unread')")


class TestScreenshots:
    def test_desk_and_field_window_sizes(self, home) -> None:
        browser = boot(home)
        browser.wait_for("""(() => { const m = document.querySelector('.odk-home-map');
            return !!(m && m.querySelector('canvas')); })()""", 120)
        time.sleep(4)  # let the map finish its first tiles
        shots = []
        for width, height in ((1440, 900), (1180, 820)):
            browser.set_viewport(width, height)
            time.sleep(2.5)
            shots.append(browser.screenshot(SHOTS / f"u4-home-desk-{width}x{height}.png"))
            overflow = browser.js("document.documentElement.scrollWidth > window.innerWidth")
            assert not overflow, f"horizontal page scroll at {width}x{height}"
        # Field mode has no Home screen: opening Home switches back to Desk, and says so.
        browser.js("odkShell.setMode('field').then(() => odkShell.open('home')).then(() => true)")
        assert browser.js("document.documentElement.dataset.mode") == "desk"
        settle(browser)
        shots.append(browser.screenshot(SHOTS / "u4-home-from-field-1180x820.png"))
        browser.set_viewport(1440, 900)
        assert all(p.stat().st_size > 20_000 for p in shots)
