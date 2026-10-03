"""The v1 Fly workspace (U6), in a real browser, against the real Api (REBUILD_PLAN 3.4).

odk.html is served by tests/ui_live_bridge.py with the production Api behind it and
driven in headless Edge/Chromium. Two aircraft:

  * the simulator (driver "mock"): connect from the CONNECTION form, confirm a preflight
    row, upload through the real confirmation dialog, then Start / Pause / Resume /
    Return home / Capture / a held Abort reach it; every simulated value on the screen
    carries the SIMULATED tag; Export log writes files; the Messages and Events tabs
    show what the Api recorded; Field mode draws the same answers with 64 px targets;
  * tests/fake_mavlink_vehicle.py over real MAVLink (driver "pymavlink"): the commands
    arrive at the vehicle as MAVLink, an autopilot's STATUSTEXT pre-arm failure reaches
    the screen, a lost link raises the banner and marks the aircraft LAST KNOWN, and
    nothing about a real link is tagged SIMULATED.

Every placement in screens/fly/surface.js is seen rendered with its data-action.
Screenshots (desk 1440x900, Field 1180x820) go to $ODK_U6_SHOTS.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("websocket")
pytest.importorskip("requests")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from ui_live_bridge import Browser, chromium_path, serve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLY = REPO / "app" / "web" / "js" / "odk" / "screens" / "fly"
FIELD_FLY = REPO / "app" / "web" / "js" / "odk" / "screens" / "field" / "fly.js"
AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]
SHOTS = Path(os.environ.get("ODK_U6_SHOTS") or Path(tempfile.gettempdir()) / "odk-u6-shots")

WATCH = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__errors = [];
  window.addEventListener('error', (e) => window.__errors.push(String(e.message)));
  window.addEventListener('unhandledrejection', (e) => window.__errors.push(String(e.reason)));
  window.__watching = new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
    if (n.classList && n.classList.contains('odk-toast')) window.__toasts.push(n.textContent);
  })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
true
"""

# What a region of the Fly screen shows, for the checks below.
REGION = "document.querySelector('[data-region={0}]')"


def placements() -> list[dict]:
    """PLACEMENTS from surface.js, read as data."""
    text = (FLY / "surface.js").read_text(encoding="utf-8")
    body = text.split("export const PLACEMENTS = [", 1)[1].split("];", 1)[0]
    return [dict(re.findall(r'(\w+): "([^"]*)"', line)) for line in body.splitlines() if "action:" in line]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    tmp = tmp_path_factory.mktemp("fly")
    cwd = os.getcwd()
    os.chdir(tmp)
    api = Api(AppSession(ProjectStore(tmp / "projects.db")))
    api.create_project("Aukerman Park", str(tmp / "project"))
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 40})["ok"]
    assert api.save_mission("grid at 40 m")["ok"]
    server, base = serve(api)
    browser = Browser(executable)
    try:
        browser.set_viewport(1440, 900)
        yield api, browser, base, tmp
    finally:
        browser.close()
        server.shutdown()
        api.disconnect_vehicle()
        os.chdir(cwd)


def boot(browser: Browser, base: str, mode: str = "desk") -> None:
    for attempt in range(2):
        browser.open(f"{base}/odk.html?mode={mode}", settle_s=0.2)
        browser.wait_for("window.__odkReady === true || window.__odkBootError", 90)
        error = browser.js("window.__odkBootError || null")
        # A reload while the previous page's polls are still in flight can drop one
        # module fetch on a loaded machine ("Failed to fetch dynamically imported
        # module"); that is the page load, not the screen, so it is retried once.
        if error is None or attempt or "Failed to fetch" not in str(error):
            break
    assert error is None, error
    browser.wait_for("odkShell.connection === 'connected' && !!odkShell.state", 60)
    browser.js(WATCH)
    browser.js("odkShell.open('fly').then(() => true)")
    browser.wait_for("document.querySelector('[data-frame]').getAttribute('data-rendered') === 'fly'", 60)


def click(browser: Browser, selector: str) -> None:
    ok = browser.js(f"""(() => {{
      const b = document.querySelector({selector!r});
      if (!b || b.disabled) return b ? 'disabled' : 'missing';
      b.click(); return 'ok';
    }})()""")
    assert ok == "ok", f"{selector}: {ok}"



def wait_api(predicate, timeout_s: float = 20.0, poll_s: float = 0.3):
    deadline = time.monotonic() + timeout_s
    value = None
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(poll_s)
    raise AssertionError(f"timed out; last value {value!r}")


def fly_actions(browser: Browser) -> list[str]:
    return browser.js("""[...document.querySelectorAll('[data-region] [data-action]')]
        .map((n) => n.getAttribute('data-action'))""")


def sim_tags(browser: Browser, where: str) -> int:
    return browser.js(f"document.querySelectorAll('{where} [data-tag=SIMULATED]').length")


# ===================================================================== the source


class TestTheSourceIsHonest:
    def test_no_artboard_figure_is_written_into_the_screen(self) -> None:
        """The artboards are full of real Aukerman numbers; none may be pasted in."""
        figures = ["55.0", "8.0 m/s", "015°", "77 / 152", "1,853", "921.6", "921.0", "40,818",
                   "259,920", "be023b45", "41.30372", "81.75222", "[battery %]", "Battery 1",
                   "Pilot A", "eBee", "EPSG:32617", "1 min 55 s", "100 %"]
        for path in [*FLY.glob("*.js"), FIELD_FLY]:
            text = path.read_text(encoding="utf-8")
            for figure in figures:
                assert figure not in text, f"{figure!r} in {path.name}"

    def test_every_placement_resolves_and_names_its_inventory_row(self) -> None:
        import json

        inventory = json.loads((REPO / "docs" / "ui" / "inventory.json").read_text(encoding="utf-8"))
        rows = {r["id"] for r in inventory["rows"] if r["workspace"] == "fly"}
        named = {p.get("inventory") for p in placements()}
        panels = set(re.findall(r'inventory: "([^"]+)"', (FLY / "surface.js").read_text(encoding="utf-8")))
        missing = rows - named - panels
        assert not missing, f"fly inventory rows with no placement: {sorted(missing)}"


# ============================================================ desk, the simulator


class TestDeskWithTheSimulator:
    def test_before_a_link_nothing_is_tagged_and_the_screen_says_so(self, stack) -> None:
        api, browser, base, _ = stack
        boot(browser, base)
        browser.wait_for("document.querySelectorAll('.fly-check').length > 0", 20)
        present = set(fly_actions(browser))
        for key in ("Connect Vehicle", "Disconnect", "Preflight", "Upload", "Load Mission Version",
                    "Export Log", "Load Flight", "Linked Progress", "Mark Corner", "Close Boundary",
                    "Clear Corners", "Arm", "Start Mission", "Pause", "RTL", "Abort"):
            assert key in present, key
        assert sim_tags(browser, "[data-region]") == 0
        text = browser.js(REGION.format("inspector") + ".textContent")
        assert "No vehicle connected" in text
        assert "no vehicle connected" in browser.js("document.querySelector('.fly-progress').textContent")
        # The checklist is run_preflight's, row for row.
        checks = api.run_preflight()["checks"]
        assert browser.js("document.querySelectorAll('.fly-check').length") == len(checks)
        assert browser.js("document.querySelector('[data-action=\"Start Mission\"]').disabled") is True

    def test_connect_from_the_form(self, stack) -> None:
        api, browser, _, _ = stack
        click(browser, "[data-control='driver:mock']")
        browser.js("(() => { const i = document.querySelector('input[name=fly-uri]'); i.value = 'mock://u6'; "
                   "i.dispatchEvent(new Event('input')); return true; })()")
        click(browser, '[data-action="Connect Vehicle"]')
        wait_api(lambda: api.get_state()["vehicle"]["connected"])
        assert api.get_state()["vehicle"]["uri"] == "mock://u6"
        browser.wait_for("document.querySelector('.fly-connection [data-tag=SIMULATED]') !== null", 15)
        browser.wait_for("document.querySelector('[data-action=\"Connect Vehicle\"]').textContent.includes('Reconnect')", 10)
        facts = browser.js("document.querySelector('.fly-connection .fly-kv').textContent")
        assert "mock" in facts and "simulator" in facts

    def test_a_preflight_row_is_confirmed_by_the_operator(self, stack) -> None:
        api, browser, _, _ = stack
        browser.wait_for("document.querySelector('.fly-check [data-action=\"Confirm Check\"]') !== null", 15)
        check_id = browser.js("document.querySelector('.fly-check [data-action=\"Confirm Check\"]')"
                              ".closest('.fly-check').getAttribute('data-check')")
        click(browser, '.fly-check [data-action="Confirm Check"]')
        wait_api(lambda: next(c for c in api.run_preflight()["checks"] if c["id"] == check_id)["confirmed"])
        browser.wait_for(f"document.querySelector('.fly-check[data-check=\"{check_id}\"]').textContent.includes('confirmed')", 15)
        counts = browser.js("document.querySelector('.fly-counts').textContent")
        report = api.run_preflight()
        assert f"{report['blocking_count']} blk" in counts
        assert f"{len(report['pending_confirmation'])} man" in counts

    def test_upload_asks_first_and_the_aircraft_then_carries_the_mission(self, stack) -> None:
        api, browser, _, _ = stack
        browser.js("odkShell.confirmAll = false; true")
        click(browser, '[data-action="Upload"]')
        browser.wait_for("document.querySelector('.modal-actions .primary') !== null", 10)
        assert "Upload the mission" in browser.js("document.querySelector('.modal-text').textContent")
        browser.js("document.querySelector('.modal-actions .primary').click(); true")
        wait_api(lambda: api.vehicle_mission()["on_vehicle"])
        browser.wait_for("document.querySelector('.fly-mission-state').textContent.includes('on the aircraft')", 15)
        browser.wait_for("document.querySelector('.fly-legend').textContent.includes('To fly')", 15)
        items = api.vehicle_mission()["item_count"]
        assert f"{items} items" in browser.js("document.querySelector('.fly-mission-state').textContent")
        assert browser.js("document.querySelector('[data-action=\"Start Mission\"]').disabled") is False

    def test_start_pause_resume_and_return_reach_the_simulator(self, stack) -> None:
        api, browser, _, _ = stack
        browser.js("odkShell.confirmAll = true; true")   # Start's dialog and its preflight override
        click(browser, '[data-action="Start Mission"]')
        wait_api(lambda: api.telemetry()["telemetry"].get("flight_mode") == "AUTO")
        assert api.telemetry()["telemetry"]["armed"] is True
        browser.wait_for("document.querySelector('.fly-mode').textContent.includes('AUTO · ARMED')", 15)
        # The aircraft climbs, then flies the route: progress follows the vehicle.
        wait_api(lambda: api.telemetry()["telemetry"]["waypoint_index"] > 0, timeout_s=60)
        browser.wait_for("/ITEM [1-9]/.test(document.querySelector('.fly-progress').textContent)", 20)
        click(browser, '.fly-commandbar [data-action="Pause"]')
        wait_api(lambda: (api.flight_session()["session"] or {}).get("state", {}).get("status") == "paused")
        browser.wait_for("document.querySelector('.fly-commandbar [data-action=\"Resume\"]') !== null", 15)
        click(browser, '.fly-commandbar [data-action="Resume"]')
        wait_api(lambda: api.flight_session()["session"]["state"]["status"] == "flying")
        assert api.telemetry()["telemetry"]["flight_mode"] == "AUTO"
        browser.wait_for("document.querySelector('.fly-legend').textContent.includes('Flown')", 20)

    def test_every_simulated_value_is_tagged_and_the_aircraft_is_where_the_api_says(self, stack) -> None:
        api, browser, _, _ = stack
        for where in (".fly-connection", ".fly-commandbar", ".fly-progress", "[data-tab=telemetry]",
                      ".fly-aircraft"):
            assert sim_tags(browser, where) >= 1, where
        label = browser.js("document.querySelector('.fly-aircraft').getAttribute('aria-label')")
        lat, lon = [float(v) for v in re.findall(r"(\d+\.\d+)°", label)]
        reported = api.telemetry()["telemetry"]
        assert abs(lat - abs(reported["latitude"])) < 5e-4 and abs(lon - abs(reported["longitude"])) < 5e-4
        stats = browser.js("[...document.querySelectorAll('[data-tab=telemetry] .fly-stat-label')].map(n => n.textContent)")
        assert stats == ["Altitude rel. home", "Ground speed", "Heading", "Battery"]
        battery = browser.js("[...document.querySelectorAll('[data-tab=telemetry] .fly-stat')][3].textContent")
        assert battery.startswith("Battery") and "%" in battery
        # What the vehicle does not report is said, not filled.
        missing = browser.js("[...document.querySelectorAll('[data-tab=telemetry] dd.is-missing')].map(n => n.textContent)")
        assert missing and set(missing) == {"not reported"}

    def test_more_offers_land_manual_capture_and_capture_reaches_the_camera(self, stack) -> None:
        api, browser, _, _ = stack
        click(browser, "[data-control='fly-more']")
        browser.wait_for("document.querySelector('.fly-more:not([hidden])') !== null", 5)
        keys = browser.js("[...document.querySelectorAll('.fly-more [data-action]')].map(n => n.getAttribute('data-action'))")
        assert keys == ["Land", "Manual Override", "Capture Now", "Disarm"]
        click(browser, '.fly-more [data-action="Capture Now"]')
        wait_api(lambda: any("Camera triggered" in m["text"] for m in api.vehicle_messages(0)["messages"]))

    def test_the_messages_tab_shows_what_the_vehicle_said(self, stack) -> None:
        api, browser, _, _ = stack
        click(browser, "[data-control='panel-tab:messages']")
        browser.wait_for("document.querySelectorAll('.fly-msg').length > 0", 10)
        shown = browser.js("[...document.querySelectorAll('.fly-msg-text')].map(n => n.textContent)")
        said = [m["text"] for m in api.vehicle_messages(0)["messages"]]
        assert "Mission started." in shown and set(shown) <= set(said)
        assert sim_tags(browser, ".fly-msgs") == len(shown)

    def test_the_events_tab_reads_the_audit_log(self, stack) -> None:
        api, browser, _, _ = stack
        click(browser, "[data-control='panel-tab:events']")
        browser.wait_for("document.querySelectorAll('.fly-event').length > 0", 15)
        types = browser.js("[...document.querySelectorAll('.fly-event-type')].map(n => n.textContent)")
        assert "vehicle command" in types and "mission uploaded" in types
        click(browser, "[data-control='panel-tab:telemetry']")

    def test_return_home_then_a_short_press_does_not_abort_but_a_hold_does(self, stack) -> None:
        api, browser, _, _ = stack
        click(browser, '.fly-commandbar [data-action="RTL"]')
        wait_api(lambda: api.telemetry()["telemetry"]["flight_mode"] in ("RTL", "LAND"))
        browser.js("odkShell.confirmAll = false; true")   # the hold is Abort's confirmation
        click(browser, "[data-control='abort-confirm']")
        browser.wait_for("!document.querySelector('.fly-abort').hidden", 5)
        aborted = lambda: any("aborted" in m["text"] for m in api.vehicle_messages(0)["messages"])  # noqa: E731
        browser.js("document.querySelector('.fly-hold').click(); true")
        time.sleep(2.0)
        assert not aborted(), "a tap is not a hold"
        browser.js("""(() => { const b = document.querySelector('.fly-hold');
            b.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })); return true; })()""")
        wait_api(aborted, timeout_s=10)
        assert api.telemetry()["telemetry"]["flight_mode"] == "LAND"
        browser.wait_for("document.querySelector('.fly-abort').hidden", 5)
        assert not browser.js("document.querySelector('.modal-actions')"), "the hold must not ask again"
        browser.js("""document.querySelector('.fly-hold').dispatchEvent(new PointerEvent('pointerup', { bubbles: true })); true""")

    def test_export_log_writes_the_flight(self, stack) -> None:
        api, browser, _, tmp = stack
        browser.js("odkShell.confirmAll = true; true")
        before = set((tmp / "project" / "flights").glob("*")) if (tmp / "project" / "flights").exists() else set()
        click(browser, '[data-action="Export Log"]')
        folder = tmp / "project" / "flights"
        wanted = {".csv", ".json", ".gpx", ".kml"}
        written = wait_api(lambda: (set(folder.glob("*")) - before)
                           if folder.exists() and wanted <= {p.suffix for p in set(folder.glob("*")) - before} else None)
        # A simulated log is exported, and says so in its file names.
        assert all("SIMULATED" in p.name for p in written), written
        browser.wait_for("window.__toasts.some((t) => t.includes('SIMULATED log'))", 10)

    def test_desk_screenshots(self, stack) -> None:
        _, browser, _, _ = stack
        click(browser, "[data-control='abort-confirm']")
        time.sleep(0.6)
        browser.js("document.querySelectorAll('.odk-toast').forEach((t) => t.remove()); true")  # transient
        browser.screenshot(SHOTS / "u6-fly-desk-1440x900.png", 1440, 900)
        click(browser, ".fly-abort-cancel")
        browser.screenshot(SHOTS / "u6-fly-desk-1100x700.png", 1100, 700)
        overflow = browser.js("(() => { const bar = document.querySelector('.fly-commandbar');"
                              " return bar.scrollWidth - bar.clientWidth; })()")
        assert overflow <= 1, "the command bar must fit the centre at the minimum window"
        browser.set_viewport(1440, 900)
        assert browser.js("window.__errors") == []


# ======================================================================= Field


class TestField:
    def test_field_draws_the_same_answers_with_64px_targets(self, stack) -> None:
        api, browser, base, _ = stack
        browser.set_viewport(1180, 820)
        # Field is reached the way an operator reaches it (the mode is persisted), on a
        # fresh page: the long desk session above leaves many requests in flight.
        browser.js("odkShell.setMode('field').then(() => true)")
        boot(browser, base, "field")
        try:
            browser.wait_for("document.querySelector('.flyf-panel') !== null", 30)
        except AssertionError as exc:
            state = browser.js("JSON.stringify({rendered: document.querySelector('[data-frame]').getAttribute('data-rendered'),"
                               " frame: document.querySelector('[data-frame]').getAttribute('data-frame'),"
                               " ws: odkShell.workspaceId, toasts: window.__toasts.slice(-6), errors: window.__errors})")
            raise AssertionError(f"{exc}: {state}") from None
        browser.wait_for("document.querySelector('.flyf-status').textContent.length > 0", 10)
        if not api.get_state()["vehicle"]["connected"]:
            # Field connects with its own yellow action, on the driver chosen there.
            click(browser, "[data-control='driver:mock']")
            click(browser, '.flyf-actions [data-action="Connect Vehicle"]')
            wait_api(lambda: api.get_state()["vehicle"]["connected"])
        browser.wait_for("document.querySelector('.flyf-status [data-tag=SIMULATED]') !== null", 20)
        browser.wait_for("document.querySelectorAll('.flyf-tile [data-tag=SIMULATED]').length >= 4", 20)
        assert "LINK" in browser.js("document.querySelector('.flyf-link').textContent")
        small = browser.js("""[...document.querySelectorAll('[data-region=left] button')]
            .filter((b) => b.offsetParent).map((b) => [b.textContent.trim(), b.getBoundingClientRect().height])
            .filter(([, h]) => h < 63.5)""")
        assert small == [], small
        # The buttons are the same nodes from poll to poll (the screen re-reads every
        # second): a press, and above all a held Abort, is never cut by a redraw.
        browser.js("window.__flyfButtons = [...document.querySelectorAll('.flyf-actions button')]; true")
        time.sleep(2.5)
        assert browser.js("""window.__flyfButtons.length > 0
            && window.__flyfButtons.every((b) => b.isConnected)"""), "Field rebuilt its action buttons on a poll"
        time.sleep(1.5)
        browser.screenshot(SHOTS / "u6-fly-field-1180x820.png", 1180, 820)
        browser.js("odkShell.setMode('desk').then(() => true)")
        browser.set_viewport(1440, 900)
        assert browser.js("window.__errors") == []


# ================================================================ a real link


class TestARealMavlinkLink:
    def test_commands_arrive_as_mavlink_and_nothing_is_tagged_simulated(self, stack) -> None:
        pytest.importorskip("pymavlink")
        from fake_mavlink_vehicle import FakeVehicle, free_udp_port

        api, browser, base, _ = stack
        api.disconnect_vehicle()
        port = free_udp_port()
        vehicle = FakeVehicle(port).start()
        try:
            boot(browser, base)
            browser.js("odkShell.confirmAll = true; true")
            click(browser, "[data-control='driver:pymavlink']")
            browser.js(f"(() => {{ const i = document.querySelector('input[name=fly-uri]'); i.value = 'udpin:127.0.0.1:{port}';"
                       " i.dispatchEvent(new Event('input')); return true; })()")
            click(browser, '[data-action="Connect Vehicle"]')
            try:
                wait_api(lambda: api.telemetry()["telemetry"].get("connected"), timeout_s=30)
            except AssertionError as exc:
                raise AssertionError(f"{exc}; toasts {browser.js('window.__toasts')}; "
                                     f"vehicle {api.get_state()['vehicle']}") from None
            browser.wait_for("document.querySelector('.fly-aircraft') !== null", 20)
            assert sim_tags(browser, "[data-region]") == 0

            # An autopilot's pre-arm refusal reaches the pilot, as the autopilot wrote it.
            vehicle.say("PreArm: Compass not calibrated", severity=2)
            browser.wait_for("document.querySelector('.fly-prearm').textContent.includes('PreArm: Compass not calibrated')", 15)
            browser.wait_for("document.querySelector('.fly-banner').textContent.includes('PreArm: Compass')", 10)

            click(browser, '[data-action="Upload"]')
            wait_api(lambda: len(vehicle.stored[0]) > 0, timeout_s=60)
            browser.wait_for("!document.querySelector('[data-action=\"Start Mission\"]').disabled", 20)
            click(browser, '[data-action="Start Mission"]')
            wait_api(lambda: any(c[0] == 300 for c in vehicle.commands))
            assert any(c[0] == 400 and c[1] == 1.0 for c in vehicle.commands), "armed before start"
            browser.wait_for("document.querySelector('.fly-commandbar [data-action=\"Pause\"]') && "
                             "!document.querySelector('.fly-commandbar [data-action=\"Pause\"]').disabled", 20)
            click(browser, '.fly-commandbar [data-action="Pause"]')
            wait_api(lambda: any(c[0] == 193 and c[1] == 0.0 for c in vehicle.commands))
            click(browser, '.fly-commandbar [data-action="RTL"]')
            wait_api(lambda: any(c[0] == 20 for c in vehicle.commands))
            wait_api(lambda: vehicle.mode_name() == "RTL")

            # A lost link: the banner says so, the aircraft is LAST KNOWN, Details offers
            # Clear alerts.
            vehicle.go_silent()
            browser.wait_for("document.querySelector('.fly-banner').textContent.includes('Link lost')", 20)
            browser.wait_for("document.querySelector('.fly-aircraft').textContent.includes('LAST KNOWN')", 10)
            assert "LAST KNOWN" in browser.js("document.querySelector('[data-tab=telemetry] .fly-tab-head').textContent")
            click(browser, "[data-control='alerts']")
            browser.wait_for("document.querySelector('.fly-alerts [data-action=\"Clear Alerts\"]') !== null", 5)
            browser.js("document.querySelectorAll('.odk-toast').forEach((t) => t.remove()); true")
            browser.screenshot(SHOTS / "u6-fly-desk-link-lost-1440x900.png", 1440, 900)
            click(browser, '.fly-alerts [data-action="Clear Alerts"]')
            vehicle.resume()
            browser.wait_for("!document.querySelector('.fly-banner').textContent.includes('Link lost')", 20)
            assert sim_tags(browser, "[data-region]") == 0
        finally:
            api.disconnect_vehicle()
            vehicle.stop()

    def test_after_the_link_closes_the_interrupted_flight_can_be_discarded(self, stack) -> None:
        api, browser, _, _ = stack
        browser.js("odkShell.refreshState().then(() => true)")
        assert api.check_interrupted_flight()["recovered"] is True
        browser.wait_for("!document.querySelector('.fly-interrupted').hidden", 20)
        keys = browser.js("[...document.querySelectorAll('.fly-interrupted [data-action]')].map(n => n.getAttribute('data-action'))")
        assert keys == ["Resume From Images", "Discard Interrupted Flight"]
        click(browser, '[data-action="Discard Interrupted Flight"]')
        wait_api(lambda: api.check_interrupted_flight()["recovered"] is False)
        browser.wait_for("document.querySelector('.fly-interrupted').hidden", 20)

    def test_every_placement_was_rendered(self, stack) -> None:
        """surface.js promises each verb on screen; the session above showed each one."""
        _, browser, _, _ = stack
        seen = set(fly_actions(browser))
        conditional = {"Resume", "Disarm", "Land", "Manual Override", "Capture Now", "Clear Alerts",
                       "Resume From Images", "Discard Interrupted Flight", "Confirm Check"}
        for placement in placements():
            if placement["action"] in conditional:
                continue  # drawn only in a state the tests above put the screen in
            assert placement["action"] in seen, placement
