"""Both UIs, in a real browser, against the real Api and a MAVLink peer.

The flight buttons were wired by name and tested by name, against a bridge that said
"ok" to everything -- so RTL sent a command the Api did not know, Capture Now a camera
action that did not exist, Export Log counted a dict as an array, and neither UI could
connect an aircraft at all. Those are disagreements between the page and the Api, and
only running the two together can see one.

tests/ui_live_bridge.py serves app/web with the production Api behind it (pywebview's
transport replaced by HTTP on loopback) and drives headless Edge/Chromium over the
DevTools protocol. tests/fake_mavlink_vehicle.py is the aircraft.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("pymavlink")
pytest.importorskip("websocket")
pytest.importorskip("requests")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from fake_mavlink_vehicle import FakeVehicle, free_udp_port  # noqa: E402
from ui_live_bridge import Browser, chromium_path, serve  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]

HELPERS = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((mutations) => mutations.forEach((m) =>
    m.addedNodes.forEach((n) => {
      if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent);
    })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
window.__click = (text) => {
  const b = [...document.querySelectorAll('.toolbar button')].find((x) => x.textContent.trim() === text);
  if (!b) throw new Error('no toolbar button ' + text);
  b.click();
  return true;
};
window.__choose = (text) => {
  const row = [...document.querySelectorAll('.modal-row')].find((x) => x.textContent.includes(text));
  if (!row) return false;
  row.click();
  return true;
};
window.__answer = (value) => {
  const input = document.querySelector('.modal input');
  if (!input) return false;
  input.value = value;
  document.querySelector('.modal-actions .primary').click();
  return true;
};
window.__yes = () => {
  const yes = document.querySelector('.modal-actions .primary');
  if (!yes) return false;
  yes.click();
  return true;
};
true
"""


@pytest.fixture
def browser():
    # One browser per test: a renderer that stalls under memory pressure then fails only
    # the test it stalled in, instead of every test after it.
    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    driver = Browser(executable)
    yield driver
    driver.close()


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("ui", str(tmp_path / "project"))
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    port = free_udp_port()
    vehicle = FakeVehicle(port).start()
    server, base = serve(api)
    yield api, vehicle, f"udpin:127.0.0.1:{port}", base
    api.disconnect_vehicle()
    server.shutdown()
    vehicle.stop()


def toast_matching(browser: Browser, needle: str, timeout_s: float = 30.0) -> str:
    """The first toast containing `needle` since the last one this test consumed."""
    found = browser.wait_for(
        f"__toasts.slice(window.__seen || 0).find((t) => t.includes({needle!r}))",
        timeout_s=timeout_s)
    browser.js("window.__seen = __toasts.length; true")
    return str(found)


class TestTheCockpitFliesTheAircraft:
    def test_connect_upload_command_and_record_from_the_flight_screen(self, browser, stack):
        api, vehicle, uri, base = stack
        browser.open(f"{base}/workspace.html", settle_s=3)
        browser.js(HELPERS)
        browser.js("[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === 'Flight').click(); true")
        time.sleep(0.5)

        # Connect: the cockpit had no way to reach an aircraft.
        browser.js("__click('Connect')")
        browser.wait_for("__choose('MAVLink autopilot')")
        browser.wait_for(f"__answer({uri!r})")
        assert "via pymavlink" in toast_matching(browser, "Connected to")
        time.sleep(2.5)
        panel = browser.js("document.querySelector('.workspace').innerText")
        assert "pymavlink" in panel and "STABILIZE" in panel
        assert "Battery 0%" not in panel

        # Upload reports what the vehicle accepted, including the fence.
        browser.js("__click('Upload')")
        browser.wait_for("__yes()")
        uploaded = toast_matching(browser, "The vehicle accepted", 60)
        assert "fence" in uploaded
        assert len(vehicle.stored[1]) == len(AOI)

        # Load Flight, on the Flight toolbar, reads the record the upload just wrote.
        summary = api.check_interrupted_flight()["summary"]
        browser.js("__click('Load Flight')")
        assert toast_matching(browser, summary[:24]).startswith(summary[:24])

        # RTL sent 'rtl', which the Api did not know.
        browser.js("__click('RTL')")
        browser.wait_for("__yes()")
        assert "Accepted by the vehicle" in toast_matching(browser, "RTL:")
        assert vehicle.mode_name() == "RTL"

        # Capture Now sent 'trigger', which the camera did not know.
        browser.js("__click('Capture Now')")
        toast_matching(browser, "Accepted by the vehicle.")
        assert any(command == 2000 for command, *_ in vehicle.commands)

        # Preflight listed Python libraries; it now checks the aircraft.
        browser.js("__click('Preflight')")
        preflight = toast_matching(browser, "confirmation")
        assert "Weather" in preflight and "torch" not in preflight

        # Export Log said "No flight has been recorded yet" after writing four files.
        browser.js("__click('Export Log')")
        assert "Wrote 4 file(s)" in toast_matching(browser, "Wrote")

        # Mark Corner refused every corner as "fix type 0".
        browser.js("__click('Mark Corner')")
        browser.wait_for("__answer('north-west')")
        assert "Corner 1 marked" in toast_matching(browser, "Corner 1")

        # The Fly menu's command items fell through to "not handled here".
        browser.js("window.odk.onMenu({action: 'fly.command', payload: {command: 'land'}}); true")
        browser.wait_for("__yes()")
        assert "Accepted" in toast_matching(browser, "land:")
        assert not browser.js("__toasts.some((t) => t.includes('not handled'))")

    def test_a_refusal_reaches_the_operator_as_an_error(self, browser, stack):
        api, vehicle, uri, base = stack
        vehicle.reject_upload = True
        api.connect_vehicle(uri, "pymavlink")
        browser.open(f"{base}/workspace.html", settle_s=3)
        browser.js(HELPERS)
        browser.js("[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === 'Flight').click(); true")
        time.sleep(0.5)
        browser.js("__click('Upload')")
        browser.wait_for("__yes()")
        assert "rejected" in toast_matching(browser, "Upload:", 60).lower()
        assert not browser.js("__toasts.some((t) => t.includes('Mission uploaded'))")

    def test_a_lost_link_shows_in_the_frame(self, browser, stack):
        api, vehicle, uri, base = stack
        api.connect_vehicle(uri, "pymavlink")
        browser.open(f"{base}/workspace.html", settle_s=3)
        browser.js("[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === 'Flight').click(); true")
        browser.wait_for("document.querySelector('.nav-right').innerText.includes('pymavlink')", 15)
        vehicle.go_silent()
        browser.wait_for("document.querySelector('.nav-right').innerText.includes('link lost')", 20)
        browser.wait_for("document.querySelector('.workspace').innerText.includes('LOST')", 20)
        # The Preflight panel ticked GPS, battery and home from the last values heard
        # while the top bar said the link was lost.
        browser.wait_for("document.querySelector('.workspace').innerText"
                         ".includes('Drone Connection — Link lost')", 20)
        panel = browser.js("document.querySelector('.workspace').innerText")
        for name in ("GPS Lock", "Home Position", "Battery"):
            assert f"{name} — Not readable" in panel, panel
        assert "Drone connected and reachable" not in panel

    def test_the_top_bar_reports_state_not_literals(self, browser, stack):
        """'API', '3 workers', a 'PK' avatar and the demo organisation were constants."""
        _, _, _, base = stack
        browser.open(f"{base}/workspace.html", settle_s=3)
        bar = browser.js("document.querySelector('.nav-right').innerText")
        assert "3 workers" not in bar and "PK" not in bar
        # The job count ("no jobs running" / "2 jobs running") and the aircraft link.
        assert "jobs running" in bar and "no aircraft" in bar
        status = browser.js("document.querySelector('.statusbar').innerText")
        assert "ui" in status   # the open project's name, not a demo site


class TestTheDefaultUiFliesTheAircraft:
    def test_connect_upload_preflight_and_fleet(self, browser, stack, tmp_path, monkeypatch):
        pytest.importorskip("sqlalchemy")
        import services.api.db as db_module

        monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(tmp_path / 'fleet.db').as_posix()}")
        monkeypatch.setattr(db_module, "_engine", None)
        monkeypatch.setattr(db_module, "_SessionLocal", None)

        api, vehicle, uri, base = stack
        browser.open(f"{base}/index.html", settle_s=4)
        status = "document.querySelector('#status-message').textContent"

        browser.js("document.querySelector('#btn-connect').click(); true")
        # The dialog's "real autopilot" option is a driver the factory accepts.
        assert browser.js("document.querySelector('#dlg-driver').value") == "pymavlink"
        browser.js(f"document.querySelector('#dlg-uri').value = {uri!r};"
                   "[...document.querySelectorAll('#modal-actions button')]"
                   ".find((x) => x.textContent === 'Connect').click(); true")
        browser.wait_for(f"{status}.includes('Connected')", 20)
        assert browser.js("document.querySelector('#vehicle-chip').textContent") == "LIVE pymavlink"
        time.sleep(2.0)
        grid = browser.js("document.querySelector('#telemetry-grid').innerText")
        assert "STABILIZE" in grid

        browser.js("document.querySelector('#btn-upload').click(); true")
        browser.wait_for(f"{status}.includes('Vehicle accepted')", 60)

        browser.js("document.querySelector('#btn-preflight').click(); true")
        browser.wait_for("!!document.querySelector('[data-confirm=weather]')", 15)
        browser.js("document.querySelector('[data-confirm=weather]').click(); true")
        browser.wait_for("[...document.querySelectorAll('#preflight-list div')]"
                         ".some((d) => d.innerText.includes('Weather') && d.innerText.includes('confirmed'))", 15)

        # Fleet was absent from this UI, and failed on a fresh database.
        browser.js("document.querySelector('.tab[data-tab=fleet]').click(); true")
        browser.js("document.querySelector('#btn-fleet-aircraft').click(); true")
        browser.wait_for("!!document.querySelector('#f-name')")
        browser.js("document.querySelector('#f-name').value = 'ODK-7';"
                   "[...document.querySelectorAll('#modal-actions button')]"
                   ".find((x) => x.textContent === 'Save').click(); true")
        browser.wait_for("document.querySelector('#fleet-list').innerText.includes('ODK-7')", 20)

        vehicle.reject_upload = True
        browser.js("document.querySelector('#btn-upload').click(); true")
        browser.wait_for(f"{status}.includes('rejected')", 60)

        vehicle.go_silent()
        browser.wait_for("document.querySelector('#vehicle-chip').textContent.startsWith('LINK LOST')", 20)
