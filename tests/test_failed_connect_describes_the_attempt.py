"""A failed connect describes the connection that was attempted, not the last one.

The re-audit connected the mock, then tried pymavlink with nothing listening. The
result said vehicle.connected:false -- correct -- but its flight_log block said
{driver: 'mock', simulated: true}: the PREVIOUS connection's log, because the session
returns before starting a log on failure and the Api read whatever log was current.
A result that describes a log for a link that never opened is a small lie; this pins
it both at the Api and through the cockpit's own Connect button in a real browser.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("pymavlink")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from fake_mavlink_vehicle import free_udp_port  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("connect", str(tmp_path / "project"))
    api.set_aoi(AOI)
    yield api
    api.disconnect_vehicle()


def _dead_uri() -> str:
    # A UDP port nobody sends heartbeats to: pymavlink waits, hears nothing, fails.
    return f"udpin:127.0.0.1:{free_udp_port()}"


def test_a_failed_link_after_a_mock_flight_reports_the_attempt(api):
    mocked = api.connect_vehicle("mock://", "mock")
    assert mocked["flight_log"]["driver"] == "mock" and mocked["flight_log"]["started"] is True
    for _ in range(3):
        api.telemetry()
        time.sleep(0.1)

    uri = _dead_uri()
    failed = api.connect_vehicle(uri, "pymavlink")
    # A link that did not open is a refusal, with the reason as the error.
    assert failed["ok"] is False
    assert failed["error"] == failed["vehicle"]["last_error"]
    assert "heartbeat" in failed["error"].lower(), failed["error"]
    assert failed["vehicle"]["connected"] is False and failed["vehicle"]["last_error"]
    log = failed["flight_log"]
    assert log["driver"] == "pymavlink", log
    assert log["uri"] == uri
    assert log["simulated"] is False
    assert log["started"] is False
    assert log["archived"] is None
    assert "not" in log["note"].lower()


def test_the_cockpit_connect_button_gets_the_attempt_back(api):
    pytest.importorskip("websocket")
    pytest.importorskip("requests")
    from ui_live_bridge import Browser, chromium_path, serve

    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    api.connect_vehicle("mock://", "mock")
    server, base = serve(api)
    browser = Browser(executable)
    try:
        browser.open(f"{base}/workspace.html", settle_s=0.5)
        # Wait for the shell itself, not a fixed delay: under load 3 s was too short.
        browser.wait_for("!!(window.pywebview && window.pywebview.api"
                         " && [...document.querySelectorAll('.nav-item')]"
                         ".some((x) => x.textContent === 'Flight'))", timeout_s=60)
        browser.js("""
window.__toasts = [];
new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
  if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent);
}))).observe(document.body, { childList: true, subtree: true });
window.__connects = [];
const real = window.pywebview.api;
window.pywebview.api = new Proxy({}, { get(_t, name) {
  const fn = real[name];
  if (name !== 'connect_vehicle' || !fn) return fn;
  return async (...args) => { const r = await fn(...args); window.__connects.push(r); return r; };
}});
window.__click = (text) => {
  const b = [...document.querySelectorAll('.toolbar button')].find((x) => x.textContent.trim() === text);
  if (!b) throw new Error('no toolbar button ' + text);
  b.click(); return true;
};
window.__choose = (text) => {
  const row = [...document.querySelectorAll('.modal-row')].find((x) => x.textContent.includes(text));
  if (!row) return false; row.click(); return true;
};
window.__answer = (value) => {
  const input = document.querySelector('.modal input');
  if (!input) return false;
  input.value = value; document.querySelector('.modal-actions .primary').click(); return true;
};
true
""")
        browser.js("[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === 'Flight').click(); true")
        browser.wait_for("[...document.querySelectorAll('.toolbar button')]"
                         ".some((x) => x.textContent.trim() === 'Connect')", timeout_s=30)
        uri = _dead_uri()
        browser.js("__click('Connect')")
        browser.wait_for("__choose('MAVLink autopilot')")
        browser.wait_for(f"__answer({uri!r})")
        result = browser.wait_for("window.__connects[0]", timeout_s=40)
        assert result["ok"] is False and "heartbeat" in result["error"].lower(), result
        assert result["vehicle"]["connected"] is False
        log = result["flight_log"]
        assert log["driver"] == "pymavlink" and log["uri"] == uri, log
        assert log["started"] is False and log["simulated"] is False
        # The operator sees the failure, not "Connected".
        browser.wait_for("__toasts.length > 0", timeout_s=10)
        toasts = browser.js("__toasts")
        assert not any("Connected to" in t for t in toasts), toasts
        assert any("heartbeat" in t.lower() for t in toasts), toasts
    finally:
        browser.close()
        server.shutdown()


def test_the_classic_connect_dialog_reports_the_refusal(api):
    """index.html's Connect dialog: a refused link says why and never 'Connected'."""
    pytest.importorskip("websocket")
    pytest.importorskip("requests")
    from ui_live_bridge import Browser, chromium_path, serve

    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    server, base = serve(api)
    browser = Browser(executable)
    try:
        browser.open(f"{base}/index.html", settle_s=0.5)
        browser.wait_for("!!(window.pywebview && window.pywebview.api"
                         " && document.querySelector('#btn-connect'))", timeout_s=60)
        status = "document.querySelector('#status-message').textContent"
        uri = _dead_uri()
        browser.js("document.querySelector('#btn-connect').click(); true")
        browser.wait_for("!!document.querySelector('#dlg-uri')")
        browser.js(f"document.querySelector('#dlg-driver').value = 'pymavlink';"
                   f"document.querySelector('#dlg-uri').value = {uri!r};"
                   "[...document.querySelectorAll('#modal-actions button')]"
                   ".find((x) => x.textContent === 'Connect').click(); true")
        browser.wait_for(f"{status}.includes('Connection failed')", timeout_s=40)
        text = browser.js(status)
        assert "heartbeat" in text.lower() and "Connected to" not in text, text
        chip = browser.js("document.querySelector('#vehicle-chip').textContent")
        assert "LIVE" not in chip, chip
    finally:
        browser.close()
        server.shutdown()
