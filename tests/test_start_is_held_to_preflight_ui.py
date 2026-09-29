"""Start Mission held to the preflight checklist, pressed in both UIs.

The cockpit's Start Mission and the classic window's Fly > Start went straight to the
aircraft whatever the checklist said. Now the pilot is shown what is not ready and asked;
declining sends nothing, and starting anyway is recorded as an override. Headless Edge,
the production Api, and tests/fake_mavlink_vehicle.py.
"""

from __future__ import annotations

import time

import pytest

from test_flight_ui_live import HELPERS, browser, stack, toast_matching  # noqa: F401
from test_start_is_held_to_preflight import _overrides, _started


# ------------------------------------------------------------------ both UIs

def _no():
    return ("(() => { const b = [...document.querySelectorAll('.modal-actions button')]"
            ".find((x) => !x.classList.contains('primary')); if (!b) return false; b.click(); return true; })()")


def test_the_cockpit_start_button_shows_what_is_not_ready_and_asks(browser, stack):
    api, vehicle, uri, base = stack
    assert api.connect_vehicle(uri, "pymavlink")["ok"]
    time.sleep(2.4)
    assert api.upload_mission()["ok"]
    vehicle.accept_modes = True
    browser.open(f"{base}/workspace.html", settle_s=3)
    browser.js(HELPERS)
    browser.js("[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === 'Flight').click(); true")
    time.sleep(1.5)
    panel = browser.js("document.querySelector('.workspace').innerText")
    assert "Not ready to start" in panel, panel

    # Start Mission -> "Arm and start?" yes -> the checklist refusal, answered No.
    browser.js("__click('Start Mission')")
    browser.wait_for("__yes()")
    question = browser.wait_for(
        "(() => { const t = document.querySelector('.modal-text');"
        " return t && t.textContent.includes('Preflight is not ready') ? t.textContent : false; })()")
    assert "Weather Acknowledged" in question and "anyway" in question, question
    browser.wait_for(_no())
    assert "preflight is not ready" in toast_matching(browser, "not started")
    time.sleep(0.5)
    assert not _started(vehicle)
    assert _overrides(api) == []

    # Again, and this time the pilot starts anyway: on the record, and the aircraft hears it.
    browser.js("__click('Start Mission')")
    browser.wait_for("__yes()")
    browser.wait_for("!!document.querySelector('.modal-text') && "
                     "document.querySelector('.modal-text').textContent.includes('Preflight is not ready')")
    browser.wait_for("__yes()")
    assert "Accepted by the vehicle" in toast_matching(browser, "Start Mission:")
    assert _started(vehicle)
    [override] = _overrides(api)
    assert "Weather Acknowledged" in override["unconfirmed"], override


def test_the_classic_start_command_shows_what_is_not_ready_and_asks(browser, stack):
    api, vehicle, uri, base = stack
    assert api.connect_vehicle(uri, "pymavlink")["ok"]
    time.sleep(2.4)
    assert api.upload_mission()["ok"]
    vehicle.accept_modes = True
    browser.open(f"{base}/index.html", settle_s=4)
    status = "document.querySelector('#status-message').textContent"
    title = "document.querySelector('#modal-title').textContent"

    browser.js("window.odk.onMenu({action: 'fly.command', payload: {command: 'start'}}); true")
    browser.wait_for(f"{title} === 'Preflight is not ready'", 20)
    assert "Weather Acknowledged" in browser.js("document.querySelector('#modal-body').innerText")
    browser.js("[...document.querySelectorAll('#modal-actions button')]"
               ".find((x) => x.textContent === 'Do not start').click(); true")
    browser.wait_for(f"{status}.startsWith('Not started')", 10)
    assert not _started(vehicle)

    browser.js("window.odk.onMenu({action: 'fly.command', payload: {command: 'start'}}); true")
    browser.wait_for(f"{title} === 'Preflight is not ready' && "
                     "!document.querySelector('#modal-backdrop').classList.contains('hidden')", 20)
    browser.js("[...document.querySelectorAll('#modal-actions button')]"
               ".find((x) => x.textContent === 'Start anyway').click(); true")
    browser.wait_for(f"{status}.includes('Command start')", 20)
    assert _started(vehicle)
    assert len(_overrides(api)) == 1

    # A confirmed manual check reads "confirmed", not "pass".
    browser.js("document.querySelector('#btn-preflight').click(); true")
    browser.wait_for("!!document.querySelector('[data-confirm=weather]')", 15)
    browser.js("document.querySelector('[data-confirm=weather]').click(); true")
    row = browser.wait_for("[...document.querySelectorAll('#preflight-list div')]"
                           ".map((d) => d.innerText).find((t) => t.includes('Weather') && t.includes('confirmed'))", 15)
    assert "pass" not in row, row
