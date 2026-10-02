"""What the Mission Planning form shows is what the planner gets, edited or not.

The cockpit sent only the fields an operator had changed. The form showed Stand-off
12 m, Altitude AGL 60 m and RTH 90 m, and with only Type changed plan_mission received
none of them and used its own 8 m, 55 m and 80 m: a multi-facade flown 7.65 m off the
wall and a tower of radius 4 flown 8 m off its surface while the screen said 12. The
Planned message never said which stand-off or altitude it had used.

Driven in headless Edge/Chromium against the production Api (tests/ui_live_bridge.py):
pick the Type, press Plan, and read what the Api was asked for and what it planned.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

R = 6371008.8 * math.pi / 180
LAT0, LON0 = 47.6, -122.3
# An L-shaped building, in metres east / north of LAT0, LON0.
L_SHAPE = [(-30, -20), (30, -20), (30, 0), (0, 0), (0, 20), (-30, 20)]


def lonlat(x, y):
    return [LON0 + x / (R * math.cos(math.radians(LAT0))), LAT0 + y / R]


def metres(lon, lat):
    return ((lon - LON0) * math.cos(math.radians(LAT0)) * R, (lat - LAT0) * R)


def distance_to_ring(p, ring):
    best = math.inf
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(ax + t * dx - p[0], ay + t * dy - p[1]))
    return best


HELPERS = r"""
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => {
    if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent); })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
window.__input = (label) => {
  const f = [...document.querySelectorAll('.field')].find((f) =>
    f.querySelector('label') && f.querySelector('label').textContent.trim() === label);
  return f ? f.querySelector('select, input') : null;
};
window.__field = (label, value) => {
  const i = window.__input(label);
  if (!i) throw new Error('no field ' + label);
  i.value = value; i.dispatchEvent(new Event('change')); return i.value;
};
window.__shown = (label) => { const i = window.__input(label); return i ? i.value : null; };
window.__click = (text) => {
  const b = [...document.querySelectorAll('.toolbar button')].find((x) => x.textContent.trim() === text);
  if (!b) throw new Error('no button ' + text); b.click(); return true;
};
window.__open = (title) => {
  [...document.querySelectorAll('.nav-item')].find((x) => x.textContent === title).click(); return true;
};
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


@pytest.fixture
def cockpit(tmp_path, monkeypatch, browser):
    """The cockpit on Mission Planning over an L-shaped footprint; records each plan request."""
    from ui_live_bridge import serve

    monkeypatch.chdir(tmp_path)
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    assert api.create_project("form", str(tmp_path / "project"))["ok"]
    assert api.set_aoi([lonlat(*p) for p in L_SHAPE])["ok"]

    asked: list[dict] = []
    real_plan = api.plan_mission

    def recording_plan(options=None):
        asked.append(dict(options or {}))
        return real_plan(options)

    api.plan_mission = recording_plan
    server, base = serve(api)
    browser.open(f"{base}/workspace.html", settle_s=3)
    browser.js(HELPERS)
    browser.js("__open('Mission Planning')")
    time.sleep(0.5)

    def plan(**fields):
        for label, value in fields.items():
            assert browser.js(f"__field({json.dumps(label)}, {json.dumps(value)})") == value
        browser.js("window.__toasts.length = 0; __click('Plan'); true")
        toast = str(browser.wait_for(
            "__toasts.find((t) => /Planned|refus|fail|cannot|error/i.test(t))", timeout_s=120))
        assert toast.startswith("Planned"), toast
        return asked[-1], toast

    try:
        yield api, plan, browser
    finally:
        server.shutdown()


def waypoints(api):
    features = (api._session.mission_plan_dict.get("geojson") or {}).get("features", [])
    return [metres(*f["geometry"]["coordinates"][:2]) for f in features
            if f["properties"].get("role") == "waypoint"]


class TestTheFormDefaultsReachThePlanner:
    def test_multi_facade_is_flown_at_the_stand_off_the_form_shows(self, cockpit):
        api, plan, browser = cockpit
        assert browser.js("__shown('Stand-off')") == "12"
        sent, toast = plan(Type="Multi-facade")

        assert sent["template"] == "multi_facade"
        assert sent["standoff_m"] == 12
        assert sent["altitude_m"] == 60
        assert sent["rth_altitude_m"] == 90       # on the Aircraft tab, never opened
        meta = (api._session.mission_plan_dict.get("flight_recipe") or {}).get("metadata") or {}
        assert meta.get("facade_standoff_m") == pytest.approx(12.0)
        poses = waypoints(api)
        assert poses
        # The facade poses sit at the stand-off, not the planner's own 8 m.
        assert min(distance_to_ring(p, L_SHAPE) for p in poses) >= 0.95 * 12
        assert "stand-off 12 m" in toast and "altitude 60 m AGL" in toast and "RTH 90 m" in toast

    def test_tower_stand_off_is_measured_from_the_surface(self, cockpit):
        api, plan, _ = cockpit
        sent, toast = plan(**{"Type": "Tower mapping", "Tower radius": "4"})
        assert sent["standoff_m"] == 12 and sent["tower_object_radius_m"] == 4
        meta = (api._session.mission_plan_dict.get("flight_recipe") or {}).get("metadata") or {}
        assert meta.get("tower_flight_radius_m") == pytest.approx(16.0)
        assert "stand-off 12 m" in toast
        # The planner lifts RTH above the tower's top; the message says what it flies.
        flown_rth = api._session.mission_plan_dict["safety_constraints"]["rth_altitude_m"]
        if abs(flown_rth - 90) > 0.05:
            assert f"RTH 90 m (planned {flown_rth:.1f} m)" in toast, toast

    def test_the_default_type_is_what_is_planned(self, cockpit):
        _, plan, browser = cockpit
        assert browser.js("__shown('Type')") == "3D modelling"
        sent, toast = plan()
        assert sent["template"] == "double_grid"
        assert sent["altitude_m"] == 60 and sent["front_overlap_pct"] == 80
        assert sent["side_overlap_pct"] == 70
        assert "altitude 60 m AGL" in toast


class TestTheFormKeepsWhatWasTyped:
    def test_a_redrawn_form_shows_the_edited_value_and_plans_it(self, cockpit):
        _, plan, browser = cockpit
        browser.js("__field('Stand-off', '15'); __field('Type', 'Multi-facade'); true")
        browser.js("__open('Processing'); true")
        time.sleep(0.3)
        browser.js("__open('Mission Planning'); true")
        time.sleep(0.5)
        assert browser.js("__shown('Stand-off')") == "15"
        assert browser.js("__shown('Type')") == "Multi-facade"
        sent, toast = plan()
        assert sent["standoff_m"] == 15 and sent["template"] == "multi_facade"
        assert "stand-off 15 m" in toast

    def test_picking_a_mission_type_in_the_list_sets_type(self, cockpit):
        _, plan, browser = cockpit
        browser.js("""[...document.querySelectorAll('.tree-node .label')]
            .find((n) => n.textContent === 'Orbit').closest('.tree-node').click(); true""")
        time.sleep(0.3)
        assert browser.js("__shown('Type')") == "Orbit"
        sent, _ = plan()
        assert sent["template"] == "orbit"
