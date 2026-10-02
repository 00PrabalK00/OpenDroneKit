"""A multi-facade mission's wall-to-wall transits go round the building, not past it.

Each wall's passes sat at the stand-off, but the leg from where one wall's serpentine
ended to where the next wall's began was flown straight. When a wall finished at its
far end, that leg ran back alongside the finished wall and cut the corner. On a
60 x 30 m rectangle at a 10 m stand-off, with the wall given as 5-30 m, waypoint 227
(30, 25) at 30 m to waypoint 228 (-40, 15) at 20 m passed the corner (-30, 15) 1.4 m
away at about 21 m, inside the wall's height. No warning was given, and the plan
exported to every format. The older leg test only asked for half the stand-off.

These tests plan through the real Api (and, in the last class, through the cockpit in
headless Edge/Chromium), export the QGC .plan, and measure every leg between exported
positions against the drawn footprint, sampled every 0.25 m. The footprint geometry is
computed here from the inputs, not read back from the planner.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

LAT0, LON0 = 52.2051, 0.1180
M_PER_DEG = 6371008.8 * math.pi / 180.0
NAV = (16, 19)

RECT = [(-30, -15), (30, -15), (30, 15), (-30, 15)]
L_CCW = [(-30, -20), (30, -20), (30, 0), (0, 0), (0, 20), (-30, 20)]
L_CW = L_CCW[::-1]
U_COURT = [(-40, -20), (40, -20), (40, 20), (15, 20), (15, 0), (-15, 0), (-15, 20), (-40, 20)]
U_NARROW = [(-30, -20), (30, -20), (30, 20), (6, 20), (6, 0), (-6, 0), (-6, 20), (-30, 20)]
# The export rounds positions to 1e-7 deg (about 1 cm); allow for it.
ROUNDING_M = 0.03


def lonlat(x: float, y: float) -> list[float]:
    return [LON0 + x / (M_PER_DEG * math.cos(math.radians(LAT0))), LAT0 + y / M_PER_DEG]


def east_north(lon: float, lat: float) -> np.ndarray:
    return np.array([(lon - LON0) * math.cos(math.radians(LAT0)) * M_PER_DEG,
                     (lat - LAT0) * M_PER_DEG])


def distance_to_walls(p, ring) -> float:
    best = float("inf")
    for a, b in zip(ring, ring[1:] + ring[:1]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * ab - p)))
    return best


def inside(p, ring) -> bool:
    result = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
                result = not result
    return result


def exported_positions(plan_path: Path) -> list[tuple[np.ndarray, float]]:
    items = json.loads(Path(plan_path).read_text())["mission"]["items"]
    return [(east_north(i["params"][5], i["params"][4]), float(i["params"][6]))
            for i in items if i["command"] in NAV and i["params"][4]]


def closest_leg(positions, ring) -> tuple[float, int, bool]:
    """(closest approach of any leg to a wall, that leg's index, any sample inside)."""
    best, where, entered = float("inf"), -1, False
    for index, ((a, _), (b, _)) in enumerate(zip(positions, positions[1:])):
        n = max(1, int(np.linalg.norm(b - a) / 0.25))
        for k in range(n + 1):
            q = a + (b - a) * (k / n)
            entered = entered or inside(q, ring)
            d = distance_to_walls(q, ring)
            if d < best:
                best, where = d, index
    return best, where, entered


def plan_multi_facade(tmp_path: Path, footprint, **options) -> tuple[dict, dict]:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("mf", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi([lonlat(x, y) for x, y in footprint])["ok"]
    planned = api.plan_mission({"template": "multi_facade", "altitude_m": 50, **options})
    assert planned["ok"], planned.get("error")
    exported = api.export_mission(None, str(tmp_path / "out"))
    assert exported["ok"], exported.get("error")
    failed = {k: v for k, v in exported["written"].items() if str(v).startswith("failed")}
    assert not failed, failed
    return planned, exported["written"]


class TestEveryLegKeepsTheStandOff:
    @pytest.mark.parametrize("name, footprint, standoff, walls", [
        ("rectangle", RECT, 10.0, None),
        ("rectangle", RECT, 15.0, None),
        ("rectangle", RECT, 8.0, None),
        ("rectangle, walls 5-30 m", RECT, 10.0, (5.0, 30.0)),
        ("L, anticlockwise", L_CCW, 10.0, None),
        ("L, clockwise", L_CW, 10.0, None),
        ("L, anticlockwise", L_CCW, 15.0, None),
        ("U with a courtyard", U_COURT, 8.0, None),
        ("U with a courtyard too narrow to enter", U_NARROW, 8.0, None),
    ])
    def test_no_leg_comes_inside_the_stand_off(self, tmp_path, name, footprint, standoff, walls):
        options = {"standoff_m": standoff}
        if walls:
            options.update(facade_bottom_altitude_m=walls[0], facade_top_altitude_m=walls[1])
        planned, written = plan_multi_facade(tmp_path, footprint, **options)
        positions = exported_positions(Path(written["qgc_plan"]))[1:]  # after the climb-out
        assert len(positions) > 20
        closest, leg, entered = closest_leg(positions, footprint)
        assert not entered, f"{name}: a leg crosses the building"
        assert closest >= 0.95 * standoff - ROUNDING_M, (
            f"{name} at {standoff} m: leg {leg} passes {closest:.2f} m from a wall "
            f"({closest / standoff:.2f}x the stand-off)")
        # Round the corner, not over the roof: a climb would be said, and none is needed.
        assert not [w for w in planned["warnings"] if "transit" in w.lower()]

    def test_the_transit_takes_no_photographs_and_faces_the_building(self, tmp_path):
        planned, _ = plan_multi_facade(tmp_path, RECT, standoff_m=10.0)
        waypoints = [f for f in planned["geojson"]["features"]
                     if f["properties"].get("role") == "waypoint"]
        transits = [f for f in waypoints if f["properties"].get("primitive") == "facade_transit"]
        assert transits, "the corners are not routed round"
        assert not any(f["properties"].get("trigger") for f in transits)


HELPERS = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((mutations) => mutations.forEach((m) =>
    m.addedNodes.forEach((n) => {
      if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent);
    })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
window.__field = (label, value) => {
  const field = [...document.querySelectorAll('.field')]
    .find((f) => f.querySelector('label') && f.querySelector('label').textContent.trim() === label);
  if (!field) throw new Error('no field ' + label);
  const input = field.querySelector('select, input');
  input.value = value;
  input.dispatchEvent(new Event('change'));
  return input.value;
};
window.__click = (text) => {
  const b = [...document.querySelectorAll('.toolbar button')].find((x) => x.textContent.trim() === text);
  if (!b) throw new Error('no toolbar button ' + text);
  b.click();
  return true;
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


class TestFromTheCockpit:
    def test_a_multi_facade_planned_and_exported_in_the_cockpit_keeps_clear(
            self, tmp_path, monkeypatch, browser):
        from ui_live_bridge import serve

        monkeypatch.chdir(tmp_path)
        api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
        assert api.create_project("mf-ui", str(tmp_path / "project"))["ok"]
        assert api.set_aoi([lonlat(x, y) for x, y in RECT])["ok"]
        server, base = serve(api)
        try:
            browser.open(f"{base}/workspace.html", settle_s=3)
            browser.js(HELPERS)
            browser.js("[...document.querySelectorAll('.nav-item')]"
                       ".find((x) => x.textContent === 'Mission Planning').click(); true")
            time.sleep(0.5)
            assert browser.js("__field('Type', 'Multi-facade')") == "Multi-facade"
            assert browser.js("__field('Stand-off', '10')") == "10"
            browser.js("__click('Plan')")
            browser.wait_for("__toasts.find((t) => /Planned|waypoint/i.test(t))", timeout_s=60)

            browser.js("window.odk.onMenu({action: 'mission.export', payload: {format: 'qgc_plan'}}); true")
            browser.wait_for("__toasts.find((t) => t.includes('Exported'))", timeout_s=60)
            plans = sorted((tmp_path / "project" / "missions").glob("*.plan"))
            assert plans, "the menu wrote no .plan"
            mission = json.loads(plans[-1].read_text())
            positions = exported_positions(plans[-1])[1:]
            assert len(positions) > 20
            # The form reached the planner: facade poses at the typed 10 m, not a grid
            # over the roof or the 12 m default.
            nearest = min(distance_to_walls(p, RECT) for p, _ in positions)
            assert abs(nearest - 10.0) < 0.6, nearest
            closest, leg, entered = closest_leg(positions, RECT)
            assert not entered
            assert closest >= 0.95 * 10.0 - ROUNDING_M, f"leg {leg} passes {closest:.2f} m from a wall"
            assert mission["mission"]["items"]
        finally:
            server.shutdown()
