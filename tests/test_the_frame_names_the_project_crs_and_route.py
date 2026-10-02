"""Two things the cockpit frame said that were not so.

* The status bar read "project EPSG:4326" for every project. shell.js took get_state's
  display_epsg -- the coordinate readout setting, always 4326 -- ahead of the project's
  own layers, so a project whose DSM is in UTM 17N, and an empty project with no CRS at
  all, both claimed WGS84.
* Mission Planning's Simulation Timeline promised "Scrub to preview the flight. Playback
  follows the aircraft along the route." over an empty frame. It is now drawn from the
  planned route, and the guide says so.

Driven through workspace.html in headless Chromium against the real Api.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from cockpit_harness import Cockpit, make_api

rasterio = pytest.importorskip("rasterio")

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]
CRS_JS = "(ODK.$$('.statusbar .mono').find((n) => /^map EPSG/.test(n.textContent)) || {}).textContent || ''"


def _utm_dsm(path):
    """A small DSM in UTM zone 17N, over the Aukerman site (a test input)."""
    from rasterio.transform import from_origin

    heights = np.full((20, 20), 300.0, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1,
                       dtype="float32", crs="EPSG:32617",
                       transform=from_origin(437000.0, 4574000.0, 1.0, 1.0)) as dst:
        dst.write(heights, 1)
    return path


def test_the_status_bar_names_the_projects_own_crs(tmp_path):
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "empty") as cockpit:
        empty = cockpit.run(f"""
            await ODK.waitFor(() => /^connected/.test((ODK.$('.statusbar .chip') || {{}}).textContent || ''));
            await ODK.waitFor(() => /project/.test({CRS_JS}), 15000);
            return {{ crs: {CRS_JS} }};
        """)
    assert empty["crs"] == "map EPSG:4326 · project CRS not known yet", empty

    added = api.add_layer_from_file(str(_utm_dsm(tmp_path / "dsm_utm.tif")))
    assert added["ok"], added
    assert added["layer"]["crs_epsg"] == 32617, added
    assert api.get_state()["display_epsg"] == 4326   # the value that used to win
    with Cockpit(api, tmp_path / "utm") as cockpit:
        utm = cockpit.run(f"""
            await ODK.waitFor(() => /32617/.test({CRS_JS}), 20000);
            return {{ crs: {CRS_JS} }};
        """)
    assert utm["crs"] == "map EPSG:4326 · project EPSG:32617", utm


def test_the_simulation_timeline_scrubs_the_planned_route(tmp_path):
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "none") as cockpit:
        before = cockpit.run("""
            await ODK.nav("Mission Planning");
            return { panel: await ODK.waitFor(() => {
              const t = ODK.panel("Simulation Timeline"); return t && !/Reading/.test(t) ? t : null; }, 15000) };
        """)
    assert before["panel"].startswith("Plan a mission"), before

    assert api.set_aoi(AOI)["ok"]
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    estimates = api.mission_estimates()["estimates"]
    waypoints = len(api._session.mission_plan_dict["waypoints"])  # noqa: SLF001
    with Cockpit(api, tmp_path / "plan") as cockpit:
        seen = cockpit.run("""
            const out = {};
            await ODK.nav("Mission Planning");
            await ODK.waitFor(() => ODK.$(".timeline-scrub"), 20000);
            out.first = ODK.$(".timeline-readout").textContent;
            out.legend = ODK.panel("Simulation Timeline");
            out.hasProfile = Boolean(ODK.$(".timeline-profile polyline"));
            const scrub = ODK.$(".timeline-scrub");
            scrub.value = scrub.max;
            scrub.dispatchEvent(new Event("input"));
            await ODK.sleep(300);
            out.last = ODK.$(".timeline-readout").textContent;
            out.max = scrub.max;
            await ODK.map();
            scrub.value = "3"; scrub.dispatchEvent(new Event("input"));
            out.marker = Boolean(await ODK.waitFor(() => ODK.$(".timeline-aircraft"), 10000));
            const play = ODK.$$(".simulation-timeline button").find((b) => b.textContent === "Play");
            play.click();
            await ODK.sleep(1500);
            out.afterPlay = Number(scrub.value);
            return out;
        """, virtual_time_ms=120000, timeout_s=480)
    assert seen["hasProfile"], seen
    assert int(seen["max"]) == waypoints - 1, seen
    assert seen["first"].startswith("0:00 of "), seen
    assert f"waypoint 1/{waypoints}" in seen["first"] and "altitude 60.0 m" in seen["first"], seen
    total = round(float(estimates["duration_min"]) * 60)
    clock = f"{total // 60}:{total % 60:02d}"
    assert seen["last"].startswith(f"{clock} of {clock}"), (clock, seen)
    assert f"waypoint {waypoints}/{waypoints}" in seen["last"], seen
    assert re.search(r"\d+\.\d m/s", seen["legend"]), seen
    assert seen["marker"], seen
    assert seen["afterPlay"] > 3, seen


def test_the_clock_shows_no_reading_before_the_application_answers(tmp_path):
    """The header read "no bridge ... 00:00:00 UTC" -- the demo's epoch -- until the
    bridge connected, and for good with no bridge. No reading is shown as none."""
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "nb", bridge=False) as cockpit:
        seen = cockpit.run("""
            await ODK.sleep(3000);
            return { header: ODK.$(".topnav .nav-right").innerText };
        """)
    assert "--:--:-- UTC" in seen["header"], seen
    assert "00:00:00" not in seen["header"], seen
