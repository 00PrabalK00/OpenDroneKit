"""The cockpit map draws the open project, and its tools draw into the application.

Audit, before this change: all eight canvas tools and the Distance and Area buttons made
zero Api calls. The shell stored `activeTool` and nothing read it, nothing ever assigned
the selection geometry, so Plan always refused with "Draw an area of interest on the map
before planning a mission" on a screen whose polygon tool could not draw one. Meanwhile
every map canvas drew the bundled example's lawnmower mission over whatever project was
open, and the corner readout said 0.0000 N 0.0000 E.

These tests drive the real page in a browser against the real Api and assert on both
ends: what the Python session now holds, and what the map now shows.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api

# A 60 x 40 m block in Ohio, well inside the AOI size limit.
W, E = -81.7520, -81.7513
S, N = 41.3040, 41.3044
RING = [[W, S], [E, S], [E, N], [W, N]]


@pytest.fixture
def api(tmp_path):
    return make_api(tmp_path)


def test_the_polygon_tool_sets_the_area_of_interest_and_plan_uses_it(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Mission Planning");
            const map = await ODK.map();
            const demoMissionBefore = Boolean(map.getSource("odk-mission-line"));
            await ODK.tool("Draw polygon");
            for (const [lon, lat] of {RING}) await ODK.click(map, lon, lat);
            await ODK.key("Enter");
            const aoiToast = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Area of interest set")));
            await ODK.press("Plan");
            const planToast = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Planned")), 30000);
            await ODK.sleep(1500);
            return {{
              demoMissionBefore,
              aoiToast, planToast,
              aoiDrawn: Boolean(map.getLayer("odk-p-aoi-line")),
              missionDrawn: Boolean(map.getLayer("odk-p-mission-line")),
              demoMissionAfter: Boolean(map.getSource("odk-mission-line")),
              readout: ODK.$$(".workspace .canvas .canvas-overlay.br").map((n) => n.innerText).join(" | "),
            }};
        """)

    # The application holds exactly what was clicked.
    assert api._session.aoi_polygon == RING
    assert cockpit.called("set_aoi"), "the polygon tool never reached the Api"
    assert result["aoiToast"].startswith("Area of interest set: 4 vertices")
    # Plan used it: a mission exists, and the toast reports its real size.
    assert api._session.mission_plan_dict, "Plan did not produce a mission from the drawn area"
    waypoints = len(api._session.mission_plan.waypoints)
    assert result["planToast"].startswith(f"Planned {waypoints} waypoints")
    assert "?" not in result["planToast"]
    # The map shows the project's own AOI and plan, and never the bundled example.
    assert result["aoiDrawn"] and result["missionDrawn"]
    assert not result["demoMissionBefore"] and not result["demoMissionAfter"]
    assert "0.0000 N" not in result["readout"]


def test_measure_reports_the_api_measurement(api, tmp_path) -> None:
    expected = api.measure_geometry([[W, S], [E, S]], "distance")["length_m"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Measurements");
            await ODK.press("Distance");
            const armed = ODK.lastToast();
            const map = await ODK.map();
            await ODK.click(map, {W}, {S});
            await ODK.click(map, {E}, {S});
            await ODK.key("Enter");
            const measured = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Distance ")));
            await ODK.press("Area");
            for (const [lon, lat] of {RING}) await ODK.click(map, lon, lat);
            await ODK.key("Enter");
            const area = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Area ")));
            return {{ armed, measured, area, selection: ODK.$(".statusbar").innerText }};
        """)
    assert cockpit.called("measure_geometry"), "Distance took no measurement"
    assert f"{expected:.2f} m" in result["measured"]
    area = api.measure_geometry(RING, "area")["area_m2"]
    assert f"{area:.1f} m²" in result["area"]
    # The drawn area is the selection Volume, Cut & Fill and Slope act on.
    assert "selection: Polygon" in result["selection"] or "Area " in result["selection"]


def test_exclusion_home_undo_and_escape(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Mission Planning");
            const map = await ODK.map();
            await ODK.tool("Exclusion zone");
            await ODK.click(map, {W}, {S});
            await ODK.click(map, {E}, {S});
            await ODK.click(map, -81.7500, 41.3000);   // a stray click...
            await ODK.key("z", {{ ctrlKey: true }});     // ...undone
            await ODK.click(map, {E}, {N});
            await ODK.key("Enter");
            const zone = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Exclusion zone added")));
            await ODK.tool("Place home");
            await ODK.click(map, {W}, {N});
            const home = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Home placed")));
            await ODK.tool("Draw polygon");
            await ODK.click(map, {W}, {S});
            await ODK.key("Escape");
            await ODK.sleep(500);
            return {{ zone, home, noFlyDrawn: Boolean(map.getLayer("odk-p-nofly-fill")),
                      markersDrawn: Boolean(map.getLayer("odk-p-markers-pt")),
                      cancelled: ODK.toasts().includes("Drawing cancelled.") }};
        """)
    assert api._session.no_fly_polygons == [[[W, S], [E, S], [E, N]]], "undo did not remove the stray point"
    markers = api.list_site_markers()["markers"]
    assert [(m["name"], m["kind"]) for m in markers] == [("Home", "takeoff")]
    assert result["noFlyDrawn"] and result["markersDrawn"]
    assert result["cancelled"]
    # Escape abandoned the half-drawn polygon rather than sending it.
    assert not cockpit.called("set_aoi")


def test_an_empty_project_says_so_instead_of_showing_the_example(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Flight");
            const map = await ODK.map();
            await ODK.sleep(1500);
            const note = ODK.$(".workspace .canvas .map-note");
            return { note: note && !note.classList.contains("hidden") ? note.innerText : "",
                     demoMission: Boolean(map.getSource("odk-mission-line")) };
        """)
    assert "Nothing in this project to place on the map yet" in result["note"]
    assert not result["demoMission"]
