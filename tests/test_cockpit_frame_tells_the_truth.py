"""The cockpit's frame -- header, status bar, banner, palette, keys -- reports what is real.

Audit, before this change, connected to a real project:

  * header: "3 workers" (this build runs jobs on threads, there are no workers), the
    avatar "PK" and "DEMO organisation", and a clock frozen at 00:00:00 UTC;
  * status bar: always "OpenDroneKit Aukerman survey", whatever project was open;
  * palette: two demo sites that did nothing when chosen, a finding "F-118" and a mission
    "Roof Block A v3" that existed nowhere; Ctrl+S, Ctrl+B, F and Ctrl+Z listed and unbound;
  * EXAMPLE DATA over real data on five screens, triggered by static menus;
  * ?demo=1 showed the machine's REAL projects under the EXAMPLE DATA banner;
  * no bridge: "No projects yet. Use New Project" -- a claim about a machine nobody asked;
  * panels rendered before pywebview's bridge arrived and never asked again;
  * Activity Log and Recent activity rendered only dashes (wrong field names);
  * six choose() pickers rendered blank rows and resolved undefined.

Each test drives the page in a browser against the real Api.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api


@pytest.fixture
def api(tmp_path):
    return make_api(tmp_path)


def test_the_header_and_status_bar_describe_the_open_project(api, tmp_path) -> None:
    api.create_project("Second Site", str(tmp_path / "second"))
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.sleep(1500);
            const header = ODK.$(".topnav .nav-right").innerText;
            const status = ODK.$(".statusbar").innerText;
            ODK.$(".searchbox").click();
            await ODK.sleep(300);
            const palette = ODK.$$(".palette-item").map((r) => r.innerText.replace(/\\s+/g, " "));
            const row = ODK.$$(".palette-item").find((r) => r.innerText.includes("Harness Site"));
            row.click();
            await ODK.waitFor(() => ODK.toasts().includes("Opened Harness Site."));
            await ODK.sleep(800);
            return { header, status, palette, after: ODK.$(".statusbar").innerText };
        """)
    header = result["header"]
    assert "3 workers" not in header and "PK" not in header and "DEMO organisation" not in header
    assert "00:00:00 UTC" not in header, "the clock is stopped in connected mode"
    assert "no jobs running" in header and "Second Site" in header
    assert "Second Site" in result["status"] and "Aukerman" not in result["status"]
    palette = " ".join(result["palette"])
    assert "F-118" not in palette and "Roof Block A" not in palette
    assert "Harness Site" in palette and "Second Site" in palette
    # Choosing a project in the palette opens it, in the application.
    assert api._session.store.get_active_project()["name"] == "Harness Site"
    assert "Harness Site" in result["after"]


def test_every_listed_shortcut_is_bound(api, tmp_path) -> None:
    api.set_aoi([[-81.7520, 41.3040], [-81.7513, 41.3040], [-81.7513, 41.3044]])
    api.plan_mission({})
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Mission Planning");
            await ODK.key("s", { ctrlKey: true });
            await ODK.answer("Block A");
            const saved = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Saved")));
            await ODK.key("b", { ctrlKey: true });
            const sides = ["left", "right"].every((side) =>
              ODK.$(`.region.${side}`).classList.contains("collapsed"));
            await ODK.key("z", { ctrlKey: true });
            const undo = ODK.lastToast();
            return { saved, sides, undo, toolbar: ODK.$(".toolbar").innerText };
        """)
    assert result["saved"] == 'Saved "Block A".'
    history = api.mission_version_history()
    assert history["ok"] and history["history"], "Ctrl+S did not save a mission version"
    assert result["sides"], "Ctrl+B did not collapse the side panels"
    assert result["undo"].startswith("Nothing to undo")
    assert "Save" in result["toolbar"].split("\n")


def test_the_banner_is_off_over_real_data_and_menus(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            const shown = {};
            for (const name of ["Home", "Mission Planning", "Measurements", "Reports", "Developers", "Settings"]) {
              await ODK.nav(name);
              await ODK.sleep(1200);
              shown[name] = !ODK.$(".demo-banner").classList.contains("hidden");
            }
            return shown;
        """)
    assert not any(result.values()), f"EXAMPLE DATA over a real project on: {result}"


def test_demo_mode_shows_the_example_and_never_reaches_python(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.sleep(2500);
            const tree = ODK.panel("Project Explorer");
            await ODK.nav("Mission Planning");
            const map = await ODK.map();
            await ODK.sleep(800);
            await ODK.press("Plan");
            return { banner: !ODK.$(".demo-banner").classList.contains("hidden"), tree,
                     header: ODK.$(".topnav .nav-right").innerText,
                     example: Boolean(map.getSource("odk-mission-line")),
                     plan: ODK.lastToast() };
        """, query="?demo=1")
    assert cockpit.calls == [], f"demo mode called the application: {cockpit.calls[:5]}"
    assert result["banner"]
    assert "OpenDroneKit Aukerman survey" in result["tree"]
    assert "Harness Site" not in result["tree"]
    assert "00:00:00 UTC" in result["header"] and "DEMO organisation" in result["header"]
    assert result["example"], "the example mission is not drawn in demo mode"
    assert "Demo mode" in result["plan"]


def test_with_no_bridge_the_panels_say_so(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile", bridge=False) as cockpit:
        result = cockpit.run("""
            await ODK.sleep(6000);
            return { tree: ODK.panel("Project Explorer"),
                     banner: !ODK.$(".demo-banner").classList.contains("hidden"),
                     status: ODK.$(".statusbar").innerText };
        """)
    assert result["tree"].startswith("Not connected to the application")
    assert "No projects yet" not in result["tree"]
    assert not result["banner"], "EXAMPLE DATA over a screen with no example on it"
    assert "not connected" in result["status"]


def test_panels_rendered_before_the_bridge_arrives_still_fill_in(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile", ready_delay_ms=2500) as cockpit:
        result = cockpit.run("""
            await ODK.waitFor(() => (ODK.panel("Project Explorer") || "").includes("Harness Site"), 15000);
            return { tree: ODK.panel("Project Explorer") };
        """)
    assert "Harness Site" in result["tree"]


def test_the_activity_log_shows_what_happened(api, tmp_path) -> None:
    api.set_aoi([[-81.7520, 41.3040], [-81.7513, 41.3040], [-81.7513, 41.3044]])
    api.plan_mission({})
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.waitFor(() => (ODK.panel("Activity Log") || "").includes("mission planned"));
            return { log: ODK.panel("Activity Log"), recent: ODK.panel("Recent activity") };
        """)
    assert "mission planned" in result["log"] and "project created" in result["log"]
    assert "template=grid" in result["log"]
    assert "mission planned" in result["recent"]


def test_string_pickers_choose_what_was_clicked(api, tmp_path) -> None:
    cameras = api.list_cameras()["cameras"]
    name = cameras[0]["name"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Mission Planning");
            await ODK.press("Cameras");
            const row = await ODK.pick({name!r});
            await ODK.answer("60");
            const said = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes(" at 60 m")));
            return {{ row, said }};
        """)
    assert result["row"].strip() == name
    described = cockpit.called("describe_camera")
    assert described and described[0]["args"][0] == name
    assert result["said"].startswith(f"{name} at 60 m: GSD")
