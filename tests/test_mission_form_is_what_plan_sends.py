"""What the Mission Properties panel shows is what Plan sends.

The panel displayed Type "3D modelling", Altitude AGL 60, overlaps 80/70 and Speed 8,
and pressing Plan without touching the form sent plan_mission({}) -- only EDITED fields
were sent -- so the planner used its own defaults and flew a 55 m grid. The operator
saw one mission and got another.

Three more ways the panel and the plan parted, fixed with it:

* a redraw of the panel (switching workspace and back) put the defaults back in the
  inputs while Plan kept sending what had been typed;
* picking a type in the Mission Types list printed its name and left the form's Type,
  which is what Plan sends, as it was;
* the Camera tab described a "P1 / 35 mm" the planner never used; it now describes the
  Camera field's camera at the Altitude field's height, from the camera database.

Driven through the shipped workspace.html in headless Chromium against the real Api.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


def _plans(cockpit):
    return [c for c in cockpit.calls if c["method"] == "plan_mission"]


def test_plan_sends_what_the_untouched_form_shows(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const out = {};
            await ODK.nav("Mission Planning");
            await ODK.waitFor(() => ODK.$('[data-key="alt"]'), 10000);
            const shown = (key) => (ODK.$(`[data-key="${key}"]`) || {}).value;
            out.shown = { type: shown("type"), alt: shown("alt"), fwd: shown("fwd"),
                          side: shown("side"), speed: shown("speed"), camera: shown("camera") };
            await ODK.press("Plan");
            out.planned = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Planned")), 30000);

            // Edit one field, leave the workspace, come back: the input must still show it.
            const alt = ODK.$('[data-key="alt"]');
            alt.value = "80"; alt.dispatchEvent(new Event("change"));
            await ODK.nav("Home");
            await ODK.nav("Mission Planning");
            await ODK.waitFor(() => ODK.$('[data-key="alt"]'), 10000);
            out.altAfterRedraw = shown("alt");
            await ODK.press("Plan");
            await ODK.sleep(200);
            out.planned2 = await ODK.waitFor(() => ODK.toasts().filter((t) => t.startsWith("Planned")).length >= 2, 30000);

            // Choosing a type in the Mission Types list is what Plan sends next.
            const row = ODK.$$(".panel .tree-node")
              .find((r) => r.textContent.trim().endsWith("2D mapping"));
            out.foundTypeRow = Boolean(row);
            if (row) row.click();
            await ODK.sleep(600);
            out.typeAfterPick = shown("type");
            await ODK.press("Plan");
            await ODK.waitFor(() => ODK.toasts().filter((t) => t.startsWith("Planned")).length >= 3, 30000);

            // The Camera tab: the camera Plan used, at the altitude on the form.
            const tab = ODK.$$(".panel-tab").find((t) => t.textContent.trim() === "Camera");
            out.foundCameraTab = Boolean(tab);
            if (tab) tab.click();
            out.camera = await ODK.waitFor(() => {
              const text = ODK.panel("Mission Properties") || "";
              return /GSD @80 m/.test(text) ? text : null;
            }, 15000);
            return out;
        """)
    plans = _plans(cockpit)
    assert len(plans) >= 3, cockpit.calls

    shown = seen["shown"]
    assert shown == {"type": "3D modelling", "alt": "60", "fwd": "80", "side": "70",
                     "speed": "8", "camera": "mavic2pro"}, shown
    first = plans[0]["args"][0]
    # Every value the panel displayed reached the planner, untouched.
    assert first["template"] == "double_grid", first
    assert first["altitude_m"] == 60, first
    assert first["front_overlap_pct"] == 80 and first["side_overlap_pct"] == 70, first
    assert first["speed_m_s"] == 8 and first["camera"] == "mavic2pro", first
    # The inert fields are labelled as such and are not sent under invented names.
    assert not {"gsd", "angle", "air", "batt"} & set(first), first
    assert plans[0]["ok"], plans[0]
    assert "Planned" in seen["planned"], seen

    assert seen["altAfterRedraw"] == "80", seen
    assert plans[1]["args"][0]["altitude_m"] == 80, plans[1]

    assert seen["foundTypeRow"], seen
    assert seen["typeAfterPick"] == "2D mapping", seen
    assert plans[2]["args"][0]["template"] == "grid", plans[2]

    assert seen["foundCameraTab"], seen
    camera = seen["camera"] or ""
    assert "dji mavic 2 pro" in camera.lower() and "p1 / 35 mm" not in camera.lower(), camera
    from mission.cameras import describe
    expected = describe("mavic2pro", altitude_m=80)["gsd_cm"]
    assert f"{expected:.2f} cm" in camera, (expected, camera)


def test_the_planned_mission_is_the_displayed_one(tmp_path):
    """The session's plan, not only the call: 60 m, double grid, mavic2pro.

    A double grid raises the overlaps to the 85/80 % reconstruction needs. That used to
    happen silently under a form reading 80/70; the Plan result now says so."""
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            await ODK.nav("Mission Planning");
            await ODK.waitFor(() => ODK.$('[data-key="alt"]'), 10000);
            await ODK.press("Plan");
            return { done: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Planned")), 30000) };
        """)
    assert "Overlap raised to 85/80 % (front/side) from the 80/70 % asked for" in seen["done"], seen
    plan = api._session.mission_plan_dict  # noqa: SLF001 - what was actually planned
    assert plan, cockpit.calls
    assert plan["altitude_m"] == pytest.approx(60.0)
    assert plan["template"] == "double_grid"
    meta = plan["flight_recipe"]["metadata"]
    assert meta["requested_front_overlap_pct"] == pytest.approx(80.0)
    assert meta["requested_side_overlap_pct"] == pytest.approx(70.0)
    assert plan["front_overlap_pct"] == pytest.approx(85.0)
    assert plan["side_overlap_pct"] == pytest.approx(80.0)
    assert plan["camera"] == "mavic2pro"
