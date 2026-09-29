"""The whole Fleet screen on a fresh install, driven like an operator.

A fresh install has no service database: no ODK_DATABASE_URL and an empty hub home.
Every fleet action -- Add Aircraft, Add Battery, Add Pilot, then selecting the aircraft
and pressing Log Maintenance and Assign Mission -- is pressed in the real cockpit against
the real Api, and what the screen shows afterwards is read back.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api

pytest.importorskip("sqlalchemy")


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    """No database URL and an empty hub home: what a first launch sees."""
    monkeypatch.delenv("ODK_DATABASE_URL", raising=False)
    monkeypatch.setenv("ODK_HUB_HOME", str(tmp_path / "hub_home"))
    monkeypatch.chdir(tmp_path)          # no legacy ./opendronekit.db either
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - the engine is cached per URL
    db_module._SessionLocal = None    # noqa: SLF001
    yield tmp_path / "hub_home" / "opendronekit.db"
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


def test_every_fleet_action_works_in_the_cockpit_on_a_fresh_install(fresh_install, tmp_path):
    assert not fresh_install.exists()
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const found = (text) => ODK.waitFor(() => ODK.toasts().find((t) => t.includes(text)), 10000);
            await ODK.nav("Fleet");
            await ODK.press("Add Aircraft");
            await ODK.answer("Survey One"); await ODK.answer("M300"); await ODK.answer("SN-1");
            const aircraft = await found("Survey One");
            await ODK.press("Add Battery");
            await ODK.answer("BAT-7"); await ODK.answer("5000"); await ODK.answer("300");
            const battery = await found("BAT-7");
            await ODK.press("Add Pilot");
            await ODK.answer("A. Pilot"); await ODK.answer("RP-123"); await ODK.answer("2030-01-31");
            const pilot = await found("A. Pilot");
            await ODK.sleep(800);
            // Pick the aircraft in the list, the way the buttons ask to be used.
            const row = await ODK.waitFor(() => ODK.$$(".tree-node").find((n) => n.textContent.includes("Survey One")), 10000);
            row.click();
            await ODK.sleep(300);
            await ODK.press("Log Maintenance");
            await ODK.answer("propeller"); await ODK.answer("replaced all four");
            const maintenance = await found("Logged");
            await ODK.press("Assign Mission");
            await ODK.answer("North Facade");
            const assigned = await found("Assigned");
            const all = ODK.toasts();
            await ODK.sleep(1500);
            return { aircraft, battery, pilot, maintenance, assigned,
                     maintPanel: ODK.panel("Maintenance"), details: ODK.panel("Aircraft Details"),
                     batteries: ODK.panel("Batteries"), pilots: ODK.panel("Pilots"),
                     all, errors: ODK.toasts().filter((t) => /error|invalid|undefined|No aircraft/i.test(t)) };
        """)
    assert seen["errors"] == [], seen
    assert seen["aircraft"] == 'Aircraft "Survey One" registered.', seen
    assert "Battery BAT-7 registered" in seen["battery"], seen
    assert "Pilot A. Pilot added" in seen["pilot"], seen
    assert seen["maintenance"].startswith("Logged propeller"), seen
    assert "North Facade" in seen["assigned"], seen
    assert "propeller: replaced all four" in seen["maintPanel"], seen
    assert "North Facade" in seen["details"], seen
    assert "BAT-7" in seen["batteries"] and "A. Pilot" in seen["pilots"], seen
    assert fresh_install.exists(), "the fleet was not written to the hub home's database"

    [aircraft] = api.list_fleet(1)["aircraft"]
    assert aircraft["name"] == "Survey One" and aircraft["assigned_mission"] == "North Facade"


def test_assigning_an_overdue_aircraft_does_not_mark_it_serviced(fresh_install, tmp_path):
    """Assign Mission was written as a maintenance record, and every maintenance record
    reset the service clock: assigning an overdue airframe made it read as serviced."""
    import services.api.db as db_module
    from services.api.models import Aircraft, Maintenance
    from sqlalchemy import select

    api = make_api(tmp_path)
    aircraft = api.add_aircraft(1, "Old Frame")
    with db_module.get_session_factory()() as db:
        row = db.get(Aircraft, aircraft["id"])
        row.flight_hours = 130.0            # 30 h past a 100 h interval
        db.commit()
    assert api.list_fleet(1)["aircraft"][0]["hours_to_service"] == -30.0

    assigned = api.assign_mission_to_aircraft(aircraft["id"], "Bridge 4")
    assert assigned["ok"] and assigned["mission"] == "Bridge 4", assigned
    [listed] = api.list_fleet(1)["aircraft"]
    assert listed["hours_to_service"] == -30.0, "an assignment reset the service clock"
    assert listed["assigned_mission"] == "Bridge 4"
    assert api.fleet_status(1)["service_due"] == ["Old Frame"]

    serviced = api.log_maintenance(aircraft["id"], "100 h inspection", "", "operator")
    assert serviced["ok"], serviced
    assert api.list_fleet(1)["aircraft"][0]["hours_to_service"] == 100.0
    with db_module.get_session_factory()() as db:
        records = list(db.scalars(select(Maintenance).order_by(Maintenance.id)))
    # A name is not a user id: it goes into the record's text, the id column stays empty.
    assert [r.performed_by for r in records] == [None, None]
    assert records[-1].description == "by operator"


def test_an_id_that_is_not_a_number_is_refused_by_name(fresh_install, tmp_path):
    api = make_api(tmp_path)
    api.add_aircraft(1, "Frame")
    refused = api.log_maintenance("a1", "inspection")
    assert refused["ok"] is False and "select an aircraft" in refused["error"], refused
    assert api.assign_mission_to_aircraft(1, " ")["ok"] is False


def test_every_fleet_action_works_in_the_classic_window_on_a_fresh_install(fresh_install, tmp_path):
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile", page="index.html") as window:
        seen = window.run("""
            const form = async (button, values) => {
              ODK.$("#status-message").textContent = "";
              ODK.$(button).click();
              await ODK.waitFor(() => !ODK.$("#modal-backdrop").classList.contains("hidden"), 5000);
              for (const [id, value] of Object.entries(values)) ODK.$("#" + id).value = value;
              ODK.$("#modal-actions button.primary").click();
              return ODK.waitFor(() => /saved|: /.test(ODK.$("#status-message").textContent)
                && ODK.$("#status-message").textContent, 10000);
            };
            await ODK.sleep(1000);
            ODK.$(".tab[data-tab=fleet]").click();
            await ODK.sleep(500);
            const out = {};
            out.aircraft = await form("#btn-fleet-aircraft", { "f-name": "Classic One", "f-model": "Mavic 3E", "f-serial": "SN-9" });
            out.battery = await form("#btn-fleet-battery", { "f-serial": "PK-1", "f-cap": "5000", "f-limit": "200" });
            out.pilot = await form("#btn-fleet-pilot", { "f-name": "B. Pilot", "f-licence": "RP-9", "f-expires": "2031-05-01" });
            out.maintenance = await form("#btn-fleet-maintenance", { "f-id": "1", "f-kind": "motor", "f-detail": "bearing" });
            out.assign = await form("#btn-fleet-assign", { "f-id": "1", "f-mission": "Roof Survey" });
            await ODK.sleep(800);
            out.list = ODK.$("#fleet-list").innerText.replace(/\s+/g, " ");
            return out;
        """)
    assert seen["aircraft"] == "Add aircraft: saved.", seen
    assert seen["battery"] == "Add battery: saved.", seen
    assert seen["pilot"] == "Add pilot: saved.", seen
    assert seen["maintenance"] == "Log maintenance: saved.", seen
    assert seen["assign"] == "Assign mission: saved.", seen
    listed = seen["list"]
    for text in ("Classic One", "on Roof Survey", "PK-1", "B. Pilot", "motor: bearing",
                 "Assigned to mission: Roof Survey"):
        assert text in listed, (text, listed)
