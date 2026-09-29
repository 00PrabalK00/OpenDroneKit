"""Add Aircraft, Add Battery and Add Pilot work on a fresh install.

Verifier, before this change: Fleet > Add Aircraft on an empty service database failed
with "IntegrityError: FOREIGN KEY constraint failed [INSERT INTO aircraft
(organization_id ...) parameters (1, ...)]". The cockpit sends organisation 1, and a
fresh service database has no organisation until something creates one. The fleet calls
now resolve the organisation the way sharing does (app/local_tenant.py).

The page is the real cockpit against the real Api and an empty service database; the
rows are read back through the service's own ORM.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api

pytest.importorskip("sqlalchemy")


@pytest.fixture
def service_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(tmp_path / 'service.db').as_posix()}")
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - the engine is cached per URL
    db_module._SessionLocal = None    # noqa: SLF001
    db_module.init_db()
    yield db_module
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


def rows(db_module, model):
    from sqlalchemy import select

    with db_module.get_session_factory()() as db:
        return list(db.scalars(select(model)))


def test_fleet_buttons_register_rows_on_a_fresh_install(service_db, tmp_path) -> None:
    from services.api.models import Aircraft, Battery, Organization, PilotProfile

    assert rows(service_db, Organization) == []
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Fleet");
            await ODK.press("Add Aircraft");
            await ODK.answer("Survey One");
            await ODK.answer("M300");
            await ODK.answer("SN-1");
            const aircraft = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("Survey One")), 10000);
            await ODK.press("Add Battery");
            await ODK.answer("BAT-7");
            await ODK.answer("5000");
            await ODK.answer("300");
            const battery = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("BAT-7")), 10000);
            await ODK.press("Add Pilot");
            await ODK.answer("A. Pilot");
            await ODK.answer("");
            await ODK.answer("");
            const pilot = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("A. Pilot")), 10000);
            return { aircraft, battery, pilot };
        """)

    assert result["aircraft"] == 'Aircraft "Survey One" registered.', result
    assert "Battery BAT-7 registered" in result["battery"], result
    assert "Pilot A. Pilot added" in result["pilot"], result
    for method in ("add_aircraft", "add_battery", "add_pilot"):
        calls = cockpit.called(method)
        assert calls and all(c["ok"] for c in calls), calls
    [org] = rows(service_db, Organization)
    [aircraft] = rows(service_db, Aircraft)
    [battery] = rows(service_db, Battery)
    [pilot] = rows(service_db, PilotProfile)
    assert aircraft.name == "Survey One" and aircraft.organization_id == org.id
    assert battery.organization_id == org.id and pilot.organization_id == org.id
    # The fleet screen reads back what was registered.
    listed = api.list_fleet(1)
    assert listed["ok"] and [a["name"] for a in listed["aircraft"]] == ["Survey One"], listed


def test_an_unknown_organisation_is_refused_not_guessed(service_db, tmp_path) -> None:
    api = make_api(tmp_path)
    assert api.add_aircraft(1, "First")["ok"]
    refused = api.add_aircraft(99, "Elsewhere")
    assert refused["ok"] is False and "No organisation with id 99" in refused["error"]
