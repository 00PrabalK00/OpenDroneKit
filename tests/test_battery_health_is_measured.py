"""A battery's health is a measurement, so a pack nobody measured shows none.

The Battery model defaulted health_pct to 100. Every pack registered from the desktop
then read "Health 100%" on the Fleet screen and in the exported register, and the
desktop had no way to enter a real reading, so it stayed that way. Add Battery also
accepted a negative cycle limit, which labelled a new 0-cycle pack "past cycle limit".

These tests press the real cockpit and classic window against the real Api on a fresh
database, and check the upgrade path for a database made while the column was NOT NULL.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from cockpit_harness import Cockpit, make_api

pytest.importorskip("sqlalchemy")


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    monkeypatch.delenv("ODK_DATABASE_URL", raising=False)
    monkeypatch.setenv("ODK_HUB_HOME", str(tmp_path / "hub_home"))
    monkeypatch.chdir(tmp_path)
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - the engine is cached per URL
    db_module._SessionLocal = None    # noqa: SLF001
    yield tmp_path / "hub_home" / "opendronekit.db"
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


def test_a_new_pack_has_no_health_until_one_is_measured(fresh_install, tmp_path):
    api = make_api(tmp_path)
    added = api.add_battery(1, "BAT-NEW", "5000", "300")
    assert added["ok"] and added["health_pct"] is None, added
    [listed] = api.list_fleet(1)["batteries"]
    assert listed["health_pct"] is None and listed["over_cycle_limit"] is False, listed

    exported = api.export_fleet(1, str(tmp_path / "export"))
    assert exported["ok"], exported
    with open(Path(exported["directory"]) / "batteries.csv", newline="", encoding="utf-8") as fh:
        [row] = list(csv.DictReader(fh))
    assert row["health_pct"] == "", row
    register = json.loads(Path(exported["files"]["json"]).read_text("utf-8"))
    assert register["batteries"][0]["health_pct"] is None, register

    logged = api.log_battery_cycle(1, added["id"], "2", "")
    assert logged["ok"] and logged["cycle_count"] == 2 and logged["health_pct"] is None, logged
    measured = api.log_battery_cycle(1, added["id"], "0", "87.5%")
    assert measured["ok"] and measured["cycle_count"] == 2 and measured["health_pct"] == 87.5
    assert api.list_fleet(1)["batteries"][0]["health_pct"] == 87.5


@pytest.mark.parametrize("capacity,limit,words", [
    (-5, 300, "Capacity (mAh) cannot be negative"),
    (5000, -1, "Cycle limit cannot be negative"),
    ("abc", 300, "must be a whole number"),
    (5000, "2.5", "must be a whole number"),
])
def test_add_battery_refuses_values_that_are_not_counts(fresh_install, tmp_path,
                                                        capacity, limit, words):
    api = make_api(tmp_path)
    refused = api.add_battery(1, "BAT-X", capacity, limit)
    assert refused["ok"] is False and words in refused["error"], refused
    assert api.list_fleet(1)["batteries"] == []


def test_a_health_reading_is_refused_not_clamped(fresh_install, tmp_path):
    api = make_api(tmp_path)
    battery = api.add_battery(1, "BAT-1", 0, 0)
    for health, words in (("150", "between 0 and 100"), ("-3", "between 0 and 100"),
                          ("good", "must be a percentage")):
        refused = api.log_battery_cycle(1, battery["id"], 1, health)
        assert refused["ok"] is False and words in refused["error"], (health, refused)
    assert api.log_battery_cycle(1, battery["id"], 0, "")["ok"] is False
    assert "No battery with id 99" in api.log_battery_cycle(1, 99, 1, "")["error"]
    assert "not a battery id" in api.log_battery_cycle(1, "b1", 1, "")["error"]
    assert api.list_fleet(1)["batteries"][0]["cycle_count"] == 0


def test_a_database_made_while_health_was_not_null_still_takes_new_packs(fresh_install,
                                                                        tmp_path):
    """The upgrade path: the old column was NOT NULL, and a new pack now stores NULL."""
    import services.api.db as db_module
    from services.api.models import Battery

    # Build the old schema exactly as the old model did, with one pack in it.
    db_module.init_db()
    with sqlite3.connect(fresh_install) as con:
        con.execute("INSERT INTO organizations (id, name, slug, created_at) "
                    "VALUES (1, 'Local organisation', 'local-desktop', '2026-01-01')")
        con.execute("DROP TABLE batteries")
        con.execute(
            "CREATE TABLE batteries (id INTEGER NOT NULL PRIMARY KEY, organization_id "
            "INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE, "
            "serial_number VARCHAR(120) NOT NULL, capacity_mah INTEGER NOT NULL, "
            "cycle_count INTEGER NOT NULL, cycle_limit INTEGER NOT NULL, "
            "health_pct FLOAT NOT NULL, retired BOOLEAN NOT NULL, last_used_at DATETIME, "
            "created_at DATETIME NOT NULL)")
        con.execute("CREATE INDEX ix_batteries_organization_id ON batteries (organization_id)")
        con.execute("INSERT INTO batteries VALUES (1, 1, 'OLD-1', 4000, 12, 300, 91.0, 0, "
                    "NULL, '2026-01-01')")
        # Registered and never used: its 100 is the old default, nobody measured it.
        con.execute("INSERT INTO batteries VALUES (2, 1, 'OLD-2', 4000, 0, 300, 100.0, 0, "
                    "NULL, '2026-01-01')")
        # Used, and 100 when last logged: that may be a reading, so it is kept.
        con.execute("INSERT INTO batteries VALUES (3, 1, 'OLD-3', 4000, 4, 300, 100.0, 0, "
                    "NULL, '2026-01-01')")
    db_module._engine = None          # noqa: SLF001 - a new launch
    db_module._SessionLocal = None    # noqa: SLF001

    api = make_api(tmp_path)
    added = api.add_battery(1, "NEW-1", 5000, 300)
    assert added["ok"] and added["health_pct"] is None, added
    by_serial = {b["serial_number"]: b for b in api.list_fleet(1)["batteries"]}
    assert by_serial["OLD-1"]["health_pct"] == 91.0 and by_serial["OLD-1"]["cycle_count"] == 12
    assert by_serial["NEW-1"]["health_pct"] is None
    assert by_serial["OLD-2"]["health_pct"] is None, by_serial["OLD-2"]
    assert by_serial["OLD-3"]["health_pct"] == 100.0
    with sqlite3.connect(fresh_install) as con:
        notnull = {row[1]: row[3] for row in con.execute("PRAGMA table_info(batteries)")}
        indexes = {row[1] for row in con.execute("PRAGMA index_list(batteries)")}
        leftovers = con.execute("SELECT name FROM sqlite_master "
                                "WHERE name = '_batteries_before_nullable'").fetchall()
    assert notnull["health_pct"] == 0 and "ix_batteries_organization_id" in indexes
    assert leftovers == []
    assert Battery.__table__.c.health_pct.nullable


def test_add_battery_and_log_a_cycle_in_the_cockpit(fresh_install, tmp_path):
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const found = (text) => ODK.waitFor(() => ODK.toasts().find((t) => t.includes(text)), 10000);
            const out = {};
            await ODK.nav("Fleet");
            await ODK.press("Add Battery");
            await ODK.answer("BAT-NEW"); await ODK.answer("5000"); await ODK.answer("-1");
            out.refused = await found("cannot be negative");
            await ODK.press("Add Battery");
            await ODK.answer("BAT-NEW"); await ODK.answer("5000"); await ODK.answer("300");
            out.added = await found("Battery BAT-NEW registered");
            await ODK.sleep(1000);
            out.before = ODK.panel("Batteries");
            const pack = await ODK.waitFor(() => ODK.$$("tr").find((r) => r.textContent.includes("BAT-NEW")), 10000);
            pack.click(); await ODK.sleep(300);
            await ODK.press("Log Battery Cycle");
            await ODK.answer("3"); await ODK.answer("");
            out.cycles = await found("BAT-NEW: 3 cycles");
            await ODK.press("Log Battery Cycle");
            await ODK.answer("1"); await ODK.answer("92");
            out.health = await found("BAT-NEW: 4 cycles, health 92%");
            await ODK.sleep(1200);
            out.after = ODK.panel("Batteries");
            out.errors = ODK.toasts().filter((t) => /undefined|NaN|invalid literal/i.test(t));
            return out;
        """)
    failed = [c for c in cockpit.calls if not c["ok"]
              and c["method"] in ("list_fleet", "fleet_status", "log_battery_cycle")]
    assert failed == [], failed
    assert seen["errors"] == [], seen
    assert "Cycle limit cannot be negative" in seen["refused"], seen
    assert "Health is blank until you measure it" in seen["added"], seen
    before = seen["before"]
    assert "BAT-NEW" in before and "100%" not in before and "—" in before, before
    assert "past cycle limit" not in before, before
    assert seen["cycles"] == "BAT-NEW: 3 cycles, health not measured.", seen
    assert "92%" in seen["after"] and "4" in seen["after"], seen
    [pack] = api.list_fleet(1)["batteries"]
    assert pack["cycle_count"] == 4 and pack["health_pct"] == 92.0 and pack["cycle_limit"] == 300


def test_battery_health_and_cycles_in_the_classic_window(fresh_install, tmp_path):
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile", page="index.html") as window:
        seen = window.run("""
            const form = async (button, values) => {
              ODK.$("#status-message").textContent = "";
              ODK.$(button).click();
              await ODK.waitFor(() => !ODK.$("#modal-backdrop").classList.contains("hidden"), 5000);
              for (const [id, value] of Object.entries(values)) ODK.$("#" + id).value = value;
              ODK.$("#modal-actions button.primary").click();
              await ODK.sleep(800);
              return ODK.$("#status-message").textContent;
            };
            await ODK.sleep(1000);
            ODK.$(".tab[data-tab=fleet]").click();
            await ODK.sleep(500);
            const out = {};
            out.negative = await form("#btn-fleet-battery", { "f-serial": "PK-1", "f-cap": "5000", "f-limit": "-1" });
            ODK.$("#modal-backdrop").classList.add("hidden");
            out.battery = await form("#btn-fleet-battery", { "f-serial": "PK-1", "f-cap": "5000", "f-limit": "200" });
            await ODK.sleep(500);
            out.before = ODK.$("#fleet-list").innerText.replace(/\\s+/g, " ");
            out.cycle = await form("#btn-fleet-cycle", { "f-id": "1", "f-cycles": "5", "f-health": "88" });
            await ODK.sleep(800);
            out.after = ODK.$("#fleet-list").innerText.replace(/\\s+/g, " ");
            return out;
        """)
    assert "Cycle limit cannot be negative" in seen["negative"], seen
    assert seen["battery"] == "Add battery: saved.", seen
    assert "PK-1" in seen["before"] and "health not measured" in seen["before"], seen
    assert "100%" not in seen["before"], seen
    assert seen["cycle"] == "Log battery cycle: saved.", seen
    assert "5 / 200 cycles · health 88%" in seen["after"], seen


def test_the_service_api_reports_an_unmeasured_pack_as_unmeasured(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("email_validator")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001
    from services.api.main import app

    try:
        with TestClient(app) as client:
            token = client.post("/auth/register", json={
                "email": "ops@example.com", "password": "longenough1",
                "organization_name": "Fleet Ops"}).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            org = client.get("/organizations", headers=headers).json()[0]["id"]
            pack = client.post(f"/organizations/{org}/batteries", headers=headers,
                               json={"serial_number": "S-1"}).json()
            assert pack["health_pct"] is None and pack["warnings"] == [], pack
            bad = client.post(f"/organizations/{org}/batteries", headers=headers,
                              json={"serial_number": "S-2", "cycle_limit": -1})
            assert bad.status_code == 422, bad.text
            logged = client.post(f"/batteries/{pack['id']}/cycles", headers=headers).json()
            assert logged["cycle_count"] == 1 and logged["health_pct"] is None, logged
    finally:
        db_module._engine = None      # noqa: SLF001
        db_module._SessionLocal = None  # noqa: SLF001


def test_a_fleet_that_could_not_be_read_is_not_shown_as_empty(fresh_install, tmp_path):
    """A failed list_fleet rendered "No batteries registered" -- an empty register
    stated as fact. Seen once while driving Log Battery Cycle under memory pressure."""
    api = make_api(tmp_path)
    api.add_battery(1, "BAT-1", 5000, 300)
    api.list_fleet = lambda *_: {"ok": False, "error": "database is locked"}
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            await ODK.nav("Fleet");
            await ODK.sleep(1500);
            return { batteries: ODK.panel("Batteries"), aircraft: ODK.panel("Aircraft"),
                     maintenance: ODK.panel("Maintenance") };
        """)
    for panel in ("batteries", "aircraft", "maintenance"):
        text = seen[panel]
        assert "Could not read this: the fleet register (database is locked)" in text, seen
        assert "No " not in text, seen
