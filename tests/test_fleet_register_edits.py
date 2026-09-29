"""The fleet register can be corrected and handed over, on a fresh install.

Driving the Fleet screen on a first launch found three things a user could not do:

  * the Home tile and the Fleet lists ask for the fleet at the same moment, and both
    created the local organisation -- the loser failed with "UNIQUE constraint failed:
    organizations.slug";
  * nothing could be removed, so a mistyped aircraft (or the "E2E aircraft" rows an old
    test tool wrote into users' databases) stayed on the screen for good;
  * Fleet's Export button exported the mission and answered "No mission has been
    planned yet".

Each is exercised through the real Api on a database the test creates, and the screens
are pressed in a real browser.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import threading

import pytest

from cockpit_harness import Cockpit

pytest.importorskip("sqlalchemy")


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    """No database URL, no hub home: the service database lands under the data root."""
    monkeypatch.delenv("ODK_DATABASE_URL", raising=False)
    monkeypatch.delenv("ODK_HUB_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - the engine is cached per URL
    db_module._SessionLocal = None    # noqa: SLF001
    yield tmp_path
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


def _api(tmp_path: Path):
    """The Api as the desktop Shell builds it: startup selects the default project."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(tmp_path / "state" / "projects.db")))
    api._session.restore_active_dataset()  # noqa: SLF001 - what Shell.__init__ does
    return api


def test_the_first_fleet_reads_at_once_do_not_race_to_create_the_organisation(
        fresh_install, monkeypatch):
    from sqlalchemy.orm import Session

    api = _api(fresh_install)
    # Widen the window between "is there an organisation?" and creating one, which is
    # where the screen's simultaneous first reads collided. Without it the collision
    # depends on thread timing and a passing run proves nothing.
    original = Session.scalar

    def slow_count(self, statement, *args, **kwargs):
        if "count(" in str(statement).lower() and "organizations" in str(statement).lower():
            import time
            time.sleep(0.2)
        return original(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "scalar", slow_count)
    results: list[dict] = []
    start = threading.Barrier(8)

    def read(method: str) -> None:
        start.wait()
        results.append(getattr(api, method)(1))

    threads = [threading.Thread(target=read, args=(m,))
               for m in ["fleet_status", "list_fleet"] * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(results) == 8
    assert all(r["ok"] for r in results), [r.get("error") for r in results if not r["ok"]]

    import services.api.db as db_module
    from services.api.models import Organization
    from sqlalchemy import select

    with db_module.get_session_factory()() as db:
        assert len(list(db.scalars(select(Organization)))) == 1


def test_remove_and_export_through_the_api(fresh_install):
    api = _api(fresh_install)
    kept = api.add_aircraft(1, "Keep", "M30T", "S-1")
    gone = api.add_aircraft(1, "E2E aircraft", "M3E", "SN-E2E")
    battery = api.add_battery(1, "BAT-9", 5000, 300)
    pilot = api.add_pilot(1, "C. Pilot", "RP-1", "2031-01-01")
    assert api.log_maintenance(gone["id"], "inspection", "e2e", "")["ok"]
    assert api.log_maintenance(kept["id"], "motor", "bearing", "operator")["ok"]

    removed = api.remove_fleet_record(1, "aircraft", gone["id"])
    assert removed["ok"] and removed["label"] == "E2E aircraft", removed
    assert removed["removed_maintenance"] == 1
    fleet = api.list_fleet(1)
    assert [a["name"] for a in fleet["aircraft"]] == ["Keep"]
    assert all(m["aircraft"] == "Keep" for m in fleet["maintenance"])

    assert api.remove_fleet_record(1, "battery", battery["id"])["ok"]
    assert api.list_fleet(1)["batteries"] == []
    # Refused by name: an id that is not there, a kind that is not a fleet record,
    # and an organisation that does not exist.
    assert "No pilot with id 999" in api.remove_fleet_record(1, "pilot", 999)["error"]
    assert "Choose what to remove" in api.remove_fleet_record(1, "hangar", 1)["error"]
    assert api.remove_fleet_record(99, "pilot", pilot["id"])["ok"] is False
    assert api.list_fleet(1)["pilots"][0]["display_name"] == "C. Pilot"

    exported = api.export_fleet(1, "")
    assert exported["ok"], exported
    out = Path(exported["directory"])
    assert out.is_dir() and out.parent.name == "exports"
    assert exported["counts"] == {"aircraft": 1, "batteries": 0, "pilots": 1, "maintenance": 1}
    with open(exported["files"]["aircraft"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["name"], r["serial_number"]) for r in rows] == [("Keep", "S-1")]
    with open(exported["files"]["maintenance"], newline="", encoding="utf-8") as handle:
        [log] = list(csv.DictReader(handle))
    assert log["aircraft"] == "Keep" and log["kind"] == "motor"
    body = json.loads(Path(exported["files"]["json"]).read_text(encoding="utf-8"))
    assert body["pilots"][0]["licence_expires_on"] == "2031-01-01"
    # An empty section is still a readable CSV: a header row, no invented records.
    assert Path(exported["files"]["batteries"]).read_text(encoding="utf-8").strip() == ""


def test_remove_record_and_export_fleet_in_the_cockpit(fresh_install):
    api = _api(fresh_install)
    with Cockpit(api, fresh_install / "profile") as cockpit:
        seen = cockpit.run("""
            const out = {};
            await ODK.nav("Fleet");
            await ODK.press("Add Aircraft"); await ODK.answer("Alpha"); await ODK.answer("M30T"); await ODK.answer("S1");
            await ODK.sleep(600);
            await ODK.press("Add Aircraft"); await ODK.answer("Bravo"); await ODK.answer(""); await ODK.answer("");
            await ODK.sleep(600);
            await ODK.press("Add Battery"); await ODK.answer("B-1"); await ODK.answer("5000"); await ODK.answer("200");
            await ODK.sleep(1000);
            const bravo = await ODK.waitFor(() => ODK.$$(".tree-node").find((n) => n.textContent.includes("Bravo")), 10000);
            bravo.click(); await ODK.sleep(300);
            await ODK.press("Log Maintenance"); await ODK.answer("motor"); await ODK.answer("swap");
            await ODK.sleep(800);
            await ODK.press("Export Fleet");
            out.exported = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Exported")), 10000);
            const pack = await ODK.waitFor(() => ODK.$$("tr").find((r) => r.textContent.includes("B-1")), 10000);
            pack.click(); await ODK.sleep(300);
            await ODK.press("Remove Record");
            out.question = (ODK.$(".modal-backdrop .modal") || {}).innerText;
            await ODK.answer();
            out.battery = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Removed battery")), 10000);
            const again = await ODK.waitFor(() => ODK.$$(".tree-node").find((n) => n.textContent.includes("Bravo")), 10000);
            again.click(); await ODK.sleep(300);
            await ODK.press("Remove Record"); await ODK.answer();
            out.aircraft = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Removed aircraft")), 10000);
            await ODK.sleep(1200);
            out.list = ODK.panel("Aircraft"); out.batteries = ODK.panel("Batteries");
            out.errors = ODK.toasts().filter((t) => /error|undefined|No mission/i.test(t));
            return out;
        """)
    assert seen["errors"] == [], seen
    assert "Exported 2 aircraft, 1 batteries, 0 pilots and 1 maintenance records to" in seen["exported"], seen
    assert 'Remove battery "B-1"?' in seen["question"], seen
    assert seen["battery"] == "Removed battery B-1.", seen
    assert seen["aircraft"] == "Removed aircraft Bravo and 1 maintenance record(s).", seen
    assert "Alpha" in seen["list"] and "Bravo" not in seen["list"], seen
    assert "No batteries registered" in seen["batteries"], seen
    assert [a["name"] for a in api.list_fleet(1)["aircraft"]] == ["Alpha"]
    assert cockpit.opened, "the export folder was not revealed"


def test_remove_and_export_in_the_classic_window(fresh_install):
    api = _api(fresh_install)
    api.add_aircraft(1, "Classic One")
    api.add_battery(1, "PK-1", 5000, 200)
    with Cockpit(api, fresh_install / "profile", page="index.html") as window:
        seen = window.run("""
            await ODK.sleep(1000);
            ODK.$(".tab[data-tab=fleet]").click();
            await ODK.sleep(800);
            const out = { before: ODK.$("#fleet-list").innerText.replace(/\\s+/g, " ") };
            ODK.$("#status-message").textContent = "";
            ODK.$("#btn-fleet-export").click();
            out.export = await ODK.waitFor(() => /Export fleet:/.test(ODK.$("#status-message").textContent)
              && ODK.$("#status-message").textContent, 10000);
            ODK.$("#status-message").textContent = "";
            ODK.$("#btn-fleet-remove").click();
            await ODK.waitFor(() => !ODK.$("#modal-backdrop").classList.contains("hidden"), 5000);
            ODK.$("#f-kind").value = "battery"; ODK.$("#f-id").value = "1";
            ODK.$("#modal-actions button.primary").click();
            out.remove = await ODK.waitFor(() => /saved|: /.test(ODK.$("#status-message").textContent)
              && ODK.$("#status-message").textContent, 10000);
            await ODK.sleep(800);
            out.after = ODK.$("#fleet-list").innerText.replace(/\\s+/g, " ");
            return out;
        """)
    assert "#1 PK-1" in seen["before"], seen
    assert "Export fleet: 1 aircraft, 1 batteries, 0 pilots, 0 maintenance records written to" in seen["export"], seen
    assert seen["remove"] == "Remove record: saved.", seen
    assert "PK-1" not in seen["after"] and "Classic One" in seen["after"], seen
