"""Three Fleet defects found driving a fresh install.

1. The classic window built its fleet list with innerHTML and did not escape names, so
   an aircraft called '<i id=xss>Kite</i>' became a live element -- and names also come
   from the shared service database.
2. With a battery or pilot picked last, the cockpit's Log Maintenance and Assign Mission
   fell back to an "Aircraft id" prompt pre-filled with 1: one Enter logged maintenance
   against whichever aircraft happened to be number 1.
3. The service's /batteries/{id}/cycles clamped a health reading into 0-100, so 150 was
   stored as a measured 100%. The desktop refuses it; the service now does too.
"""

from __future__ import annotations

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


def test_the_classic_fleet_list_shows_names_as_text(fresh_install, tmp_path):
    api = make_api(tmp_path)
    api.add_aircraft(1, "<i id=xss>Kite</i>", "<b id=xssm>M</b>", "")
    api.add_battery(1, "<img id=xssb src=x>", 0, 0)
    api.add_pilot(1, "<u id=xssp>Jo</u>", "", "")
    with Cockpit(api, tmp_path / "profile", page="index.html") as window:
        seen = window.run("""
            await ODK.sleep(1000);
            ODK.$(".tab[data-tab=fleet]").click();
            await ODK.waitFor(() => ODK.$("#fleet-list").innerText.includes("Kite"), 10000);
            return {
              text: ODK.$("#fleet-list").innerText,
              live: ["xss", "xssm", "xssb", "xssp"].filter((id) => document.getElementById(id)),
            };
        """)
    assert seen["live"] == [], seen
    assert "<i id=xss>Kite</i>" in seen["text"], seen
    assert "<img id=xssb src=x>" in seen["text"], seen
    assert "<u id=xssp>Jo</u>" in seen["text"], seen


def test_log_maintenance_with_a_battery_selected_is_refused_not_sent_to_aircraft_1(
        fresh_install, tmp_path):
    api = make_api(tmp_path)
    api.add_aircraft(1, "Hawk")
    api.add_battery(1, "P1", 0, 0)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const found = (text) => ODK.waitFor(() => ODK.toasts().find((t) => t.includes(text)), 10000);
            await ODK.nav("Fleet");
            await ODK.sleep(1200);
            const pack = await ODK.waitFor(() => ODK.$$("tr").find((r) => r.textContent.includes("P1")), 10000);
            pack.click(); await ODK.sleep(300);
            const out = {};
            await ODK.press("Log Maintenance");
            out.maintenance = await found("Log Maintenance:");
            await ODK.press("Assign Mission");
            out.assign = await found("Assign Mission:");
            // Now the aircraft: both go through.
            const hawk = await ODK.waitFor(() => ODK.$$(".tree-node").find((n) => n.textContent.includes("Hawk")), 10000);
            hawk.click(); await ODK.sleep(300);
            await ODK.press("Log Maintenance");
            await ODK.answer("motor"); await ODK.answer("");
            out.logged = await found("Logged motor");
            return out;
        """)
    assert 'battery "P1" is selected' in seen["maintenance"], seen
    assert "Select an aircraft" in seen["assign"], seen
    assert seen["logged"].startswith("Logged motor"), seen
    fleet = api.list_fleet(1)
    assert [m["kind"] for m in fleet["maintenance"]] == ["motor"], fleet["maintenance"]
    assert fleet["aircraft"][0]["assigned_mission"] in (None, ""), fleet["aircraft"]


def test_the_service_refuses_a_health_reading_outside_0_to_100(tmp_path, monkeypatch):
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
            url = f"/batteries/{pack['id']}/cycles"
            for bad in ("150", "-3", "nan"):
                refused = client.post(url, headers=headers, params={"health_pct": bad})
                assert refused.status_code == 422, (bad, refused.text)
            listed = client.get(f"/organizations/{org}/batteries", headers=headers).json()
            assert listed[0]["cycle_count"] == 0 and listed[0]["health_pct"] is None, listed
            measured = client.post(url, headers=headers, params={"health_pct": 88}).json()
            assert measured["cycle_count"] == 1 and measured["health_pct"] == 88.0, measured
    finally:
        db_module._engine = None      # noqa: SLF001
        db_module._SessionLocal = None  # noqa: SLF001
