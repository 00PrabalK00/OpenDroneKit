"""Fleet, batteries, pilots and maintenance.

These records exist to answer a question before a flight rather than after an
incident, so the tests check the arithmetic that produces the answer: is this
airframe due for service, is this pack past its cycle limit, is this pilot's
certification still current.

They also check the boundary the system must not cross. A maintenance flag is
information; nothing here refuses a flight on the operator's behalf.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def org(client):
    response = client.post("/auth/register", json={
        "email": "ops@example.com", "password": "longenough1",
        "organization_name": "Fleet Ops",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    return headers, organization_id


class TestAircraftService:
    def test_a_new_airframe_is_not_due(self, client, org):
        headers, organization_id = org
        aircraft = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                               json={"name": "M300-1", "service_interval_hours": 100.0}).json()
        assert aircraft["service_due"] is False
        assert aircraft["hours_until_service"] == pytest.approx(100.0)

    def test_flight_hours_accumulate(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1"}).json()["id"]

        client.post(f"/aircraft/{aircraft_id}/flights?hours=2.5", headers=headers)
        updated = client.post(f"/aircraft/{aircraft_id}/flights?hours=1.5", headers=headers).json()

        assert updated["flight_hours"] == pytest.approx(4.0)
        assert updated["flight_count"] == 2

    def test_service_becomes_due_at_the_interval(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1", "service_interval_hours": 10.0}).json()["id"]

        client.post(f"/aircraft/{aircraft_id}/flights?hours=9.0", headers=headers)
        assert client.get(f"/organizations/{organization_id}/aircraft",
                          headers=headers).json()[0]["service_due"] is False

        client.post(f"/aircraft/{aircraft_id}/flights?hours=2.0", headers=headers)
        assert client.get(f"/organizations/{organization_id}/aircraft",
                          headers=headers).json()[0]["service_due"] is True

    def test_recording_maintenance_resets_the_interval(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1", "service_interval_hours": 10.0}).json()["id"]
        client.post(f"/aircraft/{aircraft_id}/flights?hours=12.0", headers=headers)

        result = client.post("/maintenance", headers=headers, json={
            "aircraft_id": aircraft_id, "kind": "scheduled", "description": "100h check"})
        assert result.status_code == 201
        aircraft = result.json()["aircraft"]
        assert aircraft["service_due"] is False
        assert aircraft["hours_since_service"] == pytest.approx(0.0)
        # The hours themselves are not reset; only the interval is.
        assert aircraft["flight_hours"] == pytest.approx(12.0)

    def test_maintenance_history_is_kept(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1"}).json()["id"]
        client.post("/maintenance", headers=headers, json={
            "aircraft_id": aircraft_id, "description": "prop replacement"})

        history = client.get(f"/aircraft/{aircraft_id}/maintenance", headers=headers).json()
        assert len(history) == 1
        assert history[0]["description"] == "prop replacement"

    def test_a_negative_flight_time_is_refused(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1"}).json()["id"]
        assert client.post(f"/aircraft/{aircraft_id}/flights?hours=-3",
                           headers=headers).status_code == 422


class TestBatteries:
    def test_cycles_accumulate_and_the_limit_is_flagged(self, client, org):
        headers, organization_id = org
        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "BAT-001", "cycle_limit": 3}).json()["id"]

        for _ in range(2):
            client.post(f"/batteries/{battery_id}/cycles", headers=headers)
        assert client.get(f"/organizations/{organization_id}/batteries",
                          headers=headers).json()[0]["past_cycle_limit"] is False

        battery = client.post(f"/batteries/{battery_id}/cycles", headers=headers).json()
        assert battery["cycle_count"] == 3
        assert battery["past_cycle_limit"] is True
        assert any("cycles against a limit" in w for w in battery["warnings"])

    def test_low_health_is_flagged_even_within_the_cycle_limit(self, client, org):
        headers, organization_id = org
        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "BAT-002", "cycle_limit": 300}).json()["id"]

        battery = client.post(f"/batteries/{battery_id}/cycles?health_pct=62",
                              headers=headers).json()
        assert battery["past_cycle_limit"] is False
        assert any("Health is 62%" in w for w in battery["warnings"])

    def test_a_retired_battery_says_it_should_not_be_flown(self, client, org):
        headers, organization_id = org
        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "BAT-003"}).json()["id"]
        battery = client.post(f"/batteries/{battery_id}/retire", headers=headers).json()
        assert battery["retired"] is True
        assert any("should not be flown" in w for w in battery["warnings"])

    def test_health_is_clamped_to_a_percentage(self, client, org):
        headers, organization_id = org
        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "BAT-004"}).json()["id"]
        battery = client.post(f"/batteries/{battery_id}/cycles?health_pct=150",
                              headers=headers).json()
        assert battery["health_pct"] == pytest.approx(100.0)

    def test_only_an_admin_can_retire_a_battery(self, client, org):
        headers, organization_id = org
        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "BAT-005"}).json()["id"]
        client.post("/auth/register", json={
            "email": "pilot@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "pilot@example.com", "role": "pilot"})
        pilot = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "pilot@example.com", "password": "longenough1"}).json()["access_token"]}

        assert client.post(f"/batteries/{battery_id}/retire", headers=pilot).status_code == 403


class TestPilotCurrency:
    def test_a_current_pilot_has_no_warnings(self, client, org):
        headers, organization_id = org
        future = (date.today() + timedelta(days=365)).isoformat()
        pilot = client.post(f"/organizations/{organization_id}/pilots", headers=headers, json={
            "display_name": "A. Pilot", "licence_expires_on": future,
            "medical_expires_on": future}).json()
        assert pilot["current"] is True
        assert pilot["warnings"] == []

    def test_an_expired_licence_makes_the_pilot_not_current(self, client, org):
        headers, organization_id = org
        past = (date.today() - timedelta(days=1)).isoformat()
        pilot = client.post(f"/organizations/{organization_id}/pilots", headers=headers, json={
            "display_name": "B. Pilot", "licence_expires_on": past}).json()
        assert pilot["current"] is False
        assert any("expired" in w for w in pilot["warnings"])

    def test_an_imminent_expiry_warns_while_still_current(self, client, org):
        """A month's notice is the difference between renewing calmly and
        discovering it on the morning of a job."""
        headers, organization_id = org
        soon = (date.today() + timedelta(days=10)).isoformat()
        pilot = client.post(f"/organizations/{organization_id}/pilots", headers=headers, json={
            "display_name": "C. Pilot", "medical_expires_on": soon}).json()
        assert pilot["current"] is True
        assert any("expires on" in w for w in pilot["warnings"])

    def test_flight_hours_credit_the_named_pilot(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1"}).json()["id"]
        pilot_id = client.post(f"/organizations/{organization_id}/pilots", headers=headers,
                               json={"display_name": "D. Pilot"}).json()["id"]

        client.post(f"/aircraft/{aircraft_id}/flights?hours=3.25&pilot_id={pilot_id}",
                    headers=headers)
        pilots = client.get(f"/organizations/{organization_id}/pilots", headers=headers).json()
        assert pilots[0]["flight_hours"] == pytest.approx(3.25)

    def test_a_pilot_from_another_organization_is_refused(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "M300-1"}).json()["id"]
        assert client.post(f"/aircraft/{aircraft_id}/flights?hours=1&pilot_id=9999",
                           headers=headers).status_code == 404


class TestFleetStatus:
    def test_it_surfaces_what_needs_attention(self, client, org):
        headers, organization_id = org
        aircraft_id = client.post(f"/organizations/{organization_id}/aircraft", headers=headers,
                                  json={"name": "Due-1", "service_interval_hours": 5.0}).json()["id"]
        client.post(f"/aircraft/{aircraft_id}/flights?hours=6", headers=headers)

        battery_id = client.post(f"/organizations/{organization_id}/batteries", headers=headers,
                                 json={"serial_number": "OLD-1", "cycle_limit": 1}).json()["id"]
        client.post(f"/batteries/{battery_id}/cycles", headers=headers)

        past = (date.today() - timedelta(days=5)).isoformat()
        client.post(f"/organizations/{organization_id}/pilots", headers=headers, json={
            "display_name": "Lapsed", "licence_expires_on": past})

        status = client.get(f"/organizations/{organization_id}/fleet/status",
                            headers=headers).json()
        assert "Due-1" in status["aircraft"]["service_due"]
        assert "OLD-1" in status["batteries"]["past_cycle_limit"]
        assert "Lapsed" in status["pilots"]["not_current"]

    def test_it_states_that_it_does_not_prevent_a_flight(self, client, org):
        """The operator remains responsible; this is advisory, not an interlock."""
        headers, organization_id = org
        status = client.get(f"/organizations/{organization_id}/fleet/status",
                            headers=headers).json()
        assert "does not" in status["note"] or "remains responsible" in status["note"]

    def test_a_non_member_cannot_read_fleet_status(self, client, org):
        _, organization_id = org
        outsider = {"Authorization": "Bearer " + client.post("/auth/register", json={
            "email": "outsider@example.com", "password": "longenough1",
            "organization_name": "Elsewhere"}).json()["access_token"]}
        assert client.get(f"/organizations/{organization_id}/fleet/status",
                          headers=outsider).status_code == 404
