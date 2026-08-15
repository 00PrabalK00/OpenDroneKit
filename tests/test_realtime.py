"""Shared realtime delivery and authenticated telemetry publication."""

from __future__ import annotations

from datetime import datetime, timezone

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
def organization(client):
    response = client.post("/auth/register", json={
        "email": "pilot@example.com", "password": "longenough1",
        "organization_name": "Realtime Survey",
    })
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    return headers, token, organization_id


def telemetry_payload() -> dict:
    return {
        "aircraft_id": "aircraft-IN-17",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "mavlink:udp:14550",
        "mission_id": 42,
        "connected": True,
        "latitude": 12.9716,
        "longitude": 77.5946,
        "altitude_m": 64.2,
        "battery_pct": 71.5,
        "flight_mode": "AUTO",
        "satellites": 18,
        "ground_speed_m_s": 7.4,
        "heading_deg": 92.0,
    }


class TestSharedBroker:
    def test_independent_broker_instances_read_the_same_committed_event(self, client, organization):
        """Two instances model separate workers against the same real SQLite database."""
        from services.api.realtime import DatabaseEventBroker

        _, _, organization_id = organization
        producer = DatabaseEventBroker(retention_per_organization=200)
        consumer = DatabaseEventBroker(retention_per_organization=200)
        event_id = producer.publish(organization_id, "telemetry.updated", telemetry_payload())

        rows = consumer.read_after(organization_id, event_id - 1)
        assert [row.id for row in rows] == [event_id]
        assert rows[0].data["aircraft_id"] == "aircraft-IN-17"
        assert rows[0].data["latitude"] == pytest.approx(12.9716)

    def test_stream_declares_best_effort_boundary(self, client):
        info = client.get("/events/stream-info").json()
        assert info["broker"] == "shared_database"
        assert info["multi_worker"] is True
        assert "no client acknowledgements" in info["limitation"].lower()
        assert "exactly-once" in info["limitation"].lower()


class TestTelemetryStream:
    def test_published_telemetry_reaches_an_authenticated_socket(self, client, organization):
        headers, token, organization_id = organization
        with client.websocket_connect(
            f"/ws/organizations/{organization_id}?token={token}"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["event"] == "stream.connected"
            assert connected["info"]["broker"] == "shared_database"

            accepted = client.post(
                f"/organizations/{organization_id}/telemetry",
                headers=headers, json=telemetry_payload(),
            )
            assert accepted.status_code == 202
            assert accepted.json()["delivery"].endswith("not_acknowledged_by_clients")

            delivered = websocket.receive_json()
            assert delivered["event"] == "telemetry.updated"
            assert delivered["data"]["aircraft_id"] == "aircraft-IN-17"
            assert delivered["data"]["battery_pct"] == pytest.approx(71.5)
            assert delivered["id"] == accepted.json()["broker_event_id"]

    def test_non_pilot_cannot_publish_telemetry(self, client, organization):
        owner, _, organization_id = organization
        client.post("/auth/register", json={
            "email": "viewer@example.com", "password": "longenough1",
            "organization_name": "Personal",
        })
        client.post(f"/organizations/{organization_id}/members", headers=owner, json={
            "email": "viewer@example.com", "role": "viewer",
        })
        token = client.post("/auth/login", json={
            "email": "viewer@example.com", "password": "longenough1",
        }).json()["access_token"]
        response = client.post(
            f"/organizations/{organization_id}/telemetry",
            headers={"Authorization": f"Bearer {token}"}, json=telemetry_payload(),
        )
        assert response.status_code == 403

    def test_invalid_measured_fields_are_refused(self, client, organization):
        headers, _, organization_id = organization
        payload = telemetry_payload()
        payload["battery_pct"] = 140
        response = client.post(
            f"/organizations/{organization_id}/telemetry", headers=headers, json=payload,
        )
        assert response.status_code == 422
