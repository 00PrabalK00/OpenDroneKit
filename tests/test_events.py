"""Webhooks and live event streaming.

Deliveries go to a real local HTTP server rather than a mock, so the signature a
receiver would actually verify is the one under test. A mocked transport would prove
the mock works, not that the bytes on the wire are signable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

from services.api.routers.events import sign_payload  # noqa: E402


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
        "email": "admin@example.com", "password": "longenough1",
        "organization_name": "Events Co",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    return headers, organization_id


class Receiver:
    """A real HTTP endpoint that records what it was sent."""

    def __init__(self, status: int = 200):
        self.received: list[dict] = []
        self.status = status
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                # HTTP header names are case-insensitive and clients normalise them
                # differently, so a real receiver must not match on exact case either.
                receiver.received.append({
                    "body": body,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                })
                self.send_response(receiver.status)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):  # silence the default stderr logging
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/hook"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def wait_for(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class TestSubscription:
    def test_the_signing_secret_is_shown_once(self, client, org):
        headers, organization_id = org
        with Receiver() as receiver:
            created = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                                  json={"url": receiver.url, "events": ["flight.completed"]})
        assert created.status_code == 201
        assert created.json()["signing_secret"]

        listed = client.get(f"/organizations/{organization_id}/webhooks", headers=headers).json()
        assert "signing_secret" not in listed[0]
        assert listed[0]["secret_prefix"]

    def test_an_unknown_event_is_refused_with_the_list(self, client, org):
        """Creating a webhook that can never fire helps nobody."""
        headers, organization_id = org
        response = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                               json={"url": "https://example.com/h", "events": ["flight.landed"]})
        assert response.status_code == 422
        assert "flight.completed" in response.json()["detail"]

    def test_subscribing_to_nothing_is_refused(self, client, org):
        headers, organization_id = org
        response = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                               json={"url": "https://example.com/h", "events": []})
        assert response.status_code == 422

    def test_a_non_http_url_is_refused(self, client, org):
        headers, organization_id = org
        response = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                               json={"url": "ftp://example.com/h", "events": ["ai.completed"]})
        assert response.status_code == 422

    def test_only_an_admin_may_manage_webhooks(self, client, org):
        headers, organization_id = org
        client.post("/auth/register", json={
            "email": "eng@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "eng@example.com", "role": "engineer"})
        engineer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "eng@example.com", "password": "longenough1"}).json()["access_token"]}

        assert client.get(f"/organizations/{organization_id}/webhooks",
                          headers=engineer).status_code == 403


class TestDelivery:
    def test_an_event_reaches_a_real_receiver(self, client, org):
        headers, organization_id = org
        with Receiver() as receiver:
            client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                        json={"url": receiver.url, "events": ["defect.created"]})

            import services.api.db as db_module
            from services.api.routers.events import deliver_event

            session = db_module.get_session_factory()()
            deliver_event(session, organization_id, "defect.created", {"defect_id": 7})
            session.close()

            assert wait_for(lambda: len(receiver.received) == 1), "no delivery arrived"
            payload = json.loads(receiver.received[0]["body"])
            assert payload["event"] == "defect.created"
            assert payload["data"]["defect_id"] == 7

    def test_the_signature_verifies_with_the_issued_secret(self, client, org):
        """This is the whole point of signing: a receiver must be able to check it."""
        headers, organization_id = org
        with Receiver() as receiver:
            secret = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                                 json={"url": receiver.url,
                                       "events": ["report.generated"]}).json()["signing_secret"]

            import services.api.db as db_module
            from services.api.routers.events import deliver_event

            session = db_module.get_session_factory()()
            deliver_event(session, organization_id, "report.generated", {"report_id": 3})
            session.close()

            assert wait_for(lambda: len(receiver.received) == 1)
            delivery = receiver.received[0]
            timestamp = delivery["headers"]["x-odk-timestamp"]
            signature = delivery["headers"]["x-odk-signature"]

            expected = hmac.new(
                secret.encode(), timestamp.encode() + b"." + delivery["body"], hashlib.sha256
            ).hexdigest()
            assert hmac.compare_digest(signature, expected)

    def test_a_tampered_body_fails_the_signature(self, client, org):
        headers, organization_id = org
        with Receiver() as receiver:
            secret = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                                 json={"url": receiver.url,
                                       "events": ["ai.completed"]}).json()["signing_secret"]

            import services.api.db as db_module
            from services.api.routers.events import deliver_event

            session = db_module.get_session_factory()()
            deliver_event(session, organization_id, "ai.completed", {"job": 1})
            session.close()
            assert wait_for(lambda: len(receiver.received) == 1)

            delivery = receiver.received[0]
            tampered = delivery["body"].replace(b'"job": 1', b'"job": 999')
            expected = hmac.new(
                secret.encode(),
                delivery["headers"]["x-odk-timestamp"].encode() + b"." + tampered,
                hashlib.sha256,
            ).hexdigest()
            assert delivery["headers"]["x-odk-signature"] != expected

    def test_the_timestamp_is_inside_the_signed_material(self):
        """Otherwise a captured delivery could be replayed with a fresh timestamp."""
        body = b'{"event":"x"}'
        assert sign_payload("s", body, "111") != sign_payload("s", body, "222")

    def test_only_subscribed_events_are_delivered(self, client, org):
        headers, organization_id = org
        with Receiver() as receiver:
            client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                        json={"url": receiver.url, "events": ["flight.completed"]})

            import services.api.db as db_module
            from services.api.routers.events import deliver_event

            session = db_module.get_session_factory()()
            notified = deliver_event(session, organization_id, "defect.created", {"x": 1})
            session.close()

            assert notified == []
            assert receiver.received == []

    def test_a_failing_receiver_is_recorded_not_swallowed(self, client, org):
        """A webhook that silently stopped working is the failure mode to surface."""
        headers, organization_id = org
        with Receiver(status=500) as receiver:
            webhook_id = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                                     json={"url": receiver.url,
                                           "events": ["processing.failed"]}).json()["id"]

            import services.api.db as db_module
            from services.api.routers.events import deliver_event

            session = db_module.get_session_factory()()
            deliver_event(session, organization_id, "processing.failed", {"job": 2})
            session.close()

            assert wait_for(lambda: bool(
                client.get(f"/webhooks/{webhook_id}/deliveries", headers=headers).json()))
            deliveries = client.get(f"/webhooks/{webhook_id}/deliveries", headers=headers).json()
            assert deliveries[0]["success"] is False
            assert deliveries[0]["status_code"] == 500

    def test_an_unreachable_receiver_records_the_error(self, client, org):
        headers, organization_id = org
        # Nothing is listening on this port.
        webhook_id = client.post(f"/organizations/{organization_id}/webhooks", headers=headers,
                                 json={"url": "http://127.0.0.1:9/hook",
                                       "events": ["mission.created"]}).json()["id"]

        import services.api.db as db_module
        from services.api.routers.events import deliver_event

        session = db_module.get_session_factory()()
        deliver_event(session, organization_id, "mission.created", {"mission": 1})
        session.close()

        assert wait_for(lambda: bool(
            client.get(f"/webhooks/{webhook_id}/deliveries", headers=headers).json()))
        deliveries = client.get(f"/webhooks/{webhook_id}/deliveries", headers=headers).json()
        assert deliveries[0]["success"] is False
        assert deliveries[0]["error"]


class TestHonestyAboutGuarantees:
    def test_delivery_is_declared_best_effort(self, client):
        """Implying a durable queue that does not exist would be the lie here."""
        payload = client.get("/event-types").json()
        assert "best-effort" in payload["delivery"].lower()
        assert "no durable queue" in payload["delivery"].lower()

    def test_the_signature_scheme_is_documented(self, client):
        payload = client.get("/event-types").json()
        assert "HMAC-SHA256" in payload["signature"]

    def test_the_stream_declares_its_multi_worker_limitation(self, client):
        info = client.get("/events/stream-info").json()
        assert "in-process" in info["limitation"].lower()


class TestLiveStream:
    def test_an_unauthorised_socket_is_closed_rather_than_left_open(self, client, org):
        from fastapi.testclient import TestClient as _TestClient  # noqa: F401

        _, organization_id = org
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/organizations/{organization_id}?token=bogus") as ws:
                message = ws.receive_json()
                assert "error" in message
                # The server closes after the error; the next receive raises.
                ws.receive_json()

    def test_a_member_connects_and_is_told_the_limitations(self, client, org):
        headers, organization_id = org
        token = headers["Authorization"].split()[1]
        with client.websocket_connect(
            f"/ws/organizations/{organization_id}?token={token}"
        ) as websocket:
            message = websocket.receive_json()
            assert message["event"] == "stream.connected"
            assert "limitation" in message["info"]
