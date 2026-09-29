"""Webhooks fire from the actions they name, and the live stream carries them too.

Before this, ``deliver_event`` was called only from tests: a user could subscribe a
webhook to defect.created, create a defect, and receive nothing, with an empty
delivery log. These tests perform each real action through the API and assert a real
HTTP receiver on loopback got a signed delivery for it.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

from hub_api_helpers import configure, project_with_dataset, register, reset_engine, upload, wait  # noqa: E402


class _Receiver(BaseHTTPRequestHandler):
    received: list = []

    def log_message(self, *_):
        return

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).received.append({
            "event": self.headers["X-ODK-Event"],
            "signature": self.headers["X-ODK-Signature"],
            "timestamp": self.headers["X-ODK-Timestamp"],
            "body": body,
        })
        self.send_response(204)
        self.end_headers()


@pytest.fixture
def receiver():
    _Receiver.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_engine()


ALL_EVENTS = [
    "mission.created", "mission.updated", "flight.completed", "dataset.uploaded",
    "processing.completed", "processing.failed", "ai.completed", "defect.created",
    "defect.reviewed", "report.generated", "project.shared", "telemetry.updated",
]


def _events(timeout: float = 10.0, want: set[str] | None = None) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen = {row["event"] for row in _Receiver.received}
        if want is not None and want <= seen:
            break
        time.sleep(0.05)
    return list(_Receiver.received)


def test_every_subscribable_event_is_emitted_by_a_real_action(client, receiver):
    headers, org = register(client)
    hook = client.post(f"/organizations/{org}/webhooks", headers=headers, json={
        "url": f"http://127.0.0.1:{receiver.server_port}/hook", "events": ALL_EVENTS,
    })
    assert hook.status_code == 201, hook.text
    secret = hook.json()["signing_secret"]
    project_id, dataset_id = project_with_dataset(client, headers, org)

    upload(client, headers, dataset_id)
    defect = client.post(f"/projects/{project_id}/defects", headers=headers,
                         json={"category": "crack", "severity": "high"}).json()
    client.post(f"/defects/{defect['id']}/review", headers=headers, json={"decision": "accepted"})
    client.post(f"/projects/{project_id}/reports", headers=headers, json={"title": "R"})
    client.post(f"/projects/{project_id}/shares", headers=headers, json={})
    mission = client.post(f"/projects/{project_id}/missions", headers=headers, json={
        "name": "Deck", "aoi": {"type": "Polygon", "coordinates": [[
            [-83.0, 41.0], [-82.999, 41.0], [-82.999, 41.001], [-83.0, 41.001], [-83.0, 41.0]]]},
    })
    assert mission.status_code == 201, mission.text
    renamed = client.patch(f"/missions/{mission.json()['id']}", headers=headers,
                           json={"name": "Deck, north span"})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Deck, north span"
    aircraft = client.post(f"/organizations/{org}/aircraft", headers=headers,
                           json={"name": "M300", "model": "M300 RTK"}).json()
    client.post(f"/aircraft/{aircraft['id']}/flights?hours=0.5", headers=headers)
    client.post(f"/organizations/{org}/telemetry", headers=headers, json={
        "aircraft_id": "M300", "observed_at": "2026-09-01T10:00:00Z", "source": "test-link",
        "connected": True, "latitude": 41.0, "longitude": -83.0,
    })
    # A report job completes (processing.completed); a reconstruction over an empty
    # dataset fails (processing.failed); a heuristic AI job completes (ai.completed).
    report_job = client.post(f"/projects/{project_id}/jobs", headers=headers,
                             json={"kind": "report"}).json()
    assert wait(client, headers, f"/jobs/{report_job['id']}")["status"] == "done"
    empty = client.post(f"/projects/{project_id}/datasets", headers=headers,
                        json={"name": "Empty"}).json()["id"]
    failing = client.post(f"/projects/{project_id}/jobs", headers=headers,
                          json={"kind": "reconstruction", "dataset_id": empty}).json()
    assert wait(client, headers, f"/jobs/{failing['id']}")["status"] == "failed"
    ai = client.post(f"/projects/{project_id}/ai-jobs", headers=headers, json={
        "task": "detection", "model_key": "heuristic", "input_ref": f"dataset:{dataset_id}",
    }).json()
    assert wait(client, headers, f"/ai-jobs/{ai['id']}")["status"] == "done"

    received = _events(want=set(ALL_EVENTS))
    assert {row["event"] for row in received} == set(ALL_EVENTS)

    from services.api.routers.events import sign_payload

    for row in received:
        assert row["signature"] == sign_payload(secret, row["body"], row["timestamp"])
        assert json.loads(row["body"])["event"] == row["event"]
    share = next(json.loads(r["body"]) for r in received if r["event"] == "project.shared")
    assert "url_token" not in share["data"], "a share credential must never be broadcast"

    log = client.get(f"/webhooks/{hook.json()['id']}/deliveries", headers=headers).json()
    assert len(log) >= len(ALL_EVENTS) and all(row["success"] for row in log)


def test_unsubscribed_organizations_and_events_are_not_sent(client, receiver):
    headers, org = register(client)
    other_headers, other_org = register(client, "other@example.com", "Other Co")
    client.post(f"/organizations/{other_org}/webhooks", headers=other_headers, json={
        "url": f"http://127.0.0.1:{receiver.server_port}/hook", "events": ["defect.created"],
    })
    client.post(f"/organizations/{org}/webhooks", headers=headers, json={
        "url": f"http://127.0.0.1:{receiver.server_port}/hook", "events": ["report.generated"],
    })
    project_id, _ = project_with_dataset(client, headers, org)
    client.post(f"/projects/{project_id}/defects", headers=headers,
                json={"category": "crack", "severity": "low"})
    time.sleep(0.5)
    assert _Receiver.received == []


def test_domain_events_reach_the_live_stream(client):
    headers, org = register(client)
    token = headers["Authorization"].split()[1]
    project_id, _ = project_with_dataset(client, headers, org)
    with client.websocket_connect(f"/ws/organizations/{org}?token={token}") as socket:
        assert socket.receive_json()["event"] == "stream.connected"
        created = client.post(f"/projects/{project_id}/defects", headers=headers,
                              json={"category": "spalling", "severity": "medium"}).json()
        message = socket.receive_json()
    assert message["event"] == "defect.created"
    assert message["data"]["id"] == created["id"]
