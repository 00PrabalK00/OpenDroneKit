"""Share links.

The convenience — a client opens a URL with no account — is exactly what makes the
security properties worth testing. A share must grant view access to one project and
nothing more, must die on revocation or expiry, and must leave a record of who opened
it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def project(client):
    response = client.post("/auth/register", json={
        "email": "owner@example.com", "password": "longenough1",
        "organization_name": "Delivery Co",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                             json={"name": "Bridge 7", "client": "County Roads"}).json()["id"]
    client.post(f"/projects/{project_id}/defects", headers=headers, json={
        "category": "crack", "severity": "high", "source": "model",
        "model_key": "crack_segmentation", "confidence": 0.8, "area_m2": 0.5})
    return headers, organization_id, project_id


class TestCreation:
    def test_the_token_is_returned_once_and_not_stored(self, client, project):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={})
        assert created.status_code == 201
        assert created.json()["url_token"]

        listed = client.get(f"/projects/{project_id}/shares", headers=headers).json()
        assert "url_token" not in listed[0], "the share token must not be retrievable"
        assert listed[0]["prefix"], "a prefix is needed to identify the link in a list"

    def test_a_viewer_cannot_mint_a_share(self, client, project):
        """Sharing a project outward is not a read operation."""
        headers, organization_id, project_id = project
        client.post("/auth/register", json={
            "email": "v@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "v@example.com", "role": "viewer"})
        viewer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "v@example.com", "password": "longenough1"}).json()["access_token"]}

        assert client.post(f"/projects/{project_id}/shares", headers=viewer,
                           json={}).status_code == 403


class TestPublicAccess:
    def test_a_valid_link_opens_without_an_account(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers,
                            json={}).json()["url_token"]

        # No Authorization header at all.
        response = client.get(f"/public/shares/{token}")
        assert response.status_code == 200
        assert response.json()["project"]["name"] == "Bridge 7"

    def test_the_response_states_it_is_read_only(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers,
                            json={}).json()["url_token"]
        payload = client.get(f"/public/shares/{token}").json()
        assert "view-only" in payload["access"]

    def test_unreviewed_model_findings_carry_a_caveat(self, client, project):
        """A client seeing this data deserves the same caveat the report carries."""
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers,
                            json={}).json()["url_token"]
        payload = client.get(f"/public/shares/{token}").json()
        assert "not been confirmed" in payload["caveat"]

    def test_an_unknown_token_is_refused(self, client):
        assert client.get("/public/shares/nonsense-token").status_code == 404

    def test_content_selection_is_honoured(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers, json={
            "include_defects": False}).json()["url_token"]
        payload = client.get(f"/public/shares/{token}").json()
        assert "defects" not in payload
        assert "missions" in payload


class TestRevocationAndExpiry:
    def test_a_revoked_link_stops_working_immediately(self, client, project):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={}).json()
        token = created["url_token"]
        assert client.get(f"/public/shares/{token}").status_code == 200

        assert client.delete(f"/shares/{created['id']}", headers=headers).status_code == 204
        assert client.get(f"/public/shares/{token}").status_code == 404

    def test_a_revoked_link_is_indistinguishable_from_an_unknown_one(self, client, project):
        """Otherwise a probe learns which tokens once existed."""
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={}).json()
        client.delete(f"/shares/{created['id']}", headers=headers)

        revoked = client.get(f"/public/shares/{created['url_token']}")
        unknown = client.get("/public/shares/definitely-not-a-real-token")
        assert revoked.status_code == unknown.status_code == 404
        assert revoked.json()["detail"] == unknown.json()["detail"]

    def test_an_expired_link_is_refused(self, client, project, tmp_path):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers,
                              json={"expires_in_days": 1}).json()

        # Move the expiry into the past directly, rather than waiting a day.
        import services.api.db as db_module
        from services.api.models import ShareLink

        session = db_module.get_session_factory()()
        link = session.get(ShareLink, created["id"])
        link.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        session.commit()
        session.close()

        response = client.get(f"/public/shares/{created['url_token']}")
        assert response.status_code == 410
        assert "expired" in response.json()["detail"].lower()


class TestPassword:
    def test_a_password_protected_link_refuses_without_the_password(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers, json={
            "password": "client-secret-1"}).json()["url_token"]

        assert client.get(f"/public/shares/{token}").status_code == 401

    def test_the_correct_password_opens_it(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers, json={
            "password": "client-secret-1"}).json()["url_token"]

        response = client.get(f"/public/shares/{token}",
                              headers={"X-Share-Password": "client-secret-1"})
        assert response.status_code == 200

    def test_a_wrong_password_is_refused(self, client, project):
        headers, _, project_id = project
        token = client.post(f"/projects/{project_id}/shares", headers=headers, json={
            "password": "client-secret-1"}).json()["url_token"]

        assert client.get(f"/public/shares/{token}",
                          headers={"X-Share-Password": "guess"}).status_code == 401

    def test_the_password_is_not_stored_in_the_clear(self, client, project):
        headers, _, project_id = project
        client.post(f"/projects/{project_id}/shares", headers=headers,
                    json={"password": "client-secret-1"})

        import services.api.db as db_module
        from services.api.models import ShareLink

        session = db_module.get_session_factory()()
        link = session.scalars(__import__("sqlalchemy").select(ShareLink)).first()
        stored = link.password_hash
        session.close()
        assert "client-secret-1" not in stored
        assert stored.startswith("$2")  # bcrypt


class TestAccessLog:
    def test_a_successful_access_is_recorded(self, client, project):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={}).json()
        client.get(f"/public/shares/{created['url_token']}")

        accesses = client.get(f"/shares/{created['id']}/accesses", headers=headers).json()
        assert len(accesses) == 1
        assert accesses[0]["outcome"] == "granted"

    def test_a_failed_password_attempt_is_recorded(self, client, project):
        """Repeated failures are how a guessed token becomes visible."""
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers,
                              json={"password": "client-secret-1"}).json()
        client.get(f"/public/shares/{created['url_token']}",
                   headers={"X-Share-Password": "wrong"})

        accesses = client.get(f"/shares/{created['id']}/accesses", headers=headers).json()
        assert any(entry["outcome"] == "bad_password" for entry in accesses)

    def test_the_access_count_increments(self, client, project):
        headers, _, project_id = project
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={}).json()
        for _ in range(3):
            client.get(f"/public/shares/{created['url_token']}")

        listed = client.get(f"/projects/{project_id}/shares", headers=headers).json()
        assert listed[0]["access_count"] == 3
