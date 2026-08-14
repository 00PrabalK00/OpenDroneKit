"""Hub API: authentication, role enforcement, and mission planning through the API.

Authorisation is the part worth testing hardest. A permission bug does not look like a
bug -- the endpoint returns data, just to the wrong person -- so these tests assert the
negative cases as carefully as the positive ones.

Each test gets a fresh database file so ordering cannot make one test pass because of
another's leftovers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402

AOI = [
    [-81.7510, 41.3035],
    [-81.7490, 41.3035],
    [-81.7490, 41.3050],
    [-81.7510, 41.3050],
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client backed by an isolated database, with the app lifespan actually run."""
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-not-for-deployment")

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


def register(client, email: str, password: str = "longenough1", org: str = "") -> dict:
    response = client.post("/auth/register", json={
        "email": email, "password": password, "organization_name": org,
    })
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestHealth:
    def test_reports_status_and_backend(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["database"]["backend"] == "sqlite"

    def test_declares_that_sqlite_is_not_postgis(self, client):
        """A development database must not be mistaken for a deployment."""
        payload = client.get("/health").json()
        assert payload["database"]["postgis"] is False
        assert any("PostGIS" in w for w in payload["warnings"])


class TestAuthentication:
    def test_register_creates_an_owned_organization(self, client):
        headers = register(client, "owner@example.com", org="Acme Survey")
        organizations = client.get("/organizations", headers=headers).json()
        assert len(organizations) == 1
        assert organizations[0]["name"] == "Acme Survey"
        assert organizations[0]["role"] == "owner"

    def test_duplicate_email_is_refused(self, client):
        register(client, "dupe@example.com")
        response = client.post("/auth/register", json={
            "email": "dupe@example.com", "password": "longenough1",
        })
        assert response.status_code == 409

    def test_short_password_is_refused(self, client):
        response = client.post("/auth/register", json={"email": "x@y.com", "password": "short"})
        assert response.status_code == 422

    def test_login_returns_a_working_token(self, client):
        register(client, "login@example.com")
        response = client.post("/auth/login", json={
            "email": "login@example.com", "password": "longenough1",
        })
        assert response.status_code == 200
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        assert client.get("/auth/me", headers=headers).json()["email"] == "login@example.com"

    def test_wrong_password_is_rejected(self, client):
        register(client, "pw@example.com")
        response = client.post("/auth/login", json={
            "email": "pw@example.com", "password": "wrongpassword",
        })
        assert response.status_code == 401

    def test_unknown_email_gives_the_same_message_as_a_wrong_password(self, client):
        """Distinguishing them would disclose which accounts exist."""
        register(client, "known@example.com")
        wrong_password = client.post("/auth/login", json={
            "email": "known@example.com", "password": "wrongpassword"})
        unknown_user = client.post("/auth/login", json={
            "email": "nobody@example.com", "password": "longenough1"})
        assert wrong_password.status_code == unknown_user.status_code == 401
        assert wrong_password.json()["detail"] == unknown_user.json()["detail"]

    def test_unauthenticated_requests_are_refused(self, client):
        assert client.get("/organizations").status_code == 401
        assert client.get("/auth/me").status_code == 401

    def test_a_forged_token_is_refused(self, client):
        headers = {"Authorization": "Bearer not.a.real.token"}
        assert client.get("/auth/me", headers=headers).status_code == 401


class TestApiTokens:
    def test_secret_is_shown_once_and_then_never_again(self, client):
        headers = register(client, "tok@example.com")
        created = client.post("/auth/tokens", headers=headers, json={"name": "field tablet"})
        assert created.status_code == 201
        secret = created.json()["secret"]

        listed = client.get("/auth/tokens", headers=headers).json()
        assert len(listed) == 1
        assert "secret" not in listed[0], "the token secret must not be retrievable"

    def test_an_api_token_authenticates(self, client):
        headers = register(client, "apitok@example.com")
        secret = client.post("/auth/tokens", headers=headers, json={}).json()["secret"]
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {secret}"})
        assert response.status_code == 200
        assert response.json()["email"] == "apitok@example.com"

    def test_a_revoked_token_stops_working(self, client):
        headers = register(client, "revoke@example.com")
        created = client.post("/auth/tokens", headers=headers, json={}).json()
        secret = created["secret"]
        assert client.delete(f"/auth/tokens/{created['id']}", headers=headers).status_code == 204
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {secret}"}).status_code == 401


class TestAuthorization:
    def test_a_non_member_cannot_see_an_organization(self, client):
        owner = register(client, "owner2@example.com", org="Private Co")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]

        outsider = register(client, "outsider@example.com", org="Other Co")
        response = client.get(f"/organizations/{organization_id}/members", headers=outsider)
        # 404 rather than 403: confirming existence is itself a disclosure.
        assert response.status_code == 404

    def test_a_viewer_cannot_create_a_project(self, client):
        owner = register(client, "owner3@example.com", org="Acme")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]
        register(client, "viewer@example.com", org="Personal")
        client.post(f"/organizations/{organization_id}/members", headers=owner,
                    json={"email": "viewer@example.com", "role": "viewer"})

        viewer = {"Authorization": f"Bearer " + client.post("/auth/login", json={
            "email": "viewer@example.com", "password": "longenough1"}).json()["access_token"]}
        response = client.post(f"/organizations/{organization_id}/projects",
                               headers=viewer, json={"name": "Should fail"})
        assert response.status_code == 403

    def test_an_admin_cannot_grant_a_role_above_their_own(self, client):
        """Otherwise an admin could promote themselves by way of a second account."""
        owner = register(client, "owner4@example.com", org="Acme")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]

        register(client, "admin@example.com", org="Personal")
        register(client, "target@example.com", org="Personal2")
        client.post(f"/organizations/{organization_id}/members", headers=owner,
                    json={"email": "admin@example.com", "role": "admin"})

        admin = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "admin@example.com", "password": "longenough1"}).json()["access_token"]}
        response = client.post(f"/organizations/{organization_id}/members", headers=admin,
                               json={"email": "target@example.com", "role": "owner"})
        assert response.status_code == 403

    def test_the_last_owner_cannot_be_demoted(self, client):
        """An organisation with no owner can never be administered again."""
        owner = register(client, "solo@example.com", org="Solo Co")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]
        user_id = client.get("/auth/me", headers=owner).json()["id"]

        response = client.patch(f"/organizations/{organization_id}/members/{user_id}",
                                headers=owner, json={"role": "viewer"})
        assert response.status_code == 409

    def test_a_member_can_be_added_and_removed(self, client):
        owner = register(client, "owner5@example.com", org="Acme")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]
        register(client, "member@example.com", org="Personal")

        added = client.post(f"/organizations/{organization_id}/members", headers=owner,
                            json={"email": "member@example.com", "role": "engineer"})
        assert added.status_code == 201
        member_id = added.json()["user_id"]
        assert len(client.get(f"/organizations/{organization_id}/members", headers=owner).json()) == 2

        assert client.delete(f"/organizations/{organization_id}/members/{member_id}",
                             headers=owner).status_code == 204
        assert len(client.get(f"/organizations/{organization_id}/members", headers=owner).json()) == 1


class TestProjectsAndMissions:
    def test_plan_a_mission_over_a_drawn_area(self, client):
        headers = register(client, "planner@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "Bridge 7"}).json()["id"]

        response = client.post(f"/projects/{project_id}/missions", headers=headers, json={
            "name": "Grid 1", "template": "grid", "aoi": AOI, "altitude_m": 60.0,
        })
        assert response.status_code == 201, response.text
        mission = response.json()
        assert mission["waypoint_count"] > 0
        assert mission["distance_m"] > 0

    def test_a_mission_without_an_area_is_refused(self, client):
        """Planning must never fall back to a default polygon elsewhere in the world."""
        headers = register(client, "noaoi@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "No AOI"}).json()["id"]

        response = client.post(f"/projects/{project_id}/missions", headers=headers,
                               json={"name": "Nowhere", "template": "grid"})
        assert response.status_code == 422
        assert "area of interest" in response.json()["detail"].lower()

    def test_geojson_polygon_is_accepted_as_an_area(self, client):
        headers = register(client, "geojson@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "GeoJSON"}).json()["id"]

        response = client.post(f"/projects/{project_id}/missions", headers=headers, json={
            "name": "From GeoJSON", "template": "grid",
            "aoi": {"type": "Polygon", "coordinates": [AOI + [AOI[0]]]},
        })
        assert response.status_code == 201
        assert response.json()["waypoint_count"] > 0

    def test_mission_export_uses_the_shared_exporters(self, client):
        headers = register(client, "export@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "Export"}).json()["id"]
        mission_id = client.post(f"/projects/{project_id}/missions", headers=headers, json={
            "name": "Grid", "template": "grid", "aoi": AOI}).json()["id"]

        for fmt in ("qgc_plan", "qgc_wpl", "kml", "litchi", "dji_wpml"):
            response = client.get(f"/missions/{mission_id}/export/{fmt}", headers=headers)
            assert response.status_code == 200, f"{fmt}: {response.text}"
            assert response.json()["bytes"] > 0

    def test_unknown_export_format_is_refused_with_the_list(self, client):
        headers = register(client, "badfmt@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "Fmt"}).json()["id"]
        mission_id = client.post(f"/projects/{project_id}/missions", headers=headers, json={
            "name": "Grid", "template": "grid", "aoi": AOI}).json()["id"]

        response = client.get(f"/missions/{mission_id}/export/nonsense", headers=headers)
        assert response.status_code == 404
        assert "qgc_plan" in response.json()["detail"]

    def test_the_stored_plan_can_be_retrieved_verbatim(self, client):
        """A flown mission must always be reproducible."""
        headers = register(client, "plan@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                                 json={"name": "Plan"}).json()["id"]
        mission_id = client.post(f"/projects/{project_id}/missions", headers=headers, json={
            "name": "Grid", "template": "grid", "aoi": AOI}).json()["id"]

        payload = client.get(f"/missions/{mission_id}/plan", headers=headers).json()
        assert payload["aoi"]["type"] == "Polygon"
        assert payload["crs_epsg"] == 4326
        assert payload["plan"]


class TestAssets:
    def test_an_asset_stores_its_geometry_and_crs(self, client):
        headers = register(client, "asset@example.com", org="Survey Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]

        response = client.post(f"/organizations/{organization_id}/assets", headers=headers, json={
            "name": "Bridge 7", "asset_type": "bridge",
            "geometry": {"type": "Polygon", "coordinates": [AOI + [AOI[0]]]},
            "crs_epsg": 4326,
        })
        assert response.status_code == 201
        asset = response.json()
        assert asset["geometry"]["type"] == "Polygon"
        assert asset["crs_epsg"] == 4326


class TestAudit:
    def test_actions_are_recorded(self, client):
        headers = register(client, "audit@example.com", org="Audited Co")
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        client.post(f"/organizations/{organization_id}/projects", headers=headers,
                    json={"name": "Logged"})

        entries = client.get(f"/organizations/{organization_id}/audit", headers=headers).json()
        actions = {entry["action"] for entry in entries}
        assert "project_created" in actions
        assert "user_registered" in actions

    def test_a_viewer_cannot_read_the_audit_log(self, client):
        owner = register(client, "auditowner@example.com", org="Acme")
        organization_id = client.get("/organizations", headers=owner).json()[0]["id"]
        register(client, "auditviewer@example.com", org="Personal")
        client.post(f"/organizations/{organization_id}/members", headers=owner,
                    json={"email": "auditviewer@example.com", "role": "viewer"})

        viewer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "auditviewer@example.com", "password": "longenough1"}).json()["access_token"]}
        assert client.get(f"/organizations/{organization_id}/audit", headers=viewer).status_code == 403
