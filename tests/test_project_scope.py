"""Project-scoped membership and asset inspection history.

Access control is the one place where a well-meaning change locks someone out of their
own work, so the tests below pin the direction of every grant. Project membership adds
permission and never removes it; restriction is opt-in per project so existing projects
keep behaving as they did; and an administrator cannot be shut out of their own tenancy,
because an administrator who can be locked out cannot fix anything.

The asset timeline exists to answer the question a single survey cannot: is this
structure getting worse. So it must keep confirmed findings separate from unconfirmed
model predictions, and must not imply a trend it has not got the inspections to support.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'scope.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


def register(client, email: str, organization_name: str = "") -> dict:
    payload = {"email": email, "password": "longenough1"}
    if organization_name:
        payload["organization_name"] = organization_name
    response = client.post("/auth/register", json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()


def headers_for(response: dict) -> dict:
    return {"Authorization": f"Bearer {response['access_token']}"}


@pytest.fixture
def owner(client):
    created = register(client, "owner@example.com", organization_name="Acme Surveys")
    headers = headers_for(created)
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    return headers, organization_id


@pytest.fixture
def outsider(client):
    return headers_for(register(client, "outsider@example.com",
                                organization_name="Other Co"))


def user_id_of(client, headers) -> int:
    return client.get("/auth/me", headers=headers).json()["id"]


def make_project(client, headers, org_id, **overrides):
    payload = {"name": "Bridge 14", **overrides}
    response = client.post(f"/organizations/{org_id}/projects", headers=headers, json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()


class TestUnrestrictedProjects:
    def test_an_organisation_member_can_open_an_unrestricted_project(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)

        response = client.get(f"/projects/{project['id']}", headers=headers)
        assert response.status_code == 200

    def test_projects_are_not_restricted_by_default(self, client, owner):
        """Existing projects must keep behaving exactly as they did."""
        headers, org_id = owner
        assert make_project(client, headers, org_id)["restricted"] is False


class TestProjectMembership:
    def test_a_member_can_be_named_on_a_project(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))

        response = client.put(
            f"/projects/{project['id']}/members/{other_id}",
            headers=headers, params={"role": "engineer"})
        assert response.status_code == 201, response.text
        assert response.json()["role"] == "engineer"

    def test_members_are_listed_with_the_effective_rule_explained(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))
        client.put(f"/projects/{project['id']}/members/{other_id}",
                   headers=headers, params={"role": "engineer"})

        payload = client.get(f"/projects/{project['id']}/members", headers=headers).json()
        assert len(payload["members"]) == 1
        assert "higher of the two" in payload["note"]

    def test_adding_the_same_member_twice_updates_rather_than_duplicates(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))

        client.put(f"/projects/{project['id']}/members/{other_id}",
                   headers=headers, params={"role": "viewer"})
        client.put(f"/projects/{project['id']}/members/{other_id}",
                   headers=headers, params={"role": "admin"})

        members = client.get(f"/projects/{project['id']}/members", headers=headers).json()["members"]
        assert len(members) == 1
        assert members[0]["role"] == "admin"

    def test_an_unknown_role_is_refused_and_lists_the_real_ones(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))

        response = client.put(f"/projects/{project['id']}/members/{other_id}",
                              headers=headers, params={"role": "wizard"})
        assert response.status_code == 422
        assert "viewer" in response.json()["detail"]

    def test_an_unknown_user_is_refused(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        response = client.put(f"/projects/{project['id']}/members/99999",
                              headers=headers, params={"role": "viewer"})
        assert response.status_code == 404

    def test_a_member_can_be_removed(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))
        client.put(f"/projects/{project['id']}/members/{other_id}",
                   headers=headers, params={"role": "engineer"})

        assert client.delete(f"/projects/{project['id']}/members/{other_id}",
                             headers=headers).status_code == 204
        members = client.get(f"/projects/{project['id']}/members", headers=headers).json()["members"]
        assert members == []

    def test_removing_someone_who_is_not_a_member_is_a_clear_404(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id)
        other_id = user_id_of(client, headers_for(register(client, "engineer@example.com")))
        response = client.delete(f"/projects/{project['id']}/members/{other_id}",
                                 headers=headers)
        assert response.status_code == 404


class TestRestrictedProjects:
    def test_an_outsider_cannot_see_a_restricted_project(self, client, owner):
        """404 rather than 403: the existence of a client's job is itself disclosure."""
        headers, org_id = owner
        project = make_project(client, headers, org_id, restricted=True)

        outsider = headers_for(register(client, "outsider@example.com",
                                        organization_name="Other Co"))
        assert client.get(f"/projects/{project['id']}", headers=outsider).status_code == 404

    def test_an_admin_keeps_access_to_a_restricted_project(self, client, owner):
        """An administrator who can be locked out of their own tenancy cannot fix it."""
        headers, org_id = owner
        project = make_project(client, headers, org_id, restricted=True)
        assert client.get(f"/projects/{project['id']}", headers=headers).status_code == 200

    def test_a_restricted_project_is_hidden_from_the_listing_too(self, client, owner):
        headers, org_id = owner
        make_project(client, headers, org_id, name="Open work")
        make_project(client, headers, org_id, name="Confidential", restricted=True)

        member = headers_for(register(client, "member@example.com",
                                      organization_name="Third Co"))
        # Not a member of the organisation at all, so nothing is visible.
        response = client.get(f"/organizations/{org_id}/projects", headers=member)
        assert response.status_code in (403, 404)

    def test_the_listing_note_explains_the_restriction(self, client, owner):
        headers, org_id = owner
        project = make_project(client, headers, org_id, restricted=True)
        payload = client.get(f"/projects/{project['id']}/members", headers=headers).json()
        assert payload["restricted"] is True
        assert "not sufficient" in payload["note"]


class TestAssetTimeline:
    def _asset(self, client, headers, org_id):
        response = client.post(f"/organizations/{org_id}/assets", headers=headers,
                               json={"name": "Bridge 14", "asset_type": "bridge"})
        assert response.status_code in (200, 201), response.text
        return response.json()

    def test_an_asset_with_no_inspections_says_so_rather_than_implying_none_found(
            self, client, owner):
        headers, org_id = owner
        asset = self._asset(client, headers, org_id)

        payload = client.get(f"/assets/{asset['id']}/timeline", headers=headers).json()
        assert payload["inspection_count"] == 0
        assert payload["trend"] == "unknown"
        assert "No inspections are linked" in payload["note"]

    def test_projects_linked_to_an_asset_appear_on_its_timeline(self, client, owner):
        headers, org_id = owner
        asset = self._asset(client, headers, org_id)
        make_project(client, headers, org_id, name="Q1 survey", asset_id=asset["id"])
        make_project(client, headers, org_id, name="Q2 survey", asset_id=asset["id"])

        payload = client.get(f"/assets/{asset['id']}/timeline", headers=headers).json()
        assert payload["inspection_count"] == 2
        assert [e["project_name"] for e in payload["timeline"]] == ["Q1 survey", "Q2 survey"]

    def test_a_project_not_linked_to_the_asset_is_not_counted(self, client, owner):
        headers, org_id = owner
        asset = self._asset(client, headers, org_id)
        make_project(client, headers, org_id, name="Linked", asset_id=asset["id"])
        make_project(client, headers, org_id, name="Unrelated")

        payload = client.get(f"/assets/{asset['id']}/timeline", headers=headers).json()
        assert payload["inspection_count"] == 1

    def test_a_single_inspection_cannot_establish_a_trend(self, client, owner):
        """One survey says what is there, not whether it is getting worse."""
        headers, org_id = owner
        asset = self._asset(client, headers, org_id)
        make_project(client, headers, org_id, name="Only survey", asset_id=asset["id"])

        payload = client.get(f"/assets/{asset['id']}/timeline", headers=headers).json()
        assert payload["trend"] == "unknown"

    def test_the_timeline_warns_that_surveys_may_not_be_comparable(self, client, owner):
        headers, org_id = owner
        asset = self._asset(client, headers, org_id)
        make_project(client, headers, org_id, name="Q1", asset_id=asset["id"])

        payload = client.get(f"/assets/{asset['id']}/timeline", headers=headers).json()
        assert "not directly comparable" in payload["note"]

    def test_an_unknown_asset_is_a_404(self, client, owner):
        headers, _ = owner
        assert client.get("/assets/99999/timeline", headers=headers).status_code == 404
