"""The capabilities the desktop app could not reach, tested where they now live.

Twenty-three buttons declared themselves unavailable because fleet records, share links,
webhooks and review decisions sat behind the FastAPI service while the desktop app speaks
to app/api.py. None of it was missing -- every one carries a verified registry row -- so
what was built was a path, not a feature.

app/desktop_ops.py opens the SAME database the service uses and calls the SAME modules.
These tests hold that claim: the rows are real rows, the refusals are real refusals, and
nothing here is a second implementation that will drift from the first.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy", reason="the service database layer needs SQLAlchemy")


@pytest.fixture(scope="module")
def database():
    """A throwaway database, so a test run never touches the developer's own records."""
    path = Path(tempfile.mkdtemp()) / "desktop_ops.db"
    previous = os.environ.get("ODK_DATABASE_URL")
    os.environ["ODK_DATABASE_URL"] = f"sqlite:///{path.as_posix()}"

    # The db module caches its engine, so a URL set after first use would be ignored.
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - resetting the cache is the point
    db_module._SessionLocal = None    # noqa: SLF001
    db_module.init_db()

    yield db_module

    if previous is None:
        os.environ.pop("ODK_DATABASE_URL", None)
    else:
        os.environ["ODK_DATABASE_URL"] = previous
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


@pytest.fixture(scope="module")
def org_and_project(database):
    from services.api.models import Organization, Project

    with database.get_session_factory()() as session:
        org = Organization(name="Test org", slug="test-org")
        session.add(org)
        session.commit()
        session.refresh(org)
        project = Project(organization_id=org.id, name="Test project")
        session.add(project)
        session.commit()
        session.refresh(project)
        return org.id, project.id


class TestFleetRecordsAreReal:
    def test_an_aircraft_is_persisted(self, org_and_project) -> None:
        from app import desktop_ops

        org_id, _ = org_and_project
        created = desktop_ops.add_aircraft(org_id, "Mavic 3E", "M3E", "SN-1")
        assert created["id"]
        assert desktop_ops.fleet_status(org_id)["aircraft"] >= 1

    def test_an_aircraft_needs_a_name(self, org_and_project) -> None:
        from app import desktop_ops

        org_id, _ = org_and_project
        with pytest.raises(ValueError, match="needs a name"):
            desktop_ops.add_aircraft(org_id, "   ")

    def test_a_battery_needs_a_serial(self, org_and_project) -> None:
        """Cycles are tracked by serial; a battery without one cannot be followed."""
        from app import desktop_ops

        org_id, _ = org_and_project
        with pytest.raises(ValueError, match="serial number"):
            desktop_ops.add_battery(org_id, "")

    def test_a_bad_licence_expiry_is_refused_not_defaulted(self, org_and_project) -> None:
        """A wrong expiry is worse than a missing one: the roster would clear a pilot
        whose licence has lapsed."""
        from app import desktop_ops

        org_id, _ = org_and_project
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            desktop_ops.add_pilot(org_id, "A Pilot", "L-1", "next tuesday")

    def test_maintenance_resets_the_service_clock(self, org_and_project) -> None:
        """A record that does not move hours_at_last_service leaves the aircraft
        permanently overdue, which teaches an operator to ignore the warning."""
        from services.api.models import Aircraft

        from app import desktop_ops

        org_id, _ = org_and_project
        aircraft = desktop_ops.add_aircraft(org_id, "Service test", "M3E", "SN-SVC")

        import services.api.db as db_module

        with db_module.get_session_factory()() as session:
            row = session.get(Aircraft, aircraft["id"])
            row.flight_hours = 120.0
            row.service_interval_hours = 50.0
            row.hours_at_last_service = 0.0
            session.commit()

        assert "Service test" in desktop_ops.fleet_status(org_id)["service_due"]
        desktop_ops.log_maintenance(aircraft["id"], "100h inspection", "belts", "engineer")
        assert "Service test" not in desktop_ops.fleet_status(org_id)["service_due"]

    def test_maintenance_on_an_unknown_aircraft_is_refused(self) -> None:
        from app import desktop_ops

        with pytest.raises(ValueError, match="No aircraft"):
            desktop_ops.log_maintenance(999999, "inspection")


class TestShareLinksBehaveLikeCredentials:
    def test_the_token_is_returned_once_and_only_hashed_afterwards(self, org_and_project) -> None:
        """A link that can be read back out of the database is a credential sitting in
        every backup."""
        from app import desktop_ops

        _, project_id = org_and_project
        created = desktop_ops.create_share_link(project_id, note="client review")
        token = created["token"]
        assert len(token) > 20

        listed = desktop_ops.list_share_links(project_id)
        assert all("token" not in link for link in listed)
        assert any(link["prefix"] == created["prefix"] for link in listed)

    def test_a_link_can_be_revoked(self, org_and_project) -> None:
        from app import desktop_ops

        _, project_id = org_and_project
        created = desktop_ops.create_share_link(project_id)
        desktop_ops.revoke_share_link(created["id"])
        revoked = [l for l in desktop_ops.list_share_links(project_id) if l["id"] == created["id"]]
        assert revoked and revoked[0]["revoked"] is True

    def test_revoking_something_that_does_not_exist_is_refused(self) -> None:
        from app import desktop_ops

        with pytest.raises(ValueError, match="No share link"):
            desktop_ops.revoke_share_link(999999)


class TestWebhooks:
    def test_a_webhook_is_registered_with_its_events(self, org_and_project) -> None:
        from app import desktop_ops

        org_id, _ = org_and_project
        created = desktop_ops.add_webhook(
            org_id, "https://example.com/hook", ["job.finished"], "CI"
        )
        assert created["secret"]
        listed = desktop_ops.list_webhooks(org_id)
        assert any(h["url"] == "https://example.com/hook" for h in listed)
        assert all("secret" not in hook for hook in listed)

    def test_a_non_http_url_is_refused(self, org_and_project) -> None:
        from app import desktop_ops

        org_id, _ = org_and_project
        with pytest.raises(ValueError, match="http"):
            desktop_ops.add_webhook(org_id, "ftp://example.com/hook")


class TestReviewKeepsWhatTheModelClaimed:
    @pytest.fixture
    def project_with_finding(self, tmp_path):
        from core.annotations import create_annotation

        annotation = create_annotation(
            tmp_path, "p1", "image", "DSC00229.JPG", "point",
            {"type": "Point", "coordinates": [10, 20]}, "crack", "medium", "open",
        )
        return tmp_path, annotation.id

    @pytest.mark.parametrize(
        "decision,expected",
        [("accept", "resolved"), ("reject", "dismissed"), ("flag", "in_review")],
    )
    def test_each_decision_moves_the_status(self, project_with_finding, decision, expected) -> None:
        from app import desktop_ops

        root, annotation_id = project_with_finding
        result = desktop_ops.review_finding(root, annotation_id, decision, "reviewer")
        assert result["status"] == expected

    def test_the_model_claim_survives_the_review(self, project_with_finding) -> None:
        """ai.human_validation requires the prediction to survive alongside the human
        answer. A reviewer disagreeing with a model is evidence ABOUT the model, and
        overwriting the claim destroys it."""
        from app import desktop_ops

        root, annotation_id = project_with_finding
        result = desktop_ops.review_finding(root, annotation_id, "reject", "reviewer")
        assert result["label"] == "crack"
        assert result["severity"] == "medium"
        assert result["geometry"]["coordinates"] == [10, 20]

    def test_the_reviewer_is_recorded(self, project_with_finding) -> None:
        """Annotation has no reviewed_by field and update_annotation drops unknown keys
        silently, so a reviewer stored that way would vanish without an error."""
        from app import desktop_ops

        root, annotation_id = project_with_finding
        result = desktop_ops.review_finding(root, annotation_id, "accept", "prabal")
        assert "prabal" in (result.get("note") or "")

    def test_an_unknown_decision_is_refused(self, project_with_finding) -> None:
        from app import desktop_ops

        root, annotation_id = project_with_finding
        with pytest.raises(ValueError, match="Unknown review decision"):
            desktop_ops.review_finding(root, annotation_id, "maybe")

    def test_an_unknown_finding_is_refused(self, project_with_finding) -> None:
        from app import desktop_ops

        root, _ = project_with_finding
        with pytest.raises(ValueError, match="No annotation"):
            desktop_ops.review_finding(root, "does-not-exist", "accept")


class TestItIsTheSameDatabaseAsTheService:
    def test_the_rows_are_the_service_orm(self, org_and_project) -> None:
        """Not a parallel table. A desktop-only fleet table would diverge from the
        service the first time either side changed.
        """
        from sqlalchemy import select

        import services.api.db as db_module
        from services.api.models import Aircraft

        from app import desktop_ops

        org_id, _ = org_and_project
        desktop_ops.add_aircraft(org_id, "Shared row", "M3E", "SN-SHARED")
        with db_module.get_session_factory()() as session:
            names = [a.name for a in session.scalars(select(Aircraft))]
        assert "Shared row" in names
