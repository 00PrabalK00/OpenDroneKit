"""Share, Revoke Links, Add Webhook and New Key work on a fresh install.

Audit, before this change: on a fresh install every one of them failed with
"IntegrityError: FOREIGN KEY constraint failed". The cockpit sent DESKTOP project ids
(final_toolkit_outputs/app_state/projects.db) to rows that reference the SERVICE
database's projects and organisations -- a different file, which on a fresh install has
no organisation at all. New Key was a share link labelled as an API key.

These tests start from an empty service database, drive the buttons in the browser, and
then read the rows back through the service's own ORM.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api

pytest.importorskip("sqlalchemy")


@pytest.fixture
def service_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(tmp_path / 'service.db').as_posix()}")
    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001 - the engine is cached per URL
    db_module._SessionLocal = None    # noqa: SLF001
    db_module.init_db()
    yield db_module
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


def rows(db_module, model):
    from sqlalchemy import select

    with db_module.get_session_factory()() as db:
        return list(db.scalars(select(model)))


def test_share_and_revoke_on_a_fresh_install(service_db, tmp_path) -> None:
    from services.api.models import Organization, Project, ShareLink

    assert rows(service_db, Organization) == []
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Projects");
            await ODK.press("Share");
            await ODK.answer();                       // "Issue a share link?" -> Yes
            await ODK.answer("Client review");        // what is it for
            const dialog = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal textarea"));
            const token = dialog.value;
            ODK.$(".modal-backdrop .modal .modal-actions .tbtn.primary").click();
            const issued = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Link ")));
            await ODK.press("Revoke Links");
            await ODK.answer();
            const revoked = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Revoked")));
            return { token, issued, revoked };
        """)
    assert "FOREIGN KEY" not in str(cockpit.calls)
    links = rows(service_db, ShareLink)
    assert len(links) == 1 and links[0].revoked
    project = rows(service_db, Project)[0]
    assert project.name == "Harness Site" and links[0].project_id == project.id
    # The recipient sees the description, so it must not carry a local folder path.
    assert str(tmp_path) not in project.description
    assert result["revoked"] == "Revoked 1 link(s)."
    assert "not uploaded" in result["token"], "the link does not say what it reaches"
    import hashlib

    token = result["token"].split("\n\n")[1].strip()
    assert hashlib.sha256(token.encode()).hexdigest() == links[0].token_hash


def test_webhook_on_a_fresh_install(service_db, tmp_path) -> None:
    from services.api.models import Organization, Webhook

    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Developers");
            await ODK.press("Add Webhook");
            await ODK.answer("https://hooks.example.test/odk");
            await ODK.answer("job.finished");
            const dialog = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal textarea"));
            ODK.$(".modal-backdrop .modal .modal-actions .tbtn.primary").click();
            const said = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Webhook registered")));
            await ODK.sleep(1500);
            return { said, secret: dialog.value, panel: ODK.panel("Webhooks") };
        """)
    hooks = rows(service_db, Webhook)
    assert len(hooks) == 1 and hooks[0].url == "https://hooks.example.test/odk"
    orgs = rows(service_db, Organization)
    assert [o.slug for o in orgs] == ["local-desktop"] and hooks[0].organization_id == orgs[0].id
    assert "hooks.example.test" in result["panel"]


def test_new_key_is_an_api_token_for_a_service_account(service_db, tmp_path) -> None:
    from services.api.models import ApiToken, ShareLink, User
    from services.api.security import hash_password, hash_token

    with service_db.get_session_factory()() as db:
        db.add(User(email="pilot@example.test", display_name="Pilot",
                    password_hash=hash_password("correct horse")))
        db.commit()

    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Developers");
            await ODK.press("New Key");
            const form = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal"));
            const inputs = form.querySelectorAll("input");
            inputs[0].value = "pilot@example.test";
            inputs[1].value = "wrong";
            form.querySelector(".modal-actions .tbtn.primary").click();
            const refused = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("New Key: Incorrect")));
            await ODK.press("New Key");
            const again = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal"));
            const fields = again.querySelectorAll("input");
            const passwordType = fields[1].type;
            fields[0].value = "pilot@example.test";
            fields[1].value = "correct horse";
            again.querySelector(".modal-actions .tbtn.primary").click();
            const dialog = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal textarea"));
            const secret = dialog.value.split("\\n\\n")[1];
            ODK.$(".modal-backdrop .modal .modal-actions .tbtn.primary").click();
            return { refused, passwordType, secret };
        """)
    assert result["refused"]
    assert result["passwordType"] == "password"
    tokens = rows(service_db, ApiToken)
    assert len(tokens) == 1 and tokens[0].token_hash == hash_token(result["secret"].strip())
    assert rows(service_db, ShareLink) == [], "New Key still issues a share link"
