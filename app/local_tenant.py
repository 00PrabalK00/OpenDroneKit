"""Where the desktop app's projects meet the service database's foreign keys.

Share links and webhooks are rows in the service database (services/api), which the
desktop app opens directly through app/desktop_ops.py. Those rows reference
`projects.id` and `organizations.id` in THAT database. The desktop app's projects live
in a different one (final_toolkit_outputs/app_state/projects.db), and on a fresh install
the service database has no organisation at all. So every Share, New Key and Add Webhook
button failed with "IntegrityError: FOREIGN KEY constraint failed": the ids the cockpit
sent pointed at rows that did not exist.

This module creates the two rows those keys need, and nothing more:

  * one organisation standing for this installation, created only when the service
    database has none (an install that already has organisations uses them); and
  * for each desktop project that is shared, one service project carrying the same
    name and a description that says which desktop project it stands for.

Nothing is copied into the service project: a share link reaches what the service
holds for it, and the Api says so when the link is issued.
"""

from __future__ import annotations

from typing import Any

LOCAL_ORG_NAME = "This desktop installation"
LOCAL_ORG_SLUG = "local-desktop"


def _marker(desktop_project: dict[str, Any]) -> str:
    # The description is shown to whoever opens a share link, so it names the desktop
    # project by id only -- never the folder it lives in on this machine.
    return f"Shared from desktop project #{desktop_project['id']}"


def organization_id(db: Any, requested: int | None = None) -> int:
    """The organisation to file rows under.

    A requested id that exists is used as-is. With none requested, or with the desktop's
    default of 1 on a database that has no organisations yet, the local organisation is
    used, created on first need. Any other missing id is an error: guessing which tenant
    a webhook belongs to is not something to do silently.
    """
    from sqlalchemy import func, select

    from services.api.models import Organization

    if requested:
        found = db.get(Organization, int(requested))
        if found is not None:
            return found.id
    local = db.scalar(select(Organization).where(Organization.slug == LOCAL_ORG_SLUG))
    desktop_default = not requested or int(requested) == 1
    if local is not None and desktop_default:
        return local.id
    count = db.scalar(select(func.count()).select_from(Organization)) or 0
    if not desktop_default or (requested and count):
        raise ValueError(f"No organisation with id {requested} in the service database.")
    org = Organization(name=LOCAL_ORG_NAME, slug=LOCAL_ORG_SLUG)
    db.add(org)
    db.flush()
    return org.id


def service_project_id(db: Any, desktop_project: dict[str, Any]) -> int:
    """The service project standing for one desktop project, created on first need."""
    from sqlalchemy import select

    from services.api.models import Project

    org = organization_id(db)
    marker = _marker(desktop_project)
    found = db.scalar(select(Project).where(
        Project.organization_id == org, Project.description == marker))
    if found is not None:
        return found.id
    project = Project(organization_id=org, name=str(desktop_project.get("name") or "Project"),
                      description=marker, project_type="inspection")
    db.add(project)
    db.flush()
    return project.id
