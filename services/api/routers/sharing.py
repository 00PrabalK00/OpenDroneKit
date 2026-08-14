"""Share links for client delivery.

A client should be able to open a link and see the deliverable without an account.
That convenience is exactly what makes the security properties matter, so they are
explicit here rather than assumed:

* The token is high-entropy and only its hash is stored, so a database disclosure
  does not hand over working links.
* A link grants access to one project and nothing else, at view level only. It is not
  a session and cannot be escalated into one.
* Expiry and revocation are checked on every access, not only at creation.
* Every access is logged, because "who saw this, and when" is a question that gets
  asked after a dispute.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import Defect, Mission, Project, Role, ShareLink, ShareAccess
from ..security import CurrentUser, hash_password, require_role, verify_password

router = APIRouter(tags=["sharing"])


class ShareCreate(BaseModel):
    expires_in_days: int | None = Field(default=30, ge=1, le=3650)
    password: str = ""
    allow_download: bool = False
    note: str = ""
    include_defects: bool = True
    include_missions: bool = True


class ShareOut(BaseModel):
    id: int
    project_id: int
    prefix: str
    expires_at: datetime | None
    revoked: bool
    allow_download: bool
    password_protected: bool
    access_count: int
    created_at: datetime


class ShareCreated(ShareOut):
    # Returned once, at creation. Only the hash is stored.
    url_token: str


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    """Return a timezone-aware UTC datetime, whatever the database handed back.

    SQLite has no timezone type and returns naive datetimes, while PostgreSQL with
    `timezone=True` returns aware ones. Comparing the two raises, so an expiry check
    written for one backend fails on the other. Values are stored as UTC, so a naive
    value is interpreted as UTC rather than as local time.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _share_out(link: ShareLink) -> ShareOut:
    return ShareOut(
        id=link.id, project_id=link.project_id, prefix=link.prefix,
        expires_at=link.expires_at, revoked=link.revoked,
        allow_download=link.allow_download,
        password_protected=bool(link.password_hash),
        access_count=link.access_count, created_at=link.created_at,
    )


@router.post("/projects/{project_id}/shares", response_model=ShareCreated, status_code=201)
def create_share(
    project_id: int, payload: ShareCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> ShareCreated:
    """Mint a share link. The token is shown once and never recoverable."""
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.engineer)

    token = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)
        if payload.expires_in_days else None
    )

    link = ShareLink(
        project_id=project_id,
        token_hash=_hash_token(token),
        prefix=token[:8],
        # Reusing the password hasher: a share password deserves the same treatment
        # as an account password, since people reuse them.
        password_hash=hash_password(payload.password) if payload.password else "",
        expires_at=expires_at,
        allow_download=payload.allow_download,
        note=payload.note,
        include_defects=payload.include_defects,
        include_missions=payload.include_missions,
        created_by=user.id,
    )
    db.add(link)
    db.flush()
    record(db, action="share_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"share:{link.id}",
           detail={"expires_at": str(expires_at), "password": bool(payload.password)})
    db.commit()

    result = _share_out(link).model_dump()
    return ShareCreated(**result, url_token=token)


@router.get("/projects/{project_id}/shares", response_model=list[ShareOut])
def list_shares(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[ShareOut]:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.engineer)
    rows = db.scalars(select(ShareLink).where(ShareLink.project_id == project_id))
    return [_share_out(link) for link in rows]


@router.delete("/shares/{share_id}", status_code=204)
def revoke_share(
    share_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> None:
    """Revoke immediately. The link is dead on the next access, not at expiry."""
    link = db.get(ShareLink, share_id)
    if link is None:
        raise HTTPException(status_code=404, detail="Share link not found.")
    project = _project_or_404(db, link.project_id)
    require_role(db, user, project.organization_id, Role.engineer)

    link.revoked = True
    record(db, action="share_revoked", user_id=user.id,
           organization_id=project.organization_id, resource=f"share:{share_id}")
    db.commit()


@router.get("/shares/{share_id}/accesses")
def share_accesses(
    share_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[dict[str, Any]]:
    """Who opened this link and when."""
    link = db.get(ShareLink, share_id)
    if link is None:
        raise HTTPException(status_code=404, detail="Share link not found.")
    project = _project_or_404(db, link.project_id)
    require_role(db, user, project.organization_id, Role.engineer)

    rows = db.scalars(
        select(ShareAccess).where(ShareAccess.share_id == share_id)
        .order_by(ShareAccess.accessed_at.desc()).limit(500)
    )
    return [
        {"accessed_at": row.accessed_at.isoformat(), "client_ip": row.client_ip,
         "user_agent": row.user_agent, "outcome": row.outcome}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# public access
# ---------------------------------------------------------------------------


def _resolve_share(
    db: Session, token: str, password: str, request: Request
) -> ShareLink:
    """Validate a token, logging every attempt including the failures.

    A failed attempt is worth recording: repeated failures against one link are how
    you notice a token being guessed or a revoked link still circulating.
    """
    link = db.scalar(select(ShareLink).where(ShareLink.token_hash == _hash_token(token)))

    def log(outcome: str, matched: ShareLink | None) -> None:
        if matched is None:
            return
        db.add(ShareAccess(
            share_id=matched.id,
            client_ip=(request.client.host if request.client else ""),
            user_agent=str(request.headers.get("user-agent", ""))[:400],
            outcome=outcome,
        ))
        db.commit()

    if link is None:
        # Same message for an unknown and a revoked token: distinguishing them tells
        # a probe which tokens once existed.
        raise HTTPException(status_code=404, detail="This link is not valid.")
    if link.revoked:
        log("revoked", link)
        raise HTTPException(status_code=404, detail="This link is not valid.")
    expires_at = _as_utc(link.expires_at)
    if expires_at is not None and expires_at < datetime.now(timezone.utc):
        log("expired", link)
        raise HTTPException(status_code=410, detail="This link has expired.")
    if link.password_hash and not verify_password(password, link.password_hash):
        log("bad_password", link)
        raise HTTPException(status_code=401, detail="This link requires a password.")

    link.access_count += 1
    link.last_accessed_at = datetime.now(timezone.utc)
    log("granted", link)
    return link


@router.get("/public/shares/{token}")
def view_share(
    token: str, request: Request, db: Annotated[Session, Depends(get_db)],
    x_share_password: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    """Open a shared project. No account required, and no write access granted."""
    link = _resolve_share(db, token, x_share_password, request)
    project = db.get(Project, link.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="This link is not valid.")

    payload: dict[str, Any] = {
        "project": {
            "name": project.name, "description": project.description,
            "client": project.client, "address": project.address,
            "project_type": project.project_type,
            "longitude": project.longitude, "latitude": project.latitude,
            "crs_epsg": project.crs_epsg,
        },
        "allow_download": link.allow_download,
        "expires_at": link.expires_at.isoformat() if link.expires_at else None,
        # Said plainly so a recipient knows this is a read-only view.
        "access": "view-only; this link grants no write access and is not an account",
    }

    if link.include_missions:
        missions = db.scalars(select(Mission).where(Mission.project_id == project.id))
        payload["missions"] = [
            {"name": m.name, "template": m.template, "waypoints": m.waypoint_count,
             "distance_m": m.distance_m, "duration_min": m.duration_min,
             "crs_epsg": m.crs_epsg}
            for m in missions
        ]

    if link.include_defects:
        defects = list(db.scalars(select(Defect).where(Defect.project_id == project.id)))
        payload["defects"] = [
            {"category": d.category, "severity": d.severity, "area_m2": d.area_m2,
             "length_m": d.length_m, "review_state": d.review_state,
             "source": d.source, "longitude": d.longitude, "latitude": d.latitude,
             "crs_epsg": d.crs_epsg}
            for d in defects
        ]
        unreviewed = sum(
            1 for d in defects if d.source == "model" and d.review_state != "accepted"
        )
        if unreviewed:
            # A client seeing this data deserves the same caveat the report carries.
            payload["caveat"] = (
                f"{unreviewed} finding(s) were produced by an automated model and have "
                "not been confirmed by an inspector."
            )

    return payload
