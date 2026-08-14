"""Organizations and membership."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record, recent
from ..db import get_db
from ..models import Membership, Organization, ROLE_RANK, Role, User
from ..schemas import (
    MemberInvite,
    MemberOut,
    MemberRoleUpdate,
    OrganizationCreate,
    OrganizationOut,
)
from ..security import CurrentUser, require_role, role_in_organization
from .auth import unique_slug

router = APIRouter(prefix="/organizations", tags=["organizations"])


@router.get("", response_model=list[OrganizationOut])
def list_organizations(
    user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[OrganizationOut]:
    """Only organisations the caller belongs to. There is no global listing."""
    rows = db.execute(
        select(Organization, Membership.role)
        .join(Membership, Membership.organization_id == Organization.id)
        .where(Membership.user_id == user.id)
    ).all()
    return [
        OrganizationOut(
            id=org.id, name=org.name, slug=org.slug,
            role=role.value, created_at=org.created_at,
        )
        for org, role in rows
    ]


@router.post("", response_model=OrganizationOut, status_code=201)
def create_organization(
    payload: OrganizationCreate, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> OrganizationOut:
    organization = Organization(name=payload.name, slug=unique_slug(db, payload.name))
    db.add(organization)
    db.flush()
    db.add(Membership(user_id=user.id, organization_id=organization.id, role=Role.owner))
    record(db, action="organization_created", user_id=user.id,
           organization_id=organization.id, resource=f"org:{organization.id}")
    db.commit()
    return OrganizationOut(
        id=organization.id, name=organization.name, slug=organization.slug,
        role=Role.owner.value, created_at=organization.created_at,
    )


@router.get("/{organization_id}/members", response_model=list[MemberOut])
def list_members(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[MemberOut]:
    require_role(db, user, organization_id, Role.viewer)
    rows = db.execute(
        select(User, Membership.role)
        .join(Membership, Membership.user_id == User.id)
        .where(Membership.organization_id == organization_id)
    ).all()
    return [
        MemberOut(user_id=member.id, email=member.email,
                  display_name=member.display_name, role=role.value)
        for member, role in rows
    ]


@router.post("/{organization_id}/members", response_model=MemberOut, status_code=201)
def add_member(
    organization_id: int, payload: MemberInvite,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> MemberOut:
    """Add an existing account to this organisation.

    An admin cannot grant a role above their own, which is what stops privilege
    escalation by way of inviting yourself back at a higher level.
    """
    actor_role = require_role(db, user, organization_id, Role.admin)
    if ROLE_RANK[payload.role] > ROLE_RANK[actor_role]:
        raise HTTPException(
            status_code=403,
            detail=f"You cannot grant {payload.role.value}, which is above your own role.",
        )

    invitee = db.scalar(select(User).where(User.email == payload.email.lower()))
    if invitee is None:
        raise HTTPException(status_code=404, detail="No account with that email.")
    if role_in_organization(db, invitee, organization_id) is not None:
        raise HTTPException(status_code=409, detail="Already a member of this organization.")

    db.add(Membership(user_id=invitee.id, organization_id=organization_id, role=payload.role))
    record(db, action="member_added", user_id=user.id, organization_id=organization_id,
           resource=f"user:{invitee.id}", detail={"role": payload.role.value})
    db.commit()
    return MemberOut(user_id=invitee.id, email=invitee.email,
                     display_name=invitee.display_name, role=payload.role.value)


@router.patch("/{organization_id}/members/{member_id}", response_model=MemberOut)
def update_member_role(
    organization_id: int, member_id: int, payload: MemberRoleUpdate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> MemberOut:
    actor_role = require_role(db, user, organization_id, Role.admin)
    if ROLE_RANK[payload.role] > ROLE_RANK[actor_role]:
        raise HTTPException(status_code=403, detail="You cannot grant a role above your own.")

    membership = db.scalar(
        select(Membership).where(
            Membership.organization_id == organization_id, Membership.user_id == member_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    if ROLE_RANK[membership.role] > ROLE_RANK[actor_role]:
        raise HTTPException(status_code=403, detail="You cannot modify a more privileged member.")

    # An organisation with no owner cannot be administered again by anyone.
    if membership.role == Role.owner and payload.role != Role.owner:
        owners = db.scalars(
            select(Membership).where(
                Membership.organization_id == organization_id, Membership.role == Role.owner
            )
        ).all()
        if len(owners) <= 1:
            raise HTTPException(
                status_code=409, detail="An organization must retain at least one owner."
            )

    membership.role = payload.role
    record(db, action="member_role_changed", user_id=user.id, organization_id=organization_id,
           resource=f"user:{member_id}", detail={"role": payload.role.value})
    db.commit()

    member = db.get(User, member_id)
    return MemberOut(user_id=member.id, email=member.email,
                     display_name=member.display_name, role=payload.role.value)


@router.delete("/{organization_id}/members/{member_id}", status_code=204)
def remove_member(
    organization_id: int, member_id: int,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> None:
    actor_role = require_role(db, user, organization_id, Role.admin)
    membership = db.scalar(
        select(Membership).where(
            Membership.organization_id == organization_id, Membership.user_id == member_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    if ROLE_RANK[membership.role] > ROLE_RANK[actor_role]:
        raise HTTPException(status_code=403, detail="You cannot remove a more privileged member.")
    if membership.role == Role.owner:
        owners = db.scalars(
            select(Membership).where(
                Membership.organization_id == organization_id, Membership.role == Role.owner
            )
        ).all()
        if len(owners) <= 1:
            raise HTTPException(
                status_code=409, detail="An organization must retain at least one owner."
            )

    db.delete(membership)
    record(db, action="member_removed", user_id=user.id,
           organization_id=organization_id, resource=f"user:{member_id}")
    db.commit()


@router.get("/{organization_id}/audit")
def organization_audit(
    organization_id: int, user: CurrentUser,
    db: Annotated[Session, Depends(get_db)], limit: int = 200,
) -> list[dict]:
    require_role(db, user, organization_id, Role.admin)
    return [
        {
            "id": entry.id, "action": entry.action, "resource": entry.resource,
            "detail": entry.detail, "user_id": entry.user_id,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in recent(db, organization_id=organization_id, limit=limit)
    ]
