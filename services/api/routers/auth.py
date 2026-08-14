"""Registration, login, identity and API tokens."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import ApiToken, Membership, Organization, Role, User
from ..schemas import (
    ApiTokenCreate,
    ApiTokenCreated,
    ApiTokenOut,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from ..security import (
    SESSION_HOURS,
    CurrentUser,
    create_session_token,
    hash_password,
    hash_token,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "org"


def unique_slug(db: Session, name: str) -> str:
    base = slugify(name)
    candidate = base
    suffix = 2
    while db.scalar(select(Organization).where(Organization.slug == candidate)):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    """Create an account, and an organisation the account actually owns.

    Without that organisation the new user would authenticate successfully and then be
    permitted to see nothing, which reads as a broken login.
    """
    existing = db.scalar(select(User).where(User.email == payload.email.lower()))
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    user = User(
        email=payload.email.lower(),
        display_name=payload.display_name or payload.email.split("@")[0],
        password_hash=password_hash,
    )
    db.add(user)
    db.flush()

    org_name = payload.organization_name.strip() or f"{user.display_name}'s organization"
    organization = Organization(name=org_name, slug=unique_slug(db, org_name))
    db.add(organization)
    db.flush()
    db.add(Membership(user_id=user.id, organization_id=organization.id, role=Role.owner))

    record(db, action="user_registered", user_id=user.id,
           organization_id=organization.id, resource=f"user:{user.id}")
    db.commit()

    return TokenResponse(
        access_token=create_session_token(user.id), expires_in_hours=SESSION_HOURS
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    # One message for both branches: distinguishing them reveals which emails exist.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is disabled.")

    record(db, action="user_login", user_id=user.id, resource=f"user:{user.id}")
    db.commit()
    return TokenResponse(
        access_token=create_session_token(user.id), expires_in_hours=SESSION_HOURS
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user


@router.get("/tokens", response_model=list[ApiTokenOut])
def list_tokens(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> list[ApiToken]:
    return list(db.scalars(select(ApiToken).where(ApiToken.user_id == user.id)))


@router.post("/tokens", response_model=ApiTokenCreated, status_code=201)
def create_token(
    payload: ApiTokenCreate, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> ApiTokenCreated:
    """Issue a long-lived token for scripts and field devices.

    The secret is returned here and nowhere else; only its hash is stored.
    """
    secret, prefix = ApiToken.generate()
    token = ApiToken(
        user_id=user.id, name=payload.name or "api token",
        token_hash=hash_token(secret), prefix=prefix,
    )
    db.add(token)
    db.flush()
    record(db, action="api_token_created", user_id=user.id, resource=f"token:{token.id}")
    db.commit()

    return ApiTokenCreated(
        id=token.id, name=token.name, prefix=token.prefix,
        created_at=token.created_at, last_used_at=None, revoked=False, secret=secret,
    )


@router.delete("/tokens/{token_id}", status_code=204)
def revoke_token(
    token_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> None:
    token = db.get(ApiToken, token_id)
    if token is None or token.user_id != user.id:
        raise HTTPException(status_code=404, detail="Token not found.")
    token.revoked = True
    record(db, action="api_token_revoked", user_id=user.id, resource=f"token:{token.id}")
    db.commit()
