"""Authentication and authorisation.

Two credential types, deliberately different in lifetime:

*Session tokens* are short-lived JWTs issued at login, for a browser.
*API tokens* are long-lived opaque secrets for scripts and field devices. Only a hash
is stored, so a database disclosure does not hand over working credentials, and the
secret is shown exactly once at creation.

Authorisation asks one question everywhere: what is this user's role in the
organisation that owns the row being touched. `require_role` is the only place that
answers it, so a new endpoint cannot invent its own weaker rule.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import base64
import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import ApiToken, Membership, ROLE_RANK, Role, User

# bcrypt is used directly rather than through passlib, whose bcrypt backend reads a
# private `__about__` attribute that bcrypt 4.x removed. That shim failure surfaced as
# a bogus "password cannot be longer than 72 bytes" on an eleven-character password.

JWT_ALGORITHM = "HS256"
SESSION_HOURS = 12


# RFC 7518 section 3.2: an HMAC key for SHA-256 should be at least as long as the
# hash output. A shorter key weakens the signature rather than failing loudly.
MINIMUM_SECRET_BYTES = 32


def jwt_secret() -> str:
    """Signing secret.

    A deployment must set ODK_SECRET_KEY. The development fallback is random per
    process, so forgetting to set it invalidates sessions on restart rather than
    shipping a predictable signing key.

    A configured secret shorter than 32 bytes is strengthened by hashing rather than
    used as given, since a short key silently produces a weaker signature.
    """
    configured = os.environ.get("ODK_SECRET_KEY", "").strip()
    if configured:
        raw = configured.encode("utf-8")
        if len(raw) >= MINIMUM_SECRET_BYTES:
            return configured
        return hashlib.sha256(raw).hexdigest()

    global _EPHEMERAL_SECRET
    if not _EPHEMERAL_SECRET:
        _EPHEMERAL_SECRET = os.urandom(MINIMUM_SECRET_BYTES).hex()
    return _EPHEMERAL_SECRET


def secret_is_deployment_grade() -> tuple[bool, str]:
    """Whether the signing secret is fit for a deployment, and why not if it is not."""
    configured = os.environ.get("ODK_SECRET_KEY", "").strip()
    if not configured:
        return False, (
            "ODK_SECRET_KEY is unset: session tokens are signed with an ephemeral key "
            "and every restart invalidates existing sessions. Set it before deploying."
        )
    if len(configured.encode("utf-8")) < MINIMUM_SECRET_BYTES:
        return False, (
            f"ODK_SECRET_KEY is shorter than {MINIMUM_SECRET_BYTES} bytes. It is being "
            "stretched by hashing, but a longer secret should be configured."
        )
    return True, ""


_EPHEMERAL_SECRET = ""


def _bcrypt_input(password: str) -> bytes:
    """Prepare a password for bcrypt, which silently truncates beyond 72 bytes.

    Truncation is the dangerous behaviour: two different long passphrases sharing a
    72-byte prefix would become the same credential. Pre-hashing removes the length
    limit entirely, so a long passphrase is fully significant.
    """
    raw = password.encode("utf-8")
    if len(raw) <= 72:
        return raw
    return base64.b64encode(hashlib.sha256(raw).digest())


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    return bcrypt.hashpw(_bcrypt_input(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_bcrypt_input(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def hash_token(secret: str) -> str:
    """API tokens are high-entropy already, so a single SHA-256 is appropriate here."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def create_session_token(user_id: int, hours: int = SESSION_HOURS) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=hours)).timestamp()),
        "typ": "session",
    }
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "session":
        return None
    try:
        return int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        return None


UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authentication required.",
    headers={"WWW-Authenticate": "Bearer"},
)


def current_user(
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """Resolve the caller from a session JWT or an API token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UNAUTHENTICATED
    credential = authorization[7:].strip()
    if not credential:
        raise UNAUTHENTICATED

    user: User | None = None
    if credential.startswith("odk_"):
        token = db.scalar(
            select(ApiToken).where(
                ApiToken.token_hash == hash_token(credential), ApiToken.revoked.is_(False)
            )
        )
        if token is not None:
            token.last_used_at = datetime.now(timezone.utc)
            db.commit()
            user = db.get(User, token.user_id)
    else:
        user_id = decode_session_token(credential)
        if user_id is not None:
            user = db.get(User, user_id)

    if user is None or not user.is_active:
        raise UNAUTHENTICATED
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def role_in_organization(db: Session, user: User, organization_id: int) -> Role | None:
    membership = db.scalar(
        select(Membership).where(
            Membership.user_id == user.id, Membership.organization_id == organization_id
        )
    )
    return membership.role if membership else None


def require_role(db: Session, user: User, organization_id: int, minimum: Role) -> Role:
    """Assert the caller holds at least `minimum` in this organisation.

    A non-member gets 404, not 403: telling an outsider that an organisation exists is
    itself a disclosure.
    """
    role = role_in_organization(db, user, organization_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Organization not found.")
    if ROLE_RANK[role] < ROLE_RANK[minimum]:
        raise HTTPException(
            status_code=403,
            detail=f"Requires {minimum.value} or higher; you are {role.value}.",
        )
    return role
