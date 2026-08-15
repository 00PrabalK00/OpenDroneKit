"""Webhooks and live event streaming.

Two ways to learn that something happened, for two different consumers.

*Webhooks* are for other systems: a maintenance scheduler that wants to know a flight
finished, or a client portal that wants to know a report is ready. Deliveries are
signed so the receiver can verify the payload came from this deployment and was not
altered, and failures are recorded rather than swallowed, because a webhook that
silently stops working is worse than one that never existed.

*WebSockets* are for a browser watching a job or a flight in progress.

Delivery is best-effort and explicitly so. This is not a message broker: there is no
durable queue, and a receiver that is down during a delivery misses that event. The
endpoint says so rather than implying a guarantee it cannot keep.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db, get_session_factory
from ..models import Role, Webhook, WebhookDelivery
from ..realtime import broker
from ..security import CurrentUser, require_role

router = APIRouter(tags=["events"])

# The events a receiver may subscribe to. Kept explicit so a typo in a subscription
# is refused at creation rather than producing a webhook that never fires.
EVENT_TYPES = [
    "mission.created", "mission.updated",
    "flight.completed",
    "dataset.uploaded",
    "processing.completed", "processing.failed",
    "ai.completed",
    "defect.created", "defect.reviewed",
    "report.generated",
    "project.shared",
    "telemetry.updated",
]

DELIVERY_TIMEOUT_S = 10.0


class WebhookCreate(BaseModel):
    url: str = Field(min_length=8, max_length=800)
    events: list[str] = Field(default_factory=list)
    description: str = ""


class WebhookOut(BaseModel):
    id: int
    organization_id: int
    url: str
    events: list[str]
    description: str
    active: bool
    secret_prefix: str
    delivery_count: int
    failure_count: int
    last_delivery_at: datetime | None
    created_at: datetime


class WebhookCreated(WebhookOut):
    # Shown once. The receiver needs it to verify signatures; only a hash is stored.
    signing_secret: str


def _webhook_out(hook: Webhook) -> WebhookOut:
    try:
        events = json.loads(hook.events_json)
    except json.JSONDecodeError:
        events = []
    return WebhookOut(
        id=hook.id, organization_id=hook.organization_id, url=hook.url,
        events=events, description=hook.description, active=hook.active,
        secret_prefix=hook.secret_prefix, delivery_count=hook.delivery_count,
        failure_count=hook.failure_count, last_delivery_at=hook.last_delivery_at,
        created_at=hook.created_at,
    )


def sign_payload(secret: str, body: bytes, timestamp: str) -> str:
    """Signature a receiver can verify.

    The timestamp is inside the signed material so a captured delivery cannot be
    replayed later with a fresh timestamp header.
    """
    message = timestamp.encode("utf-8") + b"." + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


@router.get("/event-types")
def event_types() -> dict[str, Any]:
    return {
        "events": EVENT_TYPES,
        "delivery": (
            "Best-effort. There is no durable queue: a receiver that is unavailable "
            "during a delivery misses that event. Poll the relevant resource if you "
            "need certainty."
        ),
        "signature": (
            "Each delivery carries X-ODK-Timestamp and X-ODK-Signature, where the "
            "signature is HMAC-SHA256 over '<timestamp>.<body>' using the signing "
            "secret shown once at creation."
        ),
    }


@router.get("/organizations/{organization_id}/webhooks", response_model=list[WebhookOut])
def list_webhooks(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[WebhookOut]:
    require_role(db, user, organization_id, Role.admin)
    rows = db.scalars(select(Webhook).where(Webhook.organization_id == organization_id))
    return [_webhook_out(hook) for hook in rows]


@router.post("/organizations/{organization_id}/webhooks", response_model=WebhookCreated, status_code=201)
def create_webhook(
    organization_id: int, payload: WebhookCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> WebhookCreated:
    require_role(db, user, organization_id, Role.admin)

    if not payload.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="The URL must be http or https.")

    unknown = [event for event in payload.events if event not in EVENT_TYPES]
    if unknown:
        # Refusing here beats creating a webhook that can never fire.
        raise HTTPException(
            status_code=422,
            detail=f"Unknown event(s): {', '.join(unknown)}. Known: {', '.join(EVENT_TYPES)}.",
        )
    if not payload.events:
        raise HTTPException(status_code=422, detail="Subscribe to at least one event.")

    secret = secrets.token_urlsafe(32)
    hook = Webhook(
        organization_id=organization_id, url=payload.url,
        events_json=json.dumps(payload.events), description=payload.description,
        secret_hash=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        secret_prefix=secret[:8],
        # The plaintext is needed to sign outgoing deliveries. It is stored, unlike a
        # password, because the server must reproduce the signature; it is never
        # returned again after creation.
        secret_plain=secret,
        created_by=user.id,
    )
    db.add(hook)
    db.flush()
    record(db, action="webhook_created", user_id=user.id, organization_id=organization_id,
           resource=f"webhook:{hook.id}", detail={"url": payload.url})
    db.commit()

    return WebhookCreated(**_webhook_out(hook).model_dump(), signing_secret=secret)


@router.delete("/webhooks/{webhook_id}", status_code=204)
def delete_webhook(
    webhook_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> None:
    hook = db.get(Webhook, webhook_id)
    if hook is None:
        raise HTTPException(status_code=404, detail="Webhook not found.")
    require_role(db, user, hook.organization_id, Role.admin)
    db.delete(hook)
    record(db, action="webhook_deleted", user_id=user.id,
           organization_id=hook.organization_id, resource=f"webhook:{webhook_id}")
    db.commit()


@router.get("/webhooks/{webhook_id}/deliveries")
def webhook_deliveries(
    webhook_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Delivery history, successes and failures alike.

    A webhook that silently stopped working is the failure mode this exists to make
    visible.
    """
    hook = db.get(Webhook, webhook_id)
    if hook is None:
        raise HTTPException(status_code=404, detail="Webhook not found.")
    require_role(db, user, hook.organization_id, Role.admin)

    rows = db.scalars(
        select(WebhookDelivery).where(WebhookDelivery.webhook_id == webhook_id)
        .order_by(WebhookDelivery.created_at.desc()).limit(limit)
    )
    return [
        {"id": row.id, "event": row.event, "status_code": row.status_code,
         "success": row.success, "error": row.error,
         "created_at": row.created_at.isoformat()}
        for row in rows
    ]


def deliver_event(
    db: Session, organization_id: int, event: str, payload: dict[str, Any]
) -> list[int]:
    """Queue delivery to every subscribed webhook. Returns the webhook ids notified."""
    if event not in EVENT_TYPES:
        raise ValueError(f"Unknown event {event!r}")

    hooks = [
        hook for hook in db.scalars(
            select(Webhook).where(
                Webhook.organization_id == organization_id, Webhook.active.is_(True)
            )
        )
        if event in json.loads(hook.events_json or "[]")
    ]

    notified: list[int] = []
    for hook in hooks:
        notified.append(hook.id)
        threading.Thread(
            target=_send, args=(hook.id, event, payload),
            name=f"webhook-{hook.id}", daemon=True,
        ).start()
    return notified


def _send(webhook_id: int, event: str, payload: dict[str, Any]) -> None:
    """Deliver one event on a worker thread, recording the outcome either way."""
    session = get_session_factory()()
    try:
        hook = session.get(Webhook, webhook_id)
        if hook is None:
            return

        body = json.dumps({
            "event": event,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }, default=str).encode("utf-8")
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))

        request = urllib.request.Request(
            hook.url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-ODK-Event": event,
                "X-ODK-Timestamp": timestamp,
                "X-ODK-Signature": sign_payload(hook.secret_plain, body, timestamp),
            },
        )

        status_code = 0
        error = ""
        success = False
        try:
            with urllib.request.urlopen(request, timeout=DELIVERY_TIMEOUT_S) as response:
                status_code = int(response.status)
                success = 200 <= status_code < 300
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            error = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        session.add(WebhookDelivery(
            webhook_id=hook.id, event=event, status_code=status_code,
            success=success, error=error[:500],
        ))
        hook.delivery_count += 1
        hook.last_delivery_at = datetime.now(timezone.utc)
        if not success:
            hook.failure_count += 1
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# live streaming and telemetry publication
# ---------------------------------------------------------------------------


class TelemetryPublish(BaseModel):
    """One observed vehicle sample; absent sensors remain absent rather than guessed."""

    aircraft_id: str = Field(min_length=1, max_length=200)
    observed_at: datetime
    source: str = Field(min_length=1, max_length=200)
    mission_id: int | None = None
    connected: bool
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    altitude_m: float | None = None
    battery_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    flight_mode: str = ""
    satellites: int | None = Field(default=None, ge=0)
    ground_speed_m_s: float | None = Field(default=None, ge=0.0)
    heading_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    raw: dict[str, Any] = Field(default_factory=dict)


@router.post("/organizations/{organization_id}/telemetry", status_code=202)
def publish_telemetry(
    organization_id: int, payload: TelemetryPublish,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Publish an authenticated telemetry sample into the shared live stream.

    Acceptance means the shared broker committed the sample. It does not claim a
    particular browser received it; WebSockets have no acknowledgement protocol.
    """
    require_role(db, user, organization_id, Role.pilot)
    wire = payload.model_dump(mode="json")
    event_id = broker.publish(organization_id, "telemetry.updated", wire)
    return {
        "accepted": True,
        "broker_event_id": event_id,
        "event": "telemetry.updated",
        "delivery": "committed_to_shared_broker_not_acknowledged_by_clients",
    }


@router.get("/events/stream-info")
def stream_info() -> dict[str, Any]:
    """What the live stream can and cannot promise."""
    return broker.describe()


@router.websocket("/ws/organizations/{organization_id}")
async def organization_stream(websocket: WebSocket, organization_id: int, token: str = "") -> None:
    """Live events for one organization.

    The token arrives as a query parameter because a browser WebSocket cannot set an
    Authorization header. It is verified exactly as a normal request would be, and an
    unauthorised socket is closed rather than left open and silent.
    """
    from ..models import Membership, User
    from ..security import decode_session_token, hash_token
    from ..models import ApiToken

    await websocket.accept()

    session = get_session_factory()()
    try:
        user: User | None = None
        if token.startswith("odk_"):
            api_token = session.scalar(
                select(ApiToken).where(
                    ApiToken.token_hash == hash_token(token), ApiToken.revoked.is_(False)
                )
            )
            if api_token is not None:
                user = session.get(User, api_token.user_id)
        else:
            user_id = decode_session_token(token)
            if user_id is not None:
                user = session.get(User, user_id)

        membership = None
        if user is not None:
            membership = session.scalar(
                select(Membership).where(
                    Membership.user_id == user.id,
                    Membership.organization_id == organization_id,
                )
            )

        if membership is None:
            await websocket.send_json({"error": "not authorised for this organization"})
            await websocket.close(code=4403)
            return

        cursor = broker.latest_id(organization_id)
        await websocket.send_json({
            "event": "stream.connected",
            "organization_id": organization_id,
            "info": broker.describe(),
        })
    finally:
        session.close()

    try:
        while True:
            # Each worker polls the same database cursor, so an event produced by worker
            # A reaches a browser connected to worker B. The short timeout also consumes
            # pings and notices disconnects without requiring a chatty browser.
            rows = await asyncio.to_thread(broker.read_after, organization_id, cursor)
            for event in rows:
                await websocket.send_json(event.to_wire())
                cursor = event.id
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.15)
            except TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
