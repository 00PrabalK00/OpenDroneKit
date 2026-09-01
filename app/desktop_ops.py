"""The capabilities the desktop app could not reach, reached.

Twenty-three buttons in the cockpit declared themselves unavailable because fleet
records, share links, webhooks and review decisions live behind the FastAPI service in
`services/api`, while the desktop app is local-first and speaks to `app/api.py`.

They were never missing capabilities. Fleet management, sharing, webhooks, reports and
annotation review are all implemented and carry verified registry rows -- there was
simply no path from a button to them.

This module is that path, and it takes the local-first constraint seriously: it opens the
SAME database the service uses (`services/api/db.py`, SQLite by default) and works through
the SAME ORM models, rather than starting a web server and talking to it over HTTP. A user
with no network gets working fleet records; a deployment that points ODK_DATABASE_URL at
PostgreSQL gets the same rows the service sees, because it is the same table.

What this module does NOT do is reimplement any of it. Where a capability exists -- the
report engine, the annotation store, the plugin registry -- it is called, not copied. A
second implementation would drift from the first, and the first is the one with the tests.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import secrets
from pathlib import Path
from typing import Any

TOKEN_BYTES = 24


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _session():
    """A session on the service's database.

    Imported lazily: the desktop app must start on a machine where SQLAlchemy is absent,
    and the failure should arrive when someone presses a fleet button rather than at
    launch.
    """
    from services.api.db import get_session_factory, init_db

    init_db()
    return get_session_factory()()


# --------------------------------------------------------------------- fleet


def add_aircraft(organization_id: int, name: str, model: str = "", serial: str = "") -> dict[str, Any]:
    """Register an aircraft against an organisation."""
    from services.api.models import Aircraft

    if not str(name).strip():
        raise ValueError("An aircraft needs a name.")
    with _session() as db:
        aircraft = Aircraft(
            organization_id=int(organization_id),
            name=str(name).strip(),
            model=str(model).strip(),
            serial_number=str(serial).strip(),
        )
        db.add(aircraft)
        db.commit()
        db.refresh(aircraft)
        return {
            "id": aircraft.id,
            "name": aircraft.name,
            "model": aircraft.model,
            "serial_number": aircraft.serial_number,
            "flight_hours": float(aircraft.flight_hours or 0.0),
        }


def add_battery(organization_id: int, serial: str, capacity_mah: int = 0,
                cycle_limit: int = 0) -> dict[str, Any]:
    from services.api.models import Battery

    if not str(serial).strip():
        raise ValueError("A battery needs a serial number; it is how cycles are tracked.")
    with _session() as db:
        battery = Battery(
            organization_id=int(organization_id),
            serial_number=str(serial).strip(),
            capacity_mah=int(capacity_mah or 0),
            cycle_limit=int(cycle_limit or 0),
        )
        db.add(battery)
        db.commit()
        db.refresh(battery)
        return {
            "id": battery.id,
            "serial_number": battery.serial_number,
            "capacity_mah": battery.capacity_mah,
            "cycle_count": battery.cycle_count,
            "cycle_limit": battery.cycle_limit,
        }


def add_pilot(organization_id: int, display_name: str, licence_number: str = "",
              licence_expires_on: str = "") -> dict[str, Any]:
    from services.api.models import PilotProfile

    if not str(display_name).strip():
        raise ValueError("A pilot record needs a name.")
    expires: date | None = None
    if str(licence_expires_on).strip():
        try:
            expires = date.fromisoformat(str(licence_expires_on).strip())
        except ValueError as exc:
            # Refused rather than stored as today. A licence expiry that is wrong is
            # worse than one that is absent: the roster would clear a pilot who is not.
            raise ValueError(f"Licence expiry must be YYYY-MM-DD: {exc}") from exc
    with _session() as db:
        pilot = PilotProfile(
            organization_id=int(organization_id),
            display_name=str(display_name).strip(),
            licence_number=str(licence_number).strip(),
            licence_expires_on=expires,
        )
        db.add(pilot)
        db.commit()
        db.refresh(pilot)
        return {
            "id": pilot.id,
            "display_name": pilot.display_name,
            "licence_number": pilot.licence_number,
            "licence_expires_on": pilot.licence_expires_on.isoformat() if pilot.licence_expires_on else "",
        }


def log_maintenance(aircraft_id: int, kind: str, description: str = "",
                    performed_by: str = "") -> dict[str, Any]:
    """Record maintenance, and reset the aircraft's service clock.

    Both halves matter. A maintenance record that does not move hours_at_last_service
    leaves the aircraft permanently overdue, which trains an operator to ignore the
    warning.
    """
    from services.api.models import Aircraft, Maintenance

    if not str(kind).strip():
        raise ValueError("Maintenance needs a kind, so the record can be read later.")
    with _session() as db:
        aircraft = db.get(Aircraft, int(aircraft_id))
        if aircraft is None:
            raise ValueError(f"No aircraft with id {aircraft_id}.")
        record = Maintenance(
            aircraft_id=aircraft.id,
            kind=str(kind).strip(),
            description=str(description).strip(),
            hours_at_service=float(aircraft.flight_hours or 0.0),
            performed_by=str(performed_by).strip(),
            performed_at=_utc_now(),
        )
        aircraft.hours_at_last_service = float(aircraft.flight_hours or 0.0)
        aircraft.last_service_at = record.performed_at
        db.add(record)
        db.commit()
        db.refresh(record)
        return {
            "id": record.id,
            "aircraft_id": aircraft.id,
            "kind": record.kind,
            "hours_at_service": record.hours_at_service,
            "performed_at": record.performed_at.isoformat(),
        }


def fleet_status(organization_id: int) -> dict[str, Any]:
    """What is available, what is flying, and what is due for service."""
    from sqlalchemy import select

    from services.api.models import Aircraft, Battery, PilotProfile

    with _session() as db:
        aircraft = list(db.scalars(select(Aircraft).where(
            Aircraft.organization_id == int(organization_id))))
        batteries = list(db.scalars(select(Battery).where(
            Battery.organization_id == int(organization_id))))
        pilots = list(db.scalars(select(PilotProfile).where(
            PilotProfile.organization_id == int(organization_id))))

        due = [
            a.name for a in aircraft
            if a.service_interval_hours
            and float(a.flight_hours or 0) - float(a.hours_at_last_service or 0)
            >= float(a.service_interval_hours)
        ]
        return {
            "aircraft": len(aircraft),
            "batteries": len(batteries),
            "retired_batteries": sum(1 for b in batteries if b.retired),
            "pilots": len(pilots),
            "service_due": due,
        }


def list_fleet(organization_id: int) -> dict[str, Any]:
    """The fleet itself, row by row, rather than counts of it.

    fleet_status() answers "how many" -- four aircraft, three batteries, two overdue.
    That is the right answer for a dashboard tile and useless for the Fleet screen,
    which has to show which aircraft, which battery, and what is wrong with it. Having
    only the counts is why that screen was drawn from four invented aircraft, two
    invented pilots and three invented batteries with cycle counts and temperatures.

    Service due is computed here rather than stored, for the same reason it is computed
    in fleet_status(): an aircraft becomes overdue by flying, not by being edited, so a
    stored flag would be stale exactly when it matters.
    """
    from sqlalchemy import select

    from services.api.models import Aircraft, Battery, Maintenance, PilotProfile

    with _session() as db:
        aircraft = list(db.scalars(select(Aircraft).where(
            Aircraft.organization_id == int(organization_id)).order_by(Aircraft.name)))
        batteries = list(db.scalars(select(Battery).where(
            Battery.organization_id == int(organization_id)).order_by(Battery.serial_number)))
        pilots = list(db.scalars(select(PilotProfile).where(
            PilotProfile.organization_id == int(organization_id)).order_by(
                PilotProfile.display_name)))

        ids = [a.id for a in aircraft]
        recent = list(db.scalars(
            select(Maintenance).where(Maintenance.aircraft_id.in_(ids))
            .order_by(Maintenance.performed_at.desc()).limit(50)
        )) if ids else []
        names = {a.id: a.name for a in aircraft}

        def hours_to_service(a: Any) -> float | None:
            if not a.service_interval_hours:
                return None
            flown = float(a.flight_hours or 0.0) - float(a.hours_at_last_service or 0.0)
            return round(float(a.service_interval_hours) - flown, 1)

        return {
            "aircraft": [{
                "id": a.id,
                "name": a.name,
                "model": a.model or "",
                "serial_number": a.serial_number or "",
                "firmware": a.firmware or "",
                "flight_hours": round(float(a.flight_hours or 0.0), 1),
                "flight_count": int(a.flight_count or 0),
                "status": a.status or "unknown",
                # None means no interval is configured, which is not the same as
                # "not due" and must not be rendered as a comfortable number.
                "hours_to_service": hours_to_service(a),
            } for a in aircraft],
            "batteries": [{
                "id": b.id,
                "serial_number": b.serial_number or "",
                "capacity_mah": int(b.capacity_mah or 0),
                "cycle_count": int(b.cycle_count or 0),
                "cycle_limit": int(b.cycle_limit or 0),
                "health_pct": None if b.health_pct is None else round(float(b.health_pct), 1),
                "retired": bool(b.retired),
                # Past its rated cycles but not yet retired: the state that puts a pack
                # in an aircraft when it should be on a shelf.
                "over_cycle_limit": bool(
                    b.cycle_limit and int(b.cycle_count or 0) >= int(b.cycle_limit)),
            } for b in batteries],
            "pilots": [{
                "id": p.id,
                "display_name": p.display_name or "",
                "licence_number": p.licence_number or "",
                "licence_expires_on": str(p.licence_expires_on or "") or None,
                "medical_expires_on": str(p.medical_expires_on or "") or None,
                "flight_hours": round(float(p.flight_hours or 0.0), 1),
            } for p in pilots],
            "maintenance": [{
                "aircraft": names.get(m.aircraft_id, f"#{m.aircraft_id}"),
                "kind": m.kind,
                "description": m.description or "",
                "hours_at_service": round(float(m.hours_at_service or 0.0), 1),
                "performed_at": str(m.performed_at or ""),
            } for m in recent],
        }


def assign_mission(aircraft_id: int, mission_name: str) -> dict[str, Any]:
    """Note which aircraft is flying which mission.

    Recorded as a maintenance-style log entry rather than invented as a new table: the
    schema has no assignment concept, and adding one here would put the desktop app and
    the service on different schemas.
    """
    return log_maintenance(
        aircraft_id,
        kind="assignment",
        description=f"Assigned to mission: {mission_name}",
    )


# ------------------------------------------------------------------- sharing


def create_share_link(project_id: int, note: str = "", allow_download: bool = False,
                      include_defects: bool = True) -> dict[str, Any]:
    """Issue a share link and return the token ONCE.

    Only the hash is stored, so the token cannot be recovered later -- the same property
    the service enforces. A link that can be re-read from the database is a credential
    sitting in a backup.
    """
    from services.api.models import ShareLink

    token = secrets.token_urlsafe(TOKEN_BYTES)
    with _session() as db:
        link = ShareLink(
            project_id=int(project_id),
            token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            prefix=token[:8],
            note=str(note).strip(),
            allow_download=bool(allow_download),
            include_defects=bool(include_defects),
        )
        db.add(link)
        db.commit()
        db.refresh(link)
        return {
            "id": link.id,
            "prefix": link.prefix,
            "token": token,          # shown once; never stored in this form
            "allow_download": link.allow_download,
            "note": link.note,
        }


def list_share_links(project_id: int) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from services.api.models import ShareLink

    with _session() as db:
        links = db.scalars(select(ShareLink).where(ShareLink.project_id == int(project_id)))
        return [
            {"id": l.id, "prefix": l.prefix, "revoked": bool(l.revoked), "note": l.note or ""}
            for l in links
        ]


def revoke_share_link(share_id: int) -> dict[str, Any]:
    from services.api.models import ShareLink

    with _session() as db:
        link = db.get(ShareLink, int(share_id))
        if link is None:
            raise ValueError(f"No share link with id {share_id}.")
        link.revoked = True
        db.commit()
        return {"id": link.id, "revoked": True}


# ------------------------------------------------------------------ webhooks


def add_webhook(organization_id: int, url: str, events: list[str] | None = None,
                description: str = "") -> dict[str, Any]:
    """Register a webhook and return its signing secret once."""
    import json

    from services.api.models import Webhook

    if not str(url).strip().startswith(("http://", "https://")):
        raise ValueError("A webhook URL must be http or https.")
    secret = secrets.token_urlsafe(TOKEN_BYTES)
    with _session() as db:
        hook = Webhook(
            organization_id=int(organization_id),
            url=str(url).strip(),
            events_json=json.dumps(list(events or ["*"])),
            description=str(description).strip(),
            secret_hash=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            secret_prefix=secret[:8],
            active=True,
        )
        db.add(hook)
        db.commit()
        db.refresh(hook)
        return {
            "id": hook.id,
            "url": hook.url,
            "events": list(events or ["*"]),
            "secret": secret,        # shown once
            "secret_prefix": hook.secret_prefix,
        }


def list_webhooks(organization_id: int) -> list[dict[str, Any]]:
    import json

    from sqlalchemy import select

    from services.api.models import Webhook

    with _session() as db:
        hooks = db.scalars(select(Webhook).where(
            Webhook.organization_id == int(organization_id)))
        return [
            {
                "id": h.id,
                "url": h.url,
                "events": json.loads(h.events_json or "[]"),
                "active": bool(h.active),
                "delivery_count": h.delivery_count,
                "failure_count": h.failure_count,
            }
            for h in hooks
        ]


# ------------------------------------------------------------------- reports


def generate_report(project_id: str, *, title: str = "Inspection report",
                    report_type: str = "standard", author: str = "") -> dict[str, Any]:
    """Build a report from what the project actually contains.

    core.report_engine resolves the project root itself and REFUSES when the inputs are
    not there, rather than emitting a document with empty sections -- a report with no
    findings and no explanation is indistinguishable from a survey that found nothing.
    That refusal is passed straight through so the user reads the checklist.
    """
    from core.report_engine import ReportConfig, generate_report as build

    config = ReportConfig(
        project_id=str(project_id),
        title=str(title),
        report_type=str(report_type),
        author=str(author),
    )
    result = build(config)
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def report_readiness(project_id: str, *, report_type: str = "standard") -> dict[str, Any]:
    """What the report still needs, before anyone presses Generate."""
    from core.report_engine import ReportConfig, validate_report_readiness

    readiness = validate_report_readiness(
        ReportConfig(project_id=str(project_id), report_type=str(report_type))
    )
    return {
        "ok": bool(getattr(readiness, "ok", False)),
        "missing": list(getattr(readiness, "missing", []) or []),
    }


# -------------------------------------------------------------------- review


def review_finding(project_root: str | Path, annotation_id: str, decision: str,
                   reviewer: str = "operator") -> dict[str, Any]:
    """Accept, reject or flag a finding.

    The decision moves the annotation's STATUS and records who moved it. The model's own
    claim -- geometry, label, confidence, model key and digest -- is left untouched,
    because ai.human_validation requires a prediction to survive alongside the human
    answer rather than being overwritten by it. A reviewer's disagreement is evidence
    about the model, and erasing what the model said destroys it.
    """
    from core.annotations import update_annotation

    status = {
        "accept": "resolved",
        "reject": "dismissed",
        "flag": "in_review",
    }.get(str(decision).lower())
    if status is None:
        raise ValueError(f"Unknown review decision: {decision!r}. Use accept, reject or flag.")

    # The reviewer goes in the note, not a reviewed_by field: Annotation has no such
    # field and update_annotation silently drops unknown keys, so a reviewer recorded
    # that way would disappear without an error and the audit trail would read as though
    # nobody had looked at it.
    stamp = f"{decision} by {reviewer} at {_utc_now().isoformat(timespec='seconds')}"
    updated = update_annotation(
        Path(project_root),
        str(annotation_id),
        {"status": status, "note": stamp},
    )
    if updated is None:
        raise ValueError(f"No annotation with id {annotation_id!r} in this project.")
    return updated.to_dict() if hasattr(updated, "to_dict") else dict(updated)


# ------------------------------------------------------------------- plugins


def list_plugins() -> list[dict[str, Any]]:
    """What the plugin registry has actually loaded."""
    from sdk.plugins import PluginRegistry

    registry = PluginRegistry()
    discovered = getattr(registry, "specs", None) or getattr(registry, "plugins", None) or []
    out = []
    for spec in discovered:
        out.append({
            "name": getattr(spec, "name", str(spec)),
            "kind": str(getattr(spec, "kind", "")),
            "version": getattr(spec, "version", ""),
        })
    return out
