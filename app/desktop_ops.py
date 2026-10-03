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
import threading
from pathlib import Path
from typing import Any

TOKEN_BYTES = 24
_INIT_LOCK = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _session():
    """A session on the service's database.

    Imported lazily: the desktop app must start on a machine where SQLAlchemy is absent,
    and the failure should arrive when someone presses a fleet button rather than at
    launch.
    """
    from services.api.db import get_session_factory, init_db

    # Serialised: pywebview runs each Api call on its own thread, and the Fleet screen
    # asks for the list while a Save is in flight. Two concurrent create_all() calls on
    # a new SQLite file race, and the loser failed the Save with "table users already
    # exists".
    with _INIT_LOCK:
        init_db()
    return get_session_factory()()


# The organisation a single-user desktop install records against. The cockpit sends it
# when no organisation has been chosen.
LOCAL_ORGANIZATION_ID = 1


def _organization(db, organization_id: int) -> int:
    """The organisation id to record against, creating the local one on first use.

    Organisations were only ever created by the web service's registration endpoint, so
    on a fresh database -- every new desktop install -- the first fleet record failed
    with "FOREIGN KEY constraint failed" and the Fleet screen could never hold anything.

    Only the local default is created. Any other id that does not exist is refused by
    name, because recording an aircraft against an organisation nobody made would
    invent one.
    """
    from services.api.models import Organization

    wanted = int(organization_id)
    if db.get(Organization, wanted) is not None:
        return wanted
    if wanted != LOCAL_ORGANIZATION_ID:
        raise ValueError(f"No organisation with id {wanted}.")
    db.add(Organization(id=LOCAL_ORGANIZATION_ID, name="Local organisation",
                        slug="local-desktop"))
    db.flush()
    return wanted


# --------------------------------------------------------------------- fleet


def add_aircraft(organization_id: int, name: str, model: str = "", serial: str = "") -> dict[str, Any]:
    """Register an aircraft against an organisation."""
    from services.api.models import Aircraft

    if not str(name).strip():
        raise ValueError("An aircraft needs a name.")
    with _session() as db:
        aircraft = Aircraft(
            organization_id=_organization(db, organization_id),
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
    capacity = _whole_number(capacity_mah, "Capacity (mAh)")
    limit = _whole_number(cycle_limit, "Cycle limit")
    with _session() as db:
        battery = Battery(
            organization_id=_organization(db, organization_id),
            serial_number=str(serial).strip(),
            capacity_mah=capacity,
            cycle_limit=limit,
            # Not measured yet. Left empty rather than 100: nobody has tested this pack.
            health_pct=None,
        )
        db.add(battery)
        db.commit()
        db.refresh(battery)
        return _battery_row(battery)


def _whole_number(value: Any, label: str) -> int:
    """A count that cannot be negative, from a prompt's text or a number.

    Blank means 0 (not recorded). A negative cycle limit made a new 0-cycle pack read
    "past cycle limit"; a negative capacity is not a battery.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return 0
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a whole number, not {text!r}.") from None
    if number != int(number):
        raise ValueError(f"{label} must be a whole number, not {text!r}.")
    if number < 0:
        raise ValueError(f"{label} cannot be negative (got {text}).")
    return int(number)


def _battery_row(battery) -> dict[str, Any]:
    return {
        "id": battery.id,
        "serial_number": battery.serial_number,
        "capacity_mah": battery.capacity_mah,
        "cycle_count": int(battery.cycle_count or 0),
        "cycle_limit": int(battery.cycle_limit or 0),
        "health_pct": None if battery.health_pct is None else round(float(battery.health_pct), 1),
    }


def log_battery_cycle(organization_id: int, battery_id: Any, cycles: Any = 1,
                      health_pct: Any = "") -> dict[str, Any]:
    """Count charge cycles on a pack, and record its health if someone measured it.

    The desktop could register a pack but never move its cycle count or enter a
    health reading, so the Cycles column stayed 0 and Health had nothing to show.
    Health is optional here and is only ever what the operator typed: blank leaves
    the stored value (or its absence) as it was. A reading outside 0-100 is refused,
    not clamped, because a clamped number is not what was measured.
    """
    from services.api.models import Battery

    try:
        number = int(battery_id)
    except (TypeError, ValueError):
        raise ValueError(f"{battery_id!r} is not a battery id; select a battery in the "
                         "fleet list.") from None
    count = _whole_number(cycles, "Cycles")
    reading: float | None = None
    text = str(health_pct if health_pct is not None else "").strip().rstrip("%").strip()
    if text:
        try:
            reading = float(text)
        except ValueError:
            raise ValueError(f"Health must be a percentage, not {text!r}.") from None
        if not 0.0 <= reading <= 100.0:
            raise ValueError(f"Health must be between 0 and 100 % (got {text}).")
    if count == 0 and reading is None:
        raise ValueError("Nothing to record: give a number of cycles or a measured health.")
    with _session() as db:
        battery = db.get(Battery, number)
        if battery is None or int(battery.organization_id) != int(organization_id):
            raise ValueError(f"No battery with id {number} in this fleet.")
        battery.cycle_count = int(battery.cycle_count or 0) + count
        if count:
            battery.last_used_at = _utc_now()
        if reading is not None:
            battery.health_pct = reading
        db.commit()
        db.refresh(battery)
        return _battery_row(battery)


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
            organization_id=_organization(db, organization_id),
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


ASSIGNMENT_KIND = "assignment"
_ASSIGNMENT_PREFIX = "Assigned to mission: "


def _aircraft_id(aircraft_id: Any) -> int:
    try:
        return int(aircraft_id)
    except (TypeError, ValueError):
        raise ValueError(f"{aircraft_id!r} is not an aircraft id; select an aircraft in "
                         "the fleet list.") from None


def _performer(performed_by: Any) -> tuple[int | None, str]:
    """The user id column holds a user id; a name goes into the description.

    "operator" used to be written into the integer performed_by column, which SQLite
    stores without complaint and PostgreSQL rejects.
    """
    text = str(performed_by or "").strip()
    return (int(text), "") if text.isdigit() else (None, text)


def log_maintenance(aircraft_id: int, kind: str, description: str = "",
                    performed_by: str = "", *, resets_service_clock: bool = True
                    ) -> dict[str, Any]:
    """Record maintenance, and reset the aircraft's service clock.

    Both halves matter. A maintenance record that does not move hours_at_last_service
    leaves the aircraft permanently overdue, which trains an operator to ignore the
    warning. A record that is not service (an assignment) must not move it: that would
    mark an overdue airframe as serviced.
    """
    from services.api.models import Aircraft, Maintenance

    if not str(kind).strip():
        raise ValueError("Maintenance needs a kind, so the record can be read later.")
    user_id, name = _performer(performed_by)
    detail = str(description).strip()
    if name:
        detail = f"{detail} (by {name})" if detail else f"by {name}"
    with _session() as db:
        aircraft = db.get(Aircraft, _aircraft_id(aircraft_id))
        if aircraft is None:
            raise ValueError(f"No aircraft with id {aircraft_id}.")
        record = Maintenance(
            aircraft_id=aircraft.id,
            kind=str(kind).strip(),
            description=detail,
            hours_at_service=float(aircraft.flight_hours or 0.0),
            performed_by=user_id,
            performed_at=_utc_now(),
        )
        if resets_service_clock:
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
        # The latest assignment per aircraft, so the screen can say what it is flying.
        assigned: dict[int, str] = {}
        if ids:
            for m in db.scalars(select(Maintenance).where(
                    Maintenance.aircraft_id.in_(ids), Maintenance.kind == ASSIGNMENT_KIND)
                    .order_by(Maintenance.performed_at.desc(), Maintenance.id.desc())):
                text = str(m.description or "")
                if m.aircraft_id not in assigned and text.startswith(_ASSIGNMENT_PREFIX):
                    assigned[m.aircraft_id] = text[len(_ASSIGNMENT_PREFIX):]

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
                "assigned_mission": assigned.get(a.id),
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
    mission = str(mission_name or "").strip()
    if not mission:
        raise ValueError("Name the mission the aircraft is assigned to.")
    record = log_maintenance(
        aircraft_id,
        kind=ASSIGNMENT_KIND,
        description=f"{_ASSIGNMENT_PREFIX}{mission}",
        # Assigning an aircraft is not servicing it: the clock that says it is overdue
        # must keep running.
        resets_service_clock=False,
    )
    return {**record, "mission": mission}


FLEET_KINDS = ("aircraft", "battery", "pilot")


def remove_fleet_record(organization_id: int, kind: str, record_id: Any) -> dict[str, Any]:
    """Delete one aircraft, battery or pilot from this organisation's register.

    The register could only grow: a mistyped aircraft, a pack that was sold, or rows a
    test tool once wrote into the user's database stayed on the Fleet screen for good.
    An aircraft goes with its maintenance log, which describes nothing once the airframe
    is gone. A record belonging to another organisation is refused, not deleted.
    """
    from sqlalchemy import delete, func, select

    from services.api.models import Aircraft, Battery, Maintenance, PilotProfile

    models = {"aircraft": Aircraft, "battery": Battery, "pilot": PilotProfile}
    wanted = str(kind or "").strip().lower()
    if wanted not in models:
        raise ValueError(f"Choose what to remove: {', '.join(FLEET_KINDS)} (got {kind!r}).")
    try:
        number = int(record_id)
    except (TypeError, ValueError):
        raise ValueError(f"{record_id!r} is not a {wanted} id; select one in the fleet "
                         "list.") from None
    with _session() as db:
        row = db.get(models[wanted], number)
        if row is None or int(row.organization_id) != int(organization_id):
            raise ValueError(f"No {wanted} with id {number} in this fleet.")
        label = (getattr(row, "name", None) or getattr(row, "serial_number", None)
                 or getattr(row, "display_name", None) or f"#{number}")
        removed_log = 0
        if wanted == "aircraft":
            removed_log = int(db.scalar(select(func.count()).select_from(Maintenance).where(
                Maintenance.aircraft_id == number)) or 0)
            db.execute(delete(Maintenance).where(Maintenance.aircraft_id == number))
        db.delete(row)
        db.commit()
        return {"kind": wanted, "id": number, "label": str(label),
                "removed_maintenance": removed_log}


def export_fleet(organization_id: int, directory: str | Path) -> dict[str, Any]:
    """Write the fleet register as CSV files plus one JSON file, and say where.

    Every maintenance record is written, not the latest fifty the screen shows: an
    export is what an operator hands to an auditor.
    """
    import csv
    import json

    from sqlalchemy import select

    from services.api.models import Maintenance

    register = list_fleet(organization_id)
    ids = [a["id"] for a in register["aircraft"]]
    names = {a["id"]: a["name"] for a in register["aircraft"]}
    log: list[dict[str, Any]] = []
    if ids:
        with _session() as db:
            log = [{
                "aircraft_id": m.aircraft_id,
                "aircraft": names.get(m.aircraft_id, f"#{m.aircraft_id}"),
                "kind": m.kind,
                "description": m.description or "",
                "hours_at_service": round(float(m.hours_at_service or 0.0), 1),
                "performed_at": str(m.performed_at or ""),
            } for m in db.scalars(select(Maintenance).where(Maintenance.aircraft_id.in_(ids))
                                  .order_by(Maintenance.performed_at, Maintenance.id))]
    register["maintenance"] = log
    sections = ("aircraft", "batteries", "pilots", "maintenance")

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for key in sections:
        rows = register[key]
        path = out / f"{key}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            columns = list(rows[0].keys()) if rows else []
            writer.writerow(columns)
            writer.writerows([[row[c] for c in columns] for row in rows])
        files[key] = str(path)
    summary = out / "fleet_register.json"
    summary.write_text(json.dumps({
        "organization_id": int(organization_id),
        "exported_at": _utc_now().isoformat(),
        **{key: register[key] for key in sections},
    }, indent=2, default=str), encoding="utf-8")
    files["json"] = str(summary)
    return {"directory": str(out), "files": files,
            "counts": {key: len(register[key]) for key in sections}}


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
            organization_id=_organization(db, organization_id),
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
                    report_type: str = "standard", author: str = "",
                    findings_root: str | Path = "", findings_project_id: str = "") -> dict[str, Any]:
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
        findings_root=str(findings_root or ""),
        findings_project_id=str(findings_project_id or ""),
    )
    result = build(config)
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def report_readiness(project_id: str, *, report_type: str = "standard") -> dict[str, Any]:
    """What the report still needs, before anyone presses Generate."""
    from core.report_engine import ReportConfig, validate_report_readiness

    readiness = validate_report_readiness(
        ReportConfig(project_id=str(project_id), report_type=str(report_type))
    )
    # "ready", not "ok": spread into Api.ok() an "ok" key overrode the call's own, so a
    # refusal reached the UI as an error with no message. Warnings and notes were dropped.
    return {
        "ready": bool(getattr(readiness, "ok", False)),
        "missing": list(getattr(readiness, "missing", []) or []),
        "warnings": list(getattr(readiness, "warnings", []) or []),
        "notes": list(getattr(readiness, "notes", []) or []),
    }


# -------------------------------------------------------------------- review


def review_finding(project_root: str | Path, annotation_id: str, decision: str,
                   reviewer: str = "operator", comment: str = "") -> dict[str, Any]:
    """Accept, reject or flag a finding.

    The decision moves the annotation's STATUS and appends to its review history. The
    model's own claim -- geometry, label, confidence, model key and digest -- is left
    untouched, because ai.human_validation requires a prediction to survive alongside the
    human answer rather than being overwritten by it. The inspector's note is left alone
    too: it is their observation about the defect, not a log line.

    History is appended, never replaced, so a second reviewer cannot erase the first.
    A rejection takes the finding out of reports (include_in_report False), and an
    acceptance puts it back: a rejected detection printed in a report would be the
    model's claim overriding the person who looked at it.
    """
    from core.annotations import get_annotation, update_annotation

    key = str(decision).lower()
    status = {
        "accept": "resolved",
        "reject": "dismissed",
        "flag": "in_review",
    }.get(key)
    if status is None:
        raise ValueError(f"Unknown review decision: {decision!r}. Use accept, reject or flag.")

    root = Path(project_root)
    current = get_annotation(root, str(annotation_id))
    if current is None:
        raise ValueError(f"No annotation with id {annotation_id!r} in this project.")

    entry = {
        "decision": key,
        "reviewer": str(reviewer or "operator"),
        "at": _utc_now().isoformat(timespec="seconds"),
        "previous_status": current.status,
        "status": status,
    }
    if str(comment or "").strip():
        entry["comment"] = str(comment).strip()
    patch: dict[str, Any] = {
        "status": status,
        "review_history": list(current.review_history or []) + [entry],
    }
    if key == "reject":
        patch["include_in_report"] = False
    elif key == "accept":
        patch["include_in_report"] = True
    updated = update_annotation(root, str(annotation_id), patch)
    if updated is None:
        raise ValueError(f"No annotation with id {annotation_id!r} in this project.")
    return updated.to_dict() if hasattr(updated, "to_dict") else dict(updated)


# ------------------------------------------------------------------ findings
#
# Model detections used to stop at masks and overlays on disk. Nothing wrote them into
# the annotation store, so the Findings and Review panels could only ever show what a
# person had drawn, and "review a model's findings" had nothing to review.
#
# The contract matches the REST pre-label path: only a detector that ran and gave each
# region its own score becomes a finding; it arrives in review, out of the report, with
# the model key, the digest of the file that ran and its confidence fixed on the record.
# Heuristic regions are never turned into findings -- they have no score and no model,
# and a finding list full of them is what made solar look like it worked.


def _is_model_output(model_used: str) -> bool:
    return str(model_used or "").startswith("onnx:")


def findings_from_detections(project_root: str | Path, project_id: str, image_path: str,
                             *, model_used: str, model_key: str, model_sha256: str,
                             detections: list[dict[str, Any]],
                             run_tag: str = "", tags: list[str] | None = None,
                             note_suffix: str = "") -> list[dict[str, Any]]:
    """Write one reviewable finding per model detection; refuse unattributable ones.

    ``tags`` are added to every finding (licence use, "suggestion" for zero-shot
    output); ``note_suffix`` is appended to every note (the model's scope caveat). A
    detection may carry a ``condition`` from a second-stage classifier; it is written
    into that finding's note and tags with its own probability and model digest.
    """
    from core.annotations import create_annotation

    if not _is_model_output(model_used) or not model_key or not model_sha256:
        raise ValueError(
            f"Detections from {model_used or 'an unknown source'} carry no model identity; "
            "only a trained model's scored detections become findings.")
    created = []
    for hit in detections:
        confidence = hit.get("confidence")
        if confidence is None:
            raise ValueError("A model detection without a confidence cannot become a finding.")
        x1, y1, x2, y2 = (int(v) for v in hit["bbox_xyxy"])
        if x2 <= x1 or y2 <= y1:
            continue
        geometry = {"type": "Polygon", "coordinates": [[
            [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]}
        finding_tags = (["model-prelabel"] + ([run_tag] if run_tag else [])
                        + [str(t) for t in (tags or []) if t])
        note = (f"Machine pre-label from {model_used}, awaiting review. Severity is a "
                "review default, not an assessment.")
        condition = hit.get("condition")
        if isinstance(condition, dict) and condition.get("label"):
            finding_tags.append(f"condition:{condition['label']}")
            note += (f" Condition suggested by {condition.get('model_key', '')} "
                     f"(sha256 {str(condition.get('model_sha256', ''))[:16]}...): "
                     f"{condition['label']}, p={float(condition.get('probability', 0)):.3f} "
                     "over this asset's valid conditions.")
        if hit.get("mask_area_px") is not None:
            note += (f" Model mask: {int(hit['mask_area_px'])} px in the image; the finding "
                     "keeps the box.")
        if note_suffix:
            note += " " + note_suffix
        annotation = create_annotation(
            Path(project_root), str(project_id), "image", str(image_path), "rectangle",
            geometry, str(hit["label"]), "info", "in_review",
            note=note,
            include_in_report=False, created_by="model",
            model_used=model_used, model_key=model_key, model_sha256=model_sha256,
            confidence=float(confidence), tags=finding_tags,
        )
        created.append(annotation.to_dict())
    return created


def findings_from_pipeline(project_root: str | Path, project_id: str,
                           summary: dict[str, Any]) -> dict[str, Any]:
    """Turn a finished pipeline run's model detections into reviewable findings."""
    run_tag = "run " + str(Path(str(summary.get("artifacts", {}).get("run_dir", ""))).name or "")
    created = 0
    heuristic_regions = 0
    for row in summary.get("images", []) or []:
        image = str(row.get("image_path") or row.get("image") or "")
        for prefix in ("structural", "solar"):
            detections = row.get(f"{prefix}_detections") or []
            used = str(row.get(f"{prefix}_model_used", ""))
            if not detections:
                continue
            if not _is_model_output(used):
                heuristic_regions += len(detections)
                continue
            created += len(findings_from_detections(
                project_root, project_id, image, model_used=used,
                model_key=str(row.get(f"{prefix}_model_key", "")),
                model_sha256=str(row.get(f"{prefix}_model_sha256", "")),
                detections=detections, run_tag=run_tag.strip()))
    return {"findings_created": created,
            "heuristic_regions_not_recorded": heuristic_regions}


def prelabel_image(project_root: str | Path, project_id: str, image_path: str,
                   model_key: str = "structural_multiclass_detector") -> dict[str, Any]:
    """Run an installed detector on one image and store its detections as findings."""
    from core.detection import DETECTION_KINDS, load_image
    from core.model_runner import ModelRunRefused, run_registered_model
    from core.models import get_model_spec

    spec = get_model_spec(model_key)
    if spec is None:
        raise ValueError(f"{model_key!r} is not in the model registry.")
    if spec.kind not in DETECTION_KINDS:
        raise ValueError(
            f"{model_key} is a {spec.kind}; only detectors give each region its own "
            "confidence, which a pre-label needs. Run it from Run Model instead.")
    target = Path(image_path)
    if not target.is_file():
        raise ValueError(f"No such image: {image_path}")
    try:
        # Pre-labels go to review before any report, so they use the registry's
        # measured pre-label threshold where one is declared.
        result = run_registered_model(load_image(target), model_key,
                                      score_threshold=spec.prelabel_score_threshold)
    except ModelRunRefused as exc:
        raise ValueError(str(exc)) from exc
    created = findings_from_model_result(project_root, project_id, str(target.resolve()),
                                         result)
    return {"model_key": result.model_key, "model_sha256": result.model_sha256,
            "model_used": result.model_used, "findings": created,
            "count": len(created), "scope": result.scope,
            "usage_note": result.output.get("usage_note", "")}


def findings_from_model_result(project_root: str | Path, project_id: str, image_path: str,
                               result: Any) -> list[dict[str, Any]]:
    """In-review findings from a ModelRunResult, tagged with the row's licence use and
    output semantics, with the row's result note (scope caveat) on every finding."""
    from core.models import load_registry

    row = (load_registry().get("models") or {}).get(result.model_key) or {}
    tags = []
    use = str((row.get("licence") or {}).get("use") or "")
    if use:
        tags.append(f"licence:{use}")
    if str(row.get("output_semantics") or "") == "suggestion":
        tags.append("suggestion")
    return findings_from_detections(
        project_root, project_id, image_path,
        model_used=result.model_used, model_key=result.model_key,
        model_sha256=result.model_sha256, detections=result.output["detections"],
        tags=tags, note_suffix=str(row.get("result_note") or ""))


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
