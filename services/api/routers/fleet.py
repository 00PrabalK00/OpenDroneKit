"""Fleet, batteries, pilots and maintenance.

Fleet records exist to answer operational questions before a flight rather than after
an incident: is this aircraft due for service, is this battery past its cycle limit, is
this pilot's certification still current. So the endpoints compute and surface those
answers rather than only storing the fields and leaving the arithmetic to whoever reads
the list.

Nothing here grounds an aircraft on its own. The system reports what it knows and the
operator decides; a maintenance flag is information, not an interlock.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import Aircraft, Battery, Maintenance, PilotProfile, Role
from ..security import CurrentUser, require_role

router = APIRouter(tags=["fleet"])

# A lithium pack is usually considered end-of-life around here. Configurable per
# battery, because manufacturers differ and an operator may be stricter.
DEFAULT_CYCLE_LIMIT = 300
# Health below this is worth flagging even when the cycle count looks fine.
HEALTH_WARNING_PCT = 80.0


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise a stored timestamp, which SQLite returns naive and PostgreSQL aware."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class AircraftCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model: str = ""
    serial_number: str = ""
    firmware: str = ""
    service_interval_hours: float = 100.0


class AircraftOut(BaseModel):
    id: int
    organization_id: int
    name: str
    model: str
    serial_number: str
    firmware: str
    flight_hours: float
    flight_count: int
    service_interval_hours: float
    hours_since_service: float
    service_due: bool
    hours_until_service: float
    status: str
    last_service_at: datetime | None
    created_at: datetime


class BatteryCreate(BaseModel):
    serial_number: str = Field(min_length=1, max_length=120)
    capacity_mah: int = 0
    cycle_limit: int = DEFAULT_CYCLE_LIMIT


class BatteryOut(BaseModel):
    id: int
    organization_id: int
    serial_number: str
    capacity_mah: int
    cycle_count: int
    cycle_limit: int
    health_pct: float
    retired: bool
    past_cycle_limit: bool
    warnings: list[str]
    last_used_at: datetime | None


class PilotCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    licence_number: str = ""
    licence_expires_on: date | None = None
    medical_expires_on: date | None = None


class PilotOut(BaseModel):
    id: int
    organization_id: int
    display_name: str
    licence_number: str
    licence_expires_on: date | None
    medical_expires_on: date | None
    flight_hours: float
    current: bool
    warnings: list[str]


class MaintenanceCreate(BaseModel):
    aircraft_id: int
    kind: str = "scheduled"
    description: str = ""
    hours_at_service: float | None = None


def _aircraft_out(aircraft: Aircraft) -> AircraftOut:
    since_service = max(0.0, aircraft.flight_hours - aircraft.hours_at_last_service)
    interval = max(1.0, aircraft.service_interval_hours)
    return AircraftOut(
        id=aircraft.id, organization_id=aircraft.organization_id, name=aircraft.name,
        model=aircraft.model, serial_number=aircraft.serial_number,
        firmware=aircraft.firmware, flight_hours=round(aircraft.flight_hours, 2),
        flight_count=aircraft.flight_count,
        service_interval_hours=aircraft.service_interval_hours,
        hours_since_service=round(since_service, 2),
        service_due=since_service >= interval,
        hours_until_service=round(interval - since_service, 2),
        status=aircraft.status, last_service_at=aircraft.last_service_at,
        created_at=aircraft.created_at,
    )


def _battery_out(battery: Battery) -> BatteryOut:
    warnings: list[str] = []
    past_limit = battery.cycle_limit > 0 and battery.cycle_count >= battery.cycle_limit
    if past_limit:
        warnings.append(
            f"{battery.cycle_count} cycles against a limit of {battery.cycle_limit}."
        )
    if battery.health_pct < HEALTH_WARNING_PCT:
        warnings.append(f"Health is {battery.health_pct:.0f}%.")
    if battery.retired:
        warnings.append("Marked retired; it should not be flown.")

    return BatteryOut(
        id=battery.id, organization_id=battery.organization_id,
        serial_number=battery.serial_number, capacity_mah=battery.capacity_mah,
        cycle_count=battery.cycle_count, cycle_limit=battery.cycle_limit,
        health_pct=round(battery.health_pct, 1), retired=battery.retired,
        past_cycle_limit=past_limit, warnings=warnings,
        last_used_at=battery.last_used_at,
    )


def _pilot_out(pilot: PilotProfile, today: date | None = None) -> PilotOut:
    today = today or datetime.now(timezone.utc).date()
    warnings: list[str] = []

    for label, expiry in (("Licence", pilot.licence_expires_on),
                          ("Medical", pilot.medical_expires_on)):
        if expiry is None:
            continue
        if expiry < today:
            warnings.append(f"{label} expired on {expiry.isoformat()}.")
        elif expiry - today <= timedelta(days=30):
            # A month's notice is the difference between renewing calmly and
            # discovering it on the morning of a job.
            warnings.append(f"{label} expires on {expiry.isoformat()}.")

    expired = any("expired" in warning for warning in warnings)
    return PilotOut(
        id=pilot.id, organization_id=pilot.organization_id,
        display_name=pilot.display_name, licence_number=pilot.licence_number,
        licence_expires_on=pilot.licence_expires_on,
        medical_expires_on=pilot.medical_expires_on,
        flight_hours=round(pilot.flight_hours, 2),
        current=not expired, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# aircraft
# ---------------------------------------------------------------------------


@router.get("/organizations/{organization_id}/aircraft", response_model=list[AircraftOut])
def list_aircraft(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[AircraftOut]:
    require_role(db, user, organization_id, Role.viewer)
    rows = db.scalars(select(Aircraft).where(Aircraft.organization_id == organization_id))
    return [_aircraft_out(aircraft) for aircraft in rows]


@router.post("/organizations/{organization_id}/aircraft", response_model=AircraftOut, status_code=201)
def create_aircraft(
    organization_id: int, payload: AircraftCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AircraftOut:
    require_role(db, user, organization_id, Role.pilot)
    aircraft = Aircraft(organization_id=organization_id, **payload.model_dump())
    db.add(aircraft)
    db.flush()
    record(db, action="aircraft_added", user_id=user.id, organization_id=organization_id,
           resource=f"aircraft:{aircraft.id}", detail={"name": aircraft.name})
    db.commit()
    return _aircraft_out(aircraft)


@router.post("/aircraft/{aircraft_id}/flights", response_model=AircraftOut)
def log_flight(
    aircraft_id: int, hours: float, user: CurrentUser,
    db: Annotated[Session, Depends(get_db)], pilot_id: int | None = None,
) -> AircraftOut:
    """Record flight time against an aircraft, and its pilot when named."""
    aircraft = db.get(Aircraft, aircraft_id)
    if aircraft is None:
        raise HTTPException(status_code=404, detail="Aircraft not found.")
    require_role(db, user, aircraft.organization_id, Role.pilot)
    if hours <= 0:
        raise HTTPException(status_code=422, detail="Flight hours must be positive.")

    aircraft.flight_hours += float(hours)
    aircraft.flight_count += 1

    if pilot_id is not None:
        pilot = db.get(PilotProfile, pilot_id)
        if pilot is None or pilot.organization_id != aircraft.organization_id:
            raise HTTPException(status_code=404, detail="Pilot not found in this organization.")
        pilot.flight_hours += float(hours)

    record(db, action="flight_logged", user_id=user.id,
           organization_id=aircraft.organization_id, resource=f"aircraft:{aircraft_id}",
           detail={"hours": hours, "pilot_id": pilot_id})
    db.commit()
    return _aircraft_out(aircraft)


@router.get("/organizations/{organization_id}/fleet/status")
def fleet_status(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """What needs attention before the next job.

    This is the question the fleet list exists to answer, so it is computed here
    rather than left to whoever reads the rows.
    """
    require_role(db, user, organization_id, Role.viewer)

    aircraft = [_aircraft_out(a) for a in db.scalars(
        select(Aircraft).where(Aircraft.organization_id == organization_id))]
    batteries = [_battery_out(b) for b in db.scalars(
        select(Battery).where(Battery.organization_id == organization_id))]
    pilots = [_pilot_out(p) for p in db.scalars(
        select(PilotProfile).where(PilotProfile.organization_id == organization_id))]

    return {
        "aircraft": {
            "total": len(aircraft),
            "service_due": [a.name for a in aircraft if a.service_due],
            "grounded": [a.name for a in aircraft if a.status == "grounded"],
        },
        "batteries": {
            "total": len(batteries),
            "past_cycle_limit": [b.serial_number for b in batteries if b.past_cycle_limit],
            "retired": [b.serial_number for b in batteries if b.retired],
        },
        "pilots": {
            "total": len(pilots),
            "not_current": [p.display_name for p in pilots if not p.current],
            "expiring_soon": [p.display_name for p in pilots if p.current and p.warnings],
        },
        "note": (
            "These are advisories. Nothing here prevents a flight; the operator remains "
            "responsible for the decision to fly."
        ),
    }


# ---------------------------------------------------------------------------
# batteries
# ---------------------------------------------------------------------------


@router.get("/organizations/{organization_id}/batteries", response_model=list[BatteryOut])
def list_batteries(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[BatteryOut]:
    require_role(db, user, organization_id, Role.viewer)
    rows = db.scalars(select(Battery).where(Battery.organization_id == organization_id))
    return [_battery_out(battery) for battery in rows]


@router.post("/organizations/{organization_id}/batteries", response_model=BatteryOut, status_code=201)
def create_battery(
    organization_id: int, payload: BatteryCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> BatteryOut:
    require_role(db, user, organization_id, Role.pilot)
    battery = Battery(organization_id=organization_id, **payload.model_dump())
    db.add(battery)
    db.flush()
    record(db, action="battery_added", user_id=user.id, organization_id=organization_id,
           resource=f"battery:{battery.id}", detail={"serial": battery.serial_number})
    db.commit()
    return _battery_out(battery)


@router.post("/batteries/{battery_id}/cycles", response_model=BatteryOut)
def log_battery_cycle(
    battery_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
    health_pct: float | None = None,
) -> BatteryOut:
    """Record one charge cycle, optionally updating measured health."""
    battery = db.get(Battery, battery_id)
    if battery is None:
        raise HTTPException(status_code=404, detail="Battery not found.")
    require_role(db, user, battery.organization_id, Role.pilot)

    battery.cycle_count += 1
    battery.last_used_at = datetime.now(timezone.utc)
    if health_pct is not None:
        battery.health_pct = float(max(0.0, min(100.0, health_pct)))

    db.commit()
    return _battery_out(battery)


@router.post("/batteries/{battery_id}/retire", response_model=BatteryOut)
def retire_battery(
    battery_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> BatteryOut:
    battery = db.get(Battery, battery_id)
    if battery is None:
        raise HTTPException(status_code=404, detail="Battery not found.")
    require_role(db, user, battery.organization_id, Role.admin)
    battery.retired = True
    record(db, action="battery_retired", user_id=user.id,
           organization_id=battery.organization_id, resource=f"battery:{battery_id}")
    db.commit()
    return _battery_out(battery)


# ---------------------------------------------------------------------------
# pilots and maintenance
# ---------------------------------------------------------------------------


@router.get("/organizations/{organization_id}/pilots", response_model=list[PilotOut])
def list_pilots(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[PilotOut]:
    require_role(db, user, organization_id, Role.viewer)
    rows = db.scalars(select(PilotProfile).where(PilotProfile.organization_id == organization_id))
    return [_pilot_out(pilot) for pilot in rows]


@router.post("/organizations/{organization_id}/pilots", response_model=PilotOut, status_code=201)
def create_pilot(
    organization_id: int, payload: PilotCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> PilotOut:
    require_role(db, user, organization_id, Role.admin)
    pilot = PilotProfile(organization_id=organization_id, **payload.model_dump())
    db.add(pilot)
    db.flush()
    record(db, action="pilot_added", user_id=user.id, organization_id=organization_id,
           resource=f"pilot:{pilot.id}", detail={"name": pilot.display_name})
    db.commit()
    return _pilot_out(pilot)


@router.post("/maintenance", status_code=201)
def record_maintenance(
    payload: MaintenanceCreate, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Log a service, resetting the interval from the hours at which it happened."""
    aircraft = db.get(Aircraft, payload.aircraft_id)
    if aircraft is None:
        raise HTTPException(status_code=404, detail="Aircraft not found.")
    require_role(db, user, aircraft.organization_id, Role.engineer)

    hours = (payload.hours_at_service if payload.hours_at_service is not None
             else aircraft.flight_hours)
    entry = Maintenance(
        aircraft_id=aircraft.id, kind=payload.kind, description=payload.description,
        hours_at_service=float(hours), performed_by=user.id,
    )
    db.add(entry)
    aircraft.hours_at_last_service = float(hours)
    aircraft.last_service_at = datetime.now(timezone.utc)
    db.flush()

    record(db, action="maintenance_recorded", user_id=user.id,
           organization_id=aircraft.organization_id, resource=f"aircraft:{aircraft.id}",
           detail={"kind": payload.kind, "hours": hours})
    db.commit()
    return {"id": entry.id, "aircraft_id": aircraft.id, "hours_at_service": hours,
            "aircraft": _aircraft_out(aircraft).model_dump()}


@router.get("/aircraft/{aircraft_id}/maintenance")
def maintenance_history(
    aircraft_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[dict[str, Any]]:
    aircraft = db.get(Aircraft, aircraft_id)
    if aircraft is None:
        raise HTTPException(status_code=404, detail="Aircraft not found.")
    require_role(db, user, aircraft.organization_id, Role.viewer)

    rows = db.scalars(
        select(Maintenance).where(Maintenance.aircraft_id == aircraft_id)
        .order_by(Maintenance.performed_at.desc())
    )
    return [
        {"id": row.id, "kind": row.kind, "description": row.description,
         "hours_at_service": row.hours_at_service,
         "performed_at": row.performed_at.isoformat()}
        for row in rows
    ]
