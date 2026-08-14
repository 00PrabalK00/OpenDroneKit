"""Projects, assets and missions.

Mission creation calls the same `mission.planner` library the desktop application
uses. There is no second planning implementation here -- a mission planned through the
API and one planned on the desktop must be the same mission.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db, dumps_geometry, loads_geometry
from ..models import Asset, Defect, Mission, Organization, Project, ProjectMembership, Role, User
from ..schemas import (
    AssetCreate,
    AssetOut,
    MissionCreate,
    MissionOut,
    ProjectCreate,
    ProjectOut,
)
from ..security import CurrentUser, require_project_role, require_role, role_on_project

router = APIRouter(tags=["projects"])


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


@router.get("/organizations/{organization_id}/projects", response_model=list[ProjectOut])
def list_projects(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[Project]:
    require_role(db, user, organization_id, Role.viewer)
    projects = db.scalars(select(Project).where(Project.organization_id == organization_id))
    # A restricted project is filtered out of the listing for anyone who cannot open
    # it, since a project name discloses which clients an organisation is working for.
    return [p for p in projects if role_on_project(db, user, p) is not None]


@router.post("/organizations/{organization_id}/projects", response_model=ProjectOut, status_code=201)
def create_project(
    organization_id: int, payload: ProjectCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> Project:
    require_role(db, user, organization_id, Role.engineer)
    if db.get(Organization, organization_id) is None:
        raise HTTPException(status_code=404, detail="Organization not found.")

    project = Project(organization_id=organization_id, **payload.model_dump())
    db.add(project)
    db.flush()
    record(db, action="project_created", user_id=user.id,
           organization_id=organization_id, resource=f"project:{project.id}",
           detail={"name": project.name})
    db.commit()
    return project


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> Project:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)
    return project


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> None:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.admin)
    db.delete(project)
    record(db, action="project_deleted", user_id=user.id,
           organization_id=project.organization_id, resource=f"project:{project_id}")
    db.commit()


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


def _asset_out(asset: Asset) -> AssetOut:
    return AssetOut(
        id=asset.id, organization_id=asset.organization_id, name=asset.name,
        asset_type=asset.asset_type, description=asset.description,
        longitude=asset.longitude, latitude=asset.latitude,
        geometry=loads_geometry(asset.geometry_geojson),
        crs_epsg=asset.crs_epsg, created_at=asset.created_at,
    )


@router.get("/organizations/{organization_id}/assets", response_model=list[AssetOut])
def list_assets(
    organization_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[AssetOut]:
    require_role(db, user, organization_id, Role.viewer)
    rows = db.scalars(select(Asset).where(Asset.organization_id == organization_id))
    return [_asset_out(asset) for asset in rows]


@router.post("/organizations/{organization_id}/assets", response_model=AssetOut, status_code=201)
def create_asset(
    organization_id: int, payload: AssetCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> AssetOut:
    require_role(db, user, organization_id, Role.engineer)
    asset = Asset(
        organization_id=organization_id, name=payload.name, asset_type=payload.asset_type,
        description=payload.description, longitude=payload.longitude,
        latitude=payload.latitude, geometry_geojson=dumps_geometry(payload.geometry),
        crs_epsg=payload.crs_epsg,
    )
    db.add(asset)
    db.flush()
    record(db, action="asset_created", user_id=user.id, organization_id=organization_id,
           resource=f"asset:{asset.id}", detail={"name": asset.name})
    db.commit()
    return _asset_out(asset)


# ---------------------------------------------------------------------------
# missions
# ---------------------------------------------------------------------------


def _ring_from_aoi(aoi: Any) -> list[list[float]]:
    """Accept a GeoJSON polygon/feature or a bare ring, and return lon/lat vertices."""
    if aoi is None:
        return []
    if isinstance(aoi, list):
        return [[float(v[0]), float(v[1])] for v in aoi if len(v) >= 2]
    if not isinstance(aoi, dict):
        return []

    geometry = aoi.get("geometry", aoi) if aoi.get("type") == "Feature" else aoi
    if geometry.get("type") == "Polygon":
        rings = geometry.get("coordinates") or []
        if rings:
            return [[float(v[0]), float(v[1])] for v in rings[0] if len(v) >= 2]
    if geometry.get("type") == "FeatureCollection":
        for feature in geometry.get("features", []):
            ring = _ring_from_aoi(feature)
            if ring:
                return ring
    return []


@router.get("/projects/{project_id}/missions", response_model=list[MissionOut])
def list_missions(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[Mission]:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    return list(db.scalars(select(Mission).where(Mission.project_id == project_id)))


@router.post("/projects/{project_id}/missions", response_model=MissionOut, status_code=201)
def create_mission(
    project_id: int, payload: MissionCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> Mission:
    """Plan a mission over a drawn area, using the shared planning library."""
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.pilot)

    ring = _ring_from_aoi(payload.aoi)
    if len(ring) < 3:
        # Refusing beats planning over a default polygon somewhere else entirely.
        raise HTTPException(
            status_code=422,
            detail="An area of interest with at least three vertices is required to plan a mission.",
        )

    from mission.planner import MissionPlanner

    try:
        plan = MissionPlanner().generate(
            mode=payload.template,
            polygon_lonlat=ring,
            altitude_m=payload.altitude_m,
            speed_m_s=payload.speed_m_s,
            front_overlap_pct=payload.front_overlap_pct,
            side_overlap_pct=payload.side_overlap_pct,
            gimbal_tilt_deg=payload.gimbal_tilt_deg,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Mission planning failed: {exc}") from exc

    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
    version = 1 + (db.scalar(
        select(Mission).where(Mission.project_id == project_id).order_by(Mission.version.desc())
    ) or Mission(version=0)).version

    mission = Mission(
        project_id=project_id, asset_id=payload.asset_id, name=payload.name,
        template=payload.template, version=version,
        plan_json=json.dumps(plan_dict, default=str),
        aoi_geojson=json.dumps({"type": "Polygon", "coordinates": [ring + [ring[0]]]}),
        crs_epsg=payload.crs_epsg,
        waypoint_count=len(plan.waypoints),
        distance_m=float(plan.path_distance_m),
        duration_min=float(plan.estimated_time_min),
        created_by=user.id,
    )
    db.add(mission)
    db.flush()
    record(db, action="mission_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"mission:{mission.id}",
           detail={"template": payload.template, "waypoints": mission.waypoint_count})
    db.commit()
    return mission


@router.get("/missions/{mission_id}/plan")
def mission_plan(
    mission_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """The full plan as generated, so a flown mission can always be reproduced."""
    mission = db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found.")
    project = _project_or_404(db, mission.project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    try:
        plan = json.loads(mission.plan_json)
    except json.JSONDecodeError:
        plan = {}
    return {
        "mission_id": mission.id, "name": mission.name, "template": mission.template,
        "version": mission.version, "crs_epsg": mission.crs_epsg,
        "aoi": loads_geometry(mission.aoi_geojson), "plan": plan,
    }


@router.get("/missions/{mission_id}/export/{fmt}")
def export_mission(
    mission_id: int, fmt: str, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Export in any registered format, using the shared exporter library."""
    from mission.exporters import EXPORTERS

    mission = db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found.")
    project = _project_or_404(db, mission.project_id)
    require_role(db, user, project.organization_id, Role.viewer)

    if fmt not in EXPORTERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown format {fmt!r}. Available: {', '.join(sorted(EXPORTERS))}.",
        )

    import tempfile
    from pathlib import Path

    writer, suffix = EXPORTERS[fmt]
    plan = json.loads(mission.plan_json)
    target = Path(tempfile.mkdtemp()) / f"{mission.name or 'mission'}{suffix}"
    try:
        writer(target, plan)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Export failed: {exc}") from exc

    return {
        "mission_id": mission.id, "format": fmt, "suffix": suffix,
        "path": str(target), "bytes": target.stat().st_size,
    }


# ---------------------------------------------------------------------------
# project membership
# ---------------------------------------------------------------------------


def _membership_out(membership: ProjectMembership, db: Session) -> dict[str, Any]:
    user = db.get(User, membership.user_id)
    return {
        "id": membership.id,
        "project_id": membership.project_id,
        "user_id": membership.user_id,
        "email": getattr(user, "email", ""),
        "role": membership.role.value,
        "added_by": membership.added_by,
        "created_at": membership.created_at,
    }


@router.get("/projects/{project_id}/members")
def list_project_members(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Who has been named on this project, and what the organisation grants anyway."""
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.viewer)

    members = db.scalars(
        select(ProjectMembership).where(ProjectMembership.project_id == project_id)
    )
    return {
        "project_id": project_id,
        "restricted": bool(project.restricted),
        "members": [_membership_out(m, db) for m in members],
        "note": (
            "Project roles add to organisation roles; the effective role is the higher "
            "of the two. "
            + (
                "This project is restricted, so organisation membership below admin is "
                "not sufficient on its own."
                if project.restricted else
                "This project is not restricted, so any organisation member can already "
                "see it. Naming someone here can raise their role, not lower it."
            )
        ),
    }


@router.put("/projects/{project_id}/members/{user_id}", status_code=201)
def add_project_member(
    project_id: int, user_id: int, role: str,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Grant a user a role on one project."""
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.admin)

    try:
        wanted = Role(role)
    except ValueError:
        allowed = ", ".join(r.value for r in Role)
        raise HTTPException(status_code=422, detail=f"Unknown role {role!r}. Use: {allowed}.")

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    existing = db.scalar(
        select(ProjectMembership).where(
            ProjectMembership.project_id == project_id,
            ProjectMembership.user_id == user_id,
        )
    )
    if existing is not None:
        existing.role = wanted
        membership = existing
    else:
        membership = ProjectMembership(
            project_id=project_id, user_id=user_id, role=wanted, added_by=user.id)
        db.add(membership)

    db.flush()
    record(db, action="project_member_added", user_id=user.id,
           organization_id=project.organization_id, resource=f"project:{project_id}",
           detail={"member": user_id, "role": wanted.value})
    db.commit()
    return _membership_out(membership, db)


@router.delete("/projects/{project_id}/members/{user_id}", status_code=204)
def remove_project_member(
    project_id: int, user_id: int,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> None:
    project = _project_or_404(db, project_id)
    require_project_role(db, user, project, Role.admin)

    membership = db.scalar(
        select(ProjectMembership).where(
            ProjectMembership.project_id == project_id,
            ProjectMembership.user_id == user_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="That user is not a project member.")

    db.delete(membership)
    record(db, action="project_member_removed", user_id=user.id,
           organization_id=project.organization_id, resource=f"project:{project_id}",
           detail={"member": user_id})
    db.commit()


# ---------------------------------------------------------------------------
# asset inspection history
# ---------------------------------------------------------------------------


@router.get("/assets/{asset_id}/timeline")
def asset_timeline(
    asset_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Every inspection of one asset, oldest first, with what each one found.

    This is the question a persistent asset exists to answer: is this structure getting
    worse. A single survey cannot say; the sequence can. Counts are reported per
    severity and split by whether a human confirmed them, because a rising count of
    unreviewed model predictions is a different fact from a rising count of confirmed
    defects.
    """
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    require_role(db, user, asset.organization_id, Role.viewer)

    projects = [
        project
        for project in db.scalars(select(Project).where(Project.asset_id == asset_id))
        if role_on_project(db, user, project) is not None
    ]
    projects.sort(key=lambda p: p.created_at)

    entries: list[dict[str, Any]] = []
    for project in projects:
        defects = list(db.scalars(select(Defect).where(Defect.project_id == project.id)))
        by_severity: dict[str, int] = {}
        for defect in defects:
            by_severity[defect.severity] = by_severity.get(defect.severity, 0) + 1

        confirmed = sum(1 for d in defects if d.review_state == "confirmed")
        unreviewed = sum(1 for d in defects if d.review_state == "unreviewed")

        entries.append({
            "project_id": project.id,
            "project_name": project.name,
            "status": project.status,
            "inspected_at": project.created_at,
            "defect_count": len(defects),
            "by_severity": by_severity,
            "confirmed": confirmed,
            "unreviewed": unreviewed,
        })

    total = sum(e["defect_count"] for e in entries)
    trend = "unknown"
    if len(entries) >= 2:
        first, last = entries[0]["defect_count"], entries[-1]["defect_count"]
        trend = "worsening" if last > first else "improving" if last < first else "stable"

    return {
        "asset_id": asset_id,
        "asset_name": asset.name,
        "asset_type": asset.asset_type,
        "inspection_count": len(entries),
        "total_defects_recorded": total,
        "trend": trend,
        "timeline": entries,
        "note": (
            "Counts include unconfirmed model predictions, listed separately from "
            "confirmed findings. A trend across inspections reflects what was looked "
            "for as much as what is there: two surveys flown to different "
            "specifications are not directly comparable."
            if entries else
            "No inspections are linked to this asset yet. Set a project's asset_id to "
            "record one against it."
        ),
    }
