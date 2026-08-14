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
from ..models import Asset, Mission, Organization, Project, Role
from ..schemas import (
    AssetCreate,
    AssetOut,
    MissionCreate,
    MissionOut,
    ProjectCreate,
    ProjectOut,
)
from ..security import CurrentUser, require_role

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
    return list(db.scalars(select(Project).where(Project.organization_id == organization_id)))


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
    require_role(db, user, project.organization_id, Role.viewer)
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
