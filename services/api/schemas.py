"""Request and response shapes.

Kept separate from the ORM so the wire format is a deliberate decision rather than a
side effect of the schema. Password hashes and token hashes never appear here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from .models import Role


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(default="", max_length=200)
    # Creating the first organisation at signup avoids a dead-end account with
    # nothing it is permitted to see.
    organization_name: str = Field(default="", max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class OrganizationOut(BaseModel):
    id: int
    name: str
    slug: str
    role: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class MemberOut(BaseModel):
    user_id: int
    email: str
    display_name: str
    role: str


class MemberInvite(BaseModel):
    email: EmailStr
    role: Role = Role.viewer


class MemberRoleUpdate(BaseModel):
    role: Role


class ApiTokenCreate(BaseModel):
    name: str = Field(default="", max_length=200)


class ApiTokenOut(BaseModel):
    id: int
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked: bool

    model_config = {"from_attributes": True}


class ApiTokenCreated(ApiTokenOut):
    # Present exactly once, at creation. It is never recoverable afterwards.
    secret: str


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    client: str = ""
    project_type: str = "inspection"
    address: str = ""
    longitude: float | None = None
    latitude: float | None = None
    crs_epsg: int = 4326
    tags: str = ""


class ProjectOut(BaseModel):
    id: int
    organization_id: int
    name: str
    description: str
    client: str
    project_type: str
    status: str
    address: str
    longitude: float | None
    latitude: float | None
    crs_epsg: int
    tags: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AssetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    asset_type: str = "building"
    description: str = ""
    longitude: float | None = None
    latitude: float | None = None
    geometry: dict[str, Any] | None = None
    crs_epsg: int = 4326


class AssetOut(BaseModel):
    id: int
    organization_id: int
    name: str
    asset_type: str
    description: str
    longitude: float | None
    latitude: float | None
    geometry: dict[str, Any] | None
    crs_epsg: int
    created_at: datetime


class MissionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    template: str = "grid"
    asset_id: int | None = None
    # Drawn area of interest. Planning refuses to proceed without one rather than
    # falling back to a default polygon somewhere else in the world.
    aoi: dict[str, Any] | list[list[float]] | None = None
    altitude_m: float = 60.0
    speed_m_s: float = 8.0
    front_overlap_pct: float = 75.0
    side_overlap_pct: float = 65.0
    gimbal_tilt_deg: float = -90.0
    crs_epsg: int = 4326


class MissionOut(BaseModel):
    id: int
    project_id: int
    asset_id: int | None
    name: str
    template: str
    version: int
    waypoint_count: int
    distance_m: float
    duration_min: float
    crs_epsg: int
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthOut(BaseModel):
    status: str
    version: str
    database: dict[str, Any]
    capabilities: dict[str, Any]
