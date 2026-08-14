"""Database schema.

Organisation membership carries the role, not the user, because one person may be a
pilot in one organisation and a client in another. Every resource that can be read or
written hangs off an organisation so authorisation has a single question to answer:
what is this user's role in the organisation that owns this row.

Geometry is stored as GeoJSON text with an explicit `crs_epsg` column. Coordinate
reference systems are never implied here -- a geometry whose CRS is unknown is a
geometry whose measurements are meaningless.
"""

from __future__ import annotations

import enum
import secrets
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Role(str, enum.Enum):
    """Ordered least to most privileged; comparison is by RANK below, not by name."""

    client = "client"
    viewer = "viewer"
    analyst = "analyst"
    inspector = "inspector"
    engineer = "engineer"
    pilot = "pilot"
    admin = "admin"
    owner = "owner"


ROLE_RANK: dict[Role, int] = {
    Role.client: 0,
    Role.viewer: 1,
    Role.analyst: 2,
    Role.inspector: 3,
    Role.engineer: 4,
    Role.pilot: 5,
    Role.admin: 6,
    Role.owner: 7,
}


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    tokens: Mapped[list["ApiToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    projects: Mapped[list["Project"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    assets: Mapped[list["Asset"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class Membership(Base):
    """A user's role within one organisation."""

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "organization_id", name="uq_membership"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.viewer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="memberships")
    organization: Mapped[Organization] = relationship(back_populates="memberships")


class ApiToken(Base):
    """A long-lived credential. Only the hash is stored; the secret is shown once."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped[User] = relationship(back_populates="tokens")

    @staticmethod
    def generate() -> tuple[str, str]:
        """Return (secret shown once, prefix for display)."""
        secret = "odk_" + secrets.token_urlsafe(32)
        return secret, secret[:12]


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    client: Mapped[str] = mapped_column(String(200), default="")
    project_type: Mapped[str] = mapped_column(String(80), default="inspection")
    status: Mapped[str] = mapped_column(String(40), default="planned")
    address: Mapped[str] = mapped_column(String(400), default="")
    # Site location, and the CRS it is expressed in. Never implied.
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    crs_epsg: Mapped[int] = mapped_column(Integer, default=4326)
    tags: Mapped[str] = mapped_column(Text, default="")

    # The thing being inspected. Nullable because plenty of work is a one-off survey
    # with no persistent asset behind it, and inventing one would be worse than none.
    asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Whether organisation-wide membership is enough to see this project. Defaults to
    # False, which preserves the behaviour every existing project was created under:
    # turning restriction on by default would silently revoke access to live data.
    restricted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    organization: Mapped[Organization] = relationship(back_populates="projects")
    asset: Mapped["Asset | None"] = relationship(back_populates="projects")
    members: Mapped[list["ProjectMembership"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    missions: Mapped[list["Mission"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Asset(Base):
    """A persistent physical thing that is inspected repeatedly over time."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(80), default="building")
    description: Mapped[str] = mapped_column(Text, default="")
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    # GeoJSON footprint. Stored as text under SQLite; PostGIS deployments can index it.
    geometry_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs_epsg: Mapped[int] = mapped_column(Integer, default=4326)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    organization: Mapped[Organization] = relationship(back_populates="assets")
    projects: Mapped[list["Project"]] = relationship(back_populates="asset")


class ProjectMembership(Base):
    """A user's role on one project, over and above their organisation role.

    Two things are deliberately separate. An organisation role says what someone may do
    across the tenancy; a project role says what they may do on one job. The effective
    role is the higher of the two, so granting project access can only ever add
    permission -- a bug here that silently removed it would lock an operator out of
    their own survey.
    """

    __tablename__ = "project_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "project_id", name="uq_project_membership"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.viewer, nullable=False)
    added_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="members")
    user: Mapped[User] = relationship()


class Mission(Base):
    __tablename__ = "missions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    template: Mapped[str] = mapped_column(String(80), default="grid")
    version: Mapped[int] = mapped_column(Integer, default=1)
    # The plan as generated. Kept verbatim so a flown mission can always be reproduced.
    plan_json: Mapped[str] = mapped_column(Text, default="{}")
    aoi_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs_epsg: Mapped[int] = mapped_column(Integer, default=4326)
    waypoint_count: Mapped[int] = mapped_column(Integer, default=0)
    distance_m: Mapped[float] = mapped_column(Float, default=0.0)
    duration_min: Mapped[float] = mapped_column(Float, default=0.0)
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="missions")


class AuditEntry(Base):
    """Who did what, when. Written for every state-changing request."""

    __tablename__ = "audit_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    resource: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Dataset(Base):
    """A collection of captured files belonging to a project."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(80), default="imagery")
    description: Mapped[str] = mapped_column(Text, default="")
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UploadSession(Base):
    """An in-flight resumable upload.

    Chunks live on disk until the upload is finalised, so a dropped connection costs
    only the chunks still missing rather than the whole transfer.
    """

    __tablename__ = "upload_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    upload_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(400), nullable=False)
    total_bytes: Mapped[int] = mapped_column(Integer, default=0)
    chunk_size: Mapped[int] = mapped_column(Integer, default=8 * 1024 * 1024)
    # Declared by the client at the start; compared against the assembled file.
    sha256: Mapped[str] = mapped_column(String(64), default="")
    actual_sha256: Mapped[str] = mapped_column(String(64), default="")
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Job(Base):
    """A long-running processing job.

    `engine` and `crs_epsg` are recorded on the job itself so a result can always be
    traced to what produced it and to the coordinate system it is expressed in.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(60), default="reconstruction")
    engine: Mapped[str] = mapped_column(String(60), default="auto")
    profile: Mapped[str] = mapped_column(String(60), default="standard")
    options_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    percent: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    log: Mapped[str] = mapped_column(Text, default="")
    artifacts_json: Mapped[str] = mapped_column(Text, default="[]")
    # None means the run produced nothing georeferenced, which is reported rather
    # than papered over.
    crs_epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Defect(Base):
    """One inspection finding.

    The model's claim and the human's decision are separate fields on purpose. A
    prediction is never stored as verified, so a report can always distinguish what a
    model asserted from what an inspector confirmed.
    """

    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    category: Mapped[str] = mapped_column(String(80), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    description: Mapped[str] = mapped_column(Text, default="")

    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    altitude_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    geometry_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs_epsg: Mapped[int] = mapped_column(Integer, default=4326)
    # None means unmeasured, which is different from zero.
    area_m2: Mapped[float | None] = mapped_column(Float, nullable=True)
    length_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Provenance of the claim.
    source: Mapped[str] = mapped_column(String(20), default="human")
    model_key: Mapped[str] = mapped_column(String(120), default="")
    model_sha256: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The human decision, kept apart from the claim above.
    review_state: Mapped[str] = mapped_column(String(20), default="unreviewed", index=True)
    reviewed_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_note: Mapped[str] = mapped_column(Text, default="")

    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ShareLink(Base):
    """A no-account, read-only link to one project.

    Only the token hash is stored, so a database disclosure does not yield working
    links. Expiry and revocation are properties of the row and are checked on every
    access rather than only at creation.
    """

    __tablename__ = "share_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), default="")
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    allow_download: Mapped[bool] = mapped_column(Boolean, default=False)
    include_defects: Mapped[bool] = mapped_column(Boolean, default=True)
    include_missions: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ShareAccess(Base):
    """One attempt to open a share link, successful or not.

    Failures are recorded too: repeated failures against one link are how a guessed
    token or a revoked link still in circulation becomes visible.
    """

    __tablename__ = "share_accesses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    share_id: Mapped[int] = mapped_column(ForeignKey("share_links.id", ondelete="CASCADE"), index=True)
    client_ip: Mapped[str] = mapped_column(String(64), default="")
    user_agent: Mapped[str] = mapped_column(String(400), default="")
    outcome: Mapped[str] = mapped_column(String(30), default="granted")
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Aircraft(Base):
    """One airframe, with the hours that decide when it is due for service."""

    __tablename__ = "aircraft"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    model: Mapped[str] = mapped_column(String(120), default="")
    serial_number: Mapped[str] = mapped_column(String(120), default="")
    firmware: Mapped[str] = mapped_column(String(80), default="")
    flight_hours: Mapped[float] = mapped_column(Float, default=0.0)
    flight_count: Mapped[int] = mapped_column(Integer, default=0)
    service_interval_hours: Mapped[float] = mapped_column(Float, default=100.0)
    hours_at_last_service: Mapped[float] = mapped_column(Float, default=0.0)
    last_service_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="available")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Battery(Base):
    """A pack, tracked by cycles and measured health rather than age alone."""

    __tablename__ = "batteries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    serial_number: Mapped[str] = mapped_column(String(120), nullable=False)
    capacity_mah: Mapped[int] = mapped_column(Integer, default=0)
    cycle_count: Mapped[int] = mapped_column(Integer, default=0)
    cycle_limit: Mapped[int] = mapped_column(Integer, default=300)
    health_pct: Mapped[float] = mapped_column(Float, default=100.0)
    retired: Mapped[bool] = mapped_column(Boolean, default=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PilotProfile(Base):
    """Currency, which is the thing that quietly expires between jobs."""

    __tablename__ = "pilot_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    licence_number: Mapped[str] = mapped_column(String(120), default="")
    licence_expires_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    medical_expires_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    flight_hours: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Maintenance(Base):
    """A service record. The hours at which it happened reset the interval."""

    __tablename__ = "maintenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    aircraft_id: Mapped[int] = mapped_column(ForeignKey("aircraft.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(60), default="scheduled")
    description: Mapped[str] = mapped_column(Text, default="")
    hours_at_service: Mapped[float] = mapped_column(Float, default=0.0)
    performed_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Webhook(Base):
    """An outbound subscription belonging to one organisation.

    The signing secret is stored in plaintext, unlike a password, because the server
    must reproduce the HMAC for every delivery. It is returned to the caller only at
    creation.
    """

    __tablename__ = "webhooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(String(800), nullable=False)
    events_json: Mapped[str] = mapped_column(Text, default="[]")
    description: Mapped[str] = mapped_column(Text, default="")
    secret_hash: Mapped[str] = mapped_column(String(64), default="")
    secret_prefix: Mapped[str] = mapped_column(String(16), default="")
    secret_plain: Mapped[str] = mapped_column(String(120), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    delivery_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WebhookDelivery(Base):
    """One delivery attempt, successful or not.

    Failures are recorded because a webhook that silently stopped working is worse
    than one that never existed.
    """

    __tablename__ = "webhook_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    webhook_id: Mapped[int] = mapped_column(ForeignKey("webhooks.id", ondelete="CASCADE"), index=True)
    event: Mapped[str] = mapped_column(String(60), default="")
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


__all__ = [
    "Aircraft", "ApiToken", "Asset", "AuditEntry", "Base", "Battery", "Dataset",
    "Defect", "Job", "Maintenance", "Membership", "PilotProfile",
    "Mission",
    "Organization", "Project", "ROLE_RANK", "Role", "ShareAccess", "ShareLink",
    "UploadSession", "User", "Webhook", "WebhookDelivery", "utcnow",
]
