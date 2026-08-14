"""Database session and engine.

PostGIS is the target: the spec requires spatial data in native geometry types so the
database can answer "which assets are within this polygon" rather than making the
application load everything and filter in Python.

SQLite is supported as a fallback so the API can be developed and tested without a
running PostgreSQL. Geometry is stored as GeoJSON text there, which is why
`spatial_backend()` exists -- callers that need real spatial predicates must check it
rather than assume, and the API reports which backend is live so nobody mistakes a
development database for a deployment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_SQLITE_URL = "sqlite:///./opendronekit.db"


def database_url() -> str:
    """Connection string, from the environment so deployment does not need a rebuild."""
    return os.environ.get("ODK_DATABASE_URL", DEFAULT_SQLITE_URL).strip() or DEFAULT_SQLITE_URL


def is_postgres(url: str | None = None) -> bool:
    return (url or database_url()).startswith(("postgresql", "postgres"))


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    url = database_url()
    if is_postgres(url):
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    else:
        # check_same_thread=False: FastAPI serves requests on a thread pool.
        _engine = create_engine(
            url, future=True, connect_args={"check_same_thread": False}
        )

        @event.listens_for(_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            # SQLite ignores foreign keys unless asked, which would let a delete
            # silently orphan every mission under a project.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that always closes."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def spatial_backend() -> dict[str, Any]:
    """Report what the database can actually do, rather than what was hoped for."""
    url = database_url()
    if not is_postgres(url):
        return {
            "backend": "sqlite",
            "postgis": False,
            "geometry_storage": "geojson_text",
            "note": (
                "SQLite fallback: geometry is stored as GeoJSON text and spatial queries "
                "are performed in Python. Set ODK_DATABASE_URL to a PostGIS instance for "
                "native spatial indexing."
            ),
        }

    try:
        with get_engine().connect() as connection:
            version = connection.execute(text("SELECT PostGIS_Version()")).scalar()
        return {
            "backend": "postgresql",
            "postgis": True,
            "postgis_version": str(version),
            "geometry_storage": "native_geometry",
            "note": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "postgresql",
            "postgis": False,
            "geometry_storage": "geojson_text",
            "note": f"PostGIS extension not available ({exc}); geometry falls back to text.",
        }


def init_db() -> None:
    """Create tables, and the PostGIS extension when the database supports it."""
    from .models import Base  # noqa: PLC0415 - avoids a circular import at module load

    engine = get_engine()
    if is_postgres():
        with engine.connect() as connection:
            try:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                connection.commit()
            except Exception:
                # A managed database may forbid this; spatial_backend() will report it.
                connection.rollback()
    Base.metadata.create_all(bind=engine)


def dumps_geometry(geometry: dict[str, Any] | None) -> str | None:
    return json.dumps(geometry) if geometry is not None else None


def loads_geometry(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
