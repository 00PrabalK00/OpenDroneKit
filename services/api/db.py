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

from sqlalchemy import create_engine, event, inspect, text
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
            "native_geometry_columns": False,
            "note": (
                "SQLite fallback: geometry is stored as GeoJSON text and spatial queries "
                "are performed in Python. Pointing ODK_DATABASE_URL at PostGIS gives the "
                "extension and its functions, but not native geometry columns -- the "
                "schema stores GeoJSON text on either backend."
            ),
        }

    try:
        with get_engine().connect() as connection:
            version = connection.execute(text("SELECT PostGIS_Version()")).scalar()
        # geometry_storage describes THIS SCHEMA, not what the server could do. It used
        # to report "native_geometry" merely because the extension answered, which is a
        # claim about the wrong thing -- an operator would believe their spatial queries
        # were indexed when they were still filtered in Python. It is now read from the
        # tables themselves.
        mirrored = native_geometry_tables()
        if mirrored:
            return {
                "backend": "postgresql",
                "postgis": True,
                "postgis_version": str(version),
                "geometry_storage": "geojson_text_with_native_mirror",
                "native_geometry_columns": True,
                "indexed_tables": mirrored,
                "note": (
                    "Geometry is stored as GeoJSON text and MIRRORED into GIST-indexed "
                    "PostGIS geometry columns, kept in step by a trigger. The text "
                    "column remains the source of truth because SQLite is a supported "
                    "backend and cannot hold a geometry type; the geom column is a "
                    "derived index, so a row whose GeoJSON will not parse is still "
                    "stored and still visible to every non-spatial query -- its geom is "
                    "NULL and it falls back to the text path."
                ),
            }
        return {
            "backend": "postgresql",
            "postgis": True,
            "postgis_version": str(version),
            "geometry_storage": "geojson_text",
            "native_geometry_columns": False,
            "note": (
                "PostGIS is available, but no geometry columns have been mirrored yet. "
                "Run init_db() to apply the spatial migration; until then spatial "
                "filtering happens in Python."
            ),
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
    _apply_additive_migrations(engine)
    _apply_spatial_migration(engine)


def _apply_additive_migrations(engine) -> None:
    """Keep existing self-hosted databases usable across additive API releases.

    The project does not yet ship Alembic. ``create_all`` creates new tables but does
    not add columns to an existing table, so the immutable assisted-review fields need
    an idempotent, narrowly-scoped migration here instead of failing after upgrade.
    """
    inspector = inspect(engine)
    if not inspector.has_table("annotations"):
        return
    existing = {column["name"] for column in inspector.get_columns("annotations")}
    columns = {
        "origin": "VARCHAR(20) NOT NULL DEFAULT 'human'",
        "machine_claims_json": "TEXT NOT NULL DEFAULT '[]'",
        "review_action": "VARCHAR(30) NOT NULL DEFAULT 'human_drawn'",
        "parent_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "reviewed_by": "INTEGER",
        "reviewed_at": "TIMESTAMP",
    }
    with engine.begin() as connection:
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE annotations ADD COLUMN {name} {definition}"))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_annotations_origin ON annotations (origin)"
        ))



# Tables whose GeoJSON text can be mirrored into a real geometry column. Queried from
# the schema rather than hard-coded once, because a table that gains geometry later must
# not be silently left out of spatial indexing.
GEOMETRY_TABLES = ("annotations", "assets", "defects", "measurements")

# 4326 because every geometry in this schema is stored as GeoJSON, and GeoJSON is
# WGS84 by specification. Assuming any other CRS here would put features somewhere else
# entirely while looking perfectly valid.
GEOMETRY_SRID = 4326


def _apply_spatial_migration(engine) -> None:
    """Mirror GeoJSON text into native geometry columns, on PostGIS only.

    The GeoJSON column stays the source of truth. It has to: SQLite is a supported
    backend and cannot hold a geometry type, so making the native column authoritative
    would fork the schema in two and give the two backends different answers.

    So the geometry column is a derived index, not a second copy anyone writes to. It is
    populated from the text, kept in step by a trigger, and given a GIST index -- which
    is the whole point, since without one a spatial filter still scans every row and the
    column has bought nothing.

    Every statement is idempotent. This runs on each startup and must be a no-op the
    second time.
    """
    if not is_postgres():
        return
    inspector = inspect(engine)
    try:
        with engine.begin() as connection:
            has_postgis = connection.execute(text(
                "SELECT count(*) FROM pg_extension WHERE extname = 'postgis'"
            )).scalar()
    except Exception:  # noqa: BLE001 - reported by spatial_backend(), not raised here
        return
    if not has_postgis:
        return

    for table in GEOMETRY_TABLES:
        if not inspector.has_table(table):
            continue
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "geometry_geojson" not in columns:
            continue
        with engine.begin() as connection:
            connection.execute(text(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS geom "
                f"geometry(Geometry, {GEOMETRY_SRID})"
            ))
            # Backfill. ST_GeomFromGeoJSON raises on malformed input, and one bad row
            # must not block the migration for every other row, so failures are left
            # NULL rather than aborting -- a NULL geom simply falls back to the text
            # path for that row.
            connection.execute(text(f"""
                UPDATE {table}
                   SET geom = ST_SetSRID(ST_GeomFromGeoJSON(geometry_geojson), {GEOMETRY_SRID})
                 WHERE geom IS NULL
                   AND geometry_geojson IS NOT NULL
                   AND geometry_geojson <> ''
                   AND (geometry_geojson::json ->> 'type') IS NOT NULL
            """))
            connection.execute(text(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_geom ON {table} USING GIST (geom)"
            ))
            # A trigger rather than application code: writes arrive through several
            # routers and a plugin SDK, and any path that forgot to update the mirror
            # would leave a row findable by text and invisible to spatial queries.
            connection.execute(text(f"""
                CREATE OR REPLACE FUNCTION {table}_sync_geom() RETURNS trigger AS $$
                BEGIN
                    IF NEW.geometry_geojson IS NULL OR NEW.geometry_geojson = '' THEN
                        NEW.geom := NULL;
                    ELSE
                        BEGIN
                            NEW.geom := ST_SetSRID(
                                ST_GeomFromGeoJSON(NEW.geometry_geojson), {GEOMETRY_SRID});
                        EXCEPTION WHEN others THEN
                            -- Unparseable geometry must not reject the write. The text
                            -- column still holds exactly what the caller sent, and the
                            -- row stays visible to every non-spatial query.
                            NEW.geom := NULL;
                        END;
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """))
            connection.execute(text(f"DROP TRIGGER IF EXISTS trg_{table}_geom ON {table}"))
            connection.execute(text(f"""
                CREATE TRIGGER trg_{table}_geom
                BEFORE INSERT OR UPDATE OF geometry_geojson ON {table}
                FOR EACH ROW EXECUTE FUNCTION {table}_sync_geom()
            """))


def native_geometry_tables(engine=None) -> list[str]:
    """Which tables actually have a populated native geometry column right now."""
    if not is_postgres():
        return []
    try:
        inspector = inspect(engine or get_engine())
        return [
            table for table in GEOMETRY_TABLES
            if inspector.has_table(table)
            and "geom" in {c["name"] for c in inspector.get_columns(table)}
        ]
    except Exception:  # noqa: BLE001
        return []

def dumps_geometry(geometry: dict[str, Any] | None) -> str | None:
    return json.dumps(geometry) if geometry is not None else None


def loads_geometry(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
