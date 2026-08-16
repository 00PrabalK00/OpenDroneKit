"""Native geometry, verified against a real PostGIS rather than asserted.

The health endpoint used to report `native_geometry` whenever the PostGIS extension
answered a version query, while every geometry column in the schema was Text holding
GeoJSON. An operator reading that would size a query-heavy workload around spatial
indexes that did not exist.

The honest fix was to report the truth. The complete fix is this: GeoJSON text is
mirrored into a GIST-indexed geometry column, kept in step by a trigger.

The text column stays the source of truth, and that is not a shortcut. SQLite is a
supported backend and cannot hold a geometry type, so making the native column
authoritative would fork the schema and give the two backends different answers to the
same question. The geom column is a derived index.

These tests need a live PostGIS and skip without one -- a spatial test that passes
against SQLite would be proving nothing about the thing it claims to check. To run them:

    docker run -d --name odk-postgis -e POSTGRES_PASSWORD=odk -e POSTGRES_DB=odk \\
        -p 55432:5432 postgis/postgis:16-3.4
    ODK_TEST_POSTGIS=postgresql+psycopg://postgres:odk@127.0.0.1:55432/odk python -m pytest
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("sqlalchemy")

DSN = os.environ.get("ODK_TEST_POSTGIS", "")
live = pytest.mark.skipif(not DSN, reason="set ODK_TEST_POSTGIS to a running PostGIS")

POLYGON = json.dumps({
    "type": "Polygon",
    "coordinates": [[[77.40, 23.25], [77.41, 23.25], [77.41, 23.26], [77.40, 23.26], [77.40, 23.25]]],
})
POINT = json.dumps({"type": "Point", "coordinates": [78.0, 24.0]})

INSERT = """INSERT INTO assets (organization_id, name, asset_type, description,
                                crs_epsg, created_at, geometry_geojson)
            VALUES (:org, :name, 'building', '', 4326, now(), :geom) RETURNING id"""


@pytest.fixture(scope="module")
def engine():
    os.environ["ODK_DATABASE_URL"] = DSN
    from services.api import db as module

    module._engine = None  # the module caches; this fixture owns the connection
    module.init_db()
    yield module.get_engine()


@pytest.fixture
def org(engine):
    from sqlalchemy import text

    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO organizations (name, slug, created_at) "
            "VALUES ('spatial-test','spatial-test-slug', now()) ON CONFLICT DO NOTHING"
        ))
        return connection.execute(text("SELECT id FROM organizations LIMIT 1")).scalar()


@pytest.fixture(autouse=True)
def clean(engine):
    from sqlalchemy import text

    yield
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM assets WHERE name LIKE 'spatial-test%'"))


@live
class TestTheMigrationApplies:
    def test_every_geometry_table_gains_a_geom_column(self, engine) -> None:
        from services.api import db as module

        mirrored = module.native_geometry_tables(engine)
        assert set(mirrored) == set(module.GEOMETRY_TABLES)

    def test_a_gist_index_exists(self, engine) -> None:
        """Without the index a spatial filter still scans every row.

        The column alone buys nothing -- indexing is the entire reason for the mirror.
        """
        from sqlalchemy import text

        with engine.connect() as connection:
            for table in ("assets", "defects", "measurements", "annotations"):
                found = connection.execute(text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = :t AND indexname = :i"
                ), {"t": table, "i": f"ix_{table}_geom"}).scalar()
                assert found, f"{table} has no spatial index"
                assert "gist" in found.lower()

    def test_running_it_twice_is_a_no_op(self, engine) -> None:
        # It runs on every startup; a second run must not fail or duplicate anything.
        from services.api import db as module

        module.init_db()
        module.init_db()
        assert module.native_geometry_tables(engine)


@live
class TestTheTriggerKeepsTheMirrorInStep:
    def test_insert_populates_the_geometry(self, engine, org) -> None:
        """Application code never writes geom. Writes arrive through several routers and
        a plugin SDK, and any path that forgot the mirror would leave a row findable by
        text and invisible to spatial queries."""
        from sqlalchemy import text

        with engine.begin() as connection:
            asset = connection.execute(text(INSERT),
                                       {"org": org, "name": "spatial-test-a", "geom": POLYGON}).scalar()
            valid, srid = connection.execute(text(
                "SELECT ST_IsValid(geom), ST_SRID(geom) FROM assets WHERE id = :i"), {"i": asset}).one()
        assert valid is True
        assert srid == 4326, "GeoJSON is WGS84 by specification; any other SRID misplaces the feature"

    def test_update_moves_the_geometry(self, engine, org) -> None:
        from sqlalchemy import text

        with engine.begin() as connection:
            asset = connection.execute(text(INSERT),
                                       {"org": org, "name": "spatial-test-b", "geom": POLYGON}).scalar()
            connection.execute(text("UPDATE assets SET geometry_geojson = :g WHERE id = :i"),
                               {"g": POINT, "i": asset})
            kind, x = connection.execute(text(
                "SELECT ST_GeometryType(geom), ST_X(geom) FROM assets WHERE id = :i"), {"i": asset}).one()
        assert kind == "ST_Point"
        assert x == pytest.approx(78.0)

    def test_clearing_the_text_clears_the_mirror(self, engine, org) -> None:
        from sqlalchemy import text

        with engine.begin() as connection:
            asset = connection.execute(text(INSERT),
                                       {"org": org, "name": "spatial-test-c", "geom": POLYGON}).scalar()
            connection.execute(text("UPDATE assets SET geometry_geojson = NULL WHERE id = :i"),
                               {"i": asset})
            is_null = connection.execute(text(
                "SELECT geom IS NULL FROM assets WHERE id = :i"), {"i": asset}).scalar()
        assert is_null is True


@live
class TestBadGeometryDoesNotRejectTheWrite:
    def test_unparseable_geojson_is_still_stored(self, engine, org) -> None:
        """The row must survive. The text column holds exactly what the caller sent, and
        the row stays visible to every non-spatial query -- its geom is simply NULL and
        it falls back to the text path. Rejecting the write instead would lose data over
        an index."""
        from sqlalchemy import text

        with engine.begin() as connection:
            asset = connection.execute(text(INSERT),
                                       {"org": org, "name": "spatial-test-bad", "geom": "{not json}"}).scalar()
            kept, missing = connection.execute(text(
                "SELECT geometry_geojson IS NOT NULL, geom IS NULL FROM assets WHERE id = :i"),
                {"i": asset}).one()
        assert kept is True
        assert missing is True


@live
class TestSpatialQueriesActuallyWork:
    def test_st_intersects_finds_the_asset(self, engine, org) -> None:
        """The capability the whole migration exists for."""
        from sqlalchemy import text

        with engine.begin() as connection:
            connection.execute(text(INSERT), {"org": org, "name": "spatial-test-hit", "geom": POLYGON})
            found = connection.execute(text(
                "SELECT count(*) FROM assets "
                "WHERE name LIKE 'spatial-test%' "
                "AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326))"),
                {"g": POLYGON}).scalar()
        assert found == 1

    def test_a_distant_polygon_finds_nothing(self, engine, org) -> None:
        # A filter that matches everything is not a filter.
        from sqlalchemy import text

        far = json.dumps({"type": "Polygon", "coordinates":
                          [[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]]})
        with engine.begin() as connection:
            connection.execute(text(INSERT), {"org": org, "name": "spatial-test-miss", "geom": POLYGON})
            found = connection.execute(text(
                "SELECT count(*) FROM assets WHERE name LIKE 'spatial-test%' "
                "AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326))"),
                {"g": far}).scalar()
        assert found == 0

    def test_area_is_measurable_in_metres(self, engine, org) -> None:
        from sqlalchemy import text

        with engine.begin() as connection:
            asset = connection.execute(text(INSERT),
                                       {"org": org, "name": "spatial-test-area", "geom": POLYGON}).scalar()
            area = connection.execute(text(
                "SELECT ST_Area(geom::geography) FROM assets WHERE id = :i"), {"i": asset}).scalar()
        # Roughly 1.1 km x 1.1 km at this latitude.
        assert 1_000_000 < area < 1_300_000


@live
class TestTheHealthReportTellsTheTruth:
    def test_it_reports_native_columns_once_they_exist(self, engine) -> None:
        from services.api import db as module

        report = module.spatial_backend()
        assert report["native_geometry_columns"] is True
        assert report["geometry_storage"] == "geojson_text_with_native_mirror"
        assert set(report["indexed_tables"]) == set(module.GEOMETRY_TABLES)

    def test_the_note_says_the_text_column_is_authoritative(self, engine) -> None:
        """A reader must not conclude the geom column is where their data lives."""
        from services.api import db as module

        note = module.spatial_backend()["note"].lower()
        assert "source of truth" in note
        assert "sqlite" in note
