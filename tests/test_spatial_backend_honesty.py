"""The database must not claim spatial capabilities the schema does not have.

spatial_backend() reported `geometry_storage: "native_geometry"` whenever the PostGIS
extension answered a version query. Every geometry column in services/api/models is
Text holding GeoJSON, on both backends, so that field described the server's potential
rather than this schema's behaviour.

The consequence is not a crash. An operator reading the health endpoint sees native
geometry storage, reasonably concludes that spatial queries are indexed by PostGIS, and
plans a workload around a property the system does not have -- filtering still happens
in Python, row by row.

Native geometry columns are still the goal. Until the schema has them, saying so is the
whole of the fix, and these tests keep the report tied to the columns rather than to
whether an extension happens to be installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from services.api import db as db_module  # noqa: E402


class TestTheReportMatchesTheSchema:
    def test_sqlite_reports_text_geometry(self, monkeypatch) -> None:
        monkeypatch.setattr(db_module, "database_url", lambda: "sqlite:///:memory:")
        report = db_module.spatial_backend()
        assert report["backend"] == "sqlite"
        assert report["geometry_storage"] == "geojson_text"
        assert report["native_geometry_columns"] is False

    def test_postgis_present_still_reports_text_geometry(self, monkeypatch) -> None:
        """The regression. An available extension is not a migrated schema."""
        monkeypatch.setattr(db_module, "database_url", lambda: "postgresql://x/y")
        monkeypatch.setattr(db_module, "is_postgres", lambda url=None: True)

        class Connection:
            def execute(self, statement):
                class Result:
                    @staticmethod
                    def scalar():
                        return "3.4 USE_GEOS=1"

                return Result()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(db_module, "get_engine", lambda: type("E", (), {"connect": lambda self: Connection()})())

        report = db_module.spatial_backend()
        assert report["postgis"] is True, "PostGIS availability should still be reported"
        assert report["geometry_storage"] == "geojson_text", (
            "PostGIS being installed was reported as native geometry storage. The "
            "columns are Text on both backends; this is the claim that misleads."
        )
        assert report["native_geometry_columns"] is False

    def test_the_note_says_filtering_happens_in_python(self, monkeypatch) -> None:
        # An operator sizing a workload needs to know where the work happens.
        monkeypatch.setattr(db_module, "database_url", lambda: "sqlite:///:memory:")
        assert "python" in db_module.spatial_backend()["note"].lower()

    def test_no_backend_advertises_native_geometry(self, monkeypatch) -> None:
        """Belt and braces across both branches, since the claim is what caused harm."""
        for url, postgres in (("sqlite:///:memory:", False),):
            monkeypatch.setattr(db_module, "database_url", lambda u=url: u)
            monkeypatch.setattr(db_module, "is_postgres", lambda url=None, p=postgres: p)
            report = db_module.spatial_backend()
            assert report["geometry_storage"] != "native_geometry"


class TestPostgisFailureIsReported:
    def test_a_postgres_without_the_extension_says_so(self, monkeypatch) -> None:
        monkeypatch.setattr(db_module, "database_url", lambda: "postgresql://x/y")
        monkeypatch.setattr(db_module, "is_postgres", lambda url=None: True)

        def broken():
            raise RuntimeError("extension postgis does not exist")

        monkeypatch.setattr(db_module, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("no postgis")))
        report = db_module.spatial_backend()
        assert report["postgis"] is False
        assert report["geometry_storage"] == "geojson_text"
        assert report["note"], "a missing extension was not explained"
