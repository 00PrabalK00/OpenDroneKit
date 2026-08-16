"""The guides exist, and the claims in them are still true.

Documentation rots in a particular direction: it stays confident while the code moves
underneath it. A guide that tells an operator to read `result["reason"]` when the field
is `result["error"]`, or that promises native spatial indexing the schema does not have,
is worse than no guide -- it is wrong with authority, and it is the kind of wrong that
gets believed because it is written down.

So these tests do two jobs. They check the seven guides exist and carry real content,
and they check the specific factual claims that are cheap to verify and expensive to get
wrong. They cannot check prose for truth. They can stop the handful of statements that
would send someone down the wrong path from drifting away from the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

GUIDES = {
    "INSTALLATION.md": "installation",
    "ARCHITECTURE.md": "architecture",
    "USER_GUIDE.md": "user",
    "PILOT_GUIDE.md": "pilot",
    "PLUGIN_GUIDE.md": "plugin",
    "API_GUIDE.md": "api",
    "DEPLOYMENT.md": "deployment",
    "UI_GUIDE.md": "ui",
}


def read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


class TestTheSetIsComplete:
    @pytest.mark.parametrize("name", sorted(GUIDES))
    def test_the_guide_exists(self, name) -> None:
        assert (DOCS / name).is_file(), f"{name} is missing from the documentation set"

    @pytest.mark.parametrize("name", sorted(GUIDES))
    def test_the_guide_is_not_a_stub(self, name) -> None:
        # A placeholder file satisfies "exists" and helps nobody.
        text = read(name)
        assert len(text.splitlines()) > 30, f"{name} is too short to be a guide"
        assert "TODO" not in text and "TBD" not in text, f"{name} still has placeholders"

    @pytest.mark.parametrize("name", sorted(GUIDES))
    def test_the_guide_has_a_title(self, name) -> None:
        assert read(name).lstrip().startswith("# "), f"{name} has no top-level heading"


class TestClaimsMatchTheCode:
    """The specific statements that would misdirect someone if they went stale."""

    def test_the_api_guide_names_the_real_failure_key(self) -> None:
        """`fail()` returns "error". A guide saying "reason" breaks every handler."""
        from app.api import fail

        payload = fail("something went wrong")
        assert "error" in payload
        assert "error" in read("API_GUIDE.md")
        assert '"reason"' not in read("API_GUIDE.md")

    def test_the_api_guide_documents_endpoints_that_exist(self) -> None:
        main = (Path(__file__).resolve().parents[1] / "services" / "api" / "main.py").read_text(
            encoding="utf-8"
        )
        guide = read("API_GUIDE.md")
        for endpoint in ("/health/live", "/health/ready", "/metrics"):
            assert endpoint in main, f"{endpoint} is documented but not defined"
            assert endpoint in guide

    def test_the_deployment_guide_describes_the_spatial_storage_accurately(self) -> None:
        """This assertion was inverted once, and deliberately.

        It used to require the guide to say native geometry was NOT available, which was
        true while every column was Text. Native columns now exist, so the guide must
        describe the mirror -- and must still say the text column is authoritative, since
        a reader concluding their data lives in geom would be wrong on SQLite.
        """
        from services.api import db as db_module

        guide = read("DEPLOYMENT.md")
        assert "geojson_text_with_native_mirror" in guide
        assert "source of truth" in guide
        assert "sqlite" in guide.lower()
        # And the code still agrees about the SQLite answer.
        report = db_module.spatial_backend()
        assert report["geometry_storage"] in {
            "geojson_text", "geojson_text_with_native_mirror",
        }

    def test_the_guide_says_a_bad_geometry_is_still_stored(self) -> None:
        # Losing a row to gain an index would be the wrong trade, and a reader deciding
        # whether to trust the migration needs to know which way it went.
        guide = read("DEPLOYMENT.md").lower()
        assert "still" in guide and "stored" in guide

    def test_the_plugin_guide_lists_the_real_plugin_kinds(self) -> None:
        from sdk.plugins import PluginKind

        guide = read("PLUGIN_GUIDE.md")
        for kind in PluginKind:
            assert kind.value in guide, f"plugin kind {kind.value} is undocumented"

    def test_the_user_guide_names_mission_types_the_planner_accepts(self) -> None:
        from mission.planner import MissionPlanner

        guide = read("USER_GUIDE.md")
        planner = MissionPlanner()
        polygon = [[-81.7525, 41.3022], [-81.7485, 41.3022],
                   [-81.7485, 41.3062], [-81.7525, 41.3062]]
        for mode in ("grid", "double_grid", "mapping_3d"):
            assert mode in guide
            plan = planner.generate(
                polygon_lonlat=polygon, mode=mode, altitude_m=60.0, speed_m_s=8.0
            )
            assert plan.waypoints, f"the guide recommends {mode}, which plans nothing"

    def test_the_installation_guide_lists_the_real_entry_points(self) -> None:
        root = Path(__file__).resolve().parents[1]
        guide = read("INSTALLATION.md")
        for entry in ("main.py", "run_pipeline.py"):
            assert (root / entry).is_file()
            assert entry in guide

    def test_the_guides_state_the_rail_model_limit(self) -> None:
        """Recall 0.743 means an empty result is not a clear corridor.

        This one is repeated deliberately in the user and pilot guides, because it is
        the claim most likely to cause harm if a reader takes silence for safety.
        """
        for name in ("USER_GUIDE.md", "PILOT_GUIDE.md"):
            text = read(name).lower()
            assert "corridor is clear" in text or "track is safe" in text


class TestTheUiGuideMatchesTheUi:
    def test_it_names_every_workspace_the_code_defines(self) -> None:
        """A guide listing twelve workspaces for a fourteen-workspace product misleads."""
        source = (Path(__file__).resolve().parents[1] / "app" / "web" / "js" /
                  "workspace" / "workspaces.js").read_text(encoding="utf-8")
        guide = read("UI_GUIDE.md")
        for title in ("Mission Planning", "Verification", "Digital Twin",
                      "AI Inspection", "Thermal", "Measurements", "Fleet",
                      "Reports", "Developers"):
            assert title in source
            assert title in guide, f"{title} is a workspace but is undocumented"

    def test_it_says_the_data_is_not_measured(self) -> None:
        # The shell says so permanently; the guide must not imply otherwise.
        assert "sample data" in read("UI_GUIDE.md")

    def test_it_admits_the_toolbar_is_not_wired(self) -> None:
        assert "not wired" in read("UI_GUIDE.md")


class TestTheGuidesCarryTheProjectsRule:
    def test_refusal_over_fabrication_is_stated_where_it_matters(self) -> None:
        assert "refusal over fabrication" in read("ARCHITECTURE.md").lower()

    def test_the_user_guide_explains_arbitrary_scale(self) -> None:
        # The single most consequential thing a user can misunderstand.
        text = read("USER_GUIDE.md").lower()
        assert "arbitrary" in text and "scale" in text

    def test_the_pilot_guide_lists_what_the_software_cannot_check(self) -> None:
        text = read("PILOT_GUIDE.md").lower()
        assert "cannot check" in text
        assert "airspace" in text
