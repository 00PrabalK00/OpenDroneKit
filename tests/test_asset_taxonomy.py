"""The shared asset engine counts only what it can account for.

An asset inventory is a claim about the world that someone will act on: a crew is sent
to seventeen poles, a client is invoiced for a module count, a maintenance backlog is
sized from a tree survey. The claim is only as good as its ability to be checked later,
which is why the refusals here are the substance of the module rather than guardrails
around it.

Three things cannot be missing: where the asset is, how confident the model was, and
which model file produced it. Any of the three absent makes the count unauditable, and
an unauditable count is indistinguishable from a guess once the survey is a month old.
"""

from __future__ import annotations

import pytest

from core.asset_taxonomy import (
    ASSET_TYPES,
    AssetRefused,
    assets_for_domain,
    build_asset_inventory,
    canonical_asset,
    count_assets,
    filter_by_confidence,
)

MODEL = {"key": "power_asset_detector", "sha256": "a" * 64}


def point(x: float = 77.5, y: float = 12.9) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def instance(asset_type: str = "pole", confidence: float = 0.9, **kwargs) -> dict:
    base = {"asset_type": asset_type, "confidence": confidence, "geometry": point()}
    base.update(kwargs)
    return base


def inventory(instances, **kwargs) -> dict:
    return build_asset_inventory(instances, model=MODEL, crs="EPSG:4326", **kwargs)


class TestTheTaxonomyIsShared:
    def test_aliases_resolve_to_one_canonical_type(self) -> None:
        # The failure this prevents: "pole", "Pole" and "utility_pole" counted as three
        # different assets in one inventory.
        for alias in ("pole", "Pole", "utility pole", "UTILITY_POLE", "electric-pole"):
            assert canonical_asset(alias).name == "pole"

    def test_an_unknown_type_is_refused_not_passed_through(self) -> None:
        with pytest.raises(AssetRefused, match="not in the asset taxonomy"):
            canonical_asset("gribble")

    def test_the_packs_vocabularies_are_all_present(self) -> None:
        """Grounded in what the packs already detect, not invented."""
        for name in ("tower", "pole", "insulator", "conductor", "track", "signal",
                     "module", "tree", "plant", "road"):
            assert canonical_asset(name).name == name

    def test_domains_partition_the_taxonomy(self) -> None:
        seen = {asset.name for domain in {a.domain for a in ASSET_TYPES}
                for asset in assets_for_domain(domain)}
        assert seen == {asset.name for asset in ASSET_TYPES}

    def test_an_unknown_domain_is_refused(self) -> None:
        with pytest.raises(AssetRefused):
            assets_for_domain("submarine")


class TestProvenanceIsMandatory:
    def test_no_inventory_without_a_model(self) -> None:
        with pytest.raises(AssetRefused, match="model provenance"):
            build_asset_inventory([instance()], model=None, crs="EPSG:4326")

    def test_a_model_name_alone_is_not_provenance(self) -> None:
        """The digest is what survives a model being replaced in place."""
        with pytest.raises(AssetRefused, match="sha256"):
            build_asset_inventory([instance()], model={"key": "detector"}, crs="EPSG:4326")

    def test_provenance_reaches_every_feature_not_just_the_summary(self) -> None:
        # A single finding lifted out of the collection must still declare its origin.
        result = inventory([instance(), instance("tower")])
        for feature in result["features"]:
            assert feature["properties"]["model_sha256"] == MODEL["sha256"]
            assert feature["properties"]["model_key"] == MODEL["key"]


class TestLocationAndConfidence:
    def test_an_asset_without_geometry_is_refused(self) -> None:
        with pytest.raises(AssetRefused, match="cannot be located"):
            inventory([{"asset_type": "pole", "confidence": 0.9}])

    def test_an_asset_without_confidence_is_refused(self) -> None:
        with pytest.raises(AssetRefused, match="no confidence"):
            inventory([{"asset_type": "pole", "geometry": point()}])

    def test_confidence_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(AssetRefused):
            inventory([instance(confidence=1.4)])

    def test_confidence_survives_onto_the_feature(self) -> None:
        result = inventory([instance(confidence=0.42)])
        assert result["features"][0]["properties"]["confidence"] == 0.42

    def test_the_crs_must_be_stated(self) -> None:
        with pytest.raises(AssetRefused, match="CRS"):
            build_asset_inventory([instance()], model=MODEL, crs="")


class TestGeometryMatchesTheAssetKind:
    def test_a_point_asset_cannot_arrive_as_a_polygon(self) -> None:
        """How a report ends up quoting a footprint for a pole."""
        polygon = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]}
        with pytest.raises(AssetRefused, match="point in this taxonomy"):
            inventory([instance("pole", geometry=polygon)])

    def test_a_polygon_asset_accepts_its_multi_form(self) -> None:
        multi = {"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 1], [1, 1], [0, 0]]]]}
        assert inventory([instance("module", geometry=multi)])["features"]


class TestCountingIsHonest:
    def test_countable_types_are_counted(self) -> None:
        result = inventory([instance("pole"), instance("pole"), instance("tower")])
        assert count_assets(result, "pole") == 2
        assert count_assets(result, "tower") == 1

    def test_a_continuous_type_is_never_counted(self) -> None:
        """"Seven conductors" means seven spans only if somebody defined a span."""
        line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        result = inventory([instance("conductor", geometry=line)])
        with pytest.raises(AssetRefused, match="continuous"):
            count_assets(result, "conductor")

    def test_continuous_assets_are_reported_rather_than_dropped(self) -> None:
        # Absent from `counts` must not read as absent from the site.
        line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        result = inventory([instance("conductor", geometry=line)])
        assert result["summary"]["present_but_not_counted"] == {"conductor": 1}

    def test_the_summary_says_counts_are_of_detections(self) -> None:
        note = inventory([instance()])["summary"]["reading_note"].lower()
        assert "detections" in note and "missed" in note


class TestThresholdingStaysVisible:
    def test_low_confidence_instances_are_removed(self) -> None:
        result = filter_by_confidence(
            inventory([instance(confidence=0.9), instance(confidence=0.2)]), 0.5
        )
        assert result["summary"]["instance_total"] == 1
        assert count_assets(result, "pole") == 1

    def test_what_was_removed_is_reported(self) -> None:
        """A count that halved under thresholding is a different claim from one that did not."""
        result = filter_by_confidence(
            inventory([instance(confidence=0.9), instance(confidence=0.2)]), 0.5
        )
        assert result["summary"]["removed_below_threshold"] == 1
        assert result["summary"]["confidence_threshold"] == 0.5

    def test_provenance_survives_thresholding(self) -> None:
        result = filter_by_confidence(inventory([instance()]), 0.1)
        assert result["summary"]["model_sha256"] == MODEL["sha256"]

    def test_an_impossible_threshold_is_refused(self) -> None:
        with pytest.raises(AssetRefused):
            filter_by_confidence(inventory([instance()]), 1.5)


class TestTheOutputIsUsableGeoJson:
    def test_the_collection_is_well_formed(self) -> None:
        result = inventory([instance(), instance("tree")])
        assert result["type"] == "FeatureCollection"
        assert result["crs"] == "EPSG:4326"
        assert all(f["type"] == "Feature" for f in result["features"])

    def test_every_feature_has_a_stable_identifier(self) -> None:
        result = inventory([instance(), instance()])
        ids = [f["id"] for f in result["features"]]
        assert len(set(ids)) == len(ids)

    def test_an_empty_survey_produces_an_empty_inventory_not_an_error(self) -> None:
        # Finding nothing is a legitimate result and must be reportable as one.
        result = inventory([])
        assert result["features"] == []
        assert result["summary"]["instance_total"] == 0


class TestTheApiExposesIt:
    """A module nobody can reach is not a capability."""

    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("assets", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_the_taxonomy_can_be_read_before_detecting_anything(self, api) -> None:
        result = api.asset_taxonomy()
        assert result["ok"], result.get("error")
        names = {entry["name"] for entry in result["asset_types"]}
        assert {"pole", "module", "tree", "track"} <= names

    def test_a_domain_filters_the_vocabulary(self, api) -> None:
        result = api.asset_taxonomy("power")
        assert result["ok"]
        assert {entry["domain"] for entry in result["asset_types"]} == {"power"}

    def test_an_unknown_domain_is_refused_through_the_api(self, api) -> None:
        assert not api.asset_taxonomy("submarine")["ok"]

    def test_an_inventory_round_trips_through_the_api(self, api) -> None:
        result = api.build_asset_inventory(
            [instance("pole"), instance("tower")],
            model_key=MODEL["key"], model_sha256=MODEL["sha256"], crs="EPSG:4326",
        )
        assert result["ok"], result.get("error")
        assert result["summary"]["counts"] == {"pole": 1, "tower": 1}

    def test_the_api_refuses_an_inventory_with_no_provenance(self, api) -> None:
        result = api.build_asset_inventory([instance()], crs="EPSG:4326")
        assert not result["ok"]
        assert "provenance" in result["error"].lower()

    def test_the_api_applies_a_confidence_floor_and_says_so(self, api) -> None:
        result = api.build_asset_inventory(
            [instance(confidence=0.9), instance(confidence=0.1)],
            model_key=MODEL["key"], model_sha256=MODEL["sha256"], crs="EPSG:4326",
            min_confidence=0.5,
        )
        assert result["ok"], result.get("error")
        assert result["summary"]["removed_below_threshold"] == 1
