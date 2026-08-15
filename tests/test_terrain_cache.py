"""Terrain cached for field use, and refusing to fly on terrain that is not there.

The failure this guards against is quiet and outdoors. A terrain model that fails to
load leaves the planner on flat earth, and a flat-earth plan looks completely normal:
waypoints, altitudes, a clean GeoJSON. Flown with terrain-following switched on over
rising ground, it holds a constant height above the take-off point while the ground
climbs towards it.

So the tests below care less about the happy path than about the three ways coverage can
be absent -- nothing cached, cached for somewhere else, cached but not reaching far
enough -- because each calls for a different action and lumping them together as "no
terrain" tells the operator nothing.
"""

from __future__ import annotations

import json

import pytest

rasterio = pytest.importorskip("rasterio")
import numpy as np
from rasterio.transform import from_origin

from core.terrain_cache import (
    COVERAGE_MARGIN_DEG,
    TerrainCacheError,
    cache_dir,
    cache_terrain,
    coverage_report,
    describe_cache,
    load_index,
    polygon_bounds,
    raster_bounds,
    read_index,
    remove_tile,
    source_covers_area,
    terrain_for_area,
)

# A DEM in WGS84 covering roughly a kilometre around a site in Ohio.
SITE_LON, SITE_LAT = -81.7505, 41.3042


def write_dem(path, *, west=-81.76, north=41.31, size=200, step=0.0001,
              crs="EPSG:4326"):
    transform = from_origin(west, north, step, step)
    # A sloping surface, so an elevation read back is distinguishable from a constant.
    rows = np.linspace(280.0, 340.0, size, dtype="float32")
    data = np.tile(rows.reshape(-1, 1), (1, size))

    profile = {"driver": "GTiff", "height": size, "width": size, "count": 1,
               "dtype": "float32", "transform": transform}
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def site_polygon(lon=SITE_LON, lat=SITE_LAT, half=0.002):
    return [[lon - half, lat - half], [lon + half, lat - half],
            [lon + half, lat + half], [lon - half, lat + half]]


@pytest.fixture
def dem(tmp_path):
    return write_dem(tmp_path / "site_dem.tif")


class TestReadingRasters:
    def test_bounds_come_back_in_lon_lat(self, dem):
        bounds, epsg, cell = raster_bounds(dem)
        assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
        assert epsg == 4326

    def test_cell_size_is_reported_in_metres_even_for_a_degree_raster(self, dem):
        """A cell size in degrees means nothing to an operator planning a flight."""
        _, _, cell = raster_bounds(dem)
        assert 5.0 < cell < 20.0

    def test_a_raster_without_a_crs_is_refused(self, tmp_path):
        bare = write_dem(tmp_path / "bare.tif", crs=None)
        with pytest.raises(TerrainCacheError, match="no coordinate reference system"):
            raster_bounds(bare)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(TerrainCacheError, match="does not exist"):
            raster_bounds(tmp_path / "absent.tif")


class TestCaching:
    def test_a_dem_is_copied_into_the_project(self, dem, tmp_path):
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)

        assert (cache_dir(project) / "site_dem.tif").exists()
        assert tile.name == "site_dem"

    def test_the_index_records_what_ground_is_covered(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)

        tiles = load_index(project)
        assert len(tiles) == 1
        assert tiles[0].min_lon < SITE_LON < tiles[0].max_lon

    def test_caching_the_same_name_twice_replaces_rather_than_duplicates(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)
        cache_terrain(dem, project)
        assert len(load_index(project)) == 1

    def test_a_cached_tile_can_be_removed(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)

        assert remove_tile("site_dem", project) is True
        assert load_index(project) == []

    def test_removing_something_absent_is_false_not_an_error(self, tmp_path):
        assert remove_tile("never_cached", tmp_path / "project") is False

    def test_a_corrupt_index_reads_as_empty_so_callers_refuse(self, tmp_path):
        """Not knowing what is cached must not be mistaken for knowing it is fine."""
        project = tmp_path / "project"
        cache_dir(project).mkdir(parents=True)
        (cache_dir(project) / "index.json").write_text("{ not json", encoding="utf-8")

        assert load_index(project) == []

    def test_an_unreadable_index_is_distinguishable_from_an_empty_one(self, tmp_path):
        """Refusing is right, but "nothing cached" and "cannot tell" are different facts."""
        empty = tmp_path / "empty"
        corrupt = tmp_path / "corrupt"
        cache_dir(corrupt).mkdir(parents=True)
        (cache_dir(corrupt) / "index.json").write_text("{ not json", encoding="utf-8")

        assert read_index(empty) == ([], "absent")
        assert read_index(corrupt) == ([], "unreadable")


class TestCoverage:
    def test_a_covered_site_can_be_planned_offline(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)

        report = coverage_report(site_polygon(), project)
        assert report["covered"] is True
        assert "Terrain following can be planned offline" in report["detail"]

    def test_terrain_for_area_returns_the_tile(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)
        assert terrain_for_area(site_polygon(), project) is not None

    def test_nothing_cached_is_reported_as_no_cache_not_as_failure(self, tmp_path):
        report = coverage_report(site_polygon(), tmp_path / "project")

        assert report["covered"] is False
        assert report["reason"] == "no_cache"
        assert "before going out" in report["detail"]

    def test_terrain_for_somewhere_else_says_so(self, dem, tmp_path):
        """Different advice from having nothing: you cached the wrong area."""
        project = tmp_path / "project"
        cache_terrain(dem, project)

        # Australia.
        report = coverage_report(site_polygon(lon=151.2, lat=-33.8), project)
        assert report["covered"] is False
        assert report["reason"] == "elsewhere"
        assert "area you are actually flying" in report["detail"]

    def test_terrain_that_overlaps_but_does_not_contain_is_the_dangerous_case(
            self, tmp_path):
        """Partial terrain follows the ground where data reaches and flies level where
        it does not, and the transition is invisible."""
        project = tmp_path / "project"
        small = write_dem(tmp_path / "small.tif", west=-81.752, north=41.306,
                          size=40, step=0.0001)
        cache_terrain(small, project)

        report = coverage_report(site_polygon(half=0.004), project)
        assert report["covered"] is False
        assert report["reason"] == "partial"
        assert "worse than not following at all" in report["detail"]

    def test_a_site_touching_the_very_edge_is_not_counted_as_covered(self, dem, tmp_path):
        """The edge of a DEM is where its elevations are least trustworthy."""
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)

        edge = [[tile.min_lon, tile.min_lat],
                [tile.min_lon + 0.0005, tile.min_lat],
                [tile.min_lon + 0.0005, tile.min_lat + 0.0005]]
        assert coverage_report(edge, project)["covered"] is False

    def test_the_margin_is_the_documented_one(self, dem, tmp_path):
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)

        inside = [[tile.min_lon + COVERAGE_MARGIN_DEG * 2, tile.min_lat + COVERAGE_MARGIN_DEG * 2],
                  [tile.min_lon + COVERAGE_MARGIN_DEG * 3, tile.min_lat + COVERAGE_MARGIN_DEG * 2],
                  [tile.min_lon + COVERAGE_MARGIN_DEG * 3, tile.min_lat + COVERAGE_MARGIN_DEG * 3]]
        assert coverage_report(inside, project)["covered"] is True


class TestUnreadableIndex:
    def test_coverage_does_not_claim_the_cache_is_empty_when_it_cannot_be_read(
            self, dem, tmp_path):
        """The DEM may be sitting right there; saying "cache a DEM" sends the operator
        after a file they already have."""
        project = tmp_path / "project"
        cache_terrain(dem, project)
        (cache_dir(project) / "index.json").write_text("{ not json", encoding="utf-8")

        report = coverage_report(site_polygon(), project)
        assert report["covered"] is False
        assert report["reason"] == "unreadable_index"
        assert "not the same as nothing being cached" in report["detail"]

    def test_the_description_says_it_is_not_reporting_an_empty_cache(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)
        (cache_dir(project) / "index.json").write_text("{ not json", encoding="utf-8")

        described = describe_cache(project)
        assert described["index_state"] == "unreadable"
        assert "report of not knowing" in described["note"]


class TestSourceCoversTheArea:
    """The check that stops a hand-picked DEM from planning over ground it never had."""

    def test_a_dem_that_contains_the_site_is_accepted(self, dem):
        extent = source_covers_area(dem, site_polygon())
        assert extent["covered"] is True
        assert extent["cell_size_m"] > 0

    def test_a_dem_that_stops_short_is_the_dangerous_partial_case(self, tmp_path):
        small = write_dem(tmp_path / "small.tif", west=-81.752, north=41.306,
                          size=40, step=0.0001)
        extent = source_covers_area(small, site_polygon(half=0.004))

        assert extent["covered"] is False
        assert extent["reason"] == "partial"
        assert "nothing in the plan marks where that happens" in extent["detail"]

    def test_a_dem_for_another_continent_is_reported_as_elsewhere(self, dem):
        extent = source_covers_area(dem, site_polygon(lon=151.2, lat=-33.8))
        assert extent["covered"] is False
        assert extent["reason"] == "elsewhere"

    def test_a_source_that_is_not_there_is_refused_rather_than_assumed(self, tmp_path):
        extent = source_covers_area(tmp_path / "absent.tif", site_polygon())
        assert extent["covered"] is False
        assert extent["reason"] == "missing_source"

    def test_a_raster_without_a_crs_cannot_answer_the_question(self, tmp_path):
        bare = write_dem(tmp_path / "bare.tif", crs=None)
        extent = source_covers_area(bare, site_polygon())

        assert extent["covered"] is False
        assert extent["reason"] == "unreadable_source"
        assert "no coordinate reference system" in extent["detail"]


class TestMissingFiles:
    def test_a_tile_whose_file_has_gone_is_not_offered_as_coverage(self, dem, tmp_path):
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)
        from pathlib import Path

        Path(tile.path).unlink()

        assert terrain_for_area(site_polygon(), project) is None
        assert coverage_report(site_polygon(), project)["covered"] is False

    def test_the_description_names_the_tiles_whose_files_are_missing(self, dem, tmp_path):
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)
        from pathlib import Path

        Path(tile.path).unlink()

        described = describe_cache(project)
        assert described["missing"] == ["site_dem"]
        assert "cannot be flown terrain-following offline" in described["note"]

    def test_a_lost_file_is_not_reported_as_terrain_for_somewhere_else(self, dem, tmp_path):
        """The extent is still recorded, so the operator knows which DEM to fetch again
        rather than being told nothing cached is near the site."""
        project = tmp_path / "project"
        tile = cache_terrain(dem, project)
        from pathlib import Path

        Path(tile.path).unlink()

        report = coverage_report(site_polygon(), project)
        assert report["reason"] == "file_missing"
        assert "site_dem" in report["detail"]
        assert report["missing_files"] == ["site_dem"]

    def test_a_healthy_cache_describes_itself_as_complete(self, dem, tmp_path):
        project = tmp_path / "project"
        cache_terrain(dem, project)

        described = describe_cache(project)
        assert described["missing"] == []
        assert described["total_mb"] > 0


class TestBounds:
    def test_polygon_bounds_are_the_extremes(self):
        bounds = polygon_bounds([[1.0, 2.0], [3.0, 4.0], [0.5, 5.0]])
        assert bounds == (0.5, 2.0, 3.0, 5.0)

    def test_an_empty_area_is_refused(self):
        with pytest.raises(TerrainCacheError, match="no bounds"):
            polygon_bounds([])
