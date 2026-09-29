"""An ESRI ASCII grid cached for offline use keeps its .prj, and a cache that cannot place itself says so.

Found by the round-2 verifier: cache_terrain copied only the .asc into terrain_cache/
and left its .prj sidecar behind. The cached copy had no CRS, yet the index still held
the EPSG code and bounds read from the original, so terrain_coverage told the operator
"Terrain following can be planned offline" and describe_terrain_cache said every tile
was present -- for a file set_terrain_source then refused.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

EPSG = 32617
X0, Y0, CELL, N = 437_000.0, 4_573_000.0, 10.0, 500  # a 5 km UTM 17N DEM


def _api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Terrain", str(tmp_path / "project"))["ok"]
    return api


def _metric_grid(path):
    yy, xx = np.mgrid[0:N, 0:N]
    heights = (250.0 + 0.02 * xx + 0.01 * yy).astype(np.float32)
    with rasterio.open(path, "w", driver="AAIGrid", width=N, height=N, count=1,
                       dtype="float32", crs=f"EPSG:{EPSG}",
                       transform=from_origin(X0, Y0, CELL, CELL)) as dst:
        dst.write(heights, 1)
    assert path.with_suffix(".prj").exists()
    return path


def _site_ring():
    """A ~1 km square in the middle of the grid, as lon/lat."""
    xs = [X0 + 2000, X0 + 3000, X0 + 3000, X0 + 2000]
    ys = [Y0 - 3000, Y0 - 3000, Y0 - 2000, Y0 - 2000]
    lon, lat = geo.projected_to_wgs84(xs, ys, EPSG)
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
    return ring + [ring[0]]


def test_a_cached_ascii_grid_keeps_its_prj_and_is_usable(tmp_path):
    (tmp_path / "src").mkdir()
    grid = _metric_grid(tmp_path / "src" / "big_m.asc")
    api = _api(tmp_path)

    cached = api.cache_terrain(str(grid))
    assert cached["ok"], cached
    cached_path = cached["tile"]["path"]
    with rasterio.open(cached_path) as raster:
        assert raster.crs is not None and raster.crs.to_epsg() == EPSG

    # The cached copy is itself an acceptable terrain source.
    assert api.set_terrain_source(cached_path)["ok"]

    assert api.set_aoi(_site_ring())["ok"]
    coverage = api.terrain_coverage()
    assert coverage["ok"] and coverage["covered"], coverage

    described = api.describe_terrain_cache()
    assert described["unusable"] == {}
    assert "present and readable" in described["note"]


def test_a_renamed_cached_grid_carries_its_prj_under_the_new_name(tmp_path):
    (tmp_path / "src").mkdir()
    grid = _metric_grid(tmp_path / "src" / "big_m.asc")
    api = _api(tmp_path)
    cached = api.cache_terrain(str(grid), name="site_dem")
    assert cached["ok"], cached
    with rasterio.open(cached["tile"]["path"]) as raster:
        assert raster.crs.to_epsg() == EPSG


def test_a_cache_entry_whose_file_lost_its_crs_is_not_reported_as_covering(tmp_path):
    """A cache written before the fix (the .asc without its .prj) is refused in words."""
    (tmp_path / "src").mkdir()
    grid = _metric_grid(tmp_path / "src" / "big_m.asc")
    api = _api(tmp_path)
    cached = api.cache_terrain(str(grid))
    assert cached["ok"], cached

    # Reproduce the old, broken cache: the sidecar is gone but the index is intact.
    project = tmp_path / "project" / "terrain_cache"
    (project / "big_m.prj").unlink()
    index = json.loads((project / "index.json").read_text(encoding="utf-8"))
    assert index["tiles"][0]["epsg"] == EPSG

    assert api.set_aoi(_site_ring())["ok"]
    coverage = api.terrain_coverage()
    assert coverage["ok"] and not coverage["covered"], coverage
    assert coverage["reason"] == "unusable_file"
    assert "no coordinate reference system" in coverage["detail"]
    assert "can be planned offline" not in coverage["detail"]

    described = api.describe_terrain_cache()
    assert "big_m" in described["unusable"]
    assert "Every cached tile is present" not in described["note"]


def test_cache_terrain_refuses_a_grid_that_has_no_prj(tmp_path):
    (tmp_path / "src").mkdir()
    grid = _metric_grid(tmp_path / "src" / "big_m.asc")
    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(grid, bare / "big_m.asc")
    api = _api(tmp_path)
    result = api.cache_terrain(str(bare / "big_m.asc"))
    assert not result["ok"]
    assert ".prj" in result["error"]
    assert not (tmp_path / "project" / "terrain_cache" / "index.json").exists()
