"""A terrain CSV whose x/y columns hold UTM eastings/northings is not read as local metres.

Found by the round-2 verifier: x,y columns of 437000,4573000 were accepted as metres
from the mission origin, which puts the terrain ~437 km from the site; the planner only
said coverage could not be checked and "Terrain is followed where they do".
"""

from __future__ import annotations

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from core.terrain_cache import TerrainCacheError, check_terrain_file
from mission.planner import AssetReferenceFrame, _load_terrain_samples_from_csv

import pytest


def _api(tmp_path):
    api = Api(AppSession(store=ProjectStore(tmp_path / "projects.db")))
    assert api.create_project("Terrain", str(tmp_path / "project"))["ok"]
    return api


def test_utm_eastings_in_xy_columns_are_refused(tmp_path):
    csv = tmp_path / "pts.csv"
    csv.write_text("x,y,z\n437000,4573000,300\n437100,4573100,301\n437200,4573000,302\n",
                   encoding="utf-8")
    result = _api(tmp_path).set_terrain_source(str(csv))
    assert not result["ok"], result
    assert "projected coordinates" in result["error"]
    assert "lon/lat" in result["error"]
    with pytest.raises(TerrainCacheError):
        check_terrain_file(csv)
    frame = AssetReferenceFrame(asset_id="a", origin_lon=-81.75, origin_lat=41.30)
    assert _load_terrain_samples_from_csv(str(csv), frame) is None


def test_local_metre_samples_are_still_accepted(tmp_path):
    csv = tmp_path / "local.csv"
    csv.write_text("x_m,y_m,z\n0,0,300\n100,0,301\n0,100,302\n100,100,303\n",
                   encoding="utf-8")
    assert _api(tmp_path).set_terrain_source(str(csv))["ok"]
    frame = AssetReferenceFrame(asset_id="a", origin_lon=-81.75, origin_lat=41.30)
    assert _load_terrain_samples_from_csv(str(csv), frame) is not None


def test_lonlat_samples_are_still_accepted(tmp_path):
    csv = tmp_path / "ll.csv"
    csv.write_text("lon,lat,elevation\n-82.0,41.3,300\n-82.01,41.31,301\n-82.0,41.31,302\n",
                   encoding="utf-8")
    assert _api(tmp_path).set_terrain_source(str(csv))["ok"]
