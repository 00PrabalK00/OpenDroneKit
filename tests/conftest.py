"""Shared fixtures.

The repository root goes on sys.path so tests import `core` and `mission` the same
way the application does, without needing an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_root(tmp_path_factory):
    """Keep every test away from the user's real projects.

    Application state lives under one absolute data root (core/paths.py). Without this
    a test that builds `AppSession()` would open the developer's real project store --
    and block on the running desktop app's write lock, which is how the old
    cwd-relative store was found to leak. Session-scoped so module fixtures see it too.
    """
    import os

    previous = os.environ.get("ODK_DATA_DIR")
    os.environ["ODK_DATA_DIR"] = str(tmp_path_factory.mktemp("odk_data"))
    yield
    if previous is None:
        os.environ.pop("ODK_DATA_DIR", None)
    else:
        os.environ["ODK_DATA_DIR"] = previous


@pytest.fixture(autouse=True)
def _fresh_data_root_per_test(_isolated_data_root, tmp_path, monkeypatch):
    """A clean root per test, as the old cwd-relative store gave tests that chdir'd:
    two tests creating a project of the same name must not collide."""
    monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "odk_data"))


def survey_imagery_dir() -> Path:
    """The gitignored Aukerman frames: under the checkout, or where ODK_SURVEY_IMAGERY says.

    A git worktree does not carry gitignored data, and on a filesystem without junctions
    it cannot link to it either, so the location can be named instead of copied.
    """
    import os

    override = os.environ.get("ODK_SURVEY_IMAGERY", "").strip()
    if override:
        return Path(override)
    return REPO_ROOT / "training" / "data" / "aukerman_subset"


# A 40x40 m patch at 0.5 m/px: flat ground at 100 m with a 10x10x3 m block on top,
# so the correct volume is exactly 300 m^3 and can be asserted rather than eyeballed.
PIXEL_SIZE_M = 0.5
GRID = 80
GROUND_M = 100.0
BLOCK_HEIGHT_M = 3.0
BLOCK_CELLS = 20
ORIGIN_X = 500000.0
ORIGIN_Y = 4570000.0
BLOCK_VOLUME_M3 = (BLOCK_CELLS * PIXEL_SIZE_M) ** 2 * BLOCK_HEIGHT_M


@pytest.fixture
def surface_pair(tmp_path):
    """Write a DSM/DTM GeoTIFF pair with an analytically known volume."""
    from core import geo

    dtm = np.full((GRID, GRID), GROUND_M, dtype=np.float32)
    dsm = dtm.copy()
    start = (GRID - BLOCK_CELLS) // 2
    dsm[start:start + BLOCK_CELLS, start:start + BLOCK_CELLS] = GROUND_M + BLOCK_HEIGHT_M

    paths = {}
    for name, array in (("dsm", dsm), ("dtm", dtm)):
        target = tmp_path / f"{name}.tif"
        geo.write_geotiff(
            target, array, epsg=32617, west=ORIGIN_X, north=ORIGIN_Y,
            pixel_size=PIXEL_SIZE_M, cog=False,
        )
        paths[name] = target
    return paths


@pytest.fixture
def defect_layer(tmp_path):
    """A small georeferenced defect layer spanning three severities."""
    from core import geo

    features = [
        geo.polygon_feature(
            [[-81.7505, 41.3042], [-81.7504, 41.3042],
             [-81.7504, 41.3043], [-81.7505, 41.3043]],
            {"defect_id": "d1", "defect_type": "spalling", "area_m2": 0.8,
             "length_m": 1.2, "width_m": 0.4, "confidence": 0.91, "observation_count": 3},
        ),
        geo.point_feature(
            -81.7500, 41.3040,
            {"defect_id": "d2", "defect_type": "crack", "area_m2": 0.05,
             "length_m": 2.4, "width_m": 0.01, "confidence": 0.62, "observation_count": 2},
        ),
        geo.point_feature(
            -81.7501, 41.3041,
            {"defect_id": "d3", "defect_type": "efflorescence", "area_m2": 0.3,
             "length_m": 0.6, "width_m": 0.4, "confidence": 0.55, "observation_count": 1},
        ),
    ]
    target = tmp_path / "defects.geojson"
    geo.write_geojson(target, features, epsg=4326)
    return target


@pytest.fixture
def survey_polygon():
    """A small AOI over the Aukerman survey area, in lon/lat."""
    return [
        [-81.7510, 41.3035],
        [-81.7490, 41.3035],
        [-81.7490, 41.3050],
        [-81.7510, 41.3050],
    ]
