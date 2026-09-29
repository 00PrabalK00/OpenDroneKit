"""A line across a hole in the DSM has no over-the-ground length.

On the real Aukerman DSM a 100 m line came back "over the ground 96.1 m": the walked
length was summed over only the spans with elevation, four samples fell in holes, and a
figure shorter than the map distance was offered as the length over sloping ground.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

EPSG = 32617
WEST, NORTH, CELL = 500000.0, 4570000.0, 0.5


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "p.db"))
    api = Api(session)
    api.create_project("gaps", str(tmp_path / "project"))
    recon = tmp_path / "project" / "reconstruction"
    recon.mkdir(parents=True)
    dsm = np.full((80, 80), 100.0, dtype=np.float32)
    dsm[:, 40:44] = np.nan                         # a reconstruction hole across the line
    geo.write_geotiff(recon / "dsm.tif", dsm, epsg=EPSG, west=WEST, north=NORTH,
                      pixel_size=CELL, nodata=float("nan"), cog=False)
    session.restore_layers()
    return api


def _line(y_offset: float):
    x = [WEST + 2.0, WEST + 38.0]
    lon, lat = geo.projected_to_wgs84(x, [NORTH - y_offset] * 2, EPSG)
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def test_a_gap_withholds_the_ground_length_and_says_why(api):
    result = api.measure_drawn("distance", _line(20.0))
    assert result["ok"], result
    assert result["measurement"]["value"] == pytest.approx(36.0, abs=0.01)
    assert "surface_length_m" not in result
    assert "Over-the-ground length not given" in result["surface_note"]


def test_the_profile_records_the_line_not_a_partial_walk(api):
    result = api.measure_drawn("profile", _line(20.0))
    assert result["ok"], result
    assert result["surface_length_complete"] is False
    assert result["surface_length_m"] < result["planar_length_m"]
    kept = [m for m in api.list_measurements()["measurements"] if m["measurement_type"] == "profile"]
    assert kept[0]["value"] == pytest.approx(result["planar_length_m"])
