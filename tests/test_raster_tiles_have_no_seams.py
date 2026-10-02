"""Deep-zoom raster tiles leave no transparent seam along their edges.

At z22 one source pixel of a 3 cm orthomosaic spans a couple of tile pixels. The tile
renderer used to round its read window in place, which could trim up to half a source
pixel off a side, so a strip along the tile edge reprojected from outside the window:
transparent, and the map well showed through as dark lines on the tile grid. The window
is now grown outward with a margin. A synthetic, fully opaque UTM raster is enough to
prove it: inside the raster every tile pixel must be opaque, edges included.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
cv2 = pytest.importorskip("cv2")

from core import raster_tiles  # noqa: E402

LON, LAT = -81.7521, 41.3041  # inside UTM 17N


def _tile_of(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


@pytest.fixture
def opaque_ortho(tmp_path):
    from rasterio.transform import from_origin

    from core import geo

    e, n = geo.wgs84_to_projected([LON], [LAT], 32617)
    size, res = 2000, 0.033  # 66 m across at 3.3 cm/px, like the Aukerman ortho
    rng = np.random.default_rng(7)
    data = rng.integers(40, 220, size=(3, size, size), dtype=np.uint8)
    path = tmp_path / "ortho.tif"
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
                       crs="EPSG:32617", transform=from_origin(float(e[0]) - size * res / 2,
                                                               float(n[0]) + size * res / 2, res, res)) as dst:
        dst.write(data)
    return path


@pytest.mark.parametrize("z", [20, 21, 22])
def test_tiles_inside_the_raster_are_opaque_to_their_edges(opaque_ortho, z) -> None:
    info = raster_tiles.register(opaque_ortho)
    x0, y0 = _tile_of(LON, LAT, z)
    holes = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            body = raster_tiles.render_tile(info["token"], z, x0 + dx, y0 + dy)
            tile = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
            alpha = tile[..., 3]
            # Edge pixels whose neighbour two pixels in is opaque: still inside the raster.
            for edge, inner in ((alpha[0], alpha[2]), (alpha[-1], alpha[-3]),
                                (alpha[:, 0], alpha[:, 2]), (alpha[:, -1], alpha[:, -3])):
                holes += int(((edge < 255) & (inner == 255)).sum())
    assert holes == 0, f"{holes} transparent edge pixels at z{z}"
