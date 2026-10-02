"""Web-map tiles cut from a project's own GeoTIFFs, for deep zoom over the survey.

The map used to show a reconstruction as one PNG preview no wider than 1024 px. For a
300 m site flown at 3 cm/px that preview is ~0.3 m/px, so zooming in past a ~50 m
scale bar showed nothing sharper than the first view -- the survey's real resolution
never reached the screen, and the internet basemap underneath ran out of imagery at
the same depth. These tiles are read from the GeoTIFF itself at whatever zoom the map
asks, so at z21-z22 the operator sees the survey's own pixels.

Only files the application registered can be served: `register(path)` returns an
opaque token and the tile URL carries that token, never a path, so a page cannot ask
the loopback server for an arbitrary file.

The colouring matches `Api.raster_preview`: three-band rasters are shown as RGB with
nodata (or pure black, when no nodata is declared) transparent; single-band rasters are
stretched min..max over the whole file and coloured with the turbo ramp.
"""

from __future__ import annotations

import math
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TILE_SIZE = 256
# Web Mercator half-extent in metres.
_ORIGIN = 20037508.342789244
# Past the survey's native resolution the map enlarges the deepest real tile; asking
# for deeper tiles would only resample the same pixels bigger.
MAX_NATIVE_ZOOM = 24
# A single-band raster up to this many pixels is read whole for its value range, so the
# ramp (and the legend drawn from it) spans the data's own min..max. Larger ones are
# stretched from an overview and say so (value_range_source "overview").
EXACT_RANGE_MAX_PIXELS = 64_000_000

_LOCK = threading.Lock()
_BY_TOKEN: dict[str, "_Registered"] = {}
_BY_PATH: dict[str, str] = {}

_ROUTE = re.compile(r"^/raster-tiles/([A-Za-z0-9_-]{8,64})/(\d+)/(\d+)/(\d+)\.png$")

EMPTY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


@dataclass
class _Registered:
    path: str
    info: dict[str, Any]
    stretch: tuple[float, float] | None


def _mercator_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    span = 2 * _ORIGIN / (2 ** z)
    west = -_ORIGIN + x * span
    north = _ORIGIN - y * span
    return west, north - span, west + span, north


def _describe(path: str) -> tuple[dict[str, Any], tuple[float, float] | None]:
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS, so it cannot be placed on the map.")
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        # Ground size of one pixel, in metres, at the raster's centre.
        mx0, my0, mx1, my1 = transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21)
        lat = math.radians((south + north) / 2.0)
        metres_per_px = (mx1 - mx0) / max(1, src.width) * math.cos(lat)
        # The zoom at which one screen pixel covers one raster pixel.
        native = math.log2(156543.03392804097 * math.cos(lat) / max(metres_per_px, 1e-6))
        maxzoom = max(0, min(MAX_NATIVE_ZOOM, int(math.ceil(native))))
        # The GeoTIFF's own pixel size, where its CRS is in metres.
        pixel_size_m = None
        if src.crs.is_projected and abs(src.res[0] - src.res[1]) <= 1e-9 * max(src.res):
            units = (src.crs.linear_units or "").lower()
            if units in ("metre", "meter", "m", "metres", "meters"):
                pixel_size_m = float(src.res[0])
        stretch = None
        range_source = None
        if src.count < 3:
            exact = src.width * src.height <= EXACT_RANGE_MAX_PIXELS
            factor = 1.0 if exact else max(1.0, max(src.width, src.height) / 2048.0)
            range_source = "full" if exact or factor == 1.0 else "overview"
            shape = (max(1, int(src.height / factor)), max(1, int(src.width / factor)))
            band = src.read(1, out_shape=shape, masked=True).astype("float64")
            valid = band.compressed()
            valid = valid[np.isfinite(valid)]
            if valid.size:
                stretch = (float(valid.min()), float(valid.max()))
        info = {
            "bounds": [west, south, east, north],
            "minzoom": 0,
            "maxzoom": maxzoom,
            "tile_size": TILE_SIZE,
            "native_resolution_m": metres_per_px,
            "pixel_size_m": pixel_size_m,
            "bands": src.count,
            "value_range_source": range_source if stretch else None,
        }
    return info, stretch


def register(path: str | Path) -> dict[str, Any]:
    """Allow `path` to be served as tiles; return its URL template and extent."""
    key = str(Path(path).resolve())
    with _LOCK:
        token = _BY_PATH.get(key)
        if token is not None:
            entry = _BY_TOKEN[token]
            return _answer(token, entry.info, entry.stretch)
    info, stretch = _describe(key)
    with _LOCK:
        token = _BY_PATH.get(key) or secrets.token_urlsafe(12)
        _BY_PATH[key] = token
        _BY_TOKEN[token] = _Registered(key, info, stretch)
    return _answer(token, info, stretch)


def _answer(token: str, info: dict[str, Any], stretch: tuple[float, float] | None) -> dict[str, Any]:
    """What `register` hands the map.

    `value_range` is the [low, high] a single-band raster's tiles are coloured with --
    the turbo ramp runs from low to high -- so a legend drawn from it says exactly what
    each colour on the map means. None for an RGB raster, which is drawn as its own
    colours. `value_range_source` is "full" when it is the whole file's min..max and
    "overview" when the file was too large to read whole. `pixel_size_m` is the
    GeoTIFF's own pixel size (None unless its CRS is in metres); `native_resolution_m`
    is the ground size of a pixel at its centre, which sets the deepest zoom.
    """
    return {"token": token, "url": f"/raster-tiles/{token}/{{z}}/{{x}}/{{y}}.png", **info,
            "value_range": [stretch[0], stretch[1]] if stretch else None}


def render_tile(token: str, z: int, x: int, y: int) -> bytes | None:
    """The PNG for one tile, EMPTY_PNG outside the raster, or None for an unknown token."""
    import cv2
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds as window_from_bounds

    with _LOCK:
        entry = _BY_TOKEN.get(token)
    if entry is None:
        return None
    if z < 0 or z > 30 or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
        return EMPTY_PNG
    tile_bounds = _mercator_bounds(z, x, y)
    dst_transform = from_bounds(*tile_bounds, TILE_SIZE, TILE_SIZE)

    with rasterio.open(entry.path) as src:
        try:
            sw, ss, se, sn = transform_bounds("EPSG:3857", src.crs, *tile_bounds, densify_pts=21)
        except Exception:  # noqa: BLE001 - a tile far outside the raster's projection
            return EMPTY_PNG
        left, bottom, right, top = src.bounds
        if se <= left or sw >= right or sn <= bottom or ss >= top:
            return EMPTY_PNG
        # Read only the part of the file under this tile, decimated to about the
        # tile's own size so a low-zoom tile over a large survey reads the overviews
        # rather than every full-resolution pixel.
        window = window_from_bounds(max(sw, left), max(ss, bottom), min(se, right), min(sn, top),
                                    transform=src.transform)
        if window.width <= 0 or window.height <= 0:
            return EMPTY_PNG
        # Grow the window outward to whole pixels plus a margin of two read pixels, so
        # the resampling at the tile's edges has source on both sides. Rounding it in
        # place could trim up to half a pixel off a side, and at deep zoom (one source
        # pixel spans several tile pixels) that left transparent seams along tile edges.
        factor = max(1.0, max(window.width, window.height) / (TILE_SIZE * 2.0))
        pad = 2 * factor
        col0 = max(0, math.floor(window.col_off - pad))
        row0 = max(0, math.floor(window.row_off - pad))
        col1 = min(src.width, math.ceil(window.col_off + window.width + pad))
        row1 = min(src.height, math.ceil(window.row_off + window.height + pad))
        if col1 <= col0 or row1 <= row0:
            return EMPTY_PNG
        window = Window(col0, row0, col1 - col0, row1 - row0)
        factor = max(1.0, max(window.width, window.height) / (TILE_SIZE * 2.0))
        out_h = max(1, int(round(window.height / factor)))
        out_w = max(1, int(round(window.width / factor)))
        bands = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(bands, window=window, out_shape=(len(bands), out_h, out_w),
                        resampling=Resampling.bilinear if factor > 1 else Resampling.nearest)
        nodata = src.nodata
        alpha_band = None
        if src.count >= 4 and src.colorinterp[3].name == "alpha":
            alpha_band = src.read(4, window=window, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        win_transform = src.window_transform(window) * rasterio.Affine.scale(window.width / out_w,
                                                                           window.height / out_h)
        src_crs = src.crs

    if len(bands) == 3:
        if alpha_band is not None:
            valid = alpha_band > 0
        else:
            valid = np.any(data != (nodata if nodata is not None else 0), axis=0)
        rgb = np.clip(data.astype(np.float32), 0, 255).astype(np.uint8)
    else:
        band = data[0].astype(np.float64)
        valid = np.isfinite(band)
        if nodata is not None:
            valid &= ~np.isclose(band, float(nodata))
        low, high = entry.stretch or (0.0, 1.0)
        span = max(high - low, 1e-9)
        scaled = np.zeros(band.shape, dtype=np.uint8)
        scaled[valid] = np.clip((band[valid] - low) / span * 255.0, 0, 255).astype(np.uint8)
        # cv2 returns BGR; flip to RGB, as the preview does.
        rgb = np.moveaxis(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1], 2, 0)

    out = np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    for index in range(3):
        reproject(rgb[index], out[index], src_transform=win_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    reproject(valid.astype(np.uint8) * 255, mask, src_transform=win_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.nearest,
              src_nodata=None, dst_nodata=0)
    out[3] = mask
    if not mask.any():
        return EMPTY_PNG
    rgba = np.moveaxis(out, 0, 2)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return buffer.tobytes() if ok else None


def raster_tile_response(path: str) -> tuple[int, str, bytes, dict[str, str]] | None:
    """Answer `/raster-tiles/<token>/{z}/{x}/{y}.png`, or None for any other path.

    Shared by every local server that hosts the UI (the desktop shell and the test
    harnesses standing in for it), like `tile_cache.tile_response`.
    """
    match = _ROUTE.match(path)
    if not match:
        return None
    token = match.group(1)
    z, x, y = (int(value) for value in match.groups()[1:])
    try:
        body = render_tile(token, z, x, y)
    except Exception as exc:  # noqa: BLE001 - one bad tile must not take the server down
        return 500, "text/plain", f"{type(exc).__name__}: {exc}".encode("utf-8"), {}
    if body is None:
        return 404, "text/plain", b"unknown raster", {}
    return 200, "image/png", body, {"Cache-Control": "no-store", "X-ODK-Tile": "raster"}
