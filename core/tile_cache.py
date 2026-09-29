"""Offline basemap tiles on disk, shared by the desktop shell and the Hub service.

A pilot on a site with no signal still needs a map. Tiles are cached to disk before
departure and served locally afterwards. This module is the storage and download logic
with no web framework in it, so the desktop shell can serve and fill the cache itself:
the "Offline cache" basemap used to load from the Hub service at 127.0.0.1:8000, which
the desktop launcher never started -- on a disconnected laptop the map was blank.

Two properties are enforced rather than assumed:

*Nothing is fetched unless asked.* Caching is an explicit request naming an area and a
zoom range. Browsing the map never quietly downloads tiles.

*A missing tile is missing.* An uncached tile is reported as absent (the servers answer
404 with a transparent image) rather than substituted by a neighbour, so a gap in
coverage looks like a gap instead of looking like terrain.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Providers the cache may fetch from, and their real zoom ceilings. Requesting beyond
# a provider's ceiling returns nothing, which is what makes a layer look broken.
PROVIDERS: dict[str, dict[str, Any]] = {
    "satellite": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 23, "attribution": "Esri, Maxar, Earthstar Geographics",
    },
    "street": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "max_zoom": 19, "attribution": "OpenStreetMap contributors",
    },
    "topo": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 19, "attribution": "Esri, USGS, NOAA",
    },
}

USER_AGENT = "OpenDroneKit/1.0 (offline tile cache; +https://github.com/openDroneKit)"

# A 1x1 transparent PNG, returned for a tile that was never cached.
TRANSPARENT_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

# Downloading a whole city at z20 is millions of tiles. This ceiling turns an
# accidental request into a refusal rather than a runaway job filling the disk.
MAX_TILES_PER_REQUEST = 20_000


class TileCacheRefused(ValueError):
    """The request cannot be cached as asked; the message says why."""


@dataclass
class Area:
    provider: str
    west: float
    south: float
    east: float
    north: float
    min_zoom: int = 12
    max_zoom: int = 17


def tile_root() -> Path:
    from .paths import tile_cache_dir

    path = tile_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def deg_to_tile(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    """Web Mercator tile indices for a coordinate."""
    latitude = max(-85.05112878, min(85.05112878, latitude))
    n = 2.0**zoom
    x = int((longitude + 180.0) / 360.0 * n)
    radians = math.radians(latitude)
    y = int((1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * n)
    return max(0, min(int(n) - 1, x)), max(0, min(int(n) - 1, y))


def count_tiles(area: Any) -> int:
    total = 0
    for zoom in range(area.min_zoom, area.max_zoom + 1):
        x0, y0 = deg_to_tile(area.west, area.north, zoom)
        x1, y1 = deg_to_tile(area.east, area.south, zoom)
        total += (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
    return total


def tile_path(provider: str, zoom: int, x: int, y: int) -> Path:
    if provider not in PROVIDERS:
        raise TileCacheRefused(f"Unknown provider {provider!r}. Available: {', '.join(PROVIDERS)}.")
    return tile_root() / provider / str(int(zoom)) / str(int(x)) / f"{int(y)}.png"


def read_tile(provider: str, zoom: int, x: int, y: int) -> bytes | None:
    """The cached tile's bytes, or None when that tile was never cached."""
    try:
        target = tile_path(provider, zoom, x, y)
    except TileCacheRefused:
        return None
    return target.read_bytes() if target.is_file() else None


def tile_media_type(body: bytes) -> str:
    """The real image type of a cached tile, from its magic bytes.

    Tiles are stored as `<y>.png` whatever the provider sent, and the imagery providers
    send JPEG. Serving those as image/png mislabels them; say what the bytes are.
    """
    if body.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def tile_response(path: str, query: str = "") -> tuple[int, str, bytes, dict[str, str]] | None:
    """Answer `/tiles/{z}/{x}/{y}.png?provider=` from the cache, or None for other paths.

    One implementation for every local server that hosts the UI (the desktop shell, and
    the test harness that stands in for it), so the offline basemap a test sees is the
    one the application serves. A tile that was never cached is a 404 with a
    transparent image: a gap in coverage looks like a gap.
    """
    import re
    from urllib.parse import parse_qs

    match = re.match(r"^/tiles/(\d+)/(\d+)/(\d+)\.png$", path)
    if not match:
        return None
    provider = (parse_qs(query).get("provider") or ["satellite"])[0]
    z, x, y = (int(value) for value in match.groups())
    tile = read_tile(provider, z, x, y)
    if tile is None:
        return 404, "image/png", TRANSPARENT_PNG, {"X-ODK-Tile": "not-cached"}
    return 200, tile_media_type(tile), tile, {"X-ODK-Tile": "cached"}


def validate(area: Any) -> int:
    """Refuse a request that cannot be honoured; return its tile count otherwise."""
    spec = PROVIDERS.get(area.provider)
    if spec is None:
        raise TileCacheRefused(f"Unknown provider {area.provider!r}. Available: {', '.join(PROVIDERS)}.")
    if area.max_zoom > spec["max_zoom"]:
        raise TileCacheRefused(
            f"{area.provider} serves up to zoom {spec['max_zoom']}; caching beyond that "
            "would store empty tiles."
        )
    if area.min_zoom > area.max_zoom:
        raise TileCacheRefused("min_zoom must not exceed max_zoom.")
    if not (area.west < area.east and area.south < area.north):
        raise TileCacheRefused("The area is empty: west must be < east and south < north.")
    total = count_tiles(area)
    if total > MAX_TILES_PER_REQUEST:
        raise TileCacheRefused(
            f"That area needs {total:,} tiles, above the {MAX_TILES_PER_REQUEST:,} limit. "
            "Reduce the zoom range or the area."
        )
    return total


def fetch_area(
    area: Any,
    on_tile: Callable[[bool], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, int]:
    """Download every tile in the area that is not cached yet.

    `on_tile(success)` is called once per tile. A provider gap is not a failure of the
    run; it is counted and the area stays honestly incomplete.
    """
    spec = PROVIDERS[area.provider]
    fetched = failed = already = 0
    for zoom in range(area.min_zoom, area.max_zoom + 1):
        x0, y0 = deg_to_tile(area.west, area.north, zoom)
        x1, y1 = deg_to_tile(area.east, area.south, zoom)
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                if should_cancel is not None and should_cancel():
                    return {"fetched": fetched, "failed": failed, "already_cached": already,
                            "cancelled": 1}
                target = tile_path(area.provider, zoom, x, y)
                if target.exists():
                    already += 1
                    if on_tile:
                        on_tile(True)
                    continue
                url = spec["url"].format(z=zoom, x=x, y=y)
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with opener(request, timeout=20) as response:
                        body = response.read()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(body)
                    fetched += 1
                    ok = True
                except (urllib.error.URLError, OSError):
                    failed += 1
                    ok = False
                if on_tile:
                    on_tile(ok)
    return {"fetched": fetched, "failed": failed, "already_cached": already, "cancelled": 0}


def summary() -> dict[str, Any]:
    """What is actually on disk, so "offline ready" is a fact rather than a hope."""
    root = tile_root()
    out: dict[str, Any] = {"root": str(root), "providers": {}, "total_tiles": 0, "bytes": 0}
    for provider_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tiles = list(provider_dir.rglob("*.png"))
        size = sum(tile.stat().st_size for tile in tiles)
        zooms = sorted({int(t.parent.parent.name) for t in tiles if t.parent.parent.name.isdigit()})
        out["providers"][provider_dir.name] = {"tiles": len(tiles), "bytes": size, "zoom_levels": zooms}
        out["total_tiles"] += len(tiles)
        out["bytes"] += size
    return out
