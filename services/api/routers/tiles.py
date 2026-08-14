"""Offline basemap tiles.

A pilot on a site with no signal still needs a map. This caches real tiles to disk
before departure and serves them locally afterwards, which is the difference between
"offline mode" meaning a usable map and it meaning a dark rectangle.

Two properties matter and are enforced rather than assumed:

*Nothing is fetched unless asked.* Caching is an explicit request naming an area and a
zoom range. Browsing the map never quietly downloads tiles.

*A missing tile is missing.* An uncached tile returns 404 with a transparent image
rather than a substituted neighbour, so a gap in coverage looks like a gap instead of
looking like terrain.
"""

from __future__ import annotations

import math
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import Role
from ..paths import storage_root
from ..security import CurrentUser, require_role

router = APIRouter(tags=["tiles"])

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


@dataclass
class CacheJob:
    """Progress for one caching run."""

    job_id: str
    provider: str
    total: int
    fetched: int = 0
    failed: int = 0
    done: bool = False
    error: str = ""


_JOBS: dict[str, CacheJob] = {}
_LOCK = threading.Lock()


class CacheRequest(BaseModel):
    provider: str = "satellite"
    west: float
    south: float
    east: float
    north: float
    min_zoom: int = Field(default=12, ge=0, le=23)
    max_zoom: int = Field(default=17, ge=0, le=23)


def tile_root() -> Path:
    path = storage_root() / "tiles"
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


def count_tiles(request: CacheRequest) -> int:
    total = 0
    for zoom in range(request.min_zoom, request.max_zoom + 1):
        x0, y0 = deg_to_tile(request.west, request.north, zoom)
        x1, y1 = deg_to_tile(request.east, request.south, zoom)
        total += (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
    return total


def _tile_path(provider: str, zoom: int, x: int, y: int) -> Path:
    return tile_root() / provider / str(zoom) / str(x) / f"{y}.png"


def _fetch_area(job: CacheJob, request: CacheRequest) -> None:
    """Download every tile in the requested area, recording failures."""
    spec = PROVIDERS[request.provider]
    try:
        for zoom in range(request.min_zoom, request.max_zoom + 1):
            x0, y0 = deg_to_tile(request.west, request.north, zoom)
            x1, y1 = deg_to_tile(request.east, request.south, zoom)
            for x in range(min(x0, x1), max(x0, x1) + 1):
                for y in range(min(y0, y1), max(y0, y1) + 1):
                    target = _tile_path(request.provider, zoom, x, y)
                    if target.exists():
                        job.fetched += 1
                        continue

                    url = spec["url"].format(z=zoom, x=x, y=y)
                    try:
                        http_request = urllib.request.Request(
                            url, headers={"User-Agent": USER_AGENT}
                        )
                        with urllib.request.urlopen(http_request, timeout=20) as response:
                            body = response.read()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(body)
                        job.fetched += 1
                    except (urllib.error.URLError, OSError):
                        # A provider gap is not a failure of the run; it is recorded
                        # and the area stays honestly incomplete.
                        job.failed += 1
    except Exception as exc:  # noqa: BLE001
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.done = True


@router.get("/tiles/providers")
def tile_providers() -> dict[str, Any]:
    """What can be cached, and how deep each provider actually goes."""
    return {
        "providers": {
            name: {"max_zoom": spec["max_zoom"], "attribution": spec["attribution"]}
            for name, spec in PROVIDERS.items()
        },
        "max_tiles_per_request": MAX_TILES_PER_REQUEST,
        "note": (
            "Tiles are fetched only when caching is requested explicitly. Panning the "
            "map never downloads anything into the cache."
        ),
    }


@router.post("/tiles/cache", status_code=202)
def cache_area(
    payload: CacheRequest, user: CurrentUser, db: Annotated[Session, Depends(get_db)],
    organization_id: int = 0,
) -> dict[str, Any]:
    """Download tiles for an area so the map works with no connectivity."""
    if organization_id:
        require_role(db, user, organization_id, Role.pilot)

    spec = PROVIDERS.get(payload.provider)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider {payload.provider!r}. Available: {', '.join(PROVIDERS)}.",
        )
    if payload.max_zoom > spec["max_zoom"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{payload.provider} serves up to zoom {spec['max_zoom']}; caching "
                f"beyond that would store empty tiles."
            ),
        )
    if payload.min_zoom > payload.max_zoom:
        raise HTTPException(status_code=422, detail="min_zoom must not exceed max_zoom.")

    total = count_tiles(payload)
    if total > MAX_TILES_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail=(
                f"That area needs {total:,} tiles, above the {MAX_TILES_PER_REQUEST:,} "
                "limit. Reduce the zoom range or the area."
            ),
        )

    import uuid

    job = CacheJob(job_id=uuid.uuid4().hex[:12], provider=payload.provider, total=total)
    with _LOCK:
        _JOBS[job.job_id] = job

    threading.Thread(target=_fetch_area, args=(job, payload),
                     name=f"tilecache-{job.job_id}", daemon=True).start()

    if organization_id:
        record(db, action="tiles_cached", user_id=user.id, organization_id=organization_id,
               resource=f"tiles:{payload.provider}", detail={"tiles": total})
        db.commit()

    return {"job_id": job.job_id, "provider": payload.provider, "tiles": total,
            "status": "caching"}


@router.get("/tiles/cache/{job_id}")
def cache_status(job_id: str) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Cache job not found.")
    return {
        "job_id": job.job_id, "provider": job.provider, "total": job.total,
        "fetched": job.fetched, "failed": job.failed, "done": job.done,
        "error": job.error,
        "percent": round(100.0 * (job.fetched + job.failed) / max(1, job.total), 1),
    }


@router.get("/tiles/status")
def cache_summary() -> dict[str, Any]:
    """What is actually on disk, so "offline ready" is a fact rather than a hope."""
    root = tile_root()
    summary: dict[str, Any] = {"providers": {}, "total_tiles": 0, "bytes": 0}
    for provider_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tiles = list(provider_dir.rglob("*.png"))
        size = sum(tile.stat().st_size for tile in tiles)
        zooms = sorted({int(t.parent.parent.name) for t in tiles}) if tiles else []
        summary["providers"][provider_dir.name] = {
            "tiles": len(tiles), "bytes": size, "zoom_levels": zooms,
        }
        summary["total_tiles"] += len(tiles)
        summary["bytes"] += size
    return summary


@router.get("/tiles/{z}/{x}/{y}.png")
def serve_tile(z: int, x: int, y: int, provider: str = "satellite") -> Response:
    """Serve a cached tile.

    An uncached tile returns 404 with a transparent image rather than a substituted
    neighbour: a gap in coverage should look like a gap, not like terrain.
    """
    target = _tile_path(provider, z, x, y)
    if not target.exists():
        return Response(
            content=TRANSPARENT_PNG, media_type="image/png", status_code=404,
            headers={"X-ODK-Tile": "not-cached"},
        )
    return Response(
        content=target.read_bytes(), media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400", "X-ODK-Tile": "cached"},
    )
