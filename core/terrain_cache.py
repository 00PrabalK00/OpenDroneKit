"""Terrain kept on disk, so following the ground works with no signal.

Terrain-following needs elevations, and the site that needs it most is the one at the
bottom of a valley with no coverage. Fetching terrain when the mission is planned in the
office and carrying it to the field is the only arrangement that works; discovering the
DEM is unreachable while standing next to the aircraft is not.

So terrain is cached per project, with the extent it covers recorded alongside it. The
value of recording the extent is the refusal it makes possible: a mission planned over
ground the cache does not cover is reported as uncovered rather than quietly flown flat.

That refusal matters more than it sounds. A terrain model that fails to load degrades to
flat earth, and a flat-earth plan looks entirely normal -- waypoints, altitudes, a clean
GeoJSON. Flown over rising ground with terrain-following switched on, it holds a
constant height above the take-off point while the ground climbs towards it. The failure
is silent in the software and obvious only outdoors, which is the wrong order.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CACHE_DIRNAME = "terrain_cache"
INDEX_FILENAME = "index.json"

# A tile is accepted as covering a mission only if it contains the whole area with this
# much margin, in degrees. Roughly 100 m: a mission that touches the very edge of a DEM
# samples cells with no neighbours, and the edge of a terrain model is where its
# elevations are least trustworthy.
COVERAGE_MARGIN_DEG = 0.001


class TerrainCacheError(ValueError):
    """The cache cannot serve this area, and saying so is the point."""


@dataclass
class TerrainTile:
    """One cached terrain file and the ground it covers."""

    name: str
    path: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    epsg: int | None = None
    cell_size_m: float | None = None
    source: str = ""
    cached_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def covers(self, bounds: Sequence[float], margin: float = COVERAGE_MARGIN_DEG) -> bool:
        min_lon, min_lat, max_lon, max_lat = bounds
        return (self.min_lon <= min_lon - margin and self.min_lat <= min_lat - margin
                and self.max_lon >= max_lon + margin and self.max_lat >= max_lat + margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "path": self.path,
            "bounds": [self.min_lon, self.min_lat, self.max_lon, self.max_lat],
            "epsg": self.epsg, "cell_size_m": self.cell_size_m,
            "source": self.source, "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TerrainTile":
        bounds = payload.get("bounds") or [0, 0, 0, 0]
        return cls(
            name=payload.get("name", ""), path=payload.get("path", ""),
            min_lon=float(bounds[0]), min_lat=float(bounds[1]),
            max_lon=float(bounds[2]), max_lat=float(bounds[3]),
            epsg=payload.get("epsg"), cell_size_m=payload.get("cell_size_m"),
            source=payload.get("source", ""),
            cached_at=payload.get("cached_at", ""),
        )


def cache_dir(project_root: str | Path) -> Path:
    return Path(project_root) / CACHE_DIRNAME


def _index_path(project_root: str | Path) -> Path:
    return cache_dir(project_root) / INDEX_FILENAME


def read_index(project_root: str | Path) -> tuple[list[TerrainTile], str]:
    """The cached tiles, and whether the index could be read at all.

    The state matters because "nothing is cached" and "we cannot tell what is cached"
    call for opposite reactions from the operator: the first is fixed by caching a DEM,
    the second by looking at why the index will not parse, since the terrain files
    themselves may be sitting there intact.
    """
    path = _index_path(project_root)
    if not path.exists():
        return [], "absent"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], "unreadable"
    if not isinstance(payload, dict):
        return [], "unreadable"
    return [TerrainTile.from_dict(entry) for entry in payload.get("tiles", [])], "ok"


def load_index(project_root: str | Path) -> list[TerrainTile]:
    """The cached tiles. An unreadable index reads as empty, so callers refuse.

    Refusing is the safe direction, but it is not an honest answer on its own, so
    anything that reports to an operator should use `read_index` and say which of the
    two situations it is in.
    """
    return read_index(project_root)[0]


def _save_index(project_root: str | Path, tiles: Sequence[TerrainTile]) -> Path:
    path = _index_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tiles": [t.to_dict() for t in tiles],
                    "note": ("Terrain cached for offline use. A mission over ground not "
                             "covered here must not be flown terrain-following.")},
                   indent=2),
        encoding="utf-8")
    return path


def raster_bounds(path: str | Path) -> tuple[tuple[float, float, float, float], int | None, float | None]:
    """Geographic bounds of a terrain raster, with its CRS and cell size."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TerrainCacheError(
            "rasterio is required to read terrain rasters. Install it with "
            "`pip install rasterio`."
        ) from exc

    source = Path(path)
    if not source.exists():
        raise TerrainCacheError(f"{source} does not exist.")

    with rasterio.open(source) as raster:
        if raster.crs is None:
            raise TerrainCacheError(
                f"{source.name} has no coordinate reference system, so the ground it "
                "covers is unknown and it cannot be cached against an area."
            )
        epsg = raster.crs.to_epsg()
        bounds = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds, densify_pts=21)

        cell = None
        if raster.crs.is_projected:
            cell = float(abs(raster.transform.a))
        else:
            # Degrees to metres at the raster's own latitude, so the number means
            # something rather than being a fraction of a degree.
            mid_lat = (bounds[1] + bounds[3]) / 2.0
            cell = float(abs(raster.transform.a) * 111_320.0 * math.cos(math.radians(mid_lat)))

    return bounds, epsg, cell


def cache_terrain(source: str | Path, project_root: str | Path,
                  name: str = "") -> TerrainTile:
    """Copy a terrain raster into the project so it is available with no connectivity."""
    origin = Path(source)
    bounds, epsg, cell = raster_bounds(origin)

    target_dir = cache_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    tile_name = name or origin.stem
    target = target_dir / f"{tile_name}{origin.suffix}"
    if origin.resolve() != target.resolve():
        shutil.copy2(origin, target)

    tile = TerrainTile(
        name=tile_name, path=str(target),
        min_lon=bounds[0], min_lat=bounds[1], max_lon=bounds[2], max_lat=bounds[3],
        epsg=epsg, cell_size_m=round(cell, 3) if cell else None,
        source=str(origin),
    )

    tiles = [t for t in load_index(project_root) if t.name != tile_name]
    tiles.append(tile)
    _save_index(project_root, tiles)
    return tile


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Bounding box of a lon/lat ring."""
    if not polygon:
        raise TerrainCacheError("An empty area has no bounds.")
    lons = [float(p[0]) for p in polygon]
    lats = [float(p[1]) for p in polygon]
    return min(lons), min(lats), max(lons), max(lats)


def terrain_for_area(polygon: Sequence[Sequence[float]],
                     project_root: str | Path) -> TerrainTile | None:
    """The cached tile covering this area, or None when nothing does."""
    bounds = polygon_bounds(polygon)
    for tile in load_index(project_root):
        if tile.covers(bounds) and Path(tile.path).exists():
            return tile
    return None


def coverage_report(polygon: Sequence[Sequence[float]],
                    project_root: str | Path) -> dict[str, Any]:
    """Whether this area can be flown terrain-following without connectivity.

    Reports the specific reason when it cannot, because "no terrain" and "terrain that
    does not reach far enough" call for different actions.
    """
    bounds = polygon_bounds(polygon)
    tiles, state = read_index(project_root)

    if state == "unreadable":
        return {
            "covered": False,
            "reason": "unreadable_index",
            "bounds": list(bounds),
            "detail": (
                "The terrain cache index cannot be read, so what is cached is unknown. "
                "This is not the same as nothing being cached: the DEM files may be "
                f"present in {cache_dir(project_root)}. Re-cache the terrain for this "
                "site, and do not plan terrain-following until the cache answers for "
                "itself."
            ),
        }

    if not tiles:
        return {
            "covered": False,
            "reason": "no_cache",
            "bounds": list(bounds),
            "detail": (
                "No terrain is cached for this project. Cache a DEM covering the site "
                "before going out, or fly at a fixed altitude above the take-off point "
                "and accept that the height above ground will vary."
            ),
        }

    missing = [t for t in tiles if not Path(t.path).exists()]
    usable = [t for t in tiles if Path(t.path).exists()]

    # A tile that covers this ground but whose file has gone is a different problem from
    # having cached the wrong area, and the operator can act on it: the extent is still
    # recorded, so they know exactly which DEM to fetch again.
    lost_cover = [t for t in missing if t.covers(bounds)]
    if lost_cover and not any(t.covers(bounds) for t in usable):
        names = ", ".join(t.name for t in lost_cover)
        return {
            "covered": False,
            "reason": "file_missing",
            "bounds": list(bounds),
            "cached_tiles": [t.to_dict() for t in usable],
            "missing_files": [t.name for t in missing],
            "detail": (
                f"{names} covers this area but its file is gone from the cache. The "
                "terrain was cached for the right ground, so re-cache that same DEM "
                "rather than looking for a different one."
            ),
        }

    covering = [t for t in usable if t.covers(bounds)]
    if covering:
        tile = covering[0]
        return {
            "covered": True,
            "tile": tile.to_dict(),
            "bounds": list(bounds),
            "detail": (
                f"{tile.name} covers this area at about {tile.cell_size_m} m per cell. "
                "Terrain following can be planned offline."
            ),
        }

    # Something is cached but none of it reaches. Say which is closest, since that is
    # usually a tile that stops just short.
    overlapping = [
        t for t in usable
        if not (t.max_lon < bounds[0] or t.min_lon > bounds[2]
                or t.max_lat < bounds[1] or t.min_lat > bounds[3])
    ]

    return {
        "covered": False,
        "reason": "partial" if overlapping else "elsewhere",
        "bounds": list(bounds),
        "cached_tiles": [t.to_dict() for t in usable],
        "missing_files": [t.name for t in missing],
        "detail": (
            (f"{len(overlapping)} cached tile(s) overlap this area but none contains all "
             "of it. A mission planned on partial terrain follows the ground where the "
             "data reaches and flies level where it does not, which is worse than not "
             "following at all because the transition is invisible.")
            if overlapping else
            "Terrain is cached for this project, but none of it is anywhere near this "
            "site. Cache a DEM for the area you are actually flying."
        ),
    }


def describe_cache(project_root: str | Path) -> dict[str, Any]:
    """What is cached, and whether the files are still there."""
    tiles, state = read_index(project_root)
    present = [t for t in tiles if Path(t.path).exists()]
    missing = [t for t in tiles if not Path(t.path).exists()]

    total_bytes = sum(Path(t.path).stat().st_size for t in present)
    if state == "unreadable":
        note = (
            "The cache index cannot be read, so this is not a report of an empty cache "
            "-- it is a report of not knowing. Check "
            f"{_index_path(project_root)} and re-cache the terrain."
        )
    elif missing:
        note = ("Terrain listed as missing was cached but its file has since gone. Those "
                "areas cannot be flown terrain-following offline.")
    else:
        note = "Every cached tile is present."

    return {
        "tile_count": len(tiles),
        "index_state": state,
        "present": [t.to_dict() for t in present],
        "missing": [t.name for t in missing],
        "total_mb": round(total_bytes / 1_000_000, 1),
        "note": note,
    }


def source_covers_area(source: str | Path,
                       polygon: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Whether a chosen terrain file actually contains the area about to be flown.

    This is the check that makes the rest of the module more than bookkeeping. A terrain
    source is selected by hand and nothing about the file says which site it belongs to,
    so the wrong DEM -- or a right one that stops short -- plans without complaint and
    degrades to flat earth exactly where it runs out.
    """
    bounds = polygon_bounds(polygon)
    path = Path(source)
    if not path.exists():
        return {"covered": False, "reason": "missing_source", "bounds": list(bounds),
                "detail": f"The terrain source {path} does not exist."}
    try:
        raster, epsg, cell = raster_bounds(path)
    except TerrainCacheError as exc:
        return {"covered": False, "reason": "unreadable_source", "bounds": list(bounds),
                "detail": str(exc)}

    tile = TerrainTile(name=path.stem, path=str(path), min_lon=raster[0], min_lat=raster[1],
                       max_lon=raster[2], max_lat=raster[3], epsg=epsg, cell_size_m=cell)
    if tile.covers(bounds):
        return {"covered": True, "reason": "covered", "bounds": list(bounds),
                "source_bounds": list(raster), "cell_size_m": cell,
                "detail": (f"{path.name} contains the area at about {cell:.1f} m per cell."
                           if cell else f"{path.name} contains the area.")}

    overlaps = not (raster[2] < bounds[0] or raster[0] > bounds[2]
                    or raster[3] < bounds[1] or raster[1] > bounds[3])
    return {
        "covered": False,
        "reason": "partial" if overlaps else "elsewhere",
        "bounds": list(bounds),
        "source_bounds": list(raster),
        "cell_size_m": cell,
        "detail": (
            (f"{path.name} overlaps this area but does not contain all of it. Terrain "
             "following will hold to the ground where the data reaches and fly level "
             "where it does not, and nothing in the plan marks where that happens.")
            if overlaps else
            f"{path.name} covers ground nowhere near this site, so terrain following "
            "would fall back to flat earth across the whole flight."
        ),
    }


def remove_tile(name: str, project_root: str | Path) -> bool:
    tiles = load_index(project_root)
    remaining = [t for t in tiles if t.name != name]
    if len(remaining) == len(tiles):
        return False

    for tile in tiles:
        if tile.name == name:
            Path(tile.path).unlink(missing_ok=True)
    _save_index(project_root, remaining)
    return True
