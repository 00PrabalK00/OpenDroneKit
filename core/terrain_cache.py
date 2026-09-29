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
        refuse_non_metre_heights(raster.crs, source.name)
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


def refuse_non_metre_heights(crs, name: str) -> None:
    """Refuse a terrain raster whose horizontal CRS is in feet.

    Heights are read as metres everywhere terrain is used (terrain following, coverage
    height above ground). A DEM in a State Plane feet CRS almost always carries heights
    in feet too, and a GeoTIFF does not record the vertical unit, so its 770 "m" of
    ground would plan a flight -- or place an image -- as if the ground were 535 m
    higher than it is. Refused, in words.
    """
    if crs is None or not crs.is_projected:
        return
    try:
        unit, factor = crs.linear_units_factor
    except Exception:  # noqa: BLE001 - an unknown unit is as unusable as feet
        unit, factor = "an unknown unit", float("nan")
    if not (float(factor) == 1.0):
        raise TerrainCacheError(
            f"{name} is in {crs.to_string()} ({unit} horizontally). Its heights are "
            "probably in the same unit, and the raster does not record which, but "
            "terrain here is read in metres. Export the DEM with heights in metres in a "
            "metric or lon/lat CRS (a UTM zone or EPSG:4326) first."
        )


# Terrain given as a table of points rather than a raster; their columns are checked
# where they are read (mission.planner.terrain_csv_columns).
TERRAIN_SAMPLE_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xyz", ".json"}


def check_terrain_file(path: str | Path) -> None:
    """Refuse a terrain raster that cannot be placed or whose heights are not metres.

    Every raster format is checked, not only GeoTIFF: the Import Terrain picker offers
    ESRI ASCII grids (.asc, with the CRS in a sidecar .prj), and a feet .asc was accepted
    while the same DEM as a GeoTIFF was refused.
    """
    source = Path(path)
    if source.suffix.lower() in TERRAIN_SAMPLE_TABLE_SUFFIXES:
        if source.suffix.lower() == ".json":
            return
        from mission.planner import local_terrain_samples_problem, read_terrain_csv_points

        points = read_terrain_csv_points(str(source))
        if points is not None:
            world, local = points
            problem = (local_terrain_samples_problem(local, source.name)
                       if len(world) < 3 else None)
            if problem:
                raise TerrainCacheError(problem)
        return
    import rasterio

    try:
        raster = rasterio.open(source)
    except Exception as exc:  # noqa: BLE001 - any unreadable raster is refused in words
        raise TerrainCacheError(
            f"{source.name} cannot be read as a terrain raster ({exc}). Use a GeoTIFF, "
            "an ESRI ASCII grid with its .prj, or a CSV of lon/lat/elevation points."
        ) from exc
    with raster:
        if raster.crs is None:
            sidecar = (" An ESRI ASCII grid keeps its CRS in a .prj file of the same "
                       "name next to it; none was found."
                       if source.suffix.lower() in {".asc", ".grd"} else "")
            raise TerrainCacheError(
                f"{source.name} has no coordinate reference system, so the ground it "
                "covers is unknown and its heights cannot be matched to a position."
                + sidecar
            )
        refuse_non_metre_heights(raster.crs, source.name)


def _raster_files(origin: Path) -> list[Path]:
    """The raster and every sidecar GDAL reads with it (.prj, .aux.xml, .hdr ...)."""
    import rasterio

    try:
        with rasterio.open(origin) as raster:
            files = [Path(f) for f in (raster.files or [])]
    except Exception:  # noqa: BLE001 - check_terrain_file has already refused these
        files = []
    if not any(f.resolve() == origin.resolve() for f in files if f.exists()):
        files.insert(0, origin)
    return [f for f in files if f.exists() and f.is_file()]


def _copy_raster_with_sidecars(origin: Path, target_dir: Path, tile_name: str) -> list[Path]:
    """Copy a raster and its sidecars under the tile's name; return what was written."""
    written: list[Path] = []
    stem = origin.stem
    for src in _raster_files(origin):
        # g.asc -> tile.asc, g.prj -> tile.prj, g.asc.aux.xml -> tile.asc.aux.xml
        rest = src.name[len(stem):] if src.name.startswith(stem) else src.name
        dest = target_dir / f"{tile_name}{rest}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        written.append(dest)
    return written


def tile_problem(tile: "TerrainTile") -> str | None:
    """Why a cached tile's file cannot be used, or None when it can."""
    path = Path(tile.path)
    if not path.exists():
        return None  # reported separately as missing
    try:
        check_terrain_file(path)
    except TerrainCacheError as exc:
        return str(exc)
    return None


def cache_terrain(source: str | Path, project_root: str | Path,
                  name: str = "") -> TerrainTile:
    """Copy a terrain raster into the project so it is available with no connectivity."""
    origin = Path(source)
    if not origin.exists():
        raise TerrainCacheError(f"{origin} does not exist.")
    check_terrain_file(origin)
    bounds, epsg, cell = raster_bounds(origin)

    target_dir = cache_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    tile_name = name or origin.stem
    target = target_dir / f"{tile_name}{origin.suffix}"
    if origin.resolve() != target.resolve():
        copied = _copy_raster_with_sidecars(origin, target_dir, tile_name)
        # The cached copy is what gets flown, so it is the one that must place itself.
        # An ESRI ASCII grid keeps its CRS in a .prj beside it; a copy that lost it
        # would be indexed as covering the site while being unusable.
        try:
            check_terrain_file(target)
        except TerrainCacheError as exc:
            for path in copied:
                path.unlink(missing_ok=True)
            raise TerrainCacheError(
                f"{origin.name} was readable, but the copy in the terrain cache is not "
                f"({exc}) Nothing was cached."
            ) from exc

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
        if tile.covers(bounds) and Path(tile.path).exists() and tile_problem(tile) is None:
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
    present = [t for t in tiles if Path(t.path).exists()]
    problems = {t.name: tile_problem(t) for t in present}
    usable = [t for t in present if problems[t.name] is None]
    broken = [t for t in present if problems[t.name] is not None]

    # A tile whose file is there but cannot place itself (an ESRI grid whose .prj was
    # left behind, a feet DEM) covers nothing, however its index entry reads.
    broken_cover = [t for t in broken if t.covers(bounds)]
    if broken_cover and not any(t.covers(bounds) for t in usable):
        return {
            "covered": False,
            "reason": "unusable_file",
            "bounds": list(bounds),
            "cached_tiles": [t.to_dict() for t in usable],
            "unusable_files": {t.name: problems[t.name] for t in broken},
            "missing_files": [t.name for t in missing],
            "detail": " ".join(
                f"{t.name} is indexed as covering this area, but its cached file cannot "
                f"be used: {problems[t.name]} Re-cache that DEM."
                for t in broken_cover),
        }

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
    unusable = {t.name: p for t in present if (p := tile_problem(t)) is not None}

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
    elif unusable:
        note = ("Cached tile(s) " + ", ".join(sorted(unusable)) + " are present but "
                "cannot be used: " + " ".join(unusable.values()) + " Re-cache them.")
    else:
        note = "Every cached tile is present and readable."

    return {
        "tile_count": len(tiles),
        "index_state": state,
        "present": [t.to_dict() for t in present if t.name not in unusable],
        "unusable": unusable,
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
    if path.suffix.lower() in {".csv", ".tsv", ".txt", ".xyz"}:
        return _sample_file_covers_area(path, bounds)
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


def _sample_file_covers_area(path: Path, bounds) -> dict[str, Any]:
    """Coverage for a terrain CSV: its samples are lon/lat, so their extent is known.

    A CSV has no coordinate reference system to read, and the raster route used to
    say so as if the terrain had not been followed -- while the planner was following
    it. The extent comes from the samples the planner itself reads.
    """
    from mission.planner import read_terrain_csv_points

    points = read_terrain_csv_points(str(path))
    if points is None:
        return {"covered": False, "reason": "unreadable_source", "bounds": list(bounds),
                "detail": f"{path.name} could not be read as a terrain sample file."}
    world, local = points
    if len(world) < 3:
        if len(local) >= 3:
            return {"covered": False, "reason": "local_coordinates", "bounds": list(bounds),
                    "detail": (f"{path.name} gives its samples in local metres (x, y), not "
                               "longitude and latitude, so whether they reach the whole "
                               "area cannot be checked. Terrain is followed where they do.")}
        return {"covered": False, "reason": "unreadable_source", "bounds": list(bounds),
                "detail": f"{path.name} has fewer than 3 usable elevation samples."}
    lons = [p[0] for p in world]
    lats = [p[1] for p in world]
    extent = (min(lons), min(lats), max(lons), max(lats))
    tile = TerrainTile(name=path.stem, path=str(path), min_lon=extent[0], min_lat=extent[1],
                       max_lon=extent[2], max_lat=extent[3], epsg=4326, cell_size_m=None)
    if tile.covers(bounds):
        return {"covered": True, "reason": "covered", "bounds": list(bounds),
                "source_bounds": list(extent), "cell_size_m": None,
                "detail": f"{path.name} has {len(world)} samples spanning the area."}
    overlaps = not (extent[2] < bounds[0] or extent[0] > bounds[2]
                    or extent[3] < bounds[1] or extent[1] > bounds[3])
    return {
        "covered": False,
        "reason": "partial" if overlaps else "elsewhere",
        "bounds": list(bounds),
        "source_bounds": list(extent),
        "cell_size_m": None,
        "detail": (
            (f"{path.name}'s samples reach part of this area but not all of it. Terrain "
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
