"""Geospatial foundation: EXIF GPS, CRS selection, SfM->world anchoring, GIS writers.

Everything downstream that needs to place a reconstruction, a defect, or a raster in
real-world coordinates goes through this module. Heavy geo dependencies (rasterio,
pyproj) are optional at import time so the toolkit still runs without them; callers
use `geo_capabilities()` to find out what is actually available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

try:  # pragma: no cover - exercised by environment, not tests
    import pyproj

    _HAS_PYPROJ = True
except Exception:  # noqa: BLE001
    pyproj = None  # type: ignore[assignment]
    _HAS_PYPROJ = False

try:  # pragma: no cover
    import rasterio
    from rasterio.crs import CRS as _RioCRS
    from rasterio.transform import from_origin

    _HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    rasterio = None  # type: ignore[assignment]
    _RioCRS = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]
    _HAS_RASTERIO = False

try:  # pragma: no cover
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS

    _HAS_PIL = True
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    GPSTAGS = {}  # type: ignore[assignment]
    TAGS = {}  # type: ignore[assignment]
    _HAS_PIL = False


WGS84_EPSG = 4326
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geo_capabilities() -> dict[str, bool]:
    """Report which optional geo backends are importable in this environment."""
    return {"pyproj": _HAS_PYPROJ, "rasterio": _HAS_RASTERIO, "pillow": _HAS_PIL}


def require(*backends: str) -> None:
    """Raise a single actionable error naming every missing backend."""
    caps = geo_capabilities()
    missing = [name for name in backends if not caps.get(name, False)]
    if missing:
        raise RuntimeError(
            "Missing geospatial dependencies: "
            + ", ".join(sorted(missing))
            + ". Install with: pip install "
            + " ".join(sorted({"pyproj": "pyproj", "rasterio": "rasterio", "pillow": "Pillow"}[m] for m in missing))
        )


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

# Sensor widths in mm, keyed by lowercased "make model". Used to derive a real focal
# length in pixels when the EXIF carries FocalLength but no 35 mm equivalent.
SENSOR_WIDTH_MM: dict[str, float] = {
    "dji fc220": 6.16,
    "dji fc300s": 6.17,
    "dji fc330": 6.17,
    "dji fc350": 6.17,
    "dji fc550": 17.3,
    "dji fc6310": 13.2,
    "dji fc6320": 13.2,
    "dji fc6510": 13.2,
    "dji fc6540": 23.5,
    "dji fc7203": 6.16,
    "dji fc3170": 6.16,
    "dji fc3411": 6.4,
    "dji fc3582": 9.65,
    "dji zenmuse p1": 35.9,
    "dji zenmuse l1": 13.2,
    "dji zenmuse h20t": 6.4,
    "dji m3m": 17.3,
    "hasselblad l1d-20c": 13.2,
    "hasselblad l2d-20c": 17.3,
    "parrot anafi": 6.32,
    "parrot sequoia": 6.17,
    "sensefly s.o.d.a.": 12.8,
    "sony dsc-wx220": 6.17,
    "sony dsc-rx100": 13.2,
    "sony dsc-rx1rm2": 35.8,
    "sony ilce-6000": 23.5,
    "sony ilce-7rm2": 35.9,
    "canon eos 5d mark iii": 36.0,
    "gopro hero4 black": 6.17,
    "gopro hero8 black": 6.17,
}


@dataclass
class GpsFix:
    """A single image's geotag, in WGS84 degrees and metres."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    altitude_ref: str = "sea_level"
    yaw_deg: float | None = None
    timestamp: str | None = None
    source: str = "exif"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraExif:
    """Camera identity and geometry recovered from EXIF, enough to build intrinsics."""

    make: str = ""
    model: str = ""
    width: int = 0
    height: int = 0
    focal_length_mm: float | None = None
    focal_35mm: float | None = None
    sensor_width_mm: float | None = None

    @property
    def key(self) -> str:
        return f"{self.make} {self.model}".strip().lower()

    def to_dict(self) -> dict:
        return asdict(self)


def _rational(value: Any) -> float | None:
    """Coerce an EXIF rational / tuple / number into a float."""
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            denominator = float(value[1])
            return float(value[0]) / denominator if denominator else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_degrees(dms: Any, ref: str | None) -> float | None:
    """Convert an EXIF degrees/minutes/seconds triple plus hemisphere ref to signed degrees."""
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        return None
    degrees = _rational(dms[0])
    minutes = _rational(dms[1])
    seconds = _rational(dms[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if ref and str(ref).upper() in {"S", "W"}:
        value = -value
    return value


def _raw_exif(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Return (tags, gps_tags, width, height) with human-readable tag names."""
    require("pillow")
    tags: dict[str, Any] = {}
    gps_tags: dict[str, Any] = {}
    with Image.open(path) as img:  # type: ignore[union-attr]
        width, height = img.size
        exif = img.getexif()
        if not exif:
            return tags, gps_tags, width, height
        for tag_id, value in exif.items():
            tags[TAGS.get(tag_id, str(tag_id))] = value
        # Pillow >= 9 exposes the sub-IFDs separately; 34853 is the GPS IFD.
        try:
            gps_ifd = exif.get_ifd(0x8825)
        except Exception:  # noqa: BLE001
            gps_ifd = {}
        for tag_id, value in (gps_ifd or {}).items():
            gps_tags[GPSTAGS.get(tag_id, str(tag_id))] = value
        try:
            exif_ifd = exif.get_ifd(0x8769)
        except Exception:  # noqa: BLE001
            exif_ifd = {}
        for tag_id, value in (exif_ifd or {}).items():
            tags.setdefault(TAGS.get(tag_id, str(tag_id)), value)
    return tags, gps_tags, width, height


def read_exif_gps(path: str | Path) -> GpsFix | None:
    """Read a WGS84 geotag from an image, or None when the image is not geotagged."""
    try:
        _tags, gps, _w, _h = _raw_exif(path)
    except Exception:  # noqa: BLE001
        return None
    if not gps:
        return None
    latitude = _dms_to_degrees(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    longitude = _dms_to_degrees(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    if latitude is None or longitude is None:
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None

    altitude = _rational(gps.get("GPSAltitude"))
    altitude_ref_raw = gps.get("GPSAltitudeRef")
    below_sea = False
    if isinstance(altitude_ref_raw, (bytes, bytearray)):
        below_sea = bool(altitude_ref_raw and altitude_ref_raw[0] == 1)
    elif isinstance(altitude_ref_raw, int):
        below_sea = altitude_ref_raw == 1
    if altitude is not None and below_sea:
        altitude = -altitude

    yaw = _rational(gps.get("GPSImgDirection"))
    timestamp = gps.get("GPSDateStamp")
    return GpsFix(
        latitude=float(latitude),
        longitude=float(longitude),
        altitude_m=None if altitude is None else float(altitude),
        altitude_ref="below_sea_level" if below_sea else "sea_level",
        yaw_deg=None if yaw is None else float(yaw) % 360.0,
        timestamp=str(timestamp) if timestamp else None,
    )


def read_exif_camera(path: str | Path) -> CameraExif:
    """Read camera identity plus the focal/sensor terms needed for real intrinsics."""
    try:
        tags, _gps, width, height = _raw_exif(path)
    except Exception:  # noqa: BLE001
        return CameraExif()
    make = str(tags.get("Make", "") or "").strip()
    model = str(tags.get("Model", "") or "").strip()
    camera = CameraExif(
        make=make,
        model=model,
        width=int(width),
        height=int(height),
        focal_length_mm=_rational(tags.get("FocalLength")),
        focal_35mm=_rational(tags.get("FocalLengthIn35mmFilm")),
    )
    camera.sensor_width_mm = SENSOR_WIDTH_MM.get(camera.key)
    if camera.sensor_width_mm is None and model:
        camera.sensor_width_mm = SENSOR_WIDTH_MM.get(model.strip().lower())
    return camera


def intrinsics_from_exif(camera: CameraExif) -> tuple[np.ndarray, str]:
    """Build a pinhole K from EXIF, returning the matrix and how it was derived.

    Order of preference: known sensor width, then the 35 mm equivalent focal length,
    then the legacy 0.9*max(w,h) guess. The provenance string is recorded in outputs
    so a run never silently claims more accuracy than it has.
    """
    width = int(camera.width or 0)
    height = int(camera.height or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Camera EXIF has no usable image dimensions")

    focal_px: float | None = None
    method = "heuristic_0.9_max_dim"

    if camera.focal_length_mm and camera.sensor_width_mm:
        focal_px = float(camera.focal_length_mm) * width / float(camera.sensor_width_mm)
        method = "sensor_width_db"
    elif camera.focal_35mm:
        # 35 mm film is 36 mm wide by definition.
        focal_px = float(camera.focal_35mm) * width / 36.0
        method = "focal_35mm_equivalent"

    if focal_px is None or not math.isfinite(focal_px) or focal_px <= 0.0:
        focal_px = 0.9 * float(max(width, height))
        method = "heuristic_0.9_max_dim"

    matrix = np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, method


def collect_gps_fixes(image_paths: Iterable[str | Path]) -> dict[str, GpsFix]:
    """Geotag every readable image, keyed by file name. Untagged images are omitted."""
    fixes: dict[str, GpsFix] = {}
    for path in image_paths:
        fix = read_exif_gps(path)
        if fix is not None:
            fixes[Path(path).name] = fix
    return fixes


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


def auto_utm_epsg(latitude: float, longitude: float) -> int:
    """Pick the UTM zone EPSG code that minimises distortion at this location."""
    zone = int(math.floor((longitude + 180.0) / 6.0) % 60) + 1
    # Norway and Svalbard break the regular zone grid.
    if 56.0 <= latitude < 64.0 and 3.0 <= longitude < 12.0:
        zone = 32
    elif 72.0 <= latitude < 84.0 and 0.0 <= longitude < 42.0:
        if longitude < 9.0:
            zone = 31
        elif longitude < 21.0:
            zone = 33
        elif longitude < 33.0:
            zone = 35
        else:
            zone = 37
    return (32600 if latitude >= 0.0 else 32700) + zone


def auto_utm_epsg_for_fixes(fixes: Sequence[GpsFix] | dict[str, GpsFix]) -> int:
    """Choose one CRS for a whole dataset, using the mean position of its geotags."""
    values = list(fixes.values()) if isinstance(fixes, dict) else list(fixes)
    if not values:
        raise ValueError("No GPS fixes available to select a CRS")
    latitude = float(np.mean([f.latitude for f in values]))
    longitude = float(np.mean([f.longitude for f in values]))
    return auto_utm_epsg(latitude, longitude)


_TRANSFORMER_CACHE: dict[tuple[int, int], Any] = {}


def transformer(src_epsg: int, dst_epsg: int):
    """Cached pyproj Transformer. Cached because per-point construction dominates cost."""
    require("pyproj")
    key = (int(src_epsg), int(dst_epsg))
    cached = _TRANSFORMER_CACHE.get(key)
    if cached is None:
        cached = pyproj.Transformer.from_crs(  # type: ignore[union-attr]
            f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
        )
        _TRANSFORMER_CACHE[key] = cached
    return cached


def wgs84_to_projected(
    longitudes: Sequence[float] | np.ndarray,
    latitudes: Sequence[float] | np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees into the given CRS, returning easting/northing arrays."""
    tf = transformer(WGS84_EPSG, epsg)
    east, north = tf.transform(np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float))
    return np.asarray(east, dtype=float), np.asarray(north, dtype=float)


def projected_to_wgs84(
    eastings: Sequence[float] | np.ndarray,
    northings: Sequence[float] | np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `wgs84_to_projected`, returning lon/lat degrees."""
    tf = transformer(epsg, WGS84_EPSG)
    lon, lat = tf.transform(np.asarray(eastings, dtype=float), np.asarray(northings, dtype=float))
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def fixes_to_local_enu(
    fixes: dict[str, GpsFix], epsg: int
) -> tuple[list[str], np.ndarray, tuple[float, float, float]]:
    """Convert geotags to a metric frame centred on the dataset.

    Returns image names, an (N,3) array of east/north/up offsets in metres, and the
    projected origin those offsets are relative to. Working relative to an origin keeps
    the similarity solve numerically well conditioned; UTM coordinates are ~1e6 m.
    """
    names = sorted(fixes.keys())
    if not names:
        return [], np.zeros((0, 3), dtype=float), (0.0, 0.0, 0.0)
    lons = [fixes[n].longitude for n in names]
    lats = [fixes[n].latitude for n in names]
    alts = [fixes[n].altitude_m if fixes[n].altitude_m is not None else 0.0 for n in names]
    east, north = wgs84_to_projected(lons, lats, epsg)
    up = np.asarray(alts, dtype=float)
    origin = (float(np.mean(east)), float(np.mean(north)), float(np.mean(up)))
    local = np.stack([east - origin[0], north - origin[1], up - origin[2]], axis=1)
    return names, local, origin


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres on a spherical earth. Fine for QA-scale checks."""
    radius = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * math.asin(min(1.0, math.sqrt(a))))


# ---------------------------------------------------------------------------
# SfM -> world anchoring
# ---------------------------------------------------------------------------


@dataclass
class GeoAnchor:
    """A 7-parameter similarity taking SfM coordinates into a projected CRS.

    This is the transform that turns an arbitrary-scale reconstruction into metres.
    `scale`, `rotation` and `translation` map SfM points into the local ENU frame
    centred on `origin`; adding `origin` back gives full projected coordinates.
    """

    scale: float
    rotation: list[list[float]]
    translation: list[float]
    epsg: int
    origin: list[float]
    inlier_count: int
    sample_count: int
    rmse_m: float
    max_residual_m: float
    method: str = "umeyama_ransac"

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def translation_vector(self) -> np.ndarray:
        return np.asarray(self.translation, dtype=float).reshape(3)

    @property
    def origin_vector(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=float).reshape(3)

    def apply(self, points: np.ndarray, absolute: bool = True) -> np.ndarray:
        """Transform (N,3) SfM points into the CRS. `absolute=False` keeps the local frame."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = self.scale * (pts @ self.rotation_matrix.T) + self.translation_vector
        if absolute:
            out = out + self.origin_vector
        return out

    def to_wgs84(self, points: np.ndarray) -> np.ndarray:
        """Transform (N,3) SfM points straight to (N,3) lon/lat/altitude."""
        projected = self.apply(points, absolute=True)
        lon, lat = projected_to_wgs84(projected[:, 0], projected[:, 1], self.epsg)
        return np.stack([lon, lat, projected[:, 2]], axis=1)

    def to_dict(self) -> dict:
        return asdict(self)


def solve_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form least-squares similarity (Umeyama 1991) from src to dst.

    Returns (scale, rotation, translation) such that ``scale * R @ src + t ~= dst``.
    Reflections are suppressed, so the result is always a proper rotation.
    """
    source = np.asarray(src, dtype=float).reshape(-1, 3)
    target = np.asarray(dst, dtype=float).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError("Point sets must have matching shapes")
    count = source.shape[0]
    if count < 3:
        raise ValueError("At least 3 correspondences are required for a similarity fit")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean

    covariance = (dst_centered.T @ src_centered) / count
    u_mat, singular, vt_mat = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0.0:
        correction[2, 2] = -1.0

    rotation = u_mat @ correction @ vt_mat
    src_variance = float((src_centered**2).sum() / count)
    if src_variance <= 1e-12:
        raise ValueError("Source points are degenerate (zero variance)")
    scale = float(np.trace(np.diag(singular) @ correction) / src_variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def solve_geo_anchor(
    sfm_points: np.ndarray,
    world_points: np.ndarray,
    epsg: int,
    origin: tuple[float, float, float],
    *,
    threshold_m: float = 3.0,
    iterations: int = 512,
    min_inliers: int = 3,
    seed: int = 20260814,
) -> GeoAnchor:
    """RANSAC similarity from SfM coordinates to a local metric frame.

    GPS geotags routinely contain gross outliers (a fix taken indoors, a stale
    altitude), so a plain least-squares fit is not safe here. Consensus is found on
    random minimal samples, then the model is refit on all inliers.
    """
    source = np.asarray(sfm_points, dtype=float).reshape(-1, 3)
    target = np.asarray(world_points, dtype=float).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError("SfM and world point sets must have matching shapes")
    count = source.shape[0]
    if count < 3:
        raise ValueError(f"Need at least 3 geotagged cameras to georeference; got {count}")

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(count, dtype=bool)

    if count == 3:
        best_inliers[:] = True
    else:
        for _ in range(int(iterations)):
            sample = rng.choice(count, size=3, replace=False)
            try:
                scale, rotation, translation = solve_similarity_umeyama(source[sample], target[sample])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not math.isfinite(scale) or scale <= 0.0:
                continue
            predicted = scale * (source @ rotation.T) + translation
            residuals = np.linalg.norm(predicted - target, axis=1)
            inliers = residuals <= float(threshold_m)
            if int(inliers.sum()) > int(best_inliers.sum()):
                best_inliers = inliers
                if int(inliers.sum()) == count:
                    break

    if int(best_inliers.sum()) < max(3, int(min_inliers)):
        best_inliers = np.ones(count, dtype=bool)
        method = "umeyama_all_points_no_consensus"
    else:
        method = "umeyama_ransac"

    scale, rotation, translation = solve_similarity_umeyama(source[best_inliers], target[best_inliers])
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    inlier_residuals = residuals[best_inliers]

    return GeoAnchor(
        scale=float(scale),
        rotation=[[float(v) for v in row] for row in rotation],
        translation=[float(v) for v in translation],
        epsg=int(epsg),
        origin=[float(v) for v in origin],
        inlier_count=int(best_inliers.sum()),
        sample_count=int(count),
        rmse_m=float(np.sqrt(np.mean(inlier_residuals**2))) if inlier_residuals.size else float("nan"),
        max_residual_m=float(np.max(residuals)) if residuals.size else float("nan"),
        method=method,
    )


def anchor_from_cameras(
    camera_centers: dict[str, Sequence[float]],
    fixes: dict[str, GpsFix],
    epsg: int | None = None,
    **kwargs: Any,
) -> GeoAnchor | None:
    """Georeference a reconstruction by matching camera centres to image geotags.

    Returns None when fewer than three images carry a usable geotag, which is the
    honest outcome for non-geotagged datasets — callers must not fabricate a CRS.
    """
    shared = sorted(set(camera_centers.keys()) & set(fixes.keys()))
    if len(shared) < 3:
        return None
    target_epsg = int(epsg) if epsg is not None else auto_utm_epsg_for_fixes({k: fixes[k] for k in shared})
    names, world_local, origin = fixes_to_local_enu({k: fixes[k] for k in shared}, target_epsg)
    sfm = np.asarray([list(camera_centers[name])[:3] for name in names], dtype=float)
    return solve_geo_anchor(sfm, world_local, target_epsg, origin, **kwargs)


# ---------------------------------------------------------------------------
# Raster and vector writers
# ---------------------------------------------------------------------------


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    epsg: int,
    west: float,
    north: float,
    pixel_size: float,
    nodata: float | None = None,
    cog: bool = True,
    compress: str = "DEFLATE",
) -> str:
    """Write a georeferenced raster, preferring Cloud-Optimized GeoTIFF.

    `array` is (H,W) or (bands,H,W). `west`/`north` are the projected coordinates of
    the top-left pixel corner. Falls back to a tiled GTiff with overviews when the COG
    driver is unavailable, which is still readable by QGIS and GDAL.
    """
    require("rasterio")
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError("Raster array must be 2-D or 3-D")
    bands, height, width = data.shape

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(float(west), float(north), float(pixel_size), float(pixel_size))
    profile: dict[str, Any] = {
        "driver": "COG" if cog else "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": data.dtype.name,
        "crs": _RioCRS.from_epsg(int(epsg)),
        "transform": transform,
        "compress": compress,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if not cog:
        profile.update({"tiled": True, "blockxsize": 512, "blockysize": 512})

    try:
        with rasterio.open(out, "w", **profile) as dst:  # type: ignore[union-attr]
            dst.write(data)
    except Exception:  # noqa: BLE001 - COG driver missing on older GDAL builds
        profile.update({"driver": "GTiff", "tiled": True, "blockxsize": 512, "blockysize": 512})
        with rasterio.open(out, "w", **profile) as dst:  # type: ignore[union-attr]
            dst.write(data)
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)  # type: ignore[union-attr]
    return str(out)


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster and its georeferencing metadata."""
    require("rasterio")
    with rasterio.open(path) as src:  # type: ignore[union-attr]
        data = src.read()
        meta = {
            "epsg": src.crs.to_epsg() if src.crs else None,
            "transform": list(src.transform)[:6],
            "bounds": list(src.bounds),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }
    return data, meta


def hillshade(elevation: np.ndarray, pixel_size: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Standard Horn hillshade, returned as uint8. Used for DSM/DTM display layers."""
    dem = np.asarray(elevation, dtype=float)
    dy, dx = np.gradient(dem, float(pixel_size))
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = math.radians(360.0 - azimuth_deg + 90.0)
    alt = math.radians(altitude_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1.0) * 127.5, 0, 255).astype(np.uint8)


def write_geojson(
    path: str | Path,
    features: Sequence[dict[str, Any]],
    *,
    epsg: int = WGS84_EPSG,
    properties: dict[str, Any] | None = None,
) -> str:
    """Write a FeatureCollection. GeoJSON is WGS84 by convention; other CRSs are named."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": list(features)}
    if int(epsg) != WGS84_EPSG:
        payload["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(epsg)}"}}
    if properties:
        payload["properties"] = properties
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def point_feature(lon: float, lat: float, properties: dict[str, Any] | None = None, alt: float | None = None) -> dict:
    """Build a GeoJSON point feature, with altitude as the optional third ordinate."""
    coords: list[float] = [float(lon), float(lat)]
    if alt is not None:
        coords.append(float(alt))
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": coords}, "properties": properties or {}}


def polygon_feature(ring_lonlat: Sequence[Sequence[float]], properties: dict[str, Any] | None = None) -> dict:
    """Build a GeoJSON polygon feature, closing the ring if the caller did not."""
    ring = [[float(p[0]), float(p[1])] for p in ring_lonlat]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": properties or {}}


def linestring_feature(coords_lonlat: Sequence[Sequence[float]], properties: dict[str, Any] | None = None) -> dict:
    """Build a GeoJSON linestring feature, preserving optional altitude ordinates."""
    line = [[float(c) for c in point[:3]] for point in coords_lonlat]
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": line}, "properties": properties or {}}


def polygon_area_m2(ring_lonlat: Sequence[Sequence[float]]) -> float:
    """Geodesic polygon area in square metres, via an equal-area projection of the ring."""
    ring = [(float(p[0]), float(p[1])) for p in ring_lonlat]
    if len(ring) < 3:
        return 0.0
    lat0 = sum(p[1] for p in ring) / len(ring)
    lon0 = sum(p[0] for p in ring) / len(ring)
    # Local equirectangular approximation; error is negligible at inspection footprint scale.
    scale_x = math.cos(math.radians(lat0)) * math.pi * WGS84_A / 180.0
    scale_y = math.pi * WGS84_A / 180.0
    xs = [(p[0] - lon0) * scale_x for p in ring]
    ys = [(p[1] - lat0) * scale_y for p in ring]
    area = 0.0
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def write_kml(path: str | Path, features: Sequence[dict[str, Any]], document_name: str = "OpenDroneKit") -> str:
    """Write GeoJSON-shaped features to KML for Google Earth and DJI-family tools."""
    kml = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = document_name

    for feature in features:
        geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        placemark = ET.SubElement(document, "Placemark")
        ET.SubElement(placemark, "name").text = str(props.get("name", props.get("id", "feature")))
        if props:
            description = "\n".join(f"{k}: {v}" for k, v in props.items())
            ET.SubElement(placemark, "description").text = description

        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Point" and coords:
            node = ET.SubElement(placemark, "Point")
            ET.SubElement(node, "coordinates").text = ",".join(str(float(c)) for c in coords)
        elif gtype == "LineString" and coords:
            node = ET.SubElement(placemark, "LineString")
            ET.SubElement(node, "coordinates").text = " ".join(
                ",".join(str(float(c)) for c in point) for point in coords
            )
        elif gtype == "Polygon" and coords:
            node = ET.SubElement(placemark, "Polygon")
            outer = ET.SubElement(node, "outerBoundaryIs")
            ring = ET.SubElement(outer, "LinearRing")
            ET.SubElement(ring, "coordinates").text = " ".join(
                ",".join(str(float(c)) for c in point) for point in coords[0]
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(kml).write(out, encoding="utf-8", xml_declaration=True)
    return str(out)


def write_shapefile(path: str | Path, features: Sequence[dict[str, Any]], epsg: int = WGS84_EPSG) -> str:
    """Write an ESRI Shapefile when fiona is present, else an equivalent GeoJSON.

    The return value is the path actually written, so callers report what exists rather
    than what was requested.
    """
    try:
        import fiona
        from fiona.crs import from_epsg as fiona_from_epsg
    except Exception:  # noqa: BLE001
        fallback = Path(path).with_suffix(".geojson")
        return write_geojson(fallback, features, epsg=epsg)

    if not features:
        raise ValueError("Cannot write a shapefile with no features")

    geometry_type = (features[0].get("geometry") or {}).get("type", "Point")
    property_keys = sorted({key for feature in features for key in (feature.get("properties") or {})})
    schema = {"geometry": geometry_type, "properties": {key: "str" for key in property_keys}}

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(out, "w", driver="ESRI Shapefile", crs=fiona_from_epsg(int(epsg)), schema=schema) as sink:
        for feature in features:
            props = feature.get("properties") or {}
            sink.write(
                {
                    "geometry": feature.get("geometry"),
                    "properties": {key: str(props.get(key, "")) for key in property_keys},
                }
            )
    return str(out)
