"""Fast on-site coverage and quality validation utilities.

Image quality (blur, exposure) is measured from the pixels. Coverage and overlap are
measured on the ground: each image is placed from its GPS tag and sized from its camera
geometry and height above ground, and a figure that cannot be computed that way is
reported as unavailable rather than estimated from the images' appearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import cos, radians
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from mission import (
    AssetReferenceFrame,
    MissionConstraints,
    MissionPlanner,
    export_flight_recipe,
    export_geojson,
    export_qgc_wpl,
    load_flight_recipe,
)


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class CoverageValidationConfig:
    """Thresholds used during fast ingest quality validation."""

    sharpness_threshold: float = 120.0
    motion_blur_threshold: float = 0.45
    lens_blur_ratio_threshold: float = 0.5
    exposure_clip_threshold: float = 0.12
    # Kept so existing configurations still load. They thresholded an ORB match ratio,
    # which on textureless aerial imagery stayed under 0.05 for images COLMAP registered
    # without difficulty, so every image was flagged. Overlap is now geometric.
    overlap_threshold: float = 0.14
    expected_overlap_threshold: float = 0.22
    max_gap_suggestions: int = 12
    min_gap_component_px: int = 1600
    thumbnail_width: int = 180
    canvas_size_px: int = 960
    # Share of an image's footprint that its best neighbour must also cover. Below this
    # there is too little common ground to tie the image into the reconstruction.
    min_footprint_overlap: float = 0.3
    # An uncovered patch smaller than this share of one footprint is not reported.
    min_gap_fraction_of_footprint: float = 0.02
    # Height above ground, when the images do not carry it (see image_footprints).
    flight_height_agl_m: float = 0.0
    ground_elevation_m: float | None = None
    terrain_path: str = ""


def _safe_imread(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _resize_with_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if w == width:
        return image
    scale = float(width) / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def _to_gray_small(image: np.ndarray, width: int = 720) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _resize_with_width(gray, width)


def _laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _edge_sharpness_ratio(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 1.0

    cx0 = int(w * 0.25)
    cx1 = int(w * 0.75)
    cy0 = int(h * 0.25)
    cy1 = int(h * 0.75)
    center = gray[cy0:cy1, cx0:cx1]

    top = gray[: max(1, int(h * 0.2)), :]
    bottom = gray[h - max(1, int(h * 0.2)) :, :]
    left = gray[:, : max(1, int(w * 0.2))]
    right = gray[:, w - max(1, int(w * 0.2)) :]

    center_score = _laplacian_var(center) if center.size > 0 else 1.0
    edge_scores = [
        _laplacian_var(top) if top.size > 0 else 0.0,
        _laplacian_var(bottom) if bottom.size > 0 else 0.0,
        _laplacian_var(left) if left.size > 0 else 0.0,
        _laplacian_var(right) if right.size > 0 else 0.0,
    ]
    edge_score = float(np.mean(edge_scores))
    return float(edge_score / max(center_score, 1e-6))


def _motion_blur_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(gx)))
    ey = float(np.mean(np.abs(gy)))
    return float(abs(ex - ey) / max(ex + ey, 1e-6))


def _quality_metrics(gray: np.ndarray, cfg: CoverageValidationConfig) -> dict:
    sharpness = _laplacian_var(gray)
    lens_ratio = _edge_sharpness_ratio(gray)
    motion = _motion_blur_score(gray)

    dark_clip = float(np.mean(gray <= 10))
    bright_clip = float(np.mean(gray >= 245))
    mean_luma = float(gray.mean())
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))

    blur = sharpness < cfg.sharpness_threshold
    lens_blur = lens_ratio < cfg.lens_blur_ratio_threshold
    motion_blur = motion > cfg.motion_blur_threshold and sharpness < cfg.sharpness_threshold * 1.4
    underexp = dark_clip > cfg.exposure_clip_threshold
    overexp = bright_clip > cfg.exposure_clip_threshold

    flags = []
    if blur:
        flags.append("blur")
    if lens_blur:
        flags.append("lens_blur")
    if motion_blur:
        flags.append("motion_blur")
    if underexp:
        flags.append("under_exposed")
    if overexp:
        flags.append("over_exposed")

    return {
        "sharpness": float(sharpness),
        "lens_blur_ratio": float(lens_ratio),
        "motion_blur_score": float(motion),
        "mean_luma": mean_luma,
        "p10_luma": p10,
        "p90_luma": p90,
        "dark_clip_pct": dark_clip,
        "bright_clip_pct": bright_clip,
        "quality_ok": len(flags) == 0,
        "quality_flags": flags,
    }


# ---------------------------------------------------------------------------------
# Where each image actually looked
#
# The first version of this module placed images by phase-correlating each frame against
# the previous one, and fell back to a HARDCODED 180 px shift whenever the correlation was
# weak -- which, on real aerial imagery of fields and roofs, was every pair. Every
# coverage figure it produced (79.4% complete on the Aukerman set, with 12 "gaps") was
# computed on a layout it had invented. Positions now come from the images' own GPS tags,
# footprint sizes from the camera's focal length and sensor width and the height above
# ground, and when any of those is unknown the coverage figure is not computed.
# ---------------------------------------------------------------------------------


@dataclass
class ImageFootprint:
    """The ground one nadir image covered, in a local metric frame."""

    image: str
    easting: float
    northing: float
    height_agl_m: float
    height_source: str
    width_m: float
    length_m: float
    yaw_deg: float | None
    yaw_source: str
    corners: np.ndarray            # (4, 2) local metres; a square when yaw is unknown
    longitude: float
    latitude: float

    @property
    def area_m2(self) -> float:
        return float(abs(cv2.contourArea(self.corners.astype(np.float32))))

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "easting": round(self.easting, 3),
            "northing": round(self.northing, 3),
            "longitude": self.longitude,
            "latitude": self.latitude,
            "height_agl_m": round(self.height_agl_m, 2),
            "height_source": self.height_source,
            "width_m": round(self.width_m, 2),
            "length_m": round(self.length_m, 2),
            "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 1),
            "yaw_source": self.yaw_source,
        }


_XMP_NUMBER = r'="?\s*([-+]?[0-9]*\.?[0-9]+)'


def _xmp_value(raw: bytes, names: Iterable[str]) -> float | None:
    """A numeric XMP field (DJI, senseFly) as attribute or element, or None."""
    import re

    for name in names:
        match = re.search(re.escape(name).encode() + _XMP_NUMBER.encode(), raw)
        if match is None:
            match = re.search(
                rb"<" + re.escape(name).encode() + rb">\s*([-+]?[0-9]*\.?[0-9]+)", raw)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _read_xmp(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            head = handle.read(256 * 1024)
    except OSError:
        return b""
    start = head.find(b"<x:xmpmeta")
    if start < 0:
        return b""
    end = head.find(b"</x:xmpmeta>", start)
    return head[start:end if end > 0 else start + 16384]


def _terrain_sampler(terrain_path: str):
    """lon/lat -> ground elevation from a DEM GeoTIFF, or None where it has no data."""
    import rasterio
    from pyproj import Transformer

    from .terrain_cache import check_terrain_file

    # A feet DEM would put the ground ~2.3x too high and every image below it.
    check_terrain_file(terrain_path)
    dataset = rasterio.open(terrain_path)
    to_raster = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    band = dataset.read(1, masked=True)

    def sample(lon: float, lat: float) -> float | None:
        x, y = to_raster.transform(lon, lat)
        row, col = dataset.index(x, y)
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            return None
        value = band[row, col]
        if np.ma.is_masked(value) or not np.isfinite(float(value)):
            return None
        return float(value)

    return sample, dataset


def image_footprints(
    paths: Iterable[Path],
    *,
    flight_height_agl_m: float | None = None,
    ground_elevation_m: float | None = None,
    terrain_path: str = "",
) -> tuple[list[ImageFootprint], dict]:
    """Ground footprints from EXIF GPS, camera geometry and height above ground.

    Height above ground is taken, per image, from the first source available: the
    drone's own relative altitude (DJI XMP), the GPS altitude minus a terrain model
    sampled under the image, the GPS altitude minus a stated ground elevation, or a
    stated flight height. With none, the image has no footprint, and the reason is
    recorded rather than a height assumed.

    Nadir is assumed: a tilted image covers more ground than this says.
    """
    from . import geo

    paths = list(paths)
    problems: dict[str, list[str]] = {}
    sampler = None
    dataset = None
    if terrain_path:
        try:
            sampler, dataset = _terrain_sampler(terrain_path)
        except Exception as exc:  # noqa: BLE001 - an unreadable DEM is reported, not fatal
            problems.setdefault("terrain", []).append(f"{terrain_path}: {exc}")

    fixes: dict[str, object] = {}
    records: list[tuple[Path, object, object, bytes]] = []
    for path in paths:
        fix = geo.read_exif_gps(path)
        if fix is None:
            problems.setdefault("no_gps", []).append(path.name)
            continue
        camera = geo.read_exif_camera(path)
        records.append((path, fix, camera, _read_xmp(path)))
        fixes[path.name] = fix

    footprints: list[ImageFootprint] = []
    if not records:
        if dataset is not None:
            dataset.close()
        return footprints, {"problems": problems, "epsg": None}

    epsg = geo.auto_utm_epsg_for_fixes(list(fixes.values()))
    lons = [r[1].longitude for r in records]
    lats = [r[1].latitude for r in records]
    east, north = geo.wgs84_to_projected(lons, lats, epsg)
    # Where the images that get no footprint were taken: their ground is not known to
    # be covered or uncovered, and the caller must not call it a gap.
    unplaced: dict[str, tuple[float, float]] = {}

    for (path, fix, camera, xmp), x, y in zip(records, east, north):
        height, source = None, ""
        relative = _xmp_value(xmp, ["drone-dji:RelativeAltitude", "RelativeAltitude"])
        if relative is not None and relative > 0:
            height, source = relative, "xmp_relative_altitude"
        elif sampler is not None and fix.altitude_m is not None:
            ground = sampler(fix.longitude, fix.latitude)
            if ground is not None:
                height, source = fix.altitude_m - ground, "gps_altitude_minus_terrain"
        if height is None and ground_elevation_m is not None and fix.altitude_m is not None:
            height, source = fix.altitude_m - float(ground_elevation_m), "gps_altitude_minus_ground"
        if height is None and flight_height_agl_m:
            height, source = float(flight_height_agl_m), "stated_flight_height"
        if height is None or not np.isfinite(height) or height <= 0:
            problems.setdefault("no_height", []).append(path.name)
            unplaced[path.name] = (float(x), float(y))
            continue

        if not (camera.focal_length_mm and camera.sensor_width_mm and camera.width and camera.height):
            if camera.focal_35mm and camera.width and camera.height:
                width_m = height * 36.0 / float(camera.focal_35mm)
            else:
                problems.setdefault("no_camera_geometry", []).append(path.name)
                unplaced[path.name] = (float(x), float(y))
                continue
        else:
            width_m = height * float(camera.sensor_width_mm) / float(camera.focal_length_mm)
        length_m = width_m * float(camera.height) / float(camera.width)

        yaw = _xmp_value(xmp, ["drone-dji:GimbalYawDegree", "drone-dji:FlightYawDegree",
                               "Camera:Yaw"])
        yaw_source = "xmp" if yaw is not None else ""
        if yaw is None and fix.yaw_deg is not None:
            yaw, yaw_source = fix.yaw_deg, "exif_gps_img_direction"

        if yaw is None:
            # Orientation unknown: the only square every orientation of the image is
            # sure to cover is the one inscribed in its short side. Coverage drawn this
            # way is an UNDER-estimate, never an over-estimate.
            half = min(width_m, length_m) / 2.0
            corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        else:
            theta = np.radians(float(yaw))
            # Image "up" points along the heading (clockwise from north); x across it.
            across = np.array([np.cos(theta), -np.sin(theta)])
            along = np.array([np.sin(theta), np.cos(theta)])
            hw, hl = width_m / 2.0, length_m / 2.0
            corners = np.array([
                -hw * across - hl * along, hw * across - hl * along,
                hw * across + hl * along, -hw * across + hl * along,
            ])
        footprints.append(ImageFootprint(
            image=path.name, easting=float(x), northing=float(y), height_agl_m=float(height),
            height_source=source, width_m=float(width_m), length_m=float(length_m),
            yaw_deg=None if yaw is None else float(yaw), yaw_source=yaw_source or "unknown",
            corners=corners + np.array([float(x), float(y)]),
            longitude=float(fix.longitude), latitude=float(fix.latitude),
        ))
    if dataset is not None:
        dataset.close()
    return footprints, {"problems": problems, "epsg": int(epsg), "unplaced": unplaced}


def _overlap_fraction(a: ImageFootprint, b: ImageFootprint) -> float:
    """Share of footprint ``a`` that ``b`` also covers."""
    area, _ = cv2.intersectConvexConvex(a.corners.astype(np.float32), b.corners.astype(np.float32))
    return float(area) / max(a.area_m2, 1e-9)


def _suggestion_hull_local(points_xy: np.ndarray, margin_m: float = 6.0) -> np.ndarray:
    if len(points_xy) == 0:
        return np.asarray([[0.0, 0.0], [margin_m, 0.0], [0.0, margin_m]], dtype=np.float64)

    if len(points_xy) >= 3:
        hull = cv2.convexHull(points_xy.astype(np.float32))
        hull = hull.reshape(-1, 2).astype(np.float64)
    elif len(points_xy) == 2:
        a = points_xy[0]
        b = points_xy[1]
        mid = (a + b) * 0.5
        v = b - a
        n = np.array([-v[1], v[0]], dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            n = np.array([0.0, 1.0], dtype=np.float64)
        else:
            n = n / norm
        hull = np.vstack([a, b, mid + n * margin_m])
    else:
        c = points_xy[0]
        hull = np.vstack(
            [
                c + np.array([-margin_m, -margin_m]),
                c + np.array([margin_m, -margin_m]),
                c + np.array([margin_m, margin_m]),
                c + np.array([-margin_m, margin_m]),
            ]
        )

    center = hull.mean(axis=0)
    expanded = center + (hull - center) * 1.15
    return expanded.astype(np.float64)


class CoverageValidator:
    """Fast ingest validator for quality, overlap, and on-site re-fly guidance."""

    def __init__(self, config: CoverageValidationConfig | None = None):
        self.config = config or CoverageValidationConfig()
        self._planner = MissionPlanner()

    def validate(
        self,
        image_paths: Iterable[Path],
        output_dir: str | Path,
        reference_recipe_path: str = "",
        *,
        aoi_lonlat: list[list[float]] | None = None,
    ) -> dict:
        cfg = self.config
        paths = [Path(p) for p in image_paths]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        skipped_images: list[str] = []
        thumbs: dict[str, np.ndarray] = {}
        readable: list[Path] = []
        for p in paths:
            img = _safe_imread(p)
            if img is None:
                skipped_images.append(p.name)
                continue
            q = _quality_metrics(_to_gray_small(img), cfg)
            rows.append({
                "image": p.name,
                "valid": True,
                **q,
                "overlap_prev": None,
                "overlap_next": None,
                "overlap_best": None,
                "low_overlap": False,
            })
            thumbs[p.name] = _resize_with_width(img, cfg.thumbnail_width)
            readable.append(p)

        footprints, layout_meta = image_footprints(
            readable,
            flight_height_agl_m=cfg.flight_height_agl_m or None,
            ground_elevation_m=cfg.ground_elevation_m,
            terrain_path=cfg.terrain_path,
        )
        by_name = {fp.image: fp for fp in footprints}
        problems = layout_meta.get("problems", {})

        # ---- overlap, from footprint geometry --------------------------------------
        overlap_edges: list[dict] = []
        missing_pairs: list[dict] = []
        # Consecutive captures, in capture order. A pair is measured only when BOTH
        # images are on the ground: pairing across an unplaced frame compared 235 with
        # 237 (1% overlap) while 236, which lies between them, was never looked at, and
        # sent the operator to re-fly ground that had been photographed.
        pairs_not_assessed: list[dict] = []
        sequence: list[tuple[ImageFootprint, ImageFootprint]] = []
        beside_unplaced: set[str] = set()
        for before, after in zip(rows, rows[1:]):
            a, b = by_name.get(before["image"]), by_name.get(after["image"])
            if a is not None and b is not None:
                sequence.append((a, b))
                continue
            missing = [r["image"] for r in (before, after) if r["image"] not in by_name]
            pairs_not_assessed.append({
                "image_prev": before["image"], "image_curr": after["image"],
                "reason": "not placed on the ground: " + ", ".join(missing),
            })
            beside_unplaced.update(r["image"] for r in (before, after) if r["image"] in by_name)
        for prev, curr in sequence:
            forward = _overlap_fraction(prev, curr)
            edge = {
                "image_prev": prev.image,
                "image_curr": curr.image,
                "overlap": round(forward, 4),
                "distance_m": round(float(np.hypot(curr.easting - prev.easting,
                                                   curr.northing - prev.northing)), 2),
            }
            overlap_edges.append(edge)
            if forward < cfg.min_footprint_overlap:
                missing_pairs.append(edge)
        row_by_name = {r["image"]: r for r in rows}
        for prev, curr in sequence:
            share = _overlap_fraction(prev, curr)
            row_by_name[prev.image]["overlap_next"] = round(share, 4)
            row_by_name[curr.image]["overlap_prev"] = round(_overlap_fraction(curr, prev), 4)
        for fp in footprints:
            best = max((_overlap_fraction(fp, other) for other in footprints if other is not fp),
                       default=0.0)
            row = row_by_name[fp.image]
            row["overlap_best"] = round(best, 4)
            row["footprint"] = fp.to_dict()
            if best < cfg.min_footprint_overlap and fp.image in beside_unplaced:
                # Its neighbour in the capture sequence has no footprint, and may well
                # be the image that overlaps it. Not assessed, not a gap.
                row["overlap_note"] = ("Next to an image that could not be placed; its "
                                       "overlap is not assessed.")
            elif best < cfg.min_footprint_overlap:
                row["low_overlap"] = True
                row["quality_flags"] = list(row.get("quality_flags", [])) + ["overlap_gap"]
                row["quality_ok"] = False
        for row in rows:
            row["overlap_assessed"] = row["image"] in by_name

        # ---- coverage over the surveyed area ------------------------------------------
        coverage = self._coverage(footprints, layout_meta.get("epsg"), aoi_lonlat, out_dir, thumbs)
        unplaced = layout_meta.get("unplaced", {})
        suggestions, unassessed_gaps = self._gap_suggestions(
            rows, footprints, coverage, missing_pairs, layout_meta.get("epsg"), unplaced)

        quality_fail = sum(1 for row in rows if not bool(row.get("quality_ok", False)))
        layout = {
            "available": bool(coverage.get("available")),
            "basis": "exif_gps_and_camera_geometry",
            "epsg": layout_meta.get("epsg"),
            "images_placed": len(footprints),
            "images_not_placed": {k: v for k, v in problems.items() if k != "terrain"},
            "assumptions": [
                "Images are treated as nadir; an oblique image covers more ground than "
                "its footprint here.",
            ],
        }
        if problems.get("terrain"):
            layout["terrain_error"] = problems["terrain"]
        if unplaced:
            layout["not_assessed_note"] = (
                f"{len(unplaced)} image(s) with GPS could not be placed on the ground. "
                "Overlap with them is not assessed, coverage figures count only the "
                "placed images, and uncovered ground near where they were taken is "
                "listed under unassessed_gaps rather than suggested for a re-fly.")
        if not coverage.get("available"):
            layout["reason"] = coverage.get("reason", "")
        if any(fp.yaw_deg is None for fp in footprints):
            layout["assumptions"].append(
                "Some images carry no heading, so their footprint is drawn as the square "
                "inscribed in the short side. Coverage for them is under-stated, never "
                "over-stated."
            )

        result = {
            "config": asdict(cfg),
            "images": rows,
            "skipped_images": skipped_images,
            "layout": layout,
            "footprints": [fp.to_dict() for fp in footprints],
            "overlap_pairs": overlap_edges,
            "missing_pairs": missing_pairs,
            "pairs_not_assessed": pairs_not_assessed,
            "unassessed_gaps": unassessed_gaps,
            "quality_fail_count": int(quality_fail),
            "coverage_completion_pct": coverage.get("completion_pct"),
            "stereo_coverage_pct": coverage.get("stereo_pct"),
            "coverage_area_m2": coverage.get("area_m2"),
            "coverage_extent": coverage.get("extent"),
            "preview_model_path": coverage.get("mosaic_path", ""),
            "coverage_heatmap_path": coverage.get("heatmap_path", ""),
            "coverage_expected_path": coverage.get("expected_path", ""),
            "gap_suggestions": suggestions,
            "gap_suggestion_count": int(len(suggestions)),
            "gap_mission": {},
        }

        gap_payload = self._build_gap_mission(result, reference_recipe_path=reference_recipe_path,
                                              output_dir=out_dir)
        if gap_payload:
            result["gap_mission"] = gap_payload

        quality_json_path = out_dir / "coverage_validation.json"
        with quality_json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        result["coverage_json_path"] = str(quality_json_path)
        return result

    # ------------------------------------------------------------------ coverage
    def _coverage(self, footprints: list[ImageFootprint], epsg: int | None,
                  aoi_lonlat: list[list[float]] | None, out_dir: Path,
                  thumbs: dict[str, np.ndarray]) -> dict:
        if len(footprints) < 2 or epsg is None:
            reasons = []
            if not footprints:
                reasons.append(
                    "No image could be placed on the ground: coverage needs each image's "
                    "GPS position, its camera's focal length and sensor width, and its "
                    "height above ground (DJI relative altitude, a terrain model, the site's "
                    "ground elevation, or the flight height)."
                )
            else:
                reasons.append("Only one image could be placed; coverage needs at least two.")
            return {"available": False, "reason": " ".join(reasons),
                    "completion_pct": None, "stereo_pct": None}

        from . import geo

        mean_side = float(np.mean([min(fp.width_m, fp.length_m) for fp in footprints]))
        cell = max(mean_side / 40.0, 0.05)
        if aoi_lonlat and len(aoi_lonlat) >= 3:
            ex, ny = geo.wgs84_to_projected([p[0] for p in aoi_lonlat],
                                            [p[1] for p in aoi_lonlat], epsg)
            expected_poly = np.column_stack([ex, ny])
            extent = "area_of_interest"
        else:
            # The block the cameras flew: the hull of their centres. Footprints reach past
            # it, so its edges are covered; what can be uncovered is inside -- a missed
            # strip between flight lines, a dropped image.
            centres = np.array([[fp.easting, fp.northing] for fp in footprints], dtype=np.float32)
            hull = cv2.convexHull(centres).reshape(-1, 2).astype(np.float64)
            if len(hull) < 3 or cv2.contourArea(hull.astype(np.float32)) < cell * cell * 4:
                corners = np.vstack([fp.corners for fp in footprints]).astype(np.float32)
                hull = cv2.convexHull(corners).reshape(-1, 2).astype(np.float64)
            expected_poly = hull
            extent = "hull_of_image_centres"

        all_pts = np.vstack([expected_poly] + [fp.corners for fp in footprints])
        min_x, min_y = all_pts.min(axis=0) - cell
        max_x, max_y = all_pts.max(axis=0) + cell
        cols = int(np.ceil((max_x - min_x) / cell)) + 1
        rows = int(np.ceil((max_y - min_y) / cell)) + 1
        scale_down = max(1.0, max(cols, rows) / 2000.0)
        if scale_down > 1.0:
            cell *= scale_down
            cols = int(np.ceil((max_x - min_x) / cell)) + 1
            rows = int(np.ceil((max_y - min_y) / cell)) + 1

        def to_px(points: np.ndarray) -> np.ndarray:
            px = np.empty_like(points, dtype=np.float64)
            px[:, 0] = (points[:, 0] - min_x) / cell
            px[:, 1] = (max_y - points[:, 1]) / cell
            return np.round(px).astype(np.int32)

        counts = np.zeros((rows, cols), dtype=np.uint16)
        for fp in footprints:
            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.fillPoly(mask, [to_px(fp.corners)], 1)
            counts += mask
        expected = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillPoly(expected, [to_px(expected_poly)], 255)
        inside = expected > 0
        total = int(inside.sum())
        completion = float((counts[inside] >= 1).mean() * 100.0) if total else None
        stereo = float((counts[inside] >= 2).mean() * 100.0) if total else None

        # Artifacts, on the real grid.
        norm = np.clip(counts.astype(np.float32) * (255.0 / max(1, int(counts.max()))), 0, 255)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[(inside) & (counts == 0)] = (0, 0, 255)
        heat[~inside & (counts == 0)] = (0, 0, 0)
        mosaic = np.zeros((rows, cols, 3), dtype=np.uint8)
        for fp in footprints:
            thumb = thumbs.get(fp.image)
            if thumb is None:
                continue
            th, tw = thumb.shape[:2]
            src = np.float32([[0, th], [tw, th], [tw, 0], [0, 0]])
            dst = to_px(fp.corners).astype(np.float32)
            if fp.yaw_deg is None:
                # Orientation unknown: centre crop to the inscribed square.
                side = min(th, tw)
                y0, x0 = (th - side) // 2, (tw - side) // 2
                src = np.float32([[x0, y0 + side], [x0 + side, y0 + side],
                                  [x0 + side, y0], [x0, y0]])
            warp = cv2.getPerspectiveTransform(src, dst)
            placed = cv2.warpPerspective(thumb, warp, (cols, rows))
            covered = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), warp, (cols, rows)) > 0
            mosaic[covered] = placed[covered]
        heatmap_path = out_dir / "coverage_heatmap.png"
        expected_path = out_dir / "coverage_expected_mask.png"
        mosaic_path = out_dir / "preview_mosaic.png"
        cv2.imwrite(str(heatmap_path), heat)
        cv2.imwrite(str(expected_path), expected)
        cv2.imwrite(str(mosaic_path), mosaic)

        return {
            "available": True,
            "completion_pct": None if completion is None else round(completion, 2),
            "stereo_pct": None if stereo is None else round(stereo, 2),
            "area_m2": round(total * cell * cell, 1),
            "extent": extent,
            "cell_m": cell,
            "counts": counts,
            "expected": inside,
            "origin": (float(min_x), float(max_y)),
            "heatmap_path": str(heatmap_path),
            "expected_path": str(expected_path),
            "mosaic_path": str(mosaic_path),
            "mean_footprint_m2": float(np.mean([fp.area_m2 for fp in footprints])),
        }

    def _gap_suggestions(self, rows: list[dict], footprints: list[ImageFootprint],
                         coverage: dict, missing_pairs: list[dict],
                         epsg: int | None,
                         unplaced: dict[str, tuple[float, float]] | None = None,
                         ) -> tuple[list[dict], list[dict]]:
        from . import geo

        suggestions: list[dict] = []
        unassessed: list[dict] = []
        unplaced = unplaced or {}

        def located(x: float, y: float) -> dict:
            lon, lat = geo.projected_to_wgs84([x], [y], int(epsg))
            return {"easting": round(x, 2), "northing": round(y, 2),
                    "lonlat": [float(lon[0]), float(lat[0])]}

        if coverage.get("available"):
            counts, inside = coverage["counts"], coverage["expected"]
            cell = coverage["cell_m"]
            min_x, max_y = coverage["origin"]
            gap = ((inside) & (counts == 0)).astype(np.uint8) * 255
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(gap, connectivity=8)
            smallest = self.config.min_gap_fraction_of_footprint * coverage["mean_footprint_m2"]
            # An unplaced image covers roughly a footprint around where it was taken.
            reach = float(np.sqrt(coverage["mean_footprint_m2"]))
            for index in range(1, n):
                area_m2 = float(stats[index, cv2.CC_STAT_AREA]) * cell * cell
                if area_m2 < smallest:
                    continue
                cx, cy = centroids[index]
                near = []
                if unplaced:
                    rr, cc = np.nonzero(labels == index)
                    gx, gy = min_x + (cc + 0.5) * cell, max_y - (rr + 0.5) * cell
                    near = [name for name, (ux, uy) in unplaced.items()
                            if float(np.min(np.hypot(gx - ux, gy - uy))) <= reach]
                if near:
                    unassessed.append({
                        "area_m2": round(area_m2, 1),
                        **located(min_x + cx * cell, max_y - cy * cell),
                        "reason": "images taken here could not be placed on the ground",
                        "related_images": sorted(near),
                    })
                    continue
                suggestions.append({
                    "kind": "coverage_gap",
                    "priority": area_m2,
                    "area_m2": round(area_m2, 1),
                    **located(min_x + cx * cell, max_y - cy * cell),
                    "reasons": ["coverage_gap"],
                    "related_images": [],
                })
        by_name = {fp.image: fp for fp in footprints}
        for miss in missing_pairs:
            a, b = by_name.get(miss["image_prev"]), by_name.get(miss["image_curr"])
            if a is None or b is None:
                continue
            suggestions.append({
                "kind": "overlap_gap",
                "priority": (1.0 - float(miss["overlap"])) * float(coverage.get("mean_footprint_m2", 1.0)),
                **located((a.easting + b.easting) / 2.0, (a.northing + b.northing) / 2.0),
                "reasons": ["overlap_gap"],
                "related_images": [a.image, b.image],
            })
        for row in rows:
            if bool(row.get("quality_ok", True)) or row.get("quality_flags") == ["overlap_gap"]:
                continue
            fp = by_name.get(row["image"])
            item = {
                "kind": "quality_retake",
                "priority": float(len(row.get("quality_flags", []))),
                "reasons": [f for f in row.get("quality_flags", []) if f != "overlap_gap"],
                "related_images": [row["image"]],
            }
            if fp is not None:
                item.update(located(fp.easting, fp.northing))
            suggestions.append(item)

        suggestions.sort(key=lambda s: float(s.get("priority", 0.0)), reverse=True)
        return suggestions[: self.config.max_gap_suggestions], unassessed

    def _build_gap_mission(
        self,
        result: dict,
        reference_recipe_path: str,
        output_dir: Path,
    ) -> dict:
        """A re-fly mission over the located gaps, in the reference recipe's settings."""
        if not reference_recipe_path:
            return {}
        recipe_path = Path(reference_recipe_path)
        if not recipe_path.exists():
            return {}

        suggestions = [s for s in result.get("gap_suggestions", []) if s.get("lonlat")]
        if not suggestions:
            return {}

        try:
            base_recipe = load_flight_recipe(recipe_path)
        except Exception:
            return {}

        # The suggestions are real ground positions now, so the re-fly area is simply
        # their hull, grown by the stand-off -- no fitting of an invented layout onto the
        # recipe's frame, which is what the earlier version had to do.
        from . import geo

        epsg = int(result.get("layout", {}).get("epsg") or geo.auto_utm_epsg(
            suggestions[0]["lonlat"][1], suggestions[0]["lonlat"][0]))
        xs, ys = geo.wgs84_to_projected([s["lonlat"][0] for s in suggestions],
                                        [s["lonlat"][1] for s in suggestions], epsg)
        margin = max(5.0, float(base_recipe.constraints.standoff_m))
        hull_local = _suggestion_hull_local(np.column_stack([xs, ys]), margin_m=margin)
        lon, lat = geo.projected_to_wgs84(hull_local[:, 0], hull_local[:, 1], epsg)
        polygon_lonlat = [[float(a), float(b)] for a, b in zip(lon, lat)]

        constraints = MissionConstraints(
            geofence=base_recipe.constraints.geofence,
            min_altitude_m=base_recipe.constraints.min_altitude_m,
            max_altitude_m=base_recipe.constraints.max_altitude_m,
            standoff_m=base_recipe.constraints.standoff_m,
            rth_altitude_m=base_recipe.constraints.rth_altitude_m,
            rth_action=base_recipe.constraints.rth_action,
            obstacle_avoidance_profile=base_recipe.constraints.obstacle_avoidance_profile,
        )

        gap_recipe = self._planner.build_flight_recipe(
            polygon_lonlat=polygon_lonlat,
            altitude_m=float(base_recipe.metadata.get("altitude_m", max(base_recipe.constraints.min_altitude_m, 40.0))),
            front_overlap_pct=base_recipe.coverage.front_overlap_pct,
            side_overlap_pct=base_recipe.coverage.side_overlap_pct,
            mode="smart_adaptive",
            camera=str(base_recipe.metadata.get("camera", "custom")),
            constraints=constraints,
            asset_frame=base_recipe.asset_frame,
            recipe_version=int(base_recipe.version) + 1,
            metadata={
                "parent_recipe_id": base_recipe.recipe_id,
                "purpose": "coverage_gap_refly",
                "generated_by": "coverage_validator",
                "gap_suggestion_count": int(len(suggestions)),
            },
        )
        gap_plan = self._planner.compile_recipe(gap_recipe, repeat_enabled=True)

        recipe_out = output_dir / "gap_mission_recipe.json"
        qgc_out = output_dir / "gap_mission.waypoints"
        geojson_out = output_dir / "gap_mission.geojson"

        export_flight_recipe(recipe_out, gap_recipe)
        export_qgc_wpl(qgc_out, gap_plan)
        export_geojson(geojson_out, gap_plan)

        suggestions_out = output_dir / "gap_suggestions.json"
        with suggestions_out.open("w", encoding="utf-8") as handle:
            json.dump(suggestions, handle, indent=2)

        return {
            "base_recipe_path": str(recipe_path),
            "recipe_path": str(recipe_out),
            "qgc_wpl_path": str(qgc_out),
            "geojson_path": str(geojson_out),
            "suggestions_path": str(suggestions_out),
            "suggestion_count": int(len(suggestions)),
            "waypoint_count": int(len(gap_plan.waypoints)),
            "autopilot_command_count": int(len(gap_plan.autopilot_commands)),
        }
