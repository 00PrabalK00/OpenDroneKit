"""Back-project 2-D defect detections onto the reconstructed surface.

A defect found in an image is only useful for asset management once it has a location
and a size on the structure. This module casts a ray from the recovering camera through
each defect pixel, intersects it with the DSM surface, converts the hit to the output
CRS, and merges detections of the same physical defect seen from several views.

The result is a GeoJSON layer with real coordinates and real square-metre areas, which
is what the digital twin, the reports, and any GIS consumer actually need.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from . import geo


@dataclass
class CameraPose:
    """A registered view with everything needed to cast rays into the scene."""

    image: str
    rotation: np.ndarray  # 3x3, world -> camera
    translation: np.ndarray  # 3, world -> camera
    intrinsics: np.ndarray  # 3x3
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world (SfM) coordinates."""
        return -self.rotation.T @ self.translation

    def ray_direction(self, x: float, y: float) -> np.ndarray:
        """Unit ray in world coordinates through image pixel (x, y)."""
        k_inv = np.linalg.inv(self.intrinsics)
        camera_ray = k_inv @ np.array([float(x), float(y), 1.0])
        world_ray = self.rotation.T @ camera_ray
        norm = float(np.linalg.norm(world_ray)) or 1.0
        return world_ray / norm

    def project(self, world_point: np.ndarray) -> tuple[float, float, float]:
        """Project a world point back to pixel coordinates plus its depth."""
        camera_point = self.rotation @ np.asarray(world_point, dtype=float).reshape(3) + self.translation
        depth = float(camera_point[2])
        if abs(depth) < 1e-9:
            return float("nan"), float("nan"), depth
        pixel = self.intrinsics @ camera_point
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), depth


@dataclass
class SurfaceModel:
    """A georeferenced elevation grid used as the ray-intersection target."""

    elevation: np.ndarray
    west: float
    north: float
    pixel_size: float
    epsg: int

    @property
    def height(self) -> int:
        return int(self.elevation.shape[0])

    @property
    def width(self) -> int:
        return int(self.elevation.shape[1])

    def sample(self, east: float, north: float) -> float:
        """Elevation at a projected coordinate, or NaN outside the grid / in a hole."""
        col = int((east - self.west) / self.pixel_size)
        row = int((self.north - north) / self.pixel_size)
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return float("nan")
        value = float(self.elevation[row, col])
        return value if math.isfinite(value) else float("nan")

    @property
    def z_range(self) -> tuple[float, float]:
        finite = self.elevation[np.isfinite(self.elevation)]
        if finite.size == 0:
            return 0.0, 0.0
        return float(finite.min()), float(finite.max())


@dataclass
class ProjectedDefect:
    """One physical defect, located on the asset and measured in metric units."""

    defect_id: str
    defect_type: str
    longitude: float
    latitude: float
    altitude_m: float
    easting: float
    northing: float
    # None when the defect was located but its outline could not be measured (part of
    # it fell in a DSM hole): a partial outline under-states the size, and dropping the
    # defect altogether hid it from the map.
    area_m2: float | None
    length_m: float | None
    width_m: float | None
    severity: str
    confidence: float
    observation_count: int
    source_images: list[str] = field(default_factory=list)
    boundary_lonlat: list[list[float]] = field(default_factory=list)
    # True when part of the outline met a DSM cell bridged from its neighbours rather
    # than one the reconstruction measured.
    surface_interpolated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature(self) -> dict:
        """GeoJSON feature: polygon when a boundary was recovered, else a point."""
        properties = {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "area_m2": None if self.area_m2 is None else round(self.area_m2, 6),
            "length_m": None if self.length_m is None else round(self.length_m, 4),
            "width_m": None if self.width_m is None else round(self.width_m, 4),
            "sized": self.area_m2 is not None,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "source_images": ", ".join(self.source_images[:8]),
            "altitude_m": round(self.altitude_m, 3),
            # The located centre, whatever geometry the feature carries. A polygon
            # layer had no point coordinates, so risk ranking reported every defect's
            # position as null.
            "longitude": round(self.longitude, 8),
            "latitude": round(self.latitude, 8),
            "easting": round(self.easting, 3),
            "northing": round(self.northing, 3),
            "surface_interpolated": self.surface_interpolated,
        }
        if len(self.boundary_lonlat) >= 3:
            return geo.polygon_feature(self.boundary_lonlat, properties)
        return geo.point_feature(self.longitude, self.latitude, properties, alt=self.altitude_m)


def load_camera_poses(path: str | Path) -> dict[str, CameraPose]:
    """Load poses written by the COLMAP engine's `camera_poses.json`.

    Views lacking full extrinsics or intrinsics are skipped: a partial pose cannot cast
    a correct ray, and guessing one would silently misplace defects.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    poses: dict[str, CameraPose] = {}
    for record in payload.get("cameras", []):
        rotation = record.get("rotation_cam_from_world")
        translation = record.get("translation_cam_from_world")
        intrinsics = record.get("intrinsics")
        if not (rotation and translation and intrinsics):
            continue
        name = Path(str(record.get("image", ""))).name
        if not name:
            continue
        poses[name] = CameraPose(
            image=name,
            rotation=np.asarray(rotation, dtype=float).reshape(3, 3),
            translation=np.asarray(translation, dtype=float).reshape(3),
            intrinsics=np.asarray(intrinsics, dtype=float).reshape(3, 3),
            width=int(record.get("width", 0) or 0),
            height=int(record.get("height", 0) or 0),
        )
    return poses


def load_surface(dsm_path: str | Path) -> SurfaceModel:
    """Load a DSM GeoTIFF as the ray-intersection surface."""
    data, meta = geo.read_geotiff(dsm_path)
    elevation = np.asarray(data[0], dtype=float)
    nodata = meta.get("nodata")
    if nodata is not None:
        elevation = np.where(np.isclose(elevation, float(nodata)), np.nan, elevation)
    transform = meta["transform"]
    return SurfaceModel(
        elevation=elevation,
        west=float(transform[2]),
        north=float(transform[5]),
        pixel_size=abs(float(transform[0])),
        epsg=int(meta.get("epsg") or geo.WGS84_EPSG),
    )


# A ray is allowed to meet an interpolated cell only this many cells from measured data.
# Wider holes are not bridged: the surface there is unknown, not merely unsampled.
MAX_BRIDGED_CELLS = 3


def bridge_small_holes(surface: SurfaceModel, max_cells: int = MAX_BRIDGED_CELLS
                       ) -> tuple[SurfaceModel, np.ndarray]:
    """A copy of the surface with holes up to ``max_cells`` wide filled from neighbours.

    A DSM rasterised from a sparse cloud is mostly holes -- 70% of the Aukerman cells --
    and a ray aimed at a defect usually lands in one. Filling each empty cell with the
    mean of its measured neighbours, a ring at a time, bridges the gaps between samples
    without inventing a surface across a genuinely unobserved area. Returns the filled
    surface and a mask of which cells were filled, so every hit on one can be labelled.
    """
    elevation = np.array(surface.elevation, dtype=float)
    measured = np.isfinite(elevation)
    for _ in range(max(0, int(max_cells))):
        holes = ~np.isfinite(elevation)
        if not holes.any():
            break
        padded = np.pad(elevation, 1, constant_values=np.nan)
        stack = np.stack([
            padded[1 + dy:padded.shape[0] - 1 + dy, 1 + dx:padded.shape[1] - 1 + dx]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx
        ])
        known = np.isfinite(stack)
        counts = known.sum(axis=0)
        sums = np.where(known, stack, 0.0).sum(axis=0)
        fill = holes & (counts > 0)
        if not fill.any():
            break
        elevation[fill] = sums[fill] / counts[fill]
    filled = SurfaceModel(elevation=elevation, west=surface.west, north=surface.north,
                          pixel_size=surface.pixel_size, epsg=surface.epsg)
    return filled, np.isfinite(elevation) & ~measured


def intersect_surface(
    origin: np.ndarray,
    direction: np.ndarray,
    surface: SurfaceModel,
    *,
    max_distance: float | None = None,
    step: float | None = None,
) -> np.ndarray | None:
    """March a ray until it crosses the elevation surface, then refine by bisection.

    Ray marching is used rather than an analytic solve because the DSM is an arbitrary
    height field with holes. The initial step is tied to the raster resolution so no
    cell is skipped over.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction = direction / norm

    z_min, z_max = surface.z_range
    diagonal = math.hypot(surface.width * surface.pixel_size, surface.height * surface.pixel_size)
    limit = float(max_distance if max_distance is not None else diagonal * 2.0 + abs(z_max - z_min) * 4.0)
    stride = float(step if step is not None else max(surface.pixel_size, 1e-3))

    previous_gap: float | None = None
    previous_t = 0.0
    travelled = stride
    while travelled <= limit:
        point = origin + direction * travelled
        ground = surface.sample(point[0], point[1])
        if not math.isfinite(ground):
            # A hole in the DSM breaks the bracket. Carrying the last gap across it let
            # a ray that entered a hole above the surface and left it below be "hit" in
            # mid-air -- 15 m above the real DSM on the Aukerman reconstruction. A
            # crossing only counts between two consecutive samples that both have data.
            previous_gap = None
            travelled += stride
            continue
        gap = float(point[2] - ground)
        if previous_gap is not None and previous_gap > 0.0 >= gap:
            low, high = previous_t, travelled
            for _ in range(24):
                mid = 0.5 * (low + high)
                probe = origin + direction * mid
                ground_mid = surface.sample(probe[0], probe[1])
                if not math.isfinite(ground_mid):
                    # A hole inside the bracket: the crossing cannot be refined any
                    # further, so settle on the end that is known to be at or under
                    # the surface rather than the midpoint, which may be in the air.
                    low = high
                    break
                if probe[2] - ground_mid > 0.0:
                    low = mid
                else:
                    high = mid
            hit = origin + direction * (0.5 * (low + high))
            ground_hit = surface.sample(hit[0], hit[1])
            if not math.isfinite(ground_hit):
                hit = origin + direction * high
                ground_hit = surface.sample(hit[0], hit[1])
            if not math.isfinite(ground_hit):
                return None
            # The point is ON the surface: its height is the DSM's, not the ray's
            # (they agree to within the bisection tolerance when refinement finished).
            return np.array([hit[0], hit[1], ground_hit], dtype=float)
        previous_gap = gap
        previous_t = travelled
        travelled += stride
    return None


def _mask_contours(mask: np.ndarray, min_area_px: int = 12) -> list[np.ndarray]:
    """Extract defect blob outlines from a binary mask."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) >= float(min_area_px)]


def _shape_metrics(points_world: np.ndarray) -> tuple[float, float, float]:
    """Area, long axis, and short axis in metres for a projected defect boundary.

    Extents come from the principal axes of the boundary, which handles the diagonal
    cracks that a bounding box would systematically over-measure.
    """
    if points_world.shape[0] < 3:
        return 0.0, 0.0, 0.0
    planar = points_world[:, :2].astype(np.float64)
    centered = planar - planar.mean(axis=0)
    try:
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt.T
    except np.linalg.LinAlgError:
        projected = centered
    length = float(projected[:, 0].max() - projected[:, 0].min())
    width = float(projected[:, 1].max() - projected[:, 1].min()) if projected.shape[1] > 1 else 0.0

    # Shoelace on the boundary polygon, which is exact for a planar ring.
    x, y = planar[:, 0], planar[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return area, max(length, width), min(length, width)


def _cluster_by_distance(points: np.ndarray, radius: float) -> list[list[int]]:
    """Single-link clustering used to merge the same defect seen from many views.

    A defect photographed from five angles must not become five defects in the report.
    """
    count = points.shape[0]
    if count == 0:
        return []
    unvisited = set(range(count))
    clusters: list[list[int]] = []
    while unvisited:
        seed = unvisited.pop()
        cluster = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            if not unvisited:
                break
            remaining = np.fromiter(unvisited, dtype=int)
            distances = np.linalg.norm(points[remaining] - points[current], axis=1)
            near = remaining[distances <= radius]
            for index in near:
                unvisited.discard(int(index))
                cluster.append(int(index))
                frontier.append(int(index))
        clusters.append(cluster)
    return clusters


_SEVERITY_ORDER = ["informational", "low", "medium", "high", "critical"]


def _worst_severity(values: Iterable[str]) -> str:
    best = "informational"
    for value in values:
        candidate = str(value or "").strip().lower()
        if candidate in _SEVERITY_ORDER and _SEVERITY_ORDER.index(candidate) > _SEVERITY_ORDER.index(best):
            best = candidate
    return best


def project_defects(
    *,
    detections: Sequence[dict[str, Any]],
    poses: dict[str, CameraPose],
    surface: SurfaceModel,
    anchor: geo.GeoAnchor,
    mask_dir: str | Path | None = None,
    merge_radius_m: float = 0.35,
    max_boundary_points: int = 64,
    diagnostics: dict[str, int] | None = None,
) -> list[ProjectedDefect]:
    """Locate and size every detection on the reconstructed surface.

    Each detection must carry `image` plus either `mask` (a path to a binary mask) or
    `bbox` in pixel coordinates. Detections whose ray never meets the surface are
    dropped, since an unlocated defect cannot be placed on a map.

    A boundary is only sized when EVERY one of its rays reaches the surface. With one
    corner of a box lost to a DSM hole the old code measured the triangle left over --
    1.5 m2 for a 3 x 3 m footprint -- and reported it as the defect's area. Such
    boundaries are counted in ``diagnostics`` instead of being measured wrong.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.setdefault("boundaries", 0)
    diagnostics.setdefault("sized", 0)
    diagnostics.setdefault("incomplete_boundaries", 0)
    diagnostics.setdefault("unposed_images", 0)
    surface_local = SurfaceModel(
        elevation=surface.elevation,
        west=surface.west,
        north=surface.north,
        pixel_size=surface.pixel_size,
        epsg=surface.epsg,
    )

    bridged, bridged_cells = bridge_small_holes(surface_local)
    diagnostics.setdefault("sized_on_bridged_surface", 0)

    def on_bridged_cell(point: np.ndarray) -> bool:
        col = int((point[0] - bridged.west) / bridged.pixel_size)
        row = int((bridged.north - point[1]) / bridged.pixel_size)
        if 0 <= row < bridged.height and 0 <= col < bridged.width:
            return bool(bridged_cells[row, col])
        return False

    observations: list[dict[str, Any]] = []
    rotation = anchor.rotation_matrix
    origin_vec = anchor.origin_vector
    translation_vec = anchor.translation_vector

    def sfm_to_world(points: np.ndarray) -> np.ndarray:
        return anchor.scale * (points @ rotation.T) + translation_vec + origin_vec

    def world_to_sfm(points: np.ndarray) -> np.ndarray:
        local = np.asarray(points, dtype=float).reshape(-1, 3) - origin_vec - translation_vec
        return (local @ rotation) / max(anchor.scale, 1e-12)

    # Ray marching happens in projected metres, so the surface and the ray share units.
    for detection in detections:
        image_name = Path(str(detection.get("image") or detection.get("image_path") or "")).name
        pose = poses.get(image_name)
        if pose is None:
            diagnostics["unposed_images"] += 1
            continue

        boundaries: list[np.ndarray] = []
        mask_path = detection.get("mask") or detection.get("mask_path")
        if mask_path:
            candidate = Path(mask_path)
            if not candidate.is_absolute() and mask_dir:
                candidate = Path(mask_dir) / candidate.name
            if candidate.exists():
                mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if pose.width and pose.height and (mask.shape[1], mask.shape[0]) != (pose.width, pose.height):
                        mask = cv2.resize(mask, (pose.width, pose.height), interpolation=cv2.INTER_NEAREST)
                    boundaries.extend(_mask_contours(mask))
        if not boundaries:
            bbox = detection.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = [float(v) for v in bbox]
                boundaries.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float))
        if not boundaries:
            continue

        camera_center_world = sfm_to_world(pose.center.reshape(1, 3))[0]

        for boundary in boundaries:
            if boundary.shape[0] > max_boundary_points:
                step = max(1, boundary.shape[0] // max_boundary_points)
                boundary = boundary[::step]

            diagnostics["boundaries"] += 1
            directions = [
                # Rotate and scale the direction into projected space; translation does
                # not apply to a direction vector.
                anchor.scale * (rotation @ pose.ray_direction(float(p[0]), float(p[1])))
                for p in boundary
            ]
            hits = [intersect_surface(camera_center_world, d, surface_local) for d in directions]
            interpolated = False
            if any(h is None for h in hits):
                # Retry the whole outline on the surface with small holes bridged, so
                # every corner is measured the same way; label the result.
                hits = [intersect_surface(camera_center_world, d, bridged) for d in directions]
                interpolated = any(h is not None and on_bridged_cell(h) for h in hits)
            if any(h is None for h in hits) or len(hits) < 3:
                diagnostics["incomplete_boundaries"] += 1
                # Not sized -- but still placed, when the ray through the outline's
                # centre reaches the surface. A defect the camera saw does not vanish
                # from the map because a corner of its box fell in a hole.
                middle = boundary.mean(axis=0)
                ray = anchor.scale * (rotation @ pose.ray_direction(float(middle[0]), float(middle[1])))
                centre_hit = intersect_surface(camera_center_world, ray, surface_local)
                centre_bridged = False
                if centre_hit is None:
                    centre_hit = intersect_surface(camera_center_world, ray, bridged)
                    centre_bridged = centre_hit is not None and on_bridged_cell(centre_hit)
                if centre_hit is None:
                    continue
                diagnostics["located_unsized"] = diagnostics.get("located_unsized", 0) + 1
                observations.append({
                    "centroid": np.asarray(centre_hit, dtype=float),
                    "boundary": None,
                    "image": image_name,
                    "type": str(detection.get("defect_type") or detection.get("type") or "defect"),
                    "severity": str(detection.get("severity") or "informational"),
                    "confidence": float(detection.get("confidence") or detection.get("score") or 0.0),
                    "area": None,
                    "length": None,
                    "width": None,
                    "interpolated": centre_bridged,
                })
                continue
            diagnostics["sized"] += 1
            if interpolated:
                diagnostics["sized_on_bridged_surface"] += 1
            hit_array = np.asarray(hits, dtype=float)
            area, length, width = _shape_metrics(hit_array)
            if area <= 0.0:
                continue

            centroid = hit_array.mean(axis=0)
            observations.append(
                {
                    "centroid": centroid,
                    "boundary": hit_array,
                    "image": image_name,
                    "type": str(detection.get("defect_type") or detection.get("type") or "defect"),
                    "severity": str(detection.get("severity") or "informational"),
                    "confidence": float(detection.get("confidence") or detection.get("score") or 0.0),
                    "area": area,
                    "length": length,
                    "width": width,
                    "interpolated": interpolated,
                }
            )

    if not observations:
        return []

    centroids = np.asarray([obs["centroid"] for obs in observations], dtype=float)
    projected: list[ProjectedDefect] = []

    # Cluster per defect type so a crack and a spall at the same spot stay distinct.
    for defect_type in sorted({obs["type"] for obs in observations}):
        indices = [i for i, obs in enumerate(observations) if obs["type"] == defect_type]
        subset = centroids[indices]
        for cluster_number, cluster in enumerate(_cluster_by_distance(subset, float(merge_radius_m))):
            members = [observations[indices[i]] for i in cluster]
            member_centroids = np.asarray([m["centroid"] for m in members], dtype=float)
            centre = member_centroids.mean(axis=0)

            # Keep the largest single observation rather than averaging: partial views
            # of one defect would otherwise shrink its reported size.
            best = max(members, key=lambda m: -1.0 if m["area"] is None else m["area"])
            boundary_world = best["boundary"]
            centre_lon, centre_lat = geo.projected_to_wgs84([centre[0]], [centre[1]], anchor.epsg)

            hull_lonlat: list[list[float]] = []
            if boundary_world is not None and boundary_world.shape[0] >= 3:
                lon, lat = geo.projected_to_wgs84(boundary_world[:, 0], boundary_world[:, 1], anchor.epsg)
                planar = np.stack([lon, lat], axis=1).astype(np.float32)
                try:
                    hull = cv2.convexHull(planar.reshape(-1, 1, 2)).reshape(-1, 2)
                    hull_lonlat = [[float(p[0]), float(p[1])] for p in hull]
                except cv2.error:
                    hull_lonlat = [[float(a), float(b)] for a, b in zip(lon, lat)]

            projected.append(
                ProjectedDefect(
                    defect_id=f"{defect_type}-{cluster_number + 1:04d}",
                    defect_type=defect_type,
                    longitude=float(centre_lon[0]),
                    latitude=float(centre_lat[0]),
                    altitude_m=float(centre[2]),
                    easting=float(centre[0]),
                    northing=float(centre[1]),
                    area_m2=None if best["area"] is None else float(best["area"]),
                    length_m=None if best["length"] is None else float(best["length"]),
                    width_m=None if best["width"] is None else float(best["width"]),
                    severity=_worst_severity(m["severity"] for m in members),
                    confidence=float(max(m["confidence"] for m in members)),
                    observation_count=len(members),
                    source_images=sorted({m["image"] for m in members}),
                    boundary_lonlat=hull_lonlat,
                    surface_interpolated=any(m.get("interpolated") for m in members),
                )
            )

    projected.sort(key=lambda d: -1.0 if d.area_m2 is None else d.area_m2, reverse=True)
    return projected


def write_defect_layer(
    path: str | Path,
    defects: Sequence[ProjectedDefect],
    *,
    epsg: int = geo.WGS84_EPSG,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write projected defects as a GeoJSON layer ready for QGIS or a web map."""
    features = [defect.to_feature() for defect in defects]
    properties = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "defect_count": len(defects),
        "total_area_m2": round(sum(d.area_m2 or 0.0 for d in defects), 4),
        "unsized_count": sum(1 for d in defects if d.area_m2 is None),
    }
    if metadata:
        properties.update(metadata)
    return geo.write_geojson(path, features, epsg=epsg, properties=properties)


def project_run_defects(
    *,
    reconstruction_dir: str | Path,
    detections: Sequence[dict[str, Any]],
    output_path: str | Path,
    mask_dir: str | Path | None = None,
    merge_radius_m: float = 0.35,
) -> dict[str, Any]:
    """Locate a run's detections against its reconstruction, writing a GeoJSON layer.

    Returns a status dict. When the reconstruction is not georeferenced or has no DSM,
    the reason is reported and no layer is written — the caller must not present an
    unlocated defect set as if it were mapped.
    """
    recon_dir = Path(reconstruction_dir)
    pose_path = recon_dir / "camera_poses.json"
    anchor_path = recon_dir / "geo_anchor.json"
    dsm_path = recon_dir / "dsm.tif"

    if not pose_path.exists():
        return {"status": "skipped", "reason": "No camera_poses.json; reconstruction did not record poses."}
    if not anchor_path.exists():
        return {"status": "skipped", "reason": "Reconstruction is not georeferenced (no geo_anchor.json)."}
    if not dsm_path.exists():
        return {"status": "skipped", "reason": "No DSM GeoTIFF available to intersect defect rays against."}
    if not geo.geo_capabilities()["rasterio"]:
        return {"status": "skipped", "reason": "rasterio is not installed, so the DSM cannot be read."}

    poses = load_camera_poses(pose_path)
    if not poses:
        return {"status": "skipped", "reason": "camera_poses.json contains no fully-posed views."}

    anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor = geo.GeoAnchor(**anchor_payload)
    surface = load_surface(dsm_path)

    diagnostics: dict[str, int] = {}
    defects = project_defects(
        detections=detections,
        poses=poses,
        surface=surface,
        anchor=anchor,
        mask_dir=mask_dir,
        merge_radius_m=merge_radius_m,
        diagnostics=diagnostics,
    )
    note = _projection_note(diagnostics)
    if not defects:
        return {
            "status": "empty",
            "reason": "No detection could be located and sized on the reconstructed "
                      "surface. " + note,
            "input_detections": len(detections),
            "diagnostics": diagnostics,
        }

    layer_path = write_defect_layer(
        output_path,
        defects,
        metadata={"crs_epsg": anchor.epsg, "source_reconstruction": str(recon_dir)},
    )
    return {
        "status": "ok",
        "path": layer_path,
        "defect_count": len(defects),
        "input_detections": len(detections),
        "diagnostics": diagnostics,
        "note": note,
        "total_area_m2": round(sum(d.area_m2 or 0.0 for d in defects), 4),
        "unsized_count": sum(1 for d in defects if d.area_m2 is None),
        "crs_epsg": anchor.epsg,
        "defects": [d.to_dict() for d in defects],
    }


def _projection_note(diagnostics: dict[str, int]) -> str:
    """What could not be placed, in words, so a short defect list is not read as complete."""
    parts = []
    if diagnostics.get("incomplete_boundaries"):
        parts.append(
            f"{diagnostics['incomplete_boundaries']} of {diagnostics.get('boundaries', 0)} "
            "defect outline(s) had at least one ray that did not reach the DSM (a hole in "
            "the reconstruction or the edge of it) and were not sized, since a partial "
            "outline under-states the area."
        )
        if diagnostics.get("located_unsized"):
            parts.append(
                f"{diagnostics['located_unsized']} of those were still placed from the ray "
                "through their centre and appear as points with no size (sized=false)."
            )
    if diagnostics.get("sized_on_bridged_surface"):
        parts.append(
            f"{diagnostics['sized_on_bridged_surface']} outline(s) met DSM cells bridged "
            f"from neighbours at most {MAX_BRIDGED_CELLS} cells away (the DSM has holes "
            "between reconstructed points); those defects carry surface_interpolated=true."
        )
    if diagnostics.get("unposed_images"):
        parts.append(
            f"{diagnostics['unposed_images']} detection(s) came from images the "
            "reconstruction did not pose."
        )
    return " ".join(parts)
