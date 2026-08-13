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
    area_m2: float
    length_m: float
    width_m: float
    severity: str
    confidence: float
    observation_count: int
    source_images: list[str] = field(default_factory=list)
    boundary_lonlat: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature(self) -> dict:
        """GeoJSON feature: polygon when a boundary was recovered, else a point."""
        properties = {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "area_m2": round(self.area_m2, 6),
            "length_m": round(self.length_m, 4),
            "width_m": round(self.width_m, 4),
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "source_images": ", ".join(self.source_images[:8]),
            "altitude_m": round(self.altitude_m, 3),
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
        if math.isfinite(ground):
            gap = float(point[2] - ground)
            if previous_gap is not None and previous_gap > 0.0 >= gap:
                low, high = previous_t, travelled
                for _ in range(24):
                    mid = 0.5 * (low + high)
                    probe = origin + direction * mid
                    ground_mid = surface.sample(probe[0], probe[1])
                    if not math.isfinite(ground_mid):
                        break
                    if probe[2] - ground_mid > 0.0:
                        low = mid
                    else:
                        high = mid
                return origin + direction * (0.5 * (low + high))
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
) -> list[ProjectedDefect]:
    """Locate and size every detection on the reconstructed surface.

    Each detection must carry `image` plus either `mask` (a path to a binary mask) or
    `bbox` in pixel coordinates. Detections whose ray never meets the surface are
    dropped, since an unlocated defect cannot be placed on a map.
    """
    surface_local = SurfaceModel(
        elevation=surface.elevation,
        west=surface.west,
        north=surface.north,
        pixel_size=surface.pixel_size,
        epsg=surface.epsg,
    )

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

            hits: list[np.ndarray] = []
            for pixel in boundary:
                direction_sfm = pose.ray_direction(float(pixel[0]), float(pixel[1]))
                # Rotate and scale the direction into projected space; translation does
                # not apply to a direction vector.
                direction_world = anchor.scale * (rotation @ direction_sfm)
                hit = intersect_surface(camera_center_world, direction_world, surface_local)
                if hit is not None:
                    hits.append(hit)

            if len(hits) < 3:
                continue
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
            best = max(members, key=lambda m: m["area"])
            boundary_world = best["boundary"]
            lon, lat = geo.projected_to_wgs84(boundary_world[:, 0], boundary_world[:, 1], anchor.epsg)
            centre_lon, centre_lat = geo.projected_to_wgs84([centre[0]], [centre[1]], anchor.epsg)

            hull_lonlat: list[list[float]] = []
            if boundary_world.shape[0] >= 3:
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
                    area_m2=float(best["area"]),
                    length_m=float(best["length"]),
                    width_m=float(best["width"]),
                    severity=_worst_severity(m["severity"] for m in members),
                    confidence=float(max(m["confidence"] for m in members)),
                    observation_count=len(members),
                    source_images=sorted({m["image"] for m in members}),
                    boundary_lonlat=hull_lonlat,
                )
            )

    projected.sort(key=lambda d: d.area_m2, reverse=True)
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
        "total_area_m2": round(sum(d.area_m2 for d in defects), 4),
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

    defects = project_defects(
        detections=detections,
        poses=poses,
        surface=surface,
        anchor=anchor,
        mask_dir=mask_dir,
        merge_radius_m=merge_radius_m,
    )
    if not defects:
        return {
            "status": "empty",
            "reason": "No detection ray intersected the reconstructed surface.",
            "input_detections": len(detections),
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
        "total_area_m2": round(sum(d.area_m2 for d in defects), 4),
        "crs_epsg": anchor.epsg,
        "defects": [d.to_dict() for d in defects],
    }
