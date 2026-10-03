"""A building facade measured from its own reconstruction, and a mission planned on it.

A facade survey flown low and nearly level (gimbal a few degrees down, a few metres off
the wall) reconstructs into a cloud dominated by one vertical surface. This module finds
that surface, says how well the reconstruction describes it, and turns it into the
inputs a facade mission needs -- the wall's line on the ground, its base and top, and
the stand-off the survey was actually flown at -- so the next flight is planned on the
building as measured rather than on a hand-drawn line.

Everything reported comes from the georeferenced model:

* the wall plane is a RANSAC fit with a horizontal normal (a facade is vertical);
* ``flatness_rms_m`` is the RMS distance of the inlier points from that plane -- for a
  flat glass and concrete facade it is the reconstruction's noise across the wall, not
  a claim about the building;
* ``coverage_fraction`` is the share of the wall rectangle (between the inliers' 2nd and
  98th percentiles along and up the wall) that holds at least one point per cell. A
  sparse cloud covers a facade thinly by construction, so this is a sparse-coverage
  figure and is labelled so.

Heights are relative to the take-off point when the frames carry DJI's
``RelativeAltitude`` (absolute minus relative altitude gives the home elevation);
otherwise they stay in the anchor's vertical datum and the result says which.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Points further than this from the plane are not part of the wall (window reveals,
# railings, vegetation in front of it).
WALL_INLIER_M = 0.25
COVERAGE_CELL_M = 0.5


class FacadeError(ValueError):
    """The reconstruction does not describe a facade that can be planned on."""


@dataclass
class FacadeWall:
    start_xy: list[float]          # projected CRS, metres
    end_xy: list[float]
    normal_xy: list[float]         # unit, pointing from the wall towards the cameras
    epsg: int
    length_m: float
    base_z: float
    top_z: float
    height_reference: str          # "home" (take-off point) or the anchor's datum
    inlier_points: int
    total_points: int
    flatness_rms_m: float
    coverage_fraction: float
    coverage_note: str
    camera_standoff_m: dict[str, float] = field(default_factory=dict)

    @property
    def height_m(self) -> float:
        return self.top_z - self.base_z

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["height_m"] = round(self.height_m, 3)
        return out


def _ransac_vertical_plane(xyz: np.ndarray, threshold: float, iterations: int = 2000,
                           seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Best vertical plane n.x = d (n horizontal) by RANSAC on two points, refined by LSQ."""
    rng = np.random.default_rng(seed)
    xy = xyz[:, :2]
    best_mask = None
    best_count = -1
    for _ in range(iterations):
        i, j = rng.choice(len(xy), size=2, replace=False)
        direction = xy[j] - xy[i]
        norm = float(np.linalg.norm(direction))
        if norm < 0.5:
            continue
        normal = np.array([-direction[1], direction[0]]) / norm
        d = float(normal @ xy[i])
        mask = np.abs(xy @ normal - d) <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < 10:
        raise FacadeError("No vertical surface holds enough points to be a facade.")
    # Least-squares refinement on the inliers: the normal is the smallest principal axis
    # of the inliers' ground positions.
    inliers = xy[best_mask]
    centre = inliers.mean(axis=0)
    _u, _s, vt = np.linalg.svd(inliers - centre)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    d = float(normal @ centre)
    mask = np.abs(xy @ normal - d) <= threshold
    return normal, d, mask


def measure_facade(points_xyz: np.ndarray, camera_xyz: np.ndarray, epsg: int, *,
                   home_elevation: float | None = None,
                   inlier_m: float = WALL_INLIER_M) -> FacadeWall:
    """The dominant vertical surface of a georeferenced cloud, seen from these cameras."""
    xyz = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    cams = np.asarray(camera_xyz, dtype=float).reshape(-1, 3)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 50:
        raise FacadeError(f"{len(xyz)} points is too few to find a wall in.")
    if len(cams) < 2:
        raise FacadeError("At least two registered cameras are needed to orient the wall.")

    normal, d, mask = _ransac_vertical_plane(xyz, inlier_m)
    # Orient the normal towards the cameras: that is the outside of the wall.
    if float(np.median(cams[:, :2] @ normal - d)) < 0:
        normal, d = -normal, -d
    wall = xyz[mask]
    distance = wall[:, :2] @ normal - d
    flatness = float(np.sqrt(np.mean(distance ** 2)))

    along_axis = np.array([-normal[1], normal[0]])
    along = wall[:, :2] @ along_axis
    a0, a1 = np.percentile(along, [2, 98])
    z0, z1 = np.percentile(wall[:, 2], [2, 98])
    foot = normal * d
    start = foot + along_axis * a0
    end = foot + along_axis * a1

    cols = max(1, int(math.ceil((a1 - a0) / COVERAGE_CELL_M)))
    rows = max(1, int(math.ceil((z1 - z0) / COVERAGE_CELL_M)))
    inside = (along >= a0) & (along <= a1) & (wall[:, 2] >= z0) & (wall[:, 2] <= z1)
    ci = np.clip(((along[inside] - a0) / COVERAGE_CELL_M).astype(int), 0, cols - 1)
    ri = np.clip(((wall[inside, 2] - z0) / COVERAGE_CELL_M).astype(int), 0, rows - 1)
    occupied = len(set(zip(ci.tolist(), ri.tolist())))
    coverage = occupied / float(cols * rows)

    standoff = cams[:, :2] @ normal - d
    reference = "home" if home_elevation is not None else "anchor_datum"
    offset = float(home_elevation) if home_elevation is not None else 0.0
    return FacadeWall(
        start_xy=[float(v) for v in start], end_xy=[float(v) for v in end],
        normal_xy=[float(v) for v in normal], epsg=int(epsg),
        length_m=float(a1 - a0), base_z=float(z0 - offset), top_z=float(z1 - offset),
        height_reference=reference,
        inlier_points=int(mask.sum()), total_points=int(len(xyz)),
        flatness_rms_m=flatness, coverage_fraction=float(coverage),
        coverage_note=(f"Share of {COVERAGE_CELL_M} m cells on the wall holding at least one "
                       "reconstructed point. From a sparse cloud this measures where tie "
                       "points were found, not the dense surface."),
        camera_standoff_m={"min": float(standoff.min()), "median": float(np.median(standoff)),
                           "max": float(standoff.max())},
    )


def home_elevation_from_xmp(image_paths: Sequence[str | Path]) -> float | None:
    """Take-off elevation from DJI XMP: median of (absolute - relative altitude)."""
    from core.coverage_validation import _read_xmp, _xmp_value

    values = []
    for path in image_paths:
        xmp = _read_xmp(Path(path))
        absolute = _xmp_value(xmp, ["drone-dji:AbsoluteAltitude"])
        relative = _xmp_value(xmp, ["drone-dji:RelativeAltitude"])
        if absolute is not None and relative is not None:
            values.append(absolute - relative)
    return float(np.median(values)) if values else None


def camera_from_exif(image_paths: Sequence[str | Path]) -> tuple[str | None, str]:
    """The camera profile the survey frames were taken with, read from their EXIF.

    Matches on focal length (to 0.05 mm) and image width against the profile table, and
    uses the EXIF ``Model`` to choose between profiles that share a sensor (the M30T and
    Mavic 3T wide cameras are both 4.4 mm on 4000 px). Returns ``(key, reason)``; the key
    is ``None`` when the frames do not identify exactly one profile, and the reason says
    why, so the caller asks for a camera instead of planning on a guess.
    """
    from PIL import Image

    from mission.cameras import all_profiles

    seen: dict[tuple[str, float, int], int] = {}
    for path in list(image_paths)[:20]:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                sub = exif.get_ifd(0x8769)
                width = int(sub.get(0xA002) or im.width)
        except (OSError, ValueError):
            continue
        model = str(exif.get(0x0110) or "").strip()
        focal = sub.get(0x920A)
        if focal is None:
            continue
        key = (model, round(float(focal), 2), width)
        seen[key] = seen.get(key, 0) + 1
    if not seen:
        return None, "The frames carry no EXIF focal length to identify the camera by."
    model, focal, width = max(seen, key=seen.get)
    matches = [k for k, prof in all_profiles().items()
               if abs(prof.focal_mm - focal) <= 0.05 and int(prof.image_w_px) == width
               and not prof.thermal]
    token = "".join(ch for ch in model.lower() if ch.isalnum())
    named = [k for k in matches if token and k.split("_")[0] in token]
    chosen = named if named else matches
    if len(chosen) == 1:
        return chosen[0], (f"EXIF Model {model!r}, focal {focal} mm, {width} px wide "
                           f"matches the {chosen[0]} profile.")
    if not chosen:
        return None, (f"No camera profile has a {focal} mm lens on a {width} px sensor "
                      f"(EXIF Model {model!r}); add it under Cameras or choose one.")
    return None, (f"EXIF Model {model!r}, focal {focal} mm, {width} px matches "
                  f"{', '.join(sorted(chosen))}; choose which one flew.")


def facade_from_run(run_dir: str | Path, image_dir: str | Path | None = None) -> FacadeWall:
    """Measure the facade in a COLMAP run's output folder (sparse model + geo_anchor.json)."""
    import json

    import pycolmap  # type: ignore

    from core import geo

    run = Path(run_dir)
    anchor_path = run / "geo_anchor.json"
    if not anchor_path.is_file():
        raise FacadeError("The run is not georeferenced, so the wall has no position.")
    anchor = geo.GeoAnchor(**json.loads(anchor_path.read_text(encoding="utf-8")))
    sparse = run / "colmap" / "sparse"
    models = [pycolmap.Reconstruction(str(f)) for f in sparse.iterdir()
              if (f / "images.bin").is_file()]
    if not models:
        raise FacadeError(f"No COLMAP model under {sparse}.")
    model = max(models, key=lambda m: m.num_reg_images())
    xyz = np.array([p.xyz for p in model.points3D.values()], dtype=float)
    centres = np.array([img.projection_center() for img in model.images.values()
                        if img.has_pose], dtype=float)
    home = None
    if image_dir is not None:
        names = [img.name for img in model.images.values() if img.has_pose]
        home = home_elevation_from_xmp([Path(image_dir) / n for n in names])
    return measure_facade(anchor.apply(xyz), anchor.apply(centres), anchor.epsg,
                          home_elevation=home)


def plan_facade_mission(wall: FacadeWall, *, standoff_m: float | None = None,
                        depth_m: float = 5.0, camera: str = "mavic2pro",
                        mode: str = "facade", bottom_m: float | None = None,
                        top_m: float | None = None) -> tuple[Any, list[list[float]]]:
    """A facade mission on the measured wall.

    The footprint handed to the planner is the wall's ground line with a ``depth_m``
    strip behind it (away from the cameras), so its first edge is the wall and the
    planner's outward side is the side the survey was flown from. The stand-off
    defaults to the median the survey was actually flown at. Heights must be relative
    to the take-off point; a wall measured in another datum is refused.
    """
    from core import geo
    from mission.planner import MissionPlanner

    if wall.height_reference != "home":
        raise FacadeError(
            "The wall's heights are not relative to the take-off point, so a mission "
            "altitude cannot be derived from them."
        )
    start = np.asarray(wall.start_xy)
    end = np.asarray(wall.end_xy)
    inward = -np.asarray(wall.normal_xy)
    ring = [start, end, end + inward * depth_m, start + inward * depth_m]
    lon, lat = geo.projected_to_wgs84([p[0] for p in ring], [p[1] for p in ring], wall.epsg)
    polygon = [[float(a), float(b)] for a, b in zip(lon, lat)]
    standoff = float(standoff_m if standoff_m is not None else wall.camera_standoff_m["median"])
    bottom = float(bottom_m if bottom_m is not None else max(2.0, wall.base_z + 1.0))
    top = float(top_m if top_m is not None else wall.top_z)
    plan = MissionPlanner().generate(
        polygon_lonlat=polygon, mode=mode, camera=camera,
        altitude_m=top, facade_top_altitude_m=top, facade_bottom_altitude_m=bottom,
        facade_standoff_m=standoff, gimbal_tilt_deg=0.0,
    )
    return plan, polygon


def waypoint_offsets_from_wall(plan: Any, wall: FacadeWall) -> np.ndarray:
    """Signed horizontal distance of each planned waypoint from the wall (+ = outside)."""
    from core import geo

    waypoints = np.asarray([[w[0], w[1]] for w in plan.waypoints], dtype=float)
    east, north = geo.wgs84_to_projected(waypoints[:, 0], waypoints[:, 1], wall.epsg)
    xy = np.column_stack([east, north])
    normal = np.asarray(wall.normal_xy)
    d = float(normal @ np.asarray(wall.start_xy))
    return xy @ normal - d
