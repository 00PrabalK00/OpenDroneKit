"""Ground control points: surveyed marks that tell a reconstruction where it really is.

GPS EXIF puts a survey within a few metres. That is fine for a coverage map and useless
for anything a client will build against. Ground control points are positions measured
on the ground to centimetre accuracy, identified by eye in the imagery, and used to tie
the model to the real world.

The output that matters is not the transform but the residual. A reconstruction fitted
to five ground control points will always produce a transform; whether that transform is
any good is a separate question, answered only by how far each point lands from where it
was surveyed. So every result here reports per-point residuals, and a fit is never
described as accurate on the strength of having converged.

Two failure modes get explicit treatment because both produce a plausible wrong answer.
A control point marked in too few images cannot be triangulated, and including it lets a
badly constrained position drag the whole fit. And a point whose residual dwarfs the
others is usually a marking mistake -- the wrong target identified in one image -- which
is worth flagging rather than averaging away.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# A point seen in fewer images than this cannot be triangulated reliably: two rays
# intersect somewhere no matter how badly they are aimed, and a third is what makes the
# intersection meaningful.
MIN_IMAGE_MARKS = 3

# Above this, a residual is not measurement noise. Survey-grade control is expected to
# fit within a few centimetres; a point metres out has been marked on the wrong target
# or typed in wrong.
OUTLIER_RESIDUAL_M = 0.5

# What a good fit looks like for photogrammetric control. Quoted rather than enforced:
# the tolerance belongs to the job, not to this module.
GOOD_RMSE_M = 0.05


class GcpError(ValueError):
    """A control point file or marking that cannot be used as given."""


@dataclass
class GroundControlPoint:
    """One surveyed mark."""

    name: str
    x: float
    y: float
    z: float
    epsg: int
    # Accuracy claimed by whoever surveyed it, in metres. Kept because a point measured
    # to 2 cm and one measured to 50 cm should not be weighted alike, and because a fit
    # cannot be better than its control.
    accuracy_m: float = 0.02
    marks: list["ImageMark"] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return len(self.marks) >= MIN_IMAGE_MARKS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "x": self.x, "y": self.y, "z": self.z,
            "epsg": self.epsg, "accuracy_m": self.accuracy_m,
            "mark_count": len(self.marks), "usable": self.is_usable,
        }


@dataclass
class ImageMark:
    """Where a control point was identified in one photograph."""

    image: str
    pixel_x: float
    pixel_y: float
    # The size of the picture the pixel was read on. The reconstruction may have worked
    # on the same frame at another resolution, and a mark taken at face value there lands
    # somewhere else in the photograph.
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"image": self.image, "pixel_x": self.pixel_x, "pixel_y": self.pixel_y,
                "image_width": self.image_width, "image_height": self.image_height}


@dataclass
class GcpResidual:
    """How far one control point landed from where it was surveyed."""

    name: str
    dx: float
    dy: float
    dz: float
    mark_count: int

    @property
    def horizontal_m(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def total_m(self) -> float:
        return math.sqrt(self.dx ** 2 + self.dy ** 2 + self.dz ** 2)

    @property
    def is_outlier(self) -> bool:
        return self.total_m > OUTLIER_RESIDUAL_M

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dx": round(self.dx, 4), "dy": round(self.dy, 4), "dz": round(self.dz, 4),
            "horizontal_m": round(self.horizontal_m, 4),
            "total_m": round(self.total_m, 4),
            "mark_count": self.mark_count,
            "outlier": self.is_outlier,
        }


def read_gcp_file(path: str | Path, default_epsg: int | None = None
                  ) -> list[GroundControlPoint]:
    """Read control points from the CSV surveyors actually produce.

    Column order is taken from the header rather than assumed. A file of bare numbers is
    ambiguous in exactly the way that matters -- northing and easting are both large and
    both plausible in either column -- so a headerless file is refused.
    """
    source = Path(path)
    if not source.exists():
        raise GcpError(f"{source} does not exist.")

    text = source.read_text(encoding="utf-8-sig", errors="replace")
    rows = [r for r in csv.reader(text.splitlines()) if r and any(c.strip() for c in r)]
    if not rows:
        raise GcpError(f"{source.name} is empty.")

    header = [c.strip().lower() for c in rows[0]]
    aliases = {
        "name": {"name", "id", "point", "label", "gcp"},
        "x": {"x", "easting", "east", "lon", "longitude"},
        "y": {"y", "northing", "north", "lat", "latitude"},
        "z": {"z", "elevation", "height", "alt", "altitude"},
        "accuracy": {"accuracy", "accuracy_m", "sigma", "precision"},
        "epsg": {"epsg", "crs", "srid"},
    }
    index: dict[str, int] = {}
    for field_name, names in aliases.items():
        for position, column in enumerate(header):
            if column in names:
                index[field_name] = position
                break

    missing = [f for f in ("x", "y", "z") if f not in index]
    if missing:
        found = ", ".join(repr(c) for c in rows[0][:8]) or "nothing"
        raise GcpError(
            f"{source.name} has no {'/'.join(missing)} column, so which value is which "
            f"cannot be determined. Found: {found}. Add a header row -- guessing between "
            "easting and northing would place the survey kilometres away."
        )

    points: list[GroundControlPoint] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if max(index.values()) >= len(row):
            continue
        try:
            x = float(row[index["x"]])
            y = float(row[index["y"]])
            z = float(row[index["z"]])
        except ValueError:
            continue

        epsg = default_epsg
        if "epsg" in index:
            try:
                epsg = int(float(row[index["epsg"]]))
            except ValueError:
                pass
        if epsg is None:
            raise GcpError(
                f"{source.name} does not state a CRS and none was supplied. Control "
                "point coordinates are meaningless without one: the same numbers name "
                "different places in different projections."
            )

        accuracy = 0.02
        if "accuracy" in index:
            try:
                accuracy = abs(float(row[index["accuracy"]]))
            except ValueError:
                pass

        name = (row[index["name"]].strip() if "name" in index
                else f"GCP{line_number - 1:03d}")
        points.append(GroundControlPoint(name=name or f"GCP{line_number - 1:03d}",
                                         x=x, y=y, z=z, epsg=epsg,
                                         accuracy_m=accuracy))

    if not points:
        raise GcpError(f"{source.name} has a usable header but no numeric rows.")
    return points


def add_mark(point: GroundControlPoint, image: str,
             pixel_x: float, pixel_y: float,
             image_width: int = 0, image_height: int = 0) -> GroundControlPoint:
    """Record where a control point was identified in one image."""
    existing = next((m for m in point.marks if m.image == image), None)
    if existing is not None:
        existing.pixel_x = float(pixel_x)
        existing.pixel_y = float(pixel_y)
        existing.image_width = int(image_width or 0)
        existing.image_height = int(image_height or 0)
    else:
        point.marks.append(ImageMark(image=image, pixel_x=float(pixel_x),
                                     pixel_y=float(pixel_y),
                                     image_width=int(image_width or 0),
                                     image_height=int(image_height or 0)))
    return point


def residuals_from_positions(points: Sequence[GroundControlPoint],
                             computed: dict[str, tuple[float, float, float]]
                             ) -> list[GcpResidual]:
    """Compare where the reconstruction put each point against where it was surveyed.

    Only points marked in the imagery can have been placed by the reconstruction. A
    position supplied for an unmarked point came from somewhere else, and counting it
    produced a report that quoted an RMSE over points its own warning said took no part.
    """
    residuals: list[GcpResidual] = []
    for point in points:
        position = computed.get(point.name)
        if position is None or not point.marks:
            continue
        residuals.append(GcpResidual(
            name=point.name,
            dx=float(position[0]) - point.x,
            dy=float(position[1]) - point.y,
            dz=float(position[2]) - point.z,
            mark_count=len(point.marks),
        ))
    return residuals


def accuracy_report(points: Sequence[GroundControlPoint],
                    residuals: Sequence[GcpResidual], *,
                    role: str = "fit") -> dict[str, Any]:
    """What the control says about the survey's accuracy.

    Reports the residuals rather than a verdict. A fit always produces a transform;
    whether it is good enough is a question about the job's tolerance, which this module
    does not know.

    ``role`` says what the residuals are: ``"fit"`` for points the georeferencing was
    fitted to (which flatter it), ``"check"`` for points that took no part in it and
    are therefore an independent measure.
    """
    unusable = [p for p in points if not p.is_usable]
    unmarked = [p for p in points if not p.marks]

    warnings: list[str] = []
    if unmarked:
        warnings.append(
            f"{len(unmarked)} control point(s) were never marked in any image and took "
            "no part in the fit: " + ", ".join(p.name for p in unmarked[:8]) + "."
        )
    marked_but_thin = [p for p in unusable if p.marks]
    if marked_but_thin:
        warnings.append(
            f"{len(marked_but_thin)} control point(s) are marked in fewer than "
            f"{MIN_IMAGE_MARKS} images and cannot be triangulated reliably: "
            + ", ".join(p.name for p in marked_but_thin[:8]) +
            ". Two rays intersect wherever they are aimed; a third is what makes the "
            "intersection mean something."
        )

    if not residuals:
        return {
            "point_count": len(points),
            "used": 0,
            "rmse_m": None,
            "residuals": [],
            "warnings": warnings + [
                "No residuals could be computed, so this survey has no measured "
                "accuracy. Do not quote one."
            ],
            "note": "A survey with no checked control is not a controlled survey.",
        }

    horizontal = [r.horizontal_m for r in residuals]
    vertical = [abs(r.dz) for r in residuals]
    totals = [r.total_m for r in residuals]

    rmse = math.sqrt(sum(t * t for t in totals) / len(totals))
    outliers = [r for r in residuals if r.is_outlier]

    if outliers:
        warnings.append(
            f"{len(outliers)} control point(s) sit more than {OUTLIER_RESIDUAL_M} m from "
            "their surveyed position: " + ", ".join(r.name for r in outliers[:8]) +
            ". That is usually the wrong target marked in one image rather than a bad "
            "reconstruction; check the markings before accepting the fit."
        )

    best_possible = max((p.accuracy_m for p in points), default=0.0)
    if rmse < best_possible:
        warnings.append(
            f"The reported RMSE ({rmse:.3f} m) is below the accuracy claimed for the "
            f"control itself ({best_possible:.3f} m). A survey cannot be more accurate "
            "than the points it was fitted to; this suggests the fit is over-constrained "
            "or the control accuracy is understated."
        )

    if role == "check":
        note = (
            f"RMSE is computed over {len(residuals)} control point(s) located in the "
            "imagery and compared with where the georeferencing put them. They took no "
            "part in that georeferencing, so this is an independent check. "
            f"{GOOD_RMSE_M} m is quoted as a common expectation, not as this job's "
            "tolerance."
        )
    else:
        note = (
            f"RMSE is computed over the {len(residuals)} control point(s) that took part "
            "in the fit. Points used to fit a transform and then measured against it "
            "flatter the result; independent check points give a truer figure. "
            f"{GOOD_RMSE_M} m is quoted as a common expectation, not as this job's "
            "tolerance."
        )

    return {
        "point_count": len(points),
        "used": len(residuals),
        "role": role,
        "rmse_m": round(rmse, 4),
        "horizontal_rmse_m": round(
            math.sqrt(sum(h * h for h in horizontal) / len(horizontal)), 4),
        "vertical_rmse_m": round(
            math.sqrt(sum(v * v for v in vertical) / len(vertical)), 4),
        "max_residual_m": round(max(totals), 4),
        "worst_point": max(residuals, key=lambda r: r.total_m).name,
        "outlier_count": len(outliers),
        "residuals": [r.to_dict() for r in residuals],
        "warnings": warnings,
        "meets_survey_grade": rmse <= GOOD_RMSE_M,
        "note": note,
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Kept with the project
# ---------------------------------------------------------------------------

CONTROL_FILE = "gcps.json"


def save_control(project_root: str | Path, points: Sequence[GroundControlPoint]) -> Path:
    """Write the control points and every image mark into the project folder.

    They lived only in memory, so an afternoon of marking targets vanished when the
    application closed.
    """
    target = Path(project_root) / CONTROL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": [
            {**{k: v for k, v in p.to_dict().items() if k not in ("mark_count", "usable")},
             "marks": [m.to_dict() for m in p.marks]}
            for p in points
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_control(project_root: str | Path) -> list[GroundControlPoint]:
    """The project's control points, or an empty list when none were imported."""
    source = Path(project_root) / CONTROL_FILE
    if not source.is_file():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise GcpError(f"{source} cannot be read: {exc}") from exc
    points: list[GroundControlPoint] = []
    for raw in payload.get("points", []):
        point = GroundControlPoint(
            name=str(raw["name"]), x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]),
            epsg=int(raw["epsg"]), accuracy_m=float(raw.get("accuracy_m", 0.02)),
        )
        for mark in raw.get("marks", []):
            add_mark(point, str(mark["image"]), float(mark["pixel_x"]), float(mark["pixel_y"]),
                     int(mark.get("image_width") or 0), int(mark.get("image_height") or 0))
        points.append(point)
    return points


# ---------------------------------------------------------------------------
# Locating the marks in the reconstruction
# ---------------------------------------------------------------------------


@dataclass
class ViewPose:
    """One registered photograph: where the camera stood and how it sees."""

    image: str
    rotation: np.ndarray        # camera-from-world, 3x3
    translation: np.ndarray     # camera-from-world, 3
    intrinsics: np.ndarray      # K, 3x3
    width: int = 0
    height: int = 0
    model: str = ""
    params: list[float] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    def _camera(self):
        """The COLMAP camera, so lens distortion is removed exactly as COLMAP models it."""
        if not (self.model and self.params and self.width and self.height):
            return None
        try:
            import pycolmap  # type: ignore
        except ImportError:
            return None
        try:
            return pycolmap.Camera(model=self.model, width=int(self.width),
                                   height=int(self.height), params=list(self.params))
        except Exception:  # noqa: BLE001 - an unknown model falls back to the pinhole
            return None

    def _scaled(self, mark: "ImageMark") -> tuple[float, float]:
        """The mark in this camera's pixel grid."""
        x, y = float(mark.pixel_x), float(mark.pixel_y)
        if mark.image_width and mark.image_height and self.width and self.height:
            x *= self.width / float(mark.image_width)
            y *= self.height / float(mark.image_height)
        return x, y

    def ray(self, mark: "ImageMark") -> np.ndarray:
        """Unit direction, in the model frame, of the ray through a marked pixel."""
        x, y = self._scaled(mark)
        camera = self._camera()
        if camera is not None:
            normalised = np.asarray(camera.cam_from_img(np.asarray([[x, y]], dtype=float)),
                                    dtype=float).reshape(-1)[:2]
            direction_cam = np.array([normalised[0], normalised[1], 1.0])
        else:
            direction_cam = np.linalg.solve(self.intrinsics, np.array([x, y, 1.0]))
        direction = self.rotation.T @ direction_cam
        return direction / np.linalg.norm(direction)

    def project(self, point: np.ndarray) -> np.ndarray | None:
        """Pixel (in this camera's grid) where a model-frame point appears."""
        in_camera = self.rotation @ np.asarray(point, dtype=float) + self.translation
        if in_camera[2] <= 0:
            return None
        camera = self._camera()
        if camera is not None:
            return np.asarray(camera.img_from_cam(in_camera.reshape(1, 3)), dtype=float).reshape(-1)[:2]
        homogeneous = self.intrinsics @ (in_camera / in_camera[2])
        return homogeneous[:2]


def poses_from_records(records: Iterable[dict[str, Any]]) -> dict[str, ViewPose]:
    """Registered views from the ``cameras`` list a reconstruction writes to camera_poses.json."""
    poses: dict[str, ViewPose] = {}
    for record in records:
        if not record.get("rotation_cam_from_world") or not record.get("intrinsics"):
            continue
        poses[str(record["image"])] = ViewPose(
            image=str(record["image"]),
            rotation=np.asarray(record["rotation_cam_from_world"], dtype=float).reshape(3, 3),
            translation=np.asarray(record["translation_cam_from_world"], dtype=float).reshape(3),
            intrinsics=np.asarray(record["intrinsics"], dtype=float).reshape(3, 3),
            width=int(record.get("width") or 0),
            height=int(record.get("height") or 0),
            model=str(record.get("model") or ""),
            params=[float(v) for v in record.get("params") or []],
        )
    return poses


@dataclass
class LocatedPoint:
    """A control point found in the model by intersecting the rays through its marks."""

    name: str
    position: np.ndarray            # model frame
    views: int
    reprojection_px: dict[str, float]
    unregistered_images: list[str]

    @property
    def rms_reprojection_px(self) -> float:
        values = list(self.reprojection_px.values())
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "views": self.views,
            "rms_reprojection_px": round(self.rms_reprojection_px, 3),
            "reprojection_px": {k: round(v, 3) for k, v in self.reprojection_px.items()},
            "unregistered_images": list(self.unregistered_images),
        }


def locate_in_model(point: GroundControlPoint,
                    poses: dict[str, ViewPose]) -> LocatedPoint | None:
    """Intersect the rays through a point's marks. None when fewer than two views see it."""
    rays = []
    missing = []
    for mark in point.marks:
        pose = poses.get(mark.image) or poses.get(Path(mark.image).name)
        if pose is None:
            missing.append(mark.image)
            continue
        rays.append((pose, mark, pose.center, pose.ray(mark)))
    if len(rays) < 2:
        return None

    # The point closest, in the least-squares sense, to every ray.
    system = np.zeros((3, 3))
    rhs = np.zeros(3)
    for _pose, _mark, origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        rhs += projector @ origin
    if np.linalg.cond(system) > 1e12:
        return None
    position = np.linalg.solve(system, rhs)

    errors: dict[str, float] = {}
    for pose, mark, _origin, _direction in rays:
        pixel = pose.project(position)
        if pixel is None:
            # Behind a camera that marked it: the marks do not describe one point.
            return None
        x, y = pose._scaled(mark)
        errors[mark.image] = float(math.hypot(pixel[0] - x, pixel[1] - y))
    return LocatedPoint(name=point.name, position=position, views=len(rays),
                        reprojection_px=errors, unregistered_images=missing)


def surveyed_positions(points: Sequence[GroundControlPoint], epsg: int) -> np.ndarray:
    """Each point's surveyed position in ``epsg`` (heights are left as surveyed)."""
    from core import geo

    out = np.zeros((len(points), 3))
    for index, point in enumerate(points):
        x, y = point.x, point.y
        if int(point.epsg) != int(epsg):
            x, y = geo.transformer(int(point.epsg), int(epsg)).transform(x, y)
        out[index] = (float(x), float(y), float(point.z))
    if not np.all(np.isfinite(out)):
        raise GcpError(
            f"Some control points cannot be expressed in EPSG:{epsg}; check the CRS "
            "they were imported in."
        )
    return out


@dataclass
class ControlSolution:
    """What the control did to the georeferencing, and how well it fits."""

    anchor: Any | None              # geo.GeoAnchor fitted to the control, or None
    report: dict[str, Any]


def solve_with_control(points: Sequence[GroundControlPoint],
                       poses: dict[str, ViewPose],
                       epsg: int,
                       camera_anchor: Any | None = None) -> ControlSolution:
    """Georeference a model from its control points, or check it against them.

    With three or more points each marked in at least ``MIN_IMAGE_MARKS`` registered
    images, the model is fitted to the control (a 7-parameter similarity) and the
    residuals of that fit are reported -- with leave-one-out residuals too when there
    are four or more, which is the honest measure: each point checked against a fit it
    took no part in. With fewer, the camera-GPS georeferencing stands and every point
    that can be located is reported as an independent check against it.
    """
    from core import geo

    located: list[tuple[GroundControlPoint, LocatedPoint]] = []
    for point in points:
        found = locate_in_model(point, poses)
        if found is not None:
            located.append((point, found))
    usable = [(p, f) for p, f in located if f.views >= MIN_IMAGE_MARKS]
    located_info = {f.name: f.to_dict() for _p, f in located}

    if len(usable) >= 3:
        control = [p for p, _f in usable]
        world = surveyed_positions(control, epsg)
        model = np.asarray([f.position for _p, f in usable])
        origin = world.mean(axis=0)
        scale, rotation, translation = geo.solve_similarity_umeyama(model, world - origin)
        predicted = scale * (model @ rotation.T) + translation + origin
        errors = predicted - world
        fit = [GcpResidual(name=p.name, dx=float(e[0]), dy=float(e[1]), dz=float(e[2]),
                           mark_count=len(p.marks)) for p, e in zip(control, errors)]
        norms = np.linalg.norm(errors, axis=1)
        anchor = geo.GeoAnchor(
            scale=float(scale),
            rotation=[[float(v) for v in row] for row in rotation],
            translation=[float(v) for v in translation],
            epsg=int(epsg),
            origin=[float(v) for v in origin],
            inlier_count=len(control),
            sample_count=len(control),
            rmse_m=float(np.sqrt(np.mean(norms ** 2))),
            max_residual_m=float(norms.max()),
            method="ground_control",
        )
        report = accuracy_report(points, fit, role="fit")
        if len(usable) >= 4:
            checks: list[GcpResidual] = []
            for held in range(len(usable)):
                keep = [i for i in range(len(usable)) if i != held]
                s, r, t = geo.solve_similarity_umeyama(model[keep], world[keep] - origin)
                guess = s * (r @ model[held]) + t + origin
                e = guess - world[held]
                checks.append(GcpResidual(name=control[held].name, dx=float(e[0]),
                                          dy=float(e[1]), dz=float(e[2]),
                                          mark_count=len(control[held].marks)))
            report["leave_one_out"] = accuracy_report(control, checks, role="check")
        else:
            report["warnings"].append(
                "With only three usable control points every one of them is needed for "
                "the fit, so there is no independent check. A fourth point, or a check "
                "point that is not used for fitting, is what makes the RMSE mean something."
            )
        report["georeferenced_by"] = "ground_control"
    else:
        anchor = None
        if camera_anchor is not None:
            report = check_against_anchor(points, poses, camera_anchor, role="check")
        else:
            report = accuracy_report(points, [], role="check")
        report["georeferenced_by"] = "camera_gps" if camera_anchor is not None else "none"
        report["warnings"].append(
            f"{len(usable)} control point(s) are marked in at least {MIN_IMAGE_MARKS} "
            "registered images; three are needed to georeference the model with them, "
            "so they were not used for the fit."
        )
    report["located"] = located_info
    return ControlSolution(anchor=anchor, report=report)


def check_against_anchor(points: Sequence[GroundControlPoint],
                         poses: dict[str, ViewPose],
                         anchor: Any, *, role: str = "check") -> dict[str, Any]:
    """Residuals of an existing georeferencing at every control point the model can see.

    ``role`` is "fit" when that georeferencing was itself fitted to these points and
    "check" when it came from somewhere else (the camera geotags).
    """
    located = [(p, locate_in_model(p, poses)) for p in points]
    located = [(p, f) for p, f in located if f is not None]
    residuals: list[GcpResidual] = []
    if located:
        world = surveyed_positions([p for p, _f in located], int(anchor.epsg))
        placed = anchor.apply(np.asarray([f.position for _p, f in located]), absolute=True)
        residuals = [GcpResidual(name=p.name, dx=float(e[0]), dy=float(e[1]), dz=float(e[2]),
                                 mark_count=len(p.marks))
                     for (p, _f), e in zip(located, placed - world)]
    report = accuracy_report(points, residuals, role=role)
    report["located"] = {f.name: f.to_dict() for _p, f in located}
    return report
