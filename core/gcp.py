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


def epsg_from_crs_header(header: str) -> int:
    """The EPSG code an ODM ``gcp_list.txt`` header names.

    OpenDroneMap accepts three spellings on the first line: ``WGS84 UTM <zone><N|S>``,
    ``EPSG:<code>`` and a PROJ string (``+proj=utm +zone=17 +datum=WGS84 ...``). All three
    occur in published datasets (ODM's bellus uses the first, sheffield_cross the third).
    A header that names no CRS this can resolve is refused: guessing one would put the
    control in another place entirely.
    """
    import re

    text = header.strip()
    match = re.fullmatch(r"WGS\s*-?\s*84\s+UTM\s+(\d{1,2})\s*([NnSs])", text)
    if match:
        zone = int(match.group(1))
        if not 1 <= zone <= 60:
            raise GcpError(f"UTM zone {zone} in the header {text!r} does not exist.")
        return (32600 if match.group(2).upper() == "N" else 32700) + zone
    match = re.fullmatch(r"EPSG\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if text.startswith("+proj") or "+proj=" in text:
        params = dict(
            (part[1:].split("=", 1) + [""])[:2] for part in text.split() if part.startswith("+")
        )
        try:
            from pyproj import CRS

            code = CRS.from_proj4(text).to_epsg(min_confidence=70)
        except Exception:  # noqa: BLE001 - pyproj missing or the string is not one it reads
            code = None
        if code:
            return int(code)
        # WGS84 UTM written as a PROJ string resolves without pyproj's database too.
        datum = (params.get("datum") or params.get("ellps") or "").upper()
        if params.get("proj") == "utm" and params.get("zone", "").isdigit() and datum == "WGS84":
            return (32700 if "south" in params else 32600) + int(params["zone"])
        raise GcpError(
            f"The control file's CRS header {text!r} does not match an EPSG code. "
            "Reproject the control to a CRS with one (UTM, say) and import it again."
        )
    raise GcpError(
        f"The first line of an ODM control file names its CRS ('WGS84 UTM 17N', "
        f"'EPSG:32617' or a PROJ string); {text!r} is none of those."
    )


def _looks_like_odm(lines: Sequence[str]) -> bool:
    """A first line that is not a CSV header, then rows of x y z px py image."""
    if len(lines) < 2 or "," in lines[0]:
        return False
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) < 6:
            return False
        try:
            [float(t) for t in tokens[:5]]
        except ValueError:
            return False
    return True


def read_odm_gcp_file(path: str | Path) -> list[GroundControlPoint]:
    """Read OpenDroneMap's ``gcp_list.txt`` / ``gcp_file.txt`` format.

    Line one names the CRS. Every other line is one *observation*:
    ``geo_x geo_y geo_z im_x im_y image_name [gcp_name] [extra ...]``, so a point marked
    in five photographs appears on five lines. Lines are grouped into points by their
    name when the optional seventh column is given, otherwise by their coordinates --
    the same surveyed position is the same mark on the ground. Pixel positions are on
    the full-resolution photograph, as ODM defines them.
    """
    source = Path(path)
    if not source.exists():
        raise GcpError(f"{source} does not exist.")
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise GcpError(f"{source.name} is empty.")
    epsg = epsg_from_crs_header(lines[0])

    points: dict[str, GroundControlPoint] = {}
    by_position: dict[tuple[float, float, float], str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if len(tokens) < 6:
            raise GcpError(
                f"{source.name} line {line_number} has {len(tokens)} fields; an ODM "
                "control row is 'x y z pixel_x pixel_y image [name]'."
            )
        try:
            x, y, z, px, py = (float(t) for t in tokens[:5])
        except ValueError as exc:
            raise GcpError(f"{source.name} line {line_number} is not numeric: {line!r}") from exc
        if px < 0 or py < 0:
            raise GcpError(f"{source.name} line {line_number} marks a negative pixel.")
        image = tokens[5]
        key = (round(x, 3), round(y, 3), round(z, 3))
        if len(tokens) >= 7:
            name = tokens[6]
        else:
            name = by_position.get(key) or f"GCP{len(by_position) + 1:02d}"
        by_position.setdefault(key, name)
        point = points.get(name)
        if point is None:
            point = points[name] = GroundControlPoint(name=name, x=x, y=y, z=z, epsg=epsg)
        elif (round(point.x, 3), round(point.y, 3), round(point.z, 3)) != key:
            raise GcpError(
                f"{source.name} line {line_number} gives {name} a different position from "
                "an earlier line. One name must be one surveyed point."
            )
        add_mark(point, image, px, py)
    if not points:
        raise GcpError(f"{source.name} names a CRS but has no control rows.")
    return list(points.values())


def read_gcp_file(path: str | Path, default_epsg: int | None = None
                  ) -> list[GroundControlPoint]:
    """Read control points from the CSV surveyors produce, or ODM's gcp_list.txt.

    Column order is taken from the header rather than assumed. A file of bare numbers is
    ambiguous in exactly the way that matters -- northing and easting are both large and
    both plausible in either column -- so a headerless file is refused. An ODM control
    file (CRS on line one, then ``x y z px py image``) is recognised and read with its
    image marks.
    """
    source = Path(path)
    if not source.exists():
        raise GcpError(f"{source} does not exist.")

    text = source.read_text(encoding="utf-8-sig", errors="replace")
    plain = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if _looks_like_odm(plain):
        return read_odm_gcp_file(source)
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

    # A shared offset is not noise. Every point sitting the same tens of metres low is
    # the signature of heights in two vertical datums (GPS ellipsoidal heights against
    # control in orthometric heights differ by the local geoid undulation, -30 m and
    # more in places) or of a barometric altitude, and RMSE alone hides which it is.
    offsets = np.array([[r.dx, r.dy, r.dz] for r in residuals], dtype=float)
    bias = offsets.mean(axis=0)
    spread = offsets.std(axis=0) if len(residuals) > 1 else np.zeros(3)
    if len(residuals) >= 2 and abs(bias[2]) > 1.0 and abs(bias[2]) > 3.0 * max(spread[2], 0.1):
        warnings.append(
            f"Every control point sits {abs(bias[2]):.1f} m "
            f"{'below' if bias[2] < 0 else 'above'} its surveyed height (spread "
            f"{spread[2]:.2f} m). An offset shared by all points is a datum difference, "
            "not noise: check that the control and the georeferencing use the same "
            "vertical datum (ellipsoidal or orthometric heights) before reading the RMSE."
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
        "mean_offset_m": [round(float(v), 4) for v in bias],
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
    payload = {"points": control_to_payload(points)}
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
    return control_from_payload(payload.get("points", []))


def control_to_payload(points: Sequence[GroundControlPoint]) -> list[dict[str, Any]]:
    """Control points with every image mark, as JSON-ready dicts.

    The same shape is written to the project folder and sent to an OpenDroneKit server
    with a cloud reconstruction, so the server fits the model to the marks the operator
    made instead of silently dropping them.
    """
    return [
        {**{k: v for k, v in p.to_dict().items() if k not in ("mark_count", "usable")},
         "marks": [m.to_dict() for m in p.marks]}
        for p in points
    ]


def control_from_payload(raw_points: Any) -> list[GroundControlPoint]:
    """Inverse of control_to_payload; a malformed entry raises GcpError naming it."""
    if not isinstance(raw_points, list):
        raise GcpError("Ground control must be a list of points.")
    points: list[GroundControlPoint] = []
    for index, raw in enumerate(raw_points):
        try:
            point = GroundControlPoint(
                name=str(raw["name"]), x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]),
                epsg=int(raw["epsg"]), accuracy_m=float(raw.get("accuracy_m", 0.02)),
            )
            for mark in raw.get("marks", []):
                add_mark(point, str(mark["image"]), float(mark["pixel_x"]), float(mark["pixel_y"]),
                         int(mark.get("image_width") or 0), int(mark.get("image_height") or 0))
        except GcpError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GcpError(f"Ground control point {index + 1} is malformed: {exc!r}") from exc
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
    # How the position was found: rays intersected, or one ray followed to the surface
    # the model's tie points describe around the mark (a weaker placement: it inherits
    # the error of that surface, and a single mark cannot reveal a mis-marked target).
    method: str = "ray_intersection"
    surface_points: int = 0
    surface_radius_px: float = 0.0

    @property
    def rms_reprojection_px(self) -> float:
        values = list(self.reprojection_px.values())
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name, "views": self.views, "method": self.method,
            "rms_reprojection_px": round(self.rms_reprojection_px, 3),
            "reprojection_px": {k: round(v, 3) for k, v in self.reprojection_px.items()},
            "unregistered_images": list(self.unregistered_images),
        }
        if self.method != "ray_intersection":
            out["surface_tie_points"] = self.surface_points
            out["surface_radius_px"] = self.surface_radius_px
        return out


@dataclass
class ImageObservations:
    """The model's own tie points as one registered image sees them.

    ``pixels`` are the keypoints (in the camera's pixel grid) that triangulated to the
    model-frame positions in ``points``. They are what lets a control point marked in a
    single photograph be placed: its ray is followed to the surface those tie points
    describe around the mark.
    """

    pixels: np.ndarray      # (N, 2)
    points: np.ndarray      # (N, 3) model frame


# Search radii, in camera pixels, for tie points around a single mark; the first that
# holds enough points to describe the surface is used.
SURFACE_RADII_PX = (40.0, 80.0, 160.0, 320.0)
SURFACE_MIN_POINTS = 6


def _locate_on_surface(point: GroundControlPoint, pose: ViewPose, mark: ImageMark,
                       observations: ImageObservations) -> tuple[np.ndarray, int, float] | None:
    """Where the ray through one mark meets the tie-point surface around it.

    A plane is fitted to the tie points nearest the mark in that image and the ray is
    intersected with it. Returns (position, tie points used, radius) or None when the
    image has too few tie points near the mark, or the plane is edge-on to the ray.
    """
    if observations.pixels.shape[0] < SURFACE_MIN_POINTS:
        return None
    x, y = pose._scaled(mark)
    distance = np.hypot(observations.pixels[:, 0] - x, observations.pixels[:, 1] - y)
    origin = pose.center
    direction = pose.ray(mark)
    for radius in SURFACE_RADII_PX:
        near = observations.points[distance <= radius]
        if near.shape[0] < SURFACE_MIN_POINTS:
            continue
        # Robust plane: drop the points furthest from the median depth along the ray
        # (a roof edge or a tree in the neighbourhood), then fit by SVD.
        depth = (near - origin) @ direction
        keep = np.abs(depth - np.median(depth)) <= 2.5 * max(
            float(np.median(np.abs(depth - np.median(depth)))), 1e-9)
        near = near[keep] if keep.sum() >= SURFACE_MIN_POINTS else near
        centroid = near.mean(axis=0)
        _u, _s, vt = np.linalg.svd(near - centroid)
        normal = vt[-1]
        facing = float(normal @ direction)
        if abs(facing) < 0.15:
            continue
        t = float(normal @ (centroid - origin)) / facing
        if t <= 0:
            return None
        return origin + t * direction, int(near.shape[0]), float(radius)
    return None


def locate_in_model(point: GroundControlPoint,
                    poses: dict[str, ViewPose],
                    observations: dict[str, ImageObservations] | None = None
                    ) -> LocatedPoint | None:
    """Intersect the rays through a point's marks.

    With two or more registered marks the rays are intersected. With one, and the
    model's tie points for that image, the ray is followed to the surface around the
    mark instead (``views`` is then 1 and the located dict says how). None otherwise.
    """
    rays = []
    missing = []
    for mark in point.marks:
        pose = poses.get(mark.image) or poses.get(Path(mark.image).name)
        if pose is None:
            missing.append(mark.image)
            continue
        rays.append((pose, mark, pose.center, pose.ray(mark)))
    if len(rays) == 1 and observations:
        pose, mark, _origin, _direction = rays[0]
        seen = observations.get(pose.image) or observations.get(Path(pose.image).name)
        if seen is None:
            return None
        found = _locate_on_surface(point, pose, mark, seen)
        if found is None:
            return None
        position, used, radius = found
        pixel = pose.project(position)
        x, y = pose._scaled(mark)
        error = float(math.hypot(pixel[0] - x, pixel[1] - y)) if pixel is not None else float("nan")
        return LocatedPoint(name=point.name, position=position, views=1,
                            reprojection_px={mark.image: error}, unregistered_images=missing,
                            method="single_ray_on_tie_point_surface",
                            surface_points=used, surface_radius_px=radius)
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
                       camera_anchor: Any | None = None,
                       observations: dict[str, ImageObservations] | None = None
                       ) -> ControlSolution:
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
        found = locate_in_model(point, poses, observations)
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
        # Points the model can place but that were too thinly marked to take part in the
        # fit are independent check points against it.
        spare = [(p, f) for p, f in located if f.views < MIN_IMAGE_MARKS]
        if spare:
            spare_world = surveyed_positions([p for p, _f in spare], epsg)
            spare_placed = anchor.apply(np.asarray([f.position for _p, f in spare]), absolute=True)
            report["checkpoints"] = accuracy_report(
                [p for p, _f in spare],
                [GcpResidual(name=p.name, dx=float(e[0]), dy=float(e[1]), dz=float(e[2]),
                             mark_count=len(p.marks))
                 for (p, _f), e in zip(spare, spare_placed - spare_world)],
                role="check")
        if len(usable) < 4 and not spare:
            report["warnings"].append(
                "With only three usable control points every one of them is needed for "
                "the fit, so there is no independent check. A fourth point, or a check "
                "point that is not used for fitting, is what makes the RMSE mean something."
            )
        report["georeferenced_by"] = "ground_control"
    else:
        anchor = None
        if camera_anchor is not None:
            report = check_against_anchor(points, poses, camera_anchor, role="check",
                                          observations=observations)
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
                         anchor: Any, *, role: str = "check",
                         observations: dict[str, ImageObservations] | None = None
                         ) -> dict[str, Any]:
    """Residuals of an existing georeferencing at every control point the model can see.

    ``role`` is "fit" when that georeferencing was itself fitted to these points and
    "check" when it came from somewhere else (the camera geotags).
    """
    located = [(p, locate_in_model(p, poses, observations)) for p in points]
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
    single = [f.name for _p, f in located if f.views < 2]
    if single:
        report["warnings"].append(
            f"{len(single)} control point(s) are marked in one registered image only "
            f"({', '.join(single[:8])}); each was placed by following its ray to the "
            "surface of the model's tie points around the mark. That position carries the "
            "error of that surface, and a single mark cannot show a wrong target, so "
            "treat these residuals as indicative."
        )
    return report
