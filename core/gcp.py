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

    def to_dict(self) -> dict[str, Any]:
        return {"image": self.image, "pixel_x": self.pixel_x, "pixel_y": self.pixel_y}


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
             pixel_x: float, pixel_y: float) -> GroundControlPoint:
    """Record where a control point was identified in one image."""
    existing = next((m for m in point.marks if m.image == image), None)
    if existing is not None:
        existing.pixel_x = float(pixel_x)
        existing.pixel_y = float(pixel_y)
    else:
        point.marks.append(ImageMark(image=image, pixel_x=float(pixel_x),
                                     pixel_y=float(pixel_y)))
    return point


def residuals_from_positions(points: Sequence[GroundControlPoint],
                             computed: dict[str, tuple[float, float, float]]
                             ) -> list[GcpResidual]:
    """Compare where the reconstruction put each point against where it was surveyed."""
    residuals: list[GcpResidual] = []
    for point in points:
        position = computed.get(point.name)
        if position is None:
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
                    residuals: Sequence[GcpResidual]) -> dict[str, Any]:
    """What the control says about the survey's accuracy.

    Reports the residuals rather than a verdict. A fit always produces a transform;
    whether it is good enough is a question about the job's tolerance, which this module
    does not know.
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

    return {
        "point_count": len(points),
        "used": len(residuals),
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
        "note": (
            f"RMSE is computed over the {len(residuals)} control point(s) that took part "
            "in the fit. Points used to fit a transform and then measured against it "
            "flatter the result; independent check points give a truer figure. "
            f"{GOOD_RMSE_M} m is quoted as a common expectation, not as this job's "
            "tolerance."
        ),
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
