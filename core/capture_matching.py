"""Matching captured images back to the capture points that were planned.

A mission plan says where every photograph should have been taken. After the flight
there is a folder of images. Nothing in between guarantees they correspond, and the
gap is where surveys go wrong: a capture point the aircraft skipped, or triggered late,
leaves a hole in the overlap that only becomes visible when reconstruction fails days
later.

This closes that loop while the pilot is still on site. It answers three questions:

*Did every planned point produce an image.* A planned pose with no image within the
match radius is a coverage gap, reported by name.

*How closely was the plan flown.* Position deviation per image and across the flight,
which is what tells an operator whether wind pushed the aircraft off the grid.

*Which images were not planned.* Extra frames are not an error -- a pilot may capture
manually -- but they are listed rather than quietly folded into the plan.

Matching is greedy nearest-first and one-to-one. An image matches at most one planned
pose and vice versa, because claiming one photograph satisfied two capture points would
hide a gap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from . import geo

# Beyond this an image is not credibly the planned capture. Chosen to exceed normal
# GNSS error and station-keeping drift without spanning adjacent grid points, which
# are typically tens of metres apart.
DEFAULT_MATCH_RADIUS_M = 15.0

# A deviation above this is worth an operator's attention even though the point was
# matched: the aircraft flew the plan, but not closely.
DEVIATION_WARNING_M = 8.0


@dataclass
class PlannedCapture:
    """One capture point from the mission plan."""

    index: int
    longitude: float
    latitude: float
    altitude_m: float
    yaw_deg: float = 0.0
    gimbal_pitch_deg: float = 0.0
    primitive: str = ""


@dataclass
class CapturedImage:
    """One photograph, as found on the card."""

    path: str
    longitude: float | None = None
    latitude: float | None = None
    altitude_m: float | None = None
    timestamp: datetime | None = None

    @property
    def has_position(self) -> bool:
        return self.longitude is not None and self.latitude is not None


@dataclass
class Match:
    """A planned capture point paired with the image that satisfied it."""

    planned_index: int
    image_path: str
    distance_m: float
    altitude_difference_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_index": self.planned_index,
            "image": self.image_path,
            "distance_m": round(self.distance_m, 2),
            "altitude_difference_m": (
                round(self.altitude_difference_m, 2)
                if self.altitude_difference_m is not None else None
            ),
            "beyond_warning": self.distance_m > DEVIATION_WARNING_M,
        }


@dataclass
class CaptureReport:
    """What the flight actually achieved against what was planned."""

    matches: list[Match] = field(default_factory=list)
    missed: list[PlannedCapture] = field(default_factory=list)
    unplanned: list[CapturedImage] = field(default_factory=list)
    ungeotagged: list[CapturedImage] = field(default_factory=list)
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M
    planned_total: int = 0
    image_total: int = 0

    @property
    def coverage_pct(self) -> float:
        if not self.planned_total:
            return 0.0
        return 100.0 * len(self.matches) / self.planned_total

    def deviation_stats(self) -> dict[str, float]:
        if not self.matches:
            return {}
        distances = sorted(match.distance_m for match in self.matches)
        return {
            "mean_m": round(sum(distances) / len(distances), 2),
            "median_m": round(distances[len(distances) // 2], 2),
            "max_m": round(distances[-1], 2),
            "over_warning": sum(1 for d in distances if d > DEVIATION_WARNING_M),
        }

    def to_dict(self) -> dict[str, Any]:
        stats = self.deviation_stats()
        recommendations: list[str] = []
        if self.missed:
            indices = ", ".join(str(capture.index) for capture in self.missed[:10])
            recommendations.append(
                f"{len(self.missed)} planned capture point(s) produced no image "
                f"(indices {indices}{'...' if len(self.missed) > 10 else ''}). "
                "Re-fly those points before leaving the site: a gap here becomes a hole "
                "in the reconstruction."
            )
        if self.ungeotagged:
            recommendations.append(
                f"{len(self.ungeotagged)} image(s) carry no GPS position and could not be "
                "matched. Check the camera's geotagging settings; without positions these "
                "images cannot seed the reconstruction."
            )
        if stats.get("over_warning"):
            recommendations.append(
                f"{stats['over_warning']} image(s) were taken more than "
                f"{DEVIATION_WARNING_M} m from their planned point. The plan was flown, "
                "but loosely; check overlap before relying on it."
            )

        return {
            "planned_total": self.planned_total,
            "image_total": self.image_total,
            "matched": len(self.matches),
            "missed": [
                {"index": c.index, "lon": c.longitude, "lat": c.latitude,
                 "alt_m": c.altitude_m, "primitive": c.primitive}
                for c in self.missed
            ],
            "unplanned": [image.path for image in self.unplanned],
            "ungeotagged": [image.path for image in self.ungeotagged],
            "coverage_pct": round(self.coverage_pct, 1),
            "deviation": stats,
            "match_radius_m": self.match_radius_m,
            "matches": [match.to_dict() for match in self.matches],
            "recommendations": recommendations,
            "method": (
                "Images are matched to planned capture points greedily, nearest first, "
                "one-to-one within the match radius. One photograph never satisfies two "
                "capture points, because that would hide a gap."
            ),
        }


def planned_captures_from_plan(plan: dict[str, Any] | Any) -> list[PlannedCapture]:
    """Extract capture points from a mission plan.

    The flight recipe carries pose detail; bare waypoints are the fallback and lose
    yaw and gimbal, which is worth knowing when the report is read.
    """
    if not isinstance(plan, dict):
        plan = plan.to_dict() if hasattr(plan, "to_dict") else {}

    recipe = plan.get("flight_recipe") or {}
    poses = recipe.get("world_poses") or recipe.get("poses") or []

    captures: list[PlannedCapture] = []
    if poses:
        for index, pose in enumerate(poses):
            if not pose.get("trigger", True):
                # A transit waypoint is not a capture point and must not be counted
                # as a missed photograph.
                continue
            captures.append(PlannedCapture(
                index=index,
                longitude=float(pose.get("lon", pose.get("longitude", 0.0))),
                latitude=float(pose.get("lat", pose.get("latitude", 0.0))),
                altitude_m=float(pose.get("alt_m", pose.get("alt", 0.0))),
                yaw_deg=float(pose.get("yaw_deg", 0.0)),
                gimbal_pitch_deg=float(pose.get("gimbal_pitch_deg", 0.0)),
                primitive=str(pose.get("primitive", "")),
            ))
        return captures

    for index, row in enumerate(plan.get("waypoints") or []):
        if len(row) < 2:
            continue
        captures.append(PlannedCapture(
            index=index, longitude=float(row[0]), latitude=float(row[1]),
            altitude_m=float(row[2]) if len(row) >= 3 else 0.0,
        ))
    return captures


def images_from_folder(folder: str | Path) -> list[CapturedImage]:
    """Read positions and timestamps from the EXIF of every image in a folder."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder of images.")

    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG"}
    images: list[CapturedImage] = []
    for path in sorted(p for p in root.iterdir() if p.suffix in suffixes):
        fix = geo.read_exif_gps(path)
        images.append(CapturedImage(
            path=str(path),
            longitude=getattr(fix, "longitude", None) if fix else None,
            latitude=getattr(fix, "latitude", None) if fix else None,
            altitude_m=getattr(fix, "altitude", None) if fix else None,
            timestamp=getattr(fix, "timestamp", None) if fix else None,
        ))
    return images


def match_captures(
    planned: Iterable[PlannedCapture],
    images: Iterable[CapturedImage],
    *,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
) -> CaptureReport:
    """Pair images with the capture points they satisfied."""
    planned_list = list(planned)
    image_list = list(images)

    report = CaptureReport(match_radius_m=match_radius_m)
    report.planned_total = len(planned_list)
    report.image_total = len(image_list)

    positioned: list[tuple[int, CapturedImage]] = []
    for index, image in enumerate(image_list):
        if image.has_position:
            positioned.append((index, image))
        else:
            # Without a position an image cannot be placed, and guessing from filename
            # order would fabricate a correspondence.
            report.ungeotagged.append(image)

    candidates: list[tuple[float, int, int]] = []
    for plan_index, capture in enumerate(planned_list):
        for image_index, image in positioned:
            distance = geo.haversine_m(
                capture.latitude, capture.longitude, image.latitude, image.longitude
            )
            if distance <= match_radius_m:
                candidates.append((distance, plan_index, image_index))

    candidates.sort()
    used_plan: set[int] = set()
    used_image: set[int] = set()

    for distance, plan_index, image_index in candidates:
        if plan_index in used_plan or image_index in used_image:
            continue
        used_plan.add(plan_index)
        used_image.add(image_index)

        capture = planned_list[plan_index]
        image = image_list[image_index]
        altitude_difference = (
            image.altitude_m - capture.altitude_m
            if image.altitude_m is not None else None
        )
        report.matches.append(Match(
            planned_index=capture.index, image_path=image.path,
            distance_m=distance, altitude_difference_m=altitude_difference,
        ))

    report.matches.sort(key=lambda match: match.planned_index)
    report.missed = [c for i, c in enumerate(planned_list) if i not in used_plan]
    report.unplanned = [
        image for index, image in positioned if index not in used_image
    ]
    return report


def verify_capture(
    plan: dict[str, Any] | Any,
    image_folder: str | Path,
    *,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check a folder of images against the plan that produced it.

    Intended to run before the pilot leaves the site, which is the only moment when a
    missed capture point is cheap to fix.
    """
    planned = planned_captures_from_plan(plan)
    if not planned:
        return {
            "ok": False,
            "error": (
                "The plan contains no capture points, so there is nothing to verify "
                "against."
            ),
        }

    images = images_from_folder(image_folder)
    report = match_captures(planned, images, match_radius_m=match_radius_m)
    payload = {"ok": True, **report.to_dict()}

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["report_path"] = str(target)

    return payload


def pose_priors_for_reconstruction(report: CaptureReport,
                                   planned: Iterable[PlannedCapture]) -> dict[str, Any]:
    """Planned poses keyed by image name, for seeding reconstruction.

    Structure from motion recovers camera positions from the imagery itself, but it
    converges faster and more reliably from a good initial guess. The plan is exactly
    that guess -- for images that actually matched a planned point. Unmatched images
    are excluded rather than given a fabricated prior.
    """
    by_index = {capture.index: capture for capture in planned}
    priors: dict[str, Any] = {}
    for match in report.matches:
        capture = by_index.get(match.planned_index)
        if capture is None:
            continue
        priors[Path(match.image_path).name] = {
            "lon": capture.longitude, "lat": capture.latitude,
            "alt_m": capture.altitude_m, "yaw_deg": capture.yaw_deg,
            "gimbal_pitch_deg": capture.gimbal_pitch_deg,
            "planned_index": capture.index,
            "observed_offset_m": round(match.distance_m, 2),
        }
    return {
        "priors": priors,
        "count": len(priors),
        "note": (
            "Priors are supplied only for images matched to a planned capture point. "
            "Unmatched images are left for structure from motion to place, rather than "
            "being given a position they were not observed at."
        ),
    }
