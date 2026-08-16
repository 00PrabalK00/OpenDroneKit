"""What a reconstruction's coordinates actually mean, decided before anyone measures.

Structure-from-motion recovers geometry up to a similarity transform. Without something
external tying it down, the model has arbitrary position, arbitrary rotation and --
this is the part that bites -- arbitrary SCALE. It will still render, still mesh, still
produce a beautiful point cloud, and every distance measured in it will be wrong by an
unknown factor.

Geotagged imagery supplies all three. Ground control supplies all three, more precisely.
Neither present, and the reconstruction is a shape, not a survey.

So this module answers one question before a job starts: what can be claimed about the
output. A GPS-denied reconstruction is a legitimate and useful thing -- indoor, handheld,
under-bridge, ground robot -- and the failure is never that it exists. The failure is
letting someone measure in it as though the units were metres.

    from core.spatial_reference import assess_spatial_reference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Below this fraction of images carrying a fix, the geotags are too sparse to constrain
# the block: a handful of tagged frames in a long sequence pins the ends and lets the
# middle drift.
MIN_GEOTAG_FRACTION = 0.6

# Three control points is the minimum that defines a similarity transform. Fewer cannot
# resolve scale and rotation together.
MIN_GCP_COUNT = 3


@dataclass
class SpatialReference:
    """What the reconstruction's coordinates mean, and what may be claimed from them."""

    mode: str                    # georeferenced | control_referenced | arbitrary
    georeferenced: bool
    scale_is_known: bool
    image_count: int
    geotagged_count: int
    gcp_count: int
    epsg: int | None
    measurements_allowed: bool
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "georeferenced": self.georeferenced,
            "scale_is_known": self.scale_is_known,
            "measurements_allowed": self.measurements_allowed,
            "image_count": self.image_count,
            "geotagged_count": self.geotagged_count,
            "geotagged_fraction": (
                round(self.geotagged_count / self.image_count, 3) if self.image_count else 0.0
            ),
            "gcp_count": self.gcp_count,
            "epsg": self.epsg,
            "note": self.note,
            "warnings": list(self.warnings),
        }


def count_geotagged(image_paths: Sequence[str | Path]) -> int:
    """How many images carry a usable GPS fix."""
    from . import geo

    found = 0
    for path in image_paths:
        try:
            if geo.read_exif_gps(path) is not None:
                found += 1
        except Exception:  # noqa: BLE001 - an unreadable tag is simply not a fix
            continue
    return found


def assess_spatial_reference(
    image_paths: Sequence[str | Path],
    *,
    gcp_count: int = 0,
    epsg: int | None = None,
) -> SpatialReference:
    """Decide what the reconstruction's coordinates will mean, before it runs.

    Three outcomes, and the third is the one that matters:

    ``georeferenced`` -- enough geotags to constrain the block. Real coordinates, real
    scale, measurements valid.

    ``control_referenced`` -- no usable geotags but at least three GCPs. Real
    coordinates and real scale, from control rather than from the camera.

    ``arbitrary`` -- neither. The model is correct in shape and meaningless in units.
    Measurements are refused rather than returned in "model units" that a reader will
    inevitably treat as metres.
    """
    image_count = len(image_paths)
    if image_count == 0:
        raise ValueError("No images supplied; there is nothing to reconstruct.")

    geotagged = count_geotagged(image_paths)
    fraction = geotagged / image_count
    warnings: list[str] = []

    enough_geotags = fraction >= MIN_GEOTAG_FRACTION
    enough_control = gcp_count >= MIN_GCP_COUNT

    if 0 < fraction < MIN_GEOTAG_FRACTION:
        warnings.append(
            f"Only {geotagged} of {image_count} images ({fraction:.0%}) carry a GPS fix. "
            f"Below {MIN_GEOTAG_FRACTION:.0%} the tagged frames pin the ends of the block "
            "and the middle is free to drift, so these geotags are not treated as "
            "constraining the reconstruction."
        )

    if enough_geotags:
        mode, georeferenced, scaled = "georeferenced", True, True
        note = (
            f"{geotagged} of {image_count} images carry a GPS fix. The reconstruction "
            "is georeferenced and its scale comes from the camera positions; distances "
            "and areas measured in it are metres."
        )
        if not enough_control:
            warnings.append(
                "No ground control. Absolute accuracy is limited by the onboard GNSS, "
                "which for a consumer receiver is metres rather than centimetres."
            )
    elif enough_control:
        mode, georeferenced, scaled = "control_referenced", True, True
        note = (
            f"No usable geotags, but {gcp_count} ground control points define position, "
            "rotation and scale. The reconstruction is georeferenced through control and "
            "measurements are valid."
        )
    else:
        mode, georeferenced, scaled = "arbitrary", False, False
        note = (
            "GPS-denied reconstruction with no ground control. The model is correct in "
            "SHAPE but has arbitrary position, rotation and SCALE: there is nothing "
            "tying it to the ground or to a metre. It can be viewed, meshed and "
            "inspected visually. It cannot be measured, and no distance, area or volume "
            "from it means anything until at least "
            f"{MIN_GCP_COUNT} control points or a known reference length is supplied."
        )
        warnings.append(
            "Measurements are refused on this reconstruction. Reporting them in model "
            "units would invite a reader to treat them as metres, which is the failure "
            "this refusal exists to prevent."
        )

    return SpatialReference(
        mode=mode,
        georeferenced=georeferenced,
        scale_is_known=scaled,
        image_count=image_count,
        geotagged_count=geotagged,
        gcp_count=gcp_count,
        epsg=epsg if georeferenced else None,
        measurements_allowed=scaled,
        note=note,
        warnings=warnings,
    )


class MeasurementRefused(ValueError):
    """A measurement was attempted on a reconstruction with no known scale."""


def require_measurable(reference: SpatialReference, *, what: str = "measure") -> None:
    """Guard any call that returns a distance, area or volume."""
    if not reference.measurements_allowed:
        raise MeasurementRefused(
            f"Cannot {what} on an arbitrary-scale reconstruction. {reference.note}"
        )
