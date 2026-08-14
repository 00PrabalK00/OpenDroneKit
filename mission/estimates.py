"""What a mission will cost to fly: batteries, storage and time.

These are the numbers an operator needs before leaving for site, because each one
changes what goes in the case. A survey needing four battery swaps is a different job
from one needing two. A flight producing 180 GB is a different job from one producing
18. Both are cheap to know in advance and expensive to discover on site.

Every figure here is an *estimate* derived from declared equipment specifications, and
the honest thing to do with an estimate is say what it assumed. So each result carries
its assumptions, and anything that could not be derived is reported as unknown rather
than filled in with a plausible-looking number. A storage estimate that silently
guessed the camera would be worse than no estimate at all: the operator would pack
cards to match it.

The battery model is deliberately simple. Endurance is taken from the aircraft's
declared figure rather than modelled from mass, wind and battery chemistry, because a
model detailed enough to beat the manufacturer's number would need inputs an operator
does not have at planning time. What this does add is the reserve, the swap count, and
a check that no single leg exceeds what one battery can fly -- which is the failure
that strands an aircraft mid-survey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Compression is the whole difficulty in sizing imagery. These are bytes per pixel
# measured across typical UAV survey frames, not a specification: detailed scenes
# compress worse than uniform ones, so treat the result as an order-of-magnitude guide
# and pack a margin. Stated here rather than buried so the assumption can be argued
# with.
JPEG_BYTES_PER_PIXEL = 0.40
RAW_BYTES_PER_PIXEL = 1.50

# Below this a reserve stops being a reserve. Regulators and manufacturers converge on
# 20-30%; going under 15% leaves nothing for a diversion or a headwind on the way home.
MIN_SAFE_RESERVE_PCT = 15.0

DEFAULT_RESERVE_PCT = 25.0


@dataclass
class AircraftProfile:
    """Declared performance for one airframe.

    Endurance is the manufacturer's hover or cruise figure. It is optimistic in wind
    and in cold, which is what the reserve is for.
    """

    name: str = "generic multirotor"
    endurance_min: float = 25.0
    cruise_speed_m_s: float = 8.0
    battery_capacity_mah: float = 0.0
    reserve_pct: float = DEFAULT_RESERVE_PCT
    batteries_owned: int = 0

    @property
    def usable_endurance_min(self) -> float:
        """Endurance after holding back the reserve."""
        return self.endurance_min * (1.0 - self.reserve_pct / 100.0)


@dataclass
class StorageEstimate:
    """How much card space the imagery will need."""

    image_count: int
    megapixels: float
    bytes_per_image: int
    total_bytes: int
    format: str
    known_camera: bool
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1_000_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_count": self.image_count,
            "megapixels": round(self.megapixels, 1),
            "bytes_per_image": self.bytes_per_image,
            "mb_per_image": round(self.bytes_per_image / 1_000_000, 1),
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_gb, 2),
            "format": self.format,
            "known_camera": self.known_camera,
            "assumptions": self.assumptions,
        }


@dataclass
class BatteryEstimate:
    """How many packs the flight needs, and whether it fits in one."""

    flight_time_min: float
    usable_endurance_min: float
    reserve_pct: float
    batteries_required: int
    swaps_required: int
    fits_in_one_flight: bool
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flight_time_min": round(self.flight_time_min, 1),
            "usable_endurance_min": round(self.usable_endurance_min, 1),
            "reserve_pct": self.reserve_pct,
            "batteries_required": self.batteries_required,
            "swaps_required": self.swaps_required,
            "fits_in_one_flight": self.fits_in_one_flight,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
        }


def estimate_storage(image_count: int, camera: str = "custom", *,
                     image_format: str = "jpeg",
                     camera_presets: dict[str, dict[str, Any]] | None = None) -> StorageEstimate:
    """Size the imagery a mission will produce.

    An unrecognised camera is reported as such and sized from the fallback profile,
    because an operator who knows the number is a guess can pack accordingly, whereas
    one shown a confident wrong number cannot.
    """
    if image_count < 0:
        raise ValueError("Image count cannot be negative.")

    from .planner import CAMERA_PRESETS  # imported late to avoid a circular import

    presets = camera_presets if camera_presets is not None else CAMERA_PRESETS
    key = (camera or "").lower()
    known = key in presets
    profile = presets.get(key, presets["custom"])

    width = float(profile.get("image_w_px", 4000))
    # Not every preset declares a height; 3:2 is the common survey sensor aspect.
    height = float(profile.get("image_h_px", width * 2.0 / 3.0))
    pixels = width * height

    fmt = image_format.lower()
    if fmt in ("raw", "dng"):
        per_pixel, fmt = RAW_BYTES_PER_PIXEL, "raw"
    elif fmt in ("jpeg", "jpg"):
        per_pixel, fmt = JPEG_BYTES_PER_PIXEL, "jpeg"
    else:
        raise ValueError(f"Unsupported image format: {image_format!r}. Use jpeg or raw.")

    bytes_per_image = int(round(pixels * per_pixel))

    assumptions = [
        f"{width:.0f}x{height:.0f} px at {per_pixel} bytes/pixel for {fmt}.",
        "Compression varies with scene detail; treat this as a guide and carry margin.",
    ]
    if not known:
        assumptions.insert(0, (
            f"Camera {camera!r} is not in the preset database, so the fallback "
            f"{width:.0f}x{height:.0f} profile was used. The real figure will differ."
        ))
    if fmt == "jpeg":
        assumptions.append("Shooting RAW as well would roughly quadruple this.")

    return StorageEstimate(
        image_count=image_count,
        megapixels=pixels / 1_000_000,
        bytes_per_image=bytes_per_image,
        total_bytes=bytes_per_image * image_count,
        format=fmt,
        known_camera=known,
        assumptions=assumptions,
    )


def estimate_batteries(flight_time_min: float,
                       aircraft: AircraftProfile | None = None) -> BatteryEstimate:
    """Work out how many packs the flight needs, and flag what a swap cannot fix."""
    if flight_time_min < 0:
        raise ValueError("Flight time cannot be negative.")

    aircraft = aircraft or AircraftProfile()
    if aircraft.endurance_min <= 0:
        raise ValueError("Aircraft endurance must be positive to estimate batteries.")

    usable = aircraft.usable_endurance_min
    warnings: list[str] = []

    if aircraft.reserve_pct < MIN_SAFE_RESERVE_PCT:
        warnings.append(
            f"Reserve is {aircraft.reserve_pct:.0f}%, below the {MIN_SAFE_RESERVE_PCT:.0f}% "
            "that leaves room for a headwind on the return leg."
        )

    if usable <= 0:
        raise ValueError("Reserve leaves no usable endurance; check the aircraft profile.")

    required = max(1, math.ceil(flight_time_min / usable)) if flight_time_min > 0 else 1
    fits = flight_time_min <= usable

    if not fits:
        warnings.append(
            f"The mission needs {flight_time_min:.0f} min of flying but one battery gives "
            f"{usable:.0f} min usable, so it cannot be flown in a single sortie. Plan "
            f"{required} flights with a resumable split, or the aircraft will land "
            "mid-survey and leave a gap."
        )

    if aircraft.batteries_owned and required > aircraft.batteries_owned:
        warnings.append(
            f"{required} batteries are needed but {aircraft.batteries_owned} are on the "
            "profile. Either charge between sorties on site or the survey will not finish "
            "in one visit."
        )

    return BatteryEstimate(
        flight_time_min=flight_time_min,
        usable_endurance_min=usable,
        reserve_pct=aircraft.reserve_pct,
        batteries_required=required,
        swaps_required=max(0, required - 1),
        fits_in_one_flight=fits,
        warnings=warnings,
        assumptions=[
            f"{aircraft.name} endurance {aircraft.endurance_min:.0f} min with a "
            f"{aircraft.reserve_pct:.0f}% reserve held back.",
            "Manufacturer endurance is measured in still air; wind and cold reduce it.",
            "Time to land, swap and climb back to altitude is not counted.",
        ],
    )


def estimate_mission(plan: Any, *, aircraft: AircraftProfile | None = None,
                     image_format: str = "jpeg") -> dict[str, Any]:
    """Full pre-flight estimate for a generated plan.

    Accepts a MissionPlan or its dict form, so it works on a plan that has been through
    storage or an API round trip.
    """
    payload = plan if isinstance(plan, dict) else plan.to_dict()

    recipe = payload.get("flight_recipe") or {}
    poses = recipe.get("world_poses") or []
    if poses:
        image_count = sum(1 for pose in poses if pose.get("trigger", True))
    else:
        image_count = len(payload.get("waypoints") or [])

    flight_time = float(payload.get("estimated_time_min") or 0.0)
    camera = str(payload.get("camera") or "custom")

    storage = estimate_storage(image_count, camera, image_format=image_format)
    battery = estimate_batteries(flight_time, aircraft)

    return {
        "image_count": image_count,
        "distance_m": round(float(payload.get("path_distance_m") or 0.0), 1),
        "duration_min": round(flight_time, 1),
        "gsd_cm": payload.get("estimated_gsd_cm"),
        "storage": storage.to_dict(),
        "battery": battery.to_dict(),
        "warnings": battery.warnings,
        "note": (
            "Estimates from declared equipment specifications, not measurements. "
            "Storage varies with scene detail and endurance with wind and temperature."
        ),
    }
