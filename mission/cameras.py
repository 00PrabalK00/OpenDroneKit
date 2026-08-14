"""Camera profiles: the sensor geometry every planning calculation depends on.

Ground sample distance, footprint, line spacing and capture interval are all computed
from three numbers -- sensor size, focal length and pixel count. Get them wrong and the
mission is still flown, the photographs are still taken, and the survey is simply at a
different resolution than the one that was ordered. Nothing fails; the deliverable is
just not what was specified.

That is why an unknown camera is reported as unknown here rather than resolved to a
default. The planner previously fell back to a generic 1-inch sensor for any name it did
not recognise, which produces a confident GSD for a camera nobody has described. A
caller that knows the profile is a guess can warn the operator; one handed a silent
default cannot.

Profiles carry a ``source`` field distinguishing manufacturer-published figures from
ones a user typed in, because the two deserve different trust. User profiles are stored
as JSON outside the repository so an operator's fleet survives an upgrade.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Focal length below this is a fisheye or an error; above it, a telescope. Both are
# outside what these formulas describe, so they are refused rather than silently used.
MIN_FOCAL_MM = 1.0
MAX_FOCAL_MM = 500.0

# Smallest and largest plausible sensor dimensions, in millimetres.
MIN_SENSOR_MM = 1.0
MAX_SENSOR_MM = 100.0


@dataclass
class CameraProfile:
    """One camera's geometry, and where the numbers came from."""

    key: str
    name: str
    sensor_w_mm: float
    sensor_h_mm: float
    focal_mm: float
    image_w_px: int
    image_h_px: int
    thermal: bool = False
    # "manufacturer" for published specifications, "user" for operator-entered ones.
    source: str = "manufacturer"
    notes: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Refuse geometry that cannot describe a real camera."""
        if not MIN_FOCAL_MM <= self.focal_mm <= MAX_FOCAL_MM:
            raise ValueError(
                f"{self.key}: focal length {self.focal_mm} mm is outside "
                f"{MIN_FOCAL_MM}-{MAX_FOCAL_MM} mm. A wrong focal length silently "
                "changes every GSD this camera produces."
            )
        for label, value in (("width", self.sensor_w_mm), ("height", self.sensor_h_mm)):
            if not MIN_SENSOR_MM <= value <= MAX_SENSOR_MM:
                raise ValueError(
                    f"{self.key}: sensor {label} {value} mm is outside "
                    f"{MIN_SENSOR_MM}-{MAX_SENSOR_MM} mm."
                )
        if self.image_w_px < 1 or self.image_h_px < 1:
            raise ValueError(f"{self.key}: image dimensions must be positive.")

    @property
    def megapixels(self) -> float:
        return self.image_w_px * self.image_h_px / 1_000_000

    @property
    def pixel_pitch_um(self) -> float:
        """Physical size of one pixel, in micrometres."""
        return self.sensor_w_mm * 1000.0 / self.image_w_px

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_w_mm / (2.0 * self.focal_mm)))

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_h_mm / (2.0 * self.focal_mm)))

    def gsd_cm(self, altitude_m: float) -> float:
        """Ground sample distance in cm/px at a given altitude."""
        if altitude_m <= 0:
            raise ValueError("Altitude must be positive to compute a GSD.")
        return altitude_m * (self.sensor_w_mm / self.focal_mm) / self.image_w_px * 100.0

    def footprint_m(self, altitude_m: float) -> tuple[float, float]:
        """Ground area one nadir photograph covers, in metres."""
        if altitude_m <= 0:
            raise ValueError("Altitude must be positive to compute a footprint.")
        scale = altitude_m / self.focal_mm
        return self.sensor_w_mm * scale, self.sensor_h_mm * scale

    def altitude_for_gsd_m(self, gsd_cm: float) -> float:
        """The altitude that achieves a required GSD -- how a survey is actually specified."""
        if gsd_cm <= 0:
            raise ValueError("A target GSD must be positive.")
        return gsd_cm / 100.0 * self.image_w_px / (self.sensor_w_mm / self.focal_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name,
            "sensor_w_mm": self.sensor_w_mm, "sensor_h_mm": self.sensor_h_mm,
            "focal_mm": self.focal_mm,
            "image_w_px": self.image_w_px, "image_h_px": self.image_h_px,
            "megapixels": round(self.megapixels, 1),
            "pixel_pitch_um": round(self.pixel_pitch_um, 2),
            "horizontal_fov_deg": round(self.horizontal_fov_deg, 1),
            "vertical_fov_deg": round(self.vertical_fov_deg, 1),
            "thermal": self.thermal, "source": self.source, "notes": self.notes,
        }


def _profile(key, name, sw, sh, focal, iw, ih, thermal=False, notes=""):
    return CameraProfile(key, name, sw, sh, focal, iw, ih, thermal, "manufacturer", notes)


# Manufacturer-published geometry for cameras commonly used in survey and inspection.
# Sensor sizes are the active imaging area, not the nominal "1-inch" style marketing
# designation, which does not correspond to any physical dimension.
BUILT_IN: dict[str, CameraProfile] = {
    p.key: p for p in [
        _profile("mavic2pro", "DJI Mavic 2 Pro (Hasselblad L1D-20c)",
                 13.2, 8.8, 10.26, 5472, 3648),
        _profile("phantom4rtk", "DJI Phantom 4 RTK (FC6310R)",
                 13.2, 8.8, 8.8, 5472, 3648,
                 notes="RTK positioning; suits GCP-free corridor and cadastral work."),
        _profile("phantom4pro", "DJI Phantom 4 Pro (FC6310)",
                 13.2, 8.8, 8.8, 5472, 3648),
        _profile("mavic3e", "DJI Mavic 3 Enterprise (wide)",
                 17.3, 13.0, 12.29, 5280, 3956,
                 notes="Four-thirds sensor with a mechanical shutter."),
        _profile("mavic3t_wide", "DJI Mavic 3 Thermal (wide RGB)",
                 6.4, 4.8, 4.4, 4000, 3000),
        _profile("mavic3t_thermal", "DJI Mavic 3 Thermal (radiometric)",
                 7.68, 6.14, 9.1, 640, 512, thermal=True,
                 notes="Radiometric; temperature per pixel, not just an image."),
        _profile("zenmuse_p1", "DJI Zenmuse P1 (35 mm)",
                 35.9, 24.0, 35.0, 8192, 5460,
                 notes="Full-frame survey payload; the highest GSD of this set."),
        _profile("zenmuse_h20t_thermal", "DJI Zenmuse H20T (thermal)",
                 7.68, 6.14, 13.5, 640, 512, thermal=True),
        _profile("evo2pro", "Autel EVO II Pro",
                 13.2, 8.8, 10.57, 5472, 3648),
        _profile("flir_vue_pro_r_640", "FLIR Vue Pro R 640 (13 mm)",
                 10.88, 8.70, 13.0, 640, 512, thermal=True,
                 notes="Radiometric; common fixed payload on custom airframes."),
        # Kept because existing plans and tests reference it. It is a generic 1-inch
        # sensor, not any particular camera, and is labelled as such.
        CameraProfile("custom", "Generic 1-inch sensor", 13.2, 8.8, 10.0, 4000, 3000,
                      source="generic",
                      notes="Placeholder geometry. Any GSD computed from it is indicative "
                            "only; describe the real camera before relying on a survey."),
    ]
}


class UnknownCamera(KeyError):
    """The camera is not in the database, and guessing its geometry would mislead."""


def _store_path() -> Path:
    """User cameras live outside the repository so they survive an upgrade."""
    return Path.home() / ".opendronekit" / "cameras.json"


def load_user_profiles(path: Path | None = None) -> dict[str, CameraProfile]:
    store = path or _store_path()
    if not store.exists():
        return {}
    try:
        payload = json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    profiles: dict[str, CameraProfile] = {}
    for key, entry in (payload or {}).items():
        try:
            profiles[key] = CameraProfile(
                key=key, name=entry.get("name", key),
                sensor_w_mm=float(entry["sensor_w_mm"]),
                sensor_h_mm=float(entry["sensor_h_mm"]),
                focal_mm=float(entry["focal_mm"]),
                image_w_px=int(entry["image_w_px"]),
                image_h_px=int(entry["image_h_px"]),
                thermal=bool(entry.get("thermal", False)),
                source="user", notes=entry.get("notes", ""),
            )
        except (KeyError, TypeError, ValueError):
            # One malformed entry must not hide the rest of an operator's fleet.
            continue
    return profiles


def save_user_profile(profile: CameraProfile, path: Path | None = None) -> Path:
    """Add or replace a user-defined camera."""
    if profile.key in BUILT_IN:
        raise ValueError(
            f"{profile.key!r} is a built-in camera. Choose another key rather than "
            "shadowing published manufacturer geometry."
        )
    profile.validate()

    store = path or _store_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if store.exists():
        try:
            existing = json.loads(store.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            existing = {}

    entry = profile.to_dict()
    entry["source"] = "user"
    existing[profile.key] = entry
    store.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return store


def delete_user_profile(key: str, path: Path | None = None) -> bool:
    store = path or _store_path()
    if not store.exists():
        return False
    try:
        existing = json.loads(store.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return False
    if key not in existing:
        return False
    del existing[key]
    store.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return True


def all_profiles(path: Path | None = None) -> dict[str, CameraProfile]:
    """Built-in cameras plus the operator's own."""
    profiles = dict(BUILT_IN)
    profiles.update(load_user_profiles(path))
    return profiles


def resolve(name: str, path: Path | None = None) -> tuple[CameraProfile, bool]:
    """Look up a camera, reporting whether it was actually found.

    Returns the profile and a flag saying whether the name was recognised. The flag is
    the point: a caller that knows the geometry is a placeholder can say so, and a
    survey planned on a guess can be labelled as such instead of quietly shipping.
    """
    key = (name or "").strip().lower().replace(" ", "").replace("-", "")
    profiles = all_profiles(path)

    if key in profiles:
        return profiles[key], True

    for candidate_key, profile in profiles.items():
        if candidate_key.replace("_", "") == key.replace("_", ""):
            return profile, True

    return profiles["custom"], False


def require(name: str, path: Path | None = None) -> CameraProfile:
    """Look up a camera, refusing to substitute a placeholder."""
    profile, known = resolve(name, path)
    if not known:
        available = ", ".join(sorted(all_profiles(path)))
        raise UnknownCamera(
            f"Camera {name!r} is not in the database. Add it with save_user_profile, "
            f"or choose one of: {available}."
        )
    return profile


def describe(name: str, altitude_m: float = 60.0,
             path: Path | None = None) -> dict[str, Any]:
    """A camera's geometry and what it yields at a working altitude."""
    profile, known = resolve(name, path)
    payload = profile.to_dict()
    payload["known"] = known
    payload["altitude_m"] = altitude_m
    payload["gsd_cm"] = round(profile.gsd_cm(altitude_m), 3)

    width, height = profile.footprint_m(altitude_m)
    payload["footprint_m"] = [round(width, 1), round(height, 1)]

    if not known:
        payload["warning"] = (
            f"{name!r} is not a known camera. These figures come from the generic "
            "placeholder profile and do not describe your equipment. Add the real "
            "sensor geometry before planning a survey to a GSD specification."
        )
    return payload
