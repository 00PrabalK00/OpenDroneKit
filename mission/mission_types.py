"""Three mission types whose difference is the payload and the structure, not the path.

Each of these was specified and unbuilt, and each has a way of appearing to work that
this module exists to prevent.

**Pylon inspection.** A transmission tower is not a cylinder. The photographs a utility
needs are of named things at known heights -- the body, each crossarm, the insulator
strings hanging off them, the conductor attachments -- and an orbit flown at an
arbitrary spacing produces a hundred pictures of steelwork and none of the fitting that
was cracking. So the elevations are supplied, not guessed. Guessing them near live
conductors would be the kind of wrong that is measured in clearance, so an unspecified
structure is refused rather than approximated.

**Thermal.** A thermal sensor is a 640 x 512 imager. Flown at the altitude that gives a
20 megapixel RGB camera a fine GSD, it produces thermal pixels several times larger than
the defect being looked for, and the mosaic still comes out looking like a thermal
survey. So altitude here is derived from the thermal sensor, and a mission asked to fly
thermal with a camera that is not radiometric is refused.

**Multispectral.** Indices are ratios between bands, and a ratio is only comparable
between two surveys if both were referenced to a known reflectance. That is what the
calibration panel is for, and the shot has to be taken before and after the flight
because the light changes during it. A multispectral mission planned without those
captures yields NDVI that cannot be compared with last month's, which is the entire
reason anyone flies it monthly.

Every mission here is a real compiled plan from the mission engine -- the same geometry,
constraints, geofence and terrain handling as any other -- with the payload contract
attached and stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .cameras import UnknownCamera, require as require_camera
from .planner import MissionPlan, MissionPlanner


class MissionTypeRefused(ValueError):
    """The mission cannot be planned honestly with what was supplied."""


# Elements of a transmission tower a utility asks for by name. The gimbal angle is the
# one that puts the element in frame from a level orbit at its own height: a crossarm is
# looked at horizontally, an insulator string hangs below its attachment so it is looked
# at slightly downward, and the conductor attachment sits lower still.
PYLON_ELEMENT_TILTS = {
    "body": -10.0,
    "crossarm": 0.0,
    "insulator": -20.0,
    "conductor": -30.0,
}


@dataclass
class PylonElement:
    """One named thing on the structure, at the height it actually sits."""

    name: str
    height_m: float
    radius_m: float | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        if self.name not in PYLON_ELEMENT_TILTS:
            raise MissionTypeRefused(
                f"{self.name!r} is not a pylon element this engine plans for. Use one "
                f"of: {', '.join(sorted(PYLON_ELEMENT_TILTS))}."
            )
        if not math.isfinite(self.height_m) or self.height_m <= 0:
            raise MissionTypeRefused(
                f"{self.name}: height must be a positive measured value. A capture "
                "height guessed near energised conductors is a clearance problem, not a "
                "framing one."
            )

    @property
    def gimbal_tilt_deg(self) -> float:
        return PYLON_ELEMENT_TILTS[self.name]


def plan_pylon_inspection(
    planner: MissionPlanner,
    *,
    center_lonlat: Sequence[float],
    elements: Iterable[PylonElement | dict[str, Any]],
    standoff_m: float = 12.0,
    structure_radius_m: float = 3.0,
    camera: str = "mavic2pro",
    speed_m_s: float = 3.0,
    dwell_s: float = 2.0,
    **planner_kwargs: Any,
) -> dict[str, Any]:
    """One stacked orbit per named element, at the height that element sits at.

    Each level is flown at the element's own height with the gimbal set for it, so the
    deliverable is a set of photographs of identified fittings rather than a general
    circuit of the tower.
    """
    resolved: list[PylonElement] = []
    for entry in elements:
        resolved.append(entry if isinstance(entry, PylonElement)
                        else PylonElement(**dict(entry)))
    if not resolved:
        raise MissionTypeRefused(
            "A pylon inspection needs the elements to photograph and the height each "
            "one sits at. Without them the only honest plan is a general orbit, which "
            "is what this mission type exists to replace."
        )
    if standoff_m <= structure_radius_m:
        raise MissionTypeRefused(
            f"A standoff of {standoff_m} m is inside the structure radius of "
            f"{structure_radius_m} m. The flight path would pass through the tower."
        )
    if len(center_lonlat) < 2:
        raise MissionTypeRefused("A pylon inspection needs the structure centre as [lon, lat].")

    resolved.sort(key=lambda element: element.height_m)
    levels: list[dict[str, Any]] = []
    plans: list[MissionPlan] = []
    for element in resolved:
        radius = float(element.radius_m or standoff_m)
        if radius <= structure_radius_m:
            raise MissionTypeRefused(
                f"{element.name}: a capture radius of {radius} m is inside the "
                f"{structure_radius_m} m structure."
            )
        plan = planner.generate(
            mode="orbit",
            orbit_center_lonlat=[float(center_lonlat[0]), float(center_lonlat[1])],
            orbit_radius_m=radius,
            altitude_m=element.height_m,
            gimbal_tilt_deg=element.gimbal_tilt_deg,
            camera=camera,
            speed_m_s=speed_m_s,
            inspection_dwell_s=dwell_s,
            orbit_poi_yaw_lock=True,
            orbit_poi_lonlat=[float(center_lonlat[0]), float(center_lonlat[1])],
            **planner_kwargs,
        )
        plans.append(plan)
        levels.append({
            "element": element.name,
            "height_m": element.height_m,
            "radius_m": radius,
            "gimbal_tilt_deg": element.gimbal_tilt_deg,
            "waypoints": len(plan.waypoints),
            "capture_range_m": round(radius - structure_radius_m, 3),
        })

    return {
        "template": "pylon_inspection",
        "levels": levels,
        "plans": plans,
        "element_count": len(levels),
        "waypoint_count": sum(level["waypoints"] for level in levels),
        "elements_covered": sorted({level["element"] for level in levels}),
        "limits": [
            "Capture heights are the ones supplied, not measured by the aircraft. If "
            "they came from a drawing rather than a survey, the framing follows the "
            "drawing.",
            "Clearance to energised conductors is the operator's responsibility; this "
            "plan places the aircraft at the standoff it was given.",
        ],
    }


def _thermal_altitude_for_gsd(camera_key: str, target_gsd_cm: float) -> float:
    profile = require_camera(camera_key)
    if not profile.thermal:
        raise MissionTypeRefused(
            f"{profile.name} is not a radiometric thermal camera, so a thermal mission "
            "planned with it would produce ordinary photographs labelled as a thermal "
            "survey. Choose a thermal camera, or plan this as an RGB mission."
        )
    return profile.altitude_for_gsd_m(target_gsd_cm)


def plan_thermal_mission(
    planner: MissionPlanner,
    *,
    polygon_lonlat: Sequence[Sequence[float]],
    thermal_camera: str,
    target_gsd_cm: float,
    rgb_camera: str = "",
    front_overlap_pct: float = 80.0,
    side_overlap_pct: float = 70.0,
    speed_m_s: float = 4.0,
    **planner_kwargs: Any,
) -> dict[str, Any]:
    """A grid flown at the altitude the *thermal* sensor needs, with paired RGB.

    The altitude is the one that gives the thermal imager the requested ground sample
    distance. An RGB camera on the same aircraft will be far finer than that, which is
    the right way round: a thermal survey flown to the RGB's altitude is thermally
    useless, while an RGB image finer than it needs to be is merely larger.
    """
    if target_gsd_cm <= 0:
        raise MissionTypeRefused("A thermal mission needs a positive target GSD.")
    try:
        altitude_m = _thermal_altitude_for_gsd(thermal_camera, target_gsd_cm)
    except UnknownCamera as exc:
        raise MissionTypeRefused(str(exc)) from exc

    plan = planner.generate(
        polygon_lonlat=polygon_lonlat,
        mode="grid",
        camera=thermal_camera,
        altitude_m=altitude_m,
        front_overlap_pct=front_overlap_pct,
        side_overlap_pct=side_overlap_pct,
        speed_m_s=speed_m_s,
        **planner_kwargs,
    )

    paired = bool(rgb_camera)
    rgb_gsd_cm = None
    if paired:
        try:
            rgb_gsd_cm = round(require_camera(rgb_camera).gsd_cm(altitude_m), 3)
        except UnknownCamera as exc:
            raise MissionTypeRefused(str(exc)) from exc

    limits = [
        "Altitude is derived from the thermal sensor, not the RGB one. Thermal imagers "
        "are low resolution, so this altitude is lower than an RGB survey of the same "
        "area would need.",
        "Radiometric values depend on emissivity, reflected apparent temperature and "
        "atmosphere. The plan captures them; it does not correct them.",
    ]
    if not paired:
        limits.append(
            "No RGB camera was named, so the thermal frames will arrive without a "
            "visual pair. Anything found in them will be located on a thermal image "
            "alone, which is harder to show a client and harder to act on."
        )
    return {
        "template": "thermal",
        "plan": plan,
        "altitude_m": round(altitude_m, 3),
        "thermal_camera": thermal_camera,
        "thermal_gsd_cm": round(require_camera(thermal_camera).gsd_cm(altitude_m), 3),
        "paired_rgb": paired,
        "rgb_camera": rgb_camera,
        "rgb_gsd_cm": rgb_gsd_cm,
        "capture_contract": {
            "radiometric": True,
            "paired_capture": paired,
            "trigger": "per_capture_point",
        },
        "limits": limits,
    }


def plan_multispectral_mission(
    planner: MissionPlanner,
    *,
    polygon_lonlat: Sequence[Sequence[float]],
    payload_key: str,
    camera: str = "mavic3e",
    calibration_panel_lonlat: Sequence[float] | None = None,
    altitude_m: float = 60.0,
    front_overlap_pct: float = 80.0,
    side_overlap_pct: float = 75.0,
    speed_m_s: float = 6.0,
    **planner_kwargs: Any,
) -> dict[str, Any]:
    """A grid with synchronised band capture and a calibration shot at each end.

    The panel captures are part of the mission, not a note in the flight log. Without
    both of them the indices computed from this flight cannot be compared with the ones
    computed from the last, which is what a monthly vegetation survey is for.
    """
    from .payloads import UnknownPayload, get_payload

    try:
        payload = get_payload(payload_key)
    except UnknownPayload as exc:
        raise MissionTypeRefused(str(exc)) from exc
    if payload.kind != "multispectral":
        raise MissionTypeRefused(
            f"{payload.name} is a {payload.kind} payload. A multispectral mission "
            "planned around it would capture no bands and produce no index."
        )
    if not payload.bands_nm:
        raise MissionTypeRefused(
            f"{payload.name} states no band centres, so nothing downstream can know "
            "which band is which."
        )
    if payload.requires_calibration and calibration_panel_lonlat is None:
        raise MissionTypeRefused(
            f"{payload.name} needs a reflectance panel capture before and after the "
            "flight. Without the panel position this mission cannot include them, and "
            "indices from it will not be comparable with any other survey. Supply the "
            "panel location, or accept single-flight relative values explicitly."
        )

    plan = planner.generate(
        polygon_lonlat=polygon_lonlat,
        mode="grid",
        camera=camera,
        altitude_m=altitude_m,
        front_overlap_pct=front_overlap_pct,
        side_overlap_pct=side_overlap_pct,
        speed_m_s=speed_m_s,
        **planner_kwargs,
    )

    calibration_captures: list[dict[str, Any]] = []
    if calibration_panel_lonlat is not None:
        if len(calibration_panel_lonlat) < 2:
            raise MissionTypeRefused("The calibration panel position needs [lon, lat].")
        panel = [float(calibration_panel_lonlat[0]), float(calibration_panel_lonlat[1])]
        for when in ("before", "after"):
            calibration_captures.append({
                "when": when, "lonlat": panel, "command": "calibrate",
                "altitude_m": 2.0,
                "note": ("Reflectance panel, shot from directly above with the panel "
                         "filling the frame and the operator's shadow clear of it."),
            })

    return {
        "template": "multispectral",
        "plan": plan,
        "payload": payload.to_dict(),
        "bands_nm": list(payload.bands_nm),
        "band_count": len(payload.bands_nm),
        "calibration_captures": calibration_captures,
        "capture_contract": {
            "synchronised_bands": True,
            "band_count": len(payload.bands_nm),
            "calibrated": bool(calibration_captures),
            "trigger": payload.capture_command(),
        },
        "limits": [
            "All bands are captured at one trigger, so band-to-band alignment depends "
            "on the instrument, not on the flight plan.",
            "Indices are comparable between surveys only where both flights carry "
            "their panel captures and both were flown in comparable light.",
        ],
    }
