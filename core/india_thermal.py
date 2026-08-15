"""Honest browser artifacts for radiometric thermal survey products.

``core.thermal`` owns the temperature measurement.  This module does not infer
temperature from a colour palette: it packages an already-calibrated
``ThermalImage`` as a georeferenced GeoTIFF plus a small JSON grid that the
local Hub can render without a server or a GeoTIFF JavaScript dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .india_geospatial import IndiaPackRefused, IndiaPackResult
from .thermal import ThermalImage, write_thermal_geotiff


@dataclass(frozen=True)
class ThermalRegistration:
    """Evidence that a thermal grid has already been registered to an RGB image."""

    method: str
    rgb_width: int
    rgb_height: int
    residual_px: float
    validated_by: str

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("Thermal registration method is required.")
        if self.rgb_width < 1 or self.rgb_height < 1:
            raise ValueError("Registered RGB dimensions must be positive.")
        if not math.isfinite(self.residual_px) or self.residual_px < 0:
            raise ValueError("Registration residual_px must be finite and non-negative.")
        if not self.validated_by.strip():
            raise ValueError("Registration must name who or what validated it.")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "validated": True, "target": "rgb"}


def _json_temperatures(celsius: np.ndarray) -> list[float | None]:
    return [round(float(value), 6) if math.isfinite(float(value)) else None
            for value in celsius.reshape(-1)]


def write_radiometric_thermal_map(
    image: ThermalImage,
    output_dir: str | Path,
    *,
    epsg: int,
    west: float,
    north: float,
    pixel_size: float,
    registration: ThermalRegistration | None = None,
) -> IndiaPackResult:
    """Write one measured temperature map for GIS and the local Hub.

    The JSON values are copied from ``ThermalImage.celsius``.  They are not
    interpolated, recoloured values, or a visual proxy.  RGB comparison metadata
    is included only when an explicit validated registration is supplied.
    """

    field = np.asarray(image.celsius, dtype=np.float64)
    if field.ndim != 2 or not field.size:
        raise IndiaPackRefused("A thermal map requires a non-empty 2D temperature field.")
    finite = field[np.isfinite(field)]
    if not finite.size:
        raise IndiaPackRefused("The thermal image contains no valid temperature measurements.")
    if not all(math.isfinite(float(value)) for value in (west, north, pixel_size)):
        raise IndiaPackRefused("Thermal georeferencing values must be finite.")
    if pixel_size <= 0:
        raise IndiaPackRefused("Thermal pixel_size must be positive.")

    try:
        from rasterio.crs import CRS

        crs = CRS.from_epsg(int(epsg))
    except Exception as exc:  # pragma: no cover - rasterio supplies the detail
        raise IndiaPackRefused(f"Thermal EPSG:{epsg} is not a usable CRS.") from exc
    if not crs.is_projected:
        raise IndiaPackRefused(
            "Thermal survey maps require a projected CRS; degrees cannot support "
            "metric pixel size or 3D surface association."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geotiff = Path(write_thermal_geotiff(
        image, output / "thermal_temperature.tif", epsg=int(epsg),
        west=float(west), north=float(north), pixel_size=float(pixel_size),
    ))

    # Make the measurement origin explicit in the GIS artifact itself.
    import rasterio

    with rasterio.open(geotiff, "r+") as raster:
        raster.update_tags(
            ODK_VALUE_KIND="measured_surface_temperature",
            ODK_VALUE_UNIT="celsius",
            ODK_TEMPERATURE_SOURCE="radiometric_counts_via_core.thermal",
            ODK_INTERPOLATED="false",
        )

    height, width = field.shape
    payload: dict[str, Any] = {
        "type": "odk-thermal-map",
        "schema_version": 1,
        "width": int(width),
        "height": int(height),
        "crs_epsg": int(epsg),
        "transform": [float(pixel_size), 0.0, float(west),
                      0.0, -float(pixel_size), float(north)],
        "unit": "celsius",
        "values": _json_temperatures(field),
        "stats": image.stats,
        "source": image.source,
        "temperature_source": "radiometric_counts_via_core.thermal",
        "interpolated": False,
        "calibration": asdict(image.calibration),
        "geotiff": geotiff.name,
    }
    if registration is not None:
        payload["registration"] = registration.to_dict()

    manifest = output / "thermal_map.json"
    manifest.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = output / "thermal_summary.json"
    summary.write_text(json.dumps({
        "status": "complete",
        "measurement": "radiometric surface temperature",
        "unit": "celsius",
        "crs_epsg": int(epsg),
        "valid_pixels": int(finite.size),
        "stats": image.stats,
        "registration": registration.to_dict() if registration else {
            "available": False,
            "reason": "No validated RGB/thermal registration was supplied.",
        },
        "limits": [
            "Temperatures come from radiometric counts and camera calibration, not palette colours.",
            "No spatial interpolation was applied to the temperature grid.",
            "A statistical or visual hotspot is not a confirmed equipment fault.",
        ],
    }, indent=2, allow_nan=False), encoding="utf-8")

    return IndiaPackResult(
        summary_path=str(summary),
        artifact_paths=(str(geotiff), str(manifest)),
    )

