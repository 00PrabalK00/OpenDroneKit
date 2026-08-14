"""Radiometric thermal imagery.

A thermal photograph and a temperature measurement are different things. Most thermal
JPEGs are colour-mapped pictures: the palette is a rendering choice, and reading a
temperature back out of the pixel colours is guesswork. A radiometric file additionally
carries the raw sensor counts and the calibration constants needed to convert them, and
only those can be measured.

This module converts raw counts to temperature using the Planck parameters the camera
records, applies emissivity and reflected-temperature correction, and **refuses** any
file that carries no radiometric data rather than inferring temperatures from a
palette. A number invented from a colour ramp would look exactly like a measurement.

Calibration follows the FLIR convention: raw counts relate to radiance through

    raw = R1 / (R2 * (exp(B / T) - F)) - O

which inverts to

    T = B / ln(R1 / (R2 * (raw + O)) + F)

with T in kelvin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ABSOLUTE_ZERO_C = -273.15

# Stefan-Boltzmann is not needed here; atmospheric transmission is treated as unity
# unless the file states otherwise, and that assumption is reported rather than hidden.
DEFAULT_TRANSMISSION = 1.0


class NotRadiometric(ValueError):
    """Raised when a file carries no temperature data, only a rendered image."""


@dataclass
class Calibration:
    """The camera constants needed to turn raw counts into kelvin."""

    r1: float
    r2: float
    b: float
    f: float
    o: float
    emissivity: float = 0.95
    reflected_temp_c: float = 20.0
    transmission: float = DEFAULT_TRANSMISSION

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "Calibration":
        """Build from camera metadata, refusing when the constants are absent.

        Every one of R1, R2, B, F and O is required: a missing constant cannot be
        defaulted without silently changing every temperature in the image.
        """
        required = {
            "r1": ("PlanckR1", "planck_r1"),
            "r2": ("PlanckR2", "planck_r2"),
            "b": ("PlanckB", "planck_b"),
            "f": ("PlanckF", "planck_f"),
            "o": ("PlanckO", "planck_o"),
        }
        values: dict[str, float] = {}
        missing: list[str] = []
        for field, keys in required.items():
            for key in keys:
                if key in metadata:
                    values[field] = float(metadata[key])
                    break
            else:
                missing.append(keys[0])

        if missing:
            raise NotRadiometric(
                "Missing calibration constants: " + ", ".join(missing) +
                ". Without them raw counts cannot be converted to temperature, and "
                "guessing a default would change every value in the image."
            )

        return cls(
            **values,
            emissivity=float(metadata.get("Emissivity", metadata.get("emissivity", 0.95))),
            reflected_temp_c=float(
                metadata.get("ReflectedApparentTemperature",
                             metadata.get("reflected_temp_c", 20.0))
            ),
            transmission=float(
                metadata.get("AtmosphericTransmission",
                             metadata.get("transmission", DEFAULT_TRANSMISSION))
            ),
        )


# Nothing a thermal camera images is below absolute zero or above a few thousand
# kelvin. A count outside the sensor's valid range can still satisfy the formula
# arithmetically and yield an absurd temperature, so the result is range-checked as
# well as the intermediate.
MIN_PHYSICAL_KELVIN = 1.0
MAX_PHYSICAL_KELVIN = 5000.0


def raw_to_kelvin(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Convert raw sensor counts to kelvin, before emissivity correction.

    Pixels with no physical solution become NaN. They must not carry a wrapped,
    clipped or absurd value, because a temperature field is read as measurement and
    a nonsense number there is indistinguishable from a real one.
    """
    counts = np.asarray(raw, dtype=np.float64)
    denominator = calibration.r2 * (counts + calibration.o)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = calibration.r1 / denominator + calibration.f
        # log(x) needs x > 0, and log(1) == 0 would divide by zero.
        usable = np.isfinite(ratio) & (ratio > 0) & (np.abs(ratio - 1.0) > 1e-12)
        logarithm = np.log(np.where(usable, ratio, np.nan))
        kelvin = np.where(usable, calibration.b / logarithm, np.nan)

    physical = np.isfinite(kelvin) & (kelvin > MIN_PHYSICAL_KELVIN) & (kelvin < MAX_PHYSICAL_KELVIN)
    return np.where(physical, kelvin, np.nan)


def kelvin_to_raw(kelvin: np.ndarray | float, calibration: Calibration) -> np.ndarray:
    """Inverse of `raw_to_kelvin`, used to convert the reflected-ambient term."""
    temperature = np.asarray(kelvin, dtype=np.float64)
    return (
        calibration.r1 / (calibration.r2 * (np.exp(calibration.b / temperature) - calibration.f))
        - calibration.o
    )


def correct_for_emissivity(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Remove the reflected ambient component from the measured signal.

    A surface with emissivity below one also reflects its surroundings, so part of the
    measured radiance did not originate from the surface. Ignoring this reads a shiny
    metal panel as far cooler than it is.
    """
    emissivity = float(np.clip(calibration.emissivity, 0.01, 1.0))
    reflected_raw = kelvin_to_raw(calibration.reflected_temp_c - ABSOLUTE_ZERO_C, calibration)
    transmission = float(np.clip(calibration.transmission, 0.01, 1.0))

    measured = np.asarray(raw, dtype=np.float64)
    surface = (measured - (1.0 - emissivity) * transmission * reflected_raw) / (
        emissivity * transmission
    )
    return surface


def raw_to_celsius(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Full conversion: counts to surface temperature in Celsius."""
    corrected = correct_for_emissivity(raw, calibration)
    return raw_to_kelvin(corrected, calibration) + ABSOLUTE_ZERO_C


@dataclass
class ThermalImage:
    """A temperature field, not a picture."""

    celsius: np.ndarray
    calibration: Calibration
    source: str = ""

    @property
    def stats(self) -> dict[str, float]:
        finite = self.celsius[np.isfinite(self.celsius)]
        if finite.size == 0:
            return {}
        return {
            "min_c": float(finite.min()), "max_c": float(finite.max()),
            "mean_c": float(finite.mean()), "median_c": float(np.median(finite)),
            "std_c": float(finite.std()),
        }

    def region_stats(self, row0: int, col0: int, row1: int, col1: int) -> dict[str, float]:
        """Temperature statistics over a rectangular region."""
        window = self.celsius[min(row0, row1):max(row0, row1) + 1,
                              min(col0, col1):max(col0, col1) + 1]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            return {}
        return {
            "min_c": float(finite.min()), "max_c": float(finite.max()),
            "mean_c": float(finite.mean()), "pixels": int(finite.size),
        }

    def point_temperature(self, row: int, col: int) -> float:
        value = float(self.celsius[row, col])
        return value

    def anomalies(self, sigma: float = 3.0) -> dict[str, Any]:
        """Regions departing from the scene by more than `sigma` standard deviations.

        A statistical outlier is not a fault: a sunlit surface is legitimately hotter
        than its surroundings. The result names candidates for inspection and says so.
        """
        finite = self.celsius[np.isfinite(self.celsius)]
        if finite.size == 0:
            return {"hot_pixels": 0, "cold_pixels": 0, "note": "No valid temperature data."}

        mean = float(finite.mean())
        std = float(finite.std())
        if std <= 0:
            return {"hot_pixels": 0, "cold_pixels": 0,
                    "note": "The scene is isothermal; no anomaly can be distinguished."}

        hot = np.isfinite(self.celsius) & (self.celsius > mean + sigma * std)
        cold = np.isfinite(self.celsius) & (self.celsius < mean - sigma * std)
        return {
            "hot_pixels": int(hot.sum()), "cold_pixels": int(cold.sum()),
            "threshold_hot_c": round(mean + sigma * std, 2),
            "threshold_cold_c": round(mean - sigma * std, 2),
            "sigma": sigma,
            "note": (
                "These are statistical outliers, not confirmed faults. A sunlit surface "
                "is legitimately hotter than its surroundings; each candidate needs "
                "review against the visual image."
            ),
        }


def load_radiometric(
    raw_counts: np.ndarray, metadata: dict[str, Any], source: str = ""
) -> ThermalImage:
    """Build a temperature field from raw counts and camera metadata."""
    calibration = Calibration.from_metadata(metadata)
    return ThermalImage(
        celsius=raw_to_celsius(raw_counts, calibration),
        calibration=calibration,
        source=source,
    )


def load_radiometric_file(path: str | Path) -> ThermalImage:
    """Read a radiometric image from disk.

    Supported: a sidecar JSON carrying `raw` counts and calibration constants, which
    is what `exiftool -b -RawThermalImage` plus a metadata dump produces. Reading the
    embedded APP1 payload of a FLIR or DJI JPEG directly is not implemented, and this
    says so rather than falling back to the rendered pixels.
    """
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        raw = np.asarray(payload["raw"], dtype=np.float64)
        return load_radiometric(raw, payload.get("metadata", payload), source=str(target))

    raise NotRadiometric(
        f"{target.name}: extracting the embedded radiometric payload from a thermal "
        "JPEG is not implemented. Export it first, for example with "
        "`exiftool -b -RawThermalImage`, and supply the counts with the camera's "
        "Planck constants. The rendered pixels are a palette, not temperatures."
    )


def write_thermal_geotiff(
    image: ThermalImage,
    path: str | Path,
    *,
    epsg: int,
    west: float,
    north: float,
    pixel_size: float,
) -> str:
    """Write the temperature field as a georeferenced raster in Celsius.

    Values are temperatures, not palette indices, so the raster is directly usable in
    GIS and its units are unambiguous.
    """
    from . import geo

    return geo.write_geotiff(
        path, image.celsius.astype(np.float32), epsg=epsg, west=west, north=north,
        pixel_size=pixel_size, nodata=float("nan"), cog=False,
    )
