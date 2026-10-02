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
from dataclasses import dataclass, field
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
class FlirAtmosphere:
    """FLIR's atmospheric-path model, from the values the camera records.

    The path between camera and object both absorbs the object's radiation and emits
    its own; an external IR window does the same. FLIR cameras record what is needed to
    model that: atmospheric temperature, humidity, distance and the empirical
    transmission coefficients alpha1/alpha2/beta1/beta2/X. Transmission is computed as

        h2o = RH * exp(1.5587 + 0.06939 Ta - 0.00027816 Ta^2 + 0.00000068455 Ta^3)
        tau = X exp(-sqrt(d/2) (a1 + b1 sqrt(h2o))) + (1 - X) exp(-sqrt(d/2) (a2 + b2 sqrt(h2o)))

    once for each half of the path (camera to window, window to object).
    """

    atmospheric_temp_c: float
    object_distance_m: float
    relative_humidity: float  # fraction, 0..1
    ir_window_temp_c: float = 20.0
    ir_window_transmission: float = 1.0
    alpha1: float = 0.006569
    alpha2: float = 0.01262
    beta1: float = -0.002276
    beta2: float = -0.00667
    x: float = 1.9

    def transmission(self) -> float:
        """Transmission of one half of the camera-to-object path."""
        t = self.atmospheric_temp_c
        h2o = self.relative_humidity * math.exp(
            1.5587 + 0.06939 * t - 0.00027816 * t ** 2 + 0.00000068455 * t ** 3)
        root = math.sqrt(max(self.object_distance_m, 0.0) / 2.0)
        return (self.x * math.exp(-root * (self.alpha1 + self.beta1 * math.sqrt(h2o)))
                + (1 - self.x) * math.exp(-root * (self.alpha2 + self.beta2 * math.sqrt(h2o))))


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
    # Set for FLIR files, which record the atmospheric path; None keeps the simple
    # emissivity/reflection model with ``transmission`` above.
    atmosphere: FlirAtmosphere | None = None

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


def correct_for_flir_path(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Object-only counts after removing reflection, atmosphere and IR-window terms.

    The measured signal is the object's emission attenuated by both halves of the
    atmospheric path and the window, plus: the reflected surroundings, the emission of
    each air path, and the window's own emission. Each is converted to counts through
    the camera's Planck curve and removed.
    """
    atmosphere = calibration.atmosphere
    assert atmosphere is not None
    emissivity = float(np.clip(calibration.emissivity, 0.01, 1.0))
    window = float(np.clip(atmosphere.ir_window_transmission, 0.01, 1.0))
    tau = float(np.clip(atmosphere.transmission(), 0.01, 1.0))

    def counts(celsius: float) -> float:
        return float(kelvin_to_raw(celsius - ABSOLUTE_ZERO_C, calibration))

    reflected = counts(calibration.reflected_temp_c)
    air = counts(atmosphere.atmospheric_temp_c)
    window_counts = counts(atmosphere.ir_window_temp_c)
    measured = np.asarray(raw, dtype=np.float64)
    return (
        measured / (emissivity * tau * window * tau)
        - (1 - tau) / (emissivity * tau) * air
        - (1 - tau) / (emissivity * tau * window * tau) * air
        - (1 - window) / (emissivity * tau * window) * window_counts
        - (1 - emissivity) / emissivity * reflected
    )


def raw_to_celsius(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Full conversion: counts to surface temperature in Celsius."""
    if calibration.atmosphere is not None:
        corrected = correct_for_flir_path(raw, calibration)
        return raw_to_kelvin(corrected, calibration) + ABSOLUTE_ZERO_C
    corrected = correct_for_emissivity(raw, calibration)
    return raw_to_kelvin(corrected, calibration) + ABSOLUTE_ZERO_C


@dataclass
class ThermalImage:
    """A temperature field, not a picture."""

    celsius: np.ndarray
    # None for a temperature map read back from disk: its counts were converted when it
    # was written, and the constants are not carried in the raster.
    calibration: Calibration | None
    source: str = ""
    # How the temperatures were obtained and how far that has been checked: decoder,
    # unit, validation status. Reported with every product built from the frame.
    provenance: dict[str, Any] = field(default_factory=dict)

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


SIDECAR_PROVENANCE = {
    "decoder": "sidecar_json_planck",
    "temperature_kind": "radiometric_surface_temperature",
    "unit": "celsius",
    "validation": "Planck conversion verified by round trip (tests/test_thermal.py); "
                  "the counts and constants are as supplied in the sidecar.",
}


def _sniff(path: Path) -> str:
    """Which radiometric container a file is, from its bytes rather than its name."""
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head[:2] == b"\xff\xd8":
        from .thermal_dji import is_dji_rjpeg
        from .thermal_flir import is_flir_radiometric_jpeg

        if is_flir_radiometric_jpeg(path):
            return "flir_jpeg"
        if is_dji_rjpeg(path):
            return "dji_rjpeg"
        return "jpeg"
    if head in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def load_radiometric_file(path: str | Path) -> ThermalImage:
    """Read a radiometric image from disk, by what the file actually contains.

    * a sidecar JSON with ``raw`` counts and Planck constants;
    * a FLIR radiometric JPEG (FLIR cameras, DJI XT2/XT R): counts and constants read
      from its FFF records by ``core.thermal_flir``;
    * a DJI R-JPEG (H20T, H20N, H30T, M2EA, M30T, M3T, M3TD, M4T, XT S): decoded by the
      DJI Thermal SDK from the user-configured path, ``core.thermal_dji``;
    * a FLIR TLinear TIFF (Duo Pro R): ``core.thermal_tlinear``.

    Anything else is refused. The rendered pixels of a thermal JPEG are a palette, and a
    temperature read from them would be invented.
    """
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        raw = np.asarray(payload["raw"], dtype=np.float64)
        image = load_radiometric(raw, payload.get("metadata", payload), source=str(target))
        image.provenance = dict(SIDECAR_PROVENANCE)
        return image
    if not target.is_file():
        raise FileNotFoundError(str(target))

    kind = _sniff(target)
    if kind == "flir_jpeg":
        from .thermal_flir import read_flir_jpeg

        parsed = read_flir_jpeg(target)
        calibration = parsed.calibration()
        return ThermalImage(
            celsius=raw_to_celsius(parsed.counts, calibration), calibration=calibration,
            source=str(target),
            provenance={
                "decoder": "flir_fff_cleanroom_planck",
                "camera_model": parsed.metadata.get("CameraModel", ""),
                "temperature_kind": "radiometric_surface_temperature",
                "unit": "celsius",
                "validation": (
                    "Counts and Planck/atmosphere constants are read from the file. The "
                    "parser is checked against ExifTool dumps and against FLIR cameras' "
                    "own on-screen readings in sample images (tests/"
                    "test_thermal_real_data.py). This file's temperatures have NOT been "
                    "compared with an independent reference measurement."),
            },
        )
    if kind == "dji_rjpeg":
        from .thermal_dji import decode_rjpeg

        return decode_rjpeg(target)
    if kind == "tiff":
        from .thermal_tlinear import is_tlinear_tiff, read_tlinear_tiff

        if is_tlinear_tiff(target):
            return read_tlinear_tiff(target)
        raise NotRadiometric(
            f"{target.name} is a TIFF without radiometric metadata this toolkit can "
            "verify (no FLIR TLinear gain). If it is a temperature raster, open it as a "
            "temperature GeoTIFF and confirm its unit.")

    raise NotRadiometric(
        f"{target.name}: no radiometric payload was found (no FLIR FFF record, no DJI "
        "R-JPEG APP3 data). The rendered pixels are a palette, not temperatures. Supply "
        "the original radiometric file, or export the counts first, for example with "
        "`exiftool -b -RawThermalImage`, together with the camera's Planck constants.")


# The tags write_radiometric_thermal_map stamps on the temperature GeoTIFF it writes.
# A float raster without them may be a DSM or an index; reading one as degrees would
# invent temperatures, so a unit is required: declared by the file or confirmed by
# the user, never assumed.
TEMPERATURE_TAGS = {"ODK_VALUE_KIND": "measured_surface_temperature", "ODK_VALUE_UNIT": "celsius"}

# Band unit strings (GDAL band units / unit tags) that say a band holds temperature.
CELSIUS_UNITS = {"celsius", "degc", "deg c", "°c", "degree celsius", "degrees celsius"}
KELVIN_UNITS = {"kelvin", "k", "degk"}

# A confirmed unit still has to describe plausible surface temperatures. A DSM in
# metres confirmed as Celsius by mistake usually falls outside this.
PLAUSIBLE_SURFACE_C = (-90.0, 1500.0)


class UnitConfirmationNeeded(NotRadiometric):
    """The raster may hold temperatures, but it does not say so: ask the user."""


def _declared_unit(raster) -> str | None:
    tags = {k: str(v).strip().lower() for k, v in raster.tags().items()}
    band_tags = {k.lower(): str(v).strip().lower() for k, v in raster.tags(1).items()}
    if all(tags.get(k.lower(), tags.get(k)) == v for k, v in TEMPERATURE_TAGS.items()):
        return "celsius"
    unit = (raster.units[0] or "").strip().lower() if raster.units else ""
    unit = unit or band_tags.get("unit", "") or band_tags.get("units", "")
    if unit in CELSIUS_UNITS:
        return "celsius"
    if unit in KELVIN_UNITS:
        return "kelvin"
    return None


def load_temperature_geotiff(path: str | Path, confirmed_unit: str | None = None) -> ThermalImage:
    """Read a temperature GeoTIFF (band 1).

    Accepted when the raster says it holds temperature (the ODK tags this toolkit
    writes, or a Celsius/kelvin band unit), or when the user has confirmed its unit
    (``confirmed_unit`` = "celsius" or "kelvin"). Otherwise ``UnitConfirmationNeeded``
    is raised with the reason: a DSM or index read as degrees would invent
    temperatures.
    """
    import rasterio

    target = Path(path)
    with rasterio.open(target) as raster:
        declared = _declared_unit(raster)
        file_unit = (raster.units[0] or "") if raster.units else ""
        band = raster.read(1, masked=True).astype(np.float64)
    confirmed = (confirmed_unit or "").strip().lower() or None
    if confirmed not in (None, "celsius", "kelvin"):
        raise NotRadiometric(f"Unknown temperature unit {confirmed_unit!r}; use celsius or kelvin.")
    unit = declared or confirmed
    if unit is None:
        said = f"; its band unit says {file_unit!r}" if file_unit else ""
        raise UnitConfirmationNeeded(
            f"{target.name} does not say it holds temperatures (it lacks "
            f"{', '.join(f'{k}={v}' for k, v in TEMPERATURE_TAGS.items())} and has no "
            f"Celsius or kelvin band unit{said}). A DSM or index raster read as degrees "
            "would invent temperatures. If you know its values are temperatures, "
            "confirm the unit (Celsius or kelvin).")
    celsius = band.filled(np.nan)
    if unit == "kelvin":
        celsius = celsius + ABSOLUTE_ZERO_C
    finite = celsius[np.isfinite(celsius)]
    if finite.size == 0:
        raise NotRadiometric(f"{target.name} holds no valid values.")
    low, high = (float(v) for v in np.percentile(finite, [1, 99]))
    if low < PLAUSIBLE_SURFACE_C[0] or high > PLAUSIBLE_SURFACE_C[1]:
        raise NotRadiometric(
            f"{target.name}: read as {unit}, its values span {low:.1f} to {high:.1f} °C "
            "(1st to 99th percentile), which is not a plausible surface temperature. It "
            "is probably not a temperature raster.")
    provenance: dict[str, Any] = {
        "decoder": "temperature_geotiff",
        "unit": "celsius",
        "unit_source": "declared_by_file" if declared else "confirmed_by_user",
        "temperature_kind": "temperature_raster_as_supplied",
        "validation": ("Values are read as stored; how they were produced is outside "
                       "this toolkit and is not verified here."),
    }
    if declared is None:
        provenance["file_unit_metadata"] = file_unit or None
        if file_unit:
            provenance["note"] = (
                f"The file's band unit says {file_unit!r}; the user confirmed {unit}.")
    if unit == "kelvin":
        provenance["converted_from"] = "kelvin"
    return ThermalImage(celsius=celsius, calibration=None, source=str(target),
                        provenance=provenance)


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
