"""MicaSense multispectral band reading, radiometric calibration and band alignment.

Everything here is measured from the files themselves:

* Band TIFFs are decoded as stored (RedEdge-P panel captures are 12-bit packed; that
  needs ``imagecodecs`` for tifffile, and rasterio/GDAL is used when it is missing).
* Radiance follows MicaSense's published sensor model (MIT, micasense/imageprocessing
  ``image.py``): black level, vignette polynomial, row-gradient term and the per-band
  radiometric coefficients from the image XMP, divided by gain x exposure.
* Reflectance comes from one of two real references, and says which:
    - ``dls``: the DLS2 sun sensor's horizontal irradiance in the XMP
      (``reflectance = pi * L / E``);
    - ``panel``: a calibration-panel capture, using the panel albedo the camera read
      from the panel QR code (``Camera:Albedo``) and the panel area it located
      (``Camera:ReflectArea``). A band whose panel albedo is absent is refused, not
      filled with a typical value, unless the caller supplies the albedo from the
      panel's calibration sheet and names that source.
* Bands are aligned to the rig's reference band by undistorting each band with its own
  lens model and then estimating a homography by ECC on gradient images. The ECC
  correlation is reported per band so a poor alignment is visible.

The Altum LWIR band is radiometric thermal stored in centi-kelvin; it is decoded to
degrees Celsius linearly and never passed through the reflectance path.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

# Canonical band keys used across OpenDroneKit (india_agriculture.INDEX_BANDS uses these).
_BAND_ALIASES = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "near infrared": "nir",
    "red edge": "red_edge",
    "rededge": "red_edge",
    "red_edge": "red_edge",
    "panchro": "panchro",
    "pan": "panchro",
    "lwir": "lwir",
}

DLS_DOWNLOAD_NOTE = (
    "Reflectance needs either the DLS sun-sensor irradiance in the band XMP or a "
    "calibration-panel capture from the same flight."
)


class MultispectralError(RuntimeError):
    """A band or capture cannot be calibrated honestly; the message says why."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_band_pixels(path: str | Path) -> np.ndarray:
    """Decode a band TIFF to its stored integer values.

    RedEdge-P panel captures are 12-bit packed (BitsPerSample=12). tifffile decodes
    those only with ``imagecodecs``; rasterio/GDAL decodes them natively and is used
    as the fallback, so either installed reader is enough.
    """
    path = Path(path)
    errors: list[str] = []
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except Exception as exc:  # noqa: BLE001 - try the next reader, keep the reason
        errors.append(f"tifffile: {exc}")
    try:
        import warnings

        import rasterio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # band TIFFs are not georeferenced rasters
            with rasterio.open(path) as dataset:
                return np.asarray(dataset.read(1))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rasterio: {exc}")
    raise MultispectralError(
        f"Cannot decode {path.name}: " + "; ".join(errors)
        + ". 12-bit packed MicaSense TIFFs need `pip install imagecodecs` (BSD) or rasterio."
    )


_XMP_TAG = re.compile(r"<(?P<ns>[\w]+):(?P<tag>[\w]+)>(?P<body>.*?)</(?P=ns):(?P=tag)>", re.S)
_RDF_LI = re.compile(r"<rdf:li>(.*?)</rdf:li>", re.S)


def parse_xmp(xmp: str) -> dict[str, Any]:
    """Flatten MicaSense/Pix4D XMP into {tag: value}; rdf:Seq becomes a list of strings."""
    values: dict[str, Any] = {}
    for match in _XMP_TAG.finditer(xmp):
        tag, body = match.group("tag"), match.group("body")
        items = _RDF_LI.findall(body)
        if items:
            values[tag] = [item.strip() for item in items]
        elif "<" not in body:
            values[tag] = body.strip()
    return values


def _floats(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[float] = []
    for item in value:
        out.extend(float(part) for part in str(item).replace(" ", "").split(",") if part)
    return out


def _rational(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dms(value: Any, ref: Any) -> float | None:
    if not value or len(value) < 6:
        return None
    parts = [float(value[i]) / float(value[i + 1]) if value[i + 1] else 0.0 for i in (0, 2, 4)]
    degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    return -degrees if str(ref).upper() in {"S", "W"} else degrees


def _version_tuple(text: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", text.split("-")[0])
    return tuple(int(n) for n in numbers[:3])


@dataclass
class BandImage:
    """One band file of one capture, with the metadata the radiometric model needs."""

    path: str
    band_key: str
    band_name: str
    band_index: int
    capture_id: str
    camera_model: str
    camera_serial: str
    firmware: str
    width: int
    height: int
    bits_per_pixel: int
    black_level: float
    exposure_s: float
    gain: float
    radiometric_cal: list[float]
    vignette_center: list[float]
    vignette_polynomial: list[float]
    vignette_poly2d: list[float]
    vignette_poly2d_exponents: list[float]
    center_wavelength_nm: float | None
    bandwidth_nm: float | None
    focal_length_mm: float | None
    focal_plane_px_per_mm: tuple[float, float] | None
    principal_point_mm: tuple[float, float] | None
    distortion: list[float]
    rig_relatives_deg: list[float]
    rig_reference_index: int | None
    spectral_irradiance: float | None
    horizontal_irradiance: float | None
    direct_irradiance: float | None
    scattered_irradiance: float | None
    solar_elevation_rad: float | None
    irradiance_scale: float
    panel_albedo: float | None
    panel_region: list[tuple[int, int]] | None
    panel_serial: str | None
    latitude: float | None
    longitude: float | None
    altitude_m: float | None
    datetime_original: str | None
    flight_id: str | None
    _pixels: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def is_thermal(self) -> bool:
        return self.band_key == "lwir"

    def pixels(self) -> np.ndarray:
        if self._pixels is None:
            self._pixels = read_band_pixels(self.path)
        return self._pixels

    def release(self) -> None:
        self._pixels = None

    def camera_matrix(self) -> np.ndarray:
        """Pinhole K in pixels from the XMP lens model, or refuse when it is absent."""
        if not (self.focal_length_mm and self.focal_plane_px_per_mm and self.principal_point_mm):
            raise MultispectralError(f"{Path(self.path).name} carries no lens model in its XMP.")
        fx = self.focal_length_mm * self.focal_plane_px_per_mm[0]
        fy = self.focal_length_mm * self.focal_plane_px_per_mm[1]
        cx = self.principal_point_mm[0] * self.focal_plane_px_per_mm[0]
        cy = self.principal_point_mm[1] * self.focal_plane_px_per_mm[1]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def cv2_distortion(self) -> np.ndarray:
        # MicaSense stores k1, k2, k3, p1, p2; OpenCV wants k1, k2, p1, p2, k3.
        d = list(self.distortion) + [0.0] * (5 - len(self.distortion))
        return np.array([d[0], d[1], d[3], d[4], d[2]], dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_pixels", None)
        return data


def read_band_metadata(path: str | Path) -> BandImage:
    """Read one MicaSense band TIFF's EXIF/XMP without decoding its pixels."""
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        tags = {tag.code: tag.value for tag in page.tags.values()}
        width = int(page.shape[-1])
        height = int(page.shape[-2])
    xmp_raw = tags.get(700, b"")
    xmp = parse_xmp(xmp_raw.decode("utf-8", "replace") if isinstance(xmp_raw, bytes) else str(xmp_raw))
    exif = tags.get(34665) or {}
    gps = tags.get(34853) or {}
    if "BandName" not in xmp or "RadiometricCalibration" not in xmp:
        raise MultispectralError(
            f"{path.name} is not a MicaSense band image (no BandName/RadiometricCalibration XMP)."
        )
    band_name = str(xmp["BandName"])
    band_key = _BAND_ALIASES.get(band_name.strip().casefold())
    if band_key is None:
        raise MultispectralError(f"{path.name}: unknown band name {band_name!r}.")
    suffix = re.search(r"_(\d+)\.tiff?$", path.name, re.I)
    if suffix:
        band_index = int(suffix.group(1))
    else:
        band_index = int(float(xmp.get("RigCameraIndex", 0))) + 1

    black = tags.get(50714)
    if black is not None:
        black_values = [float(v) for v in (black if isinstance(black, (tuple, list)) else [black])]
        black_level = float(np.mean(black_values))
    elif xmp.get("DarkRowValue"):
        black_level = float(np.mean(_floats(xmp["DarkRowValue"])))
    else:
        black_level = 0.0
    bits = tags.get(258, 16)
    bits = int(bits[0] if isinstance(bits, (tuple, list)) else bits)

    exposure = _rational(exif.get("ExposureTime"))
    iso = exif.get("ISOSpeed") or exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")
    plane_x = _rational(exif.get("FocalPlaneXResolution"))
    plane_y = _rational(exif.get("FocalPlaneYResolution"))
    principal = _floats(xmp.get("PrincipalPoint"))
    focal = _floats(xmp.get("PerspectiveFocalLength"))

    if "IrradianceScaleToSIUnits" in xmp:
        irradiance_scale = float(xmp["IrradianceScaleToSIUnits"])
    elif "HorizontalIrradiance" in xmp:
        # DLS2 without an explicit scale reports in units of 0.01 W/m^2/nm
        # (MicaSense metadata.irradiance_scale_factor).
        irradiance_scale = 0.01
    else:
        irradiance_scale = 1.0

    def opt(tag: str) -> float | None:
        values = _floats(xmp.get(tag))
        return values[0] if values else None

    region = _floats(xmp.get("ReflectArea"))
    panel_region = (
        [(int(region[i]), int(region[i + 1])) for i in range(0, len(region) - 1, 2)]
        if len(region) >= 6 else None
    )
    altitude = _rational(gps.get("GPSAltitude"))
    if altitude is not None and gps.get("GPSAltitudeRef") in (1, b"\x01"):
        altitude = -altitude

    return BandImage(
        path=str(path),
        band_key=band_key,
        band_name=band_name,
        band_index=band_index,
        capture_id=str(xmp.get("CaptureId") or re.sub(r"_\d+\.tiff?$", "", path.name, flags=re.I)),
        camera_model=str(tags.get(272, "") or xmp.get("RigName", "")),
        camera_serial=str(exif.get("BodySerialNumber", "")),
        firmware=str(tags.get(305, "")),
        width=width,
        height=height,
        bits_per_pixel=bits,
        black_level=black_level,
        exposure_s=float(exposure) if exposure else float("nan"),
        gain=float(iso) / 100.0 if iso else float("nan"),
        radiometric_cal=_floats(xmp.get("RadiometricCalibration")),
        vignette_center=_floats(xmp.get("VignettingCenter")),
        vignette_polynomial=_floats(xmp.get("VignettingPolynomial")),
        vignette_poly2d=_floats(xmp.get("VignettingPolynomial2D")),
        vignette_poly2d_exponents=_floats(xmp.get("VignettingPolynomial2DName")),
        center_wavelength_nm=opt("CentralWavelength"),
        bandwidth_nm=opt("WavelengthFWHM"),
        focal_length_mm=focal[0] if focal else None,
        focal_plane_px_per_mm=(plane_x, plane_y) if plane_x and plane_y else None,
        principal_point_mm=(principal[0], principal[1]) if len(principal) >= 2 else None,
        distortion=_floats(xmp.get("PerspectiveDistortion")),
        rig_relatives_deg=_floats(xmp.get("RigRelatives")),
        rig_reference_index=(
            int(float(xmp["RigRelativesReferenceRigCameraIndex"]))
            if "RigRelativesReferenceRigCameraIndex" in xmp else None
        ),
        spectral_irradiance=opt("SpectralIrradiance") or opt("Irradiance"),
        horizontal_irradiance=opt("HorizontalIrradiance"),
        direct_irradiance=opt("DirectIrradiance"),
        scattered_irradiance=opt("ScatteredIrradiance"),
        solar_elevation_rad=opt("SolarElevation"),
        irradiance_scale=irradiance_scale,
        panel_albedo=opt("Albedo"),
        panel_region=panel_region,
        panel_serial=str(xmp["PanelSerial"]) if "PanelSerial" in xmp else None,
        latitude=_dms(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")),
        longitude=_dms(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")),
        altitude_m=altitude,
        datetime_original=str(exif.get("DateTimeOriginal")) if exif.get("DateTimeOriginal") else None,
        flight_id=str(xmp["FlightId"]) if "FlightId" in xmp else None,
    )


@dataclass
class Capture:
    """All bands triggered together, keyed by canonical band name."""

    capture_id: str
    bands: dict[str, BandImage]

    @property
    def name(self) -> str:
        first = next(iter(self.bands.values()))
        return re.sub(r"_\d+\.tiff?$", "", Path(first.path).name, flags=re.I)

    def band(self, key: str) -> BandImage:
        try:
            return self.bands[key]
        except KeyError as exc:
            raise MultispectralError(f"Capture {self.name} has no {key} band.") from exc

    @property
    def reflectance_keys(self) -> list[str]:
        """Narrow bands in rig order (panchromatic and thermal excluded)."""
        ordered = sorted(self.bands.values(), key=lambda b: b.band_index)
        return [b.band_key for b in ordered if b.band_key not in {"lwir", "panchro"}]


def group_captures(paths: Iterable[str | Path] | str | Path) -> list[Capture]:
    """Group band TIFFs into captures by the camera's CaptureId (file stem as fallback)."""
    if isinstance(paths, (str, Path)) and Path(paths).is_dir():
        paths = sorted(p for p in Path(paths).iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    groups: dict[str, dict[str, BandImage]] = {}
    order: list[str] = []
    for path in paths:
        try:
            band = read_band_metadata(path)
        except MultispectralError:
            continue
        if band.capture_id not in groups:
            groups[band.capture_id] = {}
            order.append(band.capture_id)
        groups[band.capture_id][band.band_key] = band
    captures = [Capture(cid, groups[cid]) for cid in order]
    captures.sort(key=lambda c: c.name)
    return captures


# ---------------------------------------------------------------------------
# Radiometry
# ---------------------------------------------------------------------------


def vignette_correction(band: BandImage) -> np.ndarray:
    """Per-pixel multiplier that undoes optical vignetting (MicaSense model)."""
    rows, cols = np.mgrid[0 : band.height, 0 : band.width].astype(np.float64)
    if len(band.vignette_center) == 2 and band.vignette_polynomial:
        cx, cy = band.vignette_center
        r = np.hypot(cols - cx, rows - cy)
        coefficients = list(reversed(band.vignette_polynomial)) + [1.0]
        return 1.0 / np.polyval(coefficients, r)
    if band.vignette_poly2d and len(band.vignette_poly2d_exponents) >= 2 * len(band.vignette_poly2d):
        xv = cols / band.width
        yv = rows / band.height
        poly = np.zeros_like(xv)
        for i, c in enumerate(band.vignette_poly2d):
            ex = band.vignette_poly2d_exponents[2 * i]
            ey = band.vignette_poly2d_exponents[2 * i + 1]
            poly += c * xv**ex * yv**ey
        return 1.0 / poly
    raise MultispectralError(f"{Path(band.path).name} has no vignette model in its XMP.")


def radiance(band: BandImage) -> np.ndarray:
    """Spectral radiance in W/m^2/sr/nm (MicaSense radiometric model)."""
    if band.is_thermal:
        raise MultispectralError("LWIR is thermal; use lwir_temperature_c().")
    if len(band.radiometric_cal) < 3:
        raise MultispectralError(f"{Path(band.path).name} lacks the 3 radiometric coefficients.")
    if not (math.isfinite(band.exposure_s) and band.exposure_s > 0 and math.isfinite(band.gain) and band.gain > 0):
        raise MultispectralError(f"{Path(band.path).name} lacks exposure time or ISO gain in EXIF.")
    a1, a2, a3 = band.radiometric_cal[:3]
    raw = band.pixels().astype(np.float64)
    rows = np.arange(band.height, dtype=np.float64)[:, None]
    row_gradient = 1.0 / (1.0 + a2 * rows / band.exposure_s - a3 * rows)
    signal = vignette_correction(band) * row_gradient * (raw - band.black_level)
    signal[signal < 0] = 0
    max_dn = float(2**band.bits_per_pixel)
    return (signal / (band.gain * band.exposure_s) * a1 / max_dn).astype(np.float32)


def lwir_temperature_c(band: BandImage) -> np.ndarray:
    """Altum LWIR: stored as centi-kelvin, decoded linearly to degrees Celsius."""
    if not band.is_thermal:
        raise MultispectralError(f"{Path(band.path).name} is not an LWIR band.")
    return (band.pixels().astype(np.float32) * 0.01 - 273.15).astype(np.float32)


def dls_irradiance(band: BandImage) -> tuple[float, str]:
    """Horizontal spectral irradiance (W/m^2/nm) at capture time, and how it was obtained."""
    model = band.camera_model.strip()
    horizontal_trusted = band.horizontal_irradiance is not None
    if horizontal_trusted and model == "Altum":
        # Firmware before 1.2.3 wrote a wrong horizontal value (MicaSense metadata.py).
        horizontal_trusted = _version_tuple(band.firmware.lstrip("v")) >= (1, 2, 3)
    if horizontal_trusted and band.horizontal_irradiance and band.horizontal_irradiance > 0:
        return band.horizontal_irradiance * band.irradiance_scale, "DLS2 HorizontalIrradiance (XMP)"
    if (
        band.direct_irradiance and band.scattered_irradiance is not None
        and band.solar_elevation_rad is not None
    ):
        value = (
            band.direct_irradiance * math.sin(band.solar_elevation_rad) + band.scattered_irradiance
        ) * band.irradiance_scale
        if value > 0:
            return value, "DLS2 direct*sin(solar elevation)+scattered (XMP)"
    raise MultispectralError(
        f"{Path(band.path).name}: no usable DLS2 horizontal irradiance in the XMP. "
        + DLS_DOWNLOAD_NOTE
    )


@dataclass(frozen=True)
class BandCalibration:
    """Multiplier from radiance to surface reflectance for one band, with its provenance."""

    band_key: str
    factor: float
    method: str  # "dls" or "panel"
    source: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _shrunk_polygon_mask(shape: tuple[int, int], polygon: Sequence[tuple[int, int]], keep: float) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float64)
    centre = pts.mean(axis=0)
    inner = centre + (pts - centre) * keep
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(inner).astype(np.int32)], 1)
    return mask.astype(bool)


def panel_statistics(band: BandImage, region: Sequence[tuple[int, int]] | None = None) -> dict[str, Any]:
    """Mean radiance over the panel's active area (inner 80% to avoid its edges)."""
    region = region or band.panel_region
    if not region:
        raise MultispectralError(
            f"{Path(band.path).name}: the camera did not record the panel area (Camera:ReflectArea)."
        )
    rad = radiance(band)
    mask = _shrunk_polygon_mask(rad.shape, region, 0.8)
    raw = band.pixels()
    saturated = float((raw[mask] >= (2**band.bits_per_pixel - 1) * 0.99).mean())
    values = rad[mask]
    return {
        "pixels": int(mask.sum()),
        "mean_radiance": float(values.mean()),
        "std_radiance": float(values.std()),
        "coefficient_of_variation": float(values.std() / max(values.mean(), 1e-12)),
        "saturated_fraction": saturated,
        "region": [list(p) for p in region],
    }


def panel_calibration(
    capture: Capture,
    albedo_overrides: Mapping[str, tuple[float, str]] | None = None,
) -> dict[str, BandCalibration]:
    """Radiance->reflectance factors from a calibration-panel capture.

    The albedo comes from the camera's reading of the panel QR code (Camera:Albedo).
    For a band where the camera recorded none, a value is used only when supplied in
    ``albedo_overrides`` as ``{band: (albedo, "where it came from")}``; otherwise that
    band is left uncalibrated.
    """
    overrides = dict(albedo_overrides or {})
    out: dict[str, BandCalibration] = {}
    for key in capture.reflectance_keys + (["panchro"] if "panchro" in capture.bands else []):
        band = capture.bands[key]
        if key in overrides:
            albedo, albedo_source = overrides[key]
            if not albedo_source.strip():
                raise MultispectralError("An albedo override must name its source.")
        elif band.panel_albedo:
            albedo, albedo_source = band.panel_albedo, f"panel QR albedo read by the camera ({band.panel_serial})"
        else:
            continue
        try:
            stats = panel_statistics(band)
        except MultispectralError:
            continue
        if stats["saturated_fraction"] > 0.01:
            continue
        out[key] = BandCalibration(
            band_key=key,
            factor=float(albedo) / stats["mean_radiance"],
            method="panel",
            source=f"{Path(band.path).name}: {albedo_source}",
            details={"albedo": float(albedo), **stats, "panel_serial": band.panel_serial},
        )
        band.release()
    return out


def dls_calibration(capture: Capture) -> dict[str, BandCalibration]:
    """Per-capture factors from the DLS2 irradiance: reflectance = pi * L / E."""
    out: dict[str, BandCalibration] = {}
    for key in capture.reflectance_keys + (["panchro"] if "panchro" in capture.bands else []):
        band = capture.bands[key]
        irradiance, how = dls_irradiance(band)
        out[key] = BandCalibration(
            band_key=key,
            factor=math.pi / irradiance,
            method="dls",
            source=f"{Path(band.path).name}: {how}",
            details={"irradiance_w_m2_nm": irradiance},
        )
    return out


def validate_panel_against_albedo(capture: Capture) -> dict[str, Any]:
    """Independent check: the DLS-derived reflectance of the panel versus its QR albedo.

    The panel calibration reproduces the albedo by construction, so it cannot validate
    itself. The DLS path does not use the panel, so its panel reading is a real
    measurement of how close sun-sensor calibration gets to a known target.
    """
    rows: dict[str, Any] = {}
    for key in capture.reflectance_keys + (["panchro"] if "panchro" in capture.bands else []):
        band = capture.bands[key]
        row: dict[str, Any] = {"band": key, "file": Path(band.path).name, "albedo": band.panel_albedo}
        if not band.panel_albedo or not band.panel_region:
            row["status"] = "unavailable"
            row["reason"] = "the camera recorded no panel albedo/area for this band"
            rows[key] = row
            continue
        try:
            stats = panel_statistics(band)
            irradiance, how = dls_irradiance(band)
        except MultispectralError as exc:
            row["status"] = "unavailable"
            row["reason"] = str(exc)
            rows[key] = row
            continue
        measured = math.pi * stats["mean_radiance"] / irradiance
        row.update(
            status="measured",
            dls_reflectance=measured,
            absolute_error=measured - band.panel_albedo,
            relative_error=(measured - band.panel_albedo) / band.panel_albedo,
            irradiance_w_m2_nm=irradiance,
            irradiance_source=how,
            panel_cv=stats["coefficient_of_variation"],
            saturated_fraction=stats["saturated_fraction"],
        )
        rows[key] = row
        band.release()
    measured_rows = [r for r in rows.values() if r.get("status") == "measured"]
    return {
        "capture": capture.name,
        "camera": next(iter(capture.bands.values())).camera_model,
        "panel_serial": next((b.panel_serial for b in capture.bands.values() if b.panel_serial), None),
        "bands": rows,
        "mean_absolute_relative_error": (
            float(np.mean([abs(r["relative_error"]) for r in measured_rows])) if measured_rows else None
        ),
        "method": "pi * mean panel radiance / DLS2 horizontal irradiance, compared with Camera:Albedo",
    }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _gradient(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0.0).astype(np.float32)
    scale = float(np.percentile(image, 99.5)) or 1.0
    image = cv2.GaussianBlur(image / scale, (0, 0), 1.0)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _rig_rotation(roll_pitch_yaw_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(v) for v in roll_pitch_yaw_deg[:3])
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _ecc_homography(
    moving: np.ndarray, fixed: np.ndarray, init: np.ndarray, levels: int = 4, finest: int = 1
) -> tuple[np.ndarray, float]:
    """Pyramid ECC homography mapping fixed-frame pixels into the moving frame.

    Refinement stops at pyramid level ``finest`` (1 = half resolution): the full-res
    pass cost up to 30 s per band for sub-pixel changes.
    """
    warp = init.astype(np.float32).copy()
    scale_down = np.diag([0.5, 0.5, 1.0]).astype(np.float32)
    scale_up = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
    for _ in range(levels - 1):
        warp = scale_down @ warp @ scale_up
    pyramid_fixed = [fixed]
    pyramid_moving = [moving]
    for _ in range(levels - 1):
        pyramid_fixed.append(cv2.pyrDown(pyramid_fixed[-1]))
        pyramid_moving.append(cv2.pyrDown(pyramid_moving[-1]))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    rho = float("nan")
    for level in reversed(range(levels)):
        if level >= finest:
            rho, warp = cv2.findTransformECC(
                pyramid_fixed[level], pyramid_moving[level], warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
            )
        if level:
            warp = scale_up @ warp @ scale_down
    return warp, float(rho)


def _max_corner_deviation(warp: np.ndarray, reference: np.ndarray, size: tuple[int, int]) -> float:
    corners = np.array([[0, 0, 1], [size[0], 0, 1], [0, size[1], 1], [size[0], size[1], 1]], dtype=np.float64).T
    a = warp.astype(np.float64) @ corners
    b = reference.astype(np.float64) @ corners
    return float(np.max(np.hypot(a[0] / a[2] - b[0] / b[2], a[1] / a[2] - b[1] / b[2])))


def _warp_agreement(moving: np.ndarray, fixed: np.ndarray, warp: np.ndarray, size: tuple[int, int]) -> float:
    """Gradient NCC of the warped band against the reference over their overlap."""
    warped = cv2.warpPerspective(
        moving, warp.astype(np.float64), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
    )
    valid = np.isfinite(warped) & np.isfinite(fixed)
    if valid.sum() < 1000:
        return float("nan")
    a = fixed[valid] - fixed[valid].mean()
    b = warped[valid] - warped[valid].mean()
    return float((a * b).sum() / max(math.sqrt(float((a * a).sum() * (b * b).sum())), 1e-12))


@dataclass
class AlignedCapture:
    """Reflectance bands of one capture, undistorted and co-registered to one frame."""

    name: str
    band_keys: list[str]
    reflectance: np.ndarray  # (bands, H, W) float32, NaN outside a band's coverage
    camera_matrix: np.ndarray  # intrinsics of the common (reference) frame
    reference_band: str
    alignment: dict[str, dict]
    calibrations: dict[str, BandCalibration]
    gps: tuple[float | None, float | None, float | None]
    datetime_original: str | None

    def valid_mask(self) -> np.ndarray:
        return np.all(np.isfinite(self.reflectance), axis=0)


def align_capture(
    capture: Capture,
    calibrations: Mapping[str, BandCalibration],
    *,
    reference: str | None = None,
    min_correlation: float = 0.5,
) -> AlignedCapture:
    """Convert each narrow band to reflectance, undistort it and register it to the reference.

    The initial homography comes from the rig's factory relative rotations
    (Camera:RigRelatives); ECC on gradient images then refines it. A band whose ECC
    correlation stays below ``min_correlation``, or whose ECC warp agrees with the
    reference band's gradients less well than the rig prediction does, keeps the rig
    estimate. A band is flagged when even the kept warp correlates below the threshold.
    """
    keys = [k for k in capture.reflectance_keys if k in calibrations]
    if len(keys) < 2:
        raise MultispectralError(f"Capture {capture.name}: fewer than two calibrated bands.")
    if reference is None:
        ref_index = next((b.rig_reference_index for b in capture.bands.values() if b.rig_reference_index is not None), None)
        reference = next(
            (b.band_key for b in capture.bands.values() if ref_index is not None and b.band_index - 1 == ref_index and b.band_key in keys),
            keys[0],
        )
    ref_band = capture.band(reference)
    k_ref = ref_band.camera_matrix()
    size = (ref_band.width, ref_band.height)

    undistorted: dict[str, np.ndarray] = {}
    for key in keys:
        band = capture.bands[key]
        refl = radiance(band) * np.float32(calibrations[key].factor)
        band.release()
        k = band.camera_matrix()
        # Undistort into the band's own pinhole frame; the homography handles the rest.
        map_x, map_y = cv2.initUndistortRectifyMap(k, band.cv2_distortion(), None, k, (band.width, band.height), cv2.CV_32FC1)
        undistorted[key] = cv2.remap(refl, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    fixed = _gradient(undistorted[reference])
    stack = np.full((len(keys), size[1], size[0]), np.nan, dtype=np.float32)
    report: dict[str, dict] = {}
    for i, key in enumerate(keys):
        band = capture.bands[key]
        if key == reference:
            stack[i] = undistorted[key]
            report[key] = {"reference": True, "ecc_correlation": 1.0, "method": "reference"}
            continue
        k_band = band.camera_matrix()
        if band.rig_relatives_deg and ref_band.rig_relatives_deg:
            r_band = _rig_rotation(band.rig_relatives_deg)
            r_ref = _rig_rotation(ref_band.rig_relatives_deg)
            init = k_band @ r_band.T @ r_ref @ np.linalg.inv(k_ref)
            init = init / init[2, 2]
        else:
            init = np.eye(3)
        moving = _gradient(undistorted[key])
        try:
            warp, rho = _ecc_homography(moving, fixed, init)
        except cv2.error:
            warp, rho = None, float("nan")
        rig_score = _warp_agreement(moving, fixed, init, size)
        ecc_score = _warp_agreement(moving, fixed, warp, size) if warp is not None else float("nan")
        deviation = _max_corner_deviation(warp, init, size) if warp is not None else float("nan")
        if warp is not None and rho >= min_correlation and ecc_score >= rig_score:
            method, flagged = "rig+ecc", False
        else:
            # The factory rig calibration predicts the inter-band warp to about a pixel at
            # survey altitude; an ECC result that did not converge, or that agrees with the
            # reference band less well than the rig model does, is not used.
            reason = (
                "ECC did not converge" if warp is None
                else f"ECC correlation {rho:.2f} < {min_correlation}" if not rho >= min_correlation
                else f"ECC agreement {ecc_score:.3f} < rig {rig_score:.3f}"
            )
            warp, method = init.astype(np.float32), f"rig ({reason})"
            flagged = not rig_score >= min_correlation
        stack[i] = cv2.warpPerspective(
            undistorted[key], warp.astype(np.float64), size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
        )
        shift = warp @ np.array([size[0] / 2, size[1] / 2, 1.0])
        report[key] = {
            "reference": False,
            "ecc_correlation": rho,
            "method": method,
            "flagged": flagged,
            "gradient_agreement_rig": rig_score,
            "gradient_agreement_ecc": ecc_score,
            "ecc_vs_rig_corner_px": deviation,
            "centre_offset_px": [float(shift[0] / shift[2] - size[0] / 2), float(shift[1] / shift[2] - size[1] / 2)],
        }
    first = capture.bands[reference]
    return AlignedCapture(
        name=capture.name,
        band_keys=keys,
        reflectance=stack,
        camera_matrix=k_ref,
        reference_band=reference,
        alignment=report,
        calibrations={k: calibrations[k] for k in keys},
        gps=(first.latitude, first.longitude, first.altitude_m),
        datetime_original=first.datetime_original,
    )


def band_to_band_agreement(aligned: AlignedCapture) -> dict[str, float]:
    """Normalised cross-correlation of each band's gradients with the reference band.

    A measured registration-quality figure: higher is better, 1.0 is the reference.
    """
    ref = _gradient(aligned.reflectance[aligned.band_keys.index(aligned.reference_band)])
    valid = aligned.valid_mask()
    out: dict[str, float] = {}
    for i, key in enumerate(aligned.band_keys):
        grad = _gradient(aligned.reflectance[i])
        a = ref[valid] - ref[valid].mean()
        b = grad[valid] - grad[valid].mean()
        out[key] = float((a * b).sum() / max(math.sqrt(float((a * a).sum() * (b * b).sum())), 1e-12))
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_reflectance_stack(aligned: AlignedCapture, path: str | Path) -> str:
    """Float32 multi-band TIFF (bands in ``band_keys`` order, NaN = no data)."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        aligned.reflectance.astype(np.float32),
        photometric="minisblack",
        planarconfig="separate",
        metadata={"bands": aligned.band_keys, "quantity": "surface_reflectance"},
    )
    return str(path)


def read_reflectance_stack(path: str | Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(str(path)), dtype=np.float32)


def _to_rational(value: float, denominator: int = 1_000_000):
    from PIL.TiffImagePlugin import IFDRational

    return IFDRational(int(round(value * denominator)), denominator)


def write_sfm_preview(aligned: AlignedCapture, path: str | Path, *, white_reflectance: float = 0.3) -> str:
    """8-bit JPEG for structure-from-motion, geotagged from the capture's EXIF GPS.

    Pixels are a fixed stretch of red/green/blue reflectance (same stretch for every
    capture). It exists so the reconstruction engine can solve camera poses; the
    measured reflectance lives in the float stack, not here.
    """
    from PIL import Image

    channels = []
    for key in ("red", "green", "blue"):
        if key in aligned.band_keys:
            channels.append(aligned.reflectance[aligned.band_keys.index(key)])
    if len(channels) < 3:
        channels = [aligned.reflectance[0]] * 3
    rgb = np.stack(channels, axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb / white_reflectance, 0.0, 1.0) ** (1.0 / 2.2)
    image = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))

    exif = Image.Exif()
    exif[0x010F] = "MicaSense"
    exif[0x0110] = "RedEdge reflectance composite"
    focal_px = float(aligned.camera_matrix[0, 0])
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0xA405] = int(round(focal_px * 36.0 / aligned.reflectance.shape[2]))  # 35 mm equivalent
    if aligned.datetime_original:
        exif_ifd[0x9003] = aligned.datetime_original
    lat, lon, alt = aligned.gps
    if lat is not None and lon is not None:
        gps = exif.get_ifd(0x8825)

        def dms(value: float) -> tuple:
            value = abs(value)
            d = int(value)
            m = int((value - d) * 60)
            s = (value - d - m / 60.0) * 3600.0
            return (_to_rational(d, 1), _to_rational(m, 1), _to_rational(s, 10000))

        gps[1] = "N" if lat >= 0 else "S"
        gps[2] = dms(lat)
        gps[3] = "E" if lon >= 0 else "W"
        gps[4] = dms(lon)
        if alt is not None:
            gps[5] = b"\x00" if alt >= 0 else b"\x01"
            gps[6] = _to_rational(abs(alt), 1000)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), quality=95, exif=exif)
    return str(path)


# ---------------------------------------------------------------------------
# End-to-end survey processing
# ---------------------------------------------------------------------------


def process_survey(
    images: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    calibration: str = "dls",
    panel_capture: Capture | None = None,
    albedo_overrides: Mapping[str, tuple[float, str]] | None = None,
    reconstruct: bool = True,
    reconstructor: Any = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Band TIFFs -> calibrated, aligned reflectance per capture -> multiband orthomosaic.

    ``calibration='dls'`` uses each capture's own sun-sensor irradiance;
    ``calibration='panel'`` uses ``panel_capture`` (which must be from the same flight
    and camera, checked by serial). The orthomosaic is produced by the COLMAP engine:
    camera poses are solved on geotagged reflectance composites, and every band is
    then orthorectified with those poses onto the same grid.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    captures = group_captures(images)
    if not captures:
        raise MultispectralError("No MicaSense band TIFFs were found.")
    panel_factors: dict[str, BandCalibration] = {}
    if calibration == "panel":
        if panel_capture is None:
            raise MultispectralError("Panel calibration needs a panel capture.")
        survey_serial = next(iter(captures[0].bands.values())).camera_serial
        panel_serial = next(iter(panel_capture.bands.values())).camera_serial
        if survey_serial and panel_serial and survey_serial != panel_serial:
            raise MultispectralError(
                f"The panel was captured by camera {panel_serial}, the survey by {survey_serial}; "
                "a panel calibrates only the camera and flight it was taken with."
            )
        panel_factors = panel_calibration(panel_capture, albedo_overrides)
    elif calibration != "dls":
        raise ValueError("calibration must be 'dls' or 'panel'.")

    stacks_dir = out / "reflectance_stacks"
    sfm_dir = out / "sfm_images"
    per_capture: list[dict] = []
    band_sources: dict[str, str] = {}
    band_keys: list[str] | None = None
    for index, capture in enumerate(captures):
        if progress:
            progress(int(40 * index / len(captures)), f"Calibrating {capture.name}")
        factors = dls_calibration(capture) if calibration == "dls" else panel_factors
        aligned = align_capture(capture, factors)
        if band_keys is None:
            band_keys = aligned.band_keys
        elif aligned.band_keys != band_keys:
            raise MultispectralError(f"Capture {capture.name} has a different band set: {aligned.band_keys}")
        stack_path = write_reflectance_stack(aligned, stacks_dir / f"{capture.name}.tif")
        jpg_path = write_sfm_preview(aligned, sfm_dir / f"{capture.name}.jpg")
        band_sources[Path(jpg_path).name] = stack_path
        valid = aligned.valid_mask()
        per_capture.append(
            {
                "capture": capture.name,
                "stack": stack_path,
                "sfm_image": jpg_path,
                "alignment": aligned.alignment,
                "gradient_agreement": band_to_band_agreement(aligned),
                "valid_fraction": float(valid.mean()),
                "reflectance_median": {
                    key: float(np.nanmedian(aligned.reflectance[i][valid]))
                    for i, key in enumerate(aligned.band_keys)
                },
                "calibration": {k: v.to_dict() for k, v in aligned.calibrations.items()},
            }
        )
    summary: dict[str, Any] = {
        "status": "calibrated",
        "calibration": calibration,
        "quantity": "surface_reflectance",
        "bands": band_keys,
        "captures": per_capture,
        "orthomosaic": None,
    }
    if reconstruct:
        summary["orthomosaic"] = reconstruct_band_orthomosaic(
            out, band_keys or [], band_sources, reconstructor=reconstructor, progress=progress
        )
    (out / "multispectral_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(out / "multispectral_summary.json")
    return summary


def reconstruct_band_orthomosaic(
    output_dir: str | Path,
    band_keys: Sequence[str],
    band_sources: Mapping[str, str],
    *,
    reconstructor: Any = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Solve poses on the geotagged composites in ``output_dir/sfm_images`` and
    orthorectify every band stack in ``band_sources`` onto one grid."""
    out = Path(output_dir)
    if reconstructor is None:
        from .reconstruction_colmap import ColmapReconstructor

        reconstructor = ColmapReconstructor(profile="standard", dense=False)
    reconstructor.band_sources = dict(band_sources)
    reconstructor.band_names = list(band_keys)
    result = reconstructor.reconstruct(
        out / "sfm_images", output_dir=out / "reconstruction",
        progress_callback=(lambda p, m: progress(40 + int(0.6 * p), m)) if progress else None,
    )
    bands_path = getattr(reconstructor, "band_orthomosaic_path", None)
    report: dict[str, Any] = {
        "reflectance_bands": bands_path,
        "rgb_preview": getattr(result, "orthomosaic_cog_path", "") or getattr(result, "orthomosaic_path", ""),
        "registered_images": getattr(result, "registered_images", None),
        "input_images": len(band_sources),
        "reprojection_error_px": getattr(result, "reprojection_error_px", None),
        "geo_rmse_m": getattr(result, "geo_rmse_m", None),
        "crs_epsg": getattr(result, "crs_epsg", None),
        "orthomosaic_gsd_m": getattr(result, "orthomosaic_gsd_m", None),
        "warnings": list(getattr(reconstructor, "warnings", [])),
    }
    if not bands_path:
        report["status"] = "unavailable"
        report["reason"] = "The reconstruction did not produce a georeferenced band orthomosaic; see warnings."
    else:
        report["status"] = "complete"
    return report


def reflectance_band_calibrations(bands: Sequence[str], source: str) -> dict:
    """ReflectanceBandCalibration entries for a band orthomosaic that already holds reflectance."""
    from .india_agriculture import ReflectanceBandCalibration

    return {
        key: ReflectanceBandCalibration(band=i + 1, scale=1.0, offset=0.0, source=source)
        for i, key in enumerate(bands)
    }
