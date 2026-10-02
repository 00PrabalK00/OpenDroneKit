"""DJI R-JPEG thermal decoding through the DJI Thermal SDK (TSDK), loaded at run time.

DJI's infrared cameras (Zenmuse H20T/H20N/H30T/XT S, M2EA, M30T, M3T, M3TD, M4T) store
raw detector data and a calibration curve in APP3 segments of the JPEG. Unlike FLIR,
the conversion to temperature is not published; only DJI's own SDK (``libdirp``)
performs it. The SDK is proprietary and is **not** shipped with OpenDroneKit: the user
downloads it under DJI's licence and points the app at it, with the
``ODK_DJI_TSDK_DIR`` environment variable or the ``dji_thermal_sdk_dir`` setting.

When it is absent the decode is refused with instructions, never approximated: the
rendered colours are a palette, and the counts alone have no published conversion.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .thermal import NotRadiometric, ThermalImage

DJI_TSDK_URL = "https://www.dji.com/downloads/softwares/dji-thermal-sdk"
ENV_VAR = "ODK_DJI_TSDK_DIR"
SETTING = "dji_thermal_sdk_dir"

_ERRORS = {
    -1: "malloc error", -2: "NULL pointer input", -3: "invalid parameters",
    -4: "invalid RAW in R-JPEG", -5: "invalid header in R-JPEG",
    -6: "invalid curve LUT in R-JPEG", -7: "R-JPEG parse error", -8: "wrong size input",
    -9: "invalid handle", -10: "wrong input image format", -11: "wrong output image format",
    -12: "unsupported function", -13: "preliminary conditions not met",
    -14: "SDK activation failed", -15: "invalid ini file", -16: "invalid sub DLL",
    -64: "unsupported super-resolution mode picture",
}


class DjiSdkUnavailable(NotRadiometric):
    """The DJI Thermal SDK is needed for this file and is not configured or not loadable."""


class _Resolution(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class _MeasurementParams(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("distance", ctypes.c_float), ("humidity", ctypes.c_float),
                ("emissivity", ctypes.c_float), ("reflection", ctypes.c_float),
                ("ambient_temp", ctypes.c_float)]


class _ApiVersion(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("api", ctypes.c_uint32), ("magic", ctypes.c_char * 8)]


def _library_name() -> str:
    return "libdirp.dll" if sys.platform == "win32" else "libdirp.so"


def _arch_dirs() -> list[str]:
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    system = "windows" if sys.platform == "win32" else "linux"
    arch = "release_x64" if bits == 64 else "release_x86"
    if system == "linux" and platform.machine().lower() in ("aarch64", "arm64"):
        arch = "release_arm64"
    return [f"tsdk-core/lib/{system}/{arch}", f"utility/bin/{system}/{arch}"]


def configured_sdk_dir() -> tuple[str, str]:
    """(path, where it came from): the env var wins over the saved setting."""
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env, f"environment variable {ENV_VAR}"
    try:
        from .settings import load_app_settings

        saved = str(getattr(load_app_settings(), SETTING, "") or "").strip()
    except Exception:  # settings unreadable: treat as not configured
        saved = ""
    if saved:
        return saved, f"setting {SETTING}"
    return "", ""


def find_library(sdk_dir: str | Path) -> Path | None:
    """The libdirp binary inside an SDK folder (its root, or the library folder itself)."""
    root = Path(sdk_dir)
    candidates = [root] + [root / sub for sub in _arch_dirs()]
    for folder in candidates:
        library = folder / _library_name()
        if library.is_file():
            return library
    return None


def _absent_message(detail: str) -> str:
    return (
        f"{detail} Decoding a DJI R-JPEG to temperature needs the DJI Thermal SDK, which "
        f"OpenDroneKit cannot ship. Download it from {DJI_TSDK_URL} (accepting DJI's "
        f"licence), extract it, and set its folder in Settings (DJI Thermal SDK folder) "
        f"or in the {ENV_VAR} environment variable. Without it no temperature is "
        "reported: the rendered pixels are a palette.")


_lock = threading.Lock()
_loaded: dict[str, ctypes.CDLL] = {}


def load_sdk() -> tuple[ctypes.CDLL, Path]:
    """Load libdirp from the configured folder, or raise DjiSdkUnavailable saying why."""
    folder, origin = configured_sdk_dir()
    if not folder:
        raise DjiSdkUnavailable(_absent_message("No DJI Thermal SDK folder is configured."))
    library = find_library(folder)
    if library is None:
        raise DjiSdkUnavailable(_absent_message(
            f"The DJI Thermal SDK folder from {origin} ({folder}) contains no "
            f"{_library_name()} for this platform (looked in the folder and "
            f"{', '.join(_arch_dirs())})."))
    key = str(library.resolve())
    with _lock:
        if key in _loaded:
            return _loaded[key], library
        try:
            if sys.platform == "win32":
                os.add_dll_directory(str(library.parent))
            sdk = ctypes.CDLL(str(library))
        except OSError as exc:
            raise DjiSdkUnavailable(_absent_message(
                f"{library} could not be loaded ({exc}).")) from exc
        handle_p = ctypes.POINTER(ctypes.c_void_p)
        sdk.dirp_create_from_rjpeg.argtypes = [ctypes.c_void_p, ctypes.c_int32, handle_p]
        sdk.dirp_create_from_rjpeg.restype = ctypes.c_int32
        sdk.dirp_destroy.argtypes = [ctypes.c_void_p]
        sdk.dirp_destroy.restype = ctypes.c_int32
        sdk.dirp_get_rjpeg_resolution.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Resolution)]
        sdk.dirp_get_rjpeg_resolution.restype = ctypes.c_int32
        sdk.dirp_measure_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int32]
        sdk.dirp_measure_ex.restype = ctypes.c_int32
        sdk.dirp_get_measurement_params.argtypes = [ctypes.c_void_p,
                                                    ctypes.POINTER(_MeasurementParams)]
        sdk.dirp_get_measurement_params.restype = ctypes.c_int32
        sdk.dirp_get_api_version.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ApiVersion)]
        sdk.dirp_get_api_version.restype = ctypes.c_int32
        _loaded[key] = sdk
        return sdk, library


def sdk_status() -> dict[str, Any]:
    """What the Settings screen shows: configured folder, whether it loads, and why not."""
    folder, origin = configured_sdk_dir()
    status: dict[str, Any] = {"configured_dir": folder, "configured_by": origin,
                              "download_url": DJI_TSDK_URL, "env_var": ENV_VAR}
    try:
        _sdk, library = load_sdk()
    except DjiSdkUnavailable as exc:
        status.update(available=False, reason=str(exc))
    else:
        status.update(available=True, library=str(library))
    return status


def _app3_segments(data: bytes) -> int:
    count, position = 0, 2
    while position + 4 <= len(data) and data[position] == 0xFF:
        marker = data[position + 1]
        if marker == 0xDA or marker == 0xD9:
            break
        length = struct.unpack(">H", data[position + 2:position + 4])[0]
        if marker == 0xE3:
            count += 1
        position += 2 + length
    return count


def is_dji_rjpeg(path: str | Path) -> bool:
    """A JPEG from a DJI camera carrying APP3 radiometric segments."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(1 << 20)
    except OSError:
        return False
    return data[:2] == b"\xff\xd8" and _app3_segments(data) > 0 and b"DJI" in data[:65536]


def _check(code: int, what: str, name: str) -> None:
    if code != 0:
        raise NotRadiometric(
            f"{name}: the DJI Thermal SDK refused the file at {what} "
            f"(code {code}: {_ERRORS.get(code, 'unknown error')}).")


def decode_rjpeg(path: str | Path) -> ThermalImage:
    """Temperatures in Celsius as the DJI Thermal SDK computes them (``dirp_measure_ex``)."""
    target = Path(path)
    sdk, library = load_sdk()
    data = target.read_bytes()
    buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    handle = ctypes.c_void_p()
    _check(sdk.dirp_create_from_rjpeg(buffer, len(data), ctypes.byref(handle)),
           "open", target.name)
    try:
        resolution = _Resolution()
        _check(sdk.dirp_get_rjpeg_resolution(handle, ctypes.byref(resolution)),
               "resolution", target.name)
        params = _MeasurementParams()
        params_ok = sdk.dirp_get_measurement_params(handle, ctypes.byref(params)) == 0
        version = _ApiVersion()
        version_ok = sdk.dirp_get_api_version(handle, ctypes.byref(version)) == 0
        celsius = np.empty((resolution.height, resolution.width), dtype=np.float32)
        _check(sdk.dirp_measure_ex(handle, celsius.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   int(celsius.nbytes)), "measure", target.name)
    finally:
        sdk.dirp_destroy(handle)

    provenance: dict[str, Any] = {
        "decoder": "dji_thermal_sdk_dirp_measure_ex",
        "sdk_library": str(library),
        "temperature_kind": "apparent_temperature_dji_sdk",
        "unit": "celsius",
        "validation": (
            "Temperatures are as the DJI Thermal SDK computes them with the measurement "
            "parameters stored in the file. They are apparent temperatures and have not "
            "been compared with an independent reference measurement."),
    }
    if version_ok:
        provenance["sdk_api_version"] = f"0x{version.api:08x}"
    if params_ok:
        provenance["measurement_params"] = {
            "distance_m": round(float(params.distance), 4),
            "relative_humidity_pct": round(float(params.humidity), 4),
            "emissivity": round(float(params.emissivity), 4),
            "reflected_temp_c": round(float(params.reflection), 4),
            "ambient_temp_c": round(float(params.ambient_temp), 4),
        }
    return ThermalImage(celsius=celsius.astype(np.float64), calibration=None,
                        source=str(target), provenance=provenance)
