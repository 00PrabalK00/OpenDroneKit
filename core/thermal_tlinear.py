"""FLIR TLinear radiometric TIFFs (FLIR Duo Pro R and Tau-core cameras).

FLIR's Duo Pro R User Guide (Rev. 1.0, December 2017, section 6.2.2 "TIFF") documents
the format: the 14-bit TIFF holds "temperature linear" counts, and

    [counts@High Resolution] * 0.04 - 273.15 = deg C   (Temp Range High, -25 to +135 C)
    [counts@Low Resolution]  * 0.40 - 273.15 = deg C   (Temp Range Low,  -40 to +550 C)

The camera records which gain applies in the XMP tag ``Camera:TlinearGain`` (Pix4D's
camera namespace). Only the two documented gains are accepted: any other value, or a
missing tag, is refused rather than guessed.

The guide also states that radiometric parameters (emissivity, background
temperature) are fixed at capture and cannot be adjusted afterwards; the TIFF does not
record which values were used, so the result is labelled as the camera's apparent
temperature.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .thermal import ABSOLUTE_ZERO_C, NotRadiometric, ThermalImage

DOCUMENTED_GAINS = (0.04, 0.40)
SOURCE_DOC = ("FLIR Duo Pro R User Guide, Rev. 1.0 (Dec 2017), section 6.2.2 TIFF: "
              "counts x 0.04 (High resolution) or x 0.40 (Low resolution) - 273.15 = deg C")
_GAIN = re.compile(rb"TlinearGain>\s*([0-9.]+)\s*<|TlinearGain=\"([0-9.]+)\"")


def _xmp(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        tags = getattr(image, "tag_v2", None)
        value = tags.get(700) if tags is not None else None
        make = str(tags.get(271, "")) if tags is not None else ""
    if isinstance(value, str):
        value = value.encode("utf-8", "replace")
    return (value or b"") + b"\nMAKE=" + make.encode("latin-1", "replace")


def tlinear_gain(path: str | Path) -> float | None:
    match = _GAIN.search(_xmp(Path(path)))
    if not match:
        return None
    return float((match.group(1) or match.group(2)).decode())


def is_tlinear_tiff(path: str | Path) -> bool:
    try:
        return tlinear_gain(path) is not None
    except Exception:
        return False


def read_tlinear_tiff(path: str | Path) -> ThermalImage:
    from PIL import Image

    target = Path(path)
    gain = tlinear_gain(target)
    if gain is None:
        raise NotRadiometric(f"{target.name} has no Camera:TlinearGain; its counts have no "
                             "documented conversion.")
    documented = next((g for g in DOCUMENTED_GAINS if abs(gain - g) < 1e-6), None)
    if documented is None:
        raise NotRadiometric(
            f"{target.name} declares TlinearGain={gain}, which FLIR's documentation does "
            f"not define (only {DOCUMENTED_GAINS}). Refusing rather than guessing.")
    with Image.open(target) as image:
        if image.mode not in ("I;16", "I;16B", "I;16L", "I"):
            raise NotRadiometric(f"{target.name} is not a single-band 16-bit TIFF "
                                 f"(mode {image.mode}).")
        counts = np.array(image, dtype=np.float64)
    celsius = counts * documented + ABSOLUTE_ZERO_C
    celsius[counts <= 0] = np.nan
    provenance: dict[str, Any] = {
        "decoder": "flir_tlinear_tiff",
        "tlinear_gain_k_per_count": documented,
        "temperature_resolution_k": documented,
        "temperature_kind": "apparent_temperature_camera_fixed_parameters",
        "unit": "celsius",
        "source_document": SOURCE_DOC,
        "validation": (
            "Conversion formula confirmed from FLIR's Duo Pro R User Guide. Emissivity "
            "and background temperature were fixed in the camera at capture and are not "
            "recorded in the file. Not compared with an independent reference "
            "measurement."),
    }
    if documented == 0.40:
        provenance["note"] = ("Low-resolution range: each count is 0.4 K, so values are "
                              "quantised to 0.4 °C steps.")
    return ThermalImage(celsius=celsius, calibration=None, source=str(target),
                        provenance=provenance)
