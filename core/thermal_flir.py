"""FLIR radiometric JPEGs: a clean-room reader for the FFF records in APP1.

A FLIR radiometric JPEG (and the FLIR-core DJI Zenmuse XT2 / XT R) carries, beside the
rendered picture, an "FFF" container split across APP1 segments that start with
``FLIR\\0``. The container holds a record directory. Two records matter here:

* the raw-data record (type 1): the 16-bit detector counts, stored either as plain
  little-endian words or as a PNG whose 16-bit samples are byte-swapped;
* the camera-info record (type 0x20): the Planck constants R1/R2/B/F/O, the
  emissivity, object distance, reflected/atmospheric/IR-window temperatures, relative
  humidity and the atmospheric-transmission coefficients.

The layout was written from the record offsets ExifTool *documents* for its FLIR tags
(the tag tables, not its source) and was checked field by field against the ExifTool
12.35 dumps shipped with the test images. No GPL code was read or copied.

The counts become temperature through ``core.thermal``'s Planck model, extended with
FLIR's atmospheric-transmission model (see ``FlirAtmosphere``). Nothing is read from
the rendered palette.
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .thermal import Calibration, FlirAtmosphere, NotRadiometric

FFF_MAGIC = b"FFF\0"
APP1_FLIR_MAGIC = b"FLIR\0"
RECORD_RAW_DATA = 0x0001
RECORD_CAMERA_INFO = 0x0020
KELVIN = 273.15


def _jpeg_segments(data: bytes):
    """Yield (marker, payload) for every JPEG marker segment before the scan data."""
    if data[:2] != b"\xff\xd8":
        return
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        if marker == 0xD9:
            return
        length = struct.unpack(">H", data[position + 2:position + 4])[0]
        payload = data[position + 4:position + 2 + length]
        yield marker, payload
        if marker == 0xDA:  # start of scan: the entropy-coded data follows
            return
        position += 2 + length


def extract_fff(data: bytes) -> bytes | None:
    """Concatenate the FLIR APP1 chunks in their stated order, or None if absent."""
    chunks: dict[int, bytes] = {}
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE1 and payload.startswith(APP1_FLIR_MAGIC):
            # FLIR\0, then a version byte, this chunk's index and the last index.
            index = payload[6]
            chunks[index] = payload[8:]
    if not chunks:
        return None
    blob = b"".join(chunks[key] for key in sorted(chunks))
    return blob if blob.startswith(FFF_MAGIC) else None


def is_flir_radiometric_jpeg(path: str | Path) -> bool:
    try:
        head = Path(path).read_bytes()
    except OSError:
        return False
    return extract_fff(head) is not None


def _directory(fff: bytes) -> list[tuple[int, int, int]]:
    """(record type, offset, length) for each record in the FFF directory."""
    # The header's version field (offset 0x14) is a small number such as 100 or 101;
    # whichever byte order reads it that way is the file's.
    for order in (">", "<"):
        version = struct.unpack(order + "I", fff[0x14:0x18])[0]
        if 100 <= version < 200:
            break
    else:
        raise NotRadiometric("FLIR FFF header has an unrecognised version or byte order.")
    directory_offset, entries = struct.unpack(order + "II", fff[0x18:0x20])
    records = []
    for index in range(entries):
        start = directory_offset + 32 * index
        entry = fff[start:start + 32]
        if len(entry) < 32:
            break
        record_type = struct.unpack(order + "H", entry[0:2])[0]
        offset, length = struct.unpack(order + "II", entry[12:20])
        if record_type and offset + length <= len(fff):
            records.append((record_type, offset, length))
    return records


def _record_order(record: bytes) -> str:
    """Raw-data and camera-info records start with the word 2 in their own byte order."""
    return "<" if struct.unpack("<H", record[0:2])[0] == 2 else ">"


def _raw_counts(record: bytes) -> np.ndarray:
    order = _record_order(record)
    width, height = struct.unpack(order + "HH", record[2:6])
    body = record[0x20:]
    if body.startswith(b"\x89PNG"):
        from PIL import Image

        with Image.open(io.BytesIO(body)) as png:
            png.load()
            counts = np.array(png, dtype=np.uint32).astype(np.uint16)
        # FLIR writes little-endian words into a PNG, whose 16-bit samples are
        # big-endian by definition, so every sample arrives byte-swapped.
        counts = counts.byteswap()
    else:
        dtype = np.dtype("<u2") if order == "<" else np.dtype(">u2")
        needed = width * height * 2
        if len(body) < needed:
            raise NotRadiometric("FLIR raw-data record is shorter than its stated size.")
        counts = np.frombuffer(body[:needed], dtype=dtype).reshape(height, width)
    if counts.shape != (height, width):
        raise NotRadiometric(
            f"FLIR raw image is {counts.shape[1]}x{counts.shape[0]}, but its record "
            f"says {width}x{height}.")
    return counts.astype(np.float64)


def _camera_info(record: bytes) -> dict[str, Any]:
    order = _record_order(record)

    def f32(offset: int) -> float:
        return float(struct.unpack(order + "f", record[offset:offset + 4])[0])

    def i32(offset: int) -> int:
        return int(struct.unpack(order + "i", record[offset:offset + 4])[0])

    def text(offset: int, size: int) -> str:
        return record[offset:offset + size].split(b"\0", 1)[0].decode("latin-1").strip()

    return {
        "Emissivity": f32(0x20),
        "ObjectDistance": f32(0x24),
        "ReflectedApparentTemperature": f32(0x28) - KELVIN,
        "AtmosphericTemperature": f32(0x2C) - KELVIN,
        "IRWindowTemperature": f32(0x30) - KELVIN,
        "IRWindowTransmission": f32(0x34),
        "RelativeHumidity": f32(0x3C),
        "PlanckR1": f32(0x58),
        "PlanckB": f32(0x5C),
        "PlanckF": f32(0x60),
        "AtmosphericTransAlpha1": f32(0x70),
        "AtmosphericTransAlpha2": f32(0x74),
        "AtmosphericTransBeta1": f32(0x78),
        "AtmosphericTransBeta2": f32(0x7C),
        "AtmosphericTransX": f32(0x80),
        "CameraModel": text(0xD4, 32),
        "PlanckO": i32(0x308),
        "PlanckR2": f32(0x30C),
        "RawValueMedian": int(struct.unpack(order + "H", record[0x338:0x33A])[0]),
        "RawValueRange": int(struct.unpack(order + "H", record[0x33C:0x33E])[0]),
    }


@dataclass
class FlirRadiometricJpeg:
    counts: np.ndarray
    metadata: dict[str, Any]

    def calibration(self) -> Calibration:
        m = self.metadata
        # ExifTool reports relative humidity in percent; some firmware stores it as a
        # fraction, some as percent. A value above 1 is percent.
        humidity = m["RelativeHumidity"]
        humidity = humidity / 100.0 if humidity > 1.0 else humidity
        return Calibration(
            r1=m["PlanckR1"], r2=m["PlanckR2"], b=m["PlanckB"], f=m["PlanckF"],
            o=float(m["PlanckO"]), emissivity=m["Emissivity"],
            reflected_temp_c=m["ReflectedApparentTemperature"],
            atmosphere=FlirAtmosphere(
                atmospheric_temp_c=m["AtmosphericTemperature"],
                object_distance_m=m["ObjectDistance"],
                relative_humidity=humidity,
                ir_window_temp_c=m["IRWindowTemperature"],
                ir_window_transmission=m["IRWindowTransmission"],
                alpha1=m["AtmosphericTransAlpha1"], alpha2=m["AtmosphericTransAlpha2"],
                beta1=m["AtmosphericTransBeta1"], beta2=m["AtmosphericTransBeta2"],
                x=m["AtmosphericTransX"],
            ),
        )


def read_flir_jpeg(path: str | Path) -> FlirRadiometricJpeg:
    """Parse the raw counts and calibration of a FLIR-format radiometric JPEG."""
    target = Path(path)
    fff = extract_fff(target.read_bytes())
    if fff is None:
        raise NotRadiometric(f"{target.name} carries no FLIR FFF radiometric record.")
    records = _directory(fff)
    raw = next((fff[o:o + n] for t, o, n in records if t == RECORD_RAW_DATA), None)
    info = next((fff[o:o + n] for t, o, n in records if t == RECORD_CAMERA_INFO), None)
    if raw is None:
        raise NotRadiometric(f"{target.name}: the FLIR record has no raw thermal image.")
    if info is None or len(info) < 0x340:
        raise NotRadiometric(
            f"{target.name}: the FLIR record has no camera-info block, so the Planck "
            "constants are missing and the counts cannot be converted.")
    metadata = _camera_info(info)
    for key in ("PlanckR1", "PlanckR2", "PlanckB"):
        if not math.isfinite(metadata[key]) or metadata[key] <= 0:
            raise NotRadiometric(f"{target.name}: {key}={metadata[key]!r} is not usable.")
    return FlirRadiometricJpeg(counts=_raw_counts(raw), metadata=metadata)
