"""Choosing the temperature range a thermal image is drawn in.

A radiometric frame holds temperatures. A screen holds 256 grey levels. Something has to
decide which temperatures map onto them, and that decision changes what the inspector
sees more than any other step in the thermal workflow.

The two ways it goes wrong are opposite and both common:

  Full range   One 80 C exhaust stack in the frame stretches the scale so far that the
               3 C delta across a wet roof -- the thing being looked for -- renders as a
               single flat grey. The defect is in the data and invisible on the screen.

  Tight range  A range chosen to show that 3 C delta clips the stack. The hottest pixels
               all saturate to the same colour, so a genuinely dangerous hotspot looks
               exactly like a merely warm one.

The second is the dangerous one, because the image looks fine. So every scaling this
module produces reports what it clipped: how many pixels, and what the true extremes were.
An inspector who cannot see that a range hid something is not inspecting, they are looking
at a picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class ScalingRefused(ValueError):
    """A range that would render nothing, or nothing legible."""


#: Palettes as (position, R, G, B) control points, interpolated across the range.
#: Ironbow is the industry default and the one the thermal processing guide expects;
#: greyscale is here because a palette makes small deltas look like structure to some
#: eyes, and an inspector second-guessing a colour can switch it off.
PALETTES: dict[str, list[tuple[float, int, int, int]]] = {
    "ironbow": [
        (0.00, 0, 0, 0), (0.25, 60, 0, 110), (0.50, 180, 40, 90),
        (0.75, 250, 140, 20), (0.90, 255, 220, 60), (1.00, 255, 255, 255),
    ],
    "greyscale": [(0.00, 0, 0, 0), (1.00, 255, 255, 255)],
    "amber": [(0.00, 0, 0, 0), (0.60, 140, 60, 0), (1.00, 255, 210, 120)],
}


@dataclass
class TemperatureScale:
    """The range an image is drawn in, and what that choice cost."""

    min_c: float
    max_c: float
    palette: str = "ironbow"
    #: What the data actually contains, so the caller can see what was excluded.
    data_min_c: float = 0.0
    data_max_c: float = 0.0
    clipped_low: int = 0
    clipped_high: int = 0
    total_pixels: int = 0
    method: str = "manual"

    @property
    def span_c(self) -> float:
        return self.max_c - self.min_c

    @property
    def clipped_fraction(self) -> float:
        if not self.total_pixels:
            return 0.0
        return (self.clipped_low + self.clipped_high) / self.total_pixels

    def hides_the_hottest(self) -> bool:
        """Whether the hottest pixels in the frame are saturated by this range.

        The failure that matters: a dangerous hotspot rendering the same colour as a
        merely warm one, in an image that looks perfectly reasonable.
        """
        return self.clipped_high > 0 and self.data_max_c > self.max_c

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_c": self.min_c,
            "max_c": self.max_c,
            "span_c": self.span_c,
            "palette": self.palette,
            "method": self.method,
            "data_min_c": self.data_min_c,
            "data_max_c": self.data_max_c,
            "clipped_low": self.clipped_low,
            "clipped_high": self.clipped_high,
            "clipped_fraction": self.clipped_fraction,
            "hides_the_hottest": self.hides_the_hottest(),
        }

    def warning(self) -> str:
        """What an operator needs told about this range, or an empty string."""
        if self.hides_the_hottest():
            return (
                f"This range saturates {self.clipped_high:,} pixel(s) above "
                f"{self.max_c:.1f} C; the frame reaches {self.data_max_c:.1f} C. The "
                "hottest area is drawn the same colour as anything else above the top of "
                "the scale, so a severe hotspot and a mild one look identical."
            )
        if self.clipped_low and self.data_min_c < self.min_c:
            return (
                f"{self.clipped_low:,} pixel(s) fall below {self.min_c:.1f} C and are "
                f"drawn as the coldest colour; the frame reaches {self.data_min_c:.1f} C."
            )
        return ""


def _finite(celsius: np.ndarray) -> np.ndarray:
    values = np.asarray(celsius, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ScalingRefused("The frame carries no finite temperatures to scale.")
    return finite


def manual_scale(celsius: np.ndarray, min_c: float, max_c: float,
                 palette: str = "ironbow") -> TemperatureScale:
    """A range the operator chose, with an honest account of what it excludes."""
    if palette not in PALETTES:
        raise ScalingRefused(
            f"{palette!r} is not a palette. Use one of: " + ", ".join(sorted(PALETTES)))
    if not np.isfinite(min_c) or not np.isfinite(max_c):
        raise ScalingRefused("A temperature range needs two finite numbers.")
    if max_c <= min_c:
        raise ScalingRefused(
            f"The top of the range ({max_c} C) must be above the bottom ({min_c} C). "
            "A zero or inverted span renders one flat colour."
        )

    finite = _finite(celsius)
    return TemperatureScale(
        min_c=float(min_c), max_c=float(max_c), palette=palette,
        data_min_c=float(finite.min()), data_max_c=float(finite.max()),
        clipped_low=int((finite < min_c).sum()),
        clipped_high=int((finite > max_c).sum()),
        total_pixels=int(finite.size),
        method="manual",
    )


def auto_scale(celsius: np.ndarray, low_percentile: float = 1.0,
               high_percentile: float = 99.0, palette: str = "ironbow") -> TemperatureScale:
    """A range from percentiles, so a handful of extreme pixels cannot flatten the rest.

    Percentiles rather than min/max because a single sun-glinted rivet or a dead pixel at
    300 C compresses everything else into two grey levels. Trimming 1% at each end is the
    difference between an image of a roof and an image of one rivet.

    It still reports what it trimmed. That is the whole contract of this module: the
    inspector is told the range excluded something, every time it did.
    """
    if palette not in PALETTES:
        raise ScalingRefused(
            f"{palette!r} is not a palette. Use one of: " + ", ".join(sorted(PALETTES)))
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ScalingRefused(
            f"Percentiles must satisfy 0 <= low < high <= 100; got {low_percentile} "
            f"and {high_percentile}."
        )

    finite = _finite(celsius)
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low:
        # A near-uniform field: every percentile lands on the same value. Widen by a
        # small amount rather than refuse, so a flat wall still renders.
        pad = max(0.5, abs(low) * 0.01)
        low, high = low - pad, high + pad

    scale = manual_scale(finite, low, high, palette)
    scale.method = f"percentile {low_percentile:g}-{high_percentile:g}"
    return scale


def anomaly_scale(celsius: np.ndarray, sigma: float = 2.0,
                  palette: str = "ironbow") -> TemperatureScale:
    """A range centred on the scene, so departures from normal stand out.

    For finding what is unusual rather than reading absolute values: the range is the mean
    plus and minus a few standard deviations, so ordinary surface temperature occupies the
    middle of the palette and an anomaly sits visibly at one end.
    """
    if sigma <= 0:
        raise ScalingRefused("Sigma must be positive.")
    finite = _finite(celsius)
    mean = float(finite.mean())
    spread = float(finite.std())
    if spread <= 0:
        spread = max(0.5, abs(mean) * 0.01)
    scale = manual_scale(finite, mean - sigma * spread, mean + sigma * spread, palette)
    scale.method = f"mean +/- {sigma:g} sigma"
    return scale


def render(celsius: np.ndarray, scale: TemperatureScale) -> np.ndarray:
    """Temperatures to an RGB image, using the palette and range given.

    Non-finite pixels render black rather than being silently mapped to the bottom of the
    scale, which would draw a dead sensor element as the coldest thing in the frame.
    """
    values = np.asarray(celsius, dtype=np.float64)
    finite_mask = np.isfinite(values)

    normalised = np.zeros(values.shape, dtype=np.float64)
    span = scale.span_c or 1.0
    normalised[finite_mask] = np.clip((values[finite_mask] - scale.min_c) / span, 0.0, 1.0)

    stops = PALETTES[scale.palette]
    positions = np.array([s[0] for s in stops], dtype=np.float64)
    channels = [np.array([s[i] for s in stops], dtype=np.float64) for i in (1, 2, 3)]

    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for index, table in enumerate(channels):
        rgb[..., index] = np.interp(normalised, positions, table).astype(np.uint8)
    rgb[~finite_mask] = 0
    return rgb


def legend(scale: TemperatureScale, steps: int = 5) -> list[dict[str, Any]]:
    """Tick values for a colour bar, so the picture carries its own units."""
    if steps < 2:
        raise ScalingRefused("A legend needs at least two ticks.")
    return [
        {
            "fraction": i / (steps - 1),
            "celsius": scale.min_c + scale.span_c * (i / (steps - 1)),
        }
        for i in range(steps)
    ]
