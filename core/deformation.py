"""How far a surface moved between two surveys, and whether that number means anything.

Deformation is the difference between two surfaces captured at different times. A model
shown a single survey has nothing to compare against and can only pattern-match to what
deformation usually looks like, which is why this is measured rather than learned.

The gate is the feature. Two surveys of the same unchanged ground never difference to
zero: each has its own vertical error, and co-registering them adds a residual on top.
Any displacement smaller than those combined is noise with a decimal point, and this
module refuses to report it. That refusal is not a safety margin bolted on afterwards --
without it, every output would be indistinguishable from measurement error, and a
subsidence report nobody can check is worse than none.

    from core.deformation import compare_surfaces
    report = compare_surfaces(earlier, later, earlier_accuracy_m=0.03,
                              later_accuracy_m=0.03, registration_residual_m=0.02)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .dsm_analysis import RasterSurface
from .slope import NotProjected


class DeformationRefused(ValueError):
    """The two surveys cannot be compared into a number worth reporting."""


# Regions smaller than this are almost always reconstruction artefacts rather than
# ground that moved.
MIN_AREA_CELLS = 9


@dataclass
class DisplacementRegion:
    """One contiguous area whose movement exceeded the detection floor."""

    area_m2: float
    max_displacement_m: float
    mean_displacement_m: float
    volume_change_m3: float
    cell_count: int
    centroid_xy: tuple[float, float]
    direction: str  # "subsidence" or "uplift"

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_m2": round(self.area_m2, 3),
            "max_displacement_m": round(self.max_displacement_m, 4),
            "mean_displacement_m": round(self.mean_displacement_m, 4),
            "volume_change_m3": round(self.volume_change_m3, 3),
            "cell_count": self.cell_count,
            "centroid_xy": [round(v, 3) for v in self.centroid_xy],
            "direction": self.direction,
        }


@dataclass
class DeformationReport:
    regions: list[DisplacementRegion] = field(default_factory=list)
    detection_floor_m: float = 0.0
    compared_cells: int = 0
    cell_size_m: float = 0.0
    note: str = ""

    @property
    def moved(self) -> bool:
        return bool(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "count": len(self.regions),
            "detection_floor_m": round(self.detection_floor_m, 4),
            "compared_cells": self.compared_cells,
            "cell_size_m": round(self.cell_size_m, 4),
            "moved": self.moved,
            "note": self.note,
        }


def detection_floor(
    earlier_accuracy_m: float,
    later_accuracy_m: float,
    registration_residual_m: float,
) -> float:
    """The smallest displacement this pair of surveys can resolve.

    The two vertical errors combine in quadrature because they are independent; the
    registration residual adds linearly because it is a systematic offset between the
    surfaces rather than random noise within either.
    """
    combined_noise = math.hypot(float(earlier_accuracy_m), float(later_accuracy_m))
    return combined_noise + abs(float(registration_residual_m))


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    rows, cols = mask.shape
    for start_r in range(rows):
        for start_c in range(cols):
            if not mask[start_r, start_c] or labels[start_r, start_c]:
                continue
            current += 1
            stack = [(start_r, start_c)]
            labels[start_r, start_c] = current
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = current
                        stack.append((nr, nc))
    return labels, current


def compare_surfaces(
    earlier: RasterSurface,
    later: RasterSurface,
    *,
    earlier_accuracy_m: float | None = None,
    later_accuracy_m: float | None = None,
    registration_residual_m: float | None = None,
    min_area_cells: int = MIN_AREA_CELLS,
) -> DeformationReport:
    """Vertical displacement between two epochs, with the detection floor enforced.

    All three uncertainties are required. Defaulting any of them would let this report a
    displacement without knowing whether the surveys could resolve it, which is the one
    thing this module exists to prevent.
    """
    for surface, label in ((earlier, "earlier"), (later, "later")):
        if surface.epsg is None:
            raise NotProjected(
                f"The {label} surface has no projected CRS. Displacement is metres and "
                "cannot be measured over degrees."
            )
    if earlier.epsg != later.epsg:
        raise DeformationRefused(
            f"The surveys are in different coordinate systems (EPSG {earlier.epsg} and "
            f"{later.epsg}). Differencing them would subtract unlike quantities."
        )
    if earlier.elevation.shape != later.elevation.shape:
        raise DeformationRefused(
            f"The surfaces have different shapes ({earlier.elevation.shape} and "
            f"{later.elevation.shape}); they are not on a common grid, so cell-by-cell "
            "differencing would compare different ground."
        )
    if not np.allclose(np.asarray(earlier.transform[:6], dtype=float),
                       np.asarray(later.transform[:6], dtype=float), atol=1e-6):
        raise DeformationRefused(
            "The surfaces have different affine transforms, so identical pixel indices "
            "refer to different ground. Resample onto a common grid first."
        )

    missing = [name for name, value in (
        ("earlier_accuracy_m", earlier_accuracy_m),
        ("later_accuracy_m", later_accuracy_m),
        ("registration_residual_m", registration_residual_m),
    ) if value is None]
    if missing:
        raise DeformationRefused(
            "Cannot report displacement without " + ", ".join(missing) + ". Two surveys "
            "of unchanged ground never difference to zero, so without these there is no "
            "way to tell movement from measurement error."
        )
    if earlier_accuracy_m <= 0 or later_accuracy_m <= 0:
        raise DeformationRefused("Survey accuracies must be positive distances in metres.")

    floor = detection_floor(earlier_accuracy_m, later_accuracy_m, registration_residual_m)
    cell = earlier.pixel_size_m
    if not np.isfinite(cell) or cell <= 0:
        raise NotProjected("The raster transform has no ground spacing to measure across.")

    both_known = np.isfinite(earlier.elevation) & np.isfinite(later.elevation)
    if not both_known.any():
        raise DeformationRefused(
            "The surveys share no cells with data in both. There is nothing to compare."
        )

    displacement = np.where(both_known, later.elevation - earlier.elevation, np.nan)
    significant = both_known & (np.abs(np.nan_to_num(displacement)) > floor)

    cell_area = earlier.pixel_area_m2
    regions: list[DisplacementRegion] = []
    # Subsidence and uplift are labelled separately rather than by absolute magnitude:
    # a pit next to a spoil heap is two findings, not one region straddling zero.
    for direction, mask in (("subsidence", significant & (displacement < 0)),
                            ("uplift", significant & (displacement > 0))):
        labels, count = _label_regions(mask)
        for index in range(1, count + 1):
            region = labels == index
            cells = int(region.sum())
            if cells < min_area_cells:
                continue
            values = displacement[region]
            rows, cols = np.nonzero(region)
            x, y = earlier.xy_of(rows, cols)
            regions.append(DisplacementRegion(
                area_m2=cells * cell_area,
                max_displacement_m=float(np.abs(values).max()),
                mean_displacement_m=float(np.abs(values).mean()),
                volume_change_m3=float(values.sum()) * cell_area,
                cell_count=cells,
                centroid_xy=(float(np.mean(x)), float(np.mean(y))),
                direction=direction,
            ))

    regions.sort(key=lambda r: r.max_displacement_m, reverse=True)
    return DeformationReport(
        regions=regions,
        detection_floor_m=floor,
        compared_cells=int(both_known.sum()),
        cell_size_m=cell,
        note=(
            f"Displacements below {floor:.4f} m are not reported. That floor is the two "
            f"surveys' vertical accuracies ({earlier_accuracy_m:.3f} m and "
            f"{later_accuracy_m:.3f} m) combined in quadrature, plus the "
            f"{registration_residual_m:.3f} m co-registration residual. A result at or "
            "under it would be indistinguishable from measurement error. Absence of a "
            "region means no movement was RESOLVABLE, not that none occurred."
        ),
    )
