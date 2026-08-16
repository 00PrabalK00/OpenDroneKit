"""Where water can collect on a surface, measured from the DSM.

This is a geometric question and is answered geometrically. A model shown a photograph
would have to infer ponding from colour and specularity, and would answer confidently on
wet-but-not-ponded membrane, on shadow, and on solar glare -- all of which look like
standing water from above. A closed depression in a surface either exists or it does not,
and its depth is a number with an error bound.

The method is sink filling. Raising every cell to the lowest elevation on its lowest path
to the surface edge produces a filled surface; the difference between filled and original
is exactly the water a depression can hold before it spills. That is the definition of
ponding, not a proxy for it.

What this reports is CAPACITY, not observation: "water can collect here to this depth",
not "water is here now". The second is an appearance question about a particular flight
and must never be merged into the first -- a dry roof with a 40 mm depression and a
flooded one look identical to this module, and conflating them would put a measurement's
authority behind a guess.

    from core.ponding import find_ponding
    report = find_ponding(surface)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dsm_analysis import RasterSurface
from .slope import NotProjected


class PondingRefused(ValueError):
    """The surface cannot support a ponding measurement worth reporting."""


# A depression shallower than this multiple of the surface's own vertical error is not
# distinguishable from noise in it. Two sigma rather than one: at one sigma roughly a
# third of pure-noise dimples would be reported as ponding.
NOISE_MULTIPLE = 2.0

# Depressions smaller than this are almost always reconstruction artefacts around edges
# and vegetation rather than real basins.
MIN_AREA_CELLS = 9


@dataclass
class Depression:
    """One closed basin, with what its numbers depend on stated."""

    area_m2: float
    max_depth_m: float
    mean_depth_m: float
    volume_m3: float
    cell_count: int
    centroid_xy: tuple[float, float]
    # Carried through from the surface rather than recomputed, so a reader can see the
    # error on every number above without going hunting.
    vertical_accuracy_m: float

    @property
    def depth_is_significant(self) -> bool:
        return self.max_depth_m > NOISE_MULTIPLE * self.vertical_accuracy_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_m2": round(self.area_m2, 3),
            "max_depth_m": round(self.max_depth_m, 4),
            "mean_depth_m": round(self.mean_depth_m, 4),
            "volume_m3": round(self.volume_m3, 4),
            "cell_count": self.cell_count,
            "centroid_xy": [round(v, 3) for v in self.centroid_xy],
            "vertical_accuracy_m": round(self.vertical_accuracy_m, 4),
            "depth_is_significant": self.depth_is_significant,
        }


@dataclass
class PondingReport:
    depressions: list[Depression] = field(default_factory=list)
    detection_floor_m: float = 0.0
    cell_size_m: float = 0.0
    note: str = ""

    @property
    def total_volume_m3(self) -> float:
        return float(sum(d.volume_m3 for d in self.depressions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "depressions": [d.to_dict() for d in self.depressions],
            "count": len(self.depressions),
            "total_volume_m3": round(self.total_volume_m3, 4),
            "detection_floor_m": round(self.detection_floor_m, 4),
            "cell_size_m": round(self.cell_size_m, 4),
            "note": self.note,
        }


def fill_sinks(elevation: np.ndarray, *, max_iterations: int = 400) -> np.ndarray:
    """Raise each cell to the lowest elevation on its lowest path to the edge.

    Planchon-Darboux: start from a surface flooded everywhere except its boundary, then
    repeatedly lower each interior cell to the highest of its own elevation and the
    lowest of its neighbours. It converges to the filled surface from above.

    NaN cells are treated as boundary. A hole in the reconstruction is somewhere water
    could drain through for all this module knows, and assuming otherwise would invent
    a basin out of missing data.
    """
    known = np.isfinite(elevation)
    if not known.any():
        raise PondingRefused("The surface has no valid elevations to measure.")

    high = float(np.nanmax(elevation)) + 1.0
    filled = np.where(known, high, np.nan)

    # Boundary cells, and any cell adjacent to no-data, drain freely.
    edge = np.zeros(elevation.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    holes = ~known
    if holes.any():
        neighbours_missing = np.zeros_like(edge)
        neighbours_missing[1:, :] |= holes[:-1, :]
        neighbours_missing[:-1, :] |= holes[1:, :]
        neighbours_missing[:, 1:] |= holes[:, :-1]
        neighbours_missing[:, :-1] |= holes[:, 1:]
        edge |= neighbours_missing
    edge &= known
    filled[edge] = elevation[edge]

    for _ in range(max_iterations):
        previous = filled.copy()
        neighbour_min = _neighbour_min(filled)
        candidate = np.maximum(elevation, neighbour_min)
        update = known & ~edge & (candidate < filled)
        filled[update] = candidate[update]
        if np.allclose(np.nan_to_num(filled), np.nan_to_num(previous), atol=1e-9):
            break
    return filled


def _neighbour_min(surface: np.ndarray) -> np.ndarray:
    """Minimum over the four-connected neighbourhood, ignoring no-data."""
    padded = np.pad(surface, 1, mode="constant", constant_values=np.nan)
    stack = np.stack([
        padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:],
    ])
    with np.errstate(all="ignore"):
        result = np.nanmin(stack, axis=0)
    return result


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Four-connected labelling, without requiring scipy."""
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


def find_ponding(
    surface: RasterSurface,
    *,
    vertical_accuracy_m: float | None = None,
    min_area_cells: int = MIN_AREA_CELLS,
) -> PondingReport:
    """Closed depressions on a surface, with the detection floor reported beside them.

    ``vertical_accuracy_m`` is required rather than defaulted. Without it there is no
    way to say which depressions are real, and a list of basins with no error bound is
    exactly the confident-but-uncheckable output this module exists to avoid.
    """
    if surface.epsg is None:
        raise NotProjected(
            "Ponding needs a projected surface: depth is metres and area is square "
            "metres, and neither means anything over degrees."
        )
    cell = surface.pixel_size_m
    if not np.isfinite(cell) or cell <= 0:
        raise NotProjected("The raster transform has no ground spacing to measure across.")

    if vertical_accuracy_m is None:
        raise PondingRefused(
            "No vertical accuracy was supplied for this surface. Ponding depths cannot "
            "be separated from reconstruction noise without one, so nothing is reported "
            "rather than reporting basins that may not exist."
        )
    if vertical_accuracy_m <= 0:
        raise PondingRefused("Vertical accuracy must be a positive distance in metres.")

    floor = NOISE_MULTIPLE * float(vertical_accuracy_m)
    filled = fill_sinks(surface.elevation)
    depth = np.where(np.isfinite(filled) & np.isfinite(surface.elevation),
                     filled - surface.elevation, 0.0)
    depth[~np.isfinite(depth)] = 0.0

    labels, count = _label_regions(depth > floor)
    cell_area = surface.pixel_area_m2
    depressions: list[Depression] = []
    for index in range(1, count + 1):
        region = labels == index
        cells = int(region.sum())
        if cells < min_area_cells:
            continue
        values = depth[region]
        rows, cols = np.nonzero(region)
        x, y = surface.xy_of(rows, cols)
        depressions.append(Depression(
            area_m2=cells * cell_area,
            max_depth_m=float(values.max()),
            mean_depth_m=float(values.mean()),
            volume_m3=float(values.sum()) * cell_area,
            cell_count=cells,
            centroid_xy=(float(np.mean(x)), float(np.mean(y))),
            vertical_accuracy_m=float(vertical_accuracy_m),
        ))

    depressions.sort(key=lambda d: d.volume_m3, reverse=True)
    return PondingReport(
        depressions=depressions,
        detection_floor_m=floor,
        cell_size_m=cell,
        note=(
            f"Depressions shallower than {floor:.3f} m are not reported: that is "
            f"{NOISE_MULTIPLE:g} x the surface's stated vertical accuracy of "
            f"{vertical_accuracy_m:.3f} m, below which a basin cannot be told from "
            "noise. This measures where water CAN collect, not whether water is "
            "present now; a dry depression and a flooded one are identical here."
        ),
    )
