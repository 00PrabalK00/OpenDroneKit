"""Ponding detection, tested against surfaces whose answer is known exactly.

This is why the capability was built as a measurement rather than a model: a synthetic
surface with a 60 mm depression of known area has a known volume, so the code can be
checked against arithmetic instead of against a corpus that does not exist.

The tests that matter most are the refusals. A basin list with no error bound, or one
computed over degrees, is confident output nobody can check -- which is worse than no
output at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dsm_analysis import RasterSurface
from core.ponding import (
    NOISE_MULTIPLE,
    PondingRefused,
    fill_sinks,
    find_ponding,
)
from core.slope import NotProjected


def _flat(size: int = 40, height: float = 10.0, cell: float = 0.5) -> RasterSurface:
    return RasterSurface(
        elevation=np.full((size, size), height, dtype=float),
        transform=(cell, 0.0, 1000.0, 0.0, -cell, 2000.0),
        epsg=32643,
    )


def _with_basin(depth: float, half: int = 4, cell: float = 0.5) -> RasterSurface:
    surface = _flat(cell=cell)
    mid = surface.elevation.shape[0] // 2
    surface.elevation[mid - half:mid + half, mid - half:mid + half] -= depth
    return surface


class TestRefusals:
    def test_a_geographic_surface_is_refused(self) -> None:
        surface = _flat()
        surface.epsg = None
        with pytest.raises(NotProjected):
            find_ponding(surface, vertical_accuracy_m=0.02)

    def test_no_vertical_accuracy_is_refused(self) -> None:
        # The point of the whole module: without an error bound there is no way to say
        # which basins are real, so it reports nothing rather than reporting guesses.
        with pytest.raises(PondingRefused, match="vertical accuracy"):
            find_ponding(_with_basin(0.1), vertical_accuracy_m=None)

    def test_a_nonsense_accuracy_is_refused(self) -> None:
        with pytest.raises(PondingRefused):
            find_ponding(_with_basin(0.1), vertical_accuracy_m=0.0)

    def test_an_empty_surface_is_refused(self) -> None:
        surface = _flat()
        surface.elevation[:] = np.nan
        with pytest.raises(PondingRefused, match="no valid elevations"):
            find_ponding(surface, vertical_accuracy_m=0.02)


class TestMeasurement:
    def test_a_flat_roof_ponds_nowhere(self) -> None:
        report = find_ponding(_flat(), vertical_accuracy_m=0.02)
        assert report.depressions == []
        assert report.total_volume_m3 == 0.0

    def test_a_known_basin_measures_correctly(self) -> None:
        # 8x8 cells at 0.5 m = 16 m^2, 0.06 m deep -> 0.96 m^3. Arithmetic, not a metric.
        surface = _with_basin(0.06, half=4, cell=0.5)
        report = find_ponding(surface, vertical_accuracy_m=0.01)
        assert len(report.depressions) == 1
        found = report.depressions[0]
        assert found.area_m2 == pytest.approx(16.0, rel=1e-6)
        assert found.max_depth_m == pytest.approx(0.06, abs=1e-6)
        assert found.volume_m3 == pytest.approx(0.96, rel=1e-3)
        assert found.depth_is_significant

    def test_two_basins_are_reported_separately(self) -> None:
        surface = _flat()
        surface.elevation[6:12, 6:12] -= 0.08
        surface.elevation[26:32, 26:32] -= 0.05
        report = find_ponding(surface, vertical_accuracy_m=0.01)
        assert len(report.depressions) == 2
        # Ordered by volume so the worst basin is what a reader sees first.
        assert report.depressions[0].volume_m3 >= report.depressions[1].volume_m3

    def test_a_basin_open_to_the_edge_is_not_ponding(self) -> None:
        # A channel running off the roof drains. Reporting it as ponding would be the
        # classic false positive: it looks like a depression in a height field but water
        # does not stay in it.
        surface = _flat()
        surface.elevation[18:22, 20:] -= 0.08
        report = find_ponding(surface, vertical_accuracy_m=0.01)
        assert report.depressions == [], "a drained channel must not be reported as ponding"


class TestDetectionFloor:
    def test_a_basin_below_the_noise_floor_is_not_reported(self) -> None:
        # 10 mm deep against a 20 mm accuracy: indistinguishable from reconstruction
        # noise, so it must not appear however real it might be.
        report = find_ponding(_with_basin(0.01), vertical_accuracy_m=0.02)
        assert report.depressions == []

    def test_the_floor_is_reported_beside_the_result(self) -> None:
        report = find_ponding(_with_basin(0.06), vertical_accuracy_m=0.015)
        assert report.detection_floor_m == pytest.approx(NOISE_MULTIPLE * 0.015)
        assert "not reported" in report.note

    def test_the_note_separates_capacity_from_observation(self) -> None:
        # "Water can collect here" and "water is here" are different claims, and the
        # report must not let a reader merge them.
        report = find_ponding(_with_basin(0.06), vertical_accuracy_m=0.01)
        assert "not whether water is" in report.note


class TestFillSinks:
    def test_no_data_drains_rather_than_holding_water(self) -> None:
        # A hole in the reconstruction might be a drain for all this knows. Treating it
        # as a wall would invent a basin out of missing data.
        surface = _flat()
        surface.elevation[15:25, 15:25] -= 0.1
        surface.elevation[20, 20] = np.nan
        report = find_ponding(surface, vertical_accuracy_m=0.01)
        for found in report.depressions:
            assert found.cell_count < 100

    def test_filling_never_lowers_the_surface(self) -> None:
        surface = _with_basin(0.07)
        filled = fill_sinks(surface.elevation)
        finite = np.isfinite(filled) & np.isfinite(surface.elevation)
        assert np.all(filled[finite] >= surface.elevation[finite] - 1e-9)
