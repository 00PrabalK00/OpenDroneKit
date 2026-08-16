"""Deformation between two epochs, and the floor that makes it meaningful.

Two surveys of unchanged ground never difference to zero. Each carries its own vertical
error and co-registering them adds a residual, so any displacement smaller than those
combined is measurement noise wearing a decimal point.

Most of these tests are about refusing rather than measuring, because the failure mode
here is not a wrong number -- it is a plausible number nobody can check.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.deformation import (
    DeformationRefused,
    compare_surfaces,
    detection_floor,
)
from core.dsm_analysis import RasterSurface
from core.slope import NotProjected


def _surface(size: int = 30, height: float = 100.0, cell: float = 1.0,
             epsg: int = 32643) -> RasterSurface:
    return RasterSurface(
        elevation=np.full((size, size), height, dtype=float),
        transform=(cell, 0.0, 500000.0, 0.0, -cell, 3000000.0),
        epsg=epsg,
    )


class TestDetectionFloor:
    def test_errors_combine_in_quadrature_plus_registration(self) -> None:
        # Independent noise adds in quadrature; a registration offset is systematic and
        # adds linearly.
        floor = detection_floor(0.03, 0.04, 0.02)
        assert floor == pytest.approx(0.05 + 0.02)

    def test_the_floor_is_reported_with_the_result(self) -> None:
        report = compare_surfaces(_surface(), _surface(), earlier_accuracy_m=0.03,
                                  later_accuracy_m=0.03, registration_residual_m=0.01)
        assert report.detection_floor_m > 0
        assert "not reported" in report.note


class TestRefusals:
    def test_missing_uncertainties_are_refused(self) -> None:
        with pytest.raises(DeformationRefused, match="registration_residual_m"):
            compare_surfaces(_surface(), _surface(), earlier_accuracy_m=0.03,
                             later_accuracy_m=0.03, registration_residual_m=None)

    def test_all_three_are_named_when_all_are_missing(self) -> None:
        with pytest.raises(DeformationRefused) as exc:
            compare_surfaces(_surface(), _surface())
        for field in ("earlier_accuracy_m", "later_accuracy_m", "registration_residual_m"):
            assert field in str(exc.value)

    def test_mismatched_crs_is_refused(self) -> None:
        with pytest.raises(DeformationRefused, match="coordinate systems"):
            compare_surfaces(_surface(epsg=32643), _surface(epsg=32644),
                             earlier_accuracy_m=0.03, later_accuracy_m=0.03,
                             registration_residual_m=0.01)

    def test_geographic_surfaces_are_refused(self) -> None:
        surface = _surface()
        surface.epsg = None
        with pytest.raises(NotProjected):
            compare_surfaces(surface, _surface(), earlier_accuracy_m=0.03,
                             later_accuracy_m=0.03, registration_residual_m=0.01)

    def test_different_grids_are_refused(self) -> None:
        with pytest.raises(DeformationRefused, match="different shapes"):
            compare_surfaces(_surface(30), _surface(40), earlier_accuracy_m=0.03,
                             later_accuracy_m=0.03, registration_residual_m=0.01)

    def test_shifted_transform_is_refused(self) -> None:
        # Same shape but different ground under the same pixel indices: cell-by-cell
        # differencing would compare two different places and report the difference as
        # deformation.
        later = _surface()
        later.transform = (1.0, 0.0, 500050.0, 0.0, -1.0, 3000000.0)
        with pytest.raises(DeformationRefused, match="affine"):
            compare_surfaces(_surface(), later, earlier_accuracy_m=0.03,
                             later_accuracy_m=0.03, registration_residual_m=0.01)

    def test_no_overlapping_data_is_refused(self) -> None:
        later = _surface()
        later.elevation[:] = np.nan
        with pytest.raises(DeformationRefused, match="nothing to compare"):
            compare_surfaces(_surface(), later, earlier_accuracy_m=0.03,
                             later_accuracy_m=0.03, registration_residual_m=0.01)


class TestMeasurement:
    def test_unchanged_ground_reports_nothing(self) -> None:
        report = compare_surfaces(_surface(), _surface(), earlier_accuracy_m=0.03,
                                  later_accuracy_m=0.03, registration_residual_m=0.01)
        assert report.regions == []
        assert not report.moved

    def test_subsidence_is_found_and_labelled(self) -> None:
        later = _surface()
        later.elevation[10:20, 10:20] -= 0.5
        report = compare_surfaces(_surface(), later, earlier_accuracy_m=0.02,
                                  later_accuracy_m=0.02, registration_residual_m=0.01)
        assert len(report.regions) == 1
        region = report.regions[0]
        assert region.direction == "subsidence"
        assert region.max_displacement_m == pytest.approx(0.5, abs=1e-6)
        assert region.area_m2 == pytest.approx(100.0)
        assert region.volume_change_m3 == pytest.approx(-50.0, rel=1e-3)

    def test_uplift_and_subsidence_are_separate_findings(self) -> None:
        # A pit beside a spoil heap is two findings, not one region straddling zero.
        later = _surface()
        later.elevation[5:12, 5:12] -= 0.4
        later.elevation[20:27, 20:27] += 0.3
        report = compare_surfaces(_surface(), later, earlier_accuracy_m=0.02,
                                  later_accuracy_m=0.02, registration_residual_m=0.01)
        assert {r.direction for r in report.regions} == {"subsidence", "uplift"}

    def test_regions_are_ordered_worst_first(self) -> None:
        later = _surface()
        later.elevation[3:10, 3:10] -= 0.2
        later.elevation[18:26, 18:26] -= 0.9
        report = compare_surfaces(_surface(), later, earlier_accuracy_m=0.02,
                                  later_accuracy_m=0.02, registration_residual_m=0.01)
        assert report.regions[0].max_displacement_m > report.regions[1].max_displacement_m


class TestTheFloorIsEnforced:
    def test_movement_under_the_floor_is_not_reported(self) -> None:
        # 30 mm of real movement against a 30 mm floor: genuinely there, genuinely not
        # resolvable, and reporting it would be indistinguishable from noise.
        later = _surface()
        later.elevation[10:20, 10:20] -= 0.03
        report = compare_surfaces(_surface(), later, earlier_accuracy_m=0.02,
                                  later_accuracy_m=0.02, registration_residual_m=0.01)
        assert report.regions == []

    def test_the_same_movement_is_found_with_better_surveys(self) -> None:
        # The floor is a property of the surveys, not of the ground. Same displacement,
        # tighter accuracies, now resolvable.
        later = _surface()
        later.elevation[10:20, 10:20] -= 0.03
        report = compare_surfaces(_surface(), later, earlier_accuracy_m=0.005,
                                  later_accuracy_m=0.005, registration_residual_m=0.002)
        assert len(report.regions) == 1

    def test_the_note_says_absence_is_not_proof(self) -> None:
        report = compare_surfaces(_surface(), _surface(), earlier_accuracy_m=0.03,
                                  later_accuracy_m=0.03, registration_residual_m=0.01)
        assert "not that none occurred" in report.note
