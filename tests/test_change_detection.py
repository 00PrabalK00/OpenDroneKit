"""Comparing two surveys of the same asset.

Volumes are asserted against surfaces whose difference is known analytically. The
defect-matching tests pin down the conservative behaviour: a defect that moved beyond
the match radius must be reported as one resolved and one new, never as the same
defect having grown, because asserting continuity that was not established would
fabricate a history for the asset.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import geo
from core.change_detection import (
    IncomparableSurveys,
    compare_defects,
    compare_surfaces,
    compare_surveys,
)

PIXEL = 0.5
GRID = 80
GROUND = 100.0
ORIGIN_X, ORIGIN_Y = 500000.0, 4570000.0


def write_surface(path, array, epsg=32617, pixel=PIXEL):
    geo.write_geotiff(path, array.astype(np.float32), epsg=epsg, west=ORIGIN_X,
                      north=ORIGIN_Y, pixel_size=pixel, cog=False)
    return path


@pytest.fixture
def flat(tmp_path):
    return write_surface(tmp_path / "before.tif", np.full((GRID, GRID), GROUND))


class TestSurfaceComparison:
    def test_material_added_is_measured_exactly(self, tmp_path, flat):
        """A 10 m x 10 m pile 2 m high is 200 m3, whatever the code thinks."""
        after = np.full((GRID, GRID), GROUND)
        start, cells, height = 30, 20, 2.0
        after[start:start + cells, start:start + cells] = GROUND + height
        later = write_surface(tmp_path / "after.tif", after)

        change = compare_surfaces(flat, later)
        expected = (cells * PIXEL) ** 2 * height
        assert change.added_volume_m3 == pytest.approx(expected, abs=1e-6)
        assert change.removed_volume_m3 == pytest.approx(0.0, abs=1e-9)
        assert change.net_volume_m3 == pytest.approx(expected, abs=1e-6)

    def test_material_removed_is_measured_exactly(self, tmp_path, flat):
        after = np.full((GRID, GRID), GROUND)
        after[30:50, 30:50] = GROUND - 2.0
        later = write_surface(tmp_path / "after.tif", after)

        change = compare_surfaces(flat, later)
        expected = (20 * PIXEL) ** 2 * 2.0
        assert change.removed_volume_m3 == pytest.approx(expected, abs=1e-6)
        assert change.net_volume_m3 == pytest.approx(-expected, abs=1e-6)

    def test_cut_and_fill_together_net_correctly(self, tmp_path, flat):
        after = np.full((GRID, GRID), GROUND)
        after[10:30, 10:30] = GROUND + 1.0   # fill
        after[50:70, 50:70] = GROUND - 1.0   # cut
        later = write_surface(tmp_path / "after.tif", after)

        change = compare_surfaces(flat, later)
        each = (20 * PIXEL) ** 2 * 1.0
        assert change.added_volume_m3 == pytest.approx(each, abs=1e-6)
        assert change.removed_volume_m3 == pytest.approx(each, abs=1e-6)
        assert change.net_volume_m3 == pytest.approx(0.0, abs=1e-6)

    def test_identical_surveys_report_no_change(self, tmp_path, flat):
        same = write_surface(tmp_path / "same.tif", np.full((GRID, GRID), GROUND))
        change = compare_surfaces(flat, same)
        assert change.net_volume_m3 == pytest.approx(0.0, abs=1e-9)
        assert change.changed_area_m2 == pytest.approx(0.0)

    def test_max_rise_and_fall_are_reported(self, tmp_path, flat):
        after = np.full((GRID, GRID), GROUND)
        after[10, 10] = GROUND + 3.5
        after[20, 20] = GROUND - 1.25
        later = write_surface(tmp_path / "after.tif", after)

        change = compare_surfaces(flat, later)
        assert change.max_rise_m == pytest.approx(3.5)
        assert change.max_fall_m == pytest.approx(1.25)

    def test_a_difference_raster_is_written_with_the_survey_crs(self, tmp_path, flat):
        after = np.full((GRID, GRID), GROUND)
        after[30:50, 30:50] = GROUND + 1.0
        later = write_surface(tmp_path / "after.tif", after)

        change = compare_surfaces(flat, later, output_path=tmp_path / "diff.tif")
        assert change.difference_path
        _, meta = geo.read_geotiff(change.difference_path)
        assert meta["epsg"] == 32617


class TestIncomparableSurveys:
    def test_different_coordinate_systems_are_refused(self, tmp_path, flat):
        """Silently resampling would produce a difference map that looks valid."""
        other = write_surface(tmp_path / "utm18.tif", np.full((GRID, GRID), GROUND),
                              epsg=32618)
        with pytest.raises(IncomparableSurveys, match="coordinate systems"):
            compare_surfaces(flat, other)

    def test_different_shapes_are_refused(self, tmp_path, flat):
        smaller = write_surface(tmp_path / "small.tif", np.full((40, 40), GROUND))
        with pytest.raises(IncomparableSurveys, match="shapes"):
            compare_surfaces(flat, smaller)

    def test_different_resolutions_are_refused(self, tmp_path, flat):
        coarse = write_surface(tmp_path / "coarse.tif", np.full((GRID, GRID), GROUND),
                               pixel=1.0)
        with pytest.raises(IncomparableSurveys, match="resolution"):
            compare_surfaces(flat, coarse)

    def test_surveys_with_no_overlapping_data_are_refused(self, tmp_path):
        empty = np.full((GRID, GRID), np.nan)
        first = write_surface(tmp_path / "a.tif", empty)
        second = write_surface(tmp_path / "b.tif", empty)
        with pytest.raises(IncomparableSurveys, match="no cell"):
            compare_surfaces(first, second)


def layer(*defects):
    """Build a defect GeoJSON layer from (type, lon, lat, area) tuples."""
    features = [
        geo.point_feature(lon, lat, {
            "defect_id": f"d{index}", "defect_type": kind, "area_m2": area,
        })
        for index, (kind, lon, lat, area) in enumerate(defects)
    ]
    return {"type": "FeatureCollection", "features": features}


class TestDefectComparison:
    def test_a_defect_present_only_later_is_new(self):
        change = compare_defects(layer(), layer(("crack", -81.75, 41.30, 0.4)))
        assert len(change.new) == 1
        assert change.resolved == []

    def test_a_defect_present_only_earlier_is_resolved(self):
        change = compare_defects(layer(("crack", -81.75, 41.30, 0.4)), layer())
        assert len(change.resolved) == 1
        assert change.new == []

    def test_a_defect_in_the_same_place_that_enlarged_is_grown(self):
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.40)),
            layer(("crack", -81.75, 41.30, 0.80)),
        )
        assert len(change.grown) == 1
        assert change.grown[0]["area_delta_m2"] == pytest.approx(0.40)

    def test_a_defect_that_shrank_is_reported_as_such(self):
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.80)),
            layer(("crack", -81.75, 41.30, 0.40)),
        )
        assert len(change.shrunk) == 1

    def test_a_small_area_difference_is_treated_as_unchanged(self):
        """Segmentation noise must not be reported as the defect growing."""
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.400)),
            layer(("crack", -81.75, 41.30, 0.415)),
        )
        assert len(change.unchanged) == 1
        assert change.grown == []

    def test_a_defect_beyond_the_match_radius_is_not_claimed_to_have_moved(self):
        """Asserting continuity that was not established would fabricate a history."""
        # ~30 m apart, well beyond the 2 m radius.
        change = compare_defects(
            layer(("crack", -81.7500, 41.3000, 0.4)),
            layer(("crack", -81.7500, 41.3003, 0.4)),
        )
        assert len(change.resolved) == 1
        assert len(change.new) == 1
        assert change.grown == [] and change.unchanged == []

    def test_a_different_defect_type_is_never_matched(self):
        """A crack becoming a spall is two findings, not one that changed."""
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.4)),
            layer(("spalling", -81.75, 41.30, 0.4)),
        )
        assert len(change.resolved) == 1
        assert len(change.new) == 1

    def test_each_defect_matches_at_most_once(self):
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.4)),
            layer(("crack", -81.75, 41.30, 0.4), ("crack", -81.750005, 41.30, 0.4)),
        )
        counts = change.to_dict()["counts"]
        assert counts["new"] == 1
        assert counts["grown"] + counts["shrunk"] + counts["unchanged"] == 1

    def test_unmeasured_defects_do_not_produce_a_growth_claim(self):
        change = compare_defects(
            layer(("crack", -81.75, 41.30, 0.0)),
            layer(("crack", -81.75, 41.30, 0.0)),
        )
        assert len(change.unchanged) == 1
        assert "undetermined" in change.unchanged[0]["note"]

    def test_the_method_is_stated_in_the_result(self):
        change = compare_defects(layer(), layer())
        assert "match radius" in change.to_dict()["method"]


class TestCombinedComparison:
    def test_missing_inputs_are_reported_not_silently_skipped(self):
        result = compare_surveys()
        assert result["surface"] is None
        assert result["defects"] is None
        assert len(result["warnings"]) == 2

    def test_a_surface_failure_does_not_lose_the_defect_comparison(self, tmp_path, flat):
        other = write_surface(tmp_path / "utm18.tif", np.full((GRID, GRID), GROUND),
                              epsg=32618)
        result = compare_surveys(
            earlier_dsm=flat, later_dsm=other,
            earlier_defects=layer(), later_defects=layer(("crack", -81.75, 41.30, 0.4)),
        )
        assert result["surface"] is None
        assert any("Surface comparison unavailable" in w for w in result["warnings"])
        assert result["defects"]["counts"]["new"] == 1
