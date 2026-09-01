"""Choosing the temperature range a thermal image is drawn in.

This decision changes what an inspector sees more than any other step in the thermal
workflow, and it fails in two opposite ways:

  Full range   One 80 C exhaust stack stretches the scale until the 3 C delta across a wet
               roof -- the thing being looked for -- renders as flat grey.
  Tight range  A range that shows the 3 C delta clips the stack, so a dangerous hotspot is
               drawn the same colour as a merely warm one.

The second is the one these tests are mostly about, because the image looks fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.thermal_scaling import (
    PALETTES,
    ScalingRefused,
    anomaly_scale,
    auto_scale,
    legend,
    manual_scale,
    render,
)


@pytest.fixture
def roof() -> np.ndarray:
    """A wet roof: 20 C surface with a 3 C damp patch, and one 80 C flue."""
    field = np.full((40, 40), 20.0)
    field[10:20, 10:20] = 23.0
    field[0, 0] = 80.0
    return field


class TestSayingWhatARangeHides:
    def test_a_range_below_the_hottest_pixel_is_reported(self, roof) -> None:
        """The dangerous case: a severe hotspot drawn the same colour as a mild one, in
        an image that looks perfectly reasonable."""
        scale = manual_scale(roof, 18.0, 25.0)
        assert scale.hides_the_hottest() is True
        assert scale.clipped_high == 1
        assert "80.0 C" in scale.warning()

    def test_a_range_covering_everything_hides_nothing(self, roof) -> None:
        scale = manual_scale(roof, 15.0, 85.0)
        assert scale.hides_the_hottest() is False
        assert scale.warning() == ""

    def test_cold_clipping_is_reported_too(self) -> None:
        field = np.full((10, 10), 20.0)
        field[0, 0] = -5.0
        scale = manual_scale(field, 10.0, 30.0)
        assert scale.clipped_low == 1
        assert "-5.0 C" in scale.warning()

    def test_the_true_extremes_are_always_carried(self, roof) -> None:
        """An inspector who cannot see what the range excluded is looking at a picture,
        not inspecting."""
        scale = manual_scale(roof, 18.0, 25.0)
        assert scale.data_max_c == pytest.approx(80.0)
        assert scale.data_min_c == pytest.approx(20.0)

    def test_the_clipped_fraction_is_reported(self, roof) -> None:
        scale = manual_scale(roof, 18.0, 25.0)
        assert scale.clipped_fraction == pytest.approx(1 / 1600, abs=1e-9)


class TestRefusingARangeThatRendersNothing:
    def test_an_inverted_range_is_refused(self, roof) -> None:
        with pytest.raises(ScalingRefused, match="above the bottom"):
            manual_scale(roof, 30.0, 10.0)

    def test_a_zero_span_is_refused(self, roof) -> None:
        """It renders one flat colour, which looks like a broken sensor."""
        with pytest.raises(ScalingRefused):
            manual_scale(roof, 20.0, 20.0)

    def test_a_non_finite_bound_is_refused(self, roof) -> None:
        with pytest.raises(ScalingRefused):
            manual_scale(roof, float("nan"), 30.0)

    def test_an_unknown_palette_names_the_real_ones(self, roof) -> None:
        with pytest.raises(ScalingRefused, match="ironbow"):
            manual_scale(roof, 10.0, 30.0, palette="rainbow")

    def test_a_frame_with_no_temperatures_is_refused(self) -> None:
        with pytest.raises(ScalingRefused):
            manual_scale(np.full((4, 4), np.nan), 10.0, 30.0)


class TestAutoScale:
    def test_one_extreme_pixel_does_not_flatten_the_scene(self, roof) -> None:
        """The whole reason for percentiles. Against min/max the 80 C flue compresses a
        3 C roof delta into almost nothing."""
        auto = auto_scale(roof)
        full = manual_scale(roof, roof.min(), roof.max())
        assert auto.span_c < full.span_c / 5

    def test_it_still_reports_what_it_trimmed(self, roof) -> None:
        """Trimming quietly would be the same failure as a bad manual range."""
        auto = auto_scale(roof)
        assert auto.clipped_high >= 1
        assert auto.warning() != ""

    def test_the_method_is_recorded(self, roof) -> None:
        assert "percentile" in auto_scale(roof, 2.0, 98.0).method

    def test_a_uniform_field_still_renders(self) -> None:
        """Every percentile lands on the same value; refusing would mean a flat wall
        cannot be displayed at all."""
        scale = auto_scale(np.full((10, 10), 20.0))
        assert scale.span_c > 0

    @pytest.mark.parametrize("low,high", [(-1.0, 99.0), (50.0, 50.0), (60.0, 40.0), (0.0, 101.0)])
    def test_impossible_percentiles_are_refused(self, roof, low, high) -> None:
        with pytest.raises(ScalingRefused):
            auto_scale(roof, low, high)


class TestAnomalyScale:
    def test_it_centres_on_the_scene(self, roof) -> None:
        scale = anomaly_scale(roof, sigma=2.0)
        midpoint = (scale.min_c + scale.max_c) / 2
        assert midpoint == pytest.approx(float(roof.mean()), abs=0.01)

    def test_a_wider_sigma_covers_more(self, roof) -> None:
        assert anomaly_scale(roof, 3.0).span_c > anomaly_scale(roof, 1.0).span_c

    def test_a_flat_field_does_not_collapse(self) -> None:
        assert anomaly_scale(np.full((8, 8), 20.0)).span_c > 0

    def test_a_non_positive_sigma_is_refused(self, roof) -> None:
        with pytest.raises(ScalingRefused):
            anomaly_scale(roof, sigma=0.0)


class TestRendering:
    def test_it_produces_an_rgb_image(self, roof) -> None:
        image = render(roof, manual_scale(roof, 15.0, 85.0))
        assert image.shape == (40, 40, 3)
        assert image.dtype == np.uint8

    def test_hotter_renders_brighter(self, roof) -> None:
        """Ironbow runs dark to light. A palette that inverted that would make every
        thermal image in every report read backwards."""
        image = render(roof, manual_scale(roof, 15.0, 85.0))
        assert image[0, 0].sum() > image[30, 30].sum()

    def test_a_dead_pixel_is_not_drawn_as_the_coldest_thing(self) -> None:
        """Mapping NaN to the bottom of the scale draws a broken sensor element as the
        coldest area in the frame, which is exactly what a thermal inspection looks for."""
        field = np.full((4, 4), 20.0)
        field[0, 0] = np.nan
        image = render(field, manual_scale(field, 10.0, 30.0))
        assert tuple(image[0, 0]) == (0, 0, 0)

    @pytest.mark.parametrize("palette", sorted(PALETTES))
    def test_every_palette_renders(self, roof, palette) -> None:
        assert render(roof, manual_scale(roof, 15.0, 85.0, palette)).shape == (40, 40, 3)

    def test_values_outside_the_range_saturate_rather_than_wrap(self, roof) -> None:
        """Wrapping would draw the hottest pixel as the coldest colour."""
        image = render(roof, manual_scale(roof, 18.0, 25.0))
        top = render(np.full((1, 1), 25.0), manual_scale(roof, 18.0, 25.0))
        assert tuple(image[0, 0]) == tuple(top[0, 0])


class TestTheLegend:
    def test_it_spans_the_range(self, roof) -> None:
        ticks = legend(manual_scale(roof, 10.0, 30.0), steps=5)
        assert ticks[0]["celsius"] == pytest.approx(10.0)
        assert ticks[-1]["celsius"] == pytest.approx(30.0)
        assert len(ticks) == 5

    def test_one_tick_is_refused(self, roof) -> None:
        with pytest.raises(ScalingRefused):
            legend(manual_scale(roof, 10.0, 30.0), steps=1)


class TestTheApiPathActuallyWorks:
    """The wrapper, not the arithmetic.

    Four features in this codebase passed their unit tests while being unreachable, so a
    module with thirty-one green tests and no exercised route to it is not finished.
    """

    @pytest.fixture
    def api(self, monkeypatch, roof):
        from app.api import Api
        from app.session import AppSession
        from core.thermal import Calibration, ThermalImage

        # Real Planck constants are required and cannot be defaulted, so the fixture
        # supplies plausible ones; nothing here converts counts, it only scales celsius.
        calibration = Calibration(r1=17096.0, r2=0.0406, b=1428.0, f=1.0, o=-56.0)
        frame = ThermalImage(celsius=roof, calibration=calibration, source="test.tif")
        monkeypatch.setattr("core.thermal.load_radiometric_file", lambda _p: frame)
        return Api(AppSession())

    def test_auto_mode_returns_a_scale_and_a_legend(self, api) -> None:
        result = api.scale_thermal("any.tif", mode="auto")
        assert result["ok"], result.get("error")
        assert result["scale"]["span_c"] > 0
        assert len(result["legend"]) >= 2

    def test_a_clipping_range_warns_through_the_api(self, api) -> None:
        """The warning is the feature. If it does not survive the wrapper, the operator
        never sees that the hottest area was saturated."""
        result = api.scale_thermal("any.tif", mode="manual", min_c=18.0, max_c=25.0)
        assert result["ok"]
        assert result["scale"]["hides_the_hottest"] is True
        assert "80.0 C" in result["warning"]

    def test_manual_mode_needs_both_bounds(self, api) -> None:
        assert not api.scale_thermal("any.tif", mode="manual", min_c=10.0)["ok"]

    def test_an_unknown_mode_names_the_real_ones(self, api) -> None:
        error = api.scale_thermal("any.tif", mode="rainbow")["error"]
        assert "auto" in error and "manual" in error and "anomaly" in error

    def test_an_inverted_range_is_refused_through_the_api(self, api) -> None:
        result = api.scale_thermal("any.tif", mode="manual", min_c=30.0, max_c=10.0)
        assert not result["ok"]
        assert "above the bottom" in result["error"]

    def test_anomaly_mode_reaches_the_scaler(self, api) -> None:
        result = api.scale_thermal("any.tif", mode="anomaly", sigma=2.0)
        assert result["ok"]
        assert "sigma" in result["scale"]["method"]
