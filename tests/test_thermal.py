"""Radiometric thermal conversion.

Temperatures are asserted by round trip: a known temperature is converted to raw
counts using the camera model, then read back through the full pipeline and required
to match within 0.1 K. That catches an inverted formula, which would still produce
plausible-looking numbers.

The refusal tests matter as much. A thermal JPEG's pixels are a palette; inferring
temperature from them would produce a number indistinguishable from a measurement.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.thermal import (
    ABSOLUTE_ZERO_C,
    Calibration,
    NotRadiometric,
    kelvin_to_raw,
    load_radiometric,
    load_radiometric_file,
    raw_to_celsius,
    raw_to_kelvin,
)

# Representative FLIR-family constants.
METADATA = {
    "PlanckR1": 17096.453, "PlanckR2": 0.046642166, "PlanckB": 1428.0,
    "PlanckF": 1.0, "PlanckO": -342.0,
    "Emissivity": 1.0, "ReflectedApparentTemperature": 20.0,
}

CALIBRATION = Calibration.from_metadata(METADATA)


def counts_for(celsius: float, calibration: Calibration = CALIBRATION) -> float:
    """Raw counts a camera would record for a black body at this temperature."""
    return float(kelvin_to_raw(celsius - ABSOLUTE_ZERO_C, calibration))


class TestCalibration:
    def test_constants_are_read(self):
        assert CALIBRATION.b == pytest.approx(1428.0)
        assert CALIBRATION.o == pytest.approx(-342.0)

    @pytest.mark.parametrize("missing", ["PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO"])
    def test_a_missing_constant_is_refused(self, missing):
        """Defaulting one would silently change every temperature in the image."""
        metadata = {k: v for k, v in METADATA.items() if k != missing}
        with pytest.raises(NotRadiometric, match=missing):
            Calibration.from_metadata(metadata)

    def test_the_error_names_every_missing_constant(self):
        with pytest.raises(NotRadiometric) as info:
            Calibration.from_metadata({"Emissivity": 0.95})
        message = str(info.value)
        for name in ("PlanckR1", "PlanckR2", "PlanckB"):
            assert name in message


class TestConversion:
    @pytest.mark.parametrize("celsius", [-20.0, 0.0, 20.0, 37.5, 100.0, 250.0])
    def test_temperatures_round_trip_within_a_tenth_of_a_kelvin(self, celsius):
        """Catches an inverted formula, which would still look plausible."""
        raw = counts_for(celsius)
        recovered = float(raw_to_celsius(np.array([[raw]]), CALIBRATION)[0, 0])
        assert recovered == pytest.approx(celsius, abs=0.1)

    def test_kelvin_conversion_is_offset_from_celsius_correctly(self):
        raw = counts_for(0.0)
        kelvin = float(raw_to_kelvin(np.array([[raw]]), CALIBRATION)[0, 0])
        assert kelvin == pytest.approx(273.15, abs=0.1)

    def test_hotter_scenes_give_higher_counts(self):
        assert counts_for(100.0) > counts_for(20.0) > counts_for(-10.0)

    def test_an_impossible_count_becomes_nan_not_a_wrapped_value(self):
        """A pixel with no physical solution must not read as a real temperature."""
        # Drives the log argument non-positive.
        impossible = np.array([[-1e9]])
        result = raw_to_celsius(impossible, CALIBRATION)
        assert np.isnan(result).all()

    def test_a_whole_image_converts(self):
        raw = np.full((8, 8), counts_for(25.0))
        celsius = raw_to_celsius(raw, CALIBRATION)
        assert celsius.shape == (8, 8)
        assert np.allclose(celsius, 25.0, atol=0.1)


class TestEmissivity:
    def test_a_low_emissivity_surface_reads_hotter_after_correction(self):
        """A shiny surface also reflects its surroundings; ignoring that reads it cold.

        With the scene cooler than the target, removing the reflected component must
        raise the reported surface temperature.
        """
        metadata = dict(METADATA, Emissivity=0.6, ReflectedApparentTemperature=10.0)
        low_emissivity = Calibration.from_metadata(metadata)

        raw = counts_for(80.0)
        uncorrected = float(raw_to_celsius(np.array([[raw]]), CALIBRATION)[0, 0])
        corrected = float(raw_to_celsius(np.array([[raw]]), low_emissivity)[0, 0])
        assert corrected > uncorrected

    def test_unit_emissivity_leaves_the_reading_unchanged(self):
        """A black body reflects nothing, so the correction must be a no-op."""
        raw = counts_for(50.0)
        black_body = Calibration.from_metadata(dict(METADATA, Emissivity=1.0))
        assert float(raw_to_celsius(np.array([[raw]]), black_body)[0, 0]) == pytest.approx(
            50.0, abs=0.1
        )

    def test_reflected_temperature_moves_the_result(self):
        raw = counts_for(60.0)
        cold_room = Calibration.from_metadata(
            dict(METADATA, Emissivity=0.7, ReflectedApparentTemperature=0.0))
        warm_room = Calibration.from_metadata(
            dict(METADATA, Emissivity=0.7, ReflectedApparentTemperature=40.0))
        assert float(raw_to_celsius(np.array([[raw]]), cold_room)[0, 0]) > \
               float(raw_to_celsius(np.array([[raw]]), warm_room)[0, 0])


class TestThermalImage:
    @pytest.fixture
    def image(self):
        raw = np.full((10, 10), counts_for(20.0))
        raw[2:4, 2:4] = counts_for(65.0)  # a hot patch
        return load_radiometric(raw, METADATA)

    def test_statistics_are_in_celsius(self, image):
        stats = image.stats
        assert stats["min_c"] == pytest.approx(20.0, abs=0.1)
        assert stats["max_c"] == pytest.approx(65.0, abs=0.1)

    def test_point_temperature(self, image):
        assert image.point_temperature(2, 2) == pytest.approx(65.0, abs=0.1)
        assert image.point_temperature(9, 9) == pytest.approx(20.0, abs=0.1)

    def test_region_statistics(self, image):
        region = image.region_stats(2, 2, 3, 3)
        assert region["pixels"] == 4
        assert region["mean_c"] == pytest.approx(65.0, abs=0.1)

    def test_anomalies_are_labelled_as_candidates_not_faults(self, image):
        """A sunlit surface is legitimately hotter; the result must not claim a fault."""
        anomalies = image.anomalies(sigma=2.0)
        assert anomalies["hot_pixels"] > 0
        assert "not confirmed faults" in anomalies["note"]

    def test_an_isothermal_scene_reports_no_anomaly_and_says_why(self):
        flat = load_radiometric(np.full((6, 6), counts_for(30.0)), METADATA)
        anomalies = flat.anomalies()
        assert anomalies["hot_pixels"] == 0
        assert "isothermal" in anomalies["note"]


class TestRefusal:
    def test_a_thermal_jpeg_is_refused_rather_than_read_from_its_palette(self, tmp_path):
        """The rendered pixels are a colour ramp; a temperature read from them is invented."""
        fake = tmp_path / "DJI_0001_T.jpg"
        fake.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")

        with pytest.raises(NotRadiometric, match="palette"):
            load_radiometric_file(fake)

    def test_the_refusal_explains_how_to_extract_the_data(self, tmp_path):
        fake = tmp_path / "FLIR0001.jpg"
        fake.write_bytes(b"\xff\xd8")
        with pytest.raises(NotRadiometric, match="exiftool"):
            load_radiometric_file(fake)

    def test_a_sidecar_json_is_accepted(self, tmp_path):
        sidecar = tmp_path / "frame.json"
        sidecar.write_text(json.dumps({
            "raw": [[counts_for(30.0), counts_for(31.0)]],
            "metadata": METADATA,
        }), encoding="utf-8")

        image = load_radiometric_file(sidecar)
        assert image.celsius.shape == (1, 2)
        assert image.point_temperature(0, 0) == pytest.approx(30.0, abs=0.1)


class TestGeoreferencedOutput:
    def test_the_raster_holds_temperatures_in_a_stated_crs(self, tmp_path):
        pytest.importorskip("rasterio")
        from core import geo
        from core.thermal import write_thermal_geotiff

        image = load_radiometric(np.full((8, 8), counts_for(42.0)), METADATA)
        path = write_thermal_geotiff(
            image, tmp_path / "thermal.tif",
            epsg=32617, west=500000.0, north=4570000.0, pixel_size=0.5,
        )

        data, meta = geo.read_geotiff(path)
        assert meta["epsg"] == 32617
        # Values are degrees Celsius, not palette indices.
        assert float(np.nanmean(data[0])) == pytest.approx(42.0, abs=0.1)
