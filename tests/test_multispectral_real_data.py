"""MicaSense multispectral on real captures: decoding, calibration, alignment, indices.

Data: ``ODK_TESTDATA_DIR`` (see odk-testdata/INDEX.md) holding
``multispectral_micasense_panels`` (MIT, MicaSense sample panel + in-flight captures)
and ``multispectral_buergewald_rededgep`` (CC0, 24 RedEdge-P survey captures). Every
test skips with the reason when that data is absent; nothing here is synthesised.

The band orthomosaic needs a full reconstruction, which is too heavy for a test run.
``tools/run_multispectral.py`` writes it; the orthomosaic tests check that real output
when ``ODK_MULTISPECTRAL_ORTHO`` (or the default output location) points at it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from core import multispectral as ms


def _testdata() -> Path | None:
    root = os.environ.get("ODK_TESTDATA_DIR", "").strip()
    return Path(root) if root else None


def _dataset(name: str) -> Path:
    root = _testdata()
    if root is None:
        pytest.skip("ODK_TESTDATA_DIR is not set; real multispectral captures are required.")
    path = root / name
    if not path.is_dir():
        pytest.skip(f"{path} is missing; see odk-testdata/INDEX.md for the dataset.")
    return path


@pytest.fixture(scope="module")
def rededge_p_panel():
    return ms.group_captures(_dataset("multispectral_micasense_panels") / "REDEDGE-P")


@pytest.fixture(scope="module")
def altum():
    return ms.group_captures(_dataset("multispectral_micasense_panels") / "ALTUM")


@pytest.fixture(scope="module")
def buergewald():
    return ms.group_captures(_dataset("multispectral_buergewald_rededgep") / "images")


class TestReading:
    def test_12bit_packed_rededge_p_band_decodes_like_gdal(self, rededge_p_panel):
        import warnings

        import rasterio

        band = rededge_p_panel[0].bands["nir"]
        assert band.bits_per_pixel == 12
        pixels = ms.read_band_pixels(band.path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(band.path) as dataset:
                reference = dataset.read(1)
        assert np.array_equal(pixels, reference)
        assert pixels.max() < 4096  # 12-bit values, not misread 16-bit garbage

    def test_survey_groups_into_24_captures_of_six_bands(self, buergewald):
        assert len(buergewald) == 24
        for capture in buergewald:
            assert set(capture.bands) == {"blue", "green", "red", "nir", "red_edge", "panchro"}
            assert capture.reflectance_keys == ["blue", "green", "red", "nir", "red_edge"]

    def test_xmp_radiometry_and_dls_are_read(self, buergewald):
        nir = buergewald[0].bands["nir"]
        assert nir.center_wavelength_nm == 842
        assert len(nir.radiometric_cal) == 3
        assert nir.black_level > 0 and nir.exposure_s > 0 and nir.gain > 0
        irradiance, how = ms.dls_irradiance(nir)
        assert "HorizontalIrradiance" in how
        assert 0.01 < irradiance < 3.0  # W/m^2/nm, after the DLS2 0.01 scale
        assert nir.latitude == pytest.approx(50.864, abs=0.01)


    def test_multisensor_ndvi_uses_reflectance_and_labels_raw_counts(self, buergewald, tmp_path):
        from core.multisensor import _frame_ndvi

        capture = buergewald[0]
        ndvi, basis, _ = _frame_ndvi(capture.bands["red"].path, capture.bands["nir"].path)
        assert basis == "surface_reflectance"
        assert 0.3 < float(np.nanmean(ndvi)) < 0.95
        import cv2

        plain = tmp_path / "plain.png"
        cv2.imwrite(str(plain), np.full((8, 8), 100, dtype=np.uint8))
        _, basis, note = _frame_ndvi(str(plain), str(plain))
        assert basis == "uncalibrated_dn_ratio" and "not reflectance" in note


class TestPanelCalibration:
    def test_panel_albedo_comes_from_the_capture_not_a_constant(self, rededge_p_panel):
        factors = ms.panel_calibration(rededge_p_panel[0])
        assert set(factors) >= {"blue", "green", "red", "nir", "red_edge"}
        for key, cal in factors.items():
            band = rededge_p_panel[0].bands[key]
            assert cal.details["albedo"] == band.panel_albedo
            assert cal.details["panel_serial"] == "RP06-2051037-OB"
            assert cal.details["saturated_fraction"] < 0.01

    def test_dls_reflectance_of_the_panel_is_measured_against_its_albedo(self, rededge_p_panel):
        report = ms.validate_panel_against_albedo(rededge_p_panel[0])
        rows = report["bands"]
        assert all(rows[k]["status"] == "measured" for k in ("blue", "green", "red", "nir", "red_edge"))
        # Sun-sensor calibration on this RedEdge-P flight reads the 48-49 % panel
        # within 2 % in NIR and within 20 % in every band (measured 2026-10-02:
        # blue -17.7 %, green -4.4 %, red +6.2 %, NIR +0.6 %, red edge +11.2 %).
        assert abs(rows["nir"]["relative_error"]) < 0.02
        for key in ("blue", "green", "red", "red_edge"):
            assert abs(rows[key]["relative_error"]) < 0.20
        assert report["mean_absolute_relative_error"] < 0.12

    def test_altum_band_without_a_panel_reading_is_not_filled_in(self, altum):
        panel = altum[0]
        report = ms.validate_panel_against_albedo(panel)
        assert report["bands"]["green"]["status"] == "unavailable"
        factors = ms.panel_calibration(panel)
        assert "green" not in factors  # no typical albedo substituted
        # An operator-supplied albedo replaces the QR value and is named as the source,
        # but cannot rescue green: the camera never located the panel in that band.
        factors = ms.panel_calibration(panel, {"green": (0.4877, "sheet"), "nir": (0.4850, "calibration sheet")})
        assert "green" not in factors
        assert factors["nir"].details["albedo"] == 0.4850
        assert "calibration sheet" in factors["nir"].source
        with pytest.raises(ms.MultispectralError):
            ms.panel_calibration(panel, {"nir": (0.49, "  ")})

    def test_a_panel_from_another_camera_is_refused(self, rededge_p_panel, buergewald, tmp_path):
        survey = _dataset("multispectral_buergewald_rededgep") / "images"
        with pytest.raises(ms.MultispectralError, match="panel calibrates only"):
            ms.process_survey(sorted(survey.glob("IMG_0483_*.tif")), tmp_path, calibration="panel",
                              panel_capture=rededge_p_panel[0], reconstruct=False)

    def test_altum_lwir_decodes_centikelvin_to_celsius(self, altum):
        for capture in altum:
            temps = ms.lwir_temperature_c(capture.bands["lwir"])
            assert temps.shape == (120, 160)
            # PROVENANCE.md: this sample spans 8-43 C.
            assert 5.0 < float(temps.min()) and float(temps.max()) < 45.0
        with pytest.raises(ms.MultispectralError):
            ms.radiance(altum[0].bands["lwir"])


class TestAlignment:
    def test_bands_register_to_the_reference(self, buergewald):
        capture = buergewald[0]
        aligned = ms.align_capture(capture, ms.dls_calibration(capture))
        assert aligned.reference_band == "green"
        assert aligned.reflectance.shape == (5, 1088, 1456)
        assert aligned.valid_mask().mean() > 0.9
        agreement = ms.band_to_band_agreement(aligned)
        # Gradient correlation with green after alignment (measured: blue 0.94,
        # red 0.57, NIR 0.60, red edge 0.80). Misregistration by the 10-30 px rig
        # offsets drops these towards zero.
        assert agreement["blue"] > 0.85
        assert agreement["red_edge"] > 0.6
        assert agreement["red"] > 0.4 and agreement["nir"] > 0.4
        for key, row in aligned.alignment.items():
            if key != "green":
                assert row["method"].startswith(("rig+ecc", "rig ("))

    def test_vegetation_reflectance_is_physically_plausible(self, buergewald):
        capture = buergewald[0]
        aligned = ms.align_capture(capture, ms.dls_calibration(capture))
        refl = {k: aligned.reflectance[i] for i, k in enumerate(aligned.band_keys)}
        valid = aligned.valid_mask()
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (refl["nir"] - refl["red"]) / (refl["nir"] + refl["red"])
        canopy = valid & (ndvi > 0.6)
        assert canopy.mean() > 0.05  # this field is visibly vegetated
        # Green vegetation: NIR plateau well above red, red absorbed.
        assert 0.2 < float(np.median(refl["nir"][canopy])) < 0.7
        assert float(np.median(refl["red"][canopy])) < 0.1


def _ortho_path() -> Path:
    override = os.environ.get("ODK_MULTISPECTRAL_ORTHO", "").strip()
    if override:
        path = Path(override)
    else:
        root = _testdata()
        if root is None:
            pytest.skip("ODK_TESTDATA_DIR is not set.")
        path = root / "_outputs" / "multispectral" / "buergewald_dls" / "reconstruction" / "orthomosaic_bands.tif"
    if not path.is_file():
        pytest.skip(f"No band orthomosaic at {path}; run tools/run_multispectral.py first.")
    return path


class TestBandOrthomosaic:
    def test_orthomosaic_holds_georeferenced_reflectance_bands(self):
        import rasterio

        with rasterio.open(_ortho_path()) as dataset:
            assert dataset.count == 5
            assert dataset.crs is not None and dataset.crs.is_projected
            assert [d for d in dataset.descriptions] == ["blue", "green", "red", "nir", "red_edge"]
            assert dataset.tags()["ODK_QUANTITY"] == "surface_reflectance"
            nir = dataset.read(4)
            red = dataset.read(3)
        valid = np.isfinite(nir) & np.isfinite(red)
        assert valid.mean() > 0.2
        assert 0.1 < float(np.median(nir[valid])) < 0.6

    def test_agriculture_pack_runs_on_the_orthomosaic(self, tmp_path):
        from core.india_agriculture import compute_vegetation_indices, create_stress_zones

        path = _ortho_path()
        calibrations = ms.reflectance_band_calibrations(
            ["blue", "green", "red", "nir", "red_edge"], "MicaSense radiance + DLS2 irradiance")
        result = compute_vegetation_indices(path, calibrations, tmp_path)
        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        assert summary["status"] == "complete"
        ndvi = summary["indices"]["NDVI"]
        ndre = summary["indices"]["NDRE"]
        assert ndvi["valid_pixel_count"] > 10_000
        assert ndvi["max"] > 0.7 and ndvi["min"] < 0.3  # vegetated rows and bare soil
        assert ndre["mean"] < ndvi["mean"]  # red edge sits between red and NIR
        stress = create_stress_zones(
            ndvi["path"], tmp_path / "stress", severe_below=0.3, moderate_below=0.5,
            validation_scope="UNVALIDATED test thresholds; RedEdge-P, Buergewald 2024-03-07",
        )
        zones = json.loads(Path(stress.summary_path).read_text(encoding="utf-8"))
        assert zones["validation_scope"].startswith("UNVALIDATED")
        total = sum(z["area_m2"] for z in zones["zones"].values())
        assert total > 0 and math.isfinite(total)
