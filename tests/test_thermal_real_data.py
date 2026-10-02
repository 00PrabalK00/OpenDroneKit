"""Thermal decoding against real, openly licensed camera files.

Data: ``ODK_TESTDATA_DIR`` points at the odk-testdata collection (see its INDEX.md and
each folder's PROVENANCE.md). The DJI R-JPEG tests also need the DJI Thermal SDK, which
OpenDroneKit does not ship: ``ODK_DJI_TSDK_DIR`` points at the folder the user
extracted it into. Each test skips with the reason when its input is absent; nothing
here is synthesised to stand in for a real file.

References used, and what each one proves:
* Boğa Çayı bridge (H20T): the authors' SDK-decoded float32 °C arrays. Proves the DJI
  path reproduces the SDK pixel for pixel. It is not an independent calibration.
* FLIR sample JPEGs: ExifTool 12.35 dumps (for parsed constants) and the measurement
  readouts the camera burned into its own rendered image (spot / box max-min-avg).
  Those readouts are FLIR firmware's own temperature computation, so they check the
  whole parse-and-convert chain against the manufacturer.
* THED-PV (FLIR Duo Pro R TLinear): conversion formula from FLIR's Duo Pro R User
  Guide. The only cross-check available is plausibility against the site's ambient
  air temperature; there is no reference thermometer reading of the panels.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.thermal import NotRadiometric, load_radiometric_file

ROOT = Path(os.environ.get("ODK_TESTDATA_DIR", "").strip() or "__unset__")
BRIDGE = ROOT / "bridge_inspection_h20t_boga_cayi"
SAMPLES = ROOT / "thermal_rjpeg_samples_thermal_parser"
FACADE = ROOT / "thermal_nvs_praschl_zenodo"
THEDPV = ROOT / "thermal_pv_thedpv_subset"
THEDPV_SESSION = THEDPV / "Raw Dataset" / "2024-12-21" / "12" / "20241221_120152"
DJI_CAMERAS = ["H20N", "H20T", "H30T", "M2EA", "M30T", "M3T", "M3TD", "M4T", "XTS"]
FLIR_FILES = ["FLIR", "FLIR_AX8", "FLIR_B60", "FLIR_E40", "FLIR_T640", "DJI_XT2", "DJI_XTR"]


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not found at {path}; set ODK_TESTDATA_DIR to the odk-testdata folder.")
    return path


def _need_sdk() -> None:
    from core.thermal_dji import DjiSdkUnavailable, load_sdk

    try:
        load_sdk()
    except DjiSdkUnavailable as exc:
        pytest.skip(f"DJI Thermal SDK unavailable: {str(exc)[:160]}")


def _sdk_dataset() -> Path:
    folder = os.environ.get("ODK_DJI_TSDK_DIR", "").strip()
    if not folder or not (Path(folder) / "dataset").is_dir():
        pytest.skip("ODK_DJI_TSDK_DIR does not point at an SDK folder with its dataset/ samples.")
    return Path(folder) / "dataset"


# --------------------------------------------------------------------------- DJI


class TestDjiThermalSdk:
    def test_bridge_h20t_matches_the_sdk_decoded_ground_truth(self):
        """Every bridge frame decodes to the authors' SDK float32 °C array."""
        _need_sdk()
        frames = sorted(glob.glob(str(_need(BRIDGE / "B1_Thermal", "Bridge thermal") / "*_T.JPG")))
        assert len(frames) == 81
        worst, means = 0.0, []
        for jpg in frames:
            image = load_radiometric_file(jpg)
            truth = np.fromfile(jpg[:-4] + ".raw", np.float32).reshape(512, 640)
            error = np.abs(image.celsius - truth.astype(np.float64))
            worst = max(worst, float(error.max()))
            means.append(float(error.mean()))
            assert image.provenance["decoder"] == "dji_thermal_sdk_dirp_measure_ex"
        # Measured 2026-10-02: max 0.0 °C, mean 0.0 °C over 81 frames (exact match).
        assert worst <= 1e-4, worst
        assert float(np.mean(means)) <= 1e-5

    @pytest.mark.parametrize("camera", DJI_CAMERAS)
    def test_every_sdk_sample_camera_decodes(self, camera):
        _need_sdk()
        files = sorted((_sdk_dataset() / camera).glob("*"))
        assert files, camera
        for path in files:
            image = load_radiometric_file(path)
            field = image.celsius
            assert field.ndim == 2 and field.shape[1] in (640, 1280)
            assert np.isfinite(field).all()
            stats = image.stats
            assert -60.0 < stats["min_c"] <= stats["max_c"] < 600.0
            assert image.provenance["unit"] == "celsius"
            assert "measurement_params" in image.provenance

    @pytest.mark.parametrize("camera", DJI_CAMERAS)
    def test_thermal_parser_dji_samples_decode(self, camera):
        _need_sdk()
        files = sorted((_need(SAMPLES, "thermal_parser samples") / camera).glob("*.[jJ][pP][gG]"))
        assert files
        for path in files:
            assert np.isfinite(load_radiometric_file(path).celsius).all()

    def test_facade_m30t_frames_decode(self):
        """Building-facade M30T survey: every frame is a DJI R-JPEG; every 10th is decoded."""
        _need_sdk()
        from core.thermal_dji import is_dji_rjpeg

        frames = sorted((_need(FACADE / "T", "Facade thermal")).glob("*.JPG"))
        assert len(frames) == 227 and all(is_dji_rjpeg(f) for f in frames)
        for path in frames[::10]:
            image = load_radiometric_file(path)
            assert image.celsius.shape == (512, 640)
            assert np.isfinite(image.celsius).all()
            assert image.provenance["measurement_params"]["emissivity"] == pytest.approx(0.95)

    def test_without_the_sdk_the_refusal_says_where_to_get_it(self, monkeypatch, tmp_path):
        from core import thermal_dji

        monkeypatch.setenv("ODK_DJI_TSDK_DIR", str(tmp_path))
        with pytest.raises(thermal_dji.DjiSdkUnavailable) as info:
            thermal_dji.load_sdk()
        message = str(info.value)
        assert thermal_dji.DJI_TSDK_URL in message
        assert "ODK_DJI_TSDK_DIR" in message and "palette" in message
        status = thermal_dji.sdk_status()
        assert status["available"] is False and status["download_url"] == thermal_dji.DJI_TSDK_URL

    def test_the_settings_value_points_the_app_at_the_sdk(self, monkeypatch, tmp_path):
        """With no env var, the folder saved through the Api is what loads the SDK."""
        _need_sdk()
        folder = os.environ["ODK_DJI_TSDK_DIR"]
        frame = _need(BRIDGE / "B1_Thermal" / "DJI_20220826113024_0100_T.JPG", "Bridge frame")
        from cockpit_harness import make_api

        api = make_api(tmp_path)
        monkeypatch.setattr("core.settings.settings_file", lambda: tmp_path / "settings.json")
        monkeypatch.delenv("ODK_DJI_TSDK_DIR")
        assert api.thermal_decoder_status()["dji_thermal_sdk"]["available"] is False
        assert api.set_dji_thermal_sdk_dir(str(tmp_path))["ok"] is False  # no libdirp there
        saved = api.set_dji_thermal_sdk_dir(folder)
        assert saved["ok"] and saved["dji_thermal_sdk"]["available"], saved
        assert saved["dji_thermal_sdk"]["configured_by"] == "setting dji_thermal_sdk_dir"
        assert api.scale_thermal(str(frame), "auto")["ok"]

    def test_a_dji_rjpeg_without_the_sdk_is_refused_not_guessed(self, monkeypatch, tmp_path):
        sample = _need(BRIDGE / "B1_Thermal" / "DJI_20220826113024_0100_T.JPG", "Bridge frame")
        from core.thermal_dji import DjiSdkUnavailable

        monkeypatch.setenv("ODK_DJI_TSDK_DIR", str(tmp_path))
        with pytest.raises(DjiSdkUnavailable):
            load_radiometric_file(sample)


# -------------------------------------------------------------------------- FLIR


def _exiftool_dump(name: str) -> dict[str, str]:
    text = (SAMPLES / f"{name}.txt").read_bytes().decode("utf-16")
    rows = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            rows[key.strip()] = value.strip()
    return rows


def _number(value: str) -> float:
    return float(re.match(r"-?[0-9.]+", value).group(0))


class TestFlirRadiometricJpeg:
    @pytest.mark.parametrize("name", FLIR_FILES)
    def test_parsed_constants_match_exiftool(self, name):
        from core.thermal_flir import read_flir_jpeg

        _need(SAMPLES / f"{name}.jpg", "FLIR sample")
        parsed = read_flir_jpeg(SAMPLES / f"{name}.jpg")
        dump = _exiftool_dump(name)
        m = parsed.metadata
        assert m["PlanckR1"] == pytest.approx(_number(dump["Planck R1"]), rel=1e-6)
        assert m["PlanckR2"] == pytest.approx(_number(dump["Planck R2"]), rel=1e-6)
        assert m["PlanckB"] == pytest.approx(_number(dump["Planck B"]), rel=1e-6)
        assert m["PlanckF"] == pytest.approx(_number(dump["Planck F"]), rel=1e-6)
        assert m["PlanckO"] == int(_number(dump["Planck O"]))
        assert m["Emissivity"] == pytest.approx(_number(dump["Emissivity"]), abs=1e-3)
        assert m["ObjectDistance"] == pytest.approx(_number(dump["Object Distance"]), abs=1e-2)
        assert m["ReflectedApparentTemperature"] == pytest.approx(
            _number(dump["Reflected Apparent Temperature"]), abs=0.05)
        assert m["AtmosphericTemperature"] == pytest.approx(
            _number(dump["Atmospheric Temperature"]), abs=0.05)
        assert m["RelativeHumidity"] * (1 if m["RelativeHumidity"] > 1 else 100) == pytest.approx(
            _number(dump["Relative Humidity"]), abs=0.05)
        assert m["IRWindowTransmission"] == pytest.approx(
            _number(dump["IR Window Transmission"]), abs=1e-3)
        height, width = parsed.counts.shape
        assert width == int(dump["Raw Thermal Image Width"])
        assert height == int(dump["Raw Thermal Image Height"])

    @pytest.mark.parametrize("name", FLIR_FILES)
    def test_every_flir_file_decodes_through_the_app_path(self, name):
        image = load_radiometric_file(_need(SAMPLES / f"{name}.jpg", "FLIR sample"))
        assert image.provenance["decoder"] == "flir_fff_cleanroom_planck"
        assert "NOT been compared" in image.provenance["validation"]
        assert np.isfinite(image.celsius).mean() > 0.99

    def test_e40_spot_matches_the_cameras_on_screen_reading(self):
        """The E40 burned "Spot 20.9 °C" into its image; Meas1 is at raw pixel (80, 60)."""
        image = load_radiometric_file(_need(SAMPLES / "FLIR_E40.jpg", "FLIR E40"))
        # Measured 2026-10-02: 20.92 °C.
        assert image.celsius[60, 80] == pytest.approx(20.9, abs=0.1)

    def test_b60_centre_spot_matches_the_cameras_on_screen_reading(self):
        """The B60 shows a centre spot of 19.0 °F (the cold sky makes the scene minimum low)."""
        image = load_radiometric_file(_need(SAMPLES / "FLIR_B60.jpg", "FLIR b60"))
        fahrenheit = image.celsius[90, 90] * 9 / 5 + 32
        # Measured 2026-10-02: 18.80 °F.
        assert fahrenheit == pytest.approx(19.0, abs=0.3)

    def test_ax8_box_max_min_avg_match_the_cameras_on_screen_reading(self):
        """The AX8 shows max 25.5, min 24.3, avg 25.1 °C over the frame."""
        stats = load_radiometric_file(_need(SAMPLES / "FLIR_AX8.jpg", "FLIR AX8")).stats
        # Measured 2026-10-02: 25.47 / 24.36 / 25.03 °C.
        assert stats["max_c"] == pytest.approx(25.5, abs=0.1)
        assert stats["min_c"] == pytest.approx(24.3, abs=0.1)
        assert stats["mean_c"] == pytest.approx(25.1, abs=0.1)

    def test_t640_max_matches_the_cameras_on_screen_reading(self):
        """The T640 shows "max 199 °F". The box outline is not recorded in the file, so
        the frame interior (3 px border excluded, where edge pixels run hot) is used."""
        image = load_radiometric_file(_need(SAMPLES / "FLIR_T640.jpg", "FLIR T640"))
        interior = image.celsius[3:-3, 3:-3]
        # Measured 2026-10-02: 199.56 °F; the display shows whole degrees.
        assert float(np.nanmax(interior)) * 9 / 5 + 32 == pytest.approx(199.0, abs=1.0)

    def test_a_palette_only_jpeg_is_still_refused(self):
        with pytest.raises(NotRadiometric, match="palette"):
            load_radiometric_file(_need(SAMPLES / "image.jpg", "palette-only sample"))


# ------------------------------------------------------------------ FLIR TLinear


def _ambient_c(day_csv: str, minute: str) -> float:
    with zipfile.ZipFile(THEDPV / "Metadata.zip") as archive:
        text = archive.read(f"Metadata/{day_csv}").decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    column = next(i for i, name in enumerate(header) if name.startswith("TAA"))
    for row in reader:
        if row and row[0].startswith(minute):
            return float(row[column])
    raise AssertionError(f"no ambient reading at {minute}")


class TestFlirTlinearTiff:
    def test_every_thedpv_frame_decodes_with_the_documented_gain(self):
        frames = sorted(_need(THEDPV_SESSION, "THED-PV session").glob("*_IR.TIFF"))
        assert len(frames) == 58
        medians = []
        for path in frames:
            image = load_radiometric_file(path)
            assert image.provenance["decoder"] == "flir_tlinear_tiff"
            assert image.provenance["tlinear_gain_k_per_count"] == 0.40
            assert "Duo Pro R User Guide" in image.provenance["source_document"]
            medians.append(image.stats["median_c"])
        # Measured 2026-10-02: frame medians 39.25 to 53.25 °C.
        ambient = _ambient_c("environmental_day2.csv", "2024-12-21 12:01")  # 24.3 °C
        # Plausibility only: sunlit modules (POA ~920 W/m²) sit above the air temperature.
        assert min(medians) > ambient + 5
        assert max(medians) < 90

    def test_an_undocumented_gain_is_refused(self, tmp_path):
        from PIL import Image
        from PIL.TiffImagePlugin import ImageFileDirectory_v2

        xmp = b'<x:xmpmeta><Camera:TlinearGain>0.10</Camera:TlinearGain></x:xmpmeta>'
        tags = ImageFileDirectory_v2()
        tags[700] = xmp
        tags.tagtype[700] = 1
        path = tmp_path / "odd_gain.tiff"
        Image.fromarray(np.full((4, 4), 780, dtype=np.uint16)).save(path, tiffinfo=tags)
        with pytest.raises(NotRadiometric, match="does not define"):
            load_radiometric_file(path)


# --------------------------------------------------------- temperature GeoTIFF


class TestTemperatureGeotiff:
    def test_the_bridge_orthomosaic_needs_its_unit_confirmed(self):
        from core.thermal import UnitConfirmationNeeded, load_temperature_geotiff

        ortho = _need(BRIDGE / "orthophoto" / "B1_Aug22.tif", "Bridge orthomosaic")
        with pytest.raises(UnitConfirmationNeeded, match="metre"):
            load_temperature_geotiff(ortho)
        image = load_temperature_geotiff(ortho, confirmed_unit="celsius")
        stats = image.stats
        # PROVENANCE.md: values 15.1 to 57.3 °C.
        assert stats["min_c"] == pytest.approx(15.1, abs=0.1)
        assert stats["max_c"] == pytest.approx(57.3, abs=0.1)
        assert image.provenance["unit_source"] == "confirmed_by_user"
        assert "metre" in image.provenance["note"]

    def test_the_api_asks_rather_than_assumes(self, tmp_path):
        from cockpit_harness import make_api

        ortho = _need(BRIDGE / "orthophoto" / "B1_Aug22.tif", "Bridge orthomosaic")
        api = make_api(tmp_path)
        refused = api.scale_thermal(str(ortho), "auto")
        assert refused["ok"] is False and refused["needs_unit_confirmation"] is True
        accepted = api.scale_thermal(str(ortho), "auto", confirmed_unit="celsius")
        assert accepted["ok"] and accepted["provenance"]["unit_source"] == "confirmed_by_user"


# ---------------------------------------- scaling and module analysis on real data


class TestDownstreamOnRealFrames:
    def test_auto_scale_on_a_decoded_dji_frame(self):
        _need_sdk()
        from core.thermal_scaling import auto_scale

        image = load_radiometric_file(
            _need(BRIDGE / "B1_Thermal" / "DJI_20220826113024_0100_T.JPG", "Bridge frame"))
        scale = auto_scale(image.celsius, 1.0, 99.0, "ironbow")
        low, high = np.percentile(image.celsius, [1, 99])
        assert scale.min_c == pytest.approx(low, abs=1e-6)
        assert scale.max_c == pytest.approx(high, abs=1e-6)

    def test_the_api_scales_a_flir_jpeg_and_a_tlinear_tiff(self, tmp_path):
        from cockpit_harness import make_api

        api = make_api(tmp_path)
        for path in (_need(SAMPLES / "FLIR_E40.jpg", "FLIR E40"),
                     next(_need(THEDPV_SESSION, "THED-PV").glob("*_IR.TIFF"))):
            result = api.scale_thermal(str(path), "anomaly")
            assert result["ok"], result
            assert result["provenance"]["unit"] == "celsius"

    def test_image_space_module_temperatures_on_a_thedpv_frame(self, tmp_path):
        """THED-PV frames are close-ups within one module and are not georeferenced, so
        the regions here are frame halves: test geometry, not module outlines."""
        from core.india_thermal import module_temperatures_in_image

        frame = next(_need(THEDPV_SESSION, "THED-PV").glob("*_IR.TIFF"))
        image = load_radiometric_file(frame)
        regions = [
            {"module_id": "left_half", "string_id": "frame",
             "polygon_px": [[0, 0], [319, 0], [319, 511], [0, 511]]},
            {"module_id": "right_half", "string_id": "frame",
             "polygon_px": [[320, 0], [639, 0], [639, 511], [320, 511]]},
        ]
        result = module_temperatures_in_image(
            frame, regions, tmp_path, region_source="test fixture: frame halves",
            warning_delta_c=5, critical_delta_c=10, hot_cell_delta_c=10,
            threshold_basis="test thresholds, no standard")
        rows = json.loads(Path(result.artifact_paths[0]).read_text(encoding="utf-8"))["modules"]
        left = image.celsius[:, :320]
        assert rows[0]["median_c"] == pytest.approx(float(np.median(left)), abs=1e-3)
        assert rows[0]["max_c"] == pytest.approx(float(left.max()), abs=1e-3)
        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        assert summary["provenance"]["decoder"] == "flir_tlinear_tiff"
        assert summary["temperature_source"].startswith("flir_tlinear_tiff")

    def test_map_placed_association_refuses_a_bare_frame_with_the_reason(self):
        from core.india_geospatial import IndiaPackRefused
        from core.india_thermal import _radiometric_georeferencing

        frame = next(_need(THEDPV_SESSION, "THED-PV").glob("*_IR.TIFF"))
        with pytest.raises(IndiaPackRefused, match="no map georeferencing"):
            _radiometric_georeferencing(frame)

    def test_a_thermal_map_records_how_its_temperatures_were_decoded(self, tmp_path):
        _need_sdk()
        from core.india_thermal import write_radiometric_thermal_map

        image = load_radiometric_file(
            _need(BRIDGE / "B1_Thermal" / "DJI_20220826113024_0100_T.JPG", "Bridge frame"))
        written = write_radiometric_thermal_map(image, tmp_path, epsg=32636,
                                                west=288200.0, north=4081290.0, pixel_size=0.05)
        manifest = next(p for p in written.artifact_paths if p.endswith("thermal_map.json"))
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        assert payload["temperature_source"] == "dji_thermal_sdk_dirp_measure_ex_via_core.thermal"
        assert payload["provenance"]["temperature_kind"] == "apparent_temperature_dji_sdk"
