"""Coverage QA places images where they were taken, or says it cannot.

The validator used to lay images out by phase-correlating each frame with the previous
one and, whenever the correlation was weak, used a hard-coded 180 px step instead. On
real aerial imagery that was every pair, so the "79.4% complete" and the twelve gap
suggestions it reported for the Aukerman set were measured on a layout it had made up;
and it flagged all 24 images as overlap gaps from an ORB match ratio that stays under
0.05 on grass even where COLMAP registers every image.

These tests write real JPEGs with real EXIF (GPS, camera model, focal length) for a
3 x 3 grid of shots with one missing, and check that the gap is found where the missing
shot should have been, that the overlap between neighbours is the geometric one, and
that with no height above ground nothing is estimated.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import geo
from core.coverage_validation import CoverageValidationConfig, CoverageValidator

Image = pytest.importorskip("PIL.Image")
from PIL.TiffImagePlugin import IFDRational  # noqa: E402


def _rational(value: float, denominator: int = 100000) -> IFDRational:
    return IFDRational(int(round(value * denominator)), denominator)

EPSG = 32617
ORIGIN = (437000.0, 4572700.0)
GROUND_M = 240.0
AGL_M = 100.0
# SONY DSC-WX220 (in the sensor table): 6.17 mm sensor, 4.45 mm lens.
# At 100 m: 138.65 m across, 104.0 m along (4:3).
FOOTPRINT_W = AGL_M * 6.17 / 4.45
SPACING = 60.0


def _dms(value: float):
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60) * 3600
    return (_rational(degrees, 1), _rational(minutes, 1), _rational(seconds))


def _shot(path, east: float, north: float, *, altitude: float | None = GROUND_M + AGL_M,
          texture_seed: int = 0):
    lon, lat = geo.projected_to_wgs84([east], [north], EPSG)
    rng = np.random.default_rng(texture_seed)
    pixels = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[0x010F] = "SONY"
    exif[0x0110] = "DSC-WX220"
    sub = exif.get_ifd(0x8769)
    sub[0x920A] = IFDRational(445, 100)         # FocalLength
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = _dms(float(lat[0]))
    gps[3] = "W"
    gps[4] = _dms(float(lon[0]))
    if altitude is not None:
        gps[5] = b"\x00"
        gps[6] = _rational(altitude, 1000)
    # Heading 0 (north), as senseFly writes it.
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
           b'Camera:Yaw="0.0"/></rdf:RDF></x:xmpmeta>')
    image.save(path, exif=exif, quality=90, xmp=xmp)
    return path


def _grid(tmp_path, skip=None, **kwargs):
    paths = []
    for row in range(3):
        for col in range(3):
            if (row, col) == skip:
                continue
            east = ORIGIN[0] + col * SPACING
            north = ORIGIN[1] + row * SPACING
            paths.append(_shot(tmp_path / f"IMG_{row}{col}.jpg", east, north,
                               texture_seed=row * 3 + col, **kwargs))
    return paths


class TestImagesArePlacedFromTheirGeotags:
    def test_footprints_come_from_gps_and_camera_geometry(self, tmp_path):
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(_grid(tmp_path), tmp_path / "out")
        assert result["layout"]["available"] is True
        assert result["layout"]["images_placed"] == 9
        first = result["footprints"][0]
        assert first["height_agl_m"] == pytest.approx(AGL_M, abs=0.01)
        assert first["height_source"] == "gps_altitude_minus_ground"
        assert first["width_m"] == pytest.approx(FOOTPRINT_W, rel=1e-3)
        assert first["yaw_source"] == "xmp"

    def test_neighbour_overlap_is_geometric(self, tmp_path):
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(_grid(tmp_path), tmp_path / "out")
        # IMG_00 -> IMG_01 are 60 m apart across a 138.65 m wide footprint.
        pair = next(p for p in result["overlap_pairs"]
                    if (p["image_prev"], p["image_curr"]) == ("IMG_00.jpg", "IMG_01.jpg"))
        assert pair["overlap"] == pytest.approx(1 - SPACING / FOOTPRINT_W, abs=0.01)
        assert pair["distance_m"] == pytest.approx(SPACING, abs=0.05)
        # Random-noise textures share no features at all; the old ORB ratio flagged
        # every one of these as an overlap gap.
        assert not any(row["low_overlap"] for row in result["images"])

    def test_a_complete_grid_is_fully_covered(self, tmp_path):
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(_grid(tmp_path), tmp_path / "out")
        assert result["coverage_completion_pct"] == pytest.approx(100.0)
        assert not [s for s in result["gap_suggestions"] if s["kind"] == "coverage_gap"]


class TestGapsAreWhereTheyAre:
    def test_a_missing_shot_is_found_at_its_position(self, tmp_path):
        # A 3x3 grid with the corner (2,2) missing; with AOI covering the full block the
        # NE corner is under-covered.
        lon, lat = geo.projected_to_wgs84(
            [ORIGIN[0], ORIGIN[0] + 2 * SPACING + 60, ORIGIN[0] + 2 * SPACING + 60, ORIGIN[0]],
            [ORIGIN[1], ORIGIN[1], ORIGIN[1] + 2 * SPACING + 45, ORIGIN[1] + 2 * SPACING + 45],
            EPSG)
        aoi = [[float(a), float(b)] for a, b in zip(lon, lat)]
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(_grid(tmp_path, skip=(2, 2)), tmp_path / "out", aoi_lonlat=aoi)
        assert result["coverage_completion_pct"] < 100.0
        gaps = [s for s in result["gap_suggestions"] if s["kind"] == "coverage_gap"]
        assert gaps
        biggest = max(gaps, key=lambda s: s["area_m2"])
        assert biggest["easting"] > ORIGIN[0] + 2 * SPACING
        assert biggest["northing"] > ORIGIN[1] + 2 * SPACING
        assert len(biggest["lonlat"]) == 2


class TestNothingIsEstimatedWithoutAHeight:
    def test_no_height_means_no_coverage_figure(self, tmp_path):
        result = CoverageValidator().validate(_grid(tmp_path), tmp_path / "out")
        assert result["coverage_completion_pct"] is None
        assert result["layout"]["available"] is False
        assert "height above ground" in result["layout"]["reason"]
        assert len(result["layout"]["images_not_placed"]["no_height"]) == 9
        assert not any(row["low_overlap"] for row in result["images"])
        assert all(row["overlap_assessed"] is False for row in result["images"])

    def test_no_gps_means_not_placed(self, tmp_path):
        path = tmp_path / "plain.jpg"
        Image.fromarray(np.zeros((50, 50, 3), np.uint8)).save(path)
        result = CoverageValidator(CoverageValidationConfig(flight_height_agl_m=100)) \
            .validate([path], tmp_path / "out")
        assert result["layout"]["images_not_placed"]["no_gps"] == ["plain.jpg"]
        assert result["coverage_completion_pct"] is None

    def test_a_stated_flight_height_is_used_and_named(self, tmp_path):
        result = CoverageValidator(CoverageValidationConfig(flight_height_agl_m=AGL_M)) \
            .validate(_grid(tmp_path, altitude=None), tmp_path / "out")
        assert result["footprints"][0]["height_source"] == "stated_flight_height"
        assert result["coverage_completion_pct"] == pytest.approx(100.0)


class TestAnImageThatCannotBePlacedIsNotAGap:
    """A terrain model with a hole left some real Aukerman images without a height.

    They were dropped from the capture sequence, so 235 was paired with 237 (6%
    overlap) and a re-fly was suggested over ground that 236 had photographed.
    """

    @staticmethod
    def _strip_over_a_hole(tmp_path):
        # Two columns of five shots northwards, 60 m apart: a footprint is 104 m long,
        # so neighbours overlap 42% and shots two apart (120 m) do not meet at all.
        paths = []
        for col in range(2):
            # Serpentine, as flown: up the first column, back down the second.
            for index in range(5):
                row = index if col == 0 else 4 - index
                paths.append(_shot(tmp_path / f"IMG_{col}{index}.jpg",
                                   ORIGIN[0] + col * SPACING, ORIGIN[1] + row * SPACING,
                                   texture_seed=col * 5 + row))
        # A DTM at the ground level with no data under the middle row.
        cell = 5.0
        west, north = ORIGIN[0] - 200.0, ORIGIN[1] + 4 * SPACING + 200.0
        height, width = int((4 * SPACING + 400) / cell), int((SPACING + 400) / cell)
        dtm = np.full((height, width), GROUND_M, dtype=np.float32)
        hole_top = int((north - (ORIGIN[1] + 2 * SPACING + 10)) / cell)
        hole_bottom = int((north - (ORIGIN[1] + 2 * SPACING - 10)) / cell)
        dtm[hole_top:hole_bottom, :] = -9999.0
        terrain = geo.write_geotiff(tmp_path / "dtm.tif", dtm, epsg=EPSG, west=west,
                                    north=north, pixel_size=cell, nodata=-9999.0)
        return paths, terrain

    def test_pairs_across_an_unplaced_image_are_not_assessed(self, tmp_path):
        paths, terrain = self._strip_over_a_hole(tmp_path)
        result = CoverageValidator(CoverageValidationConfig(terrain_path=str(terrain))) \
            .validate(paths, tmp_path / "out")
        assert sorted(result["layout"]["images_not_placed"]["no_height"]) == \
            ["IMG_02.jpg", "IMG_12.jpg"]
        pairs = {(p["image_prev"], p["image_curr"]) for p in result["overlap_pairs"]}
        # Never IMG_01 -> IMG_03: IMG_02 lies between them.
        assert ("IMG_01.jpg", "IMG_03.jpg") not in pairs
        assert ("IMG_00.jpg", "IMG_01.jpg") in pairs
        unassessed = {(p["image_prev"], p["image_curr"]) for p in result["pairs_not_assessed"]}
        assert {("IMG_01.jpg", "IMG_02.jpg"), ("IMG_02.jpg", "IMG_03.jpg")} <= unassessed
        assert result["missing_pairs"] == [
            p for p in result["missing_pairs"] if p["image_prev"][:5] != p["image_curr"][:5]]
        within_a_column = [p for p in result["missing_pairs"]
                           if p["image_prev"][:5] == p["image_curr"][:5]]
        assert within_a_column == []
        rows = {r["image"]: r for r in result["images"]}
        assert rows["IMG_01.jpg"]["low_overlap"] is False
        assert "not assessed" in result["layout"]["not_assessed_note"]

    def test_no_refly_is_suggested_over_ground_an_unplaced_image_took(self, tmp_path):
        paths, terrain = self._strip_over_a_hole(tmp_path)
        result = CoverageValidator(CoverageValidationConfig(terrain_path=str(terrain))) \
            .validate(paths, tmp_path / "out")
        # The band between rows 1 and 3 has no placed footprint over it ...
        assert result["unassessed_gaps"], result["gap_suggestions"]
        band = result["unassessed_gaps"][0]
        assert abs(band["northing"] - (ORIGIN[1] + 2 * SPACING)) < 20.0
        assert set(band["related_images"]) & {"IMG_02.jpg", "IMG_12.jpg"}
        # ... but it was photographed, so it is not a re-fly instruction.
        assert not [s for s in result["gap_suggestions"]
                    if s["kind"] in ("coverage_gap", "overlap_gap")]

    def test_a_shot_that_was_really_missed_is_still_a_gap(self, tmp_path):
        paths, _terrain = self._strip_over_a_hole(tmp_path)
        kept = [p for p in paths if p.name not in ("IMG_02.jpg", "IMG_12.jpg")]
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(kept, tmp_path / "out")
        assert ("IMG_01.jpg", "IMG_03.jpg") in {
            (p["image_prev"], p["image_curr"]) for p in result["missing_pairs"]}
        assert [s for s in result["gap_suggestions"] if s["kind"] == "coverage_gap"]
        assert result["unassessed_gaps"] == []
