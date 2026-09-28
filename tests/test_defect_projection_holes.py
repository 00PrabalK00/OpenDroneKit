"""Defect back-projection over a DSM with holes.

A DSM rasterised from a sparse reconstruction is mostly holes (70% of cells on the
Aukerman run). Two faults followed from that, both measured on the real reconstruction:

* a ray that entered a hole above the surface and left it below was "hit" in mid-air --
  z 279.7 m over a DSM whose highest cell is 264.6 m -- because the last gap was carried
  across the hole and the bisection broke on NaN and returned its midpoint;
* with one bbox corner lost in a hole, the area came from the triangle that was left:
  1.5 m2 for a 3 x 3 m footprint.

And the projected layer is polygons, which risk ranking read no position from, so every
ranked defect sat at latitude/longitude null.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core import geo
from core.defect_projection import (
    CameraPose,
    SurfaceModel,
    bridge_small_holes,
    intersect_surface,
    project_defects,
    write_defect_layer,
)
from core.risk_scoring import score_run_risk

ORIGIN = (500000.0, 4570000.0)


def _surface(elevation: np.ndarray, pixel: float = 1.0) -> SurfaceModel:
    rows, cols = elevation.shape
    return SurfaceModel(elevation=elevation.astype(float), west=ORIGIN[0],
                        north=ORIGIN[1] + rows * pixel, pixel_size=pixel, epsg=32617)


def _flat_with_band_hole(first_col: int, last_col: int, size: int = 100) -> SurfaceModel:
    elevation = np.zeros((size, size))
    elevation[:, first_col:last_col + 1] = np.nan
    return _surface(elevation)


class TestNoMidAirHits:
    def test_a_ray_crossing_the_surface_inside_a_wide_hole_finds_nothing(self):
        # Flat ground at z=0 with a 20-cell hole. The ray descends 1 m per metre and
        # reaches z=0 in the middle of the hole, where the surface is unknown.
        surface = _flat_with_band_hole(40, 59)
        origin = np.array([ORIGIN[0] + 30.0, ORIGIN[1] + 50.0, 20.0])
        direction = np.array([1.0, 0.0, -1.0])
        hit = intersect_surface(origin, direction, surface)
        # Before the fix: a point inside the hole, in the air, reported as the surface.
        assert hit is None

    def test_every_hit_is_on_the_surface(self):
        elevation = np.full((80, 80), 5.0)
        elevation[:, 30:33] = np.nan
        surface = _surface(elevation)
        origin = np.array([ORIGIN[0] + 10.0, ORIGIN[1] + 40.0, 30.0])
        for slope in (-0.3, -0.5, -0.9, -1.4, -2.5):
            hit = intersect_surface(origin, np.array([1.0, 0.0, slope]), surface)
            if hit is None:
                continue
            ground = surface.sample(hit[0], hit[1])
            assert math.isfinite(ground)
            assert hit[2] == pytest.approx(ground, abs=1e-6)

    def test_a_small_hole_is_bridged_and_labelled(self):
        surface = _flat_with_band_hole(49, 50)
        bridged, filled = bridge_small_holes(surface, max_cells=3)
        assert filled[:, 49:51].all()
        assert not filled[:, :49].any()
        origin = np.array([ORIGIN[0] + 30.0, ORIGIN[1] + 50.0, 20.0])
        hit = intersect_surface(origin, np.array([1.0, 0.0, -1.0]), bridged)
        assert hit is not None
        assert hit[2] == pytest.approx(0.0, abs=1e-6)
        assert hit[0] == pytest.approx(ORIGIN[0] + 50.0, abs=0.1)

    def test_a_wide_hole_is_not_bridged(self):
        surface = _flat_with_band_hole(30, 69)
        bridged, _ = bridge_small_holes(surface, max_cells=3)
        assert np.isnan(bridged.elevation[:, 40:60]).all()


def _nadir(image: str, east: float, north: float, height: float,
           focal: float = 1000.0) -> CameraPose:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    centre = np.array([east, north, height])
    return CameraPose(image=image, rotation=rotation, translation=-rotation @ centre,
                      intrinsics=np.array([[focal, 0, 320.0], [0, focal, 240.0], [0, 0, 1]]),
                      width=640, height=480)


def _identity_anchor() -> geo.GeoAnchor:
    return geo.GeoAnchor(scale=1.0, rotation=np.eye(3).tolist(), translation=[0.0, 0.0, 0.0],
                         epsg=32617, origin=[0.0, 0.0, 0.0], inlier_count=10,
                         sample_count=10, rmse_m=0.1, max_residual_m=0.2)


class TestPartialOutlinesAreNotMeasured:
    def test_a_bbox_with_a_corner_in_a_wide_hole_is_counted_not_shrunk(self):
        elevation = np.zeros((200, 200))
        # A hole covering the NE corner of the footprint, far wider than can be bridged.
        elevation[40:95, 105:160] = np.nan
        surface = _surface(elevation)
        cx, cy = ORIGIN[0] + 100.0, ORIGIN[1] + 100.0
        pose = _nadir("a.jpg", cx, cy, 100.0)
        # 200 x 200 px at 100 m and f=1000 px is a 20 x 20 m footprint.
        detection = {"image": "a.jpg", "bbox": [220, 140, 420, 340], "defect_type": "spalling"}
        diagnostics: dict[str, int] = {}
        defects = project_defects(detections=[detection], poses={"a.jpg": pose},
                                  surface=surface, anchor=_identity_anchor(),
                                  diagnostics=diagnostics)
        assert defects == []
        assert diagnostics["incomplete_boundaries"] == 1

    def test_a_complete_outline_is_measured_exactly(self):
        surface = _surface(np.zeros((200, 200)))
        cx, cy = ORIGIN[0] + 100.0, ORIGIN[1] + 100.0
        pose = _nadir("a.jpg", cx, cy, 100.0)
        detection = {"image": "a.jpg", "bbox": [220, 140, 420, 340], "defect_type": "spalling"}
        defects = project_defects(detections=[detection], poses={"a.jpg": pose},
                                  surface=surface, anchor=_identity_anchor())
        assert len(defects) == 1
        assert defects[0].area_m2 == pytest.approx(400.0, rel=0.01)
        assert defects[0].surface_interpolated is False


class TestRankedDefectsHaveAPosition:
    def test_polygon_defects_carry_their_location_into_risk_ranking(self, tmp_path):
        surface = _surface(np.zeros((200, 200)))
        # A real place, so the WGS84 conversion is meaningful: UTM 17N near Aukerman.
        cx, cy = ORIGIN[0] + 100.0, ORIGIN[1] + 100.0
        pose = _nadir("a.jpg", cx, cy, 100.0)
        detection = {"image": "a.jpg", "bbox": [300, 220, 340, 260],
                     "defect_type": "spalling", "confidence": 0.9}
        defects = project_defects(detections=[detection], poses={"a.jpg": pose},
                                  surface=surface, anchor=_identity_anchor())
        layer = tmp_path / "defects.geojson"
        write_defect_layer(layer, defects)
        feature = json.loads(layer.read_text())["features"][0]
        assert feature["geometry"]["type"] == "Polygon"

        ranked = score_run_risk(None, defects_geojson=layer, structure_type="bridge")
        top = ranked["defects"][0]
        lon, lat = geo.projected_to_wgs84([cx], [cy], 32617)
        assert top["longitude"] == pytest.approx(float(lon[0]), abs=1e-5)
        assert top["latitude"] == pytest.approx(float(lat[0]), abs=1e-5)

    def test_a_layer_without_centre_properties_uses_the_ring_centroid(self, tmp_path):
        ring = [[-81.75, 41.30], [-81.74, 41.30], [-81.74, 41.31], [-81.75, 41.31]]
        geo.write_geojson(tmp_path / "d.geojson",
                          [geo.polygon_feature(ring, {"defect_type": "crack", "area_m2": 1.0})])
        ranked = score_run_risk(None, defects_geojson=tmp_path / "d.geojson")
        assert ranked["defects"][0]["longitude"] == pytest.approx(-81.745)
        assert ranked["defects"][0]["latitude"] == pytest.approx(41.305)

    def test_the_ranking_score_is_not_called_integrity(self, tmp_path):
        """health_scoring owns integrity_score; the ranking's own figure used the same
        name with a different value in the same report."""
        ring = [[-81.75, 41.30], [-81.74, 41.30], [-81.74, 41.31], [-81.75, 41.31]]
        geo.write_geojson(tmp_path / "d.geojson",
                          [geo.polygon_feature(ring, {"defect_type": "crack", "area_m2": 1.0})])
        ranked = score_run_risk(None, defects_geojson=tmp_path / "d.geojson")
        assert "integrity_score" not in ranked
        assert "defect_condition_score" in ranked
