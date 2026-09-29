"""The India-first survey packs, run from the application rather than only from tests.

core/india_*.py was reachable from nothing: no Api method, no button, no route. The
semantic class maps they read came only from the server's processing runs. And a design
GeoJSON exported in UTM without a CRS member crashed with a raw PROJ error ("utm: Invalid
latitude") instead of saying what was wrong with the file.

These run each pack through its Api entry on real GeoTIFF/GeoJSON files in a project,
check the figures match the packs' own known-answer tests, and check the outputs land in
the layer tree. The semantic chain (installed ONNX model over a real orthomosaic, then a
land-GIS pack on the map it wrote) runs when the gitignored model is present; a browser
test presses Survey Packs in the real cockpit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from training.verification.conftest import (  # noqa: E402
    feature,
    line,
    point,
    polygon,
    write_geojson,
    write_raster,
)

SCHEMA = {0: "background", 1: "building", 2: "road", 3: "vegetation", 4: "water"}


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    api.create_project("Packs", str(tmp_path / "project"))
    return api


@pytest.fixture
def land(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint16)
    classes[1:3, 1:3] = 1
    classes[7:9, :] = 2
    classes[3:7, 3:7] = 3
    classes[0, 8:10] = 4
    raster = write_raster(tmp_path / "semantic.tif", classes, nodata=65535)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"classes": [{"id": k, "name": v} for k, v in SCHEMA.items()]}),
                      encoding="utf-8")
    return str(raster), str(schema)


def layer_names(api: Api) -> list[str]:
    return [layer["name"] for layer in api.list_layers()["layers"]]


class TestEachPackThroughTheApi:
    def test_land_gis(self, api, land):
        raster, schema = land
        result = api.india_land_gis(raster, schema)
        assert result["ok"], result
        assert result["summary"]["categories"]["building"]["area_m2"] == 4.0
        assert result["summary"]["parcel_boundary"]["status"] == "unavailable"
        assert any(name.startswith("land_gis:") for name in layer_names(api))
        assert Path(result["summary_path"]).is_relative_to(Path(api._session.project_root()))

    def test_encroachment_against_a_boundary(self, api, land, tmp_path):
        raster, schema = land
        boundary = write_geojson(tmp_path / "parcel.geojson", [feature(polygon(0, 0, 6, 10), parcel_id="P-1")])
        result = api.india_land_gis(raster, schema, str(boundary))
        assert result["ok"], result
        assert "before any legal conclusion" in result["summary"]["interpretation"]

    def test_design_progress(self, api, tmp_path):
        classes = np.zeros((10, 10), dtype=np.uint8)
        classes[2:6, 2:4] = 1
        classes[7:9, 7:9] = 1
        raster = write_raster(tmp_path / "construction.tif", classes, nodata=255)
        schema = tmp_path / "c.json"
        schema.write_text(json.dumps({"classes": [{"id": 0, "name": "background"},
                                                  {"id": 1, "name": "building"}]}), encoding="utf-8")
        design = write_geojson(tmp_path / "approved.geojson",
                               [feature(polygon(2, 2, 6, 6), element_id="B1", expected_class="building")])
        result = api.india_design_progress(str(design), str(raster), str(schema))
        assert result["ok"], result
        assert result["summary"]["observed_surface_coverage_percent"] == 50.0
        assert result["summary"]["contractual_completion_percent"]["status"] == "unavailable"

    def test_a_design_in_utm_without_a_crs_is_refused_in_words(self, api, tmp_path):
        classes = np.ones((10, 10), dtype=np.uint8)
        raster = write_raster(tmp_path / "construction.tif", classes, nodata=255)
        design = tmp_path / "approved_no_crs.geojson"
        design.write_text(json.dumps({"type": "FeatureCollection", "features": [
            feature(polygon(2, 2, 6, 6), element_id="B1", expected_class="building")]}),
            encoding="utf-8")
        result = api.india_design_progress(str(design), str(raster), str(self._schema(tmp_path)))
        assert not result["ok"]
        assert "not longitude/latitude" in result["error"]
        assert "PROJ" not in result["error"]

    @staticmethod
    def _schema(tmp_path):
        path = tmp_path / "schema1.json"
        path.write_text(json.dumps({"classes": [{"id": 1, "name": "building"}]}), encoding="utf-8")
        return path

    def test_canopy_cover(self, api, land):
        raster, schema = land
        result = api.india_canopy_cover(raster, schema)
        assert result["ok"], result

    def test_vegetation_indices_need_calibration(self, api, tmp_path):
        bands = np.stack([np.full((4, 4), v, dtype=np.uint16) for v in (2000, 3000, 4000, 6000)])
        raster = write_raster(tmp_path / "ms.tif", bands)
        calibrations = {role: {"band": band, "scale": 0.0001, "offset": 0, "source": "panel-v1"}
                        for role, band in (("red", 1), ("green", 2), ("nir", 4))}
        result = api.india_vegetation_indices(str(raster), calibrations)
        assert result["ok"], result
        assert result["summary"]["indices"]["NDVI"]["status"] == "available"
        assert result["summary"]["indices"]["NDRE"]["status"] == "unavailable"
        refused = api.india_vegetation_indices(str(raster), {"red": {"band": 1, "scale": 0.0001}})
        assert not refused["ok"]

    def test_road_condition(self, api, tmp_path):
        classes = np.zeros((10, 10), dtype=np.uint8)
        classes[4:7, :] = 1
        raster = write_raster(tmp_path / "road.tif", classes, nodata=255)
        schema = tmp_path / "r.json"
        schema.write_text(json.dumps({"classes": [{"id": 0, "name": "background"},
                                                  {"id": 1, "name": "road"}]}), encoding="utf-8")
        centre = write_geojson(tmp_path / "centre.geojson",
                               [feature(line(((0.5, 5.5), (9.5, 5.5))), road_id="R1")])
        props = {"evidence_source": "validated_model", "model_key": "m", "model_version": "1",
                 "confidence": 0.9}
        findings = write_geojson(tmp_path / "findings.geojson", [
            feature(point(2.5, 5.5), defect_type="pothole", severity="severe", **props)])
        result = api.india_road_condition(str(centre), str(findings), str(raster), str(schema))
        assert result["ok"], result
        assert result["summary"]["surveyed_distance_m"] == 9.0
        assert result["summary"]["road_surface_area_m2"] == 30.0

    def test_corridor_kind_is_checked(self, api, tmp_path):
        result = api.india_corridor_inspection("pipeline", "a", "b", "c", "d")
        assert not result["ok"] and "power" in result["error"]

    def test_the_semantic_packs_say_when_there_is_no_map(self, api):
        result = api.india_land_gis()
        assert not result["ok"]
        assert "no semantic map yet" in result["error"]


MODELS = Path(os.environ.get("ODK_MODELS_DIR", "models"))
ORTHO = Path(os.environ.get("ODK_DEMO_ORTHO",
                            "demo_project/aukerman_full/reconstruction/orthomosaic.tif"))


def test_the_installed_semantic_model_feeds_the_land_pack(api):
    """Real model, real orthomosaic, then a pack on the map it wrote."""
    pytest.importorskip("onnxruntime")
    model = MODELS / "shared_semantic_dinov2_vitb14_v4.onnx"
    if not model.is_file() or not ORTHO.is_file():
        pytest.skip("The semantic model or the demo orthomosaic is not present (gitignored).")
    started = api.semantic_map(str(ORTHO), str(model))
    assert started["ok"], started
    assert started["usable_labels"] == ["vegetation", "building"]
    deadline = time.time() + 600
    while True:
        job = api.job_status(started["job_id"])["job"]
        if job["status"] in ("done", "failed", "cancelled") or time.time() > deadline:
            break
        time.sleep(1)
    assert job["status"] == "done", job
    assert "treat every other class as unvalidated" in job["result"]["caveat"]
    import rasterio

    with rasterio.open(job["result"]["class_map"]) as classes:
        assert classes.crs.to_epsg() == 32617
        assert classes.width == 304 and classes.height == 231
    assert "Semantic classes" in layer_names(api)

    # The land pack, with no raster named, reads this project's latest semantic map.
    land = api.india_land_gis()
    assert land["ok"], land
    total = sum(v["area_m2"] for v in land["summary"]["categories"].values()
                if isinstance(v, dict) and "area_m2" in v)
    assert total > 0


def test_survey_packs_in_the_cockpit(api, land, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    raster, schema = land
    boundary = write_geojson(tmp_path / "parcel.geojson", [feature(polygon(0, 0, 6, 10), parcel_id="P-1")])
    results = run_actions(api, tmp_path, [
        {"name": "land", "action": "Survey Packs",
         "choose": ["Land GIS", "Choose a class raster", "Land cover vectors"]},
        {"name": "encroach", "action": "Survey Packs",
         "choose": ["Land GIS", "Choose a class raster", "Encroachment"]},
    ], picks=[raster, schema, raster, schema, str(boundary)])
    assert "[ok] Land GIS / encroachment:" in results["land"], results
    assert "building 4 m²" in results["land"]
    assert "layer(s) added" in results["land"]
    assert "[ok]" in results["encroach"] and "total(s)" in results["encroach"], results
    assert any(name.startswith("land_encroachment:") for name in layer_names(api))


def test_thermal_anomaly_and_corridor_packs_in_the_cockpit(api, tmp_path):
    """The cockpit paths the Api tests could not see.

    Module temperatures passed nine positional arguments to a method whose last five were
    keyword-only (a TypeError on every press), and its chooser offered only .tif while
    the pack reads a radiometric .json sidecar. Anomaly baseline sent one raster where
    the pack needs two. The corridor prompt suggested a capture geometry the pack refuses.
    """
    from cockpit_actions_in_a_browser import run_actions
    from test_india_anomaly import write_features
    from test_solar_thermal import _write_inputs, _write_modules
    from training.verification.test_india_assets import _validated_properties

    rgb, thermal, tie_points = _write_inputs(tmp_path)
    modules = _write_modules(tmp_path / "modules.geojson")
    rows, columns = np.indices((20, 20))
    normal = 10.0 + ((rows + columns) % 4)
    first = write_features(tmp_path / "normal-a.tif", normal)
    second = write_features(tmp_path / "normal-b.tif", np.roll(normal, 1, axis=1))
    centreline = write_geojson(tmp_path / "corridor.geojson",
                               [feature(line(((0, 5), (10, 5))), corridor="C1")])
    detections = write_geojson(tmp_path / "power.geojson", [feature(point(2, 5), **_validated_properties(
        asset_class="insulator", finding="corrosion", finding_validated=True))])

    picks: list = []
    results = run_actions(api, tmp_path, [
        {"name": "thermal", "action": "Survey Packs", "choose": "Module temperatures",
         "answers": ["field thermographer", "5", "10", "10", "IEC TS 62446-3"]},
        {"name": "baseline", "action": "Survey Packs",
         "choose": ["Anomaly baseline", "Add another", "Done"],
         "answers": ["two-person field review", "concrete deck, dry season"]},
        {"name": "power", "action": "Survey Packs",
         "choose": ["Power corridor", "close_range_oblique"],
         "answers": ["insulator model, 2cm GSD, oblique captures"]},
    ], picks=[str(thermal), str(rgb), str(tie_points), str(modules),
              str(first), str(second),
              str(detections), str(centreline)], pick_log=picks)

    assert "[ok] Module temperatures:" in results["thermal"], results
    assert "layer(s) added" in results["thermal"]
    assert "[ok] Anomaly baseline:" in results["baseline"], results
    assert "baseline written to" in results["baseline"]
    assert "[ok] Power corridor:" in results["power"], results
    assert "surveyed distance m 10" in results["power"]
    # Every file handed over is one the real chooser would have let the operator pick.
    for offered, path in picks:
        assert Path(path).suffix.lower().lstrip(".") in offered, (offered, path)

    baseline = next((Path(api._session.project_root()) / "survey_packs").rglob("baseline.json"))
    observed = normal.copy()
    observed[6:10, 8:12] = 30.0
    target = write_features(tmp_path / "observed.tif", observed)
    found = run_actions(api, tmp_path, [
        {"name": "find", "action": "Survey Packs", "choose": "Find anomalies", "answers": ["5"]},
    ], picks=[str(target), str(baseline)])
    assert "[ok] Find anomalies:" in found["find"], found
    assert "candidate count 1" in found["find"]
