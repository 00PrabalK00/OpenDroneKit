"""Every India survey pack, pressed in the real cockpit.

The earlier browser tests pressed Land GIS, encroachment, module temperatures, the
anomaly pair and the power corridor. The rest -- design progress, canopy cover,
vegetation indices, stress zones, plant count, road condition, rail corridor and the
solar inventory -- were reached only by calling the Api directly, so nothing checked the
buttons' own argument lists or what the operator was told afterwards. Three pack inputs
the library accepts were not offered at all: a canopy mask for stress zones, label-id
masks for plant count (read as binary, touching crowns merge into one plant) and
validated findings for the solar inventory. Stress zones, indices and plant count
reported only "done".

Each step here goes through Survey Packs in headless Chromium against a real Api, with
the packs' known-answer inputs, and checks the figures the operator reads.
"""

from __future__ import annotations

import json
import os
import re
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
from training.verification.test_india_assets import _validated_properties  # noqa: E402


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    api.create_project("Packs", str(tmp_path / "project"))
    return api


def _schema(path: Path, classes: dict[int, str]) -> str:
    path.write_text(json.dumps({"classes": [{"id": k, "name": v} for k, v in classes.items()]}),
                    encoding="utf-8")
    return str(path)


def test_the_construction_and_agriculture_packs_in_the_cockpit(api, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    # Design progress: half of the approved building footprint is observed.
    built = np.zeros((10, 10), dtype=np.uint8)
    built[2:6, 2:4] = 1
    construction = write_raster(tmp_path / "construction.tif", built, nodata=255)
    construction_schema = _schema(tmp_path / "c.json", {0: "background", 1: "building"})
    design = write_geojson(tmp_path / "approved.geojson",
                           [feature(polygon(2, 2, 6, 6), element_id="B1", expected_class="building")])
    # Canopy: 50 m2 of crop out of 90 m2 classified.
    field = np.zeros((10, 10), dtype=np.uint8)
    field[:5, :] = 1
    field[5:9, :] = 2
    field[9, :] = 255
    canopy = write_raster(tmp_path / "canopy.tif", field, nodata=255)
    canopy_schema = _schema(tmp_path / "a.json", {0: "background", 1: "crop", 2: "soil"})
    # Four calibrated bands; NIR rises across the columns, so NDVI falls into three zones.
    red = np.full((3, 3), 2000, dtype=np.uint16)
    green = np.full((3, 3), 3000, dtype=np.uint16)
    red_edge = np.full((3, 3), 3000, dtype=np.uint16)
    nir = np.array([[2000, 4000, 8000]] * 3, dtype=np.uint16)
    multispectral = write_raster(tmp_path / "ms.tif", np.stack([red, green, red_edge, nir]))
    # Plant crowns: two touching crowns with their own ids, and a third apart.
    crowns = np.zeros((12, 12), dtype=np.uint16)
    crowns[1:3, 1:3] = 1
    crowns[1:3, 3:5] = 2
    crowns[8:10, 2:4] = 3
    plants = write_raster(tmp_path / "plants.tif", crowns, nodata=0,
                          tags={"ODK_INSTANCE_KIND": "plant"})

    picks: list = []
    results = run_actions(api, tmp_path, [
        {"name": "design", "action": "Survey Packs",
         "choose": ["Design progress", "Choose a class raster"]},
        {"name": "canopy", "action": "Survey Packs",
         "choose": ["Canopy cover", "Choose a class raster"]},
        {"name": "indices", "action": "Survey Packs", "choose": "Vegetation indices",
         "answers": ["panel-v1", "1", "0.0001", "0", "2", "0.0001", "0",
                     "3", "0.0001", "0", "4", "0.0001", "0"]},
        {"name": "plants_binary", "action": "Survey Packs",
         "choose": ["Plant count", "binary"], "answers": ["mango crowns, orchard-A, 3cm GSD"]},
        {"name": "plants_ids", "action": "Survey Packs",
         "choose": ["Plant count", "label_ids"], "answers": ["mango crowns, orchard-A, 3cm GSD"]},
    ], picks=[str(design), str(construction), construction_schema,
              str(canopy), canopy_schema,
              str(multispectral),
              str(plants), str(plants)], pick_log=picks, workspace="processing")

    assert "[ok] Design progress:" in results["design"], results
    assert "observed surface coverage percent 50" in results["design"]
    assert "[ok] Canopy cover:" in results["canopy"], results
    assert "canopy area m2 50" in results["canopy"]
    assert "[ok] Vegetation indices:" in results["indices"], results
    assert "NDVI available" in results["indices"]
    assert "layer(s) added" in results["indices"]
    # Read as binary the two touching crowns are one plant; with their ids, two.
    assert "[ok] Plant count:" in results["plants_binary"], results
    assert "count 2" in results["plants_binary"]
    assert "count 3" in results["plants_ids"], results
    for offered, path in picks:
        assert Path(path).suffix.lower().lstrip(".") in offered, (offered, path)

    # Stress zones on the NDVI the indices pack just wrote, restricted to a canopy mask
    # that leaves out the first column.
    ndvi = next((Path(api._session.project_root()) / "survey_packs").rglob("ndvi.tif"))
    mask = np.ones((3, 3), dtype=np.uint8)
    mask[:, 0] = 0
    with __import__("rasterio").open(ndvi) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=None)
    canopy_mask = tmp_path / "canopy_mask.tif"
    with __import__("rasterio").open(canopy_mask, "w", **profile) as dst:
        dst.write(mask, 1)
    stress = run_actions(api, tmp_path, [
        {"name": "zones", "action": "Survey Packs", "choose": ["Stress zones", "No canopy mask"],
         "answers": ["0.2", "0.5", "wheat, sensor-X, vegetative stage, 20m AGL"]},
        {"name": "masked", "action": "Survey Packs",
         "choose": ["Stress zones", "Choose a canopy mask"],
         "answers": ["0.2", "0.5", "wheat, sensor-X, vegetative stage, 20m AGL"]},
    ], picks=[str(ndvi), str(ndvi), str(canopy_mask)], workspace="processing")
    assert "[ok] Stress zones:" in stress["zones"], stress
    assert "severe stress 3 m²" in stress["zones"]
    assert "moderate stress 3 m²" in stress["zones"]
    assert "reference or healthy range 3 m²" in stress["zones"]
    # The mask leaves out the first column, where every pixel is severe.
    assert "[ok] Stress zones:" in stress["masked"], stress
    assert "severe stress 0 m²" in stress["masked"], stress
    assert "moderate stress 3 m²" in stress["masked"]
    assert "reference or healthy range 3 m²" in stress["masked"]


def test_the_asset_packs_in_the_cockpit(api, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    road_classes = np.zeros((10, 10), dtype=np.uint8)
    road_classes[4:7, :] = 1
    road = write_raster(tmp_path / "road.tif", road_classes, nodata=255)
    road_schema = _schema(tmp_path / "r.json", {0: "background", 1: "road"})
    centre = write_geojson(tmp_path / "centre.geojson",
                           [feature(line(((0.5, 5.5), (9.5, 5.5))), road_id="R1")])
    road_findings = write_geojson(tmp_path / "road_findings.geojson", [
        feature(point(2.5, 5.5), **_validated_properties(defect_type="pothole", severity="severe"))])
    corridor = write_geojson(tmp_path / "corridor.geojson",
                             [feature(line(((0, 5), (10, 5))), corridor="C1")])
    rail = write_geojson(tmp_path / "rail.geojson", [feature(point(4, 5), **_validated_properties(
        asset_class="track", finding="obstacle", finding_validated=False))])
    labels = np.zeros((4, 12), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[1:3, 4:6] = 2
    labels[1:3, 7:9] = 3
    modules = write_raster(tmp_path / "modules.tif", labels, nodata=0, tags={
        "ODK_INSTANCE_KIND": "solar_module", "ODK_MODEL_KEY": "solar-yolo11l-seg",
        "ODK_MODEL_VERSION": "candidate-1"})
    layout = write_geojson(tmp_path / "layout.geojson", [
        feature(polygon(1, 1, 3, 3), module_id="M1"), feature(polygon(4, 1, 6, 3), module_id="M2"),
        feature(polygon(7, 1, 9, 3), module_id="M3"), feature(polygon(10, 1, 12, 3), module_id="M4")])
    solar_findings = write_geojson(tmp_path / "solar_findings.geojson", [
        feature(point(5, 2), **_validated_properties(finding="damaged_module", module_id=2))])

    picks: list = []
    results = run_actions(api, tmp_path, [
        {"name": "road", "action": "Survey Packs",
         "choose": ["Road condition", "Choose a class raster"]},
        {"name": "rail", "action": "Survey Packs", "choose": ["Rail corridor", "corridor_nadir_oblique"],
         "answers": ["fastener model, 1cm GSD, nadir"]},
        {"name": "solar", "action": "Survey Packs",
         "choose": ["Solar module inventory", "Choose an approved layout", "Choose validated findings"],
         "answers": ["Duke UAV module masks, 2cm GSD"]},
        {"name": "solar_bare", "action": "Survey Packs",
         "choose": ["Solar module inventory", "No approved layout", "No findings"],
         "answers": ["Duke UAV module masks, 2cm GSD"]},
    ], picks=[str(centre), str(road_findings), str(road), road_schema,
              str(rail), str(corridor),
              str(modules), str(layout), str(solar_findings),
              str(modules)], pick_log=picks, workspace="processing")

    assert "[ok] Road condition:" in results["road"], results
    assert "surveyed distance m 9" in results["road"]
    assert "road surface area m2 30" in results["road"]
    assert "[ok] Rail corridor:" in results["rail"], results
    assert "surveyed distance m 10" in results["rail"]
    assert "[ok] Solar module inventory:" in results["solar"], results
    assert "module count 3" in results["solar"]
    assert "module count 3" in results["solar_bare"], results
    for offered, path in picks:
        assert Path(path).suffix.lower().lstrip(".") in offered, (offered, path)

    summaries = sorted((Path(api._session.project_root()) / "survey_packs").rglob("*.json"))
    solar = [json.loads(p.read_text(encoding="utf-8")) for p in summaries
             if "solar" in str(p).lower()]
    with_findings = [s for s in solar if isinstance(s, dict) and s.get("findings_by_type")]
    assert with_findings and with_findings[0]["findings_by_type"] == {"damaged_module": 1}
    assert with_findings[0]["approved_layout"]["missing_module_review_count"] == 1


MODELS = Path(os.environ.get("ODK_MODELS_DIR", "models"))
ORTHO = Path(os.environ.get("ODK_DEMO_ORTHO",
                            "demo_project/aukerman_full/reconstruction/orthomosaic.tif"))


def test_semantic_map_button_then_a_pack_on_the_projects_latest_map(api, tmp_path):
    """Semantic Map on the project's real orthomosaic, then Land GIS on "latest map"."""
    pytest.importorskip("onnxruntime")
    from cockpit_actions_in_a_browser import run_actions

    model = MODELS / "shared_semantic_dinov2_vitb14_v4.onnx"
    if not model.is_file() or not ORTHO.is_file():
        pytest.skip("The semantic model or the demo orthomosaic is not present (gitignored).")
    api._session.register_raster_layer(str(ORTHO), name="Orthomosaic")
    os.environ.setdefault("ODK_MODELS_DIR", str(MODELS))
    started = run_actions(api, tmp_path, [{"name": "map", "action": "Semantic Map"}],
                          workspace="processing")
    assert "Semantic map started" in started["map"], started
    assert "vegetation and building" in started["map"]
    jobs = [j for j in api.list_jobs()["jobs"] if "semantic" in str(j.get("kind", j)).lower()]
    assert jobs, api.list_jobs()
    deadline = time.time() + 900
    while True:
        job = api.job_status(jobs[0]["id"])["job"]
        if job["status"] in ("done", "failed", "cancelled") or time.time() > deadline:
            break
        time.sleep(1)
    assert job["status"] == "done", job
    land = run_actions(api, tmp_path, [
        {"name": "land", "action": "Survey Packs",
         "choose": ["Land GIS", "This project's latest semantic map", "Land cover vectors"]},
    ], workspace="processing")
    assert "[ok] Land GIS / encroachment:" in land["land"], land
    assert "building" in land["land"] and "layer(s) added" in land["land"]
    # Water and road are predicted but not validated by the model's holdout; the
    # operator reads that next to their areas.
    assert re.search(r"water [\d.]+ m² \(UNVALIDATED\)", land["land"]), land
    assert not re.search(r"vegetation [\d.]+ m² \(UNVALIDATED\)", land["land"]), land
