"""The three survey-pack gaps the end-to-end re-audit left, pressed in the real cockpit.

1. Anomaly baseline refused every raster the application writes ("no ODK_FEATURE_SCHEMA
   tag") and nothing wrote that tag. Vegetation indices now carry it, and an untagged
   raster (orthomosaic, DSM, another tool's output) is accepted once the operator says
   what its bands are -- recorded as declared, and never allowed to contradict a tag.
2. The power/rail corridor refused a WGS84 centreline, which is what a GPS track or a web
   map exports. It is now measured in the UTM zone of its own midpoint, and says so.
3. Land GIS reported classes the semantic model does not validate (water, road) as plain
   "available". Each category now carries validated true/false/unknown, from the map's
   own manifest or the installed model's registry entry, and the cockpit says UNVALIDATED
   next to the figure.

Each runs through Survey Packs in headless Chromium against a real Api
(cockpit_actions_in_a_browser.run_actions); only the file chooser and modal answers are
supplied by the harness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core.india_anomaly import fit_validated_baseline  # noqa: E402
from core.india_geospatial import IndiaPackRefused  # noqa: E402
from training.verification.conftest import feature, write_raster  # noqa: E402
from training.verification.test_india_assets import _validated_properties  # noqa: E402


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    api.create_project("Packs", str(tmp_path / "project"))
    return api


def _untagged(path: Path, values: np.ndarray) -> str:
    from test_india_anomaly import write_features

    return str(write_features(path, values, schema=""))


# -- 1. anomaly baseline on the application's own rasters -----------------------------

def test_an_untagged_raster_needs_a_declared_schema_and_a_tag_is_never_overridden(tmp_path):
    from test_india_anomaly import write_features

    rows, columns = np.indices((20, 20))
    normal = 10.0 + ((rows + columns) % 4)
    a = _untagged(tmp_path / "a.tif", normal)
    b = _untagged(tmp_path / "b.tif", np.roll(normal, 1, axis=1))
    with pytest.raises(IndiaPackRefused, match="say what its bands are"):
        fit_validated_baseline([a, b], tmp_path / "x.json", validated_by="r", validation_scope="s")
    baseline = fit_validated_baseline([a, b], tmp_path / "base.json", validated_by="field crew",
                                      validation_scope="deck", declared_schema="dsm_elevation_m")
    assert baseline.feature_schema == "dsm_elevation_m"
    assert baseline.feature_schema_source == "declared by field crew"
    saved = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert saved["feature_schema_source"] == "declared by field crew"
    assert {s["feature_schema_source"] for s in saved["source_files"]} == {"declared"}

    tagged = str(write_features(tmp_path / "t.tif", normal, schema="surface-v1"))
    with pytest.raises(IndiaPackRefused, match="not overridden"):
        fit_validated_baseline([tagged, b], tmp_path / "y.json", validated_by="r",
                               validation_scope="s", declared_schema="dsm_elevation_m")


def test_a_vegetation_index_the_app_wrote_is_a_valid_anomaly_input(api, tmp_path):
    """The app's own index rasters now say what they are, so no declaration is needed."""
    import rasterio

    red = np.full((3, 3), 2000, dtype=np.uint16)
    nir = np.array([[2000, 4000, 8000]] * 3, dtype=np.uint16)
    raster = write_raster(tmp_path / "ms.tif", np.stack([red, nir]))
    result = api.india_vegetation_indices(str(raster), {
        "red": {"band": 1, "scale": 0.0001, "offset": 0, "source": "panel"},
        "nir": {"band": 2, "scale": 0.0001, "offset": 0, "source": "panel"}})
    assert result["ok"], result
    ndvi = next(p for p in result["artifacts"] if p.lower().endswith("ndvi.tif"))
    with rasterio.open(ndvi) as src:
        assert src.tags()["ODK_FEATURE_SCHEMA"] == "calibrated_ndvi_reflectance_index_v1"


def test_anomaly_pair_on_untagged_rasters_in_the_cockpit(api, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    rows, columns = np.indices((20, 20))
    normal = 10.0 + ((rows + columns) % 4)
    first = _untagged(tmp_path / "dsm-a.tif", normal)
    second = _untagged(tmp_path / "dsm-b.tif", np.roll(normal, 1, axis=1))
    results = run_actions(api, tmp_path, [
        {"name": "baseline", "action": "Survey Packs",
         "choose": ["Anomaly baseline", "Add another", "Done"],
         "answers": ["two-person field review", "deck, dry season", "dsm_elevation_m"]},
    ], picks=[first, second])
    assert "[ok] Anomaly baseline:" in results["baseline"], results
    baseline = next((Path(api._session.project_root()) / "survey_packs").rglob("baseline.json"))
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    assert saved["feature_schema"] == "dsm_elevation_m"
    assert saved["feature_schema_source"] == "declared by two-person field review"

    observed = normal.copy()
    observed[6:10, 8:12] = 30.0
    target = _untagged(tmp_path / "dsm-now.tif", observed)
    found = run_actions(api, tmp_path, [
        {"name": "find", "action": "Survey Packs", "choose": "Find anomalies",
         "answers": ["5", "dsm_elevation_m"]},
        {"name": "wrong", "action": "Survey Packs", "choose": "Find anomalies",
         "answers": ["5", "rgb_orthomosaic_8bit"]},
    ], picks=[target, str(baseline), target, str(baseline)])
    assert "[ok] Find anomalies:" in found["find"], found
    assert "candidate count 1" in found["find"]
    # A declaration that is not the baseline's schema is refused, not compared.
    assert "[error]" in found["wrong"] and "does not match baseline" in found["wrong"], found


# -- 2. a longitude/latitude corridor centreline ---------------------------------------

def _lonlat_geojson(path: Path, features: list[dict]) -> str:
    # No crs member: GeoJSON is WGS84 longitude/latitude by definition (RFC 7946).
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                    encoding="utf-8")
    return str(path)


def test_a_wgs84_power_centreline_is_measured_in_its_utm_zone(api, tmp_path):
    from cockpit_actions_in_a_browser import run_actions
    from pyproj import Geod

    # 0.02 degrees of longitude at 18.5 N near Pune, and one insulator on it.
    start, end = (73.80, 18.50), (73.82, 18.50)
    centreline = _lonlat_geojson(tmp_path / "line.geojson", [feature(
        {"type": "LineString", "coordinates": [list(start), list(end)]}, corridor="C1")])
    detections = _lonlat_geojson(tmp_path / "det.geojson", [feature(
        {"type": "Point", "coordinates": [73.81, 18.50]},
        **_validated_properties(asset_class="insulator", finding="corrosion",
                                finding_validated=True))])
    geodesic = Geod(ellps="WGS84").line_length([start[0], end[0]], [start[1], end[1]])

    results = run_actions(api, tmp_path, [
        {"name": "power", "action": "Survey Packs",
         "choose": ["Power corridor", "close_range_oblique"],
         "answers": ["insulator model, 2cm GSD, oblique captures"]},
    ], picks=[detections, centreline])
    assert "[ok] Power corridor:" in results["power"], results
    assert "projecting it to its UTM zone, EPSG:32643" in results["power"]
    summary = next((Path(api._session.project_root()) / "survey_packs").rglob(
        "power_inspection.json"))
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["measured_in_epsg"] == 32643
    # UTM's scale error near the zone's central meridian is under 0.1 %.
    assert data["surveyed_distance_m"] == pytest.approx(geodesic, rel=1e-3)
    assert data["asset_count"] == 1


# -- 3. unvalidated land classes are flagged -------------------------------------------

@pytest.fixture
def land(tmp_path):
    classes = np.zeros((10, 10), dtype=np.uint16)
    classes[1:3, 1:3] = 1
    classes[7:9, :] = 2
    classes[3:7, 3:7] = 3
    classes[0, 8:10] = 4
    raster = write_raster(tmp_path / "semantic.tif", classes, nodata=65535)
    names = {0: "background", 1: "building", 2: "road", 3: "vegetation", 4: "water"}
    classes_json = [{"id": k, "name": v} for k, v in names.items()]
    manifest = tmp_path / "semantic_manifest.json"
    manifest.write_text(json.dumps({"classes": classes_json,
                                    "validated_classes": ["vegetation", "building"]}),
                        encoding="utf-8")
    plain = tmp_path / "schema.json"
    plain.write_text(json.dumps({"classes": classes_json}), encoding="utf-8")
    return str(raster), str(manifest), str(plain)


def test_land_gis_flags_classes_the_model_does_not_validate(api, land, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    raster, manifest, plain = land
    direct = api.india_land_gis(raster, manifest)
    assert direct["ok"], direct
    cats = direct["summary"]["categories"]
    assert cats["building"]["validated"] is True and cats["vegetation"]["validated"] is True
    assert cats["water"]["validated"] is False and cats["road_path"]["validated"] is False
    assert "UNVALIDATED" in cats["water"]["validation"]
    assert direct["summary"]["unvalidated_categories"] == ["road_path", "water"]
    vectors = json.loads(Path(direct["summary"]["artifacts"]["vectors"]).read_text(encoding="utf-8"))
    water = [f for f in vectors["features"] if f["properties"].get("layer") == "water"]
    assert water and all(f["properties"]["validated"] is False for f in water)

    unknown = api.india_land_gis(raster, plain)["summary"]["categories"]
    assert unknown["building"]["validated"] is None   # nobody said; not claimed

    results = run_actions(api, tmp_path, [
        {"name": "land", "action": "Survey Packs",
         "choose": ["Land GIS", "Choose a class raster", "Land cover vectors"]},
        {"name": "plain", "action": "Survey Packs",
         "choose": ["Land GIS", "Choose a class raster", "Land cover vectors"]},
    ], picks=[raster, manifest, raster, plain])
    assert "[ok] Land GIS / encroachment:" in results["land"], results
    assert "water 2 m² (UNVALIDATED)" in results["land"]
    assert "road_path 20 m² (UNVALIDATED)" in results["land"]
    assert "building 4 m²," in results["land"]
    assert "building 4 m² (validation unknown)" in results["plain"], results


def test_a_map_made_by_the_installed_model_is_matched_by_its_checksum(api, land, tmp_path):
    """An older semantic manifest without validated_classes: the model is found by SHA."""
    from core.models import load_registry

    entry = (load_registry().get("models") or {}).get("shared_semantic_landcover") or {}
    if not entry.get("sha256") or not entry.get("usable_labels"):
        pytest.skip("No installed semantic model entry with a checksum in the registry.")
    raster, manifest, _plain = land
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload.pop("validated_classes")
    payload["model"] = {"checkpoint_sha256": entry["sha256"]}
    Path(manifest).write_text(json.dumps(payload), encoding="utf-8")
    cats = api.india_land_gis(raster, manifest)["summary"]["categories"]
    assert cats["water"]["validated"] is False
    assert cats["vegetation"]["validated"] is ("vegetation" in entry["usable_labels"])
    assert math.isfinite(cats["water"]["area_m2"])
