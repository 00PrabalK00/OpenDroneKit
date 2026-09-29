"""The 3D model: shown, measured on, and returned to from a saved view.

measure_in_model always refused on real reconstructions ("mesh.ply has no provenance
sidecar"), and could not have been trusted if it had not: the mesh is written in the
structure-from-motion frame, so a sidecar naming the run's CRS against it -- which a
provenance pass over the output folder would write -- would have let model units pass as
metres. Save View and Measure in 3D answered "the 3D viewer is not built".

Now the reconstruction's mesh is georeferenced with the geo_anchor.json written beside
it (a copy in the CRS, with a sidecar saying so), the cockpit draws it in a WebGL view,
clicks are picked by the application on the full mesh, and saved views store and restore
that view's camera.

The scene is a 20 x 10 x 6 m block on a ground plane, written in a model frame that is
rotated 30 degrees, scaled by 1/7 and shifted from UTM -- so every number checked here
only comes out right if the georeferencing was applied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from core import geo
from core.model_measurement import NotMetric, georeference_model, model_scale
from core.provenance import record

EPSG = 32617
ORIGIN = np.array([500000.0, 4570000.0, 100.0])
ANGLE = math.radians(30)
ROT = np.array([[math.cos(ANGLE), -math.sin(ANGLE), 0], [math.sin(ANGLE), math.cos(ANGLE), 0], [0, 0, 1]])
SCALE = 7.0
SHIFT = np.array([3.0, -4.0, 1.0])


def to_model(local):
    return (ROT.T @ ((np.asarray(local, float) - SHIFT).T)).T / SCALE


def scene_local():
    """A closed block on an open ground plane, in local metres."""
    box = [(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0),
           (0, 0, 6), (20, 0, 6), (20, 10, 6), (0, 10, 6)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    ground = [(-15, -15, 0), (35, -15, 0), (35, 25, 0), (-15, 25, 0)]
    base = len(box)
    faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(box + ground, float), np.array(faces)


@pytest.fixture
def project(tmp_path):
    store = ProjectStore(tmp_path / "projects.db")
    api = Api(AppSession(store=store))
    api.create_project("Model", str(tmp_path / "project"))
    recon = tmp_path / "project" / "reconstruction"
    recon.mkdir(parents=True)
    vertices, faces = scene_local()
    model = to_model(vertices)
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in model]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    (recon / "mesh.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    anchor = geo.GeoAnchor(scale=SCALE, rotation=ROT.tolist(), translation=SHIFT.tolist(),
                           epsg=EPSG, origin=ORIGIN.tolist(), inlier_count=8, sample_count=8,
                           rmse_m=0.8, max_residual_m=1.4)
    (recon / "geo_anchor.json").write_text(json.dumps(anchor.to_dict()), encoding="utf-8")
    return api, recon


class TestTheMeshIsMeasuredInMetres:
    def test_a_crs_recorded_against_the_raw_mesh_is_not_taken_as_metres(self, project):
        """What a provenance pass over the output folder writes: true of the run, not of
        the vertices."""
        _api, recon = project
        record(recon / "mesh.obj", engine="colmap", crs_epsg=EPSG)
        with pytest.raises(NotMetric, match="model frame"):
            model_scale(recon / "mesh.obj")

    def test_the_georeferenced_copy_is_in_the_crs(self, project):
        _api, recon = project
        copy = georeference_model(recon / "mesh.obj")
        assert model_scale(copy).epsg == EPSG
        from core.model_measurement import read_mesh

        vertices, _faces = read_mesh(copy)
        local, _ = scene_local()
        assert vertices == pytest.approx(local + ORIGIN, abs=1e-3)
        # Reused while the source is unchanged.
        assert georeference_model(recon / "mesh.obj") == copy

    def test_measure_in_model_measures_the_real_outputs(self, project):
        api, recon = project
        a, b = ORIGIN + [2, 2, 6], ORIGIN + [18, 8, 6]
        result = api.measure_in_model(str(recon / "mesh.obj"), "length", [a.tolist(), b.tolist()])
        assert result["ok"], result
        assert result["measurement"]["slope_distance_m"] == pytest.approx(math.hypot(16, 6), abs=1e-3)
        assert result["model_path"].endswith("mesh.epsg32617.obj")
        assert api.list_measurements()["measurements"][0]["measurement_type"] == "model_length"

    def test_a_model_with_no_georeferencing_is_refused(self, project):
        api, recon = project
        (recon / "geo_anchor.json").unlink()
        result = api.measure_in_model(str(recon / "mesh.obj"), "length", [[0, 0, 0], [1, 1, 1]])
        assert not result["ok"]
        assert "no provenance sidecar" in result["error"] or "not georeferenced" in result["error"]

    def test_a_pick_lands_on_the_full_mesh(self, project):
        api, recon = project
        above = (ORIGIN + [10, 5, 50]).tolist()
        hit = api.pick_model_point(str(recon / "mesh.obj"), above, [0, 0, -1])
        assert hit["ok"], hit
        assert hit["point"] == pytest.approx((ORIGIN + [10, 5, 6]).tolist(), abs=1e-6)
        miss = api.pick_model_point(str(recon / "mesh.obj"), above, [0, 0, 1])
        assert not miss["ok"] and "not on the model" in miss["error"]

    def test_the_scene_is_sent_relative_to_its_origin(self, project):
        api, _recon = project
        scene = api.model_scene()
        assert scene["ok"], scene
        assert scene["epsg"] == EPSG
        positions = np.array(scene["positions"]).reshape(-1, 3) + scene["origin"]
        local, _ = scene_local()
        assert sorted(map(tuple, np.round(positions - ORIGIN, 3))) == sorted(map(tuple, local))
        assert max(abs(v) for v in scene["positions"]) < 100      # float32-safe


def test_measure_and_saved_views_in_the_cockpit(tmp_path, project):
    """Textured Mesh, Measure in 3D (two clicks on the WebGL canvas), Save View, Open View."""
    from cockpit_actions_in_a_browser import run_actions

    api, _recon = project
    roof_a, roof_b = (ORIGIN + [2, 2, 6]).tolist(), (ORIGIN + [18, 8, 6]).tolist()
    roof_c, ground = (ORIGIN + [10, 5, 6]).tolist(), (ORIGIN + [-5, -5, 0]).tolist()
    camera = "const { currentModelViewer } = await import('./js/workspace/model3d.js'); " \
             "const v = currentModelViewer(); return v.getCamera();"
    turn = "const { currentModelViewer } = await import('./js/workspace/model3d.js'); " \
           "const v = currentModelViewer(); v.yaw += 1.2; v.distance *= 0.5; v.render(); " \
           "return v.getCamera();"
    zoom = "const { whenModelReady } = await import('./js/workspace/model3d.js'); "            "const v = await whenModelReady(); v.target = v.target.map((c, i) => i < 2 ? c * 0.3 : c); v.distance *= 0.3; v.render(); "            "return { distance: v.distance, pixels: v.size().height };"
    results = run_actions(api, tmp_path, [
        {"name": "show", "action": "Textured Mesh"},
        {"name": "height", "action": "Measure in 3D", "choose": "height",
         "modelClicks": [roof_c, ground]},
        # Closer in, as an operator zooms before picking: a pick is one screen pixel,
        # and the headless canvas is small.
        {"name": "zoom", "action": "Textured Mesh", "after": zoom},
        {"name": "length", "action": "Measure in 3D", "choose": "length",
         "modelClicks": [roof_a, roof_b]},
        {"name": "save", "action": "Save View", "answers": ["north corner"], "after": camera},
        {"name": "turn", "action": "Textured Mesh", "after": turn},
        {"name": "open", "action": "Open View", "choose": "north corner", "after": camera},
    ], workspace="twin", virtual_time_ms=20000)

    assert "[error]" not in json.dumps(results), results
    # A click is a whole screen pixel; on the small headless canvas one pixel covers
    # about a decimetre of roof, so the picks are checked to that and no finer.
    history = {m["measurement_type"]: m for m in api.list_measurements()["measurements"]}
    picked = history["model_length"]["details"]["points_xyz"]
    for got, wanted in zip(picked, [roof_a, roof_b]):
        assert got[2] == pytest.approx(wanted[2], abs=1e-6)        # on the roof itself
        assert math.dist(got[:2], wanted[:2]) < 0.25, results["zoom:after"]
    length = float(results["length"].split("Model length ")[1].split(" m")[0])
    assert length == pytest.approx(math.dist(picked[0], picked[1]), abs=1e-3)
    assert length == pytest.approx(math.hypot(16, 6), abs=0.5), results["length"]
    height = float(results["height"].split("Model height ")[1].split(" m")[0])
    assert height == pytest.approx(6.0, abs=1e-3), results["height"]
    assert "EPSG:32617" in results["length"]

    saved, turned, opened = results["save:after"], results["turn:after"], results["open:after"]
    assert turned["position"] != pytest.approx(saved["position"], abs=1.0)
    assert opened["position"] == pytest.approx(saved["position"], abs=1e-3)
    assert opened["target"] == pytest.approx(saved["target"], abs=1e-3)
    stored = api.list_views()["views"][0]
    assert stored["name"] == "north corner"
    assert list(stored["position"]) == pytest.approx(saved["position"], abs=1e-3)
    # Kept in the project's measurement history.
    kinds = [m["measurement_type"] for m in api.list_measurements()["measurements"]]
    assert sorted(kinds) == ["model_height", "model_length"]
