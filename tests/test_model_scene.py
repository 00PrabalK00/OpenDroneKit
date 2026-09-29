"""core.model_scene: the reconstruction as a drawable scene, and measurements on it.

The browser tests (test_cockpit_3d_viewer.py) drive the viewer; these pin the parts a
screen cannot show -- that thinning is reported, that clips are the export's clips, and
that a length is scaled by the recorded georeference and nothing else.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from core.geo import GeoAnchor
from core.model_clipping import Clip, ClipPlane, ClipStore
from core.model_scene import SceneRefused, build_scene, measure_scene, read_ply
from core.reconstruction import _write_ascii_ply


def _positions(scene):
    return np.frombuffer(base64.b64decode(scene["positions"]), dtype="<f4").reshape(-1, 3)


def _cloud(recon, n=1000):
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    _write_ascii_ply(recon / "reconstruction.ply", points, np.full((n, 3), 128, np.uint8))
    return points


def test_a_polygon_ply_is_fan_triangulated(tmp_path) -> None:
    path = tmp_path / "quad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
                    "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
                    "end_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n", encoding="utf-8")
    vertices, colors, faces = read_ply(path)
    assert vertices.shape == (4, 3) and colors is None
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_thinning_is_reported_not_hidden(tmp_path) -> None:
    _cloud(tmp_path, 1000)
    scene = build_scene(tmp_path, "pointcloud", max_points=300)
    assert scene["source_vertex_count"] == 1000
    assert scene["vertex_count"] < 1000 and scene["reduction"] == "every 4th point"


def test_positions_are_relative_to_the_reported_center(tmp_path) -> None:
    points = _cloud(tmp_path, 50) + 0  # file coordinates
    scene = build_scene(tmp_path, "pointcloud")
    restored = _positions(scene) + np.asarray(scene["center"])
    np.testing.assert_allclose(restored, points, atol=1e-5)


def test_visible_clips_are_applied_and_named(tmp_path) -> None:
    points = _cloud(tmp_path, 1000)
    ClipStore(tmp_path).add(Clip(name="east", kind="plane",
                                 plane=ClipPlane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))))
    scene = build_scene(tmp_path, "pointcloud")
    assert scene["clips_applied"] == ["east"]
    assert scene["vertex_count"] == int((points[:, 0] >= 0).sum())
    ClipStore(tmp_path).set_visible("east", False)
    assert build_scene(tmp_path, "pointcloud")["vertex_count"] == 1000


def test_no_model_is_a_refusal(tmp_path) -> None:
    with pytest.raises(SceneRefused, match="No mesh in this project yet"):
        build_scene(tmp_path, "mesh")


def test_length_is_scaled_by_the_anchor_and_refused_without_one(tmp_path) -> None:
    with pytest.raises(SceneRefused, match="structure-from-motion units"):
        measure_scene(tmp_path, "length", [[0, 0, 0], [1, 0, 0]])
    anchor = GeoAnchor(scale=4.0, rotation=np.eye(3).tolist(), translation=[0, 0, 0], epsg=32633,
                       origin=[500000.0, 0.0, 0.0], inlier_count=5, sample_count=5,
                       rmse_m=0.3, max_residual_m=0.5)
    (tmp_path / "geo_anchor.json").write_text(json.dumps(anchor.to_dict()), encoding="utf-8")
    m = measure_scene(tmp_path, "length", [[0, 0, 0], [3, 0, 4]])
    assert m["slope_distance_m"] == 20.0 and m["horizontal_distance_m"] == 12.0
    assert m["vertical_distance_m"] == 16.0 and m["georeference_rmse_m"] == 0.3
    area = measure_scene(tmp_path, "area", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert area["surface_area_m2"] == 16.0 and area["planimetric_area_m2"] == 16.0
