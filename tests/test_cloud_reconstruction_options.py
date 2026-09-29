"""A cloud run uses the settings the operator chose, or says which it did not.

The round-2 audit found RemoteReconstructor sent the server only engine and profile, and
the server built every reconstruction from those two. A CRS set with Set CRS, Dense on,
max image size and marked control points were dropped without a word: a run asked for
EPSG:32616 with dense on came back in EPSG:32617 with no dense points, advising the
operator to "Enable dense MVS". These drive the real client against the real
services/api app on a local port, with real COLMAP on real survey frames.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")
pytest.importorskip("uvicorn")

from conftest import survey_imagery_dir  # noqa: E402
from test_cloud_reconstruction import _desktop_api, _survey_copy, _wait_job, server  # noqa: E402,F401

from core.reconstruction_remote import RemoteReconstructor, job_options  # noqa: E402


def _headers(server) -> dict:
    return {"Authorization": f"Bearer {server['token']}"}


def _submit(server, options: dict):
    """POST a reconstruction job straight at the server, as any client would."""
    from fastapi.testclient import TestClient
    from services.api.main import app

    client = TestClient(app)
    base, project = server["endpoint"].rsplit("/projects/", 1)
    dataset = client.post(f"/projects/{project}/datasets", headers=_headers(server),
                          json={"name": "options"}).json()["id"]
    return client.post(f"/projects/{project}/jobs", headers=_headers(server),
                       json={"kind": "reconstruction", "dataset_id": dataset,
                             "engine": "auto", "profile": "fast_preview", "options": options})


def test_the_server_refuses_options_it_would_otherwise_ignore(server):
    unknown = _submit(server, {"denes": True})
    assert unknown.status_code == 422 and "Unknown reconstruction option" in unknown.text
    bad_epsg = _submit(server, {"target_epsg": 999999})
    assert bad_epsg.status_code == 422 and "target_epsg" in bad_epsg.text
    not_bool = _submit(server, {"dense": "yes"})
    assert not_bool.status_code == 422 and "true or false" in not_bool.text
    bad_control = _submit(server, {"ground_control": [{"name": "G1", "x": 1}]})
    assert bad_control.status_code == 422 and "malformed" in bad_control.text


def test_a_server_that_does_not_report_applied_options_is_named():
    """An older server ignores job options; each unapplied setting is said, not assumed."""
    from core.reconstruction import ReconstructionResult

    remote = RemoteReconstructor("http://10.0.0.5:8000/projects/1", token="t",
                                 dense=True, target_epsg=32616, max_image_size=800)
    result = ReconstructionResult.__new__(ReconstructionResult)
    result.crs_epsg, result.engine = 32617, "colmap"
    old = remote._check_applied({}, result)
    assert len(old) == 1 and "were not used on the server" in old[0]
    assert "dense" in old[0] and "target_epsg" in old[0] and "max_image_size" in old[0]
    # A current server running the custom engine applies none of them.
    custom = remote._check_applied({"options_applied": {}}, result)
    assert "did not apply" in custom[0] and "CRS (EPSG)" in custom[0]
    # A server that applied all of them, but produced another CRS anyway.
    applied = {"dense": True, "target_epsg": 32616, "max_image_size": 800}
    assert remote._check_applied({"options_applied": applied}, result) == [
        "You asked for EPSG:32616; the server produced EPSG:32617."]
    result.crs_epsg = 32616
    assert remote._check_applied({"options_applied": applied}, result) == []
    assert job_options() == {}


def test_a_cloud_run_from_the_desktop_uses_the_projects_crs_and_dense_choice(server, tmp_path):
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route.")
    rasterio = pytest.importorskip("rasterio")

    images = _survey_copy(tmp_path)
    api = _desktop_api(tmp_path)
    # EPSG:32616 is the neighbouring UTM zone: the imagery's own zone is 32617, so a
    # server that ignored the choice produces 32617, as it did before this fix.
    assert api.set_project_crs(32616)["ok"]
    started = api.run_reconstruction({
        "image_dir": str(images), "output_dir": str(tmp_path / "recon"), "profile": "fast_preview",
        "dense": True, "max_image_size": 900,
        "execution_mode": "cloud", "cloud_endpoint": server["endpoint"],
        "cloud_token": server["token"]})
    assert started["ok"], started
    job = _wait_job(api, started["job_id"])
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["execution_mode_used"] == "cloud"
    assert payload["crs_epsg"] == 32616
    with rasterio.open(payload["orthomosaic_cog_path"]) as ortho:
        assert ortho.crs.to_epsg() == 32616
    summary = json.loads((tmp_path / "recon" / "reconstruction_result.json").read_text())
    assert summary["options_applied"] == {"dense": True, "use_gpu": True,
                                          "max_image_size": 900, "target_epsg": 32616}
    request = json.loads((tmp_path / "recon" / "cloud_request.json").read_text())
    assert request["options"]["target_epsg"] == 32616 and request["options"]["dense"] is True
    warnings = " ".join(payload["warnings"])
    assert "did not apply" not in warnings and "were not used on the server" not in warnings
    # Dense was asked for: it ran, or the server says why it could not. It is never
    # "Enable dense MVS" as if the operator had left it off.
    if payload["dense_point_count"] == 0:
        assert "Dense MVS skipped" in warnings, warnings
    assert "Enable dense MVS" not in warnings


def test_marked_control_points_georeference_a_cloud_run(server, tmp_path):
    """The marks go to the server, and its fit to them comes back and moves the products."""
    pycolmap = pytest.importorskip("pycolmap")
    rasterio = pytest.importorskip("rasterio")
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route.")
    from core import geo
    from core.reconstruction_colmap import colmap_available

    if not colmap_available():
        pytest.skip("pycolmap is not available in this environment.")
    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:8] if source.is_dir() else []
    if len(frames) < 8:
        pytest.skip(f"Survey imagery not present at {source} (gitignored).")
    images = tmp_path / "images"
    images.mkdir()
    for frame in frames:
        shutil.copy2(frame, images / frame.name)

    api = _desktop_api(tmp_path)
    options = {"image_dir": str(images), "engine": "colmap", "profile": "fast", "dense": False,
               "use_gpu": False, "max_image_size": 1024}
    first = _wait_job(api, api.run_reconstruction({**options, "output_dir": str(tmp_path / "gps")})["job_id"])
    assert first["status"] == "done", first
    gps_anchor = geo.GeoAnchor(**json.loads((tmp_path / "gps" / "geo_anchor.json").read_text()))

    # Four tie points COLMAP saw in four or more frames, "surveyed" at a known offset
    # from the geotag fit; COLMAP's own observations of them are the marks.
    offset = np.array([3.0, -2.0, 1.5])
    model = pycolmap.Reconstruction(str(max((tmp_path / "gps" / "colmap" / "sparse").iterdir())))
    candidates = [(pid, p) for pid, p in model.points3D.items()
                  if p.track.length() >= 4 and p.error < 1.0]
    xyz = np.array([p.xyz for _pid, p in candidates])
    centre = xyz.mean(axis=0)
    rows = ["name,easting,northing,elevation,epsg,accuracy"]
    marks = []
    for index, direction in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        score = (xyz[:, 0] - centre[0]) * direction[0] + (xyz[:, 1] - centre[1]) * direction[1]
        _pid, point3d = candidates[int(np.argmax(score))]
        world = gps_anchor.apply(np.asarray(point3d.xyz).reshape(1, 3))[0] + offset
        rows.append(f"T{index},{world[0]:.4f},{world[1]:.4f},{world[2]:.4f},{gps_anchor.epsg},0.02")
        for element in point3d.track.elements:
            xy = model.images[element.image_id].points2D[element.point2D_idx].xy
            marks.append((f"T{index}", model.images[element.image_id].name, float(xy[0]), float(xy[1])))
    csv = tmp_path / "control.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert api.import_gcps(str(csv))["count"] == 4
    for name, image, u, v in marks:
        assert api.mark_gcp(name, image, u, v)["ok"]

    cloud = _wait_job(api, api.run_reconstruction({
        **options, "output_dir": str(tmp_path / "cloud"), "execution_mode": "cloud",
        "cloud_endpoint": server["endpoint"], "cloud_token": server["token"]})["job_id"])
    assert cloud["status"] == "done", cloud
    payload = cloud["result"]
    assert payload["execution_mode_used"] == "cloud"
    report = payload["ground_control"]
    assert report and report["used"] == 4 and report["rmse_m"] < 0.5, report
    warnings = " ".join(payload["warnings"])
    assert "Ground control points were not used" not in warnings
    assert "did not apply" not in warnings
    anchor = json.loads((tmp_path / "cloud" / "geo_anchor.json").read_text())
    assert anchor["method"] == "ground_control"
    with rasterio.open(tmp_path / "gps" / "dsm.tif") as a, rasterio.open(tmp_path / "cloud" / "dsm.tif") as b:
        shift_x = (b.bounds.left + b.bounds.right - a.bounds.left - a.bounds.right) / 2
        shift_y = (b.bounds.bottom + b.bounds.top - a.bounds.bottom - a.bounds.top) / 2
    assert shift_x == pytest.approx(offset[0], abs=1.0)
    assert shift_y == pytest.approx(offset[1], abs=1.0)


def test_naming_the_profile_again_keeps_the_callers_choices(tmp_path):
    """The server passes the job's profile to reconstruct(), which reset dense and size."""
    pytest.importorskip("pycolmap")
    import cv2
    from core.reconstruction_colmap import ColmapReconstructor

    engine = ColmapReconstructor("standard", dense=False, max_image_size=640)
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="at least 3 images"):
        engine.reconstruct(images, output_dir=tmp_path / "out", profile="high")
    assert engine.settings["dense"] is False
    assert engine.settings["max_image_size"] == 640
