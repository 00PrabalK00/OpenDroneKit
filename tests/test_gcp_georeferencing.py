"""Ground control that actually controls something.

The audit found GCPs imported into memory (lost on restart), mark_gcp reachable from no
button, no reconstruction code that ever read them, and an accuracy report that quoted
"used 2, rmse 0.0834" over points its own warning said "were never marked in any image
and took no part in the fit".

Now: control points and their marks live in the project folder; each point is located
in the model by intersecting the rays through its marks with the reconstruction's own
camera poses; three or more usable points georeference the model (every product is
written with that fit); fewer are reported as an independent check on the geotag fit.

The first half builds a scene whose answer is known exactly -- cameras, a similarity
between the model frame and UTM, a GPS fit biased by 2 m east -- and checks the Api
recovers it. The second half runs real COLMAP on real survey frames through
Api.run_reconstruction, with marks taken from COLMAP's own feature observations of four
tie points and "surveyed" coordinates deliberately shifted from the GPS fit by a known
offset, and checks the deliverables moved by that offset.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from core import geo
from core.gcp import (
    GroundControlPoint,
    accuracy_report,
    add_mark,
    poses_from_records,
    residuals_from_positions,
    solve_with_control,
)

EPSG = 32617
ORIGIN = np.array([500000.0, 4570000.0, 100.0])
WIDTH, HEIGHT, FOCAL = 1000, 800, 900.0
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])

# Model frame -> local metric frame: rotated 30 degrees about z, scaled by 7.
ANGLE = math.radians(30)
ROT = np.array([[math.cos(ANGLE), -math.sin(ANGLE), 0], [math.sin(ANGLE), math.cos(ANGLE), 0], [0, 0, 1]])
SCALE = 7.0
SHIFT = np.array([3.0, -4.0, 1.0])

# Surveyed marks on the ground, relative to ORIGIN.
CONTROL = {"G1": (-20, -15, 0.5), "G2": (22, -18, -0.3), "G3": (18, 20, 0.8),
           "G4": (-19, 17, 0.1), "G5": (1, 2, 0.4)}
CAMERAS = {f"IMG_{i}.JPG": np.array([x, y, 100.0])
           for i, (x, y) in enumerate([(-15, -10), (15, -10), (15, 10), (-15, 10), (0, 0)])}
NADIR = np.diag([1.0, -1.0, -1.0])    # x east, image-down is south, looking straight down


def to_model(local: np.ndarray) -> np.ndarray:
    return ROT.T @ ((np.asarray(local, float) - SHIFT) / SCALE)


def camera_records() -> list[dict]:
    records = []
    for name, centre in CAMERAS.items():
        rotation = NADIR @ ROT                     # camera-from-model
        translation = -rotation @ to_model(centre)
        records.append({"image": name, "rotation_cam_from_world": rotation.tolist(),
                        "translation_cam_from_world": translation.tolist(),
                        "intrinsics": K.tolist(), "width": WIDTH, "height": HEIGHT})
    return records


def pixel(camera: str, local_point) -> tuple[float, float]:
    in_camera = NADIR @ (np.asarray(local_point, float) - CAMERAS[camera])
    uvw = K @ (in_camera / in_camera[2])
    return float(uvw[0]), float(uvw[1])


def anchor_dict(bias_east: float = 0.0) -> dict:
    return geo.GeoAnchor(scale=SCALE, rotation=ROT.tolist(),
                         translation=(SHIFT + [bias_east, 0, 0]).tolist(), epsg=EPSG,
                         origin=ORIGIN.tolist(), inlier_count=5, sample_count=5,
                         rmse_m=1.2, max_residual_m=2.0, method="umeyama_ransac").to_dict()


@pytest.fixture
def project(tmp_path):
    store = ProjectStore(tmp_path / "projects.db")
    api = Api(AppSession(store=store))
    api.create_project("Controlled", str(tmp_path / "project"))
    csv = tmp_path / "control.csv"
    csv.write_text("name,easting,northing,elevation,epsg,accuracy\n" + "".join(
        f"{n},{ORIGIN[0] + x},{ORIGIN[1] + y},{ORIGIN[2] + z},{EPSG},0.02\n"
        for n, (x, y, z) in CONTROL.items()), encoding="utf-8")
    run = tmp_path / "project" / "reconstruction"
    run.mkdir(parents=True)
    (run / "camera_poses.json").write_text(json.dumps({"cameras": camera_records()}), encoding="utf-8")
    (run / "geo_anchor.json").write_text(json.dumps(anchor_dict(bias_east=2.0)), encoding="utf-8")
    return api, store, csv, run


def mark_everything(api: Api) -> None:
    for name, local in CONTROL.items():
        for camera in CAMERAS:
            u, v = pixel(camera, local)
            assert api.mark_gcp(name, camera, u, v)["ok"]


class TestThroughTheApi:
    def test_control_and_marks_survive_a_restart(self, project):
        api, store, csv, _run = project
        assert api.import_gcps(str(csv))["count"] == 5
        mark_everything(api)
        again = Api(AppSession(store=store))
        listed = again.list_gcps()
        assert listed["count"] == 5
        assert all(p["mark_count"] == 5 and p["usable"] for p in listed["points"])
        # Re-importing the same file keeps the marks.
        assert again.import_gcps(str(csv))["ok"]
        assert all(p["mark_count"] == 5 for p in again.list_gcps()["points"])

    def test_the_full_pipeline_reconstructs_with_the_marked_control(self, project):
        """Start / Run AI built its reconstruction without the project's GCPs, silently."""
        from core.pipeline import PipelineConfig, StructuralFaultPipeline
        from core.reconstruction_colmap import _HAS_PYCOLMAP

        api, _store, csv, _run = project
        api.import_gcps(str(csv))
        mark_everything(api)
        control = api._control_points()
        assert len(control) == 5
        if _HAS_PYCOLMAP:
            config = PipelineConfig()
            config.reconstruction_engine = "colmap"
            pipeline = StructuralFaultPipeline(config, ground_control=control)
            assert [p.name for p in pipeline.reconstructor.ground_control] ==                 [p.name for p in control]
        # The custom engine cannot use them, and says so rather than ignoring them.
        config = PipelineConfig()
        config.reconstruction_engine = "custom"
        custom = StructuralFaultPipeline(config, ground_control=control)
        assert not hasattr(custom.reconstructor, "control_report")

    def test_a_mark_on_a_preview_is_rescaled_to_the_photograph(self, project):
        api, _store, csv, run = project
        api.import_gcps(str(csv))
        for name, local in CONTROL.items():
            for camera in CAMERAS:
                u, v = pixel(camera, local)
                # Marked on a 500x400 preview of the 1000x800 frame.
                assert api.mark_gcp(name, camera, u / 2, v / 2, 500, 400)["ok"]
        report = api.gcp_accuracy_report(reconstruction_dir=str(run))["report"]
        assert all(r["dx"] == pytest.approx(2.0, abs=1e-6) for r in report["residuals"])

    def test_a_mark_off_the_picture_is_refused(self, project):
        api, _store, csv, _run = project
        api.import_gcps(str(csv))
        result = api.mark_gcp("G1", "IMG_0.JPG", 1200, 10, 1000, 800)
        assert not result["ok"] and "outside" in result["error"]

    def test_the_report_locates_the_points_in_the_model_and_finds_the_gps_bias(self, project):
        api, _store, csv, run = project
        api.import_gcps(str(csv))
        mark_everything(api)
        result = api.gcp_accuracy_report(reconstruction_dir=str(run))
        assert result["ok"], result
        report = result["report"]
        assert report["role"] == "check"
        assert report["georeferenced_by"] == "umeyama_ransac"
        assert report["used"] == 5
        for residual in report["residuals"]:
            assert residual["dx"] == pytest.approx(2.0, abs=1e-6)
            assert residual["dy"] == pytest.approx(0.0, abs=1e-6)
            assert residual["dz"] == pytest.approx(0.0, abs=1e-6)
        assert all(v["rms_reprojection_px"] < 1e-6 for v in report["located"].values())
        assert any("run Process again" in w for w in report["warnings"])

    def test_without_a_reconstruction_it_says_so(self, project, tmp_path):
        api, _store, csv, _run = project
        api.import_gcps(str(csv))
        result = api.gcp_accuracy_report(reconstruction_dir=str(tmp_path / "nowhere"))
        assert not result["ok"] and "run Process" in result["error"]


class TestFittingTheModelToTheControl:
    def points(self, *, marks: int = 5) -> list[GroundControlPoint]:
        points = []
        for name, (x, y, z) in CONTROL.items():
            point = GroundControlPoint(name, ORIGIN[0] + x, ORIGIN[1] + y, ORIGIN[2] + z, EPSG)
            for camera in list(CAMERAS)[:marks]:
                add_mark(point, camera, *pixel(camera, (x, y, z)))
            points.append(point)
        return points

    def test_the_control_replaces_the_biased_gps_fit(self):
        biased = geo.GeoAnchor(**anchor_dict(bias_east=2.0))
        solution = solve_with_control(self.points(), poses_from_records(camera_records()),
                                      EPSG, biased)
        anchor = solution.anchor
        assert anchor.method == "ground_control"
        assert anchor.rmse_m < 1e-6
        # The fitted model now puts a model point where it really is.
        local = np.array([5.0, -7.0, 2.0])
        placed = anchor.apply(to_model(local).reshape(1, 3))[0]
        assert placed == pytest.approx(ORIGIN + local, abs=1e-6)
        assert solution.report["role"] == "fit"
        assert solution.report["leave_one_out"]["role"] == "check"
        assert solution.report["leave_one_out"]["rmse_m"] < 1e-6

    def test_three_points_fit_but_say_there_is_no_independent_check(self):
        solution = solve_with_control(self.points()[:3], poses_from_records(camera_records()),
                                      EPSG, None)
        assert solution.anchor is not None
        assert "leave_one_out" not in solution.report
        assert any("no independent check" in w for w in solution.report["warnings"])

    def test_thinly_marked_control_checks_the_gps_fit_then_corrects_its_position(self):
        """Too few marks for a 7-parameter fit, enough to say the GPS fit is 2 m east.

        The geotag fit is checked against the control first (that check is kept as
        camera_gps_check), then translated and scaled onto it, keeping the geotag
        orientation; the leave-one-out residuals are the independent figure.
        """
        biased = geo.GeoAnchor(**anchor_dict(bias_east=2.0))
        solution = solve_with_control(self.points(marks=2), poses_from_records(camera_records()),
                                      EPSG, biased)
        report = solution.report
        assert report["camera_gps_check"]["role"] == "check"
        assert report["camera_gps_check"]["residuals"][0]["dx"] == pytest.approx(2.0, abs=1e-6)
        assert any("three are needed" in w for w in report["warnings"])
        assert report["georeferenced_by"] == "camera_gps_shifted_to_control"
        assert solution.anchor.method == "camera_gps_shifted_to_control"
        assert solution.anchor.rotation == biased.rotation
        assert report["leave_one_out"]["rmse_m"] == pytest.approx(0.0, abs=1e-6)
        assert report["shift_m"][0] == pytest.approx(-2.0, abs=1e-6)

    def test_thinly_marked_control_without_a_correction_to_make_is_a_check(self):
        exact = geo.GeoAnchor(**anchor_dict())
        solution = solve_with_control(self.points(marks=2), poses_from_records(camera_records()),
                                      EPSG, exact)
        assert solution.anchor is None
        assert solution.report["role"] == "check"
        assert solution.report["georeferenced_by"] == "camera_gps"
        assert any("three are needed" in w for w in solution.report["warnings"])


def test_unmarked_points_never_count_as_used():
    """The audit's contradiction: used 2, while every point was unmarked."""
    points = [GroundControlPoint(f"T{i}", 1.0 * i, 2.0, 3.0, EPSG) for i in range(3)]
    computed = {p.name: (p.x + 0.05, p.y, p.z) for p in points}
    report = accuracy_report(points, residuals_from_positions(points, computed))
    assert report["used"] == 0
    assert report["rmse_m"] is None
    assert any("never marked" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Real COLMAP, real frames
# ---------------------------------------------------------------------------

IMAGERY = Path(os.environ.get("ODK_SURVEY_IMAGERY", "training/data/aukerman_subset"))
OFFSET = np.array([3.0, -2.0, 1.5])     # the "survey" disagrees with the geotags by this


def _wait(api: Api, job_id: str, limit_s: float = 900.0) -> dict:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        job = api.job_status(job_id)["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(1.0)
    raise AssertionError("reconstruction did not finish")


def test_marked_control_moves_the_real_deliverables(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    rasterio = pytest.importorskip("rasterio")
    from core.reconstruction_colmap import colmap_available

    if not colmap_available():
        pytest.skip("pycolmap is not available in this environment.")
    frames = sorted(IMAGERY.glob("*.JPG"))[:8] if IMAGERY.is_dir() else []
    if len(frames) < 8:
        pytest.skip(f"Survey imagery not present at {IMAGERY} (gitignored).")
    images = tmp_path / "images"
    images.mkdir()
    for frame in frames:
        shutil.copy2(frame, images / frame.name)

    store = ProjectStore(tmp_path / "projects.db")
    api = Api(AppSession(store=store))
    api.create_project("Real control", str(tmp_path / "project"))
    # CPU SIFT, as tests/test_reconstruction_colmap.py pins it: on eight frames the GPU
    # detector's fewer keypoints decide whether anything registers at all.
    options = {"image_dir": str(images), "engine": "colmap", "profile": "fast", "dense": False,
               "use_gpu": False, "max_image_size": 1024}

    # 1. Geotags only.
    first = _wait(api, api.run_reconstruction({**options, "output_dir": str(tmp_path / "gps")})["job_id"])
    assert first["status"] == "done", first
    gps_anchor = geo.GeoAnchor(**json.loads((tmp_path / "gps" / "geo_anchor.json").read_text()))

    # 2. Four well-spread tie points seen in at least three images, with COLMAP's own
    #    observations of them as the marks -- real pixels on real frames.
    model = pycolmap.Reconstruction(str(max((tmp_path / "gps" / "colmap" / "sparse").iterdir())))
    names = {image_id: image.name for image_id, image in model.images.items()}
    candidates = [(pid, p) for pid, p in model.points3D.items() if p.track.length() >= 4
                  and p.error < 1.0]
    xyz = np.array([p.xyz for _pid, p in candidates])
    centre = xyz.mean(axis=0)
    chosen = []
    for direction in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        score = (xyz[:, 0] - centre[0]) * direction[0] + (xyz[:, 1] - centre[1]) * direction[1]
        chosen.append(candidates[int(np.argmax(score))])
    rows = ["name,easting,northing,elevation,epsg,accuracy"]
    marks = []
    for index, (pid, point3d) in enumerate(chosen):
        world = gps_anchor.apply(np.asarray(point3d.xyz).reshape(1, 3))[0] + OFFSET
        rows.append(f"T{index},{world[0]:.4f},{world[1]:.4f},{world[2]:.4f},{gps_anchor.epsg},0.02")
        for element in point3d.track.elements:
            image = model.images[element.image_id]
            xy = image.points2D[element.point2D_idx].xy
            marks.append((f"T{index}", names[element.image_id], float(xy[0]), float(xy[1])))
    csv = tmp_path / "control.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert api.import_gcps(str(csv))["count"] == 4
    for name, image, u, v in marks:
        assert api.mark_gcp(name, image, u, v)["ok"]

    # 3. The same frames again, now with control.
    second = _wait(api, api.run_reconstruction({**options, "output_dir": str(tmp_path / "gcp")})["job_id"])
    assert second["status"] == "done", second
    anchor = json.loads((tmp_path / "gcp" / "geo_anchor.json").read_text())
    assert anchor["method"] == "ground_control"
    fit = json.loads((tmp_path / "gcp" / "gcp_report.json").read_text())
    assert fit["used"] == 4 and fit["rmse_m"] < 0.5, fit
    assert fit["leave_one_out"]["role"] == "check"

    # The deliverables moved by the offset between the survey and the geotags.
    with rasterio.open(tmp_path / "gps" / "dsm.tif") as a, rasterio.open(tmp_path / "gcp" / "dsm.tif") as b:
        shift_x = (b.bounds.left + b.bounds.right - a.bounds.left - a.bounds.right) / 2
        shift_y = (b.bounds.bottom + b.bounds.top - a.bounds.bottom - a.bounds.top) / 2
        za, zb = a.read(1, masked=True), b.read(1, masked=True)
        dz = float(np.ma.median(zb)) - float(np.ma.median(za))
    assert shift_x == pytest.approx(OFFSET[0], abs=1.0)
    assert shift_y == pytest.approx(OFFSET[1], abs=1.0)
    assert dz == pytest.approx(OFFSET[2], abs=1.0)

    # And the report on that run says the fit is to the control.
    report = api.gcp_accuracy_report(reconstruction_dir=str(tmp_path / "gcp"))["report"]
    assert report["georeferenced_by"] == "ground_control"
    assert report["role"] == "fit"
    assert report["rmse_m"] < 0.5


def test_control_is_imported_marked_and_reported_from_the_cockpit(project, tmp_path):
    """Add GCPs, Mark GCP (a click on the photograph) and GCP Report, in the real Shell."""
    import cv2

    from cockpit_actions_in_a_browser import run_actions

    api, _store, _csv, _run = project
    # A control file without a CRS column: the button asks for the EPSG code.
    bare = tmp_path / "control_no_crs.csv"
    bare.write_text("name,easting,northing,elevation\n" + "".join(
        f"{n},{ORIGIN[0] + x},{ORIGIN[1] + y},{ORIGIN[2] + z}\n"
        for n, (x, y, z) in CONTROL.items()), encoding="utf-8")
    # The photographs, at twice the resolution the reconstruction's cameras record, so
    # the mark has to be carried from the preview to the frame to the camera grid.
    dataset = tmp_path / "photos"
    dataset.mkdir()
    for camera in CAMERAS:
        cv2.imwrite(str(dataset / camera), np.full((HEIGHT * 2, WIDTH * 2, 3), 90, np.uint8))
    assert api.set_active_dataset(str(dataset))["ok"]

    ui_marks = ["IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG"]
    plan = [{"name": "import", "action": "Add GCPs", "answers": [str(EPSG)]}]
    for camera in ui_marks:
        u, v = pixel(camera, CONTROL["G1"])
        plan.append({"name": f"mark {camera}", "action": "Mark GCP",
                     "choose": ["G1", camera], "imageClick": [2 * u, 2 * v, 2 * WIDTH, 2 * HEIGHT]})
    plan.append({"name": "report", "action": "GCP Report"})

    # The other points are marked through the Api between the import and the report.
    class MarkingApi:
        def __getattr__(self, name):
            attribute = getattr(api, name)
            if name == "gcp_accuracy_report":
                def report(*args, **kwargs):
                    for other, local in CONTROL.items():
                        if other == "G1":
                            continue
                        for camera in ui_marks:
                            u, v = pixel(camera, local)
                            api.mark_gcp(other, camera, u, v)
                    return attribute(*args, **kwargs)
                return report
            return attribute

    results = run_actions(MarkingApi(), tmp_path, plan, picks=[str(bare)], workspace="processing")

    assert "[ok] Imported 5 control points" in results["import"], results
    assert "needs 2 more" in results["mark IMG_0.JPG"], results
    assert "3 marks." in results["mark IMG_2.JPG"], results
    # The clicks landed where G1 is in each frame (to within a preview pixel).
    g1 = next(p for p in api.list_gcps()["points"] if p["name"] == "G1")
    for mark in g1["marks"]:
        u, v = pixel(mark["image"], CONTROL["G1"])
        assert mark["image_width"] == 2 * WIDTH
        assert mark["pixel_x"] == pytest.approx(2 * u, abs=2.0)
        assert mark["pixel_y"] == pytest.approx(2 * v, abs=2.0)
    # And the report, located from those marks, finds the 2 m bias in the GPS fit.
    import re

    rmse = float(re.search(r"GCP RMSE ([0-9.]+) m", results["report"]).group(1))
    assert rmse == pytest.approx(2.0, abs=0.05), results["report"]
    assert "independent check" in results["report"]
