"""Reconstruction, ground control and facade planning on real, openly licensed surveys.

Each test runs the real COLMAP engine on frames from the test corpus described in
``$ODK_TESTDATA_DIR/INDEX.md`` and checks figures measured from the result. None of the
inputs is synthesised. Without the corpus (``ODK_TESTDATA_DIR`` unset or a dataset
missing) or without pycolmap, the tests skip and say why; they never fail CI that lacks
the data.

* ``odm_bellus_gcp`` (CC0): Canon S110 nadir survey, ``gcp_list.txt`` with a
  ``WGS84 UTM 17N`` header and four control points marked once each. One mark cannot
  be triangulated, so each point is placed by following its ray to the tie-point
  surface and reported as a check point against the GPS georeferencing.
* ``odm_sheffield_cross`` (Unlicense): DJI Phantom 3 oblique survey, ``gcp_file.txt``
  with a PROJ-string header, five points marked in five images each. The model is
  fitted to the control and leave-one-out residuals are the independent figure.
* ``thermal_nvs_praschl_zenodo/W_subset`` (CC BY 4.0, Praschl & Schedl 2025): DJI M30T
  wide frames of a building facade flown at 3.5-13 m with the gimbal at -8 degrees. The
  wall is measured from the model and a facade mission is planned on it.

The figures in docs/features/registry.py were measured with the same code on fuller
sets (tools/run_gcp_reconstruction.py); the thresholds here leave room for the smaller
subsets and for run-to-run variation in RANSAC, not for a different answer.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "")) if os.environ.get("ODK_TESTDATA_DIR") else None


def _dataset(name: str) -> Path:
    if TESTDATA is None:
        pytest.skip("ODK_TESTDATA_DIR is not set; the real-survey corpus is not on this machine.")
    path = TESTDATA / name
    if not path.is_dir():
        pytest.skip(f"{path} is missing from the test corpus.")
    return path


def _require_colmap():
    pytest.importorskip("pycolmap")
    from core.reconstruction_colmap import colmap_available

    if not colmap_available():
        pytest.skip("pycolmap is not usable in this environment.")


def _subset(source: Path, names: list[str], target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        frame = source / name
        if not frame.is_file():
            pytest.skip(f"{frame} is missing from the corpus.")
        try:
            (target / name).hardlink_to(frame)
        except OSError:
            shutil.copy2(frame, target / name)
    return target


def _reconstruct(images: Path, out: Path, control=None, max_image_size: int = 1600):
    from core.reconstruction_colmap import ColmapReconstructor

    # CPU SIFT, as the other real-imagery tests pin it: the GPU detector finds fewer
    # keypoints and decides what registers on small subsets.
    engine = ColmapReconstructor("standard", use_gpu=False, dense=False,
                                 max_image_size=max_image_size, ground_control=control or [])
    result = engine.reconstruct(str(images), output_dir=str(out))
    return engine, result


# ---------------------------------------------------------------------------
# Parsing the real control files
# ---------------------------------------------------------------------------


def test_bellus_control_file_is_read_with_its_marks():
    from core.gcp import read_gcp_file

    points = read_gcp_file(_dataset("odm_bellus_gcp") / "gcp_list.txt")
    assert len(points) == 4
    assert {p.epsg for p in points} == {32617}
    assert all(len(p.marks) == 1 for p in points)
    assert [p.marks[0].image for p in points] == [
        "IMG_1356_RGB.jpg", "IMG_1346_RGB.jpg", "IMG_1382_RGB.jpg", "IMG_1338_RGB.jpg"]


def test_sheffield_control_file_with_a_proj_header_is_read_with_its_marks():
    from core.gcp import read_gcp_file

    points = read_gcp_file(_dataset("odm_sheffield_cross") / "gcp_file.txt")
    assert len(points) == 5
    assert {p.epsg for p in points} == {32617}
    assert all(len(p.marks) == 5 for p in points)
    assert all({m.image for m in p.marks} ==
               {"DJI_0065.JPG", "DJI_0066.JPG", "DJI_0067.JPG", "DJI_0068.JPG", "DJI_0081.JPG"}
               for p in points)


# ---------------------------------------------------------------------------
# Reconstruction with that control
# ---------------------------------------------------------------------------

SHEFFIELD_FRAMES = [f"DJI_{n:04d}.JPG" for n in range(60, 91)]


def test_sheffield_oblique_survey_is_fitted_to_its_control(tmp_path):
    """Five points x five marks: the model is fitted to them, leave-one-out is the check."""
    from core.gcp import read_gcp_file

    _require_colmap()
    root = _dataset("odm_sheffield_cross")
    control = read_gcp_file(root / "gcp_file.txt")
    images = _subset(root / "images", SHEFFIELD_FRAMES, tmp_path / "images")
    engine, result = _reconstruct(images, tmp_path / "run", control)

    assert result.registered_images >= 28, result.warnings
    assert result.reprojection_error_px < 1.5
    report = engine.control_report
    assert report["georeferenced_by"] == "ground_control"
    assert report["used"] == 5
    assert all(v["views"] == 5 and v["method"] == "ray_intersection"
               for v in report["located"].values())
    # Marks reproject onto the model within a few pixels: the marks and the model agree.
    assert max(v["rms_reprojection_px"] for v in report["located"].values()) < 5.0
    assert report["rmse_m"] < 0.5
    loo = report["leave_one_out"]
    assert loo["role"] == "check" and loo["used"] == 5
    assert loo["rmse_m"] < 1.0
    assert (tmp_path / "run" / "gcp_report.json").is_file()


BELLUS_FRAMES = [f"IMG_{n}_RGB.jpg" for n in range(1334, 1387)]


def test_bellus_single_marks_are_placed_on_the_model_and_check_the_gps_fit(tmp_path):
    """One mark per point: placed on the tie-point surface, reported as check points."""
    from core.gcp import read_gcp_file

    _require_colmap()
    root = _dataset("odm_bellus_gcp")
    control = read_gcp_file(root / "gcp_list.txt")
    images = _subset(root / "images", BELLUS_FRAMES, tmp_path / "images")
    engine, result = _reconstruct(images, tmp_path / "run", control)

    assert result.registered_images >= 40, result.warnings
    assert result.reprojection_error_px < 1.5
    report = engine.control_report
    # Too thinly marked to fit, so the geotags georeference and the control checks them.
    assert report["georeferenced_by"] == "camera_gps"
    assert report["role"] == "check"
    assert report["used"] >= 3
    assert all(v["views"] == 1 and v["method"] == "single_ray_on_tie_point_surface"
               for v in report["located"].values())
    assert any("one registered image only" in w for w in report["warnings"])
    # A consumer GPS georeference: metres, not centimetres, and the report says which.
    assert report["horizontal_rmse_m"] < 15.0
    assert report["meets_survey_grade"] is False


# ---------------------------------------------------------------------------
# Facade: reconstruction quality, and a mission on the measured wall
# ---------------------------------------------------------------------------


def test_the_hagenberg_facade_is_measured_and_a_mission_planned_on_it(tmp_path):
    from core.facade_survey import facade_from_run, plan_facade_mission, waypoint_offsets_from_wall

    _require_colmap()
    root = _dataset("thermal_nvs_praschl_zenodo") / "W_subset"
    if not root.is_dir():
        pytest.skip(f"{root} is missing from the corpus.")
    names = sorted(p.name for p in root.glob("*.JPG"))
    images = _subset(root, names, tmp_path / "images")
    _engine, result = _reconstruct(images, tmp_path / "run")

    assert result.registered_images >= 50, result.warnings
    assert result.reprojection_error_px < 1.5
    assert result.crs_epsg == 32633                       # Hagenberg, UTM 33N

    wall = facade_from_run(tmp_path / "run", images)
    assert wall.height_reference == "home"
    # The frames were flown along about 35 m of facade, 3.5 to 13 m above take-off.
    assert 20.0 < wall.length_m < 60.0
    assert wall.top_z > 8.0
    assert wall.inlier_points > 0.3 * wall.total_points
    # The facade runs east-west and the camera looked north (yaw ~1 degree): the wall's
    # outward normal points south, back at the aircraft.
    assert wall.normal_xy[1] < -0.9
    assert 2.0 < wall.camera_standoff_m["median"] < 25.0

    plan, _polygon = plan_facade_mission(wall)
    offsets = waypoint_offsets_from_wall(plan, wall)
    assert len(offsets) > 4
    assert np.all(offsets > 0.5), "a planned waypoint is at or behind the measured wall"
    assert np.median(offsets) == pytest.approx(wall.camera_standoff_m["median"], abs=1.0)
