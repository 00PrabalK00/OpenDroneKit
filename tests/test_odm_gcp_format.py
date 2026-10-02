"""OpenDroneMap's control-file format, and control marked in a single photograph.

ODM's ``gcp_list.txt`` was refused outright ("has no x/y/z column") because the reader
only knew headed CSV. The two open ODM datasets in the test corpus use the two common
header spellings: ``WGS84 UTM 17N`` (bellus) and a PROJ string (sheffield_cross). Each
row is one observation, so the reader also has to group rows into points and keep their
pixel marks -- dropping them would leave every imported point "never marked".

Bellus marks each of its four points in one image only. Two rays are needed to
triangulate, so those points could not be placed at all; they are now placed by
following the single ray to the surface of the model's tie points around the mark.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.gcp import (
    GcpError,
    GroundControlPoint,
    ImageObservations,
    ViewPose,
    add_mark,
    epsg_from_crs_header,
    locate_in_model,
    read_gcp_file,
)


class TestCrsHeader:
    @pytest.mark.parametrize("header,epsg", [
        ("WGS84 UTM 17N", 32617),
        ("WGS84 UTM 33S", 32733),
        ("EPSG:32632", 32632),
        ("epsg: 4326", 4326),
        ("+proj=utm +zone=17 +ellps=WGS84 +datum=WGS84 +units=m +no_defs", 32617),
        ("+proj=utm +zone=56 +south +datum=WGS84 +units=m +no_defs", 32756),
    ])
    def test_each_odm_spelling_resolves(self, header, epsg):
        assert epsg_from_crs_header(header) == epsg

    def test_an_unknown_header_is_refused_not_guessed(self):
        with pytest.raises(GcpError, match="names its CRS"):
            epsg_from_crs_header("Local grid, site datum")

    def test_a_proj_string_with_no_epsg_is_refused(self):
        with pytest.raises(GcpError, match="does not match an EPSG"):
            epsg_from_crs_header("+proj=tmerc +lat_0=12.3 +lon_0=45.6 +k=0.9 +x_0=7 +y_0=8 +ellps=intl +units=m")


def _write(tmp_path, text, name="gcp_list.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestOdmRows:
    def test_rows_with_the_same_position_are_one_point_with_several_marks(self, tmp_path):
        path = _write(tmp_path, "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs\n"
                      "333158.20\t3102916.18\t3\t2699.43\t1668.72\tDJI_0065.JPG\n"
                      "333158.20\t3102916.18\t3\t2706.86\t1960.27\tDJI_0066.JPG\n"
                      "333187.62\t3102954.46\t3\t1434.68\t704.34\tDJI_0065.JPG\n")
        points = read_gcp_file(path)
        assert [p.name for p in points] == ["GCP01", "GCP02"]
        assert points[0].epsg == 32617
        assert [(m.image, m.pixel_x, m.pixel_y) for m in points[0].marks] == [
            ("DJI_0065.JPG", 2699.43, 1668.72), ("DJI_0066.JPG", 2706.86, 1960.27)]
        # Pixels are on the full-resolution photograph; no preview size is implied.
        assert points[0].marks[0].image_width == 0

    def test_a_seventh_column_names_the_point(self, tmp_path):
        path = _write(tmp_path, "EPSG:32617\n"
                      "441024.15 4564001.87 345.17 2648 2147 IMG_1356_RGB.jpg gcp_a\n"
                      "441024.15 4564001.87 345.17 1200 900 IMG_1357_RGB.jpg gcp_a\n")
        (point,) = read_gcp_file(path)
        assert point.name == "gcp_a" and len(point.marks) == 2

    def test_one_name_at_two_positions_is_refused(self, tmp_path):
        path = _write(tmp_path, "EPSG:32617\n"
                      "1 2 3 10 10 a.jpg P\n"
                      "5 2 3 10 10 b.jpg P\n")
        with pytest.raises(GcpError, match="different position"):
            read_gcp_file(path)

    def test_a_short_row_is_refused_with_its_line_number(self, tmp_path):
        path = _write(tmp_path, "WGS84 UTM 17N\n1 2 3 10 10 a.jpg\n1 2 3 10\n")
        # The CRS header marks it as an ODM file, so the error names the bad line
        # instead of the CSV reader's "no x column".
        with pytest.raises(GcpError, match="line 3"):
            read_gcp_file(path)

    def test_unnamed_points_do_not_take_a_name_a_named_row_already_has(self, tmp_path):
        path = _write(tmp_path, "EPSG:32617\n"
                      "1 2 3 10 10 a.jpg GCP01\n"
                      "7 8 9 10 10 b.jpg\n")
        points = read_gcp_file(path)
        assert [(p.name, p.x) for p in points] == [("GCP01", 1.0), ("GCP02", 7.0)]

    def test_headed_csv_still_reads_as_before(self, tmp_path):
        path = _write(tmp_path, "name,easting,northing,elevation,epsg\nT1,1,2,3,32617\n",
                      name="control.csv")
        (point,) = read_gcp_file(path)
        assert point.name == "T1" and point.marks == []


def test_importing_an_odm_file_keeps_its_marks_and_merges_earlier_ones(tmp_path):
    """Import replaced every point's marks with the project's previous ones (none)."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(store=ProjectStore(tmp_path / "p.db")))
    api.create_project("ODM control", str(tmp_path / "project"))
    path = _write(tmp_path, "WGS84 UTM 17N\n"
                  "441024.15 4564001.87 345.17 2648 2147 IMG_1356_RGB.jpg\n"
                  "441004.67 4564119.58 351.33 1678 1442 IMG_1346_RGB.jpg\n")
    result = api.import_gcps(str(path))
    assert result["ok"] and result["count"] == 2 and result["marks"] == 2
    assert api.mark_gcp("GCP01", "IMG_1357_RGB.jpg", 100, 200)["ok"]
    again = api.import_gcps(str(path))
    listed = {p["name"]: p for p in api.list_gcps()["points"]}
    assert again["ok"]
    assert sorted(m["image"] for m in listed["GCP01"]["marks"]) == [
        "IMG_1356_RGB.jpg", "IMG_1357_RGB.jpg"]


class TestSingleMark:
    """A nadir camera 50 m above a flat tie-point surface at z = 0."""

    K = np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]])
    NADIR = np.diag([1.0, -1.0, -1.0])

    def pose(self):
        centre = np.array([10.0, 20.0, 50.0])
        return ViewPose(image="a.jpg", rotation=self.NADIR, translation=-self.NADIR @ centre,
                        intrinsics=self.K, width=1000, height=800)

    def surface(self, pose, tilt=0.0):
        rng = np.random.default_rng(1)
        ground = np.column_stack([rng.uniform(-10, 30, 400), rng.uniform(0, 40, 400),
                                  np.zeros(400)])
        ground[:, 2] = tilt * ground[:, 0]
        pixels = np.array([pose.project(p) for p in ground])
        return ImageObservations(pixels=pixels, points=ground)

    def test_the_ray_lands_on_the_surface_under_the_mark(self):
        pose = self.pose()
        target = np.array([14.0, 17.0, 0.0])
        u, v = pose.project(target)
        point = add_mark(GroundControlPoint("G", 0, 0, 0, 32617), "a.jpg", u, v)
        found = locate_in_model(point, {"a.jpg": pose}, {"a.jpg": self.surface(pose)})
        assert found is not None and found.views == 1
        assert found.method == "single_ray_on_tie_point_surface"
        assert found.position == pytest.approx(target, abs=1e-6)
        assert found.to_dict()["surface_tie_points"] >= 6

    def test_a_sloped_surface_is_followed_not_flattened(self):
        pose = self.pose()
        target = np.array([20.0, 25.0, 0.1 * 20.0])
        u, v = pose.project(target)
        point = add_mark(GroundControlPoint("G", 0, 0, 0, 32617), "a.jpg", u, v)
        found = locate_in_model(point, {"a.jpg": pose}, {"a.jpg": self.surface(pose, tilt=0.1)})
        assert found.position == pytest.approx(target, abs=1e-6)

    def test_without_tie_points_a_single_mark_is_not_placed(self):
        pose = self.pose()
        point = add_mark(GroundControlPoint("G", 0, 0, 0, 32617), "a.jpg", 500, 400)
        assert locate_in_model(point, {"a.jpg": pose}) is None

    def test_a_single_mark_never_takes_part_in_the_fit(self):
        """Fit needs MIN_IMAGE_MARKS views; one ray on a surface is a check point only."""
        from core.gcp import solve_with_control

        pose = self.pose()
        points = []
        for i, (x, y) in enumerate([(12, 18), (16, 24), (8, 22)]):
            u, v = pose.project(np.array([x, y, 0.0]))
            points.append(add_mark(GroundControlPoint(f"G{i}", x, y, 0, 32617), "a.jpg", u, v))
        solution = solve_with_control(points, {"a.jpg": pose}, 32617, None,
                                      observations={"a.jpg": self.surface(pose)})
        assert solution.anchor is None
        assert all(v["views"] == 1 for v in solution.report["located"].values())
        assert math.isclose(len(solution.report["located"]), 3)


def test_a_shared_vertical_offset_is_named_as_a_datum_question():
    """Bellus's check points all sat ~29 m low against the GPS fit: a datum, not noise."""
    from core.gcp import GcpResidual, accuracy_report

    points = [add_mark(GroundControlPoint(f"G{i}", 0, 0, 0, 32617), "a.jpg", 1, 1)
              for i in range(3)]
    residuals = [GcpResidual(f"G{i}", dx, dy, dz, 1)
                 for i, (dx, dy, dz) in enumerate([(1.4, -1.6, -29.0), (2.7, 6.3, -29.9),
                                                    (-0.5, 2.0, -29.4)])]
    report = accuracy_report(points, residuals, role="check")
    assert report["mean_offset_m"][2] == pytest.approx(-29.433, abs=1e-3)
    assert any("vertical datum" in w for w in report["warnings"])
    # Scattered errors with no common offset are not called a datum problem.
    scattered = [GcpResidual(f"G{i}", 0, 0, dz, 1) for i, dz in enumerate([-3.0, 2.5, 0.4])]
    assert not any("vertical datum" in w
                   for w in accuracy_report(points, scattered, role="check")["warnings"])


class TestThinControlCorrectsTheGeotagFit:
    """Single-mark control cannot carry a 7-parameter fit but can fix position and size.

    On the real bellus survey the geotag fit put every check point ~31.6 m low (heights in
    another datum) and the model 4.2 per cent small; correcting scale and translation onto
    the four single-mark points took the leave-one-out RMSE from 31.9 m to 1.57 m.
    Here the geotag fit is wrong by a known scale and offset and the surface is exact,
    so the correction must recover it exactly.
    """

    ORIGIN = [440000.0, 4564000.0, 300.0]
    MARKS = [(12, 18), (16, 24), (8, 22), (14, 30)]

    def control(self, pose, origin):
        points = []
        for i, (x, y) in enumerate(self.MARKS):
            u, v = pose.project(np.array([x, y, 0.0]))
            points.append(add_mark(GroundControlPoint(
                f"G{i}", origin[0] + x, origin[1] + y, origin[2], 32617), "a.jpg", u, v))
        return points

    def anchor(self, scale, translation):
        from core.geo import GeoAnchor

        return GeoAnchor(scale=scale, rotation=np.eye(3).tolist(), translation=list(translation),
                         epsg=32617, origin=list(self.ORIGIN), inlier_count=20,
                         sample_count=20, rmse_m=2.0, max_residual_m=4.0)

    def test_a_wrong_geotag_scale_and_datum_are_corrected(self):
        from core.gcp import solve_with_control

        single = TestSingleMark()
        pose = single.pose()
        camera = self.anchor(1 / 1.042, [1.4, -0.9, -31.6])
        solution = solve_with_control(self.control(pose, self.ORIGIN), {"a.jpg": pose}, 32617,
                                      camera, observations={"a.jpg": single.surface(pose)})
        report = solution.report
        assert report["georeferenced_by"] == "camera_gps_shifted_to_control"
        assert report["method"] == "scale_and_translation"
        assert report["scale_factor"] == pytest.approx(1.042, abs=1e-6)
        assert report["camera_gps_check"]["vertical_rmse_m"] > 30
        assert report["leave_one_out"]["role"] == "check"
        assert report["leave_one_out"]["rmse_m"] == pytest.approx(0.0, abs=1e-6)
        # The anchor the products are written with puts each point where it was surveyed.
        placed = solution.anchor.apply(np.array([[x, y, 0.0] for x, y in self.MARKS]))
        assert placed == pytest.approx(
            np.array([[self.ORIGIN[0] + x, self.ORIGIN[1] + y, self.ORIGIN[2]]
                      for x, y in self.MARKS]), abs=1e-6)
        assert solution.anchor.rotation == np.eye(3).tolist()

    def test_an_exact_geotag_fit_is_left_alone(self):
        from core.gcp import solve_with_control

        single = TestSingleMark()
        pose = single.pose()
        camera = self.anchor(1.0, [0.0, 0.0, 0.0])
        solution = solve_with_control(self.control(pose, self.ORIGIN), {"a.jpg": pose}, 32617,
                                      camera, observations={"a.jpg": single.surface(pose)})
        assert solution.anchor is None
        assert solution.report["georeferenced_by"] == "camera_gps"
        assert any("would not improve" in w
                   for w in solution.report["control_shift"]["warnings"])

    def test_two_points_are_too_few_to_correct_anything(self):
        from core.gcp import solve_with_control

        single = TestSingleMark()
        pose = single.pose()
        camera = self.anchor(1 / 1.042, [1.4, -0.9, -31.6])
        solution = solve_with_control(self.control(pose, self.ORIGIN)[:2], {"a.jpg": pose},
                                      32617, camera, observations={"a.jpg": single.surface(pose)})
        assert solution.anchor is None
        assert "control_shift" not in solution.report
