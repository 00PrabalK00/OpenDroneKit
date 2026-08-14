"""Ground control points and the accuracy they establish.

A reconstruction fitted to control points always produces a transform. Whether that
transform is any good is a separate question, and the only honest answer is the
residuals. So these tests check that residuals are reported rather than summarised into
a verdict, that a survey with no checked control refuses to quote an accuracy at all,
and that the two failure modes which produce plausible wrong answers -- a point marked
in too few images, and a point marked on the wrong target -- are surfaced instead of
averaged away.
"""

from __future__ import annotations

import pytest

from core.gcp import (
    GOOD_RMSE_M,
    MIN_IMAGE_MARKS,
    OUTLIER_RESIDUAL_M,
    GcpError,
    GroundControlPoint,
    accuracy_report,
    add_mark,
    read_gcp_file,
    residuals_from_positions,
    write_report,
)

EPSG = 32617


def point(name: str, x=437000.0, y=4572900.0, z=280.0, accuracy=0.02):
    return GroundControlPoint(name=name, x=x, y=y, z=z, epsg=EPSG, accuracy_m=accuracy)


def marked(name: str, marks: int = MIN_IMAGE_MARKS, **kwargs):
    gcp = point(name, **kwargs)
    for i in range(marks):
        add_mark(gcp, f"DSC{i:05d}.JPG", 100.0 + i, 200.0 + i)
    return gcp


def write_csv(tmp_path, text: str, name: str = "gcps.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestReading:
    def test_a_headed_file_is_read_by_column_name(self, tmp_path):
        path = write_csv(tmp_path, "name,x,y,z,epsg\n"
                                   "T1,437000.12,4572900.44,281.20,32617\n"
                                   "T2,437100.90,4572950.10,283.05,32617\n")
        points = read_gcp_file(path)

        assert [p.name for p in points] == ["T1", "T2"]
        assert points[0].x == pytest.approx(437000.12)
        assert points[0].epsg == EPSG

    def test_surveyor_column_names_are_understood(self, tmp_path):
        path = write_csv(tmp_path, "id,easting,northing,elevation\n"
                                   "A,437000.0,4572900.0,280.0\n"
                                   "B,437010.0,4572910.0,281.0\n"
                                   "C,437020.0,4572920.0,282.0\n")
        points = read_gcp_file(path, default_epsg=EPSG)
        assert len(points) == 3
        assert points[0].y == pytest.approx(4572900.0)

    def test_a_file_without_a_header_is_refused(self, tmp_path):
        """Easting and northing are both large and plausible in either column."""
        path = write_csv(tmp_path, "437000.0,4572900.0,280.0\n437010.0,4572910.0,281.0\n")
        with pytest.raises(GcpError, match="no .* column"):
            read_gcp_file(path, default_epsg=EPSG)

    def test_a_file_with_no_crs_and_none_supplied_is_refused(self, tmp_path):
        """The same numbers name different places in different projections."""
        path = write_csv(tmp_path, "name,x,y,z\nT1,437000.0,4572900.0,280.0\n")
        with pytest.raises(GcpError, match="without one"):
            read_gcp_file(path)

    def test_declared_accuracy_is_kept(self, tmp_path):
        path = write_csv(tmp_path, "name,x,y,z,accuracy\nT1,437000,4572900,280,0.15\n")
        assert read_gcp_file(path, default_epsg=EPSG)[0].accuracy_m == pytest.approx(0.15)

    def test_a_missing_file_is_refused_clearly(self, tmp_path):
        with pytest.raises(GcpError, match="does not exist"):
            read_gcp_file(tmp_path / "absent.csv")

    def test_a_header_with_no_numeric_rows_is_refused(self, tmp_path):
        path = write_csv(tmp_path, "name,x,y,z\nT1,north,west,up\n")
        with pytest.raises(GcpError, match="no numeric rows"):
            read_gcp_file(path, default_epsg=EPSG)


class TestMarking:
    def test_a_mark_records_where_the_point_was_identified(self):
        gcp = point("T1")
        add_mark(gcp, "DSC00001.JPG", 512.0, 384.0)

        assert len(gcp.marks) == 1
        assert gcp.marks[0].pixel_x == pytest.approx(512.0)

    def test_marking_the_same_image_twice_corrects_rather_than_duplicates(self):
        gcp = point("T1")
        add_mark(gcp, "DSC00001.JPG", 512.0, 384.0)
        add_mark(gcp, "DSC00001.JPG", 520.0, 390.0)

        assert len(gcp.marks) == 1
        assert gcp.marks[0].pixel_x == pytest.approx(520.0)

    def test_a_point_needs_three_images_to_be_usable(self):
        """Two rays intersect wherever they are aimed."""
        gcp = point("T1")
        add_mark(gcp, "a.JPG", 1, 1)
        add_mark(gcp, "b.JPG", 2, 2)
        assert gcp.is_usable is False

        add_mark(gcp, "c.JPG", 3, 3)
        assert gcp.is_usable is True


class TestResiduals:
    def test_a_perfect_fit_has_zero_residual(self):
        points = [marked("T1")]
        computed = {"T1": (points[0].x, points[0].y, points[0].z)}

        residuals = residuals_from_positions(points, computed)
        assert residuals[0].total_m == pytest.approx(0.0)

    def test_a_residual_measures_the_offset_in_each_axis(self):
        points = [marked("T1")]
        computed = {"T1": (points[0].x + 0.03, points[0].y - 0.04, points[0].z + 0.10)}

        residual = residuals_from_positions(points, computed)[0]
        assert residual.horizontal_m == pytest.approx(0.05, abs=1e-6)
        assert residual.dz == pytest.approx(0.10)
        assert residual.total_m == pytest.approx(0.1118, abs=1e-3)

    def test_a_point_the_reconstruction_never_placed_produces_no_residual(self):
        points = [marked("T1"), marked("T2")]
        residuals = residuals_from_positions(points, {"T1": (points[0].x, points[0].y, points[0].z)})
        assert [r.name for r in residuals] == ["T1"]


class TestAccuracyReport:
    def _fit(self, offsets):
        points = [marked(f"T{i}") for i in range(len(offsets))]
        computed = {
            p.name: (p.x + dx, p.y + dy, p.z + dz)
            for p, (dx, dy, dz) in zip(points, offsets)
        }
        return points, residuals_from_positions(points, computed)

    def test_a_tight_fit_reports_a_small_rmse(self):
        points, residuals = self._fit([(0.01, 0.01, 0.02)] * 5)
        report = accuracy_report(points, residuals)

        assert report["rmse_m"] < 0.05
        assert report["used"] == 5

    def test_the_report_names_the_worst_point(self):
        points, residuals = self._fit([(0.01, 0.0, 0.0), (0.4, 0.0, 0.0), (0.01, 0.0, 0.0)])
        report = accuracy_report(points, residuals)
        assert report["worst_point"] == "T1"

    def test_an_outlier_is_flagged_as_a_marking_mistake(self):
        """A point metres out was marked on the wrong target, not badly reconstructed."""
        points, residuals = self._fit([(0.01, 0.0, 0.0), (3.0, 0.0, 0.0)])
        report = accuracy_report(points, residuals)

        assert report["outlier_count"] == 1
        assert any("wrong target" in w for w in report["warnings"])

    def test_horizontal_and_vertical_accuracy_are_reported_separately(self):
        """They fail differently, and a client's tolerance usually differs too."""
        points, residuals = self._fit([(0.0, 0.0, 0.20)] * 3)
        report = accuracy_report(points, residuals)

        assert report["vertical_rmse_m"] == pytest.approx(0.20, abs=1e-6)
        assert report["horizontal_rmse_m"] == pytest.approx(0.0, abs=1e-9)

    def test_no_residuals_means_no_accuracy_may_be_quoted(self):
        """A survey with no checked control is not a controlled survey."""
        report = accuracy_report([marked("T1")], [])

        assert report["rmse_m"] is None
        assert any("Do not quote one" in w for w in report["warnings"])

    def test_unmarked_points_are_reported_as_taking_no_part(self):
        points = [marked("T1"), point("T2")]
        computed = {"T1": (points[0].x, points[0].y, points[0].z)}
        report = accuracy_report(points, residuals_from_positions(points, computed))

        assert any("never marked" in w for w in report["warnings"])

    def test_thinly_marked_points_are_reported_as_untriangulable(self):
        thin = marked("T2", marks=2)
        points = [marked("T1"), thin]
        computed = {p.name: (p.x, p.y, p.z) for p in points}
        report = accuracy_report(points, residuals_from_positions(points, computed))

        assert any("fewer than" in w for w in report["warnings"])

    def test_an_rmse_better_than_the_control_itself_is_flagged(self):
        """A survey cannot be more accurate than the points it was fitted to."""
        points = [marked(f"T{i}", accuracy=0.30) for i in range(4)]
        computed = {p.name: (p.x, p.y, p.z) for p in points}
        report = accuracy_report(points, residuals_from_positions(points, computed))

        assert any("cannot be more accurate" in w for w in report["warnings"])

    def test_the_note_warns_that_fitted_points_flatter_the_result(self):
        points, residuals = self._fit([(0.01, 0.01, 0.01)] * 3)
        note = accuracy_report(points, residuals)["note"]

        assert "flatter the result" in note
        assert "independent check points" in note

    def test_survey_grade_is_reported_as_a_common_expectation_not_a_verdict(self):
        points, residuals = self._fit([(0.01, 0.0, 0.0)] * 3)
        report = accuracy_report(points, residuals)

        assert report["meets_survey_grade"] is True
        assert f"{GOOD_RMSE_M}" in report["note"]
        assert "not as this job's tolerance" in report["note"]


class TestReportOutput:
    def test_a_report_can_be_written_and_read_back(self, tmp_path):
        import json

        points = [marked("T1")]
        computed = {"T1": (points[0].x + 0.02, points[0].y, points[0].z)}
        report = accuracy_report(points, residuals_from_positions(points, computed))

        path = write_report(report, tmp_path / "gcp_report.json")
        assert json.loads(path.read_text(encoding="utf-8"))["used"] == 1
