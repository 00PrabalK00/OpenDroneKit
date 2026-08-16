"""The measurement capabilities as an operator reaches them.

The module-level tests prove the maths. These prove the capability: that the API exposes
it, that a refusal reaches the caller as a legible message rather than a traceback, and
that the guarantees survive the trip through the boundary.

A module that computes correctly but is not reachable is not a capability, and the
registry treats it that way.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.api import Api

# 20x20 m building with a 10x10 bite out of the corner; the vertex at (10,10) is reflex.
L_SHAPE = [[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]]


@pytest.fixture
def api() -> Api:
    return Api()


def _write_surface(tmp_path, name: str, elevation: np.ndarray, epsg: int = 32643) -> str:
    """A small projected GeoTIFF, or skip when rasterio is unavailable."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / name
    with rasterio.open(
        path, "w", driver="GTiff", height=elevation.shape[0], width=elevation.shape[1],
        count=1, dtype="float32", crs=f"EPSG:{epsg}",
        transform=from_origin(500000.0, 3000000.0, 1.0, 1.0),
    ) as dst:
        dst.write(elevation.astype("float32"), 1)
    return str(path)


class TestPondingCapability:
    def test_it_refuses_without_a_vertical_accuracy(self, api: Api, tmp_path) -> None:
        # The refusal an operator must see: without an error bound there is no way to
        # say which basins are real.
        surface = _write_surface(tmp_path, "roof.tif", np.full((30, 30), 10.0))
        result = api.find_ponding(surface)
        assert result["ok"] is False
        assert "accuracy" in result["error"].lower()

    def test_a_flat_roof_reports_no_ponding(self, api: Api, tmp_path) -> None:
        surface = _write_surface(tmp_path, "flat.tif", np.full((30, 30), 10.0))
        result = api.find_ponding(surface, vertical_accuracy_m=0.02)
        assert result["ok"] is True
        assert result["count"] == 0

    def test_a_real_basin_is_measured_and_bounded(self, api: Api, tmp_path) -> None:
        elevation = np.full((30, 30), 10.0)
        elevation[10:20, 10:20] -= 0.08
        surface = _write_surface(tmp_path, "basin.tif", elevation)
        result = api.find_ponding(surface, vertical_accuracy_m=0.01)
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["depressions"][0]["max_depth_m"] == pytest.approx(0.08, abs=1e-3)
        assert result["detection_floor_m"] > 0

    def test_the_answer_says_it_is_capacity_not_observation(self, api: Api, tmp_path) -> None:
        elevation = np.full((30, 30), 10.0)
        elevation[10:20, 10:20] -= 0.08
        surface = _write_surface(tmp_path, "note.tif", elevation)
        result = api.find_ponding(surface, vertical_accuracy_m=0.01)
        assert "not whether water is" in result["note"]


class TestDeformationCapability:
    def test_it_refuses_without_uncertainties(self, api: Api, tmp_path) -> None:
        first = _write_surface(tmp_path, "a.tif", np.full((30, 30), 100.0))
        second = _write_surface(tmp_path, "b.tif", np.full((30, 30), 100.0))
        result = api.compare_surveys(first, second)
        assert result["ok"] is False
        assert "accuracy" in result["error"].lower()

    def test_unchanged_ground_reports_no_movement(self, api: Api, tmp_path) -> None:
        first = _write_surface(tmp_path, "a.tif", np.full((30, 30), 100.0))
        second = _write_surface(tmp_path, "b.tif", np.full((30, 30), 100.0))
        result = api.compare_surveys(
            first, second, earlier_accuracy_m=0.03, later_accuracy_m=0.03,
            registration_residual_m=0.01)
        assert result["ok"] is True
        assert result["moved"] is False

    def test_subsidence_is_found_with_its_floor_stated(self, api: Api, tmp_path) -> None:
        later = np.full((30, 30), 100.0)
        later[10:20, 10:20] -= 0.5
        first = _write_surface(tmp_path, "a.tif", np.full((30, 30), 100.0))
        second = _write_surface(tmp_path, "b.tif", later)
        result = api.compare_surveys(
            first, second, earlier_accuracy_m=0.02, later_accuracy_m=0.02,
            registration_residual_m=0.01)
        assert result["ok"] is True
        assert result["regions"][0]["direction"] == "subsidence"
        assert "not that none occurred" in result["note"]


class TestIrregularFacadeCapability:
    def test_an_l_shape_plans_and_names_its_reflex_corner(self, api: Api) -> None:
        result = api.plan_irregular_facade(L_SHAPE, standoff_m=4.0)
        assert result["ok"] is True
        assert result["footprint"]["reflex_count"] == 1
        assert result["segment_count"] > 0

    def test_no_planned_pass_lies_inside_the_building(self, api: Api) -> None:
        # The failure this capability exists to prevent, checked at the API boundary.
        from mission.footprints import point_in_polygon

        polygon = [(float(x), float(y)) for x, y in L_SHAPE]
        result = api.plan_irregular_facade(L_SHAPE, standoff_m=4.0)
        for segment in result["segments"]:
            for point in (segment["start"], segment["end"]):
                assert not point_in_polygon((point[0], point[1]), polygon)

    def test_degrees_are_refused_with_a_useful_message(self, api: Api) -> None:
        result = api.plan_irregular_facade(
            [[77.5, 12.9], [77.5001, 12.9], [77.5001, 12.9001]], standoff_m=5.0)
        assert result["ok"] is False
        assert "degrees" in result["error"]


class TestDemoCapability:
    def test_the_whole_workflow_is_marked_synthetic(self, api: Api) -> None:
        result = api.demo_workflow()
        assert result["ok"] is True
        assert result["synthetic"] is True

    def test_findings_stay_marked_when_taken_alone(self, api: Api) -> None:
        # The real risk: someone lifting one finding out of the demo and passing it on.
        for finding in api.demo_workflow()["findings"]:
            assert finding["synthetic"] is True

    def test_demo_output_is_refused_by_the_publishing_guard(self, api: Api) -> None:
        from core.demo_mode import DemoDataRefused, refuse_if_demo

        with pytest.raises(DemoDataRefused):
            refuse_if_demo(api.demo_workflow(), action="publish an inspection report")


class TestReconstructionCapabilities:
    """Dense reconstruction is a dependency question, answered before a job starts.

    pr.dense was blocked for a real reason: patch-match stereo needs a CUDA COLMAP, and
    the previous code faked it by cloning sparse points with Gaussian jitter. That fake
    was removed and tests/test_honesty.py guards its absence. What was missing is a way
    for a caller to ASK, rather than discovering it partway through a long job.
    """

    def test_it_reports_whether_dense_is_possible(self, api: Api) -> None:
        result = api.reconstruction_capabilities()
        assert result["ok"] is True
        assert isinstance(result["dense_available"], bool)

    def test_the_note_matches_the_answer(self, api: Api) -> None:
        result = api.reconstruction_capabilities()
        if result["dense_available"]:
            assert "is available" in result["note"]
        else:
            assert "NOT available" in result["note"]

    def test_it_never_promises_densification_from_a_sparse_cloud(self, api: Api) -> None:
        # The honesty rule this capability exists to hold: no post-processing turns a
        # sparse cloud into a dense one, and the report must not imply otherwise.
        result = api.reconstruction_capabilities()
        if not result["dense_available"]:
            assert "never inflated" in result["note"]

    def test_the_underlying_capability_flags_are_exposed(self, api: Api) -> None:
        # A user debugging a deployment needs to see WHICH piece is missing.
        caps = api.reconstruction_capabilities()["capabilities"]
        for key in ("pycolmap", "colmap_binary", "pycolmap_cuda", "dense_stereo"):
            assert key in caps


class TestSpatialReferenceCapability:
    """GPS-denied reconstruction, and the refusal that keeps it honest.

    The model reconstructs fine without geotags. What it cannot do is carry a metre, and
    a user who measures in it gets numbers wrong by an unknown factor with nothing on
    screen to say so.
    """

    def test_geotagged_imagery_is_reported_measurable(self, api: Api, tmp_path, monkeypatch) -> None:
        from core import geo

        paths = []
        for i in range(10):
            p = tmp_path / f"g{i}.jpg"
            p.write_bytes(b"stub")
            paths.append(str(p))
        monkeypatch.setattr(geo, "read_exif_gps", lambda path: object())

        result = api.check_spatial_reference(paths, epsg=32643)
        assert result["ok"] is True
        assert result["mode"] == "georeferenced"
        assert result["measurements_allowed"] is True

    def test_gps_denied_imagery_refuses_measurement(self, api: Api, tmp_path, monkeypatch) -> None:
        from core import geo

        paths = []
        for i in range(10):
            p = tmp_path / f"n{i}.jpg"
            p.write_bytes(b"stub")
            paths.append(str(p))
        monkeypatch.setattr(geo, "read_exif_gps", lambda path: None)

        result = api.check_spatial_reference(paths)
        assert result["ok"] is True, "assessment succeeds; it is the MEASURING that is refused"
        assert result["mode"] == "arbitrary"
        assert result["measurements_allowed"] is False
        assert "SCALE" in result["note"]

    def test_control_points_restore_measurability(self, api: Api, tmp_path, monkeypatch) -> None:
        from core import geo

        paths = []
        for i in range(10):
            p = tmp_path / f"c{i}.jpg"
            p.write_bytes(b"stub")
            paths.append(str(p))
        monkeypatch.setattr(geo, "read_exif_gps", lambda path: None)

        result = api.check_spatial_reference(paths, gcp_count=4)
        assert result["mode"] == "control_referenced"
        assert result["measurements_allowed"] is True

    def test_no_images_is_refused(self, api: Api) -> None:
        result = api.check_spatial_reference([])
        assert result["ok"] is False
        assert "nothing to reconstruct" in result["error"]
