"""India-first survey intelligence: measured change, location, and deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from core import geo
from core.change_detection import IncomparableSurveys, compare_surfaces
from core.processing_runs import (
    STATUS_COMPLETED,
    create_processing_run,
    get_processing_status,
    run_pipeline,
    validate_pipeline_inputs,
)
from core.survey_intelligence import (
    create_selected_roi_change_package,
    create_surface_change_package,
)
from core.workflows import get_workflow_template, validate_workflow_readiness


GRID = 40
PIXEL = 0.5
GROUND = 100.0
ORIGIN_X = 500_000.0
ORIGIN_Y = 2_100_000.0


def write_surface(path: Path, elevation: np.ndarray, *, west: float = ORIGIN_X) -> Path:
    geo.write_geotiff(
        path,
        elevation.astype(np.float32),
        epsg=32643,
        west=west,
        north=ORIGIN_Y,
        pixel_size=PIXEL,
        cog=False,
    )
    return path


def grid_polygon(row: int, col: int, height: int, width: int) -> list[list[float]]:
    west = ORIGIN_X + col * PIXEL
    east = west + width * PIXEL
    north = ORIGIN_Y - row * PIXEL
    south = north - height * PIXEL
    return [[west, north], [east, north], [east, south], [west, south]]


@pytest.fixture
def changed_surveys(tmp_path):
    earlier = np.full((GRID, GRID), GROUND)
    later = earlier.copy()
    later[4:14, 5:15] += 2.0   # 25 m2 * 2 m = 50 m3 rise
    later[22:32, 20:30] -= 1.0  # 25 m2 * 1 m = 25 m3 fall
    return (
        write_surface(tmp_path / "earlier.tif", earlier),
        write_surface(tmp_path / "later.tif", later),
    )


class TestSurfaceChangePackage:
    def test_measures_and_maps_where_the_surface_changed(self, tmp_path, changed_surveys):
        earlier, later = changed_surveys
        package = create_surface_change_package(
            earlier,
            later,
            tmp_path / "deliverable",
            change_threshold_m=0.10,
            min_region_area_m2=1.0,
        )

        assert package.surface_change.added_volume_m3 == pytest.approx(50.0)
        assert package.surface_change.removed_volume_m3 == pytest.approx(25.0)
        assert package.surface_change.net_volume_m3 == pytest.approx(25.0)
        assert package.surface_change.changed_area_m2 == pytest.approx(50.0)

        layer = json.loads(Path(package.regions_geojson_path).read_text(encoding="utf-8"))
        assert layer["crs"]["properties"]["name"].endswith("EPSG::32643")
        assert len(layer["features"]) == 2
        by_change = {
            feature["properties"]["change"]: feature["properties"]
            for feature in layer["features"]
        }
        assert by_change["surface_rise"]["area_m2"] == pytest.approx(25.0)
        assert by_change["surface_rise"]["volume_m3"] == pytest.approx(50.0)
        assert by_change["surface_fall"]["area_m2"] == pytest.approx(25.0)
        assert by_change["surface_fall"]["volume_m3"] == pytest.approx(25.0)

    def test_every_artifact_is_real_and_the_raster_keeps_its_grid(
        self, tmp_path, changed_surveys
    ):
        earlier, later = changed_surveys
        package = create_surface_change_package(earlier, later, tmp_path / "deliverable")

        assert all(Path(path).is_file() for path in package.artifact_paths())
        difference, metadata = geo.read_geotiff(package.difference_raster_path)
        _, later_metadata = geo.read_geotiff(later)
        assert metadata["epsg"] == 32643
        assert metadata["transform"] == later_metadata["transform"]
        assert difference[0, 4, 5] == pytest.approx(2.0)
        assert difference[0, 22, 20] == pytest.approx(-1.0)

    def test_summary_and_report_do_not_invent_a_semantic_cause(
        self, tmp_path, changed_surveys
    ):
        package = create_surface_change_package(
            *changed_surveys, tmp_path / "deliverable"
        )
        summary = json.loads(Path(package.summary_path).read_text(encoding="utf-8"))
        report = Path(package.report_path).read_text(encoding="utf-8")

        assert summary["analysis"] == "measured_surface_change"
        assert summary["units"] == {
            "elevation": "m",
            "area": "m2",
            "volume": "m3",
        }
        assert "Semantic causes require independent evidence" in summary["interpretation"]
        assert "does not prove the cause" in report
        assert "must not be relabelled as new construction" in report

    def test_noise_below_the_mapping_threshold_is_not_drawn(self, tmp_path):
        earlier = write_surface(
            tmp_path / "earlier.tif", np.full((GRID, GRID), GROUND)
        )
        noisy = np.full((GRID, GRID), GROUND + 0.02)
        later = write_surface(tmp_path / "later.tif", noisy)

        package = create_surface_change_package(
            earlier,
            later,
            tmp_path / "deliverable",
            change_threshold_m=0.05,
        )
        assert package.surface_change.changed_area_m2 == 0.0
        assert package.regions == ()

    def test_shifted_grids_are_refused_even_when_shape_and_resolution_match(self, tmp_path):
        values = np.full((GRID, GRID), GROUND)
        earlier = write_surface(tmp_path / "earlier.tif", values)
        shifted = write_surface(
            tmp_path / "shifted.tif", values, west=ORIGIN_X + PIXEL
        )

        with pytest.raises(IncomparableSurveys, match="grid origins"):
            compare_surfaces(earlier, shifted)


class TestSelectedROIChangePackage:
    def test_stockpile_roi_excludes_change_outside_the_selection(
        self, tmp_path, changed_surveys
    ):
        package = create_selected_roi_change_package(
            *changed_surveys,
            tmp_path / "roi",
            polygon_xy=grid_polygon(4, 5, 10, 10),
            roi_type="stockpile",
            roi_name="Pile A",
            change_threshold_m=0.10,
        )

        assert package.surface_change.added_volume_m3 == pytest.approx(50.0)
        assert package.surface_change.removed_volume_m3 == 0.0
        assert package.surface_change.changed_area_m2 == pytest.approx(25.0)
        assert len(package.artifact_paths()) == 5
        assert all(Path(path).is_file() for path in package.artifact_paths())

        difference, _ = geo.read_geotiff(package.difference_raster_path)
        assert difference[0, 4, 5] == pytest.approx(2.0)
        assert np.isnan(difference[0, 22, 20])

        selection = json.loads(
            Path(package.selection_geojson_path).read_text(encoding="utf-8")
        )
        properties = selection["features"][0]["properties"]
        assert properties["roi_type"] == "stockpile"
        assert properties["roi_name"] == "Pile A"

    def test_pit_roi_measures_only_the_selected_surface_fall(
        self, tmp_path, changed_surveys
    ):
        package = create_selected_roi_change_package(
            *changed_surveys,
            tmp_path / "pit",
            polygon_xy=grid_polygon(22, 20, 10, 10),
            roi_type="pit",
        )
        assert package.surface_change.added_volume_m3 == 0.0
        assert package.surface_change.removed_volume_m3 == pytest.approx(25.0)
        assert package.surface_change.net_volume_m3 == pytest.approx(-25.0)

    def test_roi_that_does_not_overlap_the_surveys_is_refused(
        self, tmp_path, changed_surveys
    ):
        outside = [
            [ORIGIN_X + 1000, ORIGIN_Y + 1000],
            [ORIGIN_X + 1010, ORIGIN_Y + 1000],
            [ORIGIN_X + 1010, ORIGIN_Y + 990],
            [ORIGIN_X + 1000, ORIGIN_Y + 990],
        ]
        with pytest.raises(IncomparableSurveys, match="inside the selected ROI"):
            create_selected_roi_change_package(
                *changed_surveys,
                tmp_path / "outside",
                polygon_xy=outside,
            )


class TestConstructionProgressWorkflow:
    def test_the_mission_pack_declares_the_real_change_stage(self):
        workflow = get_workflow_template("construction_progress")
        assert workflow.asset_type == "construction_site"
        assert workflow.required_inputs == ["earlier_dsm", "later_dsm"]
        assert workflow.processing_stages == ["survey_change"]

    def test_readiness_requires_two_existing_georeferenced_inputs(
        self, tmp_path, changed_surveys
    ):
        missing = validate_workflow_readiness(
            {"root_dir": str(tmp_path)}, "construction_progress"
        )
        assert missing.ready is False
        assert len(missing.missing_required) == 2

        earlier, later = changed_surveys
        ready = validate_workflow_readiness(
            {
                "root_dir": str(tmp_path),
                "earlier_dsm": str(earlier),
                "later_dsm": str(later),
            },
            "construction_progress",
        )
        assert ready.ready is True

    def test_processing_run_emits_the_complete_client_package(
        self, tmp_path, changed_surveys
    ):
        earlier, later = changed_surveys
        project_root = tmp_path / "project"
        run = create_processing_run(
            project_root,
            project_id="site-1",
            dataset_id="",
            workflow_id="construction_progress",
            config={
                "earlier_dsm_path": str(earlier),
                "later_dsm_path": str(later),
                "change_threshold_m": 0.10,
            },
        )

        assert validate_pipeline_inputs(project_root, run.id).ok is True
        completed = run_pipeline(project_root, run.id)
        status = get_processing_status(project_root, run.id)

        assert completed.status == STATUS_COMPLETED
        assert status.status == STATUS_COMPLETED
        assert status.completed_stages == ["survey_change"]
        assert len(completed.stages[0].output_artifacts) == 4
        assert all(Path(path).is_file() for path in completed.stages[0].output_artifacts)

    def test_missing_comparison_input_fails_readiness_instead_of_running(self, tmp_path):
        existing = write_surface(
            tmp_path / "earlier.tif", np.full((GRID, GRID), GROUND)
        )
        run = create_processing_run(
            tmp_path / "project",
            project_id="site-1",
            dataset_id="",
            workflow_id="construction_progress",
            config={"earlier_dsm_path": str(existing)},
        )
        readiness = validate_pipeline_inputs(tmp_path / "project", run.id)
        assert readiness.ok is False
        assert any("Later DSM missing" in issue for issue in readiness.issues)


class TestSelectedROIChangeWorkflow:
    def test_workflow_requires_two_surveys_and_a_polygon(self):
        workflow = get_workflow_template("stockpile_change")
        assert workflow.required_inputs == ["earlier_dsm", "later_dsm", "roi_polygon"]
        assert workflow.processing_stages == ["selected_roi_change"]

    def test_processing_run_emits_roi_scoped_artifacts(
        self, tmp_path, changed_surveys
    ):
        earlier, later = changed_surveys
        project_root = tmp_path / "project"
        run = create_processing_run(
            project_root,
            project_id="quarry-1",
            dataset_id="",
            workflow_id="stockpile_change",
            config={
                "earlier_dsm_path": str(earlier),
                "later_dsm_path": str(later),
                "roi_polygon_xy": grid_polygon(4, 5, 10, 10),
                "roi_type": "stockpile",
            },
        )
        assert validate_pipeline_inputs(project_root, run.id).ok is True
        completed = run_pipeline(project_root, run.id)
        assert completed.status == STATUS_COMPLETED
        assert completed.stages[0].id == "selected_roi_change"
        assert len(completed.stages[0].output_artifacts) == 5

    def test_processing_readiness_requires_the_roi_polygon(
        self, tmp_path, changed_surveys
    ):
        earlier, later = changed_surveys
        run = create_processing_run(
            tmp_path / "project",
            project_id="quarry-1",
            dataset_id="",
            workflow_id="stockpile_change",
            config={
                "earlier_dsm_path": str(earlier),
                "later_dsm_path": str(later),
            },
        )
        readiness = validate_pipeline_inputs(tmp_path / "project", run.id)
        assert readiness.ok is False
        assert any("ROI polygon missing" in issue for issue in readiness.issues)
