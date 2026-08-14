"""A selected stockpile becomes a mapped, reproducible client measurement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from conftest import (
    BLOCK_CELLS,
    BLOCK_VOLUME_M3,
    GRID,
    GROUND_M,
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE_M,
)
from core import geo
from core.dsm_analysis import IncompatibleReferenceSurface, estimate_volume
from core.processing_runs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    create_processing_run,
    run_pipeline_stage,
)
from core.survey_intelligence import create_stockpile_package


def block_polygon() -> list[list[float]]:
    start = (GRID - BLOCK_CELLS) // 2
    west = ORIGIN_X + start * PIXEL_SIZE_M
    east = west + BLOCK_CELLS * PIXEL_SIZE_M
    north = ORIGIN_Y - start * PIXEL_SIZE_M
    south = north - BLOCK_CELLS * PIXEL_SIZE_M
    return [[west, north], [east, north], [east, south], [west, south]]


class TestSelectedStockpilePackage:
    def test_selected_pile_volume_is_exact_and_uses_the_dtm(self, tmp_path, surface_pair):
        package = create_stockpile_package(
            surface_pair["dsm"],
            tmp_path / "deliverable",
            polygon_xy=block_polygon(),
            dtm_path=surface_pair["dtm"],
        )

        preferred = package.measurement["preferred"]
        assert preferred["reference"] == "dtm"
        assert preferred["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3)
        assert preferred["fill_volume_m3"] == 0.0
        assert package.measurement["region_area_m2"] == pytest.approx(100.0)

    def test_selection_summary_and_report_are_client_usable(self, tmp_path, surface_pair):
        package = create_stockpile_package(
            surface_pair["dsm"],
            tmp_path / "deliverable",
            polygon_xy=block_polygon(),
            dtm_path=surface_pair["dtm"],
        )

        assert all(Path(path).is_file() for path in package.artifact_paths())
        layer = json.loads(Path(package.selection_geojson_path).read_text(encoding="utf-8"))
        assert layer["crs"]["properties"]["name"].endswith("EPSG::32617")
        properties = layer["features"][0]["properties"]
        assert properties["cut_volume_m3"] == pytest.approx(BLOCK_VOLUME_M3)
        assert properties["reference"] == "dtm"

        summary = json.loads(Path(package.summary_path).read_text(encoding="utf-8"))
        report = Path(package.report_path).read_text(encoding="utf-8")
        assert summary["selection"] == "operator_polygon"
        assert summary["units"]["volume"] == "m3"
        assert "no material class" in summary["interpretation"]
        assert "does not classify the material" in report
        assert "Base surface: aligned DTM" in report

    def test_a_base_surface_is_required_instead_of_quietly_using_the_minimum(
        self, tmp_path, surface_pair
    ):
        with pytest.raises(ValueError, match="requires either an aligned DTM"):
            create_stockpile_package(
                surface_pair["dsm"],
                tmp_path / "deliverable",
                polygon_xy=block_polygon(),
            )

    def test_an_operator_supplied_plane_is_recorded(self, tmp_path, surface_pair):
        package = create_stockpile_package(
            surface_pair["dsm"],
            tmp_path / "deliverable",
            polygon_xy=block_polygon(),
            base_elevation_m=GROUND_M,
        )
        assert package.measurement["preferred"]["reference"] == "plane"
        assert package.measurement["preferred"]["reference_elevation_m"] == GROUND_M
        assert package.measurement["preferred"]["cut_volume_m3"] == pytest.approx(
            BLOCK_VOLUME_M3
        )


class TestReferenceSurfaceSafety:
    def test_a_shifted_dtm_is_refused_even_if_its_shape_and_crs_match(
        self, tmp_path, surface_pair
    ):
        shifted = tmp_path / "shifted_dtm.tif"
        geo.write_geotiff(
            shifted,
            np.full((GRID, GRID), GROUND_M, dtype=np.float32),
            epsg=32617,
            west=ORIGIN_X + PIXEL_SIZE_M,
            north=ORIGIN_Y,
            pixel_size=PIXEL_SIZE_M,
            cog=False,
        )
        with pytest.raises(IncompatibleReferenceSurface, match="does not share"):
            estimate_volume(
                surface_pair["dsm"],
                dtm_path=shifted,
                polygon_xy=block_polygon(),
            )


class TestStockpileWorkflow:
    def test_volume_stage_emits_the_mapped_package(self, tmp_path, surface_pair):
        project_root = tmp_path / "project"
        run = create_processing_run(
            project_root,
            project_id="quarry-1",
            dataset_id="",
            workflow_id="stockpile_measurement",
            config={
                "dsm_path": str(surface_pair["dsm"]),
                "dtm_path": str(surface_pair["dtm"]),
                "volume_polygon_xy": block_polygon(),
            },
        )

        result = run_pipeline_stage(project_root, run.id, "volume_estimation")
        assert result.status == STATUS_COMPLETED
        assert len(result.artifacts) == 3
        assert all(Path(path).is_file() for path in result.artifacts)

    def test_volume_stage_refuses_to_measure_everything_as_one_pile(
        self, tmp_path, surface_pair
    ):
        project_root = tmp_path / "project"
        run = create_processing_run(
            project_root,
            project_id="quarry-1",
            dataset_id="",
            workflow_id="stockpile_measurement",
            config={
                "dsm_path": str(surface_pair["dsm"]),
                "dtm_path": str(surface_pair["dtm"]),
            },
        )
        result = run_pipeline_stage(project_root, run.id, "volume_estimation")
        assert result.status == STATUS_FAILED
        assert "selected pile polygon" in str(result.error)
