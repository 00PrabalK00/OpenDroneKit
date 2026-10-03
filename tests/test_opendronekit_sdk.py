"""The public ``opendronekit`` SDK: its surface, and each workflow step against real code.

Every call goes through the desktop API the application itself runs (app.api.Api), so
these tests check results and refusals, not that a wrapper forwarded arguments.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import opendronekit as odk
import opendronekit.plugins as plugins_module
from opendronekit.plugins import PluginHost, PluginKind, PluginRegistry, PluginSpec

AOI = [[-82.2265, 41.3055], [-82.2225, 41.3055], [-82.2225, 41.3080], [-82.2265, 41.3080]]
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "E:/Shit was cluittering/Projects/odk-testdata"))
RDD_IMAGE = TESTDATA / "road_damage_rdd2022_subset" / "China_Drone" / "train" / "images" / "China_Drone_000011.jpg"


@pytest.fixture
def ws(tmp_path):
    with odk.Workspace(tmp_path / "odk-data") as workspace:
        yield workspace


@pytest.fixture
def project(ws):
    return ws.create_project("SDK test")


@pytest.fixture
def photo_dir(tmp_path):
    """A dataset of one photograph with a dark diagonal line on a textured background."""
    folder = tmp_path / "photos"
    folder.mkdir()
    rng = np.random.default_rng(7)
    image = (rng.normal(150, 12, (240, 320, 3))).clip(0, 255).astype(np.uint8)
    cv2.line(image, (30, 40), (290, 200), (25, 25, 25), 3)
    cv2.imwrite(str(folder / "wall.png"), image)
    return folder


class TestPublicSurface:
    def test_version_is_semver_and_exported(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", odk.__version__)
        assert odk.PLUGIN_API_LEVEL == "1.1"

    def test_every_public_name_exists(self):
        for name in odk.__all__:
            assert hasattr(odk, name), name

    def test_the_package_ships_type_information(self):
        assert (Path(odk.__file__).parent / "py.typed").is_file()

    def test_public_methods_are_type_hinted_and_documented(self):
        for cls in (odk.Workspace, odk.Project, odk.MissionPlan):
            for name, member in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith("_"):
                    continue
                sig = inspect.signature(member)
                assert sig.return_annotation is not inspect.Signature.empty, f"{cls.__name__}.{name}"
                for param in list(sig.parameters.values())[1:]:
                    if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                        continue
                    assert param.annotation is not inspect.Parameter.empty, \
                        f"{cls.__name__}.{name}({param.name})"
        for cls in (odk.Workspace, odk.Project, odk.MissionPlan):
            for name in ("create_project", "plan_mission", "export", "reconstruct",
                         "detect_cracks", "measure_image", "generate_report"):
                if hasattr(cls, name):
                    assert inspect.getdoc(getattr(cls, name)), f"{cls.__name__}.{name}"

    def test_importing_the_sdk_discovers_no_plugins(self):
        # Discovery is on request; importing must not import third-party code.
        fresh = PluginHost(PluginRegistry())
        assert fresh._state.report is None


class TestProjects:
    def test_create_list_open(self, ws):
        created = ws.create_project("Bridge 7", description="north span")
        assert created.root.is_dir() and created.root.parent == ws.data_dir / "projects"
        assert [p.name for p in ws.list_projects()] == ["Bridge 7"]
        assert ws.open_project("bridge 7").id == created.id
        assert ws.open_project(created.id).info.description == "north span"

    def test_a_duplicate_name_is_refused(self, ws):
        ws.create_project("Same")
        with pytest.raises(odk.Refused, match="already exists"):
            ws.create_project("same")

    def test_an_unknown_project_is_refused(self, ws):
        with pytest.raises(odk.Refused, match="No project 'ghost'"):
            ws.open_project("ghost")

    def test_two_projects_keep_their_own_missions(self, ws):
        a = ws.create_project("A")
        b = ws.create_project("B")
        plan_a = a.plan_mission(AOI, altitude_m=60)
        b.plan_mission(AOI[:3], altitude_m=90)
        assert plan_a.summary.altitude_m == 60
        a.plan_mission(AOI, altitude_m=60).save("a-grid")
        events = [e["event_type"] for e in a.audit_log()]
        assert "mission_planned" in events


class TestMission:
    def test_plan_and_export_every_builtin_format(self, project, tmp_path):
        plan = project.plan_mission(AOI, template="grid", altitude_m=70)
        assert plan.summary.waypoints == len(plan.plan["waypoints"]) > 10
        assert plan.summary.distance_m > 0 and plan.summary.gsd_cm > 0
        assert any("terrain" in w.lower() for w in plan.summary.warnings)
        written = plan.export(directory=tmp_path / "out", stem="m")
        assert set(written) == set(odk.MissionPlan.builtin_formats())
        for files in written.values():
            for path in files:
                assert path.is_file() and path.stat().st_size > 0
        qgc = json.loads((tmp_path / "out" / "m.plan").read_text(encoding="utf-8"))
        assert qgc["fileType"] == "Plan" and qgc["mission"]["items"]

    def test_an_unknown_template_is_refused_not_planned_as_a_grid(self, project):
        with pytest.raises(odk.Refused, match="Unknown mission template 'zigzag'"):
            project.plan_mission(AOI, template="zigzag")

    def test_a_degenerate_area_is_refused_with_the_apps_reason(self, project):
        with pytest.raises(odk.Refused):
            project.plan_mission(AOI[:2])

    def test_export_defaults_to_the_project_missions_folder(self, project):
        written = project.plan_mission(AOI).export(["kml"])
        assert written["kml"][0].parent == project.root / "missions"


class TestDetection:
    def test_heuristic_cracks_are_labelled_heuristic(self, project, photo_dir):
        found = project.detect_cracks(photo_dir / "wall.png", use_model=False)
        assert found.source == "heuristic" and found.model_sha256 == ""
        assert found.mask_path.is_file() and found.overlay_path.is_file()
        mask = cv2.imread(str(found.mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (240, 320) and int((mask > 0).sum()) == found.crack_pixels

    def test_the_crack_model_records_its_weights(self, project):
        from core.models import model_status

        status = model_status("crack_segmentation")
        if not status.get("exists") or not RDD_IMAGE.is_file():
            pytest.skip("crack_segmentation weights (ODK_MODEL_WEIGHTS_DIR) or the RDD "
                        "test image are not present")
        import hashlib

        found = project.detect_cracks(RDD_IMAGE)
        assert found.source == "model" and found.model_used.startswith("onnx:"), found.fallback_reason
        digest = hashlib.sha256(Path(status["path"]).read_bytes()).hexdigest()
        assert found.model_sha256 == digest

    def test_a_failing_model_is_retried_then_explained(self, project, photo_dir, monkeypatch):
        import core.detection as det
        import dataclasses

        calls = []
        real = det.detect_cracks

        def failing(image, **kwargs):
            calls.append(kwargs.get("use_model"))
            res = real(image, min_area_px=kwargs.get("min_area_px", 30), use_model=False)
            return dataclasses.replace(res, model_used="onnx_fallback_heuristic")

        monkeypatch.setattr(det, "detect_cracks", failing)
        monkeypatch.setattr("opendronekit.workspace._crack_model_failure",
                            lambda pixels: "The crack_segmentation model failed: MemoryError: x")
        found = project.detect_cracks(photo_dir / "wall.png")
        assert calls == [True, True]
        assert found.source == "heuristic" and found.model_sha256 == ""
        assert "MemoryError" in found.fallback_reason

    def test_a_missing_image_is_refused(self, project, tmp_path):
        with pytest.raises(odk.Refused, match="No such image"):
            project.detect_cracks(tmp_path / "nothing.jpg")

    def test_a_detector_plugin_runs_through_the_project(self, project, photo_dir, monkeypatch):
        class Dark:
            model_key = "darkest_region"
            model_sha256 = "0" * 64

            def detect(self, image):
                ys, xs = np.where(image.mean(axis=2) < 60)
                return [{"label": "dark", "score": 1.0,
                         "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}]

        fresh = PluginHost(PluginRegistry())
        fresh.load(entry_points=[])
        fresh.registry.register(PluginSpec(PluginKind.MODEL, "dark", Dark, version="0.0.1"))
        monkeypatch.setattr(plugins_module, "host", fresh)
        found = project.detect(photo_dir / "wall.png", detector="dark")
        assert found.model_used == "plugin:dark" and found.model_key == "darkest_region"
        box = found.detections[0]["bbox_xyxy"]
        assert box[0] <= 31 and box[2] >= 289
        assert any(e["event_type"] == "plugin_detection" for e in project.audit_log())


class TestMeasurement:
    def test_a_photo_distance_needs_a_scale(self, project, photo_dir):
        project.import_dataset(photo_dir)
        with pytest.raises(odk.Refused):
            project.measure_image("distance", "wall.png", [[0, 0], [30, 40]])

    def test_a_photo_distance_with_the_operators_scale(self, project, photo_dir):
        project.import_dataset(photo_dir)
        m = project.measure_image("distance", photo_dir / "wall.png", [[0, 0], [30, 40]],
                                  scale_mm_per_px=2.0, label="test")
        # 50 px at 2 mm/px = 100 mm, reported in metres like every desktop measurement.
        assert m.value == pytest.approx(0.1) and m.unit == "m"
        assert m.raw["scale_source"] == "entered"
        assert m.caveat and m.id
        assert any(r["id"] == m.id for r in project.measurements())

    def test_a_reference_of_known_length_sets_the_scale(self, project, photo_dir):
        project.import_dataset(photo_dir)
        m = project.measure_image("distance", "wall.png", [[0, 0], [0, 50]],
                                  reference_pixels=[[0, 0], [100, 0]], reference_length_mm=500)
        assert m.value == pytest.approx(0.25) and m.unit == "m"
        assert m.raw["scale_source"] != "entered"

    def test_a_photo_outside_the_dataset_is_refused(self, project, photo_dir, tmp_path):
        project.import_dataset(photo_dir)
        with pytest.raises(odk.Refused, match="not in the active dataset"):
            project.measure_image("distance", tmp_path / "elsewhere.png", [[0, 0], [1, 1]],
                                  scale_mm_per_px=1)

    def test_crack_length_from_a_mask(self, project, photo_dir, tmp_path):
        project.import_dataset(photo_dir)
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (10, 120), (310, 120), 255, 3)        # 300 px long
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(mask_path), mask)
        with pytest.raises(odk.Refused):
            project.measure_crack("wall.png", mask_path)
        m = project.measure_crack("wall.png", mask_path, scale_mm_per_px=0.5)
        assert m.unit == "m" and m.value == pytest.approx(0.15, rel=0.03)

    def test_a_mask_of_the_wrong_size_is_refused(self, project, photo_dir, tmp_path):
        project.import_dataset(photo_dir)
        cv2.imwrite(str(tmp_path / "small.png"), np.zeros((10, 10), np.uint8))
        with pytest.raises(odk.Refused, match="mask is 10x10"):
            project.measure_crack("wall.png", tmp_path / "small.png", scale_mm_per_px=1)

    def test_map_measurement_without_a_raster_is_refused(self, project):
        with pytest.raises(odk.Refused, match="no georeferenced raster"):
            project.measure_map("distance", AOI[:2])


class TestProcessingAndReports:
    def test_reconstruction_without_images_is_refused_before_a_job(self, project):
        with pytest.raises(odk.Refused, match="Select a dataset"):
            project.reconstruct()

    def test_a_failed_reconstruction_raises_job_failed(self, project, tmp_path):
        folder = tmp_path / "blank"
        folder.mkdir()
        for i in range(3):
            cv2.imwrite(str(folder / f"b{i}.jpg"), np.full((64, 64, 3), 128, np.uint8))
        with pytest.raises(odk.JobFailed) as caught:
            project.reconstruct(folder, engine="custom", timeout_s=300)
        assert caught.value.job["status"] == "failed"

    def test_an_empty_project_report_is_refused_with_the_checklist(self, project):
        with pytest.raises(odk.Refused, match="Report not generated"):
            project.generate_report()

    def test_a_mission_summary_report_is_written(self, project):
        project.plan_mission(AOI, altitude_m=65).save("Site grid")
        report = project.generate_report(title="Mission summary", report_type="mission_summary")
        assert report.html_path.is_file() and report.output_dir.parent == project.root / "reports"
        assert "Site grid" in report.html_path.read_text(encoding="utf-8")
        assert report.pdf_path is None

    def test_an_unknown_report_format_is_refused(self, project):
        with pytest.raises(odk.Refused, match="Unknown report format"):
            project.generate_report(formats=["odt"])
