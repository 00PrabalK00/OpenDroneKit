"""The Home workspace's backends (U4), against real files.

Home shows figures no Api method could answer before: the survey's capture window,
what a project occupies on disk, which files were exported, what a reconstruction
measured. Each is read from the files themselves (app/home_data.py). These tests hold
them to that: every figure equals what the files say, and a figure no file holds is
None rather than a default.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", REPO / "testdata"))
IMAGES = TESTDATA / "odm_aukerman" / "images"
RECON = Path(os.environ.get("ODK_AUKERMAN_FULLRES", REPO / "demo_project" / "aukerman_fullres"))
if (RECON / "reconstruction").is_dir():
    RECON = RECON / "reconstruction"

needs_images = pytest.mark.skipif(not IMAGES.is_dir(), reason="ODK_TESTDATA_DIR has no odm_aukerman/images.")
needs_recon = pytest.mark.skipif(not (RECON / "orthomosaic.tif").is_file(),
                                 reason="ODK_AUKERMAN_FULLRES is not on this machine.")


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    assert api.create_project("Home data", str(tmp_path / "project"))["ok"]
    return api


def _exif_times(folder: Path) -> list[str]:
    from PIL import Image

    stamps = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}:
            continue
        with Image.open(path) as img:
            raw = img.getexif().get_ifd(0x8769).get(36867)
        if raw:
            stamps.append(str(raw).replace(":", "-", 2).replace(" ", "T"))
    return sorted(stamps)


@needs_images
class TestTheSurveyDateIsTheCameras:
    def test_capture_window_is_the_exif_clocks(self, api, tmp_path) -> None:
        assert api.import_dataset(str(IMAGES))["ok"]
        summary = api.dataset_summary()
        assert summary["ok"]
        active = summary["active"]
        stamps = _exif_times(IMAGES)
        capture = active["capture"]
        assert capture["images"] == active["image_count"] == len([
            p for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}])
        assert capture["dated"] == len(stamps)
        assert capture["first"] == stamps[0] and capture["last"] == stamps[-1]
        assert capture["source"] == "EXIF DateTimeOriginal"
        # The Aukerman flight: 77 photographs, 2016-06-29 from 10:47 to 11:04, camera time.
        assert capture["dated"] == 77 and capture["first"].startswith("2016-06-29T10:47")
        assert capture["last"].startswith("2016-06-29T11:04") and capture["utc_offset"] is None
        assert active["geotagged_count"] == 77 and active["camera"] == "SONY DSC-WX220"
        assert active["image_bytes"] == sum(p.stat().st_size for p in IMAGES.iterdir()
                                            if p.suffix.lower() in {".jpg", ".jpeg"})
        # Cached under the project, and the cache is what a second call answers.
        cache = Path(api.get_project()["project"]["root_dir"]) / ".odk" / "capture_windows.json"
        assert cache.is_file()
        assert api.dataset_summary()["active"]["capture"] == capture

    def test_the_project_list_carries_it(self, api) -> None:
        assert api.import_dataset(str(IMAGES))["ok"]
        listed = api.home_projects()
        row = next(p for p in listed["projects"] if p["active"])
        assert row["photos"] == 77 and row["geotagged"] == 77
        assert row["survey_first"].startswith("2016-06-29") and row["status"] == ["photos imported"]

    def test_a_folder_with_no_clock_says_so(self, tmp_path) -> None:
        from PIL import Image

        from app.home_data import capture_window

        folder = tmp_path / "plain"
        folder.mkdir()
        Image.new("RGB", (8, 8)).save(folder / "a.jpg")
        window = capture_window(folder)
        assert window == {**window, "images": 1, "dated": 0, "first": None, "last": None, "duration_s": None}


class TestStorage:
    def test_bytes_are_the_files_on_disk(self, api, tmp_path) -> None:
        root = Path(api.get_project()["project"]["root_dir"])
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "a.bin").write_bytes(b"x" * 1234)
        (root / "loose.txt").write_bytes(b"y" * 10)
        outside = tmp_path / "photos"
        outside.mkdir()
        (outside / "p.jpg").write_bytes(b"z" * 500)
        api._session.store.save_dataset_entry(api._session.project_id(), "photos", str(outside), {"image_count": 1})
        storage = api.project_storage()
        assert storage["ok"]
        walked = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        assert storage["project_bytes"] == walked
        assert storage["datasets_outside"] == [{"path": str(outside), "bytes": 500, "files": 1}]
        assert {"name": "reports", "bytes": 1234, "files": 1} in storage["groups"]
        usage = shutil.disk_usage(root)
        assert storage["disk"]["total"] == usage.total


class TestExports:
    def test_mission_exports_are_listed_with_their_sizes(self, api) -> None:
        aoi = [[-81.7520, 41.3030], [-81.7500, 41.3030], [-81.7500, 41.3045], [-81.7520, 41.3045]]
        assert api.set_aoi(aoi)["ok"]
        assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
        assert api.list_exports()["files"] == []
        written = api.export_mission()
        assert written["ok"]
        listed = api.list_exports()
        files = {Path(f["path"]).resolve(): f for f in listed["files"]}
        for path in written["written"].values():
            paths = path if isinstance(path, list) else [path]
            for one in paths:
                entry = files[Path(one).resolve()]
                assert entry["bytes"] == Path(one).stat().st_size and entry["kind"] == "mission"
        assert listed["count"] == len(listed["files"])
        assert listed["total_bytes"] == sum(f["bytes"] for f in listed["files"])

    def test_an_export_to_a_chosen_folder_is_found_through_the_audit_log(self, api, tmp_path) -> None:
        aoi = [[-81.7520, 41.3030], [-81.7500, 41.3030], [-81.7500, 41.3045], [-81.7520, 41.3045]]
        api.set_aoi(aoi)
        api.plan_mission({"template": "grid", "altitude_m": 60})
        chosen = tmp_path / "elsewhere"
        written = api.export_mission(["qgc_plan"], str(chosen))
        assert written["ok"]
        names = {f["name"] for f in api.list_exports()["files"]}
        assert Path(written["written"]["qgc_plan"]).name in names

    def test_nothing_exported_is_an_empty_list(self, api) -> None:
        assert api.list_exports() == {**api.list_exports(), "files": [], "count": 0, "total_bytes": 0}


@needs_recon
class TestReconstruction:
    def test_figures_come_from_the_engines_files(self, api) -> None:
        root = Path(api.get_project()["project"]["root_dir"])
        out = root / "reconstruction"
        out.mkdir(parents=True)
        for item in RECON.iterdir():
            if item.is_file():
                shutil.copy2(item, out / item.name)
        run = api.reconstruction_summary()["runs"][0]
        anchor = json.loads((out / "geo_anchor.json").read_text(encoding="utf-8"))
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8")) if (out / "run_meta.json").is_file() else {}
        assert run["georef_rmse_m"] == anchor["rmse_m"]
        assert run["sources"]["georef_rmse_m"] == "geo_anchor.json"
        import rasterio

        with rasterio.open(out / "orthomosaic.tif") as src:
            assert run["ortho_gsd_m"] == abs(src.res[0])
        with rasterio.open(out / "dsm.tif") as src:
            assert run["dsm_gsd_m"] == abs(src.res[0])
        if meta:
            assert run["runtime_s"] == meta["elapsed_s"]
            assert run["sparse_points"] == meta["result"]["total_points"]
            assert run["registered_images"] == meta["result"]["registered_images"]
        assert run["complete"] is True

    def test_a_figure_no_file_holds_is_none(self, api) -> None:
        root = Path(api.get_project()["project"]["root_dir"])
        out = root / "reconstruction"
        out.mkdir(parents=True)
        shutil.copy2(RECON / "orthomosaic.tif", out / "orthomosaic.tif")
        run = api.reconstruction_summary()["runs"][0]
        for key in ("sparse_points", "reprojection_error_px", "georef_rmse_m", "runtime_s", "registered_images"):
            assert run[key] is None, key
        assert run["ortho_gsd_m"] and run["dsm_gsd_m"] is None

    def test_the_run_record_the_api_writes_is_read(self, api) -> None:
        root = Path(api.get_project()["project"]["root_dir"])
        out = root / "reconstruction"
        out.mkdir(parents=True)
        shutil.copy2(RECON / "orthomosaic.tif", out / "orthomosaic.tif")
        (out / "run_record.json").write_text(json.dumps({
            "elapsed_s": 12.5, "engine": "colmap", "profile": "fast_preview",
            "finished_utc": "2026-10-02T10:00:00+00:00",
            "result": {"registered_images": 9, "frame_count": 10, "total_points": 1234,
                       "reprojection_error_px": 0.9},
        }), encoding="utf-8")
        run = api.reconstruction_summary()["runs"][0]
        assert (run["runtime_s"], run["profile"], run["registered_images"], run["total_images"],
                run["sparse_points"], run["reprojection_error_px"]) == (12.5, "fast_preview", 9, 10, 1234, 0.9)
        assert run["sources"]["runtime_s"] == "run_record.json"

    def test_an_empty_project_has_no_runs(self, api) -> None:
        assert api.reconstruction_summary()["runs"] == []


class TestReportFiles:
    def test_no_report_is_a_refusal_that_says_so(self, api) -> None:
        answer = api.report_files()
        assert answer["ok"] is False and "No report" in answer["error"]
