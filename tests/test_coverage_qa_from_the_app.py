"""Coverage QA as an operator runs it: on the flown images, from the Verification toolbar.

The ground-based coverage check (tests/test_coverage_is_measured_on_the_ground.py) ran
only inside the full pipeline, after reconstruction -- far too late to send the aircraft
back while it is still on site -- and nothing in the app could start it or show what it
found. It also told a complete serpentine grid to re-fly every line change: consecutive
frames at a turn meet little, although the ground between them was photographed twice by
the neighbouring lines.

These drive Api.check_coverage as a background job on real JPEG+EXIF flights (and on the
real Aukerman frames when present), check the footprints and re-fly points reach the
layer tree, that a missing height, a feet DEM or a bad height is refused in words, and
press Coverage QA in the real cockpit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from core.coverage_validation import CoverageValidationConfig, CoverageValidator
from test_coverage_is_measured_on_the_ground import GROUND_M, ORIGIN, SPACING, _grid, _shot


@pytest.fixture
def api(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Coverage", str(tmp_path / "project"))["ok"]
    return api


def _finish(api: Api, job_id: str, timeout_s: float = 300) -> dict:
    deadline = time.time() + timeout_s
    while True:
        job = api.job_status(job_id)["job"]
        if job["status"] in ("done", "failed", "cancelled") or time.time() > deadline:
            return job
        time.sleep(0.2)


def _flight(tmp_path: Path, **kwargs) -> Path:
    folder = tmp_path / "flight"
    folder.mkdir()
    _grid(folder, **kwargs)
    return folder


class TestALineChangeIsNotAGap:
    def test_a_complete_grid_is_told_to_refly_nothing(self, tmp_path):
        # Row-major: IMG_02 -> IMG_10 is a line change 134 m apart with ~3% overlap, but
        # every metre between them is photographed by two or more images.
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(_grid(tmp_path), tmp_path / "out")
        turns = [m for m in result["missing_pairs"]
                 if (m["image_prev"], m["image_curr"]) in (("IMG_02.jpg", "IMG_10.jpg"),
                                                          ("IMG_12.jpg", "IMG_20.jpg"))]
        assert len(turns) == 2 and all(m["ground_between_stereo_covered"] for m in turns)
        assert not [s for s in result["gap_suggestions"] if s["kind"] == "overlap_gap"]

    def test_a_single_strip_with_a_missing_shot_is_still_a_gap(self, tmp_path):
        # One line only: the ground between shots two apart is seen by one image at
        # most, so the missing frame is a real re-fly.
        paths = [_shot(tmp_path / f"IMG_{i}.jpg", ORIGIN[0], ORIGIN[1] + i * SPACING,
                       texture_seed=i) for i in (0, 1, 3, 4)]
        result = CoverageValidator(CoverageValidationConfig(ground_elevation_m=GROUND_M)) \
            .validate(paths, tmp_path / "out")
        gaps = [s for s in result["gap_suggestions"] if s["kind"] == "overlap_gap"]
        assert [g["related_images"] for g in gaps] == [["IMG_1.jpg", "IMG_3.jpg"]]


class TestCoverageQaThroughTheApi:
    def test_it_places_the_images_and_adds_map_layers(self, api, tmp_path):
        folder = _flight(tmp_path, skip=(1, 1))
        assert api.set_active_dataset(str(folder))["ok"]
        started = api.check_coverage({"ground_elevation_m": GROUND_M})
        assert started["ok"] and started["image_count"] == 8, started
        job = _finish(api, started["job_id"])
        assert job["status"] == "done", job
        summary = job["result"]["coverage_summary"]
        assert summary.startswith("8 of 8 image(s) placed"), summary
        names = [layer["name"] for layer in api.list_layers()["layers"]]
        assert "Coverage footprints" in names
        footprints = json.loads(next(
            (Path(api._session.project_root()) / "verification" / "coverage")
            .rglob("footprints.geojson")).read_text(encoding="utf-8"))
        assert len(footprints["features"]) == 8
        assert footprints["features"][0]["properties"]["height_source"] == \
            "gps_altitude_minus_ground"
        report = api.coverage_qa_report()
        assert report["ok"] and report["available"] is True
        assert report["images_placed"] == 8 and report["completion_pct"] == pytest.approx(100.0)

    def test_without_a_height_it_is_reported_not_computed(self, api, tmp_path):
        folder = _flight(tmp_path)
        started = api.check_coverage({"image_dir": str(folder)})
        job = _finish(api, started["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["coverage_summary"].startswith("coverage not computed")
        assert job["result"]["layer_ids"] == []
        report = api.coverage_qa_report()
        assert report["available"] is False and report["completion_pct"] is None
        assert "height above ground" in report["reason"]

    def test_refusals_are_in_words(self, api, tmp_path, surface_pair):
        assert "Load Flight" in api.check_coverage()["error"]
        assert "No coverage check" in api.coverage_qa_report()["error"]
        folder = _flight(tmp_path)
        bad = api.check_coverage({"image_dir": str(folder), "flight_height_agl_m": -5})
        assert bad["ok"] is False and "more than 0 m" in bad["error"]
        from test_wrong_scale_and_terrain_units_are_refused import OHIO_NORTH_US_FT, _warp

        feet = _warp(surface_pair["dtm"], tmp_path / "dtm_ft.tif", OHIO_NORTH_US_FT)
        refused = api.check_coverage({"image_dir": str(folder), "terrain_path": str(feet)})
        assert refused["ok"] is False and "US survey foot" in refused["error"], refused

    def test_on_the_real_aukerman_frames(self, api):
        from conftest import survey_imagery_dir

        folder = survey_imagery_dir()
        if not folder.is_dir() or not list(folder.glob("*.JPG")):
            pytest.skip("The Aukerman survey frames are not present (gitignored).")
        started = api.check_coverage({"image_dir": str(folder), "ground_elevation_m": 235})
        job = _finish(api, started["job_id"], 600)
        assert job["status"] == "done", job
        report = api.coverage_qa_report()
        assert report["images_placed"] == report["images"] == 24
        assert report["completion_pct"] == pytest.approx(100.0)
        assert report["stereo_pct"] == pytest.approx(99.45, abs=0.2)
        result = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
        # The subset skips frames 231, 245, 248 and 250; where neighbouring lines saw the ground
        # twice that is not a re-fly, and every re-fly left is where only one image did.
        for miss in result["missing_pairs"]:
            refly = [s for s in result["gap_suggestions"] if s["kind"] == "overlap_gap"
                     and s["related_images"] == [miss["image_prev"], miss["image_curr"]]]
            assert bool(refly) is (not miss["ground_between_stereo_covered"])


def test_coverage_qa_button_in_the_cockpit(api, tmp_path):
    from cockpit_actions_in_a_browser import run_actions

    folder = _flight(tmp_path)
    assert api.set_active_dataset(str(folder))["ok"]
    after = """
      for (let i = 0; i < 400; i++) {
        const text = document.body.innerText;
        const m = text.match(/Job \\S+ finished: [^\\n]*/);
        if (m) return m[0];
        await new Promise((r) => setTimeout(r, 50));
      }
      return "no finish toast";
    """
    results = run_actions(api, tmp_path, [
        # No height typed, then the site's ground elevation.
        {"name": "qa", "action": "Coverage QA", "answers": ["", str(GROUND_M)], "after": after},
    ], workspace="verification", virtual_time_ms=30000)
    assert "[ok] Coverage QA started on 9 image(s)." in results["qa"], results
    assert "9 of 9 image(s) placed" in results["qa:after"], results
    assert "100.0% of the flown block seen" in results["qa:after"]
    report = api.coverage_qa_report()
    assert report["available"] is True and report["images_placed"] == 9
