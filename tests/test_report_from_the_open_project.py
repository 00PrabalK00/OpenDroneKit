"""Reports describe the project the desktop has open, and every format is written.

Found by driving the application: Generate Report built its document from an empty
mirror project in a separate workspace, so a project with imported imagery and a saved
mission produced "No missions saved / No datasets imported"; the PDF was always None
because the HTML-to-PDF converters were never installed; all six report types were the
same document with a different label; the recommendations were boilerplate about
cracks in reports with no cracks; and the appendix dumped absolute local paths.

Every test here goes through the real Api with a real store and a real project folder.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from test_reports_in_the_desktop_window import _geotagged_jpeg

pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")
pytest.importorskip("docx")


def make_images(folder: Path, count: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _geotagged_jpeg(folder / f"IMG_{index:04d}.jpg", 41.3038 + index * 1e-4,
                        -81.7504, 80 + 20 * index)
    return folder


@pytest.fixture
def api(tmp_path) -> Api:
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Bridge 14", str(tmp_path / "project"))["ok"]
    return api


@pytest.fixture
def surveyed(api, tmp_path) -> Api:
    """A project with imagery and a saved mission, and nothing else."""
    assert api.import_dataset(str(make_images(tmp_path / "imagery")), "North span")["ok"]
    assert api.set_aoi([[-81.7510, 41.3035], [-81.7500, 41.3035],
                        [-81.7500, 41.3045], [-81.7510, 41.3045]])["ok"]
    assert api.plan_mission({"altitude_m": 60, "mode": "grid"})["ok"]
    assert api.save_mission("Span grid", "")["ok"]
    return api


def headings(path: str) -> list[str]:
    return re.findall(r"<h2>(.*?)</h2>", Path(path).read_text(encoding="utf-8"))


def pdf_text(path: str) -> str:
    """The PDF's text with line wrapping undone, so a sentence can be searched for."""
    text = " ".join(page.extract_text() or "" for page in pypdf.PdfReader(path).pages)
    return " ".join(text.split())


def root_of(api: Api) -> Path:
    return Path(api._session.project_root())  # noqa: SLF001


class TestTheReportIsAboutTheOpenProject:
    def test_imagery_and_missions_reach_the_report(self, surveyed) -> None:
        result = surveyed.generate_report("", "Span report")
        assert result["ok"], result
        page = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "North span" in page and "Span grid" in page
        assert "3 image(s), 3 geotagged" in page
        assert "No imagery has been imported" not in page

    def test_it_is_written_inside_the_project_folder(self, surveyed) -> None:
        result = surveyed.generate_report("", "Span report")
        assert Path(result["html_path"]).is_relative_to(root_of(surveyed) / "reports")
        listed = surveyed.list_reports()["reports"]
        assert [r["id"] for r in listed] == [result["id"]]

    def test_local_folder_paths_do_not_leak_into_the_deliverable(self, surveyed) -> None:
        result = surveyed.generate_report("", "Span report", "engineering")
        page = Path(result["html_path"]).read_text(encoding="utf-8")
        assert str(root_of(surveyed)) not in page
        assert str(root_of(surveyed)) not in pdf_text(result["pdf_path"])

    def test_readiness_reads_the_same_project(self, surveyed) -> None:
        readiness = surveyed.report_readiness()
        assert readiness["ok"] and readiness["ready"]
        text = " ".join(readiness["warnings"] + readiness["notes"])
        assert "No dataset imported" not in text
        assert "No findings recorded" in text


class TestEveryFormatIsWritten:
    def test_the_pdf_exists_and_says_what_the_page_says(self, surveyed) -> None:
        """pdf_path was None for every report: weasyprint was never installed."""
        result = surveyed.generate_report("", "Span report", "standard", "",
                                          "Acme", "", "", ["html", "pdf", "docx"])
        assert result["pdf_path"] and Path(result["pdf_path"]).is_file()
        text = pdf_text(result["pdf_path"])
        assert "Span report" in text and "Span grid" in text and "Acme" in text
        from docx import Document

        assert Document(result["docx_path"]).paragraphs

    def test_every_written_file_carries_verifiable_provenance(self, surveyed) -> None:
        from core.provenance import verify

        result = surveyed.generate_report("", "Span report", "standard", "", "", "", "",
                                          ["html", "pdf", "docx"])
        for key in ("html_path", "pdf_path", "docx_path"):
            assert verify(result[key])["status"] == "verified", key


class TestReportTypesAreDifferentDocuments:
    def test_the_types_have_different_sections(self, surveyed) -> None:
        seen = {}
        for kind in ("standard", "engineering", "executive", "mission_summary",
                     "dataset_quality"):
            result = surveyed.generate_report("", kind, kind)
            assert result["ok"], result
            seen[kind] = tuple(headings(result["html_path"]))
        assert len(set(seen.values())) == len(seen)
        assert "Audit Trail" not in seen["executive"]
        assert "Provenance of Derived Files" in seen["engineering"]
        assert "Findings" not in seen["mission_summary"]

    def test_a_type_with_nothing_to_be_about_is_refused(self, api) -> None:
        refused = api.generate_report("", "Flight", "mission_summary")
        assert not refused["ok"] and "mission" in refused["error"]
        refused = api.generate_report("", "Defects", "defect_only")
        assert not refused["ok"] and "findings" in refused["error"]

    def test_an_unknown_type_is_refused_by_name(self, api) -> None:
        refused = api.generate_report("", "X", "thermal")
        assert not refused["ok"] and "standard" in refused["error"]


class TestNothingIsWrittenOnTheProjectsBehalf:
    def test_no_boilerplate_recommendations(self, surveyed) -> None:
        page = Path(surveyed.generate_report("", "R")["html_path"]).read_text(encoding="utf-8")
        assert "30 days" not in page and "Re-fly any areas" not in page
        assert "The author entered no recommendations." in page
        assert "No follow-up items arise from the recorded data." in page

    def test_follow_up_items_come_from_the_findings(self, surveyed) -> None:
        assert surveyed.record_finding("IMG_0001.jpg", "spalling", "critical", "", [10, 10, 60, 50])["ok"]
        page = Path(surveyed.generate_report("", "R", "standard", "", "", "", "", None, "",
                                             "Close lane 2.")["html_path"]).read_text(encoding="utf-8")
        assert "Close lane 2." in page
        assert "Finding 1 (spalling, critical) on IMG_0001.jpg is not yet confirmed" in page

    def test_an_empty_findings_section_says_nothing_was_recorded(self, surveyed) -> None:
        page = Path(surveyed.generate_report("", "R")["html_path"]).read_text(encoding="utf-8")
        assert "not the same as an inspection that found nothing" in page


class TestReviewReachesTheDeliverable:
    def _model_finding(self, api: Api) -> str:
        from core.annotations import create_annotation

        image = next((root_of(api).parent / "imagery").glob("*.jpg"))
        annotation = create_annotation(
            root_of(api), "1", "image", str(image), "rectangle",
            {"type": "Polygon", "coordinates": [[[5, 5], [50, 5], [50, 40], [5, 40], [5, 5]]]},
            "corrosion", "high", "open", note="model proposal", created_by="model:onnx:x.onnx")
        return annotation.id

    def test_an_unreviewed_model_finding_is_flagged(self, surveyed) -> None:
        self._model_finding(surveyed)
        result = surveyed.export_report("pdf", "Findings")
        assert "not been confirmed by an inspector" in pdf_text(result["path"])

    def test_accepting_it_clears_the_caveat_and_keeps_the_note(self, surveyed) -> None:
        finding = self._model_finding(surveyed)
        reviewed = surveyed.review_finding(finding, "accept", "P. Khare")
        assert reviewed["status"] == "resolved"
        assert reviewed["note"].startswith("model proposal")
        text = pdf_text(surveyed.export_report("pdf", "Findings")["path"])
        assert "not been confirmed by an inspector" not in text
        assert "accepted" in text

    def test_a_rejected_finding_is_not_printed_as_a_finding(self, surveyed) -> None:
        finding = self._model_finding(surveyed)
        surveyed.record_finding("IMG_0002.jpg", "crack", "low")
        surveyed.review_finding(finding, "reject", "P. Khare")
        text = pdf_text(surveyed.export_report("pdf", "Findings")["path"])
        assert "Findings (1)" in text
        assert "rejected by the reviewer" in text

    def test_exports_do_not_overwrite_each_other(self, surveyed) -> None:
        surveyed.record_finding("IMG_0002.jpg", "crack", "low")
        first = surveyed.export_report("docx")["path"]
        time.sleep(1.1)
        second = surveyed.export_report("docx")["path"]
        assert first != second and Path(first).is_file() and Path(second).is_file()


class TestRecordingAFinding:
    def test_a_finding_on_an_unknown_image_is_refused(self, surveyed) -> None:
        refused = surveyed.record_finding("DSC_9999.JPG", "crack", "high")
        assert not refused["ok"] and "Import" in refused["error"]

    def test_a_box_outside_the_photo_is_refused(self, surveyed) -> None:
        refused = surveyed.record_finding("IMG_0001.jpg", "crack", "high", "", [0, 0, 5000, 10])
        assert not refused["ok"]

    def test_the_finding_is_an_inspectors_and_is_located(self, surveyed) -> None:
        recorded = surveyed.record_finding("IMG_0001.jpg", "crack", "high", "at the joint",
                                           [10, 10, 60, 50])["finding"]
        assert recorded["created_by"] == "user"
        page = Path(surveyed.generate_report("", "R")["html_path"]).read_text(encoding="utf-8")
        assert "photo position 41.303900, -81.750400" in page
        assert "at the joint" in page


class TestReconstructionAndProvenance:
    def test_a_reconstruction_run_from_the_app_records_provenance(self, surveyed) -> None:
        """record_reconstruction_outputs had no caller: every real output was
        unattributed."""
        job = surveyed.run_reconstruction({"engine": "custom"})
        assert job["ok"], job
        deadline = time.time() + 300
        while time.time() < deadline:
            status = surveyed.job_status(job["job_id"])["job"]
            if status["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.5)
        assert status["status"] == "done", status.get("error")
        audit = surveyed.verify_provenance()
        assert audit["ok"], audit
        assert audit["total"] > 0 and not audit["unattributed"] and audit["all_verified"]

    def test_outputs_made_before_that_can_be_recorded_and_are_then_checked(self, surveyed) -> None:
        folder = root_of(surveyed) / "reconstruction"
        folder.mkdir()
        (folder / "digital_twin.json").write_text(json.dumps(
            {"engine": "colmap", "profile": "standard", "crs_epsg": 32617,
             "frame_count": 3, "registered_images": 3, "point_count": 1200}), encoding="utf-8")
        before = surveyed.verify_provenance()
        assert before["unattributed"] == ["digital_twin.json"]
        assert surveyed.record_provenance()["artifact_count"] == 1
        assert surveyed.verify_provenance()["all_verified"]
        page = Path(surveyed.generate_report("", "E", "engineering")["html_path"]).read_text(
            encoding="utf-8")
        assert "EPSG:32617" in page and "1200" in page
        # A file changed after its record is reported, not passed.
        (folder / "digital_twin.json").write_text("{}", encoding="utf-8")
        assert surveyed.verify_provenance()["modified"] == ["digital_twin.json"]


class TestThePipelineWritesItsReport:
    def _run(self, api: Api) -> dict:
        job = api.run_pipeline({"engine": "custom"})
        assert job["ok"], job
        deadline = time.time() + 600
        while time.time() < deadline:
            status = api.job_status(job["job_id"])["job"]
            if status["status"] in ("done", "failed", "cancelled"):
                return status
            time.sleep(0.5)
        raise AssertionError("pipeline did not finish")

    def test_run_full_pipeline_completes_and_writes_the_markdown(self, surveyed) -> None:
        """It raised TypeError: unexpected keyword argument 'output_root'."""
        status = self._run(surveyed)
        assert status["status"] == "done", status.get("error")
        report = Path(status["result"]["report_markdown_path"])
        text = report.read_text(encoding="utf-8")
        assert "heuristic" in text
        html = Path(surveyed.generate_report("", "D", "defect_only")["html_path"]).read_text(
            encoding="utf-8")
        assert "IMG_0000.jpg" in html and "screening indicators" in html

    def test_a_reconstruction_that_cannot_run_does_not_lose_the_run(self, api, tmp_path) -> None:
        assert api.import_dataset(str(make_images(tmp_path / "one", 1)))["ok"]
        status = self._run(api)
        assert status["status"] == "done", status.get("error")
        text = Path(status["result"]["report_markdown_path"]).read_text(encoding="utf-8")
        assert "Reconstruction FAILED" in text


class TestWhatThisInstallationCannotWrite:
    def test_a_missing_writer_is_refused_by_name_and_html_still_works(self, surveyed,
                                                                      monkeypatch) -> None:
        """The launcher's old environment had no reportlab: PDF vanished silently."""
        from core import report_engine

        real = report_engine._importable
        monkeypatch.setattr(report_engine, "_importable",
                            lambda module: False if module == "reportlab" else real(module))
        refused = surveyed.generate_report("", "R", "standard", "", "", "", "", ["html", "pdf"])
        assert not refused["ok"] and "reportlab" in refused["error"]
        readiness = surveyed.report_readiness("", "standard", ["html", "pdf"])
        assert not readiness["ready"]
        assert surveyed.generate_report("", "R", "standard", "", "", "", "", ["html"])["ok"]


class TestTheAuditTrail:
    def test_it_exports_the_desktop_events(self, surveyed) -> None:
        import csv

        surveyed.generate_report("", "R")
        result = surveyed.export_audit_log("csv")
        with open(result["path"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        kinds = [row["event_type"] for row in rows]
        assert "dataset_imported" in kinds and "mission_version_saved" in kinds
        assert "report_generated" in kinds
        assert result["events"] == len(rows)

    def test_the_report_prints_the_same_trail(self, surveyed) -> None:
        page = Path(surveyed.generate_report("", "R")["html_path"]).read_text(encoding="utf-8")
        assert "mission_version_saved" in page and "dataset_imported" in page


@pytest.mark.skipif(__import__("os").name != "nt", reason="the launcher is a Windows batch file")
def test_the_launcher_names_an_environment_that_can_write_reports(monkeypatch) -> None:
    import subprocess

    launcher = Path(__file__).resolve().parents[1] / "OpenDroneKit.bat"
    monkeypatch.setenv("ODK_LAUNCH_CHECK_ONLY", "1")
    monkeypatch.delenv("ODK_CONDA_ENV", raising=False)
    completed = subprocess.run(["cmd", "/c", str(launcher)], capture_output=True, text=True,
                               timeout=120)
    if "Could not locate conda" in completed.stdout:
        pytest.skip("no conda installation on this machine")
    assert "Using conda environment" in completed.stdout
    env = completed.stdout.split("Using conda environment:")[1].split()[0]
    # Whichever environment it picked, it either can write reports or says it cannot.
    check = subprocess.run(["cmd", "/c", "conda", "run", "-n", env, "python", "-c",
                            "import reportlab, docx"], capture_output=True, text=True, timeout=300)
    if check.returncode != 0:
        assert "cannot write PDF/Word reports" in completed.stdout


class TestPipelineDetectionsBecomeReviewableFindings:
    """Only a trained model's detections become findings; heuristic output does not."""

    def _summary(self, tmp_path: Path, image_dir: Path, used: str) -> Path:
        run = tmp_path / "analysis" / "run_20260101_000000"
        run.mkdir(parents=True)
        summary = {
            "input_image_dir": str(image_dir),
            "images": [{
                "image": "IMG_0000.jpg",
                "structural_model_used": used,
                "structural_detections": [
                    {"label": "spalling", "confidence": 0.83, "bbox_xyxy": [10, 10, 60, 40],
                     "area_px": 1500},
                ],
                "solar_model_used": "heuristic",
                "solar_detections": [],
            }],
        }
        path = run / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_model_detections_are_recorded_as_unreviewed_model_findings(self, surveyed,
                                                                         tmp_path) -> None:
        from app.report_builder import persist_pipeline_findings

        root = root_of(surveyed)
        summary = self._summary(root, tmp_path / "imagery", "onnx:structural.onnx")
        assert persist_pipeline_findings(root, 1, summary)["created"] == 1
        # Idempotent: the same run is not recorded twice.
        assert persist_pipeline_findings(root, 1, summary)["already_recorded"]
        found = surveyed.find_annotations()["annotations"]
        assert len(found) == 1
        finding = found[0]
        assert finding["created_by"] == "model:onnx:structural.onnx"
        assert finding["status"] == "open" and finding["severity"] == "info"
        assert "0.83" in finding["note"]
        text = pdf_text(surveyed.export_report("pdf")["path"])
        assert "not been confirmed by an inspector" in text

    def test_heuristic_detections_are_not_turned_into_findings(self, surveyed, tmp_path) -> None:
        from app.report_builder import persist_pipeline_findings

        root = root_of(surveyed)
        summary = self._summary(root, tmp_path / "imagery", "heuristic")
        result = persist_pipeline_findings(root, 1, summary)
        assert result["created"] == 0 and result["skipped_heuristic"] == 1
        assert surveyed.find_annotations()["annotations"] == []
