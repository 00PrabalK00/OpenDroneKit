"""A report can be produced from the window the application actually opens.

The audit that prompted this found that index.html -- the page main.py opens -- had no
report action at all; its only report path (Run Full Pipeline) crashed on a keyword
argument; findings could not be recorded from any screen; and the cockpit's Generate
Report described an empty mirror project written outside the user's folder.

These tests click through the shipped page in headless Chromium, with the real Api
behind the bridge (see desktop_window.py), and then open the files it wrote.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from desktop_window import run_page

pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")
pytest.importorskip("docx")


def _geotagged_jpeg(path: Path, lat: float, lon: float, shade: int) -> None:
    from PIL import Image

    image = Image.new("RGB", (160, 120), (shade, 120, 90))
    for x in range(40, 120):
        image.putpixel((x, 60), (20, 20, 20))  # something to crop around
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    exif[0x0110] = "Model 1"

    def dms(value: float) -> tuple[float, float, float]:
        value = abs(value)
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        return (float(degrees), float(minutes), (value - degrees - minutes / 60) * 3600)

    exif[0x8825] = {1: "N" if lat >= 0 else "S", 2: dms(lat),
                    3: "E" if lon >= 0 else "W", 4: dms(lon)}
    image.save(path, "JPEG", exif=exif)


DRIVER = r"""
const $ = (s) => document.querySelector(s);
const out = {};

async function recordFinding(image, label, severity, bbox, note) {
  $('#btn-finding').click();
  await __waitFor(() => __modalOpen('Record Finding'));
  $('#fd-image').value = image;
  $('#fd-label').value = label;
  $('#fd-severity').value = severity;
  $('#fd-bbox').value = bbox;
  $('#fd-note').value = note;
  __clickAction('Record');
  await __waitFor(() => $('#status-message').textContent.startsWith('Finding recorded: ' + label));
  await __waitFor(() => $('#modal-backdrop').classList.contains('hidden'));
}

await recordFinding('IMG_0001.jpg', 'crack', 'high', '30,40,130,80', 'hairline crack at the joint');
await recordFinding('IMG_0002.jpg', 'spalling', 'medium', '', 'surface loss near drain');
out.reportInfo = $('#report-info').textContent;

$('#btn-review').click();
await __waitFor(() => __modalOpen('Findings and Review') && document.querySelector('button[data-review]'));
const spallRow = Array.from(document.querySelectorAll('#modal-body .info > div'))
  .find((row) => row.textContent.includes('spalling'));
spallRow.querySelector('button[data-decision="accept"]').click();
await __waitFor(() => Array.from(document.querySelectorAll('#modal-body .info > div'))
  .some((row) => row.textContent.includes('spalling') && row.textContent.includes('resolved')));
out.reviewText = $('#modal-body').textContent;
__clickAction('Close');

$('#btn-report').click();
await __waitFor(() => __modalOpen('Generate Report')
  && /Ready|Warning|Missing|Note/.test($('#rp-readiness').textContent));
out.readiness = $('#rp-readiness').textContent;
$('#rp-title').value = 'Culvert 7 inspection';
$('#rp-org').value = 'Acme Surveying';
$('#rp-client').value = 'County Roads';
$('#rp-author').value = 'P. Inspector';
$('#rp-docx').checked = true;
$('#rp-recs').value = 'Seal the crack before winter.';
__clickAction('Generate');
await __waitFor(() => __modalOpen('Report Written'), 200000);
out.written = $('#modal-body').textContent;
out.openButtons = Array.from(document.querySelectorAll('#modal-body button[data-open-report]'))
  .map((b) => b.dataset.kind);
__clickAction('Close');

$('#btn-reports').click();
await __waitFor(() => __modalOpen('Generated Reports'));
out.listed = document.querySelectorAll('#modal-body button[data-kind="html"]').length;
__clickAction('Close');

out.notificationButton = $('#btn-notifications').textContent;
$('#btn-notifications').click();
await __waitFor(() => __modalOpen('Notifications'));
out.notifications = $('#modal-body').textContent;
__clickAction('Mark all read');
await __waitFor(() => $('#btn-notifications').textContent === 'Notifications');

window.odk.onMenu({ action: 'tools.audit' });
await __waitFor(() => __modalOpen('Audit Log'));
out.audit = $('#modal-body').textContent;
__finish(out);
"""


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    base = tmp_path_factory.mktemp("window")
    images = base / "imagery"
    images.mkdir()
    _geotagged_jpeg(images / "IMG_0001.jpg", 41.3038, -81.7504, 90)
    _geotagged_jpeg(images / "IMG_0002.jpg", 41.3041, -81.7505, 110)
    _geotagged_jpeg(images / "IMG_0003.jpg", 41.3044, -81.7506, 130)

    api = Api(AppSession(ProjectStore(base / "state.db")))
    root = base / "project"
    assert api.create_project("Culvert 7", str(root))["ok"]
    assert api.import_dataset(str(images), "Culvert imagery")["ok"]
    assert api.set_aoi([[-81.7510, 41.3035], [-81.7500, 41.3035],
                        [-81.7500, 41.3045], [-81.7510, 41.3045]])["ok"]
    assert api.plan_mission({"altitude_m": 60, "mode": "grid"})["ok"]
    assert api.save_mission("Culvert grid", "first flight")["ok"]

    # A job that fails for a real reason, through the public Api, so the Notifications
    # button has something true to show.
    empty = base / "empty"
    empty.mkdir()
    job = api.run_reconstruction({"image_dir": str(empty), "engine": "custom"})
    assert job["ok"], job
    for _ in range(600):
        if api.job_status(job["job_id"])["job"]["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    result = run_page(api, DRIVER, base / "profile")
    assert "error" not in result, result
    return result, root


class TestTheWindowProducesAReport:
    def test_findings_recorded_in_the_window_are_counted(self, driven) -> None:
        result, _ = driven
        assert "2 (2 not reviewed)" in result["reportInfo"]

    def test_accepting_a_finding_is_shown_as_resolved(self, driven) -> None:
        result, _ = driven
        assert "resolved" in result["reviewText"]

    def test_readiness_is_judged_on_the_open_project(self, driven) -> None:
        """It used to answer for an empty mirror and always say ready."""
        result, _ = driven
        assert "No defect run found" in result["readiness"]
        assert "No dataset imported" not in result["readiness"]

    def test_the_report_is_written_into_the_project_folder(self, driven) -> None:
        result, root = driven
        folders = [p for p in (root / "reports").iterdir() if p.is_dir()]
        assert len(folders) == 1
        assert {"html", "pdf", "docx"} <= set(result["openButtons"])
        assert str(root / "reports") in result["written"]
        assert result["listed"] == 1

    def test_the_documents_describe_this_project(self, driven) -> None:
        """Datasets, the saved mission and both findings reach every format."""
        _, root = driven
        folder = next(p for p in (root / "reports").iterdir() if p.is_dir())
        page = (folder / "report.html").read_text(encoding="utf-8")
        assert "Culvert imagery" in page and "Culvert grid" in page
        assert "hairline crack at the joint" in page
        assert "No datasets imported" not in page and "No missions saved" not in page

        text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(folder / "report.pdf").pages)
        for expected in ("Culvert 7 inspection", "Acme Surveying", "County Roads", "crack",
                         "spalling", "Seal the crack before winter.", "Culvert grid"):
            assert expected in text, expected
        reader = pypdf.PdfReader(folder / "report.pdf")
        assert sum(len(page.images) for page in reader.pages) >= 2, "no photos in the PDF"

        from docx import Document

        document = Document(str(folder / "report.docx"))
        assert len(document.inline_shapes) >= 2

    def test_the_audit_log_shows_what_each_event_was_about(self, driven) -> None:
        result, _ = driven
        assert "finding_recorded" in result["audit"]
        assert "label: crack" in result["audit"]
        assert "report_generated" in result["audit"]


class TestNotificationsReachTheDefaultWindow:
    def test_a_failed_job_is_announced_with_its_reason(self, driven) -> None:
        result, _ = driven
        assert result["notificationButton"] == "Notifications (1)"
        assert "failed" in result["notifications"].lower()
