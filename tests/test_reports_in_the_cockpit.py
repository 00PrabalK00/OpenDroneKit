"""The cockpit's Reports workspace, clicked through in a real browser against a real Api.

What the audit found here: Generate Report described an empty mirror project; Export PDF
was an alias that never produced a PDF; Branding defaulted to "DEMO organisation" and
wrote it into the client's document; the structure tree carried a hardcoded "6"
findings; the Annotate tool armed a cursor nothing read; Save Template saved a mission;
and the readiness panel claimed provenance nothing wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from desktop_window import run_page
from test_reports_in_the_desktop_window import _geotagged_jpeg

pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")
pytest.importorskip("docx")

DRIVER = r"""
const q = (s) => document.querySelector(s);
const qa = (s) => Array.from(document.querySelectorAll(s));
const seen = [];
new MutationObserver(() => qa('.toast').forEach((t) => {
  if (!seen.includes(t.textContent)) seen.push(t.textContent);
})).observe(document.body, { childList: true, subtree: true });

const tool = (label) => {
  const button = qa('.tbtn').find((b) => b.textContent === label);
  if (!button) throw new Error('no toolbar button ' + label);
  button.click();
};
async function answer(value) {
  await __waitFor(() => q('.modal-field input, .modal-field select'));
  const input = q('.modal-field input, .modal-field select');
  input.value = value;
  qa('.modal-actions .tbtn').find((b) => b.classList.contains('primary')).click();
  await __waitFor(() => !document.body.contains(input));
}
function setField(label, value) {
  const field = qa('.field').find((f) => f.querySelector('label').textContent === label);
  const input = field.querySelector('input, select');
  input.value = value;
  input.dispatchEvent(new Event('change'));
}
const out = {};

// Branding starts empty: nothing is put in the client's document on the user's behalf.
await __waitFor(() => qa('.field').some((f) => f.querySelector('label').textContent === 'Organisation'));
out.brandingDefaults = qa('.field').filter((f) => ['Organisation', 'Logo file', 'Client']
  .includes(f.querySelector('label').textContent)).map((f) => f.querySelector('input').value);
await __waitFor(() => /warning|note|ready|missing/.test(qa('.console .line').map((l) => l.textContent).join(' ')));
out.readiness = qa('.console .line').map((l) => l.textContent).join(' | ');

// Record a finding through the Annotate button, the way an inspector would.
tool('Record Finding');
await answer('IMG_0001.jpg');
await answer('efflorescence');
await answer('high');
await answer('white deposit under the deck');
await answer('20,20,120,90');
await __waitFor(() => seen.some((t) => t.startsWith('Recorded efflorescence')));

await __waitFor(() => qa('.tree-node').some((n) => n.textContent.includes('Findings')
  && n.textContent.includes('1')));
out.tree = qa('.tree-node').map((n) => n.textContent).join(' | ');

setField('Organisation', 'Acme Surveying');
setField('Client', 'County Roads');
setField('Format', 'PDF+DOCX');
tool('Generate');
await answer('Culvert 9 inspection');
await __waitFor(() => seen.some((t) => t.startsWith('Report written')), 200000);
out.generated = seen.find((t) => t.startsWith('Report written'));

tool('Export Word');
await answer('Culvert 9 word copy');
await __waitFor(() => seen.filter((t) => t.startsWith('Report written')).length >= 2, 200000);

tool('Save Template');
await answer('county');
await __waitFor(() => seen.some((t) => t.startsWith('Saved report template')));

await __waitFor(() => qa('table tr').some((r) => r.textContent.includes('Culvert 9 word copy')));
out.listed = qa('table tr').filter((r) => r.textContent.includes('Culvert 9')).length;
out.seen = seen;
__finish(out);
"""


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    base = tmp_path_factory.mktemp("cockpit")
    images = base / "imagery"
    images.mkdir()
    for index in range(3):
        _geotagged_jpeg(images / f"IMG_{index:04d}.jpg", 41.3038 + index * 1e-4, -81.7504, 90)
    api = Api(AppSession(ProjectStore(base / "state.db")))
    root = base / "project"
    assert api.create_project("Culvert 9", str(root))["ok"]
    assert api.import_dataset(str(images), "Deck imagery")["ok"]
    result = run_page(
        api, DRIVER, base / "profile", page_name="workspace.html",
        before_js="localStorage.setItem('odk.workspace.last', 'reports');",
        ready_js="Array.from(document.querySelectorAll('.tbtn')).some("
                 "(b) => b.textContent === 'Generate')",
    )
    assert "error" not in result, result
    return result, root, api


class TestTheReportsWorkspace:
    def test_branding_starts_empty(self, driven) -> None:
        result, _, _ = driven
        assert result["brandingDefaults"] == ["", "", ""]

    def test_readiness_is_about_this_project_and_claims_nothing_false(self, driven) -> None:
        result, _, _ = driven
        assert "No findings recorded" in result["readiness"]
        assert "provenance together" not in result["readiness"]

    def test_the_structure_counts_the_real_findings(self, driven) -> None:
        result, _, _ = driven
        assert "Thermal" not in result["tree"] and "Change Analysis" not in result["tree"]
        assert "Findings1" in result["tree"].replace(" ", "")

    def test_generate_writes_pdf_and_word_into_the_project(self, driven) -> None:
        result, root, _ = driven
        assert "html, pdf, docx" in result["generated"]
        folders = sorted(p for p in (root / "reports").iterdir() if p.is_dir())
        assert len(folders) == 2
        first = next(f for f in folders if "culvert-9-inspection" in f.name)
        text = " ".join((p.extract_text() or "") for p in pypdf.PdfReader(first / "report.pdf").pages)
        assert "Acme Surveying" in text and "County Roads" in text
        assert "efflorescence" in text and "DEMO" not in text
        second = next(f for f in folders if "word-copy" in f.name)
        assert (second / "report.docx").is_file() and not (second / "report.pdf").exists()

    def test_the_generated_reports_are_listed_on_the_canvas(self, driven) -> None:
        result, _, _ = driven
        assert result["listed"] == 2

    def test_save_template_saves_report_settings_not_a_mission(self, driven) -> None:
        _, _, api = driven
        templates = {t["name"]: t for t in api.report_templates()["templates"]}
        assert templates["county"]["organization"] == "Acme Surveying"
        assert not api.list_mission_versions()["versions"]
