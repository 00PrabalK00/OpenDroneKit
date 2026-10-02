"""Standards reports and report verification through the desktop Api.

Real Api, real store, real project folder, real signing key in a temporary data root.
The imagery is a fixture (three small geotagged JPEGs) because what is tested here is
the plumbing -- which facts reach the report, the manifest and the signature -- not a
measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("pypdf")
pytest.importorskip("reportlab")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from test_reports_in_the_desktop_window import _geotagged_jpeg  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch) -> Api:
    monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "odk-data"))
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Bridge 14", str(tmp_path / "project"))["ok"]
    folder = tmp_path / "imagery"
    folder.mkdir()
    for index in range(3):
        _geotagged_jpeg(folder / f"IMG_{index:04d}.jpg", 41.3038 + index * 1e-4, -81.7504,
                        80 + 20 * index)
    assert api.import_dataset(str(folder), "North span")["ok"]
    return api


def test_the_templates_are_listed_with_their_sources(api):
    result = api.standards_templates()
    assert result["ok"]
    keys = {t["key"] for t in result["templates"]}
    assert keys == {"iec62446_3", "bridge_snbi", "roof", "telecom_tower"}
    assert all(t["sources"] and t["structured_following"].startswith("Structured following")
               for t in result["templates"])


def test_entered_values_must_name_who_entered_them_and_known_items_only(api):
    assert not api.save_standards_facts("bridge_snbi", {"fields": {"cond.deck": "7"}})["ok"]
    bad = api.save_standards_facts("bridge_snbi", {"fields": {"made.up": "x"}}, "J. Roe")
    assert not bad["ok"] and "Unknown item" in bad["error"]
    bad_col = api.save_standards_facts("bridge_snbi", {"tables": {"elements": [{"cs9": 1}]}},
                                       "J. Roe")
    assert not bad_col["ok"]
    saved = api.save_standards_facts("bridge_snbi", {"fields": {"cond.deck": "7",
                                                                "cond.scour": ""}}, "J. Roe")
    assert saved["ok"]
    assert saved["fields"] == {"cond.deck": {"value": "7", "source": "entered by J. Roe"}}
    assert api.standards_facts("bridge_snbi")["fields"]["cond.deck"]["value"] == "7"


def test_readiness_counts_what_the_project_supplies(api):
    before = api.standards_report_readiness("bridge_snbi")
    assert before["ok"] and before["inputs"] == 3
    provided_before = before["completeness"]["provided_items"]
    assert provided_before == 1  # B.IE.12 equipment, from the images' EXIF
    api.save_standards_facts("bridge_snbi", {"fields": {"cond.deck": "7", "id.bridge_number": "B-14"},
                                             "tables": {"elements": [
                                                 {"element_number": 12, "total_quantity": 100,
                                                  "cs1": 90, "cs2": 10, "cs3": 0, "cs4": 0}]}},
                             "J. Roe")
    after = api.standards_report_readiness("bridge_snbi")
    assert after["completeness"]["provided_items"] == provided_before + 2
    assert {c["id"]: c["status"] for c in after["checks"]}["cs_sum:12"] == "PASS"
    assert not api.standards_report_readiness("nope")["ok"]


def test_generate_writes_a_signed_report_that_verifies(api, tmp_path):
    analysis = Path(api._session.project_root()) / "analysis" / "run_001"
    analysis.mkdir(parents=True)
    (analysis / "summary.json").write_text(json.dumps({"images": [
        {"image": "IMG_0000.jpg", "structural_model_used": "onnx:structural_v1",
         "structural_model_sha256": "cd" * 32}]}), encoding="utf-8")
    api.save_standards_facts("bridge_snbi", {"fields": {"cond.deck": "7"}}, "J. Roe")
    result = api.generate_standards_report("bridge_snbi", "Span 2 element inspection")
    assert result["ok"], result.get("error")
    assert Path(result["pdf_path"]).is_file()
    assert result["inputs"] == 3 and result["models"] == 1
    assert Path(result["output_dir"]).parent == Path(api._session.project_root()) / "reports" / "standards"
    html = Path(result["html_path"]).read_text(encoding="utf-8")
    assert "entered by J. Roe" in html and "MISSING" in html
    assert "cd" * 32 in html

    verified = api.verify_report(result["output_dir"])
    assert verified["ok"] and verified["verified"], verified["problems"]
    assert verified["signer"] == "this installation's key"
    assert verified["signer_fingerprint"] == api.report_signing_key()["fingerprint"]
    listed = api.list_standards_reports()["reports"]
    assert [r["id"] for r in listed] == [result["id"]]

    Path(api._session.project_root(), "..", "imagery", "IMG_0001.jpg").resolve().write_bytes(b"x")
    tampered = api.verify_report(result["pdf_path"])
    assert tampered["ok"] and not tampered["verified"]
    assert any("no longer match" in p for p in tampered["problems"])


def test_the_signing_key_is_reported_without_its_private_path(api):
    info = api.report_signing_key()
    assert info["ok"] and info["algorithm"] == "Ed25519" and info["fingerprint"]
    assert "private_key_path" not in info


def test_verifying_a_missing_file_says_so(api, tmp_path):
    result = api.verify_report(str(tmp_path / "nothing.pdf"))
    assert result["ok"] and not result["verified"]
    assert "does not exist" in result["problems"][0]
