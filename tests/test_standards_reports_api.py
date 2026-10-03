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
    # From the images: B.IE.12 equipment (EXIF Make/Model) and B.L.05/06 (GPS centroid).
    assert provided_before == before["completeness"]["derived_items"] >= 3
    assert before["status"] == "DRAFT" and before["todo"]
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
    assert "entered by J. Roe" in html and "Before this report can be issued" in html
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


def test_generation_is_refused_with_a_reason_when_signing_is_unavailable(api, monkeypatch):
    from core import report_signing as rs

    def unavailable():
        raise rs.SigningUnavailable("Report signing needs the 'cryptography' package")

    monkeypatch.setattr(rs, "_crypto", unavailable)
    result = api.generate_standards_report("bridge_snbi", "Refused")
    assert not result["ok"]
    assert "cryptography" in result["error"]
    assert api.list_standards_reports()["reports"] == []


# ------------------------------------------------ inspection details and records


def test_operator_profiles_fill_the_personnel_items(api):
    assert not api.save_operator_profile({"name": ""}, "Admin")["ok"]
    assert not api.save_operator_profile({"name": "A"}, "")["ok"]  # who saved it is required
    saved = api.save_operator_profile({"name": "Dana Ortiz", "role": "thermographer",
                                       "qualification": "Level 2 thermographer (ISO 18436-7)",
                                       "certificate_number": "TH-2291"}, "Admin")
    assert saved["ok"]
    pid = saved["profile"]["id"]
    assert [p["id"] for p in api.list_operator_profiles()["profiles"]] == [pid]
    bad = api.set_inspection_details("iec62446_3", {"details": {"people": {
        "people.thermographer": ["op_nope"]}}}, "Dana Ortiz")
    assert not bad["ok"]
    wrong_item = api.set_inspection_details("iec62446_3", {"details": {"people": {
        "camera.netd_k": [pid]}}}, "Dana Ortiz")
    assert not wrong_item["ok"]
    ok = api.set_inspection_details("iec62446_3", {"details": {"people": {
        "people.thermographer": [pid]}}}, "Dana Ortiz")
    assert ok["ok"], ok.get("error")
    field = {f["id"]: f for f in ok["form"]["fields"]}["people.thermographer"]
    assert field["value"] == "Dana Ortiz, Level 2 thermographer (ISO 18436-7), cert. TH-2291"
    assert field["origin"] == "profile" and "operator profile 'Dana Ortiz'" in field["source"]
    assert ok["form"]["completeness"]["reused_items"] >= 1


def test_inspection_details_round_trip_and_drive_the_form(api):
    form = api.get_inspection_details("iec62446_3")
    assert form["ok"]
    whos = {f["id"]: f["who"] for f in form["form"]["fields"]}
    assert whos["people.pv_expert"] == "profile" and whos["scope.bos_location"] == "operator"
    assert not api.set_inspection_details("iec62446_3", {"fields": {"x.y": "1"}}, "J. Roe")["ok"]
    assert not api.set_inspection_details("iec62446_3", {"fields": {"scope.bos_location": "x"}},
                                          "")["ok"]
    saved = api.set_inspection_details(
        "iec62446_3", {"details": {"client": "Sunfield Energy Ltd"},
                       "fields": {"scope.bos_location": "Inverter station 2, north fence"}},
        "J. Roe")
    assert saved["ok"], saved.get("error")
    again = api.get_inspection_details("iec62446_3")
    stored = again["stored"]
    assert stored["details"]["client"]["value"] == "Sunfield Energy Ltd"
    assert stored["details"]["client"]["entered_by"] == "J. Roe"
    field = {f["id"]: f for f in again["form"]["fields"]}["scope.bos_location"]
    assert field["value"] == "Inverter station 2, north fence"
    assert field["source"].startswith("entered by J. Roe") and field["origin"] == "operator"
    cleared = api.set_inspection_details("iec62446_3", {"fields": {"scope.bos_location": ""}},
                                         "J. Roe")
    assert "scope.bos_location" not in cleared["stored"]["fields"]


def test_module_lookup_fills_component_type_and_efficiency_from_the_cec_list(api):
    found = api.search_pv_modules("Ablytek 6MN6A270")
    assert found["ok"] and found["modules"][0]["name"] == "Ablytek 6MN6A270"
    assert "NREL SAM" in found["database"]["name"]
    unknown = api.set_inspection_details("iec62446_3", {"details": {"module_model": "Nope 9000"}},
                                         "J. Roe")
    assert not unknown["ok"] and "not in the" in unknown["error"]
    saved = api.set_inspection_details("iec62446_3", {"details": {"module_model": "Ablytek 6MN6A270"}},
                                       "J. Roe")
    fields = {f["id"]: f for f in saved["form"]["fields"]}
    # 270 W / (1.6269 m² x 1000 W/m²) = 16.60 %
    assert fields["scope.module_efficiency"]["value"].startswith("16.60 %")
    assert "STC 270 W / (A_c 1.6269 m²" in fields["scope.module_efficiency"]["source"]
    assert fields["scope.component_types"]["value"].startswith("Ablytek 6MN6A270, Mono-c-Si")
    assert fields["scope.component_types"]["origin"] == "lookup"


def test_a_default_procedure_is_only_a_suggestion_until_someone_adopts_it(api):
    before = {f["id"]: f for f in api.get_inspection_details("iec62446_3")["form"]["fields"]}
    assert before["inspection.procedure"]["status"] == "MISSING"
    assert before["inspection.procedure"]["suggestion"]["value"].startswith("UAV-borne")
    assert not api.save_procedure_template("iec62446_3", "text", "detailed", "")["ok"]
    adopted = api.save_procedure_template("iec62446_3", "Our own procedure.", "detailed (clause 7)",
                                          "Dana Ortiz")
    assert adopted["ok"] and adopted["procedure"]["adopted"]
    after = {f["id"]: f for f in api.get_inspection_details("iec62446_3")["form"]["fields"]}
    assert after["inspection.procedure"]["value"] == "Our own procedure."
    assert "edited by Dana Ortiz" in after["inspection.procedure"]["source"]
    assert after["inspection.type"]["value"] == "detailed (clause 7)"


def test_site_conditions_log_fills_cloud_and_weather(api):
    assert not api.add_site_conditions({"observed_at": "2024-12-21T12:02",
                                        "cloud_okta": 11}, "J. Roe")["ok"]
    assert not api.add_site_conditions({"observed_at": "2024-12-21T12:02",
                                        "cloud_type": "purple"}, "J. Roe")["ok"]
    rec = api.add_site_conditions({"observed_at": "2024-12-21T12:02", "cloud_okta": 1,
                                   "cloud_type": "cumulus", "wind_speed_m_s": 3.0,
                                   "notes": "clear"}, "J. Roe")
    assert rec["ok"], rec.get("error")
    assert api.list_site_conditions()["records"][0]["recorded_by"] == "J. Roe"
    # The fixture's single record is used (one record, any time) for the roof weather item.
    form = {f["id"]: f for f in api.get_inspection_details("roof")["form"]["fields"]}
    assert "cloud okta 1.0" in form["general.weather"]["value"]
    assert form["general.weather"]["origin"] == "site"


def test_a_confirmed_coa_replaces_the_suggestion_and_names_the_confirmer(api):
    from app import standards_builder
    from core import inspection_records

    assert not api.confirm_anomaly_class("iec62446_3", "f.tif:m1", 4, "Dana")["ok"]
    assert not api.confirm_anomaly_class("iec62446_3", "f.tif:m1", 2, "")["ok"]
    done = api.confirm_anomaly_class("iec62446_3", "f.tif:m1", 2, "Dana Ortiz",
                                     "single hot cell", "replace module")
    assert done["ok"] and done["confirmation"]["confirmed_by"] == "Dana Ortiz"
    facts = {"fields": {}, "tables": {"anomalies": [
        {"key": "f.tif:m1", "object": "m1", "source": "decoder",
         "suggestions": {"coa": {"value": "CoA 1", "basis": "x", "source": "y"}}}]},
        "findings": [{"key": "f.tif:m1", "severity": {"label": "CoA 1 suggested", "level": 1}}]}
    root = Path(api._session.project_root())
    standards_builder._apply_details(
        "iec62446_3", facts, inspection_records.load_inspection_details(root, "iec62446_3"), root)
    row = facts["tables"]["anomalies"][0]
    assert row["coa"] == "CoA 2" and row["action"] == "replace module"
    assert "CoA confirmed by Dana Ortiz" in row["source"] and "suggestions" not in row
    assert facts["findings"][0]["severity"] == {"label": "CoA 2", "level": 2, "confirmed": True}


def test_surface_planes_measure_tilt_and_azimuth_with_residuals(api, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    # A synthetic DSM is fine here: the test checks the geometry, not a site. A plane
    # rising 0.5 m per metre northwards faces south at atan(0.5) = 26.57°.
    rows, cols = 40, 40
    y = np.arange(rows)[:, None] * np.ones((1, cols))
    dsm = (300.0 + (rows - 1 - y) * 0.5).astype("float32")
    path = tmp_path / "dsm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1,
                       dtype="float32", crs="EPSG:32633",
                       transform=from_origin(464000, 5357400, 1.0, 1.0)) as dst:
        dst.write(dsm, 1)
    result = api.measure_surface_planes(str(path), [{"label": "array A", "polygon": [
        [464005, 5357395], [464035, 5357395], [464035, 5357365], [464005, 5357365]]}])
    assert result["ok"], result.get("error")
    surface = result["surfaces"][0]
    assert surface["tilt_deg"] == 26.6 and surface["accepted"]
    assert 178 < surface["azimuth_deg"] < 182 and surface["facing"] == "S"
    assert surface["rmse_m"] < 0.001
    assert not api.measure_surface_planes(str(path))["ok"]
