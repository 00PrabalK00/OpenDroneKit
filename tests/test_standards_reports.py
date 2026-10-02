"""Standards-structured reports: every required item is supplied with a source or MISSING.

The structural tests run on every machine. The real-data tests read the openly licensed
odk-testdata collection (``ODK_TESTDATA_DIR``; see its INDEX.md) and skip with the reason
when it is absent -- nothing here is synthesised to stand in for a real file.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")

from core import report_signing as rs  # noqa: E402
from core.standards_reports import (  # noqa: E402
    MISSING,
    NOT_CERTIFIED,
    TEMPLATES,
    beaufort_from_m_s,
    generate_standards_report,
    get_template,
    resolve,
)

REPO = Path(__file__).resolve().parents[1]
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "").strip() or "__unset__")
THEDPV = TESTDATA / "thermal_pv_thedpv_subset"
THEDPV_SESSION = THEDPV / "Raw Dataset" / "2024-12-21" / "12" / "20241221_120152"
BRIDGE = TESTDATA / "bridge_inspection_h20t_boga_cayi"
FACADE = TESTDATA / "thermal_nvs_praschl_zenodo"
REGIONS = REPO / "tests" / "fixtures" / "thedpv_module_regions.json"
CLAIMS = re.compile(r"\b(complies|compliant|compliance with|certified|conforms|conformant)\b",
                    re.IGNORECASE)


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not found at {path}; set ODK_TESTDATA_DIR to the odk-testdata folder.")
    return path


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "odk-data"))
    return tmp_path / "odk-data"


def _pdf_text(path: str) -> str:
    return "\n".join(page.extract_text() for page in pypdf.PdfReader(path).pages)


class TestTemplates:
    @pytest.mark.parametrize("key", sorted(TEMPLATES))
    def test_each_template_cites_public_sources_and_says_structured_following(self, key):
        template = TEMPLATES[key]
        assert template.sources and all(s.url.startswith("https://") for s in template.sources)
        assert template.structured_following.startswith("Structured following ")
        assert all(item.ref for item in template.items())
        ids = [item.id for item in template.items()]
        assert len(ids) == len(set(ids))

    def test_the_iec_template_carries_the_clause_8_and_table_3_items(self):
        ids = {item.id for item in TEMPLATES["iec62446_3"].items()}
        for required in ("env.irradiance_poa_w_m2", "env.wind_speed_m_s", "env.air_temperature_c",
                         "env.cloud_okta", "camera.make_model", "camera.emissivity",
                         "camera.reflected_temperature_c", "inspection.location"):
            assert required in ids
        anomalies = TEMPLATES["iec62446_3"].tables()[0]
        assert {"delta_t_k", "coa", "xy_position"} <= {c.key for c in anomalies.columns}

    def test_the_bridge_template_carries_the_snbi_element_items(self):
        table = {t.id: t for t in TEMPLATES["bridge_snbi"].tables()}["elements"]
        assert "B.CS.01-04" in table.ref
        assert [c.key for c in table.columns][-4:] == ["cs1", "cs2", "cs3", "cs4"]

    def test_an_unknown_template_is_refused(self):
        with pytest.raises(ValueError, match="Unknown standards template"):
            get_template("iso9001")


class TestNoDefaults:
    @pytest.mark.parametrize("key", sorted(TEMPLATES))
    def test_with_no_facts_every_item_is_missing_and_says_so(self, key, tmp_path):
        content = resolve(TEMPLATES[key], {})
        items = [i for s in content["sections"] for i in s["items"]]
        assert items and all(i["status"] == MISSING and i["value"] is None for i in items)
        assert content["completeness"]["provided_items"] == 0
        assert all(c["status"] in (MISSING,) for c in content["checks"])
        result = generate_standards_report(key, {}, tmp_path / "r")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert html.count(">MISSING<") >= len(items)
        assert _pdf_text(result["pdf_path"]).count("MISSING") >= len(items)

    @pytest.mark.parametrize("key", sorted(TEMPLATES))
    def test_no_report_claims_compliance_or_certification(self, key, tmp_path):
        result = generate_standards_report(key, {}, tmp_path / "r")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        # SNBI item B.IE.04 is named "Nationally Certified Bridge Inspector": a field
        # label, not a claim about the report.
        text = (re.sub(r"<[^>]+>", " ", html).replace(NOT_CERTIFIED, "")
                .replace("Nationally Certified Bridge Inspector", ""))
        assert not CLAIMS.findall(text)
        assert NOT_CERTIFIED in html
        pdf = re.sub(r"\s+", " ", _pdf_text(result["pdf_path"]))
        assert "It is not a certificate of compliance" in pdf
        assert "Structured following" in pdf

    def test_blank_values_are_not_values(self):
        content = resolve(TEMPLATES["roof"], {"fields": {
            "general.inspector": "", "general.address": None,
            "roof.gutters": {"value": "", "source": "x"}}})
        items = {i["id"]: i for s in content["sections"] for i in s["items"]}
        assert {items[k]["status"] for k in ("general.inspector", "general.address",
                                             "roof.gutters")} == {MISSING}

    def test_a_supplied_value_is_printed_with_its_source(self):
        content = resolve(TEMPLATES["roof"], {"fields": {
            "general.inspector": "J. Roe",
            "roof.covering_type": {"value": "Standing-seam metal", "source": "entered by J. Roe"}}})
        items = {i["id"]: i for s in content["sections"] for i in s["items"]}
        assert items["general.inspector"]["source"] == "entered by operator"
        assert items["roof.covering_type"]["value"] == "Standing-seam metal"
        assert content["completeness"]["provided_items"] == 2

    def test_a_table_row_missing_a_required_column_shows_missing(self):
        content = resolve(TEMPLATES["iec62446_3"], {"tables": {"anomalies": [
            {"object": "M1", "image_file": "a.tif", "delta_t_k": "+12 K"}]}})
        row = content["sections"][5]["tables"][0]["rows"][0]["cells"]
        assert row["coa"] == MISSING and row["temperature_c"] == MISSING
        assert row["serial_number"] == "-"  # optional identification, not required alone
        assert any("Class of abnormality" in m for m in content["completeness"]["missing"])


def _checks(key: str, fields: dict, tables: dict | None = None) -> dict[str, dict]:
    content = resolve(TEMPLATES[key], {"fields": fields, "tables": tables or {}})
    return {c["id"]: c for c in content["checks"]}


class TestIecChecks:
    @pytest.mark.parametrize("poa,status", [(599.9, "FAIL"), (600, "PASS"), (921, "PASS")])
    def test_irradiance_threshold_is_600_w_m2(self, poa, status):
        assert _checks("iec62446_3", {"env.irradiance_poa_w_m2": poa})["irradiance"]["status"] == status

    def test_a_missing_irradiance_is_missing_not_passed(self):
        assert _checks("iec62446_3", {})["irradiance"]["status"] == MISSING

    @pytest.mark.parametrize("speed,force", [(0.1, 0), (3.2, 2), (3.5, 3), (7.9, 4), (8.0, 5),
                                             (33.0, 12)])
    def test_beaufort_follows_the_wmo_scale(self, speed, force):
        assert beaufort_from_m_s(speed) == force

    @pytest.mark.parametrize("speed,status", [(7.9, "PASS"), (8.0, "FAIL")])
    def test_wind_limit_is_4_bft(self, speed, status):
        assert _checks("iec62446_3", {"env.wind_speed_m_s": speed})["wind"]["status"] == status

    def test_cloud_resolution_and_netd_limits(self):
        checks = _checks("iec62446_3", {"env.cloud_okta": 3, "camera.ir_resolution_px": "320 x 240",
                                        "camera.netd_k": 0.05,
                                        "camera.geometric_resolution_cm_px": 3.5})
        assert checks["cloud"]["status"] == "FAIL"
        assert checks["ir_resolution"]["status"] == "PASS"
        assert checks["netd"]["status"] == "PASS"
        assert checks["geometric_resolution"]["status"] == "FAIL"
        assert _checks("iec62446_3", {"camera.ir_resolution_px": "160 x 120"})[
            "ir_resolution"]["status"] == "FAIL"

    def test_each_abnormality_needs_two_position_identifications(self):
        checks = _checks("iec62446_3", {}, {"anomalies": [
            {"object": "A", "serial_number": "S1"},
            {"object": "B", "serial_number": "S2", "xy_position": "row 3 col 7"}]})
        assert checks["identification:A"]["status"] == "FAIL"
        assert checks["identification:B"]["status"] == "PASS"


class TestBridgeAndTowerChecks:
    def test_condition_states_must_sum_to_the_total_quantity(self):
        checks = _checks("bridge_snbi", {}, {"elements": [
            {"element_number": 12, "total_quantity": 100, "cs1": 80, "cs2": 15, "cs3": 5, "cs4": 0},
            {"element_number": 107, "total_quantity": 50, "cs1": 40, "cs2": 5, "cs3": 0, "cs4": 0},
            {"element_number": 215, "total_quantity": 10, "cs1": 10}]})
        assert checks["cs_sum:12"]["status"] == "PASS"
        assert checks["cs_sum:107"]["status"] == "FAIL"
        assert checks["cs_sum:215"]["status"] == MISSING

    def test_condition_ratings_must_be_snbi_codes(self):
        checks = _checks("bridge_snbi", {"cond.deck": "7", "cond.scour": "good"})
        assert checks["code:cond.deck"]["status"] == "PASS"
        assert checks["code:cond.scour"]["status"] == "FAIL"

    def test_a_guyed_mast_without_a_guy_record_is_missing(self):
        assert _checks("telecom_tower", {"site.structure_type": "guyed mast"})["guys"][
            "status"] == MISSING
        assert _checks("telecom_tower", {"site.structure_type": "monopole"})["guys"][
            "status"] == "N/A"


# ------------------------------------------------------------------ real data


class TestEnvironmentLogger:
    def test_a_time_with_no_reading_nearby_is_refused(self):
        from core.standards_collectors import environment_at

        source = str(_need(THEDPV / "Metadata.zip", "THED-PV metadata")) + \
            "::Metadata/environmental_day2.csv"
        found = environment_at(source, datetime(2024, 12, 21, 12, 2, 4),
                               {"poa": "G_POA_2", "air": "TAA"})
        assert found["row_time"] == "2024-12-21 12:02:00"
        assert found["values"] == {"poa": 921.0, "air": 24.2}
        with pytest.raises(ValueError, match="no reading within"):
            environment_at(source, datetime(2024, 12, 21, 23, 30), {"poa": "G_POA_2"})
        with pytest.raises(ValueError, match="matched 2 headers"):
            environment_at(source, datetime(2024, 12, 21, 12, 2), {"poa": "G_POA"})


class TestIecOnRealThermalFrames:
    @pytest.fixture
    def report(self, tmp_path) -> dict:
        _need(THEDPV_SESSION, "THED-PV session")
        from core.standards_collectors import iec_facts_from_frames

        regions = json.loads(REGIONS.read_text(encoding="utf-8"))
        frames = [THEDPV_SESSION / name for name in sorted(regions["frames"])]
        collected = iec_facts_from_frames(
            frames, REGIONS, tmp_path / "work",
            environment={"source": str(THEDPV / "Metadata.zip") + "::Metadata/environmental_day2.csv",
                         "columns": {"poa": "G_POA_2", "air": "TAA", "wind": "WV_2m",
                                     "wind_dir": "WD_2m"}},
            rgb_suffix=("_IR.TIFF", "_8b.JPG"))
        return generate_standards_report("iec62446_3", collected["facts"], tmp_path / "reports",
                                         inputs=collected["inputs"])

    def test_conditions_come_from_the_site_logger_and_pass_the_table_3_limits(self, report):
        content = json.loads((Path(report["output_dir"]) / "content.json").read_text(encoding="utf-8"))
        items = {i["id"]: i for s in content["sections"] for i in s["items"]}
        # Measured 2026-10-02: logger row 12:02 (frames 12:02:04-12:02:27).
        assert items["env.irradiance_poa_w_m2"]["value"] == 921.0
        assert items["env.air_temperature_c"]["value"] == 24.2
        assert items["env.wind_speed_m_s"]["value"] == 3.2
        assert "G_POA_2" in items["env.irradiance_poa_w_m2"]["source"]
        checks = {c["id"]: c["status"] for c in report["checks"]}
        assert checks["irradiance"] == "PASS" and checks["wind"] == "PASS"
        assert checks["cloud"] == MISSING  # the dataset records no cloud cover
        assert items["camera.make_model"]["value"] == "FLIR Duo Pro R"
        assert items["camera.serial_lens"]["value"].startswith("serial 279016")
        assert items["camera.ir_resolution_px"]["value"] == "640 x 512"

    def test_what_the_files_do_not_record_is_missing(self, report):
        content = json.loads((Path(report["output_dir"]) / "content.json").read_text(encoding="utf-8"))
        items = {i["id"]: i for s in content["sections"] for i in s["items"]}
        for missing in ("camera.emissivity", "camera.reflected_temperature_c", "camera.netd_k",
                        "inspection.location", "people.thermographer", "env.cloud_okta"):
            assert items[missing]["status"] == MISSING, missing
        rows = content["sections"][5]["tables"][0]["rows"]
        assert len(rows) == 4
        assert {r["cells"]["coa"] for r in rows} == {MISSING}  # never assigned by software

    def test_module_temperatures_and_delta_t_are_the_decoded_values(self, report, tmp_path):
        from core.india_thermal import module_temperatures_in_image

        regions = json.loads(REGIONS.read_text(encoding="utf-8"))
        name = "20241221_120227_869_IR.TIFF"
        result = module_temperatures_in_image(
            THEDPV_SESSION / name, regions["frames"][name]["regions"], tmp_path / "check",
            region_source="test", warning_delta_c=5, critical_delta_c=10, hot_cell_delta_c=10,
            threshold_basis="test")
        modules = {m["module_id"]: m for m in json.loads(
            Path(result.artifact_paths[0]).read_text(encoding="utf-8"))["modules"]}
        content = json.loads((Path(report["output_dir"]) / "content.json").read_text(encoding="utf-8"))
        rows = [r["cells"] for r in content["sections"][5]["tables"][0]["rows"] if name in r["cells"]["object"]]
        reference = (modules["upper_module"]["median_c"] + modules["lower_module"]["median_c"]) / 2
        for row in rows:
            module = modules[row["object"].split(" ")[0]]
            assert row["temperature_c"].startswith(f"median {module['median_c']:.2f}")
            assert row["delta_t_k"] == f"{module['median_c'] - reference:+.2f} K"
        # Measured 2026-10-02: 48.85 / 44.05 °C medians, so ±2.40 K about their median.
        assert sorted(r["delta_t_k"] for r in rows) == ["+2.40 K", "-2.40 K"]

    def test_the_report_is_signed_over_its_real_inputs(self, report):
        result = rs.verify_report(report["output_dir"])
        assert result["ok"], result["problems"]
        names = {i["name"] for i in result["inputs"]}
        assert {"20241221_120204_881_IR.TIFF", "20241221_120227_869_IR.TIFF",
                "20241221_120204_881_8b.JPG", "Metadata.zip", "thedpv_module_regions.json"} <= names


class TestBridgeAndRoofOnRealImagery:
    def test_bridge_report_reads_equipment_and_leaves_agency_records_missing(self, tmp_path):
        from core.standards_collectors import bridge_facts, list_images

        images = list_images(_need(BRIDGE / "B1_Wide", "bridge imagery"), "*_W.JPG")
        assert len(images) == 81
        collected = bridge_facts(images)
        report = generate_standards_report("bridge_snbi", collected["facts"], tmp_path,
                                           inputs=collected["inputs"])
        content = json.loads((Path(report["output_dir"]) / "content.json").read_text(encoding="utf-8"))
        items = {i["id"]: i for s in content["sections"] for i in s["items"]}
        assert "ZH20T" in items["ie.equipment"]["value"]
        # The photos are geotagged, but B.L.05/06 is the agency's record, not a photo centroid.
        assert items["loc.latitude"]["status"] == MISSING
        assert any("GPS extent" in o["text"] for o in content["observations"])
        assert all(items[k]["status"] == MISSING for k in items if k.startswith(("cond.", "ie.b",
                                                                               "ie.insp")))
        assert content["sections"][3]["tables"][0]["status"] == MISSING
        assert report["inputs"] == 81
        assert rs.verify_report(report["output_dir"])["ok"]

    def test_roof_report_counts_nadir_and_oblique_frames_from_the_gimbal(self, tmp_path):
        from core.standards_collectors import image_metadata, list_images, roof_facts

        images = list_images(_need(FACADE / "W_subset", "building imagery"))
        pitches = [image_metadata(p)["gimbal_pitch_deg"] for p in images]
        nadir = sum(1 for p in pitches if p is not None and p <= -80)
        collected = roof_facts(images)
        method = collected["facts"]["fields"]["general.method"]["value"]
        assert f"{nadir} near-nadir frame(s)" in method
        assert f"{len(images) - nadir} oblique frame(s)" in method
        report = generate_standards_report("roof", collected["facts"], tmp_path,
                                           inputs=collected["inputs"])
        assert report["completeness"]["provided_items"] == 2  # date and method; rest MISSING
        assert rs.verify_report(report["output_dir"])["ok"]
