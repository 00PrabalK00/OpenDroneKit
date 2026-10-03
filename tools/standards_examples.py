"""Generate one example standards-structured report per template from real open data.

    python tools/standards_examples.py --testdata <odk-testdata> --out <folder>

Data (see <odk-testdata>/INDEX.md and each PROVENANCE.md):
* IEC TS 62446-3: THED-PV (FLIR Duo Pro R TLinear frames, CC BY 4.0) with the site's
  environmental logger (POA irradiance, air temperature, wind, DustIQ soiling ratio),
  the paired visual frames as photo evidence, and the operator module outlines in
  tests/fixtures/thedpv_module_regions.json.
* Bridge (SNBI): Boga Cayi bridge, DJI H20T wide frames 0100-0180 + thermal orthomosaic,
  plus the structural detector's per-frame boxes from
  <odk-testdata>/_outputs/detectors/bridge_h20t (proposals, never findings).
* Roof: FH Hagenberg building, DJI M30T wide frames 0001-0060 (nadir roof passes and
  oblique facade passes), plus the DSM/DTM of the COLMAP run in
  <odk-testdata>/_outputs/recon/nvs_facade for the roof facets (suggested).
* Telecom tower: no open tower dataset is available here, so the report is a clean
  draft showing the structure and the checklist of what is needed.

No person, client, procedure adoption or classification is invented for the examples:
everything only a person can supply stays on each report's checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.standards_collectors import (  # noqa: E402
    bridge_facts,
    iec_facts_from_frames,
    list_images,
    roof_facts,
)
from core.standards_reports import generate_standards_report  # noqa: E402

THEDPV_SESSION = Path("thermal_pv_thedpv_subset/Raw Dataset/2024-12-21/12/20241221_120152")
REGIONS = REPO / "tests" / "fixtures" / "thedpv_module_regions.json"


def _procedure_suggestions(facts: dict, template: str) -> None:
    """Offer the built-in procedure as a suggestion (never adopted on anyone's behalf)."""
    from core.inspection_records import get_procedure

    proc = get_procedure(template)
    facts.setdefault("details", {})["procedure"] = proc
    facts.setdefault("fields", {})
    targets = {"iec62446_3": {"inspection.procedure": "text", "inspection.type": "inspection_type"},
               "bridge_snbi": {"ie.type": "inspection_type"}}.get(template, {})
    for item, key in targets.items():
        if proc.get(key) and item not in facts["fields"]:
            facts["fields"][item] = {"value": proc[key], "source": proc["source"],
                                     "origin": "procedure" if proc["adopted"] else "suggested"}


def iec(testdata: Path, out: Path) -> dict:
    session = testdata / THEDPV_SESSION
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    frames = [session / name for name in sorted(regions["frames"])]
    collected = iec_facts_from_frames(
        frames, REGIONS, out / "_work_iec",
        environment={
            "source": str(testdata / "thermal_pv_thedpv_subset" / "Metadata.zip")
            + "::Metadata/environmental_day2.csv",
            "columns": {"poa": "G_POA_2", "air": "TAA", "wind": "WV_2m", "wind_dir": "WD_2m",
                        "soiling": "Soiling ratio 1",
                        "poa_second_sensor": "G_POA_3", "ghi": "GHI_1",
                        "soiling_ratio_2": "Soiling ratio 2"},
        },
        rgb_suffix=("_IR.TIFF", "_8b.JPG"))
    facts = collected["facts"]
    facts["observations"].append({
        "text": "The logger has two plane-of-array sensors (G_POA_2 and G_POA_3). The dataset "
                "does not say which matches the plane of the imaged modules; G_POA_2 is "
                "reported as the in-plane irradiance and G_POA_3 is listed above. Both "
                "exceed 600 W/m².",
        "source": "THED-PV Metadata.zip header"})
    _procedure_suggestions(facts, "iec62446_3")
    return generate_standards_report(
        "iec62446_3", facts, out, title="THED-PV PV array",
        project="odk-testdata/thermal_pv_thedpv_subset", inputs=collected["inputs"])


def bridge(testdata: Path, out: Path) -> dict:
    root = testdata / "bridge_inspection_h20t_boga_cayi"
    det_dir = testdata / "_outputs" / "detectors" / "bridge_h20t"
    detections = None
    if (det_dir / "per_frame.json").is_file():
        detections = {
            "per_frame": det_dir / "per_frame.json", "image_dir": root / "B1_Wide",
            "mode": "structural_letterbox",
            "models": [{"name": "structural_multiclass_detector.onnx",
                        "sha256": "359817c663f4cb7772f44e4370095c116efb9d0123478bc58aa618884380c445",
                        "digest_source": "recorded in _outputs/detectors/bridge.log "
                                         "(measured, matches the model registry)"}]}
    collected = bridge_facts(list_images(root / "B1_Wide", "*_W.JPG"),
                             thermal_raster=root / "orthophoto" / "B1_Aug22.tif",
                             detections=detections)
    _procedure_suggestions(collected["facts"], "bridge_snbi")
    return generate_standards_report(
        "bridge_snbi", collected["facts"], out, title="Boga Cayi bridge",
        project="odk-testdata/bridge_inspection_h20t_boga_cayi", inputs=collected["inputs"],
        models=collected.get("models"))


def roof(testdata: Path, out: Path) -> dict:
    recon = testdata / "_outputs" / "recon" / "nvs_facade"
    dsm, dtm = recon / "dsm.tif", recon / "dtm.tif"
    have_dsm = dsm.is_file() and dtm.is_file()
    collected = roof_facts(
        list_images(testdata / "thermal_nvs_praschl_zenodo" / "W_subset"),
        dsm=dsm if have_dsm else None, dtm=dtm if have_dsm else None,
        dsm_note=("The DSM comes from a sparse COLMAP reconstruction gridded at 2.26 m "
                  "(run_summary.json), so each facet is a coarse estimate."))
    _procedure_suggestions(collected["facts"], "roof")
    return generate_standards_report(
        "roof", collected["facts"], out, title="FH Hagenberg building roof",
        project="odk-testdata/thermal_nvs_praschl_zenodo", inputs=collected["inputs"])


def tower(testdata: Path, out: Path) -> dict:
    facts = {"observations": [{"text": "No telecom tower imagery or assessment record is "
                               "available in the test data, so this draft shows the report's "
                               "structure and what must be supplied.",
                               "source": "odk-testdata/INDEX.md"}],
             "figures": [{"role": "hero", "spec": {"type": "placeholder",
                                                   "text": "No imagery yet: fly the tower "
                                                           "survey to add it"},
                          "caption": "Imagery is added when the survey is flown."}]}
    _procedure_suggestions(facts, "telecom_tower")
    return generate_standards_report("telecom_tower", facts, out,
                                     title="Telecom tower (no data yet)", project="none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testdata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", default=["iec", "bridge", "roof", "tower"])
    args = parser.parse_args(argv)
    testdata, out = Path(args.testdata), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    builders = {"iec": iec, "bridge": bridge, "roof": roof, "tower": tower}
    summary = {}
    for name in args.only:
        result = builders[name](testdata, out)
        comp = result["completeness"]
        summary[name] = {k: result[k] for k in ("pdf_path", "inputs", "signing_key_fingerprint")}
        summary[name].update({
            "provided": comp["provided_items"], "required": comp["required_items"],
            "derived": comp["derived_items"], "reused": comp["reused_items"],
            "entered": comp["entered_items"], "open_actions": comp["open_actions"],
            "open_by_who": comp["open_by_who"],
            "checks": {c["id"]: c["status"] for c in result["checks"]}})
        print(json.dumps({name: summary[name]}, indent=2))
    (out / "examples_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
