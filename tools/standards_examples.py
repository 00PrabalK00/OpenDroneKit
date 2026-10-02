"""Generate one example standards-structured report per template from real open data.

    python tools/standards_examples.py --testdata <odk-testdata> --out <folder>

Data (see <odk-testdata>/INDEX.md and each PROVENANCE.md):
* IEC TS 62446-3: THED-PV (FLIR Duo Pro R TLinear frames, CC BY 4.0) with the site's
  environmental logger (POA irradiance, air temperature, wind) and the operator module
  outlines in tests/fixtures/thedpv_module_regions.json.
* Bridge (SNBI): Boga Cayi bridge, DJI H20T wide frames 0100-0180 + thermal orthomosaic.
* Roof: FH Hagenberg building, DJI M30T wide frames 0001-0060 (nadir roof passes and
  oblique facade passes).
* Telecom tower: no open tower dataset is available here, so the report is generated
  with no inputs and every item MISSING.

Every item the data does not state is printed as MISSING. Nothing here fills a field
for the sake of the example.
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
                        "poa_second_sensor": "G_POA_3", "ghi": "GHI_1",
                        "soiling_ratio_1": "Soiling ratio 1"},
        },
        rgb_suffix=("_IR.TIFF", "_8b.JPG"))
    collected["facts"]["observations"].append({
        "text": "The logger has two plane-of-array sensors (G_POA_2 and G_POA_3). The dataset "
                "does not say which matches the plane of the imaged modules; G_POA_2 is "
                "reported as the in-plane irradiance and G_POA_3 is listed above. Both "
                "exceed 600 W/m².",
        "source": "THED-PV Metadata.zip header"})
    return generate_standards_report(
        "iec62446_3", collected["facts"], out, title="THED-PV PV array (example)",
        project="odk-testdata/thermal_pv_thedpv_subset", inputs=collected["inputs"])


def bridge(testdata: Path, out: Path) -> dict:
    root = testdata / "bridge_inspection_h20t_boga_cayi"
    collected = bridge_facts(list_images(root / "B1_Wide", "*_W.JPG"),
                             thermal_raster=root / "orthophoto" / "B1_Aug22.tif")
    return generate_standards_report(
        "bridge_snbi", collected["facts"], out, title="Boga Cayi bridge (example)",
        project="odk-testdata/bridge_inspection_h20t_boga_cayi", inputs=collected["inputs"])


def roof(testdata: Path, out: Path) -> dict:
    collected = roof_facts(list_images(testdata / "thermal_nvs_praschl_zenodo" / "W_subset"))
    return generate_standards_report(
        "roof", collected["facts"], out, title="FH Hagenberg building roof (example)",
        project="odk-testdata/thermal_nvs_praschl_zenodo", inputs=collected["inputs"])


def tower(testdata: Path, out: Path) -> dict:
    facts = {"observations": [{"text": "No telecom tower imagery or assessment record is "
                               "available in the test data, so every item is MISSING.",
                               "source": "odk-testdata/INDEX.md"}]}
    return generate_standards_report("telecom_tower", facts, out,
                                     title="Telecom tower (no data available)",
                                     project="none")


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
        summary[name] = {k: result[k] for k in ("pdf_path", "inputs", "signing_key_fingerprint")}
        summary[name]["provided"] = result["completeness"]["provided_items"]
        summary[name]["required"] = result["completeness"]["required_items"]
        summary[name]["checks"] = {c["id"]: c["status"] for c in result["checks"]}
        print(json.dumps({name: summary[name]}, indent=2))
    (out / "examples_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
