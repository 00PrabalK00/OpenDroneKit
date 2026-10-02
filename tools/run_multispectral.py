"""Run the MicaSense multispectral chain on real captures and write a measured report.

    python tools/run_multispectral.py --images <band tiff folder> --out <dir>
        [--panel <panel capture folder>] [--calibration dls|panel] [--no-reconstruct]
        [--stress-scope "..." --severe-below 0.3 --moderate-below 0.5]
        [--cover-scope "..." --max-plant-area 0.25]

Steps: group band TIFFs into captures, calibrate radiance to reflectance (DLS2 sun
sensor, or a panel capture from the same camera and flight), align the bands, solve
camera poses with the COLMAP engine and orthorectify every reflectance band, then run
the agriculture pack (NDVI / NDRE / GNDVI, and stress zones only when a validation
scope is given). When ``--panel`` is given, the panel is also used as an independent
check of the DLS calibration: the DLS reflectance of the panel against its QR albedo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import multispectral as ms  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel", default="", help="folder holding a panel capture (validation / panel calibration)")
    parser.add_argument("--calibration", choices=("dls", "panel"), default="dls")
    parser.add_argument("--no-reconstruct", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse the calibrated stacks from a previous run in --out; only reconstruct")
    parser.add_argument("--stress-scope", default="", help="crop/sensor validation scope for stress thresholds")
    parser.add_argument("--severe-below", type=float, default=None)
    parser.add_argument("--moderate-below", type=float, default=None)
    parser.add_argument("--cover-scope", default="",
                        help="validation scope for NDVI cover classes; enables canopy cover and plant count")
    parser.add_argument("--max-plant-area", type=float, default=None,
                        help="largest footprint (m2) of one plant for the crop and stage; enables plant count")
    parser.add_argument("--analyse-only", action="store_true",
                        help="reuse the band orthomosaic of a previous run in --out; only run the agriculture pack")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}", flush=True)

    if not (args.reuse or args.analyse_only):
        # The same chain the application's "Multispectral to reflectance" job runs.
        stress = ({"severe_below": args.severe_below, "moderate_below": args.moderate_below,
                   "validation_scope": args.stress_scope}
                  if args.stress_scope and args.severe_below is not None and args.moderate_below is not None
                  else None)
        cover = ({"validation_scope": args.cover_scope, "max_plant_area_m2": args.max_plant_area}
                 if args.cover_scope else None)
        report = ms.run_survey_report(args.images, out, panel=args.panel, calibration=args.calibration,
                                      reconstruct=not args.no_reconstruct, stress=stress, cover=cover,
                                      progress=progress)
        print(report["report_path"])
        return 0
    report: dict = {"images": args.images, "calibration": args.calibration}
    started = time.time()

    panel_capture = None
    if args.panel:
        panel_captures = [c for c in ms.group_captures(args.panel) if any(b.panel_albedo for b in c.bands.values())]
        if panel_captures:
            panel_capture = panel_captures[0]
            report["panel_validation"] = ms.validate_panel_against_albedo(panel_capture)

    summary_path = out / "multispectral_summary.json"
    if not summary_path.exists():
        parser.error(f"--reuse/--analyse-only need a previous run in {out}")
    if args.analyse_only:
        survey = json.loads(summary_path.read_text(encoding="utf-8"))
    else:  # --reuse
        survey = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = {Path(c["sfm_image"]).name: c["stack"] for c in survey["captures"]}
        survey["orthomosaic"] = ms.reconstruct_band_orthomosaic(out, survey["bands"], sources, progress=progress)
        summary_path.write_text(json.dumps(survey, indent=2, default=str), encoding="utf-8")
    report["survey"] = {k: v for k, v in survey.items() if k != "captures"}
    report["survey"]["capture_count"] = len(survey["captures"])

    ortho = (survey.get("orthomosaic") or {}).get("reflectance_bands")
    if ortho:
        from core.india_agriculture import compute_vegetation_indices, create_stress_zones

        source = (
            "MicaSense radiance model + DLS2 horizontal irradiance per capture (pi*L/E)"
            if args.calibration == "dls" else "MicaSense radiance model + panel QR albedo"
        )
        calibrations = ms.reflectance_band_calibrations(survey["bands"], source)
        indices = compute_vegetation_indices(ortho, calibrations, out / "agriculture")
        report["vegetation_indices"] = json.loads(Path(indices.summary_path).read_text(encoding="utf-8"))
        if args.stress_scope and args.severe_below is not None and args.moderate_below is not None:
            ndvi = report["vegetation_indices"]["indices"]["NDVI"]["path"]
            stress = create_stress_zones(
                ndvi, out / "agriculture", severe_below=args.severe_below,
                moderate_below=args.moderate_below, validation_scope=args.stress_scope,
            )
            report["stress_zones"] = json.loads(Path(stress.summary_path).read_text(encoding="utf-8"))
        if args.cover_scope:
            from core.india_agriculture import (
                analyse_canopy_cover, classify_index_cover, count_plants_from_canopy)

            ndvi = report["vegetation_indices"]["indices"]["NDVI"]["path"]
            cover = classify_index_cover(ndvi, out / "agriculture" / "cover", validation_scope=args.cover_scope)
            classes, schema = cover.artifact_paths
            report["index_cover"] = json.loads(Path(cover.summary_path).read_text(encoding="utf-8"))
            report["index_cover"]["visible_band_check"] = ms.visible_band_cover_check(ortho, classes)
            canopy = analyse_canopy_cover(classes, schema, out / "agriculture" / "cover")
            report["canopy_cover"] = json.loads(Path(canopy.summary_path).read_text(encoding="utf-8"))
            if args.max_plant_area is not None:
                plants = count_plants_from_canopy(
                    classes, schema, out / "agriculture" / "plants",
                    validation_scope=args.cover_scope, max_plant_area_m2=args.max_plant_area)
                report["plant_count"] = json.loads(Path(plants.summary_path).read_text(encoding="utf-8"))
    report["elapsed_s"] = round(time.time() - started, 1)
    path = out / "multispectral_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
