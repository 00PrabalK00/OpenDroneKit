"""Run the MicaSense multispectral chain on real captures and write a measured report.

    python tools/run_multispectral.py --images <band tiff folder> --out <dir>
        [--panel <panel capture folder>] [--calibration dls|panel] [--no-reconstruct]
        [--stress-scope "..." --severe-below 0.3 --moderate-below 0.5]

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
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"images": args.images, "calibration": args.calibration}
    started = time.time()

    panel_capture = None
    if args.panel:
        panel_captures = [c for c in ms.group_captures(args.panel) if any(b.panel_albedo for b in c.bands.values())]
        if panel_captures:
            panel_capture = panel_captures[0]
            report["panel_validation"] = ms.validate_panel_against_albedo(panel_capture)

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}", flush=True)

    summary_path = out / "multispectral_summary.json"
    if args.reuse and summary_path.exists():
        survey = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = {Path(c["sfm_image"]).name: c["stack"] for c in survey["captures"]}
        survey["orthomosaic"] = ms.reconstruct_band_orthomosaic(out, survey["bands"], sources, progress=progress)
        summary_path.write_text(json.dumps(survey, indent=2, default=str), encoding="utf-8")
    else:
        survey = ms.process_survey(
            args.images, out, calibration=args.calibration, panel_capture=panel_capture,
            reconstruct=not args.no_reconstruct, progress=progress,
        )
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
    report["elapsed_s"] = round(time.time() - started, 1)
    path = out / "multispectral_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
