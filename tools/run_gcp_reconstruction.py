"""Reconstruct a real survey with its ground control and print the measured figures.

Used to produce the evidence for the GCP and oblique reconstruction rows in
docs/features/registry.py on the open ODM datasets (bellus, sheffield_cross) and the
NVS facade set. Every number it prints is read from the COLMAP model and the reports
the engine writes; nothing is estimated here.

    python tools/run_gcp_reconstruction.py --images <dir> [--gcp <file>] --out <dir>
        [--profile standard] [--max-image-size 2000] [--subset 0054-0120]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--gcp", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--glob", default="*", help="restrict to images matching this glob")
    args = parser.parse_args(argv)

    from core.gcp import read_gcp_file
    from core.reconstruction_colmap import ColmapReconstructor

    images = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.glob != "*":
        # A subset is linked into its own folder so COLMAP sees only those frames.
        subset = out / "images"
        subset.mkdir(exist_ok=True)
        for frame in sorted(images.glob(args.glob)):
            target = subset / frame.name
            if not target.exists():
                try:
                    target.hardlink_to(frame)
                except OSError:
                    shutil.copy2(frame, target)
        images = subset

    control = read_gcp_file(args.gcp) if args.gcp else []
    engine = ColmapReconstructor(args.profile, use_gpu=not args.no_gpu, dense=False,
                                 max_image_size=args.max_image_size, ground_control=control)
    started = time.time()
    result = engine.reconstruct(
        str(images), output_dir=str(out),
        progress_callback=lambda p, m: print(f"[{p:3d}%] {m}", flush=True))
    summary = {
        "images": result.frame_count,
        "registered": result.registered_images,
        "mean_reprojection_error_px": result.reprojection_error_px,
        "sparse_points": result.total_points,
        "crs_epsg": result.crs_epsg,
        "geo_rmse_m": result.geo_rmse_m,
        "georeferenced_by": json.loads(Path(result.geo_anchor_path).read_text())["method"]
        if result.geo_anchor_path else None,
        "control_report": engine.control_report,
        "warnings": result.warnings,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
