"""Run one reconstruction of a survey with a chosen dense engine and record the evidence.

    python tools/dense_benchmark.py IMAGES OUT --engine colmap|openmvs|sparse
        [--profile standard] [--colmap PATH] [--openmvs DIR]

Writes OUT/summary.json next to the products: the capability report the run saw, the
engine that actually ran (GPU or CPU), point counts, the DSM/orthophoto ground sampling
distance, how much of the DSM grid holds a measured height, timings and every warning.
Nothing here decides an outcome; it reports what the pipeline wrote.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _raster_stats(path: str) -> dict:
    if not path or not Path(path).is_file():
        return {}
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return {"path": path}
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        valid = int((~np.ma.getmaskarray(band)).sum())
        return {
            "path": path,
            "width": src.width,
            "height": src.height,
            "pixel_size_m": [abs(src.transform.a), abs(src.transform.e)],
            "crs": str(src.crs),
            "valid_cells": valid,
            "valid_fraction": round(valid / float(src.width * src.height), 4),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images")
    parser.add_argument("out")
    parser.add_argument("--engine", choices=("colmap", "openmvs", "sparse"), default="colmap")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--colmap", default="")
    parser.add_argument("--openmvs", default="")
    args = parser.parse_args()
    if args.colmap:
        os.environ["ODK_COLMAP"] = args.colmap
    if args.openmvs:
        os.environ["ODK_OPENMVS_DIR"] = args.openmvs

    from core.reconstruction_colmap import ColmapReconstructor, engine_capabilities

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = engine_capabilities()
    started = time.time()
    stages: list[list] = []

    def progress(percent: int, message: str) -> None:
        stages.append([round(time.time() - started, 1), percent, message])
        print(f"[{time.time() - started:8.1f}s] {percent:3d}% {message}", flush=True)

    engine = ColmapReconstructor(
        profile=args.profile,
        dense=args.engine != "sparse",
        dense_engine=None if args.engine == "sparse" else args.engine,
        use_gpu=True,
    )
    result = engine.reconstruct(image_dir=args.images, output_dir=out, progress_callback=progress)
    elapsed = round(time.time() - started, 1)

    summary = {
        "requested_engine": args.engine,
        "profile": args.profile,
        "elapsed_s": elapsed,
        "capabilities": {k: v for k, v in caps.items()
                         if "dense" in k or "colmap" in k or "openmvs" in k},
        "dense_report": engine.dense_report,
        "registered_images": result.registered_images,
        "frame_count": result.frame_count,
        "total_points": result.total_points,
        "dense_point_count": result.dense_point_count,
        "products_from_dense_cloud": bool(engine._cloud_is_dense),
        "ground_sample_distance_m": result.ground_sample_distance_m,
        "orthomosaic_gsd_m": result.orthomosaic_gsd_m,
        "crs_epsg": result.crs_epsg,
        "geo_rmse_m": result.geo_rmse_m,
        "reprojection_error_px": result.reprojection_error_px,
        "dsm": _raster_stats(result.dsm_cog_path),
        "orthomosaic": _raster_stats(result.orthomosaic_cog_path),
        "textured_mesh_obj": result.textured_mesh_obj_path,
        "texture_image": result.texture_image_path,
        "warnings": result.warnings,
        "stages": stages,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "requested_engine", "elapsed_s", "registered_images", "dense_point_count",
        "ground_sample_distance_m", "orthomosaic_gsd_m", "products_from_dense_cloud")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
