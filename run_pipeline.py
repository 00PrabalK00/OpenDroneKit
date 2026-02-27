"""CLI entrypoint for the finalized structural inspection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_toolkit.core.multisensor import build_capture_bundle_from_folders
from final_toolkit.core.pipeline import PipelineConfig, run_pipeline
from final_toolkit.core.reconstruction import available_reconstruction_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline structural inspection: detection + crack forecast + custom 3D reconstruction + report."
    )
    parser.add_argument("--images", required=True, help="Input image folder (drone frames).")
    parser.add_argument("--output", default="final_toolkit_outputs", help="Output root directory.")
    parser.add_argument("--forecast-steps", type=int, default=6, help="Number of forecast steps.")
    parser.add_argument("--forecast-growth", type=float, default=4.0, help="Growth in pixels per step.")
    parser.add_argument(
        "--propagation-mode",
        choices=["geometric", "physics_informed"],
        default="geometric",
        help="Crack propagation mode.",
    )
    parser.add_argument(
        "--prop-pixel-size-mm",
        type=float,
        default=0.5,
        help="Approximate physical pixel size used for physics-informed propagation.",
    )
    parser.add_argument(
        "--prop-sigma-mpa",
        type=float,
        default=35.0,
        help="Nominal stress (MPa) for physics-informed propagation.",
    )
    parser.add_argument(
        "--prop-delta-sigma-mpa",
        type=float,
        default=12.0,
        help="Stress range (MPa) for fatigue loading (DeltaSigma).",
    )
    parser.add_argument(
        "--prop-cycles-per-year",
        type=float,
        default=1_000_000.0,
        help="Estimated load cycles per year for fatigue growth.",
    )
    parser.add_argument(
        "--prop-horizon-years",
        type=float,
        default=1.0,
        help="Forecast horizon in years for distributing fatigue cycles.",
    )
    parser.add_argument(
        "--prop-cycles-per-step",
        type=float,
        default=0.0,
        help="Optional explicit cycles per forecast step; 0 means auto from yearly cycles/horizon.",
    )
    parser.add_argument(
        "--prop-fracture-toughness",
        type=float,
        default=0.0,
        help="Optional fracture toughness KIC (MPa*sqrt(m)); 0 uses material defaults.",
    )
    parser.add_argument(
        "--prop-paris-c",
        type=float,
        default=0.0,
        help="Optional Paris-law C; 0 uses material defaults.",
    )
    parser.add_argument(
        "--prop-paris-m",
        type=float,
        default=0.0,
        help="Optional Paris-law m; 0 uses material defaults.",
    )
    parser.add_argument("--use-fenicsx", action="store_true", help="Enable optional FEniCSx bridge via cracksim.py.")
    parser.add_argument("--recon-max-points", type=int, default=150000, help="Max points kept in final cloud.")
    parser.add_argument(
        "--recon-profile",
        choices=available_reconstruction_profiles(),
        default="standard",
        help="Reconstruction profile: fast preview vs inspection-grade.",
    )
    parser.add_argument(
        "--recon-mode",
        choices=["local", "cloud"],
        default="local",
        help="Reconstruction execution mode.",
    )
    parser.add_argument(
        "--recon-no-cache",
        action="store_true",
        help="Disable reconstruction feature/match cache.",
    )
    parser.add_argument(
        "--recon-cloud-endpoint",
        default="",
        help="Cloud endpoint URL for reconstruction job submission in cloud mode.",
    )
    parser.add_argument(
        "--coverage-recipe",
        default="",
        help="Optional baseline flight recipe JSON used to generate geo-anchored gap missions.",
    )
    parser.add_argument(
        "--skip-coverage-validation",
        action="store_true",
        help="Skip fast on-site quality/coverage validation stage.",
    )
    parser.add_argument(
        "--capture-bundle",
        default="",
        help="Optional multi-sensor capture bundle JSON manifest (RGB/thermal/LiDAR/IMU/GNSS).",
    )
    parser.add_argument(
        "--bundle-root",
        default="",
        help="Optional root folder containing rgb/thermal/multispectral/lidar subfolders to auto-build a capture bundle.",
    )
    parser.add_argument(
        "--bundle-payload-id",
        default="payload",
        help="Payload id to use when auto-building a capture bundle from --bundle-root.",
    )
    parser.add_argument(
        "--calibration-profile",
        default="",
        help="Optional calibration profile JSON for the selected capture bundle.",
    )
    parser.add_argument(
        "--skip-multisensor",
        action="store_true",
        help="Skip multi-sensor alignment and QA even if --capture-bundle is provided.",
    )
    parser.add_argument(
        "--thermal-hotspot-threshold",
        type=float,
        default=65.0,
        help="Hotspot detection threshold in degC for thermal workflow.",
    )
    parser.add_argument(
        "--thermal-min-hotspot-area",
        type=int,
        default=60,
        help="Minimum connected hotspot area in pixels.",
    )
    parser.add_argument(
        "--lidar-voxel-size",
        type=float,
        default=0.08,
        help="Voxel size (m) for LiDAR preview downsampling.",
    )
    parser.add_argument(
        "--disable-structural-models",
        action="store_true",
        help="Disable model-assisted multi-class structural defect detection.",
    )
    parser.add_argument(
        "--disable-solar-models",
        action="store_true",
        help="Disable model-assisted solar defect detection.",
    )
    parser.add_argument(
        "--structural-model-key",
        default="structural_multiclass_detector",
        help="Model registry key for structural defect detector.",
    )
    parser.add_argument(
        "--solar-model-key",
        default="solar_defect_detector",
        help="Model registry key for solar defect detector.",
    )
    parser.add_argument(
        "--asset-id",
        default="",
        help="Asset identifier for temporal health tracking (defaults to dataset folder name).",
    )
    parser.add_argument(
        "--structure-type",
        default="generic",
        help="Structure category for risk modeling (e.g., bridge/building/tower/solar/pipeline/generic).",
    )
    parser.add_argument(
        "--material-type",
        default="concrete",
        help="Material category for risk modeling (e.g., concrete/steel/masonry/composite/mixed).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    capture_bundle = str(args.capture_bundle or "").strip()
    bundle_root = str(args.bundle_root or "").strip()
    if not capture_bundle and bundle_root:
        output_root = Path(args.output).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        auto_bundle = output_root / "auto_capture_bundle.json"
        build_capture_bundle_from_folders(
            root_dir=bundle_root,
            output_manifest_path=auto_bundle,
            payload_id=str(args.bundle_payload_id or "payload"),
        )
        capture_bundle = str(auto_bundle)

    config = PipelineConfig(
        forecast_steps=max(1, args.forecast_steps),
        forecast_growth_px_per_step=max(0.0, args.forecast_growth),
        propagation_mode=str(args.propagation_mode or "geometric"),
        propagation_pixel_size_mm=float(max(0.01, args.prop_pixel_size_mm)),
        propagation_sigma_nominal_mpa=float(max(0.0, args.prop_sigma_mpa)),
        propagation_delta_sigma_mpa=float(max(0.0, args.prop_delta_sigma_mpa)),
        propagation_cycles_per_year=float(max(1.0, args.prop_cycles_per_year)),
        propagation_horizon_years=float(max(0.01, args.prop_horizon_years)),
        propagation_cycles_per_step=float(max(0.0, args.prop_cycles_per_step)),
        propagation_fracture_toughness_mpa_sqrt_m=float(max(0.0, args.prop_fracture_toughness)),
        propagation_paris_c=float(max(0.0, args.prop_paris_c)),
        propagation_paris_m=float(max(0.0, args.prop_paris_m)),
        reconstruction_max_points=max(1000, args.recon_max_points),
        use_fenicsx=bool(args.use_fenicsx),
        run_coverage_validation=not bool(args.skip_coverage_validation),
        coverage_reference_recipe_path=str(args.coverage_recipe or ""),
        run_multisensor_processing=not bool(args.skip_multisensor),
        multi_sensor_bundle_path=capture_bundle,
        multi_sensor_calibration_profile_path=str(args.calibration_profile or ""),
        thermal_hotspot_threshold_c=float(args.thermal_hotspot_threshold),
        thermal_min_hotspot_area_px=max(1, int(args.thermal_min_hotspot_area)),
        lidar_voxel_size_m=max(1e-4, float(args.lidar_voxel_size)),
        reconstruction_profile=str(args.recon_profile or "standard"),
        reconstruction_execution_mode=str(args.recon_mode or "local"),
        reconstruction_use_cache=not bool(args.recon_no_cache),
        reconstruction_cloud_endpoint=str(args.recon_cloud_endpoint or ""),
        run_structural_models=not bool(args.disable_structural_models),
        run_solar_models=not bool(args.disable_solar_models),
        structural_model_key=str(args.structural_model_key or "structural_multiclass_detector"),
        solar_model_key=str(args.solar_model_key or "solar_defect_detector"),
        asset_id=str(args.asset_id or ""),
        structure_type=str(args.structure_type or "generic"),
        material_type=str(args.material_type or "concrete"),
    )
    result = run_pipeline(image_dir=args.images, output_dir=args.output, config=config)
    print(json.dumps(result.to_dict(), indent=2))

    report_path = Path(result.report_markdown_path)
    if report_path.exists():
        print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
