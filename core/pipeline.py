"""End-to-end structural inspection pipeline (offline, non-COLMAP mode)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .coverage_validation import CoverageValidationConfig, CoverageValidator
from .detection import (
    detect_cracks,
    detect_metal_defects,
    detect_solar_defects,
    detect_structural_defects,
    load_image,
)
from .health_scoring import derive_asset_id, evaluate_asset_health, load_asset_health_history
from .models import ensure_model_layout, migrate_legacy_models, model_status
from .multisensor import MultiSensorConfig, MultiSensorProcessor
from .propagation import CrackPropagationForecaster, PropagationPhysicsConfig, run_fenicsx_phasefield
from .reconstruction import CustomDroneReconstructor
from .reporting import write_json, write_markdown_report


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineConfig:
    crack_min_area_px: int = 30
    metal_min_area_px: int = 40
    forecast_steps: int = 6
    forecast_growth_px_per_step: float = 4.0
    forecast_lateral_growth: int = 1
    propagation_mode: str = "geometric"
    propagation_pixel_size_mm: float = 0.5
    propagation_sigma_nominal_mpa: float = 35.0
    propagation_delta_sigma_mpa: float = 12.0
    propagation_cycles_per_year: float = 1_000_000.0
    propagation_horizon_years: float = 1.0
    propagation_cycles_per_step: float = 0.0
    propagation_fracture_toughness_mpa_sqrt_m: float = 0.0
    propagation_paris_c: float = 0.0
    propagation_paris_m: float = 0.0
    reconstruction_max_points: int = 150_000
    use_fenicsx: bool = False
    run_coverage_validation: bool = True
    coverage_reference_recipe_path: str = ""
    coverage_sharpness_threshold: float = 120.0
    coverage_exposure_clip_threshold: float = 0.12
    coverage_overlap_threshold: float = 0.14
    run_multisensor_processing: bool = True
    multi_sensor_bundle_path: str = ""
    multi_sensor_calibration_profile_path: str = ""
    thermal_hotspot_threshold_c: float = 65.0
    thermal_min_hotspot_area_px: int = 60
    lidar_voxel_size_m: float = 0.08
    reconstruction_profile: str = "standard"
    reconstruction_execution_mode: str = "local"
    reconstruction_use_cache: bool = True
    reconstruction_cloud_endpoint: str = ""
    run_structural_models: bool = True
    run_solar_models: bool = True
    structural_model_key: str = "structural_multiclass_detector"
    solar_model_key: str = "solar_defect_detector"
    asset_id: str = ""
    structure_type: str = "generic"
    material_type: str = "concrete"


@dataclass
class PipelineResult:
    summary_json_path: str
    report_markdown_path: str
    point_cloud_path: str
    crack_point_cloud_path: str | None
    forecast_dir: str
    coverage_dir: str
    multisensor_dir: str
    run_dir: str
    health_scoring_path: str = ""
    asset_integrity_score: float = 0.0
    structural_risk_score: float = 0.0
    condition_level: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class StructuralFaultPipeline:
    """Coordinator for crack detection, propagation forecast, 3D reconstruction, and reporting."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.forecaster = CrackPropagationForecaster(
            growth_px_per_step=self.config.forecast_growth_px_per_step,
            lateral_growth=self.config.forecast_lateral_growth,
        )
        self.reconstructor = CustomDroneReconstructor(
            max_points=self.config.reconstruction_max_points,
            profile=self.config.reconstruction_profile,
            execution_mode=self.config.reconstruction_execution_mode,
            use_cache=self.config.reconstruction_use_cache,
            cloud_endpoint=self.config.reconstruction_cloud_endpoint,
        )

    @staticmethod
    def _images_in_dir(image_dir: str | Path) -> list[Path]:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        images = [p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        images.sort()
        if not images:
            raise ValueError(f"No supported images found in: {image_dir}")
        return images

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> PipelineResult:
        def _emit_progress(percent: int, message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(int(max(0, min(100, int(percent)))), str(message))
            except Exception:
                pass

        _emit_progress(1, "Preparing model layout")
        ensure_model_layout()
        model_migration = migrate_legacy_models()
        structural_model_info = model_status(self.config.structural_model_key)
        solar_model_info = model_status(self.config.solar_model_key)

        _emit_progress(5, "Loading input dataset")
        image_paths = self._images_in_dir(image_dir)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        run_dt = datetime.now(timezone.utc)
        run_utc = run_dt.isoformat()
        run_stamp = run_dt.strftime("%Y%m%d_%H%M%S")
        asset_id = derive_asset_id(self.config.asset_id, image_dir)
        run_dir = out_root / f"run_{run_stamp}"
        masks_dir = run_dir / "masks"
        overlays_dir = run_dir / "overlays"
        forecast_dir = run_dir / "forecast"
        recon_dir = run_dir / "reconstruction"
        fenicsx_dir = run_dir / "fenicsx"
        coverage_dir = run_dir / "coverage"
        multisensor_dir = run_dir / "multisensor"
        for d in (masks_dir, overlays_dir, forecast_dir, recon_dir, fenicsx_dir, coverage_dir, multisensor_dir):
            d.mkdir(parents=True, exist_ok=True)

        coverage_payload: dict = {}
        quality_by_image: dict[str, dict] = {}
        if self.config.run_coverage_validation:
            _emit_progress(12, "Running coverage validation")
            coverage_cfg = CoverageValidationConfig(
                sharpness_threshold=float(max(1.0, self.config.coverage_sharpness_threshold)),
                exposure_clip_threshold=float(np.clip(self.config.coverage_exposure_clip_threshold, 0.0, 0.95)),
                overlap_threshold=float(np.clip(self.config.coverage_overlap_threshold, 0.0, 1.0)),
            )
            coverage_validator = CoverageValidator(config=coverage_cfg)
            coverage_payload = coverage_validator.validate(
                image_paths=image_paths,
                output_dir=coverage_dir,
                reference_recipe_path=str(self.config.coverage_reference_recipe_path or ""),
            )
            for item in coverage_payload.get("images", []):
                name = str(item.get("image", "")).strip()
                if name:
                    quality_by_image[name] = item
            _emit_progress(24, "Coverage validation complete")
        else:
            _emit_progress(24, "Coverage validation skipped")

        multisensor_payload: dict = {}
        bundle_path = str(self.config.multi_sensor_bundle_path or "").strip()
        if self.config.run_multisensor_processing and bundle_path:
            _emit_progress(25, "Processing multi-sensor bundle")
            ms_cfg = MultiSensorConfig(
                thermal_hotspot_threshold_c=float(max(-50.0, self.config.thermal_hotspot_threshold_c)),
                thermal_min_hotspot_area_px=max(1, int(self.config.thermal_min_hotspot_area_px)),
                lidar_voxel_size_m=float(max(1e-4, self.config.lidar_voxel_size_m)),
            )
            ms_processor = MultiSensorProcessor(config=ms_cfg)
            try:
                multisensor_payload = ms_processor.process(
                    bundle_path=bundle_path,
                    output_dir=multisensor_dir,
                    calibration_profile_path=str(self.config.multi_sensor_calibration_profile_path or ""),
                )
            except Exception as exc:
                multisensor_payload = {
                    "bundle_manifest_path": bundle_path,
                    "qa": {
                        "valid": False,
                        "errors": [f"Multi-sensor processing failed: {exc}"],
                        "warnings": [],
                    },
                    "alignment": {},
                    "thermal": {},
                    "lidar": {},
                    "multispectral": {},
                }
            _emit_progress(35, "Multi-sensor processing complete")
        else:
            _emit_progress(35, "Multi-sensor processing skipped")

        image_rows: list[dict] = []
        crack_candidates: list[tuple[float, Path]] = []

        total_images = max(1, len(image_paths))
        for idx, image_path in enumerate(image_paths):
            image_pct = 36 + int((30.0 * float(idx)) / float(total_images))
            _emit_progress(image_pct, f"Analyzing image {idx + 1}/{len(image_paths)}: {image_path.name}")
            image = load_image(image_path)
            crack = detect_cracks(image, min_area_px=self.config.crack_min_area_px)
            metal = detect_metal_defects(image, min_area_px=self.config.metal_min_area_px)
            structural = detect_structural_defects(
                image,
                min_area_px=max(self.config.crack_min_area_px, self.config.metal_min_area_px),
                model_key=str(self.config.structural_model_key or "structural_multiclass_detector"),
                use_model=bool(self.config.run_structural_models),
            )
            solar = detect_solar_defects(
                image,
                thermal_gray=None,
                min_area_px=max(self.config.crack_min_area_px, self.config.metal_min_area_px),
                model_key=str(self.config.solar_model_key or "solar_defect_detector"),
                use_model=bool(self.config.run_solar_models),
            )

            crack_mask_path = masks_dir / f"{image_path.stem}_crack_mask.png"
            metal_mask_path = masks_dir / f"{image_path.stem}_metal_mask.png"
            structural_mask_path = masks_dir / f"{image_path.stem}_structural_mask.png"
            solar_mask_path = masks_dir / f"{image_path.stem}_solar_mask.png"
            crack_overlay_path = overlays_dir / f"{image_path.stem}_crack_overlay.png"
            metal_overlay_path = overlays_dir / f"{image_path.stem}_metal_overlay.png"
            structural_overlay_path = overlays_dir / f"{image_path.stem}_structural_overlay.png"
            solar_overlay_path = overlays_dir / f"{image_path.stem}_solar_overlay.png"

            cv2.imwrite(str(crack_mask_path), crack.mask)
            cv2.imwrite(str(metal_mask_path), metal.mask)
            cv2.imwrite(str(structural_mask_path), structural.mask)
            cv2.imwrite(str(solar_mask_path), solar.mask)
            cv2.imwrite(str(crack_overlay_path), crack.overlay)
            cv2.imwrite(str(metal_overlay_path), metal.overlay)
            cv2.imwrite(str(structural_overlay_path), structural.overlay)
            cv2.imwrite(str(solar_overlay_path), solar.overlay)

            image_rows.append(
                {
                    "image": image_path.name,
                    "crack_pixels": crack.crack_pixels,
                    "crack_ratio": crack.crack_ratio,
                    "crack_length_px": crack.estimated_length_px,
                    "crack_max_width_px": crack.estimated_max_width_px,
                    "metal_pixels": metal.defect_pixels,
                    "metal_ratio": metal.defect_ratio,
                    "metal_regions": metal.defect_regions,
                    "structural_ratio": float(structural.defect_ratio),
                    "structural_pixels": int(structural.defect_pixels),
                    "structural_detection_count": int(len(structural.detections)),
                    "structural_class_ratio": {k: float(v) for k, v in structural.class_pixel_ratio.items()},
                    "structural_detections": [d.to_dict() for d in structural.detections],
                    "structural_model_used": str(structural.model_used),
                    "structural_model_available": bool(structural.model_available),
                    "solar_ratio": float(solar.defect_ratio),
                    "solar_pixels": int(solar.defect_pixels),
                    "solar_detection_count": int(len(solar.detections)),
                    "solar_class_ratio": {k: float(v) for k, v in solar.class_pixel_ratio.items()},
                    "solar_detections": [d.to_dict() for d in solar.detections],
                    "solar_model_used": str(solar.model_used),
                    "solar_model_available": bool(solar.model_available),
                    "solar_hotspot_max_temp_c": (
                        float(solar.hotspot_max_temp_c) if solar.hotspot_max_temp_c is not None else None
                    ),
                    "crack_mask_path": str(crack_mask_path),
                    "crack_overlay_path": str(crack_overlay_path),
                    "metal_mask_path": str(metal_mask_path),
                    "metal_overlay_path": str(metal_overlay_path),
                    "structural_mask_path": str(structural_mask_path),
                    "structural_overlay_path": str(structural_overlay_path),
                    "solar_mask_path": str(solar_mask_path),
                    "solar_overlay_path": str(solar_overlay_path),
                }
            )
            quality = quality_by_image.get(image_path.name, {})
            if quality:
                image_rows[-1].update(
                    {
                        "sharpness": float(quality.get("sharpness", 0.0)),
                        "motion_blur_score": float(quality.get("motion_blur_score", 0.0)),
                        "lens_blur_ratio": float(quality.get("lens_blur_ratio", 0.0)),
                        "dark_clip_pct": float(quality.get("dark_clip_pct", 0.0)),
                        "bright_clip_pct": float(quality.get("bright_clip_pct", 0.0)),
                        "mean_luma": float(quality.get("mean_luma", 0.0)),
                        "overlap_prev": float(quality.get("overlap_prev", 0.0)),
                        "overlap_next": float(quality.get("overlap_next", 0.0)),
                        "low_overlap": bool(quality.get("low_overlap", False)),
                        "quality_ok": bool(quality.get("quality_ok", True)),
                        "quality_flags": list(quality.get("quality_flags", [])),
                    }
                )

            crack_candidates.append((crack.crack_ratio, crack_mask_path))
        _emit_progress(67, "Image defect analysis complete")

        structural_class_sum: dict[str, float] = {}
        solar_class_sum: dict[str, float] = {}
        for row in image_rows:
            for key, value in (row.get("structural_class_ratio", {}) or {}).items():
                structural_class_sum[key] = float(structural_class_sum.get(key, 0.0)) + float(value)
            for key, value in (row.get("solar_class_ratio", {}) or {}).items():
                solar_class_sum[key] = float(solar_class_sum.get(key, 0.0)) + float(value)
        divisor = float(max(1, len(image_rows)))
        structural_class_avg = {k: float(v / divisor) for k, v in structural_class_sum.items()}
        solar_class_avg = {k: float(v / divisor) for k, v in solar_class_sum.items()}

        forecast_payload: dict = {
            "target_image": None,
            "mode": str(self.config.propagation_mode or "geometric"),
            "steps": 0,
            "output_dir": str(forecast_dir),
            "frames": [],
            "metrics": [],
            "fenicsx_message": "",
            "physics_config": {},
            "physics_summary": {},
        }

        _emit_progress(68, "Running crack propagation forecast")
        propagation_mode = str(self.config.propagation_mode or "geometric").strip().lower()
        if propagation_mode not in {"geometric", "physics_informed", "physics", "fracture_mechanics"}:
            propagation_mode = "geometric"
        if propagation_mode in {"physics", "fracture_mechanics"}:
            propagation_mode = "physics_informed"

        best_ratio, best_mask_path = max(crack_candidates, key=lambda item: item[0])
        if best_ratio > 0.0:
            target_name = best_mask_path.name.replace("_crack_mask.png", "")
            base_mask = cv2.imread(str(best_mask_path), cv2.IMREAD_GRAYSCALE)
            if base_mask is not None:
                physics_cfg: PropagationPhysicsConfig | None = None
                if propagation_mode == "physics_informed":
                    physics_cfg = PropagationPhysicsConfig.from_context(
                        structure_type=str(self.config.structure_type or "generic"),
                        material_type=str(self.config.material_type or "concrete"),
                        pixel_size_mm=float(max(0.01, self.config.propagation_pixel_size_mm)),
                        sigma_nominal_mpa=(
                            None
                            if self.config.propagation_sigma_nominal_mpa <= 0.0
                            else float(self.config.propagation_sigma_nominal_mpa)
                        ),
                        delta_sigma_mpa=(
                            None
                            if self.config.propagation_delta_sigma_mpa <= 0.0
                            else float(self.config.propagation_delta_sigma_mpa)
                        ),
                        cycles_per_year=float(max(1.0, self.config.propagation_cycles_per_year)),
                        horizon_years=float(max(0.01, self.config.propagation_horizon_years)),
                        cycles_per_step=float(max(0.0, self.config.propagation_cycles_per_step)),
                        fracture_toughness_mpa_sqrt_m=(
                            None
                            if self.config.propagation_fracture_toughness_mpa_sqrt_m <= 0.0
                            else float(self.config.propagation_fracture_toughness_mpa_sqrt_m)
                        ),
                        paris_c=None if self.config.propagation_paris_c <= 0.0 else float(self.config.propagation_paris_c),
                        paris_m=None if self.config.propagation_paris_m <= 0.0 else float(self.config.propagation_paris_m),
                    )
                seq, metrics = self.forecaster.forecast(
                    base_mask,
                    steps=self.config.forecast_steps,
                    mode=propagation_mode,
                    physics=physics_cfg,
                )
                frame_paths = self.forecaster.save_sequence(seq, forecast_dir, prefix=f"{target_name}_forecast")
                metrics_payload = [m.to_dict() for m in metrics]
                forecast_payload.update(
                    {
                        "target_image": target_name,
                        "mode": propagation_mode,
                        "steps": self.config.forecast_steps,
                        "frames": frame_paths,
                        "metrics": metrics_payload,
                    }
                )
                if physics_cfg is not None:
                    forecast_payload["physics_config"] = {
                        "pixel_size_mm": float(physics_cfg.pixel_size_mm),
                        "sigma_nominal_mpa": float(physics_cfg.sigma_nominal_mpa),
                        "delta_sigma_mpa": float(physics_cfg.delta_sigma_mpa),
                        "cycles_per_year": float(physics_cfg.cycles_per_year),
                        "horizon_years": float(physics_cfg.horizon_years),
                        "cycles_per_step": float(physics_cfg.cycles_per_step),
                        "fracture_toughness_mpa_sqrt_m": float(physics_cfg.fracture_toughness_mpa_sqrt_m),
                        "paris_c": float(physics_cfg.paris_c),
                        "paris_m": float(physics_cfg.paris_m),
                    }
                if metrics_payload:
                    last = metrics_payload[-1]
                    forecast_payload["physics_summary"] = {
                        "final_crack_length_m": float(last.get("crack_length_m", 0.0) or 0.0),
                        "final_stress_intensity_mpa_sqrt_m": float(last.get("stress_intensity_mpa_sqrt_m", 0.0) or 0.0),
                        "final_delta_k_mpa_sqrt_m": float(last.get("delta_k_mpa_sqrt_m", 0.0) or 0.0),
                        "final_fatigue_da_dn_m_per_cycle": float(last.get("fatigue_da_dn_m_per_cycle", 0.0) or 0.0),
                        "final_factor_of_safety": float(last.get("factor_of_safety", 0.0) or 0.0),
                        "final_failure_probability_horizon": float(last.get("failure_probability_horizon", 0.0) or 0.0),
                    }

                if self.config.use_fenicsx:
                    ok, msg = run_fenicsx_phasefield(
                        mask_path=str(best_mask_path),
                        steps=self.config.forecast_steps,
                        output_dir=str(fenicsx_dir),
                    )
                    forecast_payload["mode"] = f"fenicsx+{propagation_mode}" if ok else f"{propagation_mode}-fenicsx-fallback"
                    forecast_payload["fenicsx_message"] = msg
        else:
            forecast_payload["fenicsx_message"] = "No crack pixels detected, forecasting skipped."
        _emit_progress(74, "Forecast stage complete")

        def _recon_progress(percent: int, message: str) -> None:
            mapped = 74 + int((22.0 * float(max(0, min(100, int(percent))))) / 100.0)
            _emit_progress(mapped, f"Reconstruction: {message}")

        recon = self.reconstructor.reconstruct(
            image_dir=image_dir,
            crack_mask_dir=masks_dir,
            output_dir=recon_dir,
            profile=self.config.reconstruction_profile,
            execution_mode=self.config.reconstruction_execution_mode,
            use_cache=self.config.reconstruction_use_cache,
            cloud_endpoint=self.config.reconstruction_cloud_endpoint,
            progress_callback=_recon_progress,
        )

        _emit_progress(96, "Computing structural health score")
        history_rows = load_asset_health_history(
            output_root=out_root,
            asset_id=asset_id,
            current_run_dir=run_dir,
            max_runs=30,
        )
        health_scoring = evaluate_asset_health(
            asset_id=asset_id,
            run_utc=run_utc,
            structure_type=str(self.config.structure_type or "generic"),
            material_type=str(self.config.material_type or "concrete"),
            image_rows=image_rows,
            forecast_payload=forecast_payload,
            reconstruction_payload=recon.to_dict(),
            coverage_payload=coverage_payload,
            history=history_rows,
        )
        health_scoring_path = write_json(run_dir / "health_scoring.json", health_scoring)

        _emit_progress(97, "Writing summary artifacts")
        payload = {
            "run_utc": run_utc,
            "input_image_dir": str(Path(image_dir).resolve()),
            "config": asdict(self.config),
            "images": image_rows,
            "multisensor": multisensor_payload,
            "defect_models": {
                "migration": model_migration,
                "structural_model": structural_model_info,
                "solar_model": solar_model_info,
                "run_structural_models": bool(self.config.run_structural_models),
                "run_solar_models": bool(self.config.run_solar_models),
                "structural_class_ratio_avg": structural_class_avg,
                "solar_class_ratio_avg": solar_class_avg,
            },
            "coverage_validation": coverage_payload,
            "forecast": forecast_payload,
            "reconstruction": recon.to_dict(),
            "health_scoring": health_scoring,
            "artifacts": {
                "run_dir": str(run_dir),
                "masks_dir": str(masks_dir),
                "overlays_dir": str(overlays_dir),
                "forecast_dir": str(forecast_dir),
                "reconstruction_dir": str(recon_dir),
                "coverage_dir": str(coverage_dir),
                "multisensor_dir": str(multisensor_dir),
                "health_scoring_path": str(health_scoring_path),
            },
        }
        if isinstance(multisensor_payload, dict):
            if multisensor_payload.get("summary_path"):
                payload["artifacts"]["multisensor_summary_path"] = str(multisensor_payload.get("summary_path"))
            ms_artifacts = multisensor_payload.get("artifacts", {})
            if isinstance(ms_artifacts, dict):
                for key, value in ms_artifacts.items():
                    payload["artifacts"][f"multisensor_{key}"] = value
        recon_dict = recon.to_dict()
        for key in (
            "orthomosaic_path",
            "dsm_path",
            "dtm_path",
            "mesh_path",
            "textured_mesh_obj_path",
            "textured_mesh_mtl_path",
            "texture_image_path",
            "digital_twin_path",
            "camera_pose_path",
            "cloud_request_path",
            "cloud_response_path",
        ):
            value = recon_dict.get(key)
            if isinstance(value, str) and value:
                payload["artifacts"][f"reconstruction_{key}"] = value
        gap_mission = coverage_payload.get("gap_mission", {}) if isinstance(coverage_payload, dict) else {}
        if isinstance(gap_mission, dict):
            for key, value in gap_mission.items():
                if key.endswith("_path"):
                    payload["artifacts"][f"coverage_{key}"] = value

        summary_json = write_json(run_dir / "summary.json", payload)
        report_md = write_markdown_report(run_dir / "inspection_report.md", payload)
        _emit_progress(100, "Pipeline complete")

        return PipelineResult(
            summary_json_path=summary_json,
            report_markdown_path=report_md,
            point_cloud_path=recon.point_cloud_path,
            crack_point_cloud_path=recon.crack_cloud_path,
            forecast_dir=str(forecast_dir),
            coverage_dir=str(coverage_dir),
            multisensor_dir=str(multisensor_dir),
            run_dir=str(run_dir),
            health_scoring_path=str(health_scoring_path),
            asset_integrity_score=float(health_scoring.get("integrity_score", 0.0)),
            structural_risk_score=float(health_scoring.get("structural_risk_score", 0.0)),
            condition_level=int(health_scoring.get("condition_level", 0)),
        )


def run_pipeline(
    image_dir: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> PipelineResult:
    pipeline = StructuralFaultPipeline(config=config)
    return pipeline.run(
        image_dir=image_dir,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
