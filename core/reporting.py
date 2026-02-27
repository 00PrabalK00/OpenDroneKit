"""Report generation utilities for structural inspection runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def write_json(path: str | Path, payload: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(target)


def write_markdown_report(path: str | Path, payload: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    run_time = payload.get("run_utc", datetime.now(timezone.utc).isoformat())
    images = payload.get("images", [])
    multisensor = payload.get("multisensor", {})
    defect_models = payload.get("defect_models", {})
    coverage = payload.get("coverage_validation", {})
    forecast = payload.get("forecast", {})
    recon = payload.get("reconstruction", {})
    health = payload.get("health_scoring", {})
    artifacts = payload.get("artifacts", {})

    lines: list[str] = []
    lines.append("# Structural Inspection Report")
    lines.append("")
    lines.append(f"- Run UTC: `{run_time}`")
    lines.append(f"- Input image directory: `{payload.get('input_image_dir', '')}`")
    lines.append(f"- Images processed: `{len(images)}`")
    lines.append("")
    lines.append("## Structural Health Scoring")
    lines.append("")
    if isinstance(health, dict) and health:
        lines.append(f"- Asset ID: `{health.get('asset_id', '[unknown]')}`")
        lines.append(f"- Structure type: `{health.get('structure_type', 'generic')}`")
        lines.append(f"- Material type: `{health.get('material_type', 'concrete')}`")
        lines.append(f"- Integrity score: `{float(health.get('integrity_score', 0.0)):.2f}/100`")
        lines.append(f"- Severity index: `{float(health.get('severity_index', 0.0)):.2f}/100`")
        lines.append(f"- Structural risk score: `{float(health.get('structural_risk_score', 0.0)):.2f}/100`")
        lines.append(
            f"- Condition grade: `{health.get('condition_grade', '')}` ({health.get('condition_label', '')})"
        )
        failure = health.get("probabilistic_failure", {})
        if isinstance(failure, dict):
            lines.append(f"- Failure probability (6 months): `{100.0 * float(failure.get('probability_6_month', 0.0)):.2f}%`")
            lines.append(f"- Failure probability (1 year): `{100.0 * float(failure.get('probability_1_year', 0.0)):.2f}%`")
            lines.append(f"- Failure probability (3 years): `{100.0 * float(failure.get('probability_3_year', 0.0)):.2f}%`")
        trend = health.get("trend", {})
        if isinstance(trend, dict):
            lines.append(f"- Trend direction: `{trend.get('trend_direction', 'stable')}`")
            lines.append(f"- Trend summary: `{trend.get('summary', '')}`")
            lines.append(f"- Integrity change / month: `{float(trend.get('integrity_change_per_month', 0.0)):.3f}`")
            lines.append(f"- Projected integrity (6 months): `{float(trend.get('projected_integrity_6m', 0.0)):.2f}`")
            lines.append(f"- Projected integrity (12 months): `{float(trend.get('projected_integrity_12m', 0.0)):.2f}`")
    else:
        lines.append("- Structural health scoring not available.")

    lines.append("")
    lines.append("## Crack and Defect Summary")
    lines.append("")
    lines.append("| Image | Crack Ratio | Crack Length (px) | Max Crack Width (px) | Metal Defect Ratio | Defect Regions | Structural Ratio | Solar Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in images:
        lines.append(
            "| {name} | {cr:.4f} | {cl:.1f} | {cw:.1f} | {mr:.4f} | {dr} | {sr:.4f} | {solr:.4f} |".format(
                name=item.get("image", ""),
                cr=float(item.get("crack_ratio", 0.0)),
                cl=float(item.get("crack_length_px", 0.0)),
                cw=float(item.get("crack_max_width_px", 0.0)),
                mr=float(item.get("metal_ratio", 0.0)),
                dr=int(item.get("metal_regions", 0)),
                sr=float(item.get("structural_ratio", 0.0)),
                solr=float(item.get("solar_ratio", 0.0)),
            )
        )
    lines.append("")
    lines.append("## Model-Assisted Defect Analysis")
    lines.append("")
    if defect_models:
        struct = defect_models.get("structural_model", {})
        solar = defect_models.get("solar_model", {})
        migration = defect_models.get("migration", {})
        lines.append(f"- Structural model key: `{struct.get('key', '')}`")
        lines.append(f"- Structural model available: `{bool(struct.get('exists', False))}`")
        lines.append(f"- Structural model path: `{struct.get('path', '')}`")
        lines.append(f"- Solar model key: `{solar.get('key', '')}`")
        lines.append(f"- Solar model available: `{bool(solar.get('exists', False))}`")
        lines.append(f"- Solar model path: `{solar.get('path', '')}`")
        lines.append(f"- Legacy model migration attempted: `{bool(migration.get('migrated', False))}`")
        lines.append(f"- Legacy migration target: `{migration.get('target', '')}`")
        struct_avg = defect_models.get("structural_class_ratio_avg", {})
        if isinstance(struct_avg, dict) and struct_avg:
            joined = ", ".join(f"{k}:{float(v):.4f}" for k, v in sorted(struct_avg.items(), key=lambda x: x[0]))
            lines.append(f"- Structural class ratio avg: `{joined}`")
        solar_avg = defect_models.get("solar_class_ratio_avg", {})
        if isinstance(solar_avg, dict) and solar_avg:
            joined = ", ".join(f"{k}:{float(v):.4f}" for k, v in sorted(solar_avg.items(), key=lambda x: x[0]))
            lines.append(f"- Solar class ratio avg: `{joined}`")
    else:
        lines.append("- Model-assisted defect summary not available.")

    lines.append("")
    lines.append("## Multi-Sensor Capture")
    lines.append("")
    if multisensor:
        lines.append(f"- Bundle: `{multisensor.get('bundle_manifest_path', '')}`")
        lines.append(f"- Sensors: `{', '.join(multisensor.get('sensors', []))}`")
        qa = multisensor.get("qa", {})
        if isinstance(qa, dict):
            lines.append(f"- QA valid: `{bool(qa.get('valid', False))}`")
            lines.append(f"- QA errors: `{len(qa.get('errors', []))}`")
            lines.append(f"- QA warnings: `{len(qa.get('warnings', []))}`")
        align = multisensor.get("alignment", {})
        if isinstance(align, dict) and align:
            lines.append(f"- Aligned frames: `{int(align.get('frame_count', 0))}`")
            lines.append(f"- Thermal match: `{float(align.get('thermal_match_pct', 0.0)):.1f}%`")
            lines.append(f"- LiDAR match: `{float(align.get('lidar_match_pct', 0.0)):.1f}%`")
            lines.append(f"- GNSS match: `{float(align.get('gnss_match_pct', 0.0)):.1f}%`")
        thermal = multisensor.get("thermal", {})
        if isinstance(thermal, dict) and thermal:
            lines.append(f"- Thermal hotspots frames: `{int(thermal.get('frames_with_hotspots', 0))}`")
            if thermal.get("global_max_temp_c") is not None:
                lines.append(f"- Thermal max temp: `{float(thermal.get('global_max_temp_c', 0.0)):.2f} C`")
        lidar = multisensor.get("lidar", {})
        if isinstance(lidar, dict) and lidar:
            lines.append(f"- LiDAR packets valid: `{int(lidar.get('valid_packet_count', 0))}/{int(lidar.get('packet_count', 0))}`")
            lines.append(f"- LiDAR points (raw/preview): `{int(lidar.get('total_points', 0))}/{int(lidar.get('preview_points', 0))}`")
            if lidar.get("preview_cloud_path"):
                lines.append(f"- LiDAR preview cloud: `{lidar.get('preview_cloud_path')}`")
        multispectral = multisensor.get("multispectral", {})
        if isinstance(multispectral, dict) and multispectral:
            lines.append(f"- NDVI frames: `{int(multispectral.get('frame_count', 0))}`")
            if multispectral.get("ndvi_mean_global") is not None:
                lines.append(f"- NDVI global mean: `{float(multispectral.get('ndvi_mean_global', 0.0)):.4f}`")
        if multisensor.get("summary_path"):
            lines.append(f"- Multi-sensor summary JSON: `{multisensor.get('summary_path')}`")
    else:
        lines.append("- Multi-sensor processing not run.")

    lines.append("")
    lines.append("## On-Site Coverage Validation")
    lines.append("")
    if coverage:
        lines.append(f"- Quality failures: `{int(coverage.get('quality_fail_count', 0))}`")
        lines.append(f"- Coverage completion: `{float(coverage.get('coverage_completion_pct', 0.0)):.1f}%`")
        lines.append(f"- Gap suggestions: `{int(coverage.get('gap_suggestion_count', 0))}`")
        if coverage.get("coverage_heatmap_path"):
            lines.append(f"- Coverage heatmap: `{coverage.get('coverage_heatmap_path')}`")
        if coverage.get("preview_model_path"):
            lines.append(f"- Preview mosaic: `{coverage.get('preview_model_path')}`")
        gap = coverage.get("gap_mission", {})
        if isinstance(gap, dict) and gap:
            lines.append("- Gap mission:")
            if gap.get("recipe_path"):
                lines.append(f"  - Flight recipe: `{gap.get('recipe_path')}`")
            if gap.get("qgc_wpl_path"):
                lines.append(f"  - QGC mission: `{gap.get('qgc_wpl_path')}`")
            if gap.get("geojson_path"):
                lines.append(f"  - GeoJSON: `{gap.get('geojson_path')}`")
            if gap.get("waypoint_count") is not None:
                lines.append(f"  - Waypoints: `{int(gap.get('waypoint_count', 0))}`")
    else:
        lines.append("- Coverage validation was not run.")

    lines.append("")
    lines.append("## Crack Propagation Forecast")
    lines.append("")
    if forecast.get("target_image"):
        lines.append(f"- Target image: `{forecast.get('target_image')}`")
        lines.append(f"- Mode: `{forecast.get('mode', 'geometric')}`")
        lines.append(f"- Steps: `{forecast.get('steps', 0)}`")
        lines.append(f"- Output dir: `{forecast.get('output_dir', '')}`")
        physics_cfg = forecast.get("physics_config", {})
        if isinstance(physics_cfg, dict) and physics_cfg:
            lines.append("- Physics configuration:")
            lines.append(f"  - Pixel size: `{float(physics_cfg.get('pixel_size_mm', 0.0)):.4f} mm/px`")
            lines.append(f"  - Sigma nominal: `{float(physics_cfg.get('sigma_nominal_mpa', 0.0)):.3f} MPa`")
            lines.append(f"  - Delta sigma: `{float(physics_cfg.get('delta_sigma_mpa', 0.0)):.3f} MPa`")
            lines.append(f"  - Cycles/year: `{float(physics_cfg.get('cycles_per_year', 0.0)):.0f}`")
            lines.append(f"  - Horizon: `{float(physics_cfg.get('horizon_years', 0.0)):.3f} years`")
            lines.append(f"  - Fracture toughness KIC: `{float(physics_cfg.get('fracture_toughness_mpa_sqrt_m', 0.0)):.3f}`")
            lines.append(f"  - Paris C,m: `{float(physics_cfg.get('paris_c', 0.0)):.3e}`, `{float(physics_cfg.get('paris_m', 0.0)):.3f}`")
        physics_summary = forecast.get("physics_summary", {})
        if isinstance(physics_summary, dict) and physics_summary:
            lines.append("- Physics summary (final step):")
            lines.append(f"  - Crack length: `{float(physics_summary.get('final_crack_length_m', 0.0)):.6f} m`")
            lines.append(
                f"  - K_I / DeltaK: `{float(physics_summary.get('final_stress_intensity_mpa_sqrt_m', 0.0)):.3f}` / "
                f"`{float(physics_summary.get('final_delta_k_mpa_sqrt_m', 0.0)):.3f}`"
            )
            lines.append(
                f"  - da/dN: `{float(physics_summary.get('final_fatigue_da_dn_m_per_cycle', 0.0)):.3e} m/cycle`"
            )
            lines.append(f"  - Factor of safety: `{float(physics_summary.get('final_factor_of_safety', 0.0)):.3f}`")
            lines.append(
                f"  - Failure probability (horizon): "
                f"`{100.0 * float(physics_summary.get('final_failure_probability_horizon', 0.0)):.2f}%`"
            )
        if forecast.get("fenicsx_message"):
            lines.append(f"- FEniCSx: `{forecast.get('fenicsx_message')}`")
    else:
        lines.append("- No crack region detected for forecasting.")

    lines.append("")
    lines.append("## 3D Reconstruction")
    lines.append("")
    lines.append(f"- Frames: `{recon.get('frame_count', 0)}`")
    lines.append(f"- Processed pairs: `{recon.get('processed_pairs', 0)}`")
    lines.append(f"- Failed pairs: `{recon.get('failed_pairs', 0)}`")
    lines.append(f"- Reconstructed points: `{recon.get('total_points', 0)}`")
    lines.append(f"- Crack-localized 3D points: `{recon.get('crack_points', 0)}`")
    if recon.get("processing_profile"):
        lines.append(f"- Processing profile: `{recon.get('processing_profile')}`")
    if recon.get("execution_mode_used"):
        req_mode = recon.get("execution_mode_requested", "local")
        used_mode = recon.get("execution_mode_used", "local")
        lines.append(f"- Execution mode: `{req_mode} -> {used_mode}`")
    if recon.get("cache_hits") is not None and recon.get("cache_misses") is not None:
        lines.append(f"- Cache hits/misses: `{int(recon.get('cache_hits', 0))}/{int(recon.get('cache_misses', 0))}`")
    lines.append(f"- Point cloud: `{recon.get('point_cloud_path', '')}`")
    lines.append(f"- Camera poses: `{recon.get('camera_pose_path', '')}`")
    if recon.get("orthomosaic_path"):
        lines.append(f"- Orthomosaic: `{recon.get('orthomosaic_path')}`")
    if recon.get("dsm_path"):
        lines.append(f"- DSM: `{recon.get('dsm_path')}`")
    if recon.get("dtm_path"):
        lines.append(f"- DTM: `{recon.get('dtm_path')}`")
    if recon.get("mesh_path"):
        lines.append(f"- Mesh OBJ: `{recon.get('mesh_path')}`")
    if recon.get("textured_mesh_obj_path"):
        lines.append(f"- Textured mesh OBJ: `{recon.get('textured_mesh_obj_path')}`")
    if recon.get("texture_image_path"):
        lines.append(f"- Texture image: `{recon.get('texture_image_path')}`")
    if recon.get("digital_twin_path"):
        lines.append(f"- Digital twin manifest: `{recon.get('digital_twin_path')}`")
    if recon.get("cloud_request_path"):
        lines.append(f"- Cloud request: `{recon.get('cloud_request_path')}`")
    if recon.get("cloud_response_path"):
        lines.append(f"- Cloud response: `{recon.get('cloud_response_path')}`")
    warnings = recon.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append(f"- Reconstruction warnings: `{'; '.join(str(w) for w in warnings)}`")
    if recon.get("crack_cloud_path"):
        lines.append(f"- Crack point cloud: `{recon.get('crack_cloud_path')}`")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")

    with target.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return str(target)
