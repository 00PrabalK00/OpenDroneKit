"""Report generation utilities for structural inspection runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _num(item: dict[str, Any], key: str, digits: int, *, ran: bool = True,
         fallback_key: str = "") -> str:
    """A row's figure for the report table, or '-' when nothing measured it."""
    value = item.get(key, item.get(fallback_key) if fallback_key else None)
    if not ran or value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def refusal_reasons(rows: list, detector: str) -> list[dict]:
    """Every distinct reason `detector` refused, with how many images each covers.

    A mixed folder (RGB photos plus undeclared grey images) refuses for different
    reasons; naming only the first would give the wrong reason for the rest.
    """
    counts: dict[str, int] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get(f"{detector}_model_used") == "refused":
            reason = str(row.get(f"{detector}_refused_reason", "") or "no reason recorded")
            counts[reason] = counts.get(reason, 0) + 1
    return [{"reason": r, "images": n} for r, n in counts.items()]


def solar_models_used(rows: list) -> list[dict]:
    """The solar models that actually read images in this run, one entry per model.

    The solar model is chosen per image from the declared imagery, so no single
    configured key describes the run.
    """
    used: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("solar_model_used", ""))
        if not model.startswith("onnx:"):
            continue
        entry = used.setdefault(model, {
            "model_used": model,
            "key": str(row.get("solar_model_key", "")),
            "sha256": str(row.get("solar_model_sha256", "")),
            "imagery": set(),
            "images": 0,
        })
        entry["images"] += 1
        if row.get("solar_imagery"):
            entry["imagery"].add(str(row["solar_imagery"]))
    return [{**e, "imagery": sorted(e["imagery"])} for e in used.values()]


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
    # The heading names the evidence the section is built from: "(heuristic)" over a
    # basis line naming a trained model contradicted itself.
    evidence = ((health.get("basis") or {}).get("defect_evidence")
                if isinstance(health, dict) else None)
    components = ((health.get("components") or {}) if isinstance(health, dict) else {})
    # No crack model read any image (a solar asset's EL cells): the index then rests on
    # the solar and metal components, which count only trained-model output.
    model_components = any(components.get(k) is not None
                           for k in ("solar_component", "metal_component"))
    trained = evidence == "trained_model" or (evidence == "not_assessed" and model_components)
    lines.append("## Structural Health Screening ("
                 + ("trained model, uncalibrated" if trained else "heuristic") + ")")
    lines.append("")
    if isinstance(health, dict) and health:
        # The score is built from the crack and metal pixel ratios of classical image
        # processing plus a geometric growth forecast of that same heuristic mask. It
        # ranks runs for a person to look at; it measures nothing about the structure.
        # A condition grade and a failure probability derived from it would present
        # screening output as a structural assessment, so they are not printed.
        basis = health.get("basis") or {}
        if basis.get("defect_evidence") == "not_assessed":
            lines.append(
                "- Basis: no crack or structural model read these images "
                f"(`{', '.join(basis.get('crack_sources') or [])}`), so those components are "
                "not assessed; solar and metal count only where a trained model produced "
                "them. This is an uncalibrated screening index for choosing what to "
                "review, not a structural assessment."
            )
        elif basis.get("defect_evidence") == "trained_model":
            lines.append(
                "- Basis: crack pixel ratios from a trained segmentation model "
                f"(`{', '.join(basis.get('crack_sources') or [])}`), with metal and solar "
                "counted only where a trained model produced them. This is an "
                "uncalibrated screening index for choosing what to review, not a "
                "structural assessment."
            )
        else:
            lines.append(
                "- Basis: crack and metal pixel ratios from classical image processing "
                "(heuristic). This is an uncalibrated screening index for choosing what "
                "to review, not a structural assessment."
            )
        lines.append(f"- Asset ID: `{health.get('asset_id', '[unknown]')}`")
        lines.append(f"- Structure type (as configured): `{health.get('structure_type', 'generic')}`")
        lines.append(f"- Material type (as configured): `{health.get('material_type', 'concrete')}`")
        if health.get("severity_index") is None:
            lines.append(
                "- Screening severity index: not assessed. "
                + str(health.get("not_assessed_reason")
                      or "No defect component was assessed in this run.")
            )
        else:
            lines.append(
                f"- Screening severity index "
                f"({'uncalibrated' if trained else 'heuristic'}): "
                f"`{float(health.get('severity_index', 0.0)):.2f}/100`"
            )
        lines.append(
            "- Condition grade, integrity score and failure probabilities: not reported. "
            "They need defect measurements (a trained model's detections confirmed by an "
            "inspector, or a georeferenced defect layer) and a hazard model calibrated "
            "for this asset; this run has only "
            + ("uncalibrated model pixel ratios, not confirmed measurements."
               if trained else "heuristic ratios.")
        )
        excluded = (health.get("components", {}) or {}).get("excluded_heuristic_images", {}) or {}
        for detector in ("metal", "solar"):
            count = int(excluded.get(detector, 0) or 0)
            if count:
                lines.append(
                    f"- {detector.capitalize()} detection on {count} image(s) came from a "
                    "heuristic, not a trained model, and was left out of these scores: "
                    f"the {detector} component is not scored, which is not the same as "
                    "no defects."
                )
        failure = health.get("probabilistic_failure", {})
        if isinstance(failure, dict) and failure.get("available") is False:
            lines.append(f"- Failure probability: not computed. {failure.get('reason', '')}")
        not_assessed = (health.get("components") or {}).get("not_assessed") or []
        if not_assessed:
            lines.append(f"- Not assessed (excluded from the index): `{', '.join(not_assessed)}`")
        trend = health.get("trend", {})
        if isinstance(trend, dict) and trend.get("trend_computed"):
            lines.append(
                f"- Screening index trend across {int(trend.get('history_points', 0))} runs "
                f"of this asset over {float(trend.get('span_days', 0.0)):.0f} days: "
                f"`{trend.get('trend_direction')}`"
            )
        elif isinstance(trend, dict) and trend:
            # With one run, or runs less than a week apart, there is no slope: the
            # "stable" default used to be printed as if a trend had been observed.
            lines.append(
                "- Trend: not computed. It needs two or more runs of this asset at least "
                f"7 days apart; this asset has {int(trend.get('history_points', 1))} run(s)"
                + (f" spanning {float(trend.get('span_days', 0.0)):.1f} days."
                   if int(trend.get('history_points', 1)) > 1 else ".")
            )
    else:
        lines.append("- Structural health screening not available.")

    lines.append("")
    lines.append("## Crack and Defect Summary")
    lines.append("")
    lines.append("Each detector column names what produced it. `heuristic` means a classical "
                 "image rule, not a trained model, and its regions carry no confidence; "
                 "`not_run` means the detector was not applicable to this asset; `refused` "
                 "means no installed model reads this imagery, so it was not looked at (see "
                 "below), which is not the same as no defects. Mask "
                 "thickness is the width of the crack MASK in pixels, not a measured crack width.")
    lines.append("")
    for detector in ("solar", "metal"):
        for entry in refusal_reasons(images, detector):
            lines.append(f"- {detector.capitalize()} detection refused on "
                         f"{entry['images']} image(s): {entry['reason']}")
    if any(item.get(f"{d}_model_used") == "refused" for item in images for d in ("solar", "metal")):
        lines.append("")
    lines.append("| Image | Crack Ratio | Crack Length (px) | Mask Thickness (px) | Crack Model | Structural Hits | Structural Model | Metal Ratio | Metal | Solar Ratio | Solar | Corrosion Grade |")
    lines.append("|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---|")
    for item in images:
        crack_ran = not str(item.get("crack_model_used", "")).startswith("not_run:") or (
            str(item.get("crack_model_used", "")) == "not_run:crack_presence_gate")
        corrosion = item.get("corrosion_severity") or {}
        grade = corrosion.get("worst_present_grade") or ("refused" if corrosion.get("refused") else "")
        lines.append(
            "| {name} | {cr} | {cl} | {cw} | {cm} | {sh} | {sm} | {mr} | {mm} | {solr} | {som} | {grade} |".format(
                name=item.get("image", ""),
                # '-' where the model did not run: a printed 0 reads as a measured zero.
                cr=_num(item, "crack_ratio", 4, ran=crack_ran),
                cl=_num(item, "crack_length_px", 1, ran=crack_ran),
                cw=_num(item, "crack_mask_thickness_px", 1, ran=crack_ran,
                        fallback_key="crack_max_width_px"),
                cm=item.get("crack_model_used", ""),
                sh=(str(int(item.get("structural_detection_count", 0) or 0))
                    if item.get("structural_run", True) is not False else "-"),
                sm=item.get("structural_model_used", ""),
                mr=f"{float(item['metal_ratio']):.4f}" if "metal_ratio" in item else "-",
                mm=item.get("metal_model_used", ""),
                solr=f"{float(item['solar_ratio']):.4f}" if "solar_ratio" in item else "-",
                som=item.get("solar_model_used", ""),
                grade=grade or "-",
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
        if solar.get("key"):
            lines.append(f"- Solar model key (forced): `{solar.get('key', '')}`")
            lines.append(f"- Solar model available: `{bool(solar.get('exists', False))}`")
            lines.append(f"- Solar model path: `{solar.get('path', '')}`")
        else:
            lines.append("- Solar model: chosen per image from the declared solar imagery")
        # What actually read the images, from the rows -- not the configured key.
        used = defect_models.get("solar_models_used")
        if used is None:
            used = solar_models_used(images)
        for entry in used:
            lines.append(
                f"- Solar model used: `{entry.get('model_used', '')}` (key "
                f"`{entry.get('key', '')}`, sha256 `{str(entry.get('sha256', ''))[:12]}`, "
                f"imagery `{', '.join(entry.get('imagery', []))}`) on "
                f"{entry.get('images', 0)} image(s)")
        if not used and any(item.get("solar_model_used") not in (None, "", "not_run")
                            for item in images):
            lines.append("- Solar model used: none -- every solar pass was refused (see above)")
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
        completion = coverage.get("coverage_completion_pct")
        if completion is None:
            reason = (coverage.get("layout") or {}).get("reason", "")
            lines.append(f"- Coverage completion: not computed. {reason}")
        else:
            lines.append(f"- Coverage completion: `{float(completion):.1f}%` "
                         f"(two or more images: `{float(coverage.get('stereo_coverage_pct') or 0.0):.1f}%`, "
                         f"over the {str(coverage.get('coverage_extent', '')).replace('_', ' ')})")
        lines.append(f"- Gap suggestions: `{int(coverage.get('gap_suggestion_count', 0))}`")
        not_assessed = (coverage.get("layout") or {}).get("not_assessed_note")
        if not_assessed:
            lines.append(f"- Not assessed: {not_assessed}")
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
    lines.append("## Crack Propagation Forecast (illustrative)")
    lines.append("")
    if forecast.get("basis") == "what_if_simulation" and forecast.get("target_image"):
        lines.append(f"- **What-if simulation, not a measurement.** {forecast.get('note', '')}")
        assumed = forecast.get("assumed_inputs", {})
        if isinstance(assumed, dict) and assumed:
            lines.append("- Assumed inputs: " + ", ".join(f"{k}={v}" for k, v in assumed.items()))
    if forecast.get("target_image"):
        if forecast.get("disclaimer"):
            lines.append(f"- **{forecast.get('disclaimer')}**")
        lines.append(
            "- Basis: the forecast grows the detected crack mask of one image. It "
            "illustrates how a crack of that shape could extend; it is not a measured "
            "crack or a prediction for this structure."
        )
        lines.append(f"- Target image: `{forecast.get('target_image')}`")
        lines.append(f"- Mode: `{forecast.get('mode', 'geometric')}`")
        lines.append(f"- Steps: `{forecast.get('steps', 0)}`")
        lines.append(f"- Output dir: `{forecast.get('output_dir', '')}`")
        physics_cfg = forecast.get("physics_config", {})
        physics_summary = forecast.get("physics_summary", {})
        if isinstance(physics_cfg, dict) and physics_cfg:
            lines.append("- Physics configuration (assumed inputs, not measured on site):")
            lines.append(f"  - Pixel size: `{float(physics_cfg.get('pixel_size_mm', 0.0)):.4f} mm/px`")
            lines.append(f"  - Sigma nominal: `{float(physics_cfg.get('sigma_nominal_mpa', 0.0)):.3f} MPa`")
            lines.append(f"  - Delta sigma: `{float(physics_cfg.get('delta_sigma_mpa', 0.0)):.3f} MPa`")
            lines.append(f"  - Cycles/year: `{float(physics_cfg.get('cycles_per_year', 0.0)):.0f}`")
            lines.append(f"  - Horizon: `{float(physics_cfg.get('horizon_years', 0.0)):.3f} years`")
            lines.append(f"  - Fracture toughness KIC: `{float(physics_cfg.get('fracture_toughness_mpa_sqrt_m', 0.0)):.3f}`")
            lines.append(f"  - Paris C,m: `{float(physics_cfg.get('paris_c', 0.0)):.3e}`, `{float(physics_cfg.get('paris_m', 0.0)):.3f}`")
            usable = isinstance(physics_summary, dict) and float(
                physics_summary.get("final_crack_length_m", 0.0) or 0.0) > 0.0
            if usable:
                lines.append(
                    "- Fracture-mechanics model output (final step), computed from the "
                    "heuristic mask and the assumed inputs above:"
                )
                lines.append(f"  - Modelled crack length: `{float(physics_summary.get('final_crack_length_m', 0.0)):.6f} m`")
                lines.append(
                    f"  - K_I / DeltaK: `{float(physics_summary.get('final_stress_intensity_mpa_sqrt_m', 0.0)):.3f}` / "
                    f"`{float(physics_summary.get('final_delta_k_mpa_sqrt_m', 0.0)):.3f}`"
                )
                lines.append(
                    f"  - da/dN: `{float(physics_summary.get('final_fatigue_da_dn_m_per_cycle', 0.0)):.3e} m/cycle`"
                )
                fos = float(physics_summary.get("final_factor_of_safety", 0.0) or 0.0)
                lines.append(
                    f"  - Modelled factor of safety: `{fos:.3f}`" if fos > 0.0
                    else "  - Factor of safety: not computed."
                )
            else:
                lines.append(
                    "- Fracture-mechanics output: not computed (the model produced no "
                    "crack length for this mask)."
                )
        else:
            lines.append(
                "- Geometric growth only: no crack length, stress intensity, factor of "
                "safety or failure probability was computed."
            )
        if forecast.get("fenicsx_message"):
            lines.append(f"- FEniCSx: `{forecast.get('fenicsx_message')}`")
    else:
        lines.append("- No crack region detected for forecasting.")

    lines.append("")
    lines.append("## 3D Reconstruction")
    lines.append("")
    if recon.get("status") == "failed":
        lines.append(f"- Reconstruction FAILED: `{recon.get('error', '')}`. Detection results "
                     "above are complete.")
        lines.append("- No 3D output, measurements or volumes were produced from this run.")
    else:
        # A failed run has no counts; zeros printed after "FAILED" read as a result.
        # "Frames: 10" read as ten frames in the model when two were never registered.
        frames = int(recon.get("frame_count", 0) or 0)
        registered = recon.get("registered_images")
        if isinstance(registered, int) and 0 < registered < frames:
            lines.append(f"- Frames: `{frames}` (`{registered}` registered, "
                         f"`{frames - registered}` not in the model)")
        elif isinstance(registered, int) and registered == frames and frames:
            lines.append(f"- Frames: `{frames}` (all registered)")
        else:
            lines.append(f"- Frames: `{frames}`")
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
    if recon.get("point_cloud_path"):
        lines.append(f"- Point cloud: `{recon.get('point_cloud_path')}`")
    if recon.get("camera_pose_path"):
        lines.append(f"- Camera poses: `{recon.get('camera_pose_path')}`")
    if recon.get("orthomosaic_path"):
        lines.append(f"- Orthomosaic: `{recon.get('orthomosaic_path')}`")
    if recon.get("dsm_path"):
        lines.append(f"- DSM: `{recon.get('dsm_path')}`")
    if recon.get("dtm_path"):
        lines.append(f"- DTM: `{recon.get('dtm_path')}`")
    if recon.get("mesh_path"):
        # mesh_path is a PLY from the COLMAP engine and an OBJ from the custom one, so
        # the label names neither format.
        lines.append(f"- Mesh: `{recon.get('mesh_path')}`")
    if recon.get("mesh_obj_path"):
        lines.append(f"- Mesh OBJ (untextured): `{recon.get('mesh_obj_path')}`")
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
    lines.append("## Measurements")
    lines.append("")
    measurements = payload.get("measurements", {}) or {}
    totals = measurements.get("totals", {}) if isinstance(measurements, dict) else {}
    terrain = measurements.get("terrain", {}) if isinstance(measurements, dict) else {}
    if totals.get("defect_count"):
        lines.append(f"- Defects measured: `{totals.get('defect_count')}`")
        lines.append(f"- Total defect area: `{totals.get('total_area_m2')} m2`")
        lines.append(f"- Total defect length: `{totals.get('total_length_m')} m`")
        for name, values in (totals.get("by_type") or {}).items():
            lines.append(
                f"  - {name}: `{values.get('count')}` covering `{values.get('area_m2')} m2`"
            )
    else:
        # Areas in square metres require the georeferenced defect layer, which only
        # the COLMAP engine produces. Say so rather than reporting a bare zero.
        lines.append(
            "- No georeferenced defect layer was available, so no real-world defect "
            "measurements could be taken."
        )
    if terrain.get("covered_area_m2"):
        lines.append(f"- Surveyed area: `{terrain.get('covered_area_m2')} m2`")
        lines.append(
            f"- Elevation range: `{terrain.get('elevation_min_m')} - "
            f"{terrain.get('elevation_max_m')} m` (relief `{terrain.get('relief_m')} m`)"
        )

    basis = measurements.get("surface_basis", {}) if isinstance(measurements, dict) else {}
    # The basis qualifies terrain figures; with none printed (no surface, or a failed
    # reconstruction) it described a surface that does not exist.
    if isinstance(basis, dict) and basis.get("reason") and terrain.get("covered_area_m2"):
        lines.append(
            f"- Surface basis: {basis.get('reason')}. Terrain figures above describe the "
            "measured cells only and are not survey-grade."
        )

    volume = payload.get("volume_estimation", {}) or {}
    if isinstance(volume, dict) and volume.get("refused"):
        lines.append("")
        lines.append("### Volume")
        lines.append("")
        lines.append(f"- {volume.get('reason')}")
    if isinstance(volume, dict) and volume.get("ok"):
        preferred = volume.get("preferred") or {}
        lines.append("")
        lines.append("### Volume")
        lines.append("")
        lines.append(f"- Reference surface: `{preferred.get('reference')}`")
        lines.append(f"- Cut (above reference): `{preferred.get('cut_volume_m3')} m3`")
        lines.append(f"- Fill (below reference): `{preferred.get('fill_volume_m3')} m3`")
        lines.append(f"- Net: `{preferred.get('net_volume_m3')} m3`")
        lines.append(f"- CRS: `EPSG:{volume.get('crs_epsg')}` at `{volume.get('pixel_size_m')} m/px`")
        if volume.get("note"):
            lines.append(f"- Note: {volume.get('note')}")

    lines.append("")
    lines.append("## Defect Risk Ranking")
    lines.append("")
    risk = payload.get("risk_scoring", {}) or {}
    if isinstance(risk, dict) and risk.get("defect_count"):
        lines.append(f"- Risk index: `{risk.get('risk_index')}`")
        lines.append(
            f"- Defect condition score: `{risk.get('defect_condition_score')}` "
            f"(grade `{risk.get('defect_condition_grade')}` - "
            f"{risk.get('defect_condition_label')}); from the ranked defects below, not "
            "the screening index in Structural Health Screening"
        )
        lines.append(f"- Basis: {risk.get('measurement_note', '')}")
        lines.append("")
        lines.append("| # | Defect | Type | Risk | Action |")
        lines.append("|---|---|---|---|---|")
        for item in (risk.get("defects") or [])[:10]:
            lines.append(
                f"| {item.get('rank')} | `{item.get('defect_id')}` | {item.get('defect_type')} "
                f"| {item.get('risk_score')} | {item.get('action')} |"
            )
    else:
        lines.append("- No defect records were available to rank.")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")

    with target.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return str(target)
