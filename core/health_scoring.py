"""Asset-level structural health scoring and temporal trend analytics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _clip100(value: float) -> float:
    return float(np.clip(float(value), 0.0, 100.0))


def _parse_utc(value: str | None) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def derive_asset_id(config_asset_id: str | None, image_dir: str | Path) -> str:
    explicit = str(config_asset_id or "").strip()
    if explicit:
        return explicit
    name = Path(image_dir).resolve().name.strip()
    return name or "asset_unknown"


def _asset_id_from_summary(summary: dict[str, Any]) -> str:
    health = summary.get("health_scoring", {})
    if isinstance(health, dict):
        aid = str(health.get("asset_id", "")).strip()
        if aid:
            return aid
    cfg = summary.get("config", {})
    if isinstance(cfg, dict):
        aid = str(cfg.get("asset_id", "")).strip()
        if aid:
            return aid
    image_dir = str(summary.get("input_image_dir", "")).strip()
    if image_dir:
        return derive_asset_id("", image_dir)
    return "asset_unknown"


def _aggregate_image_metrics(image_rows: list[dict[str, Any]]) -> dict[str, float]:
    n = float(max(1, len(image_rows)))
    crack_ratio = float(sum(_safe_float(r.get("crack_ratio", 0.0)) for r in image_rows) / n)
    crack_len = float(sum(_safe_float(r.get("crack_length_px", 0.0)) for r in image_rows) / n)
    crack_width = float(sum(_safe_float(r.get("crack_max_width_px", 0.0)) for r in image_rows) / n)
    metal_ratio = float(sum(_safe_float(r.get("metal_ratio", 0.0)) for r in image_rows) / n)
    structural_ratio = float(sum(_safe_float(r.get("structural_ratio", 0.0)) for r in image_rows) / n)
    solar_ratio = float(sum(_safe_float(r.get("solar_ratio", 0.0)) for r in image_rows) / n)
    quality_fail = float(sum(1.0 for r in image_rows if bool(r.get("quality_ok", True)) is False) / n)
    low_overlap = float(sum(1.0 for r in image_rows if bool(r.get("low_overlap", False))) / n)
    return {
        "crack_ratio_avg": crack_ratio,
        "crack_length_avg_px": crack_len,
        "crack_max_width_avg_px": crack_width,
        "metal_ratio_avg": metal_ratio,
        "structural_ratio_avg": structural_ratio,
        "solar_ratio_avg": solar_ratio,
        "quality_fail_ratio": quality_fail,
        "low_overlap_ratio": low_overlap,
    }


def _forecast_growth_metric(forecast_payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metrics = forecast_payload.get("metrics", [])
    if not isinstance(metrics, list) or len(metrics) < 2:
        return 0.0, {"initial_crack_ratio": 0.0, "final_crack_ratio": 0.0, "delta_crack_ratio": 0.0}

    first = metrics[0] if isinstance(metrics[0], dict) else {}
    last = metrics[-1] if isinstance(metrics[-1], dict) else {}
    initial = _safe_float(first.get("crack_ratio", 0.0))
    final = _safe_float(last.get("crack_ratio", 0.0))
    delta = max(0.0, final - initial)
    # 0.12 absolute crack-ratio increase is treated as high-growth.
    growth_norm = _clip01(delta / 0.12)
    return growth_norm, {
        "initial_crack_ratio": float(initial),
        "final_crack_ratio": float(final),
        "delta_crack_ratio": float(delta),
    }


def _criticality_metric(reconstruction_payload: dict[str, Any]) -> tuple[float, dict[str, float | int]]:
    total_points = int(max(0, _safe_float(reconstruction_payload.get("total_points", 0))))
    critical_count = 0
    avg_critical_risk = 0.0
    high_critical_risk = 0.0

    critical_path_raw = str(reconstruction_payload.get("critical_points_path", "") or "").strip()
    if critical_path_raw:
        critical_path = Path(critical_path_raw)
        if critical_path.exists():
            try:
                payload = json.loads(critical_path.read_text(encoding="utf-8"))
                points = payload.get("points", [])
                if isinstance(points, list):
                    risks = []
                    for item in points:
                        if isinstance(item, dict):
                            risks.append(_safe_float(item.get("criticality_score", item.get("risk_score", 0.0))))
                    if risks:
                        critical_count = int(len(risks))
                        avg_critical_risk = float(np.mean(risks))
                        high_critical_risk = float(np.percentile(risks, 90))
            except Exception:
                pass

    if critical_count == 0:
        # Fallback proxy from crack-focused 3D points if critical points file is unavailable.
        crack_points = int(max(0, _safe_float(reconstruction_payload.get("crack_points", 0))))
        critical_count = int(max(0, crack_points))
        avg_critical_risk = 0.35 if crack_points > 0 else 0.0
        high_critical_risk = 0.50 if crack_points > 0 else 0.0

    density = float(critical_count / max(1, total_points))
    # 1.5% critical-point density is treated as severe.
    density_norm = _clip01(density / 0.015)
    score_norm = _clip01(0.55 * density_norm + 0.45 * max(avg_critical_risk, high_critical_risk))
    return score_norm, {
        "total_points": int(total_points),
        "critical_points": int(critical_count),
        "critical_density": float(density),
        "avg_critical_risk": float(avg_critical_risk),
        "high_critical_risk": float(high_critical_risk),
    }


def _coverage_penalty_metric(coverage_payload: dict[str, Any], image_metrics: dict[str, float]) -> float:
    quality_fail = float(image_metrics.get("quality_fail_ratio", 0.0))
    low_overlap = float(image_metrics.get("low_overlap_ratio", 0.0))
    gap_count = int(max(0, _safe_float(coverage_payload.get("gap_suggestion_count", 0))))
    fail_count = int(max(0, _safe_float(coverage_payload.get("quality_fail_count", 0))))
    completion = _safe_float(coverage_payload.get("coverage_completion_pct", 100.0))

    gap_penalty = _clip01(gap_count / 20.0)
    fail_penalty = _clip01(fail_count / 12.0)
    completion_penalty = _clip01(max(0.0, (95.0 - completion) / 45.0))
    return _clip01(0.30 * quality_fail + 0.20 * low_overlap + 0.20 * gap_penalty + 0.20 * fail_penalty + 0.10 * completion_penalty)


def _weights_for_structure(structure_type: str) -> dict[str, float]:
    key = str(structure_type or "generic").strip().lower()
    if key == "solar":
        return {
            "crack": 0.18,
            "metal": 0.07,
            "structural": 0.10,
            "solar": 0.35,
            "growth": 0.12,
            "criticality": 0.12,
            "quality": 0.06,
        }
    if key in {"tower", "steel_tower"}:
        return {
            "crack": 0.20,
            "metal": 0.25,
            "structural": 0.20,
            "solar": 0.00,
            "growth": 0.12,
            "criticality": 0.15,
            "quality": 0.08,
        }
    # Bridge/building/generic defaults.
    return {
        "crack": 0.32,
        "metal": 0.14,
        "structural": 0.24,
        "solar": 0.08,
        "growth": 0.12,
        "criticality": 0.06,
        "quality": 0.04,
    }


def _base_hazard_per_year(structure_type: str, material_type: str) -> float:
    s = str(structure_type or "generic").strip().lower()
    m = str(material_type or "concrete").strip().lower()

    structure_base = {
        "bridge": 0.028,
        "building": 0.020,
        "tower": 0.024,
        "pipeline": 0.031,
        "solar": 0.016,
        "generic": 0.022,
    }.get(s, 0.022)
    material_adj = {
        "steel": 0.005,
        "reinforced_steel": 0.006,
        "concrete": 0.002,
        "masonry": 0.003,
        "composite": 0.002,
        "mixed": 0.003,
    }.get(m, 0.002)
    return float(max(1e-4, structure_base + material_adj))


def _grade_from_integrity(integrity_score: float) -> tuple[int, str]:
    s = float(integrity_score)
    if s >= 85.0:
        return 1, "Excellent"
    if s >= 70.0:
        return 2, "Good"
    if s >= 55.0:
        return 3, "Fair"
    if s >= 40.0:
        return 4, "Poor"
    return 5, "Critical"


def _build_timeline_trend(
    history: list[dict[str, Any]],
    current_run_utc: str,
    current_integrity: float,
    current_severity: float,
) -> dict[str, Any]:
    entries = []
    for row in history:
        if not isinstance(row, dict):
            continue
        run_utc = str(row.get("run_utc", "")).strip()
        if not run_utc:
            continue
        entries.append(
            {
                "run_utc": run_utc,
                "integrity_score": _safe_float(row.get("integrity_score", 0.0)),
                "severity_index": _safe_float(row.get("severity_index", 100.0)),
            }
        )
    entries.append(
        {
            "run_utc": str(current_run_utc),
            "integrity_score": float(current_integrity),
            "severity_index": float(current_severity),
        }
    )
    entries.sort(key=lambda r: _parse_utc(str(r.get("run_utc", ""))))

    previous = entries[-2] if len(entries) >= 2 else None
    previous_integrity = _safe_float(previous.get("integrity_score", current_integrity)) if previous else None
    previous_severity = _safe_float(previous.get("severity_index", current_severity)) if previous else None
    days_since_previous = 0.0
    delta_integrity = 0.0
    delta_severity = 0.0
    if previous is not None:
        t_prev = _parse_utc(str(previous.get("run_utc", "")))
        t_cur = _parse_utc(str(current_run_utc))
        days_since_previous = float(max(0.0, (t_cur - t_prev).total_seconds() / 86400.0))
        delta_integrity = float(current_integrity - previous_integrity)
        delta_severity = float(current_severity - previous_severity)

    slope_per_day = 0.0
    monthly_change = 0.0
    projected_6m = float(current_integrity)
    projected_12m = float(current_integrity)
    if len(entries) >= 2:
        t0 = _parse_utc(entries[0]["run_utc"])
        xs = np.asarray(
            [(_parse_utc(e["run_utc"]) - t0).total_seconds() / 86400.0 for e in entries],
            dtype=np.float64,
        )
        ys = np.asarray([_safe_float(e["integrity_score"], 0.0) for e in entries], dtype=np.float64)
        span_days = float(np.max(xs) - np.min(xs))
        if span_days >= 7.0:
            x_mean = float(xs.mean())
            y_mean = float(ys.mean())
            denom = float(np.sum((xs - x_mean) ** 2))
            if denom > 1e-9:
                slope_per_day = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
        else:
            slope_per_day = 0.0
        monthly_change = float(slope_per_day * 30.4375)
        projected_6m = _clip100(float(current_integrity + slope_per_day * 182.625))
        projected_12m = _clip100(float(current_integrity + slope_per_day * 365.25))

    trend_direction = "stable"
    if monthly_change <= -1.0:
        trend_direction = "degrading"
    elif monthly_change >= 1.0:
        trend_direction = "improving"

    if previous is None:
        summary = f"Asset baseline integrity score is {current_integrity:.1f}."
    else:
        months = days_since_previous / 30.4375 if days_since_previous > 0.0 else 0.0
        if months < 0.2:
            period = f"{days_since_previous:.1f} days"
        else:
            period = f"{months:.1f} months"
        if current_integrity < previous_integrity:
            summary = (
                f"Asset integrity score dropped from {previous_integrity:.1f} to "
                f"{current_integrity:.1f} over {period}."
            )
        elif current_integrity > previous_integrity:
            summary = (
                f"Asset integrity score improved from {previous_integrity:.1f} to "
                f"{current_integrity:.1f} over {period}."
            )
        else:
            summary = (
                f"Asset integrity score remained at {current_integrity:.1f} over "
                f"{period}."
            )

    return {
        "history_points": int(len(entries)),
        "previous_integrity_score": previous_integrity,
        "previous_severity_index": previous_severity,
        "delta_integrity_score": float(delta_integrity),
        "delta_severity_index": float(delta_severity),
        "days_since_previous": float(days_since_previous),
        "integrity_slope_per_day": float(slope_per_day),
        "integrity_change_per_month": float(monthly_change),
        "projected_integrity_6m": float(projected_6m),
        "projected_integrity_12m": float(projected_12m),
        "trend_direction": trend_direction,
        "summary": summary,
    }


def _history_entry_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    run_utc = str(summary.get("run_utc", "")).strip()
    if not run_utc:
        return None
    health = summary.get("health_scoring", {})
    if isinstance(health, dict) and health:
        return {
            "run_utc": run_utc,
            "asset_id": str(health.get("asset_id", _asset_id_from_summary(summary))),
            "integrity_score": _safe_float(health.get("integrity_score", 0.0)),
            "severity_index": _safe_float(health.get("severity_index", 100.0)),
            "structural_risk_score": _safe_float(health.get("structural_risk_score", 0.0)),
            "condition_level": int(max(1, _safe_float(health.get("condition_level", 5)))),
        }
    rows = summary.get("images", [])
    if not isinstance(rows, list) or not rows:
        return None
    img = _aggregate_image_metrics(rows)
    crack_n = _clip01(img["crack_ratio_avg"] / 0.02)
    metal_n = _clip01(img["metal_ratio_avg"] / 0.015)
    structural_n = _clip01(img["structural_ratio_avg"] / 0.03)
    severity = _clip100(100.0 * (0.45 * crack_n + 0.20 * metal_n + 0.35 * structural_n))
    integrity = _clip100(100.0 - severity)
    risk = _clip100(0.65 * severity)
    level, _ = _grade_from_integrity(integrity)
    return {
        "run_utc": run_utc,
        "asset_id": _asset_id_from_summary(summary),
        "integrity_score": float(integrity),
        "severity_index": float(severity),
        "structural_risk_score": float(risk),
        "condition_level": int(level),
    }


def load_asset_health_history(
    output_root: str | Path,
    asset_id: str,
    current_run_dir: str | Path | None = None,
    max_runs: int = 30,
    max_scan_runs: int = 160,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    if not root.exists():
        return []
    current = Path(current_run_dir).resolve() if current_run_dir else None
    run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    run_dirs.sort(key=lambda p: p.name)
    if max_scan_runs > 0 and len(run_dirs) > max_scan_runs:
        run_dirs = run_dirs[-max_scan_runs:]

    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if current is not None:
            try:
                if run_dir.resolve() == current:
                    continue
            except Exception:
                pass
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entry = _history_entry_from_summary(summary)
        if entry is None:
            continue
        if str(entry.get("asset_id", "")).strip() != str(asset_id).strip():
            continue
        entries.append(entry)

    entries.sort(key=lambda r: _parse_utc(str(r.get("run_utc", ""))))
    if max_runs > 0 and len(entries) > max_runs:
        entries = entries[-max_runs:]
    return entries


def evaluate_asset_health(
    asset_id: str,
    run_utc: str,
    structure_type: str,
    material_type: str,
    image_rows: list[dict[str, Any]],
    forecast_payload: dict[str, Any],
    reconstruction_payload: dict[str, Any],
    coverage_payload: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = image_rows if isinstance(image_rows, list) else []
    history_rows = history if isinstance(history, list) else []

    image_metrics = _aggregate_image_metrics(rows)
    growth_norm, growth_meta = _forecast_growth_metric(forecast_payload if isinstance(forecast_payload, dict) else {})
    critical_norm, critical_meta = _criticality_metric(
        reconstruction_payload if isinstance(reconstruction_payload, dict) else {}
    )
    coverage_norm = _coverage_penalty_metric(
        coverage_payload if isinstance(coverage_payload, dict) else {},
        image_metrics=image_metrics,
    )

    crack_norm = _clip01(
        0.70 * _clip01(image_metrics["crack_ratio_avg"] / 0.020)
        + 0.20 * _clip01(image_metrics["crack_max_width_avg_px"] / 8.0)
        + 0.10 * _clip01(image_metrics["crack_length_avg_px"] / 1800.0)
    )
    metal_norm = _clip01(image_metrics["metal_ratio_avg"] / 0.015)
    structural_norm = _clip01(image_metrics["structural_ratio_avg"] / 0.030)
    solar_norm = _clip01(image_metrics["solar_ratio_avg"] / 0.025)

    w = _weights_for_structure(structure_type)
    severity_norm = _clip01(
        w["crack"] * crack_norm
        + w["metal"] * metal_norm
        + w["structural"] * structural_norm
        + w["solar"] * solar_norm
        + w["growth"] * growth_norm
        + w["criticality"] * critical_norm
        + w["quality"] * coverage_norm
    )
    severity_index = _clip100(100.0 * severity_norm)
    integrity_score = _clip100(100.0 - severity_index)

    trend = _build_timeline_trend(
        history=history_rows,
        current_run_utc=str(run_utc),
        current_integrity=integrity_score,
        current_severity=severity_index,
    )
    monthly_change = _safe_float(trend.get("integrity_change_per_month", 0.0))
    trend_penalty_norm = _clip01(max(0.0, -monthly_change) / 4.0)

    structural_risk_score = _clip100(
        0.62 * severity_index
        + 20.0 * growth_norm
        + 12.0 * critical_norm
        + 6.0 * coverage_norm
        + 8.0 * trend_penalty_norm
    )

    baseline_hazard = _base_hazard_per_year(structure_type=structure_type, material_type=material_type)
    hazard_factor = (
        1.0
        + 1.8 * (structural_risk_score / 100.0)
        + 0.9 * growth_norm
        + 0.7 * critical_norm
        + 0.8 * trend_penalty_norm
    )
    adjusted_hazard = float(max(1e-6, baseline_hazard * hazard_factor))
    prob_6m = float(1.0 - math.exp(-adjusted_hazard * 0.5))
    prob_1y = float(1.0 - math.exp(-adjusted_hazard))
    prob_3y = float(1.0 - math.exp(-adjusted_hazard * 3.0))

    level, level_label = _grade_from_integrity(integrity_score)
    quality_confidence = _clip01(1.0 - coverage_norm)
    model_confidence = _clip01(0.6 + 0.4 * quality_confidence)

    timeline = list(history_rows)
    timeline.append(
        {
            "run_utc": str(run_utc),
            "asset_id": str(asset_id),
            "integrity_score": float(integrity_score),
            "severity_index": float(severity_index),
            "structural_risk_score": float(structural_risk_score),
            "condition_level": int(level),
        }
    )
    timeline.sort(key=lambda r: _parse_utc(str(r.get("run_utc", ""))))
    if len(timeline) > 30:
        timeline = timeline[-30:]

    return {
        "model_version": "health_scoring_v1",
        "generated_utc": _fmt_utc(datetime.now(timezone.utc)),
        "asset_id": str(asset_id),
        "run_utc": str(run_utc),
        "structure_type": str(structure_type or "generic"),
        "material_type": str(material_type or "concrete"),
        "severity_index": float(severity_index),
        "integrity_score": float(integrity_score),
        "structural_risk_score": float(structural_risk_score),
        "condition_level": int(level),
        "condition_grade": f"Level {level}",
        "condition_label": level_label,
        "probabilistic_failure": {
            "baseline_hazard_per_year": float(baseline_hazard),
            "adjusted_hazard_per_year": float(adjusted_hazard),
            "probability_6_month": float(prob_6m),
            "probability_1_year": float(prob_1y),
            "probability_3_year": float(prob_3y),
        },
        "components": {
            "crack_component": float(crack_norm),
            "metal_component": float(metal_norm),
            "structural_component": float(structural_norm),
            "solar_component": float(solar_norm),
            "growth_component": float(growth_norm),
            "criticality_component": float(critical_norm),
            "quality_penalty_component": float(coverage_norm),
            "weights": {k: float(v) for k, v in w.items()},
        },
        "image_metrics": image_metrics,
        "forecast_metrics": growth_meta,
        "criticality_metrics": critical_meta,
        "trend": trend,
        "confidence": {
            "quality_confidence": float(quality_confidence),
            "model_confidence": float(model_confidence),
        },
        "history": {
            "count": int(len(timeline)),
            "entries": timeline,
        },
    }
