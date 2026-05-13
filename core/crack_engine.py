"""Crack propagation engine façade — spec-named API on top of propagation.py."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import AppError, ERR_INVALID_INPUT
from .propagation import (
    CrackPropagationForecaster,
    PropagationPhysicsConfig,
    ForecastMetrics,
    run_fenicsx_phasefield,
)
from .validation import ValidationMessage, SEVERITY_ERROR, SEVERITY_WARNING


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CrackPropagationConfig:
    image_path: str
    mask_path: str | None = None
    pixel_size_mm_per_px: float = 0.5
    sigma_nominal_mpa: float = 35.0
    delta_sigma_mpa: float = 12.0
    cycles_per_year: float = 1_000_000.0
    horizon_years: float = 1.0
    steps: int = 6
    cycles_per_step: float = 0.0
    kic_mpa_sqrt_m: float | None = None
    paris_c: float | None = None
    paris_m: float | None = None
    material_profile: str | None = "concrete"
    structure_type: str = "generic"
    use_fenicsx: bool = False
    growth_px_per_step: float = 4.0
    lateral_growth: int = 1
    mode: str = "physics_informed"   # geometric | physics_informed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrackState:
    step: int
    mask_path: str
    crack_length_m: float
    width_max_px: float


@dataclass
class CrackGeometry:
    length_m: float
    length_px: float
    endpoints: list[tuple[int, int]]
    branches: int
    orientation_deg: float
    max_width_px: float


@dataclass
class StressIntensityResult:
    k_i_mpa_sqrt_m: float
    delta_k_mpa_sqrt_m: float
    crack_length_m: float
    crack_half_length_m: float
    factor_of_safety: float
    a_crit_m: float
    assumptions: list[str] = field(default_factory=list)


@dataclass
class CrackPropagationResult:
    id: str
    config: CrackPropagationConfig
    forecast_images: list[str]
    metrics: list[dict[str, Any]]
    risk_level: str
    critical_points: list[dict[str, Any]]
    summary: str
    assumptions: list[str]
    output_dir: str
    fenicsx_message: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "forecast_images": list(self.forecast_images),
            "metrics": list(self.metrics),
            "risk_level": self.risk_level,
            "critical_points": list(self.critical_points),
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "output_dir": self.output_dir,
            "fenicsx_message": self.fenicsx_message,
            "created_at": self.created_at,
        }


# ── Geometry / stress helpers ─────────────────────────────────────────────────

def extract_crack_geometry(mask_path: Path | str, pixel_size_mm_per_px: float) -> CrackGeometry:
    """Skeletonize mask and estimate crack metrics."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise AppError(ERR_INVALID_INPUT, f"Cannot read mask: {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return CrackGeometry(0.0, 0.0, [], 0, 0.0, 0.0)
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(binary > 0).astype(np.uint8)
    except Exception:
        skel = CrackPropagationForecaster._skeletonize_binary(binary)  # type: ignore[attr-defined]

    length_px = float(skel.sum())
    length_m = length_px * (float(pixel_size_mm_per_px) / 1000.0)

    # Endpoints
    neighbor = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    tip_pts = np.argwhere((skel == 1) & (neighbor == 2))
    endpoints = [(int(p[1]), int(p[0])) for p in tip_pts]

    # Branches: skeleton pixels with 3+ neighbors
    branches = int(np.sum((skel == 1) & (neighbor >= 4)))

    # Orientation via PCA of skeleton points
    pts = np.argwhere(skel == 1)
    orientation = 0.0
    if pts.shape[0] >= 2:
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal = eig_vecs[:, -1]
        orientation = float(math.degrees(math.atan2(principal[1], principal[0])))

    # Max width via distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_width_px = float(np.nan_to_num(dist.max(), nan=0.0, posinf=0.0)) * 2.0

    return CrackGeometry(
        length_m=length_m,
        length_px=length_px,
        endpoints=endpoints,
        branches=branches,
        orientation_deg=orientation,
        max_width_px=max_width_px,
    )


def estimate_stress_intensity(
    geometry: CrackGeometry,
    sigma_mpa: float,
    delta_sigma_mpa: float = 12.0,
    geometry_factor: float = 1.12,
    kic_mpa_sqrt_m: float | None = None,
) -> StressIntensityResult:
    """Simplified K_I = Y σ √(π a) for a half-crack length a."""
    a_m = max(1e-6, 0.5 * geometry.length_m)
    y = float(max(0.5, geometry_factor))
    k_i = y * float(sigma_mpa) * math.sqrt(math.pi * a_m)
    delta_k = y * float(delta_sigma_mpa) * math.sqrt(math.pi * a_m)
    kic = float(kic_mpa_sqrt_m) if kic_mpa_sqrt_m else 32.0
    a_crit = (kic / max(y * sigma_mpa, 1e-9)) ** 2 / math.pi
    fos = float(np.clip(a_crit / max(a_m, 1e-9), 0.0, 100.0))
    return StressIntensityResult(
        k_i_mpa_sqrt_m=k_i,
        delta_k_mpa_sqrt_m=delta_k,
        crack_length_m=geometry.length_m,
        crack_half_length_m=a_m,
        factor_of_safety=fos,
        a_crit_m=a_crit,
        assumptions=[
            "Through-thickness centre crack model.",
            "Geometry factor Y = 1.12 (centre crack in finite plate).",
            f"Half-length a = 0.5 * total crack length ({a_m*1000:.2f} mm).",
            f"Uses K_IC = {kic:.1f} MPa√m (material default).",
        ],
    )


def calculate_crack_growth_step(
    current_length_m: float,
    delta_k_mpa_sqrt_m: float,
    paris_c: float,
    paris_m: float,
    cycles: float,
) -> float:
    """Δa = C * (ΔK)^m * N. Returns Δa in metres."""
    if cycles <= 0 or delta_k_mpa_sqrt_m <= 0:
        return 0.0
    return float(max(0.0, paris_c) * (max(delta_k_mpa_sqrt_m, 1e-9) ** float(paris_m)) * float(cycles))


def generate_growth_overlay(
    image_path: Path | str,
    crack_states: list[CrackState],
    output_path: Path | str,
) -> Path:
    """Render predicted crack states on the source image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise AppError(ERR_INVALID_INPUT, f"Cannot read image: {image_path}")
    canvas = img.copy()
    n = max(1, len(crack_states))
    for state in crack_states:
        mask = cv2.imread(state.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != canvas.shape[:2]:
            mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        ratio = float(state.step) / float(n)
        # Cool→hot ramp by step
        color = (int(255 * (1.0 - ratio)), int(80 + 100 * ratio), int(255 * ratio))
        layer = np.zeros_like(canvas)
        layer[:] = color
        alpha = (mask > 0).astype(np.float32)[..., None] * 0.25
        canvas = (canvas * (1.0 - alpha) + layer * alpha).astype(np.uint8)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def classify_crack_risk(result: CrackPropagationResult) -> str:
    """Assign Low / Medium / High / Critical based on growth, KIC margin, area."""
    metrics = list(result.metrics or [])
    if not metrics:
        return "Low"
    last = metrics[-1]
    fos = float(last.get("factor_of_safety", 0.0) or 0.0)
    fail_p = float(last.get("failure_probability_horizon", 0.0) or 0.0)
    grew = float(last.get("crack_length_m", 0.0) or 0.0) > float(metrics[0].get("crack_length_m", 0.0) or 0.0) * 1.5
    if fos < 1.0 or fail_p > 0.5:
        return "Critical"
    if fos < 1.5 or fail_p > 0.2 or grew:
        return "High"
    if fos < 3.0 or fail_p > 0.05:
        return "Medium"
    return "Low"


def validate_crack_config(config: CrackPropagationConfig) -> list[ValidationMessage]:
    msgs: list[ValidationMessage] = []
    if not Path(config.image_path).exists():
        msgs.append(ValidationMessage("image_path", SEVERITY_ERROR, f"Image not found: {config.image_path}"))
    if config.mask_path and not Path(config.mask_path).exists():
        msgs.append(ValidationMessage("mask_path", SEVERITY_ERROR, f"Mask not found: {config.mask_path}"))
    if config.pixel_size_mm_per_px <= 0:
        msgs.append(ValidationMessage("pixel_size_mm_per_px", SEVERITY_ERROR, "Pixel size must be > 0."))
    if config.steps < 1 or config.steps > 100:
        msgs.append(ValidationMessage("steps", SEVERITY_ERROR, "Steps must be 1..100."))
    if config.sigma_nominal_mpa <= 0:
        msgs.append(ValidationMessage("sigma_nominal_mpa", SEVERITY_ERROR, "Sigma must be > 0."))
    if config.delta_sigma_mpa <= 0:
        msgs.append(ValidationMessage("delta_sigma_mpa", SEVERITY_ERROR, "Delta sigma must be > 0."))
    if config.horizon_years <= 0:
        msgs.append(ValidationMessage("horizon_years", SEVERITY_ERROR, "Horizon must be > 0."))
    return msgs


# ── Main run ──────────────────────────────────────────────────────────────────

def run_crack_propagation(config: CrackPropagationConfig, output_dir: Path | str) -> CrackPropagationResult:
    """End-to-end crack propagation forecast for a single image+mask."""
    issues = validate_crack_config(config)
    blockers = [i for i in issues if i.severity == SEVERITY_ERROR]
    if blockers:
        raise AppError(ERR_INVALID_INPUT, "Invalid crack propagation config.",
                       technical_message="; ".join(i.message for i in blockers))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_path = config.mask_path or ""
    if not mask_path:
        # Best-effort: try to derive crack mask from image
        from .detection import detect_cracks, load_image
        img = load_image(config.image_path)
        cd = detect_cracks(img)
        if cd.crack_pixels == 0:
            raise AppError(ERR_INVALID_INPUT, "No crack detected on input image. Provide a mask.")
        derived = out_dir / "derived_crack_mask.png"
        cv2.imwrite(str(derived), cd.mask)
        mask_path = str(derived)
    base_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if base_mask is None:
        raise AppError(ERR_INVALID_INPUT, f"Cannot read mask: {mask_path}")

    physics_cfg = PropagationPhysicsConfig.from_context(
        structure_type=config.structure_type,
        material_type=config.material_profile or "concrete",
        pixel_size_mm=config.pixel_size_mm_per_px,
        sigma_nominal_mpa=config.sigma_nominal_mpa,
        delta_sigma_mpa=config.delta_sigma_mpa,
        cycles_per_year=config.cycles_per_year,
        horizon_years=config.horizon_years,
        cycles_per_step=config.cycles_per_step,
        fracture_toughness_mpa_sqrt_m=config.kic_mpa_sqrt_m,
        paris_c=config.paris_c,
        paris_m=config.paris_m,
    )

    forecaster = CrackPropagationForecaster(
        growth_px_per_step=config.growth_px_per_step,
        lateral_growth=config.lateral_growth,
    )
    seq, metrics = forecaster.forecast(
        base_mask,
        steps=int(config.steps),
        mode=config.mode,
        physics=physics_cfg,
    )

    # Persist sequence
    image_name = Path(config.image_path).stem
    frame_paths = forecaster.save_sequence(seq, masks_dir, prefix=f"{image_name}_forecast")
    metrics_payload = [m.to_dict() for m in metrics]

    # FEniCSx leg
    fenicsx_msg = ""
    if config.use_fenicsx:
        ok, fenicsx_msg = run_fenicsx_phasefield(
            mask_path=str(mask_path),
            steps=int(config.steps),
            output_dir=str(out_dir / "fenicsx"),
        )

    # Overlay on source image
    states = [
        CrackState(step=i, mask_path=p, crack_length_m=float(metrics_payload[i].get("crack_length_m", 0.0) or 0.0),
                   width_max_px=float(metrics_payload[i].get("estimated_max_width_px", 0.0) or 0.0))
        for i, p in enumerate(frame_paths)
    ]
    overlay_path = out_dir / f"{image_name}_growth_overlay.png"
    try:
        generate_growth_overlay(config.image_path, states, overlay_path)
    except Exception:
        pass

    # Critical points: top-3 highest delta_k or smallest factor_of_safety
    ranked = sorted(
        metrics_payload,
        key=lambda m: (float(m.get("factor_of_safety", 0.0) or 0.0),),
    )
    critical_points = [
        {
            "step": m.get("step"),
            "crack_length_m": m.get("crack_length_m"),
            "factor_of_safety": m.get("factor_of_safety"),
            "failure_probability_horizon": m.get("failure_probability_horizon"),
        }
        for m in ranked[:3]
    ]

    last = metrics_payload[-1] if metrics_payload else {}
    summary = (
        f"Predicted crack length grows from "
        f"{(metrics_payload[0].get('crack_length_m') if metrics_payload else 0):.4f} m "
        f"to {last.get('crack_length_m', 0):.4f} m over {len(metrics_payload) - 1} steps. "
        f"FoS: {last.get('factor_of_safety', 0):.2f}. Failure probability over horizon: "
        f"{100.0 * float(last.get('failure_probability_horizon', 0.0) or 0.0):.1f}%."
    )

    result = CrackPropagationResult(
        id=str(uuid.uuid4()),
        config=config,
        forecast_images=frame_paths,
        metrics=metrics_payload,
        risk_level="Low",
        critical_points=critical_points,
        summary=summary,
        assumptions=[
            "Linear elastic fracture mechanics (LEFM) regime.",
            "Centre-cracked plate geometry (Y ≈ 1.12).",
            f"Pixel size = {physics_cfg.pixel_size_mm:.3f} mm/px.",
            f"σ_nom = {physics_cfg.sigma_nominal_mpa:.1f} MPa, Δσ = {physics_cfg.delta_sigma_mpa:.1f} MPa.",
            f"Paris C = {physics_cfg.paris_c:.2e}, m = {physics_cfg.paris_m:.2f}.",
            f"K_IC = {physics_cfg.fracture_toughness_mpa_sqrt_m:.1f} MPa√m.",
            "Estimates are engineering forecasts, not certified physical truth.",
        ],
        output_dir=str(out_dir),
        fenicsx_message=fenicsx_msg,
    )
    result.risk_level = classify_crack_risk(result)

    summary_path = out_dir / "crack_propagation.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
