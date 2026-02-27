"""Crack propagation forecasting with optional FEniCSx integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except Exception:  # pragma: no cover - optional dependency
    _skimage_skeletonize = None


@dataclass
class ForecastMetrics:
    step: int
    crack_pixels: int
    crack_ratio: float
    estimated_length_px: float
    estimated_max_width_px: float
    crack_length_m: float = 0.0
    stress_intensity_mpa_sqrt_m: float = 0.0
    delta_k_mpa_sqrt_m: float = 0.0
    fatigue_da_dn_m_per_cycle: float = 0.0
    growth_px_applied: float = 0.0
    factor_of_safety: float = 0.0
    failure_probability_horizon: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PropagationPhysicsConfig:
    pixel_size_mm: float = 0.5
    sigma_nominal_mpa: float = 35.0
    delta_sigma_mpa: float = 12.0
    cycles_per_year: float = 1_000_000.0
    horizon_years: float = 1.0
    cycles_per_step: float = 0.0
    geometry_factor: float = 1.12
    stress_concentration_factor: float = 1.0
    fracture_toughness_mpa_sqrt_m: float = 32.0
    paris_c: float = 2.5e-11
    paris_m: float = 3.2
    min_growth_px_per_step: float = 0.5
    max_growth_px_per_step: float = 15.0

    @staticmethod
    def _defaults_for_material(material_type: str) -> dict[str, float]:
        key = str(material_type or "concrete").strip().lower()
        if key == "steel":
            return {"kic": 65.0, "paris_c": 1.2e-12, "paris_m": 3.1}
        if key == "masonry":
            return {"kic": 14.0, "paris_c": 9.0e-11, "paris_m": 3.6}
        if key == "composite":
            return {"kic": 24.0, "paris_c": 3.5e-11, "paris_m": 3.3}
        if key == "mixed":
            return {"kic": 28.0, "paris_c": 2.8e-11, "paris_m": 3.3}
        return {"kic": 32.0, "paris_c": 2.5e-11, "paris_m": 3.2}

    @staticmethod
    def _load_factor_for_structure(structure_type: str) -> float:
        key = str(structure_type or "generic").strip().lower()
        return {
            "bridge": 1.25,
            "tower": 1.10,
            "pipeline": 1.30,
            "building": 1.00,
            "solar": 0.85,
            "generic": 1.00,
        }.get(key, 1.00)

    @classmethod
    def from_context(
        cls,
        structure_type: str = "generic",
        material_type: str = "concrete",
        pixel_size_mm: float = 0.5,
        sigma_nominal_mpa: float | None = None,
        delta_sigma_mpa: float | None = None,
        cycles_per_year: float = 1_000_000.0,
        horizon_years: float = 1.0,
        cycles_per_step: float = 0.0,
        fracture_toughness_mpa_sqrt_m: float | None = None,
        paris_c: float | None = None,
        paris_m: float | None = None,
    ) -> "PropagationPhysicsConfig":
        mat = cls._defaults_for_material(material_type)
        load_factor = cls._load_factor_for_structure(structure_type)
        sigma = float(35.0 * load_factor) if sigma_nominal_mpa is None else float(sigma_nominal_mpa)
        delta_sigma = float(12.0 * load_factor) if delta_sigma_mpa is None else float(delta_sigma_mpa)
        return cls(
            pixel_size_mm=float(max(0.01, pixel_size_mm)),
            sigma_nominal_mpa=float(max(0.1, sigma)),
            delta_sigma_mpa=float(max(0.05, delta_sigma)),
            cycles_per_year=float(max(1.0, cycles_per_year)),
            horizon_years=float(max(0.01, horizon_years)),
            cycles_per_step=float(max(0.0, cycles_per_step)),
            geometry_factor=1.12,
            stress_concentration_factor=float(max(0.5, load_factor)),
            fracture_toughness_mpa_sqrt_m=float(
                max(1.0, fracture_toughness_mpa_sqrt_m if fracture_toughness_mpa_sqrt_m is not None else mat["kic"])
            ),
            paris_c=float(max(1e-16, paris_c if paris_c is not None else mat["paris_c"])),
            paris_m=float(np.clip(paris_m if paris_m is not None else mat["paris_m"], 2.0, 6.0)),
            min_growth_px_per_step=0.5,
            max_growth_px_per_step=15.0,
        )


class CrackPropagationForecaster:
    """Fast geometric crack evolution forecaster for offline workflows."""

    def __init__(self, growth_px_per_step: float = 4.0, lateral_growth: int = 1):
        self.growth_px_per_step = float(max(growth_px_per_step, 0.0))
        self.lateral_growth = int(max(lateral_growth, 1))

    @staticmethod
    def _skeletonize_binary(binary_mask: np.ndarray) -> np.ndarray:
        if _skimage_skeletonize is not None:
            return _skimage_skeletonize(binary_mask > 0).astype(np.uint8)

        work = ((binary_mask > 0).astype(np.uint8) * 255).copy()
        skel = np.zeros_like(work, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(work, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(work, opened)
            skel = cv2.bitwise_or(skel, residue)
            work = eroded
            if cv2.countNonZero(work) == 0:
                break
        return (skel > 0).astype(np.uint8)

    @staticmethod
    def _skeleton(binary_mask: np.ndarray) -> np.ndarray:
        return CrackPropagationForecaster._skeletonize_binary(binary_mask)

    @staticmethod
    def _tip_points(skeleton: np.ndarray) -> np.ndarray:
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), dtype=np.uint8))
        # Endpoint in skeleton has exactly one neighbor -> sum 2 including itself.
        tips = np.argwhere((skeleton == 1) & (neighbors == 2))
        return tips

    @staticmethod
    def _tip_directions(skeleton: np.ndarray, tips: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        points = np.argwhere(skeleton == 1)
        if points.size == 0:
            return []

        tip_dirs: list[tuple[np.ndarray, np.ndarray]] = []
        for tip in tips:
            diffs = points - tip
            d2 = np.sum(diffs * diffs, axis=1)
            order = np.argsort(d2)
            direction = np.array([0.0, 0.0], dtype=np.float32)
            for idx in order:
                if d2[idx] > 0:
                    # Outward extension from nearest inner skeleton direction.
                    direction = (tip - points[idx]).astype(np.float32)
                    break
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                tip_dirs.append((tip.astype(np.float32), direction / norm))
        return tip_dirs

    @staticmethod
    def _metrics(mask: np.ndarray, step: int) -> ForecastMetrics:
        binary = (mask > 0).astype(np.uint8)
        skel = CrackPropagationForecaster._skeletonize_binary(binary)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        crack_pixels = int(binary.sum())
        total = int(binary.size)
        max_width = float(np.nan_to_num(dist.max(), nan=0.0, posinf=0.0, neginf=0.0))
        h, w = binary.shape[:2]
        return ForecastMetrics(
            step=step,
            crack_pixels=crack_pixels,
            crack_ratio=float(crack_pixels / max(total, 1)),
            estimated_length_px=float(skel.sum()),
            estimated_max_width_px=float(np.clip(max_width * 2.0, 0.0, float(max(h, w) * 2.0))),
        )

    @staticmethod
    def _with_physics(metrics: ForecastMetrics, values: dict[str, float]) -> ForecastMetrics:
        return ForecastMetrics(
            step=metrics.step,
            crack_pixels=metrics.crack_pixels,
            crack_ratio=metrics.crack_ratio,
            estimated_length_px=metrics.estimated_length_px,
            estimated_max_width_px=metrics.estimated_max_width_px,
            crack_length_m=float(values.get("crack_length_m", 0.0)),
            stress_intensity_mpa_sqrt_m=float(values.get("k_i", 0.0)),
            delta_k_mpa_sqrt_m=float(values.get("delta_k", 0.0)),
            fatigue_da_dn_m_per_cycle=float(values.get("da_dn", 0.0)),
            growth_px_applied=float(values.get("growth_px", 0.0)),
            factor_of_safety=float(values.get("fos", 0.0)),
            failure_probability_horizon=float(values.get("failure_probability", 0.0)),
        )

    @staticmethod
    def _physics_values(
        current_mask: np.ndarray,
        cfg: PropagationPhysicsConfig,
        steps: int,
    ) -> dict[str, float]:
        base = CrackPropagationForecaster._metrics(current_mask, step=0)
        px_size_m = float(max(1e-6, cfg.pixel_size_mm * 1e-3))
        crack_length_m = float(max(px_size_m, base.estimated_length_px * px_size_m))
        a_m = float(max(px_size_m, 0.5 * crack_length_m))

        y = float(max(0.8, cfg.geometry_factor * cfg.stress_concentration_factor))
        sigma = float(max(0.1, cfg.sigma_nominal_mpa))
        delta_sigma = float(max(0.01, cfg.delta_sigma_mpa))

        k_i = float(y * sigma * math.sqrt(math.pi * a_m))
        delta_k = float(y * delta_sigma * math.sqrt(math.pi * a_m))
        da_dn = float(max(0.0, cfg.paris_c * (max(delta_k, 1e-9) ** cfg.paris_m)))
        if cfg.cycles_per_step > 0.0:
            n_step = float(cfg.cycles_per_step)
        else:
            n_step = float(cfg.cycles_per_year * cfg.horizon_years / max(1, steps))
        da_step_m = float(max(0.0, da_dn * n_step))

        growth_px = float(da_step_m / px_size_m)
        growth_px = float(np.clip(growth_px, cfg.min_growth_px_per_step, cfg.max_growth_px_per_step))

        a_crit = float((cfg.fracture_toughness_mpa_sqrt_m / max(y * sigma, 1e-9)) ** 2 / math.pi)
        fos = float(np.clip(a_crit / max(a_m, 1e-9), 0.0, 100.0))
        progress = float(np.clip(a_m / max(a_crit, 1e-9), 0.0, 5.0))
        failure_probability = float(np.clip(1.0 - math.exp(-max(0.0, progress - 0.80) * 1.4), 0.0, 1.0))

        return {
            "crack_length_m": crack_length_m,
            "k_i": k_i,
            "delta_k": delta_k,
            "da_dn": da_dn,
            "growth_px": growth_px,
            "a_m": a_m,
            "a_crit": a_crit,
            "fos": fos,
            "failure_probability": failure_probability,
        }

    def _forecast_geometric(self, base_mask: np.ndarray, steps: int) -> tuple[list[np.ndarray], list[ForecastMetrics]]:
        """
        Fast geometric extrapolation from the observed crack mask.
        """
        mask = (base_mask > 0).astype(np.uint8) * 255
        sequence: list[np.ndarray] = [mask.copy()]
        metrics: list[ForecastMetrics] = [self._metrics(mask, step=0)]

        skeleton = self._skeleton(mask)
        tips = self._tip_points(skeleton)
        tip_dirs = self._tip_directions(skeleton, tips)

        if not tip_dirs:
            # No clear tips (e.g., closed contour). Fall back to isotropic growth.
            for step in range(1, steps + 1):
                grown = cv2.dilate(sequence[-1], np.ones((3, 3), np.uint8), iterations=1)
                sequence.append(grown)
                metrics.append(self._metrics(grown, step))
            return sequence, metrics

        h, w = mask.shape[:2]
        for step in range(1, steps + 1):
            grown = sequence[-1].copy()
            thickness = self.lateral_growth + max(0, step // 3)
            delta = int(round(self.growth_px_per_step))
            for tip, direction in tip_dirs:
                start = tip + direction * delta * (step - 1)
                end = tip + direction * delta * step
                x1, y1 = int(round(start[1])), int(round(start[0]))
                x2, y2 = int(round(end[1])), int(round(end[0]))
                x1 = int(np.clip(x1, 0, w - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                y2 = int(np.clip(y2, 0, h - 1))
                cv2.line(grown, (x1, y1), (x2, y2), color=255, thickness=thickness)

            grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            sequence.append(grown)
            metrics.append(self._metrics(grown, step))

        return sequence, metrics

    def _forecast_physics_informed(
        self,
        base_mask: np.ndarray,
        steps: int,
        physics: PropagationPhysicsConfig,
    ) -> tuple[list[np.ndarray], list[ForecastMetrics]]:
        """
        Physics-informed growth model:
        - stress intensity factor estimate K_I
        - fatigue crack growth via Paris law (da/dN = C*(DeltaK)^m)
        - material toughness check against critical crack length
        """
        mask = (base_mask > 0).astype(np.uint8) * 255
        sequence: list[np.ndarray] = [mask.copy()]
        initial_phys = self._physics_values(mask, physics, steps=steps)
        metrics: list[ForecastMetrics] = [self._with_physics(self._metrics(mask, step=0), initial_phys)]

        skeleton = self._skeleton(mask)
        tips = self._tip_points(skeleton)
        tip_dirs = self._tip_directions(skeleton, tips)

        h, w = mask.shape[:2]
        for step in range(1, steps + 1):
            grown = sequence[-1].copy()
            phys = self._physics_values(grown, physics, steps=steps)
            delta_px = int(max(1, round(float(phys.get("growth_px", 1.0)))))
            thickness = self.lateral_growth + max(0, delta_px // 6)

            if not tip_dirs:
                grown = cv2.dilate(grown, np.ones((3, 3), np.uint8), iterations=max(1, delta_px // 2))
                grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
                sequence.append(grown)
                metrics.append(self._with_physics(self._metrics(grown, step), phys))
                continue

            for tip, direction in tip_dirs:
                start = tip + direction * delta_px * (step - 1)
                end = tip + direction * delta_px * step
                x1, y1 = int(round(start[1])), int(round(start[0]))
                x2, y2 = int(round(end[1])), int(round(end[0]))
                x1 = int(np.clip(x1, 0, w - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                y2 = int(np.clip(y2, 0, h - 1))
                cv2.line(grown, (x1, y1), (x2, y2), color=255, thickness=thickness)

            grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            sequence.append(grown)
            metrics.append(self._with_physics(self._metrics(grown, step), phys))

        return sequence, metrics

    def forecast(
        self,
        base_mask: np.ndarray,
        steps: int = 6,
        mode: str = "geometric",
        physics: PropagationPhysicsConfig | None = None,
    ) -> tuple[list[np.ndarray], list[ForecastMetrics]]:
        if base_mask.ndim != 2:
            raise ValueError("base_mask must be a single-channel mask")
        mode_key = str(mode or "geometric").strip().lower()
        if mode_key in {"physics", "physics_informed", "fracture_mechanics"}:
            cfg = physics if physics is not None else PropagationPhysicsConfig()
            return self._forecast_physics_informed(base_mask, steps=max(1, int(steps)), physics=cfg)
        return self._forecast_geometric(base_mask, steps=max(1, int(steps)))

    @staticmethod
    def save_sequence(sequence: Iterable[np.ndarray], output_dir: str | Path, prefix: str = "forecast") -> list[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for idx, frame in enumerate(sequence):
            path = out / f"{prefix}_{idx:02d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        return paths


def run_fenicsx_phasefield(mask_path: str, steps: int, output_dir: str) -> tuple[bool, str]:
    """
    Try running legacy FEniCSx phase-field simulation from cracksim.py.
    Returns (success, message).
    """
    try:
        from cracksim import run_phasefield
    except Exception as exc:
        return False, f"Unable to import cracksim/FEniCSx stack: {exc}"

    try:
        run_phasefield(mask_path=mask_path, steps=steps, output_dir=output_dir)
        return True, f"FEniCSx simulation completed in: {output_dir}"
    except Exception as exc:
        return False, f"FEniCSx execution failed: {exc}"
