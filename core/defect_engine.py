"""Defect detection orchestration — classical + AI, per-image and per-dataset."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .data_library import get_dataset, get_image_assets
from .detection import (
    detect_cracks,
    detect_metal_defects,
    detect_solar_defects,
    detect_structural_defects,
    load_image,
    DefectHit,
    StructuralDefectResult,
    SolarDefectResult,
)
from .errors import AppError, ERR_INVALID_INPUT, ERR_MODEL_MISSING
from .models import get_model_spec, model_status as _model_status, ModelSpec, resolve_model_path
from .validation import ValidationMessage, SEVERITY_ERROR, SEVERITY_WARNING


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SEVERITY_CRIT = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MED = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class DefectDetectionConfig:
    mode: str = "hybrid"                         # classical | ai | hybrid
    model_key: str | None = "structural_multiclass_detector"
    threshold: float = 0.25
    iou_threshold: float = 0.45
    min_area_px: int = 30
    output_masks: bool = True
    output_overlay: bool = True
    classes: list[str] = field(default_factory=list)     # filter classes if non-empty
    structure_type: str = "generic"
    material_type: str = "concrete"
    solar_model_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectedDefect:
    id: str
    image_id: str
    image_path: str
    defect_type: str
    confidence: float
    severity: str
    bbox: list[int]                              # [x1, y1, x2, y2]
    area_px: int
    mask_path: str | None = None
    overlay_path: str | None = None
    notes: str = ""
    source: str = "classical"                    # classical | ai
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefectDetectionResult:
    id: str
    dataset_id: str
    config: DefectDetectionConfig
    output_dir: str
    image_count: int
    defects: list[DetectedDefect]
    model_used: str | None
    model_available: bool
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "config": self.config.to_dict(),
            "output_dir": self.output_dir,
            "image_count": self.image_count,
            "defect_count": len(self.defects),
            "model_used": self.model_used,
            "model_available": self.model_available,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "defects": [d.to_dict() for d in self.defects],
        }


@dataclass
class SeverityRules:
    critical_area_pct: float = 5.0
    high_area_pct: float = 1.0
    medium_area_pct: float = 0.25
    critical_types: tuple[str, ...] = ("rebar_exposure", "collapse", "hotspot")
    confidence_high: float = 0.8


# ── Severity classification ───────────────────────────────────────────────────

def classify_defect_severity(
    defect: DetectedDefect | dict[str, Any],
    image_area_px: int,
    rules: SeverityRules | None = None,
) -> str:
    rules = rules or SeverityRules()
    if isinstance(defect, DetectedDefect):
        area = int(defect.area_px)
        conf = float(defect.confidence)
        dtype = str(defect.defect_type)
    else:
        area = int(defect.get("area_px", 0))
        conf = float(defect.get("confidence", 0.0))
        dtype = str(defect.get("defect_type", ""))
    pct = 100.0 * area / max(1, image_area_px)
    if dtype in rules.critical_types and conf >= rules.confidence_high:
        return SEVERITY_CRIT
    if pct >= rules.critical_area_pct:
        return SEVERITY_CRIT
    if pct >= rules.high_area_pct:
        return SEVERITY_HIGH
    if pct >= rules.medium_area_pct:
        return SEVERITY_MED
    if conf >= rules.confidence_high:
        return SEVERITY_MED
    return SEVERITY_LOW


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    if mask is None or mask.size == 0:
        return [0, 0, 0, 0]
    binary = (mask > 0).astype(np.uint8)
    nonzero = np.argwhere(binary)
    if nonzero.size == 0:
        return [0, 0, 0, 0]
    y1, x1 = nonzero.min(axis=0)
    y2, x2 = nonzero.max(axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]


def _save_mask(mask: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    return out_path


def _hit_to_defect(
    hit: DefectHit,
    image_path: str,
    image_id: str,
    image_area_px: int,
    source: str,
    rules: SeverityRules,
    output_dir: Path | None,
    mask: np.ndarray | None = None,
) -> DetectedDefect:
    bbox = hit.bbox if hasattr(hit, "bbox") and hit.bbox else [0, 0, 0, 0]
    area = 0
    if mask is not None and mask.size > 0:
        area = int((mask > 0).sum())
    elif bbox and len(bbox) == 4:
        area = max(0, int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    defect_id = str(uuid.uuid4())
    mask_path: Path | None = None
    if mask is not None and output_dir is not None:
        mask_path = output_dir / "masks" / f"{Path(image_path).stem}_{hit.label}_{defect_id[:8]}.png"
        _save_mask(mask, mask_path)
    d = DetectedDefect(
        id=defect_id,
        image_id=image_id,
        image_path=image_path,
        defect_type=str(hit.label),
        confidence=float(hit.confidence),
        severity=SEVERITY_LOW,
        bbox=[int(x) for x in (bbox or [0, 0, 0, 0])],
        area_px=area,
        mask_path=str(mask_path) if mask_path else None,
        source=source,
    )
    d.severity = classify_defect_severity(d, image_area_px=image_area_px, rules=rules)
    return d


# ── Overlay ───────────────────────────────────────────────────────────────────

_SEVERITY_COLORS_BGR = {
    SEVERITY_CRIT: (0, 0, 220),
    SEVERITY_HIGH: (0, 80, 240),
    SEVERITY_MED: (0, 200, 255),
    SEVERITY_LOW: (60, 200, 80),
    SEVERITY_INFO: (200, 200, 200),
}


def create_defect_overlay(
    image_path: Path | str,
    defects: list[DetectedDefect],
    output_path: Path | str,
) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise AppError(ERR_INVALID_INPUT, f"Cannot read image: {image_path}")
    out = img.copy()
    for d in defects:
        color = _SEVERITY_COLORS_BGR.get(d.severity, (255, 255, 255))
        if d.bbox and len(d.bbox) == 4 and any(d.bbox):
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.defect_type} {d.confidence:.2f}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if d.mask_path and Path(d.mask_path).exists():
            mask = cv2.imread(d.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] == out.shape[:2]:
                color_layer = np.zeros_like(out)
                color_layer[:] = color
                alpha = (mask > 0).astype(np.float32)[..., None] * 0.35
                out = (out * (1.0 - alpha) + color_layer * alpha).astype(np.uint8)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), out)
    return out_p


# ── Per-image detection ───────────────────────────────────────────────────────

def detect_defects_classical(image_path: Path | str, config: DefectDetectionConfig) -> list[DetectedDefect]:
    image = load_image(image_path)
    h, w = image.shape[:2]
    image_area = h * w
    rules = SeverityRules()

    defects: list[DetectedDefect] = []
    crack = detect_cracks(image, min_area_px=config.min_area_px)
    if crack.crack_pixels > 0:
        bbox = _bbox_from_mask(crack.mask)
        d = DetectedDefect(
            id=str(uuid.uuid4()),
            image_id="",
            image_path=str(image_path),
            defect_type="crack",
            confidence=min(0.95, 0.4 + 50.0 * float(crack.crack_ratio)),
            severity=SEVERITY_LOW,
            bbox=bbox,
            area_px=int(crack.crack_pixels),
            source="classical",
        )
        d.severity = classify_defect_severity(d, image_area, rules)
        defects.append(d)
    metal = detect_metal_defects(image, min_area_px=config.min_area_px)
    if metal.defect_pixels > 0:
        bbox = _bbox_from_mask(metal.mask)
        d = DetectedDefect(
            id=str(uuid.uuid4()),
            image_id="",
            image_path=str(image_path),
            defect_type="corrosion",
            confidence=min(0.95, 0.4 + 50.0 * float(metal.defect_ratio)),
            severity=SEVERITY_LOW,
            bbox=bbox,
            area_px=int(metal.defect_pixels),
            source="classical",
        )
        d.severity = classify_defect_severity(d, image_area, rules)
        defects.append(d)
    return defects


def detect_defects_ai(
    image_path: Path | str,
    config: DefectDetectionConfig,
    structural: bool = True,
) -> list[DetectedDefect]:
    image = load_image(image_path)
    h, w = image.shape[:2]
    image_area = h * w
    rules = SeverityRules()
    defects: list[DetectedDefect] = []

    if structural:
        result = detect_structural_defects(
            image,
            min_area_px=config.min_area_px,
            model_key=str(config.model_key or "structural_multiclass_detector"),
            use_model=True,
        )
        if not result.model_available:
            raise AppError(
                ERR_MODEL_MISSING,
                f"Structural model {config.model_key!r} not available.",
                technical_message=f"model_used={result.model_used}, model_available={result.model_available}",
                recovery_action="Configure model path in Developer Tools.",
            )
        for hit in result.detections:
            if config.classes and hit.label not in config.classes:
                continue
            if hit.confidence < config.threshold:
                continue
            mask = result.mask_by_class.get(hit.label) if hasattr(result, "mask_by_class") else None
            defects.append(_hit_to_defect(hit, str(image_path), "", image_area, "ai", rules, output_dir=None, mask=mask))
    else:
        result_s = detect_solar_defects(
            image,
            thermal_gray=None,
            min_area_px=config.min_area_px,
            model_key=str(config.solar_model_key or config.model_key or "solar_defect_detector"),
            use_model=True,
        )
        if not result_s.model_available:
            raise AppError(
                ERR_MODEL_MISSING,
                f"Solar model {config.solar_model_key!r} not available.",
                recovery_action="Configure solar model path in Developer Tools.",
            )
        for hit in result_s.detections:
            if config.classes and hit.label not in config.classes:
                continue
            if hit.confidence < config.threshold:
                continue
            defects.append(_hit_to_defect(hit, str(image_path), "", image_area, "ai", rules, output_dir=None))
    return defects


# ── Dataset-level orchestration ───────────────────────────────────────────────

def validate_defect_config(config: DefectDetectionConfig) -> list[ValidationMessage]:
    msgs: list[ValidationMessage] = []
    if config.mode not in ("classical", "ai", "hybrid"):
        msgs.append(ValidationMessage("mode", SEVERITY_ERROR, f"Invalid mode {config.mode!r}."))
    if config.threshold < 0.0 or config.threshold > 1.0:
        msgs.append(ValidationMessage("threshold", SEVERITY_ERROR, "Threshold must be in [0, 1]."))
    if config.mode in ("ai", "hybrid") and config.model_key:
        info = _model_status(config.model_key)
        if not info.get("exists"):
            msgs.append(ValidationMessage(
                "model_key", SEVERITY_WARNING,
                f"Model {config.model_key!r} not available — AI mode will fall back to classical.",
                fix_action="Configure model path in Developer Tools.",
            ))
    return msgs


def run_defect_detection(
    project_root: Path | str,
    dataset_id: str,
    config: DefectDetectionConfig | None = None,
    image_ids: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DefectDetectionResult:
    """Run selected detection mode over dataset or selected images."""
    config = config or DefectDetectionConfig()
    project_root = Path(project_root)
    ds = get_dataset(project_root, dataset_id)
    if ds is None:
        raise AppError(ERR_INVALID_INPUT, f"Dataset not found: {dataset_id}")

    assets = get_image_assets(project_root, dataset_id, page=0, page_size=10**6)
    if image_ids:
        keep = set(image_ids)
        assets = [a for a in assets if a.id in keep]

    run_id = str(uuid.uuid4())
    out_dir = project_root / "analysis" / "defects" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    started = _now_iso()
    rules = SeverityRules()

    model_used: str | None = None
    model_available = True
    if config.mode in ("ai", "hybrid") and config.model_key:
        info = _model_status(config.model_key)
        model_used = config.model_key
        model_available = bool(info.get("exists"))

    all_defects: list[DetectedDefect] = []
    total = max(1, len(assets))
    for idx, asset in enumerate(assets):
        if progress_callback:
            try:
                progress_callback(int(100.0 * idx / total), Path(asset.file_path).name)
            except Exception:
                pass
        try:
            image = load_image(asset.file_path)
            h, w = image.shape[:2]
            image_area = h * w
        except Exception as exc:
            continue

        # Classical leg
        classical_defects: list[DetectedDefect] = []
        if config.mode in ("classical", "hybrid"):
            crack = detect_cracks(image, min_area_px=config.min_area_px)
            metal = detect_metal_defects(image, min_area_px=config.min_area_px)
            if crack.crack_pixels > 0:
                mask_path = out_dir / "masks" / f"{asset.id}_crack.png"
                _save_mask(crack.mask, mask_path)
                d = DetectedDefect(
                    id=str(uuid.uuid4()),
                    image_id=asset.id,
                    image_path=asset.file_path,
                    defect_type="crack",
                    confidence=min(0.95, 0.4 + 50.0 * float(crack.crack_ratio)),
                    severity=SEVERITY_LOW,
                    bbox=_bbox_from_mask(crack.mask),
                    area_px=int(crack.crack_pixels),
                    mask_path=str(mask_path),
                    source="classical",
                )
                d.severity = classify_defect_severity(d, image_area, rules)
                classical_defects.append(d)
            if metal.defect_pixels > 0:
                mask_path = out_dir / "masks" / f"{asset.id}_corrosion.png"
                _save_mask(metal.mask, mask_path)
                d = DetectedDefect(
                    id=str(uuid.uuid4()),
                    image_id=asset.id,
                    image_path=asset.file_path,
                    defect_type="corrosion",
                    confidence=min(0.95, 0.4 + 50.0 * float(metal.defect_ratio)),
                    severity=SEVERITY_LOW,
                    bbox=_bbox_from_mask(metal.mask),
                    area_px=int(metal.defect_pixels),
                    mask_path=str(mask_path),
                    source="classical",
                )
                d.severity = classify_defect_severity(d, image_area, rules)
                classical_defects.append(d)

        # AI leg
        ai_defects: list[DetectedDefect] = []
        if config.mode in ("ai", "hybrid") and model_available:
            try:
                result_s = detect_structural_defects(
                    image,
                    min_area_px=config.min_area_px,
                    model_key=str(config.model_key or "structural_multiclass_detector"),
                    use_model=True,
                )
                if result_s.model_available:
                    for hit in result_s.detections:
                        if config.classes and hit.label not in config.classes:
                            continue
                        if hit.confidence < config.threshold:
                            continue
                        mask = None
                        if hasattr(result_s, "mask_by_class"):
                            mask = result_s.mask_by_class.get(hit.label)
                        ai_defects.append(_hit_to_defect(
                            hit, asset.file_path, asset.id, image_area, "ai", rules,
                            output_dir=out_dir, mask=mask,
                        ))
            except Exception:
                pass

        image_defects = classical_defects + ai_defects
        all_defects.extend(image_defects)

        if config.output_overlay and image_defects:
            overlay_path = out_dir / "overlays" / f"{asset.id}.jpg"
            try:
                create_defect_overlay(asset.file_path, image_defects, overlay_path)
                for d in image_defects:
                    d.overlay_path = str(overlay_path)
            except Exception:
                pass

    finished = _now_iso()
    res = DefectDetectionResult(
        id=run_id,
        dataset_id=dataset_id,
        config=config,
        output_dir=str(out_dir),
        image_count=len(assets),
        defects=all_defects,
        model_used=model_used,
        model_available=model_available,
        started_at=started,
        finished_at=finished,
    )

    # Persist run summary
    summary_path = out_dir / "defects.json"
    summary_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    if progress_callback:
        try:
            progress_callback(100, "defect detection complete")
        except Exception:
            pass
    return res


def export_defect_table(result_or_path: DefectDetectionResult | Path | str, output_format: str = "csv") -> Path:
    """Export defects to CSV / JSON / Markdown table next to the defect run dir."""
    if isinstance(result_or_path, DefectDetectionResult):
        run_dir = Path(result_or_path.output_dir)
        defects = result_or_path.defects
    else:
        run_dir = Path(result_or_path)
        summary = run_dir if run_dir.is_file() else (run_dir / "defects.json")
        data = json.loads(Path(summary).read_text(encoding="utf-8"))
        defects = [DetectedDefect(**{k: v for k, v in d.items() if k in DetectedDefect.__dataclass_fields__})
                   for d in data.get("defects", [])]
        run_dir = Path(summary).parent

    fmt = (output_format or "csv").lower()
    if fmt == "json":
        out = run_dir / "defects_table.json"
        out.write_text(json.dumps([d.to_dict() for d in defects], indent=2), encoding="utf-8")
        return out
    if fmt in ("md", "markdown"):
        out = run_dir / "defects_table.md"
        lines = ["| Image | Type | Confidence | Severity | Area px | bbox |", "|---|---|---:|---|---:|---|"]
        for d in defects:
            lines.append(
                f"| {Path(d.image_path).name} | {d.defect_type} | {d.confidence:.2f} | {d.severity} | {d.area_px} | {d.bbox} |"
            )
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    out = run_dir / "defects_table.csv"
    cols = ["id", "image_id", "image_path", "defect_type", "confidence",
            "severity", "bbox", "area_px", "mask_path", "overlay_path", "source", "created_at"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in defects:
            row = d.to_dict()
            row["bbox"] = ",".join(str(x) for x in d.bbox)
            w.writerow({k: row.get(k, "") for k in cols})
    return out


# ── Model registry helpers (spec-named) ───────────────────────────────────────

def register_model(
    model_key: str,
    model_path: Path | str,
    model_type: str = "onnx_yolo",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Register or update a model entry in the registry JSON."""
    from .settings import configure_model_path
    # configure_model_path validates existence + writes back
    info = configure_model_path(model_key, model_path)
    # Update labels/kind if supplied
    from .models import load_registry, registry_path
    reg = load_registry()
    if labels is not None:
        reg.setdefault("models", {}).setdefault(model_key, {})["labels"] = list(labels)
        registry_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")
        info["labels"] = list(labels)
    return info


def load_model(model_key: str) -> ModelSpec:
    spec = get_model_spec(model_key)
    if spec is None:
        raise AppError(ERR_MODEL_MISSING, f"Model {model_key!r} not registered.")
    return spec


def validate_model_registry() -> dict[str, Any]:
    """Check every registered model file is present."""
    from .models import load_registry
    reg = load_registry()
    missing: list[str] = []
    present: list[str] = []
    for key in reg.get("models", {}):
        info = _model_status(key)
        if info.get("exists"):
            present.append(key)
        else:
            missing.append(key)
    return {
        "total": len(reg.get("models", {})),
        "present": present,
        "missing": missing,
        "ok": len(missing) == 0,
    }
