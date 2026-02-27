"""Model registry and lightweight model management utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str
    path: str
    labels: list[str]
    score_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: int = 640
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "path": self.path,
            "labels": list(self.labels),
            "score_threshold": float(self.score_threshold),
            "iou_threshold": float(self.iou_threshold),
            "input_size": int(self.input_size),
            "description": self.description,
        }


def toolkit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return toolkit_root().parent


def models_root() -> Path:
    return toolkit_root() / "models"


def registry_path() -> Path:
    return models_root() / "model_registry.json"


def _default_registry() -> dict[str, Any]:
    return {
        "version": 1,
        "models": {
            "structural_multiclass_detector": {
                "kind": "onnx_yolo",
                "path": "structural/structural_defect_yolov8n.onnx",
                "labels": [
                    "crack",
                    "spalling",
                    "corrosion",
                    "delamination",
                    "rebar_exposure",
                    "efflorescence",
                    "moisture_intrusion",
                ],
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
                "description": "Primary multi-class structural defect detector.",
            },
            "structural_codebrim_detector": {
                "kind": "onnx_yolo",
                "path": "structural/codebrim_concrete_damage.onnx",
                "labels": [
                    "crack",
                    "spalling",
                    "corrosion",
                    "efflorescence",
                    "rebar_exposure",
                ],
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
                "description": "CODEBRIM-style structural defects baseline (no delamination/moisture classes).",
            },
            "structural_concrete_yolov8_baseline": {
                "kind": "onnx_yolo",
                "path": "structural/concrete_defects_yolov8.onnx",
                "labels": [
                    "crack",
                    "spalling",
                    "rebar_exposure",
                ],
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
                "description": "YOLOv8 concrete defects quick-start baseline.",
            },
            "solar_defect_detector": {
                "kind": "onnx_yolo",
                "path": "solar/solar_defect_yolov8n.onnx",
                "labels": [
                    "hotspot",
                    "cell_crack",
                    "soiling",
                    "delamination",
                    "string_fault",
                ],
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
                "description": "Primary solar defect detector for RGB/thermal panel inspections.",
            },
            "solar_pv_multidefect_detector": {
                "kind": "onnx_yolo",
                "path": "solar/pv_multidefect_gbh_yolov5.onnx",
                "labels": [
                    "hotspot",
                    "cell_crack",
                    "soiling",
                    "delamination",
                    "string_fault",
                ],
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
                "description": "PV-Multi-Defect baseline mapped to toolkit solar label schema.",
            },
            "legacy_swin_unet": {
                "kind": "pytorch_checkpoint",
                "path": "legacy/swin_unet_best_model.pth",
                "labels": ["crack"],
                "score_threshold": 0.5,
                "iou_threshold": 0.0,
                "input_size": 512,
                "description": "Legacy Swin-UNet crack segmentation checkpoint.",
            },
            "legacy_crack_segmentation_unet": {
                "kind": "pytorch_checkpoint",
                "path": "legacy/crack_segmentation_unet.pth",
                "labels": ["crack"],
                "score_threshold": 0.5,
                "iou_threshold": 0.0,
                "input_size": 512,
                "description": "Alternative crack segmentation checkpoint slot.",
            },
        },
    }


def ensure_model_layout() -> None:
    root = models_root()
    (root / "structural").mkdir(parents=True, exist_ok=True)
    (root / "solar").mkdir(parents=True, exist_ok=True)
    (root / "legacy").mkdir(parents=True, exist_ok=True)
    (root / "legacy" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    reg_path = registry_path()
    if not reg_path.exists():
        reg_path.write_text(json.dumps(_default_registry(), indent=2), encoding="utf-8")


def load_registry() -> dict[str, Any]:
    ensure_model_layout()
    reg_path = registry_path()
    try:
        payload = json.loads(reg_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Registry must be a JSON object.")
    except Exception:
        payload = _default_registry()
        reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if "models" not in payload or not isinstance(payload.get("models"), dict):
        payload = _default_registry()
        reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def get_model_spec(model_key: str) -> ModelSpec | None:
    key = str(model_key or "").strip()
    if not key:
        return None
    payload = load_registry()
    models = payload.get("models", {})
    row = models.get(key)
    if not isinstance(row, dict):
        return None
    labels = row.get("labels", [])
    if not isinstance(labels, list):
        labels = []
    return ModelSpec(
        key=key,
        kind=str(row.get("kind", "unknown")),
        path=str(row.get("path", "")),
        labels=[str(v) for v in labels],
        score_threshold=float(row.get("score_threshold", 0.25)),
        iou_threshold=float(row.get("iou_threshold", 0.45)),
        input_size=int(row.get("input_size", 640)),
        description=str(row.get("description", "")),
    )


def resolve_model_path(model_key: str) -> Path | None:
    spec = get_model_spec(model_key)
    if spec is None:
        return None
    if not spec.path:
        return None
    path = Path(spec.path)
    if not path.is_absolute():
        path = models_root() / path
    return path.resolve()


def model_status(model_key: str) -> dict[str, Any]:
    spec = get_model_spec(model_key)
    if spec is None:
        return {"key": model_key, "exists": False, "error": "unknown_model_key"}
    resolved = resolve_model_path(model_key)
    if resolved is None:
        return {"key": model_key, "exists": False, "error": "invalid_model_path", "spec": spec.to_dict()}
    return {
        "key": model_key,
        "exists": resolved.exists(),
        "path": str(resolved),
        "spec": spec.to_dict(),
    }


def _legacy_checkpoint_candidates() -> list[Path]:
    ws = workspace_root()
    return [
        models_root() / "legacy" / "swin_unet_best_model.pth",
        models_root() / "legacy" / "checkpoints" / "best_model.pth",
        ws / "CConCrack_SwinUNet_Final" / "checkpoints" / "best_model.pth",
        ws / "CConCrack_SwinUNet_Final" / "checkpoints" / "latest_checkpoint.pth",
        ws / "models" / "legacy" / "swin_unet_best_model.pth",
    ]


def sync_legacy_checkpoint_archive(source_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Copy legacy checkpoint files into final_toolkit/models/legacy/checkpoints.

    This is intended for one-time consolidation of historical training outputs.
    Runtime inference still uses the registry-selected files in models/.
    """
    ensure_model_layout()
    src = (
        Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else (workspace_root() / "CConCrack_SwinUNet_Final" / "checkpoints").resolve()
    )
    dst = (models_root() / "legacy" / "checkpoints").resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists() or not src.is_dir():
        return {
            "source": str(src),
            "target_dir": str(dst),
            "copied": 0,
            "skipped": 0,
            "reason": "source_missing",
        }

    copied = 0
    skipped = 0
    moved_files: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".pth", ".pt", ".onnx", ".ckpt"}:
            continue
        target = dst / entry.name
        if target.exists():
            try:
                if target.stat().st_size == entry.stat().st_size:
                    skipped += 1
                    continue
            except Exception:
                pass
        shutil.copy2(entry, target)
        copied += 1
        moved_files.append(str(target))

    return {
        "source": str(src),
        "target_dir": str(dst),
        "copied": copied,
        "skipped": skipped,
        "files": moved_files,
    }


def migrate_legacy_models() -> dict[str, Any]:
    """
    Move/copy legacy model assets into final_toolkit/models layout.

    Current migration target:
    - CConCrack_SwinUNet_Final/checkpoints/best_model.pth
      -> final_toolkit/models/legacy/swin_unet_best_model.pth
    """
    ensure_model_layout()
    target = models_root() / "legacy" / "swin_unet_best_model.pth"
    if target.exists():
        return {"migrated": False, "target": str(target), "reason": "already_present"}

    for source in _legacy_checkpoint_candidates():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return {
                "migrated": True,
                "source": str(source),
                "target": str(target),
            }
    return {
        "migrated": False,
        "target": str(target),
        "reason": "legacy_source_not_found",
    }
