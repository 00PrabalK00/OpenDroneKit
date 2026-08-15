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
    # The digest of the file whose metrics were measured. Empty means nobody recorded
    # one, which is a different statement from the file having changed.
    sha256: str = ""
    # "installed" when weights exist and were measured; "awaiting_weights" when the entry
    # is a training or routing target with nothing behind it yet. A registry whose whole
    # purpose is to say what is real must not list the second as though it were the first.
    status: str = "installed"
    status_note: str = ""

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
            "sha256": self.sha256,
            "status": self.status,
            "status_note": self.status_note,
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
        sha256=str(row.get("sha256", "") or "").strip().lower(),
        status=str(row.get("status", "installed") or "installed"),
        status_note=str(row.get("status_note", "")),
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


# ── Model identity ────────────────────────────────────────────────────────────

# Digests are cached against (path, size, mtime) so a repeated detection does not
# re-read a 200 MB graph, while a model replaced on disk still produces a new identity.
_DIGEST_CACHE: dict[tuple[str, int, float], str] = {}


def model_identity(model_key: str) -> dict[str, str]:
    """Which model file would answer for this key, and its digest.

    The digest is computed from the installed file rather than read from the registry.
    A recorded hash describes what was installed at some point; hashing the file
    describes what is about to run, and those differ exactly when it matters -- after a
    model has been replaced, retrained, or copied in by hand.
    """
    import hashlib

    info = model_status(model_key)
    path_str = str(info.get("path", "") or "")
    if not info.get("exists") or not path_str:
        return {"model_key": model_key, "model_sha256": "", "model_path": ""}

    path = Path(path_str)
    try:
        stat = path.stat()
    except OSError:
        return {"model_key": model_key, "model_sha256": "", "model_path": path_str}

    cache_key = (path_str, stat.st_size, stat.st_mtime)
    digest = _DIGEST_CACHE.get(cache_key)
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        _DIGEST_CACHE[cache_key] = digest

    return {"model_key": model_key, "model_sha256": digest, "model_path": path_str}


def verify_model_identity(model_key: str) -> dict[str, Any]:
    """Compare the installed file against the digest the registry recorded for it.

    The registry's ``sha256`` is the digest of the file whose metrics were measured. The
    runtime digest is of the file about to answer. They diverge exactly when a model has
    been retrained, replaced or copied in by hand -- at which point the metrics on record
    describe a different file, and any report quoting them is quoting the wrong model.

    Four outcomes, deliberately distinct:

    ``verified``
        Recorded and installed digests agree.
    ``unrecorded``
        The file is installed but no digest was recorded. Not a failure; a gap. Saying
        "verified" here would be a claim nobody has earned.
    ``mismatch``
        A digest was recorded and the installed file is not it.
    ``missing``
        Nothing is installed at the registered path.
    """
    spec = get_model_spec(model_key)
    if spec is None:
        return {"model_key": model_key, "status": "unknown_model_key", "verified": False}

    identity = model_identity(model_key)
    actual = identity.get("model_sha256", "")
    expected = spec.sha256

    if not actual and spec.status == "awaiting_weights":
        # Declared empty rather than found empty. The distinction matters: one is a plan,
        # the other is a model that has gone.
        status = "awaiting_weights"
        detail = spec.status_note or (
            f"{model_key} is a registered target with no weights trained or obtained "
            "yet. It is not an available model."
        )
    elif not actual:
        status = "missing"
        detail = (
            f"No file is installed at the registered path for {model_key}, so there is "
            "nothing to verify and nothing to run."
        )
    elif not expected:
        status = "unrecorded"
        detail = (
            f"{model_key} has no recorded digest, so the installed file cannot be shown "
            "to be the one its metrics were measured on. Record the digest when the "
            "model is registered."
        )
    elif expected == actual:
        status = "verified"
        detail = f"{model_key} matches the digest recorded when it was registered."
    else:
        status = "mismatch"
        detail = (
            f"{model_key} on disk is not the file that was registered. Recorded "
            f"{expected[:16]}..., found {actual[:16]}.... Any accuracy figure on record "
            "for this key describes a different file."
        )

    return {
        "model_key": model_key,
        "status": status,
        "verified": status == "verified",
        "expected_sha256": expected,
        "actual_sha256": actual,
        "model_path": identity.get("model_path", ""),
        "detail": detail,
    }


def verify_all_models() -> dict[str, Any]:
    """Verification status for every registered model, worst news first."""
    payload = load_registry()
    rows = [verify_model_identity(key) for key in sorted(payload.get("models", {}))]
    order = {"mismatch": 0, "missing": 1, "unrecorded": 2, "awaiting_weights": 3,
             "verified": 4}
    rows.sort(key=lambda row: order.get(row["status"], 4))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "models": rows,
        "counts": counts,
        "any_mismatch": any(row["status"] == "mismatch" for row in rows),
        # A registered key with no weights is not an available model, and anything that
        # counts models should count these separately or not at all.
        "available": [row["model_key"] for row in rows if row["status"] == "verified"],
        "awaiting_weights": [row["model_key"] for row in rows
                             if row["status"] == "awaiting_weights"],
    }
