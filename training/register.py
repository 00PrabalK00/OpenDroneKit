"""Install an exported model into `models/` and record what it truly is.

This is the step that replaces the retracted manifest. The previous provenance file
listed four ONNX targets sharing one sha256 -- a single concrete-defect model copied
under solar and CODEBRIM filenames -- so registration here refuses to record anything
it has not itself hashed on disk, and refuses to install an export whose parity check
did not pass.

Registering a model updates two files:

``models/model_registry.json``
    What the pipeline looks up at inference time: path, labels, thresholds.

``models/manifests/model_provenance.json``
    What the artifact actually is: sha256, source run, training data, licence,
    and the validation metrics it earned.

    python -m training.register --run training/runs/crack_segformer_b5 \\
        --key crack_segmentation
    python -m training.register --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
PROVENANCE_PATH = MODELS_DIR / "manifests" / "model_provenance.json"
METRICS_DIR = MODELS_DIR / "metrics"

# Where each registry key's weights live, relative to models/.
KEY_SUBDIR = {
    "crack_segmentation": "structural",
    "structural_multiclass_detector": "structural",
    "structural_codebrim_detector": "structural",
    "solar_pv_multidefect_detector": "solar",
    "metal_corrosion_detector": "solar",
    "corrosion_severity_segmentation": "structural",
}

# Which prepared task fed each key, so provenance can name the training data.
KEY_TASK = {
    "crack_segmentation": "crack_seg",
    "structural_multiclass_detector": "structural_det",
    "structural_codebrim_detector": "structural_det",
    "solar_pv_multidefect_detector": "solar_det",
    "metal_corrosion_detector": "corrosion_det",
    "corrosion_severity_segmentation": "corrosion_seg",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _training_data_provenance(task_name: str | None) -> dict[str, Any]:
    """Recover which datasets and licences went into the model, for the manifest."""
    if not task_name:
        return {}
    manifest = _load_json(REPO_ROOT / "training" / "data" / "prepared" / "manifest.json", {})
    entry = (manifest.get("tasks") or {}).get(task_name)
    if not entry:
        return {"task": task_name, "note": "No prepared-data manifest found."}
    return {
        "task": task_name,
        "sources": entry.get("sources", []),
        "licenses": entry.get("licenses", {}),
        "sample_counts": entry.get("counts", {}),
    }


def register(run_dir: Path, key: str, *, force: bool = False) -> dict[str, Any]:
    export_report = _load_json(run_dir / "export_report.json", None)
    if export_report is None:
        raise SystemExit(
            f"No export_report.json in {run_dir}. Run training.export_onnx first."
        )
    if not export_report.get("parity_ok") and not force:
        raise SystemExit(
            f"Refusing to register {key}: the ONNX parity check failed "
            f"(max abs diff {export_report.get('max_abs_diff')}). "
            "Fix the export, or pass --force if you accept the discrepancy."
        )

    source = Path(export_report["onnx_path"])
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.exists():
        raise SystemExit(f"Exported model missing: {source}")

    subdir = KEY_SUBDIR.get(key, "structural")
    destination_dir = MODELS_DIR / subdir
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics names every export `best.onnx`, so installing under the source
    # filename would have each YOLO model silently overwrite the last. The registry
    # key is unique by construction, so it names the installed file.
    destination = destination_dir / f"{key}{source.suffix}"
    shutil.copy2(source, destination)

    checksum = sha256_of(destination)
    relative = destination.relative_to(MODELS_DIR).as_posix()
    metrics = export_report.get("train_metrics", {}) or {}

    # -- registry -------------------------------------------------------
    registry = _load_json(REGISTRY_PATH, {"version": 1, "models": {}})
    existing = registry["models"].get(key, {})
    entry = {
        "kind": export_report.get("kind", "onnx_yolo"),
        "path": relative,
        "labels": export_report.get("labels") or existing.get("labels") or ["crack"],
        "score_threshold": existing.get("score_threshold", 0.25),
        "iou_threshold": existing.get("iou_threshold", 0.45),
        "input_size": export_report.get("input_size", existing.get("input_size", 640)),
        "description": existing.get("description", f"Trained in run {run_dir.name}."),
        # The digest of the file whose metrics are on record. Without it the runtime's
        # identity check reports `unrecorded` -- not a failure, but not verification
        # either, and every model registered here would arrive in that state.
        "sha256": checksum,
        "status": "installed",
    }
    registry["models"][key] = entry
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    # -- provenance -----------------------------------------------------
    provenance = _load_json(PROVENANCE_PATH, {"schema_version": 2, "files": []})
    provenance["schema_version"] = 2
    provenance["status"] = "weights_installed"
    provenance.pop("note", None)
    files = [f for f in provenance.get("files", []) if f.get("registry_key") != key]
    files.append(
        {
            "registry_key": key,
            "path": relative,
            "sha256": checksum,
            "bytes": destination.stat().st_size,
            "method": "trained_in_repo",
            "source_run": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
            "architecture": export_report.get("name"),
            "input_size": export_report.get("input_size"),
            "opset": export_report.get("opset"),
            "onnx_parity_max_abs_diff": export_report.get("max_abs_diff"),
            "opencv_dnn_loadable": export_report.get("opencv_dnn_loadable"),
            "validation_metrics": metrics,
            "training_data": _training_data_provenance(KEY_TASK.get(key)),
            "registered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    provenance["files"] = files
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    # -- metrics card ---------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    card = {
        "registry_key": key,
        "architecture": export_report.get("name"),
        "sha256": checksum,
        "input_size": export_report.get("input_size"),
        "validation_metrics": metrics,
        "training_data": _training_data_provenance(KEY_TASK.get(key)),
        "onnx_parity_max_abs_diff": export_report.get("max_abs_diff"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (METRICS_DIR / f"{key}.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    result = {
        "key": key,
        "installed": relative,
        "sha256": checksum,
        "metrics": metrics,
        "parity_ok": export_report.get("parity_ok"),
    }
    print(json.dumps(result, indent=2))
    return result


def show_status() -> int:
    """Report, per registry key, whether a real weight file is present."""
    registry = _load_json(REGISTRY_PATH, {"models": {}})
    provenance = _load_json(PROVENANCE_PATH, {"files": []})
    recorded = {f["path"]: f for f in provenance.get("files", [])}

    print(f"{'key':<38} {'present':<9} {'sha256':<18} path")
    print("-" * 100)
    for key, entry in registry.get("models", {}).items():
        relative = entry.get("path", "")
        path = MODELS_DIR / relative
        present = path.exists()
        recorded_entry = recorded.get(relative)
        if present:
            actual = sha256_of(path)
            if recorded_entry and recorded_entry.get("sha256") != actual:
                marker = "SHA MISMATCH"
            else:
                marker = actual[:16]
        else:
            marker = "-"
        print(f"{key:<38} {'yes' if present else 'no':<9} {marker:<18} {relative}")

    checksums = [f.get("sha256") for f in provenance.get("files", []) if f.get("sha256")]
    duplicates = {c for c in checksums if checksums.count(c) > 1}
    if duplicates:
        print(
            "\nWARNING: several registered models share a checksum, which means one "
            "file is installed under multiple keys:"
        )
        for checksum in duplicates:
            keys = [f["registry_key"] for f in provenance["files"] if f.get("sha256") == checksum]
            print(f"  {checksum[:16]}  {', '.join(keys)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.register")
    parser.add_argument("--run", type=Path, help="Run directory holding export_report.json.")
    parser.add_argument("--key", help="Registry key to install this model under.")
    parser.add_argument("--force", action="store_true", help="Register despite a failed parity check.")
    parser.add_argument("--list", action="store_true", help="Show installed-model status and exit.")
    args = parser.parse_args(argv)

    if args.list or not args.run:
        return show_status()
    if not args.key:
        parser.error("--key is required when registering a run.")

    run_dir = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    register(run_dir, args.key, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
