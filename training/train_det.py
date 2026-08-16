"""YOLO detection trainer for the structural, solar, and corrosion models.

A thin, deliberate wrapper over ultralytics: it owns the config, resolves the
prepared ``data.yaml``, forces deterministic seeding, and -- the part that matters --
writes a metrics card carrying the real validation numbers, so nothing downstream
has to take a model's quality on trust.

Ultralytics already handles resumption (``resume=True`` picks up ``last.pt``), which
is what makes an interruptible Vast.ai instance usable for these runs.

    python -m training.train_det --config training/configs/structural_yolo11x.yaml
    python -m training.train_det --config <cfg> --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_seg import _read_config  # noqa: E402  (shared config reader)


@dataclass
class DetConfig:
    name: str = "structural_yolo11x"
    data_root: str = "training/data/prepared/structural_det"
    model_id: str = "yolo11x.pt"
    image_size: int = 1024
    batch_size: int = 8
    epochs: int = 120
    lr: float = 0.01
    optimizer: str = "auto"
    patience: int = 25
    seed: int = 1337
    workers: int = 8
    output_dir: str = "training/runs"
    # Warm-start weights. Empty means start from model_id's pretrained checkpoint.
    init_weights: str = ""
    # Augmentation. Mosaic and copy-paste matter most for the rare CODEBRIM classes.
    mosaic: float = 1.0
    close_mosaic: int = 15
    copy_paste: float = 0.3
    mixup: float = 0.1
    degrees: float = 10.0
    perspective: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    cos_lr: bool = True
    amp: bool = True

    @classmethod
    def load(cls, path: Path) -> "DetConfig":
        raw = _read_config(path)
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def train(config: DetConfig, *, resume: bool = False) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("training.train_det needs ultralytics: pip install ultralytics") from exc

    data_yaml = REPO_ROOT / config.data_root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"No prepared detection data at {data_yaml}.\n"
            f"Run: python -m training.datasets.prepare {Path(config.data_root).name}\n"
            "Roboflow sets need ROBOFLOW_API_KEY to download first."
        )

    run_root = REPO_ROOT / config.output_dir
    # Two different ways to carry on, and confusing them wastes rented time.
    #
    # --resume finishes an interrupted run: ultralytics reloads last.pt with its
    # optimizer and epoch index and trains up to the ORIGINAL epoch count. It cannot
    # take a finished 60-epoch run to 100 -- it will simply report that training is
    # already complete.
    #
    # --weights warm-starts a fresh run from a previous best.pt. That is what extends a
    # budget-capped run once more credit exists, and it is also how training continues
    # on a new rented box after the old one was destroyed, since the instance is gone
    # but the downloaded weights are not. The optimizer state does not survive this, so
    # the learning-rate schedule restarts; that is a real cost and the reason this is a
    # deliberate flag rather than the default.
    start_weights = config.model_id
    if getattr(config, "init_weights", ""):
        candidate = Path(config.init_weights)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not candidate.is_file():
            raise SystemExit(f"No weights to warm-start from at {candidate}.")
        start_weights = str(candidate)
        print(f"warm start from {candidate}", flush=True)

    model = YOLO(start_weights)

    results = model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        lr0=config.lr,
        optimizer=config.optimizer,
        patience=config.patience,
        seed=config.seed,
        workers=config.workers,
        project=str(run_root),
        name=config.name,
        exist_ok=True,
        resume=resume,
        cos_lr=config.cos_lr,
        amp=config.amp,
        mosaic=config.mosaic,
        close_mosaic=config.close_mosaic,
        copy_paste=config.copy_paste,
        mixup=config.mixup,
        degrees=config.degrees,
        perspective=config.perspective,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        plots=True,
        deterministic=True,
    )

    # Re-validate the best weights explicitly. `results` reflects the last epoch,
    # which is not necessarily the checkpoint that gets shipped.
    run_dir = run_root / config.name
    best_weights = run_dir / "weights" / "best.pt"
    metrics: dict[str, Any] = {}
    if best_weights.exists():
        validation = YOLO(str(best_weights)).val(
            data=str(data_yaml), imgsz=config.image_size, split="val", plots=False
        )
        box = getattr(validation, "box", None)
        if box is not None:
            # `box.maps` is pre-filled with the overall mAP and only overwritten for
            # classes the model actually scored on, so reporting it wholesale invents
            # a per-class number for classes that were never detected. Only classes
            # present in `ap_class_index` are reported; the rest are named explicitly.
            names = list(validation.names.values())
            evaluated = [int(i) for i in getattr(box, "ap_class_index", [])]
            per_class = {
                names[index]: float(box.maps[index])
                for index in evaluated
                if 0 <= index < len(names)
            }
            metrics = {
                "map50_95": float(box.map),
                "map50": float(box.map50),
                "map75": float(box.map75),
                "precision": float(box.mp),
                "recall": float(box.mr),
                "per_class_map50_95": per_class,
                "classes_without_predictions": [
                    name for index, name in enumerate(names) if index not in evaluated
                ],
            }

    summary = {
        "name": config.name,
        "checkpoint": str(best_weights),
        "metrics": metrics,
        "config": asdict(config),
        "results_dir": str(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train_det")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    # The corpus and the run directory move with the machine: Kaggle mounts inputs
    # read-only under /kaggle/input and a rented box has its own disk layout, while the
    # config's paths are repo-relative. Overriding them here keeps one config per model
    # rather than a near-duplicate per venue, which is how the two drift apart.
    parser.add_argument("--data-root", type=Path, help="Override the corpus location.")
    parser.add_argument("--output-dir", type=Path, help="Override where runs are written.")
    # Dataloader workers are a per-machine property, not a property of the model.
    # On Windows each worker is a fresh process that loads torch's CUDA DLLs, and four
    # of them exhausted the page file on an 8 GB laptop with WinError 1455 -- a failure
    # that says nothing about the config and everything about where it ran.
    parser.add_argument("--workers", type=int, help="Override dataloader worker count.")
    parser.add_argument(
        "--weights",
        dest="init_weights",
        help=(
            "Warm-start from an existing best.pt. Use this to extend a run past its "
            "configured epochs, or to carry on from a destroyed instance's weights. "
            "Use --resume instead to finish a run that was interrupted."
        ),
    )
    args = parser.parse_args(argv)

    if args.resume and args.init_weights:
        print(
            "--resume and --weights do different things and cannot be combined: "
            "--resume finishes an interrupted run up to its original epoch count, "
            "--weights starts a new run from existing weights."
        )
        return 2

    config = DetConfig.load(args.config)
    for key in ("epochs", "batch_size", "image_size", "data_root", "output_dir",
                "init_weights", "workers"):
        value = getattr(args, key)
        if value is None:
            continue
        # argparse hands back a Path for these, but the dataclass field is a str and the
        # summary is written as JSON. Assigning the Path straight through survives all of
        # training and then raises "Object of type PosixPath is not JSON serializable"
        # while writing the metrics card -- after the run, on a machine billed by the
        # hour, with the weights saved but no summary and no completion marker.
        if isinstance(value, Path):
            value = str(value)
        setattr(config, key, value)

    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
