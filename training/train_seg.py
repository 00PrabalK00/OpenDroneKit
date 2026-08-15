"""Binary crack segmentation trainer.

Trains a SegFormer (or any `transformers` semantic-segmentation checkpoint) on the
corpus produced by ``training.datasets.prepare``. Written to run unchanged on a
laptop GPU, a Kaggle P100, and a rented 4090, because the same script has to survive
all three: the only thing that changes between them is the config.

Two properties matter more than raw throughput here:

*Resumability.* Vast.ai interruptible instances get reclaimed without warning and
Kaggle sessions are capped at nine hours. Every epoch writes ``last.pt`` with model,
optimizer, scheduler, scaler, and epoch index, so ``--resume`` continues mid-run
rather than restarting.

*Honest metrics.* Crack pixels are 1-6% of a typical image, so pixel accuracy is
meaningless -- a model predicting all-background scores 95%+. Selection is on
foreground IoU, and the metric block reports IoU, Dice, precision, and recall
together so a degenerate model is obvious.

    python -m training.train_seg --config training/configs/crack_segformer_b5.yaml
    python -m training.train_seg --config <cfg> --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass
class SegConfig:
    """Everything a run needs, so a checkpoint can be traced back to its settings."""

    name: str = "crack_segformer_b5"
    data_root: str = "training/data/prepared/crack_seg"
    model_id: str = "nvidia/mit-b5"
    image_size: int = 1024
    batch_size: int = 2
    grad_accum: int = 8          # effective batch = batch_size * grad_accum
    epochs: int = 40
    lr: float = 6e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    num_workers: int = 4
    seed: int = 1337
    amp_dtype: str = "bf16"      # bf16 | fp16 | off
    ema_decay: float = 0.999
    dice_weight: float = 0.5     # loss = bce * (1 - w) + dice * w
    pos_weight: float = 4.0      # counters the 1-6% foreground fraction
    output_dir: str = "training/runs"
    max_train_samples: int = 0   # 0 means all; used for smoke tests
    val_interval: int = 1
    early_stop_patience: int = 0  # 0 disables

    @classmethod
    def load(cls, path: Path) -> "SegConfig":
        raw = _read_config(path)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def _read_config(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON config without requiring PyYAML for the simple case."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    # Flat `key: value` fallback so the configs stay readable without a yaml dep.
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        value = value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            parsed[key.strip()] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).replace("e-", "", 1).isdigit():
            parsed[key.strip()] = float(value) if ("." in value or "e-" in value) else int(value)
        else:
            parsed[key.strip()] = value
    return parsed


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CrackSegDataset(Dataset):
    """Image/mask pairs from a prepared split directory.

    Augmentation uses albumentations when available and falls back to flips and
    90-degree rotations otherwise, so a missing optional dependency degrades the
    augmentation strength rather than stopping the run.
    """

    def __init__(self, root: Path, split: str, image_size: int, *, train: bool, limit: int = 0):
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        if not self.image_dir.is_dir():
            raise SystemExit(
                f"No prepared data at {self.image_dir}. "
                f"Run: python -m training.datasets.prepare crack_seg"
            )
        self.items = sorted(p for p in self.image_dir.glob("*.png"))
        if limit:
            self.items = self.items[:limit]
        self.image_size = image_size
        self.train = train
        self.transform = self._build_transform()

    def _build_transform(self):
        try:
            import albumentations as A
        except ImportError:
            return None
        if not self.train:
            return A.Compose([A.Resize(self.image_size, self.image_size)])
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size), scale=(0.5, 1.0), ratio=(0.8, 1.25)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                A.HueSaturationValue(10, 20, 15, p=0.3),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.GaussNoise()], p=0.25),
                A.CoarseDropout(p=0.15),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path = self.items[index]
        mask_path = self.mask_dir / image_path.name
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if mask_path.exists():
            mask = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image, mask = _resize_pair(image, mask, self.image_size)
            if self.train and random.random() < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()

        tensor = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(tensor.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask.astype(np.float32)),
        )


def _resize_pair(image: np.ndarray, mask: np.ndarray, size: int):
    from PIL import Image

    resized_image = np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))
    # Nearest for the mask: a binary label must not acquire interpolated values.
    resized_mask = np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))
    return resized_image, resized_mask


# --------------------------------------------------------------------------
# model, loss, metrics
# --------------------------------------------------------------------------


def build_model(model_id: str):
    """Build a single-logit semantic segmentation head on the requested backbone."""
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )


def forward_logits(model, images: torch.Tensor) -> torch.Tensor:
    """Run the model and upsample logits back to input resolution.

    SegFormer emits logits at 1/4 input resolution; comparing them to a full-size
    mask without this step silently trains against a shape mismatch.
    """
    output = model(pixel_values=images)
    logits = output.logits
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
    return logits.squeeze(1)


def seg_loss(logits: torch.Tensor, targets: torch.Tensor, *, dice_weight: float, pos_weight: float):
    """Weighted BCE plus soft Dice.

    BCE alone under-segments thin cracks because the background term dominates the
    gradient; Dice alone is unstable on images with no foreground at all. The pair
    is the standard remedy.
    """
    weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2))
    union = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return bce * (1.0 - dice_weight) + dice * dice_weight


@dataclass
class MetricAccumulator:
    """Dataset-level confusion counts, accumulated over batches.

    Counts are summed across the whole split rather than averaged per batch: a
    per-batch mean of IoU over-weights images that contain almost no crack.
    """

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        predicted_bool = predicted > 0.5
        target_bool = target > 0.5
        self.true_positive += int((predicted_bool & target_bool).sum())
        self.false_positive += int((predicted_bool & ~target_bool).sum())
        self.false_negative += int((~predicted_bool & target_bool).sum())
        self.true_negative += int((~predicted_bool & ~target_bool).sum())

    def summary(self) -> dict[str, float]:
        tp, fp, fn, tn = (
            self.true_positive,
            self.false_positive,
            self.false_negative,
            self.true_negative,
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        total = tp + fp + fn + tn
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "pixel_accuracy": (tp + tn) / total if total else 0.0,
            "foreground_fraction": (tp + fn) / total if total else 0.0,
        }


class ModelEMA:
    """Exponential moving average of weights, evaluated instead of the raw model.

    EMA weights are consistently better on this task and cost one extra copy of the
    parameters, which is affordable even on the 8GB laptop.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach().float(), alpha=1 - self.decay)

    def copy_into(self, model) -> dict:
        """Swap EMA weights in, returning the originals so they can be restored."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(
            {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
             for k, v in model.state_dict().items()},
            strict=False,
        )
        return backup


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(requested: str, device: torch.device):
    """Pick an autocast dtype the current device actually supports."""
    if device.type != "cuda" or requested == "off":
        return None, False
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    # fp16 needs a gradient scaler; bf16 does not.
    return torch.float16, True


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = forward_logits(model, images)
        metrics.update(torch.sigmoid(logits.float()), targets)
    return metrics.summary()


def train(config: SegConfig, *, resume: bool = False) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = REPO_ROOT / config.output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = REPO_ROOT / config.data_root

    train_set = CrackSegDataset(
        data_root, "train", config.image_size, train=True, limit=config.max_train_samples
    )
    val_set = CrackSegDataset(data_root, "val", config.image_size, train=False)
    print(f"train={len(train_set)} val={len(val_set)} device={device}")

    # Windows spawns worker processes rather than forking, which makes a high worker
    # count slower than it is on Linux for this dataset size.
    workers = config.num_workers if os.name != "nt" else min(config.num_workers, 2)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, config.batch_size), shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )

    model = build_model(config.model_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    steps_per_epoch = max(1, len(train_loader) // config.grad_accum)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(total_steps * config.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_dtype, needs_scaler = resolve_amp(config.amp_dtype, device)
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)
    ema = ModelEMA(model, config.ema_decay) if config.ema_decay > 0 else None

    start_epoch = 0
    best_iou = 0.0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "last.pt"

    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and needs_scaler:
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        start_epoch = int(state.get("epoch", 0)) + 1
        best_iou = float(state.get("best_iou", 0.0))
        history = state.get("history", [])
        print(f"Resumed from epoch {start_epoch} (best IoU {best_iou:.4f})")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for index, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = forward_logits(model, images)
                loss = seg_loss(
                    logits.float(), targets,
                    dice_weight=config.dice_weight, pos_weight=config.pos_weight,
                ) / config.grad_accum

            scaler.scale(loss).backward() if needs_scaler else loss.backward()
            running_loss += float(loss) * config.grad_accum

            if (index + 1) % config.grad_accum == 0:
                if needs_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

            if index % 50 == 0:
                done = index + 1
                print(
                    f"  epoch {epoch} [{done}/{len(train_loader)}] "
                    f"loss={running_loss / done:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        train_loss = running_loss / max(1, len(train_loader))
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": round(time.time() - epoch_start, 1),
        }

        if (epoch + 1) % config.val_interval == 0 or epoch == config.epochs - 1:
            backup = ema.copy_into(model) if ema is not None else None
            metrics = evaluate(model, val_loader, device, autocast_dtype)
            if backup is not None:
                model.load_state_dict(backup, strict=False)
            record.update({f"val_{k}": round(v, 5) for k, v in metrics.items()})

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": ema.shadow if ema is not None else model.state_dict(),
                        "config": asdict(config),
                        "metrics": metrics,
                        "epoch": epoch,
                    },
                    run_dir / "best.pt",
                )
                record["best"] = True
            else:
                epochs_without_improvement += 1

        history.append(record)
        print(f"epoch {epoch}: {json.dumps(record)}", flush=True)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if needs_scaler else None,
                "ema": ema.shadow if ema is not None else None,
                "epoch": epoch,
                "best_iou": best_iou,
                "history": history,
                "config": asdict(config),
            },
            checkpoint_path,
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if config.early_stop_patience and epochs_without_improvement >= config.early_stop_patience:
            print(f"Early stop: no IoU improvement in {epochs_without_improvement} evals.")
            break

    summary = {
        "name": config.name,
        "best_val_iou": best_iou,
        "epochs_run": len(history),
        "checkpoint": str(run_dir / "best.pt"),
        "config": asdict(config),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train_seg")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt.")
    parser.add_argument("--epochs", type=int, help="Override the config epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override the config batch size.")
    parser.add_argument("--image-size", type=int, help="Override the config image size.")
    parser.add_argument("--max-train-samples", type=int, help="Cap training samples (smoke test).")
    # See train_det.main: the corpus and run directory move with the machine, the
    # config should not have to.
    parser.add_argument("--data-root", type=Path, help="Override the corpus location.")
    parser.add_argument("--output-dir", type=Path, help="Override where runs are written.")
    args = parser.parse_args(argv)

    config = SegConfig.load(args.config)
    for key in ("epochs", "batch_size", "image_size", "max_train_samples",
                "data_root", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)

    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
