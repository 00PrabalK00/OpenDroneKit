"""Train an image classifier over a prepared folder-classification corpus.

The corpora this serves are all severely imbalanced, and that is the whole design
problem. The infrared solar set is 10,000 normal modules against 249 hot spots: a model
that answers "No-Anomaly" to everything scores 50 per cent accuracy on it, tops the
leaderboard, and finds nothing. Accuracy is therefore not reported as the headline
figure and is not what selects the best checkpoint. Balanced accuracy -- the mean of the
per-class recalls -- is, because it goes to zero the moment a class is being ignored.

Two mechanisms counter the imbalance, both stated in the summary so a reader knows which
was used: inverse-frequency class weights in the loss, and optionally a balanced sampler
that draws each class equally often per epoch. Weighting alone is usually enough; the
sampler helps when the rare classes are very rare, at the cost of seeing the common ones
less.

Per-class recall is always written out. A single number cannot tell you that hot spots
are being missed while vegetation is detected perfectly, and on this corpus that is
exactly the failure that matters: the rare classes are the defects.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class ClsConfig:
    """Everything a run needs, so a checkpoint can be traced back to its settings."""

    name: str = "solar_thermal_cls"
    data_root: str = "training/data/prepared/solar_thermal_cls"
    model_id: str = "resnet18"          # torchvision architecture name
    pretrained: bool = True
    image_size: int = 96                # IR crops are 24x40; upsampled for the backbone
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    num_workers: int = 4
    seed: int = 1337
    amp_dtype: str = "bf16"             # bf16 | fp16 | off
    class_weighting: bool = True        # inverse-frequency weights in the loss
    balanced_sampler: bool = False      # draw each class equally often per epoch
    label_smoothing: float = 0.0
    output_dir: str = "training/runs"
    max_train_samples: int = 0          # 0 means all; used for smoke tests
    early_stop_patience: int = 0        # 0 disables

    @classmethod
    def load(cls, path: Path) -> "ClsConfig":
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            try:
                import yaml

                raw = yaml.safe_load(text) or {}
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise SystemExit(
                    "Reading a YAML config needs PyYAML. Install it, or pass a .json config."
                ) from exc
        raw = {k: v for k, v in raw.items() if not str(k).startswith("#")}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(requested: str, device: torch.device):
    if device.type != "cuda" or requested == "off":
        return None
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_transforms(size: int) -> tuple[Any, Any]:
    """Light augmentation only.

    These are thermal module crops, not photographs: a vertical flip is a plausible
    module orientation, but a colour jitter would alter the very intensities the class
    depends on, and a rotation would introduce interpolated temperatures that were never
    measured.
    """
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalise,
    ])
    evaluate = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalise,
    ])
    return train, evaluate


def build_model(model_id: str, class_count: int, *, pretrained: bool):
    from torchvision import models

    factory = getattr(models, model_id, None)
    if factory is None:
        raise SystemExit(
            f"Unknown torchvision architecture {model_id!r}. Use a name from "
            "torchvision.models, such as resnet18, resnet50 or efficientnet_b0."
        )
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)

    # Replace whichever head this architecture uses.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, class_count)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, class_count)
        else:
            last = classifier[-1]
            classifier[-1] = nn.Linear(last.in_features, class_count)
    else:
        raise SystemExit(f"{model_id} has no recognised classification head to replace.")
    return model


def class_counts(dataset: ImageFolder, class_count: int) -> np.ndarray:
    counts = np.zeros(class_count, dtype=np.int64)
    for _, target in dataset.samples:
        counts[target] += 1
    return counts


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype, class_count: int) -> dict[str, Any]:
    """Per-class recall and precision, plus the balanced accuracy that selects checkpoints."""
    model.eval()
    confusion = np.zeros((class_count, class_count), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(images)
        else:
            logits = model(images)
        predicted = logits.float().argmax(dim=1).cpu().numpy()
        for true, pred in zip(targets.numpy(), predicted):
            confusion[true, pred] += 1

    support = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
    hits = np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, hits / np.maximum(support, 1), np.nan)
        precision = np.where(predicted_totals > 0, hits / np.maximum(predicted_totals, 1), np.nan)
    observed = ~np.isnan(recall)
    balanced = float(np.mean(recall[observed])) if observed.any() else 0.0
    return {
        "accuracy": float(hits.sum() / max(1, confusion.sum())),
        "balanced_accuracy": balanced,
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "support": support.tolist(),
        "confusion": confusion.tolist(),
    }


def train(config: ClsConfig, *, resume: bool = False) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = resolve_amp(config.amp_dtype, device)

    root = Path(config.data_root)
    train_tf, eval_tf = build_transforms(config.image_size)
    train_set = ImageFolder(root / "train", transform=train_tf)
    val_set = ImageFolder(root / "val", transform=eval_tf)
    if train_set.classes != val_set.classes:
        raise SystemExit(
            "Train and val carry different classes, so a per-class metric would compare "
            f"different things: {train_set.classes} against {val_set.classes}."
        )
    classes = train_set.classes
    class_count = len(classes)
    counts = class_counts(train_set, class_count)

    if config.max_train_samples:
        keep = min(config.max_train_samples, len(train_set))
        train_set = torch.utils.data.Subset(train_set, list(range(keep)))

    sampler = None
    shuffle = True
    if config.balanced_sampler:
        weights_per_class = 1.0 / np.maximum(counts, 1)
        sample_weights = [float(weights_per_class[t]) for _, t in
                          (train_set.dataset.samples if isinstance(train_set, torch.utils.data.Subset)
                           else train_set.samples)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        shuffle = False

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=config.num_workers,
                              pin_memory=device.type == "cuda", drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=device.type == "cuda")

    model = build_model(config.model_id, class_count, pretrained=config.pretrained).to(device)

    weight_tensor = None
    if config.class_weighting:
        # Inverse frequency, normalised so the mean weight is 1 and the learning rate
        # keeps its meaning across corpora of different balance.
        inverse = counts.sum() / (class_count * np.maximum(counts, 1))
        inverse = inverse / inverse.mean()
        weight_tensor = torch.tensor(inverse, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor,
                                    label_smoothing=config.label_smoothing)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)

    def lr_at(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=autocast_dtype == torch.float16)

    run_dir = Path(config.output_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    start_epoch = 1
    best_balanced = 0.0
    history: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_balanced = float(state.get("best_balanced_accuracy", 0.0))
        history = list(state.get("history", []))
        print(f"Resumed {config.name} from epoch {state['epoch']}.")

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss = criterion(model(images), targets)
            else:
                loss = criterion(model(images), targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
            scheduler.step()

            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)

        metrics = evaluate(model, val_loader, device, autocast_dtype, class_count)
        record = {
            "epoch": epoch,
            "train_loss": round(running / max(1, seen), 5),
            "val_accuracy": round(metrics["accuracy"], 5),
            "val_balanced_accuracy": round(metrics["balanced_accuracy"], 5),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(record)
        print(f"epoch {epoch}: {json.dumps(record)}", flush=True)

        improved = metrics["balanced_accuracy"] > best_balanced
        if improved:
            best_balanced = metrics["balanced_accuracy"]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "classes": classes,
                        "config": asdict(config), "metrics": metrics}, best_path)
        else:
            epochs_without_improvement += 1

        torch.save({
            "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_balanced_accuracy": best_balanced, "history": history,
            "classes": classes, "config": asdict(config),
        }, checkpoint_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if config.early_stop_patience and epochs_without_improvement >= config.early_stop_patience:
            print(f"Early stop: no balanced-accuracy improvement in "
                  f"{epochs_without_improvement} evals.")
            break

    final = evaluate(model, val_loader, device, autocast_dtype, class_count)
    per_class = {
        name: {
            "recall": None if math.isnan(final["per_class_recall"][i]) else round(final["per_class_recall"][i], 4),
            "precision": None if math.isnan(final["per_class_precision"][i]) else round(final["per_class_precision"][i], 4),
            "val_support": final["support"][i],
            "train_instances": int(counts[i]),
        }
        for i, name in enumerate(classes)
    }
    never_recalled = [name for name, row in per_class.items()
                      if row["val_support"] and not row["recall"]]

    summary = {
        "name": config.name,
        "classes": classes,
        "best_val_balanced_accuracy": best_balanced,
        "final_val_accuracy": final["accuracy"],
        "final_val_balanced_accuracy": final["balanced_accuracy"],
        "per_class": per_class,
        "classes_never_recalled": never_recalled,
        "imbalance_handling": {
            "class_weighting": config.class_weighting,
            "balanced_sampler": config.balanced_sampler,
            "train_instances_per_class": {name: int(counts[i]) for i, name in enumerate(classes)},
        },
        "epochs_run": len(history),
        "checkpoint": str(best_path),
        "config": asdict(config),
        "reading_note": (
            "Balanced accuracy is the mean of per-class recalls and is what selected the "
            "best checkpoint. Plain accuracy is reported alongside it and is misleading "
            "on this corpus: answering with the majority class alone scores highly on it "
            "while finding none of the defects."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train_cls")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    # See train_det.main: the corpus and run directory move with the machine, the
    # config should not have to.
    parser.add_argument("--data-root", type=Path, help="Override the corpus location.")
    parser.add_argument("--output-dir", type=Path, help="Override where runs are written.")
    args = parser.parse_args(argv)

    config = ClsConfig.load(args.config)
    for key in ("epochs", "batch_size", "image_size", "data_root", "output_dir"):
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
