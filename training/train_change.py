"""Binary change detection over co-registered image pairs.

Every other trainer here takes one image per sample. Change detection takes two, and
that single structural difference is what left the mining pack unbuildable: MineNetCD
has been downloaded and usable for weeks, and the missing piece was never the data.

The pair is fused early -- the two images are stacked into a six-channel input and the
encoder's patch embedding is inflated to match, with the pretrained three-channel
weights halved and duplicated so the initial response to an unchanged pair is the same
as the pretrained response to one image. Late fusion with a siamese encoder is the more
common choice and may well be better, but it doubles the forward cost and this corpus is
100 sites; the honest reason for early fusion is that it fits the compute available.

Two properties are carried over from train_seg because the same failure modes apply:

*Selection on foreground IoU.* MineNetCD is about 11% changed pixels, so a model
predicting "no change" everywhere scores 89% pixel accuracy. Accuracy is reported and
explicitly labelled as misleading; IoU picks the checkpoint.

*Site-separated splits.* The two images of a site are the same ground at two dates, and
tiles from one site leak heavily into each other. Splits are by site, never by tile.

    python -m training.train_change --config training/configs/mining_change_b2.yaml
    python -m training.train_change --config <cfg> --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_seg import _read_config  # noqa: E402  (shared config reader)

# Pillow refuses images past ~179 megapixels as a decompression-bomb guard. MineNetCD
# rasters reach 212 MP, and that guard is designed for untrusted uploads rather than a
# scientific corpus fetched from a known source and already on disk. Raised deliberately
# and only here, so the protection stays in place for anything user-supplied.
Image.MAX_IMAGE_PIXELS = 400_000_000

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ChangeConfig:
    name: str = "mining_change_b2"
    data_root: str = "training/data/minenetcd"
    model_id: str = "nvidia/mit-b2"
    image_size: int = 512
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 60
    lr: float = 6e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    num_workers: int = 2
    seed: int = 1337
    amp_dtype: str = "bf16"
    dice_weight: float = 0.5
    # MineNetCD is ~11% changed, far less skewed than the crack corpora, so this is
    # milder than train_seg's 4.0.
    pos_weight: float = 2.0
    output_dir: str = "training/runs"
    max_train_samples: int = 0
    early_stop_patience: int = 10
    # Held-out fractions, applied per SITE rather than per tile.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_salt: str = "opendronekit-change-v1"

    @classmethod
    def load(cls, path: Path) -> "ChangeConfig":
        raw = _read_config(path)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def site_split(site: str, config: ChangeConfig) -> str:
    """Assign a whole site to one split.

    Per-tile splitting would be catastrophic here and silently so: crops from one mine
    overlap heavily, so a tile in val is very nearly a tile in train and the reported IoU
    would describe memorisation.
    """
    digest = hashlib.sha1(f"{config.split_salt}/{site}".encode("utf-8")).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    if position < config.test_fraction:
        return "test"
    if position < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


class ChangePairDataset(Dataset):
    """Co-registered pairs with a binary reference mask.

    Reads the site directories in place rather than through a prepared corpus. A prepared
    corpus assumes one image per sample throughout, and inventing a pair-shaped variant
    of that layout to serve one dataset would spread the assumption further rather than
    contain it.
    """

    def __init__(self, root: Path, split: str, config: ChangeConfig, *, train: bool):
        self.config = config
        self.train = train
        self.image_size = config.image_size
        self.sites = [
            path for path in sorted(root.iterdir())
            if path.is_dir() and (path / "ref.png").is_file()
            and (path / "im1.jpg").is_file() and (path / "im2.jpg").is_file()
            and site_split(path.name, config) == split
        ]
        if config.max_train_samples and train:
            self.sites = self.sites[: config.max_train_samples]
        if not self.sites:
            raise SystemExit(
                f"No {split} sites under {root}. Expected directories holding "
                "im1.jpg, im2.jpg and ref.png."
            )

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, index: int):
        from PIL import Image

        site = self.sites[index]
        first = np.asarray(Image.open(site / "im1.jpg").convert("RGB"))
        second = np.asarray(Image.open(site / "im2.jpg").convert("RGB"))
        mask = (np.asarray(Image.open(site / "ref.png").convert("L")) > 127).astype(np.uint8)

        if first.shape[:2] != second.shape[:2] or first.shape[:2] != mask.shape[:2]:
            # A misaligned pair would train the model on a change that is really a
            # registration error, which is exactly the wrong lesson.
            raise SystemExit(
                f"{site.name}: im1 {first.shape[:2]}, im2 {second.shape[:2]} and ref "
                f"{mask.shape[:2]} disagree; the pair is not co-registered."
            )

        first, second, mask = self._sample_window(first, second, mask)
        stacked = np.concatenate(
            [self._normalise(first), self._normalise(second)], axis=2
        )
        return (
            torch.from_numpy(stacked.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask.astype(np.float32)),
        )

    def _sample_window(self, first, second, mask):
        from PIL import Image

        size = self.image_size
        height, width = mask.shape[:2]
        if self.train and height > size and width > size:
            top = random.randrange(0, height - size)
            left = random.randrange(0, width - size)
            return (first[top:top + size, left:left + size],
                    second[top:top + size, left:left + size],
                    mask[top:top + size, left:left + size])
        # Validation sees the whole scene resized, so the metric describes the scene
        # rather than a lucky crop of it.
        resize = lambda a, mode: np.asarray(  # noqa: E731
            Image.fromarray(a).resize((size, size), mode)
        )
        return (resize(first, Image.BILINEAR), resize(second, Image.BILINEAR),
                resize(mask, Image.NEAREST))

    @staticmethod
    def _normalise(image: np.ndarray) -> np.ndarray:
        return (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def build_model(config: ChangeConfig):
    """A segmentation backbone with its stem widened to six channels."""
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("training.train_change needs transformers.") from exc

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.model_id, num_labels=1, ignore_mismatched_sizes=True
    )
    # Located by shape rather than by attribute path. transformers has already moved
    # this between segformer.encoder.patch_embeddings[0] and segformer.stages[0]
    # .patch_embeddings, and a hardcoded path fails loudly on some versions and, worse,
    # could silently find a different conv on others.
    parent = name = None
    embed = None
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            owner_path, _, attribute = module_name.rpartition(".")
            parent = model.get_submodule(owner_path) if owner_path else model
            name, embed = attribute, module
            break
    if embed is None:
        raise SystemExit(
            "Could not find a three-channel stem to widen in "
            f"{config.model_id}; the architecture is not what this trainer assumes."
        )
    if embed.in_channels == 6:
        return model

    widened = torch.nn.Conv2d(
        6, embed.out_channels, kernel_size=embed.kernel_size,
        stride=embed.stride, padding=embed.padding, bias=embed.bias is not None,
    )
    with torch.no_grad():
        # Halve and duplicate: an unchanged pair then produces the same activation the
        # pretrained stem gave for the single image, so the encoder starts from its
        # pretrained behaviour rather than from noise.
        widened.weight.copy_(torch.cat([embed.weight, embed.weight], dim=1) * 0.5)
        if embed.bias is not None:
            widened.bias.copy_(embed.bias)
    setattr(parent, name, widened)
    return model


def loss_fn(logits: torch.Tensor, target: torch.Tensor, config: ChangeConfig) -> torch.Tensor:
    weight = torch.tensor(config.pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + target.sum() + 1.0)
    return bce * (1.0 - config.dice_weight) + dice * config.dice_weight


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> dict[str, float]:
    """IoU, Dice, precision and recall together, so a degenerate model is obvious."""
    model.eval()
    tp = fp = fn = tn = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = _forward(model, images, target.shape[-2:])
        else:
            logits = _forward(model, images, target.shape[-2:])
        predicted = (torch.sigmoid(logits.float()) > 0.5)
        actual = target > 0.5
        tp += int((predicted & actual).sum())
        fp += int((predicted & ~actual).sum())
        fn += int((~predicted & actual).sum())
        tn += int((~predicted & ~actual).sum())

    denominator = tp + fp + fn
    return {
        "iou": tp / denominator if denominator else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0,
        "predicted_change_fraction": (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0,
    }


def _forward(model, images, size):
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False).squeeze(1)


def train(config: ChangeConfig, *, resume: bool = False) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    data_root = REPO_ROOT / config.data_root
    if not data_root.is_dir():
        raise SystemExit(
            f"No change-detection data at {data_root}.\n"
            "Run: python -m training.datasets.download minenetcd"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = None
    if device.type == "cuda" and config.amp_dtype != "off":
        autocast_dtype = (torch.bfloat16 if config.amp_dtype == "bf16"
                          and torch.cuda.is_bf16_supported() else torch.float16)

    train_set = ChangePairDataset(data_root, "train", config, train=True)
    val_set = ChangePairDataset(data_root, "val", config, train=False)
    print(f"train sites={len(train_set)} val sites={len(val_set)} device={device}", flush=True)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=config.num_workers)

    model = build_model(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    run_dir = REPO_ROOT / config.output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"

    start_epoch = 0
    best_iou = 0.0
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_iou = float(state.get("best_iou", 0.0))
        print(f"resumed from epoch {start_epoch}", flush=True)

    history: list[dict[str, Any]] = []
    since_improvement = 0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = 0.0
        started = time.time()
        optimiser.zero_grad(set_to_none=True)
        for step, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss = loss_fn(_forward(model, images, target.shape[-2:]), target, config)
            else:
                loss = loss_fn(_forward(model, images, target.shape[-2:]), target, config)
            (loss / config.grad_accum).backward()
            if (step + 1) % config.grad_accum == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            running += float(loss)

        metrics = evaluate(model, val_loader, device, autocast_dtype)
        record = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)),
                  "seconds": round(time.time() - started, 1), **metrics}
        improved = metrics["iou"] > best_iou
        record["best"] = improved
        history.append(record)
        print(json.dumps(record), flush=True)

        torch.save({"model": model.state_dict(), "optimizer": optimiser.state_dict(),
                    "epoch": epoch, "best_iou": max(best_iou, metrics["iou"])}, checkpoint_path)
        if improved:
            best_iou = metrics["iou"]
            torch.save({"model": model.state_dict(), "config": asdict(config),
                        "metrics": metrics}, run_dir / "best.pt")
            since_improvement = 0
        else:
            since_improvement += 1
            if config.early_stop_patience and since_improvement >= config.early_stop_patience:
                print(f"stopping: no IoU improvement in {since_improvement} epochs", flush=True)
                break

    summary = {
        "name": config.name,
        "best_val_iou": best_iou,
        "epochs_run": len(history),
        "checkpoint": str(run_dir / "best.pt"),
        "config": asdict(config),
        "history": history,
        "reading_note": (
            "IoU on changed pixels selected the checkpoint. Pixel accuracy is reported "
            "alongside and is misleading here: MineNetCD is roughly 11 per cent changed, "
            "so predicting no change anywhere scores about 89 per cent while finding "
            "nothing. predicted_change_fraction is included so a collapsed model is "
            "visible at a glance."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train_change")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    config = ChangeConfig.load(args.config)
    for key in ("epochs", "batch_size", "image_size", "data_root", "output_dir"):
        value = getattr(args, key)
        if value is None:
            continue
        # The dataclass fields are str and the summary is written as JSON.
        setattr(config, key, str(value) if isinstance(value, Path) else value)

    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
