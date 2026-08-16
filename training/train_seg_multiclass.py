"""Resumable multiclass semantic-segmentation trainer.

This is intentionally separate from :mod:`training.train_seg`: the binary crack
trainer uses a single-logit BCE/Dice objective, while mutually exclusive semantic
classes require ``num_classes`` logits and cross-entropy.  Validation reports every
class IoU and selects checkpoints on their mean, never on pixel accuracy.

    python -m training.train_seg_multiclass \
        --config training/configs/agriculture_segformer_b2_mc.yaml
    python -m training.train_seg_multiclass --config <cfg> --resume
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_config(path: Path) -> dict[str, Any]:
    """Load YAML/JSON using the same dependency-tolerant contract as train_seg."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            parsed[key.strip()] = [
                item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
            ]
        elif value.lower() in {"true", "false"}:
            parsed[key.strip()] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).replace("e-", "", 1).isdigit():
            parsed[key.strip()] = float(value) if ("." in value or "e-" in value) else int(value)
        else:
            parsed[key.strip()] = value
    return parsed


@dataclass
class SegMulticlassConfig:
    """Complete, checkpointed configuration for one multiclass run."""

    name: str = "segformer_multiclass"
    data_root: str = "training/data/prepared/agriculture_seg"
    model_id: str = "nvidia/mit-b2"
    num_classes: int = 3
    class_names: list[str] | None = None
    # Multispectral input. Empty keeps the trainer on the corpus's RGB images, which is
    # what every existing config expects.
    #
    # WeedsGalore ships five bands and the corpus carries an RGB composite of three of
    # them, because PNG cannot hold five. Crop and weed separation lives mostly in red
    # edge and near infrared -- a weed and a maize leaf differ far more in reflectance
    # than in visible colour -- so an RGB-only model is using the bands that
    # discriminate least. Point this at the cached stacks to use all five.
    band_root: str = ""
    band_names: list[str] | None = None
    image_size: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    epochs: int = 40
    lr: float = 6e-5
    encoder_lr: float | None = None
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    num_workers: int = 4
    seed: int = 1337
    amp_dtype: str = "bf16"
    ema_decay: float = 0.999
    class_weighting: str = "none"  # none | log_inverse
    class_weights: list[float] | None = None
    fliplr: float = 0.5
    flipud: float = 0.3
    degrees: float = 90.0
    output_dir: str = "training/runs"
    max_train_samples: int = 0
    val_interval: int = 1
    early_stop_patience: int = 0

    @classmethod
    def load(cls, path: Path) -> "SegMulticlassConfig":
        raw = _read_config(path)
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")
        try:
            config = cls(**raw)
            config.validate()
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid multiclass segmentation config {path}: {exc}") from exc
        return config

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2 for multiclass segmentation")
        if self.class_names is not None:
            self.class_names = [str(name) for name in self.class_names]
            if len(self.class_names) != self.num_classes:
                raise ValueError(
                    f"class_names has {len(self.class_names)} entries; "
                    f"num_classes is {self.num_classes}"
                )
            if len(set(self.class_names)) != len(self.class_names):
                raise ValueError("class_names must be unique")
        self.class_weighting = str(self.class_weighting).strip().lower()
        if self.class_weighting not in {"none", "log_inverse"}:
            raise ValueError("class_weighting must be 'none' or 'log_inverse'")
        if self.class_weights is not None:
            self.class_weights = [float(value) for value in self.class_weights]
            if len(self.class_weights) != self.num_classes:
                raise ValueError(
                    f"class_weights has {len(self.class_weights)} entries; "
                    f"num_classes is {self.num_classes}"
                )
            if any(not math.isfinite(value) or value <= 0 for value in self.class_weights):
                raise ValueError("class_weights must contain finite positive numbers")
            if self.class_weighting != "none":
                raise ValueError("use either explicit class_weights or class_weighting, not both")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be at least 1")
        if self.image_size < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("image_size, batch_size and epochs must be positive")
        if self.amp_dtype not in {"bf16", "fp16", "off"}:
            raise ValueError("amp_dtype must be 'bf16', 'fp16' or 'off'")
        for name in ("fliplr", "flipud"):
            probability = float(getattr(self, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def resolved_class_names(self) -> list[str]:
        return self.class_names or [f"class_{index}" for index in range(self.num_classes)]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


class MulticlassSegDataset(Dataset):
    """Prepared image/mask pairs with strict dense class-id validation."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        num_classes: int,
        *,
        train: bool,
        limit: int = 0,
        fliplr: float = 0.5,
        flipud: float = 0.3,
        degrees: float = 90.0,
        band_root: str = "",
    ) -> None:
        self.band_root = band_root
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise SystemExit(
                f"No prepared multiclass data at {root / split}. "
                "Expected images/ and masks/ directories."
            )
        images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit:
            images = images[:limit]
        if not images:
            raise SystemExit(f"No prepared images found in {self.image_dir}")

        self.items: list[tuple[Path, Path]] = []
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.fliplr = float(fliplr)
        self.flipud = float(flipud)
        self.degrees = max(0.0, float(degrees))
        self.class_pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        for image_path in images:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise ValueError(f"Missing segmentation mask for {image_path}: {mask_path}")
            mask = self._load_mask(mask_path)
            self.class_pixel_counts += np.bincount(
                mask.reshape(-1), minlength=self.num_classes
            )[: self.num_classes]
            self.items.append((image_path, mask_path))
        self.transform = self._build_transform()

    def _band_path(self, image_path: Path) -> Path | None:
        """The cached multispectral stack for this sample, when one is configured.

        Sample ids carry a source prefix (weeds_2023-05-25_0109) while the cached stacks
        are keyed by the capture id, so both are tried rather than assuming one shape.
        """
        if not getattr(self, "band_root", None):
            return None
        stem = image_path.stem
        for candidate in (stem, stem.split("_", 1)[-1]):
            path = Path(self.band_root) / f"{candidate}.npy"
            if path.is_file():
                return path
        return None

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(mask_path) as source:
            mask = np.asarray(source).copy()
        if mask.ndim != 2:
            raise ValueError(
                f"Mask {mask_path} must be a single-channel class-id image; "
                f"found shape {mask.shape}."
            )
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Mask {mask_path} must contain integer class ids, found {mask.dtype}.")
        invalid = np.unique(mask[(mask < 0) | (mask >= self.num_classes)])
        if invalid.size:
            values = ", ".join(str(int(value)) for value in invalid)
            raise ValueError(
                f"Mask {mask_path} contains class id(s) {values} outside "
                f"the configured range 0..{self.num_classes - 1}."
            )
        return mask.astype(np.int64, copy=False)

    def _build_transform(self):
        try:
            import albumentations as A
        except ImportError:
            return None
        if not self.train:
            return A.Compose([A.Resize(self.image_size, self.image_size)])
        transforms: list[Any] = [
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.5, 1.0), ratio=(0.8, 1.25)
            ),
            A.HorizontalFlip(p=self.fliplr),
            A.VerticalFlip(p=self.flipud),
        ]
        if self.degrees > 0:
            transforms.append(A.Rotate(limit=self.degrees, p=0.5))
        # Colour augmentation is dropped for multispectral input, and not only because
        # HueSaturationValue rejects a 5-channel array. Hue and saturation are defined
        # over visible colour; applying them across a stack that includes red edge and
        # near infrared would alter the very reflectance ratios the model is meant to
        # separate crop from weed by. Geometric augmentation and noise stay, since those
        # are band-agnostic.
        if getattr(self, "band_root", ""):
            transforms.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.GaussNoise()], p=0.25))
        else:
            transforms.extend(
                [
                    A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                    A.HueSaturationValue(10, 20, 15, p=0.3),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.GaussNoise()], p=0.25),
                    A.CoarseDropout(p=0.15),
                ]
            )
        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path, mask_path = self.items[index]
        stacked = self._band_path(image_path)
        if stacked is not None:
            image = np.load(stacked)
        else:
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB")).copy()
        mask = self._load_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask dimensions disagree for {image_path.name}: "
                f"image={image.shape[:2]}, mask={mask.shape}."
            )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image, mask = _resize_pair(image, mask, self.image_size)
            if self.train and random.random() < self.fliplr:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if self.train and random.random() < self.flipud:
                image, mask = image[::-1, :].copy(), mask[::-1, :].copy()
            if self.train and self.degrees > 0 and random.random() < 0.5:
                image, mask = _rotate_pair(image, mask, random.uniform(-self.degrees, self.degrees))

        # Validate again after augmentation: a transform that interpolates a mask is
        # a configuration defect, not a new semantic class to clamp away.
        invalid = np.unique(mask[(mask < 0) | (mask >= self.num_classes)])
        if invalid.size:
            values = ", ".join(str(int(value)) for value in invalid)
            raise ValueError(
                f"Mask {mask_path} contains class id(s) {values} after augmentation, "
                f"outside 0..{self.num_classes - 1}."
            )
        # ImageNet statistics only describe three visible bands. Extra bands are
        # normalised with the mean of those statistics rather than invented per-band
        # values: it keeps them on the same scale as RGB without asserting a
        # distribution nobody measured. Per-band statistics computed from this corpus
        # would be better and are worth doing once the model is shown to be worth it.
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        bands = image.shape[2]
        if bands != mean.shape[0]:
            extra = bands - mean.shape[0]
            mean = np.concatenate([mean, np.full(extra, float(mean.mean()), np.float32)])
            std = np.concatenate([std, np.full(extra, float(std.mean()), np.float32)])
        tensor = (image.astype(np.float32) / 255.0 - mean) / std
        return (
            torch.from_numpy(tensor.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask.astype(np.int64, copy=False).copy()),
        )


def _resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    resized_image = np.asarray(
        Image.fromarray(image).resize((size, size), resampling.BILINEAR)
    )
    resized_mask = np.asarray(
        Image.fromarray(mask.astype(np.int32), mode="I").resize((size, size), resampling.NEAREST)
    )
    return resized_image, resized_mask.astype(np.int64, copy=False)


def _rotate_pair(
    image: np.ndarray, mask: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    rotated_image = np.asarray(
        Image.fromarray(image).rotate(angle, resample=resampling.BILINEAR)
    )
    rotated_mask = np.asarray(
        Image.fromarray(mask.astype(np.int32), mode="I").rotate(
            angle, resample=resampling.NEAREST
        )
    )
    return rotated_image, rotated_mask.astype(np.int64, copy=False)


def class_weights_from_counts(
    counts: Sequence[int],
    *,
    mode: str = "log_inverse",
) -> list[float]:
    """Return normalized data-derived weights, refusing classes with no evidence."""

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("class pixel counts must be a non-empty one-dimensional sequence")
    missing = np.flatnonzero(values <= 0)
    if missing.size:
        raise ValueError(
            "Training masks contain no pixels for declared class id(s): "
            + ", ".join(str(int(index)) for index in missing)
        )
    if mode != "log_inverse":
        raise ValueError(f"Unsupported class weighting mode: {mode!r}")
    frequencies = values / values.sum()
    weights = 1.0 / np.log(1.02 + frequencies)
    weights /= weights.mean()
    return weights.astype(np.float32).tolist()


def resolve_class_weights(
    config: SegMulticlassConfig,
    counts: Sequence[int],
) -> list[float] | None:
    if config.class_weights is not None:
        return list(config.class_weights)
    if config.class_weighting == "none":
        return None
    return class_weights_from_counts(counts, mode=config.class_weighting)


@dataclass
class MulticlassIoU:
    """Dataset-level confusion matrix with explicit per-class IoU."""

    matrix: np.ndarray

    @classmethod
    def create(cls, num_classes: int) -> "MulticlassIoU":
        return cls(np.zeros((num_classes, num_classes), dtype=np.int64))

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predicted = logits.argmax(dim=1).detach().cpu().numpy().reshape(-1)
        expected = target.detach().cpu().numpy().reshape(-1)
        count = self.matrix.shape[0]
        invalid = np.unique(expected[(expected < 0) | (expected >= count)])
        if invalid.size:
            raise ValueError(
                "Metric target contains class id(s) outside the configured range: "
                + ", ".join(str(int(value)) for value in invalid)
            )
        bins = np.bincount(expected * count + predicted, minlength=count * count)
        self.matrix += bins.reshape(count, count)

    def summary(self, class_names: Sequence[str]) -> dict[str, Any]:
        if len(class_names) != self.matrix.shape[0]:
            raise ValueError("class_names length does not match the confusion matrix")
        true_positive = np.diag(self.matrix).astype(np.float64)
        support = self.matrix.sum(axis=1).astype(np.float64)
        predicted = self.matrix.sum(axis=0).astype(np.float64)
        union = support + predicted - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.zeros_like(union),
            where=union > 0,
        )
        measured = union > 0
        total = int(self.matrix.sum())
        return {
            "mean_iou": float(iou[measured].mean()) if measured.any() else 0.0,
            "per_class_iou": {
                str(name): float(iou[index]) if measured[index] else None
                for index, name in enumerate(class_names)
            },
            "per_class_support": {
                str(name): int(support[index]) for index, name in enumerate(class_names)
            },
            "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
            "confusion_matrix": self.matrix.tolist(),
        }


def build_model(model_id: str, num_classes: int, class_names: Sequence[str],
                in_channels: int = 3):
    from transformers import SegformerForSemanticSegmentation

    id2label = {index: str(name) for index, name in enumerate(class_names)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=num_classes,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    if in_channels != 3:
        _widen_stem(model, in_channels)
    return model


def _widen_stem(model, in_channels: int) -> None:
    """Accept more than three input bands without discarding the pretrained stem.

    The extra bands are initialised from the mean of the pretrained RGB filters and the
    whole stem is rescaled by 3/in_channels, so the initial activation magnitude matches
    what the pretrained encoder expects. Initialising them randomly instead would put
    noise into the first layer of an otherwise pretrained network.

    Located by shape rather than attribute path: transformers has moved this between
    segformer.encoder.patch_embeddings[0] and segformer.stages[0].patch_embeddings, and
    a hardcoded path fails on some versions and finds the wrong conv on others.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            owner_path, _, attribute = name.rpartition(".")
            parent = model.get_submodule(owner_path) if owner_path else model
            widened = torch.nn.Conv2d(
                in_channels, module.out_channels, kernel_size=module.kernel_size,
                stride=module.stride, padding=module.padding,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                average = module.weight.mean(dim=1, keepdim=True)
                extra = average.repeat(1, in_channels - 3, 1, 1)
                widened.weight.copy_(
                    torch.cat([module.weight, extra], dim=1) * (3.0 / in_channels)
                )
                if module.bias is not None:
                    widened.bias.copy_(module.bias)
            setattr(parent, attribute, widened)
            return
    raise SystemExit(
        "Could not find a three-channel stem to widen; the architecture is not what "
        "this trainer assumes."
    )


def forward_logits(model, images: torch.Tensor) -> torch.Tensor:
    output = model(pixel_values=images)
    logits = output.logits
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
    return logits


def build_optimizer(model, config: SegMulticlassConfig) -> torch.optim.Optimizer:
    if config.encoder_lr is None:
        return torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    encoder = getattr(model, "segformer", None)
    if encoder is None:
        raise ValueError(
            "encoder_lr was configured, but the model exposes no SegFormer encoder"
        )
    encoder_parameters = list(encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config.encoder_lr)},
            {"params": head_parameters, "lr": config.lr},
        ],
        weight_decay=config.weight_decay,
    )


class ModelEMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone().float()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach().float(), alpha=1.0 - self.decay
                )

    def copy_into(self, model) -> dict[str, torch.Tensor]:
        current = model.state_dict()
        backup = {key: value.detach().clone() for key, value in current.items()}
        model.load_state_dict(
            {
                key: self.shadow[key].to(value.dtype) if key in self.shadow else value
                for key, value in current.items()
            },
            strict=True,
        )
        return backup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(requested: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda" or requested == "off":
        return None, False
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    class_names: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    metrics = MulticlassIoU.create(len(class_names))
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            logits = forward_logits(model, images)
        metrics.update(logits.float(), targets)
    return metrics.summary(class_names)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_last_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    ema: ModelEMA | None,
    epoch: int,
    best_mean_iou: float,
    history: list[dict[str, Any]],
    config: SegMulticlassConfig,
    class_weights: list[float] | None,
) -> None:
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.shadow if ema is not None else None,
            "epoch": int(epoch),
            "best_mean_iou": float(best_mean_iou),
            "history": history,
            "config": asdict(config),
            "class_weights": class_weights,
        },
        path,
    )


def _restore_last_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    ema: ModelEMA | None,
    device: torch.device,
    config: SegMulticlassConfig,
) -> tuple[int, float, list[dict[str, Any]]]:
    state = torch.load(path, map_location=device, weights_only=False)
    saved_config = dict(state.get("config") or {})
    for key in ("model_id", "num_classes", "class_names"):
        if key in saved_config and saved_config[key] != asdict(config)[key]:
            raise ValueError(
                f"Cannot resume {path}: checkpoint {key}={saved_config[key]!r} "
                f"does not match config value {asdict(config)[key]!r}."
            )
    required = {"model", "optimizer", "scheduler", "epoch"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Cannot resume {path}: missing state {', '.join(missing)}")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema"):
        ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
    return (
        int(state["epoch"]) + 1,
        float(state.get("best_mean_iou", state.get("best_iou", -1.0))),
        list(state.get("history") or []),
    )


def train(config: SegMulticlassConfig, *, resume: bool = False) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = _resolve_path(config.data_root)
    run_dir = _resolve_path(config.output_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    class_names = config.resolved_class_names

    train_set = MulticlassSegDataset(
        data_root,
        "train",
        config.image_size,
        config.num_classes,
        train=True,
        limit=config.max_train_samples,
        fliplr=config.fliplr,
        flipud=config.flipud,
        degrees=config.degrees,
        band_root=config.band_root,
    )
    val_set = MulticlassSegDataset(
        data_root,
        "val",
        config.image_size,
        config.num_classes,
        train=False,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        band_root=config.band_root,
    )
    weight_values = resolve_class_weights(config, train_set.class_pixel_counts)
    print(
        f"train={len(train_set)} val={len(val_set)} device={device} "
        f"class_pixels={train_set.class_pixel_counts.tolist()} "
        f"class_weights={weight_values}",
        flush=True,
    )

    workers = config.num_workers if os.name != "nt" else min(config.num_workers, 2)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, config.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    # The band count comes from the data rather than a second config field, so the two
    # cannot disagree: a config claiming five bands over an RGB corpus would otherwise
    # build a five-channel stem and feed it three.
    in_channels = int(train_set[0][0].shape[0])
    if in_channels != 3:
        print(f"multispectral input: {in_channels} bands", flush=True)
    model = build_model(
        config.model_id, config.num_classes, class_names, in_channels=in_channels
    ).to(device)
    optimizer = build_optimizer(model, config)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / config.grad_accum))
    total_steps = optimizer_steps_per_epoch * config.epochs
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
    loss_weights = (
        torch.tensor(weight_values, device=device, dtype=torch.float32)
        if weight_values is not None
        else None
    )

    checkpoint_path = run_dir / "last.pt"
    start_epoch = 0
    best_mean_iou = -1.0
    history: list[dict[str, Any]] = []
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint does not exist: {checkpoint_path}")
        start_epoch, best_mean_iou, history = _restore_last_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            config=config,
        )
        print(
            f"Resumed from epoch {start_epoch} (best mean IoU {best_mean_iou:.4f})",
            flush=True,
        )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_start = time.time()
        for index, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                logits = forward_logits(model, images)
                loss = F.cross_entropy(logits.float(), targets, weight=loss_weights)
                scaled_loss = loss / config.grad_accum
            if needs_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += float(loss.detach().cpu())

            update = (index + 1) % config.grad_accum == 0 or index + 1 == len(train_loader)
            if update:
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
                print(
                    f"  epoch {epoch} [{index + 1}/{len(train_loader)}] "
                    f"loss={running_loss / (index + 1):.4f} "
                    f"lr={scheduler.get_last_lr()[-1]:.2e}",
                    flush=True,
                )

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "seconds": round(time.time() - epoch_start, 1),
        }
        if (epoch + 1) % config.val_interval == 0 or epoch == config.epochs - 1:
            backup = ema.copy_into(model) if ema is not None else None
            metrics = evaluate(model, val_loader, device, autocast_dtype, class_names)
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if backup is not None:
                model.load_state_dict(backup, strict=True)
            record["validation"] = metrics
            if float(metrics["mean_iou"]) > best_mean_iou:
                best_mean_iou = float(metrics["mean_iou"])
                epochs_without_improvement = 0
                _atomic_torch_save(
                    {
                        "model": best_model_state,
                        "config": asdict(config),
                        "class_names": class_names,
                        "class_weights": weight_values,
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
        _save_last_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            best_mean_iou=best_mean_iou,
            history=history,
            config=config,
            class_weights=weight_values,
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if config.early_stop_patience and epochs_without_improvement >= config.early_stop_patience:
            print(
                f"Early stop: no mean-IoU improvement in "
                f"{epochs_without_improvement} evaluations.",
                flush=True,
            )
            break

    summary = {
        "name": config.name,
        "best_val_mean_iou": best_mean_iou,
        "epochs_run": len(history),
        "class_names": class_names,
        "class_weights": weight_values,
        "checkpoint": str(run_dir / "best.pt"),
        "config": asdict(config),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train_seg_multiclass")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt.")
    parser.add_argument("--epochs", type=int, help="Override the config epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override the config batch size.")
    parser.add_argument("--image-size", type=int, help="Override the config image size.")
    parser.add_argument(
        "--max-train-samples", type=int, help="Cap training samples for a smoke run."
    )
    parser.add_argument("--data-root", type=Path, help="Override the corpus location.")
    parser.add_argument("--output-dir", type=Path, help="Override where runs are written.")
    args = parser.parse_args(argv)

    config = SegMulticlassConfig.load(args.config)
    for key in ("epochs", "batch_size", "image_size", "max_train_samples"):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    config.validate()
    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
