'''Train the shared DINOv2/UPerNet semantic model from an audited corpus.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from training.semantic_tiles import IGNORE_INDEX, SemanticTileDataset
from training.shared_semantic_model import build_dinov2_vitb14_upernet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Confusion:
    matrix: np.ndarray

    @classmethod
    def create(cls, classes: int) -> 'Confusion':
        return cls(np.zeros((classes, classes), dtype=np.int64))

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predicted = logits.argmax(dim=1).detach().cpu().numpy().reshape(-1)
        expected = target.detach().cpu().numpy().reshape(-1)
        valid = expected != IGNORE_INDEX
        expected = expected[valid]
        predicted = predicted[valid]
        if not expected.size:
            return
        count = self.matrix.shape[0]
        bins = np.bincount(expected * count + predicted, minlength=count * count)
        self.matrix += bins.reshape(count, count)

    def summary(self, class_names: list[str]) -> dict[str, Any]:
        true_positive = np.diag(self.matrix).astype(np.float64)
        expected = self.matrix.sum(axis=1).astype(np.float64)
        predicted = self.matrix.sum(axis=0).astype(np.float64)
        union = expected + predicted - true_positive
        iou = np.divide(true_positive, union, out=np.zeros_like(union), where=union > 0)
        support = expected > 0
        return {
            'mean_iou': float(iou[support].mean()) if support.any() else 0.0,
            'pixel_accuracy': float(true_positive.sum() / self.matrix.sum()) if self.matrix.sum() else 0.0,
            'per_class_iou': {
                name: float(iou[index]) if support[index] else None
                for index, name in enumerate(class_names)
            },
            'confusion_matrix': self.matrix.tolist(),
        }


def absent_class_penalty(
    logits: torch.Tensor,
    negative_mask: torch.Tensor,
    negative_classes: torch.Tensor,
) -> torch.Tensor:
    """Penalise predicting a class where the annotation says it is not.

    Cross-entropy with ignore_index scores only labelled pixels. On a corpus where one
    source labels a single class and leaves everything else unlabelled -- SpaceNet 7
    marks 96.7 per cent of each tile ignore -- that means predicting BUILDING everywhere
    costs nothing on those tiles, and the model that does it is exactly what came out:
    building on 100 per cent of the India holdout, precision 0.092.

    The fix has to respect what the annotation actually supports. An unlabelled pixel in
    an exhaustively annotated tile is evidence the class is absent, and no evidence at
    all about which class is present, so this drives p(class) down there and says nothing
    about the rest of the distribution. `-log(1 - p)` is the negative-learning form of
    cross-entropy: zero when the model already agrees, unbounded when it insists.

    Returns a zero scalar when the batch carries no such evidence, so a corpus of fully
    labelled tiles trains exactly as it did before.
    """
    if not bool(negative_mask.any()) or not bool(negative_classes.any()):
        return logits.sum() * 0.0

    probabilities = torch.softmax(logits.float(), dim=1)
    # [B, C] -> [B, C, 1, 1] against [B, 1, H, W]: a class is penalised only on tiles
    # that carry evidence for it, and only where that tile is unlabelled.
    weights = negative_classes.to(probabilities.dtype)[:, :, None, None]
    where = negative_mask.to(probabilities.dtype)[:, None, :, :]
    penalised = weights * where
    total = penalised.sum()
    if float(total) == 0.0:
        return logits.sum() * 0.0
    # Clamped because a confident wrong answer would otherwise produce inf and take the
    # step with it.
    surprise = -torch.log(torch.clamp(1.0 - probabilities, min=1e-6))
    return (surprise * penalised).sum() / total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalise(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (images - mean) / std


def require_declared_class_coverage(corpus: dict[str, Any]) -> None:
    counts = dict(corpus.get('counts') or {})
    uncovered = [int(value) for value in counts.get('uncovered_class_ids', [])]
    if uncovered:
        raise ValueError(
            'Corpus has no declared training labels for semantic class id(s): '
            + ', '.join(str(value) for value in uncovered)
        )


def class_weights_from_corpus(
    corpus: dict[str, Any],
    class_ids: list[int],
) -> list[float]:
    '''ENet-style log inverse-frequency weights normalized to mean one.'''
    raw = dict((corpus.get('counts') or {}).get('declared_class_pixel_counts') or {})
    counts = np.asarray([float(raw.get(str(class_id), 0)) for class_id in class_ids])
    if np.any(counts <= 0):
        missing = [class_ids[index] for index in np.flatnonzero(counts <= 0)]
        raise ValueError(
            'Corpus needs positive pixel counts for balanced semantic class id(s): '
            + ', '.join(str(value) for value in missing)
        )
    frequencies = counts / counts.sum()
    weights = 1.0 / np.log(1.02 + frequencies)
    weights /= weights.mean()
    return weights.astype(np.float32).tolist()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    classes: int,
    class_names: list[str],
    mixed_precision: bool,
) -> dict[str, Any]:
    model.eval()
    autocast_dtype = (
        torch.bfloat16
        if mixed_precision and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    confusion = Confusion.create(classes)
    loss_total = 0.0
    batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        for batch in loader:
            images = _normalise(batch['image'].to(device, non_blocking=True))
            masks = batch['mask'].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=mixed_precision,
            ):
                logits = model(images)
                loss = criterion(logits, masks)
            value = float(loss.detach().cpu())
            # A single non-finite batch would otherwise make the whole epoch's reported
            # loss nan, which is what hid the fp16 overflow in the first place.
            if np.isfinite(value):
                loss_total += value
                batches += 1
            confusion.update(logits, masks)
    metrics = confusion.summary(class_names)
    metrics['loss'] = loss_total / batches if batches else None
    return metrics


def train(
    config_path: str | Path,
    corpus_path: str | Path,
    run_dir: str | Path,
    *,
    dinov2_source: str | None = None,
    source: str | None = None,
    allow_cpu: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    corpus_path = Path(corpus_path)
    run_dir = Path(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    configured_schema = dict(config.get('schema') or {})
    if corpus.get('schema') != configured_schema:
        raise ValueError('Corpus schema does not exactly match the training configuration.')
    require_declared_class_coverage(corpus)
    classes = list(configured_schema['classes'])
    class_names = [str(item['name']) for item in classes]
    class_ids = [int(item['id']) for item in classes]
    training = dict(config.get('training') or {})
    seed = int(training.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and not allow_cpu:
        raise RuntimeError('Shared DINOv2 training requires CUDA unless --allow-cpu is explicit.')
    tile_size = int(training.get('tile_size', config.get('inference', {}).get('tile_size', 518)))
    train_data = SemanticTileDataset(corpus_path, 'train', tile_size=tile_size, augment=True)
    val_data = SemanticTileDataset(corpus_path, 'validation', tile_size=tile_size, augment=False)
    batch_size = int(training.get('batch_size', 2))
    workers = int(training.get('workers', 0))
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=device.type == 'cuda', drop_last=False,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == 'cuda', drop_last=False,
    )

    architecture = dict(config.get('architecture') or {})
    resolved_dinov2_source = str(
        dinov2_source or architecture.get('encoder_source') or 'facebookresearch/dinov2'
    )
    resolved_source_type = str(
        source or architecture.get('encoder_source_type') or 'github'
    )
    model = build_dinov2_vitb14_upernet(
        architecture['encoder_checkpoint'],
        len(classes),
        dinov2_source=resolved_dinov2_source,
        source=resolved_source_type,
        freeze_encoder=bool(training.get('freeze_encoder', False)),
    ).to(device)
    base_lr = float(training.get('learning_rate', 2e-4))
    encoder_lr = float(training.get('encoder_learning_rate', base_lr * 0.1))
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': encoder_lr},
            {'params': model.decoder.parameters(), 'lr': base_lr},
        ],
        weight_decay=float(training.get('weight_decay', 0.01)),
    )
    epochs = int(training.get('epochs', 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training.get('min_learning_rate', base_lr * 0.01)),
    )
    accumulation = max(1, int(training.get('gradient_accumulation', 1)))
    # Weight on the absent-class term. Set to 0 to train exactly as the first head did,
    # which is the run that predicted building everywhere on the India holdout.
    absent_weight = float(training.get('absent_class_weight', 1.0))
    mixed_precision = device.type == 'cuda' and bool(training.get('mixed_precision', True))
    # bf16 where the GPU supports it. The first run trained in fp16 and reported a train
    # loss of nan for every epoch: the ViT overflows fp16's range, the gradient scaler
    # then silently SKIPS those steps, and the model learns from an unknown subset of the
    # corpus. bf16 carries fp32's exponent range, so the overflow does not happen and no
    # scaler is needed. The nan was visible in the manifest for weeks and read as a
    # cosmetic reporting quirk.
    autocast_dtype = (
        torch.bfloat16
        if mixed_precision and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=mixed_precision and autocast_dtype is torch.float16
    )
    balance = str(training.get('class_balance', 'none')).casefold()
    if balance == 'log_inverse':
        weight_values = class_weights_from_corpus(corpus, class_ids)
        class_weights = torch.tensor(weight_values, device=device, dtype=torch.float32)
    elif balance == 'none':
        class_weights = None
    else:
        raise ValueError(f'Unsupported semantic class_balance mode: {balance!r}')
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        weight=class_weights,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_iou = -1.0
    start_epoch = 1
    corpus_sha256 = _sha256(corpus_path)
    last_checkpoint = run_dir / 'last.pt'
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(
                f'Cannot resume; checkpoint does not exist: {last_checkpoint}'
            )
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get('schema') != configured_schema:
            raise ValueError('Resume checkpoint semantic schema does not match.')
        if checkpoint.get('corpus_sha256') != corpus_sha256:
            raise ValueError('Resume checkpoint corpus hash does not match.')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        history = list(checkpoint.get('history') or [])
        best_iou = float(checkpoint.get('best_mean_iou', -1.0))
        start_epoch = int(checkpoint['epoch']) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        counted_steps = 0
        nonfinite_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            images = _normalise(batch['image'].to(device, non_blocking=True))
            masks = batch['mask'].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=mixed_precision,
            ):
                logits = model(images)
                loss = criterion(logits, masks)
                if absent_weight:
                    loss = loss + absent_weight * absent_class_penalty(
                        logits,
                        batch['negative_mask'].to(device, non_blocking=True),
                        batch['negative_classes'].to(device, non_blocking=True),
                    )
                loss = loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get('max_grad_norm', 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step_loss = float(loss.detach().cpu()) * accumulation
            if np.isfinite(step_loss):
                running_loss += step_loss
                counted_steps += 1
            else:
                # Named rather than averaged into nan: a non-finite loss means the
                # gradient scaler drops this step, so the tile taught the model nothing.
                nonfinite_steps += 1

        validation = _evaluate(
            model,
            val_loader,
            device=device,
            classes=len(classes),
            class_names=class_names,
            mixed_precision=mixed_precision,
        )
        record = {
            'epoch': epoch,
            'train_loss': running_loss / max(1, counted_steps),
            'nonfinite_steps': nonfinite_steps,
            'steps': len(train_loader),
            'learning_rates': [float(group['lr']) for group in optimizer.param_groups],
            'validation': validation,
        }
        history.append(record)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if float(validation['mean_iou']) > best_iou:
            best_iou = float(validation['mean_iou'])
            _atomic_torch_save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'validation_metrics': validation,
                'schema': configured_schema,
                'architecture': architecture,
                'corpus_manifest': str(corpus_path),
                'corpus_sha256': corpus_sha256,
                'config': config,
            }, run_dir / 'best.pt')
        scheduler.step()
        _atomic_torch_save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch,
            'history': history,
            'best_mean_iou': best_iou,
            'schema': configured_schema,
            'corpus_sha256': corpus_sha256,
        }, last_checkpoint)

    summary = {
        'status': 'trained',
        'best_mean_iou': best_iou,
        'epochs': epochs,
        'epochs_completed': len(history),
        'device': str(device),
        'schema': configured_schema,
        'class_weights': (
            None if class_weights is None else class_weights.detach().cpu().tolist()
        ),
        'corpus_sha256': corpus_sha256,
        'best_checkpoint': str(run_dir / 'best.pt'),
    }
    (run_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train the shared DINOv2/UPerNet model.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dinov2-source')
    parser.add_argument('--source', choices=('github', 'local'))
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = train(
        args.config,
        args.corpus,
        args.run_dir,
        dinov2_source=args.dinov2_source,
        source=args.source,
        allow_cpu=args.allow_cpu,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
