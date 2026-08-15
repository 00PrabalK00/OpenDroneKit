"""Multiclass segmentation training contracts exercised without starting a run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")

from training.train_seg_multiclass import (  # noqa: E402
    MulticlassIoU,
    MulticlassSegDataset,
    SegMulticlassConfig,
    _restore_last_checkpoint,
    _save_last_checkpoint,
    class_weights_from_counts,
)


CONFIG_PATH = Path("training/configs/agriculture_segformer_b2_mc.yaml")


def _write_pair(root: Path, mask: np.ndarray, name: str = "capture.png") -> Path:
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[..., 1] = 160
    Image.fromarray(image, mode="RGB").save(image_dir / name)
    mask_path = mask_dir / name
    Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
    return mask_path


def test_agriculture_config_activates_the_recorded_multiclass_intent():
    config = SegMulticlassConfig.load(CONFIG_PATH)
    assert config.num_classes == 3
    assert config.class_names == ["soil", "maize", "weed"]
    assert config.class_weighting == "log_inverse"
    assert config.encoder_lr == pytest.approx(0.000006)
    assert (config.fliplr, config.flipud, config.degrees) == (0.5, 0.5, 180.0)


def test_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("num_classes: 3\npixel_accuracy_is_enough: true\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="pixel_accuracy_is_enough"):
        SegMulticlassConfig.load(config_path)


def test_dataset_preserves_dense_class_ids_and_returns_long_targets(tmp_path):
    _write_pair(tmp_path, np.array([[0, 1], [2, 2]], dtype=np.uint8))
    dataset = MulticlassSegDataset(
        tmp_path, "train", 2, 3, train=False
    )
    image, mask = dataset[0]
    assert tuple(image.shape) == (3, 2, 2)
    assert mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, 1, 2}
    assert dataset.class_pixel_counts.tolist() == [1, 1, 2]


def test_dataset_refuses_out_of_range_id_with_the_mask_filename(tmp_path):
    mask_path = _write_pair(
        tmp_path, np.array([[0, 1], [2, 5]], dtype=np.uint8), name="bad_capture.png"
    )
    with pytest.raises(ValueError) as error:
        MulticlassSegDataset(tmp_path, "train", 2, 3, train=False)
    message = str(error.value)
    assert mask_path.name in message
    assert "class id(s) 5" in message
    assert "0..2" in message


def test_log_inverse_weighting_raises_the_rare_class_and_normalizes():
    weights = class_weights_from_counts([9000, 900, 100])
    assert np.mean(weights) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_weighting_refuses_a_declared_class_with_no_pixels():
    with pytest.raises(ValueError, match=r"class id\(s\): 2"):
        class_weights_from_counts([100, 20, 0])


def test_metrics_report_every_class_and_mean_over_classes():
    predicted = torch.tensor([[[0, 0, 0, 1, 0]]])
    target = torch.tensor([[[0, 0, 0, 1, 2]]])
    logits = torch.nn.functional.one_hot(predicted, num_classes=3).permute(0, 3, 1, 2).float()
    metrics = MulticlassIoU.create(3)
    metrics.update(logits, target)
    summary = metrics.summary(["soil", "maize", "weed"])
    assert summary["per_class_iou"] == {
        "soil": pytest.approx(0.75),
        "maize": pytest.approx(1.0),
        "weed": pytest.approx(0.0),
    }
    assert summary["mean_iou"] == pytest.approx((0.75 + 1.0 + 0.0) / 3.0)
    assert summary["pixel_accuracy"] == pytest.approx(0.8)
    assert summary["mean_iou"] != summary["pixel_accuracy"]


def test_last_checkpoint_restores_model_optimizer_scheduler_and_epoch(tmp_path):
    config = SegMulticlassConfig(
        model_id="local-tiny",
        num_classes=3,
        class_names=["soil", "maize", "weed"],
        epochs=3,
        ema_decay=0.0,
    )
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    model(torch.ones(1, 3, 2, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    saved_weight = model.weight.detach().clone()
    checkpoint = tmp_path / "last.pt"
    history = [{"epoch": 1, "validation": {"mean_iou": 0.42}}]
    _save_last_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        ema=None,
        epoch=1,
        best_mean_iou=0.42,
        history=history,
        config=config,
        class_weights=[0.5, 1.0, 1.5],
    )

    restored_model = torch.nn.Conv2d(3, 3, kernel_size=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 1.0
    )
    start_epoch, best_iou, restored_history = _restore_last_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        ema=None,
        device=torch.device("cpu"),
        config=config,
    )
    assert start_epoch == 2
    assert best_iou == pytest.approx(0.42)
    assert restored_history == history
    assert torch.equal(restored_model.weight, saved_weight)
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
