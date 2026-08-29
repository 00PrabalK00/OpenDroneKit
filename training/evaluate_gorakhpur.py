"""Score a semantic checkpoint on Gorakhpur, per class, on Indian ground.

Every India number this project has ever published measures ONE class. The SpaceNet 7
holdout labels buildings and leaves everything else at IGNORE_INDEX, so road, vegetation,
water and bare land have never been measured on Indian imagery at all -- while
`eng.semantic` claims all six.

Gorakhpur is exhaustively annotated: the annotators labelled every pixel of every tile
into one of the source classes. That is what makes a multiclass score defensible here and
not on SpaceNet 7. On an exhaustively labelled tile an unlabelled prediction is a real
error and a false positive is a real false positive, so the confusion matrix has
negatives and the metric can fail. On SpaceNet 7 it cannot: dropping ignored pixels
leaves no negatives, and a model predicting building everywhere scores a mean IoU of 1.0.

It is also 0.448 m per pixel rather than 2.91-4.77, which is the scale the model trains
at and far nearer the scale a survey is flown at. This is the closest thing to an honest
answer to "does the engine work in India" that the available data can give.

WHAT IT STILL IS NOT: 49 tiles of one city, from satellite/aerial capture rather than a
drone at 60 m, and one city cannot speak for a country. A good number here means the
engine works on Gorakhpur.

    python -m training.evaluate_gorakhpur --checkpoint runs/kernel_v3/best.pt
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.semantic_tiles import IGNORE_INDEX  # noqa: E402

GORAKHPUR_GROUP = "openearthmap::gorakhpur"


def confusion(predicted: np.ndarray, expected: np.ndarray, classes: int) -> np.ndarray:
    """Counts over pixels that carry a label. Ignored pixels are dropped, correctly.

    On an exhaustively annotated tile almost nothing is ignored, so unlike the SpaceNet 7
    holdout this leaves the negatives intact and the metric can register a false positive.
    """
    valid = (expected != IGNORE_INDEX) & (expected < classes)
    if not np.any(valid):
        return np.zeros((classes, classes), dtype=np.int64)
    truth = expected[valid].astype(np.int64)
    guess = np.clip(predicted[valid].astype(np.int64), 0, classes - 1)
    return np.bincount(
        truth * classes + guess, minlength=classes * classes
    ).reshape(classes, classes)


def scores_from(matrix: np.ndarray, names: list[str]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    ious: list[float] = []
    for index, name in enumerate(names):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index].sum()) - true_positive
        false_positive = int(matrix[:, index].sum()) - true_positive
        union = true_positive + false_positive + false_negative
        present = (true_positive + false_negative) > 0
        iou = float(true_positive / union) if union else 0.0
        per_class[name] = {
            "iou": iou,
            "precision": (
                float(true_positive / (true_positive + false_positive))
                if (true_positive + false_positive) else 0.0
            ),
            "recall": (
                float(true_positive / (true_positive + false_negative))
                if (true_positive + false_negative) else 0.0
            ),
            "labelled_px": true_positive + false_negative,
            "predicted_px": true_positive + false_positive,
            # A class that does not occur here has no score to give. Averaging a zero for
            # it would understate the model for a reason that has nothing to do with it.
            "present_in_holdout": bool(present),
        }
        if present:
            ious.append(iou)

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    return {
        "per_class": per_class,
        "mean_iou_present_classes": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": float(correct / total) if total else 0.0,
        "scored_pixels": total,
        "classes_present": [n for n in names if per_class[n]["present_in_holdout"]],
    }


def main(argv: list[str] | None = None) -> int:
    import torch
    import yaml

    from training.semantic_tiles import SemanticTileDataset

    parser = argparse.ArgumentParser(prog="python -m training.evaluate_gorakhpur")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--corpus", type=Path,
        default=REPO_ROOT / "training" / "data" / "prepared" / "gorakhpur_holdout" / "corpus.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from training.shared_semantic_model import build_dinov2_vitb14_upernet

    architecture = dict(
        state.get("architecture") or (state.get("config") or {}).get("architecture") or {}
    )
    schema = state.get("schema") or (state.get("config") or {}).get("schema") or {}
    names = [str(c["name"]) for c in (schema.get("classes") or [])]
    if not names:
        raise SystemExit("Checkpoint carries no class schema; cannot name the classes.")

    encoder_checkpoint = architecture.get(
        "encoder_checkpoint", "models/pretrained/dinov2_vitb14_pretrain.pth"
    )
    encoder_source = architecture.get("encoder_source", "facebookresearch/dinov2")
    # dinov2_source has to travel with source. The checkpoint from the Kaggle run records
    # 'facebookresearch/dinov2', which is a hub spec and not a directory; passing source
    # alone left the builder resolving it against the repository root and failing on a
    # hubconf.py that was never going to be there.
    local_encoder = (REPO_ROOT / encoder_source).is_dir()
    model = build_dinov2_vitb14_upernet(
        REPO_ROOT / encoder_checkpoint if not Path(encoder_checkpoint).is_absolute()
        else Path(encoder_checkpoint),
        len(names),
        dinov2_source=str(REPO_ROOT / encoder_source) if local_encoder else encoder_source,
        source="local" if local_encoder else "github",
    )
    model.load_state_dict(state.get("model_state") or state["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The checkpoint is 380 MB and stays referenced by `state` long after the weights are
    # in the model, which on a machine already short of memory is the difference between
    # scoring and dying on a 12 MB image allocation. Read what is still needed off it
    # first, then let it go.
    trained_config = dict(state.get("config") or {})
    del state
    gc.collect()

    # Scored at the scale the model was trained at, for the same reason the other holdout
    # is: cropping a fixed pixel count would show the model imagery it never saw.
    trained_with = trained_config.get("training") or {}
    target_gsd = trained_with.get("target_gsd")
    if target_gsd is None:
        config_path = REPO_ROOT / "training" / "configs" / "shared_semantic_dinov2_vitb14.yaml"
        if config_path.is_file():
            target_gsd = ((yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
                          .get("training") or {}).get("target_gsd")
    print(f"scoring at {float(target_gsd):.2f} m/px" if target_gsd else "scoring unharmonised",
          flush=True)

    dataset = SemanticTileDataset(
        args.corpus, "test", tile_size=518, augment=False, target_gsd=target_gsd,
    )
    dataset.samples = [
        s for s in dataset.samples if s.get("group") == GORAKHPUR_GROUP
    ]
    if not dataset.samples:
        raise SystemExit(
            f"No {GORAKHPUR_GROUP} samples in the test split of {args.corpus}. "
            "The holdout has to be pinned there or this measures nothing."
        )

    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    with torch.no_grad():
        for index in range(len(dataset.samples)):
            item = dataset[index]
            image = torch.as_tensor(item["image"]).unsqueeze(0).to(device).float()
            logits = model(image)
            predicted = torch.argmax(logits, dim=1)[0].cpu().numpy()
            matrix += confusion(predicted, np.asarray(item["mask"]), len(names))
            if (index + 1) % 10 == 0:
                print(f"  scored {index + 1}/{len(dataset.samples)} tiles", flush=True)

    report = {
        **scores_from(matrix, names),
        "site": "Gorakhpur, Uttar Pradesh, India",
        "tiles": len(dataset.samples),
        "metric": "multiclass_exhaustive",
        "checkpoint": str(args.checkpoint),
        "reading_note": (
            "Gorakhpur is exhaustively annotated, so unlike the SpaceNet 7 holdout a "
            "false positive is counted and this metric CAN fail. It is 0.448 m per pixel "
            "-- the scale the model trains at -- and is the first India evidence covering "
            "road, vegetation, water and bare land rather than buildings alone. "
            "It remains 49 tiles of ONE city, captured from above by satellite or aircraft "
            "rather than by a drone at 60 m. A good number here means the engine works on "
            "Gorakhpur; it does not make it verified for India, and it says nothing about "
            "sites whose built form differs."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
