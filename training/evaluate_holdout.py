"""Score a semantic checkpoint on the India holdout, with a metric that can fail.

The holdout is four SpaceNet 7 tiles over Mumbai, Kalyan/Thane, Tirupati and
Vijayawada. SpaceNet 7 labels buildings and nothing else, so every non-building pixel
carries background_id 255 -- IGNORE_INDEX -- and 96.7 per cent of each tile is ignored.

That is the right choice for the training loss. In a six-class schema you cannot assert
that an unlabelled pixel is road rather than vegetation or water, so declining to score
it is honest.

It is the wrong choice for evaluation, and badly so. The confusion matrix drops ignored
pixels from prediction and target alike, which leaves no negatives, which means false
positives are never counted. A model predicting BUILDING ON EVERY PIXEL scores a mean
IoU of 1.0 -- measured, not hypothesised. The holdout gates six registry rows and could
not detect the most likely failure of a segmentation model.

So the building class is scored BINARY here: labelled building against everything else,
with unlabelled treated as not-building. That is defensible in a way the multiclass
version is not -- SpaceNet 7's annotators looked at the whole tile and drew every
building they saw, so the absence of a polygon is evidence of absence for buildings,
even though it says nothing about what else is there.

    python -m training.evaluate_holdout --checkpoint training/runs/shared_semantic/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.semantic_tiles import IGNORE_INDEX  # noqa: E402

BUILDING_CLASS_ID = 1


def binary_building_scores(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Building IoU, precision and recall with unlabelled pixels counted as negatives.

    This is the whole point of the module. Under the multiclass metric a model that
    answers "building" everywhere is unfalsifiable, because the pixels it is wrong about
    are exactly the ones that get dropped. Counting them turns the same prediction into
    the failure it is.
    """
    predicted = np.asarray(predicted).reshape(-1)
    expected = np.asarray(expected).reshape(-1)
    if predicted.shape != expected.shape:
        raise ValueError("Prediction and label shapes differ.")

    # Unlabelled becomes not-building. The annotators saw the whole tile, so a missing
    # polygon is evidence of absence FOR BUILDINGS -- and for nothing else.
    truth = expected == BUILDING_CLASS_ID
    guess = predicted == BUILDING_CLASS_ID

    true_positive = int(np.count_nonzero(truth & guess))
    false_positive = int(np.count_nonzero(~truth & guess))
    false_negative = int(np.count_nonzero(truth & ~guess))
    union = true_positive + false_positive + false_negative

    return {
        "building_iou": float(true_positive / union) if union else 0.0,
        "building_precision": (
            float(true_positive / (true_positive + false_positive))
            if (true_positive + false_positive) else 0.0
        ),
        "building_recall": (
            float(true_positive / (true_positive + false_negative))
            if (true_positive + false_negative) else 0.0
        ),
        "true_positive_px": true_positive,
        "false_positive_px": false_positive,
        "false_negative_px": false_negative,
        "predicted_building_fraction": float(np.count_nonzero(guess) / guess.size),
        "labelled_building_fraction": float(np.count_nonzero(truth) / truth.size),
    }


def holdout_report(scores: dict[str, float], *, sites: list[str]) -> dict[str, Any]:
    """Wrap the numbers in what a reader needs to not over-read them."""
    return {
        **scores,
        "sites": sites,
        "metric": "binary_building",
        "reading_note": (
            "Buildings only, scored binary: unlabelled pixels count as not-building. "
            "SpaceNet 7 annotates buildings and nothing else, so this says NOTHING about "
            "road, vegetation, water or bare land performance on Indian sites -- those "
            "remain unmeasured. The imagery is a Planet mosaic measured at 2.91-4.77 m per "
            "pixel -- this note previously said 0.5 m, which was wrong by a factor of "
            "eight -- so it speaks for building extraction at satellite scale and says "
            "little about a drone survey flown at 60 m. Tiles are resampled to the training "
            "target_gsd before scoring, so a score is comparable only with others "
            "measured the same way. "
            "Compare predicted_building_fraction against labelled_building_fraction: a "
            "model predicting far more building than exists is over-segmenting, which "
            "the multiclass metric on this holdout cannot show at all."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.evaluate_holdout")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=REPO_ROOT / "training/data/prepared/shared_semantic/corpus.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    import torch

    from training.semantic_corpus import INDIA_HOLDOUT_GROUPS
    from training.semantic_tiles import SemanticTileDataset

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    held = [s for s in corpus["samples"] if s.get("group") in INDIA_HOLDOUT_GROUPS]
    if not held:
        raise SystemExit(
            "No India holdout samples in this corpus. The pins in semantic_corpus.py "
            "did not match any group, so there is nothing to report."
        )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # The encoder is built exactly as training built it, from the same retained
    # Apache-2.0 checkpoint. A holdout scored against a differently constructed model is
    # not a holdout score.
    from training.shared_semantic_model import build_dinov2_vitb14_upernet  # noqa: PLC0415

    # The trainer records the architecture at the top level of the checkpoint; the
    # config it was launched with is kept beside it.
    architecture = dict(
        state.get("architecture")
        or (state.get("config") or {}).get("architecture")
        or {}
    )
    encoder_checkpoint = architecture.get(
        "encoder_checkpoint", "models/pretrained/dinov2_vitb14_pretrain.pth"
    )
    encoder_source = architecture.get("encoder_source", "facebookresearch/dinov2")
    model = build_dinov2_vitb14_upernet(
        REPO_ROOT / encoder_checkpoint if not Path(encoder_checkpoint).is_absolute()
        else encoder_checkpoint,
        len((state.get("schema") or {}).get("classes") or state.get("classes") or []) or 6,
        dinov2_source=str(REPO_ROOT / encoder_source)
        if (REPO_ROOT / encoder_source).is_dir() else encoder_source,
        source="local" if (REPO_ROOT / encoder_source).is_dir() else "github",
    )
    model.load_state_dict(state.get("model_state") or state["model"])
    model.eval()

    # ViT-B/14 over 518px tiles is minutes per tile on a CPU and seconds on a GPU. The
    # scores are identical; only the wait is not.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate at the scale the model was TRAINED at. The trainer resamples every source
    # to config training.target_gsd, and cropping a fixed pixel count here instead would
    # show the model 4 m/px tiles it never saw -- scoring the harmonisation as if it were
    # a defect. A train/eval mismatch does not fail; it just reports the wrong number,
    # which is worse.
    # Taken from the CHECKPOINT rather than the config file on disk: the checkpoint
    # records what this model was actually trained with, and the file may have moved on
    # since. Falling back to the file only when the checkpoint predates the field.
    trained_with = (state.get("config") or {}).get("training") or {}
    target_gsd = trained_with.get("target_gsd")
    if target_gsd is None:
        import yaml  # noqa: PLC0415

        config_path = REPO_ROOT / "training" / "configs" / "shared_semantic_dinov2_vitb14.yaml"
        if config_path.is_file():
            on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            target_gsd = (on_disk.get("training") or {}).get("target_gsd")
            if target_gsd:
                print(
                    "checkpoint records no target_gsd; using the config file's "
                    f"{float(target_gsd):.2f} m/px. If this model predates the scale fix "
                    "the score below is not comparable.",
                    flush=True,
                )
    if target_gsd:
        print(f"holdout tiles harmonised to {float(target_gsd):.2f} m/px", flush=True)
    dataset = SemanticTileDataset(
        args.corpus, "test", tile_size=518, augment=False, target_gsd=target_gsd,
    )
    # The test split is not the holdout. It carries sixteen groups across two corpora,
    # and scoring all of them while the report names four Indian sites attributes a
    # number to a place that did not produce it -- the corpus is pinned by group, so the
    # dataset has to be too.
    dataset.samples = [
        sample for sample in dataset.samples
        if sample.get("group") in INDIA_HOLDOUT_GROUPS
    ]
    if not dataset.samples:
        raise SystemExit(
            "The test split contains none of the India holdout groups, so there is "
            "nothing to score them on."
        )

    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            logits = model(item["image"].unsqueeze(0).to(device))
            predictions.append(logits.argmax(dim=1).squeeze(0).cpu().numpy())
            labels.append(item["mask"].numpy())

    scores = binary_building_scores(np.concatenate([p.reshape(-1) for p in predictions]),
                                    np.concatenate([m.reshape(-1) for m in labels]))
    report = holdout_report(scores, sites=sorted(INDIA_HOLDOUT_GROUPS))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
