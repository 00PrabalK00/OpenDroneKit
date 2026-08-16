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
            "remain unmeasured. It is also 0.5 m satellite imagery, not drone capture, "
            "so it speaks for building extraction at satellite scale rather than for a "
            "survey flown at 60 m. "
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
    from training.shared_semantic_model import build_model  # noqa: PLC0415

    model = build_model(len(state.get("classes", [])) or 6)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = SemanticTileDataset(args.corpus, "test", tile_size=518, augment=False)
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            logits = model(item["image"].unsqueeze(0))
            predictions.append(logits.argmax(dim=1).squeeze(0).numpy())
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
