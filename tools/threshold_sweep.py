"""Find the decision threshold that actually maximises IoU on held-out data.

A segmentation model emits a probability per pixel; the threshold that turns those into
a mask is a separate choice, and it is usually left at whatever the code happened to
default to. That default is rarely the best one. This model was registered at 0.25 while
reporting recall 0.846 against precision 0.682 -- a 0.16 gap that says it is calling too
many pixels cracks, which is exactly what too low a threshold does.

The sweep is run on the validation split rather than the test split, because a threshold
chosen on test data is a parameter fitted to the test set, and the resulting score stops
being an estimate of anything. Whatever this picks is then reported against test
separately.

Usage:
    python tools/threshold_sweep.py --model models/structural/crack_segmentation.onnx \
        --split val --samples 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

# ImageNet statistics, matching how the model was trained.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_THRESHOLDS = [0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def find_mask(image_path: str) -> str | None:
    """Locate the mask belonging to an image, whatever extension it uses.

    Resolved through path components rather than string replacement. On Windows glob
    returns mixed separators -- ``.../val/images\name.png`` -- so replacing "/images/"
    or os.sep + "images" + os.sep silently matches neither, leaves the path unchanged,
    and hands back the image itself. Scoring a model against its own input produces a
    confident, meaningless number rather than an error.
    """
    path = Path(image_path)
    parts = list(path.parts)
    if "images" not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index("images")] = "masks"
    candidate = Path(*parts)

    if candidate.exists():
        return str(candidate)
    matches = sorted(candidate.parent.glob(candidate.stem + ".*"))
    return str(matches[0]) if matches else None


def sweep(model_path: str, split: str, samples: int, size: int,
          thresholds: list[float], output: str) -> dict:
    import onnxruntime as ort

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"provider: {session.get_providers()[0]}", flush=True)

    images = sorted(glob.glob(f"training/data/prepared/crack_seg/{split}/images/*"))
    if not images:
        raise SystemExit(f"No images found for split {split!r}.")

    # Evenly spaced rather than the first N: consecutive frames in a survey are
    # correlated, and the head of a sorted list is not a sample of anything.
    step = max(1, len(images) // samples)
    chosen = images[::step][:samples]
    print(f"scoring {len(chosen)} of {len(images)} {split} images at {size}px", flush=True)

    intersection = {t: 0 for t in thresholds}
    union = {t: 0 for t in thresholds}
    true_pos = {t: 0 for t in thresholds}
    false_pos = {t: 0 for t in thresholds}
    false_neg = {t: 0 for t in thresholds}
    scored = 0
    gt_fractions: list[float] = []

    for index, image_path in enumerate(chosen):
        mask_path = find_mask(image_path)
        if mask_path is None:
            continue
        image = cv2.imread(image_path)
        truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            continue

        image = cv2.resize(image, (size, size))
        truth = cv2.resize(truth, (size, size), interpolation=cv2.INTER_NEAREST) > 127

        # Cracks occupy a few percent of a frame. A ground truth claiming half the image
        # is not a crack mask -- it is the wrong file, or an inverted one, and scoring
        # against it would produce a confident, meaningless number.
        foreground = float(truth.mean())
        if foreground > 0.25:
            raise SystemExit(
                f"{Path(mask_path).name} has {foreground:.1%} foreground. Crack masks "
                "are a few percent; this pairing or polarity is wrong. Refusing to "
                "score against it."
            )
        gt_fractions.append(foreground)

        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((tensor - MEAN) / STD).transpose(2, 0, 1)[None]
        probability = session.run(None, {input_name: tensor})[0][0, 0]

        for threshold in thresholds:
            predicted = probability > threshold
            intersection[threshold] += int(np.logical_and(predicted, truth).sum())
            union[threshold] += int(np.logical_or(predicted, truth).sum())
            true_pos[threshold] += int(np.logical_and(predicted, truth).sum())
            false_pos[threshold] += int(np.logical_and(predicted, ~truth).sum())
            false_neg[threshold] += int(np.logical_and(~predicted, truth).sum())

        scored += 1
        if scored % 25 == 0:
            print(f"  {scored}/{len(chosen)}", flush=True)

    rows = []
    for threshold in thresholds:
        iou = intersection[threshold] / max(union[threshold], 1)
        precision = true_pos[threshold] / max(true_pos[threshold] + false_pos[threshold], 1)
        recall = true_pos[threshold] / max(true_pos[threshold] + false_neg[threshold], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({"threshold": threshold, "iou": iou, "precision": precision,
                     "recall": recall, "f1": f1})

    mean_gt = sum(gt_fractions) / max(len(gt_fractions), 1)
    print(f"mean ground-truth foreground: {mean_gt:.4f} "
          f"(training reported 0.0344 on this split)", flush=True)

    best = max(rows, key=lambda r: r["iou"])
    report = {
        "model": model_path,
        "split": split,
        "images_scored": scored,
        "image_size": size,
        "rows": rows,
        "best_threshold": best["threshold"],
        "best_iou": best["iou"],
        "note": (
            "Thresholds swept on the validation split. Choosing one on test data would "
            "fit a parameter to the test set and stop the score estimating anything."
        ),
    }

    print(f"\n{'thr':>5} {'IoU':>8} {'prec':>8} {'recall':>8} {'F1':>8}")
    for row in rows:
        marker = "  <-- best" if row["threshold"] == best["threshold"] else ""
        print(f"{row['threshold']:>5} {row['iou']:>8.4f} {row['precision']:>8.4f} "
              f"{row['recall']:>8.4f} {row['f1']:>8.4f}{marker}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nbest IoU {best['iou']:.4f} at threshold {best['threshold']}")
    print(f"written to {output}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="python tools/threshold_sweep.py")
    parser.add_argument("--model", default="models/structural/crack_segmentation.onnx")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", default="models/metrics/crack_threshold_sweep.json")
    args = parser.parse_args()

    sweep(args.model, args.split, args.samples, args.size,
          DEFAULT_THRESHOLDS, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
