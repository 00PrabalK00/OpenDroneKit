"""Measure the open-vocabulary models zero-shot on labelled open test sets in ODK_TESTDATA_DIR.

    python tools/eval_open_vocab.py --model gdino --graph <onnx> --out <dir> [--set thermal|rdd|all]
    python tools/eval_open_vocab.py --model florence --out <dir> [--set ...]

Sets (see odk-testdata/INDEX.md):

* ``thermal_pv_panel_detection_mavic3t``: all 353 images (train+val+test of the
  publisher's split; none of it was used to train either model), YOLO class 0 = PV
  panel. Prompt "solar panel."
* ``road_damage_rdd2022_subset``: 200 images, Pascal VOC. Prompt "pothole." scored
  against D40; prompt "crack." against D00, D10 and D20 merged. Other labels (Repair,
  D44, D01, Block crack) are ground truth for neither prompt, so a box on one of them
  counts as a false positive.

Grounding DINO is run once per image and prompt at a low box threshold (0.05) so the
whole precision-recall curve is available for AP50; precision and recall at the
default thresholds (box 0.35) are the same detections filtered by score, which is exact
because the post-processing scores each query independently. Florence-2 gives boxes
without scores, so its "AP50" is a single operating point (precision x recall) and is
reported as such.

Writes ``<out>/<set>__<model tag>.json`` (summary) and ``.jsonl`` (per-image boxes).
These are zero-shot numbers on a named split with N stated, and are parity/accuracy
facts about these files only -- not a calibration of the score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOW_BOX_THRESHOLD = 0.05
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25


def testdata() -> Path:
    return Path(os.environ.get("ODK_TESTDATA_DIR", "../odk-testdata"))


def thermal_items() -> list[tuple[Path, dict[str, list[list[float]]]]]:
    base = testdata() / "thermal_pv_panel_detection_mavic3t" / "Thermal PV Panel Detection Dataset for UAV Inspection"
    items = []
    for split in ("train", "valid", "val", "test"):
        for img in sorted((base / split / "images").glob("*.jpg")) if (base / split / "images").is_dir() else []:
            h, w = cv2.imread(str(img)).shape[:2]
            label = base / split / "labels" / (img.stem + ".txt")
            boxes = []
            if label.is_file():
                for line in label.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] == "0":
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            items.append((img, {"solar panel": boxes}))
    return items


RDD_CLASSES = {"pothole": {"D40"}, "crack": {"D00", "D10", "D20"}}


def rdd_items() -> list[tuple[Path, dict[str, list[list[float]]]]]:
    base = testdata() / "road_damage_rdd2022_subset"
    items = []
    for country in ("China_Drone", "India"):
        for xml in sorted((base / country / "train" / "annotations" / "xmls").glob("*.xml")) \
                or sorted((base / country / "train" / "annotations").glob("*.xml")):
            img = base / country / "train" / "images" / (xml.stem + ".jpg")
            if not img.is_file():
                continue
            gt: dict[str, list[list[float]]] = {k: [] for k in RDD_CLASSES}
            for obj in ET.parse(xml).getroot().iter("object"):
                name = (obj.findtext("name") or "").strip()
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                box = [float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")]
                for prompt, codes in RDD_CLASSES.items():
                    if name in codes:
                        gt[prompt].append(box)
            items.append((img, gt))
    return items


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of each box in a (N,4) with each in b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: np.clip(r[:, 2] - r[:, 0], 0, None) * np.clip(r[:, 3] - r[:, 1], 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(preds: list[dict], gts: list[list[float]], thr: float = 0.5) -> list[tuple[float, bool]]:
    """(score, true positive?) per prediction, greedy by score, VOC rule."""
    order = sorted(preds, key=lambda p: -p["score"])
    g = np.asarray(gts, dtype=np.float64).reshape(-1, 4)
    used = np.zeros(len(g), dtype=bool)
    out = []
    if not order:
        return out
    ious = iou(np.asarray([p["bbox"] for p in order], dtype=np.float64), g)
    for i, p in enumerate(order):
        tp = False
        if len(g):
            cand = np.where(~used, ious[i], -1.0)
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                used[j] = True
                tp = True
        out.append((p["score"], tp))
    return out


def ap_from(records: list[tuple[float, bool]], n_gt: int) -> float:
    """All-point interpolated AP (VOC 2010+)."""
    if n_gt == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1 if r[1] else 0 for r in records])
    fp = np.cumsum([0 if r[1] else 1 for r in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def summarise(per_image: list[dict], prompt: str, score_floor: float | None) -> dict:
    recs: list[tuple[float, bool]] = []
    n_gt = 0
    for row in per_image:
        preds = [p for p in row["preds"].get(prompt, []) if score_floor is None or p["score"] > score_floor]
        gts = row["gt"].get(prompt, [])
        n_gt += len(gts)
        recs.extend(match(preds, gts))
    tps = sum(1 for _, t in recs if t)
    return {"n_gt": n_gt, "n_pred": len(recs), "tp": tps,
            "precision": (tps / len(recs)) if recs else 0.0,
            "recall": (tps / n_gt) if n_gt else float("nan"),
            "ap50": ap_from(recs, n_gt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdino", "florence"], required=True)
    ap.add_argument("--graph", default="", help="Grounding DINO ONNX graph (int8 or fp32)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", default="all", choices=["thermal", "rdd", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="keep images already in <out>/<set>__<tag>.jsonl and run the rest")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    weights = Path(os.environ.get("ODK_MODEL_WEIGHTS_DIR", "../odk-weights"))

    if args.model == "gdino":
        from core.open_vocab import GroundingDinoOnnx
        d = weights / "open_vocab_grounding_dino_tiny"
        runner = GroundingDinoOnnx(Path(args.graph or d / "open_vocab_grounding_dino_tiny.onnx"),
                                   d / "open_vocab_grounding_dino_tiny__tokenizer.json",
                                   d / "open_vocab_grounding_dino_tiny__preprocessor_config.json",
                                   threads=args.threads)

        def predict(im, prompt):
            hits, _ = runner.detect(im, [prompt], box_threshold=LOW_BOX_THRESHOLD,
                                    text_threshold=DEFAULT_TEXT_THRESHOLD, tile=0)
            return [{"score": h["score"], "bbox": h["bbox"]} for h in hits]
        floor = DEFAULT_BOX_THRESHOLD
    else:
        from core.model_files import role_of
        from core.vision_language import FlorenceOnnx
        key = "vision_language_florence2_base_ft"
        d = weights / key
        runner = FlorenceOnnx({role_of(p.name, key): p for p in d.iterdir() if p.is_file()},
                              threads=args.threads)

        def predict(im, prompt):
            r = runner.run(im, "<CAPTION_TO_PHRASE_GROUNDING>", prompt)
            return [{"score": 1.0, "bbox": reg["bbox"]} for reg in r["regions"]]
        floor = None

    tag = args.tag or args.model
    sets = {"thermal": thermal_items, "rdd": rdd_items}
    for name in (["thermal", "rdd"] if args.set == "all" else [args.set]):
        items = sets[name]()
        if args.limit:
            items = items[: args.limit]
        per_image = []
        started = time.time()
        log_path = out / f"{name}__{tag}.jsonl"
        done: dict[str, dict] = {}
        if args.resume and log_path.is_file():
            # A shared machine can kill a long run (out of memory); keep finished images.
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    done[row["image"]] = row
                except (ValueError, KeyError):
                    break
        with log_path.open("w", encoding="utf-8") as log:
            for i, (img, gt) in enumerate(items):
                if img.name in done:
                    per_image.append(done[img.name])
                    log.write(json.dumps(done[img.name]) + "\n")
                    continue
                im = cv2.imread(str(img))
                preds = {prompt: predict(im, prompt) for prompt in gt}
                row = {"image": img.name, "gt": gt, "preds": preds}
                per_image.append(row)
                log.write(json.dumps(row) + "\n")
                if i % 25 == 0:
                    print(f"{name} {tag} {i}/{len(items)} {time.time() - started:.0f}s", flush=True)
        summary = {
            "set": name, "model": args.model, "tag": tag, "graph": str(args.graph),
            "n_images": len(per_image), "seconds": round(time.time() - started, 1),
            "label": ("zero-shot, Mavic 3T thermal palette images, N=%d" % len(per_image)) if name == "thermal"
                     else ("zero-shot, RDD2022 subset (120 China_Drone + 80 India), N=%d" % len(per_image)),
            "per_prompt": {},
        }
        for prompt in items[0][1] if items else []:
            summary["per_prompt"][prompt] = {
                "curve": summarise(per_image, prompt, None),
                "at_default_thresholds": summarise(per_image, prompt, floor) if floor else None,
            }
        (out / f"{name}__{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
