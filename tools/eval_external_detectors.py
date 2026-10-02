"""Measure the shipped detectors on openly licensed external data, through the app's code.

Every number this writes comes from running the installed model files through the same
functions the application calls (core.detection / core.model_runner) on real, openly
licensed imagery with its published annotations. Nothing is retrained here, nothing is
synthesised, and where a dataset has no labels for a question the output says
"unlabelled" rather than inventing a score.

Datasets (provenance in each folder's PROVENANCE.md under ODK_TESTDATA_DIR):

* ``road``  -- road_damage_rdd2022_subset: RDD2022 (CC BY 4.0), 120 China_Drone UAV
  images + 80 India vehicle images, Pascal-VOC D00/D10/D20 cracks and D40 potholes.
  There is NO road-damage model in the registry (no roads run produced weights), so
  this measures the closest shipped models on the crack classes only:
  structural_multiclass_detector's ``Crack`` class (box AP), crack_presence_classifier
  (tile and image triage) and crack_segmentation (mask vs boxes). Potholes (D40) have
  no model at all; Spallation is scored against them as an explicitly exploratory
  proxy, not as a pothole detector.
* ``pv``    -- thermal_pv_panel_detection_mavic3t (CC BY 4.0): 353 DJI Mavic 3T thermal
  frames with YOLO panel boxes and NO defect labels. No shipped model localises panels,
  so panel detection cannot be measured; the whole frame is checked for an honest
  refusal, and solar_thermal_anomaly_classifier is run on every labelled panel crop to
  report what it says (distribution and orientation stability), not an accuracy.
* ``bridge`` -- bridge_inspection_h20t_boga_cayi (CC BY 4.0): 81 DJI H20T wide RGB
  frames (4056x3040, nadir ~50 m). No defect labels exist, so this is qualitative:
  counts, overlays and the letterbox-vs-stretch effect on non-square frames.
* ``structural-split`` -- the structural_det test split the detector was validated on
  (ODK_TRAINING_DATA_DIR/prepared/structural_det/test). Re-measures mAP through the
  app's decoder (does the integration reproduce the published figure?) and measures
  letterbox against stretch on 4:3 crops of the same labelled images.

Usage::

    set ODK_TESTDATA_DIR=E:/.../odk-testdata
    set ODK_MODEL_WEIGHTS_DIR=<folder holding the registry's weight paths>
    python -m tools.eval_external_detectors road  --out docs/holdout/external_rdd2022_detectors.json
    python -m tools.eval_external_detectors pv    --out docs/holdout/external_pv_mavic3t_thermal.json
    python -m tools.eval_external_detectors bridge --out docs/holdout/external_bridge_h20t_detectors.json
    python -m tools.eval_external_detectors structural-split --out docs/holdout/structural_det_app_decoder.json

Overlays and per-image rows go to ``<ODK_TESTDATA_DIR>/_outputs/detectors/<area>/``
(override with ``--artifacts``), never into the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RDD_FOLDER = "road_damage_rdd2022_subset"
PV_FOLDER = "thermal_pv_panel_detection_mavic3t"
PV_INNER = "Thermal PV Panel Detection Dataset for UAV Inspection"
BRIDGE_FOLDER = "bridge_inspection_h20t_boga_cayi"

#: RDD2022's label map (label_map.pbtxt) -- the four core classes.
RDD_CORE = ("D00", "D10", "D20", "D40")
RDD_CRACKS = ("D00", "D10", "D20")
RDD_NAMES = {"D00": "longitudinal crack", "D10": "transverse crack",
             "D20": "alligator crack", "D40": "pothole"}


# ── datasets ─────────────────────────────────────────────────────────────────

def testdata_root() -> Path:
    root = os.environ.get("ODK_TESTDATA_DIR", "").strip()
    if not root:
        raise SystemExit("Set ODK_TESTDATA_DIR to the odk-testdata folder (see its INDEX.md).")
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(f"ODK_TESTDATA_DIR={root} is not a folder.")
    return path


def parse_voc(xml_path: Path) -> dict[str, Any]:
    """One Pascal-VOC annotation: image size and (name, xyxy) objects, as written."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None else 0
    height = int(float(size.findtext("height"))) if size is not None else 0
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xyxy = tuple(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((str(obj.findtext("name") or "").strip(), xyxy))
    return {"filename": root.findtext("filename") or xml_path.stem + ".jpg",
            "width": width, "height": height, "objects": objects}


def rdd_samples(root: Path) -> list[dict[str, Any]]:
    """Every image of the RDD subset with its parsed annotation and subset name."""
    samples = []
    for subset in ("China_Drone", "India"):
        base = root / RDD_FOLDER / subset / "train"
        for xml_path in sorted((base / "annotations" / "xmls").glob("*.xml")):
            ann = parse_voc(xml_path)
            image = base / "images" / ann["filename"]
            if not image.is_file():
                image = base / "images" / (xml_path.stem + ".jpg")
            samples.append({"subset": subset, "image": image, **ann})
    return samples


def voc_to_yolo(samples: Iterable[dict[str, Any]], prepared_root: Path, name: str,
                classes: tuple[str, ...] = RDD_CORE) -> dict[str, Any]:
    """Lay the RDD subset out as ``prepared/<name>/test/{images,labels}`` YOLO files.

    This is the input layout core/custom_model_training.py and the model tests read
    (ODK_TRAINING_DATA_DIR). Only the four core classes are written, in label-map order;
    the non-core labels (Repair, D44, D01, Block crack) are counted and left out, as the
    RDD2022 challenge itself does.
    """
    dest = prepared_root / "prepared" / name / "test"
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    written = Counter()
    dropped = Counter()
    images = 0
    for sample in samples:
        width, height = sample["width"], sample["height"]
        if not width or not height:
            img = cv2.imread(str(sample["image"]))
            height, width = img.shape[:2]
        stem = f"{sample['subset']}_{Path(sample['image']).stem}"
        shutil.copy2(sample["image"], dest / "images" / f"{stem}.jpg")
        lines = []
        for label, (x1, y1, x2, y2) in sample["objects"]:
            if label not in classes:
                dropped[label] += 1
                continue
            cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{classes.index(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            written[label] += 1
        (dest / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                     encoding="utf-8")
        images += 1
    (prepared_root / "prepared" / name / "data.yaml").write_text(
        "path: .\ntest: test/images\nnc: %d\nnames:\n%s" % (
            len(classes), "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))),
        encoding="utf-8")
    return {"folder": str(dest), "images": images, "boxes_written": dict(written),
            "labels_dropped_non_core": dict(dropped)}


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, tuple]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((cls, ((cx - bw / 2) * width, (cy - bh / 2) * height,
                            (cx + bw / 2) * width, (cy + bh / 2) * height)))
    return boxes


# ── metrics ──────────────────────────────────────────────────────────────────

def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(predictions: list[tuple[str, float, tuple]],
                      truths: dict[str, list[tuple]],
                      ignore: dict[str, list[tuple]] | None = None,
                      iou_threshold: float = 0.5) -> dict[str, Any]:
    """VOC all-point-interpolated AP of scored boxes against ground truth.

    ``predictions`` are (image_id, score, xyxy). A prediction that matches no unclaimed
    truth box but does overlap an ``ignore`` box at the IoU threshold (for example a
    transverse-crack box when scoring longitudinal cracks with a class-agnostic crack
    detector) is neither a true nor a false positive.
    """
    ignore = ignore or {}
    n_truth = sum(len(v) for v in truths.values())
    claimed = {k: [False] * len(v) for k, v in truths.items()}
    tp: list[int] = []
    fp: list[int] = []
    for image_id, _score, box in sorted(predictions, key=lambda p: -p[1]):
        candidates = truths.get(image_id, [])
        best, best_j = 0.0, -1
        for j, truth in enumerate(candidates):
            overlap = iou(box, truth)
            if overlap > best:
                best, best_j = overlap, j
        if best >= iou_threshold and not claimed[image_id][best_j]:
            claimed[image_id][best_j] = True
            tp.append(1)
            fp.append(0)
        elif any(iou(box, other) >= iou_threshold for other in ignore.get(image_id, [])):
            continue
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp) if tp else np.zeros(0)
    fp_cum = np.cumsum(fp) if fp else np.zeros(0)
    recall = tp_cum / max(n_truth, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    if n_truth == 0:
        ap = float("nan")
    elif len(tp) == 0:
        ap = 0.0
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    true_pos = int(tp_cum[-1]) if len(tp_cum) else 0
    false_pos = int(fp_cum[-1]) if len(fp_cum) else 0
    return {
        "ap": None if ap != ap else round(ap, 4),
        "truth_boxes": n_truth,
        "predictions_scored": true_pos + false_pos,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "precision": round(true_pos / max(true_pos + false_pos, 1), 4),
        "recall": round(true_pos / max(n_truth, 1), 4),
    }


def binary_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": None if precision is None else round(precision, 4),
           "recall": None if recall is None else round(recall, 4),
           "specificity": None if specificity is None else round(specificity, 4)}
    if recall is not None and specificity is not None:
        out["balanced_accuracy"] = round((recall + specificity) / 2.0, 4)
    return out


# ── model plumbing (the app's own functions) ─────────────────────────────────

def model_row(key: str) -> dict[str, Any]:
    from core.models import get_model_spec, model_identity, model_status

    status = model_status(key)
    spec = get_model_spec(key)
    if not status.get("exists"):
        raise SystemExit(
            f"{key}: weights not found (looked beside the registry and in "
            f"ODK_MODEL_WEIGHTS_DIR={os.environ.get('ODK_MODEL_WEIGHTS_DIR', '')!r}).")
    identity = model_identity(key)
    recorded = spec.sha256 if spec else ""
    return {
        "model_key": key,
        "file": Path(status["path"]).name,
        "sha256_measured": identity["model_sha256"],
        "sha256_registered": recorded,
        "sha256_matches_registry": bool(recorded) and recorded == identity["model_sha256"],
    }


def yolo_hits(image_bgr: np.ndarray, key: str, *, score_threshold: float | None = None,
              letterbox: bool = True, class_agnostic_nms: bool = False) -> list[Any]:
    """The app's YOLO decoder (core.detection._run_onnx_yolo) with the registry spec."""
    from core.detection import _run_onnx_yolo
    from core.models import get_model_spec, model_status

    spec = get_model_spec(key)
    path = Path(model_status(key)["path"])
    return _run_onnx_yolo(
        image_bgr, path, spec.labels,
        float(spec.score_threshold if score_threshold is None else score_threshold),
        float(spec.iou_threshold), int(spec.input_size), letterbox=letterbox,
        class_agnostic_nms=class_agnostic_nms)


def components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, count)]


def box_mask(shape: tuple[int, int], boxes: Iterable[tuple]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        out[int(max(0, y1)):int(np.ceil(y2)) + 1, int(max(0, x1)):int(np.ceil(x2)) + 1] = True
    return out


def save_overlay(path: Path, image: np.ndarray, max_side: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def artifacts_dir(args: argparse.Namespace, area: str) -> Path:
    if args.artifacts:
        out = Path(args.artifacts)
    else:
        out = testdata_root() / "_outputs" / "detectors" / area
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── road: RDD2022 subset ─────────────────────────────────────────────────────

#: A crack-presence tile counts as containing a crack when an annotated crack box covers
#: at least this share of the tile, and as clean when it touches no damage box at all.
#: Tiles in between are ambiguous and left out rather than guessed.
TILE_POSITIVE_SHARE = 1.0 / 16.0
#: A crack box counts as found by the segmenter when this share of it is painted crack.
SEG_BOX_HIT_SHARE = 0.005
SEG_BOX_HIT_MIN_PX = 10


NO_ROAD_MODEL = (
    "The registry has no road-damage detector. A roads YOLO11l Kaggle kernel "
    "(odk-train-roads-yolo11l) was pushed, but its run folder "
    "(training/runs/kaggle_v5/odk-train-roads-yolo11l) holds no weights and nothing was "
    "registered. Potholes (D40) and the RDD crack types therefore have no road model in "
    "this build; the numbers below are the closest shipped models (trained on concrete "
    "structures and crack tiles) on asphalt. Training a road-damage detector on RDD2022 "
    "is roadmap item 5.")


def evaluate_road(args: argparse.Namespace) -> dict[str, Any]:
    from core.detection import detect_cracks
    from core.model_runner import crack_presence

    root = testdata_root()
    samples = rdd_samples(root)
    if args.limit:
        samples = samples[: args.limit]
    out_dir = artifacts_dir(args, "road_rdd2022")
    conversion = voc_to_yolo(samples, out_dir, "rdd2022_subset")

    models = {k: model_row(k) for k in
              ("structural_multiclass_detector", "crack_presence_classifier",
               "crack_segmentation")}
    det_preds: dict[str, list] = defaultdict(list)          # label -> (img, score, box)
    truths: dict[str, dict[str, list]] = {c: defaultdict(list) for c in RDD_CORE}
    others: dict[str, list] = defaultdict(list)               # non-core damage boxes
    tile_counts = {s: Counter() for s in ("all", "China_Drone", "India")}
    image_counts = {s: Counter() for s in ("all", "China_Drone", "India")}
    seg = {variant: {s: Counter() for s in ("all", "China_Drone", "India")}
           for variant in ("model", "heuristic")}
    seg_box_hits = {variant: {c: Counter() for c in RDD_CRACKS} for variant in ("model", "heuristic")}
    rows = []
    started = time.time()

    for n, sample in enumerate(samples):
        image = cv2.imread(str(sample["image"]))
        if image is None:
            raise SystemExit(f"Could not read {sample['image']}")
        image_id = f"{sample['subset']}/{Path(sample['image']).name}"
        h, w = image.shape[:2]
        crack_boxes, damage_boxes = [], []
        for label, box in sample["objects"]:
            damage_boxes.append(box)
            if label in RDD_CORE:
                truths[label][image_id].append(box)
                if label in RDD_CRACKS:
                    crack_boxes.append(box)
            else:
                others[image_id].append(box)
        has_crack = bool(crack_boxes)

        # 1. structural detector, through the app decoder; low threshold for the AP curve.
        hits = yolo_hits(image, "structural_multiclass_detector", score_threshold=0.001)
        for hit in hits:
            det_preds[hit.label].append((image_id, float(hit.confidence), hit.bbox_xyxy))

        # 2. crack presence, the app's tiled triage at its default threshold.
        presence = crack_presence(image)
        flagged = [tuple(t["bbox_xyxy"]) for t in presence.flagged_tiles]
        tile = presence.tile_size
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                tx2, ty2 = min(w, x + tile), min(h, y + tile)
                if ty2 - y < tile // 4 or tx2 - x < tile // 4:
                    continue
                area = float((tx2 - x) * (ty2 - y))
                crack_cover = box_mask((ty2 - y, tx2 - x), [
                    (bx1 - x, by1 - y, bx2 - x, by2 - y) for bx1, by1, bx2, by2 in crack_boxes]).sum()
                touches = any(min(tx2, b[2]) > max(x, b[0]) and min(ty2, b[3]) > max(y, b[1])
                              for b in damage_boxes)
                is_flagged = (x, y, tx2, ty2) in flagged
                if crack_cover / area >= TILE_POSITIVE_SHARE:
                    truth = True
                elif not touches:
                    truth = False
                else:
                    for s in ("all", sample["subset"]):
                        tile_counts[s]["ambiguous"] += 1
                    continue
                key = ("tp" if is_flagged else "fn") if truth else ("fp" if is_flagged else "tn")
                for s in ("all", sample["subset"]):
                    tile_counts[s][key] += 1
        image_key = ("tp" if presence.any_crack else "fn") if has_crack else (
            "fp" if presence.any_crack else "tn")
        for s in ("all", sample["subset"]):
            image_counts[s][image_key] += 1

        # 3. crack segmentation (model) and the classical baseline the app falls back to.
        crack_zone = box_mask((h, w), crack_boxes)
        seg_summary = {}
        for variant, use_model in (("model", True), ("heuristic", False)):
            result = detect_cracks(image, use_model=use_model)
            mask = result.mask > 0
            comps = components(mask)
            inside = sum(1 for c in comps if (c & crack_zone).sum() >= 0.5 * c.sum())
            for s in ("all", sample["subset"]):
                seg[variant][s]["components"] += len(comps)
                seg[variant][s]["components_on_crack_boxes"] += inside
                seg[variant][s]["mask_px"] += int(mask.sum())
                seg[variant][s]["mask_px_in_crack_boxes"] += int((mask & crack_zone).sum())
                key = ("tp" if mask.any() else "fn") if has_crack else ("fp" if mask.any() else "tn")
                seg[variant][s]["image_" + key] += 1
            for label, box in sample["objects"]:
                if label not in RDD_CRACKS:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                sub = mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
                need = max(SEG_BOX_HIT_MIN_PX, SEG_BOX_HIT_SHARE * sub.size)
                seg_box_hits[variant][label]["truth"] += 1
                if sub.sum() >= need:
                    seg_box_hits[variant][label]["found"] += 1
            seg_summary[variant] = {"model_used": result.model_used,
                                    "crack_ratio": round(result.crack_ratio, 5)}
            if variant == "model" and n < args.overlays:
                save_overlay(out_dir / "overlays" / f"{image_id.replace('/', '_')}_seg.jpg",
                             result.overlay)

        if n < args.overlays:
            canvas = image.copy()
            for _label, box in sample["objects"]:
                cv2.rectangle(canvas, tuple(int(v) for v in box[:2]),
                              tuple(int(v) for v in box[2:]), (0, 255, 0), 2)
            for hit in hits:
                if hit.confidence >= 0.25:
                    cv2.rectangle(canvas, hit.bbox_xyxy[:2], hit.bbox_xyxy[2:], (0, 0, 255), 2)
                    cv2.putText(canvas, f"{hit.label} {hit.confidence:.2f}", hit.bbox_xyxy[:2],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            for box in flagged:
                cv2.rectangle(canvas, box[:2], (box[2] - 1, box[3] - 1), (0, 200, 255), 1)
            save_overlay(out_dir / "overlays" / f"{image_id.replace('/', '_')}_det.jpg", canvas)

        rows.append({
            "image": image_id,
            "truth": [label for label, _ in sample["objects"]],
            "detector_hits_at_0.25": [
                {"label": h_.label, "confidence": round(h_.confidence, 3), "bbox": list(h_.bbox_xyxy)}
                for h_ in hits if h_.confidence >= 0.25],
            "crack_presence": {"tile": tile, "flagged": len(flagged), "tiles": presence.tiles,
                               "max_probability": round(presence.max_probability, 4)},
            "segmentation": seg_summary,
        })
        if (n + 1) % 20 == 0:
            print(f"  road: {n + 1}/{len(samples)} images, {time.time() - started:.0f}s", flush=True)

    # Detector AP. "Crack" is class-agnostic about crack type, so each RDD crack class is
    # scored with the other crack classes (and non-core damage) as ignore regions.
    crack_preds = det_preds.get("Crack", [])
    operating = 0.25

    def at(preds, thr):
        return [p for p in preds if p[1] >= thr]

    per_class = {}
    for cls in RDD_CRACKS:
        ignore = defaultdict(list)
        for other in RDD_CRACKS:
            if other != cls:
                for k, v in truths[other].items():
                    ignore[k].extend(v)
        for k, v in others.items():
            ignore[k].extend(v)
        per_class[cls] = {
            "name": RDD_NAMES[cls],
            "ap50_full_curve": average_precision(crack_preds, truths[cls], ignore)["ap"],
            "at_operating_threshold": average_precision(at(crack_preds, operating), truths[cls], ignore),
        }
    merged = defaultdict(list)
    for cls in RDD_CRACKS:
        for k, v in truths[cls].items():
            merged[k].extend(v)
    merged_full = average_precision(crack_preds, merged, others)
    merged_op = average_precision(at(crack_preds, operating), merged, others)
    merged_iou30 = average_precision(crack_preds, merged, others, iou_threshold=0.3)
    spall = det_preds.get("Spallation", [])
    pothole_proxy = {
        "ap50_full_curve": average_precision(spall, truths["D40"], others)["ap"],
        "at_operating_threshold": average_precision(at(spall, operating), truths["D40"], others),
        "note": "EXPLORATORY ONLY. Spallation is concrete surface loss on structures; it "
                "was never trained on asphalt potholes and is not a pothole detector.",
    }
    label_counts = Counter({k: len(at(v, operating)) for k, v in det_preds.items()})

    def seg_report(variant: str) -> dict[str, Any]:
        out = {}
        for s, c in seg[variant].items():
            out[s] = {
                "image_level": binary_scores(c["image_tp"], c["image_fp"], c["image_fn"], c["image_tn"]),
                "components": c["components"],
                "component_precision_on_crack_boxes": round(
                    c["components_on_crack_boxes"] / c["components"], 4) if c["components"] else None,
                "mask_pixels_inside_crack_boxes": round(
                    c["mask_px_in_crack_boxes"] / c["mask_px"], 4) if c["mask_px"] else None,
            }
        out["crack_box_recall"] = {
            cls: {"found": seg_box_hits[variant][cls]["found"],
                  "truth": seg_box_hits[variant][cls]["truth"],
                  "recall": round(seg_box_hits[variant][cls]["found"] /
                                  max(1, seg_box_hits[variant][cls]["truth"]), 4)}
            for cls in RDD_CRACKS}
        return out

    truth_counts = {c: sum(len(v) for v in truths[c].values()) for c in RDD_CORE}
    images_with = {s: Counter() for s in ("China_Drone", "India")}
    for sample in samples:
        labels = {l for l, _ in sample["objects"]}
        images_with[sample["subset"]]["any_crack" if labels & set(RDD_CRACKS) else "no_crack"] += 1

    rows_path = out_dir / "per_image.json"
    rows_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {
        "evaluation": "road damage (RDD2022 subset) through the shipped crack models",
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "folder": RDD_FOLDER,
            "source": "https://doi.org/10.6084/m9.figshare.21431547 (RDD2022, CC BY 4.0)",
            "subset": "120 China_Drone UAV images (512 px, all damaged) + 80 India vehicle "
                      "images (720 px, 35 damaged), random sample seed 2022",
            "images": len(samples),
            "images_by_subset": {k: dict(v) for k, v in images_with.items()},
            "core_boxes": truth_counts,
            "converted_to_yolo": conversion,
            "contamination_check": "RDD2022 fed only training/data/prepared/roads_det, whose "
                                   "run never produced weights; crack_cls (SDNET2018 + "
                                   "surface_crack), crack_seg (crack_segmentation_kaggle, "
                                   "DeepCrack, CrackForest) and structural_det did not use it.",
        },
        "no_road_model": NO_ROAD_MODEL,
        "models": models,
        "structural_detector_crack_class": {
            "how": "Crack boxes from structural_multiclass_detector via core.detection."
                   "_run_onnx_yolo (letterbox, per-class NMS), scored at IoU 0.5 against "
                   "RDD crack boxes. Each RDD crack type is scored with the other types "
                   "as ignore regions, because the model has one undifferentiated Crack "
                   "class. 'at_operating_threshold' uses the registry's score 0.25.",
            "per_class": per_class,
            "all_cracks_merged": {"ap50_full_curve": merged_full["ap"],
                                  "ap30_full_curve": merged_iou30["ap"],
                                  "at_operating_threshold": merged_op},
            "map50_over_crack_classes": round(float(np.mean(
                [per_class[c]["ap50_full_curve"] or 0.0 for c in RDD_CRACKS])), 4),
            "pothole_D40_vs_Spallation": pothole_proxy,
            "predicted_labels_at_operating_threshold": dict(label_counts),
        },
        "crack_presence_classifier": {
            "how": "core.model_runner.crack_presence (the app's tiling, threshold 0.5). A "
                   f"tile is a crack tile when crack boxes cover >= {TILE_POSITIVE_SHARE:.4f} "
                   "of it and a clean tile when it touches no damage box; tiles in between "
                   "are counted as ambiguous and not scored. Image level: any flagged tile "
                   "vs any D00/D10/D20 box.",
            "tiles": {s: {**binary_scores(c["tp"], c["fp"], c["fn"], c["tn"]),
                          "ambiguous_not_scored": c["ambiguous"]} for s, c in tile_counts.items()},
            "images": {s: binary_scores(c["tp"], c["fp"], c["fn"], c["tn"])
                       for s, c in image_counts.items()},
        },
        "crack_segmentation": {
            "how": "core.detection.detect_cracks with the model (registry threshold 0.85) and "
                   "with use_model=False (the classical fallback). RDD has boxes, not masks, so "
                   "this is box-level: a crack box is found when >= 0.5% (min 10 px) of it is "
                   "painted; a mask component is on a crack when >= half of it lies inside a "
                   "crack box (an upper bound on precision, since a box also holds sound "
                   "pavement).",
            "model": seg_report("model"),
            "classical_baseline": seg_report("heuristic"),
        },
        "artifacts": str(out_dir),
    }


#: Said whenever the solar classifier file on disk is not the one the registry describes.
PV_IDENTITY_WARNING = (
    "The file loaded does not carry the registry's sha256, so the registry's validation "
    "figures (balanced accuracy 0.724) do NOT describe it, and these outputs describe "
    "that file only. Re-run with the registered file before quoting them for the "
    "registered model.")


# ── pv: Mavic 3T thermal panels ──────────────────────────────────────────────

def evaluate_pv(args: argparse.Namespace) -> dict[str, Any]:
    from core.detection import _is_grayscale, detect_solar_defects

    root = testdata_root() / PV_FOLDER / PV_INNER
    out_dir = artifacts_dir(args, "pv_mavic3t")
    model = model_row("solar_thermal_anomaly_classifier")
    frames = []
    for split in ("train", "val", "test"):
        frames += [(split, p) for p in sorted((root / split / "images").glob("*.jpg"))]
    if args.limit:
        frames = frames[: args.limit]

    whole_frame = Counter()
    whole_frame_reasons = Counter()
    grey_frames = 0
    top = {"as_labelled": Counter(), "rotated_portrait": Counter()}
    refusals = Counter()
    agree = 0
    crops = 0
    probabilities_no_anomaly = []
    examples: dict[str, list] = defaultdict(list)
    panel_sizes = []
    started = time.time()
    for n, (split, path) in enumerate(frames):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        grey_frames += int(_is_grayscale(image))
        frame_result = detect_solar_defects(image, imagery="thermal_module")
        whole_frame[frame_result.model_used] += 1
        if frame_result.refused:
            whole_frame_reasons[frame_result.refused_reason.split(".")[0][:120]] += 1
        auto_result = detect_solar_defects(image)
        whole_frame["auto:" + auto_result.model_used] += 1

        label_path = root / split / "labels" / (path.stem + ".txt")
        for _cls, (x1, y1, x2, y2) in read_yolo_labels(label_path, w, h):
            x1, y1 = int(max(0, round(x1))), int(max(0, round(y1)))
            x2, y2 = int(min(w, round(x2))), int(min(h, round(y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image[y1:y2, x1:x2]
            panel_sizes.append((x2 - x1, y2 - y1))
            answers = {}
            for variant, picture in (("as_labelled", crop),
                                     ("rotated_portrait", crop if crop.shape[0] >= crop.shape[1]
                                      else cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))):
                result = detect_solar_defects(picture, imagery="thermal_module")
                if result.refused:
                    refusals[f"{variant}: {result.refused_reason[:100]}"] += 1
                    answers[variant] = "refused"
                    continue
                probs = result.class_probabilities
                best = max(probs, key=probs.get)
                answers[variant] = best
                top[variant][best] += 1
                if variant == "as_labelled":
                    probabilities_no_anomaly.append(probs.get("No-Anomaly", 0.0))
                    if len(examples[best]) < 24:
                        examples[best].append((picture, probs[best], f"{path.name}:{x1},{y1}"))
            crops += 1
            agree += int(answers["as_labelled"] == answers["rotated_portrait"])
        if (n + 1) % 50 == 0:
            print(f"  pv: {n + 1}/{len(frames)} frames, {crops} panels, {time.time() - started:.0f}s",
                  flush=True)

    # A contact sheet per predicted class, for a human to look at -- not ground truth.
    sheets = {}
    for cls, items in examples.items():
        tiles = []
        for picture, prob, _ in items:
            t = cv2.resize(picture, (160, 96), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"{prob:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        name = out_dir / f"panels_predicted_{cls}.jpg"
        cv2.imwrite(str(name), sheet)
        sheets[cls] = str(name)

    total = max(1, sum(top["as_labelled"].values()))
    sizes = np.array(panel_sizes, dtype=float) if panel_sizes else np.zeros((0, 2))
    return {
        "evaluation": "solar thermal (Mavic 3T PV panels) through the shipped solar models",
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "folder": PV_FOLDER,
            "source": "https://doi.org/10.5281/zenodo.16420123 (CC BY 4.0)",
            "frames": len(frames),
            "frames_greyscale_by_app_test": grey_frames,
            "panel_boxes_scored": crops,
            "panel_box_px_median_wh": [float(np.median(sizes[:, 0])), float(np.median(sizes[:, 1]))]
            if len(sizes) else None,
            "labels": "PV panel boxes only (class 0). NO defect/anomaly labels, and the "
                      "images are Roboflow exports with the radiometric payload stripped.",
        },
        "panel_detection": {
            "measured": False,
            "why": "No shipped model localises PV panels or modules in a thermal frame. "
                   "solar_defect_detector and solar_pv_multidefect_detector are awaiting "
                   "weights; solar_module_segformer_b2 exists only as an unregistered "
                   "training run whose validation IoU stayed near 0.0007. Panel-box "
                   "precision/recall/mAP cannot be reported. Training a thermal panel "
                   "localiser on this dataset is roadmap item 5.",
            "whole_frame_handling": dict(whole_frame),
            "whole_frame_refusal_reasons": dict(whole_frame_reasons),
        },
        "thermal_anomaly_classifier_on_labelled_panels": {
            "model": model,
            "identity_warning": None if model["sha256_matches_registry"] else PV_IDENTITY_WARNING,
            "accuracy": None,
            "accuracy_why_none": "The dataset has no anomaly labels, so no accuracy, precision "
                                 "or recall can be computed. What follows is what the model "
                                 "SAID, not whether it was right.",
            "top_class_counts": dict(top["as_labelled"].most_common()),
            "top_class_share": {k: round(v / total, 4) for k, v in top["as_labelled"].most_common()},
            "top_class_counts_rotated_to_portrait": dict(top["rotated_portrait"].most_common()),
            "orientation_agreement": round(agree / max(1, crops), 4),
            "median_no_anomaly_probability": round(float(np.median(probabilities_no_anomaly)), 4)
            if probabilities_no_anomaly else None,
            "refusals": dict(refusals),
            "domain_gap": "Trained on 24x40 px (portrait) single-module crops from Raptor "
                          "Maps InfraredSolarModules. These are Mavic 3T module boxes of "
                          "median %s px (landscape), cut from palette-rendered frames whose "
                          "radiometric data was stripped, from a different camera and site. "
                          "The 'rotated_to_portrait' run turns each crop to the training "
                          "orientation; the agreement between the two runs measures how "
                          "stable the answer is to a change that should not matter." % (
                              "x".join(str(int(v)) for v in np.median(sizes, axis=0))
                              if len(sizes) else "?"),
            "contact_sheets": sheets,
        },
        "artifacts": str(out_dir),
    }


# ── bridge: H20T wide frames ─────────────────────────────────────────────────

def evaluate_bridge(args: argparse.Namespace) -> dict[str, Any]:
    from core.detection import detect_cracks, detect_structural_defects, non_defect_labels
    from core.model_runner import crack_presence

    root = testdata_root() / BRIDGE_FOLDER / "B1_Wide"
    frames = sorted(root.glob("*_W.JPG"))
    if args.limit:
        frames = frames[: args.limit]
    out_dir = artifacts_dir(args, "bridge_h20t")
    models = {k: model_row(k) for k in
              ("structural_multiclass_detector", "crack_presence_classifier", "crack_segmentation")}
    by_label = {"letterbox": Counter(), "stretch": Counter()}
    frames_with = {"letterbox": 0, "stretch": 0}
    presence_frames = 0
    flagged_tiles = 0
    tiles_total = 0
    seg_frames = 0
    seg_ratios = []
    tiled_rows: list[dict[str, Any]] = []
    rows = []
    started = time.time()
    for n, path in enumerate(frames):
        image = cv2.imread(str(path))
        result = detect_structural_defects(image)
        lb_hits = result.detections
        not_defects = non_defect_labels("structural_multiclass_detector")
        st_all = yolo_hits(image, "structural_multiclass_detector", letterbox=False)
        st_hits = [h for h in st_all if h.label not in not_defects]
        for name, hits, skipped in (("letterbox", lb_hits, result.non_defect_detections),
                                    ("stretch", st_hits,
                                     [h for h in st_all if h.label in not_defects])):
            frames_with[name] += int(bool(hits))
            for hit in hits:
                by_label[name][hit.label] += 1
            for hit in skipped:
                by_label[name]["(not a defect) " + hit.label] += 1
        presence = crack_presence(image)
        presence_frames += int(presence.any_crack)
        flagged_tiles += len(presence.flagged_tiles)
        tiles_total += presence.tiles
        cracks = detect_cracks(image)
        seg_frames += int(cracks.crack_pixels > 0)
        seg_ratios.append(cracks.crack_ratio)
        if n < args.tiled_frames:
            # The app segments the whole 4056x3040 frame at the model's 1024 px input, a
            # 4x downsample. Segment native-resolution 1024 px tiles too, to see what the
            # downsample hides -- qualitative, as nothing here is labelled.
            tiled = np.zeros(image.shape[:2], dtype=np.uint8)
            fh, fw = image.shape[:2]
            for ty in range(0, fh, 1024):
                for tx in range(0, fw, 1024):
                    part = image[ty:ty + 1024, tx:tx + 1024]
                    tiled[ty:ty + part.shape[0], tx:tx + part.shape[1]] = detect_cracks(part).mask
            tiled_rows.append({"frame": path.name, "whole_frame_crack_pixels": cracks.crack_pixels,
                               "tiled_1024_crack_pixels": int((tiled > 0).sum())})
            view = image.copy()
            view[tiled > 0] = (255, 0, 255)
            save_overlay(out_dir / "overlays" / f"{path.stem}_tiled_segmentation.jpg", view, 2400)

        canvas = image.copy()
        for hit in lb_hits:
            cv2.rectangle(canvas, hit.bbox_xyxy[:2], hit.bbox_xyxy[2:], (0, 0, 255), 6)
            cv2.putText(canvas, f"{hit.label} {hit.confidence:.2f}", hit.bbox_xyxy[:2],
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        for tile in presence.flagged_tiles:
            x1, y1, x2, y2 = tile["bbox_xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), 5)
            cv2.putText(canvas, f"crack p={tile['probability']:.2f}", (x1 + 8, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 4)
        canvas[cracks.mask > 0] = (255, 0, 255)
        save_overlay(out_dir / "overlays" / f"{path.stem}.jpg", canvas)
        rows.append({
            "frame": path.name,
            "structural_letterbox": [h.to_dict() for h in lb_hits],
            "structural_stretch": [h.to_dict() for h in st_hits],
            "crack_presence": {"tile": presence.tile_size, "tiles": presence.tiles,
                               "flagged": presence.flagged_tiles,
                               "max_probability": round(presence.max_probability, 4)},
            "crack_segmentation": {"model_used": cracks.model_used,
                                   "crack_pixels": cracks.crack_pixels,
                                   "crack_ratio": round(cracks.crack_ratio, 6)},
        })
        if (n + 1) % 10 == 0:
            print(f"  bridge: {n + 1}/{len(frames)} frames, {time.time() - started:.0f}s", flush=True)
    (out_dir / "per_frame.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {
        "evaluation": "bridge H20T wide frames through the shipped structural/crack models",
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "folder": BRIDGE_FOLDER + "/B1_Wide",
            "source": "https://doi.org/10.5281/zenodo.22298893 (CC BY 4.0)",
            "frames": len(frames),
            "frame_size": "4056x3040, nadir ~50 m AGL",
            "labels": "NONE. The dataset carries no defect annotations, so no precision, "
                      "recall or mAP can be computed; the counts below are what the models "
                      "said, reviewed by eye on the overlays.",
        },
        "models": models,
        "structural_detector": {
            "frames_with_any_detection": frames_with,
            "detections_by_label": {k: dict(v) for k, v in by_label.items()},
        },
        "crack_presence_classifier": {
            "frames_with_a_flagged_tile": presence_frames,
            "flagged_tiles": flagged_tiles,
            "tiles_scored": tiles_total,
        },
        "crack_segmentation": {
            "frames_with_crack_pixels": seg_frames,
            "median_crack_ratio": round(float(np.median(seg_ratios)), 6) if seg_ratios else None,
            "max_crack_ratio": round(float(np.max(seg_ratios)), 6) if seg_ratios else None,
            "native_resolution_tiles": tiled_rows,
            "native_resolution_note": "The app segments the whole frame resized to 1024 px. "
                                      "These rows re-run the same model on 1024 px tiles at "
                                      "full resolution for the first frames; compare the "
                                      "*_tiled_segmentation.jpg overlays by eye.",
        },
        "artifacts": str(out_dir),
    }


# ── structural_det test split: app decoder vs published figure ───────────────

def evaluate_structural_split(args: argparse.Namespace) -> dict[str, Any]:
    root = os.environ.get("ODK_TRAINING_DATA_DIR", "").strip()
    base = Path(root) if root else REPO_ROOT / "training" / "data"
    folder = base / "prepared" / "structural_det" / "test"
    if not (folder / "images").is_dir():
        raise SystemExit(f"{folder} not found; set ODK_TRAINING_DATA_DIR.")
    from core.detection import non_defect_labels
    from core.models import get_model_spec

    labels = get_model_spec("structural_multiclass_detector").labels
    not_defects = non_defect_labels("structural_multiclass_detector")
    model = model_row("structural_multiclass_detector")
    images = sorted((folder / "images").glob("*"))
    if args.limit:
        images = images[: args.limit]

    def run(crop_43: bool, letterbox: bool, agnostic: bool = False) -> dict[str, Any]:
        preds = defaultdict(list)
        truth = {c: defaultdict(list) for c in labels}
        for path in images:
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            boxes = read_yolo_labels(folder / "labels" / (path.stem + ".txt"), w, h)
            if crop_43:
                # Centre 4:3 crop (640x480 from 640x640). A box is kept when at least half
                # of it survives the crop, clipped to it; otherwise it becomes an ignore
                # region by being dropped together with any prediction on it below.
                top = (h - int(w * 3 / 4)) // 2
                image = image[top:top + int(w * 3 / 4)]
                kept = []
                for cls, (x1, y1, x2, y2) in boxes:
                    cy1, cy2 = max(y1, top), min(y2, top + image.shape[0])
                    if cy2 - cy1 >= 0.5 * (y2 - y1):
                        kept.append((cls, (x1, cy1 - top, x2, cy2 - top)))
                boxes = kept
            for cls, box in boxes:
                truth[labels[cls]][path.stem].append(box)
            for hit in yolo_hits(image, "structural_multiclass_detector", score_threshold=0.001,
                                 letterbox=letterbox, class_agnostic_nms=agnostic):
                preds[hit.label].append((path.stem, float(hit.confidence), hit.bbox_xyxy))
        per = {}
        for cls in labels:
            n_truth = sum(len(v) for v in truth[cls].values())
            if not n_truth:
                continue
            per[cls] = {"ap50": average_precision(preds[cls], truth[cls])["ap"], "truth_boxes": n_truth}
        values = [v["ap50"] for v in per.values() if v["ap50"] is not None]
        defects = [v["ap50"] for c, v in per.items() if v["ap50"] is not None and c not in not_defects]
        return {"map50": round(float(np.mean(values)), 4) if values else None,
                "map50_defect_classes": round(float(np.mean(defects)), 4) if defects else None,
                "per_class": per}

    return {
        "evaluation": "structural_det test split through the app's YOLO decoder",
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"folder": "prepared/structural_det/test", "images": len(images),
                    "note": "The split the registry's validation figures come from is val; "
                            "this is the held-out test split of the same corpus."},
        "model": model,
        "published_validation_map50": 0.417,
        "square_640_letterbox": run(False, True),
        "square_640_stretch": run(False, False),
        "crop_4x3_letterbox": run(True, True),
        "crop_4x3_stretch": run(True, False),
        "previous_app_decoder_square": run(False, False, True),
        "previous_app_decoder_4x3": run(True, False, True),
        "how": "AP50 (VOC all-point) from predictions down to score 0.001 through "
               "core.detection._run_onnx_yolo with per-class NMS; the 4:3 rows crop the "
               "centre 640x480 of each image to measure aspect handling on labelled data. "
               "'previous_app_decoder_*' is the decoder as it was before this change "
               "(stretch resize and class-agnostic NMS).",
    }


# ── entry point ──────────────────────────────────────────────────────────────

EVALUATIONS = {"road": evaluate_road, "pv": evaluate_pv, "bridge": evaluate_bridge,
               "structural-split": evaluate_structural_split}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("area", choices=sorted(EVALUATIONS))
    parser.add_argument("--out", help="Write the summary JSON here (e.g. docs/holdout/...).")
    parser.add_argument("--artifacts", help="Folder for overlays and per-image rows.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N images (smoke runs).")
    parser.add_argument("--overlays", type=int, default=24, help="Road: overlays for the first N images.")
    parser.add_argument("--tiled-frames", type=int, default=3,
                        help="Bridge: also segment the first N frames as native 1024 px tiles.")
    args = parser.parse_args(argv)
    summary = EVALUATIONS[args.area](args)
    text = json.dumps(summary, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
