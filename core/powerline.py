"""Two-stage power-line inspection: InsPLAD asset detector, then a condition classifier.

Stage 1, ``powerline_insplad_detector`` (YOLO11s, 17 asset classes, names read from the
ONNX metadata), finds components in a UAV photograph.

Stage 2, ``powerline_insplad_condition`` (EfficientNetV2-S), reads the condition of the
five defect-prone asset types from a crop of each detection. It has one head over eleven
``asset__condition`` classes, so its logits are masked to the conditions that exist for
the detected asset before the softmax: a glass insulator cannot be diagnosed with a
bird's nest, and without the mask the classifier could say so.

The crop follows the publisher's reference (github.com/Josh-E-S/powerline-inspection-
demo, app/app.py): the box is cut from the source image, its short side resized to 256
bilinearly, centre-cropped to 224, scaled to [0, 1] and ImageNet-normalised.

Licence: both models are CC BY-NC 3.0 (non-commercial) per their publisher, and
Ultralytics claims AGPL-3.0 on YOLO11 fine-tunes. Every result carries that note. Use
per the publisher is research and demonstration, not operational decisions.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .detection import letterbox_image
from .infrastructure_detectors import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LoadedModel,
    load_model,
    yolo_candidates,
)

DETECTOR_KEY = "powerline_insplad_detector"
CONDITION_KEY = "powerline_insplad_condition"

#: Detector class name -> classifier class prefix, from the publisher's reference.
#: Every other detector class is detect-only: the classifier was never trained on it.
ASSET_TO_CONDITION_PREFIX: dict[str, str] = {
    "glass insulator": "glass-insulator",
    "lightning rod suspension": "lightning-rod-suspension",
    "polymer insulator upper shackle": "polymer-insulator-upper-shackle",
    "vari-grip": "vari-grip",
    "yoke suspension": "yoke-suspension",
}

SCOPE_NOTE = ("Non-commercial (CC BY-NC 3.0); research/demonstration use per the publisher. "
              "Condition is image-level on a crop of the component, not a located defect.")


def valid_condition_mask(classes: list[str], asset_prefix: str) -> np.ndarray:
    """Which classifier outputs are conditions of this asset type."""
    mask = np.array([c.startswith(asset_prefix + "__") for c in classes], dtype=bool)
    if not mask.any():
        raise ValueError(f"No classifier class belongs to asset {asset_prefix!r}.")
    return mask


def masked_softmax(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Softmax over the valid classes only; the others get probability exactly 0."""
    z = np.where(valid, np.asarray(logits, np.float64), -np.inf)
    z = z - z[valid].max()
    e = np.where(valid, np.exp(z), 0.0)
    return e / e.sum()


def condition_crop(image_bgr: np.ndarray, box_xyxy: list[int] | tuple[int, ...]) -> np.ndarray:
    """One ``(3, 224, 224)`` classifier input from a source-image box."""
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop.")
    h, w = crop.shape[:2]
    r = 256.0 / min(h, w)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    crop = crop[max(0, top):max(0, top) + 224, max(0, left):max(0, left) + 224]
    if crop.shape[0] != 224 or crop.shape[1] != 224:
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = rgb.transpose(2, 0, 1)[None]
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)[0]


def condition_classes(model: LoadedModel) -> list[str]:
    classes = [str(c) for c in (model.row.get("output_classes") or model.labels)]
    if not classes or any("__" not in c for c in classes):
        from .model_runner import ModelRunRefused

        raise ModelRunRefused(
            f"{model.key} does not list its asset__condition classes in the registry.")
    return classes


def classify_conditions(image_bgr: np.ndarray, detections: list[dict[str, Any]],
                        model: LoadedModel) -> None:
    """Attach ``condition`` to each detection of a defect-prone asset, in place."""
    classes = condition_classes(model)
    todo = [(i, ASSET_TO_CONDITION_PREFIX[d["label"]]) for i, d in enumerate(detections)
            if d["label"] in ASSET_TO_CONDITION_PREFIX]
    for det in detections:
        det.setdefault("condition", None)
    if not todo:
        return
    batch = []
    keep = []
    for i, prefix in todo:
        x1, y1, x2, y2 = detections[i]["bbox_xyxy"]
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        batch.append(condition_crop(image_bgr, detections[i]["bbox_xyxy"]))
        keep.append((i, prefix))
    if not batch:
        return
    sess = model.session
    inp = sess.get_inputs()[0]
    logits = np.asarray(sess.run(None, {inp.name: np.stack(batch)})[0], np.float64)
    if logits.shape[1] != len(classes):
        from .model_runner import ModelRunRefused

        raise ModelRunRefused(
            f"{model.key} emits {logits.shape[1]} logits for {len(classes)} registered classes.")
    for (i, prefix), row in zip(keep, logits):
        valid = valid_condition_mask(classes, prefix)
        probs = masked_softmax(row, valid)
        top = int(probs.argmax())
        detections[i]["condition"] = {
            "label": classes[top].split("__", 1)[1],
            "class": classes[top],
            "probability": float(probs[top]),
            "probabilities": {classes[j]: float(probs[j]) for j in np.flatnonzero(valid)},
            "model_key": model.key,
            "model_sha256": model.sha256,
            "confidence_kind": "classifier_softmax_over_valid_conditions",
        }


def detect_assets(image_bgr: np.ndarray, model: LoadedModel, score_threshold: float,
                  iou_threshold: float) -> list[dict[str, Any]]:
    """Stage 1 through onnxruntime (the int8 QDQ graph is quantised for it)."""
    sess = model.session
    inp = sess.get_inputs()[0]
    size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else int(
        model.row.get("input_size") or 640)
    lb = letterbox_image(image_bgr, size)
    pred = sess.run(None, {inp.name: lb.blob.astype(np.float32)})[0][0]
    boxes, scores, class_ids, _ = yolo_candidates(pred, len(model.labels), score_threshold,
                                                  iou_threshold)
    h, w = image_bgr.shape[:2]
    out = []
    for box, score, cid in zip(boxes, scores, class_ids):
        x1 = int(np.clip(np.floor((box[0] - lb.left) / lb.gain), 0, w))
        y1 = int(np.clip(np.floor((box[1] - lb.top) / lb.gain), 0, h))
        x2 = int(np.clip(np.ceil((box[2] - lb.left) / lb.gain), 0, w))
        y2 = int(np.clip(np.ceil((box[3] - lb.top) / lb.gain), 0, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"label": model.labels[int(cid)], "confidence": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2], "area_px": (x2 - x1) * (y2 - y1)})
    return out


def run_powerline(image_bgr: np.ndarray, *, detector_key: str = DETECTOR_KEY,
                  condition_key: str | None = CONDITION_KEY,
                  score_threshold: float | None = None,
                  require_condition: bool = False) -> dict[str, Any]:
    """Detect power-line assets, then read the condition of the defect-prone ones.

    The detector must be installed and verified, or this refuses. The condition model
    is a second stage: when it is missing the assets are still returned, each eligible
    one with ``condition: None`` and the refusal reason in ``condition_refused`` --
    unless ``require_condition`` is set, in which case the whole run refuses.
    """
    from .model_runner import ModelRunRefused

    detector = load_model(detector_key, expected_kinds={"onnx_yolo"})
    threshold = float(detector.row.get("score_threshold", 0.25)
                      if score_threshold is None else score_threshold)
    detections = detect_assets(image_bgr, detector, threshold,
                               float(detector.row.get("iou_threshold", 0.45)))
    refused = ""
    condition_model: LoadedModel | None = None
    if condition_key:
        try:
            condition_model = load_model(condition_key, expected_kinds={"onnx_condition_classifier"})
        except ModelRunRefused as exc:
            if require_condition:
                raise
            refused = str(exc)
    if condition_model is not None:
        classify_conditions(image_bgr, detections, condition_model)
    else:
        for det in detections:
            det["condition"] = None
    return {
        "detections": detections,
        "detector": {"model_key": detector.key, "model_sha256": detector.sha256,
                     "model_used": detector.model_used},
        "condition_model": (None if condition_model is None else {
            "model_key": condition_model.key, "model_sha256": condition_model.sha256,
            "model_used": condition_model.model_used}),
        "condition_refused": refused,
        "condition_assets": sorted(ASSET_TO_CONDITION_PREFIX),
        "scope_note": SCOPE_NOTE,
        "score_threshold": threshold,
    }
