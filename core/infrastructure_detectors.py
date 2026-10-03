"""Pretrained infrastructure detectors: RF-DETR road damage and YOLO11-seg power lines.

Two registry kinds live here, both run through onnxruntime on the CPU:

``onnx_detr``
    RF-DETR (``road_damage_rfdetr_small``). The graph takes a 512 x 512 RGB image,
    stretched (not letterboxed) and ImageNet-normalised, and returns 300 query boxes
    (normalised cx, cy, w, h) with one logit per class. Decoding is RF-DETR's own
    post-process: sigmoid, top-k over queries x classes, no NMS.

``onnx_yolo_seg``
    Ultralytics YOLO11-seg (``powerline_ttpla_yolo11s_seg``). ``(1, 4 + nc + 32, N)``
    boxes with mask coefficients, plus prototypes ``(1, 32, 160, 160)``. Masks are
    sigmoid(coefficients @ prototypes), cropped to the box, upsampled to the input and
    un-letterboxed to the source image.

What this module will not do:

- Run a file whose digest is not the one the registry recorded. Each load hashes the
  file and refuses on a mismatch or a missing digest; a model that differs from the one
  the published metric describes is a different model.
- Return an empty list when the model is missing or the graph will not load. Both raise
  ``ModelRunRefused`` with the reason, unlike ``detection._load_onnx_net``, which
  returns ``None`` and lets callers report "nothing found".
- Present a detection as a measurement. Every result is a pre-label for review; a
  power-line mask is image geometry, not a clearance distance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detection import letterbox_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_SESSIONS: dict[tuple[str, str], Any] = {}


def _refused(message: str) -> Exception:
    from .model_runner import ModelRunRefused

    return ModelRunRefused(message)


# ----------------------------------------------------------------- loading


@dataclass
class LoadedModel:
    """A registry model whose file was found and whose digest matched the registry."""

    key: str
    kind: str
    path: Path
    sha256: str
    labels: list[str]
    row: dict[str, Any]
    session: Any = field(repr=False, default=None)

    @property
    def model_used(self) -> str:
        return f"onnx:{self.path.name}"


def verified_model_path(model_key: str) -> tuple[Any, Path, dict[str, Any], str]:
    """The registry spec, path, row and verified digest, or ModelRunRefused.

    Refuses when the key is unknown, the file is absent, no digest was recorded, or the
    file on disk is not the recorded file.
    """
    from .model_runner import _installed
    from .models import verify_model_identity

    spec, path, row = _installed(model_key)
    check = verify_model_identity(model_key)
    if check.get("status") != "verified":
        raise _refused(f"{model_key} cannot run: {check.get('detail') or check.get('status')}")
    return spec, path, row, str(check["actual_sha256"])


def _session(path: Path, sha256: str) -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - onnxruntime is a hard dependency
        raise _refused(f"onnxruntime is not importable: {exc}") from exc
    key = (str(path), sha256)
    sess = _SESSIONS.get(key)
    if sess is not None:
        return sess
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, (__import__("os").cpu_count() or 2) // 2))
    try:
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise _refused(f"The graph at {path} would not load: {exc}") from exc
    _SESSIONS[key] = sess
    return sess


def onnx_metadata_names(session: Any) -> list[str] | None:
    """Class names from Ultralytics' ``names`` metadata, in index order, if present."""
    try:
        raw = session.get_modelmeta().custom_metadata_map.get("names")
    except Exception:
        return None
    if not raw:
        return None
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(v) for v in names]
    return None


def load_model(model_key: str, *, expected_kinds: frozenset[str] | set[str]) -> LoadedModel:
    spec, path, row, digest = verified_model_path(model_key)
    if spec.kind not in expected_kinds:
        raise _refused(f"{model_key} is a {spec.kind}, not one of {sorted(expected_kinds)}.")
    sess = _session(path, digest)
    labels = list(spec.labels)
    meta = onnx_metadata_names(sess)
    if meta is not None and labels and meta != labels:
        # The graph names its own classes; a registry that disagrees would put the wrong
        # name on every box.
        raise _refused(
            f"{model_key}: the registry labels {labels} disagree with the names in the "
            f"ONNX metadata {meta}.")
    if not labels and meta:
        labels = meta
    if not labels:
        raise _refused(f"{model_key} names no classes, so its boxes cannot be labelled.")
    return LoadedModel(model_key, spec.kind, path, digest, labels, row, sess)


# ----------------------------------------------------------------- RF-DETR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_detr(boxes_cxcywh: np.ndarray, logits: np.ndarray, image_hw: tuple[int, int],
                score_threshold: float, *, num_select: int = 300,
                num_labels: int | None = None) -> list[tuple[int, float, list[float]]]:
    """RF-DETR post-process for one image.

    ``boxes_cxcywh`` is ``(Q, 4)`` normalised to [0, 1]; ``logits`` is ``(Q, C)``.
    Scores are sigmoid(logits); the top ``num_select`` (query, class) pairs are kept
    regardless of query, so one query may yield two classes, exactly as RF-DETR's
    PostProcess does. Returns ``(class_id, score, [x1, y1, x2, y2])`` in source pixels,
    highest score first, for scores strictly above the threshold. Class slots at or
    beyond ``num_labels`` (a no-object slot, if the head has one) are dropped.
    """
    boxes = np.asarray(boxes_cxcywh, dtype=np.float32).reshape(-1, 4)
    scores = _sigmoid(np.asarray(logits, dtype=np.float32))
    q, c = scores.shape
    if boxes.shape[0] != q:
        raise ValueError(f"{boxes.shape[0]} boxes for {q} logit rows.")
    flat = scores.reshape(-1)
    k = int(min(num_select, flat.size))
    top = np.argpartition(-flat, k - 1)[:k]
    top = top[np.argsort(-flat[top], kind="stable")]
    h, w = image_hw
    out: list[tuple[int, float, list[float]]] = []
    for idx in top:
        score = float(flat[idx])
        if score <= score_threshold:
            break
        query, cls = int(idx // c), int(idx % c)
        if num_labels is not None and cls >= num_labels:
            continue
        cx, cy, bw, bh = boxes[query]
        x1 = float(np.clip((cx - bw / 2) * w, 0, w))
        y1 = float(np.clip((cy - bh / 2) * h, 0, h))
        x2 = float(np.clip((cx + bw / 2) * w, 0, w))
        y2 = float(np.clip((cy + bh / 2) * h, 0, h))
        out.append((cls, score, [x1, y1, x2, y2]))
    return out


def detr_input(image_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """RF-DETR's predict() preprocessing: RGB, bilinear stretch, ImageNet normalisation."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def run_detr(image_bgr: np.ndarray, model: LoadedModel,
             score_threshold: float) -> list[dict[str, Any]]:
    sess = model.session
    inp = sess.get_inputs()[0]
    resolution = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else int(
        model.row.get("input_size") or 512)
    outputs = sess.run(None, {inp.name: detr_input(image_bgr, resolution)})
    names = [o.name for o in sess.get_outputs()]
    by_name = dict(zip(names, outputs))
    boxes_key = next((n for n in names if by_name[n].shape[-1] == 4), None)
    logits_key = next((n for n in names if n != boxes_key), None)
    if boxes_key is None or logits_key is None:
        raise _refused(f"{model.key}: outputs {names} are not RF-DETR's boxes and logits.")
    h, w = image_bgr.shape[:2]
    hits = decode_detr(by_name[boxes_key][0], by_name[logits_key][0], (h, w),
                       float(score_threshold),
                       num_select=int(model.row.get("num_select") or 300),
                       num_labels=len(model.labels))
    return [_detection(model.labels[c], s, b) for c, s, b in hits]


def _detection(label: str, score: float, box: list[float]) -> dict[str, Any]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return {"label": label, "confidence": float(score), "bbox_xyxy": [x1, y1, x2, y2],
            "area_px": int(max(0, x2 - x1) * max(0, y2 - y1))}


# ----------------------------------------------------------------- YOLO-seg


def yolo_candidates(pred: np.ndarray, num_classes: int, score_threshold: float,
                    iou_threshold: float, *, extra: int = 0) -> tuple[np.ndarray, ...]:
    """Score filter and per-class NMS on one image's ``(4 + nc + extra, N)`` YOLO output.

    Returns ``(boxes_xyxy, scores, class_ids, extra_columns)`` in the network's input
    coordinates, highest score first.
    """
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 4 + num_classes + extra:
        raise ValueError(
            f"Expected {4 + num_classes + extra} rows per anchor, got shape {arr.shape}.")
    arr = arr.T
    cls = arr[:, 4:4 + num_classes]
    class_ids = cls.argmax(1)
    scores = cls[np.arange(arr.shape[0]), class_ids]
    keep = scores > score_threshold
    arr, class_ids, scores = arr[keep], class_ids[keep], scores[keep]
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64),
             np.zeros((0, extra), np.float32))
    if arr.shape[0] == 0:
        return empty
    cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    idxs = cv2.dnn.NMSBoxesBatched(
        [[float(a), float(b), float(c), float(d)] for a, b, c, d in
         zip(xyxy[:, 0], xyxy[:, 1], bw, bh)],
        [float(s) for s in scores], [int(c) for c in class_ids],
        float(score_threshold), float(iou_threshold))
    if idxs is None or len(idxs) == 0:
        return empty
    idxs = np.asarray(idxs).reshape(-1)
    idxs = idxs[np.argsort(-scores[idxs], kind="stable")]
    return (xyxy[idxs], scores[idxs], class_ids[idxs].astype(np.int64),
            arr[idxs, 4 + num_classes:])


def decode_instance_mask(coefficients: np.ndarray, protos: np.ndarray,
                         box_input_xyxy: np.ndarray, *, input_size: int, gain: float,
                         left: int, top: int, new_w: int, new_h: int,
                         image_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """One instance's mask in source pixels, as (box-local bool mask, source box).

    Ultralytics' process_mask, then scale_masks, for one instance: sigmoid of the
    coefficient-weighted prototype sum, zeroed outside the box (at prototype scale),
    bilinearly upsampled to the network input, thresholded at 0.5, and the letterbox
    padding removed before scaling to the source image. Only the box region of the
    source image is materialised, so a 4K frame with a hundred instances does not
    allocate a hundred 4K masks.
    """
    c, mh, mw = protos.shape
    # sigmoid(x) > 0.5 exactly when x > 0, so the threshold is applied to the logit.
    # Ultralytics (8.3) crops and upsamples the logits, not the probabilities, and
    # zeroes the logit outside the box; doing the same keeps the boundary pixels equal.
    logits = (np.asarray(coefficients, np.float32).reshape(1, c)
              @ protos.reshape(c, -1)).reshape(mh, mw)
    sx, sy = mw / float(input_size), mh / float(input_size)
    bx1, by1, bx2, by2 = (float(v) for v in box_input_xyxy)
    cols = np.arange(mw, dtype=np.float32)[None, :]
    rows = np.arange(mh, dtype=np.float32)[:, None]
    inside = ((cols >= bx1 * sx) & (cols < bx2 * sx) & (rows >= by1 * sy) & (rows < by2 * sy))
    logits = np.where(inside, logits, 0.0).astype(np.float32)
    full = cv2.resize(logits, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    binary = (full > 0.0).astype(np.uint8)
    unpadded = binary[top:top + new_h, left:left + new_w]
    h, w = image_hw
    # The source box this instance occupies, from the letterbox box.
    x1 = int(np.clip(np.floor((bx1 - left) / gain), 0, w))
    y1 = int(np.clip(np.floor((by1 - top) / gain), 0, h))
    x2 = int(np.clip(np.ceil((bx2 - left) / gain), 0, w))
    y2 = int(np.clip(np.ceil((by2 - top) / gain), 0, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), bool), (x1, y1, x2, y2)
    # Map the source box back into the unpadded input, crop, and resize just that part.
    ux1, uy1 = x1 * gain, y1 * gain
    ux2, uy2 = x2 * gain, y2 * gain
    cx1, cy1 = int(np.floor(ux1)), int(np.floor(uy1))
    cx2 = int(min(unpadded.shape[1], max(cx1 + 1, np.ceil(ux2))))
    cy2 = int(min(unpadded.shape[0], max(cy1 + 1, np.ceil(uy2))))
    crop = unpadded[cy1:cy2, cx1:cx2].astype(np.float32)
    if crop.size == 0:
        return np.zeros((y2 - y1, x2 - x1), bool), (x1, y1, x2, y2)
    local = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) > 0.5
    return local, (x1, y1, x2, y2)


def mask_polygons(local: np.ndarray, origin: tuple[int, int], *,
                  epsilon_px: float = 1.5) -> list[list[list[int]]]:
    """Outer contours of a box-local mask, in source pixels, simplified."""
    if local.size == 0 or not local.any():
        return []
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = origin
    polys = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append([[int(x + ox), int(y + oy)] for x, y in approx])
    return polys


def run_yolo_seg(image_bgr: np.ndarray, model: LoadedModel, score_threshold: float,
                 iou_threshold: float, *, keep_masks: bool = False) -> list[dict[str, Any]]:
    sess = model.session
    inp = sess.get_inputs()[0]
    size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else int(
        model.row.get("input_size") or 640)
    lb = letterbox_image(image_bgr, size)
    outputs = sess.run(None, {inp.name: lb.blob.astype(np.float32)})
    pred = next(o for o in outputs if o.ndim == 3)[0]
    protos = next(o for o in outputs if o.ndim == 4)[0]
    nm = int(protos.shape[0])
    boxes, scores, class_ids, coeffs = yolo_candidates(
        pred, len(model.labels), score_threshold, iou_threshold, extra=nm)
    h, w = image_bgr.shape[:2]
    out: list[dict[str, Any]] = []
    for box, score, cid, coef in zip(boxes, scores, class_ids, coeffs):
        local, (x1, y1, x2, y2) = decode_instance_mask(
            coef, protos, box, input_size=size, gain=lb.gain, left=lb.left, top=lb.top,
            new_w=lb.new_w, new_h=lb.new_h, image_hw=(h, w))
        if x2 <= x1 or y2 <= y1:
            continue
        det = _detection(model.labels[int(cid)], float(score), [x1, y1, x2, y2])
        det["mask_area_px"] = int(local.sum())
        det["mask_polygons_px"] = mask_polygons(local, (x1, y1))
        if keep_masks:
            det["_mask_local"] = local
        out.append(det)
    return out


# ----------------------------------------------------------------- overlay


def overlay_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    from .detection import _label_color

    overlay = image_bgr.copy()
    for det in detections:
        color = _label_color(str(det["label"]))
        polys = det.get("mask_polygons_px") or []
        if polys:
            fill = overlay.copy()
            for poly in polys:
                cv2.fillPoly(fill, [np.asarray(poly, np.int32)], color)
            overlay = cv2.addWeighted(fill, 0.45, overlay, 0.55, 0)
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{det['label']} {det['confidence']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay
