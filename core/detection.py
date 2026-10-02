"""Defect detection utilities (classical + optional model-backed inference)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except Exception:  # pragma: no cover - optional dependency
    _skimage_skeletonize = None

from .measurements import skeleton_length_px
from .models import get_model_spec, model_identity, model_status

LOGGER = logging.getLogger(__name__)


_MODEL_NET_CACHE: dict[str, Any] = {}

#: Registry kinds that are YOLO-layout ONNX detectors. The EL solar model was exported
#: as "onnx_detection" and was rejected by an equality check against "onnx_yolo", so an
#: installed, measured model silently fell back to the heuristic.
DETECTION_KINDS = frozenset({"onnx_yolo", "onnx_detection"})


def _skeletonize_binary(binary_mask: np.ndarray) -> np.ndarray:
    """
    Return a 1-pixel skeleton for a binary mask.

    Uses scikit-image when available, otherwise falls back to an OpenCV
    morphological skeletonization implementation so runtime does not depend on
    scikit-image.
    """
    if _skimage_skeletonize is not None:
        return _skimage_skeletonize(binary_mask > 0).astype(np.uint8)

    work = ((binary_mask > 0).astype(np.uint8) * 255).copy()
    skel = np.zeros_like(work, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(work, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return (skel > 0).astype(np.uint8)


@dataclass
class CrackDetectionResult:
    """Result bundle for crack detection."""

    mask: np.ndarray
    overlay: np.ndarray
    crack_pixels: int
    total_pixels: int
    crack_ratio: float
    estimated_length_px: float
    estimated_max_width_px: float
    # "heuristic" when the classical path produced the mask, "onnx:<file>" when a
    # trained segmentation model did. Reported verbatim so a run can never claim a
    # model it did not use.
    model_used: str = "heuristic"
    # Which model, and the digest of the file that actually ran. Empty for the
    # heuristic path, which has no model identity to record.
    model_key: str = ""
    model_sha256: str = ""


@dataclass
class MetalDefectResult:
    """Result bundle for metal defect detection.

    Metal defects are read by the trained corrosion model (corrosion_severity_segmentation,
    SegFormer-B2 on close-range photos of corroded steel). Every pixel it grades fair,
    poor or severe is a defect pixel; `good` is sound surface. It finds corrosion and
    nothing else -- not fatigue cracks, section loss or missing fasteners.

    When the model is not installed, or cannot run, nothing is guessed: `model_used` is
    `refused`, the mask is empty and `refused_reason` says why. An empty refused result
    means "not looked at", which is not the same as "no corrosion".

    The texture-discontinuity rule that used to answer here marked a third of an aerial
    farmland photo as metal damage. It survives only as `metal_texture_heuristic`, for
    the classical modes that label their output as classical.
    """

    mask: np.ndarray
    overlay: np.ndarray
    defect_pixels: int
    total_pixels: int
    defect_ratio: float
    defect_regions: int
    model_used: str = "heuristic"
    model_key: str = ""
    model_sha256: str = ""
    class_pixel_ratio: dict[str, float] = field(default_factory=dict)
    worst_present_grade: str = ""
    refused_reason: str = ""
    scope: str = ""

    @property
    def refused(self) -> bool:
        return self.model_used == "refused"

    def to_summary(self) -> dict[str, Any]:
        return {
            "defect_pixels": int(self.defect_pixels),
            "total_pixels": int(self.total_pixels),
            "defect_ratio": float(self.defect_ratio),
            "defect_regions": int(self.defect_regions),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "worst_present_grade": self.worst_present_grade,
            "model_used": self.model_used,
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
            "refused_reason": self.refused_reason,
            "scope": self.scope,
        }


@dataclass
class DefectHit:
    label: str
    # The model's score, or None when nothing produced one. The classical paths used to
    # derive a "confidence" from region area (0.35..0.95), which read exactly like a
    # model score and meant nothing: 963 regions on a farmland photo at 0.95. A heuristic
    # region has no calibrated score, so it reports none.
    confidence: float | None
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": None if self.confidence is None else float(self.confidence),
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "area_px": int(self.area_px),
        }


@dataclass
class StructuralDefectResult:
    """Multi-class structural defect result."""

    mask: np.ndarray
    overlay: np.ndarray
    mask_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    detections: list[DefectHit] = field(default_factory=list)
    defect_pixels: int = 0
    total_pixels: int = 0
    defect_ratio: float = 0.0
    class_pixel_ratio: dict[str, float] = field(default_factory=dict)
    model_used: str = "heuristic"
    model_available: bool = False
    model_key: str = ""
    model_sha256: str = ""
    #: Boxes the model drew for a class that is not a defect (CODEBRIM's "Background"
    #: patches of sound concrete). Kept for the record, never counted as defects.
    non_defect_detections: list[DefectHit] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "non_defect_detections": [d.to_dict() for d in self.non_defect_detections],
            "defect_pixels": int(self.defect_pixels),
            "total_pixels": int(self.total_pixels),
            "defect_ratio": float(self.defect_ratio),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "detections": [d.to_dict() for d in self.detections],
            "model_used": self.model_used,
            "model_available": bool(self.model_available),
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
        }


@dataclass
class SolarDefectResult:
    """Solar-specific defect result.

    Produced by one of the two installed solar models, chosen by what the image IS:
    `el_cell` (electroluminescence image of cells -> solar_cell_defect_detector, boxes
    with detector scores) or `thermal_module` (one infrared module crop ->
    solar_thermal_anomaly_classifier, a module-level class with its probability). There
    is no model for RGB photographs of panels unless one is installed under an RGB
    solar key; otherwise, and whenever the model is missing, the result is `refused`
    with the reason and an empty mask. Nothing here is a colour or edge rule.
    """

    mask: np.ndarray
    overlay: np.ndarray
    mask_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    detections: list[DefectHit] = field(default_factory=list)
    defect_pixels: int = 0
    total_pixels: int = 0
    defect_ratio: float = 0.0
    class_pixel_ratio: dict[str, float] = field(default_factory=dict)
    model_used: str = "refused"
    model_available: bool = False
    model_key: str = ""
    model_sha256: str = ""
    # Only a radiometric reading could fill this, and no path here has one; kept so a
    # consumer can tell "not measured" (None) from a temperature.
    hotspot_max_temp_c: float | None = None
    imagery: str = ""
    refused_reason: str = ""
    scope: str = ""
    # thermal_module only: the classifier's probability for every class.
    class_probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.model_used == "refused"

    def to_summary(self) -> dict[str, Any]:
        return {
            "defect_pixels": int(self.defect_pixels),
            "total_pixels": int(self.total_pixels),
            "defect_ratio": float(self.defect_ratio),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "detections": [d.to_dict() for d in self.detections],
            "model_used": self.model_used,
            "model_available": bool(self.model_available),
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
            "hotspot_max_temp_c": self.hotspot_max_temp_c,
            "imagery": self.imagery,
            "refused_reason": self.refused_reason,
            "scope": self.scope,
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
        }


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk and raise a clear error if it fails."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def _filter_connected_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove tiny blobs from a binary mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # One lookup table indexed by label, not one full-image comparison per component:
    # a noisy 18 MP frame has tens of thousands of blobs, and the per-label loop spent
    # minutes per image here before the pipeline reached reconstruction.
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area_px)
    keep[0] = False
    lut = np.where(keep, 255, 0).astype(np.uint8)
    return lut[labels]


def _mask_to_hits(mask: np.ndarray, label: str, min_area_px: int,
                  confidence: float | None = None) -> list[DefectHit]:
    hits: list[DefectHit] = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        hits.append(
            DefectHit(
                label=label,
                confidence=None if confidence is None else float(np.clip(confidence, 0.01, 0.99)),
                bbox_xyxy=(x, y, x + w, y + h),
                area_px=area,
            )
        )
    return hits


def _label_color(label: str) -> tuple[int, int, int]:
    table = {
        "crack": (0, 0, 255),
        "corrosion": (255, 255, 0),
        "spalling": (0, 165, 255),
        "delamination": (255, 0, 255),
        "rebar_exposure": (42, 42, 165),
        "efflorescence": (255, 255, 255),
        "moisture_intrusion": (255, 0, 0),
        "hotspot": (0, 0, 255),
        "cell_crack": (0, 69, 255),
        "soiling": (32, 128, 192),
        "string_fault": (0, 255, 255),
    }
    return table.get(label, (0, 255, 0))


def _overlay_by_masks(image_bgr: np.ndarray, mask_by_class: dict[str, np.ndarray]) -> np.ndarray:
    overlay = image_bgr.copy()
    if not mask_by_class:
        return overlay
    for label, mask in mask_by_class.items():
        if mask is None or mask.size == 0:
            continue
        color = _label_color(label)
        overlay[mask > 0] = color
    return cv2.addWeighted(image_bgr, 0.66, overlay, 0.34, 0.0)


def _union_mask(mask_by_class: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for mask in mask_by_class.values():
        if mask is None or mask.size == 0:
            continue
        union = cv2.bitwise_or(union, (mask > 0).astype(np.uint8) * 255)
    return union


def _load_onnx_net(model_path: Path) -> Any | None:
    key = str(model_path.resolve())
    net = _MODEL_NET_CACHE.get(key)
    if net is not None:
        return net
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
    except Exception:
        return None
    _MODEL_NET_CACHE[key] = net
    return net


def _run_onnx_yolo(
    image_bgr: np.ndarray,
    model_path: Path,
    labels: list[str],
    score_threshold: float,
    iou_threshold: float,
    input_size: int,
    *,
    letterbox: bool = True,
    class_agnostic_nms: bool = False,
) -> list[DefectHit]:
    """Run a YOLO-layout ONNX detector and return boxes in source-image pixels.

    ``letterbox`` matches how Ultralytics trains and validates these models: the image
    is scaled to fit the square input with its aspect ratio kept and the rest padded
    with grey (114). Stretching a 4:3 drone frame to a square instead squashes every
    object by a quarter along one axis, which the model never saw in training. On a
    square image the two are identical. ``letterbox=False`` keeps the old stretch for
    comparison (tools/eval_external_detectors.py measures both).
    """
    net = _load_onnx_net(model_path)
    if net is None:
        return []

    size = int(max(64, input_size))
    h, w = image_bgr.shape[:2]
    if letterbox:
        gain = min(size / float(h), size / float(w))
        new_w, new_h = max(1, int(round(w * gain))), max(1, int(round(h * gain)))
        pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(size, size),
                                     swapRB=True, crop=False)
        sx = sy = 1.0 / gain
        off_x, off_y = float(left), float(top)
    else:
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(size, size),
            swapRB=True,
            crop=False,
        )
        sx = float(w) / float(size)
        sy = float(h) / float(size)
        off_x = off_y = 0.0
    net.setInput(blob)
    out = net.forward()
    arr = np.asarray(out)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return []
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
        arr = arr.T
    if arr.shape[1] < 6:
        return []

    label_count = max(1, len(labels))
    columns = int(arr.shape[1])
    if columns == 4 + label_count:
        # YOLOv8/11 layout: cx, cy, w, h, then one score per class (no objectness).
        cls_scores = arr[:, 4:].astype(np.float32)
        class_ids_all = np.argmax(cls_scores, axis=1)
        conf_all = cls_scores[np.arange(arr.shape[0]), class_ids_all]
    elif columns == 5 + label_count:
        # YOLOv5 layout: objectness, then class scores.
        cls_scores = arr[:, 5:].astype(np.float32)
        class_ids_all = np.argmax(cls_scores, axis=1)
        conf_all = arr[:, 4].astype(np.float32) * cls_scores[np.arange(arr.shape[0]),
                                                             class_ids_all]
    else:
        # The graph's class count does not match the registry's labels. Guessing which
        # column is which would attach the wrong names to real boxes, so refuse.
        raise ValueError(
            f"{model_path.name} emits {columns} values per box, which fits neither the "
            f"YOLOv8 ({4 + label_count}) nor the YOLOv5 ({5 + label_count}) layout for "
            f"the {label_count} registered labels.")

    keep = conf_all >= float(score_threshold)
    if not np.any(keep):
        return []
    rows = arr[keep].astype(np.float32)
    confidences = [float(c) for c in conf_all[keep]]
    class_ids = [int(c) for c in class_ids_all[keep]]
    cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    x1s = (cx - bw * 0.5 - off_x) * sx
    y1s = (cy - bh * 0.5 - off_y) * sy
    boxes_xywh = [[float(a), float(b), float(c), float(d)]
                  for a, b, c, d in zip(x1s, y1s, bw * sx, bh * sy)]

    if class_agnostic_nms:
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, score_threshold, iou_threshold)
    else:
        # Per-class NMS, as Ultralytics validates: a crack box must not suppress a
        # spalling box over the same patch of concrete.
        idxs = cv2.dnn.NMSBoxesBatched(boxes_xywh, confidences, class_ids,
                                       score_threshold, iou_threshold)
    if idxs is None or len(idxs) == 0:
        return []

    out_hits: list[DefectHit] = []
    for idx in np.asarray(idxs).reshape(-1):
        x, y, bw, bh = boxes_xywh[int(idx)]
        x1 = int(np.clip(round(x), 0, w - 1))
        y1 = int(np.clip(round(y), 0, h - 1))
        x2 = int(np.clip(round(x + bw), 0, w - 1))
        y2 = int(np.clip(round(y + bh), 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(class_ids[int(idx)])
        label = labels[cid] if 0 <= cid < len(labels) else f"class_{cid}"
        out_hits.append(
            DefectHit(
                label=label,
                confidence=float(confidences[int(idx)]),
                bbox_xyxy=(x1, y1, x2, y2),
                area_px=int((x2 - x1) * (y2 - y1)),
            )
        )
    return out_hits


def _hits_to_masks(
    image_shape: tuple[int, int],
    hits: list[DefectHit],
) -> dict[str, np.ndarray]:
    h, w = image_shape
    by_class: dict[str, np.ndarray] = {}
    for hit in hits:
        x1, y1, x2, y2 = hit.bbox_xyxy
        if hit.label not in by_class:
            by_class[hit.label] = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(by_class[hit.label], (x1, y1), (x2, y2), color=255, thickness=-1)
    return by_class


def _run_onnx_segmentation(
    image_bgr: np.ndarray,
    model_path: Path,
    threshold: float,
    input_size: int,
    *,
    raise_errors: bool = False,
) -> np.ndarray | None:
    """Run a binary segmentation ONNX model and return a full-resolution 0/1 mask.

    The exporter folds the upsample and sigmoid into the graph, so the output is
    already a probability map at input resolution and only needs thresholding and a
    resize back to the source image size.

    With ``raise_errors`` a failure raises with its real cause (a graph that will
    not load, a forward pass that ran out of memory, an output of the wrong shape)
    instead of returning ``None``, which callers can only report as "no mask".
    """
    net = _load_onnx_net(model_path)
    if net is None:
        if raise_errors:
            raise RuntimeError(f"The ONNX runtime could not load {model_path.name}.")
        return None

    size = int(max(64, input_size))
    # ImageNet normalisation, matching training. cv2.dnn applies mean subtraction
    # before scaling, so the std division is done afterwards by hand.
    blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True, crop=False
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    blob = (blob - mean) / std

    net.setInput(blob)
    try:
        output = np.asarray(net.forward())
    except Exception:
        if raise_errors:
            raise
        return None

    probability = np.squeeze(output)
    if probability.ndim != 2:
        if raise_errors:
            raise ValueError(
                f"{model_path.name} returned an output of shape {tuple(output.shape)}, "
                "not a single-channel probability map."
            )
        return None
    if probability.min() < 0.0 or probability.max() > 1.0:
        # A graph exported without the sigmoid still yields logits; apply it here
        # rather than thresholding raw scores against a probability cut-off.
        probability = 1.0 / (1.0 + np.exp(-probability))

    binary = (probability >= float(threshold)).astype(np.uint8)
    height, width = image_bgr.shape[:2]
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary


class CorrosionGradingRefused(ValueError):
    """Raised rather than returning a severity nothing measured.

    Every other detector here degrades to a heuristic when its model is missing, and
    reports `heuristic` so the caller can tell. Severity grading has no such fallback:
    colour and texture rules can find rust, but "poor versus severe" is a judgement the
    corpus taught and no threshold reproduces. A guessed grade drives a maintenance
    decision, so this refuses instead.
    """


@dataclass
class CorrosionSeverityResult:
    """A per-pixel severity grade, and the evidence for what produced it."""

    grade_index: np.ndarray
    overlay: np.ndarray
    labels: list[str]
    class_pixel_ratio: dict[str, float]
    worst_present_grade: str
    total_pixels: int
    model_used: str
    model_key: str = ""
    model_sha256: str = ""

    def to_summary(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "worst_present_grade": self.worst_present_grade,
            "total_pixels": int(self.total_pixels),
            "model_used": self.model_used,
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
        }


def _run_onnx_segmentation_multiclass(
    image_bgr: np.ndarray,
    model_path: Path,
    input_size: int,
    class_count: int,
    *,
    raise_errors: bool = False,
) -> np.ndarray | None:
    """Run a multiclass segmentation graph and return a full-resolution class index map.

    ``raise_errors`` behaves as in :func:`_run_onnx_segmentation`.

    The graph ends in a softmax over the channel axis, so the answer is the argmax and
    not a threshold: the classes are mutually exclusive and only interpretable against
    each other. Thresholding one channel of a softmax against 0.5 would silently return
    "nothing" for any pixel the model is merely unsure about.
    """
    net = _load_onnx_net(model_path)
    if net is None:
        if raise_errors:
            raise RuntimeError(f"The ONNX runtime could not load {model_path.name}.")
        return None

    size = int(max(64, input_size))
    blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True, crop=False
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    blob = (blob - mean) / std

    net.setInput(blob)
    try:
        output = np.asarray(net.forward())
    except Exception:
        if raise_errors:
            raise
        return None

    probabilities = np.squeeze(output, axis=0) if output.ndim == 4 else np.squeeze(output)
    if probabilities.ndim != 3 or probabilities.shape[0] != class_count:
        # A graph whose channel count disagrees with the registered labels would map
        # every pixel onto the wrong name. Refusing the array is the only safe answer.
        if raise_errors:
            raise ValueError(
                f"{model_path.name} returned an output of shape {tuple(output.shape)}, "
                f"which does not match its {class_count} labels."
            )
        return None

    grade = probabilities.argmax(axis=0).astype(np.uint8)
    height, width = image_bgr.shape[:2]
    if grade.shape != (height, width):
        grade = cv2.resize(grade, (width, height), interpolation=cv2.INTER_NEAREST)
    return grade


def grade_corrosion_severity(
    image_bgr: np.ndarray,
    *,
    model_key: str = "corrosion_severity_segmentation",
) -> CorrosionSeverityResult:
    """Grade every pixel on the corrosion severity scale, or refuse.

    `worst_present_grade` is the highest grade the model actually assigns to any pixel,
    read off the label order rather than a score, because the classes are ordinal: the
    last label is the worst by construction.
    """
    spec = get_model_spec(model_key)
    if spec is None:
        raise CorrosionGradingRefused(f"{model_key} is not in the model registry.")

    info = model_status(model_key)
    if not info.get("exists"):
        raise CorrosionGradingRefused(
            f"{model_key} is registered but its weights are not installed at "
            f"{info.get('path', '?')}. There is no heuristic severity grade to fall "
            "back to, so nothing is returned."
        )

    labels = list(getattr(spec, "labels", []) or [])
    if len(labels) < 2:
        raise CorrosionGradingRefused(
            f"{model_key} declares {len(labels)} labels; a severity scale needs at least two."
        )

    grade = _run_onnx_segmentation_multiclass(
        image_bgr,
        Path(str(info["path"])),
        input_size=int(getattr(spec, "input_size", 512) or 512),
        class_count=len(labels),
    )
    if grade is None:
        raise CorrosionGradingRefused(
            f"{model_key} did not produce a usable grade map -- the graph would not "
            "load, or its channel count disagrees with the registered labels."
        )

    total = int(grade.size)
    counts = np.bincount(grade.reshape(-1), minlength=len(labels))[: len(labels)]
    ratios = {label: float(counts[index]) / total for index, label in enumerate(labels)}
    present = [label for index, label in enumerate(labels) if counts[index] > 0]

    identity = model_identity(model_key)
    return CorrosionSeverityResult(
        grade_index=grade,
        overlay=_overlay_by_masks(
            image_bgr,
            {label: (grade == index).astype(np.uint8) * 255
             for index, label in enumerate(labels) if counts[index] > 0},
        ),
        labels=labels,
        class_pixel_ratio=ratios,
        worst_present_grade=present[-1] if present else "",
        total_pixels=total,
        model_used=f"onnx:{Path(str(info['path'])).name}",
        model_key=identity["model_key"],
        model_sha256=identity["model_sha256"],
    )


def detect_cracks(
    image_bgr: np.ndarray,
    min_area_px: int = 30,
    *,
    model_key: str = "crack_segmentation",
    use_model: bool = True,
) -> CrackDetectionResult:
    """Segment cracks, preferring a trained model and falling back to morphology.

    The classical path stays available and is reported as `heuristic`; it is never
    presented as a model result.
    """
    binary: np.ndarray | None = None
    model_used = "heuristic"

    if use_model:
        model_info = model_status(model_key)
        spec = get_model_spec(model_key)
        path_str = str(model_info.get("path", "")) if model_info.get("exists") else ""
        if spec is not None and path_str and spec.kind == "onnx_segmentation":
            predicted = _run_onnx_segmentation(
                image_bgr,
                Path(path_str),
                threshold=float(getattr(spec, "score_threshold", 0.5) or 0.5),
                input_size=int(getattr(spec, "input_size", 512) or 512),
            )
            if predicted is not None and predicted.any():
                binary = _filter_connected_components(
                    (predicted * 255).astype(np.uint8), min_area_px=min_area_px
                )
                binary = (binary > 0).astype(np.uint8)
                model_used = f"onnx:{Path(path_str).name}"
            elif predicted is not None:
                # The model ran and found nothing. That is a real answer, not a
                # failure, so it must not be overwritten by the heuristic.
                binary = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                model_used = f"onnx:{Path(path_str).name}"
            else:
                model_used = "onnx_fallback_heuristic"

    if binary is None:
        binary = _crack_mask_heuristic(image_bgr, min_area_px)

    return _crack_result_from_mask(image_bgr, binary, model_used, model_key)


def _identity_for(model_used: str, model_key: str) -> dict[str, str]:
    """Model identity, but only when a model actually produced the result.

    Derived here rather than assigned at each branch, so the recorded identity and the
    reported model_used cannot drift apart: one is computed from the other.
    """
    if not str(model_used).startswith("onnx:"):
        return {"model_key": "", "model_sha256": ""}
    identity = model_identity(model_key)
    return {"model_key": identity["model_key"],
            "model_sha256": identity["model_sha256"]}


def _crack_result_from_mask(
    image_bgr: np.ndarray, binary: np.ndarray, model_used: str, model_key: str = ""
) -> CrackDetectionResult:
    """Derive length, width, and overlay from a binary crack mask."""
    identity = _identity_for(model_used, model_key)
    skeleton = _skeletonize_binary(binary)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    crack_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    max_width = float(np.nan_to_num(dist_map.max(), nan=0.0, posinf=0.0, neginf=0.0))

    overlay = image_bgr.copy()
    overlay[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0.0)

    return CrackDetectionResult(
        mask=(binary * 255).astype(np.uint8),
        overlay=overlay,
        crack_pixels=crack_pixels,
        total_pixels=total_pixels,
        crack_ratio=float(crack_pixels / max(total_pixels, 1)),
        estimated_length_px=skeleton_length_px(skeleton),
        estimated_max_width_px=float(
            np.clip(max_width * 2.0, 0.0, float(max(image_bgr.shape[:2]) * 2.0))
        ),
        model_used=model_used,
        model_key=identity["model_key"],
        model_sha256=identity["model_sha256"],
    )


def _crack_mask_heuristic(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    Detect crack-like structures using morphology + edge cues.
    This is a robust fallback when no trained model is available.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Black-hat highlights thin dark lines over brighter surfaces.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, bh_kernel)

    adaptive = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    edges = cv2.Canny(smooth, 40, 120)
    mask = cv2.bitwise_or(adaptive, edges)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = _filter_connected_components(mask, min_area_px=min_area_px)

    return (mask > 0).astype(np.uint8)


def metal_texture_heuristic(image_bgr: np.ndarray, min_area_px: int = 40) -> MetalDefectResult:
    """Texture-discontinuity rule. Classical, and labelled so: `model_used='heuristic'`.

    Not a metal defect detector -- it marks sharp texture of any kind (a third of an aerial
    farmland photo). Kept only for the classical modes, which report their output as
    classical with no confidence. `detect_metal_defects` never calls it.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)

    _, thresh = cv2.threshold(lap_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = _filter_connected_components(thresh, min_area_px=min_area_px)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = 0
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area_px:
            regions += 1

    defect_pixels = int((mask > 0).sum())
    total_pixels = int(mask.size)
    defect_ratio = float(defect_pixels / max(total_pixels, 1))

    overlay = image_bgr.copy()
    overlay[mask > 0] = (255, 255, 0)
    overlay = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0.0)

    return MetalDefectResult(
        mask=mask.astype(np.uint8),
        overlay=overlay,
        defect_pixels=defect_pixels,
        total_pixels=total_pixels,
        defect_ratio=defect_ratio,
        defect_regions=regions,
        model_used="heuristic",
    )


#: The trained model that answers for metal: corrosion condition state on steel.
METAL_MODEL_KEY = "corrosion_severity_segmentation"


def _model_scope(model_key: str) -> str:
    spec = get_model_spec(model_key)
    return str(getattr(spec, "description", "") or "") if spec is not None else ""


def _refused_metal(image_bgr: np.ndarray, reason: str, model_key: str) -> MetalDefectResult:
    h, w = image_bgr.shape[:2]
    return MetalDefectResult(
        mask=np.zeros((h, w), dtype=np.uint8),
        overlay=image_bgr.copy(),
        defect_pixels=0,
        total_pixels=int(h * w),
        defect_ratio=0.0,
        defect_regions=0,
        model_used="refused",
        refused_reason=reason,
        scope=_model_scope(model_key),
    )


def detect_metal_defects(
    image_bgr: np.ndarray,
    min_area_px: int = 40,
    *,
    model_key: str = METAL_MODEL_KEY,
    use_model: bool = True,
) -> MetalDefectResult:
    """Metal (corrosion) defects from the trained corrosion model, or an honest refusal.

    Every pixel the model grades above `good` on its ordinal scale is a defect pixel;
    regions smaller than `min_area_px` are dropped. `worst_present_grade` and the
    per-grade ratios travel with the result. There is no fallback: a missing model, or
    `use_model=False`, returns `model_used='refused'` with the reason and an empty mask.
    """
    if not use_model:
        return _refused_metal(
            image_bgr,
            "Metal detection was asked to run without a model. There is no trained-free "
            "metal detector: the texture rule that used to answer here marks texture, not "
            "defects, so nothing is reported.",
            model_key,
        )
    try:
        graded = grade_corrosion_severity(image_bgr, model_key=model_key)
    except CorrosionGradingRefused as exc:
        return _refused_metal(image_bgr, f"Metal detection refused: {exc}", model_key)

    labels = list(graded.labels)
    defect = (graded.grade_index > 0).astype(np.uint8) * 255
    mask = _filter_connected_components(defect, min_area_px=max(1, int(min_area_px)))
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    regions = sum(1 for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] >= min_area_px)
    defect_pixels = int((mask > 0).sum())
    total = int(mask.size)
    by_grade = {
        label: ((graded.grade_index == index) & (mask > 0)).astype(np.uint8) * 255
        for index, label in enumerate(labels) if index > 0
    }
    by_grade = {label: m for label, m in by_grade.items() if m.any()}
    return MetalDefectResult(
        mask=(mask > 0).astype(np.uint8),
        overlay=_overlay_by_masks(image_bgr, by_grade),
        defect_pixels=defect_pixels,
        total_pixels=total,
        defect_ratio=float(defect_pixels / max(total, 1)),
        defect_regions=int(regions),
        model_used=graded.model_used,
        model_key=graded.model_key,
        model_sha256=graded.model_sha256,
        class_pixel_ratio=dict(graded.class_pixel_ratio),
        worst_present_grade=graded.worst_present_grade,
        scope=_model_scope(model_key),
    )


def _detect_spalling_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    _, rough = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_sat = (hsv[:, :, 1] < 65).astype(np.uint8) * 255
    bright = (hsv[:, :, 2] > 125).astype(np.uint8) * 255
    mask = cv2.bitwise_and(rough, low_sat)
    mask = cv2.bitwise_and(mask, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return _filter_connected_components(mask, min_area_px=max(60, min_area_px))


def _detect_efflorescence_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 170), (179, 60, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return _filter_connected_components(white, min_area_px=max(80, min_area_px))


def _detect_moisture_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < 90).astype(np.uint8) * 255
    sat = (hsv[:, :, 1] > 35).astype(np.uint8) * 255
    mask = cv2.bitwise_and(dark, sat)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return _filter_connected_components(mask, min_area_px=max(120, min_area_px))


def _detect_rebar_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rust1 = cv2.inRange(hsv, (5, 90, 40), (25, 255, 255))
    rust2 = cv2.inRange(hsv, (0, 90, 30), (8, 255, 255))
    rust = cv2.bitwise_or(rust1, rust2)
    edges = cv2.Canny(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), 60, 170)
    mask = cv2.bitwise_and(rust, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return _filter_connected_components(mask, min_area_px=max(80, min_area_px))


def _structural_fallback(image_bgr: np.ndarray, min_area_px: int) -> tuple[dict[str, np.ndarray], list[DefectHit]]:
    mask_by_class: dict[str, np.ndarray] = {}
    detections: list[DefectHit] = []

    crack = detect_cracks(image_bgr, min_area_px=min_area_px)
    if int(crack.crack_pixels) > 0:
        mask_by_class["crack"] = crack.mask
        detections.extend(_mask_to_hits(crack.mask, "crack", min_area_px))

    metal = metal_texture_heuristic(image_bgr, min_area_px=max(40, min_area_px))
    if int(metal.defect_pixels) > 0:
        mask_by_class["corrosion"] = metal.mask
        detections.extend(_mask_to_hits(metal.mask, "corrosion", min_area_px))

    for label, mask in (
        ("spalling", _detect_spalling_mask(image_bgr, min_area_px)),
        ("efflorescence", _detect_efflorescence_mask(image_bgr, min_area_px)),
        ("moisture_intrusion", _detect_moisture_mask(image_bgr, min_area_px)),
        ("rebar_exposure", _detect_rebar_mask(image_bgr, min_area_px)),
    ):
        pixels = int(np.sum(mask > 0))
        if pixels <= 0:
            continue
        mask_by_class[label] = mask
        detections.extend(_mask_to_hits(mask, label, min_area_px))

    return mask_by_class, detections


def non_defect_labels(model_key: str) -> frozenset[str]:
    """Labels a registry model predicts that name sound surface, not a defect.

    The structural detector was trained on CODEBRIM, which annotates "Background"
    boxes over undamaged concrete as negative examples, so the model draws them too.
    Counting them as defects reported sound concrete as damage. The registry row says
    which labels these are (``non_defect_labels``) rather than this code guessing.
    """
    from .models import load_registry

    row = (load_registry().get("models") or {}).get(model_key) or {}
    return frozenset(str(v) for v in (row.get("non_defect_labels") or []))


def detect_structural_defects(
    image_bgr: np.ndarray,
    min_area_px: int = 40,
    model_key: str = "structural_multiclass_detector",
    use_model: bool = True,
) -> StructuralDefectResult:
    """
    Detect broad structural defects.

    If a model is available in models/model_registry.json and weight file exists,
    ONNX YOLO inference is used. Otherwise, a robust classical fallback is used.
    """
    h, w = image_bgr.shape[:2]
    total = int(max(1, h * w))
    model_info = model_status(model_key)

    mask_by_class: dict[str, np.ndarray] = {}
    hits: list[DefectHit] = []
    non_defect_hits: list[DefectHit] = []
    model_used = "heuristic"
    model_available = bool(model_info.get("exists", False))

    # Whether the model ran at all, as distinct from whether it found anything. A
    # trained detector reporting no defects on a sound wall is giving a real answer,
    # and replacing it with the heuristic would overwrite that answer with invented
    # findings. Only an actual inference failure justifies the fallback.
    model_ran = False

    if use_model and model_available:
        spec = get_model_spec(model_key)
        path_str = str(model_info.get("path", ""))
        if spec is not None and path_str and spec.kind in DETECTION_KINDS:
            try:
                model_hits = _run_onnx_yolo(
                    image_bgr=image_bgr,
                    model_path=Path(path_str),
                    labels=spec.labels,
                    score_threshold=float(spec.score_threshold),
                    iou_threshold=float(spec.iou_threshold),
                    input_size=int(spec.input_size),
                )
                model_ran = True
            except Exception as exc:
                # Swallowing this silently would make a broken model look like a clean
                # wall, so record why the fallback happened.
                model_hits = []
                model_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("ONNX structural inference failed (%s); using heuristic.",
                               model_error)
            not_defects = non_defect_labels(model_key)
            non_defect_hits = [h for h in model_hits if h.label in not_defects]
            model_hits = [h for h in model_hits if h.label not in not_defects]
            if model_hits:
                hits = model_hits
                mask_by_class = _hits_to_masks((h, w), hits)
                model_used = f"onnx:{Path(path_str).name}"
            elif model_ran:
                # Ran, found nothing. That is the result.
                model_used = f"onnx:{Path(path_str).name}"

    if not model_ran and (not hits or not mask_by_class):
        mask_by_class, hits = _structural_fallback(image_bgr, min_area_px=min_area_px)
        if use_model and model_available:
            model_used = "onnx_fallback_heuristic"
        else:
            model_used = "heuristic"

    union = _union_mask(mask_by_class, (h, w))
    defect_pixels = int(np.sum(union > 0))
    defect_ratio = float(defect_pixels / total)
    class_pixel_ratio: dict[str, float] = {}
    for label, mask in mask_by_class.items():
        class_pixel_ratio[label] = float(np.sum(mask > 0) / total)
    overlay = _overlay_by_masks(image_bgr, mask_by_class)

    identity = _identity_for(model_used, model_key)
    return StructuralDefectResult(
        mask=union,
        overlay=overlay,
        mask_by_class=mask_by_class,
        detections=hits,
        defect_pixels=defect_pixels,
        total_pixels=total,
        defect_ratio=defect_ratio,
        class_pixel_ratio=class_pixel_ratio,
        model_used=model_used,
        model_available=model_available,
        model_key=identity["model_key"],
        model_sha256=identity["model_sha256"],
        non_defect_detections=non_defect_hits,
    )


#: What a solar image can be, and which installed model reads each kind. The two
#: shipping models answer different questions about different imagery, and neither
#: reads an RGB photograph: the EL detector's registry entry says it "must not be
#: applied to drone photographs", and the thermal classifier is a per-module classifier
#: that "cannot find modules in a survey image".
SOLAR_IMAGERY = ("auto", "el_cell", "thermal_module", "rgb_photo")
SOLAR_MODELS_BY_IMAGERY: dict[str, tuple[str, ...]] = {
    "el_cell": ("solar_cell_defect_detector",),
    "thermal_module": ("solar_thermal_anomaly_classifier",),
    # Registered RGB panel detectors. Neither has weights today; if one is installed
    # it answers for photographs, and until then photographs are refused.
    "rgb_photo": ("solar_defect_detector", "solar_pv_multidefect_detector"),
}
#: The thermal classifier was trained on 24x40 px crops of single modules. A frame this
#: much larger holds many modules, which the classifier cannot separate.
THERMAL_MODULE_MAX_SIDE_PX = 256
#: The EL detector was trained on 1024x1024 PVEL-AD cell images; EL cell images from
#: any camera are hundreds of pixels a side. A picture this small is not one -- the
#: 24x40 px infrared module crops are the usual mix-up -- and the detector's
#: "no defects" on it would read exactly like a real clean cell.
EL_CELL_MIN_SIDE_PX = 128


def _is_grayscale(image_bgr: np.ndarray) -> bool:
    """True when the three channels carry the same picture (EL and IR crops do)."""
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        return True
    step_y = max(1, image_bgr.shape[0] // 256)
    step_x = max(1, image_bgr.shape[1] // 256)
    sample = image_bgr[::step_y, ::step_x].astype(np.int16)
    spread = np.maximum(np.abs(sample[..., 0] - sample[..., 1]),
                        np.abs(sample[..., 1] - sample[..., 2]))
    return float(np.mean(spread > 6)) < 0.01


def _busbar_lines(gray: np.ndarray, side: int = 256, strips: int = 8) -> int:
    """How many evenly spaced, full-span dark lines cross a grey picture (either axis).

    The busbars of a solar cell under electroluminescence are narrow dark lines that
    cross the whole cell at even spacing. A line counts only when it is a dip in the
    row profile of at least strips-1 of `strips` vertical bands, so a crack or a stain
    that crosses part of the picture is not one.
    """
    best = 0
    for arr in (gray, gray.T):
        a = cv2.resize(arr, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        width = side // strips
        hits = np.zeros(side, dtype=np.int32)
        for s in range(strips):
            profile = a[:, s * width:(s + 1) * width].mean(axis=1)
            base = np.array([np.median(profile[max(0, i - 10):i + 11]) for i in range(side)],
                            dtype=np.float32)
            detrended = profile - base
            dip = detrended < -max(6.0, float(np.std(detrended)))
            hits += np.convolve(dip.astype(np.int32), [1, 1, 1], "same") > 0
        rows = hits >= strips - 1
        margin = int(side * 0.04)
        rows[:margin] = False
        rows[-margin:] = False
        lines: list[tuple[int, int]] = []
        i = 0
        while i < side:
            if rows[i]:
                j = i
                while j < side and rows[j]:
                    j += 1
                lines.append((i, j))
                i = j
            else:
                i += 1
        lines = [ln for ln in lines if ln[1] - ln[0] <= 10]
        if len(lines) >= 2:
            centres = np.array([(x + y) / 2.0 for x, y in lines])
            gaps = np.diff(centres)
            if gaps.min() > side * 0.12 and gaps.std() / gaps.mean() < 0.2:
                best = max(best, len(lines))
    return best


def el_cell_signature(image_bgr: np.ndarray) -> dict[str, Any]:
    """Whether a picture carries the look of an electroluminescence cell image.

    Nobody declared what the image is, so this reads it and says which signs it saw:
    greyscale, at least EL_CELL_MIN_SIDE_PX a side, and one of: evenly spaced busbar
    lines crossing the whole cell; a bright cell inside a black frame (the dark room an
    EL image is taken in); a near-black field (a dead or black-core cell); dark
    chamfered corners of a pseudo-square cell. Measured on the corpora on this machine:
    1,442 of 1,450 PVEL-AD and ELPV cells, and 2 of 1,983 concrete, pavement and
    corrosion images (formwork panel joints read as busbars). `strong` is the busbar or
    frame sign, the two that pick a model; any sign only keeps surface models off.
    """
    h, w = image_bgr.shape[:2]
    out: dict[str, Any] = {"el_cell": False, "strong": False, "signs": []}
    if not _is_grayscale(image_bgr) or min(h, w) < EL_CELL_MIN_SIDE_PX:
        return out
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    a = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    signs: list[str] = []
    if _busbar_lines(gray) >= 2:
        signs.append("busbar_lines")
    band = 6
    sides = [a[:band].mean(), a[-band:].mean(), a[:, :band].mean(), a[:, -band:].mean()]
    if max(sides) < 40 and np.percentile(a[16:-16, 16:-16], 90) > 3 * max(max(sides), 8):
        signs.append("dark_frame")
    if float(np.median(a)) < 25:
        signs.append("near_black_field")
    c = 8
    corners = [a[:c, :c].mean(), a[:c, -c:].mean(), a[-c:, :c].mean(), a[-c:, -c:].mean()]
    if max(corners) < 0.6 * float(np.median(a[64:-64, 64:-64])):
        signs.append("dark_corners")
    out["signs"] = signs
    out["el_cell"] = bool(signs)
    out["strong"] = bool({"busbar_lines", "dark_frame"} & set(signs))
    return out


def _imagery_of_model(model_key: str) -> str:
    for imagery, keys in SOLAR_MODELS_BY_IMAGERY.items():
        if model_key in keys:
            return imagery
    return ""


def _solar_refusal(image_bgr: np.ndarray, reason: str, imagery: str,
                   model_key: str = "") -> SolarDefectResult:
    h, w = image_bgr.shape[:2]
    return SolarDefectResult(
        mask=np.zeros((h, w), dtype=np.uint8),
        overlay=image_bgr.copy(),
        total_pixels=int(h * w),
        model_used="refused",
        model_available=bool(model_key) and bool(model_status(model_key).get("exists")),
        imagery=imagery,
        refused_reason=reason,
        scope=_model_scope(model_key) if model_key else "",
    )


def _thermal_to_bgr(thermal: np.ndarray) -> np.ndarray:
    """A radiometric array as the 8-bit grey picture the classifier was trained on."""
    t = np.asarray(thermal, dtype=np.float32)
    if t.ndim == 3:
        t = t[..., 0]
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    scaled = np.zeros_like(t) if hi - lo < 1e-6 else (t - lo) / (hi - lo) * 255.0
    gray = np.nan_to_num(scaled).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detect_solar_defects(
    image_bgr: np.ndarray,
    thermal_gray: np.ndarray | None = None,
    min_area_px: int = 40,
    model_key: str | None = None,
    use_model: bool = True,
    imagery: str = "auto",
) -> SolarDefectResult:
    """Solar defects from an installed solar model, or a refusal that says why.

    `imagery` says what the picture is: `el_cell`, `thermal_module` or `rgb_photo`.
    `auto` reads it from the input -- a `thermal_gray` array is a thermal module, a
    colour picture is an RGB photograph, a grey picture with the busbars or black frame
    of an EL cell (`el_cell_signature`) is an EL cell, and any other grey picture is
    refused because an EL cell image and an IR module crop are both grey and need
    different models. A named
    `model_key` also settles it. Each kind goes to the model trained on it; a model
    applied to imagery it was not trained on is refused, not run, because its answer
    would read exactly like a real one.

    The classical colour/edge rules that used to answer here (963 "defects" on a
    farmland photo) are gone. `use_model=False` is a refusal.
    """
    h, w = image_bgr.shape[:2]
    requested = str(imagery or "auto").strip().lower()
    if requested not in SOLAR_IMAGERY:
        return _solar_refusal(image_bgr, f"Unknown solar imagery {imagery!r}; expected one "
                              f"of {', '.join(SOLAR_IMAGERY)}.", requested)
    key = str(model_key or "").strip()
    grey = _is_grayscale(image_bgr)

    if requested != "auto":
        kind = requested
    elif thermal_gray is not None:
        kind = "thermal_module"
    elif key and _imagery_of_model(key):
        kind = _imagery_of_model(key)
    elif not grey:
        kind = "rgb_photo"
    elif el_cell_signature(image_bgr)["strong"]:
        # Busbars across the whole cell, or a bright cell in a black frame: an EL cell
        # image, which an IR module crop (24x40 px, below the size floor) is not.
        kind = "el_cell"
    else:
        return _solar_refusal(
            image_bgr,
            "This is a greyscale image, and both an electroluminescence cell image and an "
            "infrared module crop are greyscale; they need different models. Declare the "
            "solar imagery (el_cell or thermal_module) to run one.",
            "auto")

    if not use_model:
        return _solar_refusal(
            image_bgr,
            "Solar detection was asked to run without a model. There is no model-free "
            "solar detector: the colour and edge rules that used to answer here marked "
            "texture, not panel defects, so nothing is reported.", kind)

    if key and _imagery_of_model(key) and _imagery_of_model(key) != kind:
        return _solar_refusal(
            image_bgr,
            f"{key} reads {_imagery_of_model(key).replace('_', ' ')} imagery, and this "
            f"input is {kind.replace('_', ' ')}. Its answer on the wrong imagery would look "
            "exactly like a real one, so it was not run.", kind, key)

    if kind in ("el_cell", "thermal_module") and not grey and thermal_gray is None:
        return _solar_refusal(
            image_bgr,
            f"Declared as {kind.replace('_', ' ')} imagery, but the image is in colour. EL "
            "cell images and the infrared module crops the thermal classifier was trained "
            "on are greyscale; a colour-palette thermal image or an RGB photo is not what "
            "either model reads.", kind)

    if kind == "el_cell" and max(h, w) < EL_CELL_MIN_SIDE_PX:
        return _solar_refusal(
            image_bgr,
            f"Declared as an EL cell image, but it is only {w}x{h} px. The EL detector was "
            "trained on 1024x1024 px cell images; an image this small is not one (single "
            "infrared module crops are 24x40 px -- declare those as thermal_module).",
            kind, key or SOLAR_MODELS_BY_IMAGERY["el_cell"][0])

    candidates = (key,) if key else SOLAR_MODELS_BY_IMAGERY[kind]
    chosen = next((c for c in candidates if model_status(c).get("exists")), "")
    if not chosen:
        if kind == "rgb_photo":
            reason = (
                "No solar model for RGB photographs is installed "
                f"({', '.join(candidates)} have no weights). The installed solar models read "
                "electroluminescence cell images (solar_cell_defect_detector) and single "
                "infrared module crops (solar_thermal_anomaly_classifier); declare that "
                "imagery to use them.")
        else:
            reason = (f"{', '.join(candidates)} is not installed on this machine, and there "
                      "is no model-free solar detector to use instead.")
        return _solar_refusal(image_bgr, reason, kind, candidates[0] if candidates else "")

    spec = get_model_spec(chosen)
    path = Path(str(model_status(chosen)["path"]))
    used = f"onnx:{path.name}"
    identity = model_identity(chosen)
    scope = _model_scope(chosen)
    probabilities: dict[str, float] = {}

    if spec is not None and spec.kind == "onnx_classification":
        crop = _thermal_to_bgr(thermal_gray) if thermal_gray is not None else image_bgr
        if max(crop.shape[:2]) > THERMAL_MODULE_MAX_SIDE_PX:
            return _solar_refusal(
                image_bgr,
                f"{chosen} classifies ONE module per image (trained on 24x40 px crops). "
                f"This image is {crop.shape[1]}x{crop.shape[0]} px, a frame holding many "
                "modules; the classifier cannot locate modules, so crop them first.",
                kind, chosen)
        from .model_runner import ModelRunRefused, classify_image

        try:
            probabilities = classify_image(crop, chosen)
        except ModelRunRefused as exc:
            return _solar_refusal(image_bgr, str(exc), kind, chosen)
        top = max(probabilities, key=probabilities.get)
        hits: list[DefectHit] = []
        mask_by_class: dict[str, np.ndarray] = {}
        if top != "No-Anomaly":
            # Module-level answer: the whole crop is the module the class is about.
            hits = [DefectHit(label=top, confidence=float(probabilities[top]),
                              bbox_xyxy=(0, 0, w, h), area_px=int(h * w))]
            mask_by_class = {top: np.full((h, w), 255, dtype=np.uint8)}
    elif spec is not None and spec.kind in DETECTION_KINDS:
        if _load_onnx_net(path) is None:
            return _solar_refusal(image_bgr, f"{chosen}: the graph at {path} would not load.",
                                  kind, chosen)
        try:
            hits = _run_onnx_yolo(
                image_bgr=image_bgr, model_path=path, labels=spec.labels,
                score_threshold=float(spec.score_threshold),
                iou_threshold=float(spec.iou_threshold), input_size=int(spec.input_size))
        except Exception as exc:  # a broken graph must not read as a clean panel
            return _solar_refusal(image_bgr, f"{chosen} failed to run: "
                                  f"{type(exc).__name__}: {exc}", kind, chosen)
        mask_by_class = _hits_to_masks((h, w), hits)
    else:
        return _solar_refusal(
            image_bgr, f"{chosen} has kind {getattr(spec, 'kind', '?')!r}, which the solar "
            "pass has no runner for.", kind, chosen)

    union = _union_mask(mask_by_class, (h, w))
    total = int(max(1, h * w))
    defect_pixels = int(np.sum(union > 0))
    return SolarDefectResult(
        mask=union,
        overlay=_overlay_by_masks(image_bgr, mask_by_class),
        mask_by_class=mask_by_class,
        detections=hits,
        defect_pixels=defect_pixels,
        total_pixels=total,
        defect_ratio=float(defect_pixels / total),
        class_pixel_ratio={label: float(np.sum(m > 0) / total)
                           for label, m in mask_by_class.items()},
        model_used=used,
        model_available=True,
        model_key=identity["model_key"],
        model_sha256=identity["model_sha256"],
        imagery=kind,
        scope=scope,
        class_probabilities=dict(probabilities),
    )
