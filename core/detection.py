"""Defect detection utilities (classical + optional model-backed inference)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except Exception:  # pragma: no cover - optional dependency
    _skimage_skeletonize = None

from .models import get_model_spec, model_status


_MODEL_NET_CACHE: dict[str, Any] = {}


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


@dataclass
class MetalDefectResult:
    """Result bundle for metal defect detection."""

    mask: np.ndarray
    overlay: np.ndarray
    defect_pixels: int
    total_pixels: int
    defect_ratio: float
    defect_regions: int


@dataclass
class DefectHit:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
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

    def to_summary(self) -> dict[str, Any]:
        return {
            "defect_pixels": int(self.defect_pixels),
            "total_pixels": int(self.total_pixels),
            "defect_ratio": float(self.defect_ratio),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "detections": [d.to_dict() for d in self.detections],
            "model_used": self.model_used,
            "model_available": bool(self.model_available),
        }


@dataclass
class SolarDefectResult:
    """Solar-specific defect result."""

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
    hotspot_max_temp_c: float | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "defect_pixels": int(self.defect_pixels),
            "total_pixels": int(self.total_pixels),
            "defect_ratio": float(self.defect_ratio),
            "class_pixel_ratio": {k: float(v) for k, v in self.class_pixel_ratio.items()},
            "detections": [d.to_dict() for d in self.detections],
            "model_used": self.model_used,
            "model_available": bool(self.model_available),
            "hotspot_max_temp_c": self.hotspot_max_temp_c,
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
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == idx] = 255
    return cleaned


def _mask_to_hits(mask: np.ndarray, label: str, min_area_px: int, confidence: float) -> list[DefectHit]:
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
                confidence=float(np.clip(confidence, 0.01, 0.99)),
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


def _score_from_area(mask: np.ndarray) -> float:
    ratio = float(np.mean(mask > 0))
    return float(np.clip(0.35 + ratio * 2.0, 0.35, 0.95))


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
) -> list[DefectHit]:
    net = _load_onnx_net(model_path)
    if net is None:
        return []

    size = int(max(64, input_size))
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0 / 255.0,
        size=(size, size),
        swapRB=True,
        crop=False,
    )
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

    h, w = image_bgr.shape[:2]
    sx = float(w) / float(size)
    sy = float(h) / float(size)
    label_count = max(1, len(labels))

    boxes_xywh: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for row in arr:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size < 6:
            continue

        if row.size >= 5 + label_count:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            obj = float(row[4])
            cls_scores = row[5 : 5 + label_count]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            conf = obj * cls_conf
        else:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            conf = cls_conf

        if conf < score_threshold:
            continue

        x1 = float((cx - bw * 0.5) * sx)
        y1 = float((cy - bh * 0.5) * sy)
        ww = float(bw * sx)
        hh = float(bh * sy)
        boxes_xywh.append([x1, y1, ww, hh])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    if not boxes_xywh:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, score_threshold, iou_threshold)
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


def detect_cracks(image_bgr: np.ndarray, min_area_px: int = 30) -> CrackDetectionResult:
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

    binary = (mask > 0).astype(np.uint8)
    skeleton = _skeletonize_binary(binary)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    crack_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    crack_ratio = float(crack_pixels / max(total_pixels, 1))
    estimated_length = float(skeleton.sum())
    max_width = float(np.nan_to_num(dist_map.max(), nan=0.0, posinf=0.0, neginf=0.0))
    estimated_max_width = float(np.clip(max_width * 2.0, 0.0, float(max(gray.shape) * 2.0)))

    overlay = image_bgr.copy()
    overlay[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0.0)

    return CrackDetectionResult(
        mask=(binary * 255).astype(np.uint8),
        overlay=overlay,
        crack_pixels=crack_pixels,
        total_pixels=total_pixels,
        crack_ratio=crack_ratio,
        estimated_length_px=estimated_length,
        estimated_max_width_px=estimated_max_width,
    )


def detect_metal_defects(image_bgr: np.ndarray, min_area_px: int = 40) -> MetalDefectResult:
    """Detect metal surface anomalies using texture discontinuity cues."""
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
        detections.extend(_mask_to_hits(crack.mask, "crack", min_area_px, confidence=_score_from_area(crack.mask)))

    metal = detect_metal_defects(image_bgr, min_area_px=max(40, min_area_px))
    if int(metal.defect_pixels) > 0:
        mask_by_class["corrosion"] = metal.mask
        detections.extend(_mask_to_hits(metal.mask, "corrosion", min_area_px, confidence=_score_from_area(metal.mask)))

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
        detections.extend(_mask_to_hits(mask, label, min_area_px, confidence=_score_from_area(mask)))

    return mask_by_class, detections


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
    model_used = "heuristic"
    model_available = bool(model_info.get("exists", False))

    if use_model and model_available:
        spec = get_model_spec(model_key)
        path_str = str(model_info.get("path", ""))
        if spec is not None and path_str and spec.kind == "onnx_yolo":
            try:
                model_hits = _run_onnx_yolo(
                    image_bgr=image_bgr,
                    model_path=Path(path_str),
                    labels=spec.labels,
                    score_threshold=float(spec.score_threshold),
                    iou_threshold=float(spec.iou_threshold),
                    input_size=int(spec.input_size),
                )
            except Exception:
                model_hits = []
            if model_hits:
                hits = model_hits
                mask_by_class = _hits_to_masks((h, w), hits)
                model_used = f"onnx:{Path(path_str).name}"

    if not hits or not mask_by_class:
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
    )


def _detect_soiling_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_value = (hsv[:, :, 2] < np.percentile(hsv[:, :, 2], 35)).astype(np.uint8) * 255
    low_sat = (hsv[:, :, 1] < np.percentile(hsv[:, :, 1], 45)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(low_value, low_sat)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return _filter_connected_components(mask, min_area_px=max(120, min_area_px))


def _detect_solar_crack_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    edges = cv2.Canny(enh, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return _filter_connected_components(edges, min_area_px=max(30, min_area_px // 2))


def _detect_hotspot_mask(
    image_bgr: np.ndarray,
    thermal_gray: np.ndarray | None,
    min_area_px: int,
) -> tuple[np.ndarray, float | None]:
    if thermal_gray is not None and thermal_gray.size > 0:
        t = thermal_gray.astype(np.float32)
        thr = float(max(np.percentile(t, 99.0), np.mean(t) + 2.0 * np.std(t)))
        raw = (t >= thr).astype(np.uint8) * 255
        out = _filter_connected_components(raw, min_area_px=max(20, min_area_px // 2))
        max_temp = float(np.max(t))
        return out, max_temp

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    warm1 = cv2.inRange(hsv, (0, 90, 170), (15, 255, 255))
    warm2 = cv2.inRange(hsv, (160, 90, 170), (179, 255, 255))
    warm = cv2.bitwise_or(warm1, warm2)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return _filter_connected_components(warm, min_area_px=max(40, min_area_px)), None


def _detect_solar_delamination_mask(image_bgr: np.ndarray, min_area_px: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    resid = cv2.absdiff(gray, blur)
    _, mask = cv2.threshold(resid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return _filter_connected_components(mask, min_area_px=max(80, min_area_px))


def _solar_fallback(
    image_bgr: np.ndarray,
    thermal_gray: np.ndarray | None,
    min_area_px: int,
) -> tuple[dict[str, np.ndarray], list[DefectHit], float | None]:
    mask_by_class: dict[str, np.ndarray] = {}
    detections: list[DefectHit] = []

    hotspot_mask, hotspot_max = _detect_hotspot_mask(image_bgr, thermal_gray=thermal_gray, min_area_px=min_area_px)
    if int(np.sum(hotspot_mask > 0)) > 0:
        mask_by_class["hotspot"] = hotspot_mask
        detections.extend(_mask_to_hits(hotspot_mask, "hotspot", min_area_px=max(20, min_area_px // 2), confidence=_score_from_area(hotspot_mask)))

    for label, mask in (
        ("cell_crack", _detect_solar_crack_mask(image_bgr, min_area_px=min_area_px)),
        ("soiling", _detect_soiling_mask(image_bgr, min_area_px=min_area_px)),
        ("delamination", _detect_solar_delamination_mask(image_bgr, min_area_px=min_area_px)),
    ):
        pixels = int(np.sum(mask > 0))
        if pixels <= 0:
            continue
        mask_by_class[label] = mask
        detections.extend(_mask_to_hits(mask, label, min_area_px=max(20, min_area_px // 2), confidence=_score_from_area(mask)))

    return mask_by_class, detections, hotspot_max


def detect_solar_defects(
    image_bgr: np.ndarray,
    thermal_gray: np.ndarray | None = None,
    min_area_px: int = 40,
    model_key: str = "solar_defect_detector",
    use_model: bool = True,
) -> SolarDefectResult:
    """
    Detect solar inspection defects (hotspots, soiling, cell cracks, delamination).

    If a model is available in models/model_registry.json and weight file exists,
    ONNX YOLO inference is used. Otherwise, a robust classical fallback is used.
    """
    h, w = image_bgr.shape[:2]
    total = int(max(1, h * w))
    model_info = model_status(model_key)

    mask_by_class: dict[str, np.ndarray] = {}
    hits: list[DefectHit] = []
    model_used = "heuristic"
    model_available = bool(model_info.get("exists", False))
    hotspot_max_temp_c: float | None = None

    if use_model and model_available:
        spec = get_model_spec(model_key)
        path_str = str(model_info.get("path", ""))
        if spec is not None and path_str and spec.kind == "onnx_yolo":
            try:
                model_hits = _run_onnx_yolo(
                    image_bgr=image_bgr,
                    model_path=Path(path_str),
                    labels=spec.labels,
                    score_threshold=float(spec.score_threshold),
                    iou_threshold=float(spec.iou_threshold),
                    input_size=int(spec.input_size),
                )
            except Exception:
                model_hits = []
            if model_hits:
                hits = model_hits
                mask_by_class = _hits_to_masks((h, w), hits)
                model_used = f"onnx:{Path(path_str).name}"

    if not hits or not mask_by_class:
        mask_by_class, hits, hotspot_max_temp_c = _solar_fallback(
            image_bgr=image_bgr,
            thermal_gray=thermal_gray,
            min_area_px=min_area_px,
        )
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

    return SolarDefectResult(
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
        hotspot_max_temp_c=hotspot_max_temp_c,
    )
