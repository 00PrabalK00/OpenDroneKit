"""Tree-crown boxes on an RGB orthomosaic, with the project's DSM height when it has one.

The model is DeepForest's prebuilt tree detector (``weecology/deepforest-tree``, a
torchvision RetinaNet-ResNet50-FPN, MIT weights trained on NEON airborne RGB at about
10 cm GSD), exported to one ONNX graph that takes a 400 x 400 RGB patch and returns
boxes, scores and labels after the network's own NMS.

What this produces, and what it does not:

* Boxes, one per detected crown, in the orthomosaic's projected CRS, with the model's
  score. A box is not a crown outline: every area figure here is the area of the
  axis-aligned box, and the summary says so in words.
* A count. It is the count of boxes after a global NMS across tiles, which is what
  DeepForest's own ``predict_tile`` does; it is not a census.
* Per-crown height ONLY when the project has its own DSM and DTM, computed as
  max(DSM) - median(DTM) within the box and labelled as a photogrammetric measurement.
  Without both rasters the field is absent. Nothing here estimates a height.

It refuses, with the reason, when the raster has no projected metric CRS, is not 8-bit
RGB, when the model is missing, or when the installed file is not the file the
registry recorded. There is no heuristic fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .model_runner import ModelRunRefused

MODEL_KEY = "tree_crown_deepforest"
KIND = "onnx_tree_crowns"
PATCH_SIZE = 400            # DeepForest predict_tile default patch_size
PATCH_OVERLAP = 0.05        # DeepForest predict_tile default patch_overlap
TILE_NMS_IOU = 0.15         # DeepForest predict_tile default iou_threshold across windows
GSD_TOLERANCE = 0.25        # resample when the ortho GSD is >25% away from the target
BOX_AREA_LABEL = "box-derived, not crown outline"
HEIGHT_LABEL = "photogrammetric height (measured from DSM)"


class TreeCrownsRefused(ModelRunRefused):
    """The tree-crown run cannot answer for this raster or this installation."""


# --------------------------------------------------------------------------- model


class TreeCrownModel:
    """The verified ONNX session. Construction refuses rather than loading a wrong file."""

    def __init__(self, model_key: str = MODEL_KEY, *, allow_cpu: bool = False) -> None:
        from .models import get_model_spec, verify_model_identity

        spec = get_model_spec(model_key)
        if spec is None:
            raise TreeCrownsRefused(f"{model_key!r} is not in the model registry.")
        if spec.kind != KIND:
            raise TreeCrownsRefused(f"{model_key} is a {spec.kind}, not a tree-crown model.")
        check = verify_model_identity(model_key)
        if check["status"] in ("missing", "awaiting_weights"):
            note = spec.status_note or "Its weights are not installed on this machine."
            raise TreeCrownsRefused(f"{model_key} cannot run: {note}")
        if check["status"] != "verified":
            # "unrecorded" is refused too: a file nobody can tie to the registered
            # export is not the model whose card and licence the user is reading.
            raise TreeCrownsRefused(f"{model_key} refused: {check['detail']}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise TreeCrownsRefused("onnxruntime is required for tree-crown detection.") from exc

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif not allow_cpu:
            raise TreeCrownsRefused(
                "Tree-crown detection on the CPU is disabled by default because a survey "
                "orthomosaic takes long enough to look hung; use a GPU or allow the CPU.")
        session = ort.InferenceSession(str(check["model_path"]), providers=providers)
        device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        if device == "cpu" and not allow_cpu:
            # CUDA was listed but did not load (missing cuDNN, say): ONNX Runtime falls
            # back to the CPU quietly, and the guard must see the device that loaded.
            raise TreeCrownsRefused(
                "CUDA was listed but did not load, so this run would be on the CPU; "
                "allow the CPU explicitly to continue.")
        self.model_key = model_key
        self.spec = spec
        self.model_sha256 = str(check["actual_sha256"])
        self.model_path = str(check["model_path"])
        self.device = device
        self._session = session
        self._input = session.get_inputs()[0].name

    def predict(self, patch_rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (N, 4) xyxy in patch pixels and scores (N,) for one 400 x 400 RGB patch."""
        x = np.ascontiguousarray(patch_rgb_u8.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        boxes, scores = self._session.run(None, {self._input: x})[:2]
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(scores, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- geometry


def tile_starts(length: int, tile: int, overlap_px: int) -> list[int]:
    """Window origins covering ``length`` with ``tile``-sized windows, last one flush."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap_px)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression, highest score first."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        union = areas[i] + areas[rest] - inter
        overlap = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[overlap <= iou]
    return np.asarray(keep, dtype=int)


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    t = np.zeros(len(p)) if denom == 0 else np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def polygon_line_distance(ring_xy: Sequence[Sequence[float]], line_xy: Sequence[Sequence[float]]) -> float:
    """Plan distance in CRS units between a closed ring and a polyline; 0 if they touch."""
    ring = np.asarray(ring_xy, dtype=float)
    line = np.asarray(line_xy, dtype=float)
    if len(line) == 1:
        line = np.vstack([line, line])
    # A line vertex inside the ring is distance 0.
    for vx, vy in line:
        inside = False
        for (ax, ay), (bx, by) in zip(ring[:-1], ring[1:]):
            if (ay > vy) != (by > vy) and vx < (bx - ax) * (vy - ay) / ((by - ay) or 1e-12) + ax:
                inside = not inside
        if inside:
            return 0.0
    best = math.inf
    for a, b in zip(line[:-1], line[1:]):
        best = min(best, float(_segment_distance(ring, a, b).min()))
        for c, d in zip(ring[:-1], ring[1:]):
            if _segments_intersect(a, b, c, d):
                return 0.0
            best = min(best, float(_segment_distance(np.asarray([a, b]), c, d).min()))
    return best


def detect_in_image(image_rgb_u8: np.ndarray, model: Any) -> dict[str, Any]:
    """Crown boxes on one un-georeferenced RGB image, tiled exactly like the orthomosaic.

    No resampling happens here: a photo carries no GSD, so the boxes are only as good as
    the image's scale is close to the ~10 cm the model was trained at, and the result
    says so. Boxes are in image pixels; nothing here is in metres.
    """
    image = np.asarray(image_rgb_u8)
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise TreeCrownsRefused("Tree-crown detection needs an 8-bit RGB image.")
    height, width = image.shape[:2]
    overlap_px = int(round(PATCH_SIZE * PATCH_OVERLAP))
    all_boxes, all_scores = [], []
    for top in tile_starts(height, PATCH_SIZE, overlap_px):
        for left in tile_starts(width, PATCH_SIZE, overlap_px):
            crop = image[top:top + PATCH_SIZE, left:left + PATCH_SIZE, :3]
            th, tw = crop.shape[:2]
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            patch[:th, :tw] = crop
            boxes, scores = model.predict(patch)
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw) + left
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th) + top
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    keep = nms(boxes, scores, TILE_NMS_IOU)
    detections = [{
        "label": "Tree",
        "confidence": round(float(scores[i]), 4),
        "bbox_xyxy": [int(round(float(v))) for v in boxes[i]],
    } for i in keep]
    return {
        "detections": detections,
        "detection_count": len(detections),
        "confidence_kind": "detector_score",
        "note": ("Pixel boxes on an image with no recorded GSD, so it was not resampled to "
                 "the model's ~10 cm; boxes are not crown outlines and carry no size in "
                 "metres. Use the orthomosaic run for georeferenced crowns."),
    }


# --------------------------------------------------------------------------- raster


def _require_metric_projected(src: Any, path: str | Path) -> int:
    from .dsm_analysis import NotGeoreferenced, require_projected_epsg

    if src.crs is None:
        raise TreeCrownsRefused(
            f"{path} has no coordinate reference system, so crown boxes would be unlocated "
            "and their areas would not be in square metres.")
    if not src.crs.is_projected:
        raise TreeCrownsRefused(
            f"{path} is in a geographic CRS ({src.crs.to_string()}): its pixels are degrees, "
            "so box sizes and the ~10 cm resampling the model needs cannot be computed. "
            "Reproject it to the survey's UTM zone first.")
    epsg = src.crs.to_epsg()
    try:
        require_projected_epsg(epsg, path)
    except NotGeoreferenced as exc:
        raise TreeCrownsRefused(str(exc)) from exc
    return int(epsg)


def _box_values(src: Any, box_xy: tuple[float, float, float, float]) -> np.ndarray:
    """Finite cell values of one elevation raster inside a CRS box (empty if none)."""
    from rasterio.windows import from_bounds

    x1, y1, x2, y2 = box_xy
    window = from_bounds(x1, y1, x2, y2, transform=src.transform)
    window = window.round_offsets().round_lengths()
    col0, row0 = max(0, int(window.col_off)), max(0, int(window.row_off))
    col1 = min(src.width, int(window.col_off + max(1, window.width)))
    row1 = min(src.height, int(window.row_off + max(1, window.height)))
    if col1 <= col0 or row1 <= row0:
        return np.zeros(0)
    from rasterio.windows import Window

    data = src.read(1, window=Window(col0, row0, col1 - col0, row1 - row0), masked=True)
    values = np.asarray(data.filled(np.nan), dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values > -1e30)]
    return values


def crown_heights(boxes_crs: np.ndarray, epsg: int, dsm_path: str | Path,
                  dtm_path: str | Path) -> tuple[list[float | None], dict[str, Any]]:
    """max(DSM) - median(DTM) inside each box, or None where either raster has no data."""
    import rasterio

    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        for raster, name in ((dsm, dsm_path), (dtm, dtm_path)):
            if raster.crs is None or raster.crs.to_epsg() != epsg:
                raise TreeCrownsRefused(
                    f"{name} is not in EPSG:{epsg} like the orthomosaic, so its cells "
                    "cannot be matched to crown boxes without resampling.")
        heights: list[float | None] = []
        for box in boxes_crs:
            top = _box_values(dsm, tuple(box))
            ground = _box_values(dtm, tuple(box))
            if top.size == 0 or ground.size == 0:
                heights.append(None)
                continue
            heights.append(round(float(top.max() - np.median(ground)), 3))
    return heights, {
        "computed": True,
        "label": HEIGHT_LABEL,
        "method": "max(DSM) - median(DTM) within the crown box",
        "dsm_path": str(dsm_path),
        "dtm_path": str(dtm_path),
    }


# --------------------------------------------------------------------------- main


def detect_tree_crowns(
    ortho_path: str | Path,
    out_dir: str | Path,
    allow_cpu: bool = False,
    target_gsd_m: float = 0.10,
    *,
    model_key: str = MODEL_KEY,
    dsm_path: str | Path | None = None,
    dtm_path: str | Path | None = None,
    model: Any = None,
    progress: Callable[[int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect crowns over a georeferenced RGB orthomosaic and write GeoJSON + summary.

    ``model`` may be any object with ``predict(patch_rgb_u8) -> (boxes, scores)`` and
    the attributes ``model_key``, ``model_sha256``, ``device``; it exists so the tiling
    geometry can be tested without the weights. In normal use it is left empty and the
    registered, digest-verified ONNX model is loaded.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    from . import geo

    source = Path(ortho_path)
    if not source.is_file():
        raise TreeCrownsRefused(f"Orthomosaic not found: {source}")
    out = Path(out_dir)
    report = progress or (lambda pct, msg: None)

    with rasterio.open(source) as src:
        epsg = _require_metric_projected(src, source)
        if src.count < 3:
            raise TreeCrownsRefused(
                f"{source} has {src.count} band(s); the tree-crown model needs RGB.")
        if src.dtypes[0] != "uint8":
            raise TreeCrownsRefused(
                f"{source} is {src.dtypes[0]}, not 8-bit RGB. The model was trained on "
                "8-bit RGB; rescaling another range would be a guess.")
        if model is None:
            model = TreeCrownModel(model_key, allow_cpu=allow_cpu)

        transform = src.transform
        res_x, res_y = abs(float(transform.a)), abs(float(transform.e))
        source_gsd = (res_x + res_y) / 2.0
        target = float(target_gsd_m)
        resampled = abs(source_gsd - target) / target > GSD_TOLERANCE
        scale = source_gsd / target if resampled else 1.0  # target px per source px
        width_t = max(1, int(round(src.width * scale)))
        height_t = max(1, int(round(src.height * scale)))
        overlap_px = int(round(PATCH_SIZE * PATCH_OVERLAP))
        xs = tile_starts(width_t, PATCH_SIZE, overlap_px)
        ys = tile_starts(height_t, PATCH_SIZE, overlap_px)
        has_mask = any(flag.name in ("per_dataset", "alpha", "nodata")
                       for flag in src.mask_flag_enums[0]) if src.mask_flag_enums else False

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        skipped_empty = 0
        total = len(xs) * len(ys)
        done = 0
        for top in ys:
            for left in xs:
                if should_cancel and should_cancel():
                    raise TreeCrownsRefused("Tree-crown detection was cancelled.")
                # The patch on the target grid, and the source window it covers.
                tw = min(PATCH_SIZE, width_t - left)
                th = min(PATCH_SIZE, height_t - top)
                window = Window(left / scale, top / scale, tw / scale, th / scale)
                rgb = src.read([1, 2, 3], window=window, out_shape=(3, th, tw),
                               resampling=Resampling.bilinear if resampled else Resampling.nearest,
                               boundless=True, fill_value=0)
                if has_mask:
                    valid = src.read_masks(1, window=window, out_shape=(th, tw),
                                           resampling=Resampling.nearest, boundless=True)
                    if not valid.any():
                        skipped_empty += 1
                        done += 1
                        continue
                elif not rgb.any():
                    skipped_empty += 1
                    done += 1
                    continue
                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                patch[:th, :tw] = np.transpose(rgb, (1, 2, 0))
                boxes, scores = model.predict(patch)
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th)
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    boxes, scores = boxes[keep], scores[keep]
                    boxes[:, [0, 2]] += left
                    boxes[:, [1, 3]] += top
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                done += 1
                report(5 + int(85 * done / max(1, total)), f"Tile {done}/{total}")

    boxes_t = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    keep = nms(boxes_t, scores, TILE_NMS_IOU)
    boxes_t, scores = boxes_t[keep], scores[keep]
    # Target-grid pixels -> source pixels -> CRS. A north-up affine maps an axis-aligned
    # pixel box to an axis-aligned CRS box; corners are mapped individually so a rotated
    # affine still lands correctly.
    boxes_src = boxes_t / scale
    features_crs, features_wgs, crs_boxes = [], [], []
    for i, (bx, score) in enumerate(zip(boxes_src, scores)):
        x1, y1, x2, y2 = (float(v) for v in bx)
        corners = [transform * (cx, cy) for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        ring = [[float(x), float(y)] for x, y in corners] + [[float(corners[0][0]), float(corners[0][1])]]
        xs_c = [p[0] for p in ring]
        ys_c = [p[1] for p in ring]
        crs_boxes.append((min(xs_c), min(ys_c), max(xs_c), max(ys_c)))
        width_m = math.dist(corners[0], corners[1])
        height_m = math.dist(corners[1], corners[2])
        props = {
            "crown_id": i + 1,
            "label": "Tree",
            "score": round(float(score), 4),
            "box_width_m": round(width_m, 3),
            "box_height_m": round(height_m, 3),
            "box_area_m2": round(width_m * height_m, 3),
            "area_basis": BOX_AREA_LABEL,
            "bbox_px_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "model_key": model.model_key,
            "model_sha256": model.model_sha256,
            "output_semantics": "detection",
        }
        features_crs.append({"type": "Feature", "properties": props,
                             "geometry": {"type": "Polygon", "coordinates": [ring]}})

    height_info: dict[str, Any] = {
        "computed": False,
        "reason": "No project DSM and DTM were given; height is omitted, never estimated.",
    }
    if dsm_path and dtm_path and Path(dsm_path).is_file() and Path(dtm_path).is_file():
        heights, height_info = crown_heights(np.asarray(crs_boxes), epsg, dsm_path, dtm_path)
        for feature, value in zip(features_crs, heights):
            if value is not None:
                feature["properties"]["height_m"] = value
                feature["properties"]["height_source"] = HEIGHT_LABEL
        height_info["crowns_with_height"] = sum(v is not None for v in heights)
    elif dsm_path or dtm_path:
        height_info["reason"] = ("Only one of DSM and DTM is available; height needs both and "
                                 "is omitted, never estimated.")

    if features_crs:
        flat = np.asarray([p for f in features_crs for p in f["geometry"]["coordinates"][0]])
        lon, lat = geo.projected_to_wgs84(flat[:, 0], flat[:, 1], epsg)
        k = 0
        for feature in features_crs:
            n = len(feature["geometry"]["coordinates"][0])
            ring = [[float(lon[k + j]), float(lat[k + j])] for j in range(n)]
            k += n
            features_wgs.append({"type": "Feature", "properties": dict(feature["properties"]),
                                 "geometry": {"type": "Polygon", "coordinates": [ring]}})

    areas = np.asarray([f["properties"]["box_area_m2"] for f in features_crs], dtype=float)
    area_stats = {"label": BOX_AREA_LABEL}
    if areas.size:
        area_stats.update({
            "min_m2": round(float(areas.min()), 3),
            "median_m2": round(float(np.median(areas)), 3),
            "mean_m2": round(float(areas.mean()), 3),
            "max_m2": round(float(areas.max()), 3),
            "total_m2": round(float(areas.sum()), 3),
        })
    from .models import load_registry

    row = (load_registry().get("models") or {}).get(model.model_key) or {}
    collection_props = {"model_key": model.model_key, "model_sha256": model.model_sha256,
                        "area_basis": BOX_AREA_LABEL}
    out.mkdir(parents=True, exist_ok=True)
    crowns_path = geo.write_geojson(out / "tree_crowns.geojson", features_crs, epsg=epsg,
                                    properties=collection_props)
    wgs_path = geo.write_geojson(out / "tree_crowns_wgs84.geojson", features_wgs,
                                 properties=collection_props)
    summary = {
        "model_key": model.model_key,
        "model_used": f"onnx:{Path(getattr(model, 'model_path', '') or 'model').name}",
        "model_sha256": model.model_sha256,
        "device": model.device,
        "output_semantics": "detection",
        "orthomosaic": str(source),
        "epsg": epsg,
        "crown_count": len(features_crs),
        "count_note": ("Boxes after global NMS across tiles. Not a census: overlapping "
                       "crowns merge and small understorey trees are missed."),
        "box_area_stats": area_stats,
        "resampling": {
            "source_gsd_m": round(source_gsd, 4),
            "target_gsd_m": target,
            "resampled": bool(resampled),
            "scale": round(scale, 6),
            "note": ("Resampled to the model's ~10 cm training GSD (bilinear)."
                     if resampled else "Source GSD within 25% of 10 cm; not resampled."),
        },
        "tiling": {"patch_px": PATCH_SIZE, "overlap_px": overlap_px, "tiles": total,
                   "tiles_skipped_nodata": skipped_empty, "global_nms_iou": TILE_NMS_IOU},
        "height": height_info,
        "metric": row.get("metric", "unmeasured"),
        "scope": str(row.get("description") or ""),
        "crowns_path": crowns_path,
        "crowns_wgs84_path": wgs_path,
    }
    summary_path = out / "tree_crowns_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    report(100, f"{len(features_crs)} crown box(es)")
    return summary


def trees_near_line(crowns_path: str | Path, line_lonlat: Sequence[Sequence[float]],
                    buffer_m: float) -> dict[str, Any]:
    """Crowns whose box lies within ``buffer_m`` (plan distance) of a polyline."""
    from . import geo

    payload = json.loads(Path(crowns_path).read_text(encoding="utf-8"))
    name = ((payload.get("crs") or {}).get("properties") or {}).get("name", "")
    if "EPSG::" not in name:
        raise TreeCrownsRefused(f"{crowns_path} does not name a projected CRS.")
    epsg = int(name.rsplit("::", 1)[1])
    points = [[float(p[0]), float(p[1])] for p in line_lonlat or [] if len(p) >= 2]
    if not points:
        raise TreeCrownsRefused("No line was given.")
    if not (float(buffer_m) >= 0) or not math.isfinite(float(buffer_m)):
        raise TreeCrownsRefused("The buffer must be a finite distance in metres, 0 or more.")
    east, north = geo.wgs84_to_projected([p[0] for p in points], [p[1] for p in points], epsg)
    line_xy = np.column_stack([east, north])
    hits = []
    for feature in payload.get("features", []):
        ring = feature["geometry"]["coordinates"][0]
        distance = polygon_line_distance(ring, line_xy)
        if distance <= float(buffer_m):
            props = dict(feature.get("properties") or {})
            props["plan_distance_m"] = round(distance, 3)
            hits.append(props)
    hits.sort(key=lambda p: p["plan_distance_m"])
    return {
        "crowns": hits,
        "count": len(hits),
        "buffer_m": float(buffer_m),
        "epsg": epsg,
        "note": "plan distance only; 3-D clearance not computed",
    }
