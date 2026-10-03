"""Run any installed registry model on an image, by the kind the registry declares.

Six models were installed, verified against their digests, and unreachable: nothing in
the application could run crack_presence_classifier, the EL solar detector, the thermal
module classifier, either rail model or the land-cover model. A model on disk that no
path reaches is a claim about the product rather than a part of it.

Every result here carries what produced it -- model key, the digest of the file that
ran, the registry's own scope note -- because several of these models are only valid on
one kind of imagery (EL cells, thermal module crops, rail corridors) and the result is
the place a user reads that, not the registry file.

Nothing here falls back to a heuristic. A model that is not installed, or whose kind has
no runner, is refused with the reason; there is no classical substitute for "which of
twelve thermal anomaly classes is this module" and inventing one would be worse than
saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detection import (
    DETECTION_KINDS,
    CorrosionGradingRefused,
    _hits_to_masks,
    _load_onnx_net,
    _overlay_by_masks,
    _run_onnx_segmentation,
    _run_onnx_yolo,
    grade_corrosion_severity,
    non_defect_labels,
)
from .models import get_model_spec, load_registry, model_identity, model_status


class ModelRunRefused(ValueError):
    """The model cannot answer: not registered, not installed, or no runner for its kind."""


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass
class ModelRunResult:
    model_key: str
    kind: str
    model_used: str
    model_sha256: str
    # The registry's own description of what the model is for and what it cannot do.
    scope: str
    output: dict[str, Any] = field(default_factory=dict)
    overlay: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "kind": self.kind,
            "model_used": self.model_used,
            "model_sha256": self.model_sha256,
            "scope": self.scope,
            "output": self.output,
        }


def _registry_row(model_key: str) -> dict[str, Any]:
    row = (load_registry().get("models") or {}).get(model_key)
    return dict(row) if isinstance(row, dict) else {}


def _installed(model_key: str) -> tuple[Any, Path, dict[str, Any]]:
    spec = get_model_spec(model_key)
    if spec is None:
        raise ModelRunRefused(f"{model_key!r} is not in the model registry.")
    info = model_status(model_key)
    if not info.get("exists"):
        note = spec.status_note or "Its weights are not installed on this machine."
        raise ModelRunRefused(f"{model_key} cannot run: {note}")
    return spec, Path(str(info["path"])), _registry_row(model_key)


def _identity(model_key: str) -> str:
    return model_identity(model_key)["model_sha256"]


def classify_image(image_bgr: np.ndarray, model_key: str) -> dict[str, float]:
    """Class probabilities from a registry classifier, for the whole image as given.

    Two export conventions exist in the registry and they are not interchangeable: the
    crack classifier folds ImageNet normalisation and the softmax into its graph, the
    thermal classifier does neither. The registry row's `preprocessing` says which; a
    classifier that does not say is refused rather than guessed at, because the wrong
    convention still returns twelve numbers that sum to one.
    """
    spec, path, row = _installed(model_key)
    if spec.kind != "onnx_classification":
        raise ModelRunRefused(f"{model_key} is a {spec.kind}, not a classifier.")
    prep = row.get("preprocessing") or {}
    if not isinstance(prep, dict) or "normalisation" not in prep or "output" not in prep:
        raise ModelRunRefused(
            f"{model_key} does not record its preprocessing in the registry, so its "
            "input cannot be prepared the way it was trained.")
    classes = list(row.get("output_classes") or spec.labels)
    net = _load_onnx_net(path)
    if net is None:
        raise ModelRunRefused(f"{model_key}: the graph at {path} would not load.")

    size = int(spec.input_size or 224)
    blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=1.0 / 255.0, size=(size, size),
                                 swapRB=True, crop=False)
    if prep["normalisation"] == "imagenet":
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
    net.setInput(blob.astype(np.float32))
    scores = np.asarray(net.forward(), dtype=np.float64).reshape(-1)
    if scores.size != len(classes):
        raise ModelRunRefused(
            f"{model_key} emits {scores.size} scores for {len(classes)} registered classes; "
            "the names cannot be attached safely.")
    if prep["output"] == "logits":
        scores = np.exp(scores - scores.max())
        scores = scores / scores.sum()
    return {name: float(scores[i]) for i, name in enumerate(classes)}


@dataclass
class CrackPresenceResult:
    """Which tiles of an image the crack classifier says contain a crack."""

    tile_size: int
    tiles: int
    flagged_tiles: list[dict[str, Any]]
    max_probability: float
    threshold: float
    model_key: str
    model_sha256: str
    model_used: str

    @property
    def any_crack(self) -> bool:
        return bool(self.flagged_tiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "tiles": self.tiles,
            "flagged_tile_count": len(self.flagged_tiles),
            "flagged_tiles": self.flagged_tiles,
            "max_probability": self.max_probability,
            "threshold": self.threshold,
            "any_crack": self.any_crack,
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
            "model_used": self.model_used,
        }


def crack_presence(
    image_bgr: np.ndarray,
    *,
    model_key: str = "crack_presence_classifier",
    threshold: float = 0.5,
    max_tiles: int = 64,
) -> CrackPresenceResult:
    """Triage an image tile by tile: is there a crack in this tile?

    The classifier was trained on close-range tiles, so a whole photograph is split
    into roughly square tiles about the model's own input size (224 px), capped at
    `max_tiles` by growing the tile rather than skipping any of the image. This says
    which tiles to look at; it says nothing about where in a tile the crack is, how long
    or how wide -- crack_segmentation is the model for that.
    """
    spec, path, row = _installed(model_key)
    positive = str(row.get("positive_class") or "")
    if not positive:
        raise ModelRunRefused(f"{model_key} does not name its positive class in the registry.")
    h, w = image_bgr.shape[:2]
    tile = int(spec.input_size or 224)
    while (int(np.ceil(h / tile)) * int(np.ceil(w / tile))) > max_tiles:
        tile = int(tile * 1.5)
    flagged: list[dict[str, Any]] = []
    best = 0.0
    count = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            crop = image_bgr[y:min(h, y + tile), x:min(w, x + tile)]
            # A sliver at the edge carries too little surface to classify.
            if crop.shape[0] < tile // 4 or crop.shape[1] < tile // 4:
                continue
            count += 1
            probability = classify_image(crop, model_key)[positive]
            best = max(best, probability)
            if probability >= threshold:
                flagged.append({"bbox_xyxy": [x, y, min(w, x + tile), min(h, y + tile)],
                                "probability": round(float(probability), 4)})
    return CrackPresenceResult(
        tile_size=tile, tiles=count, flagged_tiles=flagged, max_probability=float(best),
        threshold=float(threshold), model_key=model_key,
        model_sha256=_identity(model_key), model_used=f"onnx:{path.name}",
    )


def run_registered_model(image_bgr: np.ndarray, model_key: str, *,
                         score_threshold: float | None = None) -> ModelRunResult:
    """Run one installed model on one image and return what it said, with provenance.

    ``score_threshold`` overrides the registry's detector threshold (the pre-label path
    passes the registry's measured pre-label threshold). Boxes for labels the registry
    marks as non-defects (CODEBRIM "Background") are returned apart from the detections.
    """
    declared = get_model_spec(model_key)
    if declared is not None and declared.kind == "onnx_prompt_segmentation":
        # A promptable segmenter answers a click or a box, not an image; running it with
        # no prompt would mean inventing one.
        raise ModelRunRefused(
            f"{model_key} is an interactive segmentation model and needs a prompt (a click "
            "or a box); use Segment (Api.segment_prompt).")
    spec, path, row = _installed(model_key)
    kind = spec.kind
    scope = str(row.get("description") or spec.description or "")
    used = f"onnx:{path.name}"
    h, w = image_bgr.shape[:2]

    if kind in DETECTION_KINDS:
        hits = _run_onnx_yolo(
            image_bgr, path, spec.labels,
            float(spec.score_threshold if score_threshold is None else score_threshold),
            float(spec.iou_threshold), int(spec.input_size))
        not_defects = non_defect_labels(model_key)
        non_defect_hits = [hit for hit in hits if hit.label in not_defects]
        hits = [hit for hit in hits if hit.label not in not_defects]
        overlay = _overlay_by_masks(image_bgr, _hits_to_masks((h, w), hits))
        for hit in hits:
            x1, y1, x2, y2 = hit.bbox_xyxy
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return ModelRunResult(model_key, kind, used, _identity(model_key), scope, {
            "detections": [hit.to_dict() for hit in hits],
            "detection_count": len(hits),
            "non_defect_detections": [hit.to_dict() for hit in non_defect_hits],
            "confidence_kind": "detector_score",
        }, overlay)

    if kind == "onnx_segmentation":
        mask = _run_onnx_segmentation(
            image_bgr, path, float(spec.score_threshold or 0.5), int(spec.input_size or 512))
        if mask is None:
            raise ModelRunRefused(f"{model_key} did not produce a usable mask.")
        label = spec.labels[0] if spec.labels else "foreground"
        overlay = _overlay_by_masks(image_bgr, {label: (mask > 0).astype(np.uint8) * 255})
        return ModelRunResult(model_key, kind, used, _identity(model_key), scope, {
            "label": label,
            "pixel_ratio": float(mask.mean()),
            "pixels": int(mask.sum()),
            "threshold": float(spec.score_threshold or 0.5),
        }, overlay)

    if kind == "onnx_segmentation_multiclass":
        try:
            graded = grade_corrosion_severity(image_bgr, model_key=model_key)
        except CorrosionGradingRefused as exc:
            raise ModelRunRefused(str(exc)) from exc
        return ModelRunResult(model_key, kind, graded.model_used, graded.model_sha256,
                              scope, graded.to_summary(), graded.overlay)

    if kind == "onnx_classification":
        if row.get("positive_class"):
            presence = crack_presence(image_bgr, model_key=model_key)
            overlay = image_bgr.copy()
            for tile in presence.flagged_tiles:
                x1, y1, x2, y2 = tile["bbox_xyxy"]
                cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
            return ModelRunResult(model_key, kind, used, presence.model_sha256, scope,
                                  presence.to_dict(), overlay)
        probabilities = classify_image(image_bgr, model_key)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        return ModelRunResult(model_key, kind, used, _identity(model_key), scope, {
            "top_class": ranked[0][0],
            "top_probability": float(ranked[0][1]),
            "probabilities": probabilities,
            "note": "Whole-image classification: the image must already be one "
                    "object of the kind the model was trained on.",
        }, None)

    raise ModelRunRefused(
        f"{model_key} is a {kind} model, which runs over a georeferenced orthomosaic "
        "rather than a single photograph; use the land-cover run for it."
        if kind == "onnx_shared_semantic" else
        f"{model_key} has kind {kind!r}, which has no runner in this build.")


def detector_backends() -> list[dict[str, Any]]:
    """For each detector the application runs, whether a model or a heuristic answers.

    Computed from the registry and the files on disk, never written down: the point is
    to show a user which detectors answer from a model, which fall back to a labelled
    heuristic, and which refuse (solar and metal) when their model is absent.
    """
    def row(detector: str, key: str, *, heuristic_fallback: bool, note: str = "") -> dict[str, Any]:
        info = model_status(key) if key else {"exists": False}
        spec = get_model_spec(key) if key else None
        installed = bool(info.get("exists"))
        if installed:
            backend = "model"
        elif heuristic_fallback:
            backend = "heuristic"
        else:
            backend = "unavailable"
        return {
            "detector": detector,
            "model_key": key,
            "installed": installed,
            "backend": backend,
            "status_note": (spec.status_note if spec is not None else "") or note,
        }

    def solar_row() -> dict[str, Any]:
        # Solar answers from whichever model reads the declared imagery; there is no
        # heuristic behind it any more, so "unavailable" is the honest state without one.
        from .detection import SOLAR_MODELS_BY_IMAGERY

        installed = {imagery: [k for k in keys if model_status(k).get("exists")]
                     for imagery, keys in SOLAR_MODELS_BY_IMAGERY.items()}
        readable = [imagery for imagery, keys in installed.items() if keys]
        keys = [k for ks in installed.values() for k in ks]
        return {
            "detector": "solar",
            "model_key": ", ".join(keys),
            "installed": bool(keys),
            "backend": "model" if keys else "unavailable",
            "status_note": (
                ("Reads " + ", ".join(readable) + " imagery"
                 + ("" if "rgb_photo" in readable else
                    "; RGB photographs have no installed model and are refused")
                 + ". Declare the imagery in Run AI Settings.")
                if keys else "No solar model is installed; solar detection refuses."),
        }

    return [
        row("crack", "crack_segmentation", heuristic_fallback=True),
        row("crack_presence", "crack_presence_classifier", heuristic_fallback=False),
        row("structural", "structural_multiclass_detector", heuristic_fallback=True),
        solar_row(),
        row("metal", "corrosion_severity_segmentation", heuristic_fallback=False,
            note="Metal defects are read by the corrosion model; refuses when it is absent."),
        row("corrosion_severity", "corrosion_severity_segmentation", heuristic_fallback=False,
            note="Refuses rather than guessing a grade."),
        row("solar_cell_el", "solar_cell_defect_detector", heuristic_fallback=False),
        row("solar_thermal", "solar_thermal_anomaly_classifier", heuristic_fallback=False),
        row("rail_corridor", "rail_corridor_segmentation", heuristic_fallback=False),
        row("rail_obstacle", "rail_obstacle_detector", heuristic_fallback=False),
        row("landcover", "shared_semantic_landcover", heuristic_fallback=False),
    ]


def save_overlay(result: ModelRunResult, directory: str | Path, stem: str) -> str:
    """Write the overlay next to a JSON of the result; returns the overlay path or ''."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if result.overlay is None:
        return ""
    target = out / f"{stem}_overlay.png"
    cv2.imwrite(str(target), result.overlay)
    return str(target)


def run_landcover(
    orthomosaic_path: str | Path,
    output_dir: str | Path,
    *,
    model_key: str = "shared_semantic_landcover",
    device: str = "cuda",
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Land-cover segmentation of a georeferenced orthomosaic with the shared model.

    The model's registry row says only two of its six classes hold up (vegetation and
    building); `usable_labels` is returned with the package so every consumer can see
    which polygons to trust. The manifest that names the classes and the checkpoint
    digest travels beside the ONNX file and is required: a class map with no schema is
    six anonymous integers.
    """
    from .semantic_engine import (
        ONNXSemanticPredictor,
        SemanticInferenceConfig,
        load_semantic_manifest,
        run_semantic_inference,
    )

    spec, path, row = _installed(model_key)
    if spec.kind != "onnx_shared_semantic":
        raise ModelRunRefused(f"{model_key} is a {spec.kind}, not a land-cover model.")
    manifest = path.with_name(path.name[: -len(".onnx")] + ".manifest.json")
    if not manifest.is_file():
        raise ModelRunRefused(
            f"{model_key}: the manifest {manifest.name} is missing beside the weights, so "
            "the class map could not be named.")
    schema, model, defaults = load_semantic_manifest(manifest)
    predictor = ONNXSemanticPredictor(
        path, device=device,
        mean=defaults.get("mean", (0.485, 0.456, 0.406)),
        std=defaults.get("std", (0.229, 0.224, 0.225)),
    )
    package = run_semantic_inference(
        orthomosaic_path, output_dir, schema=schema, model=model, predictor=predictor,
        config=SemanticInferenceConfig(
            tile_size=int(defaults.get("tile_size", spec.input_size or 518)),
            overlap=int(defaults.get("overlap", 126)),
            device=predictor.device,
            allow_cpu=bool(allow_cpu),
        ),
    )
    usable = [str(v) for v in (row.get("usable_labels") or [])]
    counts: dict[str, int] = {}
    try:
        features = json.loads(Path(package.polygons_path).read_text(encoding="utf-8")).get("features", [])
    except (OSError, ValueError):
        features = []
    for feature in features:
        props = feature.get("properties") or {}
        name = str(props.get("class_name") or props.get("label") or "")
        counts[name] = counts.get(name, 0) + 1
    return {
        "model_key": model_key,
        "model_used": f"onnx:{path.name}",
        "model_sha256": _identity(model_key),
        "device": predictor.device,
        "usable_labels": usable,
        # Per class, whether the registry's holdout says the class can be trusted. The
        # dead classes still appear in the output; this is how a reader tells them apart.
        "polygon_counts": [{"class": name, "polygons": n, "usable": name in usable}
                           for name, n in sorted(counts.items())],
        "scope": str(row.get("description") or ""),
        "class_map_path": package.class_map_path,
        "confidence_path": package.confidence_path,
        "polygons_path": package.polygons_path,
        "manifest_path": package.manifest_path,
    }
