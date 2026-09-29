"""Model inference over uploaded imagery, shared by AI jobs and detection jobs.

The contract is narrow on purpose. A request names a task and a registered model; the
model must be installed, must be of a kind that can answer that task, and must be the
file the registry measured. Anything else is refused before any work starts, because
the alternative -- quietly running a different model, or a heuristic, under the name
the user asked for -- produces results that look attributable and are not.

The one deliberate exception is the model key ``heuristic``. It is requested by name,
and every result it produces says ``heuristic`` rather than naming a model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HEURISTIC = "heuristic"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Which registry kinds can answer which task.
TASK_KINDS: dict[str, set[str]] = {
    "detection": {"onnx_yolo", "onnx_detection"},
    "segmentation": {"onnx_segmentation", "onnx_segmentation_multiclass"},
    "classification": {"onnx_classification"},
    "anomaly": {"onnx_classification"},
}
HEURISTIC_TASKS = {"detection", "segmentation"}

# Input contracts for the classifiers, measured against their validation tiles rather
# than assumed. The two installed classifiers were exported differently: one takes RGB
# in 0..1 and emits probabilities over [cracked, uncracked] (ImageFolder order, which
# the registry's single "crack" label does not record); the other expects ImageNet
# normalisation applied outside the graph and emits logits. Feeding either one the
# other's contract drops accuracy from ~0.98 / ~0.74 to ~0.80 / ~0.13 on held-out
# tiles. Keyed to the digest, so a retrained file under the same key is refused until
# its contract is checked again instead of being read with a stale one.
CLASSIFIER_CONTRACTS: dict[str, dict[str, Any]] = {
    "crack_presence_classifier": {
        "sha256": "92a4d1425b60a8ff96aa31de0ae404325f5a1ab47edc01f5b3acef061da34d74",
        "normalize": "none",
        "output": "probabilities",
        "classes": ["cracked", "uncracked"],
        "normal_class": "uncracked",
    },
    "solar_thermal_anomaly_classifier": {
        "sha256": "6933a09abff99bf8d4e2619a50a64498432d2877b9abd55ccab7a87ac357fd42",
        "normalize": "imagenet",
        "output": "logits",
        "classes": None,  # the registry's labels, which are in checkpoint order
        "normal_class": "No-Anomaly",
    },
}


class InferenceRefused(ValueError):
    """The request cannot be answered honestly with what is installed."""


class InferenceCancelled(Exception):
    """Raised between images when cancellation was requested."""


@dataclass(frozen=True)
class ModelChoice:
    task: str
    model_key: str
    kind: str
    path: str
    labels: list[str]
    sha256: str
    score_threshold: float
    input_size: int

    @property
    def is_heuristic(self) -> bool:
        return self.model_key == HEURISTIC

    def identity(self) -> dict[str, str]:
        return {"model_key": "" if self.is_heuristic else self.model_key,
                "model_sha256": self.sha256, "model_used": self.kind if self.is_heuristic
                else f"onnx:{Path(self.path).name}"}


def choose_model(task: str, model_key: str, *, verify_digest: bool = True) -> ModelChoice:
    """Resolve and check a model for a task, or refuse with the reason."""
    from core.models import get_model_spec, model_identity, model_status

    if task not in TASK_KINDS:
        raise InferenceRefused(
            f"Unknown AI task {task!r}. Supported: {', '.join(sorted(TASK_KINDS))}."
        )
    key = (model_key or "").strip()
    if key == HEURISTIC:
        if task not in HEURISTIC_TASKS:
            raise InferenceRefused(
                f"There is no heuristic for {task!r}; name a trained model instead."
            )
        return ModelChoice(task, HEURISTIC, HEURISTIC, "", [], "", 0.0, 0)

    spec = get_model_spec(key)
    if spec is None:
        raise InferenceRefused(f"Model {key!r} is not in the model registry.")
    if spec.kind not in TASK_KINDS[task]:
        suitable = _registered_for(task)
        raise InferenceRefused(
            f"Model {key!r} is a {spec.kind} model and cannot answer {task!r}. "
            f"Installed models for {task}: {', '.join(suitable) or 'none'}."
        )
    if spec.status != "installed":
        raise InferenceRefused(
            f"Model {key!r} is registered as {spec.status!r}: "
            f"{(spec.status_note or 'no weights').rstrip('.')}."
        )
    status = model_status(key)
    if not status.get("exists"):
        raise InferenceRefused(
            f"Model {key!r} weights are not installed on this server (expected at "
            f"{status.get('path', spec.path)}). No result can be produced without them."
        )
    digest = ""
    if verify_digest:
        digest = model_identity(key)["model_sha256"]
        if spec.sha256 and digest != spec.sha256:
            raise InferenceRefused(
                f"Model {key!r} on disk (sha256 {digest[:12]}) is not the file the registry "
                f"measured ({spec.sha256[:12]}); its published metrics do not apply."
            )
    if spec.kind == "onnx_classification":
        contract = CLASSIFIER_CONTRACTS.get(key)
        if contract is None or contract["sha256"] != (spec.sha256 or digest):
            raise InferenceRefused(
                f"Classifier {key!r} has no verified input contract for this build, so its "
                "outputs cannot be mapped to class names safely."
            )
        if task == "anomaly" and not contract.get("normal_class"):
            raise InferenceRefused(f"Classifier {key!r} has no 'normal' class to test against.")
    return ModelChoice(
        task=task, model_key=key, kind=spec.kind, path=str(status["path"]),
        labels=list(spec.labels), sha256=digest or spec.sha256,
        score_threshold=float(spec.score_threshold), input_size=int(spec.input_size),
    )


def _registered_for(task: str) -> list[str]:
    from core.models import load_registry, model_status

    rows = load_registry().get("models", {})
    return sorted(
        key for key, row in rows.items()
        if isinstance(row, dict) and row.get("kind") in TASK_KINDS[task]
        and row.get("status", "installed") == "installed" and model_status(key).get("exists")
    )


def model_catalogue() -> list[dict[str, Any]]:
    """Every registered model per task it could answer, with whether it can run here.

    Unusable models are listed with the reason rather than hidden, so a client can show
    why a choice is unavailable instead of offering a list that silently shrinks.
    """
    from core.models import get_model_spec, load_registry

    rows: list[dict[str, Any]] = []
    for task in sorted(TASK_KINDS):
        if task in HEURISTIC_TASKS:
            rows.append({"task": task, "model_key": HEURISTIC, "kind": HEURISTIC,
                         "description": "Classical image processing, not a trained model.",
                         "usable": True, "reason": ""})
        for key in sorted(load_registry().get("models", {})):
            spec = get_model_spec(key)
            if spec is None or spec.kind not in TASK_KINDS[task]:
                continue
            try:
                choose_model(task, key, verify_digest=False)
                usable, reason = True, ""
            except InferenceRefused as exc:
                usable, reason = False, str(exc)
            rows.append({"task": task, "model_key": key, "kind": spec.kind,
                         "description": spec.description, "usable": usable, "reason": reason})
    return rows


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._ -]+$")


def dataset_images(dataset_dir: Path, selector: str = "*") -> list[Path]:
    """Images in an uploaded dataset, optionally one named file."""
    if not dataset_dir.is_dir():
        raise InferenceRefused(f"The dataset has no uploaded files at {dataset_dir}.")
    selector = (selector or "*").strip()
    if selector in {"*", ""}:
        images = sorted(
            path for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise InferenceRefused("The dataset contains no uploaded images.")
        return images
    if not _SAFE_NAME.match(selector) or selector.startswith("."):
        raise InferenceRefused(f"Invalid file name {selector!r}.")
    candidate = dataset_dir / selector
    if not candidate.is_file():
        raise InferenceRefused(f"No uploaded file named {selector!r} in this dataset.")
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise InferenceRefused(f"{selector!r} is not an image this worker can read.")
    return [candidate]


def parse_input_ref(input_ref: str, dataset_id: int | None) -> tuple[int, str]:
    """Split ``dataset:<id>[/frame:<file>]`` (or a bare file name) into its parts."""
    ref = (input_ref or "").strip()
    match = re.fullmatch(r"dataset:(\d+)(?:/frame:(.+))?", ref)
    if match:
        ref_dataset = int(match.group(1))
        if dataset_id is not None and dataset_id != ref_dataset:
            raise InferenceRefused(
                f"input_ref names dataset {ref_dataset} but dataset_id is {dataset_id}."
            )
        return ref_dataset, match.group(2) or "*"
    if dataset_id is None:
        raise InferenceRefused(
            "input_ref must be 'dataset:<id>' or 'dataset:<id>/frame:<file name>', or "
            "dataset_id must be given with input_ref naming a file (or '*')."
        )
    return dataset_id, ref or "*"


def run_inference(
    choice: ModelChoice,
    images: list[Path],
    out_dir: Path,
    *,
    parameters: dict[str, Any] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Run one model over images, writing overlays/masks and a results.json.

    Returns the results document. Raises InferenceRefused if the model cannot run;
    a model that fails to load is never replaced by a heuristic here.
    """
    import cv2

    parameters = dict(parameters or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image: list[dict[str, Any]] = []
    artifacts: list[Path] = []

    for index, image_path in enumerate(images):
        if progress is not None:
            progress(int(100 * index / max(1, len(images))), f"{choice.task}: {image_path.name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            per_image.append({"image": image_path.name, "error": "unreadable image"})
            continue
        row, files = _infer_one(choice, image, image_path, out_dir, parameters)
        per_image.append(row)
        artifacts.extend(files)

    readable = [row for row in per_image if "error" not in row]
    if not readable:
        raise InferenceRefused("None of the selected images could be read.")

    document = {
        "task": choice.task,
        **choice.identity(),
        "image_count": len(images),
        "images": per_image,
        "summary": _summarise(choice, readable),
        "caveat": _caveat(choice),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    document["artifact_paths"] = [str(results_path)] + [str(path) for path in artifacts]
    if progress is not None:
        progress(100, f"{choice.task} complete")
    return document


def _caveat(choice: ModelChoice) -> str:
    if choice.is_heuristic:
        return ("Heuristic output: classical image processing, not a trained model. Treat "
                "it as a prompt for review, not a finding.")
    if choice.task in {"classification", "anomaly"}:
        return ("Model output, unreviewed. One label per whole image; the model's stated "
                "scope in the registry applies, and images outside it get confident but "
                "meaningless labels.")
    return ("Model output, unreviewed. Pixel geometry only; nothing here is georeferenced "
            "or measured in metres.")


def _net(path: str):
    from core.detection import _load_onnx_net

    net = _load_onnx_net(Path(path))
    if net is None:
        raise InferenceRefused(f"The ONNX runtime could not load {Path(path).name}.")
    return net


def _model_step(choice: ModelChoice, image_path: Path, step):
    """Run one model call, refusing with the real cause when it fails.

    Out-of-memory and runtime errors are reported as such, so a job that failed for
    lack of RAM does not read as a model that looked and found nothing.
    """
    try:
        return step()
    except InferenceRefused:
        raise
    except MemoryError as exc:
        raise InferenceRefused(
            f"{choice.model_key} ran out of memory on {image_path.name}.") from exc
    except Exception as exc:
        raise InferenceRefused(
            f"{choice.model_key} failed on {image_path.name}: "
            f"{type(exc).__name__}: {str(exc).strip()[:300]}") from exc


def _infer_one(choice: ModelChoice, image, image_path: Path, out_dir: Path,
               parameters: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    import cv2
    import numpy as np

    stem = image_path.stem
    row: dict[str, Any] = {"image": image_path.name,
                           "width": int(image.shape[1]), "height": int(image.shape[0])}
    files: list[Path] = []
    min_area = int(parameters.get("min_area_px", 40))

    if choice.task == "detection":
        from core.detection import DefectHit, _run_onnx_yolo, detect_structural_defects

        if choice.is_heuristic:
            result = detect_structural_defects(image, min_area_px=min_area, use_model=False)
            hits: list[DefectHit] = list(result.detections)
        else:
            _net(choice.path)  # a graph that will not load must fail, not return "no hits"
            threshold = float(parameters.get("score_threshold", choice.score_threshold))
            hits = _model_step(choice, image_path, lambda: _run_onnx_yolo(
                image_bgr=image, model_path=Path(choice.path), labels=choice.labels,
                score_threshold=threshold, iou_threshold=0.45, input_size=choice.input_size,
            ))
        row["detections"] = [hit.to_dict() for hit in hits]
        if hits:
            overlay = image.copy()
            for hit in hits:
                x1, y1, x2, y2 = hit.bbox_xyxy
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # A heuristic region has no confidence, and none is drawn for it.
                caption = (hit.label if hit.confidence is None
                           else f"{hit.label} {hit.confidence:.2f}")
                cv2.putText(overlay, caption, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            path = out_dir / f"{stem}_detections.jpg"
            cv2.imwrite(str(path), overlay)
            files.append(path)

    elif choice.task == "segmentation":
        if choice.is_heuristic:
            from core.detection import detect_cracks

            mask = (detect_cracks(image, min_area_px=min_area, use_model=False).mask > 0).astype(np.uint8)
            labels = ["crack"]
            index_map = mask
        elif choice.kind == "onnx_segmentation":
            from core.detection import _run_onnx_segmentation

            _net(choice.path)
            threshold = float(parameters.get("score_threshold", choice.score_threshold or 0.5))
            index_map = _model_step(choice, image_path, lambda: _run_onnx_segmentation(
                image, Path(choice.path), threshold, choice.input_size, raise_errors=True))
            labels = choice.labels[:1]
        else:
            from core.detection import _run_onnx_segmentation_multiclass

            _net(choice.path)
            index_map = _model_step(choice, image_path, lambda: _run_onnx_segmentation_multiclass(
                image, Path(choice.path), choice.input_size, len(choice.labels),
                raise_errors=True))
            labels = choice.labels
        total = float(index_map.size)
        if len(labels) == 1:
            row["pixel_fraction"] = {labels[0]: float((index_map > 0).sum() / total)}
            visual = (index_map > 0).astype(np.uint8) * 255
        else:
            row["pixel_fraction"] = {
                label: float((index_map == i).sum() / total) for i, label in enumerate(labels)
            }
            visual = (index_map.astype(np.float32) * (255.0 / max(1, len(labels) - 1))).astype(np.uint8)
        path = out_dir / f"{stem}_mask.png"
        cv2.imwrite(str(path), visual)
        files.append(path)

    else:  # classification / anomaly
        probabilities, classes = _model_step(choice, image_path, lambda: _classify(choice, image))
        order = np.argsort(-probabilities)
        top_k = int(parameters.get("top_k", 3))
        row["top"] = [{"label": classes[i], "probability": float(probabilities[i])}
                      for i in order[:max(1, top_k)]]
        row["predicted"] = classes[int(order[0])]
        if choice.task == "anomaly":
            normal = CLASSIFIER_CONTRACTS[choice.model_key]["normal_class"]
            row["anomalous"] = row["predicted"] != normal
            row["anomaly_probability"] = float(1.0 - probabilities[classes.index(normal)])
    return row, files


def _classify(choice: ModelChoice, image):
    import cv2
    import numpy as np

    contract = CLASSIFIER_CONTRACTS[choice.model_key]
    classes = list(contract["classes"] or choice.labels)
    size = int(choice.input_size)
    rgb = cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if contract["normalize"] == "imagenet":
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
    net = _net(choice.path)
    net.setInput(rgb.transpose(2, 0, 1)[None].astype(np.float32))
    scores = np.asarray(net.forward(), dtype=np.float64).reshape(-1)
    if scores.size != len(classes):
        raise InferenceRefused(
            f"{choice.model_key} emitted {scores.size} scores for {len(classes)} classes.")
    if contract["output"] == "logits":
        scores = np.exp(scores - scores.max())
        scores = scores / scores.sum()
    return scores, classes


def _summarise(choice: ModelChoice, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if choice.task == "detection":
        counts: dict[str, int] = {}
        for row in rows:
            for hit in row["detections"]:
                counts[hit["label"]] = counts.get(hit["label"], 0) + 1
        return {"detection_count": sum(counts.values()), "by_label": counts,
                "images_with_detections": sum(1 for row in rows if row["detections"])}
    if choice.task == "segmentation":
        labels = rows[0]["pixel_fraction"].keys()
        return {"mean_pixel_fraction": {
            label: sum(row["pixel_fraction"][label] for row in rows) / len(rows)
            for label in labels}}
    predicted: dict[str, int] = {}
    for row in rows:
        predicted[row["predicted"]] = predicted.get(row["predicted"], 0) + 1
    summary: dict[str, Any] = {"predicted_counts": predicted}
    if choice.task == "anomaly":
        summary["anomalous_images"] = sum(1 for row in rows if row["anomalous"])
    return summary
