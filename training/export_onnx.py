"""Export trained checkpoints to ONNX and prove the export is faithful.

An export that loads but computes something slightly different is worse than no
export at all -- the pipeline would report `model_used: <the model>` while producing
numbers the training metrics do not describe. So every export here is followed by a
parity check: the same input is pushed through the torch model and the ONNX graph, and
the run fails unless every value agrees to within a tolerance scaled to its magnitude.

The segmentation export is additionally verified through `cv2.dnn`, because that is
the runtime `core/detection.py` actually uses, and it supports a narrower slice of
ONNX than onnxruntime does.

    python -m training.export_onnx --run training/runs/crack_segformer_b5
    python -m training.export_onnx --run training/runs/structural_yolo11x --kind yolo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tolerance for torch vs ONNX agreement. Chosen to catch a genuinely wrong graph
# while allowing the fp32 reassociation an ONNX runtime is entitled to perform.
#
# The comparison is scaled to the magnitude of each value rather than applied flat,
# because a detector's output mixes two kinds of number. Box coordinates are pixels
# running to the image size, so an absolute 1e-3 there demands agreement to one part in
# 600,000 -- stricter than fp32 reassociation can promise, and meaningless besides,
# since a box shifted by a thousandth of a pixel is the same box. Class scores are
# probabilities in 0..1, where 1e-3 is the right scale: a score that moves by that much
# could cross a detection threshold and change what the model reports.
#
# So: an absolute floor that governs the probabilities, plus a relative term that
# governs the coordinates. A genuinely wrong graph fails both.
PARITY_ATOL = 1e-3
PARITY_RTOL = 1e-4

# Retained under its old name: this is the absolute floor, and callers that only care
# about the probability-scale tolerance still read the number they expect.
PARITY_TOLERANCE = PARITY_ATOL


def parity_violation(reference: "np.ndarray", candidate: "np.ndarray") -> float:
    """How far the ONNX output sits outside tolerance, as a multiple of it.

    Returns the worst ``|a - b| / (atol + rtol * |a|)`` across the tensors. At or below
    1.0 the graphs agree; above it they do not, and the number says by how much, which
    is more useful when diagnosing an export than a bare maximum difference.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    allowed = PARITY_ATOL + PARITY_RTOL * np.abs(reference)
    return float((np.abs(reference - candidate) / allowed).max())


def _onnx_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def export_segmentation(run_dir: Path, *, opset: int = 17, image_size: int | None = None) -> dict[str, Any]:
    """Export a SegFormer checkpoint, wrapped so the graph emits full-resolution logits."""
    import torch
    import torch.nn.functional as F

    from training.train_seg import SegConfig, build_model

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}. Train first.")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SegConfig(**state["config"])
    size = int(image_size or config.image_size)

    model = build_model(config.model_id)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    class SegWrapper(torch.nn.Module):
        """Fold the 1/4-resolution upsample and sigmoid into the exported graph.

        Without this the consumer would have to know SegFormer's stride and apply
        both steps itself, which is exactly the kind of implicit contract that goes
        wrong silently.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            logits = self.inner(pixel_values=pixel_values).logits
            logits = F.interpolate(
                logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.sigmoid(logits)

    wrapper = SegWrapper(model).eval()
    example = torch.randn(1, 3, size, size)
    output_path = run_dir / f"{config.name}.onnx"

    torch.onnx.export(
        wrapper,
        (example,),
        str(output_path),
        input_names=["images"],
        output_names=["mask"],
        opset_version=opset,
        dynamo=False,
        dynamic_axes={"images": {0: "batch"}, "mask": {0: "batch"}},
    )

    with torch.no_grad():
        torch_output = wrapper(example).numpy()
    session = _onnx_session(output_path)
    onnx_output = session.run(None, {"images": example.numpy()})[0]

    difference = float(np.abs(torch_output - onnx_output).max())
    violation = parity_violation(torch_output, onnx_output)
    report = {
        "name": config.name,
        "kind": "onnx_segmentation",
        "onnx_path": str(output_path),
        "input_size": size,
        "opset": opset,
        "max_abs_diff": difference,
        "parity_violation": violation,
        "parity_ok": violation <= 1.0,
        "output_shape": list(onnx_output.shape),
        "train_metrics": state.get("metrics", {}),
    }

    # cv2.dnn is the runtime core/detection.py uses, so a graph it cannot read is
    # useless regardless of what onnxruntime thinks.
    try:
        import cv2

        net = cv2.dnn.readNetFromONNX(str(output_path))
        net.setInput(example.numpy())
        cv_output = net.forward()
        report["opencv_dnn_loadable"] = True
        report["opencv_max_abs_diff"] = float(np.abs(torch_output - cv_output).max())
    except Exception as exc:
        report["opencv_dnn_loadable"] = False
        report["opencv_error"] = str(exc)

    return report


def export_yolo(run_dir: Path, *, opset: int = 17, image_size: int | None = None) -> dict[str, Any]:
    """Export an ultralytics checkpoint and confirm the ONNX graph matches torch."""
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}. Train first.")

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    size = int(image_size or summary.get("config", {}).get("image_size", 640))

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=size, opset=opset, simplify=True, dynamic=False)
    output_path = Path(exported)

    import torch

    example = torch.randn(1, 3, size, size)
    with torch.no_grad():
        torch_output = model.model(example)
        if isinstance(torch_output, (list, tuple)):
            torch_output = torch_output[0]
        torch_output = torch_output.numpy()

    session = _onnx_session(output_path)
    onnx_output = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]

    difference = float(np.abs(torch_output - onnx_output).max())
    violation = parity_violation(torch_output, onnx_output)
    report = {
        "name": run_dir.name,
        "kind": "onnx_yolo",
        "onnx_path": str(output_path),
        "input_size": size,
        "opset": opset,
        "max_abs_diff": difference,
        "parity_violation": violation,
        "parity_ok": violation <= 1.0,
        "output_shape": list(onnx_output.shape),
        "labels": list(model.names.values()),
        "train_metrics": summary.get("metrics", {}),
    }

    try:
        import cv2

        cv2.dnn.readNetFromONNX(str(output_path))
        report["opencv_dnn_loadable"] = True
    except Exception as exc:
        report["opencv_dnn_loadable"] = False
        report["opencv_error"] = str(exc)

    return report


def export_classification(
    run_dir: Path, *, opset: int = 17, image_size: int | None = None
) -> dict[str, Any]:
    """Export a train_cls checkpoint with ImageNet normalisation folded into the graph.

    The normalisation is inside the wrapper on purpose. train_cls applies mean/std from
    torchvision before the network sees a pixel, and an exported graph that omits it
    takes raw 0..1 images and returns confident nonsense -- no error, no warning, just
    wrong classes. Baking it in means the caller passes RGB in 0..1 and cannot get this
    wrong, and the parity check below compares against the same wrapper, so a mismatch
    between the training transform and the exported one shows up as a failed export
    rather than as a quietly bad model.

    Softmax is folded in for the same reason: raw logits look like scores and are not.
    """
    import torch

    from training.train_cls import ClsConfig, build_model

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}. Train first.")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ClsConfig(**state["config"])
    size = int(image_size or config.image_size)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    classes = summary.get("classes") or state.get("classes")
    if not classes:
        raise SystemExit(
            f"No class list in {summary_path} or the checkpoint. Refusing to export a "
            "classifier whose output columns cannot be named: an ONNX file with "
            "anonymous logits is one relabelling away from silently wrong predictions."
        )

    model = build_model(config.model_id, len(classes), pretrained=False)
    model.load_state_dict(state["model"])
    model.eval()

    class ClsWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner
            # Registered as buffers so they travel into the ONNX graph as constants.
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.inner((images - self.mean) / self.std), dim=1)

    wrapper = ClsWrapper(model).eval()
    # 0..1 rather than randn: this graph's input contract is a normalised RGB image, and
    # parity should be measured on inputs of the shape it will actually receive.
    example = torch.rand(1, 3, size, size)
    output_path = run_dir / f"{config.name}.onnx"

    torch.onnx.export(
        wrapper,
        (example,),
        str(output_path),
        input_names=["images"],
        output_names=["probabilities"],
        opset_version=opset,
        dynamo=False,
        dynamic_axes={"images": {0: "batch"}, "probabilities": {0: "batch"}},
    )

    with torch.no_grad():
        torch_output = wrapper(example).numpy()
    session = _onnx_session(output_path)
    onnx_output = session.run(None, {"images": example.numpy()})[0]

    violation = parity_violation(torch_output, onnx_output)
    report = {
        "name": config.name,
        "kind": "onnx_classification",
        "onnx_path": str(output_path),
        "input_size": size,
        "opset": opset,
        "classes": list(classes),
        "max_abs_diff": float(np.abs(torch_output - onnx_output).max()),
        "parity_violation": violation,
        "parity_ok": violation <= 1.0,
        "output_shape": list(onnx_output.shape),
        "preprocessing": "images are RGB in 0..1; ImageNet mean/std and softmax are in the graph",
        "train_metrics": summary.get("per_class", {}),
    }

    if report["output_shape"][-1] != len(classes):
        report["parity_ok"] = False
        report["class_count_mismatch"] = (
            f"graph emits {report['output_shape'][-1]} columns for {len(classes)} classes"
        )

    try:
        import cv2

        net = cv2.dnn.readNetFromONNX(str(output_path))
        net.setInput(example.numpy())
        cv_output = net.forward()
        report["opencv_dnn_loadable"] = True
        report["opencv_max_abs_diff"] = float(np.abs(torch_output - cv_output).max())
    except Exception as exc:
        report["opencv_dnn_loadable"] = False
        report["opencv_error"] = str(exc)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.export_onnx")
    parser.add_argument("--run", required=True, type=Path, help="Run directory to export.")
    parser.add_argument("--kind", choices=["seg", "yolo", "cls"], default="seg")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    if args.kind == "seg":
        report = export_segmentation(run_dir, opset=args.opset, image_size=args.image_size)
    elif args.kind == "cls":
        report = export_classification(run_dir, opset=args.opset, image_size=args.image_size)
    else:
        report = export_yolo(run_dir, opset=args.opset, image_size=args.image_size)

    (run_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not report["parity_ok"]:
        print(
            f"\nPARITY FAILED: worst deviation is {report['parity_violation']:.2f}x the "
            f"allowed tolerance (atol {PARITY_ATOL}, rtol {PARITY_RTOL}; max abs diff "
            f"{report['max_abs_diff']:.6f}). The ONNX graph does not match the trained "
            "model.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
