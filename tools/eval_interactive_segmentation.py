"""Measure the interactive segmenters honestly: parity with PyTorch, and box-prompt mIoU.

Each number is reported under its own name and never as defect accuracy:

parity
    Our ONNX path (core.interactive_segmentation: our preprocessing, our graphs, our
    mask selection) against the publisher's PyTorch reference on the same photos and the
    same prompts. Mask IoU between the two. This says the conversion and the
    preprocessing reproduce the reference; it says nothing about whether either is right.
    SAM 2.1-T reference: transformers Sam2Processor + Sam2Model (facebook/sam2.1-hiera-tiny).
    EfficientSAM-Ti reference: yformer/EfficientSAM build_efficient_sam (efficient_sam_vitt.pt).

neontree
    Box-prompt mask IoU against the manual crown delineations of Zenodo 15420565 (a
    subset of NeonTreeEvaluation, CC BY 4.0). Each crown is prompted with the tight
    bounding box of its own delineation; the mask kept is the one the app delivers
    (highest predicted IoU, largest region, holes filled). Reported as mIoU over N crowns.
    This is tree-crown segmentation from 10 cm aerial RGB, not defect segmentation.

ttpla
    The same box-prompt protocol on TTPLA tower polygons, when odk-testdata/ttpla_test
    exists.

quantize
    Dynamic int8 (weights) of the SAM 2.1-T graphs, for the int8 measurements only.

Usage:
    python tools/eval_interactive_segmentation.py parity --images DIR -n 20 --out parity.json
    python tools/eval_interactive_segmentation.py neontree --root DIR --out neontree.json
    python tools/eval_interactive_segmentation.py quantize --out DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core import interactive_segmentation as S  # noqa: E402

WEIGHTS = Path(os.environ.get("ODK_MODEL_WEIGHTS_DIR", "E:/Shit was cluittering/Projects/odk-weights"))
SAM_DIR = WEIGHTS / S.SAM21_KEY
EFF_DIR = WEIGHTS / S.EFFICIENTSAM_KEY


def variants() -> dict[str, S.PromptModel]:
    out: dict[str, S.PromptModel] = {}
    enc = SAM_DIR / f"{S.SAM21_KEY}__vision_encoder.onnx"
    dec = SAM_DIR / f"{S.SAM21_KEY}__mask_decoder.onnx"
    if enc.is_file() and dec.is_file():
        out["sam21_fp32"] = S.PromptModel.from_files(S.SAM21_KEY, enc, dec)
    enc8, dec8 = SAM_DIR / "_int8" / "vision_encoder_int8.onnx", SAM_DIR / "_int8" / "mask_decoder_int8.onnx"
    if enc8.is_file() and dec8.is_file():
        out["sam21_int8"] = S.PromptModel.from_files(S.SAM21_KEY, enc8, dec8)
    eff = EFF_DIR / f"{S.EFFICIENTSAM_KEY}.onnx"
    if eff.is_file():
        out["efficientsam_fp32"] = S.PromptModel.from_files(S.EFFICIENTSAM_KEY, eff)
    eff8 = EFF_DIR / "_src" / "opencv" / "image_segmentation_efficientsam_ti_2025april_int8.onnx"
    if eff8.is_file():
        out["efficientsam_int8"] = S.PromptModel.from_files(S.EFFICIENTSAM_KEY, eff8)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def ours(model: S.PromptModel, image_bgr: np.ndarray, points, box) -> S.MaskResult:
    # The machine is shared with jobs that can exhaust the commit limit for a moment; a
    # failed 12 MB allocation is retried rather than ending a 20-minute measurement.
    for attempt in range(20):
        try:
            emb = S.encode(image_bgr, model=model)
            return S.predict(emb, points, box, model=model)
        except (MemoryError, cv2.error) as exc:
            if isinstance(exc, cv2.error) and "memory" not in str(exc).lower():
                raise
            gc.collect()
            print("MemoryError; retrying in 30 s", flush=True)
            time.sleep(30)
    raise MemoryError("the machine stayed out of memory for 10 minutes")


def retry(fn):
    for attempt in range(20):
        try:
            return fn()
        except (MemoryError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "memory" not in str(exc).lower():
                raise
            gc.collect()
            print("out of memory in the reference; retrying in 30 s", flush=True)
            time.sleep(30)
    raise MemoryError("the machine stayed out of memory for 10 minutes")


def free_sessions() -> None:
    S.clear_cache()
    S._SESSIONS.clear()
    gc.collect()


# ── Parity against PyTorch ────────────────────────────────────────────────────

def parity_prompts(h: int, w: int) -> list[dict[str, Any]]:
    """Three fixed prompts per photo: a box, one include point, include + exclude."""
    return [
        {"name": "box", "points": [], "box": [0.35 * w, 0.35 * h, 0.65 * w, 0.65 * h]},
        {"name": "point", "points": [[0.5 * w, 0.5 * h, 1]], "box": None},
        {"name": "point_pos_neg", "points": [[0.5 * w, 0.5 * h, 1], [0.3 * w, 0.3 * h, 0]],
         "box": None},
    ]


class Sam2Reference:
    def __init__(self, source: Path):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.proc = Sam2Processor.from_pretrained(str(source))
        self.model = Sam2Model.from_pretrained(str(source)).eval().float()

    def __call__(self, image_bgr, points, box) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        kwargs: dict[str, Any] = {}
        if points:
            kwargs["input_points"] = [[[[float(p[0]), float(p[1])] for p in points]]]
            kwargs["input_labels"] = [[[int(p[2]) for p in points]]]
        if box is not None:
            kwargs["input_boxes"] = [[[float(v) for v in box]]]
        inputs = self.proc(images=pil, return_tensors="pt", **kwargs)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks, inputs["original_sizes"])[0]
        best = int(out.iou_scores[0, 0].argmax())
        return masks[0, best].numpy().astype(bool)


class EfficientSamReference:
    def __init__(self, repo: Path):
        import torch

        sys.path.insert(0, str(repo))
        from efficient_sam.efficient_sam import build_efficient_sam

        self.torch = torch
        self.model = build_efficient_sam(encoder_patch_embed_dim=192, encoder_num_heads=3,
                                         checkpoint=str(repo / "weights" / "efficient_sam_vitt.pt")).eval()

    def __call__(self, image_bgr, points, box) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        pts, lbl = [], []
        if box is not None:
            pts += [[box[0], box[1]], [box[2], box[3]]]
            lbl += [2, 3]
        for p in points:
            pts.append([p[0], p[1]])
            lbl.append(int(p[2]))
        P = torch.tensor([[pts]], dtype=torch.float32)
        L = torch.tensor([[lbl]], dtype=torch.float32)
        with torch.no_grad():
            logits, scores = self.model(x, P, L)
        best = int(scores[0, 0].argmax())
        return (logits[0, 0, best] >= 0).numpy()


def run_parity(args) -> dict[str, Any]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in exts)
    step = max(1, len(files) // args.n)
    chosen = files[::step][: args.n]
    models = variants()
    report: dict[str, Any] = {
        "measurement": "PARITY: mask IoU of the OpenDroneKit ONNX path against the publisher's PyTorch reference (not accuracy)",
        "images": [str(p) for p in chosen], "n_images": len(chosen),
        "image_long_side_px": args.max_side,
        "prompts_per_image": ["box: central 30% x 30%", "point: centre, include",
                              "point_pos_neg: centre include + (0.3W, 0.3H) exclude"],
        "results": {},
    }
    groups = [("sam21", lambda: Sam2Reference(SAM_DIR / "_src" / "facebook"), ["sam21_fp32", "sam21_int8"]),
              ("efficientsam", lambda: EfficientSamReference(Path(args.efficientsam_repo)),
               ["efficientsam_fp32", "efficientsam_int8"])]
    for name, make_ref, keys in groups:
        keys = [k for k in keys if k in models and (not args.only or k in args.only)]
        if not keys:
            continue
        ref = make_ref()
        rows: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
        for path in chosen:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if max(image.shape[:2]) > args.max_side:
                scale = args.max_side / float(max(image.shape[:2]))
                image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
            for prompt in parity_prompts(h, w):
                t = time.time()
                reference = retry(lambda: ref(image, prompt["points"], prompt["box"]))
                t_ref = time.time() - t
                for k in keys:
                    t = time.time()
                    res = ours(models[k], image, prompt["points"], prompt["box"])
                    rows[k].append({"image": path.name, "prompt": prompt["name"],
                                    "mask_iou": round(iou(res.mask, reference), 5),
                                    "ref_px": int(reference.sum()), "ours_px": int(res.mask.sum()),
                                    "ref_s": round(t_ref, 2), "ours_s": round(time.time() - t, 2)})
                print(name, path.name, prompt["name"],
                      {k: rows[k][-1]["mask_iou"] for k in keys}, flush=True)
            S.clear_cache()
        for k in keys:
            vals = np.array([r["mask_iou"] for r in rows[k]])
            report["results"][k] = {
                "model_file_sha256": models[k].graph_sha256,
                "companion_sha256": models[k].companion_sha256,
                "n_comparisons": int(vals.size),
                "mask_iou_mean": round(float(vals.mean()), 4),
                "mask_iou_median": round(float(np.median(vals)), 4),
                "mask_iou_min": round(float(vals.min()), 4),
                "share_iou_ge_0_95": round(float((vals >= 0.95).mean()), 4),
                "rows": rows[k],
            }
        del ref
        free_sessions()
    return report


# ── NeonTree crowns ───────────────────────────────────────────────────────────

def read_rgb_bgr(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        data = src.read([1, 2, 3])
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(data, (1, 2, 0))[..., ::-1])


def run_neontree(args) -> dict[str, Any]:
    root = Path(args.root)
    rgb_dir = root / "data" / "RGB"
    mask_dir = root / "data" / "Annotations" / "Masks"
    models = {k: v for k, v in variants().items() if not args.only or k in args.only}
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    per_image: dict[str, dict[str, Any]] = {}
    stems = [p.stem for p in sorted(mask_dir.glob("*.npy"))]
    # One model at a time, so only one model's sessions are in memory.
    for k, model in models.items():
        for stem in stems:
            image = read_rgb_bgr(rgb_dir / f"{stem}.tif")
            gt = np.load(mask_dir / f"{stem}.npy")
            if gt.ndim == 2:
                gt = gt[None]
            if gt.shape[1:] != image.shape[:2]:
                raise SystemExit(f"{stem}: mask {gt.shape} does not match image {image.shape}")
            crowns = [g for g in gt if g.any()]
            per_image[stem] = {"crowns": len(crowns),
                               "empty_masks_skipped": int(gt.shape[0] - len(crowns))}
            for idx, g in enumerate(crowns):
                ys, xs = np.nonzero(g)
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                res = ours(model, image, [], box)
                app_mask = S.filled_region(res.mask)
                rows[k].append({"image": stem, "crown": idx, "gt_px": int(g.sum()),
                                "iou_app": round(iou(app_mask, g), 5),
                                "iou_raw": round(iou(res.mask, g), 5),
                                "score": round(res.score, 4)})
            vals = [r["iou_app"] for r in rows[k] if r["image"] == stem]
            print(stem, k, len(vals), round(float(np.mean(vals)), 4) if vals else None, flush=True)
            S.clear_cache()
        free_sessions()
    report: dict[str, Any] = {
        "measurement": "box-prompt mask IoU against manual crown delineations (tree crowns, not defects)",
        "dataset": "Zenodo 15420565, Subset of NeonTreeEvaluation Dataset with Manual Delineations (CC BY 4.0), all 18 images",
        "split": "NeonTreeEvaluation manual-delineation subset, box prompt",
        "prompt": "tight bounding box of each crown's own delineation [xmin, ymin, xmax+1, ymax+1]",
        "mask_scored": "the mask the app delivers: highest predicted IoU, largest region, holes filled (iou_app); the raw thresholded mask as iou_raw",
        "images": per_image,
        "results": {},
    }
    for k, r in rows.items():
        app = np.array([x["iou_app"] for x in r])
        raw = np.array([x["iou_raw"] for x in r])
        report["results"][k] = {
            "model_file_sha256": models[k].graph_sha256,
            "companion_sha256": models[k].companion_sha256,
            "n_crowns": int(app.size), "n_images": len({x["image"] for x in r}),
            "miou": round(float(app.mean()), 4), "miou_raw_mask": round(float(raw.mean()), 4),
            "median_iou": round(float(np.median(app)), 4),
            "share_iou_ge_0_5": round(float((app >= 0.5).mean()), 4),
            "rows": r,
        }
    return report


# ── TTPLA towers (only when U5's test split is present) ───────────────────────

def run_ttpla(args) -> dict[str, Any]:
    root = Path(args.root)
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png"}}
    labels = sorted(root.rglob("*.json"))
    models = {k: v for k, v in variants().items() if not args.only or k in args.only}
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    for lab in labels:
        try:
            data = json.loads(lab.read_text(encoding="utf-8"))
        except ValueError:
            continue
        shapes = [s for s in data.get("shapes", []) if str(s.get("label", "")).startswith("cable") is False
                  and "tower" in str(s.get("label", "")).lower() and len(s.get("points", [])) >= 3]
        img_path = images.get(Path(str(data.get("imagePath", lab.stem))).stem) or images.get(lab.stem)
        if not shapes or img_path is None:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for k, model in models.items():
            for s in shapes:
                gt = np.zeros((h, w), np.uint8)
                cv2.fillPoly(gt, [np.round(np.array(s["points"], np.float64)).astype(np.int32)], 1)
                g = gt.astype(bool)
                if not g.any():
                    continue
                ys, xs = np.nonzero(g)
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                res = ours(model, image, [], box)
                rows[k].append({"image": img_path.name, "label": s["label"],
                                "iou_app": round(iou(S.filled_region(res.mask), g), 5)})
        S.clear_cache()
    out: dict[str, Any] = {"measurement": "box-prompt mask IoU on TTPLA tower polygons (not defects)",
                           "root": str(root), "results": {}}
    for k, r in rows.items():
        vals = np.array([x["iou_app"] for x in r]) if r else np.array([])
        out["results"][k] = {"n_towers": int(vals.size),
                             "miou": round(float(vals.mean()), 4) if vals.size else None,
                             "model_file_sha256": models[k].graph_sha256, "rows": r}
    return out


def run_quantize(args) -> dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(SAM_DIR / f"{S.SAM21_KEY}__vision_encoder.onnx", out / "vision_encoder_int8.onnx"),
             (SAM_DIR / f"{S.SAM21_KEY}__mask_decoder.onnx", out / "mask_decoder_int8.onnx")]
    result = {}
    for src, dst in pairs:
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        result[dst.name] = {"bytes": dst.stat().st_size, "sha256": S.file_sha256(dst)}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--images", required=True)
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--max-side", type=int, default=1600,
                   help="photos are shrunk (INTER_AREA) to this long side first, to fit a shared machine's memory")
    p.add_argument("--efficientsam-repo",
                   default=str(WEIGHTS / "_reference" / "yformer__EfficientSAM"))
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", required=True)
    p = sub.add_parser("neontree")
    p.add_argument("--root", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ttpla")
    p.add_argument("--root", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", required=True)
    p = sub.add_parser("quantize")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    fn = {"parity": run_parity, "neontree": run_neontree, "ttpla": run_ttpla,
          "quantize": run_quantize}[args.cmd]
    t = time.time()
    report = fn(args)
    report["seconds"] = round(time.time() - t, 1)
    report["measured_by"] = "opendronekit"
    report["tool"] = f"tools/eval_interactive_segmentation.py {args.cmd}"
    if getattr(args, "out", None) and args.cmd != "quantize":
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"} | {
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                    for k, v in (report.get("results") or {}).items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
