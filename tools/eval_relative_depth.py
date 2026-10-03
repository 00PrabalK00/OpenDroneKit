"""Measure the relative-depth model honestly: ONNX variant parity, and agreement with SfM.

Two measurements, reported under their own names and never as "accuracy":

parity
    fp32 vs int8 ONNX on the same photos. Spearman rho between the two maps, and AbsRel
    of the int8 map after a least-squares scale-and-shift onto the fp32 map. This says
    whether int8 reproduces fp32; it says nothing about whether either is right.

sfm
    Agreement with a COLMAP sparse reconstruction. Each 3-D point seen in an image is
    projected into that image to give a sparse depth z (in the reconstruction's own,
    arbitrary units). Because the network's output is relative *inverse* depth, a scale
    and shift are fitted in inverse-depth space (pred_inv ~ a * 1/z + b, solved for
    1/z) on a random half of the points, and AbsRel and delta1 (max ratio < 1.25) are
    evaluated on the other half. Reported per image and as the mean over N images.
    The scale-shift fit is what makes this a relative comparison; the units of the
    reconstruction never reach the result.

Usage:
    python tools/eval_relative_depth.py parity --fp32 A.onnx --int8 B.onnx --images DIR -n 20
    python tools/eval_relative_depth.py sfm --model A.onnx --sparse DIR/sparse/0 --images DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.depth import infer_onnx_network  # noqa: E402
from core.detection import load_image  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def fit_scale_shift(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares (s, t) with s * pred + t ~ target."""
    A = np.stack([pred, np.ones_like(pred)], axis=1).astype(np.float64)
    (s, t), *_ = np.linalg.lstsq(A, target.astype(np.float64), rcond=None)
    return float(s), float(t)


def parity_pair(ref: np.ndarray, other: np.ndarray, rng: np.random.Generator,
                samples: int = 200_000) -> dict[str, float]:
    r = ref.reshape(-1)
    o = other.reshape(-1)
    idx = rng.choice(r.size, size=min(samples, r.size), replace=False)
    s, t = fit_scale_shift(o, r)
    aligned = s * o + t
    good = np.abs(r) > 1e-6
    absrel = float(np.mean(np.abs(aligned[good] - r[good]) / np.abs(r[good])))
    return {"spearman_rho": spearman(r[idx], o[idx]), "absrel_aligned": absrel}


def list_images(folder: Path, n: int | None = None) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if n is not None and len(files) > n:
        step = len(files) / float(n)
        files = [files[int(i * step)] for i in range(n)]
    return files


def run_parity(fp32: Path, int8: Path, images: list[Path], seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = []
    t_fp32 = t_int8 = 0.0
    for path in images:
        img = load_image(path)
        t0 = time.perf_counter()
        # Compared at the network's resolution: the upsample is the same deterministic
        # bicubic for both, and full-resolution float maps do not fit on a busy machine.
        a = infer_onnx_network(img, fp32)
        t1 = time.perf_counter()
        b = infer_onnx_network(img, int8)
        t2 = time.perf_counter()
        t_fp32 += t1 - t0
        t_int8 += t2 - t1
        rows.append({"image": path.name, **parity_pair(a, b, rng)})
    rho = np.array([r["spearman_rho"] for r in rows])
    rel = np.array([r["absrel_aligned"] for r in rows])
    return {
        "measurement": "PARITY (int8 vs fp32 ONNX, raw network-resolution maps), not accuracy",
        "n_images": len(rows),
        "spearman_rho_mean": float(rho.mean()), "spearman_rho_min": float(rho.min()),
        "absrel_aligned_mean": float(rel.mean()), "absrel_aligned_max": float(rel.max()),
        "seconds_per_image_fp32": t_fp32 / max(1, len(rows)),
        "seconds_per_image_int8": t_int8 / max(1, len(rows)),
        "per_image": rows,
    }


def load_sparse(sparse_dir: Path) -> Any:
    import pycolmap

    return pycolmap.Reconstruction(str(sparse_dir))


def sparse_depths(recon: Any, image: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, z) in pixels and camera-frame depth for each 3-D point this image observes."""
    pose = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) \
        else image.cam_from_world
    R = np.asarray(pose.rotation.matrix())
    t = np.asarray(pose.translation).reshape(3)
    xs, ys, zs = [], [], []
    for p2d in image.points2D:
        if not p2d.has_point3D():
            continue
        X = np.asarray(recon.points3D[p2d.point3D_id].xyz)
        Xc = R @ X + t
        if Xc[2] <= 0:
            continue
        xs.append(p2d.xy[0])
        ys.append(p2d.xy[1])
        zs.append(Xc[2])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def sample_network_map(net: np.ndarray, x: np.ndarray, y: np.ndarray,
                       photo_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear sample of a network-resolution map at photo pixel coordinates.

    Equivalent, up to interpolation order, to upsampling the map to the photo and
    reading the pixel, without allocating a full-resolution float map.
    """
    from scipy.ndimage import map_coordinates

    nh, nw = net.shape
    ph, pw = photo_hw
    xn = (np.asarray(x, np.float64) + 0.5) * nw / pw - 0.5
    yn = (np.asarray(y, np.float64) + 0.5) * nh / ph - 0.5
    return map_coordinates(net.astype(np.float64), [yn, xn], order=1, mode="nearest")


def sfm_agreement_image(p: np.ndarray, z: np.ndarray,
                        rng: np.random.Generator) -> dict[str, Any] | None:
    """p: predicted relative inverse depth at each point; z: SfM camera-frame depth."""
    p = np.asarray(p, np.float64)
    if p.size < 20:
        return None
    order = rng.permutation(p.size)
    fit, ev = order[: p.size // 2], order[p.size // 2:]
    # Fit in inverse-depth space: a * pred + b ~ 1/z.
    s, t = fit_scale_shift(p[fit], 1.0 / z[fit])
    inv_hat = s * p[ev] + t
    valid = inv_hat > 1e-9
    if valid.sum() < 10:
        return None
    z_hat = 1.0 / inv_hat[valid]
    z_ev = z[ev][valid]
    absrel = float(np.mean(np.abs(z_hat - z_ev) / z_ev))
    ratio = np.maximum(z_hat / z_ev, z_ev / z_hat)
    # Baseline: a flat scene at the fit half's mean inverse depth. Nadir survey imagery
    # has little depth range, so AbsRel alone flatters any model; the baseline and the
    # rank correlation say how much of the agreement the network itself contributes.
    z_flat = 1.0 / float(np.mean(1.0 / z[fit]))
    ratio_flat = np.maximum(z_flat / z_ev, z_ev / z_flat)
    return {"n_fit": int(fit.size), "n_eval": int(valid.sum()),
            "n_eval_dropped_nonpositive": int((~valid).sum()),
            "absrel": absrel, "delta1": float(np.mean(ratio < 1.25)),
            "spearman_rho_eval": spearman(p[ev][valid], 1.0 / z_ev),
            "depth_range_p95_over_p5": float(np.percentile(z_ev, 95) / np.percentile(z_ev, 5)),
            "baseline_flat_absrel": float(np.mean(np.abs(z_flat - z_ev) / z_ev)),
            "baseline_flat_delta1": float(np.mean(ratio_flat < 1.25))}


def run_sfm(model: Path, sparse_dir: Path, image_dir: Path, *, max_images: int | None = None,
            seed: int = 0) -> dict[str, Any]:
    recon = load_sparse(sparse_dir)
    rng = np.random.default_rng(seed)
    rows = []
    images = sorted(recon.images.values(), key=lambda im: im.name)
    if max_images is not None and len(images) > max_images:
        step = len(images) / float(max_images)
        images = [images[int(i * step)] for i in range(max_images)]
    for im in images:
        path = image_dir / im.name
        if not path.is_file():
            continue
        bgr = load_image(path)
        photo_hw = bgr.shape[:2]
        cam = recon.cameras[im.camera_id]
        # Points are in the reconstruction's pixel grid; rescale to the photo's.
        sx, sy = photo_hw[1] / cam.width, photo_hw[0] / cam.height
        x, y, z = sparse_depths(recon, im)
        net = infer_onnx_network(bgr, model)
        del bgr
        if z.size == 0:
            continue
        row = sfm_agreement_image(sample_network_map(net, x * sx, y * sy, photo_hw), z, rng)
        if row is not None:
            rows.append({"image": im.name, **row})
    absrel = np.array([r["absrel"] for r in rows]) if rows else np.array([np.nan])
    d1 = np.array([r["delta1"] for r in rows]) if rows else np.array([np.nan])
    return {
        "measurement": "agreement with SfM sparse depth (scale+shift fitted in inverse depth "
                       "on a random half of each image's points, evaluated on the other half)",
        "sparse_model": str(sparse_dir),
        "n_images": len(rows),
        "n_eval_points": int(sum(r["n_eval"] for r in rows)),
        "absrel_mean": float(np.mean(absrel)), "absrel_median": float(np.median(absrel)),
        "delta1_mean": float(np.mean(d1)), "delta1_median": float(np.median(d1)),
        **{f"{k}_mean": float(np.mean([r[k] for r in rows])) if rows else float("nan")
           for k in ("spearman_rho_eval", "depth_range_p95_over_p5",
                     "baseline_flat_absrel", "baseline_flat_delta1")},
        "seed": seed,
        "per_image": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--fp32", required=True, type=Path)
    p.add_argument("--int8", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("-n", type=int, default=20)
    s = sub.add_parser("sfm")
    s.add_argument("--model", required=True, type=Path)
    s.add_argument("--sparse", required=True, type=Path)
    s.add_argument("--images", required=True, type=Path)
    s.add_argument("--max-images", type=int, default=None)
    for q in (p, s):
        q.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "parity":
        result = run_parity(args.fp32, args.int8, list_images(args.images, args.n))
    else:
        result = run_sfm(args.model, args.sparse, args.images, max_images=args.max_images)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
