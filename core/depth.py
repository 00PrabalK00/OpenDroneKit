"""Relative depth from one photograph (Depth Anything V2 Small), unitless by construction.

What this is: a per-pixel *relative inverse depth* map. Larger values are nearer to the
camera, smaller values farther, and the numbers have no unit. A monocular network of
this kind is trained with a scale- and shift-invariant loss; its output is only defined
up to an unknown affine transform, which differs from image to image. Two pixels can be
ordered ("this one is nearer"); no pixel has a distance.

What this is not, and must never become: metres, clearance to a conductor, crack depth,
bulge height, stockpile volume. Turning the output into any of those needs the scale and
shift for this exact image, which only an external measurement (photogrammetry, LiDAR, a
known target) can supply. No function in this module accepts or returns such a quantity,
and tests/test_relative_depth.py pins that no result field names a metric unit.

Use it to read structure: occlusion order, a bulge or sag that the shading hides, which
member sits in front of which. Everything it draws is labelled "nearer / farther
(relative)".

The model refuses rather than guesses: missing weights, or weights whose sha256 is not
the one the registry recorded, raise ModelRunRefused with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np

RELATIVE_DEPTH_KIND = "onnx_relative_depth"
DEFAULT_DEPTH_MODEL = "relative_depth_dav2_small"
UNITS = "relative (unitless)"
SEMANTICS = ("relative inverse depth: larger values are nearer to the camera; defined only "
             "up to an unknown per-image scale and shift, so values are not distances")
COLORBAR_LABELS = {"high": "nearer (relative)", "low": "farther (relative)",
                   "title": "nearer / farther (relative)"}

PATCH = 14
SHORT_SIDE = 518
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _refused(message: str) -> Exception:
    from .model_runner import ModelRunRefused

    return ModelRunRefused(message)


def network_size(height: int, width: int, short_side: int = SHORT_SIDE) -> tuple[int, int]:
    """The (height, width) the network sees: short side ~518, aspect kept, both x14.

    This follows the publisher's DPT preprocessing (keep_aspect_ratio, ensure_multiple_of
    14, size 518): scale so the short side is 518, then round each side to the nearest
    multiple of 14, never below 14.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image has no pixels")
    scale = float(short_side) / float(min(height, width))

    def snap(value: float) -> int:
        return max(PATCH, int(round(value / PATCH)) * PATCH)

    return snap(height * scale), snap(width * scale)


def preprocess(image_bgr: np.ndarray, short_side: int = SHORT_SIDE) -> np.ndarray:
    """BGR uint8 image -> float32 NCHW blob, RGB, ImageNet-normalised, sides x14."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    h, w = image_bgr.shape[:2]
    nh, nw = network_size(h, w, short_side)
    # Resize first, then convert: a 20 MP photo as float32 is ~240 MB, the network
    # input ~3 MB. Bicubic on uint8 differs from bicubic on float only by rounding to
    # one grey level, which the network does not resolve.
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(normed.transpose(2, 0, 1)[None], dtype=np.float32)


_SESSIONS: dict[tuple[str, int, float], Any] = {}
_LOCK = threading.Lock()


def _session(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment problem
        raise _refused("onnxruntime is required for relative depth and is not installed.") from exc
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    with _LOCK:
        sess = _SESSIONS.get(key)
        if sess is None:
            options = ort.SessionOptions()
            threads = int(os.environ.get("ODK_DEPTH_THREADS", "0") or 0)
            if threads > 0:
                options.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(path), sess_options=options,
                                        providers=["CPUExecutionProvider"])
            _SESSIONS[key] = sess
    return sess


def infer_onnx(image_bgr: np.ndarray, onnx_path: str | Path) -> np.ndarray:
    """Raw relative inverse depth at the image's own size, from a given ONNX file.

    No identity check: this is the measurement primitive (parity, SfM agreement) and
    takes an explicit file. Application code goes through relative_depth(), which
    verifies the file against the registry first.
    """
    h, w = image_bgr.shape[:2]
    out = infer_onnx_network(image_bgr, onnx_path)
    # Bicubic back to the photo's own pixel grid, as the publisher's infer_image does.
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def infer_onnx_network(image_bgr: np.ndarray, onnx_path: str | Path) -> np.ndarray:
    """Raw relative inverse depth at the network's own resolution (no upsample).

    For measurement on large photos, where a full-resolution float map is wasted memory.
    """
    blob = preprocess(image_bgr)
    sess = _session(Path(onnx_path))
    name = sess.get_inputs()[0].name
    out = np.asarray(sess.run(None, {name: blob})[0], dtype=np.float32)
    return out.reshape(out.shape[-2], out.shape[-1])


def normalise_for_display(inv_depth: np.ndarray) -> np.ndarray:
    """0..1 per image for display only (2nd-98th percentile stretch). 1 = nearest."""
    finite = inv_depth[np.isfinite(inv_depth)]
    if finite.size == 0:
        return np.zeros_like(inv_depth, dtype=np.float32)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi - lo <= 1e-12:
        return np.zeros_like(inv_depth, dtype=np.float32)
    return np.clip((inv_depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def colourise(inv_depth: np.ndarray, *, with_colorbar: bool = True) -> np.ndarray:
    """A BGR picture of the relative depth, with a 'nearer / farther (relative)' bar.

    The bar carries words, never numbers: there is no number on it that means anything.
    """
    norm = normalise_for_display(inv_depth)
    picture = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if not with_colorbar:
        return picture
    h, w = picture.shape[:2]
    bar_w = max(12, w // 40)
    pad = max(60, w // 9)
    canvas = np.full((h, w + bar_w + pad, 3), 255, dtype=np.uint8)
    canvas[:, :w] = picture
    ramp = np.linspace(255, 0, h, dtype=np.float32).astype(np.uint8).reshape(h, 1)
    canvas[:, w + 4:w + 4 + bar_w] = cv2.applyColorMap(np.repeat(ramp, bar_w, axis=1),
                                                       cv2.COLORMAP_INFERNO)
    scale = max(0.35, min(1.2, h / 1400.0))
    thick = max(1, int(round(scale * 2)))
    x = w + bar_w + 8
    for text, y in (("nearer", int(24 * scale) + 4), ("farther", h - 30 * scale - 4),
                    ("(relative)", h - 6)):
        cv2.putText(canvas, text, (x, int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6,
                    (20, 20, 20), thick, cv2.LINE_AA)
    return canvas


@dataclass
class DepthResult:
    """Relative inverse depth for one image. Deliberately has no metric field."""

    inverse_depth: np.ndarray
    overlay: np.ndarray
    model_key: str
    model_sha256: str
    model_used: str
    scope: str
    network_input_hw: tuple[int, int]
    units: str = UNITS
    semantics: str = SEMANTICS
    colorbar: dict[str, str] = field(default_factory=lambda: dict(COLORBAR_LABELS))

    def summary(self) -> dict[str, Any]:
        """JSON-safe description; the array itself is saved separately (.npy)."""
        h, w = self.inverse_depth.shape[:2]
        return {
            "units": self.units,
            "output_semantics": "relative",
            "semantics": self.semantics,
            "colorbar": dict(self.colorbar),
            "image_hw": [int(h), int(w)],
            "network_input_hw": [int(v) for v in self.network_input_hw],
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
            "model_used": self.model_used,
            "scope": self.scope,
        }


def _verified_path(model_key: str) -> tuple[Any, Path, dict[str, Any]]:
    """Installed, the right kind, and byte-identical to the registry's digest; or refuse."""
    from .model_runner import _installed
    from .models import verify_model_identity

    spec, path, row = _installed(model_key)
    if spec.kind != RELATIVE_DEPTH_KIND:
        raise _refused(f"{model_key} is a {spec.kind}, not a relative-depth model.")
    check = verify_model_identity(model_key)
    if check.get("status") != "verified":
        raise _refused(f"{model_key} cannot run: {check.get('detail') or check.get('status')}")
    return spec, path, row


def relative_depth(image_bgr: np.ndarray,
                   model_key: str = DEFAULT_DEPTH_MODEL) -> DepthResult:
    """Relative inverse depth for one BGR image, with a colourised overlay and provenance.

    Raises ModelRunRefused when the model is not registered, not installed, not a
    relative-depth model, or not the file the registry recorded.
    """
    spec, path, row = _verified_path(model_key)
    h, w = image_bgr.shape[:2]
    inv = infer_onnx(image_bgr, path)
    from .models import model_identity

    return DepthResult(
        inverse_depth=inv,
        overlay=colourise(inv),
        model_key=model_key,
        model_sha256=model_identity(model_key)["model_sha256"],
        model_used=f"onnx:{path.name}",
        scope=str(row.get("description") or spec.description or ""),
        network_input_hw=network_size(h, w),
    )


def save_depth(result: DepthResult, directory: str | Path, stem: str) -> dict[str, str]:
    """Write <stem>.npy (float32), <stem>_relative16.png, <stem>_overlay.png, <stem>.json.

    The 16-bit PNG is the per-image display stretch (0 = farthest, 65535 = nearest) and
    is as unitless as the array; the JSON says so next to it.
    """
    import json

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    npy = out / f"{stem}.npy"
    png16 = out / f"{stem}_relative16.png"
    overlay = out / f"{stem}_overlay.png"
    meta = out / f"{stem}.json"
    np.save(npy, result.inverse_depth.astype(np.float32))
    cv2.imwrite(str(png16), (normalise_for_display(result.inverse_depth) * 65535.0)
                .round().astype(np.uint16))
    cv2.imwrite(str(overlay), result.overlay)
    payload = result.summary()
    payload.update({
        "npy": npy.name,
        "png16": png16.name,
        "png16_encoding": "per-image display stretch of relative inverse depth "
                          "(2nd-98th percentile); 0 = farthest, 65535 = nearest; unitless",
        "warning": "Relative values only. Not metres; do not derive clearance, crack "
                   "depth, height or volume from this file.",
    })
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"depth_npy_path": str(npy), "depth_png16_path": str(png16),
            "overlay_path": str(overlay), "metadata_path": str(meta)}
