"""Interactive segmentation: a click or a box becomes a mask, then a polygon.

Two promptable models, registered as kind ``onnx_prompt_segmentation``:

``interactive_sam21_tiny``
    SAM 2.1 Hiera-Tiny (Meta, Apache-2.0), image mode only. Two graphs: a vision encoder
    that runs once per image, and a prompt decoder that runs once per click.
``interactive_efficientsam_ti``
    EfficientSAM-Ti (Apache-2.0, OpenCV zoo 2025april export). One graph that takes the
    image and up to six prompt points; it is the CPU default.

Neither model knows what a defect is. It masks whatever it is prompted with, and the
mask is a *suggestion* a person accepts, edits or throws away. The score is the model's
own predicted IoU for its mask: uncalibrated, and not a measurement of anything.

Preprocessing follows each publisher exactly, and both publishers resize the image
straight to a 1024 x 1024 square (aspect not kept, no padding): transformers'
Sam2ImageProcessor for SAM 2.1 (bilinear, /255, ImageNet mean/std), and EfficientSAM's
own ``preprocess`` (bilinear to 1024 square; the graph normalises internally). SAM-1's
"longest side 1024 then pad" is *not* what either model was trained with, so it is not
used. Prompt coordinates are mapped through the same resize: x * 1024 / W, y * 1024 / H.
tests/test_interactive_segmentation.py checks the whole path against the PyTorch
reference.

Coordinates. Prompts are pixel indices in the image as given (x right, y down). The
returned polygon traces the mask along pixel *edges* in the same frame (pixel (c, r)
covers [c, c+1] x [r, r+1]), so its shoelace area equals the mask's pixel count exactly.
Rasters are measured with ``raster.xy(row, col)``, which treats integers as pixel
*centres*; subtract 0.5 from each vertex before handing the polygon to that path
(``polygon_for_raster_measurement``).

Refusals. A model that is not installed, or whose files do not hash to the registry's
sha256, raises ModelRunRefused with the reason. The other model is never substituted
silently: the result always names the model that ran.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import threading
from typing import Any, Sequence

import cv2
import numpy as np

PROMPT_SEGMENTATION_KIND = "onnx_prompt_segmentation"
SAM21_KEY = "interactive_sam21_tiny"
EFFICIENTSAM_KEY = "interactive_efficientsam_ti"
INTERACTIVE_KEYS = (SAM21_KEY, EFFICIENTSAM_KEY)

SIDE = 1024
EFFICIENTSAM_MAX_POINTS = 6
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SCORE_NOTE = ("model's own predicted IoU for its mask; uncalibrated, not a measurement. "
              "The mask is a suggestion a person confirms.")


def _refused(message: str) -> Exception:
    from .model_runner import ModelRunRefused

    return ModelRunRefused(message)


# ── Model files and their digests ────────────────────────────────────────────

_DIGESTS: dict[tuple[str, int, float], str] = {}


def file_sha256(path: Path) -> str:
    """Plain byte sha256, cached against (path, size, mtime)."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    digest = _DIGESTS.get(key)
    if digest is None:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()
        _DIGESTS[key] = digest
    return digest


def _resolve_relative(relative: str) -> Path | None:
    from .models import extra_weights_root, models_root

    rel = Path(relative)
    if rel.is_absolute():
        return rel if rel.exists() else None
    local = models_root() / rel
    if local.exists():
        return local.resolve()
    extra = extra_weights_root()
    if extra is not None and (extra / rel).exists():
        return (extra / rel).resolve()
    return None


@dataclass(frozen=True)
class PromptModel:
    """An installed, verified promptable segmenter."""

    key: str
    graph: Path
    graph_sha256: str
    companions: dict[str, Path]
    companion_sha256: dict[str, str]
    scope: str

    @classmethod
    def from_files(cls, key: str, graph: str | Path, decoder: str | Path | None = None,
                   scope: str = "") -> "PromptModel":
        """A model built from explicit files, for the measurement tool (int8 variants).

        Not used by the app, which only runs what the registry verifies.
        """
        graph = Path(graph)
        companions = {"mask_decoder": Path(decoder)} if decoder else {}
        return cls(key, graph, file_sha256(graph), companions,
                   {str(p): file_sha256(p) for p in companions.values()}, scope)

    @property
    def model_used(self) -> str:
        names = [self.graph.name] + [p.name for p in self.companions.values()]
        return "onnx:" + "+".join(names)


def verified_model(model_key: str) -> PromptModel:
    """The model's files, each checked against the registry's sha256, or a refusal.

    The primary graph is checked through core.models.verify_model_identity. Companion
    graphs (the SAM 2.1 decoder) are hashed here against ``companion_sha256`` in the
    registry row, because a decoder swapped under a verified encoder is still a
    different model.
    """
    from .models import get_model_spec, load_registry, verify_model_identity

    spec = get_model_spec(model_key)
    if spec is None:
        raise _refused(f"{model_key!r} is not in the model registry.")
    if spec.kind != PROMPT_SEGMENTATION_KIND:
        raise _refused(f"{model_key} is a {spec.kind} model, not a promptable segmenter.")
    row = dict((load_registry().get("models") or {}).get(model_key) or {})
    note = spec.status_note or "Its weights are not installed on this machine."
    check = verify_model_identity(model_key)
    status = check.get("status")
    if status in ("missing", "awaiting_weights"):
        raise _refused(f"{model_key} cannot run: {note}")
    if status != "verified":
        raise _refused(f"{model_key} refused: {check.get('detail', status)}")
    graph = Path(str(check["model_path"]))

    expected = {str(k): str(v).lower() for k, v in (row.get("companion_sha256") or {}).items()}
    companions: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for relative in row.get("companions") or []:
        relative = str(relative)
        if not relative.lower().endswith(".onnx"):
            continue  # processor JSON and similar travel with the model but do not run
        path = _resolve_relative(relative)
        if path is None:
            raise _refused(f"{model_key} cannot run: its companion graph {relative} is "
                           f"missing. {note}")
        want = expected.get(relative, "")
        if not want:
            raise _refused(f"{model_key} refused: the registry records no sha256 for its "
                           f"companion {relative}, so it cannot be shown to be the "
                           "measured file.")
        got = file_sha256(path)
        if got != want:
            raise _refused(f"{model_key} refused: {relative} on disk is not the registered "
                           f"file (recorded {want[:16]}..., found {got[:16]}...).")
        # Role from the file name: "<key>__mask_decoder.onnx" -> "mask_decoder".
        companions[Path(relative).stem.split("__")[-1]] = path
        digests[relative] = got
    if model_key == SAM21_KEY and "mask_decoder" not in companions:
        raise _refused(f"{model_key} refused: the registry row names no mask_decoder graph.")
    return PromptModel(model_key, graph, str(check["actual_sha256"]), companions, digests,
                       str(row.get("description") or spec.description or ""))


# ── Model choice ──────────────────────────────────────────────────────────────

_CUDA_USABLE: bool | None = None


def cuda_available() -> bool:
    """Whether onnxruntime can actually run on the CUDA execution provider here.

    Listing CUDAExecutionProvider is not enough: onnxruntime-gpu lists it even when the
    CUDA or cuDNN libraries it needs are missing, and then quietly runs on the CPU. A
    one-node probe graph is opened on CUDA and the providers it really got are read.
    """
    global _CUDA_USABLE
    if _CUDA_USABLE is not None:
        return _CUDA_USABLE
    usable = False
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            from onnx import TensorProto, helper

            node = helper.make_node("Identity", ["x"], ["y"])
            graph = helper.make_graph(
                [node], "probe",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
            probe = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            probe.ir_version = 8
            options = ort.SessionOptions()
            options.log_severity_level = 4
            ort.set_default_logger_severity(4)  # the probe's failure is the answer, not noise
            try:
                sess = ort.InferenceSession(probe.SerializeToString(), sess_options=options,
                                            providers=["CUDAExecutionProvider"])
                usable = "CUDAExecutionProvider" in sess.get_providers()
            finally:
                ort.set_default_logger_severity(2)
    except Exception:
        usable = False
    _CUDA_USABLE = usable
    return usable


def default_model_key() -> str:
    """SAM 2.1-T when the CUDA execution provider is present, else EfficientSAM-Ti.

    This is a default, not a fallback: whichever key it names must be installed, and a
    missing one is refused rather than replaced by the other.
    """
    return SAM21_KEY if cuda_available() else EFFICIENTSAM_KEY


def choose_model_key(model_key: str = "") -> str:
    key = str(model_key or "").strip()
    return key or default_model_key()


_SESSIONS: dict[tuple, Any] = {}
_SESSION_LOCK = threading.Lock()

#: The EfficientSAM-Ti (OpenCV 2025april) graph is one model, image and prompts in. Its
#: image encoder ends at this tensor (the neck's last LayerNorm2d, [1, 256, 64, 64]); the
#: graph is cut there in memory so the encoder runs once per image and only the prompt
#: decoder runs per click. The cut graphs are the same nodes and weights, and
#: tests/test_interactive_segmentation.py checks they reproduce the whole graph.
EFFICIENTSAM_EMBEDDING = "/image_encoder/neck/neck.3/Add_1_output_0"


def _providers(prefer_cuda: bool) -> list[str]:
    if prefer_cuda and cuda_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _open_session(source: Any, name: str, providers: list[str]) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        return ort.InferenceSession(source, sess_options=options, providers=providers)
    except Exception as exc:
        raise _refused(f"{name} would not load in onnxruntime: {exc}") from exc


def _session(path: Path, prefer_cuda: bool) -> Any:
    providers = _providers(prefer_cuda)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime, providers[0])
    with _SESSION_LOCK:
        sess = _SESSIONS.get(key)
        if sess is None:
            sess = _open_session(str(path), path.name, providers)
            _SESSIONS[key] = sess
    return sess


def _efficientsam_sessions(path: Path, prefer_cuda: bool) -> tuple[Any, Any] | None:
    """(encoder, decoder) sessions cut from the single graph, or None if it cannot be cut."""
    providers = _providers(prefer_cuda)
    stat = path.stat()
    key = ("split", str(path), stat.st_size, stat.st_mtime, providers[0])
    with _SESSION_LOCK:
        pair = _SESSIONS.get(key)
        if pair is None:
            try:
                import onnx
                from onnx.utils import Extractor

                model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
                extractor = Extractor(model)
                inputs = [i.name for i in model.graph.input]
                outputs = [o.name for o in model.graph.output]
                enc = extractor.extract_model([inputs[0]], [EFFICIENTSAM_EMBEDDING])
                dec = extractor.extract_model([EFFICIENTSAM_EMBEDDING, inputs[1], inputs[2]],
                                              outputs[:2])
                del model, extractor
                pair = (_open_session(enc.SerializeToString(), path.name + " (encoder)", providers),
                        _open_session(dec.SerializeToString(), path.name + " (decoder)", providers))
            except Exception:
                pair = (None, None)
            _SESSIONS[key] = pair
    return None if pair[0] is None else pair


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _check_image(image_bgr: np.ndarray) -> None:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR uint8 image")
    if image_bgr.shape[0] < 2 or image_bgr.shape[1] < 2:
        raise ValueError("the image is too small to segment")


def _square_rgb01(image_bgr: np.ndarray, antialias: bool) -> np.ndarray:
    """RGB float32 in [0, 1], resized straight to 1024 x 1024, bilinear.

    ``antialias`` follows the publisher. SAM 2.1's processor resizes with an
    antialiased bilinear filter, for which INTER_AREA stands in when shrinking.
    EfficientSAM's own ``preprocess`` (and OpenCV's demo) use plain bilinear
    (F.interpolate / cv.resize), which INTER_LINEAR reproduces.
    """
    _check_image(image_bgr)
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    shrink = h > SIDE or w > SIDE
    method = cv2.INTER_AREA if (antialias and shrink) else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (SIDE, SIDE), interpolation=method)
    return resized.astype(np.float32) / 255.0


def preprocess_sam21(image_bgr: np.ndarray) -> np.ndarray:
    x = (_square_rgb01(image_bgr, antialias=True) - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def preprocess_efficientsam(image_bgr: np.ndarray) -> np.ndarray:
    # The 2025april graph normalises internally (EfficientSAM.preprocess); it takes [0, 1].
    x = _square_rgb01(image_bgr, antialias=False)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def to_model_space(xy: Sequence[float], hw: tuple[int, int]) -> tuple[float, float]:
    h, w = hw
    return float(xy[0]) * SIDE / float(w), float(xy[1]) * SIDE / float(h)


# ── Encoding, with a cache ────────────────────────────────────────────────────

@dataclass
class Embedding:
    model_key: str
    model_sha256: str
    image_sha256: str
    image_hw: tuple[int, int]
    arrays: tuple[np.ndarray, ...]
    model_used: str
    #: True when the encoder ran for this call; False when the cache answered.
    computed: bool = True


_CACHE: "OrderedDict[tuple[str, str], Embedding]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
CACHE_SIZE = 6


def image_sha256(image_bgr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image_bgr.shape).encode("ascii"))
    h.update(np.ascontiguousarray(image_bgr).tobytes())
    return h.hexdigest()


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def encode(image_bgr: np.ndarray, model_key: str = "", *,
           model: PromptModel | None = None) -> Embedding:
    """Run the image half of the model once; later prompts on this image reuse it.

    Cached per (image sha256, model sha256). For SAM 2.1 that is the vision encoder's
    three feature maps; for EfficientSAM, the encoder half of its graph. ``model`` lets
    the measurement tool pass a variant (int8) explicitly; the app always goes through
    the registry.
    """
    if model is None:
        model = verified_model(choose_model_key(model_key))
    key = model.key
    _check_image(image_bgr)
    img_sha = image_sha256(image_bgr)
    cache_key = (img_sha, model.graph_sha256)
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
        if hit is not None:
            _CACHE.move_to_end(cache_key)
            return Embedding(hit.model_key, hit.model_sha256, hit.image_sha256, hit.image_hw,
                             hit.arrays, hit.model_used, computed=False)
    hw = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
    if key == SAM21_KEY:
        sess = _session(model.graph, prefer_cuda=True)
        outs = sess.run(None, {sess.get_inputs()[0].name: preprocess_sam21(image_bgr)})
        arrays = tuple(np.asarray(o, dtype=np.float32) for o in outs[:3])
    else:
        blob = preprocess_efficientsam(image_bgr)
        pair = _efficientsam_sessions(model.graph, prefer_cuda=True)
        if pair is None:
            arrays = (blob,)  # the whole graph will run per prompt
        else:
            enc = pair[0]
            arrays = (blob, np.asarray(enc.run(None, {enc.get_inputs()[0].name: blob})[0],
                                       dtype=np.float32))
    emb = Embedding(key, model.graph_sha256, img_sha, hw, arrays, model.model_used)
    with _CACHE_LOCK:
        _CACHE[cache_key] = emb
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return emb


# ── Prompting ─────────────────────────────────────────────────────────────────

@dataclass
class MaskResult:
    mask: np.ndarray                 # bool, image_hw
    score: float                     # predicted IoU of the chosen mask (uncalibrated)
    polygon_px: list[list[float]]    # closed ring on pixel edges, image frame
    area_px: int                     # pixels inside polygon_px (== its shoelace area)
    model_key: str
    model_sha256: str
    model_used: str
    candidate_scores: list[float] = field(default_factory=list)
    object_score_logit: float | None = None
    components_dropped: int = 0
    holes_filled_px: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "polygon_px": self.polygon_px,
            "area_px": self.area_px,
            "score": round(float(self.score), 4),
            "score_kind": "predicted_iou",
            "score_note": SCORE_NOTE,
            "candidate_scores": [round(float(s), 4) for s in self.candidate_scores],
            "object_score_logit": (None if self.object_score_logit is None
                                   else round(float(self.object_score_logit), 4)),
            "components_dropped": self.components_dropped,
            "holes_filled_px": self.holes_filled_px,
            "model_key": self.model_key,
            "model_sha256": self.model_sha256,
            "model_used": self.model_used,
            "output_semantics": "mask",
            "status": "suggestion",
        }


def _normalise_prompts(points: Sequence[Sequence[float]] | None,
                       box: Sequence[float] | None,
                       hw: tuple[int, int]) -> tuple[list[tuple[float, float, int]],
                                                     tuple[float, float, float, float] | None]:
    h, w = hw
    pts: list[tuple[float, float, int]] = []
    for p in points or []:
        if len(p) < 2:
            raise ValueError("each point is (x, y) or (x, y, label)")
        label = int(p[2]) if len(p) >= 3 else 1
        if label not in (0, 1):
            raise ValueError("a point label is 1 (include) or 0 (exclude)")
        x, y = float(p[0]), float(p[1])
        if not (np.isfinite(x) and np.isfinite(y)) or not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"point ({x:g}, {y:g}) lies outside the image ({w} x {h}).")
        pts.append((x, y, label))
    bx = None
    if box is not None and len(box) > 0:
        if len(box) != 4:
            raise ValueError("a box is [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ValueError("the box has no area inside the image.")
        bx = (x0, y0, x1, y1)
    if bx is None and not any(label == 1 for *_, label in pts):
        raise ValueError("Give at least one include point or a box; exclude points alone "
                         "do not say what to segment.")
    return pts, bx


def _sam21_decode(model: PromptModel, emb: Embedding, pts, bx):
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    for x, y, label in pts:
        coords.append(to_model_space((x, y), emb.image_hw))
        labels.append(label)
    if bx is not None:
        coords.append(to_model_space(bx[:2], emb.image_hw))
        labels.append(2)
        coords.append(to_model_space(bx[2:], emb.image_hw))
        labels.append(3)
    sess = _session(model.companions["mask_decoder"], prefer_cuda=True)
    feeds = {
        "image_embeddings_0": emb.arrays[0],
        "image_embeddings_1": emb.arrays[1],
        "image_embeddings_2": emb.arrays[2],
        "point_coords": np.asarray(coords, dtype=np.float32).reshape(1, 1, -1, 2),
        "point_labels": np.asarray(labels, dtype=np.int64).reshape(1, 1, -1),
    }
    iou, low, obj = sess.run(None, feeds)
    return (np.asarray(iou, dtype=np.float32).reshape(-1),
            np.asarray(low, dtype=np.float32).reshape(-1, low.shape[-2], low.shape[-1]),
            float(np.asarray(obj).reshape(-1)[0]))


def _efficientsam_decode(model: PromptModel, emb: Embedding, pts, bx):
    coords: list[tuple[float, float]] = []
    labels: list[float] = []
    if bx is not None:
        coords.append(to_model_space(bx[:2], emb.image_hw))
        labels.append(2.0)
        coords.append(to_model_space(bx[2:], emb.image_hw))
        labels.append(3.0)
    for x, y, label in pts:
        coords.append(to_model_space((x, y), emb.image_hw))
        labels.append(float(label))
    if len(coords) > EFFICIENTSAM_MAX_POINTS:
        raise ValueError(
            f"EfficientSAM-Ti takes at most {EFFICIENTSAM_MAX_POINTS} prompt points (a box "
            f"counts as two); {len(coords)} were given. Clear some points, or choose "
            f"{SAM21_KEY}.")
    while len(coords) < EFFICIENTSAM_MAX_POINTS:
        coords.append((-1.0, -1.0))   # EfficientSAM's own padding: (-1, -1), label -1
        labels.append(-1.0)
    point_coords = np.asarray(coords, dtype=np.float32).reshape(1, 1, -1, 2)
    point_labels = np.asarray(labels, dtype=np.float32).reshape(1, 1, -1)
    pair = _efficientsam_sessions(model.graph, prefer_cuda=True) if len(emb.arrays) > 1 else None
    if pair is not None:
        dec = pair[1]
        names = [i.name for i in dec.get_inputs()]
        masks, iou = dec.run(None, {names[0]: emb.arrays[1], names[1]: point_coords,
                                    names[2]: point_labels})[:2]
    else:
        sess = _session(model.graph, prefer_cuda=True)
        names = [i.name for i in sess.get_inputs()]
        masks, iou = sess.run(None, {names[0]: emb.arrays[0], names[1]: point_coords,
                                     names[2]: point_labels})[:2]
    return (np.asarray(iou, dtype=np.float32).reshape(-1),
            np.asarray(masks, dtype=np.float32).reshape(-1, masks.shape[-2], masks.shape[-1]),
            None)


def logits_to_mask(logits: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample of one logit map to the image, threshold at 0 (both publishers)."""
    h, w = hw
    up = cv2.resize(logits.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return up > 0.0


def predict(embedding: Embedding,
            points: Sequence[Sequence[float]] | None = None,
            box: Sequence[float] | None = None, *,
            model: PromptModel | None = None) -> MaskResult:
    """Prompt the model: points as (x, y, 1|0), and/or a box [x0, y0, x1, y1].

    Always asks for the three candidate masks and keeps the one with the highest
    predicted IoU, as the publishers' demos do. The polygon is the largest connected
    region of that mask with its holes filled; how much was dropped or filled is
    reported, not hidden.
    """
    if model is None:
        model = verified_model(embedding.model_key)
    if model.graph_sha256 != embedding.model_sha256:
        raise _refused(f"{embedding.model_key} changed on disk since this image was encoded; "
                       "encode it again.")
    pts, bx = _normalise_prompts(points, box, embedding.image_hw)
    if embedding.model_key == SAM21_KEY:
        scores, logits, obj = _sam21_decode(model, embedding, pts, bx)
    else:
        scores, logits, obj = _efficientsam_decode(model, embedding, pts, bx)
    best = int(np.argmax(scores))
    mask = logits_to_mask(logits[best], embedding.image_hw)
    polygon, area, dropped, filled = mask_to_polygon(mask)
    return MaskResult(mask=mask, score=float(scores[best]), polygon_px=polygon, area_px=area,
                      model_key=embedding.model_key, model_sha256=embedding.model_sha256,
                      model_used=model.model_used,
                      candidate_scores=[float(s) for s in scores],
                      object_score_logit=obj, components_dropped=dropped,
                      holes_filled_px=filled)


def segment(image_bgr: np.ndarray, points=None, box=None, model_key: str = "") -> MaskResult:
    """encode() then predict(), for one-shot callers."""
    return predict(encode(image_bgr, model_key), points, box)


# ── Mask to polygon ───────────────────────────────────────────────────────────

def _filled_crop(mask: np.ndarray) -> tuple[np.ndarray, int, int, int, int] | None:
    """Largest 8-connected region with holes filled, on its bounding box plus a 1-px margin.

    Returns (crop, x_offset, y_offset, regions_dropped, hole_pixels_filled), where crop
    pixel (c, r) is image pixel (c + x_offset, r + y_offset); None for an empty mask.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = int(stats[biggest, cv2.CC_STAT_LEFT]), int(stats[biggest, cv2.CC_STAT_TOP])
    w, h = int(stats[biggest, cv2.CC_STAT_WIDTH]), int(stats[biggest, cv2.CC_STAT_HEIGHT])
    region = np.zeros((h + 2, w + 2), dtype=bool)
    region[1:-1, 1:-1] = labels[y:y + h, x:x + w] == biggest
    # Holes: background that the one-pixel margin cannot reach (4-connected).
    inv = (~region).astype(np.uint8)
    _, bg_labels = cv2.connectedComponents(inv, connectivity=4)
    holes = inv.astype(bool) & (bg_labels != bg_labels[0, 0])
    return region | holes, x - 1, y - 1, int(count - 2), int(holes.sum())


def filled_region(mask: np.ndarray) -> np.ndarray:
    """The mask the polygon describes, at full size: largest region, holes filled."""
    out = np.zeros(np.asarray(mask).shape[:2], dtype=bool)
    got = _filled_crop(mask)
    if got is None:
        return out
    crop, x0, y0, _, _ = got
    h, w = out.shape
    ys, xs = slice(max(0, y0), min(h, y0 + crop.shape[0])), slice(max(0, x0), min(w, x0 + crop.shape[1]))
    out[ys, xs] = crop[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return out


def mask_to_polygon(mask: np.ndarray) -> tuple[list[list[float]], int, int, int]:
    """Largest 8-connected region, holes filled, traced along pixel edges.

    Returns (closed ring [[x, y], ...], pixel count of the filled region, number of
    other regions dropped, hole pixels filled). The ring's shoelace area equals the
    pixel count, because every vertex is a pixel corner. The filling and tracing run on
    the region's bounding box, not the whole photo.
    """
    got = _filled_crop(mask)
    if got is None:
        return [], 0, 0, 0
    filled, x0, y0, dropped, holes = got
    ring = [[px + x0, py + y0] for px, py in _trace_outline(filled)]
    return ring, int(filled.sum()), dropped, holes


def _trace_outline(region: np.ndarray) -> list[list[float]]:
    """Exterior ring of one hole-free 8-connected region, on pixel corners."""
    from rasterio import features
    from rasterio.transform import Affine

    shapes = [geom for geom, value in features.shapes(
        region.astype(np.uint8), mask=region, connectivity=8, transform=Affine.identity())
        if value == 1]
    if not shapes:
        return []
    # One region normally yields one shape; if diagonal-only contacts split it, keep the
    # largest ring (its area is checked by the caller's tests against the pixel count).
    def ring_area(ring):
        xs = np.array([p[0] for p in ring])
        ys = np.array([p[1] for p in ring])
        return abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))) / 2.0

    rings = [s["coordinates"][0] for s in shapes]
    ring = max(rings, key=ring_area)
    ring = _drop_collinear([[float(x), float(y)] for x, y in ring])
    if ring and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring


def _drop_collinear(ring: list[list[float]]) -> list[list[float]]:
    """Remove vertices in the middle of straight runs; the area is unchanged."""
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 4:
        return ring
    out: list[list[float]] = []
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            out.append(b)
    return out


def shoelace(ring: Sequence[Sequence[float]]) -> float:
    pts = [tuple(p[:2]) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    return abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))) / 2.0


def polygon_for_raster_measurement(ring: Sequence[Sequence[float]]) -> list[list[float]]:
    """Edge-convention vertices to the pixel-centre convention raster.xy() expects."""
    pts = [[float(p[0]) - 0.5, float(p[1]) - 0.5] for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def mask_png_data_uri(mask: np.ndarray, rgba: tuple[int, int, int, int] = (0, 200, 255, 110)) -> str:
    """The mask as a transparent PNG overlay (data URI) for the viewer."""
    import base64

    m = np.asarray(mask, dtype=bool)
    out = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    out[m] = (rgba[2], rgba[1], rgba[0], rgba[3])  # cv2 writes BGRA
    okay, buf = cv2.imencode(".png", out)
    if not okay:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ── Orthomosaics: a window around the prompt ──────────────────────────────────

RASTER_SUFFIXES = {".tif", ".tiff", ".vrt", ".img"}
WINDOW_GRID = 512
MIN_WINDOW = 1024
MAX_WINDOW = 4096


@dataclass
class RasterWindow:
    image_bgr: np.ndarray
    col_off: int
    row_off: int
    crs_wkt: str
    epsg: int | None
    georeferenced: bool
    nodata_fraction: float


def prompt_window(points, box, raster_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """(col_off, row_off, width, height) of the window to read around the prompts.

    Snapped to a 512-px grid, so further clicks on the same object land in the same
    window and reuse the cached embedding. At least 1024 px a side (native GSD), or 1.5x
    the prompts' extent, capped at 4096 px; the model sees it at 1024 either way.
    """
    h, w = raster_hw
    xs: list[float] = [float(p[0]) for p in points or []]
    ys: list[float] = [float(p[1]) for p in points or []]
    if box is not None and len(box) == 4:
        xs += [float(box[0]), float(box[2])]
        ys += [float(box[1]), float(box[3])]
    if not xs:
        raise ValueError("no prompt to place the window around")
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    side = int(np.ceil(max(MIN_WINDOW, 1.5 * extent) / WINDOW_GRID) * WINDOW_GRID)
    side = min(side, MAX_WINDOW)
    if extent > side:
        raise ValueError(f"the prompts span {extent:.0f} px, more than the {MAX_WINDOW}-px "
                         "window one prompt can cover; prompt a smaller object.")
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    col = int(round((cx - side / 2) / WINDOW_GRID) * WINDOW_GRID)
    row = int(round((cy - side / 2) / WINDOW_GRID) * WINDOW_GRID)
    # Shift the snapped window so every prompt is inside it, then clamp to the raster.
    col = min(col, int(np.floor(min(xs))))
    row = min(row, int(np.floor(min(ys))))
    col = max(col, int(np.ceil(max(xs))) + 1 - side)
    row = max(row, int(np.ceil(max(ys))) + 1 - side)
    col = max(0, min(col, max(0, w - side)))
    row = max(0, min(row, max(0, h - side)))
    return col, row, min(side, w - col), min(side, h - row)


def read_raster_window(path: str | Path, points, box) -> RasterWindow:
    """RGB window of an orthomosaic around the prompts, as 8-bit BGR."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        col, row, width, height = prompt_window(points, box, (src.height, src.width))
        if src.count < 3:
            raise ValueError(f"{Path(path).name} has {src.count} band(s); segmentation "
                             "needs an RGB orthomosaic.")
        win = Window(col, row, width, height)
        data = src.read([1, 2, 3], window=win)
        valid = src.read_masks(1, window=win) > 0
        if data.dtype != np.uint8:
            data = _to_uint8(data, valid)
        rgb = np.transpose(data, (1, 2, 0))
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        crs = src.crs
        return RasterWindow(bgr, col, row, crs.to_wkt() if crs else "",
                            crs.to_epsg() if crs else None, crs is not None,
                            float(1.0 - valid.mean()) if valid.size else 0.0)


def _to_uint8(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Stretch a non-8-bit RGB window to 8 bits (2-98 percentile over valid pixels)."""
    out = np.zeros(data.shape, dtype=np.uint8)
    for b in range(data.shape[0]):
        band = data[b].astype(np.float64)
        vals = band[valid] if valid.any() else band.reshape(-1)
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        out[b] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def is_raster_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RASTER_SUFFIXES


def segment_source(source_path: str | Path, points=None, box=None,
                   model_key: str = "") -> dict[str, Any]:
    """Segment a photo, or a window of an orthomosaic, from prompts in source pixels.

    Points and box are in the source's own pixel frame (for a raster: column, row of
    the full raster). The polygon comes back in the same frame. For a raster with a CRS
    the polygon is also given in that CRS and in WGS84 lon/lat.
    """
    from .detection import load_image

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"No such image: {source}")
    key = choose_model_key(model_key)
    points = [list(p) for p in (points or [])]
    box = list(box) if box is not None and len(box) else None
    result: dict[str, Any] = {"source_path": str(source), "model_key": key,
                              "default_model_key": default_model_key(),
                              "model_choice": "user" if str(model_key or "").strip() else "default"}
    if is_raster_path(source):
        win = read_raster_window(source, points, box)
        local_pts = [[p[0] - win.col_off, p[1] - win.row_off, *p[2:]] for p in points]
        local_box = ([box[0] - win.col_off, box[1] - win.row_off,
                      box[2] - win.col_off, box[3] - win.row_off] if box else None)
        emb = encode(win.image_bgr, key)
        res = predict(emb, local_pts, local_box)
        polygon = [[x + win.col_off, y + win.row_off] for x, y in res.polygon_px]
        result.update(res.summary())
        result.update(polygon_px=polygon, window_px=[win.col_off, win.row_off,
                                                     win.image_bgr.shape[1], win.image_bgr.shape[0]],
                      mask_origin_px=[win.col_off, win.row_off],
                      mask_hw=list(res.mask.shape), georeferenced=win.georeferenced,
                      crs_epsg=win.epsg, window_nodata_fraction=round(win.nodata_fraction, 4),
                      embedding_cached=not emb.computed)
        if win.georeferenced and polygon:
            result.update(_georeference_polygon(source, polygon))
        result["_mask"] = res.mask
        return result
    image = load_image(source)
    emb = encode(image, key)
    res = predict(emb, points, box)
    result.update(res.summary())
    result.update(window_px=[0, 0, image.shape[1], image.shape[0]], mask_origin_px=[0, 0],
                  mask_hw=list(res.mask.shape), georeferenced=False, crs_epsg=None,
                  embedding_cached=not emb.computed)
    result["_mask"] = res.mask
    return result


def _georeference_polygon(raster: str | Path, polygon_px: list[list[float]]) -> dict[str, Any]:
    """The edge-convention polygon in the raster's CRS and in WGS84."""
    import rasterio
    from rasterio.warp import transform as rio_transform

    with rasterio.open(raster) as src:
        t = src.transform
        crs = src.crs
        world = [list(t * (float(x), float(y))) for x, y in polygon_px]
    out: dict[str, Any] = {"polygon_crs": [[round(a, 3), round(b, 3)] for a, b in world]}
    try:
        lon, lat = rio_transform(crs, "EPSG:4326", [p[0] for p in world], [p[1] for p in world])
        out["polygon_lonlat"] = [[float(a), float(b)] for a, b in zip(lon, lat)]
    except Exception:  # pragma: no cover - CRS without a WGS84 path
        out["polygon_lonlat"] = []
    return out
