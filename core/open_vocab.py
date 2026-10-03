"""Text-prompted object search with Grounding DINO Tiny (ONNX), as review suggestions.

A person types what to look for ("insulator. bird nest. rust stain.") and the model
returns boxes. The model has never been trained on any of OpenDroneKit's inspection
classes, so everything it returns is a *suggestion*: it goes to Review, out of every
report, tagged ``suggestion`` and ``zero-shot``, with a score that is not a calibrated
probability. A person confirms or rejects each one.

Pre- and post-processing follow the published pipeline exactly:

* caption: the prompts lower-cased, each ending in ".", joined by spaces ("a. b. c."),
  tokenised by the model's own BERT tokenizer, at most 256 tokens (the model's
  ``max_text_len``). A longer caption is refused, not truncated.
* image: RGB, resized to the graph's fixed 800x800 input (the published ONNX export's
  ``preprocessor_config.json``), rescaled by 1/255, ImageNet-normalised; ``pixel_mask``
  all ones.
* outputs: ``logits`` (1, 900, 256) and ``pred_boxes`` (1, 900, 4, normalised cxcywh),
  post-processed as transformers' ``GroundingDinoProcessor.
  post_process_grounded_object_detection``: sigmoid, max over text tokens > box
  threshold, phrase from the tokens whose probability exceeds the text threshold.

Large frames and orthomosaic tiles are cut into overlapping tiles, each run on its own,
and merged with per-label NMS: one 800x800 resize of a 5000 px frame leaves a panel
a few pixels wide.

There is no fallback. A missing or mismatched file raises ``ModelRunRefused``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model_files import ModelFilesRefused, VerifiedModel, verified_model

KIND = "onnx_open_vocab_detection"
DEFAULT_MODEL_KEY = "open_vocab_grounding_dino_tiny"
MAX_TEXT_LEN = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# A frame whose long side exceeds this is tiled when the caller does not say.
AUTO_TILE_ABOVE = 2000
AUTO_TILE_SIZE = 1024
TILE_OVERLAP = 0.2
NMS_IOU = 0.5
SUGGESTION_NOTE = ("Zero-shot suggestion from {model}: score is uncalibrated; "
                   "confirm or reject.")


def _refused(message: str) -> Exception:
    from .model_runner import ModelRunRefused

    return ModelRunRefused(message)


# --------------------------------------------------------------------------- caption


def normalise_prompts(prompts: Sequence[str] | str) -> list[str]:
    """Lower-cased, de-duplicated phrases without their trailing full stops.

    A single string is split on "." so "solar panel. crack." and ["solar panel",
    "crack"] mean the same thing.
    """
    if isinstance(prompts, str):
        prompts = prompts.split(".")
    out: list[str] = []
    for raw in prompts or []:
        phrase = " ".join(str(raw).lower().replace(".", " ").split())
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def build_caption(prompts: Sequence[str] | str) -> tuple[str, list[str]]:
    """The caption Grounding DINO expects: "a. b. c." -- each phrase ending in a full stop."""
    phrases = normalise_prompts(prompts)
    if not phrases:
        raise _refused("Grounding DINO needs a text prompt; use Suggest (text) and say "
                       "what to look for, e.g. 'insulator. bird nest.'")
    return " ".join(f"{p}." for p in phrases), phrases


@dataclass
class EncodedCaption:
    caption: str
    phrases: list[str]
    input_ids: np.ndarray        # (1, L) int64
    attention_mask: np.ndarray   # (1, L) int64
    token_type_ids: np.ndarray   # (1, L) int64
    # token index range [start, end) of each phrase inside input_ids
    phrase_spans: list[tuple[int, int]]


def encode_caption(tokenizer: Any, prompts: Sequence[str] | str) -> EncodedCaption:
    """Tokenise the caption with the model's own tokenizer, refusing over 256 tokens."""
    caption, phrases = build_caption(prompts)
    enc = tokenizer.encode(caption)
    ids = list(enc.ids)
    if len(ids) > MAX_TEXT_LEN:
        raise _refused(
            f"The prompt is {len(ids)} tokens; Grounding DINO reads at most {MAX_TEXT_LEN}. "
            "Search for fewer phrases at a time.")
    offsets = list(enc.offsets)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for phrase in phrases:
        start_char = caption.index(phrase, cursor)
        end_char = start_char + len(phrase)
        cursor = end_char
        toks = [i for i, (a, b) in enumerate(offsets)
                if b > a and a >= start_char and b <= end_char]
        spans.append((min(toks), max(toks) + 1) if toks else (0, 0))
    n = len(ids)
    return EncodedCaption(
        caption=caption, phrases=phrases,
        input_ids=np.asarray([ids], dtype=np.int64),
        attention_mask=np.asarray([list(enc.attention_mask)], dtype=np.int64),
        token_type_ids=np.zeros((1, n), dtype=np.int64),
        phrase_spans=spans,
    )


def _clean_up_tokenization(text: str) -> str:
    """transformers' ``clean_up_tokenization`` (the BERT tokenizer config enables it)."""
    return (text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
            .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m")
            .replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re"))


# ----------------------------------------------------------------- post-processing


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def post_process_grounded(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    input_ids: Sequence[int],
    *,
    box_threshold: float,
    text_threshold: float,
    image_size_hw: tuple[int, int],
    decode: Any,
) -> list[dict[str, Any]]:
    """transformers' ``post_process_grounded_object_detection`` for one image, in numpy.

    ``logits`` (Q, T), ``pred_boxes`` (Q, 4) normalised cxcywh, ``input_ids`` the caption
    tokens. ``decode`` turns a list of token ids into text the way the tokenizer's
    ``batch_decode`` does. Returns, per kept query: score, xyxy box in pixels, the
    phrase, and the per-token probabilities (for attributing the box to a prompt).
    """
    logits = np.asarray(logits, dtype=np.float32)
    boxes = np.asarray(pred_boxes, dtype=np.float32)
    probs = _sigmoid(logits)                      # (Q, T)
    scores = probs.max(axis=-1)                    # (Q,)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)
    img_h, img_w = image_size_hw
    xyxy = xyxy * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

    keep = scores > box_threshold
    ids = list(input_ids)
    out: list[dict[str, Any]] = []
    for q in np.nonzero(keep)[0]:
        posmap = probs[q] > text_threshold
        posmap[0:1] = False                        # left_idx = 0
        posmap[posmap.shape[-1] - 1:] = False      # right_idx = T - 1
        token_ids = [ids[i] for i in np.nonzero(posmap)[0] if i < len(ids)]
        out.append({
            "score": float(scores[q]),
            "bbox": [float(v) for v in xyxy[q]],
            "phrase": _clean_up_tokenization(decode(token_ids)) if token_ids else "",
            "token_probs": probs[q],
        })
    return out


def _attribute(token_probs: np.ndarray, phrases: list[str],
               spans: list[tuple[int, int]]) -> tuple[str, float]:
    """Which prompt a box answers: the phrase whose own tokens score highest."""
    best, best_p = phrases[0], -1.0
    for phrase, (a, b) in zip(phrases, spans):
        if b <= a:
            continue
        p = float(np.max(token_probs[a:b]))
        if p > best_p:
            best, best_p = phrase, p
    return best, best_p


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices, best first."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    order = np.argsort(-np.asarray(scores))
    x1, y1, x2, y2 = boxes.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = area[i] + area[rest] - inter
        ratio = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[ratio <= iou]
    return keep


def tile_windows(height: int, width: int, tile: int,
                 overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping (x0, y0, x1, y1) windows covering the whole frame."""
    if tile <= 0 or (height <= tile and width <= tile):
        return [(0, 0, width, height)]
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, step))
        s.append(n - tile)
        return sorted(set(s))

    return [(x, y, min(width, x + tile), min(height, y + tile))
            for y in starts(height) for x in starts(width)]


# --------------------------------------------------------------------------- model


class GroundingDinoOnnx:
    """One loaded Grounding DINO graph plus its tokenizer and preprocessing config."""

    def __init__(self, graph: Path, tokenizer_json: Path, preprocessor_json: Path | None = None,
                 *, providers: Sequence[str] | None = None, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(graph), sess_options=options,
            providers=list(providers or ["CPUExecutionProvider"]))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        shape = self.session.get_inputs()[0].shape
        size = {"height": 800, "width": 800}
        if preprocessor_json is not None and Path(preprocessor_json).is_file():
            cfg = json.loads(Path(preprocessor_json).read_text(encoding="utf-8"))
            size = dict(cfg.get("size") or size)
        h = shape[2] if isinstance(shape[2], int) else int(size.get("height", 800))
        w = shape[3] if isinstance(shape[3], int) else int(size.get("width", 800))
        self.input_hw = (int(h), int(w))

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w = self.input_hw
        # PIL bilinear, as the transformers image processor resizes (resample=2).
        resized = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.float32)
        x = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def raw(self, image_bgr: np.ndarray, enc: EncodedCaption) -> tuple[np.ndarray, np.ndarray]:
        pixel_values = self.preprocess(image_bgr)
        h, w = self.input_hw
        feeds = {
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids,
            "token_type_ids": enc.token_type_ids,
            "attention_mask": enc.attention_mask,
            "pixel_mask": np.ones((1, h, w), dtype=np.int64),
        }
        names = {i.name for i in self.session.get_inputs()}
        logits, boxes = self.session.run(["logits", "pred_boxes"],
                                         {k: v for k, v in feeds.items() if k in names})
        return logits[0], boxes[0]

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str] | str, *,
               box_threshold: float = 0.35, text_threshold: float = 0.25,
               tile: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        enc = encode_caption(self.tokenizer, prompts)
        height, width = image_bgr.shape[:2]
        if tile is None:
            tile = AUTO_TILE_SIZE if max(height, width) > AUTO_TILE_ABOVE else 0
        windows = tile_windows(height, width, int(tile or 0))
        found: list[dict[str, Any]] = []
        for x0, y0, x1, y1 in windows:
            crop = image_bgr[y0:y1, x0:x1]
            logits, boxes = self.raw(crop, enc)
            for hit in post_process_grounded(
                    logits, boxes, enc.input_ids[0].tolist(),
                    box_threshold=box_threshold, text_threshold=text_threshold,
                    image_size_hw=crop.shape[:2], decode=self._decode):
                label, _ = _attribute(hit.pop("token_probs"), enc.phrases, enc.phrase_spans)
                bx = hit["bbox"]
                bx = [min(max(bx[0] + x0, 0.0), width), min(max(bx[1] + y0, 0.0), height),
                      min(max(bx[2] + x0, 0.0), width), min(max(bx[3] + y0, 0.0), height)]
                found.append({**hit, "bbox": bx, "label": label})
        if len(windows) > 1 and found:
            kept: list[dict[str, Any]] = []
            for label in sorted({f["label"] for f in found}):
                group = [f for f in found if f["label"] == label]
                idx = nms(np.array([g["bbox"] for g in group]), np.array([g["score"] for g in group]))
                kept.extend(group[i] for i in idx)
            found = kept
        found.sort(key=lambda f: -f["score"])
        meta = {"caption": enc.caption, "phrases": enc.phrases, "tiles": len(windows),
                "tile_size": int(tile or 0), "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold)}
        return found, meta


_SESSIONS: dict[tuple, GroundingDinoOnnx] = {}


def load_verified(model_key: str = DEFAULT_MODEL_KEY) -> tuple[GroundingDinoOnnx, VerifiedModel]:
    """Verify every file against the registry, then load (cached per file set)."""
    try:
        model = verified_model(model_key, kind=KIND)
    except ModelFilesRefused as exc:
        raise _refused(str(exc)) from exc
    for role in ("model", "tokenizer.json"):
        if role not in model.files:
            raise _refused(f"{model_key}: the registry does not publish its {role}.")
    cache_key = tuple(sorted(model.digests.items()))
    runner = _SESSIONS.get(cache_key)
    if runner is None:
        runner = GroundingDinoOnnx(model.files["model"], model.files["tokenizer.json"],
                                   model.files.get("preprocessor_config.json"))
        _SESSIONS.clear()
        _SESSIONS[cache_key] = runner
    return runner, model


def detect_by_text(image_bgr: np.ndarray, prompts: Sequence[str] | str,
                   model_key: str = DEFAULT_MODEL_KEY, box_threshold: float = 0.35,
                   text_threshold: float = 0.25, tile: int | None = None) -> dict[str, Any]:
    """Boxes for each text phrase, as uncalibrated zero-shot suggestions.

    Returns ``detections`` in the shape ``findings_from_detections`` takes (label,
    confidence, bbox_xyxy), plus the model's identity and ``confidence_kind:
    "zero_shot_score"`` so no caller mistakes the score for a detector's.
    """
    build_caption(prompts)  # refuse an empty prompt before touching the weights
    runner, model = load_verified(model_key)
    hits, meta = runner.detect(image_bgr, prompts, box_threshold=box_threshold,
                               text_threshold=text_threshold, tile=tile)
    detections = [{
        "label": h["label"],
        "phrase": h["phrase"],
        "confidence": round(float(h["score"]), 4),
        "bbox_xyxy": [int(round(v)) for v in h["bbox"]],
    } for h in hits]
    return {
        "model_key": model.model_key,
        "model_used": model.model_used,
        "model_sha256": model.model_sha256,
        "detections": detections,
        "detection_count": len(detections),
        "confidence_kind": "zero_shot_score",
        "output_semantics": "suggestion",
        "note": SUGGESTION_NOTE.format(model=model.model_key),
        **meta,
    }
