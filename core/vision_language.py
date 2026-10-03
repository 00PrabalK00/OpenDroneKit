"""Captions, OCR and phrase grounding with Florence-2-base-ft (ONNX), all unverified.

Florence-2 writes text about an image: a caption, the text it can read (nameplates,
tower numbers, panel serials), or boxes for phrases. None of it is a measurement and
none of it has been checked by OpenDroneKit against ground truth -- captions can invent
details. Text is stored as a "machine description, unverified" note; boxes go to Review
as suggestions. Nothing reaches a report until a person accepts it.

The pipeline is the published one, run through four ONNX graphs:

* image: RGB, resized to 768x768 (bicubic), rescaled by 1/255, ImageNet-normalised
  (``preprocessor_config.json``, CLIPImageProcessor) -> ``vision_encoder`` -> 577 image
  tokens;
* prompt: the task token expanded to the processor's task prompt ("<CAPTION>" ->
  "What does the image describe?"), BART-tokenised -> ``embed_tokens``;
* encoder over [image tokens, prompt tokens] -> ``encoder_model``;
* greedy decoding with the merged decoder and its KV cache, from decoder start token 2
  with BOS forced first (``generation_config.json``), ``no_repeat_ngram_size`` 3, at most
  256 new tokens. The publisher's own examples decode with 3 beams; greedy is cheaper on
  a CPU and is what this build does.
* ``<loc_N>`` tokens are 1000 bins per axis, mapped back to pixels at the bin centre,
  exactly as ``Florence2Processor.post_process_generation``.

A missing or mismatched file raises ``ModelRunRefused``. There is no fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model_files import ModelFilesRefused, VerifiedModel, verified_model

KIND = "onnx_vision_language"
DEFAULT_MODEL_KEY = "vision_language_florence2_base_ft"
MAX_NEW_TOKENS = 256
LOC_BINS = 1000

TASK_PROMPTS = {
    "<CAPTION>": "What does the image describe?",
    "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
    "<OD>": "Locate the objects with category name in the image.",
    "<OCR>": "What is the text in the image?",
    "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
}
TASK_PROMPTS_WITH_INPUT = {
    "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
}
TASKS = tuple(TASK_PROMPTS) + tuple(TASK_PROMPTS_WITH_INPUT)
POST_PROCESSING = {
    "<CAPTION>": "pure_text", "<DETAILED_CAPTION>": "pure_text", "<OCR>": "pure_text",
    "<OCR_WITH_REGION>": "ocr", "<OD>": "description_with_bboxes",
    "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
}
TEXT_TASKS = {"<CAPTION>", "<DETAILED_CAPTION>", "<OCR>", "<OCR_WITH_REGION>"}
BOX_TASKS = {"<OD>", "<CAPTION_TO_PHRASE_GROUNDING>"}

# Florence2Processor's phrase-grounding blacklist (post_processor_config).
BANNED_GROUNDING = {
    "it", "I", "me", "mine", "you", "your", "yours", "he", "him", "his", "she", "her",
    "hers", "they", "them", "their", "theirs", "one", "oneself", "we", "us", "our", "ours",
    "its", "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "yourselves", "themselves", "this", "that", "these", "those", "who", "whom", "whose",
    "which", "what", "all", "another", "any", "anybody", "anyone", "anything", "each",
    "everybody", "everyone", "everything", "few", "many", "nobody", "none", "several",
    "some", "somebody", "someone", "something", "each other", "one another",
    "the image", "image", "images", "the", "a", "an", "a group", "other objects", "lots",
    "a set",
}
UNVERIFIED_NOTE = ("Machine description from {model}, unverified: generated text, not a "
                   "measurement; captions can invent details. Accuracy on inspection "
                   "imagery is unmeasured by OpenDroneKit.")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _refused(message: str) -> Exception:
    from .model_runner import ModelRunRefused

    return ModelRunRefused(message)


# ------------------------------------------------------------------- prompts / parsing


def task_prompt(task: str, text: str = "") -> str:
    """The processor's prompt for a task token; refuses unknown tasks and missing text."""
    if task in TASK_PROMPTS:
        return TASK_PROMPTS[task]
    if task in TASK_PROMPTS_WITH_INPUT:
        phrase = str(text or "").strip()
        if not phrase:
            raise _refused(f"{task} needs a text prompt: the phrases to locate.")
        return TASK_PROMPTS_WITH_INPUT[task].format(input=phrase)
    raise _refused(f"Unsupported Florence-2 task {task!r}; use one of {', '.join(TASKS)}.")


def dequantize(bins: Sequence[int], image_size_wh: tuple[int, int]) -> list[float]:
    """<loc_N> bins (x, y, x, y, ...) to pixel coordinates at the bin centre."""
    width, height = image_size_wh
    per_w, per_h = width / LOC_BINS, height / LOC_BINS
    out = []
    for i, b in enumerate(bins):
        out.append((int(b) + 0.5) * (per_w if i % 2 == 0 else per_h))
    return out


_BOX = re.compile(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
_PHRASE_WITH_BOXES = re.compile(r"([^<]+(?:<loc_\d+>){4,})")
_PHRASE_TEXT = re.compile(r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)")
_OCR = re.compile(r"(.+?)" + r"<loc_(\d+)>" * 8)


def _strip_special(text: str) -> str:
    return text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")


def parse_boxes(text: str, image_size_wh: tuple[int, int], *, grounding: bool) -> list[dict[str, Any]]:
    """Phrases with their boxes, as Florence2's description_with_bboxes / phrase_grounding."""
    text = _strip_special(text)
    out: list[dict[str, Any]] = []
    for phrase_text in _PHRASE_WITH_BOXES.findall(text):
        phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
        if not phrase_text:
            continue
        match = _PHRASE_TEXT.search(phrase_text)
        if not match:
            continue
        phrase = match.group().strip()
        if grounding and phrase in BANNED_GROUNDING:
            continue
        boxes = [[int(m.group(j)) for j in range(1, 5)] for m in _BOX.finditer(phrase_text)]
        if not boxes:
            continue
        phrase = phrase.encode("ascii", "ignore").decode("ascii")
        for b in boxes:
            out.append({"label": phrase, "bbox": dequantize(b, image_size_wh)})
    return out


def parse_ocr(text: str, image_size_wh: tuple[int, int]) -> list[dict[str, Any]]:
    """OCR lines with their quadrilaterals (8 bins: x1 y1 ... x4 y4)."""
    text = _strip_special(text)
    out = []
    for content, *quad in _OCR.findall(text):
        out.append({"text": content.strip(), "quad": dequantize([int(q) for q in quad], image_size_wh)})
    return out


def post_process(task: str, generated: str, image_size_wh: tuple[int, int]) -> dict[str, Any]:
    """Text and regions for a task, from the decoded sequence (special tokens kept)."""
    kind = POST_PROCESSING.get(task, "pure_text")
    if kind == "pure_text":
        return {"text": _strip_special(generated).strip(), "regions": []}
    if kind == "ocr":
        lines = parse_ocr(generated, image_size_wh)
        return {"text": "\n".join(line["text"] for line in lines), "regions": lines}
    regions = parse_boxes(generated, image_size_wh, grounding=(kind == "phrase_grounding"))
    return {"text": "; ".join(dict.fromkeys(r["label"] for r in regions)), "regions": regions}


def banned_ngram_tokens(sequence: Sequence[int], n: int) -> set[int]:
    """Tokens that would repeat an n-gram already in ``sequence`` (HF no_repeat_ngram)."""
    if n <= 0 or len(sequence) + 1 < n:
        return set()
    prefix = tuple(sequence[len(sequence) - n + 1:]) if n > 1 else ()
    banned = set()
    for i in range(len(sequence) - n + 1):
        if tuple(sequence[i:i + n - 1]) == prefix:
            banned.add(int(sequence[i + n - 1]))
    return banned


# --------------------------------------------------------------------------- model


class FlorenceOnnx:
    """Florence-2 as four ONNX sessions plus its BART tokenizer."""

    def __init__(self, files: dict[str, Path], *, providers: Sequence[str] | None = None,
                 threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        missing = [r for r in ("vision_encoder", "embed_tokens", "encoder_model",
                               "decoder_model_merged", "tokenizer.json") if r not in files]
        if missing:
            raise _refused(f"Florence-2 is missing {', '.join(missing)}.")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        prov = list(providers or ["CPUExecutionProvider"])

        def session(role: str):
            return ort.InferenceSession(str(files[role]), sess_options=options, providers=prov)

        self.vision = session("vision_encoder")
        self.embed = session("embed_tokens")
        self.encoder = session("encoder_model")
        self.decoder = session("decoder_model_merged")
        self.tokenizer = Tokenizer.from_file(str(files["tokenizer.json"]))
        gen = {}
        if files.get("generation_config.json"):
            gen = json.loads(Path(files["generation_config.json"]).read_text(encoding="utf-8"))
        self.decoder_start = int(gen.get("decoder_start_token_id", 2))
        self.bos = int(gen.get("forced_bos_token_id", gen.get("bos_token_id", 0)))
        self.eos = int(gen.get("eos_token_id", 2))
        self.no_repeat_ngram = int(gen.get("no_repeat_ngram_size", 3))
        size = {"height": 768, "width": 768}
        if files.get("preprocessor_config.json"):
            cfg = json.loads(Path(files["preprocessor_config.json"]).read_text(encoding="utf-8"))
            size = dict(cfg.get("size") or size)
        self.input_hw = (int(size.get("height", 768)), int(size.get("width", 768)))
        self._past_names = [i.name for i in self.decoder.get_inputs() if i.name.startswith("past_key_values.")]
        self._present_names = [o.name for o in self.decoder.get_outputs() if o.name.startswith("present.")]

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w = self.input_hw
        resized = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BICUBIC), dtype=np.float32)
        x = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def _embed(self, ids: Sequence[int]) -> np.ndarray:
        return self.embed.run(None, {"input_ids": np.asarray([list(ids)], dtype=np.int64)})[0]

    def generate(self, image_bgr: np.ndarray, prompt: str,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> list[int]:
        image_features = self.vision.run(None, {"pixel_values": self.preprocess(image_bgr)})[0]
        prompt_ids = self.tokenizer.encode(prompt).ids
        inputs_embeds = np.concatenate([image_features, self._embed(prompt_ids)], axis=1)
        mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
        hidden = self.encoder.run(None, {"attention_mask": mask, "inputs_embeds": inputs_embeds})[0]

        heads, head_dim = 12, 64
        for i in self.decoder.get_inputs():
            if i.name.startswith("past_key_values.") and isinstance(i.shape[1], int):
                heads = int(i.shape[1]); head_dim = int(i.shape[3]); break
        empty = np.zeros((1, heads, 0, head_dim), dtype=np.float32)
        past = {name: empty for name in self._past_names}
        sequence = [self.decoder_start]
        next_input = [self.decoder_start]
        for step in range(max_new_tokens):
            feeds = {"encoder_attention_mask": mask, "encoder_hidden_states": hidden,
                     "inputs_embeds": self._embed(next_input), **past,
                     "use_cache_branch": np.array([step > 0])}
            outputs = self.decoder.run(None, feeds)
            logits = outputs[0][0, -1].astype(np.float64)
            presents = dict(zip([o.name for o in self.decoder.get_outputs()], outputs))
            for name in self._past_names:
                present = "present." + name[len("past_key_values."):]
                if step > 0 and ".encoder." in name:
                    continue  # cross-attention cache is fixed after the first step
                past[name] = presents[present]
            if step == 0:
                token = self.bos                              # forced_bos_token_id
            elif step == max_new_tokens - 1:
                token = self.eos                              # forced_eos_token_id
            else:
                for banned in banned_ngram_tokens(sequence, self.no_repeat_ngram):
                    logits[banned] = -np.inf
                token = int(np.argmax(logits))
            sequence.append(token)
            if token == self.eos:
                break
            next_input = [token]
        return sequence

    def run(self, image_bgr: np.ndarray, task: str, text: str = "") -> dict[str, Any]:
        prompt = task_prompt(task, text)
        sequence = self.generate(image_bgr, prompt)
        generated = self.tokenizer.decode(sequence, skip_special_tokens=False)
        h, w = image_bgr.shape[:2]
        parsed = post_process(task, generated, (w, h))
        return {"task": task, "prompt": prompt, "raw": generated, **parsed}


_SESSIONS: dict[tuple, FlorenceOnnx] = {}


def load_verified(model_key: str = DEFAULT_MODEL_KEY) -> tuple[FlorenceOnnx, VerifiedModel]:
    try:
        model = verified_model(model_key, kind=KIND)
    except ModelFilesRefused as exc:
        raise _refused(str(exc)) from exc
    cache_key = tuple(sorted(model.digests.items()))
    runner = _SESSIONS.get(cache_key)
    if runner is None:
        runner = FlorenceOnnx(model.files)
        _SESSIONS.clear()
        _SESSIONS[cache_key] = runner
    return runner, model


def florence(image_bgr: np.ndarray, task: str, text: str = "",
             model_key: str = DEFAULT_MODEL_KEY) -> dict[str, Any]:
    """Run one Florence-2 task; text is unverified, boxes are suggestions."""
    task_prompt(task, text)   # refuse a bad task or a missing phrase before loading
    runner, model = load_verified(model_key)
    result = runner.run(image_bgr, task, text)
    return {
        **result,
        "model_key": model.model_key,
        "model_used": model.model_used,
        "model_sha256": model.model_sha256,
        "unverified": True,
        "metric": "unmeasured",
        "output_semantics": "suggestion",
        "confidence_kind": "none" if task in TEXT_TASKS else "zero_shot_no_score",
        "note": UNVERIFIED_NOTE.format(model=model.model_key),
    }
