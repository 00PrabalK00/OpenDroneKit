"""Grounding DINO text-prompted suggestions: refusal, caption, post-processing, Review.

The weights are not needed for most of this file. The real-inference smoke test at the
end runs only when the registered files are installed (ODK_MODEL_WEIGHTS_DIR) and is
skipped with the reason otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import models as core_models
from core.model_runner import ModelRunRefused

KEY = "open_vocab_grounding_dino_tiny"


# ----------------------------------------------------------------- fake registry


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """A registry with one open-vocab row whose files live under tmp_path/models."""
    root = tmp_path / "models"
    weights = root / KEY
    graph_sha = _write(weights / f"{KEY}.onnx", b"not really a graph")
    tok_sha = _write(weights / f"{KEY}__tokenizer.json", b"{}")
    row = {
        "kind": "onnx_open_vocab_detection",
        "path": f"{KEY}/{KEY}.onnx",
        "labels": [],
        "sha256": graph_sha,
        "status": "installed",
        "status_note": "Grounding DINO Tiny is not installed.",
        "output_semantics": "suggestion",
        "download": {"files": [
            {"name": f"{KEY}.onnx", "sha256": graph_sha},
            {"name": f"{KEY}__tokenizer.json", "sha256": tok_sha},
        ]},
    }
    reg = root / "model_registry.json"
    reg.write_text(json.dumps({"models": {KEY: row}}), encoding="utf-8")
    monkeypatch.setattr(core_models, "models_root", lambda: root)
    monkeypatch.setattr(core_models, "registry_path", lambda: reg)
    monkeypatch.delenv("ODK_MODEL_WEIGHTS_DIR", raising=False)
    return weights


def test_missing_model_refuses_with_the_status_note(fake_registry):
    from core.open_vocab import detect_by_text

    (fake_registry / f"{KEY}.onnx").unlink()
    with pytest.raises(ModelRunRefused, match="not installed"):
        detect_by_text(np.zeros((64, 64, 3), np.uint8), ["solar panel"], KEY)


def test_missing_companion_file_refuses(fake_registry):
    from core.open_vocab import detect_by_text

    (fake_registry / f"{KEY}__tokenizer.json").unlink()
    with pytest.raises(ModelRunRefused, match="tokenizer.json is missing"):
        detect_by_text(np.zeros((64, 64, 3), np.uint8), ["solar panel"], KEY)


def test_sha256_mismatch_refuses_before_loading(fake_registry):
    from core.open_vocab import detect_by_text

    (fake_registry / f"{KEY}__tokenizer.json").write_bytes(b'{"swapped": true}')
    with pytest.raises(ModelRunRefused, match="not the registered file"):
        detect_by_text(np.zeros((64, 64, 3), np.uint8), ["solar panel"], KEY)


def test_empty_prompt_refuses_before_touching_weights(fake_registry):
    from core.open_vocab import detect_by_text

    with pytest.raises(ModelRunRefused, match="needs a text prompt"):
        detect_by_text(np.zeros((64, 64, 3), np.uint8), ["  ", "."], KEY)


def test_run_registered_model_says_it_needs_a_text_prompt(fake_registry):
    from core.model_runner import run_registered_model

    with pytest.raises(ModelRunRefused, match="needs a text prompt"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), KEY)


# ---------------------------------------------------------------------- caption


def test_caption_is_lower_case_phrases_each_ending_in_a_full_stop():
    from core.open_vocab import build_caption

    caption, phrases = build_caption(["Solar Panel", "bird  nest.", "solar panel"])
    assert caption == "solar panel. bird nest."
    assert phrases == ["solar panel", "bird nest"]
    assert build_caption("insulator. Rust stain.")[0] == "insulator. rust stain."


class _FakeEncoding:
    def __init__(self, caption: str, n_extra: int = 0):
        words = caption.replace(".", " .").split()
        self.ids, self.offsets, pos = [101], [(0, 0)], 0
        for w in words:
            start = caption.index(w, pos)
            self.ids.append(1000 + len(self.ids)); self.offsets.append((start, start + len(w)))
            pos = start + len(w)
        self.ids += [5] * n_extra; self.offsets += [(0, 0)] * n_extra
        self.ids.append(102); self.offsets.append((0, 0))
        self.attention_mask = [1] * len(self.ids)


class _FakeTokenizer:
    def __init__(self, n_extra: int = 0):
        self.n_extra = n_extra

    def encode(self, caption):
        return _FakeEncoding(caption, self.n_extra)


def test_encode_caption_builds_all_text_inputs_and_phrase_spans():
    from core.open_vocab import encode_caption

    enc = encode_caption(_FakeTokenizer(), ["solar panel", "crack"])
    n = enc.input_ids.shape[1]
    assert enc.input_ids.dtype == np.int64 and enc.input_ids.shape == (1, n)
    assert enc.attention_mask.shape == (1, n) and enc.token_type_ids.shape == (1, n)
    assert not enc.token_type_ids.any()
    # [CLS] solar panel . crack . [SEP]
    assert enc.phrase_spans == [(1, 3), (4, 5)]


def test_caption_over_256_tokens_is_refused_not_truncated():
    from core.open_vocab import encode_caption

    with pytest.raises(ModelRunRefused, match="at most 256"):
        encode_caption(_FakeTokenizer(n_extra=300), ["solar panel"])


# --------------------------------------------------------------- post-processing


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_post_processing_matches_transformers_on_hand_made_logits():
    """Sigmoid, max over tokens > box threshold, phrase from tokens > text threshold,
    first and last of the 256 text slots never used, cxcywh -> xyxy in pixels."""
    from core.open_vocab import post_process_grounded

    Q, T = 4, 256
    logits = np.full((Q, T), -20.0, np.float32)
    ids = [101, 7001, 7002, 1012, 7003, 1012, 102]   # [CLS] solar panel . crack . [SEP]
    logits[0, 1] = _logit(0.9); logits[0, 2] = _logit(0.6)      # "solar panel"
    logits[1, 4] = _logit(0.5)                                    # "crack"
    logits[2, 4] = _logit(0.3)                                    # below box threshold
    logits[3, 0] = _logit(0.99)                                   # only [CLS]: kept, no phrase
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1],
                      [0.5, 0.5, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1]], np.float32)
    vocab = {7001: "solar", 7002: "panel", 7003: "crack", 1012: ".", 101: "[CLS]", 102: "[SEP]"}
    hits = post_process_grounded(logits, boxes, ids, box_threshold=0.35, text_threshold=0.25,
                                 image_size_hw=(100, 200),
                                 decode=lambda t: " ".join(vocab[i] for i in t))
    assert len(hits) == 3
    assert hits[0]["phrase"] == "solar panel"
    assert hits[0]["score"] == pytest.approx(0.9, abs=1e-5)
    assert hits[0]["bbox"] == pytest.approx([80.0, 30.0, 120.0, 70.0], abs=1e-3)
    assert hits[1]["phrase"] == "crack" and hits[1]["score"] == pytest.approx(0.5, abs=1e-5)
    assert hits[1]["bbox"] == pytest.approx([40.0, 20.0, 60.0, 30.0], abs=1e-3)
    assert hits[2]["phrase"] == ""


def test_tiles_cover_the_frame_and_nms_merges_duplicates():
    from core.open_vocab import nms, tile_windows

    wins = tile_windows(3000, 4000, 1024)
    covered = np.zeros((3000, 4000), bool)
    for x0, y0, x1, y1 in wins:
        assert x1 - x0 <= 1024 and y1 - y0 <= 1024
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    assert tile_windows(500, 600, 1024) == [(0, 0, 600, 500)]
    keep = nms(np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float),
               np.array([0.5, 0.9, 0.4]))
    assert keep == [1, 2]


# ------------------------------------------------------------------- Review


def test_suggestions_are_in_review_tagged_and_kept_out_of_reports(tmp_path, monkeypatch):
    from app import desktop_ops
    from core import open_vocab
    from core.annotations import list_annotations
    from core.report_engine import findings_from_annotations

    image = tmp_path / "frame.jpg"
    import cv2
    cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))

    class Spec:
        kind = "onnx_open_vocab_detection"

    monkeypatch.setattr(core_models, "get_model_spec", lambda key: Spec())

    def fake_detect(img, prompts, model_key, **kw):
        return {"model_key": model_key, "model_used": f"onnx:{model_key}.onnx",
                "model_sha256": "ab" * 32, "confidence_kind": "zero_shot_score",
                "detections": [{"label": "bird nest", "phrase": "bird nest",
                                "confidence": 0.41, "bbox_xyxy": [10, 10, 40, 30]}]}

    monkeypatch.setattr(open_vocab, "detect_by_text", fake_detect)
    out = desktop_ops.suggest_objects(tmp_path, "p1", str(image), ["bird nest"], KEY)
    assert out["count"] == 1 and out["confidence_kind"] == "zero_shot_score"
    assert "waiting for review" in out["message"]
    (stored,) = list_annotations(tmp_path, "p1")
    assert stored.status == "in_review" and stored.include_in_report is False
    assert stored.created_by == "model" and stored.confidence == pytest.approx(0.41)
    assert stored.model_key == KEY and stored.model_sha256 == "ab" * 32
    assert {"suggestion", "zero-shot"} <= set(stored.tags)
    assert any(KEY.replace("_", "-") in t or KEY in t for t in stored.tags)
    assert "Zero-shot suggestion" in stored.note and "uncalibrated" in stored.note
    assert findings_from_annotations(list_annotations(tmp_path, "p1"), [tmp_path]) == []


def test_findings_from_detections_keeps_its_default_tags():
    import inspect

    from app.desktop_ops import findings_from_detections

    params = inspect.signature(findings_from_detections).parameters
    assert params["tags"].default is None and params["note"].default is None


# ------------------------------------------------------------ real inference


def _installed() -> str:
    try:
        from core.model_files import verified_model

        verified_model(KEY)
        return ""
    except Exception as exc:  # noqa: BLE001 - the reason is the skip message
        return str(exc)


@pytest.mark.skipif(bool(_installed()), reason=f"Grounding DINO weights not installed: {_installed()}")
def test_real_inference_smoke():
    from core.open_vocab import detect_by_text

    img = np.full((600, 800, 3), 200, np.uint8)
    img[200:400, 300:500] = (30, 30, 30)
    out = detect_by_text(img, ["black square"], KEY)
    assert out["output_semantics"] == "suggestion"
    assert out["confidence_kind"] == "zero_shot_score"
    assert len(out["model_sha256"]) == 64
    for d in out["detections"]:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        assert 0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 600
        assert 0.35 < d["confidence"] <= 1.0
    assert out["detections"], "a plain dark square on grey should be found"
