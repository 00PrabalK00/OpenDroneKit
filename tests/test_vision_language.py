"""Florence-2: <loc_N> parsing, refusal, and captions stored as unverified notes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import models as core_models
from core.model_runner import ModelRunRefused

KEY = "vision_language_florence2_base_ft"


def test_loc_bins_map_to_pixel_bin_centres():
    from core.vision_language import dequantize

    # 1000 bins over 2000 x 1000 px: bin 0 -> centre 1 px (x) / 0.5 px (y).
    assert dequantize([0, 0, 999, 999], (2000, 1000)) == pytest.approx([1.0, 0.5, 1999.0, 999.5])


def test_phrase_grounding_and_od_parsing():
    from core.vision_language import post_process

    raw = "</s><s>solar panel<loc_100><loc_200><loc_300><loc_400>it<loc_1><loc_2><loc_3><loc_4></s>"
    out = post_process("<CAPTION_TO_PHRASE_GROUNDING>", raw, (1000, 500))
    assert [r["label"] for r in out["regions"]] == ["solar panel"]   # "it" is banned
    assert out["regions"][0]["bbox"] == pytest.approx([100.5, 100.25, 300.5, 200.25])
    od = post_process("<OD>", "<s>car<loc_0><loc_0><loc_499><loc_499>car<loc_500><loc_500>"
                              "<loc_999><loc_999></s>", (1000, 1000))
    assert [r["label"] for r in od["regions"]] == ["car", "car"]
    assert od["text"] == "car"


def test_ocr_with_region_parses_quadrilaterals():
    from core.vision_language import post_process

    raw = "<s>T-104" + "".join(f"<loc_{v}>" for v in (10, 10, 90, 10, 90, 30, 10, 30)) + "</s>"
    out = post_process("<OCR_WITH_REGION>", raw, (1000, 1000))
    assert out["text"] == "T-104"
    assert len(out["regions"][0]["quad"]) == 8


def test_caption_is_plain_text():
    from core.vision_language import post_process

    assert post_process("<CAPTION>", "</s><s>A solar farm.</s>", (10, 10)) == {
        "text": "A solar farm.", "regions": []}


def test_no_repeat_ngram_bans_the_repeating_token():
    from core.vision_language import banned_ngram_tokens

    assert banned_ngram_tokens([5, 6, 7, 5, 6], 3) == {7}
    assert banned_ngram_tokens([5, 6], 3) == set()


def test_unknown_task_and_missing_phrase_refuse():
    from core.vision_language import task_prompt

    with pytest.raises(ModelRunRefused, match="Unsupported"):
        task_prompt("<VQA>")
    with pytest.raises(ModelRunRefused, match="needs a text prompt"):
        task_prompt("<CAPTION_TO_PHRASE_GROUNDING>", "  ")
    assert task_prompt("<CAPTION>") == "What does the image describe?"


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    root = tmp_path / "models"
    weights = root / KEY
    weights.mkdir(parents=True)
    data = b"decoder"
    (weights / f"{KEY}__decoder_model_merged_int8.onnx").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    row = {"kind": "onnx_vision_language",
           "path": f"{KEY}/{KEY}__decoder_model_merged_int8.onnx", "sha256": sha,
           "status": "installed", "status_note": "Florence-2-base-ft is not installed.",
           "download": {"files": [
               {"name": f"{KEY}__decoder_model_merged_int8.onnx", "sha256": sha},
               {"name": f"{KEY}__vision_encoder_int8.onnx", "sha256": "0" * 64}]}}
    reg = root / "model_registry.json"
    reg.write_text(json.dumps({"models": {KEY: row}}), encoding="utf-8")
    monkeypatch.setattr(core_models, "models_root", lambda: root)
    monkeypatch.setattr(core_models, "registry_path", lambda: reg)
    monkeypatch.delenv("ODK_MODEL_WEIGHTS_DIR", raising=False)
    return weights


def test_missing_graph_refuses(fake_registry):
    from core.vision_language import florence

    with pytest.raises(ModelRunRefused, match="vision_encoder_int8.onnx is missing"):
        florence(np.zeros((32, 32, 3), np.uint8), "<CAPTION>", model_key=KEY)


def test_missing_model_refuses_with_status_note(fake_registry):
    from core.vision_language import florence

    for p in fake_registry.iterdir():
        p.unlink()
    with pytest.raises(ModelRunRefused, match="not installed"):
        florence(np.zeros((32, 32, 3), np.uint8), "<CAPTION>", model_key=KEY)


def test_run_registered_model_routes_to_describe_image(fake_registry):
    from core.model_runner import run_registered_model

    with pytest.raises(ModelRunRefused, match="needs a text prompt"):
        run_registered_model(np.zeros((32, 32, 3), np.uint8), KEY)


def test_caption_is_stored_unverified_in_review_and_out_of_reports(tmp_path, monkeypatch):
    import cv2

    from app import desktop_ops
    from core import vision_language
    from core.annotations import list_annotations
    from core.report_engine import findings_from_annotations

    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((40, 60, 3), np.uint8))

    def fake_florence(img, task, text="", model_key=KEY):
        return {"task": task, "text": "A tower with a red light.", "regions": [],
                "model_key": model_key, "model_used": "onnx:decoder.onnx",
                "model_sha256": "cd" * 32, "unverified": True, "metric": "unmeasured",
                "note": vision_language.UNVERIFIED_NOTE.format(model=model_key)}

    monkeypatch.setattr(vision_language, "florence", fake_florence)
    out = desktop_ops.describe_image(tmp_path, "p1", str(image), "<CAPTION>")
    assert out["unverified"] is True and out["metric"] == "unmeasured"
    (note,) = list_annotations(tmp_path, "p1")
    assert note.annotation_type == "text" and note.created_by == "model"
    assert note.status == "in_review" and note.include_in_report is False
    assert note.confidence is None
    assert "unverified" in note.tags and "unverified" in note.note.lower()
    assert "unmeasured" in note.note and "A tower with a red light." in note.note
    assert findings_from_annotations(list_annotations(tmp_path, "p1"), [tmp_path]) == []


def test_grounded_boxes_become_unscored_suggestions(tmp_path, monkeypatch):
    import cv2

    from app import desktop_ops
    from core import vision_language
    from core.annotations import list_annotations

    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((40, 60, 3), np.uint8))

    def fake_florence(img, task, text="", model_key=KEY):
        return {"task": task, "text": "insulator",
                "regions": [{"label": "insulator", "bbox": [1.0, 2.0, 20.0, 30.0]}],
                "model_key": model_key, "model_used": "onnx:decoder.onnx",
                "model_sha256": "cd" * 32, "unverified": True, "metric": "unmeasured",
                "note": "n"}

    monkeypatch.setattr(vision_language, "florence", fake_florence)
    out = desktop_ops.describe_image(tmp_path, "p1", str(image),
                                     "<CAPTION_TO_PHRASE_GROUNDING>", "insulator")
    assert out["count"] == 1
    (finding,) = list_annotations(tmp_path, "p1")
    assert finding.confidence is None and "suggestion" in finding.tags
    assert finding.status == "in_review" and not finding.include_in_report
    assert "gives no score" in finding.note


def _installed() -> str:
    try:
        from core.model_files import verified_model

        verified_model(KEY)
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)


@pytest.mark.skipif(bool(_installed()), reason=f"Florence-2 weights not installed: {_installed()}")
def test_real_inference_smoke():
    from core.vision_language import florence

    img = np.full((300, 400, 3), 220, np.uint8)
    img[100:200, 150:250] = (0, 0, 200)
    out = florence(img, "<CAPTION>", model_key=KEY)
    assert isinstance(out["text"], str) and out["text"].strip()
    assert out["unverified"] is True and out["metric"] == "unmeasured"
    assert len(out["model_sha256"]) == 64
    od = florence(img, "<OD>", model_key=KEY)
    for r in od["regions"]:
        x1, y1, x2, y2 = r["bbox"]
        assert 0 <= x1 <= x2 <= 400 and 0 <= y1 <= y2 <= 300
