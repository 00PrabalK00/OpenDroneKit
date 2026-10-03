"""InsPLAD two-stage runner (core/powerline.py): condition logits masked per asset."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core import models as models_module
from core.model_runner import ModelRunRefused, run_registered_model
from core.powerline import (
    ASSET_TO_CONDITION_PREFIX,
    condition_crop,
    masked_softmax,
    valid_condition_mask,
)

CLASSES = [
    "glass-insulator__good", "glass-insulator__missing-cap",
    "lightning-rod-suspension__good", "lightning-rod-suspension__rust",
    "polymer-insulator-upper-shackle__good", "polymer-insulator-upper-shackle__rust",
    "vari-grip__bird-nest", "vari-grip__good", "vari-grip__rust",
    "yoke-suspension__good", "yoke-suspension__rust",
]
NAMES = ["yoke", "yoke suspension", "spacer", "stockbridge damper", "lightning rod shackle",
         "lightning rod suspension", "polymer insulator", "glass insulator", "tower id plate",
         "vari-grip", "polymer insulator lower shackle", "polymer insulator upper shackle",
         "polymer insulator tower shackle", "glass insulator big shackle",
         "glass insulator small shackle", "glass insulator tower shackle", "spiral damper"]


def test_every_condition_asset_is_a_detector_class_with_conditions():
    for asset, prefix in ASSET_TO_CONDITION_PREFIX.items():
        assert asset in NAMES
        assert valid_condition_mask(CLASSES, prefix).sum() >= 2


def test_logits_are_masked_to_the_assets_valid_conditions_before_softmax():
    logits = np.zeros(len(CLASSES))
    logits[CLASSES.index("vari-grip__bird-nest")] = 20.0  # invalid for a glass insulator
    logits[CLASSES.index("glass-insulator__missing-cap")] = 1.0
    valid = valid_condition_mask(CLASSES, "glass-insulator")
    probs = masked_softmax(logits, valid)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[~valid].max() == 0.0
    assert CLASSES[int(probs.argmax())] == "glass-insulator__missing-cap"
    # The softmax is over the two valid logits only: e^1 / (e^1 + e^0).
    assert probs.max() == pytest.approx(np.e / (np.e + 1.0))
    # Unmasked, the bird's nest would have won -- which is what the mask prevents.
    assert CLASSES[int(logits.argmax())] == "vari-grip__bird-nest"


def test_vari_grip_keeps_all_three_conditions():
    valid = valid_condition_mask(CLASSES, "vari-grip")
    assert [c for c, v in zip(CLASSES, valid) if v] == [
        "vari-grip__bird-nest", "vari-grip__good", "vari-grip__rust"]


def test_an_asset_with_no_conditions_is_refused():
    with pytest.raises(ValueError):
        valid_condition_mask(CLASSES, "spacer")


def test_condition_crop_shape_and_normalisation():
    image = np.zeros((300, 500, 3), np.uint8)
    image[:, :, 2] = 255  # red in BGR
    x = condition_crop(image, [10, 20, 110, 70])
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    # Red channel after ImageNet normalisation: (1 - 0.485) / 0.229.
    assert x[0].mean() == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)
    assert x[2].mean() == pytest.approx((0 - 0.406) / 0.225, abs=1e-4)


# ----------------------------------------------------------------- end to end, tiny graphs

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(models_module, "models_root", lambda: root)
    monkeypatch.setattr(models_module, "registry_path", lambda: root / "model_registry.json")
    models_module._DIGEST_CACHE.clear()

    def write(models):
        (root / "model_registry.json").write_text(
            json.dumps({"version": 1, "models": models}), encoding="utf-8")

    return type("Fixture", (), {"root": root, "write": staticmethod(write)})


def _detector_and_classifier(root):
    from test_infrastructure_detectors import _constant_graph

    pred = np.zeros((1, 4 + 17, 2), np.float32)
    pred[0, :4, 0] = [20, 20, 16, 16]; pred[0, 4 + NAMES.index("glass insulator"), 0] = 0.9
    pred[0, :4, 1] = [48, 48, 12, 12]; pred[0, 4 + NAMES.index("spacer"), 1] = 0.8
    names = "{" + ", ".join(f"{i}: '{n}'" for i, n in enumerate(NAMES)) + "}"
    det = _constant_graph(root / "u5_test/det.onnx", [1, 3, 64, 64], {"output0": pred},
                          metadata={"names": names})
    logits = np.zeros((1, len(CLASSES)), np.float32)
    logits[0, CLASSES.index("vari-grip__bird-nest")] = 20.0
    logits[0, CLASSES.index("glass-insulator__missing-cap")] = 1.0
    cls = _constant_graph(root / "u5_test/cls.onnx", [1, 3, 224, 224], {"logits": logits})
    return det, cls


def test_two_stage_run_masks_condition_and_skips_detect_only_assets(registry):
    det, cls = _detector_and_classifier(registry.root)
    registry.write({
        "pl": {"kind": "onnx_yolo", "path": "u5_test/det.onnx", "labels": NAMES,
               "sha256": det, "input_size": 64, "condition_model": "plc",
               "licence": {"use": "non-commercial"}},
        "plc": {"kind": "onnx_condition_classifier", "path": "u5_test/cls.onnx",
                "labels": CLASSES, "output_classes": CLASSES, "sha256": cls,
                "condition_of": "pl"},
    })
    result = run_registered_model(np.full((64, 64, 3), 128, np.uint8), "pl")
    dets = {d["label"]: d for d in result.output["detections"]}
    glass = dets["glass insulator"]["condition"]
    assert glass["label"] == "missing-cap" and glass["model_key"] == "plc"
    assert glass["model_sha256"] == cls
    assert set(glass["probabilities"]) == {"glass-insulator__good",
                                           "glass-insulator__missing-cap"}
    assert dets["spacer"]["condition"] is None
    assert result.model_sha256 == det
    assert "non-commercial" in result.output["usage_note"]


def test_missing_condition_model_keeps_assets_and_says_why(registry):
    det, _ = _detector_and_classifier(registry.root)
    registry.write({
        "pl": {"kind": "onnx_yolo", "path": "u5_test/det.onnx", "labels": NAMES,
               "sha256": det, "input_size": 64, "condition_model": "plc"},
        "plc": {"kind": "onnx_condition_classifier", "path": "u5_test_absent/cls.onnx",
                "labels": CLASSES, "output_classes": CLASSES, "sha256": "0" * 64},
    })
    result = run_registered_model(np.full((64, 64, 3), 128, np.uint8), "pl")
    assert all(d["condition"] is None for d in result.output["detections"])
    assert "cannot run" in result.output["condition_refused"]


def test_condition_classifier_alone_is_refused(registry):
    _, cls = _detector_and_classifier(registry.root)
    registry.write({"plc": {"kind": "onnx_condition_classifier", "path": "u5_test/cls.onnx",
                            "labels": CLASSES, "sha256": cls, "condition_of": "pl"}})
    with pytest.raises(ModelRunRefused, match="not run on a whole photograph"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), "plc")


def test_prelabel_writes_condition_into_the_finding(registry, tmp_path):
    import cv2

    from app import desktop_ops

    det, cls = _detector_and_classifier(registry.root)
    registry.write({
        "pl": {"kind": "onnx_yolo", "path": "u5_test/det.onnx", "labels": NAMES,
               "sha256": det, "input_size": 64, "condition_model": "plc",
               "licence": {"use": "non-commercial"},
               "result_note": "Non-commercial; research/demonstration use per the publisher."},
        "plc": {"kind": "onnx_condition_classifier", "path": "u5_test/cls.onnx",
                "labels": CLASSES, "output_classes": CLASSES, "sha256": cls},
    })
    image = tmp_path / "tower.jpg"
    cv2.imwrite(str(image), np.full((64, 64, 3), 128, np.uint8))
    out = desktop_ops.prelabel_image(tmp_path, "p1", str(image), "pl")
    glass = next(f for f in out["findings"] if f["label"] == "glass insulator")
    assert glass["status"] == "in_review" and "condition:missing-cap" in glass["tags"]
    assert "licence:non-commercial" in glass["tags"]
    assert "research/demonstration" in glass["note"]


def test_real_insplad_two_stage_smoke():
    from core.models import verify_model_identity

    for key in ("powerline_insplad_detector", "powerline_insplad_condition"):
        if verify_model_identity(key)["status"] != "verified":
            pytest.skip(f"{key} weights not installed and verified here.")
    image = np.full((720, 1280, 3), 120, np.uint8)
    result = run_registered_model(image, "powerline_insplad_detector")
    assert result.output["condition_model"]["model_key"] == "powerline_insplad_condition"
    assert isinstance(result.output["detections"], list)
