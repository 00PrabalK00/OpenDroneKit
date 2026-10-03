"""RF-DETR road damage and YOLO11-seg power-line runners (core/infrastructure_detectors.py).

The decode tests use hand-made tensors whose right answer is known. The refusal and
pre-label tests build tiny real ONNX graphs with constant outputs, so the whole path
-- registry, digest check, onnxruntime, decode, findings -- runs without any weights.
The smoke tests at the end run the released weights when they are installed and skip
with the reason when they are not.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core import models as models_module
from core.detection import DETECTION_KINDS, YOLO_BOX_KINDS, letterbox_image
from core.infrastructure_detectors import (
    decode_detr,
    decode_instance_mask,
    mask_polygons,
    yolo_candidates,
)
from core.model_runner import ModelRunRefused, run_registered_model

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

pytest.importorskip("onnxruntime")


# ----------------------------------------------------------------- fixtures


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


def _constant_graph(path: Path, input_shape: list[int],
                    outputs: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> str:
    """An ONNX graph that ignores its image and returns fixed tensors. Returns its sha256."""
    nodes, outs = [], []
    # Use the input so onnxruntime keeps it: out = const + 0 * reduce_mean(images).
    nodes.append(helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["m", "zero"], ["z"]))
    inits = [numpy_helper.from_array(np.array(0.0, np.float32), "zero")]
    for name, value in outputs.items():
        inits.append(numpy_helper.from_array(value.astype(np.float32), f"{name}_c"))
        nodes.append(helper.make_node("Add", [f"{name}_c", "z"], [name]))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    graph = helper.make_graph(
        nodes, "constant", [helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)],
        outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _detr_outputs() -> dict[str, np.ndarray]:
    # Two queries, four classes plus a no-object slot (RF-DETR's head has num_classes+1).
    dets = np.array([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]]], np.float32)
    labels = np.full((1, 2, 5), -9.0, np.float32)
    labels[0, 0, 3] = _logit(0.9)   # query 0: pothole 0.9
    labels[0, 1, 1] = _logit(0.6)   # query 1: transverse crack 0.6
    labels[0, 1, 4] = _logit(0.95)  # query 1: no-object slot, must never become a box
    return {"dets": dets, "labels": labels}


DETR_LABELS = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


# ----------------------------------------------------------------- kinds


def test_new_kinds_are_detectors_but_not_yolo_box_kinds():
    assert {"onnx_detr", "onnx_yolo_seg"} <= DETECTION_KINDS
    assert not ({"onnx_detr", "onnx_yolo_seg"} & YOLO_BOX_KINDS)
    assert YOLO_BOX_KINDS <= DETECTION_KINDS


# ----------------------------------------------------------------- DETR decode


def test_detr_decode_scores_boxes_and_order():
    out = _detr_outputs()
    hits = decode_detr(out["dets"][0], out["labels"][0], (200, 400), 0.5, num_labels=4)
    assert [h[0] for h in hits] == [3, 1]
    assert hits[0][1] == pytest.approx(0.9, abs=1e-5)
    assert hits[1][1] == pytest.approx(0.6, abs=1e-5)
    # cx 0.5, cy 0.5, w 0.2, h 0.4 on a 400 x 200 image.
    assert hits[0][2] == pytest.approx([160.0, 60.0, 240.0, 140.0], abs=1e-3)
    assert hits[1][2] == pytest.approx([80.0, 40.0, 120.0, 60.0], abs=1e-3)


def test_detr_decode_threshold_is_strict_and_drops_no_object_slot():
    out = _detr_outputs()
    hits = decode_detr(out["dets"][0], out["labels"][0], (200, 400), 0.6 + 1e-6, num_labels=4)
    assert [h[0] for h in hits] == [3]
    # Without num_labels the 0.95 no-object slot would come first; with it, never.
    every = decode_detr(out["dets"][0], out["labels"][0], (200, 400), 0.0, num_labels=4)
    assert all(c < 4 for c, _, _ in every)


def test_detr_decode_one_query_can_yield_two_classes():
    dets = np.array([[0.5, 0.5, 0.5, 0.5]], np.float32)
    logits = np.array([[_logit(0.8), _logit(0.7), -9, -9]], np.float32)
    hits = decode_detr(dets, logits, (100, 100), 0.5)
    assert sorted(c for c, _, _ in hits) == [0, 1]


def test_detr_decode_clips_to_the_image():
    dets = np.array([[0.02, 0.98, 0.2, 0.2]], np.float32)
    logits = np.array([[_logit(0.9)]], np.float32)
    (_, _, box), = decode_detr(dets, logits, (100, 100), 0.5)
    assert box[0] == 0.0 and box[3] == 100.0


# ----------------------------------------------------------------- YOLO-seg decode


def test_yolo_candidates_layout_and_nms():
    n = 4
    pred = np.zeros((4 + 2 + 32, n), np.float32)
    # Two overlapping Line boxes (NMS keeps the better), one Tower box, one below threshold.
    pred[:4, 0] = [100, 100, 40, 40]; pred[4, 0] = 0.9
    pred[:4, 1] = [102, 101, 40, 40]; pred[4, 1] = 0.8
    pred[:4, 2] = [300, 300, 50, 80]; pred[5, 2] = 0.7
    pred[:4, 3] = [500, 500, 10, 10]; pred[4, 3] = 0.1
    pred[6, 0] = 1.0  # first mask coefficient of the kept Line box
    boxes, scores, cls, coef = yolo_candidates(pred, 2, 0.25, 0.5, extra=32)
    assert list(cls) == [0, 1]
    assert scores == pytest.approx([0.9, 0.7])
    assert boxes[0] == pytest.approx([80, 80, 120, 120])
    assert coef.shape == (2, 32) and coef[0, 0] == 1.0
    with pytest.raises(ValueError):
        yolo_candidates(pred, 3, 0.25, 0.5, extra=32)


def test_yolo_seg_mask_decode_against_a_synthetic_proto():
    """A 1280 x 720 image letterboxed to 640: gain 0.5, 140 px of padding top and bottom.

    One prototype is +10 over a rectangle and -10 elsewhere; the coefficient selects it.
    The decoded mask must be that rectangle, clipped to the box, in source pixels.
    """
    image = np.zeros((720, 1280, 3), np.uint8)
    lb = letterbox_image(image, 640)
    assert (lb.gain, lb.left, lb.top, lb.new_w, lb.new_h) == (0.5, 0, 140, 640, 360)
    protos = np.full((32, 160, 160), -10.0, np.float32)
    # Rectangle in input coordinates x 200..400, y 240..400 -> proto /4.
    protos[0, 60:100, 50:100] = 10.0
    coef = np.zeros(32, np.float32)
    coef[0] = 1.0
    # The box covers only the left half of the rectangle: x 200..300 in input pixels.
    box = np.array([200, 240, 300, 400], np.float32)
    local, (x1, y1, x2, y2) = decode_instance_mask(
        coef, protos, box, input_size=640, gain=lb.gain, left=lb.left, top=lb.top,
        new_w=lb.new_w, new_h=lb.new_h, image_hw=(720, 1280))
    # Source box: x (200-0)/0.5 .. (300)/0.5, y (240-140)/0.5 .. (400-140)/0.5.
    assert (x1, y1, x2, y2) == (400, 200, 600, 520)
    assert local.shape == (320, 200)
    # Inside the rectangle and the box, the mask is on (allow a 4 px bilinear border).
    assert local[8:-8, 8:-8].all()
    assert 0.95 <= local.mean() <= 1.0
    polys = mask_polygons(local, (x1, y1))
    assert polys and min(p[0] for p in polys[0]) >= 400 and max(p[0] for p in polys[0]) <= 600


def test_yolo_seg_mask_is_empty_outside_the_proto_region():
    protos = np.full((32, 160, 160), -10.0, np.float32)
    protos[0, 0:10, 0:10] = 10.0
    coef = np.zeros(32, np.float32)
    coef[0] = 1.0
    local, _ = decode_instance_mask(
        coef, protos, np.array([300, 300, 400, 400], np.float32), input_size=640,
        gain=1.0, left=0, top=0, new_w=640, new_h=640, image_hw=(640, 640))
    assert not local.any()


# ----------------------------------------------------------------- refusal


def test_missing_file_refuses(registry):
    registry.write({"road": {"kind": "onnx_detr", "path": "u5_test_absent/road.onnx",
                             "labels": DETR_LABELS, "sha256": "0" * 64,
                             "status_note": "Download road_damage_rfdetr_small.onnx."}})
    with pytest.raises(ModelRunRefused, match="Download road_damage_rfdetr_small"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), "road")


def test_digest_mismatch_refuses(registry):
    _constant_graph(registry.root / "u5_test/road.onnx", [1, 3, 64, 64], _detr_outputs())
    registry.write({"road": {"kind": "onnx_detr", "path": "u5_test/road.onnx",
                             "labels": DETR_LABELS, "sha256": "f" * 64}})
    with pytest.raises(ModelRunRefused, match="not the file that was registered"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), "road")


def test_unrecorded_digest_refuses(registry):
    _constant_graph(registry.root / "u5_test/road.onnx", [1, 3, 64, 64], _detr_outputs())
    registry.write({"road": {"kind": "onnx_detr", "path": "u5_test/road.onnx",
                             "labels": DETR_LABELS}})
    with pytest.raises(ModelRunRefused, match="no recorded digest"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), "road")


def test_registry_labels_must_match_onnx_metadata(registry):
    seg = _seg_outputs()
    digest = _constant_graph(registry.root / "u5_test/seg.onnx", [1, 3, 64, 64], seg,
                             metadata={"names": "{0: 'Line', 1: 'Tower'}"})
    registry.write({"seg": {"kind": "onnx_yolo_seg", "path": "u5_test/seg.onnx",
                            "labels": ["Tower", "Line"], "sha256": digest}})
    with pytest.raises(ModelRunRefused, match="disagree"):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), "seg")


# ----------------------------------------------------------------- run + prelabel


def _seg_outputs() -> dict[str, np.ndarray]:
    pred = np.zeros((1, 4 + 2 + 32, 3), np.float32)
    pred[0, :4, 0] = [20, 32, 16, 16]; pred[0, 4, 0] = 0.9  # Line
    pred[0, :4, 1] = [44, 32, 12, 30]; pred[0, 5, 1] = 0.8  # Tower
    pred[0, 6, :] = 1.0
    protos = np.full((1, 32, 16, 16), 5.0, np.float32)
    return {"output0": pred, "output1": protos}


def test_run_registered_model_detr_and_seg_carry_identity(registry):
    d1 = _constant_graph(registry.root / "u5_test/road.onnx", [1, 3, 64, 64], _detr_outputs())
    d2 = _constant_graph(registry.root / "u5_test/seg.onnx", [1, 3, 64, 64], _seg_outputs(),
                         metadata={"names": "{0: 'Line', 1: 'Tower'}"})
    registry.write({
        "road": {"kind": "onnx_detr", "path": "u5_test/road.onnx", "labels": DETR_LABELS,
                 "sha256": d1, "score_threshold": 0.5,
                 "licence": {"use": "commercial"}},
        "seg": {"kind": "onnx_yolo_seg", "path": "u5_test/seg.onnx", "labels": ["Line", "Tower"],
                "sha256": d2, "score_threshold": 0.25, "iou_threshold": 0.7,
                "licence": {"use": "copyleft"},
                "result_note": "Geometry only, not clearance."},
    })
    road = run_registered_model(np.zeros((100, 200, 3), np.uint8), "road")
    assert road.model_sha256 == d1 and road.model_used == "onnx:road.onnx"
    assert [d["label"] for d in road.output["detections"]] == ["pothole", "transverse_crack"]
    assert road.output["detections"][0]["bbox_xyxy"] == [80, 30, 120, 70]
    assert "commercial" in road.output["usage_note"]
    assert road.overlay is not None and road.overlay.shape == (100, 200, 3)

    seg = run_registered_model(np.zeros((64, 64, 3), np.uint8), "seg")
    labels = [d["label"] for d in seg.output["detections"]]
    assert labels == ["Line", "Tower"]
    line = seg.output["detections"][0]
    assert line["bbox_xyxy"] == [12, 24, 28, 40]
    assert line["mask_area_px"] > 0 and line["mask_polygons_px"]
    assert "not clearance" in seg.output["usage_note"]


def test_prelabel_accepts_new_kinds_and_writes_in_review_findings(registry, tmp_path):
    import cv2

    from app import desktop_ops
    from core.annotations import list_annotations

    d1 = _constant_graph(registry.root / "u5_test/road.onnx", [1, 3, 64, 64], _detr_outputs())
    d2 = _constant_graph(registry.root / "u5_test/seg.onnx", [1, 3, 64, 64], _seg_outputs())
    registry.write({
        "road": {"kind": "onnx_detr", "path": "u5_test/road.onnx", "labels": DETR_LABELS,
                 "sha256": d1, "score_threshold": 0.5, "licence": {"use": "commercial"}},
        "seg": {"kind": "onnx_yolo_seg", "path": "u5_test/seg.onnx", "labels": ["Line", "Tower"],
                "sha256": d2, "licence": {"use": "copyleft"},
                "result_note": "TTPLA: geometry only, not clearance."},
    })
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.full((64, 64, 3), 90, np.uint8))
    project = tmp_path / "project"
    project.mkdir()
    for key, sha, use in (("road", d1, "commercial"), ("seg", d2, "copyleft")):
        out = desktop_ops.prelabel_image(project, "p1", str(image), key)
        assert out["count"] == 2 and out["model_sha256"] == sha
        for finding in out["findings"]:
            assert finding["status"] == "in_review"
            assert finding["include_in_report"] is False
            assert finding["created_by"] == "model"
            assert finding["model_key"] == key and finding["model_sha256"] == sha
            assert 0.0 < finding["confidence"] <= 1.0
            assert f"licence:{use}" in finding["tags"]
    seg_notes = [a.note for a in list_annotations(project, "p1") if a.model_key == "seg"]
    assert seg_notes and all("not clearance" in n for n in seg_notes)


def test_prelabel_refuses_a_missing_new_kind_model(registry, tmp_path):
    import cv2

    from app import desktop_ops

    registry.write({"seg": {"kind": "onnx_yolo_seg", "path": "u5_test_absent/seg.onnx",
                            "labels": ["Line", "Tower"], "sha256": "0" * 64}})
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="cannot run"):
        desktop_ops.prelabel_image(tmp_path, "p1", str(image), "seg")


# ----------------------------------------------------------------- real weights


def _real(key: str):
    from core.models import verify_model_identity

    check = verify_model_identity(key)
    if check["status"] != "verified":
        pytest.skip(f"{key} weights not installed and verified here ({check['status']}); "
                    "set ODK_MODEL_WEIGHTS_DIR to the folder holding them.")


def _testdata(*parts: str) -> Path:
    root = os.environ.get("ODK_TESTDATA_DIR", "").strip()
    path = Path(root, *parts) if root else Path("__absent__")
    if not path.exists():
        pytest.skip(f"Test image {'/'.join(parts)} not found under ODK_TESTDATA_DIR.")
    return path


def test_real_rfdetr_on_an_rdd2022_drone_frame():
    _real("road_damage_rfdetr_small")
    from core.detection import load_image

    image = load_image(_testdata("road_damage_rdd2022_subset", "China_Drone", "train", "images",
                                 "China_Drone_000128.jpg"))
    result = run_registered_model(image, "road_damage_rfdetr_small")
    assert result.kind == "onnx_detr"
    for det in result.output["detections"]:
        assert det["label"] in DETR_LABELS and 0.0 < det["confidence"] <= 1.0


def test_real_ttpla_seg_on_a_test_frame():
    _real("powerline_ttpla_yolo11s_seg")
    from core.detection import load_image

    image = load_image(_testdata("ttpla_test", "images", "04_2220.jpg"))
    result = run_registered_model(image, "powerline_ttpla_yolo11s_seg")
    dets = result.output["detections"]
    assert dets, "TTPLA test frame 04_2220 has five annotated cables; expected detections."
    assert {d["label"] for d in dets} <= {"Line", "Tower"}
    assert all(d["mask_area_px"] >= 0 for d in dets)
