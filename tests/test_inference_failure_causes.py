"""A model that fails says why, instead of reading as a model that found nothing.

The core segmentation helpers return ``None`` on any failure, which the AI worker
used to report as "produced no usable mask" or "output does not match its N labels"
even when the real cause was an out-of-memory forward pass. These tests run real
(tiny) ONNX graphs through OpenCV's runtime and check the reported cause.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
cv2 = pytest.importorskip("cv2")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from core.detection import (  # noqa: E402
    _MODEL_NET_CACHE,
    _run_onnx_segmentation,
    _run_onnx_segmentation_multiclass,
)
from services.api.inference import InferenceRefused, ModelChoice, _infer_one  # noqa: E402


def _save(graph, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.save(model, str(path))
    _MODEL_NET_CACHE.pop(str(path.resolve()), None)
    return path


def _two_channel_segmenter(path: Path) -> Path:
    """A 1x1 convolution emitting 2 channels: a real graph with the wrong class count."""
    weight = numpy_helper.from_array(np.ones((2, 3, 1, 1), dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"])], "two_channel",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 64, 64])],
        initializer=[weight],
    )
    return _save(graph, path)


def _unrunnable(path: Path) -> Path:
    """A graph built for 64 px input whose forward pass fails at any other size."""
    weight = numpy_helper.from_array(np.ones((3 * 64 * 64, 2), dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["x"], ["f"], axis=1),
         helper.make_node("MatMul", ["f", "w"], ["y"])], "unrunnable",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[weight],
    )
    return _save(graph, path)


def _choice(path: Path, labels: list[str], kind: str, size: int = 64) -> ModelChoice:
    return ModelChoice(task="segmentation", model_key="test_segmenter", kind=kind,
                       path=str(path), labels=labels, sha256="", score_threshold=0.5,
                       input_size=size)


IMAGE = np.full((80, 96, 3), 127, dtype=np.uint8)


def test_default_behaviour_of_the_core_helpers_is_unchanged(tmp_path):
    graph = _two_channel_segmenter(tmp_path / "two.onnx")
    assert _run_onnx_segmentation_multiclass(IMAGE, graph, 64, 4) is None


def test_a_class_count_mismatch_names_the_shape(tmp_path):
    graph = _two_channel_segmenter(tmp_path / "two.onnx")
    with pytest.raises(ValueError, match=r"\(1, 2, 64, 64\).*4 labels"):
        _run_onnx_segmentation_multiclass(IMAGE, graph, 64, 4, raise_errors=True)

    with pytest.raises(InferenceRefused, match=r"test_segmenter failed on photo\.jpg: "
                                               r"ValueError: .*\(1, 2, 64, 64\)"):
        _infer_one(_choice(graph, ["a", "b", "c", "d"], "onnx_segmentation_multiclass"),
                   IMAGE, Path("photo.jpg"), tmp_path, {})


def test_a_failed_forward_pass_reports_the_runtime_error_not_an_empty_mask(tmp_path):
    graph = _unrunnable(tmp_path / "bad.onnx")
    net = cv2.dnn.readNetFromONNX(str(graph))  # it loads; only running it fails
    assert net is not None

    assert _run_onnx_segmentation(IMAGE, graph, 0.5, 96) is None  # the old, silent answer
    with pytest.raises(cv2.error):
        _run_onnx_segmentation(IMAGE, graph, 0.5, 96, raise_errors=True)

    with pytest.raises(InferenceRefused) as refused:
        _infer_one(_choice(graph, ["crack"], "onnx_segmentation", size=96),
                   IMAGE, Path("photo.jpg"), tmp_path, {})
    message = str(refused.value)
    assert message.startswith("test_segmenter failed on photo.jpg: error: ")
    assert "OpenCV" in message
    assert "no usable mask" not in message


def test_running_out_of_memory_is_reported_as_such(tmp_path):
    from services.api.inference import _model_step

    def exhausted():
        raise MemoryError()

    with pytest.raises(InferenceRefused, match="ran out of memory on photo.jpg"):
        _model_step(_choice(tmp_path / "x.onnx", ["crack"], "onnx_segmentation"),
                    Path("photo.jpg"), exhausted)
