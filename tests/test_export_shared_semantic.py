'''Shared semantic exports create the exact honest runtime manifest contract.'''

from __future__ import annotations

import json

import pytest

from core.semantic_engine import load_semantic_manifest

# The training modules import torch at module scope, and torch is not a runtime
# dependency of this project -- the runtime loads ONNX through cv2.dnn. Without this the
# import raises during collection and takes the WHOLE suite down with exit code 2, which
# is how CI stayed red while every individual test was fine. CI installs the CPU wheel so
# these still run there; the skip is for machines that only ever run the shipped code.
pytest.importorskip("torch", reason="training-only dependency; the runtime uses cv2.dnn")

from core.semantic_engine import sha256_onnx_model  # noqa: E402
from training.export_shared_semantic import build_runtime_manifest, sha256_file  # noqa: E402


def test_runtime_manifest_uses_onnx_hash_and_numeric_metrics(tmp_path):
    onnx = tmp_path / 'model.onnx'
    onnx.write_bytes(b'full trained encoder and decoder')
    schema = {
        'id': 'shared',
        'version': '1.0.0',
        'classes': [
            {'id': 0, 'name': 'background', 'color_rgb': [0, 0, 0], 'background': True},
            {'id': 1, 'name': 'building', 'color_rgb': [255, 0, 0]},
        ],
    }
    manifest = build_runtime_manifest(
        onnx_path=onnx,
        model_key='shared_semantic',
        model_version='1.0.0',
        schema=schema,
        validation_metrics={
            'mean_iou': 0.71,
            'pixel_accuracy': 0.92,
            'per_class_iou': {'building': 0.65},
        },
        inference={'tile_size': 518, 'overlap': 126},
        training_origin='site-separated fixture',
    )
    # The whole model, not the graph file. These are the same bytes here because this
    # fixture has no external-data sidecar, but they are NOT the same function: the real
    # shared_semantic export is a 1.1 MB graph beside a 378 MB .onnx.data, and
    # sha256_file would identify 0.3% of it. See tests/test_onnx_external_data_digest.py.
    assert manifest['model']['checkpoint_sha256'] == sha256_onnx_model(onnx)
    assert manifest['model']['task_trained'] is True
    assert manifest['model']['validation_metrics'] == {
        'mean_iou': 0.71,
        'pixel_accuracy': 0.92,
    }
    assert manifest['inference']['mean'] == [0.485, 0.456, 0.406]

    path = tmp_path / 'model.manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    loaded_schema, loaded_model, inference = load_semantic_manifest(path)
    assert loaded_schema.id == 'shared'
    assert loaded_model.checkpoint_sha256 == sha256_onnx_model(onnx)
    assert inference['tile_size'] == 518


def test_runtime_manifest_refuses_missing_numeric_evidence(tmp_path):
    onnx = tmp_path / 'model.onnx'
    onnx.write_bytes(b'model')
    with pytest.raises(ValueError, match='numeric validation metrics'):
        build_runtime_manifest(
            onnx_path=onnx,
            model_key='shared',
            model_version='1',
            schema={'id': 's', 'version': '1', 'classes': [1, 2]},
            validation_metrics={'per_class': {}},
            inference={},
            training_origin='fixture',
        )


class TestParityIsCheckedOnTheLabelNotTheLogit:
    """The gate that decides whether an exported graph may ship.

    Nothing downstream consumes a logit: the semantic engine takes an argmax and
    polygonises the label raster. So the export has to be judged on whether any pixel
    changes class, with the logit gap kept as a drift alarm rather than as the claim.

    This is not a hypothetical split. The shared_semantic export failed a 1e-4 logits
    gate at 5.3e-4 with every one of 1,073,296 pixels agreeing on its class -- a correct
    graph rejected by a bound tuned for a shallower network. Loosening the bound alone
    would have been the wrong fix, because it also loosens the case these tests pin: an
    export whose labels really did move.
    """

    @staticmethod
    def _logits(*rows):
        # (1, classes, 1, N): one pixel per column, so a class flip is easy to state.
        import numpy as np

        return np.asarray([rows], dtype=np.float32).reshape(1, len(rows), 1, -1)

    def test_a_tiny_numeric_difference_that_changes_no_label_passes(self) -> None:
        from training.export_shared_semantic import verify_parity

        reference = self._logits([5.0, 1.0, 2.0], [0.0, 9.0, 3.0])
        candidate = reference + 4e-4  # a uniform nudge; every argmax is unmoved
        report = verify_parity([(reference, candidate)], tolerance=1e-3)
        assert report["label_disagreements"] == 0
        assert report["pixels_compared"] == 3

    def test_a_single_flipped_pixel_is_refused(self) -> None:
        """The failure the gate exists for, and the one a loose tolerance would miss."""
        import numpy as np
        import pytest as _pytest

        from training.export_shared_semantic import verify_parity

        reference = self._logits([5.0, 1.0, 2.0], [4.9, 9.0, 3.0])
        candidate = np.array(reference)
        candidate[0, 1, 0, 0] = 6.0  # class 1 now wins that pixel; class 0 did before
        with _pytest.raises(RuntimeError, match="different class"):
            verify_parity([(reference, candidate)], tolerance=1e9)

    def test_a_flip_is_refused_even_when_the_logit_gap_is_within_tolerance(self) -> None:
        """A near-tie pixel can change class on a difference far below any sane bound.

        This is why the label check cannot be replaced by a tighter number.
        """
        import numpy as np
        import pytest as _pytest

        from training.export_shared_semantic import verify_parity

        reference = self._logits([1.00000, 2.0], [1.00001, 2.0])
        candidate = np.array(reference)
        candidate[0, 0, 0, 1] = 1.00002  # moved by 1e-5, and it wins
        with _pytest.raises(RuntimeError, match="different class"):
            verify_parity([(reference, candidate)], tolerance=1e-3)

    def test_agreement_on_labels_does_not_excuse_a_wide_logit_gap(self) -> None:
        """Passing on labels alone would let a genuinely broken graph through.

        Two graphs can agree on every argmax while one of them has drifted far enough
        that the next checkpoint, or the next opset, will not.
        """
        import pytest as _pytest

        from training.export_shared_semantic import verify_parity

        reference = self._logits([9.0, 0.0], [0.0, 9.0])
        candidate = self._logits([9.5, 0.0], [0.0, 9.0])
        with _pytest.raises(RuntimeError, match="only by luck"):
            verify_parity([(reference, candidate)], tolerance=1e-3)

    def test_a_changed_output_shape_is_refused(self) -> None:
        from training.export_shared_semantic import verify_parity
        import pytest as _pytest

        reference = self._logits([1.0, 2.0], [3.0, 4.0])
        candidate = self._logits([1.0, 2.0], [3.0, 4.0], [0.0, 0.0])
        with _pytest.raises(RuntimeError, match="returns"):
            verify_parity([(reference, candidate)], tolerance=1e-3)

    def test_verifying_nothing_is_refused_rather_than_reported_as_parity(self) -> None:
        """An empty probe list must not read as "checked, and fine"."""
        import pytest as _pytest

        from training.export_shared_semantic import verify_parity

        with _pytest.raises(ValueError):
            verify_parity([], tolerance=1e-3)


class TestNonFiniteMetricsAreNotMeasurements:
    """NaN is not a number and it is not JSON either.

    The shared_semantic checkpoint carried validation loss = NaN beside a real mean_iou
    of 0.6128 -- IoU is computed from the confusion matrix and survives a tile whose loss
    is undefined, typically one where every pixel is masked out. Two things went wrong
    when that reached the manifest: a reader sees "loss: NaN" listed among the validation
    metrics as though it were one, and json.dump writes the bare token NaN, which Python
    re-reads happily and every other parser rejects.
    """

    @staticmethod
    def _manifest(metrics, tmp_path):
        from training.export_shared_semantic import build_runtime_manifest

        onnx = tmp_path / 'm.onnx'
        onnx.write_bytes(b'weights')
        return build_runtime_manifest(
            onnx_path=onnx,
            model_key='shared',
            model_version='1.0.0',
            schema={'id': 's', 'version': '1', 'classes': []},
            validation_metrics=metrics,
            inference={},
            training_origin='fixture',
        )

    def test_a_nan_is_kept_out_of_the_metrics(self, tmp_path) -> None:
        manifest = self._manifest({'mean_iou': 0.61, 'loss': float('nan')}, tmp_path)
        assert manifest['model']['validation_metrics'] == {'mean_iou': 0.61}

    def test_the_real_metric_beside_it_survives(self, tmp_path) -> None:
        """One undefined loss must not discard a measurement that IS defined."""
        manifest = self._manifest({'mean_iou': 0.61, 'loss': float('nan')}, tmp_path)
        assert manifest['model']['validation_metrics']['mean_iou'] == 0.61

    def test_the_dropped_metric_is_named_rather_than_erased(self, tmp_path) -> None:
        """Someone deciding whether to trust the model should see the loss was undefined."""
        manifest = self._manifest({'mean_iou': 0.61, 'loss': float('nan')}, tmp_path)
        assert manifest['model']['non_finite_metrics'] == ['loss']

    def test_infinity_is_treated_the_same_way(self, tmp_path) -> None:
        manifest = self._manifest({'mean_iou': 0.61, 'loss': float('inf')}, tmp_path)
        assert 'loss' not in manifest['model']['validation_metrics']
        assert manifest['model']['non_finite_metrics'] == ['loss']

    def test_a_manifest_of_nothing_but_nan_is_refused(self, tmp_path) -> None:
        """Filtering must not turn "no evidence" into "an empty metrics block"."""
        with pytest.raises(ValueError):
            self._manifest({'loss': float('nan')}, tmp_path)

    def test_the_manifest_serialises_as_portable_json(self, tmp_path) -> None:
        """The file has to be readable by parsers that are not Python's."""
        manifest = self._manifest({'mean_iou': 0.61, 'loss': float('nan')}, tmp_path)
        text = json.dumps(manifest, allow_nan=False)  # raises if any NaN survived
        assert 'NaN' not in text
