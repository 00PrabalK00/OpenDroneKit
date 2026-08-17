"""A model that does not fit in one file must still be identified by one digest.

An ONNX graph over 2 GB of initialisers cannot hold its own weights, and torch's
exporter reaches for external data well before that limit: shared_semantic exports as a
1.1 MB graph beside a 378 MB `.onnx.data` sidecar carrying 99.7% of the model.

That breaks the guarantee this project makes about model identity. Metrics are tied to a
checkpoint by a recorded sha256, and both the exporter and the runtime used to hash the
graph file alone -- so the sidecar could be swapped for entirely different weights while
the digest, and every published number resting on it, stayed valid.

These tests pin the fix: the digest covers every file the model is made of, and the
runtime computes it the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.semantic_engine import onnx_model_files, sha256_onnx_model


@pytest.fixture
def model(tmp_path: Path) -> Path:
    """A graph with an external-data sidecar, shaped like a real export."""
    graph = tmp_path / "m.onnx"
    graph.write_bytes(b"graph")
    (tmp_path / "m.onnx.data").write_bytes(b"the actual weights")
    return graph


class TestEveryFileCounts:
    def test_the_sidecar_is_part_of_the_model(self, model) -> None:
        names = [path.name for path in onnx_model_files(model)]
        assert names == ["m.onnx", "m.onnx.data"]

    def test_the_graph_comes_first(self, model) -> None:
        assert onnx_model_files(model)[0].name == "m.onnx"

    def test_replacing_the_weights_changes_the_digest(self, model) -> None:
        """The failure the whole thing exists to prevent.

        Different weights, untouched graph. A graph-only hash reports the same model.
        """
        before = sha256_onnx_model(model)
        (model.parent / "m.onnx.data").write_bytes(b"different weights!")
        assert sha256_onnx_model(model) != before

    def test_renaming_the_sidecar_changes_the_digest(self, model) -> None:
        """A renamed sidecar does not load, so it must not hash as the same model."""
        before = sha256_onnx_model(model)
        (model.parent / "m.onnx.data").rename(model.parent / "m.onnx.weights")
        assert sha256_onnx_model(model) != before

    def test_the_digest_is_stable_across_calls(self, model) -> None:
        assert sha256_onnx_model(model) == sha256_onnx_model(model)


class TestOnlyTheModelCounts:
    def test_the_manifest_and_report_are_not_swept_in(self, model) -> None:
        """They sit in the same directory and describe the model rather than being it.

        Their names do not carry the graph's filename as a prefix, which is what keeps
        them out -- worth a test, because a looser glob would make the digest change
        every time the manifest was rewritten.
        """
        (model.parent / "m.manifest.json").write_text("{}", encoding="utf-8")
        (model.parent / "m.export.json").write_text("{}", encoding="utf-8")
        assert [path.name for path in onnx_model_files(model)] == ["m.onnx", "m.onnx.data"]

    def test_an_unrelated_model_beside_it_is_not_swept_in(self, model) -> None:
        (model.parent / "other.onnx").write_bytes(b"another model entirely")
        assert [path.name for path in onnx_model_files(model)] == ["m.onnx", "m.onnx.data"]

    def test_a_single_file_model_still_works(self, tmp_path: Path) -> None:
        # Most models here have no sidecar at all; they must not need one.
        alone = tmp_path / "small.onnx"
        alone.write_bytes(b"self contained")
        assert onnx_model_files(alone) == [alone]
        assert len(sha256_onnx_model(alone)) == 64
