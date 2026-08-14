"""Dataset preparation: split stability, label decoding, and class-name hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from training.datasets.prepare import (
    TASKS,
    _binarise,
    _clean_class_name,
    _parse_bsds_seg,
    _read_yolo_class_names,
    deterministic_split,
    read_yolo_license,
    resolve_tasks,
)


class TestDeterministicSplit:
    def test_same_sample_always_lands_in_the_same_split(self):
        """Re-running prepare must never move a sample from val into train.

        If it did, every metric produced before the move would be silently
        invalidated by validation data leaking into training.
        """
        for sample in [f"img_{i:04d}" for i in range(200)]:
            first = deterministic_split(sample, salt="crack_seg")
            second = deterministic_split(sample, salt="crack_seg")
            assert first == second

    def test_salt_separates_tasks(self):
        assignments_a = [deterministic_split(f"s{i}", salt="a") for i in range(200)]
        assignments_b = [deterministic_split(f"s{i}", salt="b") for i in range(200)]
        assert assignments_a != assignments_b

    def test_proportions_are_close_to_the_configured_fractions(self):
        from training.datasets.prepare import TEST_FRACTION, VAL_FRACTION

        samples = [f"img_{i:05d}" for i in range(5000)]
        counts = {"train": 0, "val": 0, "test": 0}
        for sample in samples:
            counts[deterministic_split(sample, salt="crack_seg")] += 1

        total = len(samples)
        assert counts["test"] / total == pytest.approx(TEST_FRACTION, abs=0.02)
        assert counts["val"] / total == pytest.approx(VAL_FRACTION, abs=0.02)


class TestBsdsSegParsing:
    def test_run_length_rows_decode_to_a_mask(self, tmp_path):
        """CrackForest ships BSDS `.seg` text; label 1 is crack, bounds inclusive."""
        seg = tmp_path / "001.seg"
        seg.write_text(
            "width 10\nheight 4\nsegments 2\ndata\n"
            "0 0 0 9\n"
            "1 1 2 4\n"
            "0 2 0 9\n"
            "0 3 0 9\n",
            encoding="utf-8",
        )
        mask = _parse_bsds_seg(seg)
        assert mask is not None
        assert mask.shape == (4, 10)
        # Columns 2..4 inclusive on row 1.
        assert mask[1, 2] == 255 and mask[1, 4] == 255
        assert mask[1, 5] == 0
        assert mask[0].sum() == 0

    def test_missing_header_yields_nothing_rather_than_a_wrong_mask(self, tmp_path):
        seg = tmp_path / "bad.seg"
        seg.write_text("data\n0 0 0 9\n", encoding="utf-8")
        assert _parse_bsds_seg(seg) is None


class TestMaskBinarisation:
    @pytest.mark.parametrize(
        "array",
        [
            np.array([[0, 255]], dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8) * 255,
            np.dstack([np.array([[0, 255]], dtype=np.uint8)] * 3),
        ],
    )
    def test_any_label_encoding_collapses_to_zero_or_255(self, array):
        result = _binarise(array)
        assert set(np.unique(result)) <= {0, 255}
        assert result.dtype == np.uint8

    def test_antialiased_edges_are_thresholded_at_mid_grey(self):
        result = _binarise(np.array([[0, 100, 128, 200, 255]], dtype=np.uint8))
        assert list(result[0]) == [0, 0, 255, 255, 255]


class TestClassNames:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0 Efflorescence", "Efflorescence"),
            ("12  CorrosionStain", "CorrosionStain"),
            ("Crack", "Crack"),
            ("  Spallation ", "Spallation"),
            ("3", "3"),
        ],
    )
    def test_index_prefixes_are_stripped(self, raw, expected):
        """CODEBRIM's mirror bakes the class index into the name."""
        assert _clean_class_name(raw) == expected

    def test_inline_list_form_is_parsed(self, tmp_path):
        (tmp_path / "data.yaml").write_text(
            "names: ['0 Crack', 'Spall']\nnc: 2\n", encoding="utf-8"
        )
        assert _read_yolo_class_names(tmp_path) == ("Crack", "Spall")

    def test_block_list_form_is_parsed(self, tmp_path):
        (tmp_path / "data.yaml").write_text(
            "names:\n- 0 Crack\n- 1 Spall\nnc: 2\n", encoding="utf-8"
        )
        assert _read_yolo_class_names(tmp_path) == ("Crack", "Spall")

    def test_licence_is_read_from_the_export(self, tmp_path):
        (tmp_path / "data.yaml").write_text(
            "names:\n- Crack\nroboflow:\n  license: CC BY 4.0\n", encoding="utf-8"
        )
        assert read_yolo_license(tmp_path) == "CC BY 4.0"


class TestTaskCatalogue:
    def test_groups_expand_without_duplicates(self):
        specs = resolve_tasks(["crack", "crack_seg"])
        names = [spec.name for spec in specs]
        assert len(names) == len(set(names))

    def test_unknown_task_is_rejected(self):
        with pytest.raises(KeyError):
            resolve_tasks(["not_a_task"])

    def test_every_task_declares_a_known_kind(self):
        for spec in TASKS.values():
            assert spec.kind in {"segmentation", "classification", "detection"}
            assert spec.datasets, f"{spec.name} lists no source datasets"
