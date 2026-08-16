"""Detection class indices must not depend on the machine that built the corpus.

This exists because of a real failure. PVEL-AD was prepared on a rented Linux box and
evaluated against a copy prepared on Windows. The adapter, the source data and the
command were identical; the only difference was the order the filesystem yielded
samples in, and that order used to decide the class numbering. Linux gave finger=1,
crack=2. Windows gave crack=1, finger=2.

The consequence was not an error. The trained weights scored 0.002 and 0.014 on exactly
those two classes and sensible numbers on the other six, which reads as "the model
cannot detect its two most common defects" -- a plausible, specific, entirely wrong
conclusion that cost most of a morning and nearly got a good model discarded.

Sorting is the fix, and these tests are here so nobody restores encounter order because
it looks more natural.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.datasets.prepare import (
    DetSample,
    PreparedTask,
    _voc_to_yolo,
    _write_detection,
)


def _task(tmp_path: Path, class_names: list[str]) -> PreparedTask:
    return PreparedTask(task="t", kind="detection", root=tmp_path, class_names=list(class_names))


def _label(tmp_path: Path, name: str, rows: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _image(tmp_path: Path, name: str) -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (32, 32)).save(path)
    return path


class TestClassOrder:
    def test_indices_follow_sorted_names_not_arrival(self, tmp_path: Path) -> None:
        image = _image(tmp_path, "a.png")
        label = _label(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.1 0.1"])
        task = _task(tmp_path / "out", ["zebra", "alpha"])
        _write_detection(
            [DetSample("s1", image, label, ("zebra", "alpha"), split="train")], task, salt="s"
        )
        assert task.class_names == ["alpha", "zebra"]

    def test_two_corpora_from_the_same_classes_agree(self, tmp_path: Path) -> None:
        # The same class set introduced in opposite orders must produce the same map.
        results = []
        for order in (["finger", "crack"], ["crack", "finger"]):
            image = _image(tmp_path, f"{order[0]}.png")
            label = _label(tmp_path, f"{order[0]}.txt", ["0 0.5 0.5 0.2 0.2"])
            task = _task(tmp_path / f"out_{order[0]}", order)
            _write_detection(
                [DetSample("s", image, label, tuple(order), split="train")], task, salt="s"
            )
            results.append(task.class_names)
        assert results[0] == results[1], (
            "Two corpora built from the same class names disagree on their ordering. "
            "Weights trained on one and evaluated on the other will silently mis-score."
        )

    def test_data_yaml_matches_the_recorded_names(self, tmp_path: Path) -> None:
        image = _image(tmp_path, "b.png")
        label = _label(tmp_path, "b.txt", ["0 0.5 0.5 0.2 0.2"])
        out = tmp_path / "out"
        task = _task(out, ["delta", "bravo", "charlie"])
        _write_detection(
            [DetSample("s", image, label, ("delta", "bravo", "charlie"), split="train")],
            task,
            salt="s",
        )
        text = (out / "data.yaml").read_text(encoding="utf-8")
        for index, name in enumerate(task.class_names):
            assert f"  {index}: {name}" in text


class TestVocConversion:
    @staticmethod
    def _voc(tmp_path: Path, name: str, objects: list[tuple[str, int, int, int, int]]) -> Path:
        body = "".join(
            f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
            f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
            for n, a, b, c, d in objects
        )
        path = tmp_path / name
        path.write_text(
            f"<annotation><size><width>100</width><height>100</height></size>{body}</annotation>",
            encoding="utf-8",
        )
        return path

    def test_unlisted_classes_are_dropped_not_renumbered(self, tmp_path: Path) -> None:
        # This is how the long-tail PVEL-AD defects leave the corpus. If a dropped class
        # ever fell through to index 0 it would be silently relabelled as another defect.
        path = self._voc(tmp_path, "v.xml", [("keep", 10, 10, 50, 50), ("drop", 1, 1, 9, 9)])
        merged = ["keep"]
        index_of = {"keep": 0}
        lines = _voc_to_yolo(DetSample("s", path, path, ("keep",)), merged, index_of)
        assert len(lines) == 1
        assert lines[0].startswith("0 ")

    def test_boxes_are_normalised_against_the_declared_size(self, tmp_path: Path) -> None:
        path = self._voc(tmp_path, "w.xml", [("keep", 0, 0, 50, 100)])
        merged, index_of = ["keep"], {"keep": 0}
        line = _voc_to_yolo(DetSample("s", path, path, ("keep",)), merged, index_of)[0]
        _, cx, cy, bw, bh = line.split()
        assert float(cx) == pytest.approx(0.25)
        assert float(cy) == pytest.approx(0.5)
        assert float(bw) == pytest.approx(0.5)
        assert float(bh) == pytest.approx(1.0)

    def test_a_box_outside_the_image_is_clamped_or_dropped(self, tmp_path: Path) -> None:
        path = self._voc(tmp_path, "x.xml", [("keep", 90, 90, 200, 200)])
        merged, index_of = ["keep"], {"keep": 0}
        lines = _voc_to_yolo(DetSample("s", path, path, ("keep",)), merged, index_of)
        for line in lines:
            values = [float(v) for v in line.split()[1:]]
            assert all(0.0 <= v <= 1.0 for v in values), line
