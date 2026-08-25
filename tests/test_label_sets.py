"""Boxes a user drew become a corpus, or the build refuses and says what is wrong.

ai.custom_training claimed users could label, split, train and deploy. They could do
three of those. Labelling meant arriving with YOLO text files already written, which is
not labelling -- it is having labelled. This is the missing half, and its refusals carry
the weight: a corpus is built once and its metric is read many times, so a box that
cannot be learned from has to stop the build rather than produce a warning nobody reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.custom_training import CorpusRefused
from core.label_sets import (
    Box,
    LabelledRegion,
    build_detection_corpus,
    regions_from_payload,
)


def image_bytes(seed: int) -> bytes:
    """A distinct PNG per seed, so content digests differ the way real photographs do."""
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (seed % 256, (seed * 7) % 256, (seed * 13) % 256)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


@pytest.fixture
def corpus_images(tmp_path):
    """Enough images that the split can put boxes in every partition."""
    paths = []
    for index in range(40):
        path = tmp_path / f"frame_{index:03d}.png"
        path.write_bytes(image_bytes(index))
        paths.append(path)
    return paths


def region(path: Path, label: str = "crack", count: int = 1) -> LabelledRegion:
    boxes = [
        Box(label=label, x=0.1 + 0.05 * i, y=0.1, width=0.2, height=0.2)
        for i in range(count)
    ]
    return LabelledRegion(path=path, boxes=boxes)


class TestABoxHasToBeADrawableThing:
    def test_a_zero_area_box_is_refused(self) -> None:
        """Almost always a click that was meant to be a drag."""
        with pytest.raises(CorpusRefused, match="zero area"):
            Box(label="crack", x=0.5, y=0.5, width=0.0, height=0.1)

    def test_a_box_outside_the_image_is_refused(self) -> None:
        with pytest.raises(CorpusRefused, match="outside the image"):
            Box(label="crack", x=0.9, y=0.5, width=0.4, height=0.1)

    def test_an_unlabelled_box_is_refused(self) -> None:
        with pytest.raises(CorpusRefused, match="needs a class label"):
            Box(label="  ", x=0.1, y=0.1, width=0.2, height=0.2)

    def test_a_box_that_exactly_fills_the_image_is_allowed(self) -> None:
        """The whole frame is a legitimate annotation, and floating point must not
        turn 1.0 into a refusal."""
        assert Box(label="crack", x=0.0, y=0.0, width=1.0, height=1.0)

    def test_yolo_lines_are_centre_based(self) -> None:
        line = Box(label="crack", x=0.1, y=0.2, width=0.4, height=0.2).to_yolo(3)
        index, cx, cy, width, height = line.split()
        assert index == "3"
        assert float(cx) == pytest.approx(0.3)
        assert float(cy) == pytest.approx(0.3)
        assert float(width) == pytest.approx(0.4)


class TestTheCorpusRefusesWhatCannotBeLearned:
    def test_too_few_boxes_for_a_class_is_refused(self, tmp_path, corpus_images) -> None:
        regions = [region(path) for path in corpus_images[:5]]
        with pytest.raises(CorpusRefused, match="Too few boxes"):
            build_detection_corpus(regions, tmp_path / "out")

    def test_an_image_with_no_boxes_stops_the_build(self, tmp_path, corpus_images) -> None:
        """The common cause is an image someone opened and forgot, and training on it
        teaches the model the defect is absent."""
        regions = [region(path, count=2) for path in corpus_images[:20]]
        regions.append(LabelledRegion(path=corpus_images[20], boxes=[]))
        with pytest.raises(CorpusRefused, match="not marked as deliberately empty"):
            build_detection_corpus(regions, tmp_path / "out")

    def test_an_image_marked_empty_on_purpose_is_accepted(self, tmp_path, corpus_images) -> None:
        """A confirmed negative is real evidence; an accident is not. Only the user
        can tell them apart, so the distinction is carried explicitly."""
        regions = [region(path, count=2) for path in corpus_images[:30]]
        regions.append(
            LabelledRegion(path=corpus_images[30], boxes=[], confirmed_empty=True)
        )
        report = build_detection_corpus(regions, tmp_path / "out")
        assert report["empty_images"] == 1

    def test_a_corpus_of_only_negatives_is_refused(self, tmp_path, corpus_images) -> None:
        regions = [
            LabelledRegion(path=path, boxes=[], confirmed_empty=True)
            for path in corpus_images[:20]
        ]
        with pytest.raises(CorpusRefused, match="nothing to detect"):
            build_detection_corpus(regions, tmp_path / "out")

    def test_a_missing_file_is_refused(self, tmp_path, corpus_images) -> None:
        regions = [region(path, count=2) for path in corpus_images[:20]]
        regions.append(region(tmp_path / "never_written.png", count=2))
        with pytest.raises(CorpusRefused, match="not on disk"):
            build_detection_corpus(regions, tmp_path / "out")

    def test_a_class_with_too_few_held_out_boxes_is_refused(self, tmp_path, corpus_images) -> None:
        """Below the floor, recall is an accident of which images landed in validation.

        Built by giving one class plenty of boxes concentrated in few images, so it
        clears the training floor and still cannot be measured.
        """
        regions = [region(path, label="crack", count=2) for path in corpus_images[:20]]
        regions.append(region(corpus_images[21], label="spall", count=12))
        with pytest.raises(CorpusRefused, match="held-out boxes to measure"):
            build_detection_corpus(regions, tmp_path / "out")


class TestWhatItWrites:
    @pytest.fixture
    def built(self, tmp_path, corpus_images):
        regions = [region(path, count=2) for path in corpus_images]
        return build_detection_corpus(regions, tmp_path / "out"), tmp_path / "out"

    def test_it_writes_a_yolo_layout_a_trainer_can_read(self, built) -> None:
        report, root = built
        for split in ("train", "val", "test"):
            assert (root / split / "images").is_dir()
            assert (root / split / "labels").is_dir()
        assert (root / "data.yaml").is_file()

    def test_the_data_yaml_names_every_class_in_index_order(self, built) -> None:
        report, root = built
        text = (root / "data.yaml").read_text(encoding="utf-8")
        assert f"nc: {len(report['classes'])}" in text
        for index, name in enumerate(report["classes"]):
            assert f"  {index}: {name}" in text

    def test_every_image_has_a_label_file(self, built) -> None:
        report, root = built
        for split in ("train", "val", "test"):
            images = sorted((root / split / "images").glob("*.png"))
            for image in images:
                label = root / split / "labels" / f"{image.stem}.txt"
                assert label.is_file(), f"{image.name} has no label file"

    def test_the_report_is_written_beside_the_corpus(self, built) -> None:
        report, root = built
        saved = json.loads((root / "corpus_report.json").read_text(encoding="utf-8"))
        assert saved["total_boxes"] == report["total_boxes"]
        assert "labelling" in saved["reading_note"]

    def test_the_same_image_under_two_names_is_counted_once(self, tmp_path, corpus_images) -> None:
        """Leakage arrives as one photograph saved twice, which a filename check misses."""
        copy = tmp_path / "renamed_copy.png"
        copy.write_bytes(corpus_images[0].read_bytes())
        regions = [region(path, count=2) for path in corpus_images]
        regions.append(region(copy, count=2))
        report = build_detection_corpus(regions, tmp_path / "out")
        assert report["duplicates_removed"] == 1

    def test_splits_are_stable_against_added_images(self, tmp_path, corpus_images) -> None:
        """Keyed on content, so adding photographs does not reshuffle what was held out
        and quietly invalidate every earlier number."""
        first = build_detection_corpus(
            [region(path, count=2) for path in corpus_images], tmp_path / "a"
        )
        extra = tmp_path / "extra.png"
        extra.write_bytes(image_bytes(999))
        second = build_detection_corpus(
            [region(path, count=2) for path in corpus_images] + [region(extra, count=2)],
            tmp_path / "b",
        )
        held_out_first = sorted(p.name for p in (tmp_path / "a/val/images").glob("*.png"))
        held_out_second = sorted(p.name for p in (tmp_path / "b/val/images").glob("*.png"))
        assert set(held_out_first) <= set(held_out_second)


class TestThePayloadFromTheUI:
    def test_it_parses_what_the_canvas_posts(self) -> None:
        regions = regions_from_payload([
            {"path": "a.png", "boxes": [
                {"label": "crack", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}]},
            {"path": "b.png", "boxes": [], "confirmed_empty": True},
        ])
        assert regions[0].boxes[0].label == "crack"
        assert regions[1].confirmed_empty is True

    def test_a_payload_without_a_path_is_refused(self) -> None:
        with pytest.raises(CorpusRefused, match="needs a path"):
            regions_from_payload([{"boxes": []}])

    def test_malformed_geometry_is_refused_at_the_boundary(self) -> None:
        """The UI clamps, but the API must not trust that it did."""
        with pytest.raises(CorpusRefused):
            regions_from_payload([
                {"path": "a.png", "boxes": [
                    {"label": "crack", "x": 0.9, "y": 0.9, "width": 0.5, "height": 0.5}]},
            ])


class TestTheApiExposesIt:
    """The desktop bridge is how the labelling canvas reaches the builder."""

    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("boxes", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_drawn_boxes_build_a_corpus_through_the_api(self, api, tmp_path, corpus_images) -> None:
        labelled = [
            {"path": str(path), "boxes": [
                {"label": "crack", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                {"label": "crack", "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2},
            ]}
            for path in corpus_images
        ]
        result = api.build_detection_corpus(labelled, str(tmp_path / "api_out"))
        assert result["ok"], result.get("error")
        assert result["classes"] == ["crack"]
        assert Path(result["data_yaml"]).is_file()

    def test_a_refusal_reaches_the_caller_as_a_reason(self, api, tmp_path, corpus_images) -> None:
        labelled = [
            {"path": str(path), "boxes": [
                {"label": "crack", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}]}
            for path in corpus_images[:4]
        ]
        result = api.build_detection_corpus(labelled, str(tmp_path / "thin_out"))
        assert not result["ok"]
        assert "Too few boxes" in result["error"]

    def test_bad_geometry_is_refused_at_the_api_boundary(self, api, tmp_path, corpus_images) -> None:
        """The canvas clamps, but the API must not assume the caller was the canvas."""
        labelled = [{"path": str(corpus_images[0]), "boxes": [
            {"label": "crack", "x": 0.9, "y": 0.9, "width": 0.5, "height": 0.5}]}]
        result = api.build_detection_corpus(labelled, str(tmp_path / "bad_out"))
        assert not result["ok"]
        assert "outside the image" in result["error"]
