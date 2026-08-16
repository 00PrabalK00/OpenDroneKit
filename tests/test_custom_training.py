"""A user's own corpus is refused unless it can produce a measurable model.

Someone labelling their own defects is doing the most valuable thing in this toolkit and
the most fragile. The failure modes are silent: a corpus builds, trains and reports
without anyone noticing that one class had four examples, that the same photograph sat
in both training and validation, or that the headline number came from eleven images.

None of those look like errors. They look like a model that did unusually well.

So the refusals are the feature, and these tests are mostly about them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.custom_training import (
    CorpusRefused,
    LabelledImage,
    build_custom_corpus,
    split_for,
)


def write_image(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def corpus(tmp_path: Path, spec: dict[str, int], *, prefix: str = "") -> list[LabelledImage]:
    """`spec` maps label to how many distinct images that label gets."""
    samples = []
    for label, count in spec.items():
        for index in range(count):
            path = write_image(tmp_path, f"{prefix}{label}_{index}.jpg",
                               f"{prefix}{label}-{index}".encode())
            samples.append(LabelledImage(path=path, label=label))
    return samples


class TestSplittingIsStable:
    def test_the_same_image_lands_in_the_same_split(self) -> None:
        """Adding photographs must not reshuffle what was previously held out.

        A reshuffle silently invalidates every metric measured before it, and nothing
        about the new run looks different.
        """
        first = split_for("abc123", salt="s")
        assert split_for("abc123", salt="s") == first

    def test_different_salts_give_different_corpora(self) -> None:
        digests = [f"{i:040x}" for i in range(200)]
        a = [split_for(d, salt="one") for d in digests]
        b = [split_for(d, salt="two") for d in digests]
        assert a != b

    def test_the_split_is_keyed_on_content_not_name(self, tmp_path: Path) -> None:
        # Same bytes under two names must resolve identically, which is what makes the
        # duplicate check able to catch a renamed copy.
        one = LabelledImage(write_image(tmp_path, "a.jpg", b"same"), "crack")
        two = LabelledImage(write_image(tmp_path, "b.jpg", b"same"), "crack")
        assert one.content_digest() == two.content_digest()


class TestStratification:
    """Every class keeps its share of every split.

    A whole-corpus hash is deterministic but not proportional. On 30 images per class it
    handed one class a single validation example -- enough to make its recall 0.0 or 1.0
    and get the corpus refused for the wrong reason.
    """

    def test_each_class_appears_in_each_split(self, tmp_path: Path) -> None:
        splits, _ = build_custom_corpus(
            corpus(tmp_path, {"crack": 30, "spall": 30}), salt="s"
        )
        for name, entries in splits.items():
            labels = {s.label for s in entries}
            assert labels == {"crack", "spall"}, f"{name} is missing a class: {labels}"

    def test_proportions_hold_within_a_class(self, tmp_path: Path) -> None:
        splits, _ = build_custom_corpus(
            corpus(tmp_path, {"crack": 100, "spall": 100}), salt="s"
        )
        for label in ("crack", "spall"):
            train = sum(1 for s in splits["train"] if s.label == label)
            val = sum(1 for s in splits["val"] if s.label == label)
            assert train == 70, f"{label} train share drifted: {train}"
            assert val == 20, f"{label} val share drifted: {val}"

    def test_no_sample_is_lost_or_duplicated_to_rounding(self, tmp_path: Path) -> None:
        samples = corpus(tmp_path, {"crack": 37, "spall": 13})
        splits, report = build_custom_corpus(samples, salt="s", min_validation_per_class=2)
        assert sum(len(v) for v in splits.values()) == len(samples)

    def test_the_assignment_is_stable_across_runs(self, tmp_path: Path) -> None:
        samples = corpus(tmp_path, {"crack": 40, "spall": 40})
        first, _ = build_custom_corpus(samples, salt="s")
        second, _ = build_custom_corpus(samples, salt="s")
        for name in first:
            assert [s.path for s in first[name]] == [s.path for s in second[name]]


class TestRefusals:
    def test_an_empty_corpus_is_refused(self) -> None:
        with pytest.raises(CorpusRefused, match="No labelled images"):
            build_custom_corpus([], salt="s")

    def test_a_single_class_corpus_is_refused(self, tmp_path: Path) -> None:
        """One class scores perfectly by answering the only thing it knows."""
        with pytest.raises(CorpusRefused, match="at least two classes"):
            build_custom_corpus(corpus(tmp_path, {"crack": 40}), salt="s")

    def test_a_thin_class_is_refused_with_its_count(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusRefused, match=r"spall \(4\)"):
            build_custom_corpus(corpus(tmp_path, {"crack": 40, "spall": 4}), salt="s")

    def test_an_unlabelled_image_is_refused(self, tmp_path: Path) -> None:
        samples = corpus(tmp_path, {"crack": 20, "spall": 20})
        samples.append(LabelledImage(write_image(tmp_path, "x.jpg", b"x"), "  "))
        with pytest.raises(CorpusRefused, match="no label"):
            build_custom_corpus(samples, salt="s")

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        samples = corpus(tmp_path, {"crack": 20, "spall": 20})
        samples.append(LabelledImage(tmp_path / "absent.jpg", "crack"))
        with pytest.raises(CorpusRefused, match="does not exist"):
            build_custom_corpus(samples, salt="s")

    def test_the_same_image_with_two_labels_is_refused(self, tmp_path: Path) -> None:
        """Guessing which label is right bakes a mistake into every model trained on it."""
        samples = corpus(tmp_path, {"crack": 20, "spall": 20})
        samples.append(LabelledImage(write_image(tmp_path, "dup_a.jpg", b"shared"), "crack"))
        samples.append(LabelledImage(write_image(tmp_path, "dup_b.jpg", b"shared"), "spall"))
        with pytest.raises(CorpusRefused, match="different labels"):
            build_custom_corpus(samples, salt="s")

    def test_a_class_that_cannot_be_measured_is_refused(self, tmp_path: Path) -> None:
        """A class with one held-out example has a recall of 0.0 or 1.0, not a measurement."""
        with pytest.raises(CorpusRefused, match="too few validation examples"):
            build_custom_corpus(
                corpus(tmp_path, {"crack": 60, "spall": 12}),
                salt="s",
                min_validation_per_class=8,
            )


class TestLeakage:
    def test_duplicates_are_removed_before_splitting(self, tmp_path: Path) -> None:
        samples = corpus(tmp_path, {"crack": 30, "spall": 30})
        # The same photograph saved twice under different names -- the common case.
        original = samples[0]
        samples.append(LabelledImage(
            write_image(tmp_path, "copy.jpg", Path(original.path).read_bytes()),
            original.label,
        ))
        splits, report = build_custom_corpus(samples, salt="s")
        assert report.duplicates_removed == 1
        assert any("inflated" in w for w in report.warnings)

    def test_no_digest_appears_in_two_splits(self, tmp_path: Path) -> None:
        splits, _ = build_custom_corpus(corpus(tmp_path, {"crack": 60, "spall": 60}), salt="s")
        seen: dict[str, str] = {}
        for name, entries in splits.items():
            for sample in entries:
                digest = sample.content_digest()
                assert seen.get(digest, name) == name
                seen[digest] = name


class TestTheReportIsHonest:
    def test_every_class_is_counted_per_split(self, tmp_path: Path) -> None:
        splits, report = build_custom_corpus(
            corpus(tmp_path, {"crack": 60, "spall": 60}), salt="s"
        )
        assert set(report.classes) == {"crack", "spall"}
        assert report.counts["train"]
        assert report.counts["val"]

    def test_imbalance_is_named_rather_than_refused(self, tmp_path: Path) -> None:
        """Imbalance is a fact of defect data; refusing it would block the real case."""
        _, report = build_custom_corpus(
            corpus(tmp_path, {"sound": 400, "crack": 30}), salt="s"
        )
        assert any("per-class recall" in w for w in report.warnings)

    def test_a_small_corpus_says_it_is_a_first_indication(self, tmp_path: Path) -> None:
        _, report = build_custom_corpus(corpus(tmp_path, {"crack": 30, "spall": 30}), salt="s")
        assert any("first indication" in w for w in report.warnings)

    def test_the_note_says_counts_are_of_labels_not_of_the_world(self, tmp_path: Path) -> None:
        _, report = build_custom_corpus(corpus(tmp_path, {"crack": 60, "spall": 60}), salt="s")
        note = report.to_dict()["reading_note"].lower()
        assert "labelling" in note

    def test_totals_match_the_split_contents(self, tmp_path: Path) -> None:
        splits, report = build_custom_corpus(
            corpus(tmp_path, {"crack": 60, "spall": 60}), salt="s"
        )
        assert sum(len(v) for v in splits.values()) == report.total_samples


class TestTheApiExposesIt:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("custom", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_a_good_corpus_builds_through_the_api(self, api, tmp_path) -> None:
        labelled = [{"path": str(s.path), "label": s.label}
                    for s in corpus(tmp_path, {"crack": 40, "spall": 40})]
        result = api.build_training_corpus(labelled)
        assert result["ok"], result.get("error")
        assert set(result["classes"]) == {"crack", "spall"}
        assert result["splits"]["val"]

    def test_a_thin_class_is_refused_through_the_api(self, api, tmp_path) -> None:
        labelled = [{"path": str(s.path), "label": s.label}
                    for s in corpus(tmp_path, {"crack": 40, "spall": 3}, prefix="api_")]
        result = api.build_training_corpus(labelled)
        assert not result["ok"]
        assert "too few examples" in result["error"]

    def test_the_warnings_reach_the_caller(self, api, tmp_path) -> None:
        labelled = [{"path": str(s.path), "label": s.label}
                    for s in corpus(tmp_path, {"sound": 400, "crack": 30}, prefix="warn_")]
        result = api.build_training_corpus(labelled)
        assert result["ok"], result.get("error")
        assert any("per-class recall" in w for w in result["warnings"])
