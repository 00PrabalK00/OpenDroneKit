"""A read-only corpus must still be trainable.

This exists because of four consecutive Kaggle failures that produced no diagnosis
between them. Every kernel reported a bare non-zero exit, Kaggle's own captured log came
back zero bytes, and the standing hypothesis was that a P100 could not fit the model --
so batch size and image size were cut, twice, for nothing.

The actual fault was one line: the trainer repaired the corpus's ``data.yaml`` in place,
and a Kaggle dataset is mounted read-only under /kaggle/input.

    OSError: [Errno 30] Read-only file system: '/kaggle/input/odk-corrosion-det-corpus/data.yaml'

The lesson worth keeping is not about YAML. A trainer that can only run where it may
write to the dataset is a trainer that cannot run on any managed platform, and the cost
of finding that out was paid in silence rather than in an error message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.train_det import _localise_data_yaml

CORPUS_YAML = "path: D:/Projects/OpenDroneKit/training/data/prepared/x\ntrain: train\nval: val\n"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "data.yaml").write_text(CORPUS_YAML, encoding="utf-8")
    return root / "data.yaml"


class TestWritableCorpus:
    def test_the_stale_absolute_path_is_repaired_in_place(self, corpus: Path) -> None:
        used = _localise_data_yaml(corpus, corpus.parent / "runs")
        assert used == corpus
        assert corpus.read_text(encoding="utf-8").splitlines()[0] == "path: ."

    def test_an_already_correct_path_is_left_alone(self, corpus: Path) -> None:
        corpus.write_text("path: .\ntrain: train\n", encoding="utf-8")
        used = _localise_data_yaml(corpus, corpus.parent / "runs")
        assert used == corpus
        assert corpus.read_text(encoding="utf-8") == "path: .\ntrain: train\n"


class TestReadOnlyCorpus:
    """The Kaggle case: /kaggle/input cannot be written to."""

    @pytest.fixture
    def read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import training.train_det as module

        monkeypatch.setattr(module.os, "access", lambda *args, **kwargs: False)

    def test_training_uses_a_repaired_copy_elsewhere(
        self, corpus: Path, read_only: None, tmp_path: Path
    ) -> None:
        workdir = tmp_path / "runs"
        used = _localise_data_yaml(corpus, workdir)
        assert used != corpus, "a read-only corpus was repaired in place; this is the EROFS crash"
        assert used.parent == workdir

    def test_the_source_corpus_is_not_modified(
        self, corpus: Path, read_only: None, tmp_path: Path
    ) -> None:
        _localise_data_yaml(corpus, tmp_path / "runs")
        assert corpus.read_text(encoding="utf-8") == CORPUS_YAML

    def test_the_copy_records_an_absolute_path_to_the_data(
        self, corpus: Path, read_only: None, tmp_path: Path
    ) -> None:
        """"path: ." in a relocated copy would point at the copy, not at the images.

        Ultralytics resolves a relative ``path`` against the yaml's own directory, so the
        copy must name the corpus outright. Getting this wrong fails as "no images found"
        rather than as a path error, which is a slower thing to diagnose.
        """
        used = _localise_data_yaml(corpus, tmp_path / "runs")
        recorded = used.read_text(encoding="utf-8").splitlines()[0].split(":", 1)[1].strip()
        assert Path(recorded).resolve() == corpus.parent.resolve()

    def test_the_rest_of_the_corpus_definition_survives_the_copy(
        self, corpus: Path, read_only: None, tmp_path: Path
    ) -> None:
        used = _localise_data_yaml(corpus, tmp_path / "runs")
        text = used.read_text(encoding="utf-8")
        assert "train: train" in text
        assert "val: val" in text

    def test_a_write_that_fails_despite_looking_writable_still_falls_back(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.access can report a directory writable when the write still fails.

        It consults permission bits, which say nothing about a read-only mount. The
        fallback has to be driven by the failed write, not only by the check.
        """
        import training.train_det as module

        monkeypatch.setattr(module.os, "access", lambda *args, **kwargs: True)
        original = Path.write_text

        def refuse(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self == corpus:
                raise OSError(30, "Read-only file system")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", refuse)
        used = _localise_data_yaml(corpus, tmp_path / "runs")
        assert used != corpus
