"""The things that killed three rented training sessions, checked on a laptop.

shared_semantic failed on Kaggle three times in a row. Not once for a modelling reason:

  1. The corpus was uploaded correctly and the kernel's corpus finder only understood
     per-split directory layouts, so it walked past a perfectly good manifest and
     reported the dataset missing.
  2. The DINOv2 checkpoint is a 350 MB binary that is not in the repository, so a fresh
     clone had the code and not the weights and raised FileNotFoundError after the
     corpus had uploaded and the GPU had been paid for.
  3. `--source github` was handed the LOCAL PATH from the config, and torch.hub tried to
     read `training/sources/dinov2` as owner/repo:
     `ValueError: too many values to unpack (expected 2)`.

Every one of them is a launch-contract mistake -- an argument, a path or a file that had
to line up before a single batch could run -- and every one was discoverable in
milliseconds on this machine. They cost hours each instead because nothing checked them
until a remote GPU did.

These tests are that check. None of them needs a GPU, a network or a corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "kaggle_kernel.py"
MODEL = REPO_ROOT / "training" / "shared_semantic_model.py"
CONFIG = REPO_ROOT / "training" / "configs" / "shared_semantic_dinov2_vitb14.yaml"


@pytest.fixture(scope="module")
def kernel_script() -> str:
    """The script the generator would actually push for the semantic config."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.kaggle_kernel import kernel_script as build

    return build("shared_semantic_dinov2_vitb14", "semantic",
                 "prabalkhare00/odk-shared-semantic-corpus")


class TestTheGeneratedCommandMatchesTheTrainerCLI:
    """Failure 3, and the family it belongs to."""

    def test_the_semantic_trainer_is_given_a_corpus_not_a_data_root(self, kernel_script) -> None:
        # train_shared_semantic takes --corpus and --run-dir; the other trainers take
        # --data-root and --output-dir. Passing the wrong pair fails at argparse, after
        # the repo clone and the torch reinstall have already been paid for.
        assert "--corpus" in kernel_script
        assert "--run-dir" in kernel_script
        assert "--data-root" not in kernel_script.split('if "{trainer}"')[-1][:600]

    def test_the_corpus_argument_points_at_the_manifest(self, kernel_script) -> None:
        assert "corpus.json" in kernel_script

    def test_every_flag_the_kernel_passes_is_one_the_trainer_accepts(self, kernel_script) -> None:
        """The check that generalises: no invented flags."""
        trainer = (REPO_ROOT / "training" / "train_shared_semantic.py").read_text(encoding="utf-8")
        accepted = set(re.findall(r"add_argument\('(--[a-z0-9-]+)'", trainer))
        accepted |= set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', trainer))
        block = kernel_script.split('if "{trainer}" == "training.train_shared_semantic":')[-1]
        block = block.split("else:")[0]
        used = set(re.findall(r'"(--[a-z0-9-]+)"', block))
        unknown = used - accepted
        assert not unknown, f"kernel passes flags the trainer does not accept: {sorted(unknown)}"


class TestTheCorpusFinderKnowsThisShape:
    """Failure 1."""

    def test_a_manifest_only_corpus_is_recognised(self, kernel_script) -> None:
        # A semantic corpus has no train/ val/ test/ directories -- the split lives in
        # the manifest. The finder must know that shape or it reports a good upload as
        # a missing dataset, which sends you to fix the upload.
        body = kernel_script.split("def looks_like_corpus")[-1].split("found = None")[0]
        assert "corpus.json" in body


class TestTheEncoderCanActuallyLoad:
    """Failures 2 and 3, checked without downloading anything."""

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return MODEL.read_text(encoding="utf-8")

    def test_a_missing_checkpoint_is_fetchable_when_the_source_is_github(self, source) -> None:
        # The weights are not in the repository, so a fresh clone must be able to get
        # them rather than raising after the corpus has uploaded.
        assert "fetch_weights" in source
        assert "pretrained=fetch_weights" in source

    def test_a_missing_checkpoint_still_refuses_when_there_is_nothing_to_fetch_from(self, source) -> None:
        assert "source != 'github'" in source

    def test_a_local_path_is_not_handed_to_github(self, source) -> None:
        """The reconciliation. Without it torch.hub reads a path as owner/repo."""
        assert "count('/') != 1" in source
        assert "facebookresearch/dinov2" in source

    def test_the_substitution_is_announced(self, source) -> None:
        # Quietly swapping the encoder source would hide a real configuration mistake.
        block = source.split("count('/') != 1")[-1][:400]
        assert "print(" in block

    def test_the_config_still_points_at_the_offline_clone(self) -> None:
        """Offline stays the default: an air-gapped site must not need a network."""
        config = CONFIG.read_text(encoding="utf-8")
        assert "encoder_source: training/sources/dinov2" in config
        assert "encoder_source_type: local" in config


class TestTheKernelReportsRatherThanVanishing:
    """Why any of this was diagnosable at all."""

    def test_the_kernel_tees_its_own_log(self, kernel_script) -> None:
        # Kaggle's captured log came back zero bytes for four consecutive failures. The
        # tee into /kaggle/working is the only reason these three causes were ever read.
        assert "run.log" in kernel_script
        assert "/kaggle/working" in kernel_script

    def test_resources_are_printed_around_training(self, kernel_script) -> None:
        # So an OOM can be told apart from a crash without guessing.
        assert "report_resources" in kernel_script


class TestEveryConfigReachesATrainerThatAcceptsIt:
    """The check that generalises past shared_semantic.

    Routing used to be substring matching over the config text, which failed three ways:
    it read words out of COMMENTS (a comment mentioning class_names sent a binary
    PV-extent config to the multiclass trainer), it could not tell change detection from
    segmentation because they share an encoder, and it sent a corrosion severity config
    to the binary trainer, which rejected num_classes after the corpus had uploaded and
    the GPU had been paid for.

    Every one of those surfaces identically -- "Unknown config keys" on a rented machine,
    minutes into a session. This runs on a laptop in under a second.
    """

    @staticmethod
    def _configs():
        return sorted((REPO_ROOT / "training" / "configs").glob("*.yaml"))

    def test_there_are_configs_to_check(self) -> None:
        # Guards the whole class: a glob that silently matched nothing would pass every
        # test below while checking nothing at all.
        assert len(self._configs()) >= 10

    def test_every_config_routes_to_a_trainer_that_accepts_all_its_keys(self) -> None:
        import dataclasses
        import importlib
        import sys

        yaml = pytest.importorskip("yaml")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.kaggle_kernel import TRAINER_FOR_KIND, infer_kind

        failures = []
        for path in self._configs():
            text = path.read_text(encoding="utf-8")
            kind = infer_kind(text)
            module = importlib.import_module(TRAINER_FOR_KIND[kind])
            config_class = next(
                (v for v in vars(module).values()
                 if dataclasses.is_dataclass(v) and v.__name__.endswith("Config")),
                None,
            )
            if config_class is None:
                continue
            unknown = set(yaml.safe_load(text) or {}) - set(config_class.__dataclass_fields__)
            if unknown:
                failures.append(f"{path.stem} -> {kind} rejects {sorted(unknown)}")
        assert not failures, "configs routed to a trainer that will refuse them:\n" + "\n".join(failures)

    def test_a_segformer_config_never_routes_to_the_classifier(self) -> None:
        """A nvidia/mit encoder is a segmentation backbone. That is not up for inference.

        The classifier's config happens to accept every key a SegFormer config sets, so
        matching on keys alone sent segmentation work to the classifier.
        """
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.kaggle_kernel import infer_kind

        for path in self._configs():
            text = path.read_text(encoding="utf-8")
            if "model_id: nvidia/mit-" in text:
                assert infer_kind(text) in {"segmentation", "segmentation_multiclass", "change"}, (
                    f"{path.stem} routed away from the segmentation family"
                )

    def test_change_detection_is_recognised(self) -> None:
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.kaggle_kernel import infer_kind

        mining = REPO_ROOT / "training" / "configs" / "mining_change_b2.yaml"
        assert infer_kind(mining.read_text(encoding="utf-8")) == "change"

    def test_a_multiclass_config_is_not_sent_to_the_binary_trainer(self) -> None:
        """The corrosion failure, pinned."""
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.kaggle_kernel import infer_kind

        corrosion = REPO_ROOT / "training" / "configs" / "corrosion_segformer_b2_cs.yaml"
        assert infer_kind(corrosion.read_text(encoding="utf-8")) == "segmentation_multiclass"
