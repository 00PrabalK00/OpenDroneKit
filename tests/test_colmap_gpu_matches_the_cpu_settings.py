"""The GPU path must run the profile the operator chose, and admit what it costs.

Turning on CUDA moved feature extraction and matching from pycolmap to the native COLMAP
binary. The two are the same engine reading the same database, so it looked like a pure
speed change. It was not, in two separate ways.

**The bug.** `_run_sparse_native()` forwarded `max_num_features` and nothing else, while
the pycolmap path also set `max_image_size` and `num_threads` from the profile. So the
GPU path ran at COLMAP's default resolution regardless of whether the operator picked
fast, balanced or accurate. On eight frames of the Aukerman survey the CPU path
registered six images and the GPU path three, and one run registered none at all and
raised

    COLMAP could not register any images. The dataset may lack overlap, be motion
    blurred, or be too small for structure-from-motion.

which blames the imagery for a setting the engine never received. That is the same shape
as the mission form sending `gsd` to a planner that reads `target_gsd_cm`: the control
was wired, the value was read, and it stopped at a boundary.

**The trade that is not a bug.** With the settings equalised, a gap remains, and it is a
property of the detector rather than of this code. Measured through the same binary with
identical options on the same eight frames:

    use_gpu=0   97,934 keypoints
    use_gpu=1   80,256 keypoints

SiftGPU finds about a fifth fewer keypoints than the CPU detector. On a comfortable
survey that costs nothing. On a marginal one it decides how many images register. An
operator who enabled the GPU for speed did not knowingly change the detector, so the
engine has to say so when the result comes back short.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from core import reconstruction_colmap as engine

SOURCE = Path(engine.__file__).read_text(encoding="utf-8")


def body(name: str) -> str:
    """The CODE of a method, with comments and the docstring removed.

    Written the naive way first, and it was vacuous: this file explains the bug at
    length in a comment inside _run_sparse_native, and that comment contains the string
    "max_image_size". The guard passed on the prose describing the defect while the
    defect was present. A test that a comment can satisfy is not a test.
    """
    section = SOURCE.split(f"def {name}")[1].split("\n    def ")[0]
    # Drop the docstring, which is everything up to the closing triple quote.
    if '"""' in section:
        head, _, rest = section.partition('"""')
        if '"""' in rest:
            section = head + rest.split('"""', 1)[1]
    return "\n".join(
        line for line in section.splitlines()
        if not line.lstrip().startswith("#")
    )


class TestBothPathsRunTheSameProfile:
    """Whatever the CPU path reads from settings, the GPU path must send too."""

    def test_the_native_extractor_forwards_max_image_size(self) -> None:
        native = body("_run_sparse_native")
        assert "max_image_size" in native, (
            "the GPU path does not send max_image_size, so the quality profile the "
            "operator chose is not the profile that runs"
        )

    def test_every_setting_the_cpu_path_reads_reaches_the_gpu_path(self) -> None:
        """The invariant, rather than a list of three flag names.

        A setting added to _extraction_options() later and not to the native command
        would reintroduce exactly this bug, silently.
        """
        cpu = body("_extraction_options")
        native = body("_run_sparse_native")

        read_by_cpu = {
            key for key in ("max_image_size", "max_num_features")
            if f'self.settings["{key}"]' in cpu
        }
        assert read_by_cpu, "the CPU path reads no settings; this test is looking in the wrong place"

        missing = sorted(key for key in read_by_cpu if key not in native)
        assert not missing, f"the GPU path never sends: {missing}"

    def test_thread_capping_is_not_lost_on_the_gpu_path(self) -> None:
        """One full-resolution pyramid per thread is what exhausts memory on a laptop.

        The cap exists because the extractor crashed outright without it, and the GPU
        path reads the same imagery on the same machine.
        """
        native = body("_run_sparse_native")
        assert "_worker_threads" in native

    def test_flag_names_are_probed_and_not_assumed(self) -> None:
        """COLMAP 4.1 renamed some knobs and kept others, inconsistently.

        `max_image_size` and `num_threads` moved to FeatureExtraction.*; max_num_features
        stayed on SiftExtraction.*. Guessing either way gets

            Failed to parse options - unrecognised option

        and the binary exits non-zero, which sends the whole run back to the CPU while
        the progress line still says the GPU is being used.
        """
        native = body("_run_sparse_native")
        assert "extract_help" in native
        assert 'f"--FeatureExtraction.{name}"' in native or "FeatureExtraction." in native
        assert "SiftExtraction." in native


class TestTheDetectorChangeIsReported:
    def test_a_short_registration_on_the_gpu_names_gpu_sift(self) -> None:
        assert "_used_gpu_sift" in SOURCE
        warning = SOURCE.split("_used_gpu_sift", 2)[2]
        assert "GPU SIFT" in warning
        # It has to say what to do, not only what happened.
        assert "GPU disabled" in warning or "disabling the GPU" in warning

    def test_it_records_what_ran_rather_than_what_was_requested(self) -> None:
        """The binary can fall back to the CPU detector by itself.

        Reporting GPU SIFT's weaker recall on a run that used the CPU detector would be
        a confident, wrong explanation for a bad reconstruction -- worse than silence,
        because it sends the operator to re-run something that would not change.
        """
        native = body("_run_sparse_native")
        assert "SIFT CPU feature extractor" in native
        set_line = [l for l in native.splitlines() if "_used_gpu_sift = True" in l]
        assert set_line, "the flag is never set on the native path"

    def test_the_default_is_that_it_did_not_run(self) -> None:
        assert "self._used_gpu_sift = False" in SOURCE


class TestTheSplitModelWarningKnowsHowManyImagesThereWere:
    def test_image_count_reaches_mapping_on_the_native_path(self) -> None:
        """It was omitted, so a split reconstruction reported "3 of 0 images"."""
        block = SOURCE.split("if self._run_sparse_native(")[1][:600]
        assert "_map_from_database(" in block
        mapped = block.split("_map_from_database(")[1].split(")")[0]
        assert "image_count" in mapped, "the native path maps without the image count"
