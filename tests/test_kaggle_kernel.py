"""A generated kernel must not make the run bigger than the config asked for.

The GPU helper is called gpu_memory_overrides and its docstring says it shrinks a run to
fit. It returned a fixed ["--image-size", "640", "--batch-size", "8"], which is a cap
only if every config asks for more than that. The agriculture configs ask for 512, so on
a 16 GB card the helper written to make runs fit raised the resolution by a quarter and
both arms died with CUDA out of memory at 15.6 GB of 15.89 GB.

That failure costs a full queue wait to discover and looks like a corpus too big for the
machine rather than a generator that argued with its own config. So the clamp is pinned
here, where it fails in a second instead.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from tools.kaggle_kernel import CONFIGS, infer_kind, kernel_script  # noqa: E402


def script_for(config_name: str) -> str:
    text = (CONFIGS / f"{config_name}.yaml").read_text(encoding="utf-8")
    return kernel_script(config_name, infer_kind(text), "user/some-corpus")


class TestTheGpuCapIsACeilingNotAnOverride:
    def test_the_config_sizes_are_baked_into_the_kernel(self) -> None:
        """Without them the cap has nothing to compare against and can only guess."""
        settings = yaml.safe_load(
            (CONFIGS / "agriculture_segformer_b2_mc.yaml").read_text(encoding="utf-8")
        )
        script = script_for("agriculture_segformer_b2_mc")
        assert f"CONFIG_IMAGE_SIZE = {settings['image_size']}" in script
        assert f"CONFIG_BATCH_SIZE = {settings['batch_size']}" in script

    def test_the_cap_takes_a_minimum_rather_than_a_constant(self) -> None:
        script = script_for("agriculture_segformer_b2_mc")
        assert "min(CONFIG_IMAGE_SIZE, 640)" in script
        assert "min(CONFIG_BATCH_SIZE, 8)" in script

    def test_a_config_smaller_than_the_cap_is_left_alone(self) -> None:
        """512 is already under 640; overriding it upward is what broke both runs."""
        script = script_for("agriculture_segformer_b2_mc")
        assert "the config already fits" in script

    def test_the_hardcoded_upsize_is_gone(self) -> None:
        script = script_for("agriculture_segformer_b2_mc")
        assert '["--image-size", "640", "--batch-size", "8"]' not in script


class TestTheMultispectralRunGetsItsBands:
    def test_the_kernel_points_band_root_at_the_mounted_dataset(self) -> None:
        """The config's band_root is repo-relative and does not exist on a rented box.

        Without this the trainer refuses on the first tile -- which is the correct
        behaviour, and still a failed run if nothing supplies the real location.
        """
        script = script_for("agriculture_segformer_b2_ms")
        assert '"--band-root"' in script
        assert "weedsgalore_bands" in script

    def test_it_searches_beside_the_corpus_not_only_inside_it(self) -> None:
        """The resolver descends to the directory holding train/val/test, so it lands on
        <dataset>/agriculture_seg while the stacks sit at <dataset>/weedsgalore_bands.

        Looking only inside the corpus found nothing, passed no --band-root, and the
        trainer refused on the first tile. The refusal was right; the lookup was not.
        """
        script = script_for("agriculture_segformer_b2_ms")
        assert "corpus.parent" in script


class TestTheTrainerChoiceIsReadFromTheConfig:
    @pytest.mark.parametrize(
        "config_name,expected",
        [
            ("agriculture_segformer_b2_mc", "segmentation_multiclass"),
            ("agriculture_segformer_b2_ms", "segmentation_multiclass"),
            ("corrosion_segformer_b2_cs", "segmentation_multiclass"),
        ],
    )
    def test_multiclass_configs_route_to_the_multiclass_trainer(
        self, config_name: str, expected: str
    ) -> None:
        text = (CONFIGS / f"{config_name}.yaml").read_text(encoding="utf-8")
        assert infer_kind(text) == expected
