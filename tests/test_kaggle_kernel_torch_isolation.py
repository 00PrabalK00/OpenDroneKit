"""The kernel replaces torch on disk mid-run, so it must not use torch in-process after.

`install_compatible_torch` pip-installs a different torch over the one this process has
already imported. Its own docstring says the swap cannot be seen by the running
interpreter, which is why the capability check it performs runs in a subprocess.

`restore_checkpoint` did not get that treatment, and the bill came due: its `torch.load`
raised ModuleNotFoundError for both mounted checkpoints, the error was reported as
"unreadable (ModuleNotFoundError)" with no module named, and the fall-through branch --
"no mounted checkpoint was trained on this corpus" -- correctly described a situation that
was not true. A resume that was sitting on disk was discarded and twelve hours of GPU
retrained from epoch 0.

Nothing here could catch that, because every existing kernel test proves the script parses,
imports and resolves its names. It does all three. The invariant that was actually broken
is this one, so it gets a test of its own.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

KERNEL_DIR = REPO_ROOT / "training" / "kaggle"


def generated_kernels() -> list[Path]:
    return sorted(KERNEL_DIR.glob("*/*.py"))


def function_named(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


@pytest.mark.skipif(not generated_kernels(), reason="no kernels generated in this tree")
@pytest.mark.parametrize("script", generated_kernels(), ids=lambda p: p.parent.name)
def test_no_torch_load_in_the_parent_process(script: Path):
    """torch.load anywhere in the kernel script itself runs against a swapped-out torch."""
    tree = ast.parse(script.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "load"
            and isinstance(target.value, ast.Name)
            and target.value.id == "torch"
        ):
            offenders.append(node.lineno)

    assert not offenders, (
        f"{script.relative_to(REPO_ROOT)} calls torch.load in the kernel process at lines "
        f"{offenders}. install_compatible_torch replaced torch on disk after this process "
        "imported it, so the load fails with ModuleNotFoundError and a real checkpoint "
        "gets reported as belonging to another corpus. Read it in a subprocess."
    )


@pytest.mark.skipif(not generated_kernels(), reason="no kernels generated in this tree")
@pytest.mark.parametrize("script", generated_kernels(), ids=lambda p: p.parent.name)
def test_an_unreadable_checkpoint_names_the_reason(script: Path):
    """Reporting only the exception type is what made this cost a second session.

    "unreadable (ModuleNotFoundError)" hides the one fact -- which module -- that
    identifies the fault from the log alone.
    """
    source = script.read_text(encoding="utf-8")
    restore = function_named(ast.parse(source), "restore_checkpoint")
    if restore is None:
        pytest.skip("this kernel does not restore checkpoints")

    body = ast.get_source_segment(source, restore) or ""
    assert "type(exc).__name__" not in body, (
        "restore_checkpoint reports the exception class and drops the message"
    )
    assert "stderr" in body, (
        "restore_checkpoint should surface the failed probe's stderr, so an unreadable "
        "checkpoint says why"
    )


def test_extra_datasets_survive_regeneration(tmp_path, monkeypatch):
    """A mount added by hand was deleted by the next regenerate, three times running.

    The uploaded checkpoint has to be in dataset_sources or there is nothing under
    /kaggle/input to restore from -- which is the same lost session by a different route.
    """
    import kaggle_kernel

    metadata = kaggle_kernel.kernel_metadata(
        "shared_semantic_dinov2_vitb14",
        "someone",
        "someone/corpus",
        gpu=True,
        internet=True,
        extra_datasets=("someone/checkpoint",),
    )
    assert metadata["dataset_sources"] == ["someone/corpus", "someone/checkpoint"]
    # The corpus stays first; resolve_corpus and CORPUS are written from it.
    assert metadata["dataset_sources"][0] == "someone/corpus"


def test_the_live_kernel_still_mounts_its_checkpoint():
    """The v4 resume depends on this exact mount being present."""
    metadata_path = KERNEL_DIR / "shared_semantic_dinov2_vitb14" / "kernel-metadata.json"
    if not metadata_path.is_file():
        pytest.skip("kernel not generated in this tree")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "prabalkhare00/odk-semantic-v4-ckpt" in metadata["dataset_sources"], (
        "the epoch-35 checkpoint is not mounted; a re-push would train from scratch again"
    )
