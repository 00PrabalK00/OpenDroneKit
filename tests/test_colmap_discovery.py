"""Finding the COLMAP binary decides whether dense stereo is offered at all.

`dense_stereo` in the capability report is computed from whether a native binary was
found. So a search that misses an installed build does not merely inconvenience the
operator -- it makes the application report that this machine cannot do dense
reconstruction, which is a false statement about the hardware.

That is what was happening here. The official Windows CUDA archive is 359 MB and
extracting it to Program Files needs administrator rights that a normal user of this
application does not have. The search looked in Program Files and on PATH and nowhere
else, so the ordinary outcome of a per-user install was an invisible binary and a
capability report that said no.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.reconstruction_colmap import colmap_executable, engine_capabilities


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch):
    """An executable that exists, in a directory of our choosing."""
    def build(root: Path, name: str = "colmap.exe") -> Path:
        target = root / "bin" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        return target

    # Nothing on PATH, so each test controls exactly one discovery route.
    monkeypatch.setattr("core.reconstruction_colmap.shutil.which", lambda _: None)
    monkeypatch.delenv("ODK_COLMAP", raising=False)
    return build


class TestTheOverrideWins:
    def test_a_directory_is_accepted(self, tmp_path, fake_colmap, monkeypatch):
        fake_colmap(tmp_path)
        monkeypatch.setenv("ODK_COLMAP", str(tmp_path))
        assert colmap_executable() == str(tmp_path / "bin" / "colmap.exe")

    def test_the_executable_itself_is_accepted(self, tmp_path, fake_colmap, monkeypatch):
        exe = fake_colmap(tmp_path)
        monkeypatch.setenv("ODK_COLMAP", str(exe))
        assert colmap_executable() == str(exe)

    def test_a_wrong_override_reports_nothing_rather_than_falling_back(
        self, tmp_path, fake_colmap, monkeypatch
    ):
        """An operator who names a build wants that build.

        Silently using a different one would make the capability report describe
        something other than what was asked for.
        """
        fake_colmap(tmp_path)
        monkeypatch.setenv("ODK_COLMAP", str(tmp_path / "does-not-exist"))
        assert colmap_executable() is None


@pytest.mark.skipif(os.name != "nt", reason="the per-user location is Windows-specific")
class TestThePerUserInstall:
    def test_a_localappdata_install_is_found(self, tmp_path, fake_colmap, monkeypatch):
        """The case that was failing: no admin rights, so no Program Files install."""
        fake_colmap(tmp_path / "COLMAP")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert colmap_executable() == str(tmp_path / "COLMAP" / "bin" / "colmap.exe")

    def test_dense_stereo_is_offered_once_the_binary_is_visible(
        self, tmp_path, fake_colmap, monkeypatch
    ):
        """The claim that actually matters to a user.

        Whether the application says this machine can do dense reconstruction has to
        follow from whether a binary is reachable, not from where it happens to live.
        """
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert engine_capabilities()["dense_stereo"] is False

        fake_colmap(tmp_path / "COLMAP")
        assert engine_capabilities()["dense_stereo"] is True


class TestTheRealInstall:
    def test_the_installed_binary_reports_cuda(self) -> None:
        """Runs only where COLMAP is actually installed.

        'with CUDA' in the banner is the build flag. It is not proof the kernels match
        this GPU -- a build without kernels for the local architecture still prints it
        and then fails at the first launch -- so this asserts discovery and the flag,
        and the GPU itself is proven by running a real extraction, not by a unit test.
        """
        import subprocess

        found = colmap_executable()
        if not found:
            pytest.skip("no COLMAP binary on this machine")
        result = subprocess.run([found, "-h"], capture_output=True, text=True, timeout=60)
        banner = (result.stdout or "") + (result.stderr or "")
        assert "COLMAP" in banner, banner[:200]


class TestTheNativeGpuPathAsksTheRightSubcommand:
    """Each COLMAP subcommand must be probed against its own --help.

    COLMAP 4.1 renamed the GPU switches to FeatureExtraction.use_gpu and
    FeatureMatching.use_gpu. An unrecognised option is fatal to the binary, so the names
    are probed rather than assumed -- but the first version of that probe read the
    MATCHER's flag out of the EXTRACTOR's help, where it can never appear. The probe fell
    through to the legacy name, the matcher rejected it, the native path returned False,
    and the run quietly finished on the CPU while the progress message still read
    "Extracting SIFT features on the GPU".

    Silent fallback to the slow path, with the UI claiming the fast one. Exactly the
    class of false statement this codebase is built to refuse.
    """

    def test_the_matcher_flag_is_not_read_from_the_extractor_help(self) -> None:
        import inspect

        from core.reconstruction_colmap import ColmapReconstructor

        source = inspect.getsource(ColmapReconstructor._run_sparse_native)
        assert 'options_of(f"{matcher}_matcher")' in source, (
            "the matcher's GPU flag must be chosen from the matcher's own --help"
        )
        # One shared help string for both subcommands is the bug itself.
        assert "help_text" not in source, (
            "a single help text reused across subcommands reintroduces the fallback bug"
        )

    def test_the_real_binary_names_both_flags_in_its_own_help(self) -> None:
        """Against the installed COLMAP, so a future version rename is caught here."""
        import subprocess

        from core.reconstruction_colmap import colmap_executable

        binary = colmap_executable()
        if not binary:
            pytest.skip("no COLMAP binary on this machine")

        for subcommand, expected in (
            ("feature_extractor", "FeatureExtraction.use_gpu"),
            ("exhaustive_matcher", "FeatureMatching.use_gpu"),
        ):
            result = subprocess.run(
                [binary, subcommand, "--help"], capture_output=True, text=True, timeout=60
            )
            text = (result.stdout or "") + (result.stderr or "")
            assert expected in text, (
                f"{subcommand} does not offer {expected}; the probe's fallback name would "
                "be used and this build may reject it"
            )
