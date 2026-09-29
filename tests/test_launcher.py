"""The launcher starts an environment that can run every feature, and says when it can't.

OpenDroneKit.bat activated `cc-env`, a Python 3.10 env missing eighteen packages from
requirements.txt, so Fleet, share links, webhooks, PDF/DOCX reports and the Hub all
failed one button at a time while launch looked fine. And it started `pythonw` from
whatever folder it was run in, with no console, so a failed import produced nothing at
all.

These drive the real batch file (in its dry-run mode, which stops short of opening the
window) and the real entry point in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BAT = REPO_ROOT / "OpenDroneKit.bat"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the launcher is a Windows batch file")


def _conda_env_exists(name: str) -> bool:
    prefix = Path(sys.executable).resolve().parent
    # sys.executable is <base>/envs/<env>/python.exe or <base>/python.exe
    base = prefix.parent.parent if prefix.parent.name == "envs" else prefix
    return (base / "envs" / name / "python.exe").is_file()


def _run_bat(tmp_path: Path, **env_overrides: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"ODK_LAUNCH_DRYRUN": "1", **env_overrides})
    env.pop("ODK_PYTHON", None) if "ODK_PYTHON" not in env_overrides else None
    return subprocess.run(
        ["cmd", "/c", str(BAT), "--ui", "cockpit"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
        stdin=subprocess.DEVNULL,
    )


@windows_only
def test_the_launcher_defaults_to_the_complete_environment(tmp_path) -> None:
    if not _conda_env_exists("odk"):
        pytest.skip("no conda env named odk on this machine")
    done = _run_bat(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "requirements satisfied" in done.stdout
    launch = next(line for line in done.stdout.splitlines() if "would launch" in line)
    assert "\\envs\\odk\\pythonw.exe" in launch.lower().replace("/", "\\")
    assert "--ui cockpit" in launch
    # It starts in the toolkit folder, not where it was invoked from.
    cwd = next(line for line in done.stdout.splitlines() if line.startswith("[OpenDroneKit] cwd:"))
    assert Path(cwd.split("cwd:", 1)[1].strip()).resolve() == REPO_ROOT.resolve()


@windows_only
def test_an_incomplete_environment_is_reported_before_launch(tmp_path) -> None:
    """The old default env is the real example: the check names what is missing and
    the launcher does not start silently."""
    if not _conda_env_exists("cc-env"):
        pytest.skip("no incomplete conda env (cc-env) on this machine to launch against")
    done = _run_bat(tmp_path, ODK_CONDA_ENV="cc-env", ODK_ON_MISSING_DEPS="cancel")
    assert done.returncode != 0
    assert "SQLAlchemy" in done.stdout
    assert "pip install -r" in done.stdout
    assert "would launch" not in done.stdout


@windows_only
def test_the_launcher_explains_an_environment_that_does_not_exist(tmp_path) -> None:
    done = _run_bat(tmp_path, ODK_CONDA_ENV="odk-does-not-exist-xyz")
    assert done.returncode != 0
    assert "conda env create -f environment.yml" in done.stdout
    assert "would launch" not in done.stdout


def test_check_deps_reads_requirements_txt(tmp_path) -> None:
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--check-deps"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    from app.dependencies import check, read_requirements

    names = {r.name.lower() for r in read_requirements()}
    assert {"pywebview", "sqlalchemy", "fastapi", "reportlab", "python-docx"} <= names
    missing = check()
    assert done.returncode == (1 if missing or sys.version_info < (3, 11) else 0), done.stdout
    for problem in missing:
        assert problem.requirement.name in done.stdout


def test_a_missing_package_is_reported_not_swallowed(tmp_path) -> None:
    """Block `webview` the way an environment without it would, and run the real
    entry point: the failure must reach the startup log, not vanish with stderr."""
    data = tmp_path / "data"
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'webview' or name.startswith('webview.'):\n"
        "            raise ModuleNotFoundError(\"No module named 'webview'\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(blocker), "ODK_DATA_DIR": str(data), "ODK_NO_DIALOG": "1"})
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    log = data / "logs" / "startup-error.log"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "No module named 'webview'" in text
    assert "pip install -r requirements.txt" in text
    assert str(log) in done.stderr


def test_environment_yml_installs_requirements_txt() -> None:
    spec = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "name: odk" in spec
    assert "python=3.11" in spec
    assert "-r requirements.txt" in spec
    bat = BAT.read_text(encoding="utf-8")
    assert 'set "ODK_CONDA_ENV=odk"' in bat
    assert "cc-env" not in bat
    assert '--check-deps' in bat
    assert 'cd /d "%ROOT%"' in bat


def test_installation_guide_matches_the_launcher() -> None:
    guide = (REPO_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    for fact in ("conda env create -f environment.yml", "python main.py --check-deps",
                 "ODK_CONDA_ENV", "ODK_DATA_DIR", "--ui cockpit", "startup-error.log"):
        assert fact in guide, fact
