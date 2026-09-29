"""Is this Python environment able to run every OpenDroneKit feature?

The launcher used to start whichever conda env it was pointed at and let the gaps
surface one button at a time: Fleet answered "No module named 'sqlalchemy'", the report
formats list came back empty, the Hub could not start. The shipped env (cc-env) was
missing eighteen of the packages in requirements.txt and nothing said so.

This reads requirements.txt -- the single list -- and checks each distribution is
installed at the required minimum, using only the standard library (plus `packaging`
when present for version comparison), so it runs before anything heavy is imported and
works in an environment that is missing almost everything.

    python main.py --check-deps     # exit 0 when complete, 1 otherwise
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
MIN_PYTHON = (3, 11)

# Without these the window cannot open at all; everything else costs one feature.
LAUNCH_CRITICAL = {"pywebview"}

# Drop-in builds published under another distribution name. The import is identical,
# so any one of them satisfies the requirement: a GPU machine has onnxruntime-gpu and
# must not be told onnxruntime is missing.
ALTERNATIVES = {
    "onnxruntime": ("onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"),
    "opencv-python": ("opencv-python-headless", "opencv-contrib-python",
                      "opencv-contrib-python-headless"),
}

_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


@dataclass(frozen=True)
class Requirement:
    name: str
    spec: str  # e.g. ">=2.0", or "" for any version


@dataclass(frozen=True)
class Problem:
    requirement: Requirement
    installed: str | None  # None when absent

    def describe(self) -> str:
        if self.installed is None:
            return f"{self.requirement.name}{self.requirement.spec}: not installed"
        return f"{self.requirement.name}{self.requirement.spec}: found {self.installed}"


def read_requirements(path: Path = REQUIREMENTS) -> list[Requirement]:
    seen: dict[str, Requirement] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name = match.group(1)
        spec = match.group(3).split(";", 1)[0].strip().replace(" ", "")
        seen.setdefault(name.lower(), Requirement(name, spec))
    return list(seen.values())


def _satisfies(installed: str, spec: str) -> bool:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(installed) in SpecifierSet(spec)
    except Exception:  # noqa: BLE001 - no `packaging`, or an unparseable local version
        # Presence is the part that breaks features; a version we cannot compare is
        # reported as satisfied rather than inventing a failure.
        return True


def _installed_version(name: str) -> str | None:
    for candidate in (name, *ALTERNATIVES.get(name.lower(), ())):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def check(requirements: list[Requirement] | None = None) -> list[Problem]:
    """Every requirement that is absent or too old in the running interpreter."""
    problems: list[Problem] = []
    for requirement in requirements if requirements is not None else read_requirements():
        installed = _installed_version(requirement.name)
        if installed is None:
            problems.append(Problem(requirement, None))
            continue
        if not _satisfies(installed, requirement.spec):
            problems.append(Problem(requirement, installed))
    return problems


def python_problem() -> str | None:
    if sys.version_info < MIN_PYTHON:
        found = ".".join(str(part) for part in sys.version_info[:3])
        wanted = ".".join(str(part) for part in MIN_PYTHON)
        return f"Python {wanted}+ is required; this is Python {found} ({sys.executable})."
    return None


def report() -> tuple[int, str]:
    """(exit code, human-readable report) for the launcher."""
    lines: list[str] = []
    python_issue = python_problem()
    problems = check()
    if python_issue:
        lines.append(python_issue)
    if problems:
        lines.append(f"{len(problems)} package(s) from requirements.txt are missing or too old "
                     f"in {sys.executable}:")
        lines.extend(f"  - {problem.describe()}" for problem in problems)
        lines.append("")
        lines.append("Install them into this environment with:")
        lines.append(f'  "{sys.executable}" -m pip install -r "{REQUIREMENTS}"')
        lines.append("or create the reference environment: conda env create -f environment.yml")
    if not lines:
        return 0, f"All {len(read_requirements())} requirements satisfied in {sys.executable}."
    return 1, "\n".join(lines)


def launch_blockers() -> list[Problem]:
    """Missing packages without which the window cannot open."""
    return [p for p in check() if p.requirement.name.lower() in LAUNCH_CRITICAL]
