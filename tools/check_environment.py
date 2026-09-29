"""Check that the Python running OpenDroneKit has what requirements.txt asks for.

The launcher runs this before the app starts. A missing package does not stop the
desktop shell from opening -- features import their dependencies lazily -- so without
this check the first sign of, say, SQLAlchemy being absent is a Fleet button failing
with "No module named 'sqlalchemy'" long after launch.

    python tools/check_environment.py            # list what is missing, exit 1 if any
    python tools/check_environment.py --install  # pip install the missing ones

Only distributions are checked, by the names requirements.txt uses. Versions are not
compared: an installed but older package is a different problem from an absent one,
and pip reports it when asked to install.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Builds that install the same importable package under another distribution name. A
# machine with onnxruntime-gpu has onnxruntime; asking pip for the CPU wheel on top of it
# would break the GPU one.
EQUIVALENTS = {
    "onnxruntime": ("onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"),
    "opencv-python": ("opencv-python-headless", "opencv-contrib-python",
                      "opencv-contrib-python-headless"),
    "pillow": ("pillow-simd",),
}


def requirements(path: Path = REQUIREMENTS) -> list[tuple[str, str]]:
    """(distribution name, full requirement line) for every requirement in the file."""
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec or spec.startswith("-"):
            continue
        match = _NAME.match(spec)
        if match:
            found.setdefault(match.group(1).lower(), spec)
    return list(found.items())


def missing(path: Path = REQUIREMENTS) -> list[str]:
    """The requirement lines whose distribution is not installed in this interpreter."""
    absent = []
    for name, spec in requirements(path):
        if not any(_installed(candidate) for candidate in (name, *EQUIVALENTS.get(name, ()))):
            absent.append(spec)
    return absent


def _installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install", action="store_true",
                        help="pip install the missing requirements into this interpreter")
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    args = parser.parse_args(argv)

    absent = missing(args.requirements)
    if not absent:
        print(f"[OpenDroneKit] All {len(requirements(args.requirements))} requirements are "
              f"installed in {sys.executable}.")
        return 0

    print(f"[OpenDroneKit] {len(absent)} requirement(s) missing from {sys.executable}:")
    for spec in absent:
        print(f"    {spec}")
    if not args.install:
        print("[OpenDroneKit] Features that need them will report the missing package "
              "when used.")
        return 1

    result = subprocess.run([sys.executable, "-m", "pip", "install", *absent], check=False)
    if result.returncode != 0:
        print("[OpenDroneKit] pip could not install every package; see its output above.")
        return result.returncode
    still = missing(args.requirements)
    if still:
        print("[OpenDroneKit] Still missing after install: " + ", ".join(still))
        return 1
    print("[OpenDroneKit] Everything requirements.txt lists is now installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
