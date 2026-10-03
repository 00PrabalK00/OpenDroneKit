"""Entry point of an installed OpenDroneKit (copied to the install root as launch.py).

    runtime\\pythonw.exe launch.py                          the application window
    runtime\\python.exe  launch.py --self-test [--pause]    headless start-up check
    runtime\\python.exe  launch.py --download-models --all  first-run weights download

An installed build keeps user state outside the install folder, so an update or an
uninstall never touches projects or downloaded weights:

    ODK_DATA_DIR            default %LOCALAPPDATA%\\OpenDroneKit\\data
    ODK_MODEL_WEIGHTS_DIR   default %LOCALAPPDATA%\\OpenDroneKit\\models

Either can still be set in the environment to point somewhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ODK = HERE / "odk"


def _defaults() -> None:
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "OpenDroneKit"
    os.environ.setdefault("ODK_INSTALLED", "1")
    os.environ.setdefault("ODK_DATA_DIR", str(local / "data"))
    os.environ.setdefault("ODK_MODEL_WEIGHTS_DIR", str(local / "models"))
    Path(os.environ["ODK_MODEL_WEIGHTS_DIR"]).mkdir(parents=True, exist_ok=True)


def main(argv: list[str]) -> int:
    # The install folder is read-only in spirit (and in fact under Program Files): every
    # .pyc was compiled at build time and is listed in install_manifest.json, so the
    # runtime must not write new ones beside the sources.
    sys.dont_write_bytecode = True
    _defaults()
    if str(ODK) not in sys.path:
        sys.path.insert(0, str(ODK))
    os.chdir(ODK)
    pause = "--pause" in argv
    argv = [a for a in argv if a != "--pause"]
    try:
        if argv[:1] == ["--download-models"]:
            from core.model_download import main as download

            return download(argv[1:])
        import main as odk_main

        return odk_main.main(argv)
    finally:
        if pause and sys.stdin is not None:
            try:
                input("\nPress Enter to close.")
            except EOFError:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
