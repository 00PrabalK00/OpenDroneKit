"""OpenDroneKit desktop entry point.

Launches the native application window. The UI is HTML/JS rendered by the operating
system's built-in webview (Edge WebView2 on Windows, WebKit elsewhere), with a real
OS menu bar and a MapLibre GL map canvas.

    python main.py            # normal launch
    python main.py --debug    # with developer tools
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow launching from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _missing_dependency_message(exc: ImportError) -> str:
    return (
        f"OpenDroneKit could not start: {exc}\n\n"
        "Install the runtime dependencies with:\n"
        "    pip install -r requirements.txt\n\n"
        "The desktop shell needs `pywebview`. On Windows it also needs the Microsoft "
        "Edge WebView2 runtime, which is present on Windows 11 by default and available "
        "from https://developer.microsoft.com/microsoft-edge/webview2/ otherwise."
    )


def main(argv: list[str] | None = None) -> int:
    try:
        from app.shell import main as shell_main
    except ImportError as exc:
        print(_missing_dependency_message(exc), file=sys.stderr)
        return 1
    return shell_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
