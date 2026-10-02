"""OpenDroneKit desktop entry point.

Launches the native application window. The UI is HTML/JS rendered by the operating
system's built-in webview (Edge WebView2 on Windows, WebKit elsewhere), with a real
OS menu bar and a MapLibre GL map canvas.

    python main.py                 # normal launch (classic single-screen UI)
    python main.py --ui cockpit    # the fourteen-workspace cockpit
    python main.py --debug         # with developer tools
    python main.py --check-deps    # verify requirements.txt is installed, then exit
    python main.py --self-test     # headless start-up check (no window), JSON report
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Allow launching from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _missing_dependency_message(exc: BaseException) -> str:
    return (
        f"OpenDroneKit could not start: {exc}\n\n"
        "Install the runtime dependencies with:\n"
        "    pip install -r requirements.txt\n\n"
        "The desktop shell needs `pywebview`. On Windows it also needs the Microsoft "
        "Edge WebView2 runtime, which is present on Windows 11 by default and available "
        "from https://developer.microsoft.com/microsoft-edge/webview2/ otherwise."
    )


def _has_console() -> bool:
    """False under pythonw.exe, where stderr goes nowhere."""
    if sys.stderr is None:
        return False
    return not Path(sys.executable).name.lower().startswith("pythonw")


def _startup_log(message: str) -> Path | None:
    """Append the failure to <data root>/logs/startup-error.log and return the path."""
    try:
        from core.paths import logs_dir

        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "startup-error.log"
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {sys.executable}\n{message}\n\n")
        return target
    except Exception:  # noqa: BLE001 - logging must never mask the original failure
        return None


def report_startup_failure(message: str) -> None:
    """Make a startup failure visible however the app was launched.

    The launcher runs `pythonw`, which has no console: printing to stderr meant a
    double-click that failed produced nothing at all -- no window, no error, no log.
    Now the failure is always logged, printed when there is a console, and shown in a
    native message box when there is not.
    """
    log = _startup_log(message)
    full = message + (f"\n\nDetails were written to:\n{log}" if log else "")
    if sys.stderr is not None:
        try:
            print(full, file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("ODK_NO_DIALOG") == "1" or _has_console():
        return
    if sys.platform == "win32":
        try:
            import ctypes

            mb_iconerror = 0x10
            ctypes.windll.user32.MessageBoxW(None, full, "OpenDroneKit could not start", mb_iconerror)
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--check-deps" in args:
        from app.dependencies import report

        code, text = report()
        print(text)
        return code

    if "--self-test" in args:
        from app.selftest import main as self_test

        return self_test()

    try:
        from app.shell import UsageError, main as shell_main
    except ImportError as exc:
        report_startup_failure(_missing_dependency_message(exc))
        return 1

    try:
        return shell_main(args)
    except UsageError as exc:
        # A bad --ui or ODK_UI value. Refused, as before, but now visibly: under
        # pythonw argparse's message went to a stderr that does not exist.
        report_startup_failure(
            f"OpenDroneKit was started with an option it does not understand.\n\n{exc}\n\n"
            "Check the ODK_UI environment variable and the arguments passed to "
            "OpenDroneKit.bat / main.py (valid interfaces: classic, cockpit)."
        )
        return 2
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - a windowless crash must still be seen
        report_startup_failure(
            f"OpenDroneKit stopped with an error: {exc}\n\n{traceback.format_exc()}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
