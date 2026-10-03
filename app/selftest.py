"""Headless start-up check: prove an installation runs without opening a window.

    python main.py --self-test            # prints a JSON report, exit 0 when it starts

What it exercises is the real start-up path minus the window: the desktop shell module
and pywebview import, .NET (pythonnet) loads the way pywebview's WinForms/WebView2
backend needs it, the bridge object the window binds is constructed, and a handful of
its read-only methods answer. Each check records its own error, so a broken install
says which part is broken. Nothing is contacted on the network.

When ``install_manifest.json`` sits beside the payload (an installed build), every file
it lists is hashed and compared, so a damaged or tampered install is reported.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]


def _progress(message: str) -> None:
    """One line per check on stderr, so a slow first start shows where it is.

    The JSON report goes to stdout at the end; on a cold machine (antivirus scanning a
    freshly installed 1.2 GB tree) the imports alone can take many minutes, and a caller
    with a deadline needs to know which check it was in.
    """
    try:
        print(f"[self-test] {message}", file=sys.stderr, flush=True)
    except (OSError, ValueError, AttributeError):  # pythonw: no stderr
        pass


def _check(name: str, fn: Callable[[], Any], results: list[dict[str, Any]],
           required: bool = True) -> Any:
    started = time.perf_counter()
    _progress(f"{name} ...")
    try:
        value = fn()
        results.append({"check": name, "ok": True, "required": required,
                        "detail": value, "seconds": round(time.perf_counter() - started, 2)})
        _progress(f"{name}: ok ({results[-1]['seconds']} s)")
        return value
    except BaseException as exc:  # noqa: BLE001 - the report is the point
        results.append({"check": name, "ok": False, "required": required,
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=3),
                        "seconds": round(time.perf_counter() - started, 2)})
        _progress(f"{name}: FAILED ({results[-1]['seconds']} s) {results[-1]['error'][:200]}")
        return None


def webview2_runtime_version() -> str:
    """The installed Edge WebView2 runtime version, read from the registry ('' if absent)."""
    if sys.platform != "win32":
        return ""
    import winreg

    guid = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    keys = [
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{guid}"),
    ]
    for hive, path in keys:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, "pv")
                if value and value != "0.0.0.0":
                    return str(value)
        except OSError:
            continue
    return ""


def run_self_test(verify_manifest: bool = True) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    _check("python", lambda: {"version": sys.version.split()[0], "executable": sys.executable,
                              "prefix": sys.prefix}, results)

    def deps() -> dict[str, Any]:
        from app.dependencies import report

        code, text = report()
        if code:
            raise RuntimeError(text)
        return {"requirements": "all installed"}

    _check("requirements", deps, results)
    _check("import webview", lambda: __import__("webview").__name__, results)

    def dotnet() -> str:
        if sys.platform != "win32":
            return "not windows"
        import clr  # noqa: F401  (pythonnet: what pywebview's WinForms backend loads)
        import System  # type: ignore[import-not-found]

        return str(System.Environment.Version)

    _check("pythonnet (.NET)", dotnet, results)

    def webview2() -> str:
        version = webview2_runtime_version()
        if not version and sys.platform == "win32":
            raise RuntimeError("Microsoft Edge WebView2 runtime not found; the window cannot open.")
        return version

    _check("WebView2 runtime", webview2, results)
    _check("import app.shell", lambda: __import__("app.shell").__name__, results)
    for module in ("onnxruntime", "rasterio", "pyproj", "cv2", "numpy", "pycolmap", "open3d"):
        _check(f"import {module}", lambda m=module: getattr(__import__(m), "__version__", "ok"),
               results)

    api = _check("construct Api", lambda: __import__("app.api", fromlist=["Api"]).Api(), results)
    if api is not None:
        results[-1]["detail"] = type(api).__name__

        def call(method: str, *args: Any) -> Callable[[], Any]:
            def run() -> Any:
                reply = getattr(api, method)(*args)
                if not isinstance(reply, dict) or not reply.get("ok"):
                    raise RuntimeError(f"{method} -> {reply!r}"[:500])
                return reply

            return run

        caps = _check("api.capabilities", call("capabilities"), results)
        if caps:
            results[-1]["detail"] = {
                "gpu_summary": caps["capabilities"].get("gpu_summary"),
                "onnxruntime_providers": caps["capabilities"]["gpu_detection"]["onnxruntime"]["providers"],
                "adapters": [a["name"] for a in caps["capabilities"]["gpu_detection"]["adapters"]],
                "pycolmap": caps["capabilities"].get("pycolmap"),
                "open3d": caps["capabilities"].get("open3d"),
            }
        downloads = _check("api.model_downloads", call("model_downloads"), results)
        if downloads:
            results[-1]["detail"] = {k: downloads[k] for k in ("missing", "downloadable", "not_published")}
        _check("api.update_settings", call("update_settings"), results)
        _check("api.check_for_update (opt-in off, no network)", call("check_for_update"), results)

    manifest = ROOT.parent / "install_manifest.json"
    if verify_manifest and manifest.is_file():
        def verify() -> dict[str, Any]:
            sys.path.insert(0, str(ROOT))
            from tools.installer.manifest import load_manifest, verify_tree

            data = load_manifest(manifest)
            problems = verify_tree(ROOT.parent, data)
            if problems:
                raise RuntimeError(f"{len(problems)} file(s) differ: {problems[:5]}")
            return {"files_verified": len(data["files"]), "signed": data.get("signed")}

        _check("install manifest", verify, results)

    passed = all(r["ok"] for r in results if r["required"])
    return {"ok": passed, "installed": os.environ.get("ODK_INSTALLED") == "1",
            "root": str(ROOT), "checks": results}


def main() -> int:
    report = run_self_test()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1
