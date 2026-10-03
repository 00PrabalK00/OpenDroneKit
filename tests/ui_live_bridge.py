"""Serve the real UI against a real Api, and drive it in a real browser.

Every other UI test in this repository runs the JavaScript against a fake bridge that
answers "ok" to everything, or reads the source as text. Both passed while the cockpit's
RTL button sent a command the Api did not know, Capture Now sent a camera action that
did not exist, and Export Log reported "no flight recorded" over four files it had just
written. A fake bridge cannot disagree with the UI about a name; only the real Api can.

So this puts the two together the way the desktop shell does -- pywebview exposes the
Api object to the page as ``window.pywebview.api`` -- except over HTTP on loopback:

  * ``serve(api)`` serves ``app/web`` and answers ``POST /__odk_api/<method>`` by calling
    that method on the Api with the JSON arguments. A small script injected into every
    page defines ``window.pywebview.api`` as a proxy onto that endpoint and fires
    ``pywebviewready``, which is all the UI ever uses of pywebview.
  * ``Browser`` starts headless Edge/Chromium with the DevTools protocol open and
    evaluates JavaScript in the page, so a test clicks the same buttons a user does.

The only thing substituted is pywebview's transport. The UI code and the Api are the
production ones.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"

BRIDGE_JS = """
(function () {
  const api = new Proxy({}, {
    get(_target, name) {
      if (typeof name !== "string" || name === "then") return undefined;
      return async (...args) => {
        const response = await fetch("/__odk_api/" + name, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ args }),
        });
        return response.json();
      };
    },
  });
  window.pywebview = { api };
  window.__odkCalls = [];
  const ready = () => window.dispatchEvent(new Event("pywebviewready"));
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => setTimeout(ready, 0));
  } else {
    setTimeout(ready, 0);
  }
})();
"""

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json", ".ico": "image/x-icon",
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(api: Any) -> tuple[ThreadingHTTPServer, str]:
    """Serve the UI with `api` behind it. Returns the server and its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # noqa: D401 - quiet test output
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0].lstrip("/") or "index.html"
            if path == "__odk_bridge.js":
                self._send(200, BRIDGE_JS.encode("utf-8"), _CONTENT_TYPES[".js"])
                return
            target = (WEB_ROOT / path).resolve()
            if WEB_ROOT not in target.parents or not target.is_file():
                self._send(404, b"not found", "text/plain")
                return
            body = target.read_bytes()
            if target.suffix == ".html":
                # Before any application script, exactly as pywebview injects its own.
                body = body.replace(b"<head>", b'<head><script src="/__odk_bridge.js"></script>', 1)
            self._send(200, body, _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if not self.path.startswith("/__odk_api/"):
                self._send(404, b"not found", "text/plain")
                return
            name = self.path[len("/__odk_api/"):]
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            method = getattr(api, name, None) if not name.startswith("_") else None
            if not callable(method):
                result: Any = {"ok": False, "error": f"unwired: {name}"}
            else:
                try:
                    result = method(*payload.get("args", []))
                except Exception as exc:  # noqa: BLE001 - reported to the page like guard() does
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(result, default=str).encode("utf-8"), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def chromium_path() -> str | None:
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if edge.is_file():
        return str(edge)
    for name in ("google-chrome", "google-chrome-stable", "msedge", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


class Browser:
    """Headless Chromium driven over the DevTools protocol."""

    def __init__(self, executable: str, window_size: tuple[int, int] = (1600, 1000),
                 log_path: str | Path | None = None):
        import requests
        import websocket

        self.port = free_port()
        self.profile = tempfile.mkdtemp(prefix="odk-ui-")
        # The browser's own log, when asked for: a renderer that dies (out of memory,
        # a GPU process crash) says so there and nowhere else.
        self._log = open(log_path, "w", encoding="utf-8", errors="replace") if log_path else None
        self.process = subprocess.Popen(
            [executable, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
             "--no-first-run", "--no-default-browser-check", "--disable-extensions",
             "--disable-background-networking", "--disable-component-update",
             "--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader",
             f"--window-size={window_size[0]},{window_size[1]}", "--hide-scrollbars",
             f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile}",
             *(["--enable-logging=stderr", "--v=0"] if self._log else []),
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=self._log or subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30.0
        pages: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                pages = [t for t in requests.get(f"http://127.0.0.1:{self.port}/json", timeout=2).json()
                         if t.get("type") == "page"]
                if pages:
                    break
            except Exception:  # noqa: BLE001 - not listening yet
                pass
            time.sleep(0.2)
        if not pages:
            self.close()
            raise RuntimeError("The browser did not open a DevTools page.")
        self.socket = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], timeout=60,
                                                   suppress_origin=True)
        self._id = 0
        self.page_ws_url = pages[0]["webSocketDebuggerUrl"]
        self._send("Page.enable")
        self._send("Runtime.enable")
        self._send("Inspector.enable")  # so a crashed renderer is reported, not a timeout
        # The viewport, pinned: headless --window-size is the window, not the page, and
        # a recording needs the page to be exactly the frame size.
        self._send("Emulation.setDeviceMetricsOverride", width=window_size[0],
                   height=window_size[1], deviceScaleFactor=1, mobile=False)

    def _send(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        wanted = self._id
        self.socket.send(json.dumps({"id": wanted, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("method") in ("Inspector.targetCrashed", "Inspector.detached"):
                raise RuntimeError(f"{method}: the page is gone ({message['method']}, "
                                   f"{message.get('params')})")
            if message.get("id") == wanted:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def open(self, url: str, settle_s: float = 2.0) -> None:
        self._send("Page.navigate", url=url)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.js("document.readyState") == "complete":
                break
            time.sleep(0.1)
        time.sleep(settle_s)

    def js(self, expression: str, *, timeout_s: float = 60.0) -> Any:
        """Evaluate in the page, awaiting a promise, and return the value."""
        result = self._send("Runtime.evaluate", expression=expression, awaitPromise=True,
                            returnByValue=True, timeout=int(timeout_s * 1000))
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            text = (detail.get("exception") or {}).get("description") or detail.get("text")
            raise RuntimeError(f"page error: {text}")
        return (result.get("result") or {}).get("value")

    def wait_for(self, expression: str, timeout_s: float = 20.0, poll_s: float = 0.2) -> Any:
        """Poll a JS expression until it is truthy; return its value."""
        deadline = time.monotonic() + timeout_s
        value = None
        while time.monotonic() < deadline:
            value = self.js(expression)
            if value:
                return value
            time.sleep(poll_s)
        raise AssertionError(f"timed out waiting for: {expression} (last value {value!r})")

    def screenshot(self, path: str | Path, quality: int | None = None) -> Path:
        """Write the page as it is now to `path` (PNG, or JPEG when quality is given)."""
        import base64

        params: dict[str, Any] = {"format": "png"}
        if quality is not None:
            params = {"format": "jpeg", "quality": int(quality)}
        data = self._send("Page.captureScreenshot", **params)["data"]
        target = Path(path)
        target.write_bytes(base64.b64decode(data))
        return target

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception:  # noqa: BLE001
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.process.kill()
        if self._log:
            self._log.close()
        shutil.rmtree(self.profile, ignore_errors=True)
