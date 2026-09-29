"""Run the real workspace UI against a real Api in headless Chromium, and drive it.

The --dump-dom tests in browser_evidence.py load a page and read what it settled to.
That cannot click a toolbar button, answer the dialog it opens and then read what the
Findings table says, which is what a user of AI Inspection does. This does, over the
DevTools protocol.

Only the transport is replaced. In the desktop app pywebview injects window.pywebview.api
and forwards each call to app.api.Api; here a small script defines the same object and
forwards each call over HTTP to an Api instance in the test process. The page, its
modules and the Api are the real ones.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any

from browser_evidence import SANDBOX_FLAGS, STARTUP_FLAGS, chromium

WEB = Path(__file__).resolve().parents[1] / "app" / "web"

BRIDGE = b"""<script>
window.pywebview = { api: new Proxy({}, { get(_target, name) {
  if (typeof name !== "string" || name === "then") return undefined;
  return async (...args) => {
    const response = await fetch("/__api/" + name, { method: "POST", body: JSON.stringify(args) });
    return response.json();
  };
}})};
</script>"""

# Small helpers the tests call by name, so each test reads as the steps a user takes.
HELPERS = r"""
window.T = {
  sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  async until(fn, ms, what) {
    const started = Date.now();
    while (Date.now() - started < ms) { const v = fn(); if (v) return v; await T.sleep(200); }
    throw new Error("timed out waiting for " + what);
  },
  tool(name) {
    const button = [...document.querySelectorAll("button.tbtn")].find((b) => b.textContent.trim() === name);
    if (!button) throw new Error("no toolbar button " + name);
    button.click();
    return true;
  },
  modal() { const all = document.querySelectorAll(".modal"); return all[all.length - 1] || null; },
  waitModal(text, ms = 120000) {
    return T.until(() => {
      const m = T.modal();
      return m && m.querySelector(".modal-title").textContent.includes(text) ? m : null;
    }, ms, "dialog " + text);
  },
  async choose(title, rowText, ms) {
    const m = await T.waitModal(title, ms);
    const row = [...m.querySelectorAll(".modal-row")].find((r) => r.textContent.includes(rowText));
    if (!row) throw new Error("no option " + rowText);
    row.click();
    return true;
  },
  async ok(title, ms) { (await T.waitModal(title, ms)).querySelector(".modal-actions .primary").click(); return true; },
  async fill(title, values) {
    const m = await T.waitModal(title);
    const inputs = [...m.querySelectorAll("input, select, textarea")];
    values.forEach((v, i) => { inputs[i].value = v; inputs[i].dispatchEvent(new Event("input")); });
    m.querySelector(".modal-actions .primary").click();
    return true;
  },
  toasts() { return [...document.querySelectorAll(".toast")].map((t) => t.textContent); },
  toast(pattern, ms = 120000) {
    const re = new RegExp(pattern);
    return T.until(() => T.toasts().find((t) => re.test(t)), ms, "message " + pattern);
  },
  table(header) {
    const t = [...document.querySelectorAll("table")].find((x) =>
      [...x.querySelectorAll("th")].some((h) => h.textContent === header));
    if (!t) return null;
    return [...t.querySelectorAll("tr")].filter((r) => r.querySelectorAll("td").length)
      .map((r) => [...r.querySelectorAll("td")].map((c) => c.textContent));
  },
  clickRow(header, index) {
    const t = [...document.querySelectorAll("table")].find((x) =>
      [...x.querySelectorAll("th")].some((h) => h.textContent === header));
    [...t.querySelectorAll("tr")].filter((r) => r.querySelectorAll("td").length)[index].click();
    return true;
  },
  props() { return [...document.querySelectorAll("table.props tr")].map((r) => r.textContent); },
  drag(canvas, x0, y0, x1, y1) {
    const rect = canvas.getBoundingClientRect();
    const fire = (type, x, y) => canvas.dispatchEvent(new PointerEvent(type, {
      bubbles: true, clientX: rect.left + x, clientY: rect.top + y, pointerId: 1, buttons: 1 }));
    fire("pointerdown", x0, y0); fire("pointermove", (x0 + x1) / 2, (y0 + y1) / 2);
    fire("pointermove", x1, y1); fire("pointerup", x1, y1);
  },
};
true;
"""


def _serve(api: Any) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                pass  # the page navigated away mid-request

        def do_POST(self) -> None:  # noqa: N802
            name = self.path.split("/__api/", 1)[1]
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"[]")
            method = getattr(api, name, None)
            if name.startswith("_") or not callable(method):
                result: Any = {"ok": False, "error": f"unwired: {name}"}
            else:
                result = method(*args)
            self._send(json.dumps(result, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            relative = self.path.split("?", 1)[0].lstrip("/") or "workspace.html"
            path = (WEB / relative).resolve()
            if WEB.resolve() not in path.parents or not path.is_file():
                self._send(b"", "text/plain", 404)
                return
            data = path.read_bytes()
            if path.suffix == ".html":
                data = data.replace(b"<head>", b"<head>" + BRIDGE, 1)
            types = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
                     ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
                     ".png": "image/png"}
            self._send(data, types.get(path.suffix, "application/octet-stream"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LiveUI:
    """The workspace page, served against `api`, open in headless Chromium."""

    def __init__(self, api: Any, workspace: str, page: str = "workspace.html",
                 ready: str = "!!document.querySelector('button.tbtn')") -> None:
        import requests
        import websocket

        self.server = _serve(api)
        self.profile = tempfile.TemporaryDirectory(prefix="odk-live-ui-")
        port = _free_port()
        self.process = subprocess.Popen(
            [str(chromium("driving the workspace UI")), "--headless=new", *SANDBOX_FLAGS,
             *STARTUP_FLAGS, f"--remote-debugging-port={port}",
             f"--user-data-dir={self.profile.name}", "--window-size=1600,1000", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages: list[dict[str, Any]] = []
        for _ in range(120):
            try:
                tabs = requests.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
                pages = [t for t in tabs if t.get("type") == "page"
                         and not str(t.get("url", "")).startswith(("edge:", "chrome:"))]
                if pages:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        assert pages, "Chromium started but offered no page to drive"
        self.socket = websocket.create_connection(
            pages[0]["webSocketDebuggerUrl"], timeout=1800, suppress_origin=True)
        self.counter = 0
        url = f"http://127.0.0.1:{self.server.server_port}/{page}"
        self._send("Page.navigate", url=url)
        self.js(f"(async () => {{ await new Promise((r) => setTimeout(r, 1500)); "
                f"localStorage.setItem('odk.workspace.last', '{workspace}'); return true; }})()")
        self._send("Page.navigate", url=url)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if self.js(f"document.readyState === 'complete' && ({ready})"):
                    break
            except RuntimeError:
                pass
            time.sleep(0.5)
        self.js(HELPERS)

    def _send(self, method: str, **params: Any) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.counter:
                return message

    def js(self, expression: str) -> Any:
        """Evaluate in the page, awaiting a promise; a page exception is raised here."""
        message = self._send("Runtime.evaluate", expression=expression,
                             returnByValue=True, awaitPromise=True)
        result = message.get("result", {})
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError((details.get("exception") or {}).get("description")
                               or details.get("text"))
        return result.get("result", {}).get("value")

    def close(self) -> None:
        try:
            self.socket.close()
        finally:
            # The whole tree: a headless browser leaves GPU, network and renderer
            # processes behind when only its first process is terminated.
            if self.process.poll() is None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                               capture_output=True, check=False) if _windows() else \
                    self.process.kill()
            self.process.wait(timeout=30)
            self.server.shutdown()
            self.profile.cleanup()


def _windows() -> bool:
    import os

    return os.name == "nt"
