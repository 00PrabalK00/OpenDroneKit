"""Press cockpit toolbar buttons in headless Chromium against a real ``app.api.Api``.

Shared by the analysis evidence tests. app/web is served from a loopback port and a
stand-in for the pywebview bridge forwards every call to the real Api over HTTP, so
what is asserted is what the application answered. The only things answered by the
harness are the ones a headless page cannot do: the native file chooser (``picks``, in
order), modal prompts (``answers``), choosers (``choose``: the first option starting
with that text) and a click on the map (``click``: delivered as its lon/lat through the
same draw module a real click reaches).
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from browser_evidence import dump_dom_command

WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="css/workspace.css"></head>
<body data-result="pending"><div id="app"></div><pre id="out"></pre>
<script src="vendor/maplibre/maplibre-gl.js"></script>
<script>
  window.pywebview = { api: new Proxy({}, { get: (_t, name) => (typeof name !== "string")
    ? undefined : (...args) => fetch("/api/" + name, { method: "POST",
        body: JSON.stringify(args) }).then((r) => r.json()) }) };
</script>
<script type="module">
import { Shell } from "./js/workspace/shell.js";
import { addVertex, canDraw, finishDrawing } from "./js/workspace/draw.js";
import { currentImage } from "./js/workspace/viewstate.js";

const PLAN = __PLAN__;
const WORKSPACE = __WORKSPACE__;
const log = [];
const shell = new Shell(document.getElementById("app"));
const toast = shell.toast.bind(shell);
shell.toast = (text, kind) => { log.push({ text: String(text), kind: kind || "" }); toast(text, kind); };
let answers = [];
let chosen = [];
const context = shell.actionContext.bind(shell);
shell.actionContext = () => ({ ...context(),
  prompt: async () => answers.shift() ?? "",
  choose: async (_title, options) => {
    const prefix = chosen.length > 1 ? chosen.shift() : (chosen[0] ?? "");
    return options.find((o) => String(o).startsWith(prefix)) ?? null;
  } });
shell.confirmAll = true;
const settle = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (test, n = 400) => { for (let i = 0; i < n && !test(); i++) await settle(25); };
const final = (before) => log.slice(before).filter((t) => (t.kind === "ok" || t.kind === "error")
  && !/^Click |click points|draw the/i.test(t.text));

try {
  await shell.refreshState();
  if (WORKSPACE) shell.open(WORKSPACE);
  const results = {};
  for (const step of PLAN) {
    const before = log.length;
    answers = [...(step.answers || [])];
    chosen = [].concat(step.choose ?? "");
    const running = shell.runAction(step.action);
    // Real click events on the photograph, at the screen point that shows this pixel of
    // the source frame -- on the whole-frame preview, or on a close-up of a window of it.
    const clicks = step.imageClick ? [].concat(step.imageClicks || [step.imageClick, step.imageClick]) : [];
    for (let k = 0; k < clicks.length; k += 1) {
      await until(() => log.slice(before).filter((t) => /^Click /.test(t.text)).length > k);
      await settle(50);
      const shown = currentImage();
      const image = document.querySelector("img.canvas-product");
      const box = image.getBoundingClientRect();
      const nw = image.naturalWidth, nh = image.naturalHeight;
      const fit = Math.min(box.width / nw, box.height / nh);
      const [sx, sy] = clicks[k];
      const [cx, cy, cw, ch] = shown.crop || [0, 0, shown.source_width, shown.source_height];
      const clientX = box.left + (box.width - nw * fit) / 2 + ((sx - cx) * nw / cw) * fit;
      const clientY = box.top + (box.height - nh * fit) / 2 + ((sy - cy) * nh / ch) * fit;
      image.dispatchEvent(new MouseEvent("click", { clientX, clientY, bubbles: true }));
    }
    // Real click events on the 3D model's canvas, at the screen point where the viewer
    // draws each absolute CRS point.
    for (let k = 0; k < (step.modelClicks || []).length; k += 1) {
      await until(() => log.slice(before).filter((t) => /^Click /.test(t.text)).length > k);
      await settle(50);
      const { currentModelViewer } = await import("./js/workspace/model3d.js");
      const viewer = currentModelViewer();
      const at = viewer.project(step.modelClicks[k]);
      viewer.canvas.dispatchEvent(new MouseEvent("click", { clientX: at.x, clientY: at.y, bubbles: true }));
    }
    if (step.click) {
      await until(() => canDraw());
      await until(() => log.slice(before).some((t) => /^Click /.test(t.text)));
      addVertex(step.click);
    }
    if (step.draw) {
      await running;
      step.draw.forEach((p) => addVertex(p));
      finishDrawing();
    }
    await running;
    await until(() => final(before).length > 0);
    await settle(50);
    results[step.name] = final(before).map((t) => `[${t.kind}] ${t.text}`).join(" ");
    // A read-back after the step, in the page (the test's own expression).
    if (step.after) results[`${step.name}:after`] = await (new Function(`return (async () => { ${step.after} })();`))();
  }
  document.getElementById("out").textContent = JSON.stringify(results);
  document.body.dataset.result = "done";
} catch (error) {
  document.body.dataset.result = "error:" + (error && error.message);
}
</script></body></html>"""


def run_actions(api: Any, tmp_path: Path, plan: list[dict], *, picks: list[str] = (),
                workspace: str = "measurements", virtual_time_ms: int = 15000) -> dict:
    """Run each step of ``plan`` in the real Shell; return the final toast per step."""
    page = (PAGE.replace("__PLAN__", json.dumps(plan))
            .replace("__WORKSPACE__", json.dumps(workspace))).encode("utf-8")
    queue = list(picks)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/harness.html":
                return self._send(page, "text/html")
            target = (WEB_ROOT / path.lstrip("/")).resolve()
            if WEB_ROOT not in target.parents or not target.is_file():
                return self._send(b"", "text/plain", 404)
            types = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}
            return self._send(target.read_bytes(),
                              types.get(target.suffix, "application/octet-stream"))

        def do_POST(self):
            name = self.path.rsplit("/", 1)[-1]
            length = int(self.headers.get("Content-Length") or 0)
            args = json.loads(self.rfile.read(length) or b"[]")
            if name.startswith("_") or not callable(getattr(api, name, None)):
                result = {"ok": False, "error": f"unwired: {name}"}
            elif name in ("pick_file", "pick_folder"):
                # The native chooser, answered in order.
                result = {"ok": True, "path": queue.pop(0) if queue else ""}
            else:
                result = getattr(api, name)(*args)
            return self._send(json.dumps(result, default=str).encode("utf-8"),
                              "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run(dump_dom_command(
            "a cockpit action evidence test", tmp_path / "edge-profile",
            f"http://127.0.0.1:{server.server_port}/harness.html",
            virtual_time_ms=virtual_time_ms, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    dom = completed.stdout
    if 'data-result="done"' not in dom:
        state = re.search(r'data-result="([^"]*)"', dom)
        raise AssertionError(f"harness did not finish ({state.group(1) if state else 'no page'}): "
                             f"{dom[-1500:]} {completed.stderr[-800:]}")
    raw = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)
    return json.loads(html.unescape(raw))
