"""Load the real desktop page (app/web/index.html) in headless Chromium against a real Api.

pywebview is the only thing replaced: its bridge object becomes a fetch() to a loopback
server that calls the SAME Api instance the window would. Everything the user touches --
the HTML, app.js, the modal dialogs, the buttons -- is the shipped code, and everything
behind the bridge is the shipped Python. A driver script clicks through the page the way
a person would and writes what it saw into <body data-result=...>, which the test reads
back from the settled DOM.

The MapLibre canvas is replaced by a no-op object. It needs WebGL and remote tiles, is not
what these tests are about, and a tile fetch that never settles would stall the virtual
clock the run depends on.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from browser_evidence import dump_dom_command

WEB = Path(__file__).resolve().parents[1] / "app" / "web"

BRIDGE_JS = r"""
(function () {
  class NoMap {
    constructor() { this.vectorLayers = new Map(); this.rasterLayers = new Map();
                    this.map = { getZoom: () => 0 }; }
    getGeometry() { return {}; }
  }
  window.OdkMap = new Proxy(NoMap, { construct(target, args) {
    return new Proxy(new target(...args), { get(obj, key) {
      return key in obj ? obj[key] : () => undefined; } });
  } });
  window.OdkMapUtils = { flatten: () => [] };
  async function invoke(name, args) {
    const response = await fetch('/__api/' + name, { method: 'POST', body: JSON.stringify(args) });
    return response.json();
  }
  window.pywebview = { api: new Proxy({}, { get: (_t, name) => (...args) => invoke(name, args) }) };
  window.__driverLog = [];
})();
"""

HELPERS_JS = r"""
window.__sleep = (ms) => new Promise((r) => setTimeout(r, ms));
window.__waitFor = async (predicate, ms = 60000) => {
  const start = Date.now();
  while (Date.now() - start < ms) {
    try { if (predicate()) return true; } catch (e) { /* not yet */ }
    await window.__sleep(50);
  }
  throw new Error('timed out waiting for ' + predicate);
};
window.__modalOpen = (title) => !document.querySelector('#modal-backdrop').classList.contains('hidden')
  && document.querySelector('#modal-title').textContent === title;
window.__clickAction = (label) => {
  const button = Array.from(document.querySelectorAll('#modal-actions button'))
    .find((b) => b.textContent === label);
  if (!button) throw new Error('no modal button ' + label);
  button.click();
};
window.__finish = (value) => {
  document.body.setAttribute('data-result', encodeURIComponent(JSON.stringify(value)));
};
"""


CLASSIC_READY = ("document.querySelector('#status-message').textContent"
                 " === 'OpenDroneKit ready.'")


def run_page(api: Any, driver_js: str, profile_dir: Path, *, budget_ms: int = 120000,
             timeout_s: int = 240, page_name: str = "index.html",
             ready_js: str = CLASSIC_READY, before_js: str = "") -> dict[str, Any]:
    """Serve a page with `api` behind the bridge, run `driver_js`, return its result.

    index.html (the default window) and workspace.html (the cockpit) are both served as
    shipped, with the bridge injected ahead of the application's own module code.
    """
    lock = threading.Lock()
    page = (WEB / page_name).read_text(encoding="utf-8")
    injected = ('<script src="/__bridge.js"></script>\n<script src="/__helpers.js"></script>\n'
                + (f"<script>{before_js}</script>\n" if before_js else ""))
    if page_name == "index.html":
        # The map scripts are dropped (see module docstring); app.js is kept as shipped.
        page = re.sub(r'<script src="vendor/maplibre/[^"]+"></script>\s*', "", page)
        page = page.replace('<script src="js/map.js"></script>', "")
        page = page.replace('<script src="js/app.js"></script>',
                            '<script src="js/app.js"></script>\n' + injected)
    else:
        page = page.replace("<body>", "<body>\n" + injected, 1)
    page = page.replace("</body>", '<script src="/__driver.js"></script>\n</body>', 1)
    driver = (
        "window.addEventListener('load', async () => {\n"
        "  window.dispatchEvent(new Event('pywebviewready'));\n"
        f"  await window.__waitFor(() => {ready_js});\n"
        "  try {\n" + driver_js + "\n  } catch (err) {\n"
        "    window.__finish({ error: String(err && err.stack || err),"
        " text: document.body.innerText.slice(0, 3000) });\n"
        "  }\n});\n"
    )
    served = {"/__page.html": ("text/html", page), "/__bridge.js": ("text/javascript", BRIDGE_JS),
              "/__helpers.js": ("text/javascript", HELPERS_JS),
              "/__driver.js": ("text/javascript", driver)}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB), **kwargs)

        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, kind: str, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path in served:
                self._send(*served[self.path])
            else:
                super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            name = self.path.rsplit("/", 1)[-1]
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "[]")
            with lock:
                method = getattr(api, name, None)
                if method is None or name.startswith("_"):
                    result = {"ok": False, "error": f"Unknown API method: {name}"}
                else:
                    try:
                        result = method(*args)
                    except Exception as exc:  # noqa: BLE001 - reported to the page
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send("application/json", json.dumps(result, default=str))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run(
            dump_dom_command("the desktop window evidence test", profile_dir,
                             f"http://127.0.0.1:{server.server_port}/__page.html",
                             virtual_time_ms=budget_ms, webgl=False),
            capture_output=True, text=True, timeout=timeout_s, check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    match = re.search(r'data-result="([^"]*)"', completed.stdout)
    assert match, "the driver never finished: " + completed.stdout[-2000:]
    from urllib.parse import unquote

    return json.loads(unquote(html.unescape(match.group(1))))
