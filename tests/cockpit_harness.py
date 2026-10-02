"""Drive the real cockpit, in a real browser, against the real Api.

The cockpit's own node harness (click_every_button.mjs) answers every bridge call with a
plausible success from a stub. That is how a map whose drawing tools read no clicks, a
demo mode that loaded the real project, and a header that said "3 workers" all passed:
the harness checked that the wiring survived the round trip, and nothing checked what
the round trip did.

This serves app/web exactly as pywebview does, with one difference: `window.pywebview.api`
is a thin fetch shim that forwards each call to an `app.api.Api` in this process. The Api
is the real one, on a store in the test's temporary directory. The page is rendered by
headless Chromium (browser_evidence.py) and a scenario script -- written in each test, run
as a module after the shell -- clicks, types and reads the screen like an operator, then
writes what it saw into <pre id="odk-result"> for the test to assert on.

Only the native dialogs are replaced, because they are the operating system's, not ours:
pick_folder and pick_file answer with the paths a test sets, and open_path records instead
of launching Explorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any

from browser_evidence import dump_dom_command

WEB = Path(__file__).resolve().parents[1] / "app" / "web"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
}

# Helpers every scenario can use. `ODK` is a global so scenarios stay short.
SCENARIO_PRELUDE = r"""
// Headless Chromium's virtual clock runs ahead of MapLibre's web workers, which work in
// real time: a wait measured only in virtual milliseconds can pass before a map has loaded.
// A pending fetch pauses the virtual clock, so each wait also holds one open for real time.
const tick = (ms) => fetch(`/__tick?ms=${Math.min(ms, 400)}`).catch(() => null);
const sleep = (ms) => Promise.all([new Promise((r) => setTimeout(r, ms)), tick(ms)]);
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const ODK = {
  sleep, $, $$,
  toasts: () => $$(".toast").map((t) => t.textContent),
  lastToast: () => { const t = $$(".toast"); return t.length ? t[t.length - 1].textContent : ""; },
  async waitFor(test, ms = 15000, step = 100) {
    for (let waited = 0; waited < ms; waited += step) {
      const value = await test();
      if (value) return value;
      await sleep(step);
    }
    return null;
  },
  async nav(title) {
    const button = $$(".nav-items .nav-item").find((b) => b.textContent === title);
    if (!button) throw new Error(`no workspace ${title}`);
    button.click();
    await sleep(600);
  },
  async press(label) {
    const button = $$(".toolbar .tbtn").find((b) => b.textContent === label);
    if (!button) throw new Error(`no toolbar button ${label}`);
    button.click();
    await sleep(300);
  },
  async tool(title) {
    const button = $(`.workspace .canvas .canvas-tools .tool[title="${title}"]`);
    if (!button) throw new Error(`no canvas tool ${title}`);
    button.click();
    await sleep(100);
  },
  panel(title) {
    const p = $$(".panel").find((x) => ((x.querySelector(".panel-header") || {}).innerText || "").trim().startsWith(title));
    return p ? p.querySelector(".panel-body").innerText.replace(/\s+/g, " ").trim() : null;
  },
  async answer(value) {
    const modal = await ODK.waitFor(() => $(".modal-backdrop .modal"), 5000);
    if (!modal) throw new Error("no dialog appeared");
    const input = modal.querySelector("input, textarea");
    if (input && value !== undefined) {
      input.value = value;
      input.dispatchEvent(new Event("input"));
    }
    modal.querySelector(".modal-actions .tbtn.primary").click();
    await sleep(300);
  },
  async pick(labelText) {
    const modal = await ODK.waitFor(() => $(".modal-backdrop .modal .modal-list"), 5000);
    if (!modal) throw new Error("no chooser appeared");
    const rows = $$(".modal-backdrop .modal .modal-row");
    const row = labelText === undefined ? rows[0] : rows.find((r) => r.textContent.includes(labelText));
    if (!row) throw new Error(`no row ${labelText}; rows: ${rows.map((r) => r.textContent).join(" | ")}`);
    const text = row.textContent;
    row.click();
    await sleep(300);
    return text;
  },
  async map() {
    const draw = await import("/js/workspace/draw.js");
    return ODK.waitFor(() => { const m = draw.currentMap(); return m && m._odkReady ? m : null; }, 20000);
  },
  async click(map, lon, lat) {
    map.fire("click", { lngLat: new window.maplibregl.LngLat(lon, lat), point: map.project([lon, lat]) });
    await sleep(50);
  },
  async key(key, opts = {}) {
    document.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, ...opts }));
    await sleep(200);
  },
  calls: async () => (await (await fetch("/__calls")).json()),
};
"""


@dataclass
class Cockpit:
    """A served cockpit bound to one Api. Use as a context manager."""

    api: Any
    profile_dir: Path
    picks: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    #: Offline basemap tiles the page asked for: (path, HTTP status).
    tiles: list[tuple[str, int]] = field(default_factory=list)
    #: Project raster tiles (core.raster_tiles) the page asked for: (path, HTTP status).
    raster_tiles: list[tuple[str, int]] = field(default_factory=list)
    ready_delay_ms: int = 0
    bridge: bool = True
    #: The page under test: the cockpit, or "index.html" for the classic window.
    page: str = "workspace.html"
    #: Browser window size, e.g. "1600,900". Headless defaults to a window small enough
    #: that the classic window's map is 21 px tall and cannot frame anything.
    window_size: str = ""
    #: Script run in <head> before any page script, e.g. to keep a handle on each map.
    head_script: str = ""
    #: Let the page reach the internet. Off by default: every host but loopback fails
    #: to resolve, as on a field laptop. Only tests that need a real provider set it.
    network: bool = False

    def __post_init__(self) -> None:
        api = self.api
        cockpit = self

        # Native dialogs are the operating system's; they answer with what the test set.
        api.pick_folder = lambda: {"ok": True, "path": cockpit.picks.get("folder", "")}
        api.pick_file = lambda extensions=None: {"ok": True, "path": cockpit.picks.get("file", "")}

        def open_path(path: str) -> dict[str, Any]:
            cockpit.opened.append(str(path))
            return {"ok": True, "opened": str(path)}

        api.open_path = open_path
        self.methods = sorted(
            name for name in dir(api) if not name.startswith("_") and callable(getattr(api, name))
        )
        self._scenario = ""
        self._query = ""

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                return

            def _send(self, body: bytes, kind: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/__tick":
                    import time as _time
                    wanted = self.path.split("ms=", 1)[-1] if "ms=" in self.path else "50"
                    _time.sleep(min(float(wanted or 50), 400.0) / 1000.0)
                    return self._send(b"{}", "application/json")
                if path == "/__tiles":
                    return self._send(json.dumps(cockpit.tiles).encode(), "application/json")
                if path == "/__calls":
                    return self._send(json.dumps(cockpit.calls, default=str).encode(), "application/json")
                if path == "/__scenario.html":
                    return self._send(cockpit._page().encode("utf-8"), "text/html; charset=utf-8")
                # The offline basemap, answered by the code the desktop shell uses.
                from core.raster_tiles import raster_tile_response
                from core.tile_cache import tile_response

                tile = tile_response(path, self.path.split("?", 1)[1] if "?" in self.path else "")
                if tile is not None:
                    code, kind, body, _ = tile
                    cockpit.tiles.append((self.path, code))
                    return self._send(body, kind, code)
                # The project's own rasters, cut into tiles by the code the shell uses.
                raster = raster_tile_response(path)
                if raster is not None:
                    code, kind, body, _ = raster
                    cockpit.raster_tiles.append((path, code))
                    return self._send(body, kind, code)
                target = (WEB / path.lstrip("/")).resolve()
                if WEB.resolve() not in target.parents or not target.is_file():
                    return self._send(b"not found", "text/plain", 404)
                kind = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
                return self._send(target.read_bytes(), kind)

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/__api/"):
                    return self._send(b"{}", "application/json", 404)
                name = self.path[len("/__api/"):]
                length = int(self.headers.get("Content-Length", 0))
                args = json.loads(self.rfile.read(length) or b"[]")
                try:
                    result = getattr(api, name)(*args)
                except Exception as exc:  # noqa: BLE001 -- reported, as pywebview would
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                cockpit.calls.append({
                    "method": name, "args": args,
                    "ok": not (isinstance(result, dict) and result.get("ok") is False),
                    "error": result.get("error") if isinstance(result, dict) else None,
                })
                return self._send(json.dumps(result, default=str).encode(), "application/json")

        # The listen backlog defaults to 5. A map opening fires a burst of tile and
        # module requests, and on a loaded Windows machine the overflow was refused:
        # "Failed to fetch", and tiles MapLibre marks errored and never asks for again.
        server_class = type("CockpitServer", (ThreadingHTTPServer,),
                            {"request_queue_size": 128, "daemon_threads": True})
        self.server = server_class(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "Cockpit":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def _page(self) -> str:
        html = (WEB / self.page).read_text(encoding="utf-8")
        shim = ""
        if self.bridge:
            shim = (
                "<script>(function(){const api={};"
                f"for (const m of {json.dumps(self.methods)}) api[m]=async(...a)=>"
                "(await fetch('/__api/'+m,{method:'POST',body:JSON.stringify(a)})).json();"
                f"const announce=()=>{{window.pywebview={{api}};"
                "window.dispatchEvent(new Event('pywebviewready'));};"
                # pywebview injects its bridge after the page has loaded, so the delay
                # reproduces the first-render race rather than hiding it.
                f"window.addEventListener('DOMContentLoaded',()=>setTimeout(announce,{self.ready_delay_ms}));"
                "})();</script>"
            )
        scenario = (
            '<pre id="odk-result">pending</pre>'
            '<script type="module">'
            + SCENARIO_PRELUDE
            + "(async () => { let result; try { await ODK.sleep(1500); "
            + "result = await (async () => {" + self._scenario + "})(); }"
            + " catch (error) { result = { scenarioError: String(error && error.stack || error) }; }"
            + " document.getElementById('odk-result').textContent = JSON.stringify(result); })();"
            + "</script>"
        )
        if self.head_script:
            shim = "<script>" + self.head_script + "</script>" + shim
        html = html.replace("<head>", "<head>" + shim, 1)
        return html.replace("</body>", scenario + "</body>", 1)

    def run(self, scenario: str, *, query: str = "", virtual_time_ms: int = 60000,
            timeout_s: float = 240) -> dict[str, Any]:
        """Load the cockpit, run `scenario` (the body of an async function), return what
        it returned."""
        self._scenario = scenario
        argv = dump_dom_command(
            "the cockpit end-to-end evidence", self.profile_dir,
            f"{self.url}/__scenario.html{query}", virtual_time_ms=virtual_time_ms,
        )
        # Every host but this loopback fails to resolve at once, so the satellite basemap
        # cannot hold the virtual clock waiting on a network this test must not use.
        if not self.network:
            argv.insert(-1, "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1")
        if self.window_size:
            argv.insert(-1, f"--window-size={self.window_size}")
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                                   encoding="utf-8", errors="replace")
        found = re.search(r'<pre id="odk-result">(.*?)</pre>', completed.stdout, re.DOTALL)
        assert found, f"the page never rendered a result:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        raw = found.group(1)
        assert raw != "pending", "the scenario did not finish inside the virtual time budget"
        result = json.loads(_unescape(raw))
        assert "scenarioError" not in result, result.get("scenarioError")
        return result

    def called(self, method: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]


def _unescape(text: str) -> str:
    return (text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            .replace("&#39;", "'").replace("&amp;", "&"))


def make_api(tmp_path: Path):
    """A real Api on a store in tmp_path, with one project open."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    store = ProjectStore(tmp_path / "state" / "projects.db")
    session = AppSession(store)
    api = Api(session)
    created = api.create_project("Harness Site", str(tmp_path / "project"))
    assert created["ok"], created
    return api


def write_geotagged_jpeg(path: Path, lon: float, lat: float, alt_m: float = 60.0) -> Path:
    """A small JPEG carrying a real EXIF GPS tag, read back by core.geo like any camera's."""
    from fractions import Fraction

    from PIL import Image

    def dms(value: float) -> tuple:
        value = abs(value)
        degrees = int(value)
        minutes_float = (value - degrees) * 60
        minutes = int(minutes_float)
        seconds = Fraction((minutes_float - minutes) * 60).limit_denominator(100000)
        return (Fraction(degrees), Fraction(minutes), seconds)

    image = Image.new("RGB", (64, 48), (90, 110, 70))
    exif = Image.Exif()
    exif[0x010F] = "HarnessCam"
    exif[0x0110] = "HC-1"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N" if lat >= 0 else "S"
    gps[2] = dms(lat)
    gps[3] = "E" if lon >= 0 else "W"
    gps[4] = dms(lon)
    gps[5] = 0
    gps[6] = Fraction(alt_m).limit_denominator(1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", exif=exif)
    return path
