"""The cockpit's measurement buttons, clicked in a real browser against the real Api.

Every measurement button in the cockpit used to fail: Volume and Cut & Fill asked
measure_on_raster for a kind it does not have, Slope passed null as the raster, Stockpile
called build_asset_inventory with no instances, Compare called compare_surveys with no
arguments, and the Distance and Area tools set a variable nothing read. The text-level
tests all passed throughout, because each named an Api method that existed.

So this serves app/web from a loopback port, stands in for pywebview with a bridge that
forwards every call to a real ``app.api.Api`` over HTTP, and drives the real Shell in
headless Edge/Chromium: draw on the map (through the same vertex/finish functions a map
click and a double-click call), press the toolbar buttons, and read back what the
application answered. The Api writes to a real project whose DSM and DTM have a known
volume, so the numbers asserted are the measurement, not a round trip of a stub.

The only stand-ins are at the boundary a headless page cannot cross: the modal prompts are
answered by the harness (WebView2 dialogs need a person), and a map click is delivered as
its lon/lat because the headless page has no pointer.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from browser_evidence import dump_dom_command
from conftest import BLOCK_VOLUME_M3, GRID, BLOCK_CELLS, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"
EPSG = 32617

HARNESS = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="css/workspace.css"></head>
<body data-result="pending"><div id="app"></div><pre id="out"></pre>
<script src="vendor/maplibre/maplibre-gl.js"></script>
<script>
  // The pywebview bridge, forwarded to a real Api over HTTP.
  window.pywebview = { api: new Proxy({}, { get: (_t, name) => (typeof name !== "string")
    ? undefined : (...args) => fetch("/api/" + name, { method: "POST",
        body: JSON.stringify(args) }).then((r) => r.json()) }) };
</script>
<script type="module">
import { Shell } from "./js/workspace/shell.js";
import { addVertex, finishDrawing } from "./js/workspace/draw.js";

const PLAN = __PLAN__;
const log = [];
const shell = new Shell(document.getElementById("app"));
const toast = shell.toast.bind(shell);
shell.toast = (text, kind) => { log.push({ text: String(text), kind: kind || "" }); toast(text, kind); };
const answers = [];
const context = shell.actionContext.bind(shell);
shell.actionContext = () => ({ ...context(), prompt: async () => answers.shift() ?? "" });
shell.confirmAll = true;
const settle = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (test) => { for (let i = 0; i < 200 && !test(); i++) await settle(25); };

try {
  await shell.refreshState();
  shell.open("measurements");
  const results = {};
  for (const step of PLAN) {
    const before = log.length;
    answers.splice(0, answers.length, ...(step.answers || []));
    const running = shell.runAction(step.action);
    if (step.draw) {
      await running;
      step.draw.forEach((p) => addVertex(p));
      finishDrawing();
    } else {
      await running;
    }
    await until(() => log.slice(before).some((t) => t.kind === "ok" || t.kind === "error")
      && !log.slice(before).every((t) => /click points|draw the/i.test(t.text)));
    await settle(50);
    results[step.name] = log.slice(before).map((t) => `[${t.kind}] ${t.text}`);
  }
  const history = await window.pywebview.api.list_measurements();
  results.history = (history.measurements || []).map((m) => m.measurement_type);
  await settle(300);
  results.historyPanel = [...document.querySelectorAll("td")].map((td) => td.textContent).join("|");
  document.getElementById("out").textContent = JSON.stringify(results);
  document.body.dataset.result = "done";
} catch (error) {
  document.body.dataset.result = "error:" + (error && error.message);
}
</script></body></html>"""


def _lonlat(points_xy):
    lon, lat = geo.projected_to_wgs84([p[0] for p in points_xy], [p[1] for p in points_xy], EPSG)
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def _serve(api: Api, plan: list[dict], pick: str):
    page = HARNESS.replace("__PLAN__", json.dumps(plan)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 - silence the test output
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
            extra = getattr(type(self), "extra_pages", {})
            if path in extra:
                return self._send(extra[path], "text/html")
            target = (WEB_ROOT / path.lstrip("/")).resolve()
            if WEB_ROOT not in target.parents or not target.is_file():
                return self._send(b"", "text/plain", 404)
            types = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
                     ".png": "image/png", ".svg": "image/svg+xml"}
            return self._send(target.read_bytes(), types.get(target.suffix, "application/octet-stream"))

        def do_POST(self):
            name = self.path.rsplit("/", 1)[-1]
            length = int(self.headers.get("Content-Length") or 0)
            args = json.loads(self.rfile.read(length) or b"[]")
            if name.startswith("_") or not callable(getattr(api, name, None)):
                result = {"ok": False, "error": f"unwired: {name}"}
            elif name == "pick_file":
                result = {"ok": True, "path": pick}   # the native dialog, answered
            else:
                result = getattr(api, name)(*args)
            return self._send(json.dumps(result, default=str).encode("utf-8"), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def project(tmp_path, surface_pair):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    api.create_project("Browser survey", str(tmp_path / "project"))
    session.register_raster_layer(surface_pair["dsm"], name="DSM")
    session.register_raster_layer(surface_pair["dtm"], name="DTM")
    # An earlier survey on its own grid: flat ground, before the block was placed.
    earlier = tmp_path / "earlier_dsm.tif"
    geo.write_geotiff(earlier, np.full((60, 60), 100.0, dtype=np.float32), epsg=EPSG,
                      west=ORIGIN_X - 2.0, north=ORIGIN_Y + 2.0, pixel_size=0.7, cog=False)
    return api, str(earlier)


def test_measurement_buttons_measure_the_project_in_a_browser(tmp_path, project):
    api, earlier = project
    start = (GRID - BLOCK_CELLS) // 2
    margin = 2
    west = ORIGIN_X + (start - margin) * PIXEL_SIZE_M
    east = ORIGIN_X + (start + BLOCK_CELLS + margin) * PIXEL_SIZE_M
    north = ORIGIN_Y - (start - margin) * PIXEL_SIZE_M
    south = ORIGIN_Y - (start + BLOCK_CELLS + margin) * PIXEL_SIZE_M
    square = _lonlat([(west, south), (east, south), (east, north), (west, north)])
    mid_y = ORIGIN_Y - (start + BLOCK_CELLS / 2) * PIXEL_SIZE_M
    line = _lonlat([(ORIGIN_X + 2.0, mid_y), (ORIGIN_X + 38.0, mid_y)])

    plan = [
        {"name": "distance", "action": "Distance", "draw": line},
        {"name": "area", "action": "Area", "draw": square},
        {"name": "volume", "action": "Volume"},
        {"name": "cutfill", "action": "Cut & Fill", "answers": ["101"]},
        {"name": "slope", "action": "Slope"},
        {"name": "stockpile", "action": "Stockpile", "answers": [""]},
        {"name": "profile", "action": "Profile", "draw": line},
        {"name": "ponding", "action": "Ponding", "answers": ["0.05"]},
        {"name": "compare", "action": "Compare Dates", "answers": ["0.05, 0.05, 0.05"]},
        {"name": "crs", "action": "Set CRS", "answers": ["32617"]},
    ]
    server, thread = _serve(api, plan, earlier)
    try:
        completed = subprocess.run(dump_dom_command(
            "the cockpit measurement evidence test", tmp_path / "edge-profile",
            f"http://127.0.0.1:{server.server_port}/harness.html",
            virtual_time_ms=60000, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    dom = completed.stdout
    assert 'data-result="done"' in dom, dom[-3000:]
    raw = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)
    import html

    results = json.loads(html.unescape(raw))
    text = {k: " ".join(v) if isinstance(v, list) else v for k, v in results.items()}

    # Each button reported the measurement, not an error.
    for step in ("distance", "area", "volume", "cutfill", "slope", "stockpile", "profile",
                 "ponding", "compare", "crs"):
        assert "[error]" not in text[step], (step, text[step])
    assert "distance 36 m" in text["distance"]
    assert "over the ground" in text["distance"]
    assert f"{BLOCK_VOLUME_M3:,.0f} m³ above" in text["volume"].replace(" ", ",")
    assert "the dtm" in text["volume"]
    assert "a 101 m plane" in text["cutfill"]
    assert "slope mean" in text["slope"]
    assert "Report:" in text["stockpile"]
    assert "profile 36 m long" in text["profile"]
    assert "detection floor" in text["ponding"] or "depression" in text["ponding"]
    assert "Added" in text["compare"]
    assert "region(s) moved" in text["compare"]
    assert "EPSG:32617" in text["crs"]

    # And each was kept.
    assert sorted(results["history"]) == sorted(
        ["distance", "area", "volume", "cut_fill", "slope", "stockpile", "profile",
         "surface_change"])
    assert "volume" in results["historyPanel"]


MAP_HARNESS = r"""<!doctype html><html><head><meta charset="utf-8"></head>
<body data-result="pending"><div id="map" style="width:640px;height:400px"></div><pre id="out"></pre>
<script src="vendor/maplibre/maplibre-gl.js"></script>
<script src="vendor/maplibre/maplibre-gl-draw.js"></script>
<script src="js/map.js"></script>
<script>
const LINE = __LINE__;
const post = (name, ...args) => fetch("/api/" + name, { method: "POST",
  body: JSON.stringify(args) }).then((r) => r.json());
const style = { version: 8, sources: {}, layers: [{ id: "bg", type: "background",
  paint: { "background-color": "#0d1117" } }] };
let viewer;
viewer = new OdkMap("map", { center: LINE[0], zoom: 17, style,
  // The same call app.js makes from its onMeasureDone.
  onMeasureDone: async (kind, points) => {
    const result = await post("measure_drawn", kind, points);
    document.getElementById("out").textContent = JSON.stringify({ kind, points: points.length, result });
    document.body.dataset.result = "done";
  },
  onReady: () => {
    viewer.setTool("measure-distance");
    for (const p of LINE) viewer.map.fire("click", { lngLat: { lng: p[0], lat: p[1] } });
    const last = LINE[LINE.length - 1];
    viewer.map.fire("click", { lngLat: { lng: last[0], lat: last[1] } });
    viewer.map.fire("dblclick", { lngLat: { lng: last[0], lat: last[1] }, preventDefault() {} });
  } });
</script></body></html>"""


def test_the_default_map_saves_a_measurement_on_double_click(tmp_path, project):
    """The launcher's map measured only in the status bar and kept nothing."""
    api, _earlier = project
    start = (GRID - BLOCK_CELLS) // 2
    mid_y = ORIGIN_Y - (start + BLOCK_CELLS / 2) * PIXEL_SIZE_M
    line = _lonlat([(ORIGIN_X + 2.0, mid_y), (ORIGIN_X + 38.0, mid_y)])
    page = MAP_HARNESS.replace("__LINE__", json.dumps(line))
    server, thread = _serve(api, [], "")
    server.RequestHandlerClass.extra_pages = {"/map.html": page.encode("utf-8")}
    try:
        completed = subprocess.run(dump_dom_command(
            "the default-map measurement evidence test", tmp_path / "edge-map",
            f"http://127.0.0.1:{server.server_port}/map.html",
            virtual_time_ms=20000, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    dom = completed.stdout
    assert 'data-result="done"' in dom, dom[-2000:]
    import html

    payload = json.loads(html.unescape(re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)))
    assert payload["kind"] == "distance"
    assert payload["points"] == 2
    assert payload["result"]["ok"], payload
    assert payload["result"]["measurement"]["value"] == pytest.approx(36.0, abs=0.01)
    assert api.list_measurements()["count"] == 1
