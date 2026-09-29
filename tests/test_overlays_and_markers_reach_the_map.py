"""CAD overlays and site markers: placed where they belong, and actually drawn.

The audit found both recorded and never shown. A DXF imported with the wrong EPSG was
written out as an overlay full of Infinity -- not even valid JSON -- and reported as a
success; a correct one was written in UTM although GeoJSON is WGS84; images were routed
to the DXF reader; nothing ever drew an overlay; and a site marker, including a hazard,
never appeared on the map the route was planned over. A marker 30 km from the site was
accepted without a word.

The Api half runs against a real project with a real DSM. The browser half drives the
real cockpit Shell in headless Chromium against that Api: press Import CAD, press Add
Marker and click the map, then read back what the MapLibre map itself holds.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from browser_evidence import dump_dom_command
from conftest import GRID, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

pytest.importorskip("rasterio")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from core import geo  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"
EPSG = 32617


def _dxf(points_xy: list[tuple[float, float]]) -> str:
    """An ASCII DXF: one LINE and one closed LWPOLYLINE over the given UTM points."""
    (x0, y0), (x1, y1) = points_xy[0], points_xy[1]
    rows = ["0", "SECTION", "2", "ENTITIES",
            "0", "LINE", "10", f"{x0}", "20", f"{y0}", "11", f"{x1}", "21", f"{y1}",
            "0", "LWPOLYLINE", "70", "1"]
    for x, y in points_xy:
        rows += ["10", f"{x}", "20", f"{y}"]
    rows += ["0", "INSERT", "2", "BLOCK", "0", "ENDSEC", "0", "EOF"]
    return "\n".join(rows) + "\n"


SITE_CORNERS = [
    (ORIGIN_X + 5.0, ORIGIN_Y - 5.0),
    (ORIGIN_X + 35.0, ORIGIN_Y - 5.0),
    (ORIGIN_X + 35.0, ORIGIN_Y - 35.0),
    (ORIGIN_X + 5.0, ORIGIN_Y - 35.0),
]


@pytest.fixture
def project(tmp_path, surface_pair):
    store = ProjectStore(tmp_path / "projects.db")
    session = AppSession(store=store)
    api = Api(session)
    api.create_project("Overlay survey", str(tmp_path / "project"))
    session.register_raster_layer(surface_pair["dsm"], name="DSM")
    drawing = tmp_path / "site_plan.dxf"
    drawing.write_text(_dxf(SITE_CORNERS), encoding="utf-8")
    return api, session, drawing


class TestCadOverlayThroughTheApi:
    def test_the_overlay_is_written_in_wgs84_and_lands_on_the_survey(self, project):
        api, session, drawing = project
        result = api.import_cad_overlay(str(drawing), EPSG, "plan")
        assert result["ok"], result
        assert result["alignment"]["overlaps"] is True
        assert result["skipped"] == {"INSERT": 1}

        geojson = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        coords = [c for f in geojson["features"] for c in f["geometry"]["coordinates"]]
        lon, lat = geo.projected_to_wgs84([ORIGIN_X + 5.0], [ORIGIN_Y - 5.0], EPSG)
        assert all(np.isfinite(c).all() for c in np.asarray(coords))
        # Degrees near the survey, not UTM metres.
        assert abs(coords[0][0] - float(lon[0])) < 1e-6
        assert abs(coords[0][1] - float(lat[0])) < 1e-6
        # And it is a layer the tree lists.
        assert any(l["id"] == result["layer_id"] for l in api.list_layers()["layers"])

        listed = api.list_cad_overlays()["overlays"]
        assert [o["name"] for o in listed] == ["plan"]
        assert listed[0]["geojson"]["features"]

    def test_the_wrong_crs_is_refused_and_nothing_is_written(self, project):
        api, _session, drawing = project
        result = api.import_cad_overlay(str(drawing), 4326, "wrong")
        assert not result["ok"]
        assert "not in EPSG:4326" in result["error"]
        assert api.list_cad_overlays()["overlays"] == []

    def test_a_plain_image_is_placed_as_a_georeferenced_raster(self, project, tmp_path):
        import cv2
        import rasterio

        api, _session, _drawing = project
        picture = tmp_path / "scan.png"
        cv2.imwrite(str(picture), np.full((40, 60, 3), 200, dtype=np.uint8))
        result = api.place_image_overlay(str(picture), ORIGIN_X, ORIGIN_Y - 40.0,
                                         ORIGIN_X + 40.0, ORIGIN_Y, EPSG, "scan")
        assert result["ok"], result
        with rasterio.open(result["path"]) as src:
            assert src.crs.to_epsg() == EPSG
            assert src.bounds.left == pytest.approx(ORIGIN_X)
            assert src.bounds.top == pytest.approx(ORIGIN_Y)
        assert any(o["kind"] == "image" for o in api.list_cad_overlays()["overlays"])

    def test_overlays_come_back_after_a_restart(self, project, tmp_path):
        api, session, drawing = project
        assert api.import_cad_overlay(str(drawing), EPSG, "plan")["ok"]
        # A new session over the same store: what the next launch sees.
        fresh = AppSession(store=session.store)
        fresh.set_active_project(session.project_id())
        assert any(l.name == "CAD: plan" for l in fresh.layers.values())


class TestSiteMarkersThroughTheApi:
    def test_a_marker_kilometres_from_the_site_is_kept_but_questioned(self, project):
        api, session, _drawing = project
        lon, lat = geo.projected_to_wgs84([ORIGIN_X + 20.0], [ORIGIN_Y - 20.0], EPSG)
        session.aoi_polygon = [[float(lon[0]) + dx, float(lat[0]) + dy]
                               for dx, dy in ((-0.001, -0.001), (0.001, -0.001),
                                              (0.001, 0.001), (-0.001, 0.001))]
        near = api.add_site_marker("Mast", "hazard", [[float(lon[0]), float(lat[0])]], "", 15)
        assert near["ok"] and "warning" not in near
        far = api.add_site_marker("Typo", "hazard", [[float(lon[0]) + 0.35, float(lat[0])]])
        assert far["ok"]
        assert "km from the area of interest" in far["warning"]
        listed = api.list_site_markers()
        assert len(listed["geojson"]["features"]) == 2


HARNESS = r"""<!doctype html><html><head><meta charset="utf-8">
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
import { addVertex, canDraw } from "./js/workspace/draw.js";
import { shownVectors } from "./js/workspace/mapview.js";

const MARKER = __MARKER__;
const log = [];
const shell = new Shell(document.getElementById("app"));
const toast = shell.toast.bind(shell);
shell.toast = (text, kind) => { log.push({ text: String(text), kind: kind || "" }); toast(text, kind); };
const answers = [];
const context = shell.actionContext.bind(shell);
shell.actionContext = () => ({ ...context(),
  prompt: async () => answers.shift() ?? "",
  choose: async (_title, options) => options.find((o) => o.startsWith("hazard")) });
shell.confirmAll = true;
const settle = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (test) => { for (let i = 0; i < 400 && !test(); i++) await settle(25); };
const outcome = (before) => log.slice(before).filter((t) => t.kind === "ok" || t.kind === "error")
  .map((t) => `[${t.kind}] ${t.text}`).filter((t) => !/Click where/.test(t));

try {
  await shell.refreshState();
  shell.open("measurements");
  await until(() => canDraw());
  const results = { canDraw: canDraw() };

  let before = log.length;
  answers.splice(0, answers.length, "32617");
  await shell.runAction("Import CAD");
  await until(() => outcome(before).length);
  results.cad = outcome(before);

  before = log.length;
  answers.splice(0, answers.length, "Mast");
  const running = shell.runAction("Add Marker");
  await until(() => log.slice(before).some((t) => /Click where/.test(t.text)));
  addVertex(MARKER);             // the map click, delivered as its lon/lat
  await running;
  await until(() => outcome(before).length);
  results.marker = outcome(before);

  await until(() => (shownVectors()["site-markers"] || 0) >= 1 && shownVectors()["cad:site_plan"]);
  results.shown = shownVectors();
  document.getElementById("out").textContent = JSON.stringify(results);
  document.body.dataset.result = "done";
} catch (error) {
  document.body.dataset.result = "error:" + (error && error.message);
}
</script></body></html>"""


def _serve(api: Api, page: bytes, pick: str):
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
            elif name == "pick_file":
                result = {"ok": True, "path": pick}   # the native dialog, answered
            else:
                result = getattr(api, name)(*args)
            return self._send(json.dumps(result, default=str).encode("utf-8"),
                              "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_import_cad_and_add_marker_draw_on_the_cockpit_map(tmp_path, project):
    api, _session, drawing = project
    centre = GRID * PIXEL_SIZE_M / 2.0
    lon, lat = geo.projected_to_wgs84([ORIGIN_X + centre], [ORIGIN_Y - centre], EPSG)
    marker = [float(lon[0]), float(lat[0])]
    page = HARNESS.replace("__MARKER__", json.dumps(marker)).encode("utf-8")
    server, thread = _serve(api, page, str(drawing))
    try:
        completed = subprocess.run(dump_dom_command(
            "the CAD overlay and site marker map test", tmp_path / "edge-profile",
            f"http://127.0.0.1:{server.server_port}/harness.html",
            # The steps settle in under 8 s of virtual time. A longer budget only keeps
            # the live map rendering in software after the answer is already written.
            virtual_time_ms=15000, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    dom = completed.stdout
    assert 'data-result="done"' in dom, (dom[-3000:], completed.stderr[-2000:], completed.returncode)
    results = json.loads(html.unescape(re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)))
    assert results["canDraw"] is True
    assert any("Placed 2 entities from site_plan" in t for t in results["cad"]), results["cad"]
    assert any('Added hazard "Mast"' in t for t in results["marker"]), results["marker"]

    # What the MapLibre map holds, not what was asked for.
    assert results["shown"].get("cad:site_plan") == 2, results["shown"]
    assert results["shown"].get("site-markers") == 1, results["shown"]

    # The click placed the marker where it was clicked.
    stored = api.list_site_markers()["markers"][0]
    assert stored["points"][0] == pytest.approx(marker, abs=1e-9)
