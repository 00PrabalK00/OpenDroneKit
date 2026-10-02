"""CAD overlays and site markers reach the v1 map, in the v1 paint.

The port of tests/test_overlays_and_markers_reach_the_map.py (whose Api half stays where
it is and still runs) to the new map module: js/odk/map/furniture.js on mapview.js.
The audit behind the original found both recorded and never drawn -- a hazard marked on
the site did not appear on the map the route was planned over. Here the real Api, a
real DSM and the real verbs (Import CAD, Add Marker from js/workspace/actions.js) run
in headless Chromium against the v1 map, and what is read back is what MapLibre holds:

* the overlay's two entities and the marker are sources on the map;
* the marker sits where the map was clicked;
* the Layers popover lists both under Overlays and can hide them (casing included);
* the overlay and the marker are restyled by map/theme.js, desk and Field.
"""

from __future__ import annotations

import json

import pytest

from conftest import GRID, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M
from test_overlays_and_markers_reach_the_map import EPSG, SITE_CORNERS, _dxf
from ui_v1_map_harness import MapBrowser, MapServer, chromium_path, stage_page

pytest.importorskip("rasterio")
pytestmark = pytest.mark.skipif(chromium_path() is None, reason="No Chromium or Edge to drive.")

SCRIPT = r"""
import { mountProjectMap } from "./js/odk/map/furniture.js";
import { ACTIONS } from "./js/workspace/actions.js";
import { addVertex, armDrawing, canDraw } from "./js/workspace/draw.js";
import { refreshProjectOverlays, shownVectors } from "./js/workspace/mapview.js";

const MARKER = __MARKER__;
const out = window.__u3 = { toasts: [], errors: [] };
window.addEventListener("unhandledrejection", (e) => out.errors.push(String(e.reason && e.reason.stack || e.reason)));
const settle = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (test, n = 800) => { for (let i = 0; i < n && !test(); i++) await settle(25); return test(); };
try {
  const mounted = await mountProjectMap(document.getElementById("map"), {
    centre: MARKER, zoom: 17, notify: (text, kind) => out.toasts.push([kind, text]) });
  const map = out.map = mounted.map;
  await until(() => map._odkProjectLayers && canDraw());

  // The context the v1 shell hands a verb: answers queued for its questions, the map
  // click delivered as the lon/lat it lands on.
  const answers = [];
  const ctx = {
    prompt: async () => answers.shift() ?? "",
    choose: async (_title, options) => options.find((o) => o.startsWith("hazard")),
    confirm: async () => true,
    toast: (text, kind) => out.toasts.push([kind, text]),
    pickPoint: () => new Promise((resolve) => {
      armDrawing("point", (geometry) => resolve(geometry.coordinates), () => resolve(null));
      setTimeout(() => addVertex(MARKER), 50);
    }),
  };
  answers.push("32617");
  out.cad = await ACTIONS["Import CAD"].run(ctx);
  answers.push("Mast", "0");
  out.marker = await ACTIONS["Add Marker"].run(ctx);
  refreshProjectOverlays();   // what the shell does on {refresh: true}
  await until(() => (shownVectors()["site-markers"] || 0) >= 1 && shownVectors()["cad:site_plan"]);
  out.shown = shownVectors();
  await until(() => map.getLayer("odk-v-cad_site_plan-line-casing"));

  const paint = (id, name) => map.getLayer(id) ? map.getPaintProperty(id, name) : null;
  const read = () => ({
    cadLine: paint("odk-v-cad_site_plan-line", "line-color"),
    cadWidth: paint("odk-v-cad_site_plan-line", "line-width"),
    cadCasing: paint("odk-v-cad_site_plan-line-casing", "line-width"),
    markerColour: paint("odk-v-site-markers-point", "circle-color"),
    markerRadius: paint("odk-v-site-markers-point", "circle-radius"),
  });
  out.desk = read();
  document.documentElement.setAttribute("data-mode", "field");
  await until(() => map._odkTheme === "field");
  out.field = read();
  document.documentElement.setAttribute("data-mode", "desk");
  await until(() => map._odkTheme === "desk");

  // The Layers popover lists both, and hiding one hides its casing too.
  document.querySelector("[data-control=layers]").click();
  await until(() => document.querySelector(".odk-mapf-pop [data-overlay]"));
  out.overlays = [...document.querySelectorAll(".odk-mapf-pop [data-overlay]")].map((r) =>
    [r.dataset.overlay, r.textContent.trim()]);
  document.querySelector('.odk-mapf-pop [data-overlay="cad:site_plan"] input').click();
  out.hidden = ["odk-v-cad_site_plan-line", "odk-v-cad_site_plan-line-casing"].map((id) =>
    map.getLayoutProperty(id, "visibility"));
  out.key = [...document.querySelectorAll(".odk-mapf-legend .odk-geom-item")].map((i) => i.textContent);
  out.dbg = map.getStyle().layers.filter((l) => l.id.startsWith("odk-v-site")).map((l) => [l.id,
    map.getLayoutProperty(l.id, "visibility") || "", JSON.stringify(map.getSource(l.source)._data).slice(0, 200)]);
  out.done = true;
} catch (error) {
  out.error = String(error && error.stack || error);
}
"""


@pytest.fixture
def project(tmp_path, surface_pair):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(store=ProjectStore(tmp_path / "projects.db")))
    api.create_project("Overlay survey", str(tmp_path / "project"))
    api._session.register_raster_layer(surface_pair["dsm"], name="DSM")
    drawing = tmp_path / "site_plan.dxf"
    drawing.write_text(_dxf(SITE_CORNERS), encoding="utf-8")
    return api, drawing


def test_import_cad_and_add_marker_draw_on_the_v1_map(tmp_path, project) -> None:
    from core import geo

    api, drawing = project
    centre = GRID * PIXEL_SIZE_M / 2.0
    lon, lat = geo.projected_to_wgs84([ORIGIN_X + centre], [ORIGIN_Y - centre], EPSG)
    marker = [float(lon[0]), float(lat[0])]
    server = MapServer(api, stage_page(SCRIPT.replace("__MARKER__", json.dumps(marker))), pick=str(drawing))
    browser = MapBrowser(chromium_path())
    try:
        browser.viewport(1440, 900)
        browser.open(f"{server.base}/__stage.html", settle_s=0.5)
        browser.wait_for("window.__u3 && (window.__u3.done || window.__u3.error)", timeout_s=240)
        result = browser.js("JSON.parse(JSON.stringify(window.__u3, (k, v) => k === 'map' ? undefined : v))")
    finally:
        browser.close()
        server.close()

    assert not result.get("error"), result.get("error")
    assert "Placed 2 entities from site_plan" in result["cad"]["message"], result["cad"]
    assert 'Added hazard "Mast"' in result["marker"]["message"], result["marker"]

    # What the MapLibre map holds, not what was asked for.
    assert result["shown"].get("cad:site_plan") == 2, result["shown"]
    assert result["shown"].get("site-markers") == 1, result["shown"]

    # The click placed the marker where it was clicked.
    stored = api.list_site_markers()["markers"][0]
    assert stored["points"][0] == pytest.approx(marker, abs=1e-9)

    # Restyled for each mode by map/theme.js: a hairline on a dark casing at the desk,
    # a 4 px stroke on an 8 px ink casing in the field.
    assert result["desk"] == {"cadLine": "#B4BCC8", "cadWidth": 1, "cadCasing": 3,
                              "markerColour": "#F5A524", "markerRadius": 5}, result["desk"]
    assert result["field"] == {"cadLine": "#FFFFFF", "cadWidth": 4, "cadCasing": 8,
                               "markerColour": "#B91C1C", "markerRadius": 9}, result["field"]

    overlays = dict(result["overlays"])
    assert overlays["cad:site_plan"] == "CAD · site_plan (2)"
    assert overlays["site-markers"] == "Site markers (1)"
    assert result["hidden"] == ["none", "none"]
    # Hidden overlays leave the key; the marker stays in it.
    assert "site marker" in result["key"], result["dbg"]
    assert "site marker" in result["key"] and "CAD overlay" not in result["key"], result["key"]
