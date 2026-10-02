"""The v1 map module on the real Aukerman survey: furniture, layers, deep zoom, themes.

U3 of the rebuild (docs/ui/ARCHITECTURE.md). js/odk/map/furniture.js and theme.js sit on
the cockpit's mapview.js (extended, not forked) and are driven here in headless
Chromium against the production Api, over a copy of the full-resolution Aukerman
reconstruction (orthomosaic 0.033 m/px, DSM 1.23 m/px, EPSG:32617). Nothing the page
shows is checked against a number written into the UI: every expectation is read from
the Api, from the GeoTIFFs, or from MapLibre's own view.

What is proved, against the map the operator sees:

* the orthomosaic is drawn from tiles cut from the GeoTIFF (Api.raster_tiles) at
  z21-z22, and is sharper there than the old single preview;
* no Esri "Map data not yet available" placeholder reaches the screen;
* the scale bar's metres-per-pixel matches MapLibre's own projection within 2 %, and its
  label, bar length and 1 : N ratio agree with it;
* the CRS chip shows the project's EPSG and copies PROJ's WKT;
* the north arrow turns with the bearing and a click puts north up;
* the Layers popover's visibility, opacity and remove reach both the Api and the map;
* the elevation legend is the range the DSM tiles are coloured with;
* Field mode draws every line on an 8 px ink casing under a 4 px stroke.

Screenshots of the desk and field map views are written to ODK_U3_SCREENS (or the
test's temporary folder) for comparison with the SysAnalyse / SysFieldAnalyse artboards.

The survey is not in the repository. ODK_AUKERMAN_FULLRES names the folder holding
its reconstruction/ (orthomosaic.tif, dsm.tif, dtm.tif, dsm_hillshade.tif,
camera_track.geojson); the tests skip when it is absent.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ui_v1_map_harness import MapBrowser, MapServer, chromium_path, decode_png, stage_page

pytest.importorskip("rasterio")

REPO = Path(__file__).resolve().parents[1]
# The variable may name the survey folder or its reconstruction/ itself.
AUKERMAN = Path(os.environ.get("ODK_AUKERMAN_FULLRES", REPO / "demo_project" / "aukerman_fullres"))
if (AUKERMAN / "reconstruction").is_dir():
    AUKERMAN = AUKERMAN / "reconstruction"
EPSG = 32617
IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"

needs_survey = pytest.mark.skipif(not (AUKERMAN / "orthomosaic.tif").is_file(),
                                  reason="The full-resolution Aukerman survey is not on this machine "
                                         "(set ODK_AUKERMAN_FULLRES).")
needs_browser = pytest.mark.skipif(chromium_path() is None, reason="No Chromium or Edge to drive.")
pytestmark = [needs_survey, needs_browser]


def _esri_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{IMAGERY}/tilemap/10/0/0/1/1", timeout=8) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - offline is a reason to skip, not to fail
        return False


# ---------------------------------------------------------------------- the page

SCRIPT = r"""
import { mountProjectMap, metresPerPixel } from "./js/odk/map/furniture.js";
import * as mapview from "./js/workspace/mapview.js";
import * as theme from "./js/odk/map/theme.js";

const u3 = window.__u3 = { toasts: [], errors: [], mapview, theme, metresPerPixel };
window.addEventListener("error", (e) => u3.errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => u3.errors.push(String(e.reason && e.reason.stack || e.reason)));
try {
  const mounted = await mountProjectMap(document.getElementById("map"), {
    centre: __CENTRE__, zoom: 17,
    notify: (text, kind) => u3.toasts.push([kind, text]),
  });
  u3.map = mounted.map;
  u3.furniture = mounted.furniture;
  u3.map.on("error", (e) => u3.errors.push(String(e && e.error && e.error.message || e)));
  u3.settle = () => new Promise((resolve) => {
    const map = u3.map;
    const done = () => setTimeout(resolve, 60);
    map.once("idle", done);
    map.triggerRepaint();
  });
  u3.box = () => u3.map.getContainer().getBoundingClientRect().toJSON();
  u3.rasterOf = (name) => (u3.map._odkProjectLayers || []).find((d) => d.layer.name === name) || null;
  u3.ready = true;
} catch (error) {
  u3.error = String(error && error.stack || error);
}
"""


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(h))


def _sharpness(rgb: np.ndarray) -> float:
    """Mean absolute Laplacian of the grey image: how much fine detail reached the screen."""
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(np.abs(cv2.Laplacian(grey, cv2.CV_64F)).mean())


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    root = tmp_path_factory.mktemp("u3-map")
    os.environ.setdefault("ODK_STORAGE_PATH", str(root / "storage"))
    api = Api(AppSession(ProjectStore(root / "projects.db")))
    assert api.create_project("Aukerman Park", str(root / "project"))["ok"]
    session = api._session
    layers = {}
    for filename, name in (("orthomosaic.tif", "Orthomosaic"), ("dsm.tif", "DSM"),
                           ("dtm.tif", "DTM"), ("dsm_hillshade.tif", "DSM hillshade")):
        layers[name] = session.register_raster_layer(AUKERMAN / filename, name=name)["id"]
    layers["Camera positions"] = session.register_vector_layer(
        AUKERMAN / "camera_track.geojson", name="Camera positions")["id"]
    # What the operator would see after choosing these in the Layers panel.
    api.set_layer_visible(layers["DTM"], False)
    api.set_layer_visible(layers["DSM hillshade"], False)
    api.set_layer_opacity(layers["DSM"], 0.85)

    # A survey area over the middle of the site, drawn through the Api like Plan does.
    cameras = json.loads((AUKERMAN / "camera_track.geojson").read_text(encoding="utf-8"))["features"]
    lon0 = float(np.mean([f["geometry"]["coordinates"][0] for f in cameras]))
    lat0 = float(np.mean([f["geometry"]["coordinates"][1] for f in cameras]))
    per_lat = 111_320.0
    per_lon = per_lat * math.cos(math.radians(lat0))
    ring = [[lon0 + x / per_lon, lat0 + y / per_lat]
            for x, y in ((-115, 58), (62, 70), (125, -18), (62, -76), (-90, -63))]
    assert api.set_aoi(ring)["ok"]

    ortho_tiles = api.raster_tiles(layers["Orthomosaic"])
    assert ortho_tiles["ok"] and ortho_tiles["maxzoom"] >= 21, ortho_tiles

    page = stage_page(SCRIPT.replace("__CENTRE__", json.dumps([lon0, lat0])))
    server = MapServer(api, page)
    browser = MapBrowser(chromium_path())
    browser.viewport(1440, 900)
    try:
        browser.open(f"{server.base}/__stage.html", settle_s=0.5)
        browser.wait_for("window.__u3 && (window.__u3.ready || window.__u3.error)", timeout_s=60)
        error = browser.js("window.__u3.error || ''")
        assert not error, error
        try:
            browser.wait_for("(window.__u3.map._odkProjectLayers || []).length >= 3", timeout_s=150)
        except AssertionError as missing:
            raise AssertionError((browser.js("JSON.stringify(window.__u3.errors)"), server.calls)) from missing
        browser.wait_for("document.querySelector('[data-furniture]').dataset.crsEpsg", timeout_s=30)
        browser.js("window.__u3.settle()")
        yield {"api": api, "browser": browser, "server": server, "layers": layers,
               "centre": (lon0, lat0), "ortho_tiles": ortho_tiles, "root": root}
    finally:
        browser.close()
        server.close()


def _screens_dir(stage) -> Path:
    return Path(os.environ.get("ODK_U3_SCREENS") or (stage["root"] / "screens"))


def _jump(browser, lon: float, lat: float, zoom: float, bearing: float = 0.0) -> None:
    browser.js(f"""(async () => {{ window.__u3.map.jumpTo({{ center: [{lon}, {lat}], zoom: {zoom},
      bearing: {bearing} }}); await window.__u3.settle(); }})()""", timeout_s=280)


def _map_shot(browser, inset: int = 0) -> np.ndarray:
    box = browser.js("window.__u3.box()")
    clip = {"x": box["x"] + inset, "y": box["y"] + inset,
            "width": box["width"] - 2 * inset, "height": box["height"] - 2 * inset}
    return decode_png(browser.screenshot(clip=clip))


# ------------------------------------------------------------------------ tests


class TestTheMapFurniture:
    def test_the_crs_chip_shows_the_projects_epsg_and_copies_its_wkt(self, stage) -> None:
        browser, api = stage["browser"], stage["api"]
        expected = api.describe_crs(EPSG)
        chip = browser.js("document.querySelector('[data-control=copy-crs]').textContent")
        assert chip == f"EPSG:{EPSG} · {expected['short']}", chip
        assert expected["short"] == "UTM 17N"
        copied = browser.js("""(async () => {
          let text = null;
          Object.defineProperty(navigator, 'clipboard', { configurable: true,
            value: { writeText: async (t) => { text = t; } } });
          document.querySelector('[data-control=copy-crs]').click();
          for (let i = 0; i < 40 && text === null; i++) await new Promise((r) => setTimeout(r, 25));
          return text;
        })()""")
        assert copied == expected["wkt"]
        assert copied.startswith("PROJCRS[") and "UTM zone 17N" in copied

    @pytest.mark.parametrize("zoom", [15.4, 17.0, 19.3, 21.7])
    def test_the_scale_bar_is_true_to_maplibres_projection(self, stage, zoom) -> None:
        browser = stage["browser"]
        lon, lat = stage["centre"]
        _jump(browser, lon, lat, zoom)
        seen = browser.js("""(() => {
          const map = window.__u3.map;
          const box = map.getContainer().getBoundingClientRect();
          const cx = box.width / 2, cy = box.height / 2;
          const a = map.unproject([cx - 50, cy]), b = map.unproject([cx + 50, cy]);
          const f = document.querySelector('[data-furniture]');
          const bar = f.querySelector('.odk-mapf-scale-bar').getBoundingClientRect();
          return { a: [a.lng, a.lat], b: [b.lng, b.lat], mpp: Number(f.dataset.mpp),
                   metres: Number(f.dataset.scaleMetres), barPx: bar.width,
                   label: f.querySelector('.odk-mapf-scale-labels').lastChild.textContent,
                   ratio: f.querySelector('.odk-mapf-scale-ratio').textContent };
        })()""")
        truth = _haversine(tuple(seen["a"]), tuple(seen["b"])) / 100.0
        assert seen["mpp"] == pytest.approx(truth, rel=0.02), (seen, truth)
        # The bar is as long as the distance it is labelled with.
        assert seen["barPx"] * truth == pytest.approx(seen["metres"], rel=0.02), seen
        value, unit = seen["label"].split(" ")
        metres = float(value.replace(",", "")) * {"km": 1000.0, "m": 1.0, "cm": 0.01}[unit]
        assert metres == pytest.approx(seen["metres"], rel=1e-6), seen
        # 1 : N at 96 dpi, from the same metres per pixel.
        denominator = float(seen["ratio"].split(":")[1].split("at")[0].replace(",", ""))
        assert denominator == pytest.approx(truth * 96 / 0.0254, rel=0.02), seen

    def test_the_north_arrow_turns_with_the_map_and_puts_north_up(self, stage) -> None:
        browser = stage["browser"]
        lon, lat = stage["centre"]
        _jump(browser, lon, lat, 18, bearing=37)
        turned = browser.js("""(() => {
          const north = document.querySelector('[data-control=north-up]');
          return { transform: north.querySelector('svg').style.transform,
                   bearing: Number(document.querySelector('[data-furniture]').dataset.bearing),
                   label: north.getAttribute('aria-label') };
        })()""")
        assert turned["bearing"] == pytest.approx(37)
        assert turned["transform"] == "rotate(-37deg)", turned
        assert "37°" in turned["label"]
        after = browser.js("""(async () => {
          document.querySelector('[data-control=north-up]').click();
          await new Promise((r) => setTimeout(r, 700));
          return { bearing: window.__u3.map.getBearing(),
                   transform: document.querySelector('[data-control=north-up] svg').style.transform };
        })()""")
        assert abs(after["bearing"]) < 0.01, after
        assert after["transform"] in ("rotate(0deg)", "rotate(-0deg)"), after

    def test_layers_popover_toggles_and_fades_a_layer_through_the_api(self, stage) -> None:
        browser, api, layers = stage["browser"], stage["api"], stage["layers"]
        lon, lat = stage["centre"]
        _jump(browser, lon, lat, 17)
        dsm = layers["DSM"]
        opened = browser.js("""(async () => {
          document.querySelector('[data-control=layers]').click();
          for (let i = 0; i < 80 && !document.querySelector('.odk-mapf-pop [data-layer]'); i++)
            await new Promise((r) => setTimeout(r, 25));
          const pop = document.querySelector('.odk-mapf-pop');
          return { hidden: pop.hidden,
                   rows: [...pop.querySelectorAll('[data-layer]')].map((r) => r.dataset.layer),
                   basemaps: [...pop.querySelectorAll('[data-basemap]')].map((r) =>
                     [r.dataset.basemap, Boolean(r.querySelector('.odk-mapf-net')), r.querySelector('input').checked]),
                   actions: [...pop.querySelectorAll('[data-action]')].map((b) => b.dataset.action) };
        })()""")
        assert not opened["hidden"]
        assert set(opened["rows"]) == set(layers.values())
        basemaps = {key: (network, checked) for key, network, checked in opened["basemaps"]}
        # Offline first, the online services marked as leaving the machine.
        assert basemaps["offline"] == (False, True)
        assert all(basemaps[k][0] for k in ("satellite", "street", "topo"))
        assert "Remove Layer" in opened["actions"] and "Cache Basemap View" in opened["actions"]

        state = browser.js(f"""(async () => {{
          const row = document.querySelector('.odk-mapf-pop [data-layer="{dsm}"]');
          const box = row.querySelector('input[type=checkbox]');
          const mapIds = window.__u3.rasterOf('DSM').mapLayers;
          const vis = () => window.__u3.map.getLayoutProperty(mapIds[0], 'visibility') || 'visible';
          const before = vis();
          box.click();
          for (let i = 0; i < 400 && vis() !== 'none'; i++) await new Promise((r) => setTimeout(r, 25));
          const hidden = vis();
          box.click();
          for (let i = 0; i < 400 && vis() !== 'visible'; i++) await new Promise((r) => setTimeout(r, 25));
          const shown = vis();
          const range = row.querySelector('input[type=range]');
          range.value = '40';
          range.dispatchEvent(new Event('input', {{ bubbles: true }}));
          range.dispatchEvent(new Event('change', {{ bubbles: true }}));
          // The opacity is painted at once and saved on "change"; wait for the Api to have it.
          for (let i = 0; i < 400; i++) {{
            const listed = await window.pywebview.api.list_layers();
            if (Math.abs(((listed.layers || []).find((l) => l.id === "{dsm}") || {{}}).opacity - 0.4) < 1e-6) break;
            await new Promise((r) => setTimeout(r, 50));
          }}
          return {{ before, hidden, shown,
                    opacity: window.__u3.map.getPaintProperty(mapIds[0], 'raster-opacity'),
                    output: row.querySelector('output').textContent }};
        }})()""")
        assert (state["before"], state["hidden"], state["shown"]) == ("visible", "none", "visible"), state
        assert state["opacity"] == pytest.approx(0.4) and state["output"] == "40%"
        listed = {l["id"]: l for l in api.list_layers()["layers"]}
        assert listed[dsm]["visible"] is True and listed[dsm]["opacity"] == pytest.approx(0.4)

        # Hide it for the deep-zoom checks that follow; the Api records it.
        browser.js(f"""(async () => {{
          const mapIds = window.__u3.rasterOf('DSM').mapLayers;
          document.querySelector('.odk-mapf-pop [data-layer="{dsm}"] input[type=checkbox]').click();
          for (let i = 0; i < 400 && window.__u3.map.getLayoutProperty(mapIds[0], 'visibility') !== 'none'; i++)
            await new Promise((r) => setTimeout(r, 25));
          document.querySelector('[data-control=layers]').click();
        }})()""")
        assert {l["id"]: l for l in api.list_layers()["layers"]}[dsm]["visible"] is False


class TestDeepZoom:
    def test_the_orthomosaic_is_drawn_from_its_own_tiles_at_z21_and_z22_and_is_sharp(self, stage) -> None:
        browser, server = stage["browser"], stage["server"]
        west, south, east, north = stage["ortho_tiles"]["bounds"]
        lon, lat = (west + east) / 2, (south + north) / 2
        sharp, preview = {}, {}
        token = stage["ortho_tiles"]["token"]
        native = stage["ortho_tiles"]["maxzoom"]
        for zoom in (21.0, 22.0):
            _jump(browser, lon, lat, zoom)
            # The tiles the map is drawing the orthomosaic with right now, by level.
            levels = browser.js("""(() => {
              const map = window.__u3.map;
              const id = window.__u3.rasterOf('Orthomosaic').mapLayers[0];
              const caches = map.style.sourceCaches || map.style.tileManagers || {};
              const cache = caches[map.getLayer(id).source];
              return [...new Set(cache.getVisibleCoordinates().map((t) => t.canonical.z))];
            })()""")
            # The survey's own native level: z22 for 3.3 cm/px at this latitude.
            assert levels == [native], (zoom, levels)
            served = {int(p.split("/")[3]) for p, code in server.raster_tiles if code == 200 and token in p}
            assert native in served, sorted(served)
            shot = _map_shot(browser, inset=200)
            sharp[zoom] = _sharpness(shot)
            # Survey pixels on screen, not an empty well.
            assert shot.reshape(-1, 3).std(axis=0).mean() > 8, zoom
            # The route the map used before: one preview no wider than 1024 px, drawn
            # over the same view. The tiles must carry clearly more detail.
            browser.js("""(async () => {
              const map = window.__u3.map;
              const entry = window.__u3.rasterOf('Orthomosaic');
              const p = await window.pywebview.api.raster_preview(entry.layer.id);
              map.addSource('u3-preview', { type: 'image', url: p.data_uri, coordinates: p.coordinates });
              map.addLayer({ id: 'u3-preview', type: 'raster', source: 'u3-preview' });
              await window.__u3.settle();
            })()""", timeout_s=90)
            preview[zoom] = _sharpness(_map_shot(browser, inset=200))
            browser.js("""(async () => { const map = window.__u3.map;
              map.removeLayer('u3-preview'); map.removeSource('u3-preview'); await window.__u3.settle(); })()""")
        # The preview is ~27 cm per pixel (8192 px of 3.3 cm squeezed into 1024). At z21
        # a screen pixel is ~6 cm, at z22 ~3 cm: the tiles' own pixels, enlarged less.
        assert sharp[21.0] > 3.0 * preview[21.0], (sharp, preview)
        assert sharp[22.0] > 1.5 * preview[22.0], (sharp, preview)
        # The resolution chip reports the product on top, from the Api.
        info = browser.js("document.querySelector('.odk-mapf-info').textContent")
        assert info.startswith("ORTHOMOSAIC · ") and info.endswith(" m/px"), info
        # The GeoTIFF's own pixel size, not the Mercator estimate.
        import rasterio
        with rasterio.open(AUKERMAN / "orthomosaic.tif") as src:
            assert float(info.split(" · ")[1].split()[0]) == pytest.approx(src.res[0], abs=0.0005)

    def test_no_esri_placeholder_reaches_the_screen_at_z21(self, stage) -> None:
        """Past the ortho's west edge, where the basemap shows: imagery or the enlarged
        deepest level, never Esri's grey 'Map data not yet available'."""
        if not _esri_reachable():
            pytest.skip("Esri World Imagery is not reachable (offline).")
        browser = stage["browser"]
        west, south, east, north = stage["ortho_tiles"]["bounds"]
        browser.js("""(async () => { window.__u3.mapview.setBasemap('satellite', { keepLayers: true });
          await window.__u3.settle(); })()""", timeout_s=90)
        try:
            for _ in range(3):  # let basemap-depth.js ask the tilemap, then redraw
                _jump(browser, west - 0.0006, (south + north) / 2, 21.0)
            shot = _map_shot(browser, inset=120)
            grey = (np.abs(shot.astype(int) - 204).max(axis=2) <= 3).mean()
            assert grey < 0.02, grey
            assert browser.js("window.__u3.map.getStyle().layers.some((l) => l.id === 'basemap')")
        finally:
            browser.js("""(async () => { window.__u3.mapview.setBasemap('offline', { keepLayers: true });
              await window.__u3.settle(); })()""", timeout_s=90)


class TestLegendAndThemes:
    def test_the_elevation_legend_is_the_range_the_dsm_is_coloured_with(self, stage) -> None:
        import rasterio

        browser, api, layers = stage["browser"], stage["api"], stage["layers"]
        api.set_layer_visible(layers["DSM"], True)
        browser.js("""(async () => { await window.__u3.furniture.redraw(); await window.__u3.settle(); })()""",
                   timeout_s=90)
        browser.wait_for("document.querySelector('.odk-mapf-legend').dataset.min")
        legend = browser.js("""(() => { const l = document.querySelector('.odk-mapf-legend');
          return { min: Number(l.dataset.min), max: Number(l.dataset.max), hidden: l.hidden,
                   ticks: [...l.querySelectorAll('.odk-mapf-ticks span')].map((s) => s.textContent),
                   head: l.querySelector('.odk-mapf-legend-head').textContent,
                   key: [...l.querySelectorAll('.odk-geom-item')].map((i) => i.textContent) }; })()""")
        tiles = api.raster_tiles(layers["DSM"])
        assert [legend["min"], legend["max"]] == pytest.approx(tiles["value_range"])
        with rasterio.open(AUKERMAN / "dsm.tif") as src:
            band = src.read(1, masked=True)
            assert legend["min"] == pytest.approx(float(band.min()), abs=0.05)
            assert legend["max"] == pytest.approx(float(band.max()), abs=0.05)
        assert legend["ticks"][0] == f"{legend['min']:.1f} m" and legend["ticks"][-1] == f"{legend['max']:.1f} m"
        assert "DSM" in legend["head"] and not legend["hidden"]
        assert "Survey area" in legend["key"] and "Camera positions" in legend["key"], legend["key"]
        # One casing for the whole key: sentence case, as the layer names are.
        assert all(item[:1].isupper() for item in legend["key"]), legend["key"]

    def test_desk_map_view(self, stage) -> None:
        browser = stage["browser"]
        browser.js("""(async () => { const m = window.__u3.map;
          m.fitBounds(m._odkProjectBounds, { padding: 24, duration: 0 }); await window.__u3.settle(); })()""",
                   timeout_s=90)
        path = _screens_dir(stage) / "u3-map-desk.png"
        browser.screenshot(path)
        assert path.stat().st_size > 50_000

    def test_field_mode_draws_an_ink_casing_under_every_line(self, stage) -> None:
        browser = stage["browser"]
        browser.viewport(1180, 820)
        state = browser.js("""(async () => {
          document.documentElement.setAttribute('data-mode', 'field');
          const map = window.__u3.map;
          for (let i = 0; i < 80 && map._odkTheme !== 'field'; i++) await new Promise((r) => setTimeout(r, 25));
          map.resize();
          map.fitBounds(map._odkProjectBounds, { padding: 24, duration: 0 });
          await window.__u3.settle();
          const lines = map.getStyle().layers.filter((l) => l.type === 'line' && !l.id.endsWith('-casing')
            && window.__u3.theme.roleOf(l.id));
          const order = map.getStyle().layers.map((l) => l.id);
          return { theme: map._odkTheme, lines: lines.map((l) => ({
            id: l.id, width: map.getPaintProperty(l.id, 'line-width'),
            casing: map.getLayer(l.id + '-casing') ? {
              width: map.getPaintProperty(l.id + '-casing', 'line-width'),
              color: map.getPaintProperty(l.id + '-casing', 'line-color'),
              below: order.indexOf(l.id + '-casing') === order.indexOf(l.id) - 1 } : null })),
            rendered: map.queryRenderedFeatures({ layers: ['odk-p-aoi-line-casing'] }).length,
            stack: getComputedStyle(document.querySelector('[data-control=zoom-in]')).height };
        })()""", timeout_s=90)
        assert state["theme"] == "field"
        assert state["lines"], state
        for line in state["lines"]:
            assert line["width"] == 4, line
            assert line["casing"] == {"width": 8, "color": "#0A0A0A", "below": True}, line
        assert state["rendered"] > 0, "the casing is in the style but drew nothing"
        assert state["stack"] == "64px"
        # The casing is on screen: ink pixels along the boundary of the survey area.
        shot = _map_shot(browser)
        ink = (shot.max(axis=2) <= 16).mean()
        assert ink > 0.002, ink
        path = _screens_dir(stage) / "u3-map-field.png"
        browser.screenshot(path)

        back = browser.js("""(async () => {
          document.documentElement.setAttribute('data-mode', 'desk');
          const map = window.__u3.map;
          for (let i = 0; i < 80 && map._odkTheme !== 'desk'; i++) await new Promise((r) => setTimeout(r, 25));
          return { width: map.getPaintProperty('odk-p-aoi-line-casing', 'line-width'),
                   dash: map.getPaintProperty('odk-p-aoi-line', 'line-dasharray') ?? null };
        })()""")
        browser.viewport(1440, 900)
        # Desk draws the survey area as a dashed rule, not in the accent.
        assert back == {"width": 3.5, "dash": [4, 3]}, back


class TestRemove:
    def test_remove_takes_the_layer_off_the_api_and_the_popover(self, stage) -> None:
        browser, api, layers = stage["browser"], stage["api"], stage["layers"]
        dtm = layers["DTM"]
        result = browser.js(f"""(async () => {{
          document.querySelector('[data-control=layers]').click();
          for (let i = 0; i < 80 && !document.querySelector('.odk-mapf-pop [data-layer="{dtm}"]'); i++)
            await new Promise((r) => setTimeout(r, 25));
          document.querySelector('.odk-mapf-pop [data-layer="{dtm}"] [data-action="Remove Layer"]').click();
          for (let i = 0; i < 120 && document.querySelector('.odk-mapf-pop [data-layer="{dtm}"]'); i++)
            await new Promise((r) => setTimeout(r, 25));
          return {{ gone: !document.querySelector('.odk-mapf-pop [data-layer="{dtm}"]'),
                    toasts: window.__u3.toasts.map((t) => t.join(' ')) }};
        }})()""", timeout_s=60)
        assert result["gone"], result
        assert dtm not in {l["id"] for l in api.list_layers()["layers"]}
        assert any("Removed DTM from the layer list" in t for t in result["toasts"]), result["toasts"]
        # The file itself is untouched.
        assert (AUKERMAN / "dtm.tif").is_file()


# ------------------------------------------------------- the finish review (U3)

MOUNT = """(async () => {
  const f = await import('./js/odk/map/furniture.js');
  document.querySelectorAll('.u3-extra').forEach((n) => n.remove());
  const wrap = document.createElement('div');
  wrap.className = 'u3-extra';
  wrap.style.cssText = 'position:fixed;left:0;top:0;width:1100px;height:640px;z-index:5;display:flex;';
  __SCREEN_ATTR__
  const box = document.createElement('div');
  box.style.cssText = 'flex:1 1 auto;position:relative;';
  wrap.appendChild(box);
  document.body.appendChild(wrap);
  const m = await f.mountProjectMap(box, __OPTIONS__);
  window.__extra = m;
  for (let i = 0; i < 600 && !(m.map._odkProjectLayers || []).length; i++) await new Promise((r) => setTimeout(r, 50));
  await new Promise((r) => { m.map.once('idle', () => setTimeout(r, 80)); m.map.triggerRepaint(); });
  return true;
})()"""

VISIBILITY = """(() => {
  const m = window.__extra.map;
  const vis = (id) => m.getLayer(id) ? (m.getLayoutProperty(id, 'visibility') || 'visible') : null;
  const cams = (m._odkProjectLayers || []).find((d) => d.layer.name === 'Camera positions');
  const rasters = (m._odkProjectLayers || []).filter((d) => d.layer.kind !== 'vector');
  const legend = window.__extra.furniture.node.querySelector('.odk-mapf-legend');
  return { aoi: vis('odk-p-aoi-line'), aoiCasing: vis('odk-p-aoi-line-casing'),
           mission: vis('odk-p-mission-line'), cameras: cams ? vis(cams.mapLayers[2]) : null,
           raster: rasters.length ? vis(rasters[0].mapLayers[0]) : null,
           hidden: window.__extra.furniture.state().viewHidden.sort(),
           key: [...legend.querySelectorAll('.odk-geom-item')].map((i) => i.textContent) };
})()"""


def _mount_extra(browser, *, screen_attr: str = "", options: str = "{}") -> None:
    browser.js(MOUNT.replace("__SCREEN_ATTR__", screen_attr).replace("__OPTIONS__", options), timeout_s=240)


class TestFinishReview:
    def test_analyse_opens_on_the_raster_with_the_capture_geometry_off(self, stage) -> None:
        browser, api, layers = stage["browser"], stage["api"], stage["layers"]
        assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
        lon, lat = stage["centre"]
        # The screen is read from the frame's [data-screen], as the shell marks it.
        _mount_extra(browser, screen_attr="wrap.setAttribute('data-screen', 'analyse');",
                     options=json.dumps({"centre": [lon, lat], "zoom": 17}))
        seen = browser.js(VISIBILITY)
        assert seen["hidden"] == ["aoi", "cameras", "mission"], seen
        assert (seen["aoi"], seen["aoiCasing"], seen["mission"], seen["cameras"]) == ("none",) * 4, seen
        assert seen["raster"] == "visible", seen
        assert not {"Survey area", "Flight line", "Camera positions"} & set(seen["key"]), seen

        # Off in the view, not in the project; one click in the Layers popover brings each back.
        cams = layers["Camera positions"]
        assert {l["id"]: l for l in api.list_layers()["layers"]}[cams]["visible"] is True
        back = browser.js(f"""(async () => {{
          const f = window.__extra.furniture;
          const pop = f.openLayers();
          for (let i = 0; i < 80 && !pop.querySelector('[data-layer="{cams}"]'); i++) await new Promise((r) => setTimeout(r, 25));
          const row = pop.querySelector('[data-layer="{cams}"] input[type=checkbox]');
          const before = row.checked;
          const overlays = Object.fromEntries([...pop.querySelectorAll('[data-overlay]')].map((r) =>
            [r.dataset.overlay, r.querySelector('input').checked]));
          row.click();
          pop.querySelector('[data-overlay=aoi] input').click();
          pop.querySelector('[data-overlay=mission] input').click();
          await new Promise((r) => setTimeout(r, 200));
          f.closeLayers();
          return {{ before, overlays }};
        }})()""")
        assert back["before"] is False and back["overlays"]["aoi"] is False and back["overlays"]["mission"] is False, back
        seen = browser.js(VISIBILITY)
        assert (seen["aoi"], seen["mission"], seen["cameras"]) == ("visible",) * 3, seen
        assert seen["hidden"] == [], seen
        # Turning a view-hidden layer on asked the Api nothing.
        assert {l["id"]: l for l in api.list_layers()["layers"]}[cams]["visible"] is True

    def test_plan_and_fly_keep_the_capture_geometry_on(self, stage) -> None:
        browser = stage["browser"]
        lon, lat = stage["centre"]
        for screen in ("plan", "fly"):
            _mount_extra(browser, options=json.dumps({"centre": [lon, lat], "zoom": 17, "screen": screen}))
            seen = browser.js(VISIBILITY)
            assert seen["hidden"] == [], (screen, seen)
            assert (seen["aoi"], seen["mission"], seen["cameras"]) == ("visible",) * 3, (screen, seen)
        browser.js("document.querySelectorAll('.u3-extra').forEach((n) => n.remove()); true")

    def test_the_layers_popover_stops_at_the_bottom_of_the_map(self, stage) -> None:
        browser = stage["browser"]
        geo = browser.js("""(async () => {
          // A strip below the map, as the desk frame's footer is.
          const stageNode = document.getElementById('stage');
          stageNode.style.bottom = '160px';
          window.__u3.map.resize();
          await new Promise((r) => setTimeout(r, 100));
          const f = window.__u3.furniture;
          f.closeLayers();
          const pop = f.openLayers();
          for (let i = 0; i < 80 && !pop.querySelector('[data-layer]'); i++) await new Promise((r) => setTimeout(r, 25));
          await new Promise((r) => setTimeout(r, 100));
          const p = pop.getBoundingClientRect();
          const m = window.__u3.map.getContainer().getBoundingClientRect();
          const button = pop.querySelector('.odk-mapf-foot [data-action="Cache Basemap View"]');
          const b = button.getBoundingClientRect();
          const out = { popBottom: p.bottom, mapBottom: m.bottom, scrolls: pop.scrollHeight > pop.clientHeight,
                        button: { scrollW: button.scrollWidth, clientW: button.clientWidth, right: b.right,
                                  popRight: p.left + pop.clientWidth, wrap: getComputedStyle(button).whiteSpace,
                                  shrink: getComputedStyle(button).flexShrink } };
          f.closeLayers();
          stageNode.style.bottom = '';
          window.__u3.map.resize();
          return out;
        })()""")
        assert geo["popBottom"] <= geo["mapBottom"] - 7.5, geo
        assert geo["scrolls"], geo
        # The label fits inside its border, and the button clears the scrollbar.
        button = geo["button"]
        assert button["scrollW"] <= button["clientW"] and button["wrap"] == "nowrap" and button["shrink"] == "0", button
        assert button["right"] <= button["popRight"], button

    def test_the_neatline_rules_the_rasters_full_extent_on_desk_only(self, stage) -> None:
        browser = stage["browser"]
        seen = browser.js("""(() => {
          const m = window.__u3.map;
          const order = m.getStyle().layers.map((l) => l.id);
          const shown = (m._odkProjectLayers || []).filter((d) => d.layer.kind !== 'vector'
            && m.getLayoutProperty(d.mapLayers[0], 'visibility') !== 'none');
          const box = [Infinity, Infinity, -Infinity, -Infinity];
          shown.forEach((d) => { const b = d.tiles.bounds; box[0] = Math.min(box[0], b[0]); box[1] = Math.min(box[1], b[1]);
            box[2] = Math.max(box[2], b[2]); box[3] = Math.max(box[3], b[3]); });
          const data = m.getSource('odk-neatline').serialize().data;
          const lastRaster = order.reduce((at, id, i) => (/^odk-p-raster-/.test(id) ? i : at), -1);
          const firstVector = order.findIndex((id) => /^odk-p-(vector|aoi|mission|nofly|markers)/.test(id));
          return { layer: m.getLayer('odk-neatline').type, color: m.getPaintProperty('odk-neatline', 'line-color'),
                   width: m.getPaintProperty('odk-neatline', 'line-width'), at: order.indexOf('odk-neatline'),
                   lastRaster, firstVector, box, coords: data.features[0].geometry.coordinates };
        })()""")
        assert seen["layer"] == "line" and seen["color"] == "#A3A3A3" and seen["width"] == 1, seen
        assert seen["at"] == seen["lastRaster"] + 1, seen
        assert seen["firstVector"] < 0 or seen["at"] < seen["firstVector"], seen
        w, s, e, n = seen["box"]
        assert np.allclose(seen["coords"], [[w, n], [e, n], [e, s], [w, s], [w, n]]), seen
        field = browser.js("""(async () => {
          document.documentElement.setAttribute('data-mode', 'field');
          await new Promise((r) => setTimeout(r, 300));
          const n = window.__u3.map.getSource('odk-neatline').serialize().data.features.length;
          document.documentElement.setAttribute('data-mode', 'desk');
          await new Promise((r) => setTimeout(r, 300));
          return [n, window.__u3.map.getSource('odk-neatline').serialize().data.features.length];
        })()""")
        assert field == [0, 1], field

    def test_sample_elevation_reads_the_dsms_own_pixel(self, stage) -> None:
        import rasterio
        from core import geo

        api = stage["api"]
        with rasterio.open(AUKERMAN / "dsm.tif") as src:
            band = src.read(1, masked=True)
            rows, cols = np.nonzero(~np.ma.getmaskarray(band))
            for k in (0, len(rows) // 3, len(rows) // 2, len(rows) - 1):
                r, c = int(rows[k]), int(cols[k])
                x, y = src.xy(r, c)
                lon, lat = geo.projected_to_wgs84([x], [y], EPSG)
                answer = api.sample_elevation(float(lon[0]), float(lat[0]))
                assert answer["ok"] and answer["dsm"] == "DSM" and answer["epsg"] == EPSG, answer
                assert answer["z"] == pytest.approx(float(band[r, c]), abs=1e-4), (answer, r, c)
            west, south, east, north = src.bounds
            off_lon, off_lat = geo.projected_to_wgs84([west - 500.0], [north + 500.0], EPSG)
        outside = api.sample_elevation(float(off_lon[0]), float(off_lat[0]))
        assert outside["ok"] and outside["z"] is None and outside["reason"] == "outside the DSM", outside
