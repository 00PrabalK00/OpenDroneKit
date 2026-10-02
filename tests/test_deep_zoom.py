"""Zooming in on the site: "I can't zoom into the map more than 50 m".

Two things stopped at about that depth, and both looked like the map refusing to zoom:

* The basemap. Esri's tiled services answer a level they hold no imagery at with HTTP
  200 and a grey "Map data not yet available" picture. At the Aukerman site that is every
  level past z19 (the service's own tilemap says so). The classic window declared the
  satellite source to z23, so from about z19.5 MapLibre asked for z20+ tiles and painted
  the placeholder over the whole view -- measured before this fix: 98% of the canvas at
  z21 was the placeholder's grey (204,204,204). The offline tile cache stored those
  placeholders too, whenever someone cached deeper than the imagery goes.
* The survey. Project rasters were drawn as one PNG preview no wider than 1024 px, so a
  3 cm/px orthomosaic reached the screen at ~10 cm/px and zooming in only enlarged it.

Now the Esri sources fetch through js/basemap-depth.js, which caps the source at the
deepest level the service's tilemap lists for the view and replaces any placeholder that
still arrives with the enlarged ancestor tile; core/tile_cache never stores or serves a
placeholder; and project rasters are served as tiles cut from the GeoTIFF itself
(core/raster_tiles.py), sharp to the survey's native zoom, with the map allowed to z24.

The UI tests drive index.html (the classic window, the default launch) and
workspace.html (the cockpit) in headless Chromium against the real Api, the same way
the other browser tests here do. The ones marked for the network talk to Esri for real
and skip when it cannot be reached.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from cockpit_harness import Cockpit, make_api

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PLACEHOLDER = FIXTURES / "esri_no_imagery_placeholder.jpg"

# The Aukerman demo survey (gitignored; copied before use). ODK_DEMO_ORTHO names it
# elsewhere, as for tests/test_classic_offline_basemap.py.
AUKERMAN = Path(os.environ.get(
    "ODK_DEMO_ORTHO", REPO / "demo_project" / "aukerman_full" / "reconstruction" / "orthomosaic.tif")).parent
AUKERMAN_SITE = (-81.7521, 41.3041)  # lon, lat
IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"


def _esri_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{IMAGERY}/tilemap/10/0/0/1/1", timeout=10) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - offline is a reason to skip, not to fail
        return False


needs_esri = pytest.mark.skipif(not _esri_reachable(),
                                reason="Esri World Imagery is not reachable from this machine (offline).")


# --------------------------------------------------------------- tile cache (disk)


@pytest.fixture
def tile_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    from core.tile_cache import tile_root

    return tile_root()


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_the_placeholder_fixture_is_the_one_the_code_recognises() -> None:
    from core.tile_cache import PLACEHOLDER_SHA256, is_placeholder

    body = PLACEHOLDER.read_bytes()
    assert hashlib.sha256(body).hexdigest() in PLACEHOLDER_SHA256
    assert is_placeholder(body)
    # A real imagery tile is not mistaken for it, nor is anything else of its size.
    assert not is_placeholder(body[:-1] + b"\x00")
    js = (REPO / "app" / "web" / "js" / "basemap-depth.js").read_text(encoding="utf-8")
    assert hashlib.sha256(body).hexdigest() in js and str(len(body)) in js


def test_caching_never_stores_the_placeholder_and_asks_esri_for_a_404(tile_cache) -> None:
    from core.tile_cache import Area, fetch_area, read_tile

    placeholder = PLACEHOLDER.read_bytes()
    real = b"\xff\xd8\xff" + b"imagery" * 400
    asked: list[str] = []

    def opener(request, timeout=20):
        url = request.full_url
        asked.append(url)
        zoom = int(url.split("/tile/")[1].split("/")[0])
        if zoom == 18:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _Response(placeholder if zoom >= 19 else real)

    lon, lat = AUKERMAN_SITE
    area = Area("satellite", lon - 0.0002, lat - 0.0002, lon + 0.0002, lat + 0.0002, 17, 19)
    counts = fetch_area(area, opener=opener)
    assert all("blankTile=false" in url for url in asked), asked
    assert counts["fetched"] >= 1 and counts["failed"] == 0
    assert counts["no_imagery"] == len([u for u in asked if "/tile/17/" not in u])
    stored = sorted(p.relative_to(tile_cache).as_posix() for p in tile_cache.rglob("*.png"))
    assert stored and all(p.startswith("satellite/17/") for p in stored), stored
    for path in stored:
        z, x, y = path.split("/")[1:]
        assert read_tile("satellite", int(z), int(x), int(y[:-4])) == real


def test_a_cached_placeholder_is_dropped_and_served_as_not_cached(tile_cache) -> None:
    from core.tile_cache import summary, tile_response

    placeholder = PLACEHOLDER.read_bytes()
    for z in (19, 20):
        folder = tile_cache / "satellite" / str(z) / "1"
        folder.mkdir(parents=True)
        (folder / "1.png").write_bytes(b"\xff\xd8\xff" + b"real" * 300 if z == 19 else placeholder)
    # A second placeholder, only ever found by the summary's sweep.
    (tile_cache / "satellite" / "20" / "1" / "2.png").write_bytes(placeholder)

    code, _, _, headers = tile_response("/tiles/20/1/1.png", "provider=satellite")
    assert code == 404 and headers["X-ODK-Tile"] == "not-cached"
    assert not (tile_cache / "satellite" / "20" / "1" / "1.png").exists()

    status = summary()
    assert status["placeholders_removed"] == 1
    # The offline map is capped at the deepest cached level: placeholders do not count.
    assert status["providers"]["satellite"]["zoom_levels"] == [19]
    assert tile_response("/tiles/19/1/1.png", "provider=satellite")[0] == 200


@needs_esri
def test_esri_at_aukerman_really_stops_at_z19_and_caching_z17_to_z21_stores_only_imagery(tile_cache) -> None:
    from core.tile_cache import Area, deg_to_tile, fetch_area, is_placeholder, tile_response

    lon, lat = AUKERMAN_SITE
    # What the service says exists here, level by level.
    depth = None
    for z in range(15, 24):
        x, y = deg_to_tile(lon, lat, z)
        with urllib.request.urlopen(f"{IMAGERY}/tilemap/{z}/{y}/{x}/1/1", timeout=20) as response:
            if json.loads(response.read())["data"][0]:
                depth = z
    assert depth == 19, f"Esri's imagery depth at Aukerman changed: z{depth}"

    area = Area("satellite", lon - 0.00005, lat - 0.00005, lon + 0.00005, lat + 0.00005, 17, 21)
    counts = fetch_area(area)
    assert counts["failed"] == 0, counts
    assert counts["no_imagery"] >= 2, counts  # z20 and z21 at least
    stored = list(tile_cache.rglob("*.png"))
    levels = sorted({int(p.parent.parent.name) for p in stored})
    assert levels == [17, 18, 19], levels
    assert not any(is_placeholder(p.read_bytes()) for p in stored)
    for z in (20, 21):
        x, y = deg_to_tile(lon, lat, z)
        assert tile_response(f"/tiles/{z}/{x}/{y}.png", "provider=satellite")[0] == 404


# ---------------------------------------------------------- project raster tiles


def _mercator_tile(lon: float, lat: float, z: int) -> tuple[int, int, float, float]:
    """Tile x, y and the pixel position of (lon, lat) inside that tile."""
    n = 2 ** z
    fx = (lon + 180.0) / 360.0 * n
    fy = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return int(fx), int(fy), (fx - int(fx)) * 256, (fy - int(fy)) * 256


def _decode(body: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def _checker_survey(path: Path, gsd_m: float = 0.033, size_px: int = 3000, square_px: int = 4) -> Path:
    """A test raster: a red/blue checkerboard of 4-pixel squares at 3.3 cm/px, north of
    the Aukerman site. A fixture for resolution, not a survey -- nothing here is shown as
    measured ground."""
    import rasterio
    from rasterio.transform import from_origin

    from core import geo

    east, north = geo.wgs84_to_projected([AUKERMAN_SITE[0]], [AUKERMAN_SITE[1] + 0.004], 32617)
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    red = ((yy // square_px + xx // square_px) % 2 == 0)
    data = np.zeros((3, size_px, size_px), dtype=np.uint8)
    data[0][red] = 255
    data[2][~red] = 255
    with rasterio.open(path, "w", driver="GTiff", width=size_px, height=size_px, count=3, dtype="uint8",
                       crs="EPSG:32617", transform=from_origin(float(east[0]), float(north[0]), gsd_m, gsd_m),
                       tiled=True, compress="deflate") as dst:
        dst.write(data)
    return path


def _pure_fraction(rgba: np.ndarray) -> float:
    """Share of opaque pixels that are the checkerboard's own red or blue, not a blend."""
    opaque = rgba[..., 3] > 0
    r, g, b = (rgba[..., i].astype(int) for i in range(3))
    pure = ((r > 225) & (g < 30) & (b < 30)) | ((b > 225) & (g < 30) & (r < 30))
    return float((pure & opaque).sum()) / max(1, int(opaque.sum()))


@pytest.fixture
def ortho(tmp_path) -> Path:
    if not (AUKERMAN / "orthomosaic.tif").is_file():
        pytest.skip("The Aukerman demo survey is not in this checkout.")
    target = tmp_path / "orthomosaic.tif"
    shutil.copy(AUKERMAN / "orthomosaic.tif", target)
    return target


def test_raster_tiles_carry_the_orthomosaics_own_pixels_at_deep_zoom(ortho) -> None:
    import rasterio

    from core import raster_tiles

    info = raster_tiles.register(ortho)
    # The demo survey is 1.19 m/px, so its native zoom is 17 at this latitude; past it
    # the map enlarges the z17 tile.
    assert info["maxzoom"] == 17 and info["minzoom"] == 0
    assert info["native_resolution_m"] == pytest.approx(1.19, abs=0.03)

    lon, lat = AUKERMAN_SITE
    x, y, px, py = _mercator_tile(lon, lat, 17)
    code, kind, body, _ = raster_tiles.raster_tile_response(f"/raster-tiles/{info['token']}/17/{x}/{y}.png")
    assert (code, kind) == (200, "image/png")
    tile = _decode(body)
    with rasterio.open(ortho) as src:
        from core import geo

        e, n = geo.wgs84_to_projected([lon], [lat], 32617)
        row, col = src.index(float(e[0]), float(n[0]))
        truth = src.read(window=((row, row + 1), (col, col + 1)))[:3, 0, 0].astype(int)
    seen = tile[int(py), int(px), :3].astype(int)
    assert tile[int(py), int(px), 3] == 255
    assert np.abs(seen - truth).max() <= 40, (seen, truth)

    # Outside the survey: an empty tile, not a stretched edge.
    far = raster_tiles.raster_tile_response(f"/raster-tiles/{info['token']}/17/0/0.png")
    assert far[0] == 200 and _decode(far[2])[..., 3].max() == 0
    # Only registered rasters, and only by token.
    assert raster_tiles.raster_tile_response("/raster-tiles/notatoken000/17/1/1.png")[0] == 404
    assert raster_tiles.raster_tile_response("/raster-tiles/../../etc/1/1/1.png") is None


def test_a_centimetre_survey_is_served_at_its_own_resolution_not_the_preview(tmp_path) -> None:
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from core import raster_tiles

    api = Api(AppSession(ProjectStore(tmp_path / "p.db")))
    assert api.create_project("hr", str(tmp_path / "project"))["ok"]
    layer = api.add_layer_from_file(str(_checker_survey(tmp_path / "hires.tif")))["layer"]

    tiles = api.raster_tiles(layer["id"])
    assert tiles["ok"] and tiles["maxzoom"] == 22, tiles
    lon, lat = (tiles["bounds"][0] + tiles["bounds"][2]) / 2, (tiles["bounds"][1] + tiles["bounds"][3]) / 2
    x, y, _, _ = _mercator_tile(lon, lat, 22)
    code, _, body, _ = raster_tiles.raster_tile_response(tiles["url"].format(z=22, x=x, y=y))
    assert code == 200
    sharp = _pure_fraction(_decode(body))

    # The old route: one preview of at most 1024 px for the whole survey.
    import base64

    preview = api.raster_preview(layer["id"])
    blurred = _pure_fraction(_decode(base64.b64decode(preview["data_uri"].split(",", 1)[1])))
    assert sharp >= 0.5, sharp
    assert blurred <= 0.2, blurred


# ------------------------------------------------------------------ the two UIs

# Keeps a handle on every map the page makes, with a readable canvas, and counts the
# basemap tiles MapLibre actually loaded by level.
RECORD_MAPS = r"""
(function(){ window.__odkMaps = []; let lib;
Object.defineProperty(window, 'maplibregl', { configurable: true, get(){ return lib; }, set(v){
  const Base = v.Map;
  class Recorded extends Base { constructor(o){
    o = Object.assign({}, o, {canvasContextAttributes: Object.assign({}, (o || {}).canvasContextAttributes || {}, {preserveDrawingBuffer: true})});
    super(o); window.__odkMaps.push(this);
    const m = this; m.__loadedZ = {};
    m.on('sourcedata', (e) => { if (e.sourceId === 'basemap' && e.tile && e.tile.state === 'loaded') {
      const z = e.tile.tileID.canonical.z; m.__loadedZ[z] = (m.__loadedZ[z] || 0) + 1; } });
  } }
  v.Map = Recorded; lib = v; } });
})();
"""

START = {
    "index.html": 'await ODK.waitFor(() => window.odk && window.odk.ready, 40000);',
    "workspace.html": 'await ODK.nav("Map").catch(() => null);',
}

MAP_HELPERS = r"""
  const map = await ODK.waitFor(() => (window.__odkMaps || []).find((m) =>
    m.getContainer().isConnected && m.getContainer().clientWidth > 0 && m.isStyleLoaded()), 40000);
  if (!map) return { noMap: true };
  const scale = () => ((map.getContainer().querySelector('.maplibregl-ctrl-scale') || {}).textContent || '')
    .replace(/\u00a0/g, ' ');
  const settle = async (rounds) => {
    for (let i = 0; i < rounds; i++) { await fetch('/__tick?ms=300'); await ODK.sleep(100);
      if (i > 8 && map.areTilesLoaded()) break; }
    await ODK.sleep(500);
  };
  const pixels = () => {
    map.triggerRepaint();
    const c = map.getCanvas(); const c2 = document.createElement('canvas');
    c2.width = c.width; c2.height = c.height;
    const ctx = c2.getContext('2d'); ctx.drawImage(c, 0, 0);
    const out = [];
    for (let i = 1; i < 24; i++) for (let j = 1; j < 16; j++) {
      const d = ctx.getImageData(Math.floor(c2.width * i / 24), Math.floor(c2.height * j / 16), 1, 1).data;
      out.push([d[0], d[1], d[2]]);
    }
    return out;
  };
  const rasterLayers = () => map.getStyle().layers.filter((l) => l.type === 'raster' && l.source !== 'basemap')
    .map((l) => { const s = map.getStyle().sources[l.source];
      return { id: l.id, maxzoom: l.maxzoom === undefined ? null : l.maxzoom,
               visibility: map.getLayoutProperty(l.id, 'visibility') || 'visible',
               type: s.type, tiles: (s.tiles || [])[0] || '', sourceMaxzoom: s.maxzoom }; });
"""


def _project_with(api, tmp_path: Path, files: dict[str, Path]) -> None:
    root = Path(api._session.project_root()) / "reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        shutil.copy(source, root / name)
    assert api._session.restore_layers()


def _is(colour, target, tolerance) -> bool:
    return all(abs(int(a) - b) <= tolerance for a, b in zip(colour, target))


@pytest.mark.parametrize("page", ["index.html", "workspace.html"])
def test_the_map_zooms_past_z21_and_the_orthomosaic_stays_drawn(tmp_path, page) -> None:
    if not (AUKERMAN / "orthomosaic.tif").is_file():
        pytest.skip("The Aukerman demo survey is not in this checkout.")
    api = make_api(tmp_path)
    _project_with(api, tmp_path, {"orthomosaic.tif": AUKERMAN / "orthomosaic.tif"})
    lon, lat = AUKERMAN_SITE
    scenario = START[page] + MAP_HELPERS + r"""
      await ODK.waitFor(() => rasterLayers().length, 30000);
      map.jumpTo({ center: [%(lon)f, %(lat)f], zoom: 17 });
      await settle(10);
      // The scroll wheel, as an operator zooms: a burst of wheel-up events on the canvas.
      const canvas = map.getCanvas(); const box = canvas.getBoundingClientRect();
      // Until z21 or 200 notches: on a loaded machine each notch may move less.
      for (let i = 0; i < 200 && map.getZoom() < 21; i++) {
        canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: -240, deltaMode: 0, bubbles: true, cancelable: true,
          clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 }));
        await ODK.sleep(50);
      }
      await settle(20);
      const wheel = { zoom: map.getZoom(), scale: scale() };
      // The + button, all the way in.
      for (let i = 0; i < 12 && map.getZoom() < 22; i++) {
        map.getContainer().querySelector('.maplibregl-ctrl-zoom-in').click(); await ODK.sleep(350); }
      await settle(10);
      const plus = { zoom: map.getZoom(), scale: scale() };
      map.jumpTo({ center: [%(lon)f, %(lat)f], zoom: 21 });
      await settle(40);
      return { maxZoom: map.getMaxZoom(), wheel, plus, z21: { zoom: map.getZoom(), scale: scale() },
               layers: rasterLayers(), pixels: pixels() };
    """ % {"lon": lon, "lat": lat}
    with Cockpit(api, tmp_path / "profile", page=page, window_size="1600,900",
                 head_script=RECORD_MAPS) as cockpit:
        result = cockpit.run(scenario, virtual_time_ms=200000, timeout_s=480)
    assert not result.get("noMap"), result
    assert result["maxZoom"] >= 22
    assert result["wheel"]["zoom"] >= 21, result["wheel"]
    assert result["plus"]["zoom"] >= 22, result["plus"]
    # "50 m" was where it stopped; street-furniture scale is a few metres or less.
    for reading in (result["wheel"]["scale"], result["plus"]["scale"], result["z21"]["scale"]):
        value, unit = reading.split()
        assert unit in {"m", "cm"} and (unit == "cm" or float(value) <= 5), reading

    ortho = [layer for layer in result["layers"] if "/raster-tiles/" in layer["tiles"]]
    assert ortho, result["layers"]
    for layer in ortho:
        # Tiles from the GeoTIFF, drawn at every zoom: no layer maxzoom to hide it.
        assert layer["type"] == "raster" and layer["maxzoom"] is None and layer["visibility"] == "visible"
    served = [code for path, code in cockpit.raster_tiles if "/17/" in path]
    assert served and set(served) == {200}, cockpit.raster_tiles
    # At z21 over the survey the view is the orthomosaic: not the empty background.
    background = sum(_is(p, (13, 17, 23), 8) for p in result["pixels"])
    assert background / len(result["pixels"]) < 0.05, result["pixels"][:10]


@pytest.mark.parametrize("page", ["index.html", "workspace.html"])
def test_a_centimetre_survey_is_sharp_at_z22_in_the_ui(tmp_path, page) -> None:
    api = make_api(tmp_path)
    survey = _checker_survey(tmp_path / "orthomosaic.tif")
    _project_with(api, tmp_path, {"orthomosaic.tif": survey})
    layer = next(l for l in api.list_layers()["layers"] if l["kind"] != "vector")
    west, south, east, north = api.raster_tiles(layer["id"])["bounds"]
    scenario = START[page] + MAP_HELPERS + r"""
      await ODK.waitFor(() => rasterLayers().length, 30000);
      map.jumpTo({ center: [%(lon)f, %(lat)f], zoom: 22 });
      await settle(40);
      return { zoom: map.getZoom(), layers: rasterLayers(), pixels: pixels() };
    """ % {"lon": (west + east) / 2, "lat": (south + north) / 2}
    with Cockpit(api, tmp_path / "profile", page=page, window_size="1600,900",
                 head_script=RECORD_MAPS) as cockpit:
        result = cockpit.run(scenario, virtual_time_ms=200000, timeout_s=480)
    assert result["zoom"] == pytest.approx(22)
    assert any(l["sourceMaxzoom"] == 22 for l in result["layers"]), result["layers"]
    pure = sum(_is(p, (255, 0, 0), 40) or _is(p, (0, 0, 255), 40) for p in result["pixels"])
    # The 1024 px preview blends these 13 cm squares into purple; tiles keep them apart.
    assert pure / len(result["pixels"]) >= 0.5, result["pixels"][:20]


@needs_esri
@pytest.mark.parametrize("track", [True, False], ids=["tilemap-cap", "placeholder-replacement-only"])
@pytest.mark.parametrize("page", ["index.html", "workspace.html"])
def test_at_z21_next_to_aukerman_the_basemap_is_imagery_not_the_placeholder(tmp_path, page, track) -> None:
    api = make_api(tmp_path)
    head = RECORD_MAPS
    if not track:
        # The belt without the braces: no tilemap cap, so Esri is asked for z22 and sends
        # its placeholder; the esri:// protocol must still keep it off the map.
        head += ("(function(){let d;Object.defineProperty(window,'ODKBasemapDepth',{configurable:true,"
                 "get(){return d;},set(v){v.track=()=>null;d=v;}});})();")
    lon, lat = AUKERMAN_SITE[0], AUKERMAN_SITE[1] + 0.0034  # 400 m north: off the survey
    scenario = START[page] + MAP_HELPERS + r"""
      await ODK.sleep(1500);
      map.jumpTo({ center: [%(lon)f, %(lat)f], zoom: 21 });
      await settle(80);
      const src = map.getSource('basemap');
      return { zoom: map.getZoom(), sourceMaxzoom: src && src.maxzoom, depth: map._odkBasemapDepth ?? null,
               tiles: ((src && src.tiles) || [''])[0], loadedZ: map.__loadedZ,
               stats: window.ODKBasemapDepth && window.ODKBasemapDepth.stats, pixels: pixels() };
    """ % {"lon": lon, "lat": lat}
    with Cockpit(api, tmp_path / "profile", page=page, window_size="1600,900",
                 head_script=head, network=True) as cockpit:
        result = cockpit.run(scenario, virtual_time_ms=200000, timeout_s=480)
    assert result["tiles"].startswith("esri://"), result["tiles"]
    grey = sum(_is(p, (204, 204, 204), 3) for p in result["pixels"]) / len(result["pixels"])
    background = sum(_is(p, (13, 17, 23), 8) for p in result["pixels"]) / len(result["pixels"])
    assert grey < 0.02, (grey, result["stats"], result["loadedZ"])
    assert background < 0.05, (background, result["loadedZ"])
    if track:
        # Capped at what the tilemap lists here, and nothing deeper was loaded.
        assert result["depth"] == 19 and result["sourceMaxzoom"] == 19, result
        assert max(int(z) for z in result["loadedZ"]) <= 19, result["loadedZ"]
    else:
        # Esri did send placeholders for z22, and every one was replaced.
        assert result["stats"]["placeholders"] >= 1, result["stats"]
        assert result["stats"]["enlarged"] == result["stats"]["placeholders"], result["stats"]
