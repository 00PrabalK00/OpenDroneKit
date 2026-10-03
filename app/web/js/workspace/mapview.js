/* A real MapLibre map inside a workspace canvas, or an honest placeholder.
 *
 * The framework does not depend on this. A canvas is a mounting point, and whether a
 * live map, a 3D viewport or a placeholder occupies it is a per-workspace decision --
 * which is why adding this changes no other file.
 *
 * Two failure modes are handled rather than hidden, because both are ordinary in the
 * field. MapLibre may be absent (a stripped deployment). The network may be absent,
 * which for an offline-first product is not a failure at all -- it is Tuesday. In both
 * cases the canvas says which of the two happened instead of showing an empty grey
 * rectangle that could mean anything.
 */

import { el } from "./primitives.js";
import { attachDrawing } from "./draw.js";
import { tryCall } from "./api.js";

/* Vector layers every canvas map draws: the project's CAD overlays and site markers,
   plus any vector layer the operator picked. The overlays used to be written to disk
   and listed by nobody, and the markers were never drawn -- a hazard recorded on the
   site did not appear on the map the pilot planned over. */
const mounted = new Set();
const pinned = new Map();

const PALETTE = {
  cad: "#22d3ee",
  markers: "#ef4444",
  layer: "#a78bfa",
};

const sourceId = (id) => `odk-v-${String(id).replace(/[^A-Za-z0-9_-]/g, "_")}`;

function drawPinned(map) {
  if (!map.isStyleLoaded || !map.isStyleLoaded()) {
    /* isStyleLoaded() is false while any source is still fetching (a raster layer being
       added, a basemap tile in flight). Returning then dropped the overlays for good;
       wait for the map to go idle and draw them. */
    if (!map.__odkVectorsPending && map.once) {
      map.__odkVectorsPending = true;
      map.once("idle", () => { map.__odkVectorsPending = false; drawPinned(map); });
    }
    return;
  }
  for (const [id, entry] of pinned) {
    const source = sourceId(id);
    const existing = map.getSource(source);
    if (existing) { existing.setData(entry.geojson); continue; }
    map.addSource(source, { type: "geojson", data: entry.geojson });
    const colour = PALETTE[entry.kind] || PALETTE.layer;
    map.addLayer({ id: `${source}-fill`, type: "fill", source,
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: { "fill-color": colour, "fill-opacity": 0.15 } });
    map.addLayer({ id: `${source}-line`, type: "line", source,
      filter: ["!=", ["geometry-type"], "Point"],
      paint: { "line-color": colour, "line-width": 1.5 } });
    map.addLayer({ id: `${source}-point`, type: "circle", source,
      filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 5, "circle-color": colour,
        "circle-stroke-color": "#0d1015", "circle-stroke-width": 1 } });
  }
}

/** Draw a vector layer on every canvas map, now and on any map mounted later. */
export function showVector(id, geojson, kind = "layer") {
  if (!geojson) return;
  pinned.set(id, { geojson, kind });
  mounted.forEach((map) => drawPinned(map));
}

/** What is actually on a mounted map, by id, with its feature count -- read back from
    the map's own sources, so a layer that was pinned but never reached a map is absent. */
export function shownVectors() {
  const shown = {};
  for (const [id, entry] of pinned) {
    if ([...mounted].some((map) => map.getSource && map.getSource(sourceId(id)))) {
      shown[id] = ((entry.geojson || {}).features || []).length;
    }
  }
  return shown;
}

/** The MapLibre layer ids that draw a pinned vector (CAD overlay, site markers...), for
    a control that shows or hides it. Only the ids present on `map` are returned. */
export function vectorLayerIds(map, id) {
  const source = sourceId(id);
  return [`${source}-fill`, `${source}-line`, `${source}-point`]
    .filter((layer) => map && map.getLayer && map.getLayer(layer));
}

async function loadProjectOverlays(map) {
  const overlays = await tryCall("list_cad_overlays");
  for (const overlay of (overlays && overlays.overlays) || []) {
    if (overlay.kind === "vector" && overlay.geojson) {
      pinned.set(`cad:${overlay.name}`, { geojson: overlay.geojson, kind: "cad" });
    }
  }
  const markers = await tryCall("list_site_markers");
  if (markers && markers.geojson) {
    pinned.set("site-markers", { geojson: markers.geojson, kind: "markers" });
  }
  drawPinned(map);
}

/** Re-read overlays and markers into every mounted map (after an import or a marker). */
export function refreshProjectOverlays() {
  mounted.forEach((map) => loadProjectOverlays(map));
}

/* Style built here rather than imported from js/map.js: that module belongs to the hub
 * and carries its own draw tooling and event wiring. Borrowing it would couple two UIs
 * that should be free to change independently. */
const OFFLINE_STYLE = {
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": "#0d1117" } }],
};

/* Basemaps the native View menu can choose between.

   `offline` is the tile cache on disk, served by the desktop shell from its own
   loopback origin (tiles/{z}/{x}/{y}.png). Before, the cockpit had exactly two styles:
   internet satellite imagery, or a dark background -- so without a network the map
   was blank even on a machine whose cache covered the site. */
const TILE_ORIGIN = typeof location !== "undefined" && /^https?:$/.test(location.protocol)
  ? location.origin : "http://127.0.0.1:8000";
/* Esri answers a level it has no imagery at with a grey "Map data not yet available"
   picture (HTTP 200), and how deep it goes changes from place to place -- z19 at the
   Aukerman site. js/basemap-depth.js fetches Esri tiles through an esri:// protocol
   that turns that answer into "no tile here", and caps the source at the deepest level
   the service lists for the view, so zooming in enlarges the last real tile. `maxzoom`
   on those two is the ceiling the service could ever reach, not a promise. */
const DEPTH = typeof window !== "undefined" ? window.ODKBasemapDepth || null : null;
const ESRI_PROTOCOL = Boolean(DEPTH && typeof window !== "undefined" && DEPTH.registerProtocol(window.maplibregl));
const esri = (url) => (ESRI_PROTOCOL ? DEPTH.esriUrl(url) : url);
const BASEMAP_TILES = {
  satellite: { tiles: [esri("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")], maxzoom: 23, attribution: "Esri World Imagery" },
  street: { tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], maxzoom: 19, attribution: "OpenStreetMap contributors" },
  topo: { tiles: [esri("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}")], maxzoom: 19, attribution: "Esri, USGS, NOAA" },
  offline: { tiles: [`${TILE_ORIGIN}/tiles/{z}/{x}/{y}.png?provider=satellite`], maxzoom: 20, attribution: "Locally cached tiles" },
};
let currentBasemap = "satellite";

function styleFor(key) {
  const spec = BASEMAP_TILES[key] || BASEMAP_TILES.satellite;
  return {
    version: 8,
    sources: { basemap: { type: "raster", tiles: spec.tiles, tileSize: 256, maxzoom: spec.maxzoom, attribution: spec.attribution } },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#0d1117" } },
      { id: "basemap", type: "raster", source: "basemap" },
    ],
  };
}

function liveMaps() {
  for (const map of [...mounted]) {
    if (!map.getContainer || !map.getContainer().isConnected) mounted.delete(map);
  }
  return [...mounted];
}

/* Swap only the basemap source of a loaded map, under everything else. setStyle()
   replaces the whole style, which drops the project's rasters and geometry with it. */
function replaceBasemapSource(map, key) {
  const spec = styleFor(key);
  if (map.getLayer("basemap")) map.removeLayer("basemap");
  if (map.getSource("basemap")) map.removeSource("basemap");
  map.addSource("basemap", spec.sources.basemap);
  const above = ((map.getStyle() || {}).layers || []).find((layer) => layer.id !== "bg");
  map.addLayer({ id: "basemap", type: "raster", source: "basemap" }, above ? above.id : undefined);
}

/** Switch every map on screen (and every map mounted later) to this basemap.
    `keepLayers` swaps the basemap under a loaded map's own layers instead of
    replacing its style (the v1 Layers popover). */
export function setBasemap(key, { keepLayers = false } = {}) {
  if (!BASEMAP_TILES[key]) return false;
  currentBasemap = key;
  for (const map of liveMaps()) {
    if (keepLayers && mapReady(map)) {
      replaceBasemapSource(map, key);
      continue;
    }
    map.setStyle(styleFor(key));
    // A new style drops every added source; the overlays and markers go back on top.
    drawPinned(map);
  }
  return true;
}

export function basemapNames() {
  return Object.keys(BASEMAP_TILES);
}

/** The basemap every map is drawing (a key of basemapNames()). */
export function currentBasemapKey() {
  return currentBasemap;
}

/** Each basemap with what a chooser needs to say about it: whether it leaves this
    machine (`network`), its attribution, and the deepest level the source declares. */
export function basemapCatalogue() {
  return Object.entries(BASEMAP_TILES).map(([key, spec]) => ({
    key, attribution: spec.attribution, maxzoom: spec.maxzoom,
    network: !spec.tiles.every((t) => t.startsWith(TILE_ORIGIN)),
    // Fetched through js/basemap-depth.js, so capped at the imagery that exists here.
    depthTracked: spec.tiles.some((t) => t.startsWith("esri://")),
  }));
}

/** The area shown by the map currently on screen, or null when none is. */
export function visibleBounds() {
  const map = liveMaps().pop();
  if (!map) return null;
  const b = map.getBounds();
  return { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth(), zoom: map.getZoom() };
}

/* Whether sources and layers can be added yet.
 *
 * Not isStyleLoaded(): that stays false while any tile is still outstanding, and offline
 * the satellite tiles never arrive -- so every overlay was skipped for the whole session
 * on exactly the machines this product is built for. The style itself is usable from
 * the "load" event on. */
export function mapReady(map) {
  return Boolean(map && map._odkReady);
}

/** Run fn once the map can take sources: now, or on its load event. */
export function whenMapReady(map, fn) {
  if (mapReady(map)) fn();
  else map.once("load", fn);
}

export function mapAvailable() {
  return typeof window !== "undefined" && typeof window.maplibregl !== "undefined";
}

/**
 * Mount a map into a canvas element.
 *
 * Returns the map instance, or null when one could not be created. Null is a legitimate
 * answer -- callers keep their placeholder rather than assuming a map appeared.
 */
export function mountMap(container, {
  centre = [77.4022, 23.2591],
  zoom = 15,
  offline = false,
  onReady,
  // The v1 interface (js/odk/map) draws its own furniture: scale, north, zoom stack and
  // cursor readout. `controls: false` leaves MapLibre's out, and `onNotice` receives
  // the offline / no-tiles notices instead of a chip appended to the container.
  controls = true,
  onNotice = null,
  // Extra MapLibre Map options (e.g. canvasContextAttributes), merged last.
  mapOptions = {},
} = {}) {
  if (!mapAvailable()) {
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip warn", text: "map library not loaded" }),
    ]));
    return null;
  }

  const holder = el("div", { style: "position:absolute;inset:0" });
  container.insertBefore(holder, container.firstChild);

  let map;
  try {
    map = new window.maplibregl.Map({
      container: holder,
      style: offline ? OFFLINE_STYLE : styleFor(currentBasemap),
      center: centre,
      zoom,
      // Deep enough for a centimetre survey (z22 is ~3 cm per screen pixel at mid
      // latitudes). Past a basemap's real depth it is enlarged; project rasters stay
      // sharp to their own native zoom.
      maxZoom: 24,
      attributionControl: false,
      ...mapOptions,
    });
  } catch (error) {
    // A map that cannot construct must not take the workspace down with it: the panels
    // around it are still useful, and a blank canvas beats a blank application.
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip error", text: `map failed: ${error.message}` }),
    ]));
    return null;
  }

  // Clicks become vertices while a measurement tool is armed; otherwise they are ignored.
  attachDrawing(map);

  if (controls) {
    map.addControl(new window.maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-right");
  }

  // Tile errors are reported once, not per tile: a site with no connectivity would
  // otherwise fill the console with hundreds of identical failures and bury anything
  // else. Offline is the expected state for this product, so it is stated calmly.
  //
  // With no network the online basemap is switched to the local tile cache, and the
  // chip says so: whatever was cached before going to site is shown, the rest stays
  // honestly empty.
  mounted.add(map);
  let reportedOffline = false;
  let tileLoaded = false;
  map.on("sourcedata", (event) => {
    if (event && event.sourceId === "basemap" && event.tile && event.tile.state === "loaded") tileLoaded = true;
  });
  // The basemap's real depth for the view on screen (see js/basemap-depth.js).
  if (DEPTH) {
    DEPTH.track(map, () => {
      if (offline || currentBasemap === "offline") return null;
      const spec = BASEMAP_TILES[currentBasemap];
      return spec ? spec.maxzoom : null;
    });
  }
  map.on("error", (event) => {
    // A level with no imagery is coverage, not a lost connection.
    if (DEPTH && DEPTH.isNoImagery(event)) return;
    const message = String(event?.error?.message || "");
    if (!/tile|fetch|network|Failed/i.test(message) || reportedOffline) return;
    reportedOffline = true;
    const fallBack = !offline && currentBasemap !== "offline" && !tileLoaded;
    if (fallBack) {
      map.setStyle(styleFor("offline"));
      drawPinned(map);
    }
    if (onNotice) {
      onNotice({ code: fallBack ? "offline-fallback" : "tiles-unavailable",
                 basemap: fallBack ? "offline" : (offline ? null : currentBasemap) });
      return;
    }
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip warn", text: fallBack
        ? "no network — showing the offline tile cache"
        : "basemap tiles unavailable — offline" }),
    ]));
  });

  // The style takes sources from "style.load". "load" also waits for the first basemap
  // tiles, so a tile request that never answers would keep the project off the map.
  map.once("style.load", () => { map._odkReady = true; });
  map.on("load", () => {
    map._odkReady = true;
    const placeholder = container.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    loadProjectOverlays(map);
    if (onReady) onReady(map);
  });

  // A live coordinate readout, because a survey interface that cannot tell you where
  // the cursor is has left something out. (The v1 furniture has its own.)
  if (!controls) return map;
  const readout = el("div", { class: "canvas-overlay br", html: "—" });
  container.appendChild(readout);
  map.on("mousemove", (event) => {
    readout.innerHTML =
      `${event.lngLat.lat.toFixed(5)} N &nbsp; ${event.lngLat.lng.toFixed(5)} E ` +
      `&nbsp; z${map.getZoom().toFixed(1)}`;
  });

  return map;
}

/** Draw a mission's flight lines and capture points onto a mounted map. */
export function showMission(map, { line = [], captures = [] } = {}) {
  if (!map || !mapReady(map)) return false;

  if (line.length > 1) {
    const id = "odk-mission-line";
    if (map.getSource(id)) removeLayersOn(map, id), map.removeSource(id);
    map.addSource(id, {
      type: "geojson",
      data: { type: "Feature", geometry: { type: "LineString", coordinates: line } },
    });
    map.addLayer({
      id, type: "line", source: id,
      paint: { "line-color": "#3b82f6", "line-width": 1.5, "line-opacity": 0.9 },
    });
  }

  if (captures.length) {
    const id = "odk-captures";
    if (map.getSource(id)) removeLayersOn(map, id), map.removeSource(id);
    map.addSource(id, {
      type: "geojson",
      data: {
        type: "FeatureCollection",
        features: captures.map((c, index) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: c },
          properties: { seq: index + 1 },
        })),
      },
    });
    map.addLayer({
      id, type: "circle", source: id,
      paint: {
        "circle-radius": 2.5,
        "circle-color": "#22d3ee",
        "circle-stroke-width": 0.5,
        "circle-stroke-color": "#0d1015",
      },
    });
  }
  return true;
}

/* ------------------------------------------------------------ project content */

/*
 * What the open project has that belongs on a map.
 *
 * Every map canvas drew the bundled example's lawnmower mission and capture dots,
 * whatever project was open and whether or not it had a mission -- a flight plan over
 * Ohio on a project in Pune. The layer panel toggled visibility on layers the map never
 * showed. Everything below is read from the application: the drawn area of interest and
 * no-fly zones, the planned mission, the site markers, and every visible layer, rasters
 * as georeferenced images and vectors as GeoJSON.
 */

const PROJECT_IDS = [];

/* Every layer drawing from `source`, removed so the source itself can be. Other modules
   stack layers on these sources (js/odk/map/theme.js puts a casing under each line),
   and MapLibre refuses to remove a source a layer still uses. */
function removeLayersOn(map, source) {
  for (const layer of [...(((map.getStyle() || {}).layers) || [])]) {
    if (layer.source === source && map.getLayer(layer.id)) map.removeLayer(layer.id);
  }
}

function clearProject(map) {
  for (const id of PROJECT_IDS.splice(0)) {
    if (map.getLayer(id)) map.removeLayer(id);
  }
  for (const layer of [...(((map.getStyle() || {}).layers) || [])]) {
    if (String(layer.source || "").startsWith("odk-p-")) removeLayersOn(map, layer.source);
  }
  const sources = Object.keys((map.getStyle() || {}).sources || {});
  for (const id of sources) if (id.startsWith("odk-p-")) map.removeSource(id);
}

function addLayer(map, spec, before) {
  map.addLayer(spec, before && map.getLayer(before) ? before : undefined);
  PROJECT_IDS.push(spec.id);
}

function extend(bounds, coordinates) {
  const walk = (c) => {
    if (typeof c[0] === "number") {
      bounds[0] = Math.min(bounds[0], c[0]); bounds[1] = Math.min(bounds[1], c[1]);
      bounds[2] = Math.max(bounds[2], c[0]); bounds[3] = Math.max(bounds[3], c[1]);
    } else c.forEach(walk);
  };
  walk(coordinates);
}

/* A project raster served as tiles. No layer maxzoom: it stays drawn at every zoom,
 * and past `maxzoom` (the survey's native zoom) MapLibre enlarges the deepest tile. */
function rasterTileSource(tiles) {
  const url = /^https?:/.test(tiles.url) ? tiles.url : `${location.origin}${tiles.url}`;
  const source = { type: "raster", tiles: [url], tileSize: tiles.tile_size || 256,
                   minzoom: tiles.minzoom || 0, maxzoom: tiles.maxzoom ?? 22 };
  if (Array.isArray(tiles.bounds) && tiles.bounds.length === 4) source.bounds = tiles.bounds;
  return source;
}

function geometriesOf(geojson) {
  if (!geojson) return [];
  if (geojson.type === "FeatureCollection") return (geojson.features || []).map((f) => f.geometry).filter(Boolean);
  if (geojson.type === "Feature") return geojson.geometry ? [geojson.geometry] : [];
  return [geojson];
}

/**
 * Draw the project's own content onto a mounted map and return its bounds
 * [west, south, east, north], or null when the project has nothing to place.
 *
 * `ask` is tryCall: every read may be refused, and a refusal leaves that item off the map
 * rather than stopping the rest.
 *
 * One draw at a time per map: a draw clears the project and then awaits each layer, so
 * two overlapping draws (the map's first draw and a screen's redraw) would both add the
 * same source ids. A second call waits for the first to finish, then draws.
 */
export function showProject(map, ask) {
  if (!map) return Promise.resolve(null);
  const previous = map._odkShowing || Promise.resolve();
  const next = previous.catch(() => null).then(() => drawProject(map, ask));
  map._odkShowing = next;
  return next;
}

async function drawProject(map, ask) {
  if (!map || !mapReady(map)) return null;
  const [state, mission, markers, listed] = await Promise.all([
    ask("get_state"), ask("mission_geojson"), ask("list_site_markers"), ask("list_layers"),
  ]);
  if (!map._container || !map._container.isConnected) return null;
  clearProject(map);
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  let placed = 0;
  // What went on the map for each project layer, read by js/odk/map/furniture.js (the
  // Layers popover, the elevation legend, the resolution chip).
  const drawn = [];

  // Rasters first so every vector sits on top of them.
  const layers = ((listed && listed.layers) || []).filter((l) => l.visible !== false);
  for (const layer of layers.filter((l) => l.kind !== "vector")) {
    const id = `odk-p-raster-${placed}`;
    // Tiles cut from the GeoTIFF itself (Api.raster_tiles), so zooming in on a
    // centimetre survey shows its own pixels; the single 1024 px preview only got
    // bigger. The preview stays the fallback for a page with no server behind it.
    const tiles = /^https?:$/.test(location.protocol) ? await ask("raster_tiles", layer.id) : null;
    if (tiles && typeof tiles.url === "string" && tiles.url.includes("{z}")) {
      map.addSource(id, rasterTileSource(tiles));
      addLayer(map, { id, type: "raster", source: id,
                      paint: { "raster-opacity": layer.opacity ?? 0.9, "raster-fade-duration": 0 } });
      if (Array.isArray(tiles.bounds) && tiles.bounds.length === 4) {
        const [w, s, e, n] = tiles.bounds;
        extend(bounds, [[w, n], [e, n], [e, s], [w, s]]);
      }
      drawn.push({ layer, mapLayers: [id], source: "tiles", tiles });
      placed += 1;
      continue;
    }
    const preview = await ask("raster_preview", layer.id);
    if (!preview || !preview.data_uri || !preview.coordinates) continue;
    map.addSource(id, { type: "image", url: preview.data_uri, coordinates: preview.coordinates });
    addLayer(map, { id, type: "raster", source: id,
                    paint: { "raster-opacity": layer.opacity ?? 0.9 } });
    extend(bounds, preview.coordinates);
    drawn.push({ layer, mapLayers: [id], source: "preview", tiles: null });
    placed += 1;
  }
  for (const layer of layers.filter((l) => l.kind === "vector")) {
    const read = await ask("read_vector_layer", layer.id);
    if (!read || !read.geojson) continue;
    const id = `odk-p-vector-${placed}`;
    map.addSource(id, { type: "geojson", data: read.geojson });
    addLayer(map, { id: `${id}-fill`, type: "fill", source: id,
                    filter: ["==", ["geometry-type"], "Polygon"],
                    paint: { "fill-color": "#a78bfa", "fill-opacity": 0.2 } });
    addLayer(map, { id: `${id}-line`, type: "line", source: id,
                    paint: { "line-color": "#a78bfa", "line-width": 1.5 } });
    addLayer(map, { id: `${id}-pt`, type: "circle", source: id,
                    filter: ["==", ["geometry-type"], "Point"],
                    paint: { "circle-radius": 3, "circle-color": "#a78bfa" } });
    geometriesOf(read.geojson).forEach((g) => extend(bounds, g.coordinates));
    drawn.push({ layer, mapLayers: [`${id}-fill`, `${id}-line`, `${id}-pt`], source: "geojson", tiles: null });
    placed += 1;
  }

  const aoi = (state && state.aoi) || [];
  if (aoi.length >= 3) {
    map.addSource("odk-p-aoi", { type: "geojson",
      data: { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...aoi, aoi[0]]] } } });
    addLayer(map, { id: "odk-p-aoi-fill", type: "fill", source: "odk-p-aoi",
                    paint: { "fill-color": "#22c55e", "fill-opacity": 0.08 } });
    addLayer(map, { id: "odk-p-aoi-line", type: "line", source: "odk-p-aoi",
                    paint: { "line-color": "#22c55e", "line-width": 2 } });
    extend(bounds, aoi);
    placed += 1;
  }

  const noFly = (state && state.no_fly) || [];
  if (noFly.length) {
    map.addSource("odk-p-nofly", { type: "geojson", data: { type: "FeatureCollection",
      features: noFly.map((ring) => ({ type: "Feature", properties: {},
        geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] } })) } });
    addLayer(map, { id: "odk-p-nofly-fill", type: "fill", source: "odk-p-nofly",
                    paint: { "fill-color": "#ef4444", "fill-opacity": 0.25 } });
    addLayer(map, { id: "odk-p-nofly-line", type: "line", source: "odk-p-nofly",
                    paint: { "line-color": "#ef4444", "line-width": 1.5 } });
    noFly.forEach((ring) => extend(bounds, ring));
    placed += 1;
  }

  const plan = mission && mission.geojson;
  if (plan && (plan.features || []).length) {
    map.addSource("odk-p-mission", { type: "geojson", data: plan });
    addLayer(map, { id: "odk-p-mission-line", type: "line", source: "odk-p-mission",
                    filter: ["==", ["geometry-type"], "LineString"],
                    paint: { "line-color": "#3b82f6", "line-width": 1.5, "line-opacity": 0.9 } });
    addLayer(map, { id: "odk-p-mission-pt", type: "circle", source: "odk-p-mission",
                    filter: ["==", ["geometry-type"], "Point"],
                    paint: { "circle-radius": 2.5, "circle-color": "#22d3ee",
                             "circle-stroke-width": 0.5, "circle-stroke-color": "#0d1015" } });
    geometriesOf(plan).forEach((g) => extend(bounds, g.coordinates));
    placed += 1;
  }

  const marks = markers && markers.geojson;
  if (marks && (marks.features || []).length) {
    map.addSource("odk-p-markers", { type: "geojson", data: marks });
    addLayer(map, { id: "odk-p-markers-pt", type: "circle", source: "odk-p-markers",
                    paint: { "circle-radius": 5, "circle-color": "#f43f5e",
                             "circle-stroke-width": 1.5, "circle-stroke-color": "#ffffff" } });
    geometriesOf(marks).forEach((g) => extend(bounds, g.coordinates));
    placed += 1;
  }

  // Drawing in progress stays above everything the project holds.
  for (const id of ["odk-draw-fill", "odk-draw-line", "odk-draw-points", "odk-draw-vertex"]) {
    if (map.getLayer(id)) map.moveLayer(id);
  }
  map._odkPlaced = placed;
  const extent = placed && Number.isFinite(bounds[0]) ? bounds : null;
  map._odkProjectLayers = drawn;
  map._odkProjectBounds = extent;
  // Hidden layers are not drawn, but a Layers control still lists them.
  map._odkProjectListed = (listed && listed.layers) || [];
  const container = map.getContainer && map.getContainer();
  if (container && typeof CustomEvent === "function") {
    container.dispatchEvent(new CustomEvent("odk:project", { bubbles: true, detail: { layers: drawn, bounds: extent } }));
  }
  return extent;
}

/** Fit a map to [west, south, east, north], keeping a point from zooming to the tiles. */
export function fitTo(map, bounds) {
  if (!map || !bounds) return false;
  const [w, s, e, n] = bounds;
  if (w === e && s === n) {
    map.jumpTo({ center: [w, s], zoom: 17 });
    return true;
  }
  map.fitBounds([[w, s], [e, n]], { padding: 40, maxZoom: 19, duration: 0 });
  return true;
}
