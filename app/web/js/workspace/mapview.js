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

/* Style built here rather than imported from js/map.js: that module belongs to the hub
 * and carries its own draw tooling and event wiring. Borrowing it would couple two UIs
 * that should be free to change independently. */
const OFFLINE_STYLE = {
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": "#0d1117" } }],
};

const SATELLITE_STYLE = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Esri World Imagery",
    },
  },
  layers: [{ id: "basemap", type: "raster", source: "basemap" }],
};

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
      style: offline ? OFFLINE_STYLE : SATELLITE_STYLE,
      center: centre,
      zoom,
      attributionControl: false,
    });
  } catch (error) {
    // A map that cannot construct must not take the workspace down with it: the panels
    // around it are still useful, and a blank canvas beats a blank application.
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip error", text: `map failed: ${error.message}` }),
    ]));
    return null;
  }

  map.addControl(new window.maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-right");

  // Tile errors are reported once, not per tile: a site with no connectivity would
  // otherwise fill the console with hundreds of identical failures and bury anything
  // else. Offline is the expected state for this product, so it is stated calmly.
  let reportedOffline = false;
  map.on("error", (event) => {
    const message = String(event?.error?.message || "");
    if (!/tile|fetch|network|Failed/i.test(message) || reportedOffline) return;
    reportedOffline = true;
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip warn", text: "basemap tiles unavailable — offline" }),
    ]));
  });

  map.on("load", () => {
    map._odkReady = true;
    const placeholder = container.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    if (onReady) onReady(map);
  });

  // A live coordinate readout, because a survey interface that cannot tell you where
  // the cursor is has left something out.
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
    if (map.getSource(id)) map.removeLayer(id), map.removeSource(id);
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
    if (map.getSource(id)) map.removeLayer(id), map.removeSource(id);
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

function clearProject(map) {
  for (const id of PROJECT_IDS.splice(0)) {
    if (map.getLayer(id)) map.removeLayer(id);
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
 */
export async function showProject(map, ask) {
  if (!map || !mapReady(map)) return null;
  const [state, mission, markers, listed] = await Promise.all([
    ask("get_state"), ask("mission_geojson"), ask("list_site_markers"), ask("list_layers"),
  ]);
  if (!map._container || !map._container.isConnected) return null;
  clearProject(map);
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  let placed = 0;

  // Rasters first so every vector sits on top of them.
  const layers = ((listed && listed.layers) || []).filter((l) => l.visible !== false);
  for (const layer of layers.filter((l) => l.kind !== "vector")) {
    const preview = await ask("raster_preview", layer.id);
    if (!preview || !preview.data_uri || !preview.coordinates) continue;
    const id = `odk-p-raster-${placed}`;
    map.addSource(id, { type: "image", url: preview.data_uri, coordinates: preview.coordinates });
    addLayer(map, { id, type: "raster", source: id,
                    paint: { "raster-opacity": layer.opacity ?? 0.9 } });
    extend(bounds, preview.coordinates);
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
  for (const id of ["odk-draw-fill", "odk-draw-line", "odk-draw-points"]) {
    if (map.getLayer(id)) map.moveLayer(id);
  }
  map._odkPlaced = placed;
  return placed && Number.isFinite(bounds[0]) ? bounds : null;
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
