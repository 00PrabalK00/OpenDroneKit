/*
 * Drawing on the cockpit map, and what each drawn shape is for.
 *
 * Eight canvas tools and the Distance, Area and Measure buttons all armed themselves and
 * then nothing read a single click: the shell stored `activeTool` and no code looked at
 * it, and nothing ever assigned the selection geometry that Volume, Cut & Fill and Slope
 * ask for. So Plan always answered "Draw an area of interest on the map before planning"
 * on a screen whose polygon tool could not draw one.
 *
 * Every tool here ends in the application, not in the browser. An area of interest goes
 * to set_aoi, an exclusion zone to set_no_fly_zones, a home point and a note to the site
 * markers, and a measurement to measure_geometry -- so a number on screen is the number
 * Python computed from the clicked positions, and a refusal from Python is shown as one.
 */

import { call } from "./api.js";

/* The tool names the canvas buttons and toolbar actions use, and what each one draws.
 *   points  how many clicks finish it on their own (0 = until double-click / Enter)
 *   shape   what is drawn while clicking */
export const TOOLS = {
  pan: { shape: null, hint: "" },
  polygon: { shape: "polygon", points: 0, hint: "Click the corners of the area of interest. Double-click or Enter to finish." },
  exclusion: { shape: "polygon", points: 0, hint: "Click the corners of the no-fly zone. Double-click or Enter to finish." },
  line: { shape: "line", points: 0, hint: "Click along the line. Double-click or Enter to finish." },
  measure: { shape: "line", points: 0, hint: "Click along the path to measure. Double-click or Enter to finish." },
  distance: { shape: "line", points: 0, hint: "Click along the path to measure. Double-click or Enter to finish." },
  area: { shape: "polygon", points: 0, hint: "Click the corners of the area to measure. Double-click or Enter to finish." },
  box: { shape: "box", points: 2, hint: "Click two opposite corners of the selection." },
  home: { shape: "point", points: 1, hint: "Click where the aircraft takes off and lands." },
  annotate: { shape: "point", points: 1, hint: "Click the spot to note." },
};

/* The canvas buttons are labelled for a person; these are the tools they arm. */
export const TOOL_BY_TITLE = {
  "Pan": "pan",
  "Box select": "box",
  "Draw polygon": "polygon",
  "Draw line": "line",
  "Measure": "measure",
  "Place home": "home",
  "Exclusion zone": "exclusion",
};

const state = {
  map: null,
  tool: "pan",
  points: [],
  hooks: {
    toast: () => {},
    prompt: async () => null,
    geometry: () => {},
    changed: () => {},
  },
  // The last shape a tool produced, as GeoJSON, whichever map it was drawn on.
  selection: null,
  history: [],
};

const DRAW_SOURCE = "odk-draw";

/** Wire the drawing to the shell: where to report, how to ask, what to update. */
export function configureDrawing(hooks) {
  Object.assign(state.hooks, hooks);
}

/** The shape last drawn, as a GeoJSON geometry in longitude/latitude, or null. */
export function selectionGeometry() {
  return state.selection;
}

/** Measurements taken this session, newest first. */
export function measurementHistory() {
  return state.history.slice();
}

export function activeTool() {
  return state.tool;
}

export function currentMap() {
  return state.map && state.map._container && state.map._container.isConnected ? state.map : null;
}

/** A map has been mounted on the visible canvas; drawing now happens on it. */
export function attachMap(map) {
  state.map = map;
  state.points = [];
  map.on("click", (event) => onClick(map, event));
  map.on("dblclick", (event) => {
    if (!drawing()) return;
    event.preventDefault();
    finish();
  });
  map.on("mousemove", (event) => {
    if (!drawing() || !state.points.length) return;
    paint([...state.points, [event.lngLat.lng, event.lngLat.lat]]);
  });
  if (map._odkReady) ensureDrawLayers(map);
  else map.once("load", () => ensureDrawLayers(map));
  applyCursor();
}

/** Arm a tool. Returns the sentence to show the operator, or null for an unknown tool. */
export function armTool(tool) {
  const spec = TOOLS[tool];
  if (!spec) return null;
  state.tool = tool;
  state.points = [];
  paint([]);
  applyCursor();
  if (tool === "pan") return "Pan: drag to move the map.";
  if (!currentMap()) {
    return "There is no map on this workspace to draw on. Switch to Mission Planning, "
      + "Measurements or Home.";
  }
  return spec.hint;
}

/** Remove the last clicked point of the shape in progress. */
export function undoPoint() {
  if (!drawing() || !state.points.length) return false;
  state.points.pop();
  paint(state.points);
  return true;
}

/** Abandon the shape in progress and go back to panning. */
export function stopDrawing() {
  const was = drawing();
  state.tool = "pan";
  state.points = [];
  paint([]);
  applyCursor();
  return was;
}

/** Finish the shape in progress (double-click or Enter). */
export async function finish() {
  const spec = TOOLS[state.tool];
  if (!spec || !spec.shape) return;
  const points = dedupe(state.points);
  const tool = state.tool;
  state.points = [];
  paint([]);
  try {
    await complete(tool, points);
  } catch (error) {
    state.hooks.toast(`${label(tool)}: ${error.message || error}`, "error");
  }
  // Measuring is repeated, so those tools stay armed; a drawn area or a placed point is
  // a single act, and leaving the tool armed turns the next pan into another corner.
  if (!["measure", "distance", "area"].includes(tool) && state.tool === tool) stopDrawing();
}

function drawing() {
  return Boolean(TOOLS[state.tool] && TOOLS[state.tool].shape);
}

function label(tool) {
  return {
    polygon: "Area of interest", exclusion: "Exclusion zone", line: "Line",
    measure: "Measure", distance: "Distance", area: "Area", box: "Box select",
    home: "Home", annotate: "Annotate",
  }[tool] || tool;
}

function onClick(map, event) {
  if (!drawing()) return;
  const point = [event.lngLat.lng, event.lngLat.lat];
  state.points.push(point);
  paint(state.points);
  const spec = TOOLS[state.tool];
  if (spec.points && state.points.length >= spec.points) finish();
}

function dedupe(points) {
  // A double-click lands two clicks on the same spot before the dblclick event.
  const out = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (!last || Math.abs(last[0] - p[0]) > 1e-9 || Math.abs(last[1] - p[1]) > 1e-9) out.push(p);
  }
  return out;
}

function boxRing([a, b]) {
  return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
}

async function complete(tool, points) {
  const say = state.hooks.toast;
  const need = { polygon: 3, exclusion: 3, area: 3, line: 2, measure: 2, distance: 2, box: 2 }[tool] || 1;
  if (points.length < need) {
    say(`${label(tool)}: needs at least ${need} point${need === 1 ? "" : "s"}; `
      + `${points.length} clicked. Nothing was changed.`, "warn");
    return;
  }

  if (tool === "polygon") {
    const result = await call("set_aoi", points);
    select({ type: "Polygon", coordinates: [[...points, points[0]]] });
    say(`Area of interest set: ${result.vertices} vertices, `
      + `${(result.area_m2 / 10000).toFixed(2)} ha. Plan will use it.`, "ok");
    state.hooks.changed();
    return;
  }

  if (tool === "exclusion") {
    const current = await call("get_state");
    const zones = [...(current.no_fly || []), points];
    const result = await call("set_no_fly_zones", zones);
    say(`Exclusion zone added. ${result.count} no-fly zone(s) now constrain planning.`, "ok");
    state.hooks.changed();
    return;
  }

  if (tool === "line" || tool === "measure" || tool === "distance") {
    const result = await call("measure_geometry", points, "distance");
    select({ type: "LineString", coordinates: points });
    record({ kind: "distance", value: result.length_m, unit: "m", vertices: result.vertices });
    say(tool === "line"
      ? `Line drawn: ${formatLength(result.length_m)}. Profile and Slope act on it.`
      : `Distance ${formatLength(result.length_m)} over ${result.vertices} points `
        + `(${result.method}).`, "ok");
    return;
  }

  if (tool === "area" || tool === "box") {
    const ring = tool === "box" ? boxRing(points) : points;
    const result = await call("measure_geometry", ring, "area");
    select({ type: "Polygon", coordinates: [[...ring, ring[0]]] });
    record({ kind: "area", value: result.area_m2, unit: "m²", perimeter_m: result.perimeter_m });
    say(`${tool === "box" ? "Selected" : "Area"} ${formatArea(result.area_m2)}, `
      + `perimeter ${formatLength(result.perimeter_m)}. Volume, Cut & Fill and Slope act on it.`, "ok");
    return;
  }

  if (tool === "home") {
    const result = await call("add_site_marker", "Home", "takeoff", [points[0]],
                              "Placed on the cockpit map");
    say(`Home placed at ${points[0][1].toFixed(6)}, ${points[0][0].toFixed(6)}. `
      + `${(result.markers || []).length} site marker(s).`, "ok");
    state.hooks.changed();
    return;
  }

  if (tool === "annotate") {
    const note = await state.hooks.prompt("Note for this spot", "");
    if (!note) {
      say("Annotate: no note given, nothing recorded.", "warn");
      return;
    }
    const result = await call("add_site_marker", note, "landmark", [points[0]], note);
    say(`Noted "${note}". ${(result.markers || []).length} site marker(s).`, "ok");
    state.hooks.changed();
  }
}

function select(geometry) {
  state.selection = geometry;
  state.hooks.geometry(geometry);
}

function record(entry) {
  state.history.unshift({ ...entry, at: new Date().toISOString() });
  state.history = state.history.slice(0, 50);
  if (typeof window !== "undefined") window.dispatchEvent(new Event("odk:measured"));
}

export function formatLength(metres) {
  const m = Number(metres);
  return m >= 1000 ? `${(m / 1000).toFixed(3)} km` : `${m.toFixed(2)} m`;
}

export function formatArea(squareMetres) {
  const a = Number(squareMetres);
  return a >= 10000 ? `${(a / 10000).toFixed(3)} ha (${Math.round(a).toLocaleString()} m²)`
    : `${a.toFixed(1)} m²`;
}

/* ---------------------------------------------------------- map rendering */

function ensureDrawLayers(map) {
  if (map.getSource(DRAW_SOURCE)) return;
  map.addSource(DRAW_SOURCE, { type: "geojson", data: emptyCollection() });
  map.addLayer({
    id: `${DRAW_SOURCE}-fill`, type: "fill", source: DRAW_SOURCE,
    filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "fill-color": "#f59e0b", "fill-opacity": 0.15 },
  });
  map.addLayer({
    id: `${DRAW_SOURCE}-line`, type: "line", source: DRAW_SOURCE,
    paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [2, 1] },
  });
  map.addLayer({
    id: `${DRAW_SOURCE}-points`, type: "circle", source: DRAW_SOURCE,
    filter: ["==", ["geometry-type"], "Point"],
    paint: { "circle-radius": 4, "circle-color": "#f59e0b",
             "circle-stroke-color": "#0d1015", "circle-stroke-width": 1 },
  });
  map.doubleClickZoom.disable();
}

function emptyCollection() {
  return { type: "FeatureCollection", features: [] };
}

function paint(points) {
  const map = currentMap();
  if (!map || !map.getSource(DRAW_SOURCE)) return;
  const spec = TOOLS[state.tool] || {};
  const features = points.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
  if (points.length >= 2) {
    let ring = points;
    if (spec.shape === "box") ring = boxRing(points.slice(0, 2));
    const closed = spec.shape === "polygon" || spec.shape === "box";
    features.push({
      type: "Feature", properties: {},
      geometry: closed && ring.length >= 3
        ? { type: "Polygon", coordinates: [[...ring, ring[0]]] }
        : { type: "LineString", coordinates: ring },
    });
  }
  map.getSource(DRAW_SOURCE).setData({ type: "FeatureCollection", features });
}

function applyCursor() {
  const map = currentMap();
  if (!map) return;
  map.getCanvas().style.cursor = drawing() ? "crosshair" : "";
}

if (typeof document !== "undefined") {
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !drawing()) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    event.preventDefault();
    finish();
  });
}
