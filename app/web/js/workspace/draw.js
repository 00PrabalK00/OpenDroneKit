/*
 * Drawing on the cockpit map, and what each drawn shape is for.
 *
 * Eight canvas tools and the Distance, Area and Measure buttons all armed themselves and
 * then nothing read a single click: the shell stored `activeTool` and no code looked at
 * it, and nothing ever assigned the selection geometry that Volume, Cut & Fill and Slope
 * ask for. So Plan always answered "Draw an area of interest on the map before planning"
 * on a screen whose polygon tool could not draw one, and every measurement ran on null.
 *
 * Two layers, one module:
 *
 *  - the drawing engine (armDrawing / addVertex / finishDrawing / cancelDrawing) collects
 *    lon/lat vertices, shows them, and hands the finished GeoJSON geometry to whoever
 *    armed it. It computes nothing;
 *  - the tools (armTool, TOOLS) say what a finished shape is for. Every tool ends in the
 *    application, not in the browser: an area of interest goes to set_aoi, an exclusion
 *    zone to set_no_fly_zones, a home point and a note to the site markers, and a
 *    measurement to the Api -- so a number on screen is the number Python computed from
 *    the clicked positions, and a refusal from Python is shown as one.
 *
 * Click adds a vertex. Double-click or Enter finishes. Escape (the shell's "stop"
 * shortcut) cancels; Backspace or Ctrl+Z removes the last vertex.
 */

import { call } from "./api.js";

const SOURCE = "odk-draw";

const state = {
  // engine
  mode: null,          // "line" | "polygon" | "point" | "box" | null
  points: [],
  onFinish: null,
  onCancel: null,
  finished: null,      // the last finished geometry, kept on screen
  maps: new Set(),
  listeners: new Set(),
  // tools
  map: null,           // the map most recently mounted: the one tools draw on
  tool: "pan",
  hooks: {
    toast: () => {},
    prompt: async () => null,
    geometry: () => {},
    changed: () => {},
    // What a measuring or selecting tool's shape is for is the shell's to decide (it
    // knows which action armed it). Without a hook the tool measures on its own.
    drawn: null,
  },
  // The last shape a tool produced, as GeoJSON, whichever map it was drawn on.
  selection: null,
  history: [],
};

/* ================================================================ engine */

function notify() {
  state.listeners.forEach((fn) => fn(drawingState()));
}

export function drawingState() {
  return {
    mode: state.mode,
    points: state.points.map((p) => [...p]),
    finished: state.finished,
  };
}

export function onDrawingChange(fn) {
  state.listeners.add(fn);
  return () => state.listeners.delete(fn);
}

/** Whether any map is mounted to draw on. */
export function canDraw() {
  prune();
  return state.maps.size > 0;
}

/**
 * Start collecting vertices. `onFinish(geometry)` receives a GeoJSON geometry;
 * `onCancel()` runs if the operator presses Esc instead. A "point" finishes on the first
 * click and a "box" on the second (two opposite corners).
 */
export function armDrawing(mode, onFinish, onCancel) {
  if (state.onCancel) {
    const previous = state.onCancel;
    state.onCancel = null;
    previous();
  }
  state.mode = ["polygon", "point", "box"].includes(mode) ? mode : "line";
  state.points = [];
  state.onFinish = onFinish || null;
  state.onCancel = onCancel || null;
  state.maps.forEach((map) => {
    // A double-click finishes the shape; letting it also zoom moves the map under the
    // operator at the moment they are checking what they drew.
    if (map.doubleClickZoom) map.doubleClickZoom.disable();
    if (map.getCanvas) map.getCanvas().style.cursor = "crosshair";
  });
  render();
  notify();
}

export function cancelDrawing() {
  const cancelled = state.onCancel;
  state.mode = null;
  state.points = [];
  state.onFinish = null;
  state.onCancel = null;
  restoreMaps();
  render();
  notify();
  if (cancelled) cancelled();
}

export function clearSelection() {
  state.finished = null;
  render();
  notify();
}

export function addVertex(lngLat) {
  if (!state.mode) return false;
  const lng = Number(lngLat[0]);
  const lat = Number(lngLat[1]);
  if (!Number.isFinite(lng) || !Number.isFinite(lat)) return false;
  const last = state.points[state.points.length - 1];
  // The two clicks of a double-click arrive as two vertices on the same spot.
  if (last && Math.abs(last[0] - lng) < 1e-12 && Math.abs(last[1] - lat) < 1e-12) return false;
  state.points.push([lng, lat]);
  render();
  notify();
  if (state.mode === "point" || (state.mode === "box" && state.points.length >= 2)) finish();
  return true;
}

export function removeLastVertex() {
  if (!state.mode || !state.points.length) return false;
  state.points.pop();
  render();
  notify();
  return true;
}

function boxRing([a, b]) {
  return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
}

/** Complete the shape. Returns the geometry, or a string saying why it cannot. */
export function finishDrawing() {
  if (!state.mode) return "Nothing is being drawn.";
  const needed = { polygon: 3, point: 1, box: 2 }[state.mode] || 2;
  if (state.points.length < needed) {
    return `A ${state.mode} needs at least ${needed} point(s); ${state.points.length} placed.`;
  }
  const coordinates = state.points.map((p) => [...p]);
  let geometry;
  if (state.mode === "polygon") {
    geometry = { type: "Polygon", coordinates: [[...coordinates, coordinates[0]]] };
  } else if (state.mode === "box") {
    const ring = boxRing(coordinates.slice(0, 2));
    geometry = { type: "Polygon", coordinates: [[...ring, ring[0]]] };
  } else if (state.mode === "point") {
    geometry = { type: "Point", coordinates: coordinates[0] };
  } else {
    geometry = { type: "LineString", coordinates };
  }
  const done = state.onFinish;
  state.finished = geometry;
  state.mode = null;
  state.points = [];
  state.onFinish = null;
  state.onCancel = null;
  restoreMaps();
  render();
  notify();
  if (done) done(geometry);
  return geometry;
}

/** The vertices of a geometry as a flat list, without the polygon's closing repeat. */
export function vertices(geometry) {
  if (!geometry) return [];
  if (geometry.type === "Point") return [geometry.coordinates];
  if (geometry.type === "Polygon") {
    const ring = geometry.coordinates[0] || [];
    return ring.slice(0, ring.length > 1 ? -1 : ring.length);
  }
  return geometry.coordinates || [];
}

function restoreMaps() {
  state.maps.forEach((map) => {
    if (map.doubleClickZoom) map.doubleClickZoom.enable();
    if (map.getCanvas) map.getCanvas().style.cursor = "";
  });
}

function prune() {
  state.maps.forEach((map) => {
    // The dock rebuilds canvases on every render; a map whose container has left the
    // document is gone for good.
    const container = map.getContainer ? map.getContainer() : null;
    if (container && container.isConnected === false) state.maps.delete(map);
  });
}

function features(hover) {
  const out = [];
  const shape = state.mode ? null : state.finished;
  if (shape) out.push({ type: "Feature", geometry: shape, properties: { role: "selection" } });
  if (state.mode && state.points.length) {
    const points = hover ? [...state.points, hover] : state.points;
    let ring = points;
    if (state.mode === "box" && points.length >= 2) {
      const box = boxRing(points.slice(0, 2));
      ring = [...box, box[0]];
    } else if (state.mode === "polygon" && points.length > 2) {
      ring = [...points, points[0]];
    }
    if (ring.length > 1) {
      out.push({ type: "Feature", geometry: { type: "LineString", coordinates: ring },
        properties: { role: "drawing" } });
    }
    state.points.forEach((p) => out.push({
      type: "Feature", geometry: { type: "Point", coordinates: p }, properties: { role: "vertex" },
    }));
  }
  return { type: "FeatureCollection", features: out };
}

function render(hover) {
  prune();
  state.maps.forEach((map) => {
    if (!map.isStyleLoaded || !map.isStyleLoaded()) return;
    const data = features(hover);
    const source = map.getSource(SOURCE);
    if (source) {
      source.setData(data);
      return;
    }
    map.addSource(SOURCE, { type: "geojson", data });
    map.addLayer({
      id: `${SOURCE}-fill`, type: "fill", source: SOURCE,
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: { "fill-color": "#f59e0b", "fill-opacity": 0.18 },
    });
    map.addLayer({
      id: `${SOURCE}-line`, type: "line", source: SOURCE,
      filter: ["!=", ["geometry-type"], "Point"],
      paint: { "line-color": "#f59e0b", "line-width": 2 },
    });
    map.addLayer({
      id: `${SOURCE}-vertex`, type: "circle", source: SOURCE,
      filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 4, "circle-color": "#f59e0b",
        "circle-stroke-color": "#0d1015", "circle-stroke-width": 1 },
    });
  });
}

/** Called for every map mounted on a canvas. Attaching the same map twice is harmless. */
export function attachDrawing(map) {
  state.map = map;
  if (state.maps.has(map)) return;
  state.maps.add(map);
  map.on("click", (event) => { addVertex([event.lngLat.lng, event.lngLat.lat]); });
  map.on("dblclick", (event) => {
    if (!state.mode) return;
    if (event.preventDefault) event.preventDefault();
    finish();
  });
  map.on("mousemove", (event) => {
    if (!state.mode || !state.points.length || !event.lngLat) return;
    render([event.lngLat.lng, event.lngLat.lat]);
  });
  map.on("load", () => render());
  if (map.once) map.once("remove", () => state.maps.delete(map));
  if (state.mode) {
    if (map.doubleClickZoom) map.doubleClickZoom.disable();
    if (map.getCanvas) map.getCanvas().style.cursor = "crosshair";
  }
}

let keysBound = false;
export function bindDrawingKeys(target = typeof document !== "undefined" ? document : null) {
  if (keysBound || !target || !target.addEventListener) return;
  keysBound = true;
  target.addEventListener("keydown", (event) => {
    if (!state.mode) return;
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(target.activeElement?.tagName);
    if (typing) return;
    if (event.key === "Enter") { event.preventDefault(); finish(); }
    else if (event.key === "Backspace") { event.preventDefault(); removeLastVertex(); }
    // Escape is the shell's "stop" shortcut, which cancels through stopDrawing() and
    // says so; cancelling here as well would leave it nothing to report.
  });
}

/* ================================================================= tools */

/* The tool names the canvas buttons and toolbar actions use, and what each one draws.
 *   shape  the engine mode (null = not a drawing tool) */
export const TOOLS = {
  pan: { shape: null, hint: "" },
  polygon: { shape: "polygon", hint: "Click the corners of the area of interest. Double-click or Enter to finish." },
  exclusion: { shape: "polygon", hint: "Click the corners of the no-fly zone. Double-click or Enter to finish." },
  line: { shape: "line", hint: "Click along the line. Double-click or Enter to finish." },
  measure: { shape: "line", hint: "Click along the path to measure. Double-click or Enter to finish." },
  distance: { shape: "line", hint: "Click along the path to measure. Double-click or Enter to finish." },
  area: { shape: "polygon", hint: "Click the corners of the area to measure. Double-click or Enter to finish." },
  profile: { shape: "line", hint: "Click along the line to profile. Double-click or Enter to finish." },
  "select-polygon": { shape: "polygon", hint: "Click the corners of the area to act on. Double-click or Enter to finish." },
  "select-line": { shape: "line", hint: "Click along the line to act on. Double-click or Enter to finish." },
  box: { shape: "box", hint: "Click two opposite corners of the selection." },
  home: { shape: "point", hint: "Click where the aircraft takes off and lands." },
  annotate: { shape: "point", hint: "Click the spot to note." },
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

// Tools whose shape the shell turns into a measurement or a selection, when it is wired.
const SHELL_TOOLS = new Set(["measure", "distance", "area", "profile", "select-polygon", "select-line"]);
// Measuring is repeated, so these stay armed; a drawn area or a placed point is a single
// act, and leaving the tool armed turns the next pan into another corner.
const REPEATING = new Set(["measure", "distance", "area"]);

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
  const map = state.map;
  const container = map && (map.getContainer ? map.getContainer() : map._container);
  return container && container.isConnected ? map : null;
}

/** A map has been mounted on the visible canvas; drawing now happens on it. */
export function attachMap(map) {
  attachDrawing(map);
}

/** Arm a tool. Returns the sentence to show the operator, or null for an unknown tool. */
export function armTool(tool) {
  const spec = TOOLS[tool];
  if (!spec) return null;
  if (!spec.shape) {
    stopDrawing();
    return "Pan: drag to move the map.";
  }
  if (!currentMap() && !canDraw()) {
    stopDrawing();
    return "There is no map on this workspace to draw on. Switch to Mission Planning, "
      + "Measurements or Home.";
  }
  state.tool = tool;
  armDrawing(spec.shape, (geometry) => { onToolFinished(tool, geometry); }, () => {
    if (state.tool === tool) state.tool = "pan";
  });
  return spec.hint;
}

/** Remove the last clicked point of the shape in progress. */
export function undoPoint() {
  return removeLastVertex();
}

/** Abandon the shape in progress and go back to panning. Returns whether one was. */
export function stopDrawing() {
  const was = Boolean(state.mode);
  state.tool = "pan";
  if (was) cancelDrawing();
  return was;
}

/** Finish the shape in progress (double-click or Enter), or say why it cannot be. */
export function finish() {
  const tool = state.tool;
  const result = finishDrawing();
  if (typeof result === "string" && result !== "Nothing is being drawn.") {
    state.hooks.toast(`${label(tool)}: ${result} Nothing was changed.`, "warn");
  }
  return result;
}

async function onToolFinished(tool, geometry) {
  try {
    await complete(tool, geometry);
  } catch (error) {
    state.hooks.toast(`${label(tool)}: ${error.message || error}`, "error");
  }
  if (REPEATING.has(tool) && state.tool === tool && !state.mode) {
    armDrawing(TOOLS[tool].shape, (next) => { onToolFinished(tool, next); }, () => {
      if (state.tool === tool) state.tool = "pan";
    });
  } else if (state.tool === tool && !state.mode) {
    state.tool = "pan";
  }
}

function label(tool) {
  return {
    polygon: "Area of interest", exclusion: "Exclusion zone", line: "Line",
    measure: "Measure", distance: "Distance", area: "Area", box: "Box select",
    home: "Home", annotate: "Annotate", profile: "Profile",
    "select-polygon": "Select Area", "select-line": "Select Line",
  }[tool] || tool;
}

async function complete(tool, geometry) {
  const say = state.hooks.toast;
  const points = vertices(geometry);

  if (SHELL_TOOLS.has(tool) && state.hooks.drawn) {
    select(geometry);
    await state.hooks.drawn(tool, geometry);
    return;
  }

  if (tool === "polygon") {
    const result = await call("set_aoi", points);
    select(geometry);
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

  if (["line", "measure", "distance", "profile", "select-line"].includes(tool)) {
    const result = await call("measure_geometry", points, "distance");
    select(geometry);
    recordMeasurement({ kind: "distance", value: result.length_m, unit: "m", vertices: result.vertices });
    say(tool === "line" || tool === "select-line"
      ? `Line drawn: ${formatLength(result.length_m)}. Profile and Slope act on it.`
      : `Distance ${formatLength(result.length_m)} over ${result.vertices} points `
        + `(${result.method}).`, "ok");
    return;
  }

  if (["area", "box", "select-polygon"].includes(tool)) {
    const result = await call("measure_geometry", points, "area");
    select(geometry);
    recordMeasurement({ kind: "area", value: result.area_m2, unit: "m²", perimeter_m: result.perimeter_m });
    say(`${tool === "area" ? "Area" : "Selected"} ${formatArea(result.area_m2)}, `
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

/** Keep a measurement in this session's history, newest first. */
export function recordMeasurement(entry) {
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
