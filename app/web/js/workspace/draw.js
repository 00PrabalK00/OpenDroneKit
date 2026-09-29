/*
 * Drawing a line or a polygon on the canvas map, for the measurement tools.
 *
 * The Distance and Area buttons used to set `activeTool` and nothing ever read it, and
 * Volume, Slope and Stockpile read `selectionGeometry`, which nothing ever assigned --
 * so every measurement in the cockpit ran on null. This is the missing piece between a
 * click on the map and an Api call: it collects lon/lat vertices, shows them, and hands
 * the finished geometry to whoever armed it.
 *
 * It computes nothing. Lengths, areas and volumes come from the Api, measured on the
 * project's own georeferenced rasters; a number worked out here in the browser would
 * be a planar guess in degrees presented beside survey-grade figures.
 *
 * Click adds a vertex. Double-click or Enter finishes. Escape or Backspace cancels /
 * removes the last vertex.
 */

const SOURCE = "odk-draw";

const state = {
  mode: null,          // "line" | "polygon" | null
  points: [],
  onFinish: null,
  onCancel: null,
  finished: null,      // the last finished geometry, kept on screen
  maps: new Set(),
  listeners: new Set(),
};

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
  return state.maps.size > 0;
}

/**
 * Start collecting vertices. `onFinish(geometry)` receives a GeoJSON geometry;
 * `onCancel()` runs if the operator presses Esc instead. A "point" finishes on the first
 * click.
 */
export function armDrawing(mode, onFinish, onCancel) {
  if (state.onCancel) {
    const previous = state.onCancel;
    state.onCancel = null;
    previous();
  }
  state.mode = mode === "polygon" ? "polygon" : mode === "point" ? "point" : "line";
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
  if (last && last[0] === lng && last[1] === lat) return false;
  state.points.push([lng, lat]);
  render();
  notify();
  if (state.mode === "point") finishDrawing();
  return true;
}

export function removeLastVertex() {
  if (!state.mode || !state.points.length) return false;
  state.points.pop();
  render();
  notify();
  return true;
}

/** Complete the shape. Returns the geometry, or a string saying why it cannot. */
export function finishDrawing() {
  if (!state.mode) return "Nothing is being drawn.";
  const needed = state.mode === "polygon" ? 3 : state.mode === "point" ? 1 : 2;
  if (state.points.length < needed) {
    return `A ${state.mode} needs at least ${needed} point(s); ${state.points.length} placed.`;
  }
  const coordinates = state.points.map((p) => [...p]);
  const geometry = state.mode === "polygon"
    ? { type: "Polygon", coordinates: [[...coordinates, coordinates[0]]] }
    : state.mode === "point"
      ? { type: "Point", coordinates: coordinates[0] }
      : { type: "LineString", coordinates };
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

function features() {
  const out = [];
  const shape = state.mode ? null : state.finished;
  if (shape) out.push({ type: "Feature", geometry: shape, properties: { role: "selection" } });
  if (state.mode && state.points.length) {
    const ring = state.mode === "polygon" && state.points.length > 2
      ? [...state.points, state.points[0]] : state.points;
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

function render() {
  state.maps.forEach((map) => {
    // The dock rebuilds canvases on every render; a map whose container has left the
    // document is gone for good.
    const container = map.getContainer ? map.getContainer() : null;
    if (container && container.isConnected === false) {
      state.maps.delete(map);
      return;
    }
    if (!map.isStyleLoaded || !map.isStyleLoaded()) return;
    const data = features();
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

/** Called by mapview for every map it mounts. */
export function attachDrawing(map) {
  state.maps.add(map);
  map.on("click", (event) => { addVertex([event.lngLat.lng, event.lngLat.lat]); });
  map.on("dblclick", (event) => {
    if (!state.mode) return;
    if (event.preventDefault) event.preventDefault();
    finishDrawing();
  });
  map.on("load", render);
  if (map.once) map.once("remove", () => state.maps.delete(map));
  if (state.mode && map.doubleClickZoom) map.doubleClickZoom.disable();
}

let keysBound = false;
export function bindDrawingKeys(target = typeof document !== "undefined" ? document : null) {
  if (keysBound || !target || !target.addEventListener) return;
  keysBound = true;
  target.addEventListener("keydown", (event) => {
    if (!state.mode) return;
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(target.activeElement?.tagName);
    if (typing) return;
    if (event.key === "Enter") { event.preventDefault(); finishDrawing(); }
    else if (event.key === "Escape") { event.preventDefault(); cancelDrawing(); }
    else if (event.key === "Backspace") { event.preventDefault(); removeLastVertex(); }
  });
}
