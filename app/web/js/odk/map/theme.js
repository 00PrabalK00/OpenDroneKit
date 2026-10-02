/* Map paint for the two modes: desk "Survey Instrument" and Field Daylight.
 *
 * mapview.js and draw.js add the geometry (ids prefixed odk-) with the cockpit's own
 * colours. This module restyles every one of those layers for the mode the map sits in,
 * so the same project reads as the approved artboards in both:
 *
 *   desk   brand blue for the area and measurements, a light flight line, each line on
 *          a thin dark casing so it holds on bright imagery (SysPlan, SysAnalyse);
 *   field  every line drawn as an 8 px ink casing under a 4 px bright stroke, so it
 *          reads in sunlight on a tablet (SysFieldPlan, SysFieldAnalyse).
 *
 * The mode is read from the nearest [data-mode] ancestor of the map, and re-applied when
 * that attribute changes or the map gains layers (a basemap switch drops and re-adds
 * them). Applying is idempotent: a property already at its value is not set again, so
 * listening to the map's own style events does not loop.
 */

/* ------------------------------------------------------------------ palettes */

// Desk tokens (css/odk/tokens.css). Restated here because MapLibre paint takes
// colours, not custom properties.
const DESK = {
  well: "#0A0D11", accent: "#2F86FF", accentHover: "#6BA8FF", text: "#E6EAF0",
  text2: "#B4BCC8", warn: "#F5A524", danger: "#F06A6A",
};
const FIELD = {
  well: "#0A0A0A", ink: "#0A0A0A", white: "#FFFFFF", action: "#FFD400",
  selection: "#0047B3", danger: "#B91C1C",
};

/** Field lines: the casing and stroke widths the spec fixes (8 px ink under 4 px). */
export const FIELD_CASING_PX = 8;
export const FIELD_STROKE_PX = 4;

/*
 * Paint per role and mode. A role's `line` may carry a `casing` (drawn as a separate
 * line layer directly under the stroke) and a `dash` in line-width units.
 */
export const THEMES = {
  desk: {
    background: { color: DESK.well },
    aoi: {
      fill: { color: DESK.accent, opacity: 0.1 },
      line: { color: DESK.accent, width: 2, casing: { color: DESK.well, width: 4, opacity: 0.55 } },
    },
    nofly: {
      fill: { color: DESK.danger, opacity: 0.16 },
      line: { color: DESK.danger, width: 1.5, casing: { color: DESK.well, width: 3.5, opacity: 0.55 } },
    },
    flight: {
      line: { color: DESK.text, width: 1.25, casing: { color: DESK.well, width: 3.5, opacity: 0.6 } },
    },
    waypoint: { circle: { radius: 2.5, color: DESK.accent, stroke: DESK.well, strokeWidth: 1 } },
    marker: {
      fill: { color: DESK.warn, opacity: 0.12 },
      line: { color: DESK.warn, width: 1.5, casing: { color: DESK.well, width: 3.5, opacity: 0.55 } },
      circle: { radius: 5, color: DESK.warn, stroke: DESK.well, strokeWidth: 1.5 },
    },
    cad: {
      fill: { color: DESK.text2, opacity: 0.06 },
      line: { color: DESK.text2, width: 1, casing: { color: DESK.well, width: 3, opacity: 0.5 } },
      circle: { radius: 3, color: DESK.text2, stroke: DESK.well, strokeWidth: 1 },
    },
    vector: {
      fill: { color: DESK.accentHover, opacity: 0.12 },
      line: { color: DESK.accentHover, width: 1.5, casing: { color: DESK.well, width: 3.5, opacity: 0.55 } },
      // A ring, so a layer's points (photo positions, survey points) never read as the
      // plan's filled capture dots drawn in the same blue family.
      circle: { radius: 3.5, color: DESK.well, stroke: DESK.accentHover, strokeWidth: 1.5 },
    },
    measure: {
      fill: { color: DESK.accent, opacity: 0.18 },
      line: { color: DESK.accent, width: 1.5, casing: { color: DESK.well, width: 3.5, opacity: 0.55 } },
      circle: { radius: 4, color: DESK.text, stroke: DESK.accent, strokeWidth: 2 },
    },
  },
  field: {
    background: { color: FIELD.well },
    aoi: {
      fill: { color: FIELD.white, opacity: 0.14 },
      line: { color: FIELD.white, width: FIELD_STROKE_PX, dash: [3, 1.5],
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
    },
    nofly: {
      fill: { color: FIELD.danger, opacity: 0.25 },
      line: { color: FIELD.danger, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
    },
    flight: {
      line: { color: FIELD.action, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
    },
    waypoint: { circle: { radius: 5, color: FIELD.action, stroke: FIELD.ink, strokeWidth: 3 } },
    marker: {
      fill: { color: FIELD.danger, opacity: 0.2 },
      line: { color: FIELD.danger, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
      circle: { radius: 9, color: FIELD.danger, stroke: FIELD.ink, strokeWidth: 3 },
    },
    cad: {
      fill: { color: FIELD.white, opacity: 0.08 },
      line: { color: FIELD.white, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
      circle: { radius: 6, color: FIELD.white, stroke: FIELD.ink, strokeWidth: 3 },
    },
    vector: {
      fill: { color: FIELD.white, opacity: 0.12 },
      line: { color: FIELD.white, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
      circle: { radius: 6, color: FIELD.white, stroke: FIELD.ink, strokeWidth: 3 },
    },
    measure: {
      fill: { color: FIELD.action, opacity: 0.24 },
      line: { color: FIELD.action, width: FIELD_STROKE_PX,
              casing: { color: FIELD.ink, width: FIELD_CASING_PX, opacity: 1 } },
      circle: { radius: 8, color: FIELD.white, stroke: FIELD.ink, strokeWidth: 3 },
    },
  },
};

/* --------------------------------------------------------------------- roles */

const CASING = "-casing";

/* Which role a layer id plays. Order matters: the first match wins. */
const ROLES = [
  [/^odk-draw-/, "measure"],
  [/^odk-p-aoi-/, "aoi"],
  [/^odk-p-nofly-/, "nofly"],
  [/^odk-(p-mission-line|mission-line)$/, "flight"],
  [/^odk-(p-mission-pt|captures)$/, "waypoint"],
  [/^odk-p-markers-/, "marker"],
  [/^odk-v-site-markers-/, "marker"],
  [/^odk-v-cad_/, "cad"],
  [/^odk-v-/, "vector"],
  [/^odk-p-vector-/, "vector"],
];

/** The role of a map layer id, or null for one this module leaves alone. */
export function roleOf(id) {
  const text = String(id || "");
  if (text === "bg") return "background";
  if (text.endsWith(CASING) || /^odk-p-raster-/.test(text)) return null;
  for (const [pattern, role] of ROLES) if (pattern.test(text)) return role;
  return null;
}

/** "desk" or "field", from the nearest [data-mode] ancestor of `node`. */
export function modeFor(node) {
  const host = node && node.closest ? node.closest("[data-mode]") : null;
  const mode = (host && host.getAttribute("data-mode"))
    || (typeof document !== "undefined" && document.documentElement.getAttribute("data-mode"));
  return mode === "field" ? "field" : "desk";
}

/* ---------------------------------------------------------------- applying */

function same(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/* null clears a property (a dash set in Field mode must not survive the switch to Desk). */
function setPaint(map, id, name, value) {
  if (value === undefined) return;
  const current = map.getPaintProperty(id, name);
  if (value === null ? current === undefined || current === null : same(current, value)) return;
  map.setPaintProperty(id, name, value);
}

function paintFor(type, spec) {
  if (type === "background") return { "background-color": spec.color };
  if (type === "fill" && spec.fill) {
    return { "fill-color": spec.fill.color, "fill-opacity": spec.fill.opacity };
  }
  if (type === "line" && spec.line) {
    return { "line-color": spec.line.color, "line-width": spec.line.width,
             "line-dasharray": spec.line.dash || null };
  }
  if (type === "circle" && spec.circle) {
    return { "circle-radius": spec.circle.radius, "circle-color": spec.circle.color,
             "circle-stroke-color": spec.circle.stroke, "circle-stroke-width": spec.circle.strokeWidth };
  }
  return {};
}

/* The casing under a line layer: same source, filter and visibility, drawn first. */
function syncCasing(map, layer, casing, order) {
  const id = layer.id + CASING;
  const exists = Boolean(map.getLayer(id));
  if (!casing) {
    if (exists) map.removeLayer(id);
    return;
  }
  const visibility = (layer.layout && layer.layout.visibility) || "visible";
  if (!exists) {
    const spec = {
      id, type: "line", source: layer.source,
      layout: { "line-join": "round", "line-cap": "round", visibility },
      paint: { "line-color": casing.color, "line-width": casing.width, "line-opacity": casing.opacity },
    };
    if (layer["source-layer"]) spec["source-layer"] = layer["source-layer"];
    if (layer.filter) spec.filter = layer.filter;
    map.addLayer(spec, layer.id);
    return;
  }
  // A layer moved to the top (mapview lifts the drawing layers) leaves its casing
  // behind; it goes back directly under its line.
  if (order && order[order.indexOf(layer.id) - 1] !== id) map.moveLayer(id, layer.id);
  setPaint(map, id, "line-color", casing.color);
  setPaint(map, id, "line-width", casing.width);
  setPaint(map, id, "line-opacity", casing.opacity);
  if (map.getLayoutProperty(id, "visibility") !== visibility) map.setLayoutProperty(id, "visibility", visibility);
}

/**
 * Restyle every odk- layer on `map` for `mode` ("desk" | "field"; default: the mode of
 * the map's container). Returns the number of layers styled.
 */
export function applyTheme(map, mode) {
  if (!map || !map.getStyle) return 0;
  let style;
  try { style = map.getStyle(); } catch { return 0; }
  if (!style || !style.layers) return 0;
  const chosen = mode || modeFor(map.getContainer && map.getContainer());
  const theme = THEMES[chosen] || THEMES.desk;
  let styled = 0;
  const order = style.layers.map((layer) => layer.id);
  for (const layer of style.layers) {
    const role = roleOf(layer.id);
    if (!role || !theme[role]) continue;
    const spec = theme[role];
    for (const [name, value] of Object.entries(paintFor(layer.type, spec))) {
      setPaint(map, layer.id, name, value);
    }
    if (layer.type === "line") {
      syncCasing(map, layer, spec.line && spec.line.casing, order);
      // Round joins: a right-angled corner of a survey area should not spike.
      if (map.getLayoutProperty(layer.id, "line-join") !== "round") map.setLayoutProperty(layer.id, "line-join", "round");
    }
    styled += 1;
  }
  map._odkTheme = chosen;
  return styled;
}

/**
 * Show or hide map layers by id, with any casing this module drew under them -- a
 * hidden boundary must not leave its black outline behind.
 */
export function setLayersVisible(map, ids, visible) {
  const value = visible ? "visible" : "none";
  for (const id of ids || []) {
    for (const target of [id, id + CASING]) {
      if (map.getLayer(target) && map.getLayoutProperty(target, "visibility") !== value) {
        map.setLayoutProperty(target, "visibility", value);
      }
    }
  }
}

const bound = new WeakSet();

/**
 * Keep `map` styled for its mode: now, whenever layers are added or the basemap is
 * switched, and whenever the [data-mode] of its host changes. Returns a function that
 * stops watching.
 */
export function bindTheme(map) {
  if (!map || bound.has(map)) return () => {};
  bound.add(map);
  let pending = false;
  const schedule = () => {
    if (pending) return;
    pending = true;
    setTimeout(() => {
      pending = false;
      try { applyTheme(map); } catch { /* the style is mid-swap; the next styledata retries */ }
    }, 0);
  };
  map.on("styledata", schedule);
  if (map.loaded && map.loaded()) schedule();
  map.on("load", schedule);

  let observer = null;
  if (typeof MutationObserver === "function" && typeof document !== "undefined") {
    observer = new MutationObserver(schedule);
    const container = map.getContainer && map.getContainer();
    const host = (container && container.closest && container.closest("[data-mode]")) || document.documentElement;
    observer.observe(host, { attributes: true, attributeFilter: ["data-mode"] });
    if (host !== document.documentElement) {
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-mode"] });
    }
  }
  return () => {
    map.off("styledata", schedule);
    map.off("load", schedule);
    if (observer) observer.disconnect();
    bound.delete(map);
  };
}
