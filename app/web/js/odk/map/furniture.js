/* Map furniture for the v1 interface, on a live MapLibre map.
 *
 * mapview.js mounts the map and draws the project (extended, not forked: this module
 * asks it for `controls: false` and reads what showProject drew). Everything a survey
 * map carries around its edge is built here, in the approved artboards' layout
 * (SysAnalyse for desk, SysFieldAnalyse for Field):
 *
 *   top left      CRS chip (the project's EPSG; a click copies the WKT), the resolution
 *                 of the product on top, and any basemap notice. Field: the base-map
 *                 switch (Photo / Height / Shade) above the CRS.
 *   top right     zoom in, zoom out, fit to the survey, Layers (desk), 3D when the screen
 *                 offers it; the north arrow under it, turning with the bearing (a click
 *                 puts north up).
 *   bottom left   a true scale bar and its ratio. Field: the cursor position beside it.
 *   bottom right  the elevation legend, from the range the DSM tiles are coloured with,
 *                 and a key to the geometry actually on the map.
 *
 * Nothing here is a sample value. The CRS, resolution and elevation range are Api
 * answers (get_project_crs, describe_crs, list_layers, raster_tiles); the scale and the
 * cursor position are arithmetic on the map's own view.
 */

import { el, icon, popover, northArrow } from "../components.js";
import { tryCall } from "../../workspace/api.js";
import {
  basemapCatalogue, currentBasemapKey, fitTo, mountMap, setBasemap, showProject, shownVectors,
  vectorLayerIds, visibleBounds,
} from "../../workspace/mapview.js";
import { drawingState, onDrawingChange } from "../../workspace/draw.js";
import { ACTIONS as MAP_ACTIONS } from "./actions.js";
import { applyTheme, bindTheme, modeFor, roleOf, setLayersVisible } from "./theme.js";

/* ------------------------------------------------------------ arithmetic */

const EARTH_CIRCUMFERENCE_M = 40075016.686;
/* CSS pixels per metre at 100% zoom: 96 per inch. */
const PX_PER_M = 96 / 0.0254;

/** Ground metres per CSS pixel at the map's centre (MapLibre's 512 px world tile). */
export function metresPerPixel(map) {
  const { lat } = map.getCenter();
  return (EARTH_CIRCUMFERENCE_M * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** map.getZoom());
}

/** The longest 1-2-5 distance whose bar fits in `maxPx`. */
export function niceScale(mpp, maxPx) {
  const limit = mpp * maxPx;
  const power = 10 ** Math.floor(Math.log10(limit));
  const metres = [5, 2, 1].map((f) => f * power).find((d) => d <= limit) || power;
  return { metres, px: metres / mpp };
}

/* The unit a distance is written in, and its value in that unit. */
function inUnit(metres) {
  if (metres >= 1000) return { unit: "km", factor: 1000 };
  if (metres >= 1) return { unit: "m", factor: 1 };
  return { unit: "cm", factor: 0.01 };
}

export function formatDistance(metres) {
  const { unit, factor } = inUnit(metres);
  return `${trim(metres / factor)} ${unit}`;
}

function trim(value) {
  return Number(value.toPrecision(3)).toLocaleString("en-US");
}

/** "1 : 1,680": ground distance over the same distance on a 96 dpi screen. */
export function scaleRatio(mpp) {
  // Three significant figures, the same rule as the footer's scale (shell.formatScale),
  // so the map and the footer never show two different ratios for one view.
  const denominator = mpp * PX_PER_M;
  const step = 10 ** Math.max(0, Math.floor(Math.log10(Math.max(1, denominator))) + 1 - 3);
  const rounded = Math.round(denominator / step) * step;
  return `1 : ${rounded.toLocaleString("en-US")}`;
}

export function formatLonLat({ lng, lat }) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lng >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}° ${ns} · ${Math.abs(lng).toFixed(5)}° ${ew}`;
}

function formatResolution(m) {
  if (!Number.isFinite(m)) return null;
  const digits = m < 0.1 ? 3 : m < 10 ? 2 : 1;
  return `${m.toFixed(digits)} m/px`;
}

const ELEVATION = /\b(dsm|dtm|dem|elevation|height|terrain)\b/i;
const NOT_ELEVATION = /hillshade|slope|aspect|ndvi|thermal/i;

/* --------------------------------------------------------------- the CRS */

/* The CRS the products are in: the project's choice, else the one its rasters share. */
async function projectCrs(ask, layers) {
  const chosen = await ask("get_project_crs");
  let epsg = chosen && chosen.epsg ? Number(chosen.epsg) : null;
  let source = epsg ? "project" : null;
  if (!epsg) {
    const counts = new Map();
    for (const layer of layers || []) {
      if (layer.kind === "vector" || !layer.crs_epsg) continue;
      counts.set(layer.crs_epsg, (counts.get(layer.crs_epsg) || 0) + 1);
    }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    if (top) { epsg = Number(top[0]); source = "layers"; }
  }
  if (!epsg) return null;
  const described = await ask("describe_crs", epsg);
  return {
    epsg, source,
    short: (described && described.short) || null,
    name: (described && described.name) || null,
    wkt: (described && described.wkt) || null,
  };
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through: WebView2 without clipboard permission */ }
  const area = el("textarea", { class: "odk-visually-hidden", "aria-hidden": "true" });
  area.value = text;
  document.body.appendChild(area);
  area.select();
  let copied = false;
  try { copied = document.execCommand("copy"); } catch { copied = false; }
  area.remove();
  return copied;
}

/* ------------------------------------------------------------- the mount */

const BASE_LABELS = [
  [/hillshade/i, "Shade"],
  [/ortho/i, "Photo"],
  [/\bdtm\b/i, "Ground"],
  [/\b(dsm|dem|elevation|height)\b/i, "Height"],
];

function baseLabel(layer) {
  const name = layer.name || layer.id;
  const hit = BASE_LABELS.find(([pattern]) => pattern.test(name));
  return hit ? hit[1] : name;
}

const GEOMETRY_LABELS = {
  measure: "measurement", aoi: "survey area", nofly: "no-fly zone", flight: "flight line",
  waypoint: "capture point", marker: "site marker", cad: "CAD overlay", vector: "vector layer",
};

/**
 * Put the furniture on `map`, inside `host` (the map's positioned container).
 *
 *   ask        Api reader, (method, ...args) => result | null. Default tryCall.
 *   run        (actionKey, payload) => result: the shell's runAction. Default runs the
 *              map's own verbs (map/actions.js) with a minimal context.
 *   onToggle3D shows a "3D" button in the stack when given.
 *   onCursor   ({lng, lat}) on every pointer move (the desk footer readout).
 *   notify     (text, kind) for a toast; kind in ok | warn | danger.
 *   fitOnLoad  fit the view to the project once it has been drawn (default true).
 *
 * Returns { node, refresh(), update(), state(), openLayers(), destroy() }.
 */
export function mountMapFurniture(map, host, {
  ask = tryCall,
  run = null,
  onToggle3D = null,
  onCursor = null,
  notify = null,
  fitOnLoad = true,
} = {}) {
  const say = (text, kind = "ok") => { if (notify) notify(text, kind); };
  const runAction = run || (async (key, payload) => {
    const entry = MAP_ACTIONS[key];
    if (!entry) return { error: `unknown action ${key}` };
    try {
      return await entry.run({ payload, confirm: async () => true, toast: say });
    } catch (error) {
      return { error: String((error && error.message) || error) };
    }
  });

  const state = {
    crs: null, layers: [], drawn: [], bounds: null, notice: null, fitted: false,
    cursor: null, mpp: null, scale: null,
  };

  const node = el("div", { class: "odk-mapf", "data-furniture": "" });
  const tl = el("div", { class: "odk-mapf-tl" });
  const tr = el("div", { class: "odk-mapf-tr" });
  const bl = el("div", { class: "odk-mapf-bl" });
  const br = el("div", { class: "odk-mapf-br" });
  node.append(tl, tr, bl, br);
  host.appendChild(node);

  /* -- top left: base switch (field), CRS, resolution, notice -------------- */
  const base = el("div", { class: "odk-mapf-base", role: "group", "aria-label": "Base map" });
  const crsChip = el("button", { type: "button", class: "odk-map-chip odk-mapf-crs", "data-control": "copy-crs",
    title: "Copy the WKT of this CRS", text: "CRS not reported" });
  crsChip.disabled = true;
  // Field shows the CRS as a label: copying WKT is desk work, and a button there would
  // have to be a 64 px target in the way of the map.
  const crsLabel = el("span", { class: "odk-map-chip odk-mapf-crs-label", hidden: "" });
  const infoChip = el("span", { class: "odk-map-chip odk-mapf-info", hidden: "" });
  const noticeChip = el("span", { class: "odk-map-chip odk-mapf-notice", role: "status", hidden: "" });
  tl.append(base, crsChip, crsLabel, infoChip, noticeChip);

  crsChip.addEventListener("click", async () => {
    if (!state.crs || !state.crs.wkt) return;
    const copied = await copyText(state.crs.wkt);
    if (copied) {
      crsChip.setAttribute("data-copied", "");
      const label = crsChip.textContent;
      crsChip.textContent = "WKT copied";
      setTimeout(() => { crsChip.removeAttribute("data-copied"); crsChip.textContent = label; }, 1400);
      say(`WKT for EPSG:${state.crs.epsg} copied.`, "ok");
    } else {
      say("The clipboard refused the WKT. Settings › Units & CRS shows it to copy by hand.", "warn");
    }
  });

  /* -- top right: zoom / fit / layers / 3D, north ---------------------------- */
  const stackButton = (iconName, label, control, onClick) => {
    const b = el("button", { type: "button", class: "odk-icon-btn", "aria-label": label, title: label,
      "data-control": control });
    b.appendChild(icon(iconName));
    b.addEventListener("click", onClick);
    return b;
  };
  const zoomIn = stackButton("plus", "Zoom in", "zoom-in", () => map.zoomIn());
  const zoomOut = stackButton("minus", "Zoom out", "zoom-out", () => map.zoomOut());
  const fit = stackButton("fit", "Fit to survey extent", "fit-extent", () => {
    if (state.bounds) fitTo(map, state.bounds);
  });
  fit.disabled = true;
  const layersButton = stackButton("layers", "Layers", "layers", () => {});
  const buttons = [zoomIn, zoomOut, fit, layersButton];
  if (onToggle3D) {
    const three = el("button", { type: "button", class: "odk-icon-btn odk-mapf-3d", "aria-label": "Toggle 3D view",
      title: "Toggle 3D view", "data-control": "toggle-3d", text: "3D" });
    three.addEventListener("click", () => onToggle3D());
    buttons.push(three);
  }
  const stack = el("div", { class: "odk-map-stack", role: "toolbar", "aria-label": "Map view" }, buttons);

  const arrow = northArrow({ bearing: 0 });
  const north = el("button", { type: "button", class: "odk-north", "data-control": "north-up",
    "aria-label": "North up", title: "North up" }, [...arrow.childNodes]);
  north.addEventListener("click", () => map.easeTo({ bearing: 0, pitch: 0, duration: 300 }));
  tr.append(stack, north);

  /* -- bottom left: scale bar, ratio, cursor (field) ------------------------ */
  const scaleLabels = el("div", { class: "odk-mapf-scale-labels" }, [
    el("span", { text: "0" }), el("span"), el("span"),
  ]);
  const scaleBar = el("div", { class: "odk-mapf-scale-bar", "aria-hidden": "true" }, [el("span"), el("span")]);
  const scaleRatioNode = el("span", { class: "odk-mapf-scale-ratio" });
  const scale = el("div", { class: "odk-mapf-scale", role: "img" }, [scaleLabels, scaleBar, scaleRatioNode]);
  const cursorBox = el("div", { class: "odk-mapf-cursor", "aria-live": "off" });
  bl.append(scale, cursorBox);

  /* -- bottom right: legend ------------------------------------------------- */
  const legend = el("div", { class: "odk-mapf-legend", hidden: "" });
  br.append(legend);

  /* ---------------------------------------------------------------- updates */

  const maxBarPx = () => (modeFor(host) === "field" ? 116 : 225);

  function updateScale() {
    const mpp = metresPerPixel(map);
    const { metres, px } = niceScale(mpp, maxBarPx());
    state.mpp = mpp;
    state.scale = { metres, px };
    const width = `${px.toFixed(1)}px`;
    scaleLabels.style.width = width;
    scaleBar.style.width = width;
    // The midpoint, in the end label's unit: 0 · 500 · 1 km.
    scaleLabels.children[1].textContent = trim(metres / 2 / inUnit(metres).factor);
    scaleLabels.children[2].textContent = formatDistance(metres);
    const ratio = scaleRatio(mpp);
    scaleRatioNode.textContent = `${ratio} at 100%`;
    scale.setAttribute("aria-label", `Scale bar ${formatDistance(metres)}, ${ratio}`);
    node.dataset.mpp = String(mpp);
    node.dataset.scaleMetres = String(metres);
    node.dataset.scalePx = String(px);
  }

  function updateNorth() {
    const bearing = map.getBearing();
    const svg = north.querySelector("svg");
    if (svg) svg.style.transform = `rotate(${-bearing}deg)`;
    node.dataset.bearing = String(bearing);
    north.setAttribute("aria-label", Math.abs(bearing) < 0.05 ? "North up"
      : `North up (map turned ${Math.round(((bearing % 360) + 360) % 360)}°)`);
  }

  function updateCursor() {
    const at = state.cursor || map.getCenter();
    cursorBox.textContent = formatLonLat(at);
  }

  function update() {
    updateScale();
    updateNorth();
    updateCursor();
  }

  function renderCrs() {
    const crs = state.crs;
    const field = modeFor(host) === "field";
    crsChip.hidden = field;
    crsLabel.hidden = !field;
    if (!crs) {
      crsChip.textContent = "CRS not reported";
      crsLabel.textContent = "CRS not reported";
      crsChip.disabled = true;
      delete node.dataset.crsEpsg;
      return;
    }
    const short = crs.short || crs.name || "";
    crsChip.textContent = field
      ? [short, `EPSG:${crs.epsg}`].filter(Boolean).join(" · ")
      : [`EPSG:${crs.epsg}`, short].filter(Boolean).join(" · ");
    crsLabel.textContent = crsChip.textContent;
    crsChip.disabled = !crs.wkt;
    crsChip.title = crs.source === "project"
      ? `${crs.name || ""} (project CRS). Click to copy the WKT.`
      : `${crs.name || ""} (the CRS of the project's rasters; none chosen). Click to copy the WKT.`;
    node.dataset.crsEpsg = String(crs.epsg);
  }

  /* The raster on top that the map draws from tiles, for the resolution chip. */
  function topRaster() {
    const rasters = state.drawn.filter((d) => d.layer.kind !== "vector" && isShown(d));
    return rasters[rasters.length - 1] || null;
  }

  function isShown(entry) {
    return entry.mapLayers.some((id) => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
  }

  function renderInfo() {
    const top = topRaster();
    if (!top) { infoChip.hidden = true; return; }
    const name = String(top.layer.name || top.layer.id).toUpperCase();
    const res = top.tiles ? formatResolution(Number(top.tiles.native_resolution_m)) : null;
    infoChip.textContent = top.source === "preview" ? `${name} · preview` : [name, res].filter(Boolean).join(" · ");
    infoChip.hidden = false;
  }

  function renderNotice() {
    const notice = state.notice;
    noticeChip.hidden = !notice;
    if (!notice) return;
    noticeChip.textContent = notice.code === "offline-fallback"
      ? "No network · showing the offline cache"
      : notice.basemap === "offline" ? "Offline cache: no tiles for this view"
        : "Basemap unreachable · offline";
  }

  /* The elevation layer drawn on the map, with the range its colours stand for. */
  function elevationEntry() {
    const shown = state.drawn.filter((d) => isShown(d) && d.tiles && Array.isArray(d.tiles.value_range)
      && d.tiles.bands === 1 && ELEVATION.test(d.layer.name || "") && !NOT_ELEVATION.test(d.layer.name || ""));
    return shown[shown.length - 1] || null;
  }

  function geometryOnMap() {
    let layers = [];
    try { layers = (map.getStyle() || {}).layers || []; } catch { layers = []; }
    // Sources that are on the map but empty draw nothing and stay out of the key: a
    // pinned overlay with no features, the drawing layer with no drawing. (Project
    // sources are only added when they hold something.)
    const pinned = new Set(Object.entries(shownVectors()).filter(([, count]) => count > 0)
      .flatMap(([id]) => vectorLayerIds(map, id)));
    const drawing = drawingState();
    const drawn = Boolean(drawing.finished || drawing.points.length);
    const roles = new Set();
    for (const layer of layers) {
      const role = roleOf(layer.id);
      if (!role || role === "background" || !GEOMETRY_LABELS[role]) continue;
      if (map.getLayoutProperty(layer.id, "visibility") === "none") continue;
      if (layer.id.startsWith("odk-v-") && !pinned.has(layer.id)) continue;
      if (layer.id.startsWith("odk-draw-") && !drawn) continue;
      roles.add(role);
    }
    return [...roles];
  }

  /* Whether a drawn vector layer holds only points (camera positions): a dot in the key. */
  function pointsOnly(entry) {
    const first = map.getLayer(entry.mapLayers[0]);
    let features = [];
    try { features = first ? map.querySourceFeatures(first.source) : []; } catch { features = []; }
    return features.length > 0 && features.every((f) => /Point/.test((f.geometry || {}).type || ""));
  }

  function renderLegend() {
    const elevation = elevationEntry();
    const roles = geometryOnMap();
    const parts = [];
    if (elevation) {
      const [low, high] = elevation.tiles.value_range.map(Number);
      const mid = (low + high) / 2;
      parts.push(
        el("div", { class: "odk-mapf-legend-head" }, [
          el("span", { text: "Elevation" }),
          el("span", { text: `${elevation.layer.name || elevation.layer.id} · m` }),
        ]),
        el("div", { class: "odk-mapf-ramp", role: "img",
          "aria-label": `Turbo colour ramp from ${low.toFixed(1)} m to ${high.toFixed(1)} m` }),
        el("div", { class: "odk-mapf-ticks" }, [low, mid, high].map((v) => el("span", { text: `${v.toFixed(1)} m` }))),
      );
      legend.dataset.min = String(low);
      legend.dataset.max = String(high);
    } else {
      delete legend.dataset.min;
      delete legend.dataset.max;
    }
    // Project vector layers are named by the layer ("Camera positions"); the rest by role.
    const vectors = state.drawn.filter((d) => d.layer.kind === "vector" && isShown(d))
      .map((d) => ({ role: "vector", label: d.layer.name || d.layer.id, swatch: pointsOnly(d) ? "vector-point" : "vector" }));
    const items = roles.flatMap((role) => (role === "vector" && vectors.length
      ? vectors : [{ role, label: GEOMETRY_LABELS[role], swatch: role === "measure" ? "measurement" : role }]));
    if (items.length) {
      parts.push(el("div", { class: "odk-geom-key" }, items.map(({ label, swatch }) =>
        el("div", { class: "odk-geom-item" }, [
          el("span", { class: `odk-geom-swatch odk-geom-swatch--${swatch}`, "aria-hidden": "true" }),
          el("span", { text: label }),
        ]))));
    }
    legend.replaceChildren(...parts);
    legend.hidden = !parts.length;
  }

  /* Field: one raster at a time, chosen by what it shows. */
  function renderBaseSwitch() {
    const rasters = state.layers.filter((l) => l.kind !== "vector" && l.crs_epsg);
    if (rasters.length < 2) { base.replaceChildren(); base.hidden = true; return; }
    base.hidden = false;
    // The raster on top is the one the map is showing; the switch shows one at a time.
    const top = topRaster();
    base.replaceChildren(...rasters.map((layer) => {
      const b = el("button", { type: "button", "data-control": "base-map", text: baseLabel(layer),
        "aria-pressed": String(Boolean(top && top.layer.id === layer.id)), title: layer.name || layer.id });
      b.addEventListener("click", () => chooseBase(layer, rasters));
      return b;
    }));
  }

  async function chooseBase(chosen, rasters) {
    for (const layer of rasters) {
      const want = layer.id === chosen.id;
      if ((layer.visible !== false) !== want) await ask("set_layer_visible", layer.id, want);
    }
    await redraw();
  }

  function renderAll() {
    renderCrs();
    renderInfo();
    renderNotice();
    renderLegend();
    renderBaseSwitch();
    layersButton.hidden = modeFor(host) === "field";
    update();
  }

  /* ----------------------------------------------------------- the project */

  async function readProject() {
    state.drawn = map._odkProjectLayers || [];
    state.bounds = map._odkProjectBounds || null;
    const listed = await ask("list_layers");
    state.layers = (listed && listed.layers) || map._odkProjectListed || [];
    state.crs = await projectCrs(ask, state.layers);
    fit.disabled = !state.bounds;
    renderAll();
  }

  /** Draw the project again (after a layer change) and re-read everything. */
  async function redraw() {
    await showProject(map, ask);
  }

  const onProject = () => {
    readProject().then(() => {
      if (fitOnLoad && !state.fitted && state.bounds) { state.fitted = true; fitTo(map, state.bounds); }
    });
  };
  host.addEventListener("odk:project", onProject);

  /* ----------------------------------------------------- Layers popover */

  const pop = popover({ anchor: layersButton, title: "Layers", content: el("div") });
  pop.node.classList.add("odk-mapf-pop");
  const popBody = pop.node.querySelector(".odk-popover-body");
  layersButton.addEventListener("click", () => {
    if (pop.node.hidden) return;
    buildLayers();
    // Open beside the stack, not over it.
    const r = layersButton.getBoundingClientRect();
    const width = pop.node.offsetWidth || 340;
    const top = Math.max(8, r.top);
    pop.node.style.left = `${Math.max(8, r.left - width - 8)}px`;
    pop.node.style.top = `${top}px`;
    // It scrolls inside the window rather than running off its bottom edge.
    pop.node.style.maxHeight = `${Math.max(160, window.innerHeight - top - 8)}px`;
  });

  const group = (label, children) => el("div", { class: "odk-mapf-group", role: "group", "aria-label": label }, [
    el("span", { class: "odk-label", text: label }), ...children,
  ]);

  async function buildLayers() {
    const [cache, listed] = await Promise.all([ask("basemap_cache_status"), ask("list_layers")]);
    state.layers = (listed && listed.layers) || state.layers;
    popBody.replaceChildren(basemapGroup(cache), projectGroup(), overlayGroup(), cacheFoot(cache));
  }

  function cacheNote(cache) {
    const sat = cache && cache.providers && cache.providers.satellite;
    const levels = sat && Array.isArray(sat.zoom_levels) ? sat.zoom_levels : [];
    if (!levels.length) return "nothing cached yet";
    const tiles = Number(sat.tiles ?? sat.tile_count);
    const range = levels.length > 1 ? `z${Math.min(...levels)}–${Math.max(...levels)}` : `z${levels[0]}`;
    return Number.isFinite(tiles) ? `${range} · ${tiles.toLocaleString("en-US")} tiles` : range;
  }

  function basemapGroup(cache) {
    const current = currentBasemapKey();
    const names = { offline: "Offline cache", satellite: "Satellite · Esri", street: "Street · OSM",
                    topo: "Topographic · Esri" };
    const rows = basemapCatalogue().map((entry) => {
      const radio = el("input", { type: "radio", name: "odk-basemap", value: entry.key });
      radio.checked = entry.key === current;
      radio.addEventListener("change", () => chooseBasemap(entry.key));
      let note = "";
      if (entry.key === "offline") note = cacheNote(cache);
      else if (entry.key === current && entry.depthTracked && Number.isFinite(map._odkBasemapDepth)) {
        note = `imagery to z${map._odkBasemapDepth} here`;
      }
      return el("label", { class: "odk-mapf-row", "data-basemap": entry.key }, [
        radio,
        el("span", { class: "odk-mapf-row-name", text: names[entry.key] || entry.key }),
        note ? el("span", { class: "odk-mapf-row-note", text: note }) : null,
        entry.network ? el("span", { class: "odk-mapf-net", title: "Fetched from the internet", text: "NETWORK" }) : null,
      ]);
    });
    return group("Basemap", rows);
  }

  function chooseBasemap(key) {
    state.notice = null;
    setBasemap(key, { keepLayers: true });
    renderNotice();
    say(`Basemap: ${key === "offline" ? "offline cache" : key}.`, "ok");
  }

  function drawnFor(layerId) {
    return state.drawn.find((d) => d.layer.id === layerId) || null;
  }

  function projectGroup() {
    if (!state.layers.length) {
      return group("Project layers", [el("p", { class: "odk-mapf-empty",
        text: "No layers yet. Products appear here once a reconstruction has run." })]);
    }
    // Top of the map first.
    const rows = [...state.layers].reverse().map((layer) => layerRow(layer));
    return group("Project layers", rows);
  }

  function layerRow(layer) {
    const visible = layer.visible !== false;
    const box = el("input", { type: "checkbox", "aria-label": `Show ${layer.name || layer.id}` });
    box.checked = visible;
    const drawn = drawnFor(layer.id);
    const note = !layer.crs_epsg && layer.kind !== "vector" ? "no CRS"
      : drawn && drawn.tiles ? `to z${drawn.tiles.maxzoom}`
        : drawn && drawn.source === "preview" ? "preview" : layer.crs_epsg ? `EPSG:${layer.crs_epsg}` : "";
    const remove = el("button", { type: "button", class: "odk-icon-btn odk-mapf-remove", "data-action": "Remove Layer",
      "aria-label": `Remove ${layer.name || layer.id}`, title: "Remove from the layer list" });
    remove.appendChild(icon("close", { size: 14 }));
    remove.addEventListener("click", async (event) => {
      event.preventDefault();
      const result = await runAction("Remove Layer", { layer });
      if (result && result.error) say(result.error, "danger");
      else if (result && result.message) say(result.message, "ok");
      if (result && !result.skipped && !result.error) {
        // The list answers at once: a removed layer must not linger, with live controls,
        // for as long as the map takes to redraw. The notes refresh after the redraw.
        await buildLayers();
        await redraw();
        buildLayers();
      }
    });

    const range = el("input", { type: "range", min: "0", max: "100", step: "1",
      "aria-label": `Opacity of ${layer.name || layer.id}` });
    const opacity = Number.isFinite(Number(layer.opacity)) ? Number(layer.opacity) : 1;
    range.value = String(Math.round(opacity * 100));
    const out = el("output", { text: `${range.value}%` });
    range.addEventListener("input", () => {
      out.textContent = `${range.value}%`;
      paintOpacity(layer, Number(range.value) / 100);
    });
    range.addEventListener("change", async () => {
      const result = await ask("set_layer_opacity", layer.id, Number(range.value) / 100);
      if (!result) say(`${layer.name || layer.id}: the opacity was not saved.`, "warn");
      else layer.opacity = Number(range.value) / 100;
    });
    range.disabled = !visible;

    box.addEventListener("change", async () => {
      const result = await ask("set_layer_visible", layer.id, box.checked);
      if (!result) { box.checked = !box.checked; say(`${layer.name || layer.id}: the change was refused.`, "warn"); return; }
      layer.visible = box.checked;
      range.disabled = !box.checked;
      const entry = drawnFor(layer.id);
      if (entry) setLayersVisible(map, entry.mapLayers, box.checked);
      else if (box.checked) await redraw();
      renderAll();
    });

    const swatch = layer.kind === "vector" ? "odk-swatch odk-swatch--line"
      : drawn && drawn.tiles && drawn.tiles.bands === 1 ? "odk-swatch odk-swatch--ramp" : "odk-swatch odk-swatch--polygon";
    return el("div", { class: "odk-mapf-layer", "data-layer": layer.id }, [
      el("label", { class: "odk-mapf-row" }, [
        box, el("span", { class: swatch, "aria-hidden": "true" }),
        el("span", { class: "odk-mapf-row-name", text: layer.name || layer.id, title: layer.path || "" }),
        note ? el("span", { class: "odk-mapf-row-note", text: note }) : null,
        remove,
      ]),
      el("div", { class: "odk-mapf-opacity" }, [range, out]),
    ]);
  }

  function paintOpacity(layer, value) {
    const entry = drawnFor(layer.id);
    if (!entry) return;
    for (const id of entry.mapLayers) {
      const mapLayer = map.getLayer(id);
      if (!mapLayer) continue;
      if (mapLayer.type === "raster") map.setPaintProperty(id, "raster-opacity", value);
      else if (mapLayer.type === "line") map.setPaintProperty(id, "line-opacity", value);
      else if (mapLayer.type === "circle") map.setPaintProperty(id, "circle-opacity", value);
      else if (mapLayer.type === "fill") map.setPaintProperty(id, "fill-opacity", 0.2 * value);
    }
  }

  /* Overlays: CAD drawings and site markers (pinned), and the project's own geometry.
     Showing or hiding them is a view choice; nothing is saved. */
  function overlayGroup() {
    const rows = [];
    const toggle = (label, ids, key) => {
      if (!ids.length) return;
      const box = el("input", { type: "checkbox", "aria-label": `Show ${label}` });
      box.checked = ids.some((id) => map.getLayoutProperty(id, "visibility") !== "none");
      box.addEventListener("change", () => { setLayersVisible(map, ids, box.checked); renderLegend(); });
      rows.push(el("label", { class: "odk-mapf-row", "data-overlay": key }, [
        box, el("span", { class: "odk-mapf-row-name", text: label }),
      ]));
    };
    for (const [id, count] of Object.entries(shownVectors())) {
      const label = id === "site-markers" ? "Site markers" : id.startsWith("cad:") ? `CAD · ${id.slice(4)}` : id;
      toggle(`${label} (${count})`, vectorLayerIds(map, id), id);
    }
    const present = (ids) => ids.filter((id) => map.getLayer(id));
    toggle("Survey area", present(["odk-p-aoi-fill", "odk-p-aoi-line"]), "aoi");
    toggle("No-fly zones", present(["odk-p-nofly-fill", "odk-p-nofly-line"]), "nofly");
    toggle("Mission", present(["odk-p-mission-line", "odk-p-mission-pt"]), "mission");
    toggle("Measurements", present(["odk-draw-fill", "odk-draw-line", "odk-draw-vertex"]), "measure");
    if (!rows.length) rows.push(el("p", { class: "odk-mapf-empty", text: "No overlays on this map." }));
    return group("Overlays", rows);
  }

  function cacheFoot(cache) {
    const button = el("button", { type: "button", class: "odk-btn odk-btn--secondary", "data-action": "Cache Basemap View",
      text: "Cache this view" });
    button.addEventListener("click", async () => {
      button.disabled = true;
      const result = await runAction("Cache Basemap View", { view: visibleBounds() });
      button.disabled = false;
      if (result && result.error) say(result.error, "danger");
      else if (result && result.message) say(result.message, "ok");
    });
    const total = cache && Number.isFinite(Number(cache.max_tiles_per_request))
      ? `Up to ${Number(cache.max_tiles_per_request).toLocaleString("en-US")} satellite tiles per request; nothing downloads while panning.`
      : "Satellite tiles for the area on screen, for a site with no signal.";
    return el("div", { class: "odk-mapf-foot" }, [el("span", { class: "odk-mapf-row-note", text: total }), button]);
  }

  /* -------------------------------------------------------------- wiring */

  const onMove = () => update();
  const onMouse = (event) => {
    state.cursor = event.lngLat;
    updateCursor();
    if (onCursor) onCursor(event.lngLat);
  };
  const onStyle = () => { renderInfo(); renderLegend(); };
  map.on("move", onMove);
  map.on("rotate", updateNorth);
  map.on("mousemove", onMouse);
  map.on("idle", onStyle);
  const unbindTheme = bindTheme(map);
  const stopDrawingWatch = onDrawingChange(() => renderLegend());

  const modeObserver = typeof MutationObserver === "function"
    ? new MutationObserver(() => { renderAll(); applyTheme(map); })
    : null;
  if (modeObserver) {
    modeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-mode"] });
    const hostMode = host.closest && host.closest("[data-mode]");
    if (hostMode && hostMode !== document.documentElement) {
      modeObserver.observe(hostMode, { attributes: true, attributeFilter: ["data-mode"] });
    }
  }

  renderAll();
  if (map._odkProjectLayers) onProject();
  else readProject();

  return {
    node,
    update,
    /** Re-read the project (CRS, layers) after a change elsewhere. */
    refresh: readProject,
    redraw,
    /** mapview's onNotice: an offline basemap or a cache with no tiles here. */
    notice(value) { state.notice = value; renderNotice(); },
    openLayers() { if (pop.node.hidden) layersButton.click(); return pop.node; },
    closeLayers() { if (!pop.node.hidden) pop.close(); },
    state: () => ({ ...state, crs: state.crs && { ...state.crs } }),
    destroy() {
      map.off("move", onMove);
      map.off("rotate", updateNorth);
      map.off("mousemove", onMouse);
      map.off("idle", onStyle);
      host.removeEventListener("odk:project", onProject);
      unbindTheme();
      stopDrawingWatch();
      if (modeObserver) modeObserver.disconnect();
      pop.close();
      pop.node.remove();
      node.remove();
    },
  };
}

/**
 * Mount a map with the v1 furniture into `container`: mapview's map with MapLibre's own
 * controls left out, the project drawn, the mode's paint applied.
 *
 * The basemap starts on the offline tile cache (REBUILD_PLAN 2.5): the online services
 * are opt-ins in the Layers popover, labelled NETWORK. `basemap` picks another start.
 *
 * Returns { map, furniture } or null when no map could be made (mapview says why in
 * the container). Options pass through to mountMap and mountMapFurniture.
 */
export async function mountProjectMap(container, options = {}) {
  container.classList.add("odk-map");
  setBasemap(options.basemap || "offline");
  let furniture = null;
  const pendingNotices = [];
  let drawn = false;
  const draw = (ready) => {
    if (drawn) return;
    drawn = true;
    showProject(ready, options.ask || tryCall);
  };
  const map = mountMap(container, {
    ...options,
    controls: false,
    onNotice: (value) => { if (furniture) furniture.notice(value); else pendingNotices.push(value); },
    onReady: (ready) => {
      draw(ready);
      if (options.onReady) options.onReady(ready);
    },
  });
  if (!map) return null;
  // The project goes on as soon as the style can take it, not after the first basemap
  // tiles: offline, those may be slow to answer or never come.
  map.once("style.load", () => setTimeout(() => draw(map), 0));
  furniture = mountMapFurniture(map, container, options);
  pendingNotices.forEach((value) => furniture.notice(value));
  releaseWhenDetached(container, map, furniture);
  return { map, furniture };
}

/**
 * The shell empties its regions on every screen change and mode switch without telling
 * what was in them. A map left behind keeps its WebGL context (the browser allows about
 * sixteen, then drops the oldest, which can be the one on screen) and its Layers
 * popover, which lives on <body> and stayed open over the next screen. Once the
 * container has been out of the document for a moment, the popover closes and the map
 * is released. A screen that moves its map to another region inside that moment keeps it.
 */
function releaseWhenDetached(container, map, furniture) {
  let pending = null;
  const watch = new MutationObserver(() => {
    if (container.isConnected) { clearTimeout(pending); pending = null; return; }
    furniture.closeLayers();
    if (pending) return;
    pending = setTimeout(() => {
      pending = null;
      if (container.isConnected) return;
      watch.disconnect();
      furniture.destroy();
      try { map.remove(); } catch { /* already gone */ }
    }, 1500);
  });
  watch.observe(document.body, { childList: true, subtree: true });
}
