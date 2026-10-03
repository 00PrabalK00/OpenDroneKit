/* Plan: what the Plan screen adds to the project map (js/odk/map, U3).

   mapview.showProject draws the project -- rasters, camera positions, the area of
   interest, no-fly zones, the planned route and capture points, site markers -- and
   map/theme.js paints them for the mode. This module adds what only Plan shows, all of
   it read from mission_overview (the plan's own numbers, mission/overview.py):

     * the photo footprint of each nadir capture point (planned, at the planned height);
     * START / END labels at the first and last waypoint;
     * the waypoint card on hover or pick: position, altitude and its reference, yaw and
       gimbal, the footprint, the ground under it when terrain is loaded;
     * a cursor the altitude profile and Play move along the route;
     * the drawn route and interest regions the next plan will use, and facade passes;
     * the vertex editor for the area of interest (drag a corner, drag a midpoint to add
       one, Delete removes the picked one): every change goes to set_aoi.

   Nothing here plans or measures: a moved corner is sent to the Api and the screen
   re-reads what the Api then holds. */

import { el } from "../../components.js";
import { lastError, tryCall } from "../../../workspace/api.js";
import { mapReady, showProject } from "../../../workspace/mapview.js";
import { drawingState } from "../../../workspace/draw.js";
import { refreshLive, live } from "../../../workspace/live.js";
import { modeFor } from "../../map/theme.js";
import { fmtLat, fmtLon, fmtNum, form, frameWords } from "./model.js";

const SRC = {
  footprints: "odk-plan-footprints",
  cursor: "odk-plan-cursor",
  route: "odk-plan-route",
  facade: "odk-plan-facade",
  edit: "odk-plan-edit",
};

const PAINT = {
  desk: { foot: "#6BA8FF", cursor: "#E6EAF0", ring: "#2F86FF", route: "#F5A524", well: "#0A0D11", edit: "#2F86FF" },
  field: { foot: "#0047B3", cursor: "#FFFFFF", ring: "#0A0A0A", route: "#0047B3", well: "#0A0A0A", edit: "#0047B3" },
};

const empty = () => ({ type: "FeatureCollection", features: [] });
const point = (c, props = {}) => ({ type: "Feature", properties: props, geometry: { type: "Point", coordinates: c } });

function pad(index) {
  return String(index).padStart(3, "0");
}

/** The signature of what showProject draws, so the project is redrawn only when it changed. */
function geometryKey(overview) {
  if (!overview) return "";
  const plan = overview.plan;
  return JSON.stringify([overview.aoi && overview.aoi.polygon, overview.no_fly_count,
    plan && [plan.template, plan.waypoint_count, plan.distance_m, plan.altitude_m, plan.line_bearing_deg]]);
}

/**
 * Attach the Plan layers to a mounted project map. Returns the controller:
 *   update(overview)      new data from mission_overview
 *   select(index)         pin the card and cursor on a waypoint (null: none)
 *   play(onStep)          move the cursor along the route; resolves when done
 *   stop()                stop playing
 *   setFootprints(on)     show / hide the footprints
 *   editVertices(on)      start / end the vertex editor
 *   showFacade(segments)  draw facade passes (facade_footprint)
 *   destroy()
 */
export function attachPlanMap(map, host, { notify = () => {}, onSelect = () => {} } = {}) {
  const state = {
    overview: null, rows: [], selected: null, hover: null, footprints: true,
    key: null, editing: false, ring: [], picked: null, drag: null, playing: null, facade: [],
  };

  const card = el("div", { class: "odk-plan-card", hidden: "", role: "status", "aria-live": "polite" });
  const startTag = el("div", { class: "odk-plan-endtag odk-plan-endtag--start" });
  const endTag = el("div", { class: "odk-plan-endtag" });
  const editHint = el("div", { class: "odk-plan-edit-hint", hidden: "" });
  host.append(card, editHint);
  const markers = [];
  const Marker = window.maplibregl && window.maplibregl.Marker;
  if (Marker) {
    markers.push(new Marker({ element: startTag, anchor: "left", offset: [12, 0] }));
    markers.push(new Marker({ element: endTag, anchor: "right", offset: [-12, 0] }));
  }

  const mode = () => modeFor(host);
  const paint = () => PAINT[mode()] || PAINT.desk;

  /* ------------------------------------------------------------- layers */

  function ensureLayers() {
    if (!mapReady(map)) return false;
    const p = paint();
    const add = (id, spec) => { if (!map.getLayer(id)) map.addLayer({ id, ...spec }); };
    for (const id of Object.values(SRC)) if (!map.getSource(id)) map.addSource(id, { type: "geojson", data: empty() });
    // Every footprint as a faint fill: overlapping photos build up, so the map shows
    // where the plan photographs the ground once, twice or more. The hovered or picked
    // capture point's own footprint is outlined.
    add(`${SRC.footprints}-line`, { type: "fill", source: SRC.footprints,
      paint: { "fill-color": p.foot, "fill-opacity": 0.022 } });
    add(`${SRC.footprints}-hl`, { type: "line", source: SRC.footprints,
      filter: ["==", ["get", "index"], -1],
      paint: { "line-color": p.foot, "line-width": 1.5, "line-dasharray": [3, 2] } });
    add(`${SRC.route}-line`, { type: "line", source: SRC.route,
      paint: { "line-color": p.route, "line-width": 2, "line-dasharray": [2, 1] } });
    add(`${SRC.facade}-line`, { type: "line", source: SRC.facade,
      paint: { "line-color": p.route, "line-width": 2 } });
    add(`${SRC.edit}-line`, { type: "line", source: SRC.edit, filter: ["==", ["geometry-type"], "LineString"],
      paint: { "line-color": p.edit, "line-width": 2 } });
    add(`${SRC.edit}-mid`, { type: "circle", source: SRC.edit, filter: ["==", ["get", "role"], "mid"],
      paint: { "circle-radius": 4, "circle-color": p.well, "circle-stroke-color": p.edit, "circle-stroke-width": 1.5 } });
    add(`${SRC.edit}-vertex`, { type: "circle", source: SRC.edit, filter: ["==", ["get", "role"], "vertex"],
      paint: { "circle-radius": 5, "circle-color": p.edit, "circle-stroke-color": p.well, "circle-stroke-width": 1.5 } });
    add(`${SRC.edit}-picked`, { type: "circle", source: SRC.edit, filter: ["==", ["get", "role"], "picked"],
      paint: { "circle-radius": 7, "circle-color": p.cursor, "circle-stroke-color": p.edit, "circle-stroke-width": 2 } });
    add(`${SRC.cursor}-pt`, { type: "circle", source: SRC.cursor,
      paint: { "circle-radius": 7, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": p.cursor, "circle-stroke-width": 2 } });
    for (const id of [`${SRC.footprints}-line`, `${SRC.footprints}-hl`]) {
      map.setLayoutProperty(id, "visibility", state.footprints ? "visible" : "none");
    }
    // Ours above the project's geometry, the editor and cursor on top.
    for (const id of [`${SRC.footprints}-line`, `${SRC.footprints}-hl`, `${SRC.route}-line`, `${SRC.facade}-line`,
      `${SRC.edit}-line`, `${SRC.edit}-mid`, `${SRC.edit}-vertex`, `${SRC.edit}-picked`, `${SRC.cursor}-pt`]) {
      if (map.getLayer(id)) map.moveLayer(id);
    }
    return true;
  }

  function repaint() {
    if (!ensureLayers()) return;
    const p = paint();
    const set = (id, name, value) => { if (map.getLayer(id)) map.setPaintProperty(id, name, value); };
    set(`${SRC.footprints}-line`, "fill-color", p.foot);
    set(`${SRC.footprints}-hl`, "line-color", p.foot);
    set(`${SRC.route}-line`, "line-color", p.route);
    set(`${SRC.facade}-line`, "line-color", p.route);
    set(`${SRC.edit}-line`, "line-color", p.edit);
    set(`${SRC.edit}-vertex`, "circle-color", p.edit);
    set(`${SRC.edit}-vertex`, "circle-stroke-color", p.well);
    set(`${SRC.edit}-mid`, "circle-stroke-color", p.edit);
    set(`${SRC.edit}-mid`, "circle-color", p.well);
    set(`${SRC.cursor}-pt`, "circle-stroke-color", p.cursor);
  }

  function setData(id, data) {
    const source = map.getSource(id);
    if (source) source.setData(data);
  }

  function drawAll() {
    if (!ensureLayers()) return;
    const plan = state.overview && state.overview.plan;
    setData(SRC.footprints, (plan && plan.footprints) || empty());
    const routeFeatures = [];
    if (form.route && form.route.length >= 2) {
      routeFeatures.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: form.route } });
    }
    for (const ring of form.regions) {
      routeFeatures.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [...ring, ring[0]] } });
    }
    setData(SRC.route, { type: "FeatureCollection", features: routeFeatures });
    setData(SRC.facade, { type: "FeatureCollection", features: state.facade.map((s) => ({
      type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [s.start_lonlat, s.end_lonlat] } })) });
    drawEditor();
    drawCursor();
    placeEnds();
  }

  function placeEnds() {
    const rows = state.rows;
    if (!markers.length) return;
    if (!rows.length) { markers.forEach((m) => m.remove()); return; }
    const first = rows[0];
    const last = rows[rows.length - 1];
    startTag.textContent = `START · WP ${pad(first.index)}`;
    endTag.textContent = `END · WP ${pad(last.index)}`;
    markers[0].setLngLat([first.lon, first.lat]).addTo(map);
    markers[1].setLngLat([last.lon, last.lat]).addTo(map);
  }

  /* --------------------------------------------------------------- card */

  function rowAt(index) {
    return state.rows.find((r) => r.index === index) || null;
  }

  function showCard(row, pinned) {
    if (!row) { card.hidden = true; return; }
    const plan = state.overview.plan;
    const lines = plan.line_count || null;
    const fp = row.footprint_m;
    const kv = (k, v) => [el("span", { class: "odk-plan-card-k", text: k }), el("span", { class: "odk-plan-card-v", text: v })];
    card.replaceChildren(
      el("div", { class: "odk-plan-card-head" }, [
        el("span", { class: "odk-plan-card-id", text: `WP ${pad(row.index)}` }),
        el("span", { class: "odk-plan-card-line", text: row.line && lines ? `LINE ${row.line} / ${lines}` : "turn" }),
      ]),
      el("div", { class: "odk-plan-card-grid" }, [
        ...kv("LAT", fmtLat(row.lat)),
        ...kv("LON", fmtLon(row.lon)),
        ...kv("ALT", row.alt_m == null ? "not reported" : `${fmtNum(row.alt_m, 1)} m ${frameWords(plan.altitude_frame)}`),
        ...kv("YAW · GIMBAL", `${row.yaw_deg == null ? "—" : `${Math.round(row.yaw_deg)}°`} · ${row.gimbal_deg == null ? "—" : `${Math.round(row.gimbal_deg)}°`}`),
        ...kv("FOOTPRINT", fp ? `${fmtNum(fp[0], 1)} × ${fmtNum(fp[1], 1)} m` : "not drawn"),
        ...kv("ALONG ROUTE", `${fmtNum(row.along_m, 0)} m`),
        ...(row.ground_m != null ? kv("GROUND", `${fmtNum(row.ground_m, 1)} m`) : []),
        ...(row.trigger ? [] : kv("PHOTO", "no trigger")),
      ]),
    );
    card.toggleAttribute("data-pinned", Boolean(pinned));
    card.hidden = false;
    const at = map.project([row.lon, row.lat]);
    const w = host.clientWidth; const h = host.clientHeight;
    const cw = card.offsetWidth || 216; const ch = card.offsetHeight || 140;
    let left = at.x + 14; let top = at.y + 14;
    if (left + cw > w - 8) left = at.x - cw - 14;
    if (top + ch > h - 8) top = at.y - ch - 14;
    card.style.left = `${Math.max(8, left)}px`;
    card.style.top = `${Math.max(8, top)}px`;
    if (map.getLayer(`${SRC.footprints}-hl`)) map.setFilter(`${SRC.footprints}-hl`, ["==", ["get", "index"], row.index]);
  }

  function drawCursor() {
    const row = state.selected != null ? rowAt(state.selected) : null;
    setData(SRC.cursor, row ? { type: "FeatureCollection", features: [point([row.lon, row.lat])] } : empty());
    if (row) showCard(row, true);
    else if (state.hover == null) {
      card.hidden = true;
      if (map.getLayer(`${SRC.footprints}-hl`)) map.setFilter(`${SRC.footprints}-hl`, ["==", ["get", "index"], -1]);
    }
  }

  const onMove = (event) => {
    if (state.editing || state.drag || drawingState().mode) return;
    const hit = map.getLayer("odk-p-mission-pt")
      ? map.queryRenderedFeatures(event.point, { layers: ["odk-p-mission-pt"] }) : [];
    const feature = hit.find((f) => (f.properties || {}).role === "waypoint");
    const index = feature ? Number(feature.properties.index) : null;
    map.getCanvas().style.cursor = feature ? "pointer" : "";
    if (index === state.hover) return;
    state.hover = index;
    if (index != null) showCard(rowAt(index), false);
    else drawCursor();
  };
  const onClick = (event) => {
    if (state.editing || drawingState().mode) return;
    const hit = map.getLayer("odk-p-mission-pt")
      ? map.queryRenderedFeatures(event.point, { layers: ["odk-p-mission-pt"] }) : [];
    const feature = hit.find((f) => (f.properties || {}).role === "waypoint");
    api.select(feature ? Number(feature.properties.index) : null);
  };
  map.on("mousemove", onMove);
  map.on("click", onClick);
  const onView = () => { if (!card.hidden) showCard(rowAt(state.selected ?? state.hover), state.selected != null); };
  map.on("move", onView);

  /* ------------------------------------------------------- vertex editor */

  function drawEditor() {
    if (!state.editing || state.ring.length < 3) { setData(SRC.edit, empty()); return; }
    const ring = state.ring;
    const features = [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [...ring, ring[0]] } }];
    ring.forEach((c, i) => {
      const next = ring[(i + 1) % ring.length];
      features.push(point([(c[0] + next[0]) / 2, (c[1] + next[1]) / 2], { role: "mid", i }));
    });
    ring.forEach((c, i) => features.push(point(c, { role: i === state.picked ? "picked" : "vertex", i })));
    setData(SRC.edit, { type: "FeatureCollection", features });
  }

  function editorHit(pointPx) {
    const layers = [`${SRC.edit}-picked`, `${SRC.edit}-vertex`, `${SRC.edit}-mid`].filter((id) => map.getLayer(id));
    const hit = map.queryRenderedFeatures([[pointPx.x - 6, pointPx.y - 6], [pointPx.x + 6, pointPx.y + 6]], { layers });
    const vertex = hit.find((f) => f.properties.role !== "mid");
    return vertex || hit[0] || null;
  }

  async function commitRing(said) {
    const before = state.overview && state.overview.aoi ? state.overview.aoi.source : "";
    const source = /^edited/.test(before || "") ? before : `edited${before ? ` · ${before}` : ""}`;
    const result = await tryCall("set_aoi", state.ring, source);
    if (!result) {
      notify(`Edit vertices: ${lastError.get("set_aoi") || "the area was refused."} The previous area is kept.`, "warn");
      state.ring = ((state.overview && state.overview.aoi && state.overview.aoi.polygon) || []).map((c) => [...c]);
      drawEditor();
      return;
    }
    if (state.overview && state.overview.plan) form.dirty = true;
    notify(`${said} The area is ${result.vertices} vertices; Generate re-plans over it.`, "ok");
    refreshLive();
  }

  const onDown = (event) => {
    if (!state.editing) return;
    const hit = editorHit(event.point);
    if (!hit) { state.picked = null; drawEditor(); return; }
    event.preventDefault();
    const i = Number(hit.properties.i);
    if (hit.properties.role === "mid") {
      state.ring.splice(i + 1, 0, [event.lngLat.lng, event.lngLat.lat]);
      state.drag = { index: i + 1, added: true };
      state.picked = i + 1;
    } else {
      state.drag = { index: i, added: false };
      state.picked = i;
    }
    map.dragPan.disable();
    drawEditor();
  };
  const onDrag = (event) => {
    if (!state.drag) {
      if (state.editing) map.getCanvas().style.cursor = editorHit(event.point) ? "grab" : "";
      return;
    }
    state.drag.moved = true;
    state.ring[state.drag.index] = [event.lngLat.lng, event.lngLat.lat];
    drawEditor();
  };
  const onUp = () => {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    map.dragPan.enable();
    if (drag.moved || drag.added) commitRing(drag.added ? "Vertex added." : "Vertex moved.");
  };
  map.on("mousedown", onDown);
  map.on("mousemove", onDrag);
  map.on("mouseup", onUp);

  const onKey = (event) => {
    if (!state.editing) return;
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes((document.activeElement || {}).tagName);
    if (typing) return;
    if ((event.key === "Delete" || event.key === "Backspace") && state.picked != null) {
      event.preventDefault();
      if (state.ring.length <= 3) {
        notify("Edit vertices: an area needs at least three corners.", "warn");
        return;
      }
      state.ring.splice(state.picked, 1);
      state.picked = null;
      drawEditor();
      commitRing("Vertex removed.");
    } else if (event.key === "Escape") {
      api.editVertices(false);
    }
  };
  document.addEventListener("keydown", onKey);

  /* ---------------------------------------------------------- controller */

  const api = {
    get state() { return state; },
    async update(overview) {
      state.overview = overview;
      state.rows = (overview && overview.plan && overview.plan.waypoints) || [];
      if (state.selected != null && !rowAt(state.selected)) state.selected = null;
      const key = geometryKey(overview);
      if (state.key !== null && key !== state.key && mapReady(map)) {
        await showProject(map, tryCall);
      }
      state.key = key;
      if (state.editing && !state.drag) {
        state.ring = ((overview && overview.aoi && overview.aoi.polygon) || []).map((c) => [...c]);
        if (state.ring.length < 3) api.editVertices(false);
      }
      drawAll();
    },
    select(index) {
      state.selected = index;
      drawCursor();
      onSelect(index == null ? null : rowAt(index));
    },
    setFootprints(on) {
      state.footprints = Boolean(on);
      if (ensureLayers()) {
        for (const id of [`${SRC.footprints}-line`, `${SRC.footprints}-hl`]) {
          map.setLayoutProperty(id, "visibility", state.footprints ? "visible" : "none");
        }
      }
    },
    editVertices(on) {
      const ring = (state.overview && state.overview.aoi && state.overview.aoi.polygon) || [];
      state.editing = Boolean(on) && ring.length >= 3;
      state.ring = ring.map((c) => [...c]);
      state.picked = null;
      editHint.hidden = !state.editing;
      editHint.textContent = "Drag a corner · drag a midpoint to add one · Delete removes the picked corner · Esc ends";
      if (state.editing) { card.hidden = true; state.selected = null; drawCursor(); }
      map.getCanvas().style.cursor = "";
      drawEditor();
      host.dispatchEvent(new CustomEvent("odk:plan-editing", { bubbles: true, detail: { editing: state.editing } }));
      return state.editing;
    },
    showFacade(segments) {
      state.facade = segments || [];
      drawAll();
    },
    play(onStep) {
      api.stop();
      const rows = state.rows;
      if (!rows.length) return Promise.resolve();
      return new Promise((resolve) => {
        const started = performance.now();
        // A fixed preview pace along the route, not the aircraft's speed.
        const total = Math.min(Math.max(rows.length * 40, 3000), 15000);
        const tick = (now) => {
          if (!state.playing) { resolve(); return; }
          const t = Math.min(1, (now - started) / total);
          const row = rows[Math.min(rows.length - 1, Math.floor(t * rows.length))];
          api.select(row.index);
          if (onStep) onStep(row);
          if (t >= 1) { state.playing = null; resolve(); return; }
          state.playing = requestAnimationFrame(tick);
        };
        state.playing = requestAnimationFrame(tick);
      });
    },
    stop() {
      if (state.playing) cancelAnimationFrame(state.playing);
      state.playing = null;
    },
    repaint,
    destroy() {
      api.stop();
      map.off("mousemove", onMove);
      map.off("click", onClick);
      map.off("move", onView);
      map.off("mousedown", onDown);
      map.off("mousemove", onDrag);
      map.off("mouseup", onUp);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("odk:mode", onMode);
      host.removeEventListener("odk:project", onProject);
      markers.forEach((m) => m.remove());
      card.remove();
      editHint.remove();
    },
  };

  const onMode = () => setTimeout(repaint, 0);
  // The project is drawn (and redrawn) after these layers exist; keep ours on top.
  const onProject = () => drawAll();
  host.addEventListener("odk:project", onProject);
  window.addEventListener("odk:mode", onMode);
  if (mapReady(map)) setTimeout(drawAll, 0); else map.once("load", () => setTimeout(drawAll, 0));
  return api;
}

/**
 * A hidden live() reader of mission_overview for a map: re-reads whenever the project
 * changes (the shell's refresh) and hands the answer to `onData`. Returns the node to
 * keep in the document (live() stops when it leaves).
 */
export function overviewFeed(onData) {
  const node = live({
    calls: ["mission_overview"],
    empty: "The mission could not be read.",
    render: ([overview]) => {
      try { onData(overview); } catch (error) { console.error("plan map:", error); window.__odkPlanMapError = String(error && error.stack || error); }
      return el("span");
    },
  });
  node.hidden = true;
  node.classList.add("odk-plan-feed");
  return node;
}

