/* Plan: desk renderer (SysPlan artboard).

     left       MISSION TEMPLATES grouped Area / Structure / Capture ("Show n more"),
                AREA OF INTEREST (area, vertices, where it came from; Edit vertices,
                Import KML, and the area's other verbs)
     main       the mission bar (name, type, versions; map tools; footprints and ortho
                toggles; undo / redo), the project map with the plan's own layers
                (./map.js), and the strip: WAYPOINTS / DISTANCE / FLIGHT TIME and the
                altitude profile, which scrubs and plays the route
     inspector  Plan: PARAMETERS (camera, two-way GSD / altitude...), Advanced, the
                type's own inputs, SAFETY, EXPORT, Generate and Send to Fly;
                Segments: battery sorties and estimates; Site: markers and no-fly zones

   Every panel is a live() read of the Api (model.js CALLS) and re-reads when an action
   changes the project. Every verb button carries data-action and runs through the
   shell; the rest are controls (data-control). No figure is written here: each comes
   from an Api answer, or the panel says it is not reported. */

import {
  button, checkRow, el, iconButton, panelTabs, popover, section, select, stat, status,
} from "../../components.js";
import { call, tryCall } from "../../../workspace/api.js";
import { live } from "../../../workspace/live.js";
import { mountProjectMap } from "../../map/furniture.js";
import {
  ADVANCED, ALTITUDE_MODES, EXPORT_FORMATS, GROUP_LABELS, PARAMETERS, REGION_TEMPLATES, ROUTE_TEMPLATES,
  TEMPLATE_INPUTS, currentTemplate, exportChoice, fieldValue, fmtBytes, fmtInt, fmtNum, fmtStamp, form,
  frameWords, latestExport, missionOptions, setValue, templateLabel, templateList,
} from "./model.js";
import { canRedo, canUndo, observe } from "./history.js";
import { attachPlanMap, overviewFeed } from "./map.js";

export const SCREEN = "plan";

/* Template icons, drawn from the SysPlan artboard (stroke, 24 px box). */
const TEMPLATE_ICONS = {
  grid: '<path d="M5 4v16M10 4v16M15 4v16M20 4v16"></path><path d="M5 4h5M10 20h5M15 4h5"></path>',
  double_grid: '<path d="M4 6h16M4 12h16M4 18h16M6 4v16M12 4v16M18 4v16"></path>',
  corridor: '<path d="M3 18C8 17 9 7 21 6"></path><path d="M3 13C7 12 8 3 18 3M6 21C11 20 12 10 21 10" stroke-dasharray="2 2"></path>',
  smart_adaptive: '<path d="M4 20l5-8 4 4 7-12"></path><circle cx="20" cy="4" r="1.5"></circle>',
  solar_inspection: '<path d="M4 9h16l-2 10H6z"></path><path d="M9 9l-1 10M15 9l1 10M5 14h14"></path><path d="M12 2v3"></path>',
  magnetic_mapping: '<path d="M6 3v9a6 6 0 0 0 12 0V3"></path><path d="M6 7h4M14 7h4"></path>',
  facade: '<path d="M5 21V4h10v17M15 9h4v12M3 21h18"></path><path d="M8 8h1M11 8h1M8 12h1M11 12h1M8 16h1M11 16h1"></path>',
  facade_mapping: '<path d="M5 21V4h10v17M15 9h4v12M3 21h18"></path>',
  multi_facade: '<path d="M3 21V8h7v13M10 21V4h7v17M17 21v-9h4v9"></path>',
  tower_mapping: '<path d="M12 2v20M8 22l4-16 4 16M7 10h10M9 16h6"></path>',
  orbit: '<circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="2"></circle><path d="M20 12l-2-2M20 12l2-2"></path>',
  roof_inspection: '<path d="M3 12l9-7 9 7M6 10v10h12V10"></path>',
  wind_turbine: '<path d="M12 11V22M12 11L6 5M12 11l8-2M12 11l-2 7"></path>',
  pylon_inspection: '<path d="M8 22l4-20 4 20M5 7h14M6 12h12"></path>',
  panorama: '<circle cx="12" cy="12" r="9"></circle><ellipse cx="12" cy="12" rx="9" ry="3.5"></ellipse>',
  bubble_360: '<circle cx="12" cy="12" r="9"></circle><ellipse cx="12" cy="12" rx="3.5" ry="9"></ellipse><path d="M3 12h18"></path>',
  waypoints: '<circle cx="5" cy="18" r="2"></circle><circle cx="12" cy="7" r="2"></circle><circle cx="19" cy="16" r="2"></circle><path d="M6.5 16.5l4-7.5M13.5 8.5l4 6"></path>',
  linear_inspection: '<path d="M3 12h18"></path><path d="M6 9v6M12 9v6M18 9v6"></path>',
  lateral_capture: '<path d="M3 16h18"></path><path d="M6 16V8M12 16V8M18 16V8"></path>',
  linked_mission: '<circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="12" r="3"></circle><path d="M9 12h6"></path>',
  closed_loop: '<path d="M5 8a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4z"></path>',
  dome_inspection: '<path d="M4 18a8 8 0 0 1 16 0M3 18h18"></path>',
  box_inspection: '<path d="M4 8l8-4 8 4v8l-8 4-8-4z"></path><path d="M4 8l8 4 8-4M12 12v8"></path>',
  thermal: '<path d="M10 14V4a2 2 0 0 1 4 0v10a4 4 0 1 1-4 0z"></path>',
  multispectral: '<path d="M4 18l4-12 4 12M12 18l4-12 4 12"></path>',
};

/* Map tool icons (SysPlan). */
const TOOL_ICONS = {
  select: '<path d="M5 3l14 8-6 2-2 6z"></path>',
  area: '<path d="M4 18L7 5l12 3-3 12z"></path><circle cx="4" cy="18" r="1.5"></circle><circle cx="7" cy="5" r="1.5"></circle>',
  vertices: '<rect x="3" y="3" width="5" height="5"></rect><rect x="16" y="3" width="5" height="5"></rect><rect x="9.5" y="16" width="5" height="5"></rect><path d="M8 5.5h8M6 8l5 8M18 8l-5 8"></path>',
  rotate: '<path d="M20 12a8 8 0 1 1-2.3-5.7"></path><path d="M20 4v5h-5"></path>',
  measure: '<path d="M3 17L17 3l4 4L7 21z"></path><path d="M7 13l2 2M10 10l2 2M13 7l2 2"></path>',
  more: '<circle cx="5" cy="12" r="1.5"></circle><circle cx="12" cy="12" r="1.5"></circle><circle cx="19" cy="12" r="1.5"></circle>',
  undo: '<path d="M9 14L4 9l5-5"></path><path d="M4 9h11a5 5 0 0 1 0 10h-3"></path>',
  redo: '<path d="M15 14l5-5-5-5"></path><path d="M20 9H9a5 5 0 0 0 0 10h3"></path>',
  play: '<path d="M7 5l12 7-12 7z"></path>',
  stop: '<rect x="6" y="6" width="12" height="12"></rect>',
  chevron: '<path d="M6 9l6 6 6-6"></path>',
};

function svgIcon(paths, size = 16) {
  const span = document.createElement("span");
  span.className = "odk-plan-ico";
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  return span;
}

/* Screen CSS lives beside the screen (docs/ui/ARCHITECTURE.md); linked once, and
   awaited, because the map measures its container when it is made: a map made before
   this sheet applies is sized for a collapsed box and fitted to the wrong view. */
function ensureStyles() {
  const existing = document.querySelector("link[data-odk-screen-css=plan]");
  if (existing) return existing._loaded || Promise.resolve();
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./plan.css", import.meta.url).href;
  link.setAttribute("data-odk-screen-css", "plan");
  link._loaded = new Promise((done) => {
    link.addEventListener("load", () => done(), { once: true });
    link.addEventListener("error", () => done(), { once: true });
  });
  document.head.appendChild(link);
  return link._loaded;
}

/* ------------------------------------------------------------ verbs */

let shellRef = null;

/** A verb button: data-action, run through the shell with an optional payload. */
function act(key, label, { variant = "secondary", payload = null, disabled = false, title, iconName, className } = {}) {
  const node = button({
    label, action: key, variant, disabled, title, iconName,
    onClick: () => { if (shellRef) shellRef.runAction(key, typeof payload === "function" ? payload() : payload); },
  });
  if (className) node.classList.add(className);
  return node;
}

/** An icon verb button for the tool bar. */
function toolButton(key, label, iconPaths, { pressed } = {}) {
  const node = el("button", { type: "button", class: "odk-plan-tool", "data-action": key, "aria-label": label, title: label });
  node.appendChild(svgIcon(iconPaths));
  if (pressed !== undefined) node.setAttribute("aria-pressed", pressed ? "true" : "false");
  node.addEventListener("click", () => { if (shellRef) shellRef.runAction(key); });
  return node;
}

/** A menu of verbs on a small button; items: [{key, label, payload, hint}]. */
/* One popover per menu: a panel re-read by live() builds its menus again, and the old
   popover (mounted on the page, not in the panel) goes with the old anchor. */
const POPS = new Map();
function ownPopover(name, spec) {
  const previous = POPS.get(name);
  if (previous) { previous.close(); previous.node.remove(); }
  const pop = popover(spec);
  pop.node.setAttribute("data-plan-menu", name);
  POPS.set(name, pop);
  return pop;
}

function verbMenu(anchor, title, items) {
  const content = el("div", { class: "odk-plan-menu" });
  const pop = ownPopover(title, { anchor, title, content });
  for (const item of items) {
    if (item.rule) { content.appendChild(el("div", { class: "odk-plan-menu-rule", role: "separator" })); continue; }
    const row = el("button", { type: "button", class: "odk-menu-item", "data-action": item.key }, [
      el("span", { class: "odk-plan-menu-label", text: item.label }),
      item.hint ? el("span", { class: "odk-plan-menu-hint", text: item.hint }) : null,
    ]);
    row.addEventListener("click", () => {
      pop.close();
      if (shellRef) shellRef.runAction(item.key, item.payload || null);
    });
    content.appendChild(row);
  }
  return pop;
}

/* ------------------------------------------------------- screen state */

const view = { overview: null, footprints: true, mapCtl: null, showAll: false, tab: "plan", lastPlanKey: null };
const hosts = {};

function rerender(...names) {
  for (const name of names) if (hosts[name] && hosts[name]._settle) hosts[name]._settle();
}

/** A plan's identity: a new one (planned, loaded) settles the form's "changed" flag. */
function planKey(plan) {
  return plan ? JSON.stringify([plan.template, plan.waypoint_count, plan.distance_m, plan.altitude_m,
    plan.line_bearing_deg, plan.camera, plan.front_overlap_pct, plan.side_overlap_pct, plan.speed_m_s]) : null;
}

function takeOverview(overview) {
  if (!overview) return;
  view.overview = overview;
  observe(overview.aoi);
  const key = planKey(overview.plan);
  if (key && view.lastPlanKey && key !== view.lastPlanKey) form.dirty = false;
  if (key) view.lastPlanKey = key;
}

/* =============================================================== left */

function templatesPanel() {
  return live({
    calls: ["mission_templates", "list_mission_versions", "mission_overview"],
    empty: "Mission types could not be read from the planner.",
    isEmpty: ([templates]) => !templates,
    render: ([templates, versions, overview]) => {
      takeOverview(overview);
      const plan = overview && overview.plan;
      const rows = templateList(templates, versions, plan);
      const active = currentTemplate(plan);
      const hidden = rows.filter((r) => !r.primary && r.key !== active).length;
      const list = el("div", { class: "odk-plan-templates" });
      list.appendChild(el("div", { class: "odk-plan-panel-head" }, [
        el("span", { class: "odk-label", text: "MISSION TEMPLATES" }),
        el("span", { class: "odk-label odk-num", text: String(rows.length) }),
      ]));
      let group = null;
      for (const row of rows) {
        if (!view.showAll && !row.primary && row.key !== active) continue;
        if (row.group !== group) {
          group = row.group;
          list.appendChild(el("div", { class: "odk-plan-group odk-label", text: (GROUP_LABELS[group] || group).toUpperCase() }));
        }
        const on = row.key === active;
        const node = el("button", {
          type: "button", class: "odk-plan-template", "data-control": `template:${row.key}`,
          "data-template": row.key, title: row.hint ? `${row.label} (${row.hint})` : row.label,
          ...(on ? { "aria-current": "true" } : {}),
        }, [
          svgIcon(TEMPLATE_ICONS[row.key] || TEMPLATE_ICONS.waypoints),
          el("span", { class: "odk-plan-template-name", text: row.label }),
          row.waypoints != null ? el("span", { class: "odk-plan-template-wp", title: `Waypoints: ${row.waypointsFrom}`,
            text: `${fmtInt(row.waypoints)} wp` }) : null,
        ]);
        node.addEventListener("click", () => {
          form.template = row.key === (plan && plan.template) && !form.dirty ? null : row.key;
          if (row.key !== (plan && plan.template)) form.dirty = true;
          if (row.key === "multi_facade") form.template = row.key;
          rerender("templates", "inspector", "bar");
        });
        list.appendChild(node);
      }
      if (hidden || view.showAll) {
        const more = el("button", { type: "button", class: "odk-plan-more", "data-control": "templates:more",
          text: view.showAll ? "Show fewer" : `Show ${hidden} more` });
        more.addEventListener("click", () => { view.showAll = !view.showAll; rerender("templates"); });
        list.appendChild(more);
      }
      return list;
    },
  });
}

function aoiPanel() {
  return live({
    calls: ["mission_overview"],
    empty: "The area of interest could not be read.",
    render: ([overview]) => {
      takeOverview(overview);
      const aoi = overview.aoi || {};
      const box = el("div", { class: "odk-plan-aoi" });
      box.appendChild(el("div", { class: "odk-plan-panel-head" }, [
        el("span", { class: "odk-label", text: "AREA OF INTEREST" }),
        aoi.set ? status("ok", "SET") : status("idle", "NOT SET"),
      ]));
      const kv = el("dl", { class: "odk-plan-kv" });
      const pair = (k, v, dim) => kv.append(el("dt", { text: k }), el("dd", { class: dim ? "dim" : "", text: v }));
      if (aoi.set) {
        pair("AREA", `${fmtInt(aoi.area_m2)} m²`);
        pair("VERTICES", String(aoi.vertices));
        pair("SOURCE", aoi.source || "not recorded", true);
      } else {
        kv.append(el("dd", { class: "odk-plan-hint", text: "Draw the area on the map, import a boundary, or outline the camera positions." }));
      }
      if (overview.no_fly_count) pair("NO-FLY", `${overview.no_fly_count} zone(s)`);
      if (overview.boundary_marks) pair("CORNERS", `${overview.boundary_marks} flown`);
      box.appendChild(kv);
      const more = iconButton({ text: "", label: "More area actions", control: "aoi:more" });
      more.appendChild(svgIcon(TOOL_ICONS.more));
      verbMenu(more, "Area of interest", [
        { key: "Draw Area", label: "Draw area" },
        { key: "Area From Photos", label: "Outline the camera positions", hint: "hull" },
        { key: "Facade Footprint", label: "Read as a building footprint…" },
        { rule: true },
        { key: "Mark Corner", label: "Fly the corners: mark corner", hint: "aircraft" },
        { key: "Close Boundary", label: "Close the flown boundary" },
        { key: "Clear Corners", label: "Clear flown corners" },
        { rule: true },
        { key: "Zoom to Area", label: "Fit to area" },
        { key: "Clear Area", label: "Clear area" },
        { key: "Clear No-fly", label: "Clear no-fly zones" },
      ]);
      box.appendChild(el("div", { class: "odk-plan-aoi-actions" }, [
        act("Edit Vertices", "Edit vertices", { disabled: !aoi.set }),
        act("Import Boundary", "Import KML", { title: "KML, KMZ, GeoJSON, GPX or CSV" }),
        more,
      ]));
      return box;
    },
  });
}

/* =============================================================== bar */

function missionBar() {
  const bar = el("div", { class: "odk-plan-bar" });
  hosts.bar = live({
    calls: ["mission_overview", "mission_templates"],
    empty: "No mission read.",
    render: ([overview, templates]) => {
      takeOverview(overview);
      const plan = overview.plan;
      const saved = overview.saved;
      const name = saved ? `${saved.name} · v${saved.version_num}` : plan ? "unsaved plan" : "no mission planned";
      const chosen = currentTemplate(plan);
      const versions = el("button", { type: "button", class: "odk-plan-versions", "data-control": "versions",
        "aria-label": "Mission versions", title: "Versions" }, [svgIcon(TOOL_ICONS.chevron, 14)]);
      verbMenu(versions, "Mission versions", [
        { key: "Save", label: "Save version" },
        { key: "Load Version", label: "Open a saved version…" },
        { key: "History", label: "Restore as newest…" },
        { rule: true },
        { key: "Compare Versions", label: "Compare versions…" },
        { key: "Compare Specifications", label: "Compare specifications…" },
        { key: "Repeat Survey", label: "Repeat survey…" },
      ]);
      return el("div", { class: "odk-plan-bar-id" }, [
        el("span", { class: "odk-label", text: "MISSION" }),
        el("span", { class: `odk-plan-name${saved ? "" : " odk-plan-name--unsaved"}`, text: name }),
        chosen ? el("span", { class: "odk-chip odk-plan-type", text: templateLabel(templates, chosen).toUpperCase() }) : null,
        versions,
      ]);
    },
  });
  const tools = el("div", { class: "odk-plan-tools", role: "toolbar", "aria-label": "Map tools" }, [
    toolButton("Select", "Select", TOOL_ICONS.select),
    toolButton("Draw Area", "Draw area", TOOL_ICONS.area),
    toolButton("Edit Vertices", "Edit vertices", TOOL_ICONS.vertices, { pressed: false }),
    toolButton("Rotate Flight Lines", "Rotate flight lines", TOOL_ICONS.rotate),
    toolButton("Distance", "Measure distance", TOOL_ICONS.measure),
  ]);
  const more = el("button", { type: "button", class: "odk-plan-tool", "data-control": "tools:more",
    "aria-label": "More map tools", title: "More map tools" }, [svgIcon(TOOL_ICONS.more)]);
  verbMenu(more, "Map tools", [
    { key: "Draw No-fly", label: "No-fly zone" },
    { key: "Draw Route", label: "Route line", hint: "linear, waypoint, solar, facade" },
    { key: "Draw Interest Region", label: "Interest region", hint: "smart adaptive" },
    { key: "Clear Route", label: "Clear route and regions" },
    { rule: true },
    { key: "Place Home", label: "Place home" },
    { key: "Add Marker", label: "Site marker…" },
    { key: "Import CAD", label: "CAD or image overlay…" },
    { rule: true },
    { key: "Area", label: "Measure area" },
    { key: "Zoom to Mission", label: "Fit to mission" },
  ]);
  tools.appendChild(more);

  hosts.layers = live({
    calls: ["list_layers"],
    empty: "",
    isEmpty: () => false,
    render: ([listed]) => {
      const layers = (listed && listed.layers) || [];
      const ortho = layers.find((l) => l.kind !== "vector" && /ortho/i.test(l.name || ""));
      const foot = checkRow({ label: "Footprints", checked: view.footprints, onChange: (on) => {
        view.footprints = on;
        if (view.mapCtl) view.mapCtl.setFootprints(on);
      } });
      foot.setAttribute("data-control", "footprints");
      const row = el("div", { class: "odk-plan-toggles" }, [foot]);
      if (ortho) {
        const orthoRow = checkRow({ label: ortho.name, checked: ortho.visible !== false, onChange: async (on) => {
          await call("set_layer_visible", ortho.id, on);
          if (view.mapCtl) view.mapCtl.state.key = null;
          rerender("layers");
          if (shellRef) shellRef.refreshState();
        } });
        orthoRow.setAttribute("data-control", "layer:ortho");
        orthoRow.title = "Show or hide the orthomosaic (set_layer_visible)";
        row.appendChild(orthoRow);
      }
      return row;
    },
  });

  const undo = toolButton("Undo Area Edit", "Undo area edit", TOOL_ICONS.undo);
  const redo = toolButton("Redo Area Edit", "Redo area edit", TOOL_ICONS.redo);
  undo.classList.add("odk-plan-tool--plain");
  redo.classList.add("odk-plan-tool--plain");
  const syncUndo = () => { undo.disabled = !canUndo(); redo.disabled = !canRedo(); };
  syncUndo();
  hosts.undo = live({ calls: ["mission_overview"], empty: "", isEmpty: () => false,
    render: ([overview]) => { takeOverview(overview); syncUndo(); return el("span"); } });
  hosts.undo.hidden = true;

  bar.append(hosts.bar, el("div", { class: "odk-plan-spacer" }), tools,
    el("span", { class: "odk-plan-sep", "aria-hidden": "true" }), hosts.layers,
    el("span", { class: "odk-plan-sep", "aria-hidden": "true" }), undo, redo, hosts.undo);
  return bar;
}

/* ============================================================== strip */

function strip() {
  return live({
    calls: ["mission_overview"],
    empty: "The mission could not be read.",
    render: ([overview]) => {
      takeOverview(overview);
      const plan = overview.plan;
      const node = el("div", { class: "odk-plan-strip-body" });
      const figure = (label, value, unit, under) => el("div", { class: "odk-plan-figure" }, [
        el("span", { class: "odk-label", text: label }),
        value == null ? el("span", { class: "odk-plan-figure-missing", text: plan ? "not reported" : "no plan" })
          : el("span", { class: "odk-plan-figure-value" }, [value, unit ? el("span", { class: "odk-plan-figure-unit", text: ` ${unit}` }) : null]),
        el("span", { class: "odk-plan-figure-under", text: under || "" }),
      ]);
      const lines = plan ? (plan.line_count ? `${plan.line_count} lines` : `${plan.leg_count} legs`) : "";
      node.appendChild(el("div", { class: "odk-plan-figures" }, [
        figure("WAYPOINTS", plan ? fmtInt(plan.waypoint_count) : null, "", plan ? `${fmtInt(plan.trigger_count)} triggers` : ""),
        figure("DISTANCE", plan ? fmtInt(plan.distance_m) : null, "m", lines),
        figure("FLIGHT TIME", plan ? fmtNum(plan.duration_min, 1) : null, "min",
          plan && plan.cruise_speed_m_s != null ? `at ${fmtNum(plan.cruise_speed_m_s, 1)} m/s` : ""),
      ]));
      node.appendChild(profile(plan));
      return node;
    },
  });
}

const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (name, attrs = {}) => {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
};

/** The altitude profile: altitude (and the ground, when sampled) along the route. */
function profile(plan) {
  const box = el("div", { class: "odk-plan-profile" });
  const terrain = plan && plan.terrain;
  const terrainText = !plan ? "" : terrain && terrain.sampled ? `TERRAIN ${terrain.source_path.split(/[\\/]/).pop()}`
    : terrain && terrain.note ? "TERRAIN not sampled" : "TERRAIN not loaded";
  const playing = Boolean(view.mapCtl && view.mapCtl.state.playing);
  const play = el("button", { type: "button", class: "odk-plan-play", "data-action": "Play Route",
    "aria-label": playing ? "Stop" : "Play the route", title: "Play the route on the map" }, [svgIcon(playing ? TOOL_ICONS.stop : TOOL_ICONS.play, 14)]);
  play.disabled = !plan || !(plan.waypoints || []).length;
  play.addEventListener("click", () => {
    if (view.mapCtl && view.mapCtl.state.playing) { view.mapCtl.stop(); rerender("strip"); return; }
    if (shellRef) shellRef.runAction("Play Route");
  });
  box.appendChild(el("div", { class: "odk-plan-profile-head" }, [
    el("span", { class: "odk-label", text: `ALTITUDE PROFILE${plan ? ` · ${frameWords(plan.altitude_frame).toUpperCase()}` : ""}` }),
    el("span", { class: `odk-plan-terrain${terrain && terrain.sampled ? "" : " odk-plan-terrain--warn"}`,
      title: (terrain && terrain.note) || "", text: terrainText }),
    play,
  ]));
  const rows = (plan && plan.waypoints) || [];
  if (rows.length < 2) {
    box.appendChild(el("div", { class: "odk-plan-profile-empty", text: plan ? "The plan has too few waypoints for a profile." : "Generate a plan to see its altitude along the route." }));
    return box;
  }
  const W = 1000; const H = 64;
  const length = rows[rows.length - 1].along_m || 1;
  const alts = rows.map((r) => r.alt_m).filter((v) => v != null);
  const grounds = rows.map((r) => r.ground_m).filter((v) => v != null);
  const relative = plan.altitude_frame === "relative_to_home" && terrain && terrain.home_ground_m != null;
  const groundRel = (g) => (relative ? g - terrain.home_ground_m : g);
  const values = [...alts, ...(relative || plan.altitude_frame === "amsl" ? grounds.map(groundRel) : [])];
  const lo = Math.min(...values, 0); const hi = Math.max(...values);
  const span = Math.max(hi - lo, 1);
  const x = (d) => (d / length) * W;
  const y = (v) => H - 4 - ((v - lo) / span) * (H - 12);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", class: "odk-plan-profile-svg",
    role: "img", "aria-label": "Altitude along the route" });
  if (grounds.length && (relative || plan.altitude_frame === "amsl")) {
    const pts = rows.filter((r) => r.ground_m != null).map((r) => `${x(r.along_m)},${y(groundRel(r.ground_m))}`);
    svg.appendChild(svgEl("polygon", { class: "odk-plan-profile-ground",
      points: `${x(rows[0].along_m)},${H} ${pts.join(" ")} ${x(length)},${H}` }));
  }
  svg.appendChild(svgEl("polyline", { class: "odk-plan-profile-alt",
    points: rows.filter((r) => r.alt_m != null).map((r) => `${x(r.along_m)},${y(r.alt_m)}`).join(" ") }));
  for (const r of rows) {
    if (!r.trigger) continue;
    svg.appendChild(svgEl("line", { class: "odk-plan-profile-tick", x1: x(r.along_m), x2: x(r.along_m), y1: H - 3, y2: H }));
  }
  const cursor = svgEl("line", { class: "odk-plan-profile-cursor", x1: -10, x2: -10, y1: 0, y2: H });
  svg.appendChild(cursor);
  const chart = el("div", { class: "odk-plan-profile-chart" });
  const label = el("span", { class: "odk-plan-profile-label", hidden: "" });
  chart.append(svg, label);
  const axis = el("div", { class: "odk-plan-profile-axis" });
  const steps = [0, 0.25, 0.5, 0.75, 1];
  steps.forEach((t) => axis.appendChild(el("span", { text: t === 1 ? `${fmtInt(length * t)} m` : fmtInt(length * t) })));
  const altLabel = el("span", { class: "odk-plan-profile-alt-label", text: `${fmtNum(hi, 1)} m` });
  chart.appendChild(altLabel);
  box.append(chart, axis);

  const moveTo = (row) => {
    if (!row) { cursor.setAttribute("x1", "-10"); cursor.setAttribute("x2", "-10"); label.hidden = true; return; }
    const px = x(row.along_m);
    cursor.setAttribute("x1", String(px)); cursor.setAttribute("x2", String(px));
    label.hidden = false;
    label.textContent = `WP ${String(row.index).padStart(3, "0")} · ${fmtInt(row.along_m)} m`;
    label.style.left = `${Math.min(88, (px / W) * 100)}%`;
  };
  const nearest = (clientX) => {
    const r = svg.getBoundingClientRect();
    const d = ((clientX - r.left) / Math.max(r.width, 1)) * length;
    let best = rows[0];
    for (const row of rows) if (Math.abs(row.along_m - d) < Math.abs(best.along_m - d)) best = row;
    return best;
  };
  chart.addEventListener("mousemove", (event) => {
    const row = nearest(event.clientX);
    moveTo(row);
    if (view.mapCtl && !view.mapCtl.state.playing) view.mapCtl.select(row.index);
  });
  chart.addEventListener("mouseleave", () => {
    if (view.mapCtl && view.mapCtl.state.playing) return;
    moveTo(null);
    if (view.mapCtl) view.mapCtl.select(null);
  });
  chart.setAttribute("data-control", "profile-scrub");
  box._moveTo = moveTo;
  view.profileMove = moveTo;
  return box;
}

/* ========================================================== inspector */

function numberField(field, plan, { readonly = false, onCommit, note } = {}) {
  const id = `plan-${field.key}`;
  const value = fieldValue(field, plan);
  const input = el("input", { id, inputmode: "decimal", autocomplete: "off", "data-field": field.key,
    ...(readonly ? { readonly: "" } : {}), placeholder: readonly ? "—" : "default" });
  input.value = value;
  const box = el("div", { class: `odk-input odk-input--mono odk-input--num${readonly ? " odk-input--readonly" : ""}` },
    [input, field.unit ? el("span", { class: "odk-input-unit", text: field.unit }) : null]);
  if (!readonly) {
    input.addEventListener("input", () => setValue(field.key, input.value));
    input.addEventListener("change", () => { if (onCommit) onCommit(input.value, input); markDirty(); });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); input.blur(); if (shellRef) shellRef.runAction("Plan"); }
    });
  }
  const wrap = el("div", { class: "odk-field" }, [el("label", { for: id, text: field.label }), box]);
  if (note) wrap.appendChild(el("span", { class: "odk-plan-field-note", text: note }));
  return wrap;
}

function textField(field) {
  const id = `plan-${field.key}`;
  if (field.kind === "yesno") {
    return select({ label: field.label, value: form.values[field.key] || "", mono: false,
      options: [{ value: "", label: "Planner default" }, { value: "yes", label: "Yes" }, { value: "no", label: "No" }],
      onChange: (v) => { setValue(field.key, v); markDirty(); } });
  }
  const input = el("input", { id, autocomplete: "off", "data-field": field.key, placeholder: field.required ? "required" : "optional" });
  input.value = form.values[field.key] || "";
  input.addEventListener("input", () => setValue(field.key, input.value));
  input.addEventListener("change", markDirty);
  return el("div", { class: "odk-field odk-plan-wide" }, [el("label", { for: id, text: field.label }),
    el("div", { class: "odk-input" }, [input])]);
}

function markDirty() {
  const node = document.querySelector("[data-plan-status]");
  if (node) {
    node.replaceChildren(status("warn", "Changed since this plan: Generate to apply"));
  }
}

function cameraPicker(cameras, plan, payloads) {
  const current = form.camera || (plan && plan.camera) || null;
  const known = (cameras || []).find((c) => c.key === current);
  const trigger = el("button", { type: "button", class: "odk-plan-picker", "data-control": "camera-picker",
    "aria-label": `Camera: ${current || "planner default"}` }, [
    el("span", { class: "odk-plan-picker-value", text: current || "planner default" }),
    form.payload ? el("span", { class: "odk-plan-picker-sub", text: `+ ${form.payload}` }) : null,
    svgIcon(TOOL_ICONS.chevron, 12),
  ]);
  const content = el("div", { class: "odk-plan-cams" });
  const pop = ownPopover("Camera and payload", { anchor: trigger, title: "Camera and payload", content });
  const tabs = panelTabs([{ id: "camera", label: "Camera" }, { id: "payload", label: "Payload" }], {
    active: "camera", label: "Camera or payload", onSelect: (id) => { camList.hidden = id !== "camera"; payList.hidden = id !== "payload"; },
  });
  const camList = el("div", { class: "odk-plan-cam-list" });
  for (const cam of cameras || []) {
    const row = el("button", { type: "button", class: "odk-menu-item", "data-control": `camera:${cam.key}`,
      ...(cam.key === current ? { "aria-current": "true" } : {}) }, [
      el("span", { class: "odk-plan-menu-label", text: cam.name }),
      el("span", { class: "odk-plan-menu-hint", text: `${cam.key}${cam.thermal ? " · thermal" : ""}` }),
    ]);
    row.addEventListener("click", () => {
      form.camera = cam.key;
      form.dirty = true;
      pop.close();
      rerender("inspector");
    });
    camList.appendChild(row);
  }
  const camVerbs = el("div", { class: "odk-plan-cam-verbs" }, [
    act("Describe Camera", "Describe", { payload: () => ({ camera: current, altitude_m: Number(fieldValue(PARAMETERS[0], plan)) || undefined }), disabled: !current }),
    act("Add Camera", "Add camera…"),
    act("Camera Capabilities", "Connected camera", { title: "What the connected aircraft's camera reports" }),
    act("Cameras", "All at an altitude…", { variant: "ghost" }),
  ]);
  camList.appendChild(camVerbs);
  const payList = el("div", { class: "odk-plan-cam-list", hidden: "" });
  const none = el("button", { type: "button", class: "odk-menu-item", "data-control": "payload:none",
    ...(!form.payload ? { "aria-current": "true" } : {}) }, [el("span", { class: "odk-plan-menu-label", text: "No payload beyond the camera" })]);
  none.addEventListener("click", () => { form.payload = null; form.dirty = true; pop.close(); rerender("inspector"); });
  payList.appendChild(none);
  for (const p of payloads || []) {
    const key = p.key || p.name;
    const row = el("button", { type: "button", class: "odk-menu-item", "data-control": `payload:${key}`,
      ...(form.payload === key ? { "aria-current": "true" } : {}) }, [
      el("span", { class: "odk-plan-menu-label", text: p.name || key }),
      el("span", { class: "odk-plan-menu-hint", text: p.kind || "" }),
    ]);
    row.addEventListener("click", () => { form.payload = key; form.dirty = true; pop.close(); rerender("inspector"); });
    payList.appendChild(row);
  }
  payList.appendChild(el("div", { class: "odk-plan-cam-verbs" }, [act("Payloads", "Describe…"), act("Add Payload", "Add payload…")]));
  content.append(tabs, camList, payList);
  return el("div", { class: "odk-field odk-plan-wide" }, [
    el("span", { class: "odk-plan-field-label", text: "Camera" }), trigger,
    known && known.source !== "manufacturer" ? el("span", { class: "odk-plan-field-note", text: `profile: ${known.source}` }) : null,
  ]);
}

function safetyRows(overview) {
  const plan = overview.plan;
  const adj = (plan && plan.safety_adjustments) || {};
  const rows = el("div", { class: "odk-plan-rows" });
  const row = (label, value, tone, extra) => rows.appendChild(el("div", { class: "odk-plan-row" }, [
    el("span", { class: "odk-plan-row-label", text: label }),
    el("span", { class: "odk-plan-row-value" }, [tone ? el("span", { class: `odk-dot odk-dot--${tone}`, "aria-hidden": "true" }) : null, value]),
    extra || null,
  ]));
  if (plan) {
    row("Geofence", `inside AOI · ${fmtInt(adj.geofence_projections ?? 0)} wp projected`, "ok");
    row("No-fly zones", `${overview.no_fly_count} zone(s) · ${fmtInt(adj.no_fly_projections ?? 0)} projections`,
      adj.no_fly_projections ? "warn" : "ok");
    row("Obstacle detours", fmtInt(adj.obstacle_detours ?? 0));
    row("Altitude clamps", fmtInt(adj.altitude_clamps ?? 0), adj.altitude_clamps ? "warn" : null);
    const t = plan.terrain || {};
    const words = t.sampled ? `${t.model_type} · sampled` : t.follow ? `${t.model_type} · ${t.model_source}`
      : `none · ${t.model_type === "flat" ? "flat plane" : t.model_type}`;
    row("Terrain model", words, t.follow || t.sampled ? "ok" : "warn");
  } else {
    row("Geofence", "no plan yet");
    row("No-fly zones", `${overview.no_fly_count} zone(s)`);
    row("Terrain model", overview.terrain_source ? overview.terrain_source.split(/[\\/]/).pop() : "none loaded",
      overview.terrain_source ? "ok" : "warn");
  }
  rows.appendChild(el("div", { class: "odk-plan-row-verbs" }, [
    act("Load Terrain", "Load terrain"),
    act("Cache Terrain", "Cache"),
    act("Terrain Details", "Details", { variant: "ghost" }),
    overview.terrain_source ? act("Clear Terrain", "Clear", { variant: "ghost" }) : null,
  ].filter(Boolean)));
  rows.appendChild(el("div", { class: "odk-plan-row-verbs" }, [
    act("Check Hazards", "Obstacle clearance…", { disabled: !plan }),
    act("Validate", "Check site"),
  ]));
  const warnings = overview.warnings || [];
  if (warnings.length) {
    const details = el("details", { class: "odk-plan-notes" }, [
      el("summary", { "data-control": "planner-notes", text: `Planner notes (${warnings.length})` }),
      ...warnings.map((w) => el("p", { text: w })),
    ]);
    rows.appendChild(details);
  }
  return rows;
}

function exportBlock(listedExports, plan) {
  const latest = latestExport(listedExports);
  const box = el("div", { class: "odk-plan-export" });
  for (const f of EXPORT_FORMATS) {
    const made = latest.byFormat[f.key];
    const label = el("label", { class: "odk-plan-export-row", "data-control": `export:${f.key}` });
    const box2 = el("input", { type: "checkbox" });
    box2.checked = exportChoice.has(f.key);
    box2.addEventListener("change", () => {
      if (box2.checked) exportChoice.add(f.key); else exportChoice.delete(f.key);
      const n = exportChoice.size;
      const primary = document.querySelector('[data-action="Export Mission Files"]');
      if (primary) primary.textContent = n === 1 ? "Export 1 format" : `Export ${n} formats`;
      if (primary) primary.disabled = !n || !plan;
    });
    label.append(box2,
      el("span", { class: "odk-plan-export-name", text: f.label }),
      el("span", { class: "odk-plan-export-ext", text: made && made.parts > 1 ? `${f.ext} × ${made.parts}` : f.ext }),
      el("span", { class: "odk-plan-export-size", title: made ? `Written ${fmtStamp(latest.stamp)}` : "Not exported yet",
        text: made ? fmtBytes(made.bytes) : "—" }));
    box.appendChild(label);
  }
  if (latest.stamp) {
    box.appendChild(el("div", { class: "odk-plan-export-last" }, [
      el("span", { text: `Last export ${fmtStamp(latest.stamp)}` }),
      act("Open Export Folder", "Open folder", { variant: "ghost" }),
    ]));
  }
  return box;
}

function inspectorPlan() {
  return live({
    calls: ["mission_overview", "list_cameras", "mission_templates", "list_mission_exports", "list_payloads"],
    empty: "The mission could not be read.",
    isEmpty: ([overview]) => !overview,
    render: ([overview, cameras, templates, exports, payloads]) => {
      takeOverview(overview);
      const plan = overview.plan;
      const template = currentTemplate(plan);
      const sameType = !form.template || !plan || form.template === plan.template;
      const shownPlan = sameType ? plan : null;
      const node = el("div", { class: "odk-plan-inspector" });
      const title = template ? templateLabel(templates, template).toUpperCase() : "MISSION";
      const generate = act("Plan", plan ? "Re-plan" : "Generate", { variant: plan && !form.dirty ? "secondary" : "primary",
        title: "Plan over the area with these parameters (Ctrl P)", className: "odk-plan-generate" });
      generate.disabled = !(overview.aoi && overview.aoi.set) && !["linked_mission"].includes(template);
      node.appendChild(el("div", { class: "odk-plan-insp-head" }, [
        el("span", { class: "odk-label", text: `${title} · PARAMETERS` }), generate,
      ]));
      const statusLine = el("div", { class: "odk-plan-status", "data-plan-status": "" }, [
        !overview.aoi || !overview.aoi.set ? status("idle", "Draw or import the area of interest to plan")
          : form.dirty || !sameType ? status("warn", "Changed since this plan: Generate to apply")
            : plan ? status("ok", overview.saved ? `Planned · saved as ${overview.saved.name} v${overview.saved.version_num}` : "Planned · not saved")
              : status("idle", "No mission planned yet"),
      ]);
      node.appendChild(statusLine);

      const params = el("div", { class: "odk-plan-grid" });
      params.appendChild(cameraPicker((cameras && cameras.cameras) || [], shownPlan, (payloads && payloads.payloads) || []));
      const camera = form.camera || (plan && plan.camera) || null;
      let gsdInput = null;
      for (const field of PARAMETERS) {
        if (field.key === "gsd_cm") {
          const wrap = numberField(field, shownPlan, { onCommit: async (value, input) => {
            if (!camera) { shellRef && shellRef.toast("GSD: choose a camera first; the altitude is solved for it.", "warn"); return; }
            const solved = await tryCall("altitude_for_gsd", camera, Number(value));
            if (!solved) { shellRef && shellRef.toast(`GSD ${value} cm/px could not be solved for ${camera}.`, "warn"); return; }
            setValue("altitude_m", String(solved.altitude_m));
            const alt = document.querySelector('[data-field="altitude_m"]');
            if (alt) alt.value = String(solved.altitude_m);
            input.closest(".odk-field").querySelector(".odk-plan-field-note").textContent = `altitude ${fmtNum(solved.altitude_m, 1)} m for ${camera}`;
          }, note: " " });
          gsdInput = wrap.querySelector("input");
          params.appendChild(wrap);
          continue;
        }
        const onCommit = field.key === "altitude_m" ? async (value) => {
          if (!camera || !gsdInput || !(Number(value) > 0)) return;
          const described = await tryCall("describe_camera", camera, Number(value));
          if (described && described.camera) {
            gsdInput.value = fmtNum(described.camera.gsd_cm, 2);
            form.values.gsd_cm = gsdInput.value;
            gsdInput.closest(".odk-field").querySelector(".odk-plan-field-note").textContent = `at ${fmtNum(value, 1)} m, ${camera}`;
          }
        } : null;
        params.appendChild(numberField(field, shownPlan, { readonly: field.readonly, onCommit }));
      }
      node.appendChild(params);

      // The type's own inputs (thermal camera, structure heights, a route...).
      const inputs = TEMPLATE_INPUTS[template] || [];
      const needsRoute = ROUTE_TEMPLATES.has(template);
      const needsRegions = REGION_TEMPLATES.has(template);
      if (inputs.length || needsRoute || needsRegions || template === "multi_facade") {
        const extra = el("div", { class: "odk-plan-grid" });
        for (const field of inputs) {
          if (field.kind === "camera") {
            const options = [{ value: "", label: field.required ? "Choose…" : "None" },
              ...((cameras && cameras.cameras) || []).filter((c) => !field.thermal || c.thermal)
                .map((c) => ({ value: c.key, label: c.name }))];
            extra.appendChild(Object.assign(select({ label: field.label, options, value: form.values[field.key] || "", mono: false,
              onChange: (v) => { setValue(field.key, v); markDirty(); } }), { className: "odk-field odk-plan-wide" }));
          } else if (field.kind === "payload") {
            const options = [{ value: "", label: "Choose…" },
              ...((payloads && payloads.payloads) || []).map((p) => ({ value: p.key || p.name, label: p.name || p.key }))];
            extra.appendChild(Object.assign(select({ label: field.label, options, value: form.values[field.key] || "", mono: false,
              onChange: (v) => { setValue(field.key, v); markDirty(); } }), { className: "odk-field odk-plan-wide" }));
          } else if (field.text || field.kind) {
            extra.appendChild(textField(field));
          } else {
            extra.appendChild(numberField(field, null));
          }
        }
        const verbs = [];
        if (needsRoute) {
          verbs.push(act("Draw Route", form.route ? `Route drawn (${form.route.length} points) · redraw` : "Draw route"));
        }
        if (needsRegions) verbs.push(act("Draw Interest Region", `Interest region (${form.regions.length})`));
        if ((needsRoute && form.route) || (needsRegions && form.regions.length)) verbs.push(act("Clear Route", "Clear", { variant: "ghost" }));
        if (template === "multi_facade") verbs.push(act("Facade Footprint", "Read the footprint…"));
        if (verbs.length) extra.appendChild(el("div", { class: "odk-plan-row-verbs odk-plan-wide" }, verbs));
        node.appendChild(section({ label: `${title} · OWN INPUTS`, children: [extra], className: "odk-plan-sec" }));
      }

      // Advanced: everything else plan_mission reads.
      const adv = el("details", { class: "odk-plan-advanced", ...(form.openAdvanced ? { open: "" } : {}) });
      adv.addEventListener("toggle", () => { form.openAdvanced = adv.open; });
      adv.appendChild(el("summary", { "data-control": "advanced", text: "Advanced" }));
      const advGrid = el("div", { class: "odk-plan-grid" });
      for (const field of ADVANCED) advGrid.appendChild(numberField(field, shownPlan));
      const modeSel = select({ label: "Altitude reference", options: ALTITUDE_MODES, value: form.altitudeMode, mono: false,
        onChange: (v) => { form.altitudeMode = v; form.dirty = true; markDirty(); } });
      modeSel.classList.add("odk-plan-wide");
      advGrid.appendChild(modeSel);
      adv.appendChild(advGrid);
      node.appendChild(adv);

      node.appendChild(section({ label: "SAFETY", children: [safetyRows(overview)], className: "odk-plan-sec" }));
      node.appendChild(section({ label: "EXPORT", aside: `${EXPORT_FORMATS.length} FORMATS`,
        children: [exportBlock(exports, plan)], className: "odk-plan-sec" }));

      const n = exportChoice.size;
      const exportBtn = act("Export Mission Files", n === 1 ? "Export 1 format" : `Export ${n} formats`, {
        variant: "primary", payload: () => ({ formats: [...exportChoice] }), disabled: !plan || !n });
      node.appendChild(el("div", { class: "odk-plan-insp-foot" }, [
        exportBtn,
        act("Send to Fly", "Send to Fly", { disabled: !plan }),
      ]));
      const allFormats = act("Export", "All formats", { variant: "ghost", disabled: !plan, title: "Write every format (Mission › Export All Formats)" });
      node.appendChild(el("div", { class: "odk-plan-insp-foot odk-plan-insp-foot--minor" }, [allFormats]));
      return node;
    },
  });
}

function inspectorSegments() {
  return live({
    calls: ["mission_overview", "plan_battery_segments", "mission_estimates"],
    empty: "The plan's sorties could not be read.",
    isEmpty: ([overview]) => !overview,
    render: ([overview, segments, estimates]) => {
      const node = el("div", { class: "odk-plan-inspector" });
      if (!overview.plan) {
        node.appendChild(el("p", { class: "odk-plan-hint", text: "Generate a plan to split it into battery sorties and see what it needs." }));
        return node;
      }
      const e = (estimates && estimates.estimates) || {};
      const battery = e.battery || {};
      const storage = e.storage || {};
      node.appendChild(el("div", { class: "odk-plan-insp-head" }, [
        el("span", { class: "odk-label", text: "SORTIES" }),
        act("Battery Segments", "Re-split"),
      ]));
      const grid = el("div", { class: "odk-plan-stats" }, [
        stat({ label: "Batteries", value: battery.batteries_required ?? null }),
        stat({ label: "Usable per battery", value: fmtNum(battery.usable_endurance_min, 1), unit: "min" }),
        stat({ label: "Photos", value: fmtInt(e.image_count) }),
        stat({ label: "Storage", value: fmtNum(storage.total_gb, 2), unit: "GB" }),
      ]);
      node.appendChild(grid);
      if (segments && segments.note) node.appendChild(el("p", { class: "odk-plan-hint", text: segments.note }));
      const splits = (segments && segments.splits) || [];
      if (splits.length) {
        const list = el("ol", { class: "odk-plan-splits" });
        splits.forEach((s, i) => list.appendChild(el("li", {}, [
          el("span", { text: `Sortie ${i + 1}` }),
          el("span", { class: "odk-num", text: [s.start_index != null && s.end_index != null ? `WP ${s.start_index}–${s.end_index}` : "",
            s.duration_min != null ? `${fmtNum(s.duration_min, 1)} min` : ""].filter(Boolean).join(" · ") || JSON.stringify(s) }),
        ])));
        node.appendChild(list);
      }
      const notes = [...(battery.assumptions || []), ...(storage.assumptions || [])];
      if (e.note) notes.unshift(e.note);
      if (notes.length) {
        node.appendChild(el("details", { class: "odk-plan-notes", open: "" }, [
          el("summary", { "data-control": "estimate-notes", text: "What these rest on" }),
          ...notes.map((t) => el("p", { text: t })),
        ]));
      }
      node.appendChild(el("div", { class: "odk-plan-row-verbs" }, [act("Mission Estimates", "Estimate again", { variant: "ghost" })]));
      return node;
    },
  });
}

function inspectorSite() {
  return live({
    calls: ["list_site_markers", "mission_overview"],
    empty: "Site markers could not be read.",
    isEmpty: ([markers]) => !markers,
    render: ([listed, overview]) => {
      const node = el("div", { class: "odk-plan-inspector" });
      const markers = (listed && listed.markers) || [];
      node.appendChild(el("div", { class: "odk-plan-insp-head" }, [
        el("span", { class: "odk-label", text: `SITE MARKERS · ${markers.length}` }),
        act("Add Marker", "Add marker"),
      ]));
      if (!markers.length) node.appendChild(el("p", { class: "odk-plan-hint", text: "No markers on this site. Place home, or add a hazard, gate or landing zone." }));
      const list = el("div", { class: "odk-plan-rows" });
      for (const m of markers) {
        list.appendChild(el("div", { class: "odk-plan-row" }, [
          el("span", { class: "odk-plan-row-label", text: m.name }),
          el("span", { class: "odk-plan-row-value", text: [m.kind, m.radius_m ? `r ${fmtNum(m.radius_m, 0)} m` : ""].filter(Boolean).join(" · ") }),
          act("Remove Marker", "Remove", { variant: "ghost", payload: { id: m.id }, className: "odk-plan-row-act" }),
        ]));
      }
      node.appendChild(list);
      node.appendChild(el("div", { class: "odk-plan-row-verbs" }, [act("Place Home", "Place home"), act("Remove Marker", "Remove…", { variant: "ghost" })]));
      node.appendChild(section({ label: `NO-FLY ZONES · ${overview ? overview.no_fly_count : 0}`, className: "odk-plan-sec", children: [
        el("div", { class: "odk-plan-row-verbs" }, [act("Draw No-fly", "Draw no-fly zone"), act("Clear No-fly", "Clear", { variant: "ghost" })]),
      ] }));
      node.appendChild(section({ label: "FLY THE CORNERS", className: "odk-plan-sec", children: [
        el("p", { class: "odk-plan-hint", text: overview && overview.boundary_marks
          ? `${overview.boundary_marks} corner(s) marked from the aircraft.` : "Mark the boundary from the aircraft's position, corner by corner (needs a connected aircraft)." }),
        el("div", { class: "odk-plan-row-verbs" }, [act("Mark Corner", "Mark corner"), act("Close Boundary", "Close"), act("Clear Corners", "Clear", { variant: "ghost" })]),
      ] }));
      return node;
    },
  });
}

/* =============================================================== render */

let keyHandler = null;

export async function render(regions /* { left, main, inspector } */, context) {
  await ensureStyles();
  shellRef = context.shell;
  const { left, main, inspector } = regions;
  if (view.mapCtl) { view.mapCtl.destroy(); view.mapCtl = null; }

  // The Plan form owns the mission options the core "Plan" verb sends.
  context.provide("missionOptions", () => missionOptions(view.overview));

  /* left */
  hosts.templates = templatesPanel();
  hosts.aoi = aoiPanel();
  left.replaceChildren(el("div", { class: "odk-plan-left" }, [hosts.templates, hosts.aoi]));

  /* main */
  const mapHost = el("div", { class: "odk-plan-map", "data-plan-map": "" });
  const legend = el("div", { class: "odk-plan-legend", hidden: "" });
  hosts.strip = strip();
  main.replaceChildren(el("div", { class: "odk-plan-main" }, [
    missionBar(), mapHost, el("div", { class: "odk-plan-strip" }, [hosts.strip]),
  ]));
  mapHost.appendChild(legend);

  /* inspector */
  const tabsHost = el("div", { class: "odk-plan-tabs" });
  const body = el("div", { class: "odk-plan-insp-body" });
  const showTab = (id) => {
    view.tab = id;
    hosts.inspector = id === "segments" ? inspectorSegments() : id === "site" ? inspectorSite() : inspectorPlan();
    body.replaceChildren(hosts.inspector);
  };
  tabsHost.appendChild(panelTabs([{ id: "plan", label: "Plan" }, { id: "segments", label: "Sorties" }, { id: "site", label: "Site" }],
    { active: view.tab, label: "Inspector", onSelect: showTab }));
  inspector.replaceChildren(el("div", { class: "odk-plan-insp" }, [tabsHost, body]));
  showTab(view.tab);

  /* the map */
  const centre = view.overview && view.overview.aoi && view.overview.aoi.polygon && view.overview.aoi.polygon[0];
  const mounted = await mountProjectMap(mapHost, {
    screen: "plan",
    // Plan is about the capture geometry: camera positions, the mission and the survey
    // area open on (U3's per-screen defaults; set explicitly so a change there cannot
    // switch them off here).
    visibility: { cameras: true, mission: true, aoi: true, nofly: true },
    ...(centre ? { centre, zoom: 16 } : {}),
    notify: (text, kind) => context.shell.toast(text, kind),
    run: (key, payload) => context.shell.runAction(key, payload),
  });
  if (mounted && mounted.map) {
    const ctl = attachPlanMap(mounted.map, mapHost, {
      notify: (text, kind) => context.shell.toast(text, kind),
      onSelect: (row) => { if (view.profileMove && !(view.mapCtl && view.mapCtl.state.playing)) view.profileMove(row); },
    });
    ctl.setFootprints(view.footprints);
    view.mapCtl = ctl;
    mapHost.appendChild(overviewFeed((overview) => {
      takeOverview(overview);
      ctl.update(overview);
      renderLegend(legend, overview);
      context.setFooter(overview.plan ? `ALT FRAME ${overview.plan.altitude_frame}` : "ALT FRAME no plan");
    }));
    mapHost.addEventListener("odk:plan-editing", (event) => {
      const b = document.querySelector('.odk-plan-tools [data-action="Edit Vertices"]');
      if (b) b.setAttribute("aria-pressed", event.detail.editing ? "true" : "false");
    });
  }

  /* screen events: the verbs that only this screen can show */
  if (!window.__odkPlanBound) {
    window.__odkPlanBound = true;
    window.addEventListener("odk:plan", (event) => {
      const verb = event.detail && event.detail.verb;
      const ctl = view.mapCtl;
      if (verb === "edit-vertices" && ctl) ctl.editVertices(!ctl.state.editing);
      else if (verb === "play" && ctl) {
        rerender("strip");
        ctl.play((row) => { if (view.profileMove) view.profileMove(row); }).then(() => rerender("strip"));
        setTimeout(() => rerender("strip"), 0);
      } else if (verb === "facade" && ctl) ctl.showFacade(event.detail.segments);
      else if (["route", "regions"].includes(verb)) { if (ctl) ctl.update(view.overview); rerender("inspector"); }
      else if (verb === "heading") markDirty();
    });
  }
  if (keyHandler) document.removeEventListener("keydown", keyHandler);
  keyHandler = (event) => {
    const frame = document.querySelector(".odk-frame");
    if (!frame || frame.getAttribute("data-screen") !== "plan") return;
    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && (event.key === "p" || event.key === "P")) {
      event.preventDefault();
      context.shell.runAction("Plan");
    }
  };
  document.addEventListener("keydown", keyHandler);
}

function renderLegend(legend, overview) {
  const plan = overview && overview.plan;
  const aoi = overview && overview.aoi;
  const items = [];
  if (aoi && aoi.set) items.push(["boundary", `Boundary · ${aoi.vertices} vx`]);
  if (plan) {
    items.push(["path", `Flight path · ${plan.line_count ? `${plan.line_count} lines` : `${plan.leg_count} legs`}`]);
    items.push(["waypoint", `Waypoint · ${plan.trigger_count === plan.waypoint_count ? "trigger" : `${plan.trigger_count} triggers`}`]);
    if (plan.footprints && plan.footprints.features.length && view.footprints) items.push(["footprint", "Photo footprint"]);
  }
  if (overview && overview.no_fly_count) items.push(["nofly", `No-fly · ${overview.no_fly_count}`]);
  legend.replaceChildren(...items.map(([kind, text]) => el("span", { class: "odk-plan-legend-item" }, [
    el("span", { class: `odk-plan-swatch odk-plan-swatch--${kind}`, "aria-hidden": "true" }), el("span", { text }),
  ])));
  legend.hidden = !items.length;
}
