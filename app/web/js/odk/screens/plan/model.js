/* Plan: data. The Api calls the screen reads, the mission form the operator edits, and
   the values derived from both, shared by the desk renderer (./desk.js) and the Field
   renderer (./field.js) so the two modes show the same plan.

   Every figure on the Plan screen comes from an Api answer read here: mission_overview
   (the area and the plan, mission/overview.py), mission_templates, list_cameras,
   list_mission_versions, list_mission_exports, plan_battery_segments, mission_estimates.
   The form holds only what the operator typed; a field left empty is sent nowhere, so
   the planner's own default is what gets planned -- and once a plan exists the field
   shows the value that plan was made with (docs/ui/ARCHITECTURE.md, "Honesty rules"). */

import { tryCall } from "../../../workspace/api.js";

/** Api methods this screen reads. Each entry: { method, args }. */
export const CALLS = [
  { method: "mission_overview", args: [] },
  { method: "mission_templates", args: [] },
  { method: "list_cameras", args: [] },
  { method: "list_payloads", args: [] },
  { method: "list_mission_versions", args: [] },
  { method: "list_mission_exports", args: [] },
  { method: "list_layers", args: [] },
  { method: "list_site_markers", args: [] },
  { method: "plan_battery_segments", args: [] },
  { method: "mission_estimates", args: [] },
  { method: "describe_terrain_cache", args: [] },
];

/** Turn the Api results (keyed by method) into what the renderers draw. */
export function derive(results) {
  const overview = results.mission_overview || null;
  const plan = (overview && overview.plan) || null;
  return {
    ...results,
    overview,
    plan,
    aoi: (overview && overview.aoi) || null,
    template: currentTemplate(plan),
    templates: templateList(results.mission_templates, results.list_mission_versions, plan),
  };
}

/* ------------------------------------------------------------ the form */

/* The parameters the inspector shows, the planner option each one sends, and where the
   plan reports the value it was made with. `planned` reads the overview's plan. */
export const PARAMETERS = [
  { key: "altitude_m", label: "Altitude", unit: "m", planned: (p) => p.altitude_m, digits: 1 },
  { key: "gsd_cm", label: "GSD", unit: "cm/px", planned: (p) => p.gsd_cm, digits: 2, solves: "altitude_m" },
  { key: "front_overlap_pct", label: "Front overlap", unit: "%", planned: (p) => p.front_overlap_pct, digits: 0 },
  { key: "side_overlap_pct", label: "Side overlap", unit: "%", planned: (p) => p.side_overlap_pct, digits: 0 },
  { key: "speed_m_s", label: "Speed", unit: "m/s", planned: (p) => p.speed_m_s, digits: 1 },
  { key: "gimbal_tilt_deg", label: "Gimbal", unit: "°", planned: (p) => p.gimbal_tilt_deg, digits: 0 },
  { key: "line_spacing_m", label: "Line spacing", unit: "m", planned: (p) => p.line_spacing_m, digits: 1, readonly: true },
  { key: "capture_spacing_m", label: "Trigger every", unit: "m", planned: (p) => p.capture_spacing_m, digits: 1, readonly: true },
];

/* Advanced: what plan_mission reads beyond the parameters. Blank = the planner's own
   default (it says which in its answer). */
export const ADVANCED = [
  { key: "line_bearing_deg", label: "Line heading", unit: "°", planned: (p) => p.line_bearing_deg, digits: 0 },
  { key: "dwell_s", label: "Dwell", unit: "s" },
  { key: "min_altitude_m", label: "Min altitude", unit: "m", planned: (p) => p.min_altitude_m },
  { key: "max_altitude_m", label: "Max altitude", unit: "m", planned: (p) => p.max_altitude_m },
  { key: "rth_altitude_m", label: "RTH altitude", unit: "m", planned: (p) => p.rth_altitude_m },
  { key: "standoff_m", label: "Stand-off", unit: "m" },
  { key: "wind_speed_m_s", label: "Wind", unit: "m/s" },
  { key: "wind_direction_deg", label: "Wind from", unit: "°" },
  { key: "wind_gust_m_s", label: "Gusts", unit: "m/s" },
  { key: "endurance_min", label: "Endurance", unit: "min" },
  { key: "reserve_pct", label: "Reserve", unit: "%" },
];

/* The altitude reference: above ground (needs terrain) or above sea level. */
export const ALTITUDE_MODES = [
  { value: "", label: "Planner default" },
  { value: "agl", label: "Above ground (AGL)" },
  { value: "amsl", label: "Above sea level (AMSL)" },
];

/* Inputs only some mission types read (cockpit mission form, MISSION_OPTION_KEYS). */
export const TEMPLATE_INPUTS = {
  thermal: [
    { key: "thermal_camera", label: "Thermal camera", kind: "camera", thermal: true, required: true },
    { key: "target_gsd_cm", label: "Thermal GSD", unit: "cm/px", required: true },
    { key: "rgb_camera", label: "Paired RGB camera", kind: "camera" },
  ],
  multispectral: [
    { key: "payload_key", label: "Multispectral payload", kind: "payload", required: true },
    { key: "calibration_panel_lonlat", label: "Calibration panel (lon, lat)", text: true },
    { key: "accept_relative_values", label: "Accept relative values", kind: "yesno" },
  ],
  pylon_inspection: [
    { key: "pylon_elements", label: "Elements (name@height[:radius])", text: true },
    { key: "structure_radius_m", label: "Structure radius", unit: "m" },
  ],
  dome_inspection: [
    { key: "dome_radius_m", label: "Dome radius", unit: "m" },
    { key: "dome_base_altitude_m", label: "Dome base", unit: "m" },
  ],
  box_inspection: [{ key: "box_altitude_levels_m", label: "Levels (m, comma separated)", text: true }],
  facade: [
    { key: "facade_bottom_altitude_m", label: "Wall bottom", unit: "m" },
    { key: "facade_top_altitude_m", label: "Wall top", unit: "m" },
  ],
  facade_mapping: [
    { key: "facade_bottom_altitude_m", label: "Wall bottom", unit: "m" },
    { key: "facade_top_altitude_m", label: "Wall top", unit: "m" },
  ],
  multi_facade: [
    { key: "facade_bottom_altitude_m", label: "Wall bottom", unit: "m" },
    { key: "facade_top_altitude_m", label: "Wall top", unit: "m" },
  ],
  tower_mapping: [
    { key: "tower_bottom_altitude_m", label: "Tower bottom", unit: "m" },
    { key: "tower_top_altitude_m", label: "Tower top", unit: "m" },
    { key: "tower_object_radius_m", label: "Tower radius", unit: "m" },
  ],
  wind_turbine: [
    { key: "turbine_hub_height_m", label: "Hub height", unit: "m" },
    { key: "turbine_blade_length_m", label: "Blade length", unit: "m" },
    { key: "turbine_rotor_facing_deg", label: "Rotor faces", unit: "°" },
    { key: "turbine_nacelle_length_m", label: "Nacelle length", unit: "m" },
    { key: "turbine_blade_angles_deg", label: "Blade angles (°, comma separated)", text: true },
  ],
  orbit: [{ key: "orbit_radius_m", label: "Orbit radius", unit: "m" }],
  linked_mission: [{ key: "linked_version_ids", label: "Saved version ids (comma separated)", text: true }],
};

/* Types that fly a line the operator draws (plan_mission path_lonlat), and the type
   that adapts to drawn interest regions. */
export const ROUTE_TEMPLATES = new Set(["linear_inspection", "lateral_capture", "waypoints",
  "solar_inspection", "facade", "facade_mapping"]);
export const REGION_TEMPLATES = new Set(["smart_adaptive"]);

/* What the operator has set on this screen, kept while the app is open. */
export const form = {
  template: null,     // null: the plan's own type, else the planner default
  camera: null,       // null: the plan's own camera, else the planner default
  payload: null,
  altitudeMode: "",
  values: {},         // key -> string as typed
  route: null,        // [[lon, lat], ...] drawn with the Route tool
  regions: [],        // [[[lon, lat], ...], ...] drawn with the Interest tool
  dirty: false,       // changed since the plan on screen was made
};

const NUMBER_TEXT = /^-?\d*\.?\d+(e-?\d+)?$/i;
const ALL_FIELDS = () => [...PARAMETERS, ...ADVANCED, ...Object.values(TEMPLATE_INPUTS).flat()];

/** The planner options for the form as it stands (shell.provide("missionOptions")). */
export function missionOptions(view = null) {
  const plan = view && view.plan;
  const options = {};
  const template = form.template || (plan && plan.template) || null;
  if (template) options.template = template;
  const camera = form.camera || (plan && plan.camera) || null;
  if (camera) options.camera = camera;
  if (form.payload) options.payload = form.payload;
  if (form.altitudeMode) options.terrain_follow_mode = form.altitudeMode;
  const allowed = new Set([...PARAMETERS, ...ADVANCED, ...(TEMPLATE_INPUTS[template] || [])].map((f) => f.key));
  const fields = ALL_FIELDS();
  for (const [key, raw] of Object.entries(form.values)) {
    const text = String(raw ?? "").trim();
    if (!text || !allowed.has(key)) continue;
    const field = fields.find((f) => f.key === key);
    if (field && field.readonly) continue;
    if (key === "gsd_cm") continue; // solved into altitude_m through altitude_for_gsd
    if (field && field.kind === "yesno") { options[key] = text; continue; }
    options[key] = field && field.text ? text : NUMBER_TEXT.test(text) ? Number(text) : text;
  }
  // A plan already on screen keeps the inputs it was made with unless they are changed:
  // re-planning after moving one vertex must not reset the altitude to the default.
  if (plan && (!form.template || form.template === plan.template)) {
    for (const field of [...PARAMETERS, ...ADVANCED]) {
      if (field.readonly || field.key === "gsd_cm" || !field.planned) continue;
      if (options[field.key] !== undefined) continue;
      // The heading is carried as the planner's own angle, which reproduces the lines
      // exactly; a compass bearing is only sent when the operator sets one.
      if (field.key === "line_bearing_deg") {
        if (plan.flight_direction_deg != null) options.flight_direction_deg = plan.flight_direction_deg;
        continue;
      }
      const value = field.planned(plan);
      if (value !== null && value !== undefined && value !== "") options[field.key] = value;
    }
  }
  if (form.route && form.route.length >= 2 && ROUTE_TEMPLATES.has(template)) options.path_lonlat = form.route;
  if (form.regions.length && REGION_TEMPLATES.has(template)) options.interest_regions = form.regions;
  return options;
}

/** The value a field shows: what was typed, else what the plan on screen was made with. */
export function fieldValue(field, plan) {
  const typed = form.values[field.key];
  if (typed !== undefined && typed !== "") return String(typed);
  if (plan && field.planned && (!form.template || form.template === plan.template)) {
    const value = field.planned(plan);
    if (value !== null && value !== undefined && Number.isFinite(Number(value))) {
      return field.digits === undefined ? String(Math.round(Number(value) * 100) / 100)
        : Number(value).toFixed(field.digits);
    }
  }
  return "";
}

export function setValue(key, value) {
  form.values[key] = value;
  form.dirty = true;
}

/** Forget what was typed (after a plan is made from it, or a version is loaded). */
export function settleForm() {
  form.values = {};
  form.dirty = false;
}

/** The mission type on screen: the one chosen here, else the plan's. */
export function currentTemplate(plan) {
  return form.template || (plan && plan.template) || null;
}

/* --------------------------------------------------------- templates */

/* The artboard's order inside each group; a type not listed follows in the Api's order. */
const ORDER = ["grid", "double_grid", "corridor", "smart_adaptive", "solar_inspection",
  "magnetic_mapping", "thermal", "multispectral", "closed_loop",
  "facade", "multi_facade", "tower_mapping", "orbit", "roof_inspection", "wind_turbine",
  "pylon_inspection", "facade_mapping", "dome_inspection", "box_inspection",
  "panorama", "bubble_360", "waypoints", "linear_inspection", "lateral_capture", "linked_mission"];
/* Shown before "Show n more" (the artboard's list). */
export const PRIMARY = new Set(["grid", "double_grid", "corridor", "smart_adaptive",
  "solar_inspection", "magnetic_mapping", "facade", "multi_facade", "tower_mapping", "orbit",
  "roof_inspection", "wind_turbine", "pylon_inspection", "panorama", "bubble_360",
  "waypoints", "linear_inspection"]);

export const GROUP_LABELS = { area: "Area", structure: "Structure", capture: "Capture", other: "Other" };

/** "Grid survey (2D mapping)" -> ["Grid survey", "2D mapping"]. */
export function splitLabel(label) {
  const m = /^(.*?)\s*\((.*)\)\s*$/.exec(String(label || ""));
  return m ? [m[1], m[2]] : [String(label || ""), ""];
}

/** A saved version's summary, whichever way the store returned it. */
export function versionSummary(v) {
  let summary = v.plan_summary ?? v.plan_summary_json ?? {};
  if (typeof summary === "string") {
    try { summary = JSON.parse(summary); } catch { summary = {}; }
  }
  return summary || {};
}

/**
 * The grouped template list. Each row: {key, label, hint, group, primary, waypoints,
 * waypointsFrom}. `waypoints` is the plan on screen's count for its own type, else the
 * newest saved version of that type's count, else null -- never a figure for a type
 * nobody has planned.
 */
export function templateList(templates, versions, plan) {
  if (!templates || !templates.groups) return [];
  const labels = templates.labels || {};
  const saved = {};
  for (const v of ((versions && versions.versions) || []).slice()
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || ""))
      || (Number(b.version_num) || 0) - (Number(a.version_num) || 0))) {
    if (saved[v.template]) continue;
    const s = versionSummary(v);
    const count = Number(s.waypoints);
    if (Number.isFinite(count) && count > 0) {
      saved[v.template] = { count, from: `${v.mission_name || "mission"} v${v.version_num}` };
    }
  }
  const rows = [];
  for (const [group, names] of Object.entries(templates.groups)) {
    const ordered = names.slice().sort((a, b) => {
      const ia = ORDER.indexOf(a); const ib = ORDER.indexOf(b);
      return (ia < 0 ? ORDER.length : ia) - (ib < 0 ? ORDER.length : ib);
    });
    for (const key of ordered) {
      const [label, hint] = splitLabel(labels[key] || key);
      let waypoints = null;
      let waypointsFrom = "";
      if (plan && plan.template === key) {
        waypoints = plan.waypoint_count;
        waypointsFrom = "the plan on screen";
      } else if (saved[key]) {
        waypoints = saved[key].count;
        waypointsFrom = `saved ${saved[key].from}`;
      }
      rows.push({ key, label, hint, group, primary: PRIMARY.has(key), waypoints, waypointsFrom });
    }
  }
  return rows;
}

/** A template's short label from the Api's labels ("Grid survey"). */
export function templateLabel(templates, key) {
  const labels = (templates && templates.labels) || {};
  return splitLabel(labels[key] || String(key || "").replace(/_/g, " "))[0];
}

/* ----------------------------------------------------------- formats */

/** Every export the planner writes, in the artboard's order, with its file type. */
export const EXPORT_FORMATS = [
  { key: "qgc_plan", label: "QGroundControl", ext: ".plan" },
  { key: "dji_wpml", label: "DJI WPML", ext: ".kmz" },
  { key: "litchi", label: "Litchi", ext: ".csv" },
  { key: "kml", label: "KML", ext: ".kml" },
  { key: "geojson", label: "GeoJSON", ext: ".geojson" },
  { key: "qgc_wpl", label: "QGC WPL 110", ext: ".waypoints" },
];

/** Formats ticked for export (kept while the app is open). */
export const exportChoice = new Set(["qgc_plan", "dji_wpml", "litchi"]);

/** The newest export's files grouped by format: {format: {bytes, parts, path}}. */
export function latestExport(listed) {
  const run = listed && listed.exports && listed.exports[0];
  if (!run) return { stamp: null, byFormat: {}, directory: listed ? listed.directory : null };
  const byFormat = {};
  for (const file of run.files || []) {
    const entry = byFormat[file.format] || (byFormat[file.format] = { bytes: 0, parts: 0, path: file.path });
    entry.bytes += Number(file.bytes) || 0;
    entry.parts += 1;
  }
  return { stamp: run.stamp, byFormat, directory: listed.directory };
}

/* ------------------------------------------------------------ format */

export function fmtInt(value) {
  return Number.isFinite(Number(value)) && value !== null && value !== ""
    ? Math.round(Number(value)).toLocaleString("en-US") : null;
}

export function fmtNum(value, digits = 1) {
  return Number.isFinite(Number(value)) && value !== null && value !== "" ? Number(value).toFixed(digits) : null;
}

export function fmtBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n)) return null;
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(1)} KB`;
}

/** "41.30362° N" / "81.75226° W". */
export function fmtLat(lat) {
  return `${Math.abs(lat).toFixed(5)}° ${lat >= 0 ? "N" : "S"}`;
}
export function fmtLon(lon) {
  return `${Math.abs(lon).toFixed(5)}° ${lon >= 0 ? "E" : "W"}`;
}

/** "20261003T030115Z" -> "2026-10-03 03:01 UTC". */
export function fmtStamp(stamp) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(String(stamp || ""));
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} UTC` : String(stamp || "");
}

/** The words for the altitude reference the plan reports. */
export function frameWords(frame) {
  return { relative_to_home: "rel. home", amsl: "AMSL", agl: "AGL" }[frame] || String(frame || "");
}

/** Read every call in CALLS once (Field mode; the desk panels read through live()). */
export async function readAll(ask = tryCall) {
  const entries = await Promise.all(CALLS.map(async ({ method, args }) => [method, await ask(method, ...args)]));
  return derive(Object.fromEntries(entries));
}
