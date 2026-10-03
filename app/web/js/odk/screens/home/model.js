/* Home: data. The Api calls each Home panel reads, and the arithmetic that turns their
   answers into what is drawn -- shared by the desk renderer (./desk.js) and any later
   renderer, so both show the same figures.

   Every figure on Home comes from one of these calls. A value the Api did not report
   is null here and renders "not reported"; nothing in this file supplies a default
   figure. Formatting (separators, units, local time) is arithmetic on the Api's
   numbers, not data. */

/** Api methods each Home panel reads. Each panel is one live() block in ./desk.js. */
export const PANELS = {
  projects: ["home_projects"],
  status: ["project_storage", "get_state", "model_status"],
  header: ["get_project", "dataset_summary", "get_project_crs", "list_annotation_tags"],
  tiles: ["dataset_summary", "get_state", "list_mission_versions", "find_annotations", "pipeline_reports"],
  map: ["reconstruction_summary"],
  reconstruction: ["reconstruction_summary"],
  measurements: ["list_measurements"],
  report: ["list_reports", "report_files", "report_readiness"],
  activity: ["audit_log", "dataset_summary"],
  notifications: ["list_notifications"],
  exports: ["list_exports"],
};

/** Api methods this screen reads. Each entry: { method, args }. */
export const CALLS = [...new Set(Object.values(PANELS).flat())].map((method) => ({ method, args: [] }));

/* ------------------------------------------------------------- formatting */

const NUM = new Intl.NumberFormat("en-US");
const given = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

/** A whole number with separators, or null. */
export function count(value) {
  return given(value) ? NUM.format(Number(value)) : null;
}

/** A number to `digits` decimals with separators, or null. */
export function fixed(value, digits) {
  if (!given(value)) return null;
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Bytes in SI units, as report and storage sizes are printed ("2.37 MB"). */
export function bytes(value) {
  if (!given(value)) return null;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = Number(value);
  let i = 0;
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i += 1; }
  const digits = i === 0 ? 0 : v < 10 ? 2 : v < 100 ? 1 : 0;
  return `${v.toFixed(digits)} ${units[i]}`;
}

/** Exact bytes with separators ("259,920 B"), the export list's figure. */
export function exactBytes(value) {
  const c = count(value);
  return c === null ? null : `${c} B`;
}

const pad = (n) => String(n).padStart(2, "0");

/** A stamp the Api wrote in UTC ("...Z", "+00:00", or naive = UTC), as a Date. */
export function parseUtc(iso) {
  if (!iso) return null;
  let text = String(iso).trim();
  if (!/[zZ]$|[+-]\d\d:?\d\d$/.test(text)) text += "Z";
  const d = new Date(text);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** "HH:MM" on this machine's clock. */
export function localTime(date) {
  return date ? `${pad(date.getHours())}:${pad(date.getMinutes())}` : null;
}

/** "YYYY-MM-DD" on this machine's clock. */
export function localDate(date) {
  return date ? `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` : null;
}

/** "UTC−4": this machine's offset at `date` (the activity list's clock). */
export function zoneLabel(date = new Date()) {
  const minutes = -date.getTimezoneOffset();
  if (minutes === 0) return "UTC";
  const sign = minutes > 0 ? "+" : "−";
  const h = Math.floor(Math.abs(minutes) / 60);
  const m = Math.abs(minutes) % 60;
  return `UTC${sign}${h}${m ? `:${pad(m)}` : ""}`;
}

/** "YYYY-MM-DD HH:MM UTC", the project's created line as the artboard prints it. */
export function utcStamp(iso) {
  const d = parseUtc(iso);
  if (!d) return null;
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} `
    + `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

/** Degrees as "41.30410° N  81.75212° W". */
export function latLon(lon, lat) {
  if (!given(lon) || !given(lat)) return null;
  const ns = Number(lat) >= 0 ? "N" : "S";
  const ew = Number(lon) >= 0 ? "E" : "W";
  return `${Math.abs(Number(lat)).toFixed(5)}° ${ns}  ${Math.abs(Number(lon)).toFixed(5)}° ${ew}`;
}

/** The WGS 84 UTM zone an EPSG code names ("UTM 17N"), or null. Arithmetic on the code. */
export function utmZone(epsg) {
  const code = Number(epsg);
  const north = 32600;
  const south = 32700;
  const zones = 60;
  if (code > north && code <= north + zones) return `UTM ${code - north}N`;
  if (code > south && code <= south + zones) return `UTM ${code - south}S`;
  return null;
}

/** "m2" -> "m²". */
export function unitLabel(unit) {
  return ({ m2: "m²", m3: "m³", deg: "°" })[unit] || unit || "";
}

/** The file's own base name. */
export function baseName(path) {
  return String(path || "").split(/[\\/]/).filter(Boolean).pop() || "";
}

/* ------------------------------------------------------------------ derive */

/** A polygon has at least three vertices: the fewest a boundary can be measured with. */
export const MIN_POLYGON = 3;

/** Survey date and window from dataset_summary().active.capture (EXIF clocks). */
export function survey(capture) {
  if (!capture || !capture.first) return null;
  const date = capture.first.slice(0, 10);
  const end = capture.last ? capture.last.slice(0, 10) : date;
  const from = capture.first.slice(11, 16);
  const to = capture.last ? capture.last.slice(11, 16) : from;
  const clock = capture.utc_offset ? `UTC${capture.utc_offset}` : "local";
  return {
    date: end !== date ? `${date} – ${end}` : date,
    window: `${from}–${to} ${clock}`,
    dated: capture.dated,
    images: capture.images,
    clock: capture.clock,
  };
}

/** The mission tile: the newest saved version's summary. */
export function mission(state, versions) {
  const saved = ((versions && versions.versions) || [])[0] || null;
  if (!saved) return { saved: null, plannedUnsaved: Boolean(state && state.has_mission) };
  let summary = {};
  try { summary = JSON.parse(saved.plan_summary_json || "{}") || {}; } catch { summary = {}; }
  return {
    saved: {
      template: summary.template || saved.template || null,
      waypoints: summary.waypoints ?? null,
      altitude_m: summary.altitude_m ?? null,
      gsd_cm: summary.gsd_cm ?? null,
      distance_m: summary.distance_m ?? null,
      name: saved.mission_name || null,
      version: saved.version_num ?? null,
    },
    plannedUnsaved: Boolean(state && state.has_mission),
  };
}

/* An audit event's name, spelled out; anything not listed is its own name in words. */
const TITLES = {
  project_created: "Project created",
  dataset_imported: "Photos imported",
  reconstruction_complete: "Reconstruction complete",
  report_generated: "Report generated",
  report_exported: "Findings exported",
  mission_planned: "Mission planned",
  mission_version_saved: "Mission version saved",
  mission_exported: "Mission exported",
  flight_log_exported: "Flight log exported",
};

function eventDetail(p) {
  const parts = [];
  const add = (v) => { if (v !== null && v !== undefined && v !== "") parts.push(String(v)); };
  if (given(p.images)) add(`${count(p.images)} photos`);
  if (given(p.waypoints)) add(`${count(p.waypoints)} wp`);
  add(p.template);
  if (p.mission_name && given(p.version_num)) add(`${p.mission_name} v${p.version_num}`);
  if (Array.isArray(p.formats)) add(`${p.formats.length} formats`);
  add(p.engine);
  if (given(p.findings)) add(`${count(p.findings)} findings`);
  add(baseName(p.folder || p.directory || p.file || p.output || ""));
  if (!parts.length) {
    for (const [k, v] of Object.entries(p)) {
      if (v === null || v === undefined || v === "" || typeof v === "object") continue;
      add(/path|dir|root/i.test(k) ? baseName(v) : `${k.replace(/_/g, " ")} ${v}`);
      if (parts.length >= 2) break;
    }
  }
  return parts.slice(0, 3).join(" · ");
}

/** One audit event as {when, title, detail, tone}. */
export function activity(event) {
  let payload = {};
  try {
    payload = typeof event.payload_json === "string" ? JSON.parse(event.payload_json || "{}") : (event.payload_json || {});
  } catch { payload = {}; }
  const type = String(event.event_type || "");
  const words = type.replace(/_/g, " ");
  return {
    id: event.id,
    type,
    when: parseUtc(event.created_at),
    title: TITLES[type] || (words ? words[0].toUpperCase() + words.slice(1) : "Event"),
    detail: eventDetail(payload || {}),
    tone: /fail|refus|error|mismatch/.test(type) ? "danger"
      : /complete|generated|exported|saved|imported|created/.test(type) ? "ok" : "running",
  };
}

/** Sort orders for the project list tabs. */
export const PROJECT_ORDER = {
  recent: (a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")),
  name: (a, b) => String(a.name || "").localeCompare(String(b.name || ""), undefined, { sensitivity: "base" }),
  status: (a, b) => (Number(b.processed) - Number(a.processed))
    || (Number(b.reports_count || 0) - Number(a.reports_count || 0))
    || (Number(b.photos || 0) - Number(a.photos || 0))
    || String(a.name || "").localeCompare(String(b.name || "")),
};

/** "Processed · report ready" from the Api's status words. */
export function statusLine(project) {
  const words = (project.status || []).join(" · ");
  return words ? words[0].toUpperCase() + words.slice(1) : null;
}

/** Results keyed by method (live() hands them over positionally). */
export function derive(results, methods = []) {
  if (!Array.isArray(results)) return { ...results };
  return Object.fromEntries(methods.map((name, i) => [name, results[i]]));
}
