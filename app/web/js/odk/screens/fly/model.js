/* Fly: data. The Api calls the screen reads, the live() panels built on them and the
   values derived from them, shared by the desk renderer (./desk.js) and the Field
   renderer (../field/fly.js), so both modes show the same answers.

   Every figure on the Fly screen comes from one of these calls: telemetry, the
   vehicle's own messages, the flight session, the preflight checklist and the mission
   the vehicle acknowledged. Nothing here invents a value. A field the vehicle did not
   report stays null and is drawn as "not reported"; a value from a simulated driver
   keeps the Api's `is_simulated` so the renderer tags it (docs/ui/ARCHITECTURE.md,
   "Honesty rules"). The only literals below are timings and the deterministic
   arithmetic of distances along a route. */

import { tryCall } from "../../../workspace/api.js";
import { live } from "../../../workspace/live.js";

/** How often the screen asks: the aircraft every second, the rest every five. */
export const FAST_MS = 1000;
export const SLOW_MS = 5000;

/** Read every poll (about once a second): what changes while the aircraft flies. */
export const FAST_CALLS = [
  { method: "telemetry", args: [] },
  { method: "control_state", args: [] },
];

/** Read every few seconds, and after any command: the slower-moving picture. */
export const SLOW_CALLS = [
  { method: "get_state", args: [] },
  { method: "run_preflight", args: [] },
  { method: "vehicle_mission", args: [] },
  { method: "flight_session", args: [] },
  { method: "check_interrupted_flight", args: [] },
  { method: "list_mission_versions", args: [] },
  { method: "list_notifications", args: [true] },
];

/** Every Api method this screen reads (the inventory and the honesty guard list these). */
export const CALLS = [
  ...FAST_CALLS,
  ...SLOW_CALLS,
  { method: "vehicle_messages", args: [0] },
  { method: "audit_log", args: [200] },
];

/** The drivers the CONNECTION segmented control offers, in the artboard's order, with
    the URI each one is usually reached on (an editable default, not a reading). */
export const DRIVERS = [
  { id: "pymavlink", label: "pymavlink", uri: "udpin:0.0.0.0:14550",
    hint: "ArduPilot or PX4 over MAVLink: a real aircraft, SITL, or a Mission Planner forward" },
  { id: "mavsdk", label: "mavsdk", uri: "udp://:14540", hint: "PX4 through MAVSDK" },
  { id: "mock", label: "mock", uri: "mock://", hint: "The built-in simulator. Nothing it reports is measured." },
];

/** Audit events that belong to a flight, for the Events tab. */
export const FLIGHT_EVENTS = new Set([
  "vehicle_connected", "vehicle_command", "mission_uploaded", "preflight_run", "preflight_override",
  "manual_control_requested", "camera_command", "flight_log_exported", "flight_state_discarded",
  "mission_resumed", "linked_progress_checked", "mission_version_loaded_for_flight",
  "boundary_corner_marked", "boundary_closed", "boundary_marks_cleared",
]);

/* Below this share of packets received the bridge's own link measure is "degraded". */
const DEGRADED_BELOW_PCT = 80;

/* MAV_CMD_NAV_WAYPOINT: the items a pilot counts as waypoints. */
const MAV_CMD_NAV_WAYPOINT = 16;

const finite = (value) => typeof value === "number" && Number.isFinite(value);

/** The link as the station sees it: "none" | "ok" | "degraded" | "lost". */
export function linkState(telemetry) {
  if (!telemetry) return "none";
  if (telemetry.link_lost || telemetry.stale) return "lost";
  if (!telemetry.connected) return "none";
  // Degraded is measured (packets lost on the link); a link with no measure is not
  // called degraded.
  if (finite(telemetry.link_quality_pct) && telemetry.link_quality_pct < DEGRADED_BELOW_PCT) return "degraded";
  return "ok";
}

/** Preflight rows and their counts, straight from run_preflight. */
export function preflight(result) {
  if (!result || !Array.isArray(result.checks)) return null;
  const rows = result.checks.map((check) => {
    let tone = "ok";
    if (check.severity === "blocking" && !check.confirmed) tone = "danger";
    else if (check.requires_manual && !check.confirmed) tone = "warn";
    else if (check.severity === "warning") tone = "warn";
    return {
      id: check.id,
      name: check.name || check.id,
      message: check.message || "",
      fix: check.fix_action || "",
      manual: Boolean(check.requires_manual),
      confirmed: Boolean(check.confirmed),
      note: check.operator_note || "",
      severity: check.severity,
      tone,
    };
  });
  // Blocking first, then what waits for the operator, then warnings, then passes: the
  // order a pilot reads a checklist in.
  const rank = { danger: 0, warn: 1, ok: 2 };
  rows.sort((a, b) => rank[a.tone] - rank[b.tone] || Number(b.manual && !b.confirmed) - Number(a.manual && !a.confirmed));
  const count = (fn) => rows.filter(fn).length;
  return {
    rows,
    ok: count((r) => r.tone === "ok"),
    manual: count((r) => r.manual && !r.confirmed),
    blocking: count((r) => r.severity === "blocking" && !r.confirmed),
    warnings: count((r) => r.severity === "warning"),
    canStart: Boolean(result.can_start),
    simulated: result.simulated === true,
  };
}

/** Metres between two {lat, lon} points (haversine on the mean Earth radius). */
export function metresBetween(a, b) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * Where the aircraft is along the mission it acknowledged.
 * `uploaded` is vehicle_mission().uploaded (positional items with their seq);
 * `telemetry.waypoint_index` is the item seq the vehicle reports as current.
 * Returns flown / remaining point lists and along-route distances, or null.
 */
export function progress(uploaded, telemetry) {
  if (!Array.isArray(uploaded) || !uploaded.length) return null;
  const reported = Boolean(telemetry && (telemetry.connected || telemetry.link_lost));
  const current = reported && finite(telemetry.waypoint_index) ? telemetry.waypoint_index : null;
  const total = reported && finite(telemetry.waypoint_total) ? telemetry.waypoint_total : null;
  const position = reported && finite(telemetry.latitude) && finite(telemetry.longitude)
    && (telemetry.latitude || telemetry.longitude)
    ? { lat: telemetry.latitude, lon: telemetry.longitude } : null;
  const started = current !== null && current > 0;
  const nextIndex = started ? uploaded.findIndex((p) => p.seq >= current) : 0;
  const next = nextIndex < 0 ? uploaded.length : nextIndex;
  const flown = uploaded.slice(0, next);
  const remaining = uploaded.slice(next);
  const along = (points) => points.reduce((sum, p, i) => (i ? sum + metresBetween(points[i - 1], p) : sum), 0);
  const routeLength = along(uploaded);
  const ahead = along(remaining);
  const toNext = position && remaining.length && started ? metresBetween(position, remaining[0]) : 0;
  const left = ahead + toNext;
  /* The aircraft counts mission items (MISSION_CURRENT): a survey route carries camera
     and gimbal items between its waypoints, so "item 9 / 2283" is the vehicle's own
     count and "760 wp" the plan's. The operator flies waypoints, so the acknowledged
     NAV_WAYPOINT items give "WP n / N" from the same answer; null when none were sent. */
  const waypoints = uploaded.filter((p) => p.command === MAV_CMD_NAV_WAYPOINT);
  const wpTotal = waypoints.length || null;
  let wpCurrent = null;
  if (wpTotal !== null && current !== null) {
    const nextWp = started ? waypoints.findIndex((p) => p.seq >= current) : 0;
    wpCurrent = !started ? 0 : nextWp < 0 ? wpTotal : nextWp + 1;
  }
  return {
    current, total, wpCurrent, wpTotal, flown, remaining, position, started,
    routeLength,
    flownMetres: Math.max(0, routeLength - left),
    leftMetres: left,
    fraction: routeLength > 0 ? Math.min(1, Math.max(0, (routeLength - left) / routeLength)) : null,
    firstSeq: uploaded[0].seq,
    lastSeq: uploaded[uploaded.length - 1].seq,
    points: uploaded.length,
  };
}

/** What the command bar may offer, from what the vehicle and the session report. */
export function commandState(telemetry, session, mission) {
  const lost = linkState(telemetry) === "lost";
  const connected = Boolean(telemetry && telemetry.connected);
  const armed = Boolean(telemetry && telemetry.armed);
  const status = session && session.state ? session.state.status : null;
  const mode = telemetry && telemetry.flight_mode ? String(telemetry.flight_mode) : null;
  const paused = status === "paused";
  const flying = ["starting", "flying", "paused", "rth"].includes(status) || (armed && mode === "AUTO");
  const onVehicle = Boolean(mission && mission.on_vehicle);
  return {
    connected, lost, armed, mode, status, paused, flying, onVehicle,
    canArm: connected && !armed,
    canDisarm: connected && armed && !flying,
    canStart: connected && onVehicle && !flying,
    canPause: connected && flying && !paused,
    canResume: connected && paused,
    canReturn: connected || lost,
    canAbort: connected || lost,
    canUpload: connected && Boolean(mission && mission.planned),
  };
}

/** Ask every call in `calls`; resolves {method: result | null}. */
export async function read(calls) {
  const answers = await Promise.all(calls.map((c) => tryCall(c.method, ...(c.args || []))));
  const results = {};
  calls.forEach((c, i) => { results[c.method] = answers[i]; });
  return results;
}

/** Turn the Api results (keyed by method) into what the renderers draw. */
export function derive(results) {
  const telemetryEnvelope = results.telemetry || null;
  const telemetry = telemetryEnvelope ? telemetryEnvelope.telemetry || null : null;
  const state = results.get_state || null;
  const mission = results.vehicle_mission || null;
  const sessionEnvelope = results.flight_session || null;
  const session = sessionEnvelope ? sessionEnvelope.session || null : null;
  const control = results.control_state ? results.control_state.state || null : null;
  const link = state && state.vehicle ? state.vehicle : null;
  return {
    telemetry,
    link,
    connected: Boolean(telemetry && (telemetry.connected || telemetry.link_lost)),
    simulated: Boolean((telemetry && telemetry.is_simulated) || (link && link.is_simulated)),
    linkState: linkState(telemetry),
    control,
    session,
    sessionReason: sessionEnvelope ? sessionEnvelope.reason || "" : "",
    mission,
    planned: mission ? mission.planned : null,
    preflight: preflight(results.run_preflight),
    interrupted: results.check_interrupted_flight || null,
    versions: results.list_mission_versions ? results.list_mission_versions.versions || [] : [],
    unread: results.list_notifications ? results.list_notifications.unread ?? null : null,
    notifications: results.list_notifications ? results.list_notifications.notifications || [] : [],
    project: state ? state.project || null : null,
    progress: progress(mission ? mission.uploaded : null, telemetry),
    commands: commandState(telemetry, session, mission),
  };
}

/** The Events tab: flight entries of the project's audit log, re-read every few seconds. */
export function eventsPanel(render) {
  return live({
    calls: [() => ["audit_log", CALLS.find((c) => c.method === "audit_log").args[0]]],
    every: SLOW_MS,
    empty: "Nothing recorded yet for this project.",
    isEmpty: ([answer]) => !answer || !(answer.events || []).some((e) => FLIGHT_EVENTS.has(e.event_type || e.action || e.event)),
    render: ([answer]) => render((answer.events || []).filter((e) => FLIGHT_EVENTS.has(e.event_type || e.action || e.event))),
  });
}

/** The SESSION section's state line: the flight session and any interrupted flight. */
export function sessionPanel(render) {
  return live({
    calls: ["flight_session", "check_interrupted_flight"],
    every: SLOW_MS,
    empty: "No flight session: connect a vehicle to start one.",
    isEmpty: () => false,
    render: ([session, interrupted]) => render(session, interrupted),
  });
}

/**
 * The vehicle's messages, kept across polls: a growing list read with `since`, so a
 * poll carries only what is new. `onMessages(all, fresh)` after each read.
 */
export function messageFeed(onMessages) {
  let since = 0;
  const all = [];
  return async function poll() {
    const answer = await tryCall("vehicle_messages", since);
    if (!answer) return all;
    const fresh = answer.messages || [];
    if (answer.reset) all.length = 0;   // a new connection started a new log
    since = answer.last_seq ?? since;
    all.push(...fresh);
    if (all.length > MESSAGES_SHOWN) all.splice(0, all.length - MESSAGES_SHOWN);
    onMessages(all, fresh, answer);
    return all;
  };
}

const MESSAGES_SHOWN = 300;

/* ------------------------------------------------------------- formatting
   How a reported value is written. None of these make a value up: each returns null
   for a missing input, and the renderers draw null as "not reported". */

export const isNum = finite;

/** A number to `digits` places, or null. */
export function fixed(value, digits = 1) {
  return finite(value) ? value.toFixed(digits) : null;
}

/** Metres, or kilometres from one kilometre up. */
export function distance(metres) {
  if (!finite(metres)) return null;
  return metres >= 1000 ? `${(metres / 1000).toFixed(2)} km` : `${metres.toFixed(1)} m`;
}

/** Seconds as s, min s, or h min. */
export function duration(seconds) {
  if (!finite(seconds) || seconds < 0) return null;
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}

/** mm:ss, for a heartbeat age. */
export function clock(seconds) {
  if (!finite(seconds) || seconds < 0) return null;
  const s = Math.floor(seconds);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/** Degrees with the hemisphere letter. */
export function latitude(value) {
  return finite(value) ? `${Math.abs(value).toFixed(5)}° ${value >= 0 ? "N" : "S"}` : null;
}
export function longitude(value) {
  return finite(value) ? `${Math.abs(value).toFixed(5)}° ${value >= 0 ? "E" : "W"}` : null;
}

/** Whole degrees, three digits. */
export function heading(value) {
  return finite(value) ? `${String(Math.round(((value % 360) + 360) % 360)).padStart(3, "0")}°` : null;
}

/** GPS_RAW_INT fix_type as a pilot says it. */
export const GPS_FIX = { 0: "no GPS", 1: "no fix", 2: "2D fix", 3: "3D fix", 4: "DGPS", 5: "RTK float", 6: "RTK fixed" };

/** hh:mm:ss from an ISO time, in the operator's clock. */
export function timeOfDay(iso) {
  const date = iso ? new Date(iso) : null;
  if (!date || Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

/** MAV_SEVERITY to a status tone. */
export function severityTone(severity) {
  if (!finite(severity)) return "idle";
  if (severity <= 3) return "danger";
  if (severity === 4) return "warn";
  return "ok";
}
