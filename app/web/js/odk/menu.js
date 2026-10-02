/* The native menu bar, bridged into the v1 interface.
 *
 * app/shell.py builds the operating-system menu and dispatches every item into the page
 * as `window.odk.onMenu({action, payload})`. Every id it sends has an entry in MENU
 * below; tests/test_menu_coverage.py reads both lists and fails when they drift, then
 * presses every item against the real Api in a browser and fails on any
 * "not handled".
 *
 * An entry is declarative where it can be:
 *   { open: "<ws>" }               switch to the workspace the item belongs to first,
 *                                  so the operator sees what it did;
 *   { action: "<ACTIONS key>" }    run the SAME verb a button runs (with the payload),
 *                                  never a second implementation of it;
 *   { handler: "<name>" }          a menu item that is not one verb: a function in
 *                                  HANDLERS that reaches the Api or says why it cannot.
 */

import { call, connected, tryCall } from "../workspace/api.js";
import { setBasemap, basemapNames, visibleBounds, fitTo } from "../workspace/mapview.js";
import { currentMap } from "../workspace/draw.js";
import { has } from "./registry.js";

export const MENU = {
  /* File */
  "project.new": { action: "New Project", open: "home" },
  "project.open": { action: "Open", open: "home" },
  "project.reveal": { handler: "revealProject" },
  "data.import_imagery": { action: "Import", open: "home" },
  "data.import_terrain": { handler: "importTerrain", open: "plan" },
  "data.import_vector": { action: "Add Layer", open: "analyse" },
  "data.import_raster": { action: "Add Layer", open: "analyse" },

  /* Mission */
  "mission.plan": { action: "Plan", open: "plan" },
  "mission.settings": { handler: "missionSettings", open: "plan" },
  "mission.clear_aoi": { handler: "clearAoi", open: "plan" },
  "mission.clear_nofly": { handler: "clearNoFly", open: "plan" },
  "mission.save": { action: "Save", open: "plan" },
  "mission.history": { action: "History", open: "plan" },
  "mission.export_all": { action: "Export", open: "plan" },
  "mission.export": { handler: "exportMission", open: "plan" },

  /* Fly */
  "fly.connect": { action: "Connect", open: "fly" },
  "fly.disconnect": { action: "Disconnect", open: "fly" },
  "fly.upload": { action: "Upload", open: "fly" },
  "fly.command": { handler: "vehicleCommand", open: "fly" },

  /* Analysis */
  "analysis.pipeline": { action: "Start", open: "process" },
  "analysis.models": { action: "Verify Models", open: "settings" },

  /* Reconstruct */
  "recon.run": { action: "Process", open: "process" },
  "recon.settings": { handler: "reconSettings", open: "process" },
  "recon.reveal": { handler: "revealProject" },

  /* Reports */
  "report.generate": { action: "Generate Report", open: "report" },
  "report.export": { handler: "exportFindings", open: "report" },
  "report.list": { action: "Open Report", open: "report" },
  "report.pipeline": { action: "Inspection Report", open: "report" },
  "report.finding": { action: "Record Finding", open: "review" },
  "report.review": { handler: "review", open: "review" },
  "report.provenance": { handler: "provenance", open: "report" },
  "report.audit_export": { handler: "auditExport", open: "report" },
  "report.notifications": { handler: "notifications", open: "home" },

  /* View */
  "view.panel": { handler: "panel" },
  "view.basemap": { handler: "basemap" },
  "view.cache_tiles": { handler: "cacheTiles" },
  "view.zoom_aoi": { handler: "zoomAoi" },
  "view.zoom_mission": { handler: "zoomMission" },
  "view.field_mode": { handler: "fieldMode" },
  "view.settings": { handler: "settings" },
  "view.palette": { handler: "palette" },
  "view.toggle_panels": { handler: "togglePanels" },

  /* Tools */
  "tools.measure": { handler: "measure", open: "analyse" },
  "tools.capabilities": { handler: "capabilities", open: "settings" },
  "tools.audit": { handler: "audit" },

  /* Help */
  "help.about": { handler: "about", open: "settings" },
  "help.shortcuts": { handler: "shortcuts" },
};

/* The View menu's panels, and the screen each now lives on (REBUILD_PLAN 2.7). */
export const PANEL_SCREEN = { layers: "analyse", mission: "plan", telemetry: "fly", jobs: "process" };

/* Each vehicle verb asks first, in words that name what it does. */
export const VEHICLE_QUESTION = {
  arm: "Arm the aircraft?",
  start: "Start the uploaded mission?",
  pause: "Pause the aircraft?",
  resume: "Resume the mission?",
  rth: "Return to home?",
  land: "Land now?",
  abort: "ABORT the flight?",
};

const LABELS = {
  qgc_plan: "QGroundControl .plan", qgc_wpl: "QGC WPL .waypoints", dji_wpml: "DJI WPML .kmz",
  litchi: "Litchi CSV", kml: "KML", geojson: "GeoJSON",
};

/** One audit event as a line: time, what happened, and the first few payload fields. */
export function auditLine(e) {
  let detail = "";
  try {
    const payload = typeof e.payload_json === "string" ? JSON.parse(e.payload_json || "{}") : (e.payload_json || {});
    detail = Object.entries(payload || {})
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .slice(0, 4)
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v).slice(0, 60) : v}`)
      .join(" ");
  } catch { detail = String(e.payload_json || ""); }
  const kind = String(e.event_type || e.action || e.message || "").replace(/_/g, " ");
  const when = String(e.created_at || e.at || e.timestamp || "").slice(0, 19).replace("T", " ") || "time not recorded";
  return `${when}  ${[kind, detail].filter(Boolean).join(" — ") || "(unnamed event)"}`;
}

async function projectRoot() {
  const state = await tryCall("get_state");
  return state && state.project && (state.project.root_dir || state.project.path);
}

/** Menu items that are not one verb. Each reaches the Api or says why it cannot. */
export const HANDLERS = {
  async revealProject(shell) {
    const root = await projectRoot();
    if (!root) return shell.toast("Open project folder: no project is open.", "warn");
    await call("open_path", String(root));
    return shell.toast(`Opened ${root}`, "ok");
  },

  async importTerrain(shell) {
    const file = await call("pick_file", ["tif", "tiff", "asc", "csv"]);
    if (!file || !file.path) {
      return shell.toast("Import terrain: no file chosen; the terrain source is unchanged.", "warn");
    }
    const result = await call("set_terrain_source", file.path);
    const coverage = result && result.coverage;
    shell.refreshLive();
    if (!coverage) {
      return shell.toast(`Terrain source: ${result.path}. Draw the area of interest to check that `
        + "it covers the site; Generate will follow it.", "ok");
    }
    if (coverage.covered) {
      return shell.toast(`Terrain source: ${result.path}. ${coverage.detail || "It covers the area."}`, "ok");
    }
    return shell.toast(`Terrain source: ${result.path}, but ${coverage.detail
      || "it does not cover the area of interest."}`, "warn");
  },

  missionSettings(shell) {
    return shell.toast("Mission settings: the Plan inspector's PARAMETERS. Generate uses what is set there.", "ok");
  },

  reconSettings(shell) {
    return shell.toast("Reconstruction settings: the Process ENGINE section and the New job dialog.", "ok");
  },

  async clearAoi(shell) {
    await call("set_aoi", []);
    await shell.refreshState();
    shell.refreshLive();
    return shell.toast("Area of interest cleared.", "ok");
  },

  async clearNoFly(shell) {
    await call("set_no_fly_zones", []);
    await shell.refreshState();
    shell.refreshLive();
    return shell.toast("No-fly zones cleared.", "ok");
  },

  async exportMission(shell, payload) {
    const format = String(payload.format || "");
    if (!format) return shell.runAction("Export");
    const result = await call("export_mission", [format]);
    const written = Object.entries((result && result.written) || {});
    const failed = written.filter(([, v]) => String(v).startsWith("failed"));
    const files = written.filter(([, v]) => !String(v).startsWith("failed")).map(([, v]) => v);
    const label = LABELS[format] || format;
    if (!files.length) {
      return shell.toast(`Export ${label}: ${failed.map(([, v]) => v).join("; ") || "nothing was written."}`,
        "danger");
    }
    return shell.toast(`Exported ${label}: ${files.join(", ")}`
      + (failed.length ? `; ${failed.map(([k, v]) => `${k} ${v}`).join("; ")}` : ""), failed.length ? "warn" : "ok");
  },

  async vehicleCommand(shell, payload) {
    const command = String(payload.command || "");
    if (!command) return shell.toast("Vehicle command: the menu sent no command.", "warn");
    if (!connected()) return shell.toast(`${command}: the application is not connected.`, "warn");
    const question = VEHICLE_QUESTION[command] || `Send "${command}" to the aircraft?`;
    if (!(await shell.actionContext().confirm(question))) return null;
    /* What the vehicle answered, not that something was sent. */
    const result = await shell.sendVehicleCommand(command);
    if (!result) return shell.toast(`${command}: not started; preflight is not ready.`, "warn");
    const detail = result && result.result;
    await shell.refreshState();
    return shell.toast(`${command}: ${(detail && detail.message) || "accepted by the vehicle."}`, "ok");
  },

  async exportFindings(shell, payload) {
    const format = String(payload.format || "pdf").toLowerCase();
    const formats = await call("report_formats");
    if (!(formats.formats || []).includes(format)) {
      return shell.toast(`Export findings as ${format.toUpperCase()}: this build cannot write ${format} `
        + `(available: ${(formats.formats || []).join(", ") || "none"}).`, "warn");
    }
    const saved = await tryCall("report_templates");
    const last = (saved && saved.templates && saved.templates[0]) || {};
    const result = await call("export_report", format, last.title || "Inspection findings",
      last.organization || "", last.client || "", last.author || "", "", last.logo_path || "");
    return shell.toast(`Findings ${format.toUpperCase()}: ${result.findings} finding(s) written to ${result.path}`, "ok");
  },

  review(shell) {
    return shell.toast("Findings and review: pick a finding in the review queue, then Accept, Reject or Flag.", "ok");
  },

  async provenance(shell) {
    const result = await call("verify_provenance");
    const lines = [
      `Folder: ${result.directory || ""}`,
      `Verified: ${(result.verified || []).length} of ${result.total || 0}`,
      `Unattributed: ${(result.unattributed || []).length}`,
      `Modified: ${(result.modified || []).length}`,
      ...Object.entries(result.results || {}).map(([name, r]) => `${name}: ${r.status}`),
    ];
    return shell.reveal("Provenance", lines.join("\n"));
  },

  async auditExport(shell) {
    const result = await call("export_audit_log", "csv");
    return shell.toast(`Audit log: ${result.events} event(s) written to ${result.path}.`, "ok");
  },

  async notifications(shell) {
    const result = await call("list_notifications");
    const lines = (result.notifications || []).map((n) =>
      `${String(n.created_utc || "").slice(0, 19)}  ${n.read ? " " : "•"} ${n.title}${n.detail ? ` — ${n.detail}` : ""}`);
    return shell.reveal(`Notifications (${result.unread || 0} unread)`, lines.join("\n") || "No notifications.");
  },

  async panel(shell, payload) {
    const target = PANEL_SCREEN[payload.panel];
    if (!target) return shell.toast(`View: there is no ${payload.panel || "unnamed"} panel.`, "warn");
    await shell.open(target);
    return shell.toast(`${payload.panel} panel: ${shell.screenLabel(target)}.`, "ok");
  },

  basemap(shell, payload) {
    const key = String(payload.basemap || "");
    if (!setBasemap(key)) {
      return shell.toast(`Unknown basemap ${key}. Available: ${basemapNames().join(", ")}.`, "warn");
    }
    shell.noteBasemap(key);
    return shell.toast(`Basemap: ${key}.`, "ok");
  },

  async cacheTiles(shell) {
    const view = visibleBounds();
    if (!view) return shell.toast("Cache map: open a screen with a map to choose the area to cache.", "warn");
    const minZoom = Math.max(0, Math.floor(view.zoom));
    const maxZoom = Math.min(19, minZoom + 3);
    const result = await call("cache_basemap_area", view.west, view.south, view.east, view.north,
      minZoom, maxZoom, "satellite");
    shell.noteNetworkUse("basemap cache download");
    shell.toast(`Caching ${result.tiles} tiles (z${minZoom}-${maxZoom}) for offline use…`, "running");
    return shell.watchJob(result.job_id);
  },

  async zoomAoi(shell) {
    const state = await tryCall("get_state");
    const ring = (state && state.aoi) || [];
    if (ring.length < 3) return shell.toast("Zoom to area: no area of interest is drawn.", "warn");
    return shell.fitMap(ring, "area of interest");
  },

  async zoomMission(shell) {
    const geo = await tryCall("mission_geojson");
    const coords = [];
    const walk = (c) => { if (typeof c[0] === "number") coords.push(c); else c.forEach(walk); };
    for (const f of (geo && (geo.geojson || geo).features) || []) {
      if (f && f.geometry && f.geometry.coordinates) walk(f.geometry.coordinates);
    }
    if (!coords.length) return shell.toast("Zoom to mission: no mission is planned.", "warn");
    return shell.fitMap(coords, "mission");
  },

  fieldMode(shell) {
    return shell.setMode(shell.mode === "field" ? "desk" : "field");
  },

  settings(shell) {
    return shell.open("settings");
  },

  palette(shell) {
    return shell.openPalette();
  },

  togglePanels(shell) {
    return shell.toggleSidePanels();
  },

  measure(shell, payload) {
    return shell.runAction(payload.mode === "area" ? "Area" : "Distance");
  },

  async capabilities(shell) {
    const result = await call("capabilities");
    const caps = (result && result.capabilities) || result || {};
    const lines = Object.entries(caps)
      .filter(([key]) => key !== "ok")
      .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`);
    return shell.reveal("Environment capabilities", lines.join("\n") || "Nothing reported.");
  },

  async audit(shell) {
    const log = await call("audit_log", 100);
    const lines = ((log && log.events) || []).map(auditLine);
    return shell.reveal("Audit log (newest first)", lines.join("\n") || "No events recorded for this project.");
  },

  async about(shell) {
    const state = await tryCall("get_state");
    const version = state && state.version;
    return shell.toast(`OpenDroneKit${version ? ` ${version}` : ""}: offline-first drone survey and inspection.`, "ok");
  },

  shortcuts(shell) {
    return shell.showShortcuts();
  },
};

/** Run one menu item. Never throws: a failure is said in a toast. */
export async function dispatchMenu(shell, action, payload = {}) {
  const route = MENU[action];
  if (!route) {
    // Named rather than ignored: a menu item nobody wired says so.
    shell.toast(`Menu action not handled here: ${action}`, "warn");
    return { error: `unhandled ${action}` };
  }
  try {
    if (route.open) await shell.open(route.open);
    if (route.action) {
      if (!has(route.action)) throw new Error(`no ACTIONS key "${route.action}"`);
      return await shell.runAction(route.action, payload);
    }
    return await HANDLERS[route.handler](shell, payload || {});
  } catch (error) {
    shell.toast(`${action}: ${error.message || error}`, "danger");
    return { error: String(error.message || error) };
  }
}

/** window.odk.onMenu, for app/shell.py's evaluate_js. */
export function installMenu(shell) {
  window.odk = window.odk || {};
  window.odk.onMenu = ({ action, payload } = {}) => {
    if (!action) return Promise.resolve(null);
    // pywebview's evaluate_js serialises what the page returns: hand back plain data,
    // never a live object (the palette handler resolves to the Palette, which is circular).
    return dispatchMenu(shell, action, payload || {}).then((result) => {
      try { return result === undefined ? null : JSON.parse(JSON.stringify(result)); } catch { return null; }
    });
  };
  window.odk.menuIds = () => Object.keys(MENU);
}

/* `fitTo` and `currentMap` are re-exported for the shell's fitMap(). */
export { fitTo, currentMap };
