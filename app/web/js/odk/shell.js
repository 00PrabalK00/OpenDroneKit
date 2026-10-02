/* The v1 shell: one frame, nine screens, two modes, every verb.
 *
 * The frozen contract (docs/ui/ARCHITECTURE.md, "Shell contract"). Browser harnesses
 * drive the application only through these members, so a ported cockpit test changes
 * its import and nothing else:
 *
 *   new Shell(root)        mounts the frame into root, reads the persisted mode
 *   open(id)               opens a screen; accepts new ids and every old cockpit id;
 *                          resolves (true) once the screen has rendered
 *   runAction(key, p)      runs a merged-registry ACTIONS key with actionContext();
 *                          resolves to the action's result, {error} for an unknown
 *                          key, and never throws
 *   actionContext()        the ctx every action receives (members listed at the method)
 *   toast(text, kind)      kind ok | warn | danger | running (old "info"/"error" too);
 *                          returns the node
 *   confirmAll             true: ctx.confirm() answers yes with no dialog (tests only)
 *   refreshState()         re-reads get_state and re-settles every live panel
 *   setMode(mode)          "desk" | "field": <html data-mode>, localStorage odk.mode,
 *                          the frame rebuilt for the mode, "odk:mode" for the map theme
 *
 * Also: shell.drawing = draw.js's addVertex / finishDrawing / canDraw, and the image
 * view keeps the cockpit's img.canvas-product.
 *
 * Every value the frame shows -- project, CRS, cursor, scale, vehicle, jobs, network --
 * is read from the Api or from the map, or is the word for its absence. Nothing is
 * filled in to look complete.
 */

import { el, showToast, popover, simulatedTag, selection } from "./components.js";
import { Frame } from "./frame.js";
import { Palette } from "./palette.js";
import { installMenu, dispatchMenu } from "./menu.js";
import { ACTIONS, UNWIRED, describeMeasurement, prerequisite, resolve, has } from "./registry.js";
import { DESK_SCREENS, FIELD_SCREENS, SETTINGS, WORKSPACE_ALIASES } from "./screens/index.js";
import { call, connected, lastError, tryCall, whenReady } from "../workspace/api.js";
import { DEMO, SAMPLE_SENTINELS, demoEnabled } from "../workspace/demo.js";
import { FORM_VALUES } from "../workspace/primitives.js";
import {
  addVertex, armDrawing, armTool, bindDrawingKeys, canDraw, configureDrawing, currentMap,
  finishDrawing, formatArea, formatLength, recordMeasurement, stopDrawing, undoPoint, vertices,
} from "../workspace/draw.js";
import { fitTo, refreshProjectOverlays } from "../workspace/mapview.js";
import { currentImage, currentView, onViewChange, setImage, setView } from "../workspace/viewstate.js";
import { ask, askOne, choose as chooseModal, confirmAsk, labelBoxes, reveal, showResult } from "../workspace/modal.js";
import { refreshLive } from "../workspace/live.js";
import { armModelTool, configureViewer, currentViewer } from "../workspace/viewer3d.js";

export const MODE_KEY = "odk.mode";
export const WORKSPACE_KEY = "odk.workspace";
const MODES = new Set(["desk", "field"]);

const ALL_SCREENS = [...DESK_SCREENS, SETTINGS];
const SCREEN_BY_ID = Object.fromEntries(ALL_SCREENS.map((s) => [s.id, s]));
const FIELD_IDS = new Set(FIELD_SCREENS.map((s) => s.id));

/** The canonical screen id for a new or old (cockpit) id, or null. */
export function canonicalScreen(id) {
  const key = String(id || "");
  if (SCREEN_BY_ID[key]) return key;
  return WORKSPACE_ALIASES[key] || null;
}

/* Every shortcut listed here is bound in onKey() and runs from the palette. A shortcut
   list is a claim about the program; the cockpit once advertised four that did nothing. */
const SHORTCUTS = [
  { keys: "Ctrl/⌘ K", what: "Command palette", name: "palette" },
  { keys: "/", what: "Search", name: "search" },
  { keys: "1 – 8", what: "Switch workspace (Field: 1 – 5)", name: null },
  { keys: "Ctrl/⌘ ,", what: "Settings", name: "settings" },
  { keys: "Ctrl/⌘ Shift F", what: "Switch Desk / Field mode", name: "mode" },
  { keys: "Ctrl/⌘ B", what: "Hide or show both side panels", name: "sides" },
  { keys: "F11", what: "Full-screen canvas", name: "full" },
  { keys: "F", what: "Fit the map to the project", name: "fit" },
  { keys: "Ctrl/⌘ S", what: "Save the planned mission", name: "save" },
  { keys: "Ctrl/⌘ Z", what: "Undo the last drawn point", name: "undo" },
  { keys: "Enter", what: "Finish the shape being drawn", name: null },
  { keys: "Esc", what: "Stop drawing / close", name: "stop" },
  { keys: "?", what: "Keyboard shortcuts", name: "help" },
];

/* The mission form's field keys and the names plan_mission reads (cockpit shell.js,
   kept as one explicit map: the short names are what a panel shows, the long ones the
   Api contract). A screen that owns the mission form may replace the whole builder with
   shell.provide("missionOptions", fn). */
const MISSION_OPTION_KEYS = {
  alt: "altitude_m", fwd: "front_overlap_pct", side: "side_overlap_pct", speed: "speed_m_s",
  standoff: "standoff_m", rth: "rth_altitude_m", camera: "camera",
  thermal_camera: "thermal_camera", target_gsd_cm: "target_gsd_cm", rgb_camera: "rgb_camera",
  payload_key: "payload_key", calibration_panel_lonlat: "calibration_panel_lonlat",
  accept_relative_values: "accept_relative_values", pylon_elements: "pylon_elements",
  dome_radius_m: "dome_radius_m", box_altitude_levels_m: "box_altitude_levels_m",
  facade_bottom_altitude_m: "facade_bottom_altitude_m", facade_top_altitude_m: "facade_top_altitude_m",
  tower_bottom_altitude_m: "tower_bottom_altitude_m", tower_top_altitude_m: "tower_top_altitude_m",
  tower_object_radius_m: "tower_object_radius_m", turbine_rotor_facing_deg: "turbine_rotor_facing_deg",
  turbine_hub_height_m: "turbine_hub_height_m", turbine_blade_length_m: "turbine_blade_length_m",
  turbine_nacelle_length_m: "turbine_nacelle_length_m", turbine_blade_angles_deg: "turbine_blade_angles_deg",
  linked_version_ids: "linked_version_ids",
};
/* Shown on the form, read by no planner option: sent nowhere rather than under an
   invented name (gsd is solved through altitude_for_gsd by the Plan screen). */
const MISSION_FIELDS_NOT_PLANNED = new Set(["gsd", "angle", "air", "batt"]);
/* The pilot-facing mission labels and the template ids the planner takes. */
const MISSION_TEMPLATES = {
  "2D mapping": "grid", "3D modelling": "double_grid", "Roof mapping": "grid",
  "Roof inspection": "roof_inspection", "Facade mapping": "facade_mapping", "Facade inspection": "facade",
  "Multi-facade": "multi_facade", "Closed loop": "closed_loop", "Linear mapping": "corridor",
  "Linear inspection": "linear_inspection", "Tower mapping": "tower_mapping", Orbit: "orbit",
  Waypoints: "waypoints", Panorama: "panorama", "Solar inspection": "solar_inspection",
  "Wind turbine": "wind_turbine", "Magnetic mapping": "magnetic_mapping", "Complex facade": "multi_facade",
  "L-shaped": "grid", "Utility pylon": "pylon_inspection", Thermal: "thermal", Multispectral: "multispectral",
  "Dome inspection": "dome_inspection", "Box inspection": "box_inspection", "360 bubble": "bubble_360",
  "Lateral capture": "lateral_capture", "Smart adaptive": "smart_adaptive", "Linked mission": "linked_mission",
};

/** What a finished job produced, in the words a user acts on (ported from the cockpit). */
function jobOutcome(result) {
  if (!result || typeof result !== "object") return "";
  const parts = [];
  if (result.findings_created != null) parts.push(`${result.findings_created} model finding(s) to review`);
  if (result.heuristic_regions_not_recorded) {
    parts.push(`${result.heuristic_regions_not_recorded} heuristic region(s) not recorded as findings`);
  }
  for (const [name, count] of Object.entries(result.heuristic_images_not_scored || {})) {
    if (count) parts.push(`${name} ran as a heuristic on ${count} image(s), not scored`);
  }
  for (const [name, info] of Object.entries(result.detectors_refused || {})) {
    if (!info || !info.images) continue;
    const reasons = Array.isArray(info.reasons) && info.reasons.length ? info.reasons
      : [{ images: info.images, reason: info.reason }];
    for (const r of reasons) parts.push(`${name} refused on ${r.images} image(s): ${r.reason}`);
  }
  for (const r of result.surface_models_not_run || []) {
    if (r && r.images) parts.push(`crack/structural models not run on ${r.images} image(s): ${r.reason}`);
  }
  if (result.health_not_assessed) parts.push("no health score: no defect model read these images");
  if (result.reconstruction_status === "failed") parts.push("reconstruction failed; detection results kept");
  if (result.report_markdown_path) parts.push("inspection report written (Report › Inspection report)");
  if (result.metrics && typeof result.metrics.map50 === "number") {
    parts.push(`validation mAP50 ${result.metrics.map50.toFixed(3)}`);
  }
  if (result.parity_ok === true && result.key) parts.push(`installed as ${result.key}`);
  if (result.coverage_summary) parts.push(String(result.coverage_summary));
  return parts.length ? `: ${parts.join("; ")}` : "";
}

/* ------------------------------------------------------- map arithmetic */

/** "41.30410° N 81.75212° W" */
export function formatLonLat(lon, lat) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}° ${ns} ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

/** WGS84 -> UTM easting/northing in a given zone (Snyder / USGS series, deterministic). */
export function toUtm(lon, lat, zone, south = false) {
  const a = 6378137.0;
  const f = 1 / 298.257223563;
  const k0 = 0.9996;
  const e2 = f * (2 - f);
  const ep2 = e2 / (1 - e2);
  const rad = Math.PI / 180;
  const phi = lat * rad;
  const lambda0 = ((zone - 1) * 6 - 180 + 3) * rad;
  const N = a / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  const T = Math.tan(phi) ** 2;
  const C = ep2 * Math.cos(phi) ** 2;
  const A = Math.cos(phi) * (lon * rad - lambda0);
  const M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
    - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * phi)
    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * phi)
    - (35 * e2 ** 3 / 3072) * Math.sin(6 * phi));
  const easting = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    + 500000;
  let northing = k0 * (M + N * Math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
    + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720));
  if (south) northing += 10000000;
  return { easting, northing };
}

/** The UTM zone an EPSG code names (WGS 84 / UTM, 326zz north, 327zz south), or null. */
export function utmZoneOf(epsg) {
  const code = Number(epsg);
  if (code >= 32601 && code <= 32660) return { zone: code - 32600, south: false };
  if (code >= 32701 && code <= 32760) return { zone: code - 32700, south: true };
  return null;
}

/** Map scale denominator at 96 dpi for a MapLibre zoom (512 px tiles) and latitude. */
export function scaleDenominator(zoom, lat) {
  const metresPerPixel = (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** zoom);
  return (metresPerPixel * 96) / 0.0254;
}

/** "1 : 1,680" -- three significant figures, which is what a scale is good to. */
export function formatScale(denominator) {
  if (!Number.isFinite(denominator) || denominator <= 0) return null;
  const digits = Math.floor(Math.log10(denominator)) + 1;
  const step = 10 ** Math.max(0, digits - 3);
  return `1 : ${(Math.round(denominator / step) * step).toLocaleString("en-US")}`;
}

/** "WGS 84 / UTM zone 17N" -> "UTM 17N"; other names as they are. */
function shortCrsName(name) {
  const m = /UTM zone (\d+)\s*([NS])/i.exec(String(name || ""));
  return m ? `UTM ${m[1]}${m[2].toUpperCase()}` : String(name || "");
}

function isLoopback(url) {
  try {
    const u = new URL(url, window.location.href);
    if (["data:", "blob:", "about:", "esri:"].includes(u.protocol)) return true;
    return u.origin === window.location.origin || ["127.0.0.1", "localhost", "[::1]"].includes(u.hostname);
  } catch { return true; }
}

/* ================================================================ Shell */

export class Shell {
  constructor(root, { mode } = {}) {
    this.root = root;
    root.classList.add("odk-shell");
    this.settings = FORM_VALUES;
    this.providers = {};
    this.confirmAll = false;
    this.connection = demoEnabled() ? "demo" : "disconnected";
    this.state = null;
    this.project = null;
    this.crs = null;
    this.jobs = [];
    this.vehicle = null;
    this.log = [];
    this.networkUses = new Set();
    this.footerContext = null;
    this.fieldSubtitle = null;
    this.modules = new Map();
    this.renderToken = 0;
    this.drawing = { addVertex, finishDrawing, canDraw };

    const html = document.documentElement.getAttribute("data-mode");
    this.mode = MODES.has(mode) ? mode : MODES.has(html) ? html : "desk";
    document.documentElement.setAttribute("data-mode", this.mode);

    let stored = null;
    try { stored = localStorage.getItem(WORKSPACE_KEY); } catch { /* private window */ }
    this.workspaceId = canonicalScreen(stored) || "home";
    if (this.mode === "field" && !FIELD_IDS.has(this.workspaceId)) this.workspaceId = "plan";
    this.requestedId = this.workspaceId;

    this.palette = new Palette(this);
    this.frame = new Frame(root, {
      desk: DESK_SCREENS,
      field: FIELD_SCREENS,
      onTab: (id) => this.open(id),
      onSettings: () => this.open("settings"),
      onToggleMode: () => this.setMode(this.mode === "field" ? "desk" : "field"),
      onSearch: (text) => this.openPalette(text),
      onProjectMenu: (anchor) => this.projectMenu(anchor),
      onSessionLog: (anchor) => this.sessionLog(anchor),
    });
    this.regions = this.frame.render(this.mode);

    this.bindDrawing();
    this.bindKeys();
    this.watchNetwork();
    installMenu(this);
    selection.on("*", (kind, value) => this.onSelection(kind, value));
    onViewChange(() => this.renderImageView());

    this.updateChrome();
    this.ready = this.open(this.workspaceId);
    this.connect();
  }

  /* ------------------------------------------------------------ screens */

  /** The screens for the current mode, with their number key, for tabs and palette. */
  screens() {
    const list = this.mode === "field" ? FIELD_SCREENS : DESK_SCREENS;
    return [...list.map((s, i) => ({ ...s, key: String(i + 1) })),
      ...(this.mode === "field" ? [] : [{ ...SETTINGS, key: "Ctrl ," }])];
  }

  screenLabel(id) {
    const screen = SCREEN_BY_ID[canonicalScreen(id)];
    return screen ? screen.label : String(id);
  }

  async screenModule(id, mode) {
    const path = mode === "field" ? `./screens/field/${id}.js` : `./screens/${id}/desk.js`;
    if (!this.modules.has(path)) this.modules.set(path, import(path));
    return this.modules.get(path);
  }

  /**
   * Open a screen. New ids and every old cockpit id are accepted. In Field mode a screen
   * with no Field renderer (Home, Review, Fleet, Settings) is opened by switching back to
   * Desk, which is the only place it exists. Resolves true once rendered, false for an
   * id that names no screen.
   */
  async open(id) {
    const target = canonicalScreen(id);
    if (!target) {
      this.toast(`Open: there is no screen called ${id}.`, "warn");
      return false;
    }
    this.requestedId = String(id);
    if (this.mode === "field" && !FIELD_IDS.has(target)) {
      this.workspaceId = target;
      try { localStorage.setItem(WORKSPACE_KEY, target); } catch { /* private window */ }
      await this.setMode("desk");
      this.toast(`${this.screenLabel(target)} has no Field screen, so it opened in Desk mode.`, "ok");
      return true;
    }
    this.workspaceId = target;
    try { localStorage.setItem(WORKSPACE_KEY, target); } catch { /* private window */ }
    // Each screen opens on its own canvas: carrying a photograph or a mesh across a
    // switch shows the previous instrument through the new one.
    setImage(null);
    setView("map");
    this.frame.setActive(target);
    this.footerContext = null;
    this.fieldSubtitle = null;
    await this.renderScreen();
    this.updateChrome();
    setTimeout(() => this.applyBanner(), 0);
    window.dispatchEvent(new CustomEvent("odk:screen", { detail: { id: target, mode: this.mode } }));
    return true;
  }

  /** Draw the current screen into the frame regions. */
  async renderScreen() {
    const token = ++this.renderToken;
    const id = this.workspaceId;
    const mode = this.mode;
    let module;
    try {
      module = await this.screenModule(id, mode);
    } catch (error) {
      this.toast(`${this.screenLabel(id)}: the screen could not load (${error.message || error}).`, "danger");
      return;
    }
    if (token !== this.renderToken) return;
    for (const node of Object.values(this.regions)) node.replaceChildren();
    this.renderImageView();
    try {
      await module.render(this.regions, this.screenContext(id));
    } catch (error) {
      this.toast(`${this.screenLabel(id)}: ${error.message || error}`, "danger");
    }
    if (token === this.renderToken) this.frame.node.setAttribute("data-rendered", id);
  }

  /** What a screen renderer receives besides its regions. */
  screenContext(id) {
    return {
      shell: this,
      screen: id,
      mode: this.mode,
      actions: this.actionContext(),
      /** A screen's own footer segment (desk), e.g. "ALT FRAME relative_to_home". */
      setFooter: (text) => { this.footerContext = text || null; this.updateChrome(); },
      /** The Field header's second line, e.g. "Plan · grid survey". */
      setSubtitle: (text) => { this.fieldSubtitle = text || null; this.updateChrome(); },
      /** Replace a context builder (missionOptions, reconstructionOptions...). */
      provide: (name, fn) => this.provide(name, fn),
    };
  }

  /** Let a screen supply a builder that actions read (e.g. the Plan form's options). */
  provide(name, fn) {
    if (typeof fn === "function") this.providers[name] = fn;
    else delete this.providers[name];
  }

  /* --------------------------------------------------------------- mode */

  async setMode(mode, { persist = true } = {}) {
    if (!MODES.has(mode)) throw new Error(`unknown mode ${mode}`);
    document.documentElement.setAttribute("data-mode", mode);
    if (persist) { try { localStorage.setItem(MODE_KEY, mode); } catch { /* private window */ } }
    const changed = mode !== this.mode;
    this.mode = mode;
    if (mode === "field" && !FIELD_IDS.has(this.workspaceId)) this.workspaceId = "plan";
    if (changed || !this.frame.node) {
      this.palette.close();
      this.closePopovers();
      this.regions = this.frame.render(mode);
    }
    this.frame.setActive(this.workspaceId);
    this.updateChrome();
    await this.renderScreen();
    this.updateChrome();
    setTimeout(() => this.applyBanner(), 0);
    // The map re-applies its paint (map/theme.js listens); maps re-measure.
    window.dispatchEvent(new CustomEvent("odk:mode", { detail: { mode } }));
    window.dispatchEvent(new Event("resize"));
    return mode;
  }

  toggleSidePanels() {
    if (this.mode !== "desk") return false;
    const both = this.frame.collapsed.left && this.frame.collapsed.inspector;
    this.frame.setCollapsed("left", !both);
    this.frame.setCollapsed("inspector", !both);
    return !both;
  }

  /* ----------------------------------------------------------- bridge */

  /** Wait for the bridge; once it answers, read the state and redraw. */
  async connect() {
    const ready = await whenReady();
    if (!ready || !connected()) {
      if (!this.awaitingBridge) {
        this.awaitingBridge = true;
        window.addEventListener("pywebviewready", () => { this.awaitingBridge = false; this.connect(); }, { once: true });
      }
      this.updateChrome();
      if (this.connection === "demo") this.startPolling();
      return;
    }
    if (this.connection === "connected") return;
    if (this.connection !== "demo") this.connection = "connected";
    await this.refreshState();
    // The first render asked before the bridge answered; ask again now.
    await this.renderScreen();
    this.startPolling();
  }

  /** Ask the application what is open. Re-settles every live panel (REFRESH_EVENT). */
  async refreshState() {
    // A layer may have been added or removed: ask for the DSM again on the next move.
    this.elevNoDsm = false;
    const state = await tryCall("get_state");
    this.stateUnread = !state && this.connection === "connected";
    if (this.stateUnread) this.retryStateRead();
    if (state) {
      this.state = {
        project: state.project || state.active_project || null,
        dataset: state.dataset || state.active_dataset || null,
        layers: state.layers || [],
        aoi: state.aoi || [],
        version: state.version || null,
        vehicle: state.vehicle || null,
        dataRoot: (state.data_paths && state.data_paths.data_root) || null,
      };
      this.project = this.state.project;
      const crs = this.project ? await tryCall("get_project_crs") : null;
      this.crs = crs ? { epsg: crs.epsg ?? null, name: crs.name || "" } : null;
    }
    const jobs = await tryCall("list_jobs");
    if (jobs) this.jobs = jobs.jobs || [];
    const telemetry = await tryCall("telemetry");
    this.vehicle = (telemetry && telemetry.telemetry) || null;
    this.updateChrome();
    this.applyBanner();
    refreshLive();
    return this.state;
  }

  refreshLive() {
    refreshLive();
  }

  retryStateRead() {
    if (this.stateRetryTimer) return;
    this.stateRetryTimer = setTimeout(async () => {
      this.stateRetryTimer = null;
      if (this.connection !== "connected" || !this.stateUnread) return;
      await this.refreshState();
      if (!this.stateUnread) this.renderScreen();
    }, 1000);
  }

  /** Jobs every 5 s, the aircraft every 3 s, the cursor and scale from the map live. */
  startPolling() {
    if (this.pollTimer) return;
    let ticks = 0;
    this.pollTimer = setInterval(async () => {
      ticks += 1;
      this.bindMap();
      if (this.connection === "disconnected") return;
      if (ticks % 5 === 0) {
        const jobs = await tryCall("list_jobs");
        if (jobs) this.jobs = jobs.jobs || [];
        this.updateChrome();
      }
      if (ticks % 3 === 0) {
        const telemetry = await tryCall("telemetry");
        this.vehicle = (telemetry && telemetry.telemetry) || null;
        this.updateChrome();
      }
    }, 1000);
  }

  stateSummary() {
    return {
      projectOpen: Boolean(this.state && this.state.project),
      datasetSelected: Boolean(this.state && this.state.dataset),
    };
  }

  /* ------------------------------------------------------- the chrome */

  /** Header, footer and Field header, from what the application reported. */
  updateChrome() {
    const frame = this.frame;
    if (!frame || !frame.node) return;
    const live = this.connection === "connected";
    const demo = this.connection === "demo";
    const project = this.project;

    // Project label. "· OFFLINE" exactly while the network state is "local · no network".
    const offline = this.networkState().text === "local · no network";
    if (demo) frame.setProject(DEMO.SITES[0] || "Example project");
    else if (!live) frame.setProject("Not connected", { none: true });
    else if (project) frame.setProject(String(project.name || `Project ${project.id}`), { offline });
    else frame.setProject("No project open", { none: true, offline });

    // Field header second line: the screen, then what the screen says about itself.
    frame.setFieldSubtitle([this.screenLabel(this.workspaceId), this.fieldSubtitle].filter(Boolean).join(" · "));
    frame.setFieldSimulated(this.vehicle && this.vehicle.connected ? this.vehicle : null);

    // CRS.
    let crsText = null;
    let crsTitle = "";
    // The EPSG the cursor's E/N are given in: the project's, else its layers'.
    this.footerEpsg = null;
    if (live && project) {
      const layerEpsg = ((this.state && this.state.layers) || []).map((l) => l.crs_epsg).filter(Boolean);
      const fromLayers = layerEpsg.find((e) => Number(e) !== 4326) || layerEpsg[0];
      if (this.crs && this.crs.epsg) {
        this.footerEpsg = this.crs.epsg;
        crsText = `EPSG:${this.crs.epsg} · ${shortCrsName(this.crs.name)}`;
        crsTitle = `Project CRS: ${this.crs.name}`;
      } else if (fromLayers) {
        this.footerEpsg = fromLayers;
        // The zone is read off the EPSG code itself (326zz / 327zz), not looked up.
        const zone = utmZoneOf(fromLayers);
        crsText = `EPSG:${fromLayers} · ${zone ? `UTM ${zone.zone}${zone.south ? "S" : "N"}` : "from layers"}`;
        crsTitle = "No project CRS is set; this is the CRS the project's layers are in.";
      } else {
        crsText = "CRS auto · UTM from imagery";
        crsTitle = (this.crs && this.crs.name) || "No CRS chosen; reconstruction picks the UTM zone.";
      }
    } else if (live) {
      crsText = "CRS · no project";
    }
    frame.setFooter("crs", crsText, { title: crsTitle });

    // Cursor and scale: only with a map on screen; otherwise said.
    if (!this.cursorBound) {
      frame.setFooter("cursor", "cursor · no map on this screen", { tone: "dim" });
      frame.setFooter("elev", null);
      frame.setFooter("scale", null);
    }

    // Vehicle.
    const v = this.vehicle;
    if (v && v.link_lost) {
      frame.setFooter("vehicle", "aircraft link lost", { tone: "danger" });
    } else if (v && v.connected) {
      const tag = simulatedTag(v, { small: true });
      frame.setFooter("vehicle", [el("span", { text: `driver ${v.driver || "unknown"} · ${v.flight_mode || "mode not reported"}` }), tag],
        { tone: "live" });
    } else if (live && this.workspaceId === "fly") {
      frame.setFooter("vehicle", "no vehicle connected");
    } else {
      frame.setFooter("vehicle", null);
    }

    // Jobs.
    const jobs = [...(this.jobs || [])].sort((a, b) => String(a.started_utc || "").localeCompare(String(b.started_utc || "")));
    const running = jobs.filter((j) => ["running", "pending"].includes(j.status));
    const focus = running[running.length - 1] || jobs[jobs.length - 1];
    if (focus) {
      const n = jobs.indexOf(focus) + 1;
      const word = { done: "complete", running: "running", pending: "queued", failed: "failed", cancelled: "cancelled" }[focus.status]
        || focus.status || "state not reported";
      const pct = focus.status === "running" && Number.isFinite(focus.percent) ? ` ${focus.percent}%` : "";
      frame.setFooter("jobs", `job ${n} of ${jobs.length} · ${word}${pct}`, {
        tone: focus.status === "failed" ? "danger" : running.length ? "live" : "",
        title: `${focus.name || "job"}${focus.message ? ` — ${focus.message}` : ""}`,
      });
    } else if (live && this.workspaceId === "process") {
      frame.setFooter("jobs", "no jobs recorded");
    } else {
      frame.setFooter("jobs", null);
    }

    frame.setFooter("context", this.footerContext);
    const last = this.log[this.log.length - 1];
    frame.setFooter("message", last ? last.text : "session log", { tone: last ? last.tone : "dim" });

    frame.setNetwork(this.networkState());
  }

  /** "local · no network" only while it is true of this window. */
  networkState() {
    if (this.connection === "demo") return { text: "example data · no network", tone: "sim" };
    if (this.connection !== "connected") return { text: "not connected to the application", tone: "warn" };
    const uses = [...this.networkUses];
    // A fixed-length label: the count, never the list, which could run off the footer.
    // The hosts are named in the tooltip, the session log and the Layers popover.
    if (uses.length) {
      return { text: `local · network in use (${uses.length})`, fieldText: `Network in use · ${uses.length}`,
               tone: "warn", title: `Network used this session: ${uses.join(", ")}` };
    }
    return { text: "local · no network", tone: "ok", title: "Nothing this window asked for has left the machine." };
  }

  /** Something this window asked for went off the machine; the footer and the log say so. */
  noteNetworkUse(what) {
    if (!what || this.networkUses.has(what)) return;
    this.networkUses.add(what);
    // One log entry per host, so the session log names every one.
    this.log.push({ at: new Date(), text: `Network in use: ${what} (${this.networkUses.size} this session)`, tone: "warn" });
    if (this.log.length > 200) this.log.shift();
    this.updateChrome();
  }

  /** A basemap was chosen; the online ones are network. */
  noteBasemap(key) {
    this.basemap = key;
    if (["satellite", "street", "topo"].includes(key)) this.noteNetworkUse(`${key} basemap`);
  }

  /** Every request the page itself makes, from the browser's own resource timing. */
  watchNetwork() {
    if (typeof PerformanceObserver === "undefined") return;
    const see = (entries) => {
      for (const entry of entries) {
        if (!isLoopback(entry.name)) {
          try { this.noteNetworkUse(new URL(entry.name).hostname); } catch { /* not a URL */ }
        }
      }
    };
    try {
      see(performance.getEntriesByType("resource"));
      new PerformanceObserver((list) => see(list.getEntries())).observe({ type: "resource", buffered: true });
    } catch { /* an engine without resource timing */ }
  }

  /** Follow the map on screen (when there is one) for the cursor and scale segments. */
  bindMap() {
    const map = currentMap();
    if (map === this.boundMap) {
      if (!map && this.cursorBound) { this.cursorBound = false; this.updateChrome(); }
      return;
    }
    this.boundMap = map;
    this.cursorBound = Boolean(map);
    if (!map) { this.updateChrome(); return; }
    const scale = () => {
      const c = map.getCenter();
      const text = formatScale(scaleDenominator(map.getZoom(), c.lat));
      this.frame.setFooter("scale", text, { title: "At 96 dpi, at the centre of the map" });
    };
    const cursor = (lon, lat) => {
      const parts = [formatLonLat(lon, lat)];
      const utm = this.footerEpsg ? utmZoneOf(this.footerEpsg) : null;
      if (utm) {
        const { easting, northing } = toUtm(lon, lat, utm.zone, utm.south);
        parts.push(`E ${Math.round(easting).toLocaleString("en-US")} N ${Math.round(northing).toLocaleString("en-US")}`);
      }
      this.frame.setFooter("cursor", parts.join("   "), { tone: "live" });
    };
    map.on("mousemove", (e) => {
      if (this.boundMap !== map) return;
      cursor(e.lngLat.lng, e.lngLat.lat);
      this.sampleElevation(e.lngLat.lng, e.lngLat.lat);
    });
    map.on("move", () => { if (this.boundMap === map) scale(); });
    const c = map.getCenter();
    cursor(c.lng, c.lat);
    this.sampleElevation(c.lng, c.lat);
    scale();
  }

  /**
   * The footer's ELEV: the project DSM's own value under the cursor (Api.sample_elevation,
   * read from the GeoTIFF). "ELEV —" with the reason in the tooltip when there is no DSM,
   * no data there, or no answer; never a number the DSM did not hold. One request in
   * flight at a time, always for the latest position.
   */
  sampleElevation(lon, lat) {
    const say = (text, title, tone = "") => this.frame.setFooter("elev", text, { title, tone });
    if (this.connection !== "connected") { say("ELEV —", "Not connected to the application."); return; }
    if (!this.frame.footer.elev) return; // Field has no footer
    if (this.elevNoDsm) { say("ELEV —", "No DSM is loaded in this project."); return; }
    this.elevNext = [lon, lat];
    if (this.elevBusy) return;
    this.elevBusy = true;
    (async () => {
      try {
        while (this.elevNext && !this.elevNoDsm) {
          const [x, y] = this.elevNext;
          this.elevNext = null;
          const answer = await tryCall("sample_elevation", x, y);
          if (!this.cursorBound) break;
          if (!answer) say("ELEV —", "The elevation could not be read here.");
          else if (!answer.dsm) { this.elevNoDsm = true; say("ELEV —", answer.reason || "No DSM is loaded in this project."); }
          else if (!Number.isFinite(answer.z)) say("ELEV —", `${answer.dsm}: ${answer.reason || "no data under the cursor"}.`);
          else {
            say(`ELEV ${answer.z.toFixed(2)} m`, `${answer.dsm} under the cursor`
              + `${answer.epsg ? ` (EPSG:${answer.epsg})` : ""}, read from the GeoTIFF`, "live");
          }
        }
      } finally {
        this.elevBusy = false;
      }
    })();
  }

  /** Fit the map on screen to coordinates, or say there is none. */
  fitMap(coords, what) {
    const map = currentMap();
    if (!map) {
      this.toast(`Zoom to ${what}: there is no map on this screen.`, "warn");
      return { skipped: "no map" };
    }
    const xs = coords.map((c) => c[0]);
    const ys = coords.map((c) => c[1]);
    fitTo(map, [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
    this.toast(`Zoomed to the ${what}.`, "ok");
    return { message: `Zoomed to the ${what}.` };
  }

  /** A search_places result: fly the map there, or say where it is. */
  flyTo(place) {
    const map = currentMap();
    if (!map) {
      this.toast(`${place.name}: ${formatLonLat(place.lon, place.lat)}. Open Plan or Analyse to see it on the map.`, "ok");
      return;
    }
    if (place.bounds && place.bounds.length === 4) fitTo(map, place.bounds);
    else map.jumpTo({ center: [place.lon, place.lat], zoom: Math.max(map.getZoom(), 16) });
    this.toast(`${place.name}: ${formatLonLat(place.lon, place.lat)}.`, "ok");
  }

  /* -------------------------------------------------------------- banner */

  /** Rows no Api call produced, or the demo's own sentinels (ported from the cockpit). */
  showingSampleContent() {
    for (const region of Object.values(this.regions || {})) {
      for (const node of region.querySelectorAll("[data-rows]")) {
        if (!node.closest("[data-live]") && !node.closest("[data-reference]")) return true;
      }
      const text = region.innerText || "";
      if (text && SAMPLE_SENTINELS.some((mark) => mark && text.includes(mark))) return true;
    }
    return false;
  }

  applyBanner() {
    const synthetic = this.connection === "demo" || this.showingSampleContent();
    this.frame.setBanner(synthetic, DEMO.BANNER, DEMO.PROVENANCE);
    return synthetic;
  }

  /* ------------------------------------------------------------- toasts */

  /**
   * A message where the operator is looking. kind: ok | warn | danger | running (the
   * cockpit's "info" and "error" are accepted). Every toast is kept in the session log,
   * which the footer message opens.
   */
  toast(text, kind = "ok") {
    const tone = { info: /…$/.test(String(text)) ? "running" : "ok", error: "danger", success: "ok" }[kind] || kind;
    const safe = ["ok", "warn", "danger", "running", "sim"].includes(tone) ? tone : "ok";
    const message = String(text ?? "");
    const split = message.indexOf(": ");
    const [title, detail] = split > 0 && split <= 48 ? [message.slice(0, split), message.slice(split + 2)] : [message, ""];
    const node = showToast({ title, detail, tone: safe }, { root: this.root, ms: safe === "danger" ? 9000 : 5000 });
    node.classList.add("toast"); // the cockpit harnesses watch for .toast
    const stack = node.parentNode;
    while (stack && stack.children.length > 4) stack.firstElementChild.remove();
    this.log.push({ at: new Date(), text: message, tone: safe });
    if (this.log.length > 200) this.log.shift();
    if (this.frame && this.frame.node) {
      this.frame.setFooter("message", message, { tone: safe });
    }
    return node;
  }

  /** A dialog to read (not a toast): provenance, capabilities, the audit log. */
  reveal(title, text) {
    return reveal(title, text, 28);
  }

  /* ----------------------------------------------------------- popovers */

  closePopovers() {
    for (const pop of [this.projectPop, this.logPop]) if (pop) { pop.close(); pop.node.remove(); }
    this.projectPop = null;
    this.logPop = null;
  }

  /** The header project label: the projects on this machine, each of which opens. */
  async projectMenu(anchor) {
    if (!this.projectPop || this.projectPop.anchor !== anchor) {
      if (this.projectPop) this.projectPop.node.remove();
      this.projectList = el("div", { class: "odk-project-list" });
      this.projectPop = popover({ anchor, title: "Projects", content: this.projectList });
      this.projectPop.anchor = anchor;
      this.projectPop.open();
    }
    if (this.projectPop.node.hidden) return;
    const list = this.projectList;
    list.replaceChildren(el("div", { class: "odk-popover-note", text: "Reading projects…" }));
    const listing = connected() ? await tryCall("list_projects") : null;
    list.replaceChildren();
    const projects = (listing && listing.projects) || [];
    if (!connected()) list.appendChild(el("div", { class: "odk-popover-note", text: "Not connected to the application." }));
    else if (!listing) list.appendChild(el("div", { class: "odk-popover-note", text: lastError.get("list_projects") || "Could not read the projects." }));
    else if (!projects.length) list.appendChild(el("div", { class: "odk-popover-note", text: "No projects on this machine yet." }));
    for (const p of projects.slice(0, 30)) {
      const current = this.project && p.id === this.project.id;
      const b = el("button", { type: "button", class: "odk-menu-item", "data-control": `project:${p.id}`,
        ...(current ? { "aria-current": "true" } : {}) }, [
        el("span", { class: "odk-menu-item-label", text: String(p.name || p.id) }),
        current ? el("span", { class: "odk-menu-item-hint", text: "open" }) : null,
      ]);
      b.addEventListener("click", () => { this.projectPop.close(); this.openProject(p.id, p.name); });
      list.appendChild(b);
    }
    const create = el("button", { type: "button", class: "odk-menu-item odk-menu-item--rule", "data-action": "New Project",
      text: "New project…" });
    create.addEventListener("click", () => { this.projectPop.close(); this.runAction("New Project"); });
    const all = el("button", { type: "button", class: "odk-menu-item", "data-control": "open:home", text: "All projects" });
    all.addEventListener("click", () => { this.projectPop.close(); this.open("home"); });
    list.append(create, all);
  }

  /** The footer message: every toast this session, newest first. */
  sessionLog(anchor) {
    if (!this.logPop || this.logPop.anchor !== anchor) {
      if (this.logPop) this.logPop.node.remove();
      this.logList = el("div", { class: "odk-log-list" });
      this.logPop = popover({ anchor, title: "Session log", content: this.logList });
      this.logPop.anchor = anchor;
      this.logPop.node.classList.add("odk-popover--log");
      this.logPop.open();
    }
    if (this.logPop.node.hidden) return;
    const rows = this.log.slice().reverse().map((entry) => el("div", { class: "odk-log-row" }, [
      el("span", { class: `odk-dot odk-dot--${entry.tone}`, "aria-hidden": "true" }),
      el("time", { class: "odk-log-time", text: entry.at.toTimeString().slice(0, 8) }),
      el("span", { class: "odk-log-text", text: entry.text }),
    ]));
    this.logList.replaceChildren(...(rows.length ? rows : [el("div", { class: "odk-popover-note", text: "Nothing has happened yet this session." })]));
    // Above the footer, right-aligned to the message it opened from.
    const r = anchor.getBoundingClientRect();
    const box = this.logPop.node;
    box.style.top = `${Math.max(8, r.top - box.offsetHeight - 4)}px`;
    box.style.left = `${Math.max(8, Math.min(r.right, window.innerWidth - 8) - box.offsetWidth)}px`;
  }

  /* ---------------------------------------------------- palette, help */

  openPalette(initial = "") {
    return this.palette.open(initial);
  }

  shortcutList() {
    return SHORTCUTS;
  }

  showShortcuts() {
    if (this.root.querySelector("[data-shortcuts]")) return;
    const close = () => { node.remove(); document.removeEventListener("keydown", onKey, true); };
    const onKey = (e) => {
      // A mode switch re-renders the root and takes the overlay with it; a listener left
      // behind would swallow the next Escape meant for a popover or a drawing.
      if (!node.isConnected) { document.removeEventListener("keydown", onKey, true); return; }
      if (e.key === "Escape" || e.key === "?") { e.preventDefault(); e.stopPropagation(); close(); }
    };
    const done = el("button", { type: "button", class: "odk-btn odk-btn--secondary", "data-control": "close-shortcuts", text: "Close" });
    done.addEventListener("click", close);
    const node = el("div", { class: "odk-palette-backdrop", "data-shortcuts": "" }, [
      el("div", { class: "odk-palette odk-shortcuts", role: "dialog", "aria-modal": "true", "aria-label": "Keyboard shortcuts" }, [
        el("div", { class: "odk-shortcuts-head" }, [el("span", { class: "odk-label", text: "Keyboard shortcuts" })]),
        el("dl", { class: "odk-shortcuts-list" }, SHORTCUTS.flatMap((s) => [
          el("dt", {}, [el("kbd", { class: "odk-kbd", text: s.keys })]), el("dd", { text: s.what }),
        ])),
        el("div", { class: "odk-shortcuts-foot" }, [done]),
      ]),
    ]);
    node.addEventListener("mousedown", (e) => { if (e.target === node) close(); });
    document.addEventListener("keydown", onKey, true);
    this.root.appendChild(node);
    done.focus();
  }

  /** What a keyboard shortcut (or its palette row) does. */
  shortcut(name) {
    switch (name) {
      case "palette": this.openPalette(); return;
      case "search": this.openPalette(); return;
      case "settings": this.open("settings"); return;
      case "mode": this.setMode(this.mode === "field" ? "desk" : "field"); return;
      case "help": this.showShortcuts(); return;
      case "save": this.runAction("Save"); return;
      case "sides":
        if (this.mode !== "desk") { this.toast("Side panels: Field mode has one panel and it stays.", "ok"); return; }
        this.toggleSidePanels();
        return;
      case "full":
        this.fullCanvas = !this.fullCanvas;
        this.frame.setFullCanvas(this.fullCanvas);
        return;
      case "fit": {
        const map = currentMap();
        if (!map) { this.toast("Fit: there is no map on this screen.", "warn"); return; }
        if (!fitTo(map, map._odkBounds)) this.toast("Fit: nothing in this project to fit to yet.", "warn");
        return;
      }
      case "undo":
        this.toast(undoPoint() ? "Removed the last point." : "Nothing to undo: no shape is being drawn.", "ok");
        return;
      case "stop":
        if (stopDrawing()) this.toast("Drawing cancelled.", "ok");
        return;
      default:
    }
  }

  bindKeys() {
    bindDrawingKeys(document);
    window.addEventListener("odk:action", (event) => { if (event && event.detail) this.runAction(String(event.detail)); });
    window.addEventListener("odk:say", (event) => {
      const { text, kind } = event.detail || {};
      if (text) this.toast(text, kind || "ok");
    });
    document.addEventListener("keydown", (event) => this.onKey(event));
  }

  onKey(event) {
    const meta = event.ctrlKey || event.metaKey;
    const tag = document.activeElement && document.activeElement.tagName;
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(tag) || (document.activeElement && document.activeElement.isContentEditable);
    const overlay = document.querySelector(".modal-backdrop, [data-palette], [data-shortcuts]");
    const key = String(event.key || "").toLowerCase();
    if (meta && key === "k") { event.preventDefault(); this.openPalette(); return; }
    if (overlay) return; // a dialog owns the keyboard
    if (meta && event.shiftKey && key === "f") { event.preventDefault(); this.shortcut("mode"); return; }
    if (meta && key === ",") { event.preventDefault(); this.shortcut("settings"); return; }
    if (event.key === "F11") { event.preventDefault(); this.shortcut("full"); return; }
    if (meta && key === "s") { event.preventDefault(); this.shortcut("save"); return; }
    if (meta && key === "b") { event.preventDefault(); this.shortcut("sides"); return; }
    if (meta && key === "z" && !typing) { event.preventDefault(); this.shortcut("undo"); return; }
    if (typing || meta || event.altKey) return;
    if (event.key === "/") { event.preventDefault(); this.shortcut("search"); return; }
    if (event.key === "?") { event.preventDefault(); this.shortcut("help"); return; }
    if (key === "f") { this.shortcut("fit"); return; }
    if (event.key === "Escape") { this.shortcut("stop"); return; }
    if (/^[1-9]$/.test(event.key)) {
      const list = this.mode === "field" ? FIELD_SCREENS : DESK_SCREENS;
      const screen = list[Number(event.key) - 1];
      if (screen) this.open(screen.id);
    }
  }

  /* -------------------------------------------------------- image view */

  /**
   * A photograph or a raster preview, over the map: img.canvas-product, as the cockpit
   * drew it, so pickImagePixel() and the ported harnesses find it. U3's imageview may
   * replace this with shell.provide("imageView", (host, image) => node).
   */
  renderImageView() {
    const host = this.frame && this.frame.overlay;
    if (!host) return;
    const image = currentImage();
    if (currentView() !== "image" || !image || !image.data_uri) {
      host.replaceChildren();
      host.hidden = true;
      return;
    }
    host.hidden = false;
    if (this.providers.imageView) { host.replaceChildren(this.providers.imageView(host, image)); return; }
    const back = el("button", { type: "button", class: "odk-btn odk-btn--secondary", "data-control": "image:close",
      text: "Back to map" });
    back.addEventListener("click", () => { setImage(null); setView("map"); });
    const dims = image.source_width && image.source_height ? `${image.source_width} × ${image.source_height} px` : "";
    host.replaceChildren(el("div", { class: "odk-shell-imageview" }, [
      el("div", { class: "odk-shell-imageview-bar" }, [
        el("span", { class: "odk-label", text: "Image" }),
        el("span", { class: "odk-shell-imageview-name", text: image.name || "" }),
        dims ? el("span", { class: "odk-shell-imageview-dims", text: dims }) : null,
        el("span", { class: "odk-header-spacer" }),
        back,
      ]),
      el("div", { class: "odk-shell-imageview-well" }, [
        el("img", { class: "canvas-product", src: image.data_uri, alt: image.name || "image" }),
      ]),
    ]));
  }

  async pickImagePixel(message) {
    let image = null;
    for (let i = 0; i < 200 && !image; i += 1) {
      image = this.root.querySelector("img.canvas-product");
      if (!image || !image.naturalWidth) {
        image = null;
        await new Promise((r) => setTimeout(r, 25));
      }
    }
    if (!image) return null;
    const picked = currentImage() || {};
    this.toast(message || "Click the point on the photograph (Esc to cancel).", "ok");
    return new Promise((resolveClick) => {
      const done = (value) => {
        image.removeEventListener("click", onClick);
        document.removeEventListener("keydown", onKey);
        resolveClick(value);
      };
      const onKey = (event) => { if (event.key === "Escape") done(null); };
      const onClick = (event) => {
        const box = image.getBoundingClientRect();
        const nw = image.naturalWidth;
        const nh = image.naturalHeight;
        const fit = Math.min(box.width / nw, box.height / nh);
        const x = (event.clientX - box.left - (box.width - nw * fit) / 2) / fit;
        const y = (event.clientY - box.top - (box.height - nh * fit) / 2) / fit;
        if (x < 0 || y < 0 || x > nw || y > nh) { done(null); return; }
        const width = Number(picked.source_width) || nw;
        const height = Number(picked.source_height) || nh;
        const [cx, cy, cw, ch] = picked.crop || [0, 0, width, height];
        done({ x: cx + (x * cw) / nw, y: cy + (y * ch) / nh, width, height });
      };
      image.addEventListener("click", onClick);
      document.addEventListener("keydown", onKey);
    });
  }

  async showImage(name, crop = null) {
    if (!connected()) { this.toast(`${name}: connect a project to preview the image.`, "warn"); return; }
    this.toast(`Loading ${name}…`, "running");
    const preview = crop ? await tryCall("image_preview", name, 1400, crop) : await tryCall("image_preview", name);
    if (!preview || preview.ok === false) {
      this.toast(`${name}: ${(preview && preview.error) || lastError.get("image_preview") || "no preview available."}`, "warn");
      return;
    }
    setImage(preview);
    setView("image");
    this.toast(`${preview.name}: ${preview.source_width} × ${preview.source_height} px`, "ok");
  }

  async showLayer(layerId, name) {
    if (!connected()) { this.toast(`${name}: connect a project to open layers.`, "warn"); return; }
    const raster = await tryCall("raster_preview", layerId);
    if (raster && raster.data_uri) {
      setImage({ name, data_uri: raster.data_uri, coordinates: raster.coordinates });
      setView("image");
      this.toast(`Showing ${name}.`, "ok");
      return;
    }
    const vector = await tryCall("read_vector_layer", layerId);
    if (vector && vector.geojson) {
      const features = vector.geojson.features || [];
      const kinds = [...new Set(features.map((f) => (f.geometry || {}).type).filter(Boolean))];
      const { showVector } = await import("../workspace/mapview.js");
      showVector(String(layerId), vector.geojson, String(name).startsWith("CAD:") ? "cad" : "layer");
      setImage(null);
      setView("map");
      this.toast(`${name}: ${features.length} feature${features.length === 1 ? "" : "s"}`
        + `${kinds.length ? ` (${kinds.join(", ")})` : ""}.`, "ok");
      return;
    }
    const why = lastError.get("raster_preview") || lastError.get("read_vector_layer");
    this.toast(`${name}: ${why || "this layer cannot be displayed."}`, "warn");
  }

  /* --------------------------------------------------------- selection */

  /** What happens when the operator picks something in a panel (ported from the cockpit). */
  onSelection(kind, value) {
    if (!value) return;
    const name = value.label || value.name || value.id || value.key || "unnamed";
    switch (kind) {
      case "job":
        this.selectedJobId = value.job || value.id || null;
        if (this.selectedJobId) this.watchJob(this.selectedJobId);
        return;
      case "finding":
      case "anomaly":
        this.selectedFindingId = value.full_id || value.id || value.label || null;
        return;
      case "image":
      case "capture":
        this.showImage(String(value.label || value.name || name));
        return;
      case "report":
        this.selectedReportId = value.id || null;
        return;
      case "aircraft":
      case "battery":
      case "pilot": {
        const recordId = Number.isInteger(value.fleetId) ? value.fleetId : null;
        if (kind === "aircraft") this.selectedFleetId = recordId;
        this.selectedFleetRecord = recordId ? { kind, id: recordId, name } : null;
        return;
      }
      case "project":
        this.openProject(value.id ?? value.project_id, name);
        return;
      case "layer":
        this.showLayer(value.id || value.layer_id || value.key || name, name);
        return;
      case "tool": {
        const tool = String(value.action || value.label || name).replace(/ line$/, "");
        if (has(tool)) this.runAction(tool);
        else this.toast(`${tool}: no such tool.`, "warn");
        return;
      }
      case "endpoint":
        if (value.method) this.selectedEndpoint = { method: value.method, path: value.path };
        return;
      case "setting":
        this.settings[value.key] = value.value;
        if (["camera", "alt", "payload_key", "rth"].includes(value.key)) refreshLive();
        return;
      case "template":
        this.settings.type = String(value.label || name);
        return;
      default:
    }
  }

  async openProject(projectId, name) {
    if (projectId == null) { this.toast(`${name}: this project has no id, so it cannot be opened.`, "warn"); return false; }
    if (!connected()) { this.toast(`${name}: the application is not connected.`, "warn"); return false; }
    const opened = await tryCall("set_active_project", projectId);
    if (!opened) {
      this.toast(`${name}: ${lastError.get("set_active_project") || "could not be opened."}`, "warn");
      return false;
    }
    setImage(null);
    setView("map");
    await this.refreshState();
    await this.renderScreen();
    this.toast(`Opened ${name}.`, "ok");
    return true;
  }

  /* ----------------------------------------------------------- actions */

  bindDrawing() {
    configureDrawing({
      toast: (text, kind) => this.toast(text, kind),
      prompt: (label, fallback) => askOne(label, label, fallback),
      workspaceId: () => this.requestedId,
      geometry: (geometry) => { this.selectionGeometry = geometry; },
      changed: async () => { await this.refreshState(); },
      drawn: (tool, geometry) => {
        let armed = this.armedTool && this.armedTool.tool === tool ? this.armedTool : null;
        if (!armed) {
          const found = Object.entries(ACTIONS).find(([, e]) => e && e.tool === tool);
          armed = { entry: found ? found[1] : {}, action: found ? found[0] : tool };
        }
        return this.onGeometryDrawn(armed.entry, armed.action, geometry);
      },
    });
    configureViewer({
      toast: (text, kind) => this.toast(text, kind),
      prompt: (label, fallback) => askOne(label, label, fallback),
      changed: () => refreshLive(),
    });
  }

  /**
   * Run an ACTIONS key against the real application. Resolves to the action's result,
   * or {error} / {skipped}; never throws. Everything is also said in a toast.
   */
  async runAction(key, payload = null) {
    try {
      return await this.runActionUnsafe(String(key), payload);
    } catch (error) {
      const message = String((error && error.message) || error);
      this.toast(`${key}: ${message}`, "danger");
      return { error: message };
    }
  }

  async runActionUnsafe(action, payload) {
    if (UNWIRED[action]) {
      this.toast(`${action}: not available. ${UNWIRED[action]}`, "warn");
      return { error: `not available: ${UNWIRED[action]}` };
    }
    const entry = resolve(action);
    if (!entry) {
      this.toast(`${action}: no such action.`, "warn");
      return { error: `unknown action ${action}` };
    }
    // Client-side behaviour before the prerequisite gate: a view, a tool, a screen.
    if (entry.view) {
      setImage(null);
      setView(entry.view);
      await this.renderScreen();
      this.toast(`View: ${action}`, "ok");
      return { message: `View: ${action}`, view: entry.view };
    }
    if (entry.tool) {
      this.armedTool = { tool: entry.tool, entry, action };
      const onModel = currentViewer();
      let said = onModel ? armModelTool(entry.tool) : null;
      const onMap = !said;
      if (onMap) said = armTool(entry.tool);
      this.activeTool = entry.tool;
      if (said) {
        this.toast(`${action}: ${said}`, !onMap || currentMap() ? "ok" : "warn");
        return { message: said, tool: entry.tool };
      }
      const why = "this tool draws on a 3D model and needs the 3D view open.";
      this.toast(`${action}: ${why}`, "warn");
      return { skipped: why };
    }
    if (entry.workspace) {
      await this.open(entry.workspace);
      return { message: `Opened ${this.screenLabel(entry.workspace)}.` };
    }

    if (this.connection === "connected" && !(this.state && this.state.project)
        && prerequisite(action, this.stateSummary())) {
      // "No project open" is only as true as the last read; ask again before refusing.
      await this.refreshState();
    }
    const blocked = prerequisite(action, this.stateSummary());
    if (blocked) {
      this.toast(`${action}: ${blocked}`, "warn");
      return { skipped: blocked };
    }
    if (entry.confirm && !this.confirmAll && !(await confirmAsk(entry.confirm))) {
      return { skipped: "Cancelled." };
    }

    const progress = this.toast(`${action}…`, "running");
    let result;
    try {
      result = entry.vehicle
        ? await this.runVehicle(entry.vehicle, action)
        : await entry.run({ ...this.actionContext(payload), payload: payload || {} });
    } catch (error) {
      progress.remove();
      const message = String((error && error.message) || error);
      this.toast(`${action}: ${message}`, "danger");
      return { error: message };
    }
    progress.remove();
    if (result && result.skipped) {
      this.toast(`${action}: ${result.skipped}`, result.level || "ok");
      return result;
    }
    const message = (result && result.message) || `${action} done.`;
    this.toast(message, (result && result.level) || "ok");
    if (result && result.job) this.watchJob(result.job);
    if (result && result.refresh) {
      await this.refreshState();
      refreshProjectOverlays();
    }
    return result || { message };
  }

  /** Arm, start, RTH...: Start is held to the preflight checklist, with an explicit override. */
  async sendVehicleCommand(command) {
    try {
      return await call("vehicle_command", command);
    } catch (error) {
      if (!(error.result && error.result.preflight_blocked)) throw error;
      const question = `${error.message} Start the mission anyway? This is recorded as your preflight override.`;
      if (!this.confirmAll && !(await confirmAsk(question))) return null;
      return await call("vehicle_command", command, { override_preflight: true });
    }
  }

  async runVehicle(command, action) {
    const result = await this.sendVehicleCommand(command);
    if (!result) return { skipped: "not started; preflight is not ready.", level: "warn" };
    const answer = (result && result.result && result.result.message) || "accepted by the vehicle.";
    return { message: `${action}: ${answer}`, refresh: true };
  }

  async onGeometryDrawn(entry, action, geometry) {
    this.selectionGeometry = geometry;
    const points = vertices(geometry);
    const pending = this.pendingSelectionAction;
    this.pendingSelectionAction = null;
    if (pending) { await this.runAction(pending); return; }
    if (!entry.measure) {
      this.toast(`${geometry.type === "Polygon" ? "Area" : "Line"} selected (${points.length} points). `
        + "Volume, Cut & Fill, Slope and Stockpile use it.", "ok");
      return;
    }
    if (!connected()) { this.toast(`${action}: connect to the application to measure.`, "warn"); return; }
    try {
      const parts = [];
      if (entry.measure === "distance" || entry.measure === "area") {
        const flat = await call("measure_geometry", points, entry.measure);
        if (entry.measure === "distance") {
          recordMeasurement({ kind: "distance", value: flat.length_m, unit: "m", vertices: flat.vertices });
          parts.push(`Distance ${formatLength(flat.length_m)} over ${flat.vertices} points (${flat.method}).`);
        } else {
          recordMeasurement({ kind: "area", value: flat.area_m2, unit: "m²", perimeter_m: flat.perimeter_m });
          parts.push(`Area ${formatArea(flat.area_m2)}, perimeter ${formatLength(flat.perimeter_m)}.`);
        }
        try {
          const surface = await call("measure_drawn", entry.measure, points);
          this.lastMeasurement = surface;
          parts.push(`On the project raster: ${describeMeasurement(surface)}`);
        } catch (error) {
          const why = String(error.message || error);
          if (!/no georeferenced raster/i.test(why)) parts.push(`Raster measurement refused: ${why}`);
        }
        this.toast(parts.join(" "), "ok");
      } else {
        const result = await call("measure_drawn", entry.measure, points);
        this.lastMeasurement = result;
        this.toast(`${action}: ${describeMeasurement(result)}`, "ok");
      }
      refreshLive();
    } catch (error) {
      this.toast(`${action}: ${error.message || error}`, "danger");
    }
  }

  requireSelection(shape, action, prompt) {
    const geometry = this.selectionGeometry;
    const wanted = shape === "polygon" ? "Polygon" : "LineString";
    if (geometry && geometry.type === wanted) return vertices(geometry);
    this.pendingSelectionAction = action;
    this.activeTool = `select-${shape}`;
    armDrawing(shape, (drawn) => this.onGeometryDrawn({}, action, drawn));
    this.toast(prompt || `${action}: draw the ${shape === "polygon" ? "area" : "line"} on the map, `
      + "then double-click or press Enter.", "ok");
    return null;
  }

  /** Follow a background job to the end, in one toast that updates in place. */
  watchJob(jobId) {
    if (!jobId) return null;
    this.selectedJobId = jobId;
    this.watching = this.watching || new Set();
    if (this.watching.has(jobId)) return null;
    this.watching.add(jobId);
    let node = null;
    const say = (text) => {
      if (node && node.isConnected) {
        const title = node.querySelector(".odk-toast-title");
        if (title) title.textContent = text;
      } else {
        node = this.toast(text, "running");
      }
    };
    const finish = (text, tone) => {
      this.watching.delete(jobId);
      if (node) node.remove();
      this.toast(text, tone);
    };
    const tick = async () => {
      const envelope = await tryCall("job_status", jobId);
      if (!envelope) { this.watching.delete(jobId); return; }
      const status = envelope.job || envelope;
      const state = status.state || status.status;
      const percent = status.progress ?? status.percent;
      const index = this.jobs.findIndex((j) => j.id === jobId);
      if (index >= 0) this.jobs[index] = { ...this.jobs[index], ...status };
      else this.jobs.push(status);
      this.updateChrome();
      if (["done", "finished", "complete"].includes(state)) {
        finish(`Job ${jobId} finished${jobOutcome(status.result)}.`, "ok");
        await this.refreshState();
        this.showFinishedPipelineReport(status.result);
        return;
      }
      if (state === "failed" || state === "error") { finish(`Job ${jobId} failed: ${status.error || "no reason given"}`, "danger"); return; }
      if (state === "cancelled") { finish(`Job ${jobId} cancelled.`, "warn"); return; }
      say(`Job ${jobId}: ${status.message || state}${percent != null ? ` ${percent}%` : ""}`);
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 800);
    return { message: `Watching job ${jobId}.` };
  }

  async showFinishedPipelineReport(result) {
    const path = result && result.report_markdown_path;
    if (!path) return;
    const runId = String(path).split(/[\\/]/).slice(-2)[0];
    const report = await tryCall("read_pipeline_report", runId);
    if (report && typeof report.markdown === "string") reveal(`Inspection report: ${report.id}`, report.markdown, 28);
  }

  /**
   * What every action receives. Same members the cockpit provided, plus open, refresh,
   * map, selection and drawing:
   *   prompt choose ask form showResult labelBoxes confirm reveal showText toast
   *   missionOptions reconstructionOptions pipelineOptions inspectionOptions reportOptions
   *   selectedReport selectedJob selectedFinding selectedFleetId selectedFleetRecord
   *   selectedEndpoint selectionGeometry selectionVertices requireSelection pickPoint
   *   pickImagePixel showImage showView setSetting resetLayout organizationId projectId
   *   openWorkspace open workspaceId refresh map selection drawing payload
   */
  actionContext(payload = null) {
    const shell = this;
    const raw = () => shell.settings || {};
    const ctx = {
      prompt: (label, fallback) => askOne(label, label, fallback),
      choose: (title, options) => chooseModal(title, options),
      ask: (title, fields) => ask(title, fields),
      form: (title, fields) => ask(title, fields),
      showResult: (title, content) => showResult(title, content),
      labelBoxes: (title, spec) => labelBoxes(title, spec),
      confirm: (question) => (shell.confirmAll ? Promise.resolve(true) : confirmAsk(question)),
      reveal: (text, title) => reveal(title || "Copy this now — it is not stored", text),
      showText: (title, text) => reveal(title, text, 28),
      toast: (text, kind) => shell.toast(text, kind || "ok"),

      missionOptions: () => (shell.providers.missionOptions ? shell.providers.missionOptions()
        : defaultMissionOptions(raw())),
      reconstructionOptions: () => {
        if (shell.providers.reconstructionOptions) return shell.providers.reconstructionOptions();
        const r = raw();
        const options = { engine: "auto", profile: "standard" };
        // Reconstruct › Run Reconstruction (COLMAP / fallback engine) names the engine.
        if (payload && payload.engine) options.engine = String(payload.engine);
        else if (r.engine) options.engine = String(r.engine);
        if (r.profile) options.profile = String(r.profile);
        if (r.dense !== undefined && r.dense !== "") options.dense = String(r.dense).toLowerCase() === "yes";
        if (r.epsg) options.epsg = Number(r.epsg);
        if (r.run_on === "OpenDroneKit server") {
          options.execution_mode = "cloud";
          options.cloud_endpoint = String(r.cloud_endpoint || "").trim();
          options.cloud_token = String(r.cloud_token || "").trim();
        }
        return options;
      },
      pipelineOptions: () => {
        const options = ctx.reconstructionOptions();
        const agl = Number(raw().agl);
        if (raw().agl !== undefined && raw().agl !== "" && Number.isFinite(agl) && agl > 0) {
          options.flight_height_agl_m = agl;
        }
        return options;
      },
      inspectionOptions: () => {
        const r = raw();
        const options = ctx.pipelineOptions();
        for (const key of ["structure_type", "material_type", "solar_imagery"]) if (r[key]) options[key] = String(r[key]);
        for (const key of ["run_solar_detection", "run_metal_detection", "run_structural_detection", "crack_presence_gate"]) {
          const value = String(r[key] ?? "").toLowerCase();
          if (value === "on") options[key] = true;
          else if (value === "off") options[key] = false;
        }
        return options;
      },
      reportOptions: () => {
        if (shell.providers.reportOptions) return shell.providers.reportOptions();
        const r = raw();
        const format = String(r.fmt || "PDF").toLowerCase();
        const formats = { pdf: ["html", "pdf"], docx: ["html", "docx"], "pdf+docx": ["html", "pdf", "docx"],
          html: ["html"] }[format] || ["html", "pdf"];
        return { reportType: String(r.tpl || "standard"), format, formats,
          organization: String(r.org || ""), client: String(r.client || ""),
          author: String(r.author || ""), logo: String(r.logo || "") };
      },

      selectedReport: () => shell.selectedReportId || null,
      selectedJob: () => shell.selectedJobId || null,
      selectedFinding: () => shell.selectedFindingId || null,
      selectedFleetId: () => shell.selectedFleetId || null,
      selectedFleetRecord: () => shell.selectedFleetRecord || null,
      selectedEndpoint: () => shell.selectedEndpoint || null,
      selectionGeometry: () => shell.selectionGeometry || null,
      selectionVertices: () => vertices(shell.selectionGeometry),
      requireSelection: (shape, action, prompt) => shell.requireSelection(shape, action, prompt),
      pickPoint: (message) => {
        if (!canDraw()) return Promise.resolve(null);
        shell.toast(message || "Click the position on the map (Esc to type it instead).", "ok");
        return new Promise((done) => {
          armDrawing("point", (geometry) => done(geometry.coordinates), () => done(null));
        });
      },
      pickImagePixel: (message) => shell.pickImagePixel(message),
      showImage: (name, crop) => shell.showImage(name, crop),
      showView: (view) => { setImage(null); setView(view); return shell.renderScreen(); },
      setSetting: (key, value) => { shell.settings[key] = value; },
      resetLayout: () => {
        shell.frame.setCollapsed("left", false);
        shell.frame.setCollapsed("inspector", false);
        if (shell.fullCanvas) { shell.fullCanvas = false; shell.frame.setFullCanvas(false); }
      },
      organizationId: () => (shell.state && shell.state.project && shell.state.project.organization_id) || 1,
      projectId: () => (shell.state && shell.state.project && shell.state.project.id) || 1,

      // The id the operator asked for, so actions shared by two old screens (Fleet,
      // Verification) still know which one pressed them.
      workspaceId: () => shell.requestedId,
      openWorkspace: (id) => shell.open(id),
      open: (id) => shell.open(id),
      refresh: () => shell.refreshState(),
      map: () => currentMap(),
      selection,
      drawing: shell.drawing,
      payload: payload || {},
    };
    return ctx;
  }
}

/** The mission form's values, under the names plan_mission reads (cockpit behaviour). */
export function defaultMissionOptions(raw) {
  const numeric = {};
  for (const [key, value] of Object.entries(raw || {})) {
    const asNumber = Number(value);
    numeric[key] = value !== "" && value !== null && typeof value !== "boolean" && Number.isFinite(asNumber)
      ? asNumber : value;
  }
  const options = {};
  for (const [formKey, apiKey] of Object.entries(MISSION_OPTION_KEYS)) {
    if (numeric[formKey] !== undefined && numeric[formKey] !== "") options[apiKey] = numeric[formKey];
  }
  // Only the mission fields: the report and processing forms share the settings store,
  // and are not planning inputs. The fields the planner has no option for go nowhere.
  for (const key of MISSION_FIELDS_NOT_PLANNED) delete options[key];
  if (numeric.type !== undefined) {
    const template = MISSION_TEMPLATES[String(numeric.type)] || (Object.values(MISSION_TEMPLATES).includes(numeric.type) ? numeric.type : null);
    if (template) options.template = template;
  }
  return options;
}

export { dispatchMenu };
