/* The application shell: navigation, contextual toolbar, status bar, command palette.
 *
 * Four persistent layers, and the third is the one that matters. Nav, toolbar and status
 * frame a dockable workspace that changes completely between tasks while the project
 * underneath does not. An operator switching from planning to flight is not navigating
 * to another page -- they are turning to a different instrument on the same aircraft.
 */

import { Dock } from "./dock.js";
import { el, selection } from "./primitives.js";
import { WORKSPACES, WORKSPACE_BY_ID, auditLine } from "./workspaces.js";
import { DATA, DEMO, SAMPLE_SENTINELS, demoEnabled } from "./demo.js";
import { call, connected, lastError, tryCall, whenReady } from "./api.js";
import { ACTIONS, UNWIRED, describeMeasurement, prerequisite, resolve } from "./actions.js";
import {
  armDrawing, armTool, bindDrawingKeys, canDraw, configureDrawing, currentMap, formatArea,
  formatLength, recordMeasurement, stopDrawing, undoPoint, vertices,
} from "./draw.js";
import { basemapNames, fitTo, refreshProjectOverlays, setBasemap, showVector, visibleBounds } from "./mapview.js";
import { currentImage, setImage, setView } from "./viewstate.js";
import { ask, askOne, choose as chooseModal, confirmAsk, labelBoxes, reveal, showResult } from "./modal.js";
import { refreshLive } from "./live.js";
import { armModelTool, configureViewer, currentViewer } from "./viewer3d.js";

const LAST_WORKSPACE = "odk.workspace.last";

/**
 * What a finished job produced, in the words a user acts on. "Job finished" said nothing
 * about a Run AI whose reconstruction failed, or how many findings are waiting for review.
 */
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
  if (result.reconstruction_status === "failed") parts.push("reconstruction failed; detection results kept");
  if (result.metrics && typeof result.metrics.map50 === "number") {
    parts.push(`validation mAP50 ${result.metrics.map50.toFixed(3)}`);
  }
  if (result.parity_ok === true && result.key) parts.push(`installed as ${result.key}`);
  return parts.length ? `: ${parts.join("; ")}` : "";
}

/* Safety-critical actions must never be one careless click from Save. They are styled
 * apart and listed here so the styling cannot drift from the meaning. */
const DANGER = new Set(["Abort", "Land", "RTL", "Manual Override", "Cancel"]);
const PRIMARY = new Set(["Plan", "Process", "Run AI", "Generate", "Start", "Generate Report"]);

/* Every shortcut listed here is bound in bindKeys() and runs from the palette.
 *
 * The list used to advertise Ctrl+S, Ctrl+B, F and Ctrl+Z, and the keydown handler bound
 * none of them -- so the palette documented keys that did nothing, and choosing one there
 * ran `() => {}`. A shortcut list is a claim about the program. */
const SHORTCUTS = [
  ["Ctrl/⌘ K", "Command palette", "palette"],
  ["Ctrl/⌘ S", "Save the planned mission", "save"],
  ["Ctrl/⌘ B", "Toggle side panels", "sides"],
  ["F", "Fit the map to the project", "fit"],
  ["F11", "Full-screen canvas", "full"],
  ["1 – 9", "Switch workspace", null],
  ["Ctrl/⌘ Z", "Undo the last drawn point", "undo"],
  ["Esc", "Stop drawing", "stop"],
];

/* The mission form's field keys, and what plan_mission actually reads.
 *
 * Kept as an explicit map rather than renaming the form fields, because the short names
 * are what the panel displays and the long ones are the API contract. Either could
 * reasonably change; this is the one place that has to know about both.
 */
const MISSION_OPTION_KEYS = {
  alt: "altitude_m",
  fwd: "front_overlap_pct",
  side: "side_overlap_pct",
  speed: "speed_m_s",
  standoff: "standoff_m",
  rth: "rth_altitude_m",
  // Mission-type inputs, already named as plan_mission reads them.
  thermal_camera: "thermal_camera",
  target_gsd_cm: "target_gsd_cm",
  rgb_camera: "rgb_camera",
  payload_key: "payload_key",
  calibration_panel_lonlat: "calibration_panel_lonlat",
  accept_relative_values: "accept_relative_values",
  pylon_elements: "pylon_elements",
  dome_radius_m: "dome_radius_m",
  box_altitude_levels_m: "box_altitude_levels_m",
  linked_version_ids: "linked_version_ids",
};

/* Fields the form shows that plan_mission has no option for.
 *
 * Listed rather than mapped to an invented name. Writing `gsd: "target_gsd_cm"` here
 * would look like wiring and do nothing, which is the failure this whole change exists
 * to remove -- the first version of this map did exactly that for two of them.
 *
 * Each is inert for a reason, and each is a real gap worth closing later:
 *   gsd    the planner solves altitude FROM overlap and camera, not from a target GSD.
 *          mp.gsd computes it the other way; wiring the two is its own piece of work.
 *   angle  line heading is chosen by the grid generator; there is no override yet.
 *   air    aircraft model affects battery and speed limits, which the planner does not
 *          currently take from the form.
 *   batt   battery count is an estimate output, not a planning input.
 */
const MISSION_FIELDS_NOT_PLANNED = new Set(["gsd", "angle", "air", "batt"]);

/* Processing Settings fields run_reconstruction has no option for.
 *
 * imgsize  max image size is a property of the chosen profile, not a separate input;
 *          the profiles set it to 2400 / 4500 / 9000 features and their own resolution.
 * mesh     mesh depth is fixed in the meshing step and not currently exposed.
 *
 * Listed here for the same reason as the mission fields: a control on screen that does
 * nothing is worse than one that is absent, and naming it makes the gap visible.
 */
const PROCESSING_FIELDS_NOT_USED = new Set(["imgsize", "mesh"]);

/* The human labels in the Mission Types list, and the templates they mean.
 *
 * The list is written for a pilot ("3D modelling", "Facade inspection"); the planner
 * takes ids. Selecting one used to send the label straight through, where
 * _normalize_template did not recognise it and fell back to "grid".
 */
const MISSION_TEMPLATES = {
  "2D mapping": "grid",
  "3D modelling": "double_grid",
  "Roof mapping": "grid",
  "Roof inspection": "roof_inspection",
  "Facade mapping": "facade_mapping",
  "Facade inspection": "facade",
  "Multi-facade": "multi_facade",
  "Closed loop": "closed_loop",
  "Linear mapping": "corridor",
  "Linear inspection": "linear_inspection",
  "Tower mapping": "tower_mapping",
  "Orbit": "orbit",
  "Waypoints": "waypoints",
  "Panorama": "panorama",
  "Solar inspection": "solar_inspection",
  "Wind turbine": "wind_turbine",
  "Magnetic mapping": "magnetic_mapping",
  "Complex facade": "multi_facade",
  "L-shaped": "grid",
  // The three below are planned by mission/mission_types.py, which plan_mission routes
  // to by these names. They used to map to tower_mapping and grid, so choosing
  // "Thermal" planned an ordinary RGB grid at the RGB camera's altitude.
  "Utility pylon": "pylon_inspection",
  "Thermal": "thermal",
  "Multispectral": "multispectral",
  "Dome inspection": "dome_inspection",
  "Box inspection": "box_inspection",
  "360 bubble": "bubble_360",
  "Lateral capture": "lateral_capture",
  "Smart adaptive": "smart_adaptive",
  "Linked mission": "linked_mission",
};

export class Shell {
  constructor(root) {
    this.root = root;
    this.workspaceId = localStorage.getItem(LAST_WORKSPACE) || "home";
    if (!WORKSPACE_BY_ID[this.workspaceId]) this.workspaceId = "home";
    // Three states rather than two, because "no projects" and "no connection" are
    // different answers and a user acts on them differently.
    this.mode = demoEnabled() ? "demo" : "disconnected";
    this.build();
    this.bindKeys();
    this.bindDrawing();
    this.connect();
  }

  /** The map tools report here, and what they draw becomes the selection. */
  bindDrawing() {
    configureDrawing({
      toast: (text, kind) => this.toast(text, kind),
      prompt: (label, fallback) => askOne(label, label, fallback),
      // Which screen the button was pressed on, for verbs shared by several toolbars.
      workspaceId: () => this.workspaceId,
      geometry: (geometry) => {
        this.selectionGeometry = geometry;
        this.selectionLabel.textContent = `selection: ${geometry.type}`;
      },
      changed: async () => {
        await this.refreshState();
        refreshLive();
      },
      // A measuring or selecting tool's shape: measured or kept the way the action that
      // armed it says (the canvas's own Measure button counts as the Measure action).
      drawn: (tool, geometry) => {
        let armed = this.armedTool && this.armedTool.tool === tool ? this.armedTool : null;
        if (!armed) {
          const found = Object.entries(ACTIONS).find(([, e]) => e && e.tool === tool);
          armed = { entry: found ? found[1] : {}, action: found ? found[0] : tool };
        }
        return this.onGeometryDrawn(armed.entry, armed.action, geometry);
      },
    });
    // The 3D viewer reports and asks through the same channels as the map drawing.
    configureViewer({
      toast: (text, kind) => this.toast(text, kind),
      prompt: (label, fallback) => askOne(label, label, fallback),
      changed: () => refreshLive(),
    });
    this.root.addEventListener("odk:say", (event) => {
      const { text, kind } = event.detail || {};
      if (text) this.toast(text, kind || "info");
    });
  }

  /** What a keyboard shortcut or its palette entry does. */
  shortcut(name) {
    switch (name) {
      case "palette": this.openPalette(); return;
      case "save": this.runAction("Save"); return;
      case "full": this.dock.toggleCanvasFullScreen(); return;
      case "sides": this.dock.toggleSidePanels(); return;
      case "fit": {
        const map = currentMap();
        if (!map) { this.toast("Fit: there is no map on this workspace.", "warn"); return; }
        if (!fitTo(map, map._odkBounds)) this.toast("Fit: nothing in this project to fit to yet.", "warn");
        return;
      }
      case "undo":
        this.toast(undoPoint() ? "Removed the last point." : "Nothing to undo: no shape is being drawn.",
          "info");
        return;
      case "stop":
        if (stopDrawing()) this.toast("Drawing cancelled.", "info");
        return;
      default:
    }
  }

  /**
   * Ask the bridge what is really there, and stop showing demo content the moment it
   * answers. Demo mode stays on if it was explicitly requested -- someone demonstrating
   * the product on a machine with real projects still wants the demo.
   */
  async connect() {
    // Drawn once from what is known now, so the frame never shows a stale literal
    // while the bridge is being asked.
    this.renderIndicators();
    const ready = await whenReady();
    if (!ready || !connected()) {
      // A slow start fires pywebviewready after whenReady() has given up. Staying
      // "disconnected" for the whole session over a few hundred milliseconds is the
      // wrong answer; connect when the bridge does arrive.
      if (!this.awaitingBridge && typeof window !== "undefined") {
        this.awaitingBridge = true;
        window.addEventListener("pywebviewready", () => { this.awaitingBridge = false; this.connect(); }, { once: true });
      }
      return;
    }
    if (this.mode === "demo") return;
    if (this.mode === "connected") return;
    this.mode = "connected";
    await this.refreshState();
    // The first workspace was rendered before the bridge answered, so every live panel
    // in it asked, got nothing, and showed its empty state ("No projects yet", "No
    // project open") while the status bar said "connected -- 1 project(s)". Render it
    // again now that there is something to ask.
    this.open(this.workspaceId);
    this.pollChrome();
  }

  /** Keep jobs and the Hub state current while the window is open. */
  pollChrome() {
    if (this.chromeTimer) return;
    const tick = async () => {
      const jobs = await tryCall("list_jobs");
      if (jobs) this.jobs = jobs.jobs || [];
      try {
        const response = await fetch("odk-runtime.json", { cache: "no-store" });
        this.hubRuntime = response.ok ? await response.json() : { hub_status: "not in desktop shell" };
      } catch {
        this.hubRuntime = { hub_status: "unknown" };
      }
      this.updateChrome();
    };
    tick();
    this.chromeTimer = setInterval(tick, 5000);
  }

  /**
   * Whether any visible panel is currently rendering the built-in sample.
   *
   * Uses the demo constants themselves as the test, so this cannot drift away from what
   * the panels actually draw: if a panel stops using DEMO.ORG the sentinel stops
   * matching, and if a new panel starts using it the banner appears without anyone
   * having to remember to wire it.
   *
   * A heuristic, deliberately biased towards showing the banner. A banner over real data
   * is a moment of confusion; real-looking numbers that were invented are the failure
   * this whole codebase is arranged to prevent.
   */
  showingSampleContent() {
    if (!this.workspaceEl) return false;

    /* Structural, because the string list did not work.
     *
     * This used to search the rendered text for sentinels: the demo organisation's name
     * and three demo site names. That catches a panel whose sample content happens to
     * mention a demo site, and misses every panel that invents something else. Fifty of
     * them do -- a Processing Queue of "#4471 / Feature matching / w-02", a Fleet
     * readout of "Aircraft 6, Available 4", flight telemetry, alerts, battery estimates,
     * maintenance records. None contains a sentinel, so all of them rendered in
     * connected mode with the banner hidden.
     *
     * That is the same failure the banner was written to fix, one level up: the
     * disclosure mechanism was itself a list of strings, so it could only disclose the
     * fabrications someone had already thought of.
     *
     * Now: anything that draws rows marks itself, live() marks its subtree as answered,
     * and the content is synthetic exactly when there are rows no API call produced.
     */
    const rows = this.workspaceEl.querySelectorAll("[data-rows]");
    for (const node of rows) {
      if (!node.closest("[data-live]") && !node.closest("[data-reference]")) return true;
    }

    // Kept as well. DATA-driven panels render real-looking prose rather than rows --
    // a project name in a heading, a site in a caption -- and those have no [data-rows]
    // to find.
    const text = this.workspaceEl.innerText || "";
    return !!text && SAMPLE_SENTINELS.some((mark) => mark && text.includes(mark));
  }

  /** The frame reflects the state: the banner only exists while content is synthetic. */
  applyMode() {
    if (!this.banner) return;
    // Synthetic whenever the panels are NOT showing what the application returned.
    // Keying this on demo mode alone was wrong and briefly made things worse: with no
    // bridge the panels still render the structural sample, so hiding the banner left
    // synthetic content on screen with nothing saying so. Connected is the only state
    // in which what you are looking at was measured.
    // The line above used to be `this.mode !== "connected"`, on the stated reasoning
    // that "connected is the only state in which what you are looking at was measured".
    // That premise is false, and the comment two lines up already says why: panels fall
    // back to the structural sample when they have nothing wired, and they do that in
    // connected mode too.
    //
    // Measured on the real window, connected, with a 77-image project open: Processing
    // showed "DEMO site 1/2/3" and "1,842 images", did NOT show the project's actual 77,
    // and the banner was collapsed to height 0 -- invented figures presented with the
    // disclaimer suppressed BECAUSE the app knew it was connected. That is the one thing
    // this project does not do.
    //
    // So the banner now follows what is on screen rather than what the bridge says.
    //
    // Disconnected no longer counts as synthetic by itself: with no bridge every live()
    // panel says "Not connected" rather than drawing anything, so there is no example on
    // screen to label, and "EXAMPLE DATA" over an empty cockpit was its own false claim.
    // Anything that still renders the sample is caught by showingSampleContent().
    const synthetic = this.mode === "demo" || this.showingSampleContent();
    this.banner.classList.toggle("hidden", !synthetic);
    this.updateChrome();
    if (this.connectionChip) {
      this.connectionChip.textContent =
        this.mode === "connected"
          ? `connected — ${this.projects ? this.projects.length : 0} project(s)`
          : this.mode === "demo"
            ? "demo mode"
            : "not connected to a project";
      this.connectionChip.className =
        this.mode === "connected" ? "chip ok" : "chip warn";
    }
  }

  build() {
    this.root.innerHTML = "";
    this.root.className = "shell";

    this.nav = this.buildNav();
    this.toolbar = el("div", { class: "toolbar" });
    this.workspaceEl = el("div", { class: "workspace" });
    this.status = this.buildStatus();

    // Above everything, in the frame itself, and impossible to scroll or clip away.
    // The previous marking was a chip at the far end of the status bar, which at 1600px
    // was cut off the screen entirely -- so the shell showed invented survey data with
    // nothing at all to say so.
    this.banner = el("div", { class: "demo-banner", text: DEMO.BANNER });
    this.banner.title = DEMO.PROVENANCE;
    // Hidden unless the content really is synthetic. A banner that is always there is
    // one nobody reads, which is how the old status-bar chip stopped working.
    this.banner.classList.toggle("hidden", this.mode !== "demo");
    this.root.append(this.banner, this.nav, this.toolbar, this.workspaceEl, this.status);

    this.dock = new Dock(this.workspaceEl);
    this.open(this.workspaceId);
    this.applyMode();
    this.installMenuBridge();

    // Every selection is echoed in the status bar. It is the cheapest possible proof
    // that the panels are wired to each other rather than merely sitting side by side.
    selection.on("*", (kind, value) => this.onSelection(kind, value));
  }

  /**
   * What happens when the user picks something in a panel.
   *
   * Every tree and table already published its selection onto the bus, and the only
   * subscriber wrote an 11px label into the status bar -- so clicking a job, a model or
   * an image looked exactly like clicking nothing. Publishing without a consequence is
   * the same failure as a button with no handler, one layer further in.
   *
   * Each branch below is the smallest honest consequence for that kind: show what is
   * known about the thing, or act on it. Where the application has nothing more to say,
   * it says that rather than staying silent.
   */
  onSelection(kind, value) {
    if (!value) return;
    const name = value.label || value.name || value.id || value.key || "—";
    this.selectionLabel.textContent = `${kind}: ${name}`;

    switch (kind) {
      case "job": {
        // Cancel needs a job to cancel, and picking one in the queue is how you say
        // which. It used to require a job id nobody could see.
        this.selectedJobId = value.job || value.id || null;
        if (this.selectedJobId) {
          this.toast(`Job ${this.selectedJobId} selected. Cancel acts on this one.`, "ok");
          this.watchJob(this.selectedJobId);
        }
        return;
      }
      case "model": {
        const model = (DATA.models || []).find((m) => m.key === name || m.key === value.key);
        this.toast(model
          ? `${model.key}: ${model.headline || "no headline metric"} · ${model.input_size}px · sha ${model.sha256}`
          : `${name}: not in the installed registry.`, "ok");
        return;
      }
      case "finding":
      case "anomaly": {
        // A finding is reviewable, and the review buttons ask for an id. Remembering it
        // here is what lets Accept/Reject/Flag act on what is on screen.
        // The table shows the first eight characters of the id; review needs all of it.
        // Selecting a row used to hand review_finding the short form, which matched no
        // annotation, so Accept on a selected finding always failed.
        this.selectedFindingId = value.full_id || value.id || value.label || null;
        this.toast(`${name} selected. Accept, Reject and Flag act on this finding.`, "ok");
        return;
      }
      case "image":
      case "capture": {
        // Fetch the actual frame and draw it. The UI is served from app/web, so it
        // cannot read a survey folder elsewhere on disk -- the bytes have to come back
        // through the bridge as a data URI or the canvas has nothing to show.
        this.showImage(String(value.label || value.name || name));
        return;
      }
      case "product": {
        const product = (DATA.products || {})[String(name).toLowerCase()];
        if (product) {
          setImage(null);
          setView(String(name).toLowerCase());
          this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
          this.toast(`Showing ${name}.`, "ok");
        } else {
          this.toast(`${name}: no rendered product for this one yet.`);
        }
        return;
      }
      case "report": {
        // Open Report acts on the report picked in the list.
        this.selectedReportId = value.id || null;
        this.toast(`${name} selected. Open Report opens it.`, "ok");
        return;
      }
      case "aircraft":
      case "battery":
      case "pilot": {
        // The fleet buttons take an id. Selecting the row is how you choose it.
        this.selectedFleetId = value.id || null;
        this.toast(`${kind} ${name} selected${this.selectedFleetId ? ` (id ${this.selectedFleetId})` : ""}.`, "ok");
        return;
      }
      case "project": {
        // Selecting a project used to announce that Open would act on it, which is a
        // sentence about a button rather than a consequence. Picking a project in the
        // list is the operator saying which one they want; open it.
        this.openProject(value.id ?? value.project_id, name);
        return;
      }
      case "layer": {
        // A layer is the reconstruction's own output -- an orthomosaic, a DSM, a camera
        // track. Naming it back at the operator was the clearest case of the panel
        // printing text: the application is holding the picture and describing it.
        this.showLayer(value.id || value.layer_id || value.key || name, name);
        return;
      }
      case "tool": {
        // The Measurements workspace lists its tools in a tree. Picking one used to print
        // its name; it now does what the toolbar button of the same name does.
        const tool = String(value.action || value.label || name).replace(/ line$/, "");
        if (resolve(tool)) this.runAction(tool);
        else this.toast(`${tool}: no such tool.`, "warn");
        return;
      }
      case "endpoint": {
        if (!value.method) return;
        this.selectedEndpoint = { method: value.method, path: value.path };
        this.toast(`${value.method} ${value.path}${value.summary ? ` — ${value.summary}` : ""}. `
          + "Copy cURL and Send Request act on this one.", "ok");
        return;
      }
      case "setting": {
        // Kept, so Plan uses what the operator typed. Without this the field updated
        // its own input and the planner ran on defaults -- the worst kind of broken,
        // because the screen agreed with the user and the output did not.
        this.settings = this.settings || {};
        this.settings[value.key] = value.value;
        this.toast(`${value.key} = ${value.value}`, "ok");
        return;
      }
      default:
        this.toast(`${kind}: ${name}`, "ok");
    }
  }

  /** Open the project the operator picked, and rebuild the screen around it. */
  async openProject(projectId, name) {
    if (projectId == null) {
      this.toast(`${name}: this project has no id, so it cannot be opened.`, "warn");
      return;
    }
    if (!connected()) {
      this.toast(`${name}: the Python bridge is not connected.`, "warn");
      return;
    }
    const opened = await tryCall("set_active_project", projectId);
    if (!opened) {
      this.toast(`${name}: ${lastError.get("set_active_project") || "could not be opened."}`, "warn");
      return;
    }
    // A different project means different datasets, layers, jobs and findings, so the
    // canvas must not keep showing the last one's picture.
    setImage(null);
    setView("map");
    await this.refreshState();
    refreshLive();
    this.toast(`Opened ${name}.`, "ok");
  }

  /**
   * Draw a map layer on the canvas.
   *
   * Rasters come back as a PNG data URI, which the canvas can already show. Vectors have
   * no picture, so rather than pretending, this reports what the layer actually contains
   * -- how many features and of what geometry. That is a real answer to "what is this",
   * which is the question selecting it was asking.
   */
  async showLayer(layerId, name) {
    if (!connected()) {
      this.toast(`${name}: connect a project to open layers.`, "warn");
      return;
    }
    const raster = await tryCall("raster_preview", layerId);
    if (raster && raster.data_uri) {
      setImage({ name, data_uri: raster.data_uri, coordinates: raster.coordinates });
      setView("image");
      this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
      this.toast(`Showing ${name}.`, "ok");
      return;
    }

    const vector = await tryCall("read_vector_layer", layerId);
    if (vector && vector.geojson) {
      const features = vector.geojson.features || [];
      const kinds = [...new Set(features.map((f) => (f.geometry || {}).type).filter(Boolean))];
      // Drawn, not just counted: CAD overlays, boundaries and camera tracks go on the map.
      showVector(String(layerId), vector.geojson, String(name).startsWith("CAD:") ? "cad" : "layer");
      // Leaving the previously drawn raster up would caption someone else's picture with
      // this layer's name. Go back to the map, which is where geometry belongs.
      setImage(null);
      setView("map");
      this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
      this.toast(
        `${name}: ${features.length} feature${features.length === 1 ? "" : "s"}`
        + `${kinds.length ? ` (${kinds.join(", ")})` : ""}.`,
        "ok",
      );
      return;
    }

    // Both refused, so say which reason applies rather than falling back to the name.
    const why = lastError.get("raster_preview") || lastError.get("read_vector_layer");
    this.toast(`${name}: ${why || "this layer cannot be displayed."}`, "warn");
  }

  /**
   * Wait for one click on the photograph shown on the canvas.
   *
   * The canvas draws a downscaled preview with object-fit: contain, so the click is
   * mapped through the letterboxing to the preview's pixels and then scaled to the
   * source frame's size -- a ground control mark is a pixel on the photograph, not on
   * the screen.
   */
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
    return new Promise((resolve) => {
      const done = (value) => {
        image.removeEventListener("click", onClick);
        document.removeEventListener("keydown", onKey);
        resolve(value);
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
        // A close-up covers a window of the frame, not all of it.
        const [cx, cy, cw, ch] = picked.crop || [0, 0, width, height];
        done({ x: cx + (x * cw) / nw, y: cy + (y * ch) / nh, width, height });
      };
      image.addEventListener("click", onClick);
      document.addEventListener("keydown", onKey);
    });
  }

  /** Draw one image from the active dataset on the canvas. */
  async showImage(name, crop = null) {
    if (!connected()) {
      this.toast(`${name}: connect a project to preview the image.`, "warn");
      return;
    }
    this.toast(`Loading ${name}…`);
    const preview = crop ? await tryCall("image_preview", name, 1400, crop)
      : await tryCall("image_preview", name);
    if (!preview || preview.ok === false) {
      this.toast(`${name}: ${(preview && preview.error) || "no preview available."}`, "warn");
      return;
    }
    setImage(preview);
    setView("image");
    this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
    this.toast(`${preview.name} — ${preview.source_width}×${preview.source_height}`, "ok");
  }

  /**
   * The native menu bar, which did nothing at all in this UI.
   *
   * app/shell.py dispatches every menu item as `window.odk.onMenu({action, payload})`.
   * That hook was defined only in app.js -- the classic single-screen UI -- so in the
   * cockpit all thirty-three items were inert: File, Mission, Fly, Analysis, Reconstruct,
   * View, Tools, Help. Nothing threw, because evaluate_js on an undefined property is
   * simply a no-op, which is why it looked like the menu was decorative.
   *
   * Each entry maps to the SAME handler the toolbar uses, rather than a second
   * implementation of the same verb. A menu item and a button that drift apart are two
   * behaviours for one action, and the one nobody tests is the one that breaks.
   */
  installMenuBridge() {
    /* Every item in app/shell.py's menu bar is handled here; tests/test_menu_coverage.py
       holds the two lists together. Seven used to fall through to "Menu action not
       handled here" (Connect Vehicle, Arm, Start, Pause, Resume, Return to Home, Abort)
       and several reached the wrong verb: Model Manager started the full reconstruction
       pipeline, Mission History ran Simulate, Environment Capabilities ran Preflight. */
    const toWorkspace = {
      "mission.plan": "planning",
      "mission.settings": "planning",
      "mission.history": "planning",
      "fly.connect": "flight",
      "fly.command": "flight",
      "fly.disconnect": "flight",
      "fly.upload": "flight",
      "recon.run": "processing",
      "recon.settings": "processing",
      "analysis.pipeline": "inspection",
      "analysis.models": "settings",
      "tools.measure": "measurements",
    };

    const toAction = {
      "project.new": "New Project",
      "project.open": "Open",
      "data.import_imagery": "Import",
      "data.import_raster": "Add Layer",
      "data.import_vector": "Add Layer",
      "mission.plan": "Plan",
      "mission.save": "Save",
      "mission.export": "Export",
      "mission.export_all": "Export",
      "mission.history": "History",
      "recon.run": "Process",
      "analysis.pipeline": "Start",
      "analysis.models": "Verify Models",
      "fly.upload": "Upload",
    };

    // Where each View-menu panel lives in the cockpit.
    const panelWorkspace = { layers: "twin", mission: "planning", telemetry: "flight", jobs: "processing" };

    // The vehicle verbs, with the same confirmation the flight toolbar asks for.
    const vehicleQuestion = {
      arm: "Arm the aircraft?",
      start: "Start the uploaded mission?",
      pause: "Pause the aircraft?",
      resume: "Resume the mission?",
      rth: "Return to home?",
      abort: "ABORT the flight?",
    };

    window.odk = window.odk || {};
    window.odk.onMenu = async ({ action, payload } = {}) => {
      if (!action) return;
      payload = payload || {};

      // Jump to where the action belongs first, so the user can see what it did.
      const workspace = toWorkspace[action];
      if (workspace && WORKSPACE_BY_ID[workspace]) this.open(workspace);

      const mapped = toAction[action];
      if (mapped) {
        await this.runAction(mapped);
        return;
      }

      try {
        await this.handleMenu(action, payload, { panelWorkspace, vehicleQuestion });
      } catch (error) {
        this.toast(`${action}: ${error.message || error}`, "error");
      }
    };
  }

  /** Menu items that are not one toolbar verb. Each one reaches the Api or says why not. */
  async handleMenu(action, payload, { panelWorkspace, vehicleQuestion }) {
    switch (action) {
      case "project.reveal":
      case "recon.reveal": {
        const state = await tryCall("get_state");
        const root = state && state.project && (state.project.root_dir || state.project.path);
        if (root) await tryCall("open_path", String(root));
        this.toast(root ? `Opened ${root}` : "No project folder to open.", root ? "ok" : "warn");
        return;
      }
      case "mission.clear_aoi":
        await call("set_aoi", []);
        this.toast("Area of interest cleared.", "ok");
        return;
      case "mission.clear_nofly":
        await call("set_no_fly_zones", []);
        this.toast("No-fly zones cleared.", "ok");
        return;
      case "mission.settings":
        this.toast("Mission settings are the parameter panel in this Planning workspace; Plan uses what is typed there.", "ok");
        return;
      case "recon.settings":
        this.toast("Reconstruction settings are the Processing Settings panel here; Process uses them.", "ok");
        return;
      case "data.import_terrain": {
        const file = await call("pick_file", ["tif", "tiff", "asc", "csv"]);
        if (!file || !file.path) return;
        await call("set_terrain_source", file.path);
        this.toast(`Terrain source: ${file.path}`, "ok");
        return;
      }
      case "fly.connect":
        // The Connect action offers every driver the Api accepts (pymavlink, MAVSDK,
        // simulated) and reports a failed link as a failure rather than "connected".
        await this.runAction("Connect");
        return;
      case "fly.disconnect":
        await this.runAction("Disconnect");
        return;
      case "fly.command": {
        const command = String(payload.command || "");
        if (!command) return;
        if (!connected()) { this.toast(`${command}: the Python bridge is not connected.`, "warn"); return; }
        const question = vehicleQuestion[command] || `Send "${command}" to the aircraft?`;
        if (!this.confirmAll && !(await confirmAsk(question))) return;
        /* What the vehicle answered, not that something was sent: call() throws on a
           denial or on silence, and an acceptance says so. */
        const result = await call("vehicle_command", command);
        const detail = result && result.result;
        this.toast(`${command}: ${(detail && detail.message) || "accepted by the vehicle."}`, "ok");
        await this.refreshState();
        return;
      }
      case "view.panel": {
        const target = panelWorkspace[payload.panel];
        if (!target) { this.toast(`No ${payload.panel} panel in the cockpit.`, "warn"); return; }
        this.open(target);
        this.toast(`${payload.panel} panel: ${WORKSPACE_BY_ID[target].title} workspace.`, "ok");
        return;
      }
      case "view.zoom_aoi":
      case "view.zoom_mission":
        setView("map");
        setImage(null);
        this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
        this.toast("Back to the map.", "ok");
        return;
      case "view.basemap": {
        const key = String(payload.basemap || "");
        if (setBasemap(key)) this.toast(`Basemap: ${key}.`, "ok");
        else this.toast(`Unknown basemap ${key}. Available: ${basemapNames().join(", ")}.`, "warn");
        return;
      }
      case "view.cache_tiles": {
        const view = visibleBounds();
        if (!view) { this.toast("Open a workspace with a map to choose the area to cache.", "warn"); return; }
        const minZoom = Math.max(0, Math.floor(view.zoom));
        const maxZoom = Math.min(19, minZoom + 3);
        const result = await call("cache_basemap_area", view.west, view.south, view.east, view.north, minZoom, maxZoom, "satellite");
        this.toast(`Caching ${result.tiles} tiles (z${minZoom}-${maxZoom}) for offline use…`, "ok");
        this.watchJob(result.job_id);
        return;
      }
      case "tools.measure":
        await this.runAction(payload.mode === "area" ? "Area" : "Distance");
        return;
      case "tools.capabilities": {
        const caps = await call("capabilities");
        const lines = Object.entries(caps || {})
          .filter(([key]) => key !== "ok")
          .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`);
        await reveal("Environment capabilities", lines.join("\n") || "Nothing reported.");
        return;
      }
      case "tools.audit": {
        const log = await call("audit_log", 100);
        const lines = ((log && log.events) || []).map((e) => { const l = auditLine(e); return `${l.t}  ${l.text}`; });
        await reveal("Audit log (newest first)", lines.join("\n") || "No events recorded for this project.");
        return;
      }
      /* The Reports menu the default window gained. Each goes where the cockpit already
         does the same thing, or asks the Api directly. */
      case "report.generate":
        this.open("reports");
        await this.runAction("Generate Report");
        return;
      case "report.export":
        this.open("reports");
        await this.runAction(String(payload.format || "pdf").toLowerCase() === "docx" ? "Export Word" : "Export PDF");
        return;
      case "report.list":
        this.open("reports");
        await this.runAction("Open Report");
        return;
      case "report.finding":
        this.open("reports");
        await this.runAction("Record Finding");
        return;
      case "report.review":
        this.open("inspection");
        this.toast("Findings and review: select a finding in AI Inspection, then Accept, Reject or Flag.", "ok");
        return;
      case "report.provenance": {
        const result = await call("verify_provenance");
        const lines = [
          `Folder: ${result.directory || ""}`,
          `Verified: ${(result.verified || []).length} of ${result.total || 0}`,
          `Unattributed: ${(result.unattributed || []).length}`,
          `Modified: ${(result.modified || []).length}`,
          ...Object.entries(result.results || {}).map(([name, r]) => `${name}: ${r.status}`),
        ];
        await reveal("Provenance", lines.join("\n"));
        return;
      }
      case "report.audit_export": {
        const result = await call("export_audit_log", "csv");
        this.toast(`Audit log: ${result.events} event(s) written to ${result.path}.`, "ok");
        return;
      }
      case "report.notifications": {
        const result = await call("list_notifications");
        const lines = (result.notifications || []).map((n) =>
          `${String(n.created_utc || "").slice(0, 19)}  ${n.read ? " " : "•"} ${n.title}${n.detail ? ` — ${n.detail}` : ""}`);
        await reveal(`Notifications (${result.unread || 0} unread)`, lines.join("\n") || "No notifications.");
        return;
      }
      case "help.shortcuts":
        this.openPalette();
        return;
      case "help.about":
        this.toast("OpenDroneKit — offline-first drone inspection toolkit.", "ok");
        return;
      default:
        // Named rather than ignored: a menu item nobody wired should say so, in the
        // same way an unwired button does.
        this.toast(`Menu action not handled here: ${action}`, "warn");
    }
  }

  buildNav() {
    const items = el("div", { class: "nav-items" });
    for (const workspace of WORKSPACES) {
      const button = el("button", {
        class: "nav-item" + (workspace.id === this.workspaceId ? " active" : ""),
        text: workspace.title,
      });
      button.onclick = () => this.open(workspace.id);
      items.appendChild(button);
    }
    this.navItems = items;

    const search = el("div", { class: "searchbox" }, [
      el("span", { text: "⌕" }),
      el("span", { text: "Search projects, missions, findings…", style: "flex:1 1 auto" }),
      el("kbd", { text: "Ctrl K" }),
    ]);
    search.onclick = () => this.openPalette();

    /* Every indicator here used to be a literal: a green "API" dot whether or not a
       bridge existed, "3 workers" in a build that runs jobs on threads in this process,
       the avatar "PK" and "DEMO organisation" over a real project, and a clock stopped
       at the epoch. Each is now set from the connection state by updateChrome(). */
    this.bridgeDot = el("span", { class: "status-dot idle" });
    this.bridgeText = el("span", { class: "nav-bridge", text: " bridge…" });
    this.jobsDot = el("span", { class: "status-dot idle" });
    this.jobsText = el("span", { class: "nav-jobs", text: " jobs —" });
    this.orgText = el("span", { class: "nav-project", text: "" });
    // The Hub the desktop shell starts (odk-runtime.json), polled by pollChrome().
    this.hubDot = el("span", { class: "status-dot idle" });
    this.hubText = el("span", { class: "nav-hub", text: " Hub —" });
    return el("header", { class: "topnav" }, [
      el("div", { class: "brand" }, [el("span", { class: "mark" }), el("span", { text: "OpenDroneKit" })]),
      items,
      el("div", { class: "nav-right" }, [
        search,
        el("span", { class: "sep" }),
        el("span", { title: "Connection to the Python application" }, [this.bridgeDot, this.bridgeText]),
        el("span", { title: "Background jobs in this application" }, [this.jobsDot, this.jobsText]),
        el("span", { title: "The Hub web service this window started" }, [this.hubDot, this.hubText]),
        // The aircraft link, on every screen: a lost link must be visible in the frame.
        this.vehicleIndicator = el("span", { title: "Aircraft link" }, [
          this.vehicleDot = el("span", { class: "status-dot idle" }),
          this.vehicleLabel = el("span", { text: " no aircraft" }),
        ]),
        el("span", { title: "Projects are stored on this machine" }, [
          el("span", { class: "status-dot idle" }), el("span", { text: " local-first" })]),
        el("span", { class: "sep" }),
        this.clock = el("span", { class: "mono", text: "--:--:-- UTC" }),
        el("span", { class: "sep" }),
        this.orgText,
      ]),
    ]);
  }

  buildStatus() {
    this.selectionLabel = el("span", { text: "no selection" });
    // Was DEMO.SITES[0] -- "OpenDroneKit Aukerman survey" -- whatever project was open.
    this.siteDot = el("span", { class: "status-dot idle" });
    this.siteText = el("span", { class: "status-project", text: " not connected" });
    this.crsText = el("span", { class: "mono", text: "map EPSG:4326" });
    return el("footer", { class: "statusbar" }, [
      el("span", {}, [this.siteDot, this.siteText]),
      el("span", { class: "sep" }),
      this.crsText,
      el("span", { class: "sep" }),
      this.selectionLabel,
      el("span", { class: "spacer" }),
      // The banner above carries this now. A chip at the end of the status bar was the
      // only marking on the whole shell, and at 1600px it was clipped off the screen --
      // so the one thing saying "none of this was measured" was the first thing to go.
      this.connectionChip = el("span", { class: "chip warn", text: "not connected to a project" }),
      el("span", { class: "sep" }),
      el("span", { class: "mono", text: "v1.0" }),
    ]);
  }

  open(id) {
    const workspace = WORKSPACE_BY_ID[id];
    if (!workspace) return;
    this.workspaceId = id;
    // Each workspace opens on its own canvas. Carrying a thermal or point-cloud view
    // across a workspace switch leaves the operator looking at the previous instrument
    // through the new one, and there would be no way back to the map.
    setView("map");
    setImage(null);
    this.canvasView = "map";
    localStorage.setItem(LAST_WORKSPACE, id);

    [...this.navItems.children].forEach((button, index) =>
      button.classList.toggle("active", WORKSPACES[index].id === id));

    this.buildToolbar(workspace);
    this.dock.render(workspace);

    // Re-check per workspace: whether the panels on screen are showing the sample varies
    // between them, so a banner decided once at startup would be wrong on most of them.
    // Deferred a tick because the dock's panels populate asynchronously and the text to
    // test does not exist yet on this one.
    setTimeout(() => this.applyMode(), 0);
  }

  buildToolbar(workspace) {
    this.toolbar.innerHTML = "";
    for (const action of workspace.toolbar || []) {
      if (action === "|") { this.toolbar.appendChild(el("span", { class: "tsep" })); continue; }
      const classes = ["tbtn"];
      if (DANGER.has(action)) classes.push("danger");
      else if (PRIMARY.has(action)) classes.push("primary");
      const button = el("button", { class: classes.join(" "), text: action });
      button.onclick = () => this.runAction(action);
      this.toolbar.appendChild(button);
    }

    this.toolbar.appendChild(el("span", { class: "tsep" }));

    const layout = el("button", { class: "tbtn", text: "Layout ▾" });
    layout.onclick = () => this.layoutMenu(layout);
    this.toolbar.appendChild(layout);

    const full = el("button", { class: "tbtn", text: "⛶ Canvas" });
    full.onclick = () => this.dock.toggleCanvasFullScreen();
    this.toolbar.appendChild(full);
  }

  layoutMenu(anchor) {
    const existing = document.querySelector(".palette-backdrop");
    if (existing) existing.remove();
    const hidden = this.dock.hiddenPanels();
    const items = [
      { label: "Reset this workspace layout", run: () => this.dock.resetLayout() },
      ...hidden.map((def) => ({ label: `Show “${def.title}”`, run: () => this.dock.showPanel(def.id) })),
    ];
    this.palette(items, "Layout");
  }

  /**
   * Run a toolbar action against the real application.
   *
   * Everything is reported where the user is looking -- a toast over the canvas -- and
   * appended to the console. The old version wrote to the status bar in 11px text at the
   * bottom of the screen, which is indistinguishable from the app doing nothing.
   */
  async runAction(action) {
    const missing = UNWIRED[action];
    if (missing) {
      this.toast(`${action}: not available. ${missing}`, "warn");
      return;
    }

    const entry = resolve(action);
    if (!entry) {
      this.toast(`${action}: no handler. This is a gap, not a refusal.`, "warn");
      return;
    }

    // Local behaviour BEFORE the prerequisite gate. A view mode, a canvas tool and a
    // workspace jump are client decisions that need no bridge -- and the gate refuses
    // everything with "Not connected to the application" when there is none, which
    // silently disabled every view button in a browser and in the demo.
    if (entry.view) {
      this.canvasView = entry.view;
      this.workspaceEl.dataset.view = entry.view;
      // Repaint, rather than setting an attribute nobody reads. Recording the mode and
      // leaving the canvas untouched is exactly as dead as the stub this replaced --
      // the button reported success and the screen did not move.
      setView(entry.view);
      this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
      this.toast(`View: ${action}`, "ok");
      return;
    }
    if (entry.tool) {
      // Armed on the map drawing layer, which reads the clicks. Storing the name on the
      // shell and toasting "Draw on the canvas" was all this did before, and nothing ever
      // read it back: Distance and Area took no measurement at all.
      // A 3D model on screen takes the tool; otherwise the map drawing layer does.
      // Which action armed it, so a finished shape is measured the way that action says.
      this.armedTool = { tool: entry.tool, entry, action };
      const onModel = currentViewer();
      let said = onModel ? armModelTool(entry.tool) : null;
      const onMap = !said;
      if (onMap) said = armTool(entry.tool);
      this.activeTool = entry.tool;
      this.workspaceEl.dataset.tool = entry.tool;
      if (said) this.toast(`${action}: ${said}`, !onMap || currentMap() ? "ok" : "warn");
      else this.toast(`${action}: this tool draws on a 3D model, and needs the 3D view open. `
        + "Use Textured Mesh or Point Cloud first.", "warn");
      return;
    }
    if (entry.workspace) {
      this.open(entry.workspace);
      this.toast(`Opened ${action}.`, "ok");
      return;
    }

    const blocked = prerequisite(action, this.stateSummary());
    if (blocked) {
      this.toast(`${action}: ${blocked}`, "warn");
      return;
    }

    // Anything that starts real work or moves an aircraft asks first. A misclick on
    // Abort must not be the same gesture as a misclick on Pan.
    const question = entry.confirm;
    // confirmAll lets the click harness exercise the confirming actions; a real session
    // never sets it, so a user always gets the question.
    if (question && !this.confirmAll && !(await confirmAsk(question))) return;

    this.toast(`${action}…`);
    try {
      const result = entry.vehicle
        ? await this.runVehicle(entry.vehicle, action)
        : await entry.run(this.actionContext());
      if (result && result.skipped) {
        this.toast(`${action}: ${result.skipped}`);
        return;
      }
      const message = (result && result.message) || `${action} done.`;
      this.toast(message, (result && result.level) || "ok");
      if (result && result.job) this.watchJob(result.job);
      if (result && result.refresh) {
        await this.refreshState();
        // A new overlay or marker belongs on the map the operator is looking at.
        refreshProjectOverlays();
        this.dock.refreshLive();
        // Live panels (and the maps) re-read what the action changed.
        refreshLive();
      }
    } catch (error) {
      // Shown, not swallowed. A refusal from the Api is the most useful thing this
      // application produces and it used to vanish into a rejected promise.
      this.toast(`${action}: ${error.message || error}`, "error");
    }
  }

  /**
   * A line or polygon was finished on the map.
   *
   * It becomes the selection that Volume, Cut & Fill, Slope, Stockpile and Compare act
   * on. For the measuring tools it is also measured straight away, on the project's own
   * rasters -- the browser never works out a length itself.
   */
  async onGeometryDrawn(entry, action, geometry) {
    this.selectionGeometry = geometry;
    if (this.selectionLabel) this.selectionLabel.textContent = `selection: ${geometry.type}`;
    const points = vertices(geometry);
    const pending = this.pendingSelectionAction;
    this.pendingSelectionAction = null;
    if (pending) {
      await this.runAction(pending);
      return;
    }
    if (!entry.measure) {
      this.toast(`${geometry.type === "Polygon" ? "Area" : "Line"} selected `
        + `(${points.length} points). Volume, Cut & Fill, Slope and Stockpile use it.`, "ok");
      return;
    }
    if (!connected()) {
      this.toast(`${action}: connect to the application to measure.`, "warn");
      return;
    }
    try {
      /* Two measurements, each saying what it is. The drawn positions give a horizontal
         length or area on the ground anywhere (measure_geometry); the project's own
         georeferenced raster, when there is one, gives the planar figure in its CRS
         and the length over the surface (measure_drawn). A profile only exists on the
         raster. Neither is worked out in the browser. */
      const parts = [];
      if (entry.measure === "distance" || entry.measure === "area") {
        const flat = await call("measure_geometry", points, entry.measure);
        if (entry.measure === "distance") {
          recordMeasurement({ kind: "distance", value: flat.length_m, unit: "m", vertices: flat.vertices });
          parts.push(`Distance ${formatLength(flat.length_m)} over ${flat.vertices} points (${flat.method}).`);
        } else {
          recordMeasurement({ kind: "area", value: flat.area_m2, unit: "m²", perimeter_m: flat.perimeter_m });
          parts.push(`Area ${formatArea(flat.area_m2)}, perimeter ${formatLength(flat.perimeter_m)}. `
            + "Volume, Cut & Fill and Slope act on it.");
        }
        try {
          const surface = await call("measure_drawn", entry.measure, points);
          this.lastMeasurement = surface;
          parts.push(`On the project raster: ${describeMeasurement(surface)}`);
        } catch (error) {
          // No raster in the project is not an error of this measurement; any other
          // refusal (a raster in feet, an unprojected CRS) is shown as one.
          const why = String(error.message || error);
          if (!/no georeferenced raster/i.test(why)) parts.push(`Raster measurement refused: ${why}`);
        }
        this.toast(parts.join(" "), "ok");
      } else {
        const result = await call("measure_drawn", entry.measure, points);
        this.lastMeasurement = result;
        this.toast(`${action}: ${describeMeasurement(result)}`, "ok");
      }
      this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
    } catch (error) {
      this.toast(`${action}: ${error.message || error}`, "error");
    }
  }

  /**
   * The drawn selection, or -- when there is none of the right shape -- start drawing
   * one and run `action` again once it is finished. Returns the vertices or null.
   */
  requireSelection(shape, action, prompt) {
    const geometry = this.selectionGeometry;
    const wanted = shape === "polygon" ? "Polygon" : "LineString";
    if (geometry && geometry.type === wanted) return vertices(geometry);
    this.pendingSelectionAction = action;
    this.activeTool = `select-${shape}`;
    armDrawing(shape, (drawn) => this.onGeometryDrawn({}, action, drawn));
    this.toast(prompt || `${action}: draw the ${shape === "polygon" ? "area" : "line"} on `
      + "the map, then double-click or press Enter.", "ok");
    return null;
  }

  async runVehicle(command, action) {
    /* What the vehicle answered, not that something was sent: call() throws on a
       denial or on silence, and an acceptance says so. */
    const result = await call("vehicle_command", command);
    const answer = (result && result.result && result.result.message) || "accepted by the vehicle.";
    return { message: `${action}: ${answer}`, refresh: true };
  }

  /** What the actions need from the shell, in one place they can all reach. */
  actionContext() {
    return {
      // Drawn by the application, not by the webview. WebView2 does not implement
      // window.prompt and pywebview does not route script dialogs, so every one of
      // these returned null instantly and the action stopped with "no name given" --
      // twenty-six buttons that looked completely dead and raised nothing.
      prompt: (label, fallback) => askOne(label, label, fallback),
      choose: (title, options) => chooseModal(title, options),
      ask: (title, fields) => ask(title, fields),
      showResult: (title, content) => showResult(title, content),
      labelBoxes: (title, spec) => labelBoxes(title, spec),
      form: (title, fields) => ask(title, fields),
      missionOptions: () => {
        // What the mission panels were actually edited to, under the names the planner
        // reads.
        //
        // The form's keys are short -- type, alt, fwd, side, standoff -- and plan_mission
        // reads template, altitude_m, front_overlap_pct, side_overlap_pct, standoff_m.
        // Not one of them matched. opts.get(...) found nothing every time and fell back
        // to its defaults, so pressing Plan produced a nadir GRID at 55 m no matter what
        // the operator typed: mission type, altitude, overlap, stand-off and speed were
        // all discarded silently.
        //
        // Measured before the fix: {type: "Facade inspection", alt: 40, standoff: 12}
        // planned template=grid at altitude 55. That is the worst shape a bug can take
        // here -- a plausible mission that is not the one that was asked for, with
        // nothing on screen to say so.
        const raw = this.settings || {};
        const numeric = {};
        for (const [key, value] of Object.entries(raw)) {
          const asNumber = Number(value);
          numeric[key] = value !== "" && Number.isFinite(asNumber) ? asNumber : value;
        }

        const options = {};
        for (const [formKey, apiKey] of Object.entries(MISSION_OPTION_KEYS)) {
          if (numeric[formKey] !== undefined && numeric[formKey] !== "") {
            options[apiKey] = numeric[formKey];
          }
        }
        // Anything the form gained since this map was written is passed through under its
        // own name, so a new field reaches the planner if the planner happens to read it.
        // The known-inert ones are deliberately NOT passed: sending "gsd" to an API that
        // ignores it is indistinguishable from wiring it.
        for (const [key, value] of Object.entries(numeric)) {
          if (key === "type") continue;
          if (key in MISSION_OPTION_KEYS) continue;
          if (MISSION_FIELDS_NOT_PLANNED.has(key)) continue;
          if (!(key in options)) options[key] = value;
        }

        if (numeric.type !== undefined) {
          const template = MISSION_TEMPLATES[String(numeric.type)];
          if (template) options.template = template;
        }
        return options;
      },
      reconstructionOptions: () => {
        // Was a hardcoded {engine: "auto", profile: "standard"}, so the Processing
        // Settings panel -- Profile, Max image size, Dense cloud, Mesh depth -- was read
        // by nobody. An operator selecting the high-accuracy profile, or turning dense
        // off to get a result before lunch, got a standard run with dense left at its
        // default either way, and nothing said so.
        //
        // run_reconstruction reads engine, profile, dense and epsg. profile and dense
        // were already the panel's own field names; they simply never left the browser.
        const raw = this.settings || {};
        const options = { engine: "auto", profile: "standard" };

        if (raw.profile) options.profile = String(raw.profile);
        // The select offers yes/no; the Api takes a boolean or nothing at all, and
        // "no" is truthy in JavaScript. Passing the string straight through would have
        // turned dense ON when the operator asked for it off.
        if (raw.dense !== undefined && raw.dense !== "") {
          options.dense = String(raw.dense).toLowerCase() === "yes";
        }
        // Chosen with Set CRS. Empty means the reconstruction picks the UTM zone itself.
        if (raw.epsg) options.epsg = Number(raw.epsg);
        return options;
      },
      // The pipeline also runs coverage QA, which sizes each photograph's footprint from
      // its height above ground. Reconstruction alone does not read it.
      pipelineOptions: () => {
        const raw = this.settings || {};
        const options = this.actionContext().reconstructionOptions();
        const agl = Number(raw.agl);
        if (raw.agl !== undefined && raw.agl !== "" && Number.isFinite(agl) && agl > 0) {
          options.flight_height_agl_m = agl;
        }
        return options;
      },
      /* Run AI: the reconstruction settings plus what the Run AI Settings panel says
       * the asset is and which optional detectors to run. "auto" sends nothing, so
       * the pipeline decides from the declared structure and material. */
      inspectionOptions: () => {
        const raw = this.settings || {};
        const options = this.actionContext().pipelineOptions();
        for (const key of ["structure_type", "material_type"]) {
          if (raw[key]) options[key] = String(raw[key]);
        }
        for (const key of ["run_solar_detection", "run_metal_detection", "crack_presence_gate"]) {
          const value = String(raw[key] ?? "").toLowerCase();
          if (value === "on") options[key] = true;
          else if (value === "off") options[key] = false;
        }
        return options;
      },
      /* What the Report panel was set to.
       *
       * Generate Report used to call generate_report("", title, "standard", "") with the
       * type and author hardcoded, so Report, Format, Organisation and Logo were all
       * read by nobody -- the third panel in this shell with that fault. */
      reportOptions: () => {
        const raw = this.settings || {};
        const format = String(raw.fmt || "PDF").toLowerCase();
        const formats = { pdf: ["html", "pdf"], docx: ["html", "docx"],
                          "pdf+docx": ["html", "pdf", "docx"], html: ["html"] }[format]
          || ["html", "pdf"];
        return {
          reportType: String(raw.tpl || "standard"),
          format,
          formats,
          organization: String(raw.org || ""),
          client: String(raw.client || ""),
          author: String(raw.author || ""),
          logo: String(raw.logo || ""),
        };
      },
      selectedReport: () => this.selectedReportId || null,
      openWorkspace: (id) => this.open(id),
      // Which workspace the button was pressed on, for a toolbar name two screens share.
      workspaceId: () => this.workspaceId,
      selectedJob: () => this.selectedJobId || null,
      selectedFinding: () => this.selectedFindingId || null,
      selectedFleetId: () => this.selectedFleetId || null,
      selectionGeometry: () => this.selectionGeometry || null,
      selectionVertices: () => vertices(this.selectionGeometry),
      // The drawn area or line, or null after starting a drawing that will re-run the
      // action once it is finished.
      requireSelection: (shape, action, prompt) => this.requireSelection(shape, action, prompt),
      // One click on the map, as [lon, lat]; null when there is no map to click (the
      // caller then asks for typed coordinates) or when the operator presses Esc.
      pickPoint: (message) => {
        if (!canDraw()) return Promise.resolve(null);
        this.toast(message || "Click the position on the map (Esc to type it instead).", "ok");
        return new Promise((resolve) => {
          armDrawing("point", (geometry) => resolve(geometry.coordinates), () => resolve(null));
        });
      },
      // Switch the canvas to a view (the mesh, say), as the view buttons do.
      showView: (view) => {
        this.canvasView = view;
        this.workspaceEl.dataset.view = view;
        setImage(null);
        setView(view);
        this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
      },
      toast: (text, kind) => this.toast(text, kind || "ok"),
      // Put one dataset photograph on the canvas.
      showImage: (name, crop) => this.showImage(name, crop),
      // One click on the photograph on the canvas, in the photograph's own full-
      // resolution pixels: {x, y, width, height}. null on Esc or a click off the picture.
      pickImagePixel: (message) => this.pickImagePixel(message),
      selectedEndpoint: () => this.selectedEndpoint || null,
      setSetting: (key, value) => {
        this.settings = this.settings || {};
        this.settings[key] = value;
      },
      resetLayout: () => this.dock.resetLayout(),
      // The organisation and project the session is actually on. Defaulting to 1 rather
      // than refusing keeps a single-tenant desktop install usable; a multi-tenant
      // deployment sets them from the opened project.
      organizationId: () => (this.state && this.state.organization_id) || 1,
      projectId: () => (this.state && this.state.project && this.state.project.id) || 1,
      // A secret the application will never show again. A toast disappears after four
      // seconds, which is not long enough to copy a token, so this blocks until the
      // user dismisses it.
      reveal: (text, title) => reveal(title || "Copy this now — it is not stored", text),
    };
  }

  stateSummary() {
    return {
      projectOpen: Boolean(this.state && this.state.project),
      datasetSelected: Boolean(this.state && this.state.dataset),
    };
  }

  /** The header and status bar, from what the application actually reported. */
  updateChrome() {
    if (!this.bridgeDot) return;
    const project = (this.state && this.state.project) || null;
    const jobs = this.jobs || [];
    const running = jobs.filter((j) =>
      ["running", "queued", "pending"].includes(j.status || j.state)).length;

    if (this.mode === "demo") {
      this.bridgeDot.className = "status-dot idle";
      this.bridgeText.textContent = " demo — not connected";
      this.jobsDot.className = "status-dot idle";
      this.jobsText.textContent = " no jobs (example)";
      this.orgText.textContent = DEMO.ORG;
    } else if (this.mode === "connected") {
      this.bridgeDot.className = "status-dot ok";
      this.bridgeText.textContent = " bridge";
      this.jobsDot.className = `status-dot ${running ? "ok" : "idle"}`;
      this.jobsText.textContent = running
        ? ` ${running} job${running === 1 ? "" : "s"} running`
        : ` no jobs running${jobs.length ? ` (${jobs.length} recorded)` : ""}`;
      this.orgText.textContent = project ? String(project.name || "") : "";
    } else {
      this.bridgeDot.className = "status-dot warn";
      this.bridgeText.textContent = " no bridge";
      this.jobsDot.className = "status-dot idle";
      this.jobsText.textContent = " jobs unknown";
      this.orgText.textContent = "";
    }

    // The Hub service the desktop shell started, as its runtime file reports it.
    if (this.hubDot) {
      const hub = this.hubRuntime || {};
      const live = this.mode === "connected";
      const kind = hub.hub_status === "running" ? "ok" : hub.hub_status === "starting" ? "idle" : "warn";
      this.hubDot.className = `status-dot ${live ? kind : "idle"}`;
      this.hubText.textContent = live ? ` Hub ${hub.hub_status || "unknown"}` : " Hub —";
      this.hubText.title = hub.hub_error || (hub.hub_api ? `Hub service at ${hub.hub_api}` : "");
    }

    const name = this.mode === "demo" ? DEMO.SITES[0] : project && project.name;
    this.siteDot.className = `status-dot ${name ? "ok" : "idle"}`;
    this.siteText.textContent = name ? ` ${name}`
      : this.mode === "connected" ? " no project open" : " not connected";
    // The map is drawn in WGS84 longitude/latitude, which is what the cursor readout
    // reports. The project's own CRS, when it has one, is a different thing and is named
    // beside it rather than conflated with it.
    const layerCrs = ((this.state && this.state.layers) || [])
      .map((l) => l.crs_epsg).find((epsg) => epsg && Number(epsg) !== 4326);
    const projectCrs = this.mode === "demo" ? DATA.project.epsg
      : (project && project.epsg) || (this.state && this.state.epsg) || layerCrs;
    this.crsText.textContent = projectCrs
      ? `map EPSG:4326 · project ${String(projectCrs).startsWith("EPSG") ? projectCrs : `EPSG:${projectCrs}`}`
      : "map EPSG:4326";
  }

  /** Ask the application what is open, so prerequisites are real rather than assumed. */
  async refreshState() {
    const state = await tryCall("get_state");
    if (state) {
      this.state = {
        project: state.project || state.active_project || null,
        dataset: state.dataset || state.active_dataset || null,
        epsg: state.display_epsg || null,
        layers: state.layers || [],
      };
    }
    const projects = await tryCall("list_projects");
    if (projects) this.projects = Array.isArray(projects.projects) ? projects.projects : [];
    const jobs = await tryCall("list_jobs");
    this.jobs = (jobs && jobs.jobs) || [];
    const telemetry = await tryCall("telemetry");
    this.vehicle = (telemetry && telemetry.telemetry) || null;
    this.applyMode();
    this.renderIndicators();
  }

  /** The top and status bar indicators, from the state just read.
   *
   * API, jobs, Hub and project are drawn by updateChrome(); the aircraft link is drawn
   * here because it is polled on its own, faster clock. */
  renderIndicators() {
    this.updateChrome();
    const set = (dot, label, tone, text) => {
      if (!dot || !label) return;
      dot.className = `status-dot ${tone}`;
      label.textContent = ` ${text}`;
    };
    if (this.mode === "demo") {
      set(this.vehicleDot, this.vehicleLabel, "idle", "no aircraft (demo)");
      return;
    }
    const v = this.vehicle;
    if (v && v.link_lost) {
      set(this.vehicleDot, this.vehicleLabel, "warn", "aircraft link lost");
    } else if (v && v.connected) {
      set(this.vehicleDot, this.vehicleLabel, v.is_simulated ? "warn" : "ok",
          `${v.is_simulated ? "SIMULATED " : ""}${v.driver || "aircraft"} · ${v.flight_mode || "mode ?"}`);
    } else {
      set(this.vehicleDot, this.vehicleLabel, "idle", "no aircraft");
    }
  }

  /** Follow a background job to completion, reporting progress as it goes. */
  watchJob(jobId) {
    this.selectedJobId = jobId;
    const tick = async () => {
      const envelope = await tryCall("job_status", jobId);
      if (!envelope) return;
      // The record is nested: the Api answers ok(job={...}). Reading the state off the
      // envelope gave undefined for every job that ever ran, so the terminal branches
      // below were unreachable and this polled a finished reconstruction forever --
      // products on disk, and a toast still saying it was working.
      const status = envelope.job || envelope;
      const state = status.state || status.status;
      const percent = status.progress ?? status.percent;
      if (state === "done" || state === "finished" || state === "complete") {
        this.toast(`Job ${jobId} finished${jobOutcome(status.result)}.`, "ok");
        await this.refreshState();
        this.dock.refreshLive();
        refreshLive();
        return;
      }
      if (state === "failed" || state === "error") {
        this.toast(`Job ${jobId} failed: ${status.error || "no reason given"}`, "error");
        return;
      }
      if (state === "cancelled") {
        this.toast(`Job ${jobId} cancelled.`);
        return;
      }
      this.toast(`Job ${jobId}: ${status.message || state}${percent != null ? ` ${percent}%` : ""}`);
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 800);
  }

  /**
   * A message where the user is actually looking.
   *
   * Over the canvas rather than in the status bar, because the status bar is where the
   * previous version reported everything and it read as silence.
   */
  toast(text, kind = "info") {
    if (!this.toasts) {
      this.toasts = el("div", { class: "toasts" });
      this.root.appendChild(this.toasts);
    }
    const item = el("div", { class: `toast ${kind}`, text });
    this.toasts.appendChild(item);
    setTimeout(() => item.remove(), kind === "error" ? 9000 : 4500);
    if (this.selectionLabel) this.selectionLabel.textContent = text;
  }

  /* --------------------------------------------------------- command palette */

  openPalette() {
    const items = [
      ...WORKSPACES.map((w) => ({ kind: "workspace", label: w.title, hint: "switch", run: () => this.open(w.id) })),
      { kind: "layout", label: "Reset workspace layout", run: () => this.dock.resetLayout() },
      { kind: "view", label: "Toggle full-screen canvas", hint: "F11", run: () => this.dock.toggleCanvasFullScreen() },
      ...SHORTCUTS.filter(([, , name]) => name && name !== "palette").map(([keys, what, name]) => ({
        kind: "shortcut", label: what, hint: keys, run: () => this.shortcut(name),
      })),
      // The projects on this machine, each of which opens. This listed two demo site
      // names that did nothing when chosen, plus a finding and a mission that existed
      // nowhere -- on every install, connected or not.
      ...(this.mode === "connected" ? (this.projects || []) : []).slice(0, 40).map((p) => ({
        kind: "project", label: String(p.name), hint: "open", run: () => this.openProject(p.id, p.name),
      })),
    ];
    this.palette(items, "");
  }

  palette(items, initial) {
    const input = el("input", { placeholder: "Type a command, workspace, project or finding…", value: initial || "" });
    const list = el("div", { class: "palette-list" });
    const backdrop = el("div", { class: "palette-backdrop" }, [
      el("div", { class: "palette" }, [input, list]),
    ]);

    let active = 0;
    const draw = () => {
      const query = input.value.trim().toLowerCase();
      const matches = items.filter((item) => !query || item.label.toLowerCase().includes(query));
      list.innerHTML = "";
      if (!matches.length) {
        list.appendChild(el("div", { class: "palette-empty", text: "Nothing matches." }));
        return;
      }
      active = Math.min(active, matches.length - 1);
      matches.forEach((item, index) => {
        const row = el("div", { class: "palette-item" + (index === active ? " active" : "") }, [
          item.kind ? el("span", { class: "kind", text: item.kind }) : null,
          el("span", { text: item.label }),
          item.hint ? el("span", { class: "hint", text: item.hint }) : null,
        ]);
        row.onclick = () => { backdrop.remove(); item.run(); };
        list.appendChild(row);
      });
      list._matches = matches;
    };

    input.addEventListener("input", () => { active = 0; draw(); });
    input.addEventListener("keydown", (event) => {
      const matches = list._matches || [];
      if (event.key === "ArrowDown") { active = Math.min(active + 1, matches.length - 1); draw(); event.preventDefault(); }
      if (event.key === "ArrowUp") { active = Math.max(active - 1, 0); draw(); event.preventDefault(); }
      if (event.key === "Enter" && matches[active]) { backdrop.remove(); matches[active].run(); }
      if (event.key === "Escape") backdrop.remove();
    });
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) backdrop.remove(); });

    document.body.appendChild(backdrop);
    draw();
    input.focus();
    input.select();
  }

  /* ------------------------------------------------------------- keyboard */

  bindKeys() {
    // Enter finishes a drawing, Esc cancels it, Backspace removes the last vertex.
    bindDrawingKeys(document);
    // The canvas's own tool icons (draw polygon, draw line, measure) run the toolbar
    // action of the same meaning.
    if (typeof window !== "undefined" && window.addEventListener) {
      window.addEventListener("odk:action", (event) => {
        if (event && event.detail) this.runAction(String(event.detail));
      });
    }
    document.addEventListener("keydown", (event) => {
      const meta = event.ctrlKey || event.metaKey;
      const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
      const key = String(event.key || "").toLowerCase();
      if (meta && key === "k") { event.preventDefault(); this.openPalette(); return; }
      if (event.key === "F11") { event.preventDefault(); this.shortcut("full"); return; }
      if (meta && key === "s") { event.preventDefault(); this.shortcut("save"); return; }
      if (meta && key === "b") { event.preventDefault(); this.shortcut("sides"); return; }
      if (meta && key === "z" && !typing) { event.preventDefault(); this.shortcut("undo"); return; }
      if (!meta && !typing && key === "f") { this.shortcut("fit"); return; }
      if (event.key === "Escape" && !document.querySelector(".modal-backdrop, .palette-backdrop")) {
        this.shortcut("stop");
        return;
      }
      // Digits switch workspace, but not while typing into a field.
      if (!meta && /^[1-9]$/.test(event.key) && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        const workspace = WORKSPACES[Number(event.key) - 1];
        if (workspace) this.open(workspace.id);
      }
    });

    let ticks = 0;
    const tick = async () => {
      // The epoch while the shell is showing demo content: a live wall clock beside
      // synthetic sites is the detail that makes the whole screen read as a real shift
      // in progress, and it is the one number here that is genuinely true -- which is
      // exactly what makes it misleading in this company.
      //
      // Only then, though. This used to show the epoch in every mode, so a connected
      // operator saw a clock stopped at 00:00:00 over their own project: a frozen reading
      // presented as the time, the same fault pointing the other way.
      this.clock.textContent = this.mode === "connected"
        ? `${new Date().toISOString().slice(11, 19)} UTC`
        : `${DEMO.CLOCK} UTC`;
      // The aircraft indicator follows the link, so a lost link shows up in the frame
      // on every screen, not only on the Flight workspace.
      ticks += 1;
      if (this.mode === "connected" && ticks % 3 === 0) {
        const telemetry = await tryCall("telemetry");
        this.vehicle = (telemetry && telemetry.telemetry) || null;
        this.renderIndicators();
      }
    };
    tick();
    setInterval(tick, 1000);
  }
}
