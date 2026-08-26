/* The application shell: navigation, contextual toolbar, status bar, command palette.
 *
 * Four persistent layers, and the third is the one that matters. Nav, toolbar and status
 * frame a dockable workspace that changes completely between tasks while the project
 * underneath does not. An operator switching from planning to flight is not navigating
 * to another page -- they are turning to a different instrument on the same aircraft.
 */

import { Dock } from "./dock.js";
import { el, selection } from "./primitives.js";
import { WORKSPACES, WORKSPACE_BY_ID } from "./workspaces.js";
import { DATA, DEMO, demoEnabled } from "./demo.js";
import { call, connected, lastError, tryCall, whenReady } from "./api.js";
import { ACTIONS, UNWIRED, prerequisite, resolve } from "./actions.js";
import { setImage, setView } from "./viewstate.js";
import { askOne, choose as chooseModal, confirmAsk, reveal } from "./modal.js";

const LAST_WORKSPACE = "odk.workspace.last";

/* Safety-critical actions must never be one careless click from Save. They are styled
 * apart and listed here so the styling cannot drift from the meaning. */
const DANGER = new Set(["Abort", "Land", "RTL", "Manual Override", "Cancel"]);
const PRIMARY = new Set(["Plan", "Process", "Run AI", "Generate", "Start", "Generate Report"]);

const SHORTCUTS = [
  ["Ctrl/⌘ K", "Command palette"],
  ["Ctrl/⌘ S", "Save"],
  ["Ctrl/⌘ B", "Toggle side panels"],
  ["F", "Fit view"],
  ["F11", "Full-screen canvas"],
  ["1 – 9", "Switch workspace"],
  ["Ctrl/⌘ Z", "Undo"],
];

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
    this.connect();
  }

  /**
   * Ask the bridge what is really there, and stop showing demo content the moment it
   * answers. Demo mode stays on if it was explicitly requested -- someone demonstrating
   * the product on a machine with real projects still wants the demo.
   */
  async connect() {
    const ready = await whenReady();
    if (!ready || !connected()) return;
    if (this.mode === "demo") return;
    this.mode = "connected";
    const projects = await tryCall("list_projects");
    this.projects = Array.isArray(projects && projects.projects) ? projects.projects : [];
    await this.refreshState();
  }

  /** The frame reflects the state: the banner only exists while content is synthetic. */
  applyMode() {
    if (!this.banner) return;
    // Synthetic whenever the panels are NOT showing what the application returned.
    // Keying this on demo mode alone was wrong and briefly made things worse: with no
    // bridge the panels still render the structural sample, so hiding the banner left
    // synthetic content on screen with nothing saying so. Connected is the only state
    // in which what you are looking at was measured.
    const synthetic = this.mode !== "connected";
    this.banner.classList.toggle("hidden", !synthetic);
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
        this.selectedFindingId = value.id || value.label || null;
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

  /** Draw one image from the active dataset on the canvas. */
  async showImage(name) {
    if (!connected()) {
      this.toast(`${name}: connect a project to preview the image.`, "warn");
      return;
    }
    this.toast(`Loading ${name}…`);
    const preview = await tryCall("image_preview", name);
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
    const toWorkspace = {
      "mission.plan": "planning",
      "mission.settings": "planning",
      "mission.history": "planning",
      "fly.connect": "flight",
      "fly.command": "flight",
      "fly.disconnect": "flight",
      "recon.settings": "processing",
      "analysis.pipeline": "inspection",
      "analysis.models": "inspection",
      "tools.measure": "measurements",
      "tools.audit": "developers",
      "view.panel": null,
    };

    const toAction = {
      "project.new": "New Project",
      "project.open": "Open",
      "data.import_imagery": "Import",
      "data.import_raster": "New Folder",
      "data.import_vector": "New Folder",
      "mission.plan": "Plan",
      "mission.save": "Save",
      "mission.export": "Export",
      "mission.export_all": "Export",
      "recon.run": "Process",
      "analysis.pipeline": "Start",
      "analysis.models": "Run AI",
      "fly.upload": "Upload",
      "tools.capabilities": "Preflight",
      "mission.history": "Simulate",
    };

    window.odk = window.odk || {};
    window.odk.onMenu = async ({ action, payload } = {}) => {
      if (!action) return;

      // Jump to where the action belongs first, so the user can see what it did.
      const workspace = toWorkspace[action];
      if (workspace && WORKSPACE_BY_ID[workspace]) this.open(workspace);

      const mapped = toAction[action];
      if (mapped) {
        await this.runAction(mapped);
        return;
      }

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
          await tryCall("set_aoi", null);
          this.toast("Area of interest cleared.", "ok");
          return;
        case "mission.clear_nofly":
          await tryCall("set_no_fly_zones", []);
          this.toast("No-fly zones cleared.", "ok");
          return;
        case "data.import_terrain": {
          const file = await call("pick_file", ["tif", "tiff", "asc", "csv"]);
          if (!file || !file.path) return;
          await tryCall("set_terrain_source", file.path);
          this.toast(`Terrain source: ${file.path}`, "ok");
          return;
        }
        case "fly.disconnect":
          await tryCall("disconnect_vehicle");
          this.toast("Vehicle disconnected.", "ok");
          return;
        case "view.zoom_aoi":
        case "view.zoom_mission":
          setView("map");
          setImage(null);
          this.dock.render(WORKSPACE_BY_ID[this.workspaceId]);
          this.toast("Back to the map.", "ok");
          return;
        case "view.basemap":
          this.toast("Basemap is chosen on the canvas toolbar.", "warn");
          return;
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
    };
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

    return el("header", { class: "topnav" }, [
      el("div", { class: "brand" }, [el("span", { class: "mark" }), el("span", { text: "OpenDroneKit" })]),
      items,
      el("div", { class: "nav-right" }, [
        search,
        el("span", { class: "sep" }),
        el("span", {}, [el("span", { class: "status-dot ok" }), el("span", { text: " API" })]),
        el("span", {}, [el("span", { class: "status-dot ok" }), el("span", { text: " 3 workers" })]),
        el("span", {}, [el("span", { class: "status-dot idle" }), el("span", { text: " offline mode" })]),
        el("span", { class: "sep" }),
        this.clock = el("span", { class: "mono", text: "--:--:-- UTC" }),
        el("span", { class: "sep" }),
        el("span", { class: "avatar", text: "PK" }),
        el("span", { text: DEMO.ORG }),
      ]),
    ]);
  }

  buildStatus() {
    this.selectionLabel = el("span", { text: "no selection" });
    return el("footer", { class: "statusbar" }, [
      el("span", {}, [el("span", { class: "status-dot ok" }), el("span", { text: ` ${DEMO.SITES[0]}` })]),
      el("span", { class: "sep" }),
      el("span", { class: "mono", text: "EPSG:4326" }),
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
      this.activeTool = entry.tool;
      this.workspaceEl.dataset.tool = entry.tool;
      this.toast(`Tool armed: ${action}. Draw on the canvas.`, "ok");
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
      this.toast(message, "ok");
      if (result && result.job) this.watchJob(result.job);
      if (result && result.refresh) await this.refreshState();
    } catch (error) {
      // Shown, not swallowed. A refusal from the Api is the most useful thing this
      // application produces and it used to vanish into a rejected promise.
      this.toast(`${action}: ${error.message || error}`, "error");
    }
  }

  async runVehicle(command, action) {
    await call("vehicle_command", command);
    return { message: `${action} sent.` };
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
      missionOptions: () => {
        // What the mission panels were actually edited to. Numeric where the planner
        // expects a number: a string altitude reaches Python and fails there, which
        // reads as a planner bug rather than a form that never converted its input.
        const raw = this.settings || {};
        const options = {};
        for (const [key, value] of Object.entries(raw)) {
          const asNumber = Number(value);
          options[key] = value !== "" && Number.isFinite(asNumber) ? asNumber : value;
        }
        return options;
      },
      reconstructionOptions: () => ({ engine: "auto", profile: "standard" }),
      selectedJob: () => this.selectedJobId || null,
      selectedFinding: () => this.selectedFindingId || null,
      selectedFleetId: () => this.selectedFleetId || null,
      selectionGeometry: () => this.selectionGeometry || null,
      resetLayout: () => this.dock.resetLayout(),
      // The organisation and project the session is actually on. Defaulting to 1 rather
      // than refusing keeps a single-tenant desktop install usable; a multi-tenant
      // deployment sets them from the opened project.
      organizationId: () => (this.state && this.state.organization_id) || 1,
      projectId: () => (this.state && this.state.project && this.state.project.id) || 1,
      // A secret the application will never show again. A toast disappears after four
      // seconds, which is not long enough to copy a token, so this blocks until the
      // user dismisses it.
      reveal: (text) => reveal("Copy this now — it is not stored", text),
    };
  }

  stateSummary() {
    return {
      projectOpen: Boolean(this.state && this.state.project),
      datasetSelected: Boolean(this.state && this.state.dataset),
    };
  }

  /** Ask the application what is open, so prerequisites are real rather than assumed. */
  async refreshState() {
    const state = await tryCall("get_state");
    if (state) {
      this.state = {
        project: state.project || state.active_project || null,
        dataset: state.dataset || state.active_dataset || null,
      };
    }
    const jobs = await tryCall("list_jobs");
    this.jobs = (jobs && jobs.jobs) || [];
    this.applyMode();
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
        this.toast(`Job ${jobId} finished.`, "ok");
        await this.refreshState();
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
      ...SHORTCUTS.map(([keys, what]) => ({ kind: "shortcut", label: what, hint: keys, run: () => {} })),
      { kind: "project", label: DEMO.SITES[0], hint: "project", run: () => {} },
      { kind: "project", label: DEMO.SITES[1], hint: "project", run: () => {} },
      { kind: "finding", label: "F-118 crack — Roof Block A", hint: "finding", run: () => this.open("inspection") },
      { kind: "mission", label: "Roof Block A v3", hint: "mission", run: () => this.open("planning") },
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
    document.addEventListener("keydown", (event) => {
      const meta = event.ctrlKey || event.metaKey;
      if (meta && event.key.toLowerCase() === "k") { event.preventDefault(); this.openPalette(); return; }
      if (event.key === "F11") { event.preventDefault(); this.dock.toggleCanvasFullScreen(); return; }
      // Digits switch workspace, but not while typing into a field.
      if (!meta && /^[1-9]$/.test(event.key) && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        const workspace = WORKSPACES[Number(event.key) - 1];
        if (workspace) this.open(workspace.id);
      }
    });

    const tick = () => {
      // The epoch while the shell is showing demo content. A live wall clock beside
      // synthetic sites is the detail that makes the whole screen read as a real shift
      // in progress, and it is the one number here that is genuinely true -- which is
      // exactly what makes it misleading in this company.
      this.clock.textContent = `${DEMO.CLOCK} UTC`;
    };
    tick();
    setInterval(tick, 1000);
  }
}
