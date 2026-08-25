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
import { DEMO, demoEnabled } from "./demo.js";
import { connected, tryCall, whenReady } from "./api.js";

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
    this.applyMode();
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

    // Every selection is echoed in the status bar. It is the cheapest possible proof
    // that the panels are wired to each other rather than merely sitting side by side.
    selection.on("*", (kind, value) => {
      this.selectionLabel.textContent = `${kind}: ${value?.label || value?.id || value?.name || "—"}`;
    });
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

  runAction(action) {
    // No action here mutates a survey. The shell is the framework; wiring an action to
    // the API is a per-action decision, and a button that silently did nothing real
    // would be worse than one that says so.
    this.selectionLabel.textContent = `action: ${action} (not wired to the API yet)`;
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
