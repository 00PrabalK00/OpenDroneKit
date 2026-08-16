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
    this.build();
    this.bindKeys();
  }

  build() {
    this.root.innerHTML = "";
    this.root.className = "shell";

    this.nav = this.buildNav();
    this.toolbar = el("div", { class: "toolbar" });
    this.workspaceEl = el("div", { class: "workspace" });
    this.status = this.buildStatus();

    this.root.append(this.nav, this.toolbar, this.workspaceEl, this.status);

    this.dock = new Dock(this.workspaceEl);
    this.open(this.workspaceId);

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
        el("span", { text: "Northern Infrastructure" }),
      ]),
    ]);
  }

  buildStatus() {
    this.selectionLabel = el("span", { text: "no selection" });
    return el("footer", { class: "statusbar" }, [
      el("span", {}, [el("span", { class: "status-dot ok" }), el("span", { text: " Bhopal Warehouse" })]),
      el("span", { class: "sep" }),
      el("span", { class: "mono", text: "EPSG:32643" }),
      el("span", { class: "sep" }),
      this.selectionLabel,
      el("span", { class: "spacer" }),
      // Said plainly, permanently, in the frame of the application: this shell shows
      // structure, not survey results. A number here has not been measured.
      el("span", { class: "chip warn", text: "sample data — not connected to a project" }),
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
      { kind: "project", label: "Bhopal Warehouse", hint: "project", run: () => {} },
      { kind: "project", label: "NH-46 Corridor", hint: "project", run: () => {} },
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
      this.clock.textContent = `${new Date().toISOString().slice(11, 19)} UTC`;
    };
    tick();
    setInterval(tick, 1000);
  }
}
