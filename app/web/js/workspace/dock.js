/* The dock: regions, panels, splitters, tab stacks, and layout persistence.
 *
 * Everything visible in OpenDroneKit is composed from this. A workspace is not a page --
 * it is an arrangement of panels around a canvas, and switching workspace changes the
 * arrangement without touching the project underneath. That distinction is the whole
 * design: an operator moving from planning to flight to verification is looking at one
 * survey through different instruments, not visiting three websites.
 *
 * Layout is persisted per workspace, so a user who widens Telemetry during a flight finds
 * it wide next time. Persistence is keyed by workspace id and panel id rather than by
 * position, so adding a panel in a later release does not scramble a saved layout -- it
 * appears at its default size and everything else stays where the user put it.
 */

const STORE_KEY = "odk.workspace.layout.v1";

export class Dock {
  constructor(root, { onLayoutChange } = {}) {
    this.root = root;
    this.onLayoutChange = onLayoutChange || (() => {});
    this.panels = new Map();       // id -> {def, el, region}
    this.regions = {};
    this.workspaceId = null;
    this.sizes = {};               // persisted region sizes and panel state
    this._drag = null;
  }

  /* ------------------------------------------------------------- persistence */

  _load() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); }
    catch { return {}; }           // corrupt storage must not brick the app
  }

  _save() {
    const all = this._load();
    all[this.workspaceId] = this.sizes;
    try { localStorage.setItem(STORE_KEY, JSON.stringify(all)); } catch { /* private mode */ }
    this.onLayoutChange(this.workspaceId, this.sizes);
  }

  resetLayout() {
    const all = this._load();
    delete all[this.workspaceId];
    try { localStorage.setItem(STORE_KEY, JSON.stringify(all)); } catch { /* ignore */ }
    this.sizes = {};
    this.render(this.workspace);
  }

  /* ----------------------------------------------------------------- render */

  render(workspace) {
    this.workspace = workspace;
    this.workspaceId = workspace.id;
    this.sizes = this._load()[workspace.id] || {};
    this.panels.clear();
    this.root.innerHTML = "";

    const left = this._region("left", workspace.left || []);
    const centre = this._centre(workspace);
    const right = this._region("right", workspace.right || []);

    if (left) {
      this.root.appendChild(left);
      this.root.appendChild(this._splitter("vertical", "left"));
    }
    this.root.appendChild(centre);
    if (right) {
      this.root.appendChild(this._splitter("vertical", "right"));
      this.root.appendChild(right);
    }
    this.regions = { left, centre, right };
  }

  _regionWidth(side, fallback) {
    const stored = this.sizes[`region.${side}`];
    return Math.max(140, Math.min(620, stored || fallback));
  }

  _region(side, panelDefs) {
    if (!panelDefs.length) return null;
    const el = document.createElement("div");
    el.className = `region ${side}`;
    el.dataset.region = side;
    el.style.width = `${this._regionWidth(side, side === "left" ? 240 : 300)}px`;
    for (const def of panelDefs) el.appendChild(this._panel(def, side));
    this._enableDrop(el, side);
    return el;
  }

  _centre(workspace) {
    const el = document.createElement("div");
    el.className = "region centre";
    el.dataset.region = "centre";

    const stack = document.createElement("div");
    stack.className = "centre-stack";
    stack.appendChild(workspace.canvas ? workspace.canvas() : this._emptyCanvas());
    el.appendChild(stack);

    const bottomDefs = workspace.bottom || [];
    if (bottomDefs.length) {
      el.appendChild(this._splitter("horizontal", "bottom"));
      const bottom = document.createElement("div");
      bottom.className = "region bottom";
      bottom.dataset.region = "bottom";
      bottom.style.height = `${Math.max(80, Math.min(520, this.sizes["region.bottom"] || 190))}px`;
      // Bottom panels sit side by side: a timeline next to a log next to a queue is how
      // the region is actually read, and stacking them would waste the width.
      bottom.style.flexDirection = "row";
      for (const def of bottomDefs) {
        const panel = this._panel(def, "bottom");
        panel.style.flex = `${def.flex || 1} 1 0`;
        bottom.appendChild(panel);
      }
      this._enableDrop(bottom, "bottom");
      el.appendChild(bottom);
      this.regions.bottom = bottom;
    }
    return el;
  }

  _emptyCanvas() {
    const el = document.createElement("div");
    el.className = "canvas";
    el.innerHTML = `<div class="placeholder"><div>
      <div class="big">No canvas for this workspace</div>
      <div class="small">Workspaces without a primary view are still composed from panels.</div>
    </div></div>`;
    return el;
  }

  /* ------------------------------------------------------------------ panels */

  _panel(def, region) {
    const el = document.createElement("section");
    el.className = "panel" + (def.grow === false ? "" : " grow");
    el.dataset.panelId = def.id;
    if (def.height) el.style.flex = `0 0 ${def.height}px`;

    const state = this.sizes[`panel.${def.id}`] || {};
    if (state.collapsed) el.classList.add("collapsed");

    const header = document.createElement("header");
    header.className = "panel-header";
    header.draggable = true;
    header.innerHTML = `
      <span class="title">${def.title}</span>
      ${def.count != null ? `<span class="count">${def.count}</span>` : ""}
      <span class="spacer"></span>
      <button class="pbtn" data-act="collapse" title="Collapse / expand">${state.collapsed ? "▸" : "▾"}</button>
      <button class="pbtn" data-act="pop" title="Open in a new window (multi-monitor)">⧉</button>
      <button class="pbtn" data-act="close" title="Hide panel">✕</button>`;
    el.appendChild(header);

    const body = document.createElement("div");
    body.className = "panel-body" + (def.pad ? " pad" : "");

    // Tabs stack several views into one panel, which is how a small window keeps
    // everything reachable instead of hiding half of it.
    if (def.tabs && def.tabs.length) {
      const tabs = document.createElement("div");
      tabs.className = "panel-tabs";
      const active = state.tab != null ? state.tab : 0;
      def.tabs.forEach((tab, index) => {
        const button = document.createElement("button");
        button.className = "panel-tab" + (index === active ? " active" : "");
        button.textContent = tab.title;
        button.onclick = () => {
          this.sizes[`panel.${def.id}`] = { ...state, tab: index };
          this._save();
          tabs.querySelectorAll(".panel-tab").forEach((t, i) =>
            t.classList.toggle("active", i === index));
          body.innerHTML = "";
          body.appendChild(tab.render());
        };
        tabs.appendChild(button);
      });
      el.appendChild(tabs);
      body.appendChild(def.tabs[active].render());
    } else if (def.render) {
      body.appendChild(def.render());
    }
    el.appendChild(body);

    header.addEventListener("click", (event) => {
      const act = event.target.dataset?.act;
      if (!act) return;
      if (act === "collapse") this._toggleCollapse(def.id, el, event.target);
      if (act === "close") { el.classList.add("hidden"); this._setState(def.id, { hidden: true }); }
      if (act === "pop") this._popOut(def);
    });

    header.addEventListener("dragstart", (event) => {
      this._drag = { panelId: def.id, from: region };
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", def.id);
    });
    header.addEventListener("dragend", () => {
      this._drag = null;
      this.root.querySelectorAll(".region").forEach((r) => r.classList.remove("drop-active"));
    });

    if (state.hidden) el.classList.add("hidden");
    this.panels.set(def.id, { def, el, region });
    return el;
  }

  _setState(id, patch) {
    this.sizes[`panel.${id}`] = { ...(this.sizes[`panel.${id}`] || {}), ...patch };
    this._save();
  }

  _toggleCollapse(id, el, button) {
    const collapsed = el.classList.toggle("collapsed");
    if (button) button.textContent = collapsed ? "▸" : "▾";
    this._setState(id, { collapsed });
  }

  /** Multi-monitor: a panel becomes its own window and keeps rendering. */
  _popOut(def) {
    const win = window.open("", `odk-${def.id}`, "width=520,height=680");
    if (!win) return;                      // popup blocked; the panel stays docked
    win.document.title = `${def.title} — OpenDroneKit`;
    const link = win.document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("../../css/workspace.css", import.meta.url).href;
    win.document.head.appendChild(link);
    win.document.body.style.background = "var(--bg-panel)";
    const holder = win.document.createElement("div");
    holder.className = "panel grow";
    holder.style.height = "100vh";
    holder.style.margin = "0";
    holder.innerHTML = `<header class="panel-header"><span class="title">${def.title}</span></header>`;
    const body = win.document.createElement("div");
    body.className = "panel-body" + (def.pad ? " pad" : "");
    const view = def.tabs ? def.tabs[0].render() : (def.render ? def.render() : document.createElement("div"));
    body.appendChild(view);
    holder.appendChild(body);
    win.document.body.appendChild(holder);
  }

  showPanel(id) {
    const entry = this.panels.get(id);
    if (!entry) return;
    entry.el.classList.remove("hidden");
    this._setState(id, { hidden: false });
  }

  hiddenPanels() {
    return [...this.panels.values()]
      .filter((p) => p.el.classList.contains("hidden"))
      .map((p) => p.def);
  }

  /* ----------------------------------------------------- moving panels about */

  _enableDrop(regionEl, side) {
    regionEl.addEventListener("dragover", (event) => {
      if (!this._drag) return;
      event.preventDefault();
      regionEl.classList.add("drop-active");
    });
    regionEl.addEventListener("dragleave", () => regionEl.classList.remove("drop-active"));
    regionEl.addEventListener("drop", (event) => {
      if (!this._drag) return;
      event.preventDefault();
      regionEl.classList.remove("drop-active");
      const entry = this.panels.get(this._drag.panelId);
      if (!entry || entry.region === side) return;
      // Bottom lays out horizontally; left and right vertically. A panel moved between
      // them has to take its new region's flow, or it lands stretched wrong.
      entry.el.style.flex = side === "bottom" ? "1 1 0" : "";
      regionEl.appendChild(entry.el);
      entry.region = side;
      this._setState(this._drag.panelId, { region: side });
    });
  }

  /* -------------------------------------------------------------- splitters */

  _splitter(orientation, key) {
    const el = document.createElement("div");
    el.className = `splitter ${orientation}`;
    el.addEventListener("mousedown", (event) => {
      event.preventDefault();
      el.classList.add("dragging");
      const start = orientation === "vertical" ? event.clientX : event.clientY;
      const target = key === "bottom" ? this.regions.bottom : this.regions[key];
      if (!target) return;
      const initial = orientation === "vertical" ? target.offsetWidth : target.offsetHeight;

      const move = (moveEvent) => {
        const now = orientation === "vertical" ? moveEvent.clientX : moveEvent.clientY;
        let delta = now - start;
        // Right region and bottom region grow in the opposite direction to the drag.
        if (key === "right" || key === "bottom") delta = -delta;
        const next = Math.max(120, initial + delta);
        if (orientation === "vertical") target.style.width = `${next}px`;
        else target.style.height = `${next}px`;
      };
      const up = () => {
        el.classList.remove("dragging");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        this.sizes[`region.${key}`] =
          orientation === "vertical" ? target.offsetWidth : target.offsetHeight;
        this._save();
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
    return el;
  }

  /* ------------------------------------------------------------ full screen */

  toggleCanvasFullScreen() {
    const hide = !this.root.classList.contains("canvas-only");
    this.root.classList.toggle("canvas-only", hide);
    for (const side of ["left", "right"]) {
      const region = this.regions[side];
      if (region) region.classList.toggle("collapsed", hide);
    }
    if (this.regions.bottom) this.regions.bottom.classList.toggle("collapsed", hide);
    this.root.querySelectorAll(".splitter").forEach((s) => s.classList.toggle("hidden", hide));
    return hide;
  }
}
