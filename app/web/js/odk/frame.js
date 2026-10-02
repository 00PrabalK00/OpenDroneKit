/* The frame: header, demo banner, side panels, footer (desk) or the Field Daylight
 * header and bottom tab bar (field). DOM only -- the shell (js/odk/shell.js) owns the
 * state and tells the frame what to show.
 *
 * Desk (SysHome / SysPlan ... artboards): 48 px header with mark, wordmark, eight
 * workspace tabs, a 240 px search field with a "/" hint, the project label, the Field
 * mode toggle and the Settings gear; 264 px left panel; 312 px inspector; 28 px footer
 * (cursor, elevation, CRS, scale, vehicle / job, message, network). Both panels collapse.
 *
 * Field (SysFieldPlan / SysFieldFly): 72 px black header with the mark, the project
 * name over the screen line, the network status box and "Desk mode"; the map and the
 * side panel; an 88 px bottom tab bar (Plan, Fly, Process, Analyse, Report).
 *
 * Nothing here holds a value of its own: every figure the frame draws is handed to it by
 * the shell from an Api answer, or is the word for its absence.
 */

import { el, icon, tabs, simulatedTag } from "./components.js";

export const COLLAPSE_KEY = { left: "odk.panel.left", inspector: "odk.panel.inspector" };

const SVG = (paths, { size = 16, stroke = 2 } = {}) => {
  const span = document.createElement("span");
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="${stroke}" stroke-linecap="square" aria-hidden="true">${paths}</svg>`;
  return span.firstChild;
};

/* The artboards' own glyphs, where components.ICONS has none. */
const GLYPH = {
  cog: '<circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"></path>',
  sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>',
  monitor: '<rect x="3" y="4" width="18" height="12"></rect><path d="M8 20h8M12 16v4"></path>',
  panelLeft: '<rect x="3" y="4" width="18" height="16"></rect><path d="M9 4v16"></path>',
  panelRight: '<rect x="3" y="4" width="18" height="16"></rect><path d="M15 4v16"></path>',
  chevronDown: '<path d="M6 9l6 6 6-6"></path>',
  /* Field tab bar (SysFieldPlan). */
  plan: '<path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z"></path><path d="M9 3v15M15 6v15"></path>',
  fly: '<path d="M3 11l18-8-8 18-2-8z"></path>',
  process: '<path d="M12 3l9 5-9 5-9-5z"></path><path d="M3 13l9 5 9-5"></path>',
  analyse: '<path d="M4 20l6-8 4 4 6-10"></path>',
  report: '<path d="M6 3h9l4 4v14H6z"></path><path d="M14 3v5h5M9 13h7M9 17h7"></path>',
};

function storedCollapse(side) {
  try { return localStorage.getItem(COLLAPSE_KEY[side]) === "collapsed"; } catch { return false; }
}

function storeCollapse(side, collapsed) {
  try { localStorage.setItem(COLLAPSE_KEY[side], collapsed ? "collapsed" : "open"); } catch { /* private window */ }
}

export class Frame {
  /**
   * hooks: { onTab(id), onSettings(), onToggleMode(), onSearch(text), onProjectMenu(anchor),
   *          onSessionLog(anchor), desk: [{id,label}], field: [{id,label}] }
   */
  constructor(root, hooks) {
    this.root = root;
    this.hooks = hooks;
    this.collapsed = { left: storedCollapse("left"), inspector: storedCollapse("inspector") };
    this.footer = {};
  }

  /** Build the frame for a mode; returns the regions screens draw into. */
  render(mode) {
    this.mode = mode;
    this.node = mode === "field" ? this.buildField() : this.buildDesk();
    this.root.replaceChildren(this.node);
    this.applyCollapse();
    return this.regions;
  }

  /* ------------------------------------------------------------------ desk */

  buildDesk() {
    const { hooks } = this;
    this.tabNav = tabs(hooks.desk, { label: "Workspaces", onSelect: (id) => hooks.onTab(id) });
    this.tabNav.classList.add("odk-header-tabs");

    this.searchInput = el("input", {
      type: "search", class: "odk-header-search-input", placeholder: "Search projects, actions, places",
      "aria-label": "Search projects, actions and places", "aria-keyshortcuts": "/ Control+K",
      "data-control": "search", autocomplete: "off", spellcheck: "false",
    });
    // The field opens the palette: typing here is typing there.
    this.searchInput.addEventListener("focus", () => {
      const text = this.searchInput.value;
      this.searchInput.value = "";
      this.searchInput.blur();
      hooks.onSearch(text);
    });
    const search = el("label", { class: "odk-header-search" }, [
      icon("search", { size: 14 }), this.searchInput,
      el("kbd", { class: "odk-kbd", text: "/", "aria-hidden": "true" }),
    ]);

    this.projectButton = el("button", {
      type: "button", class: "odk-header-project", "data-control": "project-menu",
      "aria-haspopup": "dialog", title: "Projects on this machine",
    }, [this.projectText = el("span", { class: "odk-header-project-name", text: "" }),
      this.projectNet = el("span", { class: "odk-header-project-net", hidden: "" }),
      SVG(GLYPH.chevronDown, { size: 12 })]);
    this.projectButton.addEventListener("click", () => hooks.onProjectMenu(this.projectButton));

    const modeButton = el("button", {
      type: "button", class: "odk-btn odk-btn--secondary odk-header-mode", "data-control": "mode:field",
      "aria-label": "Switch to Field mode", title: "Field mode: daylight, 64 px targets (Ctrl+Shift+F)",
    }, [SVG(GLYPH.sun, { size: 14 }), "Field mode"]);
    modeButton.addEventListener("click", () => hooks.onToggleMode());

    this.settingsButton = el("button", {
      type: "button", class: "odk-header-settings", "data-control": "open:settings",
      "aria-label": "Settings", title: "Settings",
    }, [SVG(GLYPH.cog, { size: 16 })]);
    this.settingsButton.addEventListener("click", () => hooks.onSettings());

    const header = el("header", { class: "odk-frame-header", "data-region": "header" }, [
      el("span", { class: "odk-brand" }, [
        el("img", { src: "assets/brand/mark-dark.svg", alt: "" }), "OpenDroneKit",
      ]),
      this.tabNav,
      el("div", { class: "odk-header-spacer" }),
      search,
      this.projectButton,
      modeButton,
      this.settingsButton,
    ]);

    this.banner = this.buildBanner();
    const left = el("div", { class: "odk-region", "data-region": "left" });
    const main = el("div", { class: "odk-region odk-frame-canvas", "data-region": "main" });
    const inspector = el("div", { class: "odk-region", "data-region": "inspector" });
    this.handles = {
      left: this.collapseHandle("left", "left panel", GLYPH.panelLeft),
      inspector: this.collapseHandle("inspector", "inspector", GLYPH.panelRight),
    };
    this.overlay = el("div", { class: "odk-main-overlay" });

    const footer = el("footer", { class: "odk-frame-footer", "data-region": "footer" }, [
      this.footer.cursor = el("span", { class: "odk-footer-seg", "data-footer": "cursor" }),
      this.footer.elev = el("span", { class: "odk-footer-seg", "data-footer": "elev", hidden: "" }),
      this.footer.crs = el("span", { class: "odk-footer-seg", "data-footer": "crs" }),
      this.footer.scale = el("span", { class: "odk-footer-seg", "data-footer": "scale" }),
      this.footer.vehicle = el("span", { class: "odk-footer-seg", "data-footer": "vehicle" }),
      this.footer.jobs = el("span", { class: "odk-footer-seg", "data-footer": "jobs" }),
      this.footer.context = el("span", { class: "odk-footer-seg", "data-footer": "context" }),
      this.footer.message = el("button", {
        type: "button", class: "odk-footer-seg odk-footer-message", "data-footer": "message",
        "data-control": "session-log", title: "Session log", "aria-haspopup": "dialog",
      }),
      this.footer.net = el("span", { class: "odk-footer-seg odk-footer-net", "data-footer": "net", role: "status" }),
    ]);
    this.footer.message.addEventListener("click", () => this.hooks.onSessionLog(this.footer.message));

    this.regions = { left, main, inspector };
    return el("div", { class: "odk-frame", "data-frame": "desk" }, [
      header,
      this.banner,
      el("aside", { class: "odk-frame-left", id: "odk-panel-left", "aria-label": "Panel" }, [left]),
      el("main", { class: "odk-frame-main" }, [main, this.overlay, this.handles.left, this.handles.inspector]),
      el("aside", { class: "odk-frame-inspector", id: "odk-panel-inspector", "aria-label": "Inspector" }, [inspector]),
      footer,
    ]);
  }

  collapseHandle(side, name, glyph) {
    const button = el("button", {
      type: "button", class: `odk-collapse odk-collapse--${side}`, "data-control": `collapse:${side}`,
      "aria-controls": side === "left" ? "odk-panel-left" : "odk-panel-inspector",
    }, [SVG(glyph, { size: 14 })]);
    button.addEventListener("click", () => this.setCollapsed(side, !this.collapsed[side]));
    button._name = name;
    return button;
  }

  /* ----------------------------------------------------------------- field */

  buildField() {
    const { hooks } = this;
    this.fieldTitle = el("span", { class: "odk-field-title", text: "" });
    this.fieldSubtitle = el("span", { class: "odk-field-subtitle", text: "" });
    this.fieldSim = el("span", { class: "odk-field-sim" });
    this.fieldNet = el("div", { class: "odk-field-net", role: "status", "data-footer": "net" });
    const desk = el("button", {
      type: "button", class: "odk-field-toggle", "data-control": "mode:desk", "aria-label": "Switch to Desk mode",
    }, [SVG(GLYPH.monitor, { size: 24, stroke: 2.4 }), "Desk mode"]);
    desk.addEventListener("click", () => hooks.onToggleMode());

    const header = el("header", { class: "odk-frame-header odk-field-header", "data-region": "header" }, [
      el("img", { class: "odk-field-mark", src: "assets/brand/mark-dark.svg", alt: "OpenDroneKit" }),
      el("div", { class: "odk-field-heading" }, [this.fieldTitle, this.fieldSubtitle]),
      el("div", { class: "odk-header-spacer" }),
      this.fieldSim,
      this.fieldNet,
      desk,
    ]);

    this.tabNav = el("nav", { class: "odk-field-tabbar", "aria-label": "Workspaces" });
    for (const item of hooks.field) {
      const a = el("a", { href: `#${item.id}`, class: "odk-field-tab", "data-control": `tab:${item.id}` },
        [SVG(GLYPH[item.id] || GLYPH.plan, { size: 24, stroke: 2.4 }), el("span", { text: item.label })]);
      a.addEventListener("click", (event) => { event.preventDefault(); hooks.onTab(item.id); });
      this.tabNav.appendChild(a);
    }

    this.banner = this.buildBanner();
    const left = el("div", { class: "odk-region", "data-region": "left" });
    const main = el("div", { class: "odk-region odk-frame-canvas", "data-region": "main" });
    this.overlay = el("div", { class: "odk-main-overlay" });
    this.handles = null;
    this.footer = { net: this.fieldNet };
    this.regions = { left, main };
    return el("div", { class: "odk-frame", "data-frame": "field" }, [
      header,
      this.banner,
      el("main", { class: "odk-frame-main" }, [main, this.overlay]),
      el("aside", { class: "odk-frame-left", id: "odk-panel-left", "aria-label": "Panel" }, [left]),
      this.tabNav,
    ]);
  }

  buildBanner() {
    this.bannerText = el("span", { class: "odk-banner-text" });
    return el("div", { class: "odk-banner", role: "status", hidden: "" }, [
      el("span", { class: "odk-dot odk-dot--sim", "aria-hidden": "true" }), this.bannerText,
    ]);
  }

  /* ---------------------------------------------------------------- state */

  setActive(id) {
    if (!this.tabNav) return;
    this.tabNav.querySelectorAll("[data-control^='tab:']").forEach((a) => {
      if (a.getAttribute("data-control") === `tab:${id}`) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    if (this.settingsButton) {
      if (id === "settings") this.settingsButton.setAttribute("aria-current", "page");
      else this.settingsButton.removeAttribute("aria-current");
    }
    if (this.node) this.node.setAttribute("data-screen", id);
  }

  /**
   * Project label (desk header, field title). `none`: no project open / not connected.
   * `offline`: the desk label adds "· OFFLINE" -- only while nothing has left the machine.
   */
  setProject(text, { none = false, offline = false } = {}) {
    if (this.projectText) {
      this.projectText.textContent = text;
      this.projectButton.classList.toggle("is-empty", none);
      this.projectNet.hidden = !offline;
      this.projectNet.textContent = offline ? "· OFFLINE" : "";
    }
    if (this.fieldTitle) this.fieldTitle.textContent = text;
  }

  setFieldSubtitle(text) {
    if (this.fieldSubtitle) this.fieldSubtitle.textContent = text || "";
  }

  /** A field header SIMULATED tag, only from a record the Api marked. */
  setFieldSimulated(record) {
    if (!this.fieldSim) return;
    const tag = simulatedTag(record);
    this.fieldSim.replaceChildren(...(tag ? [tag] : []));
    if (tag) tag.textContent = "SIMULATED FLIGHT";
  }

  /**
   * One footer segment. `value` is text, a node, or null (the segment is hidden).
   * `tone` colours it (ok | warn | danger | live); `title` is the tooltip.
   */
  setFooter(name, value, { tone = "", title = "" } = {}) {
    const node = this.footer[name];
    if (!node) return;
    if (value === null || value === undefined || value === "") {
      node.hidden = true;
      node.replaceChildren();
      return;
    }
    node.hidden = false;
    if (typeof value === "string") node.textContent = value;
    else node.replaceChildren(...[].concat(value).filter(Boolean));
    node.dataset.tone = tone;
    if (title) node.title = title; else if (name !== "message") node.removeAttribute("title");
  }

  /**
   * The network status: footer segment (desk) or header box (field). A fixed-length
   * label (`text`, or `fieldText` in Field); the hosts themselves go in `title`.
   */
  setNetwork({ text, fieldText = "", tone, title = "" }) {
    const node = this.footer.net;
    if (!node) return;
    node.dataset.tone = tone;
    const words = this.mode === "field"
      ? fieldText || text.charAt(0).toUpperCase() + text.slice(1)
      : text;
    node.title = title || words;
    node.replaceChildren(el("span", { class: `odk-dot odk-dot--${tone}`, "aria-hidden": "true" }),
      el("span", { class: "odk-net-text", text: words }));
  }

  setBanner(visible, text, title) {
    if (!this.banner) return;
    this.banner.hidden = !visible;
    this.bannerText.textContent = text || "";
    this.banner.title = title || "";
  }

  setCollapsed(side, collapsed, { persist = true } = {}) {
    this.collapsed[side] = Boolean(collapsed);
    if (persist) storeCollapse(side, this.collapsed[side]);
    this.applyCollapse();
    window.dispatchEvent(new Event("resize")); // maps re-measure their container
  }

  applyCollapse() {
    if (!this.node || this.mode !== "desk") return;
    for (const side of ["left", "inspector"]) {
      const collapsed = this.collapsed[side];
      if (collapsed) this.node.setAttribute(`data-${side}`, "collapsed");
      else this.node.removeAttribute(`data-${side}`);
      const handle = this.handles && this.handles[side];
      if (handle) {
        handle.setAttribute("aria-expanded", collapsed ? "false" : "true");
        const label = `${collapsed ? "Show" : "Hide"} ${handle._name} (Ctrl+B toggles both)`;
        handle.setAttribute("aria-label", label);
        handle.title = label;
      }
      const panel = this.node.querySelector(side === "left" ? ".odk-frame-left" : ".odk-frame-inspector");
      if (panel) panel.inert = collapsed;
    }
  }

  /** Full-screen canvas: both panels away, without forgetting their saved state. */
  setFullCanvas(on) {
    if (!this.node) return;
    if (on) this.node.setAttribute("data-full", "");
    else this.node.removeAttribute("data-full");
    window.dispatchEvent(new Event("resize"));
  }
}
