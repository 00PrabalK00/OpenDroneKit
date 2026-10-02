/* Component factories for the v1 interface (desk "Survey Instrument" + Field Daylight).
 *
 * Each factory returns a DOM node whose class names are styled by css/odk/components.css
 * (desk geometry) and css/odk/field.css (field overrides). Nothing here sets a colour or
 * a size inline: the mode is chosen by [data-mode] on an ancestor, so one node renders
 * correctly in either mode.
 *
 * Two rules every screen relies on (docs/ui/ARCHITECTURE.md):
 *
 *   1. data-action. A button that runs a verb carries data-action="<ACTIONS key>", so
 *      tests and the command palette find it by key, never by its label. A button that
 *      only changes the UI (dismiss, open a popover, switch a tab) must say so with
 *      `control: "<name>"`, which renders data-control. button() throws when given
 *      neither, so the rule cannot be forgotten quietly.
 *
 *   2. Tags come from Api fields. simulatedTag()/demoTag() take the record and render
 *      only when the Api marked it (`is_simulated`, `demo`/`is_demo`). A screen never
 *      decides on its own that a value is simulated, and never omits a tag the Api set.
 *
 * The DOM helper `el` and the table builder are reused from js/workspace/primitives.js,
 * which stays the single implementation of both until the final unit moves it to
 * js/core/.
 */

import { el, table as primitiveTable, selection } from "../workspace/primitives.js";

export { el, selection };

/* ---------------------------------------------------------------- icons */

const svg = (paths, { size = 16, stroke = 2, viewBox = "0 0 24 24" } = {}) =>
  `<svg width="${size}" height="${size}" viewBox="${viewBox}" fill="none" stroke="currentColor" `
  + `stroke-width="${stroke}" aria-hidden="true">${paths}</svg>`;

/** Inline stroke icons (no icon font, nothing fetched). */
export const ICONS = {
  measure: '<path d="M3 17L17 3l4 4L7 21z"></path><path d="M7 13l2 2M10 10l2 2M13 7l2 2"></path>',
  plus: '<path d="M12 5v14M5 12h14"></path>',
  minus: '<path d="M5 12h14"></path>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"></path>',
  close: '<path d="M6 6l12 12M18 6L6 18"></path>',
  search: '<circle cx="11" cy="11" r="7"></circle><path d="M20 20l-3.5-3.5"></path>',
  chevron: '<path d="M6 9l6 6 6-6"></path>',
  sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>',
  monitor: '<rect x="3" y="4" width="18" height="12"></rect><path d="M8 20h8M12 16v4"></path>',
  centre: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v4M12 18v4M2 12h4M18 12h4"></path>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"></path><path d="M13 7l4 4"></path>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"></path><path d="M3 13l9 5 9-5"></path>',
  gear: '<circle cx="12" cy="12" r="3"></circle><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"></path>',
  folder: '<path d="M3 6h6l2 2h10v11H3z"></path>',
};

/** An icon as a DOM node. */
export function icon(name, opts = {}) {
  const path = ICONS[name];
  if (!path) throw new Error(`unknown icon ${name}`);
  const span = document.createElement("span");
  span.className = "odk-icon";
  span.style.display = "inline-flex";
  span.innerHTML = svg(path, opts);
  return span.firstChild;
}

/* -------------------------------------------------------------- buttons */

function marks(node, { action, control }) {
  if (action) node.setAttribute("data-action", action);
  else if (control) node.setAttribute("data-control", control);
  else {
    throw new Error(
      "A button needs data-action (an ACTIONS key) or an explicit `control` name: "
      + "see docs/ui/ARCHITECTURE.md, 'The data-action rule'.");
  }
  return node;
}

/**
 * A text button.
 *   variant: primary | secondary | ghost | danger | commit
 *   action:  ACTIONS key (renders data-action) -- or control: "<ui-only name>"
 */
export function button({ label, action, control, variant = "secondary", iconName, disabled = false,
  title, onClick, type = "button" } = {}) {
  const node = el("button", {
    type, class: `odk-btn odk-btn--${variant}`, ...(title ? { title } : {}),
  });
  if (iconName) node.appendChild(icon(iconName));
  node.appendChild(document.createTextNode(label));
  if (disabled) node.disabled = true;
  if (onClick) node.addEventListener("click", onClick);
  return marks(node, { action, control });
}

/** A square icon button. `label` is required: it is the aria-label. */
export function iconButton({ iconName, text, label, action, control, pressed, disabled = false, onClick } = {}) {
  if (!label) throw new Error("icon buttons need an aria-label");
  const node = el("button", { type: "button", class: "odk-icon-btn", "aria-label": label, title: label });
  if (iconName) node.appendChild(icon(iconName));
  else node.textContent = text || "";
  if (pressed !== undefined) node.setAttribute("aria-pressed", pressed ? "true" : "false");
  if (disabled) node.disabled = true;
  if (onClick) node.addEventListener("click", onClick);
  return marks(node, { action, control });
}

/** Icon buttons joined with 1px gaps (desk) or 8px gaps (field). */
export function iconGroup(buttons) {
  return el("div", { class: "odk-icon-group" }, buttons);
}

/* --------------------------------------------------------------- inputs */

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

/**
 * A labelled input. `mono` + `numeric` give the right-aligned data field from the
 * artboards; `unit` renders after the value. `readonly` gives the surface-2 fill.
 */
export function input({ label, value = "", unit, placeholder, readonly = false, mono = false,
  numeric = false, type = "text", name, onChange, iconName } = {}) {
  const id = nextId("odk-in");
  const field = el("input", {
    id, type, ...(name ? { name } : {}), ...(placeholder ? { placeholder } : {}),
  });
  field.value = value;
  if (readonly) field.readOnly = true;
  if (onChange) field.addEventListener("change", () => onChange(field.value));
  const box = el("div", {
    class: ["odk-input", readonly ? "odk-input--readonly" : "", mono ? "odk-input--mono" : "",
      numeric ? "odk-input--num" : ""].filter(Boolean).join(" "),
  }, [iconName ? icon(iconName, { size: 14 }) : null, field,
    unit ? el("span", { class: "odk-input-unit", text: unit }) : null]);
  return el("div", { class: "odk-field" }, [label ? el("label", { for: id, text: label }) : null, box]);
}

/** A select styled as an input, with the chevron. options: [{value, label}] or strings. */
export function select({ label, options = [], value, onChange, mono = true } = {}) {
  const id = nextId("odk-sel");
  const node = el("select", { id }, options.map((o) => {
    const opt = typeof o === "string" ? { value: o, label: o } : o;
    return el("option", { value: opt.value, text: opt.label, ...(opt.value === value ? { selected: "" } : {}) });
  }));
  if (onChange) node.addEventListener("change", () => onChange(node.value));
  const box = el("div", { class: `odk-input odk-input--readonly${mono ? " odk-input--mono" : ""}` },
    [node, icon("chevron", { size: 12 })]);
  return el("div", { class: "odk-field" }, [label ? el("label", { for: id, text: label }) : null, box]);
}

/** The search field (header and panels). */
export function search({ label, placeholder = "Search", onInput } = {}) {
  const wrap = input({ label, placeholder, iconName: "search" });
  const field = wrap.querySelector("input");
  field.type = "search";
  if (!label) field.setAttribute("aria-label", placeholder);
  if (onInput) field.addEventListener("input", () => onInput(field.value));
  return wrap;
}

/** A 36px (desk) / 64px (field) checkbox row. */
export function checkRow({ label, checked = false, disabled = false, onChange } = {}) {
  const box = el("input", { type: "checkbox" });
  box.checked = checked;
  box.disabled = disabled;
  if (onChange) box.addEventListener("change", () => onChange(box.checked));
  return el("label", { class: `odk-check-row${disabled ? " odk-check-row--disabled" : ""}` },
    [box, el("span", { text: label })]);
}

/* -------------------------------------------------- chips, tags, status */

const TONES = new Set(["", "ok", "warn", "danger"]);

/** A mono chip. tone: "" | ok | warn | danger (toned chips get a status dot). */
export function chip(text, tone = "") {
  if (!TONES.has(tone)) throw new Error(`unknown chip tone ${tone}`);
  const node = el("span", { class: `odk-chip${tone ? ` odk-chip--${tone}` : ""}` });
  if (tone) node.appendChild(el("span", { class: `odk-dot odk-dot--${tone}`, "aria-hidden": "true" }));
  node.appendChild(document.createTextNode(text));
  return node;
}

/** The purple tag. Use simulatedTag()/demoTag() from screens; this is the raw node. */
export function tag(text, { small = false } = {}) {
  return el("span", { class: `odk-tag${small ? " odk-tag--small" : ""}`, "data-tag": text, text });
}

/** "SIMULATED", only when the Api marked the record simulated. Otherwise null. */
export function simulatedTag(record, opts) {
  return record && record.is_simulated === true ? tag("SIMULATED", opts) : null;
}

/** "DEMO RECORD", only when the Api marked the record a demo record. Otherwise null. */
export function demoTag(record, opts) {
  return record && (record.demo === true || record.is_demo === true) ? tag("DEMO RECORD", opts) : null;
}

/** A status dot. tone: ok | warn | danger | sim | running | idle. */
export function dot(tone) {
  return el("span", { class: `odk-dot odk-dot--${tone}`, "aria-hidden": "true" });
}

/** A dot plus a short status text. */
export function status(tone, text) {
  return el("span", { class: "odk-status", "data-status": tone }, [dot(tone), el("span", { text })]);
}

/* ---------------------------------------------------------------- toast */

/**
 * A toast: full outline and a status dot, never a left-border card.
 * tone ok|warn|danger|sim|running. danger uses role=alert.
 */
export function toast({ title, detail, detailMono = false, tone = "ok", link, onDismiss } = {}) {
  const close = el("button", {
    type: "button", class: "odk-toast-close", "aria-label": "Dismiss", "data-control": "dismiss-toast",
    html: svg(ICONS.close, { size: 14 }),
  });
  const node = el("div", {
    class: `odk-toast odk-toast--${tone}`, role: tone === "danger" ? "alert" : "status",
  }, [
    dot(tone),
    el("div", { class: "odk-toast-body" }, [
      el("span", { class: "odk-toast-title", text: title }),
      detail ? el("span", { class: `odk-toast-detail${detailMono ? " odk-toast-detail--mono" : ""}`, text: detail }) : null,
      link || null,
    ]),
    close,
  ]);
  close.addEventListener("click", () => { node.remove(); if (onDismiss) onDismiss(); });
  return node;
}

/** Show a toast in the page's toast stack; it dismisses itself after `ms`. */
export function showToast(opts, { ms = 6000, root = document.body } = {}) {
  let stack = root.querySelector(":scope > .odk-toasts");
  if (!stack) {
    stack = el("div", { class: "odk-toasts", "aria-live": "polite" });
    root.appendChild(stack);
  }
  const node = toast(opts);
  stack.appendChild(node);
  if (ms > 0) setTimeout(() => node.remove(), ms);
  return node;
}

/* ----------------------------------------------------------------- tabs */

/**
 * Workspace tabs (header). items: [{id, label}]. Active gets aria-current="page".
 * These switch screens, so they are controls (data-control="tab:<id>").
 */
export function tabs(items, { active, onSelect, label = "Workspaces", className = "odk-tab" } = {}) {
  const nav = el("nav", { class: "odk-tabs", "aria-label": label });
  for (const item of items) {
    const a = el("a", { href: `#${item.id}`, class: className, "data-control": `tab:${item.id}`, text: item.label });
    if (item.id === active) a.setAttribute("aria-current", "page");
    a.addEventListener("click", (event) => {
      event.preventDefault();
      nav.querySelectorAll("[aria-current]").forEach((n) => n.removeAttribute("aria-current"));
      a.setAttribute("aria-current", "page");
      if (onSelect) onSelect(item.id);
    });
    nav.appendChild(a);
  }
  return nav;
}

/** The Field bottom tab bar: same contract as tabs(), field geometry. */
export function fieldTabBar(items, opts = {}) {
  const nav = tabs(items, { ...opts, className: "odk-field-tab", label: opts.label || "Field workspaces" });
  nav.className = "odk-field-tabbar";
  return nav;
}

/** Panel tabs (role=tablist). items: [{id, label, count}] -- count must come from the Api. */
export function panelTabs(items, { active, onSelect, label = "Panel" } = {}) {
  const list = el("div", { class: "odk-panel-tabs", role: "tablist", "aria-label": label });
  for (const item of items) {
    const tab = el("button", {
      type: "button", class: "odk-panel-tab", role: "tab", "data-control": `panel-tab:${item.id}`,
      "aria-selected": item.id === active ? "true" : "false",
    }, [item.label, item.count != null ? el("span", { class: "odk-num", text: String(item.count) }) : null]);
    tab.addEventListener("click", () => {
      list.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", "false"));
      tab.setAttribute("aria-selected", "true");
      if (onSelect) onSelect(item.id);
    });
    list.appendChild(tab);
  }
  return list;
}

/* ---------------------------------------------------------------- table */

/**
 * A table. columns: [{title, key | value(row), num, dim}]. Wraps primitives.table, so
 * rows publish to the selection bus with `selectKind` exactly as the cockpit's did, and
 * the data-rows marker the honesty guards count is kept.
 */
export function table(columns, rows, opts = {}) {
  const node = primitiveTable(columns, rows, opts);
  node.classList.add("odk-table");
  const ths = node.querySelectorAll("thead th");
  columns.forEach((c, i) => {
    if (c.num && ths[i]) ths[i].classList.add("num");
    if (c.title && ths[i]) ths[i].setAttribute("scope", "col");
  });
  node.querySelectorAll("tbody tr").forEach((tr) => {
    tr.querySelectorAll("td").forEach((td, i) => {
      const c = columns[i] || {};
      if (c.num) td.classList.add("num");
      if (c.dim) td.classList.add("dim");
    });
  });
  return node;
}

/* ----------------------------------------------------------- layer rows */

/**
 * A layer row. swatch: polygon | point | line | ramp. A row the Api cannot back is
 * `disabled` with a `note` stating why ("not available").
 */
export function layerRow({ label, checked = false, swatch, note, selected = false, disabled = false, onToggle } = {}) {
  const box = el("input", { type: "checkbox" });
  box.checked = checked;
  box.disabled = disabled;
  if (onToggle) box.addEventListener("change", () => onToggle(box.checked));
  return el("label", {
    class: ["odk-layer-row", selected ? "odk-layer-row--selected" : "", disabled ? "odk-layer-row--disabled" : ""]
      .filter(Boolean).join(" "),
  }, [
    box,
    el("span", { class: "odk-layer-name", text: label }),
    note ? el("span", { class: "odk-layer-note", text: note }) : null,
    swatch && !disabled ? el("span", { class: `odk-swatch odk-swatch--${swatch}`, "aria-hidden": "true" }) : null,
  ]);
}

/* ---------------------------------------------------------------- stats */

/**
 * A labelled value. `value` must come from an Api result; null/undefined renders
 * "not reported" rather than a placeholder number. size: value | hero.
 */
export function stat({ label, value, unit, size = "value" } = {}) {
  const missing = value === null || value === undefined || value === "";
  const valueNode = el("span", { class: "odk-stat-value" });
  if (missing) valueNode.appendChild(el("span", { class: "odk-stat-missing", text: "not reported" }));
  else {
    valueNode.appendChild(document.createTextNode(String(value)));
    if (unit) valueNode.appendChild(el("span", { class: "odk-stat-unit", text: unit }));
  }
  return el("div", { class: `odk-stat odk-stat--${size}` }, [
    el("span", { class: "odk-stat-label", text: label }), valueNode,
  ]);
}

/* -------------------------------------------------------------- popover */

/**
 * A popover anchored to `anchor`. Returns {node, open, close, toggle}. Closes on
 * Escape and on a click outside. Items: [{label, action}] render as menu buttons that
 * follow the data-action rule.
 */
export function popover({ anchor, title, items = [], content } = {}) {
  const body = el("div", { class: "odk-popover-body" });
  for (const item of items) {
    const b = marks(el("button", { type: "button", class: "odk-menu-item", text: item.label }), item);
    if (item.disabled) b.disabled = true;
    if (item.onClick) b.addEventListener("click", item.onClick);
    body.appendChild(b);
  }
  if (content) body.appendChild(content);
  const node = el("div", { class: "odk-popover", role: "dialog", hidden: "" }, [
    title ? el("div", { class: "odk-popover-head odk-label", text: title }) : null, body,
  ]);
  if (title) node.setAttribute("aria-label", title);
  // Mounted inside the anchor's mode host, so a popover opened from a Field board is
  // drawn in Field style even when the page itself is in Desk mode.
  const modeHost = anchor && anchor.closest ? anchor.closest("[data-mode]") : null;
  (modeHost && modeHost !== document.documentElement ? modeHost : document.body).appendChild(node);
  const place = () => {
    if (!anchor) return;
    const r = anchor.getBoundingClientRect();
    node.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - node.offsetWidth - 8))}px`;
    node.style.top = `${r.bottom + 4}px`;
  };
  const onKey = (e) => { if (e.key === "Escape") api.close(); };
  const onDoc = (e) => { if (!node.contains(e.target) && e.target !== anchor && !anchor?.contains(e.target)) api.close(); };
  const api = {
    node,
    open() {
      node.hidden = false; place();
      if (anchor) anchor.setAttribute("aria-expanded", "true");
      document.addEventListener("keydown", onKey);
      setTimeout(() => document.addEventListener("mousedown", onDoc), 0);
    },
    close() {
      node.hidden = true;
      if (anchor) anchor.setAttribute("aria-expanded", "false");
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDoc);
    },
    toggle() { if (node.hidden) api.open(); else api.close(); },
  };
  if (anchor) {
    anchor.setAttribute("aria-haspopup", "dialog");
    anchor.setAttribute("aria-expanded", "false");
    anchor.addEventListener("click", () => api.toggle());
  }
  return api;
}

/* -------------------------------------------------------- map furniture */

/** CRS chip. Text comes from get_project_crs; click copies `wkt` when given. */
export function crsChip({ text, wkt } = {}) {
  const node = el("button", { type: "button", class: "odk-map-chip", "data-control": "copy-crs", title: "Copy WKT", text });
  if (wkt) node.addEventListener("click", () => navigator.clipboard?.writeText(wkt).catch(() => {}));
  return node;
}

/** Zoom / tool stack: icon buttons with 1px gaps (desk), 8px (field). */
export function mapStack(buttons) {
  return el("div", { class: "odk-map-stack" }, buttons);
}

/** Scale bar. `label` and `ratio` are computed by the map from zoom and latitude. */
export function scaleBar({ label, ratio, widthPx = 96 } = {}) {
  const half = `${Math.round(widthPx / 2)}px`;
  return el("div", { class: "odk-scalebar", "aria-label": `Scale ${label}` }, [
    el("span", { class: "odk-scalebar-bar", "aria-hidden": "true" }, [
      el("span", { style: `width:${half}` }), el("span", { style: `width:${half}` }),
    ]),
    el("span", { text: label }),
    ratio ? el("span", { class: "odk-scalebar-ratio", text: ratio }) : null,
  ]);
}

/** North arrow; rotates with the map bearing (degrees). */
export function northArrow({ bearing = 0 } = {}) {
  const node = el("div", { class: "odk-north", role: "img", "aria-label": "North" });
  node.innerHTML = '<svg width="14" height="18" viewBox="0 0 14 18" aria-hidden="true">'
    + '<path class="odk-north-outline" d="M7 1L13 17L7 13L1 17Z"></path>'
    + '<path class="odk-north-fill" d="M7 1L7 13L1 17Z"></path></svg><span>N</span>';
  const setBearing = (deg) => { node.querySelector("svg").style.transform = `rotate(${-deg}deg)`; };
  setBearing(bearing);
  node.setBearing = setBearing;
  return node;
}

/**
 * Elevation legend on the turbo ramp. `min`/`max` come from raster_preview; the ticks
 * are evenly spaced between them (deterministic arithmetic, not data).
 */
export function elevationLegend({ min, max, ticks = 5, unit = "m", heightPx = 132 } = {}) {
  const labels = [];
  if (Number.isFinite(min) && Number.isFinite(max)) {
    for (let i = ticks - 1; i >= 0; i -= 1) labels.push((min + ((max - min) * i) / (ticks - 1)).toFixed(1));
  }
  return el("div", { class: "odk-legend", role: "img", "aria-label": `Elevation legend, ${unit}` }, [
    el("div", { class: "odk-legend-ramp", style: `height:${heightPx}px` }),
    el("div", { class: "odk-legend-ticks", style: `height:${heightPx}px` },
      labels.length ? labels.map((t) => el("span", { text: t })) : [el("span", { text: "not reported" })]),
  ]);
}

/** Geometry key: measurement, vertex, flight line, simulated track. */
export function geometryKey(items = [
  { kind: "measurement", label: "measurement" },
  { kind: "vertex", label: "vertex · camera" },
  { kind: "flight", label: "flight line" },
  { kind: "simulated", label: "simulated track" },
]) {
  return el("div", { class: "odk-geom-key" }, items.map((i) =>
    el("div", { class: "odk-geom-item" }, [
      el("span", { class: `odk-geom-swatch odk-geom-swatch--${i.kind}`, "aria-hidden": "true" }),
      el("span", { text: i.label }),
    ])));
}

/* -------------------------------------------------------------- section */

/** A panel section with an 11px mono uppercase label. */
export function section({ label, aside, children = [], className = "" } = {}) {
  return el("section", { class: `odk-section ${className}`.trim(), "aria-label": label }, [
    el("div", { class: "odk-section-head" }, [
      el("span", { class: "odk-label", text: label }),
      aside ? (typeof aside === "string" ? el("span", { class: "odk-label", text: aside }) : aside) : null,
    ]),
    el("div", { class: "odk-section-body" }, children),
  ]);
}

/** The current mode of the nearest [data-mode] ancestor (or the document). */
export function modeOf(node = document.documentElement) {
  const host = node.closest ? node.closest("[data-mode]") : null;
  return (host && host.getAttribute("data-mode")) || "desk";
}
