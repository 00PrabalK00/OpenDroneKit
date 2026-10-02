/* odk.html?gallery -- every component on the SysTokens reference board, rendered by
 * js/odk/components.js, once in desk mode and once in field mode on the same page.
 *
 * A specimen page, not a product screen: the sample values are the ones printed on the
 * approved SysTokens board (Aukerman Park reference data), and the page says so. No
 * product screen may take a value from here. Each specimen is wrapped in
 * [data-component="<name>"] so tests/test_ui_v1_gallery.py can check that every one
 * rendered in both boards.
 */

import {
  el, button, iconButton, iconGroup, input, select, search, checkRow, chip, tag, demoTag,
  simulatedTag, status, toast, tabs, fieldTabBar, panelTabs, table, layerRow, stat, popover,
  crsChip, mapStack, scaleBar, northArrow, elevationLegend, geometryKey, section, icon,
} from "./components.js";
import { DESK_SCREENS, FIELD_SCREENS } from "./screens/index.js";

/** Component names every board must render (the test reads this list from the page). */
export const COMPONENTS = [
  "colour", "ramp", "type", "space", "button-primary", "button-secondary", "button-ghost",
  "button-danger", "button-commit", "icon-button", "input", "input-focus", "select", "search",
  "workspace-tabs", "panel-tabs", "check-row", "layer-rows", "chips", "tags", "status", "toast",
  "toast-danger", "table", "fleet-row", "stat", "popover", "crs-chip", "map-stack", "scale-bar",
  "north-arrow", "elevation-legend", "geometry-key",
];

/** Field-only parts of the layout (header and bottom tab bar). */
export const FIELD_ONLY = ["field-header", "field-tabbar"];

const COLOUR_TOKENS = [
  ["--bg", "bg · app canvas"], ["--surface-1", "surface-1 · header, panels, footer"],
  ["--surface-2", "surface-2 · controls"], ["--surface-3", "surface-3 · selected row"],
  ["--map-well", "map well"], ["--line", "line · rule"], ["--line-strong", "line-strong · hover edge"],
  ["--text", "text · primary"], ["--text-2", "text-2 · secondary"], ["--muted", "muted · labels"],
  ["--accent", "accent · selection"], ["--accent-hover", "accent-hover · links"],
  ["--action", "action · the one action"], ["--on-action", "on-action · text on action"],
  ["--ok", "ok"], ["--warn", "warn"], ["--danger", "danger"], ["--sim", "simulated / demo tag"],
];

const STYLE = `
.gal { padding: 24px; display: flex; flex-direction: column; gap: 24px; }
.gal-board { display: flex; flex-direction: column; gap: 24px; padding: 24px; border: var(--rule-strong) solid var(--line); }
.gal-board-head { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.gal-board-head h2 { font-size: var(--fs-title); font-weight: var(--fw-strong); }
.gal-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 24px; align-items: start; }
.gal-specimen { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.gal-specimen > .odk-label { display: block; }
.gal-wrap { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.gal-grid { display: grid; grid-template-columns: max-content repeat(2, minmax(0, 1fr)); gap: 8px; align-items: center; }
.gal-swatches { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px 8px; }
.gal-swatch-value { font-family: var(--font-mono); font-size: var(--fs-label); }
.gal-swatch-name { font-size: var(--fs-small); color: var(--text-2); line-height: 1.3; }
.gal-type-row { display: flex; align-items: baseline; gap: 12px; border-bottom: var(--rule) solid var(--line); padding: 4px 0; }
.gal-type-row > .odk-label { width: 72px; flex-shrink: 0; }
.gal-space { display: flex; align-items: center; gap: 12px; }
.gal-space-bar { height: 16px; background: var(--accent); }
.gal-map { position: relative; height: 240px; background: var(--map-well); border: var(--rule) solid var(--line); }
.gal-map > * { position: absolute; }
.gal-footer { height: var(--footer-h, 28px); display: flex; align-items: center; gap: 24px; padding: 0 16px;
  background: var(--surface-1); border-top: 1px solid var(--line); font-family: var(--font-mono);
  font-size: var(--fs-label); color: var(--muted); }
.gal-footer .net { margin-left: auto; color: var(--ok); }
.gal-pop-host { min-height: 64px; }
[data-mode="field"] .gal-footer { display: none; }
[data-mode="field"] .gal-row { grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); }
[data-mode="field"] .gal-swatch-value { font-family: var(--font-ui); font-weight: 700; font-size: 14px; }
[data-mode="field"] .gal-swatch-name { font-size: 14px; color: var(--text); }
[data-mode="field"] .gal-map { height: 380px; border-width: var(--rule-strong); background: #3A3A3A; }
`;

const spec = (name, label, children) =>
  el("div", { class: "gal-specimen", "data-component": name }, [
    el("span", { class: "odk-label", text: label }), ...[].concat(children),
  ]);

function colours(board) {
  const grid = el("div", { class: "gal-swatches" });
  for (const [token, name] of COLOUR_TOKENS) {
    const value = el("span", { class: "gal-swatch-value", "data-token": token });
    grid.appendChild(el("div", { class: "odk-swatch-card" }, [
      el("div", { class: "odk-swatch-chip", style: `background:var(${token})` }), value,
      el("span", { class: "gal-swatch-name", text: `${name}` }),
    ]));
  }
  // Values are read back from the computed tokens, so the board documents what the
  // stylesheet actually says rather than a copy of it.
  requestAnimationFrame(() => {
    const style = getComputedStyle(board);
    grid.querySelectorAll("[data-token]").forEach((n) => {
      n.textContent = style.getPropertyValue(n.dataset.token).trim().toUpperCase();
    });
  });
  return grid;
}

function typeScale() {
  const row = (label, node) => el("div", { class: "gal-type-row" }, [el("span", { class: "odk-label", text: label }), node]);
  return el("div", {}, [
    row("hero", stat({ label: "Area", value: "23,592", unit: "m²", size: "hero" })),
    row("value", stat({ label: "Perimeter", value: "609.7", unit: "m" })),
    row("title", el("span", { style: "font-size:var(--fs-title);font-weight:var(--fw-strong)", text: "Aukerman Park" })),
    row("body", el("span", { text: "Orthomosaic · 0.033 m/px" })),
    row("small", el("span", { style: "font-size:var(--fs-small);color:var(--muted)", text: "Volume above terrain" })),
    row("data", el("span", { class: "odk-mono", text: "41.30410° N 81.75212° W" })),
    row("label", el("span", { class: "odk-label", style: "color:var(--text)", text: "Reconstruction" })),
  ]);
}

function spaceScale() {
  return el("div", {}, [
    ...[["--sp-1", "icon–text"], ["--sp-2", "inline gap"], ["--sp-3", "field stack"], ["--sp-4", "panel pad"], ["--sp-5", "section"]]
      .map(([t, n]) => el("div", { class: "gal-space" }, [
        el("span", { class: "odk-label", style: "width:48px", text: t.replace("--", "") }),
        el("span", { class: "gal-space-bar", style: `width:var(${t})` }),
        el("span", { class: "odk-text-2", text: n }),
      ])),
  ]);
}

function buttonGrid(variant, label, options) {
  const forced = button({ ...options, variant, label });
  forced.classList.add("is-hover");
  return [
    el("span", { class: "odk-label", text: variant }),
    button({ ...options, variant, label }),
    forced,
    button({ ...options, variant, label, disabled: true }),
  ];
}

function buttons() {
  const head = () => ["", "default", "hover", "disabled"].map((t) => el("span", { class: "odk-label", text: t }));
  const grid = (variant, label, options) => el("div", {
    class: "gal-grid", style: "grid-template-columns:max-content repeat(3,minmax(0,1fr))", "data-component": `button-${variant}`,
  }, [...head(), ...buttonGrid(variant, label, options)]);
  return [
    grid("primary", "Generate", { action: "Generate" }),
    grid("secondary", "Export", { action: "Export" }),
    grid("ghost", "Cancel", { control: "cancel" }),
    grid("danger", "Abort", { action: "Abort" }),
    grid("commit", "Add to report", { control: "specimen" }),
  ];
}

function iconButtons() {
  const hover = iconButton({ iconName: "measure", label: "Measure, hover", action: "Measure" });
  hover.classList.add("is-hover");
  return el("div", { class: "gal-wrap" }, [
    iconGroup([
      iconButton({ iconName: "measure", label: "Measure", action: "Measure" }),
      iconButton({ iconName: "measure", label: "Measure, active", action: "Measure", pressed: true }),
    ]),
    hover,
    iconButton({ iconName: "measure", label: "Measure, disabled", action: "Measure", disabled: true }),
    iconButton({ iconName: "centre", label: "Centre on me", control: "specimen" }),
  ]);
}

function inputs() {
  const focused = input({ label: "Ground sample distance · focus", value: "1.29", unit: "cm/px", mono: true, numeric: true });
  focused.querySelector(".odk-input").classList.add("is-focus");
  return [
    spec("input", "Input", input({ label: "Flight altitude", value: "55", unit: "m", mono: true, numeric: true })),
    spec("input-focus", "Input · focus", focused),
    spec("select", "Select", select({ label: "Coordinate system", options: ["EPSG:32617 · UTM 17N"], value: "EPSG:32617 · UTM 17N" })),
    spec("search", "Search", search({ label: "Search", placeholder: "Layers, findings, coordinates" })),
  ];
}

function layerRows() {
  const hover = layerRow({ label: "Camera positions · 77", checked: true, swatch: "point", note: "hover" });
  hover.classList.add("is-hover");
  return el("div", {}, [
    layerRow({ label: "Measurement · polygon", checked: true, swatch: "polygon", selected: true }),
    hover,
    layerRow({ label: "Grid mission · 152 wp", swatch: "line" }),
    layerRow({ label: "DSM · 1.23 m/px", checked: true, swatch: "ramp" }),
    layerRow({ label: "Thermal · radiometric", disabled: true, note: "not available" }),
  ]);
}

function statuses() {
  return el("div", { class: "gal-wrap", style: "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 12px" }, [
    status("ok", "local · no network"), status("warn", "preview quality"), status("danger", "cannot run"),
    status("sim", "simulated telemetry"), status("running", "running · selected"), status("idle", "not started"),
  ]);
}

function modelTable() {
  const rows = [
    { model: "crack_segmentation", metric: "IoU 0.637", sha: "3391eca1" },
    { model: "corrosion_severity_seg…", metric: "severe rec. 0.788", sha: "edcecd84" },
    { model: "solar_cell_defect_detector", metric: "mAP50 0.884", sha: "a57676a7" },
    { model: "solar_thermal_anomaly_cl…", metric: "bal. acc 0.724", sha: "6933a09a" },
  ];
  const node = table([
    { title: "Model", key: "model" },
    { title: "Registry metric", key: "metric", num: true },
    { title: "SHA256", key: "sha", num: true, dim: true },
  ], rows);
  const trs = node.querySelectorAll("tbody tr");
  trs[0].classList.add("is-hover");
  trs[1].classList.add("selected");
  return node;
}

function fleetRow() {
  // The tag is drawn because the record says demo: true, the way a fleet record from the
  // Api does -- never because the screen assumes it.
  const record = { name: "eBee - Aukerman", hours: "0.29 h · 1 flt", service: "100 h to svc", demo: true };
  return table([
    { title: "Aircraft", key: "name" },
    { title: "Hours", key: "hours", num: true },
    { title: "Service", key: "service", num: true, dim: true },
    { title: "", value: (r) => demoTag(r, { small: true }) || "" },
  ], [record]);
}

function mapFurniture() {
  const measureOn = iconButton({ iconName: "measure", label: "Measure", action: "Measure", pressed: true });
  return el("div", { class: "gal-map", role: "img", "aria-label": "Map furniture specimen" }, [
    el("div", { style: "left:12px;top:12px", "data-component": "crs-chip" }, [crsChip({ text: "EPSG:32617 · UTM 17N" })]),
    el("div", { style: "right:12px;top:12px", "data-component": "map-stack" }, [mapStack([
      iconButton({ text: "+", label: "Zoom in", control: "zoom-in" }),
      iconButton({ text: "−", label: "Zoom out", control: "zoom-out" }),
      iconButton({ iconName: "fit", label: "Fit to extent", control: "fit" }),
      measureOn,
    ])]),
    el("div", { style: "left:12px;bottom:12px", "data-component": "scale-bar" }, [scaleBar({ label: "50 m", ratio: "1 : 1,500" })]),
    el("div", { style: "right:72px;bottom:12px", "data-component": "north-arrow" }, [northArrow({ bearing: 0 })]),
  ]);
}

function fieldHeader() {
  return el("header", { class: "odk-field-header", "data-component": "field-header" }, [
    el("span", { class: "odk-brand" }, [el("img", { src: "assets/brand/mark-dark.svg", alt: "" }), "Aukerman Park"]),
    el("div", { style: "flex-grow:1" }),
    el("div", { class: "odk-field-net", role: "status" }, [el("span", { class: "odk-dot" }), "Local · no network"]),
    (() => {
      const b = el("button", { type: "button", class: "odk-field-toggle", "data-control": "mode:desk", "aria-label": "Switch to Desk mode" });
      b.appendChild(icon("monitor", { size: 22, stroke: 2.4 }));
      b.appendChild(document.createTextNode("Desk mode"));
      return b;
    })(),
  ]);
}

function board(mode) {
  const node = el("section", {
    class: "gal-board", "data-mode": mode, "data-board": mode,
    "aria-label": mode === "field" ? "Field mode, direction C Field Daylight" : "Desk mode, direction A Survey Instrument",
  });
  const title = mode === "field" ? "C · Field Daylight · field mode" : "A · Survey Instrument · desk";

  const popAnchor = button({ label: "Layers", control: "open-layers", variant: "secondary", iconName: "layers" });
  const popHost = el("div", { class: "gal-pop-host", "data-component": "popover" }, [popAnchor]);

  node.append(...[
    el("div", { class: "gal-board-head" }, [
      el("h2", { text: title }),
      chip(mode === "field" ? "FIELD 1180×820" : "DESK 1440×900"),
    ]),
    mode === "field" ? fieldHeader() : null,
    el("div", { class: "gal-row" }, [
      section({ label: "01 · Colour", aside: "tokens", children: [
        el("div", { "data-component": "colour" }, [colours(node)]),
        spec("ramp", "Elevation ramp · turbo · the only gradient", el("div", { class: "odk-ramp" })),
      ] }),
      section({ label: "02 · Type", children: [el("div", { "data-component": "type" }, [typeScale()])] }),
      section({ label: "03 · Space", children: [el("div", { "data-component": "space" }, [spaceScale()])] }),
    ]),
    el("div", { class: "gal-row" }, [
      section({ label: "04 · Buttons · inputs · tabs", children: [
        ...buttons(),
        spec("icon-button", "Icon buttons", iconButtons()),
        ...inputs(),
        spec("workspace-tabs", "Workspace tabs",
          mode === "field"
            ? tabs(FIELD_SCREENS, { active: "analyse", label: "Workspace tabs specimen" })
            : tabs(DESK_SCREENS, { active: "analyse", label: "Workspace tabs specimen" })),
        spec("panel-tabs", "Panel tabs", panelTabs([
          { id: "layers", label: "Layers" }, { id: "findings", label: "Findings", count: 0 }, { id: "prov", label: "Provenance" },
        ], { active: "layers", label: "Panel tabs specimen" })),
      ] }),
      section({ label: "05 · Layers · chips · status", children: [
        spec("check-row", "Checkbox rows", [checkRow({ label: "Include audit trail", checked: true }), checkRow({ label: "Dense reconstruction", disabled: true })]),
        spec("layer-rows", "Layer rows", layerRows()),
        spec("chips", "Chips", el("div", { class: "gal-wrap" }, [
          chip("EPSG:32617"), chip("77 / 77 IMAGES"), chip("RMSE 1.11 M", "ok"), chip("PREVIEW DSM", "warn"), chip("WEIGHTS MISSING", "danger"),
        ])),
        spec("tags", "Tags (rendered from Api flags)", el("div", { class: "gal-wrap" }, [
          simulatedTag({ is_simulated: true }), demoTag({ demo: true }), tag("SIMULATED POSITION"),
        ])),
        spec("status", "Status", statuses()),
        spec("toast", "Toast", toast({ title: "Exported mission · .kmz", detail: "grid · 152 wp · 14,626 B", detailMono: true, tone: "ok" })),
        spec("popover", "Popover", popHost),
      ] }),
      section({ label: "06 · Table · toast", children: [
        spec("table", "Table · hover row 1 · selected row 2", modelTable()),
        spec("fleet-row", "Fleet row with demo fields", fleetRow()),
        spec("toast-danger", "Refusal", toast({
          title: "Refused · nothing was generated",
          detail: "corrosion_severity_segmentation cannot run: Its weights are not installed on this machine.",
          tone: "danger",
          link: el("a", { href: "#", class: "odk-toast-link", "data-control": "specimen", text: "Install weights" }),
        })),
        spec("stat", "Stat", el("div", { class: "gal-wrap", style: "gap:24px" }, [
          stat({ label: "Waypoints", value: "152", size: "hero" }),
          stat({ label: "Flight time", value: "5.1", unit: "min" }),
          stat({ label: "Battery", value: null }),
        ])),
      ] }),
    ]),
    section({ label: "07 · Map furniture", children: [
      el("div", { class: "gal-row" }, [
        mapFurniture(),
        el("div", { class: "gal-wrap", style: "align-items:flex-start;gap:24px" }, [
          spec("elevation-legend", "Elevation legend · DSM", elevationLegend({ min: 230.3, max: 264.1 })),
          spec("geometry-key", "Geometry", geometryKey()),
        ]),
      ]),
      el("div", { class: "gal-footer" }, [
        el("span", { text: "41.30410° N  81.75212° W" }), el("span", { text: "EPSG:32617" }),
        el("span", { text: "1 : 1,500" }), el("span", { class: "net", text: "● local · no network" }),
      ]),
    ] }),
    mode === "field"
      ? el("div", { "data-component": "field-tabbar" }, [fieldTabBar(FIELD_SCREENS, { active: "analyse" })])
      : null,
  ].filter(Boolean));

  // Open the popover so it is visible in the specimen.
  requestAnimationFrame(() => {
    const pop = popover({ anchor: popAnchor, title: "Map layers", items: [
      { label: "Add overlay…", action: "Import CAD" },
      { label: "Cache visible map", control: "specimen" },
      { label: "Street basemap · network", control: "specimen", disabled: true },
    ] });
    pop.node.style.position = "static";
    pop.node.hidden = false;
    pop.node.setAttribute("data-specimen", "open");
    popHost.appendChild(pop.node);
  });
  return node;
}

export function mountGallery(root) {
  document.title = "OpenDroneKit · component gallery";
  document.head.appendChild(el("style", { text: STYLE }));
  root.replaceChildren(el("div", { class: "gal", "data-gallery": "" }, [
    el("header", { class: "odk-frame-header", style: "height:56px" }, [
      el("span", { class: "odk-brand" }, [el("img", { src: "assets/brand/mark-dark.svg", alt: "" }), "OpenDroneKit"]),
      el("span", { class: "odk-text-2", text: "Design system v1 · component gallery · specimen values from the approved SysTokens board" }),
    ]),
    board("desk"),
    board("field"),
  ]));
  window.__odkGallery = { components: COMPONENTS, fieldOnly: FIELD_ONLY };
}
