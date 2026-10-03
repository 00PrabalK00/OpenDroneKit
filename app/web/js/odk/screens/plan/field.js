/* Field Plan (SysFieldPlan artboard): the same plan as the desk screen, for a tablet in
   the sun. Reads the same model (./model.js) through live(), so both modes show the
   Api's figures.

     main   the project map (Field paint: ink casings, yellow route), the plan's
            START / END and footprints (./map.js), and the band of four figures:
            WAYPOINTS (the one yellow tile), FLIGHT TIME, DISTANCE, GROUND DETAIL
     left   (the side panel, on the right) MISSION TYPE tiles, ALTITUDE stepper,
            OVERLAP / SPEED / LINES, the terrain warning, Send to aircraft (the screen's
            one yellow action) and Export

   A type tile or an altitude step re-plans at once through the core "Plan" verb: in
   the field there is no separate Generate step to forget. */

import { el } from "../../components.js";
import { live } from "../../../workspace/live.js";
import { mountProjectMap } from "../../map/furniture.js";
import {
  EXPORT_FORMATS, exportChoice, fmtInt, fmtNum, form, frameWords, missionOptions, templateLabel, templateList,
} from "./model.js";
import { observe } from "./history.js";
import { attachPlanMap, overviewFeed } from "./map.js";

export const SCREEN = "plan";
export const MODE = "field";

/* The three types the artboard puts on tiles; every other type is one tap away in
   "More types", so none is lost in Field mode. */
const TILES = ["grid", "double_grid", "orbit"];

const TILE_ICONS = {
  grid: '<path d="M5 4v16M10 4v16M15 4v16M20 4v16"></path><path d="M5 4h5M10 20h5M15 4h5"></path>',
  double_grid: '<path d="M4 6h16M4 12h16M4 18h16M6 4v16M12 4v16M18 4v16"></path>',
  orbit: '<circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="2"></circle><path d="M20 12l-2-2M20 12l2-2"></path>',
  more: '<circle cx="5" cy="12" r="1.8"></circle><circle cx="12" cy="12" r="1.8"></circle><circle cx="19" cy="12" r="1.8"></circle>',
  send: '<path d="M3 11l18-8-8 18-2-8z"></path>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"></path>',
  warn: '<path d="M12 3l10 18H2z"></path><path d="M12 10v5M12 18v.5"></path>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"></path><path d="M13 7l4 4"></path>',
};

function ico(paths, size = 24) {
  const span = document.createElement("span");
  span.className = "odk-planf-ico";
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  return span;
}

/* Screen CSS lives beside the screen (docs/ui/ARCHITECTURE.md); linked once, and
   awaited, because the map measures its container when it is made: a map made before
   this sheet applies is sized for a collapsed box and fitted to the wrong view. */
function ensureStyles() {
  const existing = document.querySelector("link[data-odk-screen-css=plan]");
  if (existing) return existing._loaded || Promise.resolve();
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./plan.css", import.meta.url).href;
  link.setAttribute("data-odk-screen-css", "plan");
  link._loaded = new Promise((done) => {
    link.addEventListener("load", () => done(), { once: true });
    link.addEventListener("error", () => done(), { once: true });
  });
  document.head.appendChild(link);
  return link._loaded;
}

const state = { overview: null, ctl: null, timer: null, templates: null };
let shellRef = null;

function run(key, payload = null) {
  if (shellRef) return shellRef.runAction(key, payload);
  return null;
}

/** Re-plan shortly after the last tap, so three taps on + plan once. */
function replanSoon() {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => run("Plan"), 450);
}

function take(overview) {
  if (!overview) return;
  state.overview = overview;
  observe(overview.aoi);
}

/* ------------------------------------------------------------ panel */

function sidePanel() {
  return live({
    calls: ["mission_overview", "mission_templates", "list_mission_versions"],
    empty: "The mission could not be read.",
    isEmpty: ([overview]) => !overview,
    render: ([overview, templates, versions]) => {
      take(overview);
      const plan = overview.plan;
      state.templates = templates;
      const rows = templateList(templates, versions, plan);
      const active = form.template || (plan && plan.template) || null;
      const node = el("div", { class: "odk-planf-panel" });

      node.appendChild(el("div", { class: "odk-planf-label", text: "MISSION TYPE" }));
      const tiles = el("div", { class: "odk-planf-tiles" });
      const shown = TILES.includes(active) || !active ? TILES : [...TILES.slice(0, 2), active];
      for (const key of shown) {
        const row = rows.find((r) => r.key === key);
        if (!row) continue;
        const tile = el("button", { type: "button", class: "odk-planf-tile", "data-control": `template:${key}`,
          "aria-pressed": key === active ? "true" : "false" }, [
          ico(TILE_ICONS[key] || TILE_ICONS.grid, 28),
          el("span", { class: "odk-planf-tile-name", text: row.label }),
          el("span", { class: "odk-planf-tile-wp", text: row.waypoints != null ? `${fmtInt(row.waypoints)} wp` : "not planned" }),
        ]);
        tile.addEventListener("click", () => {
          if (key === active) return;
          form.template = key;
          form.dirty = true;
          replanSoon();
          if (node.parentNode && node.parentNode._settle) node.parentNode._settle();
        });
        tiles.appendChild(tile);
      }
      const more = el("button", { type: "button", class: "odk-planf-tile odk-planf-more", "data-control": "templates:more",
        "aria-label": `More mission types (${rows.length})` },
        [ico(TILE_ICONS.more, 28), el("span", { class: "odk-planf-tile-name", text: "More" }),
          el("span", { class: "odk-planf-tile-wp", text: `${rows.length} types` })]);
      more.addEventListener("click", async () => {
        const picked = shellRef && await shellRef.actionContext().choose("Mission type",
          rows.map((r) => ({ label: r.label, value: r.key, hint: r.waypoints != null ? `${fmtInt(r.waypoints)} wp` : r.hint })));
        if (!picked) return;
        form.template = picked;
        form.dirty = true;
        replanSoon();
      });
      tiles.appendChild(more);
      node.appendChild(tiles);

      // ALTITUDE: the plan's own altitude, stepped and re-planned.
      node.appendChild(el("div", { class: "odk-planf-label", text: "ALTITUDE" }));
      const typed = Number(form.values.altitude_m);
      const altitude = Number.isFinite(typed) && form.values.altitude_m !== undefined && form.values.altitude_m !== ""
        ? typed : plan ? plan.altitude_m : null;
      const step = (delta) => {
        const base = Number.isFinite(Number(altitude)) && altitude !== null ? Number(altitude) : null;
        if (base === null) {
          shellRef && shellRef.toast("Altitude: plan once first, then step it.", "warn");
          return;
        }
        form.values.altitude_m = String(Math.max(1, base + delta));
        form.dirty = true;
        valueNode.textContent = fmtNum(form.values.altitude_m, 0);
        replanSoon();
      };
      const minus = el("button", { type: "button", class: "odk-planf-step", "data-control": "altitude:down", "aria-label": "Lower by 5 m", text: "−" });
      const plus = el("button", { type: "button", class: "odk-planf-step", "data-control": "altitude:up", "aria-label": "Raise by 5 m", text: "+" });
      minus.addEventListener("click", () => step(-5));
      plus.addEventListener("click", () => step(5));
      const valueNode = el("span", { class: "odk-planf-alt-value", text: altitude != null ? fmtNum(altitude, 0) : "—" });
      node.appendChild(el("div", { class: "odk-planf-stepper" }, [minus,
        el("div", { class: "odk-planf-alt" }, [valueNode, el("span", { class: "odk-planf-alt-unit", text: "m" })]), plus]));
      node.appendChild(el("p", { class: "odk-planf-note", text: plan
        ? (plan.altitude_frame === "relative_to_home" ? "Above launch point" : `Altitude ${frameWords(plan.altitude_frame)}`)
        : "Plan once to set the altitude." }));

      const cell = (label, value) => el("div", { class: "odk-planf-cell" }, [
        el("span", { class: "odk-planf-cell-label", text: label }),
        el("span", { class: "odk-planf-cell-value", text: value == null ? "—" : value }),
      ]);
      node.appendChild(el("div", { class: "odk-planf-cells" }, [
        cell("OVERLAP", plan && plan.front_overlap_pct != null ? `${fmtNum(plan.front_overlap_pct, 0)} / ${fmtNum(plan.side_overlap_pct, 0)} %` : null),
        cell("SPEED", plan && plan.cruise_speed_m_s != null ? `${fmtNum(plan.cruise_speed_m_s, 0)} m/s` : null),
        cell("LINES", plan && plan.line_spacing_m ? `${fmtNum(plan.line_spacing_m, 1)} m` : null),
      ]));

      const terrain = plan && plan.terrain;
      if (!plan || !(terrain && (terrain.follow || terrain.sampled))) {
        node.appendChild(el("div", { class: "odk-planf-warn", role: "note" }, [
          ico(TILE_ICONS.warn, 26),
          el("p", {}, [el("strong", { text: "No terrain model." }), " Height is from the launch point, not the ground."]),
        ]));
      }

      const send = el("button", { type: "button", class: "odk-planf-send", "data-action": "Send to Fly" }, [
        ico(TILE_ICONS.send, 28),
        el("span", { class: "odk-planf-send-text" }, [
          el("span", { class: "odk-planf-send-title", text: "Send to aircraft" }),
          el("span", { class: "odk-planf-send-sub", text: "opens Fly with Upload ready" }),
        ]),
      ]);
      send.disabled = !plan;
      send.addEventListener("click", () => run("Send to Fly"));
      node.appendChild(send);

      const chosen = EXPORT_FORMATS.filter((f) => exportChoice.has(f.key));
      const exp = el("button", { type: "button", class: "odk-planf-export", "data-action": "Export Mission Files" }, [
        ico(TILE_ICONS.download, 24),
        el("span", { class: "odk-planf-export-title", text: "Export" }),
        el("span", { class: "odk-planf-export-sub", text: chosen.map((f) => f.ext).join(" · ") || "no format ticked" }),
      ]);
      exp.disabled = !plan || !chosen.length;
      exp.addEventListener("click", () => run("Export Mission Files", { formats: [...exportChoice] }));
      node.appendChild(exp);
      return node;
    },
  });
}

/* ------------------------------------------------------------- band */

function band() {
  return live({
    calls: ["mission_overview"],
    empty: "The mission could not be read.",
    render: ([overview]) => {
      take(overview);
      const plan = overview.plan;
      const fig = (label, value, unit, yellow = false) => el("div", { class: `odk-planf-fig${yellow ? " odk-planf-fig--action" : ""}` }, [
        el("span", { class: "odk-planf-fig-label", text: label }),
        value == null ? el("span", { class: "odk-planf-fig-missing", text: plan ? "not reported" : "no plan" })
          : el("span", { class: "odk-planf-fig-value" }, [value, unit ? el("span", { class: "odk-planf-fig-unit", text: ` ${unit}` }) : null]),
      ]);
      return el("div", { class: "odk-planf-band" }, [
        fig("WAYPOINTS", plan ? fmtInt(plan.waypoint_count) : null, "", true),
        fig("FLIGHT TIME", plan ? fmtNum(plan.duration_min, 1) : null, "min"),
        fig("DISTANCE", plan ? fmtInt(plan.distance_m) : null, "m"),
        fig("GROUND DETAIL", plan ? fmtNum(plan.gsd_cm, 2) : null, "cm/px"),
      ]);
    },
  });
}

/* ------------------------------------------------------------ render */

export async function render(regions /* { left, main } */, context) {
  await ensureStyles();
  shellRef = context.shell;
  const { left, main } = regions;
  if (state.ctl) { state.ctl.destroy(); state.ctl = null; }
  context.provide("missionOptions", () => missionOptions(state.overview));

  left.replaceChildren(sidePanel());
  const mapHost = el("div", { class: "odk-planf-map" });
  const legend = el("div", { class: "odk-planf-legend", hidden: "" });
  const edit = el("button", { type: "button", class: "odk-planf-edit", "data-action": "Edit Vertices",
    "aria-label": "Edit boundary", title: "Edit boundary" }, [ico(TILE_ICONS.edit, 26)]);
  edit.addEventListener("click", () => run("Edit Vertices"));
  main.replaceChildren(el("div", { class: "odk-planf-main" }, [mapHost, band()]));
  mapHost.append(legend, edit);

  const centre = state.overview && state.overview.aoi && state.overview.aoi.polygon && state.overview.aoi.polygon[0];
  const mounted = await mountProjectMap(mapHost, {
    screen: "plan",
    visibility: { cameras: true, mission: true, aoi: true, nofly: true },
    ...(centre ? { centre, zoom: 16 } : {}),
    notify: (text, kind) => context.shell.toast(text, kind),
    run: (key, payload) => context.shell.runAction(key, payload),
  });
  if (!mounted || !mounted.map) return;
  const ctl = attachPlanMap(mounted.map, mapHost, { notify: (text, kind) => context.shell.toast(text, kind) });
  ctl.setFootprints(false);
  state.ctl = ctl;
  mapHost.appendChild(overviewFeed((overview) => {
    take(overview);
    ctl.update(overview);
    const plan = overview.plan;
    const aoi = overview.aoi || {};
    const items = [];
    if (plan) items.push(["path", "Flight path", ""]);
    if (aoi.set) items.push(["boundary", "Boundary", `${fmtInt(aoi.area_m2)} m²`]);
    legend.replaceChildren(...items.map(([kind, text, value]) => el("div", { class: "odk-planf-legend-item" }, [
      el("span", { class: `odk-planf-swatch odk-planf-swatch--${kind}`, "aria-hidden": "true" }),
      el("span", { class: "odk-planf-legend-text", text }),
      value ? el("span", { class: "odk-planf-legend-value", text: value }) : null,
    ])));
    legend.hidden = !items.length;
    const type = plan ? templateLabel(state.templates, plan.template) : "no plan";
    context.setSubtitle(plan ? `${type.toLowerCase()} · ${aoi.vertices}-vertex boundary`
      : aoi.set ? `${aoi.vertices}-vertex boundary · not planned` : "no area drawn");
  }));
}
