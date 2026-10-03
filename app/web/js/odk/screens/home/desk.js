/* Home: desk renderer (SysHome artboard).

   Left: the projects on this machine, filter and order, then the machine's state
   (sync, storage, models, build) and New project. Centre: the open project -- its
   header, five summary tiles, the map, the reconstruction, the newest measurements and
   the report. Inspector: recent activity and notifications, then the exports.

   Every panel is a live() block over the Api (./model.js PANELS): it says "Reading…"
   until the answer arrives, re-reads on REFRESH_EVENT (any verb that changed the
   project), and says what is missing instead of filling the gap. Buttons that run a
   verb carry data-action; buttons that only move the interface carry data-control. */

import { el, button, chip, dot, popover, panelTabs, icon } from "../../components.js";
import { live } from "../../../workspace/live.js";
import { tryCall } from "../../../workspace/api.js";
import { mountProjectMap } from "../../map/furniture.js";
import { setLayersVisible } from "../../map/theme.js";
import * as M from "./model.js";

export const SCREEN = "home";

/* How many rows a list shows before "View all"; a layout choice, not data. */
const LIMITS = { activity: 8, notifications: 16, exports: 6, measurements: 4, runs: 2, tags: 8, mountFrames: 120, bbox: 4, framePadding: 16 };
const SORT_KEY = "odk.home.sort";
const TAG_KEY = "odk.review.tag";

/* Screen CSS lives beside the screen (docs/ui/ARCHITECTURE.md); loaded once. */
function ensureStyles() {
  if (document.querySelector("link[data-screen-css=home]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./home.css", import.meta.url).href;
  link.setAttribute("data-screen-css", "home");
  document.head.appendChild(link);
}

/* A few stroke icons the shared set does not carry; same 24-unit grid, 2 px stroke. */
const GLYPHS = {
  more: '<circle cx="5" cy="12" r="1.2"></circle><circle cx="12" cy="12" r="1.2"></circle><circle cx="19" cy="12" r="1.2"></circle>',
  filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"></path>',
  upload: '<path d="M12 16V4M6 10l6-6 6 6M4 20h16"></path>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"></path>',
};
function glyph(name, size = "14") {
  const span = document.createElement("span");
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="2" aria-hidden="true">${GLYPHS[name]}</svg>`;
  return span.firstChild;
}

function withGlyph(node, name) {
  node.prepend(glyph(name, "16"));
  return node;
}

/* Popovers live on <body>; each re-read of a panel builds a new one, so the one it
   replaces is removed rather than left hidden behind it. */
const POPOVERS = new Map();
function replacePopover(name, pop) {
  const old = POPOVERS.get(name);
  if (old && old !== pop) { old.close(); old.node.remove(); }
  POPOVERS.set(name, pop);
  return pop;
}

const missing = (text = "not reported") => el("span", { class: "odk-home-missing", text });
const label = (text) => el("span", { class: "odk-label", text });

/** A card: mono label head (with an aside), body. */
function card({ title, aside, where, className = "", body }) {
  return el("section", { class: `odk-home-card ${className}`.trim(), "aria-label": title, "data-where": where }, [
    el("header", { class: "odk-home-card-head" }, [
      el("span", { class: "odk-label odk-home-card-title", text: title }), aside || null,
    ]),
    body,
  ]);
}

export function render(regions, context) {
  ensureStyles();
  const { left, main, inspector } = regions;
  const shell = context.shell;

  /* A verb button: data-action, runs through the shell (prerequisites, toasts, refresh). */
  const act = (action, text, opts = {}) => button({
    label: text, action, variant: opts.variant || "secondary", iconName: opts.iconName, title: opts.title,
    onClick: (event) => { event.stopPropagation(); shell.runAction(action, opts.payload || null); },
  });
  /* An interface move: data-control="open:<screen>". */
  const go = (screen, text, opts = {}) => button({
    label: text, control: `open:${screen}`, variant: opts.variant || "ghost", iconName: opts.iconName,
    onClick: () => shell.open(screen),
  });

  left.replaceChildren(leftPanel({ shell, act, go }));
  main.replaceChildren(mainPanel({ shell, act, go }));
  if (inspector) inspector.replaceChildren(inspectorPanel({ shell, act }));
  context.setFooter(null);
}

/* =================================================================== left */

function leftPanel({ shell, act, go }) {
  let sort = "recent";
  try { sort = localStorage.getItem(SORT_KEY) || "recent"; } catch { /* private window */ }
  if (!M.PROJECT_ORDER[sort]) sort = "recent";
  let filter = "";
  let projects = [];
  const list = el("div", { class: "odk-home-projects", role: "listbox", "aria-label": "Projects" });
  const countNode = el("span", { class: "odk-label odk-home-count" });

  const draw = () => {
    const wanted = filter.trim().toLowerCase();
    const rows = projects
      .filter((p) => !wanted || String(p.name || "").toLowerCase().includes(wanted))
      .sort(M.PROJECT_ORDER[sort]);
    list.replaceChildren(...rows.map((p) => projectRow(p, shell)));
    if (!rows.length) {
      list.appendChild(el("p", { class: "odk-home-note",
        text: projects.length ? `No project name contains "${filter.trim()}".` : "No projects on this machine yet." }));
    }
  };

  const listing = live({
    calls: ["home_projects"],
    empty: "Could not read the projects on this machine.",
    render: ([answer]) => {
      projects = (answer && answer.projects) || [];
      const n = M.count(projects.length);
      countNode.textContent = `${n} local`;
      draw();
      return list;
    },
  });

  const search = el("input", { type: "search", placeholder: "Filter", "aria-label": "Filter projects" });
  search.addEventListener("input", () => { filter = search.value; draw(); });
  const tabs = el("div", { class: "odk-home-sort", role: "tablist", "aria-label": "Order projects by" },
    [["recent", "Recent"], ["name", "Name"], ["status", "Status"]].map(([id, text]) => {
      const tab = el("button", { type: "button", role: "tab", class: "odk-home-sort-tab", "data-control": `sort:${id}`,
        "aria-selected": id === sort ? "true" : "false", text });
      tab.addEventListener("click", () => {
        sort = id;
        try { localStorage.setItem(SORT_KEY, id); } catch { /* private window */ }
        tabs.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
        draw();
      });
      return tab;
    }));

  const statusRows = live({
    calls: ["project_storage", "get_state", "model_status"],
    empty: "Could not read this machine's state.",
    isEmpty: (r) => r.every((x) => x === null),
    render: ([storage, state, models]) => machineStatus({ storage, state, models, act, go }),
  });

  const folderButton = el("button", {
    type: "button", class: "odk-home-icon-btn", "data-action": "Open Project Folder",
    "aria-label": "Open project folder", title: "Open project folder",
  }, [icon("folder")]);
  folderButton.addEventListener("click", () => shell.runAction("Open Project Folder"));

  return el("div", { class: "odk-home-left", "data-where": "home.projects" }, [
    el("div", { class: "odk-home-left-head" }, [label("Projects"), countNode]),
    el("div", { class: "odk-home-filter" }, [
      el("label", { class: "odk-home-filter-box" }, [glyph("filter", "12"), search]),
    ]),
    tabs,
    listing,
    el("div", { class: "odk-home-left-foot" }, [
      el("div", { "data-where": "home.projects.status" }, [statusRows]),
      el("div", { class: "odk-home-left-actions", "data-where": "home.projects.footer" }, [
        act("New Project", "New project", { iconName: "plus" }),
        folderButton,
      ]),
    ]),
  ]);
}

function projectRow(p, shell) {
  const photos = M.count(p.photos);
  const surveyed = p.survey_first ? String(p.survey_first).slice(0, 10) : null;
  const facts = [photos !== null ? `${photos} photos` : "no photos", surveyed].filter(Boolean).join(" · ");
  const words = M.statusLine(p);
  const node = el("button", {
    type: "button", role: "option", class: "odk-home-project", "data-control": `project:${p.id}`,
    "aria-selected": p.active ? "true" : "false", title: p.root_dir || "",
  }, [
    dot(p.processed ? "ok" : "idle"),
    el("span", { class: "odk-home-project-body" }, [
      el("span", { class: "odk-home-project-name" }, [
        el("span", { class: "odk-home-project-title", text: String(p.name || "Untitled") }),
        el("span", { class: "odk-home-project-id", text: `#${p.id}` }),
      ]),
      el("span", { class: "odk-home-project-facts", text: facts }),
      el("span", { class: `odk-home-project-status${p.processed ? " is-ok" : ""}`, text: words || "" }),
      p.folder_exists === false ? el("span", { class: "odk-home-project-warn", text: "folder missing" }) : null,
    ]),
  ]);
  node.addEventListener("click", () => { if (!p.active) shell.openProject(p.id, p.name); });
  return node;
}

function machineStatus({ storage, state, models, act, go }) {
  const project = (state && state.project) || {};
  const rows = el("dl", { class: "odk-home-status" });
  const row = (name, value, extra, title) => {
    rows.append(el("dt", { text: name }), el("dd", { ...(title ? { title } : {}) }, [value, extra || null]));
  };
  row("SYNC", el("span", { text: project.sync_status || "not reported" }),
    act("Sync", "Check", { variant: "ghost", title: "Count what is on this machine against what the service has" }));
  let used = null;
  let tip = "";
  if (storage) {
    const outside = Number(storage.datasets_outside_bytes || 0);
    used = M.bytes(Number(storage.project_bytes || 0) + outside);
    const free = storage.disk ? M.bytes(storage.disk.free) : null;
    tip = [`Project folder ${M.bytes(storage.project_bytes)} in ${M.count(storage.project_files)} files`,
      outside ? `photos kept outside it ${M.bytes(outside)}` : "",
      free ? `${free} free on ${storage.disk.path}` : ""].filter(Boolean).join(" · ");
    used = used && free ? `${used} · ${free} free` : used;
  }
  row("STORAGE", used ? el("span", { text: used }) : missing(), null, tip);
  const list = (models && models.models) || [];
  const present = list.filter((m) => m.exists).length;
  row("MODELS", models ? el("span", { text: `${M.count(present)} of ${M.count(list.length)} installed` }) : missing());
  const build = go("settings", (state && state.version) || "not reported", { variant: "ghost" });
  build.setAttribute("title", "System status and model verification: Settings");
  row("BUILD", build);
  return rows;
}

/* =================================================================== main */

function mainPanel({ shell, act, go }) {
  const header = live({
    calls: ["get_project", "dataset_summary", "get_project_crs", "list_annotation_tags"],
    empty: "No project is open.",
    isEmpty: ([project]) => !project,
    render: ([project, datasets, crs, tags]) => projectHeader({ shell, act, go, project, datasets, crs, tags }),
  });
  const tiles = live({
    calls: ["dataset_summary", "get_state", "list_mission_versions", "find_annotations", "pipeline_reports"],
    empty: "Could not read the project summary.",
    isEmpty: ([, state]) => !state,
    render: ([datasets, state, versions, findings, runs]) =>
      summaryTiles({ shell, act, datasets, state, versions, findings, runs }),
  });

  const mapHost = el("div", { class: "odk-home-map", "data-where": "home.map.canvas" });
  const mapTitle = live({
    calls: ["reconstruction_summary"],
    empty: "",
    isEmpty: () => false,
    render: ([recon]) => {
      const run = recon && (recon.runs || []).find((r) => r.orthomosaic);
      const gsd = run ? M.fixed(run.ortho_gsd_m, 3) : null;
      return el("span", { class: "odk-label odk-home-card-title",
        text: gsd ? `Orthomosaic · ${gsd} m/px` : "Project map" });
    },
  });
  const mapCard = el("section", { class: "odk-home-card odk-home-card--map", "aria-label": "Map", "data-where": "home.map" }, [
    el("header", { class: "odk-home-card-head" }, [mapTitle, go("analyse", "Open map", { variant: "ghost" })]),
    mapHost,
  ]);
  mountMapWhenPlaced(mapHost, shell);

  const recon = live({
    calls: ["reconstruction_summary"],
    empty: "Could not read this project's reconstructions.",
    render: ([summary]) => reconstructionCard({ summary, go }),
  });
  const measures = live({
    calls: ["list_measurements"],
    empty: "Could not read this project's measurements.",
    render: ([listing]) => measurementsCard({ listing, go }),
  });
  const report = live({
    calls: ["list_reports", "report_files", "report_readiness"],
    empty: "Could not read this project's reports.",
    isEmpty: ([listing, , readiness]) => !listing && !readiness,
    render: ([listing, files, readiness]) => reportCard({ listing, files, readiness, act }),
  });

  return el("div", { class: "odk-home-main", "data-where": "home.main" }, [
    el("div", { "data-where": "home.header" }, [header]),
    el("div", { "data-where": "home.tiles" }, [tiles]),
    el("div", { class: "odk-home-grid" }, [
      mapCard,
      el("div", { class: "odk-home-slot", "data-where": "home.reconstruction" }, [recon]),
      el("div", { class: "odk-home-slot", "data-where": "home.measurements" }, [measures]),
      el("div", { class: "odk-home-slot", "data-where": "home.report" }, [report]),
    ]),
  ]);
}

/* The map is mounted once the card is in the document and has a size; a refresh
   re-reads the panels around it but never remounts it. */
function mountMapWhenPlaced(host, shell) {
  let tries = 0;
  const attempt = () => {
    if (!host.isConnected || host.clientWidth === 0) {
      if (tries++ < LIMITS.mountFrames) requestAnimationFrame(attempt);
      return;
    }
    // The card is the orthomosaic's preview (the artboard): when the project has one,
    // the other rasters (DSM, DTM, hillshade) start off on this map only. A view
    // choice, never saved; Analyse and the Layers popover show them all.
    // It opens framed on the orthomosaic's own extent (else on everything the project
    // drew), once; after that the operator's pan and zoom are left alone.
    let framed = false;
    host.addEventListener("odk:project", (event) => {
      const drawn = (event.detail && event.detail.layers) || [];
      const map = host.__odkMap;
      if (!map) return;
      const ortho = drawn.find((d) => d.layer && d.layer.kind === "raster" && /ortho/i.test(d.layer.name || ""));
      if (ortho) {
        for (const d of drawn) {
          if (d !== ortho && d.layer && d.layer.kind === "raster") {
            setLayersVisible(map, (d.mapLayers || []).filter((id) => map.getLayer(id)), false);
          }
        }
      }
      if (framed) return;
      const box = (ortho && ((ortho.tiles && ortho.tiles.bounds) || ortho.layer.bounds_lonlat)) || event.detail.bounds;
      if (Array.isArray(box) && box.length === LIMITS.bbox && box.every(Number.isFinite)) {
        framed = true;
        map.fitBounds([[box[0], box[1]], [box[2], box[3]]], { padding: LIMITS.framePadding, duration: 0 });
      }
    });
    mountProjectMap(host, {
      screen: "home",
      visibility: { mission: false },
      fitOnLoad: false,
      run: (key, payload) => shell.runAction(key, payload),
      notify: (text, kind) => shell.toast(text, kind),
    }).then((mounted) => { if (mounted) host.__odkMap = mounted.map; })
      .catch((error) => shell.toast(`Map: ${error.message || error}`, "warn"));
  };
  requestAnimationFrame(attempt);
}

function projectHeader({ shell, act, go, project, datasets, crs, tags }) {
  const p = (project && project.project) || {};
  const active = datasets && datasets.active;
  const centre = active && active.center_lonlat;
  const place = centre ? M.latLon(centre[0], centre[1]) : null;
  const epsg = crs && crs.epsg;
  const crsText = epsg ? [`EPSG:${epsg}`, M.utmZone(epsg)].filter(Boolean).join(" · ")
    : crs ? `CRS ${crs.name || "automatic"}` : null;
  const created = M.utcStamp(p.created_at);

  const more = el("button", {
    type: "button", class: "odk-home-icon-btn odk-home-more", "data-control": "project-more",
    "aria-label": "More project actions", title: "More project actions",
  }, [glyph("more", "16")]);
  const pop = popover({
    anchor: more,
    title: "Project",
    items: [
      { label: "Open another project…", action: "Open", onClick: () => { pop.close(); shell.runAction("Open"); } },
      { label: "Share read-only link…", action: "Share", onClick: () => { pop.close(); shell.runAction("Share"); } },
      { label: "Revoke share links", action: "Revoke Links", onClick: () => { pop.close(); shell.runAction("Revoke Links"); } },
      { label: "Open project folder", action: "Open Project Folder",
        onClick: () => { pop.close(); shell.runAction("Open Project Folder"); } },
    ],
  });
  pop.node.setAttribute("data-where", "home.header.more");
  replacePopover("more", pop);

  const tagList = ((tags && tags.tags) || []).slice(0, LIMITS.tags);
  const tagRow = tagList.length ? el("div", { class: "odk-home-tags", "data-where": "home.header.tags" },
    tagList.map((t) => {
      const name = typeof t === "string" ? t : (t.tag || t.name || "");
      const n = typeof t === "object" ? (t.count ?? null) : null;
      const b = el("button", { type: "button", class: "odk-chip odk-home-tag", "data-control": `review-tag:${name}`,
        title: `Review the findings tagged ${name}`, text: n !== null ? `${name} · ${M.count(n)}` : name });
      b.addEventListener("click", () => {
        try { localStorage.setItem(TAG_KEY, name); } catch { /* private window */ }
        shell.open("review");
      });
      return b;
    })) : null;

  return el("div", { class: "odk-home-header" }, [
    el("div", { class: "odk-home-title" }, [
      el("span", { class: "odk-label", text: [`Project #${p.id ?? ""}`, created ? `Created ${created}` : null]
        .filter(Boolean).join(" · ") }),
      el("div", { class: "odk-home-name" }, [el("h1", { text: p.name || "Untitled project" }), more]),
      el("span", { class: "odk-home-place", text: [place, crsText].filter(Boolean).join(" · ")
        || "No position yet: import geotagged photos" }),
      tagRow,
    ]),
    el("div", { class: "odk-home-header-actions" }, [
      go("plan", "New mission", { variant: "primary", iconName: "plus" }),
      withGlyph(button({ label: "Import photos", action: "Import", variant: "secondary",
        onClick: () => shell.runAction("Import") }), "upload"),
      go("analyse", "Open analysis", { variant: "secondary", iconName: "measure" }),
    ]),
  ]);
}

function tile({ name, value, unit, sub, subTone, where, control, onClick, title }) {
  const valueNode = value === null || value === undefined
    ? missing() : el("span", { class: "odk-home-tile-value" }, [String(value),
      unit ? el("span", { class: "odk-home-tile-unit", text: unit }) : null]);
  const attrs = { class: "odk-home-tile", ...(where ? { "data-where": where } : {}), ...(title ? { title } : {}) };
  const children = [
    el("span", { class: "odk-label", text: name }),
    valueNode,
    el("span", { class: `odk-home-tile-sub${subTone ? ` is-${subTone}` : ""}`, text: sub || "" }),
  ];
  if (control) {
    const b = el("button", { type: "button", ...attrs, "data-control": control }, children);
    b.classList.add("is-button");
    if (onClick) b.addEventListener("click", onClick);
    return b;
  }
  return el("div", attrs, children);
}

function summaryTiles({ shell, act, datasets, state, versions, findings, runs }) {
  const active = datasets && datasets.active;
  const camera = active && active.camera;
  const photos = tile({
    name: "Photos",
    value: active ? M.count(active.image_count) : null,
    unit: active && active.geotagged_count !== null && active.geotagged_count !== undefined
      ? `/ ${M.count(active.geotagged_count)} geotagged` : "",
    sub: active ? (camera || "camera not in EXIF") : "no dataset selected",
    subTone: active ? "" : "warn",
    where: "home.tiles.photos",
    control: "datasets",
    title: active ? active.folder : "Choose or import a dataset",
  });
  datasetPopover(photos, datasets, shell, act);

  const when = M.survey(active && active.capture);
  const surveyTile = tile({
    name: "Survey date",
    value: when ? when.date : null,
    sub: when ? when.window : active ? "no EXIF capture time" : "",
    title: when ? `${M.count(when.dated)} of ${M.count(when.images)} photos dated; ${when.clock}, from EXIF DateTimeOriginal` : "",
  });
  surveyTile.querySelector(".odk-home-tile-value")?.classList.add("is-mono");

  const aoi = (state && state.aoi) || [];
  const boundary = tile({ name: "Site boundary", value: null, sub: aoi.length >= M.MIN_POLYGON ? "" : "draw one in Plan",
    subTone: aoi.length >= M.MIN_POLYGON ? "" : "warn" });
  if (aoi.length < M.MIN_POLYGON) boundary.querySelector(".odk-home-missing").textContent = "none";
  if (aoi.length >= M.MIN_POLYGON) {
    boundary.querySelector(".odk-home-missing").textContent = "Measuring…";
    tryCall("measure_geometry", aoi, "area").then((m) => {
      const value = boundary.querySelector(".odk-home-missing");
      const sub = boundary.querySelector(".odk-home-tile-sub");
      if (!m) { if (value) value.textContent = "not reported"; return; }
      const area = el("span", { class: "odk-home-tile-value" }, [M.count(Math.round(m.area_m2)),
        el("span", { class: "odk-home-tile-unit", text: "m²" })]);
      value?.replaceWith(area);
      sub.textContent = `${M.count(m.vertices)} vertices · ${M.fixed(m.perimeter_m, 0)} m`;
      boundary.title = m.method || "";
    });
  }

  const plan = M.mission(state, versions);
  const saved = plan.saved;
  const missionTile = tile({
    name: saved && saved.template ? `${String(saved.template).replace(/_/g, " ")} mission` : "Mission",
    value: saved ? M.count(saved.waypoints) : null,
    unit: saved ? "wp" : "",
    sub: saved ? [saved.altitude_m !== null ? `${M.fixed(saved.altitude_m, 0)} m` : null,
      saved.gsd_cm !== null ? `${M.fixed(saved.gsd_cm, 2)} cm GSD` : null].filter(Boolean).join(" · ")
      : plan.plannedUnsaved ? "planned, not saved" : "none saved",
    subTone: saved ? "" : "warn",
    title: saved ? `${saved.name} v${saved.version}` : "",
  });
  if (!saved) missionTile.querySelector(".odk-home-missing").textContent = "none";

  const found = findings ? findings.count ?? (findings.annotations || []).length : null;
  const detectorRuns = runs ? (runs.reports || []).length : null;
  const findingsTile = tile({
    name: "Findings",
    value: found !== null ? M.count(found) : null,
    sub: detectorRuns === null ? "" : detectorRuns ? `${M.count(detectorRuns)} detector runs` : "no detector run",
    subTone: detectorRuns === 0 ? "warn" : "",
  });

  return el("div", { class: "odk-home-tiles", role: "group", "aria-label": "Project summary" },
    [photos, surveyTile, boundary, missionTile, findingsTile]);
}

function datasetPopover(anchor, datasets, shell, act) {
  const rows = (datasets && datasets.datasets) || [];
  const body = el("div", { class: "odk-home-datasets", "data-where": "home.tiles.datasets" });
  if (!rows.length) body.appendChild(el("p", { class: "odk-home-note", text: "No photos imported into this project yet." }));
  for (const d of rows) {
    const use = act("Select Dataset", d.active ? "In use" : "Use", { variant: "ghost", payload: { path: d.path } });
    if (d.active) { use.disabled = true; use.classList.add("is-active"); use.title = "The active dataset"; }
    else if (!d.exists) { use.disabled = true; use.title = "The folder is not on disk"; }
    body.appendChild(el("div", { class: "odk-home-dataset", title: d.path }, [
      el("span", { class: "odk-home-dataset-name", text: d.name }),
      el("span", { class: "odk-home-dataset-facts", text: [
        d.image_count !== null && d.image_count !== undefined ? `${M.count(d.image_count)} photos` : null,
        d.exists ? null : "folder missing"].filter(Boolean).join(" · ") }),
      use,
    ]));
  }
  body.appendChild(button({ label: "Import photos…", action: "Import", variant: "secondary",
    onClick: () => { pop.close(); shell.runAction("Import"); } }));
  replacePopover("datasets", popover({ anchor, title: "Datasets", content: body }));
  const pop = POPOVERS.get("datasets");
}

/* ------------------------------------------------------------ reconstruction */

function reconstructionCard({ summary, go }) {
  const runs = ((summary && summary.runs) || []).slice(0, LIMITS.runs).reverse();
  const running = (summary && summary.running) || [];
  const newest = runs[runs.length - 1];
  const engine = newest && newest.engine ? String(newest.engine).toUpperCase() : null;
  let badge = null;
  if (running.length) badge = chip("RUNNING", "warn");
  else if (newest && newest.complete) {
    badge = chip(newest.dense === true ? "COMPLETE · DENSE" : newest.dense === false ? "COMPLETE · SPARSE" : "COMPLETE", "ok");
  }
  if (!runs.length) {
    return card({
      title: "Reconstruction", where: "home.reconstruction", className: "odk-home-card--recon", aside: badge,
      body: el("div", { class: "odk-home-empty" }, [
        el("p", { text: running.length ? "A reconstruction is running. Its figures appear here when it finishes."
          : "No reconstruction in this project yet. Process turns the photos into an orthomosaic, DSM and point cloud." }),
        go("process", "Open Process", { variant: "secondary" }),
      ]),
    });
  }
  const head = el("tr", {}, [el("th", { scope: "col", text: "Metric" }),
    ...runs.map((r, i) => el("th", { scope: "col", class: `num${i === runs.length - 1 ? " is-newest" : ""}`,
      title: r.directory, text: (r.profile || r.name || "run").replace(/_/g, " ") }))]);
  const cell = (r, value) => el("td", { class: "num", title: r.sources ? Object.entries(r.sources)
    .map(([k, v]) => `${k}: ${v}`).join("\n") : "" }, [value === null || value === undefined ? missing() : value]);
  const rows = [
    ["Registered images", (r) => (r.registered_images !== null && r.registered_images !== undefined
      ? `${M.count(r.registered_images)} / ${M.count(r.total_images) ?? "?"}` : null)],
    ["Sparse points", (r) => M.count(r.sparse_points)],
    ["Reprojection error", (r) => (M.fixed(r.reprojection_error_px, 3) ? `${M.fixed(r.reprojection_error_px, 3)} px` : null)],
    ["Georef RMSE", (r) => (M.fixed(r.georef_rmse_m, 2) ? `${M.fixed(r.georef_rmse_m, 2)} m` : null)],
    ["Ortho / DSM GSD", (r) => (r.ortho_gsd_m || r.dsm_gsd_m
      ? `${M.fixed(r.ortho_gsd_m, 3) ?? "—"} / ${M.fixed(r.dsm_gsd_m, 2) ?? "—"} m` : null)],
    ["Runtime", (r) => (M.fixed(r.runtime_s, 1) ? `${M.fixed(r.runtime_s, 1)} s` : null)],
  ];
  const table = el("table", { class: "odk-table odk-home-recon-table" }, [
    el("thead", {}, [head]),
    el("tbody", {}, rows.map(([name, fn]) => el("tr", {}, [el("td", { text: name }), ...runs.map((r) => cell(r, fn(r)))]))),
  ]);
  const reason = newest && newest.dense === false && newest.dense_reason ? String(newest.dense_reason) : null;
  const note = reason ? el("span", { class: "odk-home-recon-note", title: reason }, [dot("warn"),
    el("span", { text: `Dense MVS not run: ${reason.split(":")[0]}` })]) : el("span");
  return card({
    title: `Reconstruction${engine ? ` · ${engine}` : ""}`, where: "home.reconstruction",
    className: "odk-home-card--recon", aside: badge,
    body: el("div", { class: "odk-home-recon" }, [table,
      el("footer", { class: "odk-home-card-foot" }, [note, go("process", "View log", { variant: "ghost" })])]),
  });
}

/* -------------------------------------------------------------- measurements */

const MEASURE_NAMES = {
  distance: "Distance", area: "Area", polygon_area: "Area", perimeter: "Perimeter", volume: "Volume",
  angle: "Angle", crack_length: "Crack length", slope: "Slope", elevation_profile: "Profile",
};

function measurementsCard({ listing, go }) {
  const rows = (listing && listing.measurements) || [];
  const newest = rows.slice(0, LIMITS.measurements);
  const titled = newest.find((m) => m.label);
  const aside = go("analyse", "Open in Analyse", { variant: "ghost" });
  if (!rows.length) {
    return card({ title: "Measurements", aside, where: "home.measurements",
      body: el("div", { class: "odk-home-empty" }, [el("p", {
        text: "No measurements yet. Distance, area and volume measured in Analyse are listed here." })]) });
  }
  const body = el("div", { class: "odk-home-measures" }, newest.map((m) => {
    const value = M.fixed(m.value, m.unit === "m3" || m.unit === "m2" ? 1 : 2);
    return el("div", { class: "odk-home-measure", title: [m.label, m.created_at].filter(Boolean).join(" · ") }, [
      el("span", { text: MEASURE_NAMES[m.measurement_type] || String(m.measurement_type || "").replace(/_/g, " ") }),
      el("span", { class: "odk-home-measure-value" }, [value ?? "not reported",
        el("span", { class: "odk-home-tile-unit", text: M.unitLabel(m.unit) })]),
    ]);
  }));
  const sources = [...new Set(rows.map((m) => m.source_type).filter(Boolean))];
  const chips = el("div", { class: "odk-home-chips" }, [
    chip(`${M.count(rows.length)} recorded`),
    ...sources.map((s) => chip(`on ${s}`)),
  ]);
  return card({
    title: `Measurements${titled ? ` · ${titled.label}` : ""}`, aside, where: "home.measurements",
    body: el("div", { class: "odk-home-measures-wrap" }, [body, chips]),
  });
}

/* -------------------------------------------------------------------- report */

function reportCard({ listing, files, readiness, act }) {
  const reports = (listing && listing.reports) || [];
  const latest = files && files.report ? files.report : reports.find(Boolean) || null;
  const ready = readiness ? readiness.ready === true : null;
  const badge = ready === null ? null : ready ? chip("READY", "ok") : chip("NOT READY", "warn");
  if (!latest) {
    const missingList = (readiness && readiness.missing) || [];
    return card({
      title: "Report", aside: badge, where: "home.report",
      body: el("div", { class: "odk-home-empty" }, [
        el("p", { text: ready ? "No report written yet. Everything a report needs is in the project."
          : "No report written yet." }),
        missingList.length ? el("ul", { class: "odk-home-missing-list" }, missingList.map((m) => el("li", { text: m }))) : null,
        act("Generate Report", "Generate report", { variant: ready ? "primary" : "secondary" }),
      ]),
    });
  }
  const formats = (files && files.formats) || [];
  const created = M.parseUtc(latest.created_at);
  const pages = files && files.pages;
  const facts = [pages ? `${M.count(pages)} pages` : null,
    created ? `${M.localDate(created)} ${M.localTime(created)}` : null].filter(Boolean).join(" · ");
  const findingsNote = latest.findings === 0
    ? el("p", { class: "odk-home-report-note" }, [dot("warn"),
      el("span", { text: "Findings (0). Not the same as a clean inspection." })])
    : latest.findings ? el("p", { class: "odk-home-report-note", text: `Findings (${M.count(latest.findings)}).` }) : null;
  return card({
    title: `Report · ${String(latest.report_type || "standard").replace(/_/g, " ")}`, aside: badge, where: "home.report",
    body: el("div", { class: "odk-home-report" }, [
      el("h2", { text: latest.title || "Report" }),
      el("span", { class: "odk-home-report-facts", text: facts || "date not recorded" }),
      el("div", { class: "odk-home-chips" }, formats.map((f) => {
        const size = f.exists ? M.bytes(f.bytes) : "missing";
        return chip(`${f.kind.toUpperCase()} ${size}`, f.exists ? "" : "danger");
      })),
      findingsNote,
      (latest.warnings || []).length ? el("p", { class: "odk-home-report-warn", title: latest.warnings.join(" "),
        text: latest.warnings.join(" ") }) : null,
      el("div", { class: "odk-home-report-actions" }, [
        act("Open Report", "Open report", { variant: "secondary" }),
        act("Generate Report", "Regenerate", { variant: "ghost" }),
      ]),
    ]),
  });
}

/* =============================================================== inspector */

function inspectorPanel({ shell, act }) {
  let tab = "activity";
  const activityHost = el("div", { class: "odk-home-feed" });
  const noticeHost = el("div", { class: "odk-home-feed", hidden: "" });

  const activityPanel = live({
    calls: ["audit_log", "dataset_summary"],
    empty: "No activity recorded for this project.",
    isEmpty: ([log]) => !log,
    render: ([log, datasets]) => activityList({ log, datasets }),
  });
  activityHost.appendChild(activityPanel);
  activityHost.appendChild(el("div", { class: "odk-home-feed-foot" }, [
    act("View Activity", "View all activity", { variant: "ghost" })]));

  const noticePanel = live({
    calls: ["list_notifications"],
    empty: "Could not read this project's notifications.",
    render: ([answer]) => notificationList({ answer, act, setCount }),
  });
  noticeHost.appendChild(noticePanel);

  const tabs = panelTabs([{ id: "activity", label: "Activity" }, { id: "notifications", label: "Notifications" }], {
    active: tab, label: "Recent activity",
    onSelect: (id) => { tab = id; activityHost.hidden = id !== "activity"; noticeHost.hidden = id !== "notifications"; },
  });
  function setCount(unread) {
    const t = tabs.querySelector('[data-control="panel-tab:notifications"]');
    if (!t) return;
    t.querySelector(".odk-num")?.remove();
    if (unread) t.appendChild(el("span", { class: "odk-num", text: M.count(unread) }));
    markHomeTab(unread);
  }

  const now = new Date();
  const exportsPanel = live({
    calls: ["list_exports"],
    empty: "Could not read this project's exports.",
    render: ([answer]) => exportsList({ answer, act, shell }),
  });

  return el("div", { class: "odk-home-inspector" }, [
    el("section", { class: "odk-home-activity", "aria-label": "Recent activity", "data-where": "home.activity" }, [
      el("div", { class: "odk-home-section-head" }, [label("Recent activity"),
        el("span", { class: "odk-label odk-home-zone", text: `${M.localDate(now)} · ${M.zoneLabel(now)}` })]),
      el("div", { class: "odk-home-tabs" }, [tabs]),
      activityHost,
      el("div", { "data-where": "home.notifications" }, [noticeHost]),
    ]),
    el("section", { class: "odk-home-exports", "aria-label": "Exports", "data-where": "home.exports" }, [exportsPanel]),
  ]);
}

/* The unread dot on the Home tab: drawn from list_notifications, never assumed. */
function markHomeTab(unread) {
  const tab = document.querySelector('[data-control="tab:home"]');
  if (!tab) return;
  tab.querySelector(".odk-home-unread")?.remove();
  if (unread) {
    tab.appendChild(el("span", { class: "odk-home-unread", role: "img", "aria-label": `${M.count(unread)} unread notifications` }));
  }
}

function activityList({ log, datasets }) {
  const events = ((log && log.events) || []).map(M.activity);
  const capture = datasets && datasets.active && datasets.active.capture;
  const rows = events.slice(0, LIMITS.activity);
  const newest = rows.find((e) => e.when);
  const day = newest ? M.localDate(newest.when) : null;
  const list = el("ol", { class: "odk-home-events" }, rows.map((e) => el("li", { class: "odk-home-event" }, [
    el("time", { datetime: e.when ? e.when.toISOString() : "", title: e.when ? e.when.toString() : "",
      text: e.when ? (M.localDate(e.when) === day ? M.localTime(e.when) : M.localDate(e.when).slice(5)) : "—" }),
    dot(e.tone),
    el("span", { class: "odk-home-event-body" }, [
      el("span", { class: "odk-home-event-title", text: e.title }),
      e.detail ? el("span", { class: "odk-home-event-detail", text: e.detail }) : null,
    ]),
  ])));
  // The survey itself, from the photographs' clocks: a fact about the project the audit
  // log cannot hold, because the flight happened before the project existed.
  const when = M.survey(capture);
  if (when) {
    list.appendChild(el("li", { class: "odk-home-event is-capture", title: "From the photographs' EXIF clocks" }, [
      el("time", { text: capture.first.slice(0, 4) }), dot("idle"),
      el("span", { class: "odk-home-event-body" }, [
        el("span", { class: "odk-home-event-title", text: `Photos captured · ${capture.first.slice(5, 10)}` }),
        el("span", { class: "odk-home-event-detail", text: `${when.window} · ${M.count(when.images)} photos` }),
      ]),
    ]));
  }
  if (!rows.length && !when) return el("p", { class: "odk-home-note", text: "No activity recorded for this project yet." });
  return list;
}

function notificationList({ answer, act, setCount }) {
  const items = (answer && answer.notifications) || [];
  const unread = answer ? Number(answer.unread || 0) : 0;
  setTimeout(() => setCount(unread), 0);
  const list = el("ol", { class: "odk-home-events" });
  for (const n of items.slice().reverse().slice(0, LIMITS.notifications)) {
    const when = M.parseUtc(n.created_utc);
    list.appendChild(el("li", { class: `odk-home-event${n.read ? " is-read" : ""}`, "data-where": "home.notifications.row" }, [
      el("time", { text: when ? M.localTime(when) : "—", title: when ? when.toString() : "" }),
      dot(n.level === "error" ? "danger" : n.level === "warning" ? "warn" : n.read ? "idle" : "running"),
      el("span", { class: "odk-home-event-body" }, [
        el("span", { class: "odk-home-event-title", text: n.title || "Notification" }),
        n.detail ? el("span", { class: "odk-home-event-detail", text: n.detail }) : null,
      ]),
      n.read ? null : act("Mark Notification Read", "Mark read", { variant: "ghost", payload: { id: n.id } }),
    ]));
  }
  return el("div", {}, [
    items.length ? list : el("p", { class: "odk-home-note", text: "No notifications. Finished jobs and exports are announced here." }),
    el("div", { class: "odk-home-feed-foot" }, [
      act("Mark All Notifications Read", "Mark all read", { variant: "ghost" }),
      act("Clear Notifications", "Clear", { variant: "ghost" }),
    ]),
  ]);
}

function exportsList({ answer, act, shell }) {
  const files = (answer && answer.files) || [];
  const newest = files.find(Boolean);
  const head = el("div", { class: "odk-home-section-head" }, [
    label(`Exports${answer ? ` · ${M.count(answer.count)} files` : ""}`),
    newest ? act("Open Exports Folder", "Open folder", { variant: "ghost", payload: { folder: newest.folder } }) : null,
  ]);
  if (!files.length) {
    return el("div", {}, [head, el("p", { class: "odk-home-note",
      text: "Nothing exported yet. Missions, flight logs, fleet records and audit logs written from this project are listed here." })]);
  }
  const list = el("ul", { class: "odk-home-export-list", "data-where": "home.exports.row" }, files.slice(0, LIMITS.exports).map((f) => {
    const b = el("button", { type: "button", class: "odk-home-export", "data-action": "Open Export", title: f.path }, [
      el("span", { class: "odk-home-export-name", text: `${f.extension || f.name} · ${f.format}` }),
      el("span", { class: "odk-home-export-size", text: M.exactBytes(f.bytes) || "not reported" }),
    ]);
    b.addEventListener("click", () => shell.runAction("Open Export", { path: f.path }));
    return el("li", {}, [b]);
  }));
  return el("div", {}, [head, list]);
}
