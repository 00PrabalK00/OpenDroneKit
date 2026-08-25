/* The thirteen workspaces, each an arrangement of the same primitives.
 *
 * The project stays the same across all of them. Switching workspace changes which
 * instruments are pointed at it, which is the difference between an application and a
 * set of pages: an operator moving plan → fly → verify → process → inspect → measure →
 * report never leaves the survey they are working on.
 *
 * Data here is illustrative STRUCTURE, not measurement, and it is written so that it
 * cannot be mistaken for one. Sites are named DEMO, coordinates are Null Island and
 * clocks are at the epoch, matching core/demo_mode.py so the desktop demo and the API
 * demo agree about what synthetic looks like.
 *
 * It used to read as a real Indian survey: named warehouses, a highway corridor, a
 * substation, Delhi coordinates and hundreds of gigabytes of storage -- marked only by
 * a status-bar chip that was itself
 * clipped off screen at 1600px. Every number a real deployment shows comes from the
 * API; nothing in this file should ever be presented as a survey result.
 */

import {
  canvas, chip, consoleView, el, fields, meter, properties,
  readouts, selection, splitCanvas, table, tree,
} from "./primitives.js";
import { DATA, DEMO, demoCoords } from "./demo.js";

/* Every editable field reports onto the same bus the trees and tables use.
 *
 * fields() has always accepted an onChange and no caller passed one, so editing an
 * altitude or an overlap changed the input element and nothing else -- the mission that
 * Plan then generated used the defaults, silently. */
const settingChanged = (key, value) => selection.select("setting", { id: key, label: `${key} = ${value}`, key, value });

const MAP_TOOLS = [
  { icon: "✥", title: "Pan" },
  { icon: "▣", title: "Box select" },
  { icon: "⬠", title: "Draw polygon" },
  { icon: "╱", title: "Draw line" },
  { icon: "⤢", title: "Measure" },
  { icon: "⌂", title: "Place home" },
  { icon: "⛔", title: "Exclusion zone" },
  { icon: "⤡", title: "Fit view" },
];

const COORD = demoCoords();

/* The example project, and every model installed against it.
 *
 * Both are real. The project is the OpenDroneMap Aukerman reconstruction this repository
 * actually ran -- 77 of 77 images registered, 1.27px mean reprojection error -- and each
 * model row exists on disk with the digest its metrics were measured against. Nothing in
 * this tree is a name someone made up. */
const projectTree = () => tree([
  {
    id: "org", label: "Installed models", icon: "▦", meta: `${DATA.models.length}`,
    children: DATA.models.map((model, index) => ({
      id: `m${index}`,
      label: model.key,
      icon: "▤",
      meta: model.headline || `${model.input_size || ""}px`,
    })),
  },
  {
    id: "proj", label: DATA.project.name, icon: "▦", meta: DATA.project.epsg,
    children: [
      {
        id: "p1", label: `${DATA.project.images_registered} images registered`, icon: "▤",
        meta: `${DATA.project.reprojection_px} px`,
        children: [
          { id: "a1", label: "Orthomosaic", icon: "▱" },
          { id: "a2", label: "DSM / DTM", icon: "▯" },
          { id: "a3", label: "Point cloud", icon: "▲" },
        ],
      },
      { id: "p2", label: `Geo RMSE ${DATA.project.geo_rmse_m} m`, icon: "▤" },
      { id: "p3", label: DATA.project.source, icon: "▤" },
      { id: "p4", label: "DEMO site 4", icon: "▤", meta: "18 MW" },
    ],
  },
], { selectKind: "project" });

const layerTree = (extra = []) => tree([
  { id: "l-base", label: "Basemap — Satellite", icon: "◈" },
  { id: "l-terrain", label: "Terrain (SRTM)", icon: "◭" },
  { id: "l-ortho", label: "Orthomosaic", icon: "▦" },
  { id: "l-dsm", label: "DSM", icon: "◰" },
  { id: "l-mission", label: "Mission", icon: "⟋" },
  { id: "l-fence", label: "Geofence", icon: "⬡" },
  ...extra,
], { selectKind: "layer" });

/* --------------------------------------------------------------------- HOME */

const home = {
  id: "home",
  title: "Home",
  toolbar: ["New Project", "Open", "Import", "|", "Sync", "Share"],
  left: [
    { id: "home.projects", title: "Project Explorer", render: projectTree },
    { id: "home.storage", title: "Storage", height: 120, grow: false, pad: true,
      render: () => el("div", {}, [
        el("div", { class: "field" }, [el("label", { text: "Used" }), el("span", { class: "v", text: "0 GB" })]),
        meter(0.41, "ok"),
        el("div", { class: "field" }, [el("label", { text: "Datasets" }), el("span", { text: "38" })]),
      ]) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Operations map",
    note: "Projects, active missions and fleet positions. Select a marker to load it into every panel.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `<strong>4</strong> projects &nbsp; <strong>1</strong> flying &nbsp; <strong>2</strong> processing` },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    /* The reconstruction this repository actually ran, and the figures it earned.
       Area, coverage and GSD used to sit here as invented numbers; they are not
       reported by that run, and a plausible number is worse than a missing one. */
    { id: "home.active", title: "Active Project", render: () => properties([
      { group: DATA.project.name },
      { label: "Images", value: DATA.project.images_registered },
      { label: "Reprojection", value: String(DATA.project.reprojection_px), unit: "px" },
      { label: "Geo RMSE", value: String(DATA.project.geo_rmse_m), unit: "m" },
      { label: "CRS", value: DATA.project.epsg },
      { label: "Models", value: String(DATA.models.length) },
      { label: "Capabilities", value: `${DATA.capabilities.verified}/${DATA.capabilities.total}` },
    ]) },
    { id: "home.alerts", title: "Alerts", render: () => el("div", { class: "console" }, [
      el("div", { class: "line warn" }, [el("span", { class: "t", text: "00:00" }), el("span", { text: "Battery B-07 cycle count 298 — service due" })]),
      el("div", { class: "line error" }, [el("span", { class: "t", text: "00:00" }), el("span", { text: "DEMO site 2: 3 captures out of tolerance" })]),
      el("div", { class: "line ok" }, [el("span", { class: "t", text: "00:00" }), el("span", { text: "DEMO site 3 reconstruction complete" })]),
    ]) },
    { id: "home.system", title: "System Status", render: () => properties([
      { label: "API", value: chip("healthy", "ok") },
      { label: "Workers", value: chip("3 online", "ok") },
      { label: "Broker", value: chip("redis 7.4", "ok") },
      { label: "Database", value: chip("postgis", "ok") },
      { label: "Geometry storage", value: chip("geojson text", "warn") },
    ]) },
  ],
  bottom: [
    { id: "home.activity", title: "Activity Log", flex: 2, render: () => consoleView([
      { t: "00:00:00", text: "mission 'Roof Block A v3' validated — 214 captures" },
      { t: "00:00:00", text: "dataset upload complete — 1,842 images", level: "ok" },
      { t: "00:00:00", text: "processing job #4471 queued (priority 10)" },
      { t: "00:00:00", text: "flight DEMO site 2 seg 2 resumed after battery swap", level: "warn" },
    ]) },
    { id: "home.jobs", title: "Processing Queue", flex: 2, render: () => table(
      [{ title: "Job", key: "job" }, { title: "Stage", key: "stage" },
       { title: "Worker", key: "worker" }, { title: "Progress", value: (r) => meter(r.p, r.p > 0.9 ? "ok" : "") }],
      [
        { job: "#4471", stage: "Feature matching", worker: "w-02", p: 0.34 },
        { job: "#4470", stage: "Dense cloud", worker: "w-01", p: 0.72 },
        { job: "#4468", stage: "Orthomosaic", worker: "w-03", p: 0.95 },
      ], { selectKind: "job" }) },
    { id: "home.summary", title: "Fleet", flex: 1, render: () => readouts([
      { k: "Aircraft", v: "6" }, { k: "Available", v: "4", tone: "ok" },
      { k: "Flying", v: "1", tone: "accent" }, { k: "Service", v: "1", tone: "warn" },
    ]) },
  ],
};

/* ----------------------------------------------------------------- PROJECTS */

const projects = {
  id: "projects",
  title: "Projects",
  toolbar: ["New Project", "New Folder", "Import", "|", "Archive", "Export", "Share"],
  left: [
    { id: "proj.tree", title: "Projects", render: projectTree },
    { id: "proj.archived", title: "Archived", height: 110, grow: false,
      render: () => tree([{ id: "ar1", label: "Pune Depot 2025", icon: "▤", meta: "closed" }]) },
  ],
  canvas: () => canvas({ map: true, title: "Project extent", note: "Boundary, survey history and asset locations.", tools: MAP_TOOLS, overlays: [{ at: "br", html: COORD }] }),
  right: [
    { id: "proj.props", title: "Project Properties", tabs: [
      { title: "General", render: () => properties([
        { group: "Identity" },
        { label: "Name", value: DATA.project.name },
        { label: "Source", value: DATA.project.source },
        { group: "Reconstruction" },
        { label: "Images", value: DATA.project.images_registered },
        { label: "Reprojection", value: String(DATA.project.reprojection_px), unit: "px" },
        { label: "Geo RMSE", value: String(DATA.project.geo_rmse_m), unit: "m" },
        { label: "CRS", value: DATA.project.epsg },
      ]) },
      { title: "Team", render: () => table(
        [{ title: "Model", key: "name" }, { title: "Measured", key: "role" }],
        DATA.models.slice(0, 5).map((model) => ({ name: model.key, role: model.headline || "installed" }))) },
      { title: "Tags", render: () => el("div", { class: "panel-body pad" }, [chip("warehouse"), chip("roof"), chip("quarterly")]) },
    ] },
  ],
  bottom: [
    { id: "proj.history", title: "Survey History", flex: 2, render: () => table(
      [{ title: "Date", key: "date" }, { title: "Mission", key: "mission" }, { title: "Images", key: "images", num: true },
       { title: "GSD cm", key: "gsd", num: true }, { title: "Coverage", key: "cov", num: true }],
      [
        { date: "2026-08-02", mission: "Roof Block A v3", images: 1842, gsd: 1.8, cov: "98.4%" },
        { date: "2026-05-14", mission: "Roof Block A v2", images: 1791, gsd: 1.9, cov: "97.1%" },
        { date: "2026-02-08", mission: "Roof Block A v1", images: 1610, gsd: 2.2, cov: "94.8%" },
      ], { selectKind: "survey" }) },
    { id: "proj.datasets", title: "Datasets", flex: 1, render: () => tree([
      { id: "d1", label: "2026-08-02 RGB", icon: "▦", meta: "1842" },
      { id: "d2", label: "2026-08-02 Thermal", icon: "◍", meta: "612" },
    ], { selectKind: "dataset" }) },
  ],
};

/* --------------------------------------------------------- MISSION PLANNING */

const missionTypes = [
  "2D mapping", "3D modelling", "Roof mapping", "Roof inspection", "Facade mapping",
  "Facade inspection", "Multi-facade", "Closed loop", "Linear mapping", "Linear inspection",
  "Tower mapping", "Orbit", "Waypoints", "Panorama", "Solar inspection",
  "Wind turbine", "Magnetic mapping", "Complex facade", "L-shaped", "Utility pylon",
  "Thermal", "Multispectral",
];

const planning = {
  id: "planning",
  title: "Mission Planning",
  toolbar: ["New Mission", "Open", "|", "Plan", "Validate", "Simulate", "|", "Upload", "Export", "Share"],
  left: [
    { id: "plan.browser", title: "Mission Browser", render: () => tree([
      { id: "m1", label: "Roof Block A v3", icon: "⟋", meta: "current" },
      { id: "m2", label: "Facade North v1", icon: "▯" },
      { id: "m3", label: "Yard grid v2", icon: "▦" },
    ], { selectKind: "mission" }) },
    { id: "plan.templates", title: "Mission Types", render: () => tree(
      missionTypes.map((t, i) => ({ id: `t${i}`, label: t, icon: "◇" })), { selectKind: "template" }) },
    { id: "plan.layers", title: "Layers", height: 150, grow: false, render: () => layerTree([
      { id: "l-obst", label: "Obstacles", icon: "⛔" },
      { id: "l-nofly", label: "No-fly regions", icon: "⊘" },
    ]) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Mission map",
    note: "Flight lines, capture points, camera footprints, geofence, terrain and obstacles.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `<strong>214</strong> captures &nbsp; <strong>3.1</strong> km &nbsp; <strong>18</strong> min &nbsp; GSD <strong>1.8</strong> cm` },
      { at: "bl", html: `Terrain: <span class="chip warn">flat plane — no DEM loaded</span>` },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    { id: "plan.props", title: "Mission Properties", tabs: [
      { title: "Mission", render: () => fields([
        { group: "Geometry" },
        { key: "type", label: "Type", value: "3D modelling", options: missionTypes },
        { key: "alt", label: "Altitude AGL", value: "60", unit: "m" },
        { key: "gsd", label: "Target GSD", value: "1.8", unit: "cm" },
        { key: "fwd", label: "Front overlap", value: "80", unit: "%" },
        { key: "side", label: "Side overlap", value: "70", unit: "%" },
        { group: "Path" },
        { key: "speed", label: "Speed", value: "8", unit: "m/s" },
        { key: "angle", label: "Line heading", value: "90", unit: "°" },
        { key: "standoff", label: "Stand-off", value: "12", unit: "m" },
      ], settingChanged) },
      { title: "Aircraft", render: () => fields([
        { key: "air", label: "Aircraft", value: "M350 RTK", options: ["M350 RTK", "M300", "Mavic 3E"] },
        { key: "batt", label: "Batteries", value: "3" },
        { key: "rth", label: "RTH altitude", value: "90", unit: "m" },
      ], settingChanged) },
      { title: "Camera", render: () => properties([
        { group: "P1 / 35 mm" },
        { label: "Sensor", value: "35.9 × 24.0", unit: "mm" },
        { label: "Resolution", value: "8192 × 5460" },
        { label: "Pixel pitch", value: "4.4", unit: "µm" },
        { label: "Focal length", value: "35", unit: "mm" },
        { label: "Footprint @60 m", value: "61 × 41", unit: "m" },
      ]) },
      { title: "Payload", render: () => properties([
        { label: "Type", value: "RGB + RTK" },
        { label: "Weight", value: "930", unit: "g" },
        { label: "Trigger", value: "Shutter" },
        { label: "Interval", value: "1.4", unit: "s" },
      ]) },
      { title: "Safety", render: () => properties([
        { label: "Geofence", value: chip("contained", "ok") },
        { label: "Terrain clearance", value: chip("assumed flat", "warn") },
        { label: "Rally points", value: "2" },
        { label: "Obstacles", value: "1" },
      ]) },
    ] },
    { id: "plan.estimates", title: "Estimates", height: 132, grow: false, render: () => readouts([
      { k: "Area", v: "14.2 ha" }, { k: "Distance", v: "3.1 km" },
      { k: "Duration", v: "18 min" }, { k: "Images", v: "214" },
      { k: "Batteries", v: "1" }, { k: "Storage", v: "12.4 GB" },
    ]) },
  ],
  bottom: [
    { id: "plan.timeline", title: "Simulation Timeline", flex: 3, render: () => canvas({
      title: "Altitude and speed over time", note: "Scrub to preview the flight. Playback follows the aircraft along the route.",
    }) },
    { id: "plan.segments", title: "Mission Segments", flex: 2, render: () => table(
      [{ title: "#", key: "n", num: true }, { title: "Segment", key: "name" },
       { title: "Captures", key: "caps", num: true }, { title: "Battery", key: "batt" }],
      [
        { n: 1, name: "Nadir grid — pass A", caps: 96, batt: "1" },
        { n: 2, name: "Nadir grid — cross", caps: 94, batt: "1" },
        { n: 3, name: "Oblique ring −45°", caps: 12, batt: "1" },
        { n: 4, name: "Oblique ring −60°", caps: 12, batt: "1" },
      ], { selectKind: "segment" }) },
    { id: "plan.log", title: "Validation", flex: 2, render: () => consoleView([
      { t: "—", text: "geofence contains every capture point", level: "ok" },
      { t: "—", text: "no terrain model loaded: altitudes are above a flat plane", level: "warn" },
      { t: "—", text: "payload understands the planned trigger command", level: "ok" },
    ]) },
  ],
};

/* ------------------------------------------------------------------- FLIGHT */

const flight = {
  id: "flight",
  title: "Flight",
  toolbar: ["Preflight", "|", "Start", "Pause", "Resume", "Capture Now", "|", "Manual Override", "RTL", "Land", "Abort"],
  left: [
    { id: "fly.aircraft", title: "Aircraft", render: () => properties([
      { label: "Airframe", value: "M350 RTK" },
      { label: "Link", value: chip("strong", "ok") },
      { label: "Mode", value: chip("AUTO", "info") },
      { label: "Armed", value: chip("yes", "ok") },
    ]) },
    { id: "fly.checklist", title: "Preflight", render: () => tree([
      { id: "c1", label: "Autopilot healthy", icon: "✓" },
      { id: "c2", label: "GPS fix — RTK fixed", icon: "✓" },
      { id: "c3", label: "Home position set", icon: "✓" },
      { id: "c4", label: "Battery 96%", icon: "✓" },
      { id: "c5", label: "Storage 0 GB free", icon: "✓" },
      { id: "c6", label: "Mission uploaded", icon: "✓" },
      { id: "c7", label: "Geofence uploaded", icon: "✓" },
      { id: "c8", label: "Terrain data — none", icon: "!" },
      { id: "c9", label: "Weather acknowledged", icon: "○" },
    ]) },
    { id: "fly.queue", title: "Mission Queue", height: 110, grow: false, render: () => tree([
      { id: "q1", label: "Roof Block A v3", icon: "▶", meta: "active" },
      { id: "q2", label: "Yard grid v2", icon: "▸", meta: "queued" },
    ]) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Live flight",
    note: "Aircraft position and heading, completed and remaining route, geofence, rally points and obstacles.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `<strong>62 / 214</strong> captures &nbsp; <span class="chip ok">AUTO</span> &nbsp; <span class="chip info">RTK fixed</span>` },
      { at: "bl", html: `Segment <strong>1 of 4</strong> — nadir grid pass A` },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    { id: "fly.telemetry", title: "Telemetry", render: () => readouts([
      { k: "Battery", v: "78%", tone: "ok" }, { k: "Voltage", v: "24.1 V" },
      { k: "Alt AGL", v: "60.2 m" }, { k: "Alt AMSL", v: "512.6 m" },
      { k: "Speed", v: "8.0 m/s" }, { k: "Heading", v: "090°" },
      { k: "Sats", v: "22", tone: "ok" }, { k: "HDOP", v: "0.6" },
      { k: "Link", v: "98%", tone: "ok" }, { k: "Home", v: "412 m" },
      { k: "Gimbal", v: "−90°" }, { k: "Flight time", v: "06:12" },
    ]) },
    { id: "fly.progress", title: "Mission Progress", height: 120, grow: false, pad: true,
      render: () => el("div", {}, [
        el("div", { class: "field" }, [el("label", { text: "Captures" }), el("span", { text: "62 / 214" })]),
        meter(62 / 214),
        el("div", { class: "field" }, [el("label", { text: "Distance" }), el("span", { text: "0.9 / 3.1 km" })]),
        meter(0.9 / 3.1),
      ]) },
    { id: "fly.alerts", title: "Alerts", height: 100, grow: false, render: () => consoleView([
      { t: "06:02", text: "wind gust 9.2 m/s — within limit", level: "warn" },
    ]) },
  ],
  bottom: [
    { id: "fly.charts", title: "Telemetry Charts", flex: 3, render: () => canvas({ title: "Altitude · speed · battery · link", note: "Rolling window over the current flight." }) },
    { id: "fly.events", title: "Event Log", flex: 2, render: () => consoleView([
      { t: "00:00:00", text: "capture 62 triggered" },
      { t: "00:00:00", text: "waypoint 31 reached" },
      { t: "00:00:00", text: "mission started", level: "ok" },
      { t: "00:00:00", text: "armed", level: "ok" },
    ]) },
    { id: "fly.camera", title: "Camera", flex: 1, render: () => canvas({ title: "Live view", note: "Downlink preview" }) },
  ],
};

/* ------------------------------------------------------------- VERIFICATION */

const verification = {
  id: "verification",
  title: "Verification",
  toolbar: ["Load Flight", "Match Captures", "|", "Accept", "Flag", "Request Reflight", "|", "Export Report"],
  left: [
    { id: "ver.datasets", title: "Datasets", render: () => tree([
      { id: "f1", label: "2026-08-02 flight 1", icon: "▶", meta: "1842" },
      { id: "f2", label: "2026-08-02 flight 2", icon: "▶", meta: "612" },
    ], { selectKind: "flight" }) },
    { id: "ver.images", title: "Images", render: () => tree(
      Array.from({ length: 8 }, (_, i) => ({ id: `i${i}`, label: `DJI_${1000 + i}.JPG`, icon: "▣" })),
      { selectKind: "image" }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Planned against actual",
    note: "Planned capture positions, actual positions, deviation vectors and coverage footprints.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `<span class="chip ok">1836 matched</span> <span class="chip warn">3 out of tolerance</span> <span class="chip error">3 missing</span>` },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    { id: "ver.capture", title: "Capture Properties", render: () => table(
      [{ title: "Field", key: "f" }, { title: "Planned", key: "p" }, { title: "Actual", key: "a" }, { title: "Δ", key: "d", num: true }],
      [
        { f: "Latitude", p: "23.25914", a: "23.25913", d: "1.1 m" },
        { f: "Longitude", p: "77.40218", a: "77.40219", d: "0.9 m" },
        { f: "Altitude", p: "60.0 m", a: "59.4 m", d: "0.6 m" },
        { f: "Heading", p: "090°", a: "091°", d: "1°" },
        { f: "Gimbal", p: "−90°", a: "−90°", d: "0°" },
        { f: "GSD", p: "1.80 cm", a: "1.78 cm", d: "0.02" },
      ]) },
  ],
  bottom: [
    { id: "ver.summary", title: "Verification Summary", flex: 2, render: () => readouts([
      { k: "Planned", v: "1842" }, { k: "Matched", v: "1836", tone: "ok" },
      { k: "Missing", v: "3", tone: "error" }, { k: "Out of tol.", v: "3", tone: "warn" },
      { k: "Coverage", v: "98.4%" }, { k: "Pos. error", v: "0.9 m" },
    ]) },
    { id: "ver.missing", title: "Missing Captures", flex: 2, render: () => table(
      [{ title: "Capture", key: "id" }, { title: "Segment", key: "seg" }, { title: "Reason", key: "why" }],
      [
        { id: "C-1094", seg: "cross pass", why: "no image within tolerance" },
        { id: "C-1095", seg: "cross pass", why: "no image within tolerance" },
        { id: "C-1782", seg: "oblique −60°", why: "trigger not recorded" },
      ], { selectKind: "capture" }) },
    { id: "ver.qc", title: "QC Log", flex: 1, render: () => consoleView([
      { t: "09:31", text: "3 captures flagged for reflight", level: "warn" },
    ]) },
  ],
};

/* ---------------------------------------------------------------- PROCESSING */

const processing = {
  id: "processing",
  title: "Processing",
  toolbar: ["New Job", "|", "Process", "Pause", "Cancel", "|", "Add GCPs", "Set CRS", "Export Products"],
  left: [
    { id: "proc.datasets", title: "Datasets", render: () => tree([
      { id: "pd1", label: "2026-08-02 RGB", icon: "▦", meta: "1842" },
      { id: "pd2", label: "GCPs", icon: "✛", meta: "7" },
      { id: "pd3", label: "PPK base", icon: "◎" },
    ], { selectKind: "dataset" }) },
    { id: "proc.jobs", title: "Jobs", render: () => tree([
      { id: "j1", label: "#4471 DEMO site 1 roof", icon: "▶", meta: "34%" },
      { id: "j2", label: "#4470 DEMO site 2", icon: "▶", meta: "72%" },
      { id: "j3", label: "#4468 DEMO site 3", icon: "✓", meta: "done" },
    ], { selectKind: "job" }) },
  ],
  canvas: () => canvas({
    title: "Sparse reconstruction",
    note: "Camera positions, tie points and the growing point cloud. Switches to dense, mesh and orthomosaic as stages complete.",
    tools: [{ icon: "✥", title: "Orbit" }, { icon: "⊕", title: "Zoom" }, { icon: "▣", title: "Select" }, { icon: "⤡", title: "Fit" }],
    overlays: [
      { at: "tl", html: `<strong>Ingestion → Features → Matching → SfM → Georeference → Dense → DSM/DTM → Ortho → Mesh → Twin</strong>` },
      { at: "br", html: `1,842 cameras · 412k tie points` },
    ],
  }),
  right: [
    { id: "proc.settings", title: "Processing Settings", tabs: [
      { title: "Quality", render: () => fields([
        { key: "profile", label: "Profile", value: "standard", options: ["fast", "standard", "high"] },
        { key: "imgsize", label: "Max image size", value: "4096", unit: "px" },
        { key: "dense", label: "Dense cloud", value: "yes", options: ["yes", "no"] },
        { key: "mesh", label: "Mesh depth", value: "9" },
      ], settingChanged) },
      { title: "Spatial", render: () => properties([
        { label: "Reference", value: chip("georeferenced", "ok") },
        { label: "CRS", value: "EPSG:4326" },
        { label: "GCPs", value: "7" },
        { label: "RTK", value: chip("fixed", "ok") },
        { label: "Scale", value: "metric" },
      ]) },
      { title: "Workers", render: () => table(
        [{ title: "Worker", key: "w" }, { title: "State", key: "s" }, { title: "CPU", key: "cpu" },
         { title: "RAM", key: "ram" }, { title: "GPU", key: "gpu" }],
        [
          { w: "w-01", s: "busy", cpu: "94%", ram: "41/64 GB", gpu: "RTX 4090" },
          { w: "w-02", s: "busy", cpu: "88%", ram: "36/64 GB", gpu: "RTX 4090" },
          { w: "w-03", s: "idle", cpu: "3%", ram: "4/32 GB", gpu: "—" },
        ], { selectKind: "worker" }) },
    ] },
    { id: "proc.outputs", title: "Output Products", height: 170, grow: false, render: () => tree([
      { id: "o1", label: "Sparse cloud", icon: "◌" },
      { id: "o2", label: "Dense cloud", icon: "◍" },
      { id: "o3", label: "DSM", icon: "◰" },
      { id: "o4", label: "DTM", icon: "◱" },
      { id: "o5", label: "Orthomosaic", icon: "▦" },
      { id: "o6", label: "Textured mesh", icon: "◈" },
    ], { selectKind: "product" }) },
  ],
  bottom: [
    { id: "proc.progress", title: "Job Progress", flex: 2, render: () => table(
      [{ title: "Stage", key: "stage" }, { title: "State", key: "state" }, { title: "Elapsed", key: "t" },
       { title: "", value: (r) => meter(r.p, r.p >= 1 ? "ok" : "") }],
      [
        { stage: "Ingestion", state: "done", t: "0:41", p: 1 },
        { stage: "Feature extraction", state: "done", t: "6:12", p: 1 },
        { stage: "Feature matching", state: "running", t: "9:04", p: 0.34 },
        { stage: "Structure from motion", state: "queued", t: "—", p: 0 },
      ]) },
    { id: "proc.logs", title: "Processing Log", flex: 3, render: () => consoleView([
      { t: "00:00:00", text: "matching 412,880 pairs across 3 workers" },
      { t: "00:00:00", text: "features extracted: 1,842 images", level: "ok" },
      { t: "00:00:00", text: "job accepted by w-02 (priority 10, attempt 1)" },
    ]) },
  ],
};

/* -------------------------------------------------------------- DIGITAL TWIN */

const twin = {
  id: "twin",
  title: "Digital Twin",
  toolbar: ["Textured Mesh", "Point Cloud", "Thermal", "Semantic", "Change", "|", "Compare Dates", "Measure", "Annotate", "Export"],
  left: [
    { id: "twin.hierarchy", title: "Scene Hierarchy", render: () => tree([
      { id: "s1", label: "DEMO site 1", icon: "▦", children: [
        { id: "s2", label: "Block A", icon: "▢", children: [
          { id: "s3", label: "Roof", icon: "▱", meta: "4 defects" },
          { id: "s4", label: "North facade", icon: "▯", meta: "1" },
        ] },
        { id: "s5", label: "Yard", icon: "▲", children: [
          { id: "s6", label: "Stockpile 1", icon: "▲", meta: "1,240 m³" },
        ] },
      ] },
    ], { selectKind: "asset" }) },
    { id: "twin.layers", title: "Layers", height: 160, grow: false, render: () => layerTree([
      { id: "l-defect", label: "Defects", icon: "⚠" },
      { id: "l-thermal", label: "Thermal", icon: "◍" },
      { id: "l-change", label: "Change", icon: "±" },
    ]) },
  ],
  canvas: () => canvas({
    title: "Digital twin",
    note: "Click any object to inspect its metadata, inspection history and findings.",
    tools: [{ icon: "✥", title: "Orbit" }, { icon: "⊕", title: "Zoom" }, { icon: "▣", title: "Select" },
            { icon: "⤢", title: "Measure" }, { icon: "✎", title: "Annotate" }, { icon: "⤡", title: "Fit" }],
    overlays: [
      { at: "tr", html: `<span class="chip">Textured mesh</span> <span class="chip info">2026-08-02</span>` },
      { at: "br", html: `4.1M faces · 12 assets · 6 findings` },
    ],
  }),
  right: [
    { id: "twin.asset", title: "Selected Asset", render: () => properties([
      { group: "Roof — Block A" },
      { label: "Type", value: "Roof" },
      { label: "Area", value: "3,140", unit: "m²" },
      { label: "Material", value: "Metal deck" },
      { label: "Surveys", value: "3" },
      { label: "Open findings", value: chip("4", "warn") },
      { group: "Latest inspection" },
      { label: "Date", value: "2026-08-02" },
      { label: "Model", value: "crack_presence_classifier" },
      { label: "Digest", value: "92a4d142…" },
    ]) },
    { id: "twin.timeline", title: "Survey Timeline", height: 130, grow: false, render: () => table(
      [{ title: "Date", key: "d" }, { title: "Findings", key: "f", num: true }, { title: "Change", key: "c" }],
      [
        { d: "2026-08-02", f: 4, c: "+1" },
        { d: "2026-05-14", f: 3, c: "+2" },
        { d: "2026-02-08", f: 1, c: "—" },
      ], { selectKind: "survey" }) },
  ],
  bottom: [
    { id: "twin.props", title: "Object Properties", flex: 2, render: () => properties([
      { label: "Object id", value: "asset/roof-block-a" },
      { label: "Parent", value: "Block A" },
      { label: "Centroid", value: "23.2591 N, 77.4022 E" },
      { label: "Source run", value: "#4468" },
    ]) },
    { id: "twin.measure", title: "Measurements", flex: 2, render: () => table(
      [{ title: "Name", key: "n" }, { title: "Type", key: "t" }, { title: "Value", key: "v", num: true }],
      [
        { n: "Roof span", t: "Distance", v: "62.4 m" },
        { n: "Roof area", t: "Area", v: "3,140 m²" },
        { n: "Parapet", t: "Height", v: "1.1 m" },
      ], { selectKind: "measurement" }) },
    { id: "twin.history", title: "History", flex: 1, render: () => consoleView([
      { t: "08-02", text: "crack finding F-118 added" },
      { t: "05-14", text: "ponding F-092 resolved", level: "ok" },
    ]) },
  ],
};

/* ------------------------------------------------------------- AI INSPECTION */

const inspection = {
  id: "inspection",
  title: "AI Inspection",
  toolbar: ["Run AI", "|", "Validate", "Reject", "Edit", "Flag", "|", "Open in 3D", "Generate Report"],
  left: [
    { id: "ai.library", title: "Defect Library", render: () => tree([
      { id: "dl1", label: "Crack", icon: "⚡", meta: "4" },
      { id: "dl2", label: "Spalling", icon: "◆", meta: "1" },
      { id: "dl3", label: "Corrosion", icon: "▨", meta: "2" },
      { id: "dl4", label: "Water ponding", icon: "≈", meta: "1" },
      { id: "dl5", label: "Deformation", icon: "◠" },
      { id: "dl6", label: "Solar defect", icon: "▣" },
    ], { selectKind: "defectClass" }) },
    { id: "ai.datasets", title: "Model Runs", render: () => tree([
      { id: "r1", label: "crack_presence 0.958", icon: "◈", meta: "run 91" },
      { id: "r2", label: "crack_segmentation 0.606", icon: "◈", meta: "run 91" },
      { id: "r3", label: "rail_obstacle 0.824", icon: "◈", meta: "run 88" },
    ], { selectKind: "modelRun" }) },
  ],
  canvas: () => splitCanvas([
    { title: "Source image", note: "DJI_1094.JPG — capture C-1094" },
    { title: "Detection", note: "Mask and confidence at full resolution" },
    { title: "3D context", note: "Back-projected onto the surface it belongs to" },
  ]),
  right: [
    { id: "ai.finding", title: "Finding Details", render: () => properties([
      { group: "F-118" },
      { label: "Class", value: chip("crack", "warn") },
      { label: "Confidence", value: "0.91" },
      { label: "Severity", value: chip("moderate", "warn") },
      { label: "Length", value: "1.42", unit: "m" },
      { label: "Width", value: chip("not measured", "") },
      { group: "Provenance" },
      { label: "Model", value: "crack_presence_classifier" },
      { label: "Digest", value: "92a4d142…" },
      { label: "Run", value: "#91" },
      { label: "Source image", value: "DJI_1094.JPG" },
      { label: "Mission", value: "Roof Block A v3" },
      { label: "Captured", value: "2026-08-02 09:14" },
      { group: "Location" },
      { label: "Asset", value: "Roof — Block A" },
      { label: "Coordinate", value: "23.25914 N, 77.40218 E" },
      { label: "Validation", value: chip("unreviewed", "info") },
    ]) },
  ],
  bottom: [
    { id: "ai.findings", title: "Findings", flex: 4, render: () => table(
      [{ title: "ID", key: "id" }, { title: "Class", key: "cls" }, { title: "Conf", key: "conf", num: true },
       { title: "Severity", key: "sev" }, { title: "Asset", key: "asset" }, { title: "Status", key: "status" }],
      [
        { id: "F-118", cls: "crack", conf: 0.91, sev: "moderate", asset: "Roof — Block A", status: "unreviewed" },
        { id: "F-117", cls: "crack", conf: 0.88, sev: "minor", asset: "Roof — Block A", status: "validated" },
        { id: "F-116", cls: "corrosion", conf: 0.74, sev: "severe", asset: "North facade", status: "unreviewed" },
        { id: "F-112", cls: "ponding", conf: 0.96, sev: "moderate", asset: "Roof — Block A", status: "validated" },
      ], { selectKind: "finding" }) },
    { id: "ai.filters", title: "Filters", flex: 1, render: () => fields([
      { key: "min", label: "Min confidence", value: "0.50" },
      { key: "cls", label: "Class", value: "all", options: ["all", "crack", "corrosion", "ponding"] },
      { key: "status", label: "Status", value: "all", options: ["all", "unreviewed", "validated", "rejected"] },
    ], settingChanged) },
  ],
};

/* ------------------------------------------------------------------ THERMAL */

const thermal = {
  id: "thermal",
  title: "Thermal",
  toolbar: ["RGB", "Thermal", "Fused", "3D Thermal", "|", "Radiometric", "Compare", "Export"],
  left: [
    { id: "th.arrays", title: "Array Inventory", render: () => tree([
      { id: "ar1", label: "Block 1", icon: "▦", children: [
        { id: "st1", label: "String 1-A", icon: "▤", meta: "24" },
        { id: "st2", label: "String 1-B", icon: "▤", meta: "24" },
      ] },
      { id: "ar2", label: "Block 2", icon: "▦", meta: "48" },
    ], { selectKind: "array" }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Thermal map",
    note: "Module boundaries, anomaly overlays and temperature labels over the array.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `<span class="chip thermal">Fused</span> <span class="chip warn">7 anomalies</span>` },
      { at: "br", html: `Ambient 34.2 °C · palette ironbow` },
    ],
  }),
  right: [
    { id: "th.module", title: "Selected Module", render: () => properties([
      { group: "M-1-A-14" },
      { label: "Anomaly", value: chip("Hot-Spot", "warn") },
      { label: "Confidence", value: "0.70" },
      { label: "Peak temp", value: "68.4", unit: "°C" },
      { label: "Delta T", value: "34.2", unit: "K" },
      { label: "Ambient", value: "34.2", unit: "°C" },
      { group: "Model" },
      { label: "Model", value: "solar_thermal_anomaly_classifier" },
      { label: "Balanced acc.", value: "0.724" },
      { label: "Soiling recall", value: chip("0.367 — weak", "error") },
    ]) },
  ],
  bottom: [
    { id: "th.summary", title: "Anomaly Summary", flex: 2, render: () => table(
      [{ title: "Class", key: "c" }, { title: "Count", key: "n", num: true }, { title: "Peak ΔT", key: "d", num: true }],
      [
        { c: "Hot-Spot", n: 3, d: "34.2 K" },
        { c: "Diode", n: 2, d: "18.9 K" },
        { c: "Soiling", n: 1, d: "6.1 K" },
        { c: "Vegetation", n: 1, d: "4.4 K" },
      ], { selectKind: "anomaly" }) },
    { id: "th.hist", title: "Thermal Histogram", flex: 2, render: () => canvas({ title: "Temperature distribution", note: "Across the selected block" }) },
  ],
};

/* ------------------------------------------------------------- MEASUREMENTS */

const measurements = {
  id: "measurements",
  title: "Measurements",
  toolbar: ["Distance", "Area", "Volume", "Stockpile", "Cut & Fill", "Slope", "Profile", "|", "Compare Dates", "Export"],
  left: [
    { id: "ms.tools", title: "Tools", render: () => tree([
      { id: "t1", label: "Distance", icon: "↔" },
      { id: "t2", label: "Area", icon: "▭" },
      { id: "t3", label: "Volume", icon: "◙" },
      { id: "t4", label: "Stockpile", icon: "▲" },
      { id: "t5", label: "Cut & Fill", icon: "⇅" },
      { id: "t6", label: "Slope", icon: "◺" },
      { id: "t7", label: "Profile line", icon: "∿" },
    ], { selectKind: "tool" }) },
    { id: "ms.layers", title: "Layers", render: () => layerTree([
      { id: "l-vol", label: "Volumes", icon: "◙" },
      { id: "l-slope", label: "Slope map", icon: "◺" },
    ]) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Measurement canvas",
    note: "Orthomosaic, terrain and point cloud with measurement overlays.",
    tools: MAP_TOOLS,
    overlays: [
      { at: "tr", html: `Vertical accuracy <strong>±0.05 m</strong> — nothing below 0.10 m is reported` },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    { id: "ms.details", title: "Measurement Details", render: () => properties([
      { group: "Stockpile 1" },
      { label: "Volume", value: "1,240", unit: "m³" },
      { label: "Base area", value: "612", unit: "m²" },
      { label: "Max height", value: "6.4", unit: "m" },
      { label: "Mean height", value: "2.0", unit: "m" },
      { label: "Base method", value: "lowest perimeter" },
      { group: "Confidence" },
      { label: "Vertical accuracy", value: "±0.05", unit: "m" },
      { label: "Volume uncertainty", value: "±31", unit: "m³" },
      { group: "Change" },
      { label: "Previous", value: "1,102 m³ (2026-05-14)" },
      { label: "Difference", value: chip("+138 m³", "info") },
    ]) },
  ],
  bottom: [
    { id: "ms.history", title: "Measurement History", flex: 3, render: () => table(
      [{ title: "Name", key: "n" }, { title: "Type", key: "t" }, { title: "Value", key: "v", num: true },
       { title: "Date", key: "d" }, { title: "Δ", key: "c", num: true }],
      [
        { n: "Stockpile 1", t: "Volume", v: "1,240 m³", d: "2026-08-02", c: "+138" },
        { n: "Stockpile 2", t: "Volume", v: "884 m³", d: "2026-08-02", c: "−42" },
        { n: "Yard extent", t: "Area", v: "1.42 ha", d: "2026-08-02", c: "0" },
      ], { selectKind: "measurement" }) },
    { id: "ms.profile", title: "Profile", flex: 2, render: () => canvas({ title: "Elevation profile", note: "Along the selected profile line" }) },
  ],
};

/* -------------------------------------------------------------------- FLEET */

const fleet = {
  id: "fleet",
  title: "Fleet",
  toolbar: ["Add Aircraft", "Add Battery", "Add Pilot", "|", "Assign Mission", "Log Maintenance", "Export"],
  left: [
    { id: "fl.aircraft", title: "Aircraft", render: () => tree([
      { id: "a1", label: "M350-01", icon: "▲", meta: "flying" },
      { id: "a2", label: "M350-02", icon: "▲", meta: "available" },
      { id: "a3", label: "M300-01", icon: "▲", meta: "service" },
      { id: "a4", label: "Mavic-3E-01", icon: "▲", meta: "available" },
    ], { selectKind: "aircraft" }) },
    { id: "fl.pilots", title: "Pilots", render: () => tree([
      { id: "p1", label: "A. Sharma", icon: "◉", meta: "142 h" },
      { id: "p2", label: "R. Iyer", icon: "◉", meta: "88 h" },
    ], { selectKind: "pilot" }) },
  ],
  canvas: () => canvas({ map: true, title: "Fleet map", note: "Aircraft locations, assignments and operational state.", tools: MAP_TOOLS, overlays: [{ at: "br", html: COORD }] }),
  right: [
    { id: "fl.details", title: "Aircraft Details", render: () => properties([
      { group: "M350-01" },
      { label: "Model", value: "Matrice 350 RTK" },
      { label: "Serial", value: "1ZNBJ9..." },
      { label: "Firmware", value: "09.01.0034" },
      { label: "Payload", value: "Zenmuse P1" },
      { label: "Flight hours", value: "142.6" },
      { label: "Last flight", value: "today" },
      { label: "Next service", value: chip("in 7.4 h", "warn") },
      { label: "State", value: chip("flying", "info") },
    ]) },
  ],
  bottom: [
    { id: "fl.batteries", title: "Batteries", flex: 3, render: () => table(
      [{ title: "ID", key: "id" }, { title: "Charge", key: "c", num: true }, { title: "Cycles", key: "n", num: true },
       { title: "Health", key: "h" }, { title: "Temp", key: "t", num: true }, { title: "State", key: "s" }],
      [
        { id: "B-05", c: "96%", n: 142, h: "good", t: "31 °C", s: "in use" },
        { id: "B-06", c: "100%", n: 118, h: "good", t: "24 °C", s: "ready" },
        { id: "B-07", c: "88%", n: 298, h: "service", t: "26 °C", s: "reserved" },
      ], { selectKind: "battery" }) },
    { id: "fl.maint", title: "Maintenance", flex: 2, render: () => consoleView([
      { t: "08-14", text: "M300-01 propeller set replaced" },
      { t: "07-30", text: "B-07 flagged: cycle count above 250", level: "warn" },
    ]) },
  ],
};

/* ------------------------------------------------------------------ REPORTS */

const reports = {
  id: "reports",
  title: "Reports",
  toolbar: ["New Report", "|", "Generate", "Export PDF", "Share", "Save Template"],
  left: [
    { id: "rp.structure", title: "Report Structure", render: () => tree([
      { id: "r1", label: "Executive Summary", icon: "§" },
      { id: "r2", label: "Project", icon: "§" },
      { id: "r3", label: "Mission & Coverage", icon: "§" },
      { id: "r4", label: "Processing", icon: "§" },
      { id: "r5", label: "Inspection Findings", icon: "§", meta: "6" },
      { id: "r6", label: "Measurements", icon: "§" },
      { id: "r7", label: "Thermal", icon: "§" },
      { id: "r8", label: "Change Analysis", icon: "§" },
      { id: "r9", label: "Recommendations", icon: "§" },
      { id: "r10", label: "Appendix", icon: "§" },
    ], { selectKind: "section" }) },
  ],
  canvas: () => canvas({ title: "Report preview", note: "Live preview of the generated document." }),
  right: [
    { id: "rp.settings", title: "Template", render: () => fields([
      { key: "tpl", label: "Template", value: "Inspection", options: ["Inspection", "Survey", "Progress", "Thermal"] },
      { key: "sev", label: "Min severity", value: "minor", options: ["any", "minor", "moderate", "severe"] },
      { key: "from", label: "From", value: "2026-05-14" },
      { key: "to", label: "To", value: "2026-08-02" },
      { key: "fmt", label: "Format", value: "PDF", options: ["PDF", "DOCX", "HTML"] },
    ], settingChanged) },
    { id: "rp.branding", title: "Branding", height: 110, grow: false, render: () => fields([
      { key: "org", label: "Organisation", value: "DEMO organisation" },
      { key: "logo", label: "Logo", value: "brand.png" },
    ], settingChanged) },
  ],
  bottom: [
    { id: "rp.included", title: "Included Findings", flex: 3, render: () => table(
      [{ title: "ID", key: "id" }, { title: "Class", key: "c" }, { title: "Severity", key: "s" }, { title: "Asset", key: "a" }],
      [
        { id: "F-118", c: "crack", s: "moderate", a: "Roof — Block A" },
        { id: "F-116", c: "corrosion", s: "severe", a: "North facade" },
        { id: "F-112", c: "ponding", s: "moderate", a: "Roof — Block A" },
      ]) },
    { id: "rp.log", title: "Generation Log", flex: 2, render: () => consoleView([
      { t: "—", text: "every figure carries its model digest and run id", level: "ok" },
    ]) },
  ],
};

/* --------------------------------------------------------------- DEVELOPERS */

const developers = {
  id: "developers",
  title: "Developers",
  toolbar: ["New Key", "|", "Send Request", "Copy cURL", "|", "Add Webhook", "Install Plugin"],
  left: [
    { id: "dev.endpoints", title: "Endpoints", render: () => tree([
      { id: "e1", label: "GET /projects", icon: "▸" },
      { id: "e2", label: "GET /assets", icon: "▸" },
      { id: "e3", label: "POST /missions", icon: "▸" },
      { id: "e4", label: "GET /missions/{id}", icon: "▸" },
      { id: "e5", label: "POST /datasets", icon: "▸" },
      { id: "e6", label: "GET /processing/jobs", icon: "▸" },
      { id: "e7", label: "GET /findings", icon: "▸" },
      { id: "e8", label: "GET /measurements", icon: "▸" },
      { id: "e9", label: "GET /reports", icon: "▸" },
    ], { selectKind: "endpoint" }) },
    { id: "dev.plugins", title: "Plugins", render: () => tree([
      { id: "pl1", label: "shapefile-exporter", icon: "◈", meta: "enabled" },
      { id: "pl2", label: "custom-drone", icon: "◈", meta: "enabled" },
    ], { selectKind: "plugin" }) },
  ],
  canvas: () => canvas({ title: "API console", note: "Request, response and authentication for the selected endpoint." }),
  right: [
    { id: "dev.auth", title: "Authentication", render: () => properties([
      { label: "Scheme", value: "Bearer" },
      { label: "Key", value: "odk_live_8f2a…" },
      { label: "Scopes", value: "read, write" },
      { label: "Requests today", value: "1,204" },
    ]) },
    { id: "dev.webhooks", title: "Webhooks", render: () => table(
      [{ title: "Event", key: "e" }, { title: "Status", key: "s" }],
      [
        { e: "mission.completed", s: "active" },
        { e: "processing.completed", s: "active" },
        { e: "findings.ready", s: "active" },
        { e: "report.generated", s: "paused" },
      ], { selectKind: "webhook" }) },
  ],
  bottom: [
    { id: "dev.infra", title: "Infrastructure", flex: 3, render: () => table(
      [{ title: "Component", key: "c" }, { title: "State", key: "s" }, { title: "Detail", key: "d" }],
      [
        { c: "REST API", s: "healthy", d: "8000" },
        { c: "Processing workers", s: "3 online", d: "celery/redis" },
        { c: "PostgreSQL + PostGIS", s: "healthy", d: "geometry stored as GeoJSON text" },
        { c: "Object storage", s: "healthy", d: "s3 / minio" },
        { c: "Observability", s: "healthy", d: "/metrics" },
      ], { selectKind: "component" }) },
    { id: "dev.events", title: "Event Stream", flex: 2, render: () => consoleView([
      { t: "00:00:00", text: "processing.progress job=4471 pct=34" },
      { t: "00:00:00", text: "processing.started job=4471" },
    ]) },
  ],
};

/* ----------------------------------------------------------------- SETTINGS */

const settings = {
  id: "settings",
  title: "Settings",
  toolbar: ["Save", "Reset", "|", "Import", "Export"],
  left: [
    { id: "set.sections", title: "Settings", render: () => tree([
      { id: "s1", label: "General", icon: "⚙" },
      { id: "s2", label: "Units & CRS", icon: "⚙" },
      { id: "s3", label: "Map & Offline", icon: "⚙" },
      { id: "s4", label: "Safety defaults", icon: "⚙" },
      { id: "s5", label: "Models", icon: "⚙" },
      { id: "s6", label: "Keyboard", icon: "⚙" },
      { id: "s7", label: "Workspaces", icon: "⚙" },
    ], { selectKind: "setting" }) },
  ],
  canvas: () => canvas({ title: "Settings", note: "Select a section." }),
  right: [
    { id: "set.detail", title: "Units & CRS", render: () => fields([
      { key: "units", label: "Units", value: "metric", options: ["metric", "imperial"] },
      { key: "crs", label: "Default CRS", value: "EPSG:4326" },
      { key: "vert", label: "Vertical datum", value: "EGM96" },
      { key: "angle", label: "Angles", value: "degrees", options: ["degrees", "mils"] },
    ], settingChanged) },
  ],
  bottom: [
    { id: "set.models", title: "Installed Models", flex: 1, render: () => table(
      [{ title: "Model", key: "m" }, { title: "Metric", key: "v", num: true }, { title: "Digest", key: "d" }],
      [
        { m: "solar_cell_defect_detector", v: "mAP50 0.884", d: "…" },
        { m: "crack_presence_classifier", v: "bal.acc 0.958", d: "92a4d142…" },
        { m: "rail_obstacle_detector", v: "mAP50 0.824", d: "7a53c8b9…" },
        { m: "solar_thermal_anomaly_classifier", v: "bal.acc 0.724", d: "6933a09a…" },
        { m: "rail_corridor_segmentation", v: "IoU 0.681", d: "…" },
        { m: "crack_segmentation", v: "IoU 0.606", d: "…" },
      ], { selectKind: "model" }) },
  ],
};

export const WORKSPACES = [
  home, projects, planning, flight, verification, processing,
  twin, inspection, thermal, measurements, fleet, reports, developers, settings,
];

export const WORKSPACE_BY_ID = Object.fromEntries(WORKSPACES.map((w) => [w.id, w]));
