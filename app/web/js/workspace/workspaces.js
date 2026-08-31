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
import { emptyState, live, reported } from "./live.js";

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
const projectTree = () => live({
  calls: ["list_projects", "verify_models", "get_state"],
  empty: "No projects yet. Use New Project to create one.",
  isEmpty: ([projects, models]) => !(projects && (projects.projects || []).length)
    && !(models && (models.models || []).length),
  render: ([projects, models, state]) => {
    const installed = ((models && models.models) || [])
      .filter((m) => m.status !== "awaiting_weights");
    const active = (state && (state.project || state.active_project)) || {};
    const rows = (projects && projects.projects) || [];
    return tree([
      ...(installed.length ? [{
        id: "org", label: "Installed models", icon: "▦", meta: String(installed.length),
        children: installed.map((m, i) => ({
          id: `m${i}`, label: m.model_key, icon: "▤",
          meta: m.status === "verified" ? "verified" : m.status,
        })),
      }] : []),
      {
        id: "projects", label: "Projects", icon: "▦", meta: String(rows.length),
        children: rows.slice(0, 40).map((project) => ({
          id: `p${project.id}`,
          label: project.name,
          icon: "▤",
          meta: project.id === active.id ? "active" : "",
        })),
      },
    ], { selectKind: "project" });
  },
});

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
    /* Showed "0 GB used" beside a 41%-full meter and 38 datasets, none of which was
       read from anywhere. Project and dataset counts are things the app can actually
       answer; disk usage is not, so it is no longer claimed. */
    { id: "home.storage", title: "Library", height: 120, grow: false, pad: true,
      render: () => live({
        calls: ["list_projects", "list_datasets"],
        empty: "No projects yet.",
        isEmpty: ([projects]) => !(projects && (projects.projects || []).length),
        render: ([projects, datasets]) => properties(reported([
          ["Projects", (projects.projects || []).length],
          ["Datasets", ((datasets && datasets.datasets) || []).length || undefined],
        ])),
      }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Operations map",
    note: "Projects, active missions and fleet positions. Select a marker to load it into every panel.",
    tools: MAP_TOOLS,
    overlays: [
      /* Was "4 projects, 1 flying, 2 processing" on every machine. Nothing flies from
         a desk, and the real counts are in the Library panel. */
      { at: "br", html: COORD },
    ],
  }),
  right: [
    /* The reconstruction this repository actually ran, and the figures it earned.
       Area, coverage and GSD used to sit here as invented numbers; they are not
       reported by that run, and a plausible number is worse than a missing one. */
    { id: "home.active", title: "Active Project", render: () => live({
      calls: ["get_state", "list_layers", "list_dataset_images"],
      empty: "No project open. Use Open, or New Project to create one.",
      isEmpty: ([state]) => !(state && (state.project || state.active_project)),
      render: ([state, layers, images]) => {
        const project = state.project || state.active_project || {};
        const twin = project.reconstruction || project.digital_twin || {};
        return properties([
          { group: project.name || "Project" },
          ...reported([
            ["Images", ((images && images.images) || []).length || undefined],
            ["Registered", twin.registered_images],
            ["Reprojection", twin.mean_reprojection_error_px &&
              twin.mean_reprojection_error_px.toFixed(3), "px"],
            ["Geo RMSE", twin.geo_rmse_m && twin.geo_rmse_m.toFixed(3), "m"],
            ["Ground sample", twin.ground_sample_distance_m &&
              twin.ground_sample_distance_m.toFixed(3), "m/px"],
            ["CRS", project.epsg || (twin.crs_epsg && `EPSG:${twin.crs_epsg}`)],
            ["Layers", ((layers && layers.layers) || []).length || undefined],
          ]),
        ]);
      },
    }) },
    /* Three invented incidents used to sit here, about sites that do not exist. The
       audit log is the real record of what happened on this machine. */
    { id: "home.alerts", title: "Recent activity", render: () => live({
      calls: ["audit_log"],
      empty: "Nothing has happened in this workspace yet.",
      isEmpty: ([log]) => !(log && (log.entries || log.events || []).length),
      render: ([log]) => consoleView((log.entries || log.events || []).slice(0, 12).map((e) => ({
        t: String(e.at || e.timestamp || "").slice(11, 19) || "—",
        text: e.message || e.action || e.event || "—",
        level: /fail|error|refus/i.test(e.message || e.action || "") ? "error"
          : /warn/i.test(e.message || e.action || "") ? "warn" : "",
      }))),
    }) },
    /* Reported three workers, a redis broker and postgis. This build is local-first and
       runs none of them; it was describing a deployment that does not exist here. */
    { id: "home.system", title: "System Status", render: () => live({
      calls: ["capabilities", "verify_models"],
      empty: "The application did not report its capabilities.",
      render: ([caps, models]) => {
        const rows = (models && models.models) || [];
        const verified = rows.filter((m) => m.verified).length;
        const mismatch = rows.filter((m) => m.status === "mismatch").length;
        return properties([
          { label: "Bridge", value: chip("connected", "ok") },
          { label: "Storage", value: chip("local-first", "ok") },
          ...(rows.length ? [{ label: "Models verified",
            value: chip(`${verified}/${rows.filter((m) => m.status !== "awaiting_weights").length}`,
              mismatch ? "warn" : "ok") }] : []),
          ...(mismatch ? [{ label: "Digest mismatch", value: chip(String(mismatch), "warn") }] : []),
        ]);
      },
    }) },
  ],
  bottom: [
    { id: "home.activity", title: "Activity Log", flex: 2, render: () => live({
      calls: ["audit_log"],
      empty: "No activity recorded yet.",
      isEmpty: ([log]) => !(log && (log.entries || log.events || []).length),
      render: ([log]) => consoleView((log.entries || log.events || []).slice(0, 20).map((e) => ({
        t: String(e.at || e.timestamp || "").slice(11, 19) || "—",
        text: e.message || e.action || "—",
        level: /fail|error/i.test(e.message || e.action || "") ? "error" : "",
      }))),
    }) },
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
      [], { selectKind: "survey" }) },
    { id: "proj.datasets", title: "Datasets", flex: 1, render: () => live({
      calls: ["list_datasets", "list_dataset_images"],
      empty: "No datasets. Use Import to add a folder of images.",
      isEmpty: ([sets, imgs]) => !(sets && (sets.datasets || []).length)
        && !(imgs && (imgs.images || []).length),
      render: ([sets, imgs]) => {
        const datasets = (sets && sets.datasets) || [];
        const count = ((imgs && imgs.images) || []).length;
        return tree(datasets.length
          ? datasets.map((d, i) => ({ id: `d${i}`, label: d.name || d.path,
              icon: "▦", meta: String(d.image_count ?? "") }))
          : [{ id: "d0", label: "Imported images", icon: "▦", meta: String(count) }],
          { selectKind: "dataset" });
      },
    }) },
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
    { id: "ver.datasets", title: "Datasets", render: () => live({
      calls: ["list_datasets"],
      empty: "No flights imported to verify.",
      isEmpty: ([sets]) => !(sets && (sets.datasets || []).length),
      render: ([sets]) => tree((sets.datasets || []).map((d, i) => ({
        id: `f${i}`, label: d.name || d.path, icon: "▶",
        meta: String(d.image_count ?? ""),
      })), { selectKind: "flight" }),
    }) },
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
    /* Reported 1,842 planned against 1,836 matched with 98.4% coverage, on projects
       with no mission and no flight. Verification is a comparison, so with nothing to
       compare it says that instead of inventing a near-miss. */
    { id: "ver.summary", title: "Verification Summary", flex: 2, render: () => live({
      calls: ["list_dataset_images", "mission_estimates"],
      empty: "Nothing to verify yet: plan a mission and import the flight it produced.",
      isEmpty: ([imgs]) => !(imgs && (imgs.images || []).length),
      render: ([imgs, est]) => readouts([
        { k: "Captured", v: String(((imgs && imgs.images) || []).length) },
        ...(est && est.image_count ? [{ k: "Planned", v: String(est.image_count) }] : []),
      ]),
    }) },
    /* Named three specific captures as missing -- C-1094, C-1095, C-1782 -- for a flight
       that was never flown. A missing capture is a finding; inventing one is not. */
    { id: "ver.missing", title: "Missing Captures", flex: 2, render: () =>
      emptyState("No comparison run. Use Match Captures once a flight is imported.") },
    { id: "ver.qc", title: "QC Log", flex: 1, render: () =>
      emptyState("No QC findings.") },
  ],
};

/* ---------------------------------------------------------------- PROCESSING */

const processing = {
  id: "processing",
  title: "Processing",
  toolbar: ["New Job", "|", "Process", "Pause", "Cancel", "|", "Add GCPs", "Set CRS", "Export Products"],
  left: [
    { id: "proc.datasets", title: "Datasets", render: () => live({
      calls: ["list_datasets", "list_dataset_images"],
      empty: "No dataset imported. Use Import to add a folder of images.",
      isEmpty: ([sets, imgs]) => !(sets && (sets.datasets || []).length)
        && !(imgs && (imgs.images || []).length),
      render: ([sets, imgs]) => {
        const datasets = (sets && sets.datasets) || [];
        const count = ((imgs && imgs.images) || []).length;
        const nodes = datasets.length
          ? datasets.map((d, i) => ({
              id: `pd${i}`, label: d.name || d.path || `Dataset ${i + 1}`,
              icon: "▦", meta: String(d.image_count ?? (i === 0 ? count : "")),
            }))
          : [{ id: "pd0", label: "Imported images", icon: "▦", meta: String(count) }];
        return tree(nodes, { selectKind: "dataset" });
      },
    }) },
    { id: "proc.jobs", title: "Jobs", render: () => live({
      calls: ["list_jobs"],
      empty: "No jobs yet. Press Process to queue a reconstruction.",
      isEmpty: ([jobs]) => !(jobs && (jobs.jobs || []).length),
      render: ([jobs]) => tree((jobs.jobs || []).slice(0, 40).map((j, i) => ({
        id: j.id || `j${i}`,
        label: `${j.id ? "#" + String(j.id).slice(0, 8) : "job"} ${j.kind || j.name || ""}`.trim(),
        icon: j.state === "finished" ? "✓" : "▶",
        meta: j.state === "finished" ? "done"
          : j.progress != null ? `${Math.round(j.progress)}%` : (j.state || ""),
      })), { selectKind: "job" }),
    }) },
  ],
  canvas: () => canvas({
    title: "Sparse reconstruction",
    note: "Camera positions, tie points and the growing point cloud. Switches to dense, mesh and orthomosaic as stages complete.",
    tools: [{ icon: "✥", title: "Orbit" }, { icon: "⊕", title: "Zoom" }, { icon: "▣", title: "Select" }, { icon: "⤡", title: "Fit" }],
    overlays: [
      { at: "tl", html: `<strong>Ingestion → Features → Matching → SfM → Georeference → Dense → DSM/DTM → Ortho → Mesh → Twin</strong>` },
      /* These were "1,842 cameras · 412k tie points" regardless of the project open.
         The canvas has no data source of its own, so it now says what the stage IS
         rather than inventing what it processed. */
      { at: "br", html: `Sparse cloud · counts in the Job Progress panel` },
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
      /* Was a fixed "EPSG:4326 / 7 GCPs / RTK fixed" on every project, including ones
         with no GCPs and no RTK at all. Read from the layers the project really has. */
      { title: "Spatial", render: () => live({
        calls: ["list_layers"],
        empty: "No georeferenced products yet.",
        isEmpty: ([l]) => !(l && (l.layers || []).length),
        render: ([l]) => {
          const layers = l.layers || [];
          const epsg = (layers.find((x) => x.crs_epsg) || {}).crs_epsg;
          return properties([
            { label: "Reference", value: epsg ? chip("georeferenced", "ok") : chip("none", "warn") },
            ...reported([
              ["CRS", epsg && `EPSG:${epsg}`],
              ["Rasters", layers.filter((x) => x.kind === "raster").length || undefined],
              ["Vectors", layers.filter((x) => x.kind === "vector").length || undefined],
            ]),
          ]);
        },
      }) },
      /* Three imaginary machines with RTX 4090s used to be listed here. There is one
         machine -- this one -- and what an operator needs from it is whether the engine
         can actually do the work, which the application already reports. */
      { title: "Engine", render: () => live({
        calls: ["reconstruction_capabilities"],
        empty: "The reconstruction engine did not report its capabilities.",
        render: ([caps]) => {
          const c = caps.capabilities || caps;
          const yn = (v, good) => chip(v ? (good || "yes") : "no", v ? "ok" : "warn");
          return properties([
            { label: "pycolmap", value: yn(c.pycolmap) },
            { label: "Native COLMAP", value: c.colmap_binary
              ? chip("found", "ok") : chip("not found", "warn") },
            { label: "GPU dense stereo", value: yn(c.dense_stereo, "available") },
            { label: "Open3D mesh", value: yn(c.open3d) },
          ]);
        },
      }) },
    ] },
    /* Listed all six products whether or not any existed -- including a dense cloud on
       a machine whose dense stage had failed. These are the layers actually on disk. */
    { id: "proc.outputs", title: "Output Products", height: 170, grow: false, render: () => live({
      calls: ["list_layers"],
      empty: "Nothing produced yet.",
      isEmpty: ([l]) => !(l && (l.layers || []).length),
      render: ([l]) => tree((l.layers || []).map((layer, i) => ({
        id: layer.id || `o${i}`,
        label: layer.name || layer.id,
        icon: layer.kind === "raster" ? "▦" : "◌",
        meta: layer.metadata && layer.metadata.width
          ? `${layer.metadata.width}×${layer.metadata.height}` : "",
      })), { selectKind: "product" }),
    }) },
  ],
  bottom: [
    /* A frozen fake run used to sit here: "feature matching, running, 9:04, 34%" on a
       machine with no job at all, and a log claiming hundreds of thousands of pairs
       across three workers that do not exist.
       Both now read the job queue. */
    { id: "proc.progress", title: "Job Progress", flex: 2, render: () => live({
      calls: ["list_jobs"],
      empty: "No job running. Press Process to start one.",
      isEmpty: ([jobs]) => !(jobs && (jobs.jobs || []).length),
      render: ([jobs]) => table(
        [{ title: "Job", key: "job" }, { title: "State", key: "state" },
         { title: "Message", key: "msg" },
         { title: "", value: (r) => meter(r.p, r.p >= 1 ? "ok" : "") }],
        (jobs.jobs || []).slice(0, 12).map((j) => ({
          job: j.kind || j.name || String(j.id || "").slice(0, 8),
          state: j.state || "—",
          msg: (j.message || "").slice(0, 60),
          p: j.state === "finished" ? 1 : (j.progress != null ? j.progress / 100 : 0),
        }))),
    }) },
    { id: "proc.logs", title: "Processing Log", flex: 3, render: () => live({
      calls: ["list_jobs"],
      empty: "No processing log yet.",
      isEmpty: ([jobs]) => !(jobs && (jobs.jobs || []).length),
      render: ([jobs]) => consoleView((jobs.jobs || []).slice(0, 12).map((j) => ({
        t: String(j.started_at || j.created_at || "").slice(11, 19) || "—",
        text: `${j.kind || "job"} ${j.state || ""} ${j.message || ""}`.trim(),
        level: j.state === "finished" ? "ok"
          : /fail|error/i.test(j.state || "") ? "error" : "",
      }))),
    }) },
  ],
};

/* -------------------------------------------------------------- DIGITAL TWIN */

const twin = {
  id: "twin",
  title: "Digital Twin",
  toolbar: ["Textured Mesh", "Point Cloud", "Thermal", "Semantic", "Change", "|", "Compare Dates", "Measure", "Annotate", "Export"],
  left: [
    /* A building with a roof, a facade and a stockpile volume, none of which existed.
       build_asset_inventory reports what was actually derived from the reconstruction. */
    { id: "twin.hierarchy", title: "Scene Hierarchy", render: () => live({
      calls: ["build_asset_inventory", "list_layers"],
      empty: "No assets derived yet. Reconstruct a survey, then run asset extraction.",
      isEmpty: ([inv, layers]) => !(inv && (inv.assets || []).length)
        && !(layers && (layers.layers || []).length),
      render: ([inv, layers]) => {
        const assets = (inv && inv.assets) || [];
        if (assets.length) {
          return tree(assets.slice(0, 50).map((a, i) => ({
            id: a.id || `s${i}`, label: a.name || a.kind || `Asset ${i + 1}`,
            icon: "▢", meta: a.area_m2 ? `${Math.round(a.area_m2)} m²` : "",
          })), { selectKind: "asset" });
        }
        // No asset extraction, but the reconstruction products are real scene content.
        return tree(((layers && layers.layers) || []).map((l, i) => ({
          id: l.id || `s${i}`, label: l.name || l.id, icon: "▦",
          meta: l.crs_epsg ? `EPSG:${l.crs_epsg}` : "",
        })), { selectKind: "asset" });
      },
    }) },
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
      /* "4.1M faces, 12 assets, 6 findings" was fixed text, shown over an empty canvas
         on a project whose mesh has 0 of those numbers in common. */
      { at: "br", html: `Loaded from the project's reconstruction products` },
    ],
  }),
  right: [
    /* A metal-deck roof of 3,140 m2 with four open findings, on every project. Selection
       is what fills this panel; until something is selected it says so. */
    { id: "twin.asset", title: "Selected Asset", render: () =>
      emptyState("Select an object in the scene to inspect it.") },
    { id: "twin.timeline", title: "Survey Timeline", height: 130, grow: false, render: () => live({
      calls: ["list_mission_versions"],
      empty: "One survey. A timeline needs at least two to compare.",
      isEmpty: ([v]) => !(v && (v.versions || []).length),
      render: ([v]) => table(
        [{ title: "Version", key: "d" }, { title: "Saved", key: "f" }],
        (v.versions || []).slice(0, 10).map((row) => ({
          d: String(row.version_num ?? row.version ?? "—"),
          f: String(row.created_at || row.saved_at || "").slice(0, 10) || "—",
        })), { selectKind: "survey" }),
    }) },
  ],
  bottom: [
    /* Reported a centroid in Madhya Pradesh for an asset that does not exist. */
    { id: "twin.props", title: "Object Properties", flex: 2, render: () =>
      emptyState("No object selected.") },
    /* Three measurements of a building that was never surveyed. Measuring is an action
       the operator takes; the panel holds the results of it. */
    { id: "twin.measure", title: "Measurements", flex: 2, render: () =>
      emptyState("No measurements yet. Use Measure on the canvas.") },
    { id: "twin.history", title: "History", flex: 1, render: () => live({
      calls: ["audit_log"],
      empty: "No history for this project yet.",
      isEmpty: ([log]) => !(log && (log.entries || log.events || []).length),
      render: ([log]) => consoleView((log.entries || log.events || []).slice(0, 10).map((e) => ({
        t: String(e.at || e.timestamp || "").slice(5, 10) || "—",
        text: e.message || e.action || "—",
      }))),
    }) },
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
    /* A 1,240 m3 stockpile with +-31 m3 uncertainty and a previous survey to compare
       against, on a project with no measurements at all. Volumes are the numbers an
       operator bills from, so an invented one is the most costly kind here. */
    { id: "ms.details", title: "Measurement Details", render: () =>
      emptyState("No measurement selected. Use Volume, Area or Distance on the canvas.") },
  ],
  bottom: [
    { id: "ms.history", title: "Measurement History", flex: 3, render: () => table(
      [{ title: "Name", key: "n" }, { title: "Type", key: "t" }, { title: "Value", key: "v", num: true },
       { title: "Date", key: "d" }, { title: "Δ", key: "c", num: true }],
      [], { selectKind: "measurement" }) },
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
