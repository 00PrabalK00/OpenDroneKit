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

/* A readout in the corner of the canvas, asking the application for its numbers.
 *
 * Six overlays used to carry figures written into this file: the mission map's capture
 * count, distance, duration and GSD; the flight screen's progress, mode and fix quality;
 * the verification screen's matched and out-of-tolerance counts; the thermal screen's
 * anomaly count and ambient temperature; and the measurement screen's vertical accuracy.
 *
 * They survived every earlier fabrication sweep because those searched panel bodies, and
 * an overlay is not a panel. That is worth naming: the sweeps were not careless, they
 * were scoped to a shape, and the fabrication had another shape. An overlay sits in the
 * corner of the view, where every instrument an operator has used puts its readout, so a
 * fixed number there is read as a measurement of what is on screen more readily than a
 * table would be.
 *
 * The measurement one was the most serious. "Vertical accuracy +/-0.05 m -- nothing below
 * 0.10 m is reported" states the threshold that decides whether a deformation is real,
 * and there is no such constant: detection_floor() in core/deformation.py derives it per
 * comparison from each survey's own accuracy and the registration residual between them.
 * A fixed floor printed beside a measurement claims a precision the data may not support,
 * and it is exactly the figure a reader would cite to argue that a movement was real.
 */
function readout({ calls, isEmpty, empty, render }) {
  return live({
    calls, isEmpty, empty,
    render: (r) => el("div", { class: "canvas-readout" }, render(r)),
  });
}

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
    const active = (state && state.project) || {};
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
      isEmpty: ([state]) => !(state && state.project),
      render: ([state, layers, images]) => {
        const project = state.project || {};
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
      isEmpty: ([log]) => !(log && (log.events || []).length),
      render: ([log]) => consoleView((log.events || []).slice(0, 12).map((e) => ({
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
      isEmpty: ([log]) => !(log && (log.events || []).length),
      render: ([log]) => consoleView((log.events || []).slice(0, 20).map((e) => ({
        t: String(e.at || e.timestamp || "").slice(11, 19) || "—",
        text: e.message || e.action || "—",
        level: /fail|error/i.test(e.message || e.action || "") ? "error" : "",
      }))),
    }) },
    /* Three jobs at 34, 72 and 95 per cent on workers w-01 to w-03. This build runs
       jobs on threads in this process; there are no workers, and a Worker column
       describes a deployment that does not exist. */
    { id: "home.jobs", title: "Processing Queue", flex: 2, render: () => live({
      calls: ["list_jobs"],
      empty: "Nothing running. Import a dataset and press Process.",
      isEmpty: ([j]) => !(j && (j.jobs || []).length),
      render: ([j]) => table(
        [{ title: "Job", key: "name" }, { title: "State", key: "status" },
         { title: "Detail", key: "message" },
         { title: "Progress", value: (r) => meter((r.percent || 0) / 100,
             r.status === "failed" ? "error" : r.percent >= 100 ? "ok" : "") }],
        (j.jobs || []).slice(0, 50).map((job) => ({
          id: job.id, name: job.name, status: job.status,
          /* A failed job's error is the only thing worth reading on this row. */
          message: job.error || job.message || "",
          percent: job.percent || 0,
        })), { selectKind: "job" }),
    }) },
    /* Six aircraft, four available, one flying, one in service -- on a machine with
       no fleet registered. This is what fleet_status() is for: it returns exactly these
       counts, and always did. */
    { id: "home.summary", title: "Fleet", flex: 1, render: () => live({
      calls: ["fleet_status"],
      empty: "No fleet registered.",
      isEmpty: ([f]) => !f || f.aircraft === undefined,
      render: ([f]) => readouts([
        { k: "Aircraft", v: String(f.aircraft ?? 0) },
        { k: "Batteries", v: String(f.batteries ?? 0) },
        { k: "Retired", v: String(f.retired_batteries ?? 0),
          tone: (f.retired_batteries || 0) ? "warn" : "" },
        { k: "Pilots", v: String(f.pilots ?? 0) },
        /* The only one that asks for an action today. */
        { k: "Service due", v: String((f.service_due || []).length),
          tone: (f.service_due || []).length ? "warn" : "ok" },
      ]),
    }) },
  ],
};

/* ----------------------------------------------------------------- PROJECTS */

const projects = {
  id: "projects",
  title: "Projects",
  toolbar: ["New Project", "New Folder", "Import", "|", "Archive", "Export", "Share"],
  left: [
    { id: "proj.tree", title: "Projects", render: projectTree },
    /* One archived project that was never created. Nothing in this build archives a
       project, so the honest state is empty rather than a plausible entry. */
    { id: "proj.archived", title: "Archived", height: 110, grow: false,
      render: () => emptyState("No archived projects.") },
  ],
  canvas: () => canvas({ map: true, title: "Project extent", note: "Boundary, survey history and asset locations.", tools: MAP_TOOLS, overlays: [{ at: "br", html: COORD }] }),
  right: [
    { id: "proj.props", title: "Project Properties", tabs: [
      /* Read straight from the bundled example, so every project showed the same
         name, the same registered-image count, the same reprojection error and the same
         geo RMSE. Reprojection error and RMSE are the two numbers a surveyor quotes to
         say the work is good; they are not decoration. */
      { title: "General", render: () => live({
        calls: ["get_project", "list_layers"],
        empty: "No project open.",
        isEmpty: ([p]) => !(p && p.project),
        render: ([p, l]) => {
          const project = p.project;
          const layers = ((l && l.layers) || []).filter((x) => x.crs_epsg);
          const codes = [...new Set(layers.map((x) => x.crs_epsg))];
          return properties([
            { group: "Identity" },
            ...reported([
              ["Name", project.name],
              ["Created", String(project.created_at || "").slice(0, 10) || undefined],
              ["Root", project.root_dir],
            ]),
            { group: "Products" },
            ...reported([
              ["Georeferenced layers", layers.length || undefined],
              ["CRS", codes.length ? codes.map((c) => `EPSG:${c}`).join(", ") : undefined],
            ]),
            /* Accuracy is deliberately absent rather than zeroed. It comes from the
               reconstruction report and the control points, and a project that has
               neither has no accuracy to quote. */
            { label: "Accuracy", value: "reported on Verification, from control points" },
          ]);
        },
      }) },
      /* Labelled "Team" and listing MODELS, with columns Model and Measured. Whatever
         this once was, the tab title and its contents had come apart, and a person
         looking for who worked on a survey found a list of neural networks. There is no
         team concept in this build -- no users, no assignment -- so the tab is named for
         what it shows, and shows it from the registry rather than the example file. */
      { title: "Models", render: () => live({
        calls: ["verify_models"],
        empty: "No models installed.",
        isEmpty: ([r]) => !(r && (r.models || []).length),
        render: ([r]) => table(
          [{ title: "Model", key: "name" }, { title: "State", key: "state" }],
          (r.models || []).map((m) => ({ name: m.model_key, state: m.status })),
          { selectKind: "model" }),
      }) },
      /* Three chips -- warehouse, roof, quarterly -- that belonged to no project. */
      { title: "Tags", render: () => live({
        calls: ["list_annotation_tags"],
        empty: "No tags yet. Select findings and tag them to group this project's work.",
        isEmpty: ([t]) => !(t && (t.tags || []).length),
        render: ([t]) => el("div", { class: "panel-body pad" },
          (t.tags || []).map((tag) => chip(`${tag.tag} (${tag.count})`))),
      }) },
    ] },
  ],
  bottom: [
    /* Five column headings over zero rows. Not fabricated, but an empty grid reads as
       "this project has no surveys" when it really means "nothing was ever asked". The
       datasets in a project ARE its survey history, so it now says which. */
    { id: "proj.history", title: "Survey History", flex: 2, render: () => live({
      calls: ["list_datasets"],
      empty: "No surveys yet. Import a folder of images to start one.",
      isEmpty: ([d]) => !(d && (d.datasets || []).length),
      render: ([d]) => table(
        [{ title: "Survey", key: "name" }, { title: "Imported", key: "date" },
         { title: "Images", key: "images", num: true }],
        (d.datasets || []).map((set) => ({
          name: set.name || set.path,
          date: String(set.created_at || "").slice(0, 10) || "\u2014",
          images: set.image_count ?? "\u2014",
        })), { selectKind: "survey" }),
    }) },
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
    /* Three invented missions -- "Roof Block A v3", "Facade North v1", "Yard grid v2",
       one of them marked "current". This was the last panel in the cockpit still
       rendering a literal, and a bad one to leave: the planner's browser is a list of
       saved work, so picking a mission from it that does not exist is a click that can
       only fail.

       list_mission_versions("") returns every version in the project, newest first, so
       the browser shows each mission once at its highest version. */
    { id: "plan.browser", title: "Mission Browser", render: () => live({
      calls: ["list_mission_versions"],
      empty: "No saved missions in this project. Plan one, then Save.",
      isEmpty: ([v]) => !(v && (v.versions || []).length),
      render: ([v]) => {
        const latest = new Map();
        for (const row of v.versions || []) {
          const name = row.mission_name || "(unnamed)";
          const seen = latest.get(name);
          if (!seen || (row.version_num || 0) > (seen.version_num || 0)) latest.set(name, row);
        }
        return tree([...latest.values()].map((row, i) => ({
          id: `m${i}`,
          label: row.mission_name || "(unnamed)",
          icon: "\u25c7",
          meta: `v${row.version_num ?? 1}`,
        })), { selectKind: "mission" });
      },
    }) },
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
      /* Was a fixed "214 captures, 3.1 km, 18 min, GSD 1.8 cm", printed over whatever
         mission was open -- including one that had not been planned yet. */
      { at: "tr", node: readout({
        calls: ["mission_estimates"],
        empty: "Plan a mission to see captures, distance, duration and GSD.",
        isEmpty: ([e]) => !(e && e.estimates),
        render: ([e]) => {
          const est = e.estimates;
          return [
            chip(`${est.image_count ?? "\u2014"} captures`),
            chip(est.distance_m ? `${(est.distance_m / 1000).toFixed(2)} km` : "\u2014"),
            chip(est.duration_min ? `${Math.round(est.duration_min)} min` : "\u2014"),
            /* estimate_mission() leaves gsd_cm null when no camera is declared, and
               saying so beats printing a GSD the plan cannot support. */
            chip(est.gsd_cm ? `GSD ${Number(est.gsd_cm).toFixed(1)} cm` : "GSD not reported"),
          ];
        },
      }) },
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
    /* Six fixed numbers -- 14.2 ha, 3.1 km, 18 min, 214 images, 1 battery, 12.4 GB --
       shown whether or not a mission had been planned. "Batteries 1" is the one that
       sends someone to site with a single pack. */
    { id: "plan.estimates", title: "Estimates", height: 132, grow: false, render: () => live({
      calls: ["mission_estimates"],
      empty: "Plan a mission to estimate its distance, duration, batteries and storage.",
      isEmpty: ([e]) => !(e && e.estimates),
      render: ([e]) => {
        const est = e.estimates;
        const battery = est.battery || {};
        const storage = est.storage || {};
        return readouts([
          { k: "Images", v: String(est.image_count ?? "\u2014") },
          { k: "Distance", v: est.distance_m ? `${(est.distance_m / 1000).toFixed(2)} km` : "\u2014" },
          { k: "Duration", v: est.duration_min ? `${Math.round(est.duration_min)} min` : "\u2014" },
          /* Batteries REQUIRED, and flagged when it is more than one, because that
             changes what has to be in the car. */
          { k: "Batteries", v: String(battery.batteries_required ?? "\u2014"),
            tone: battery.fits_in_one_flight === false ? "warn" : "" },
          { k: "Storage", v: storage.total_gb !== undefined ? `${storage.total_gb} GB` : "\u2014",
            /* A storage figure computed for an unknown camera is a guess about file
               size, and the operator should know which kind of number they have. */
            tone: storage.known_camera === false ? "warn" : "" },
        ]);
      },
    }) },
  ],
  bottom: [
    { id: "plan.timeline", title: "Simulation Timeline", flex: 3, render: () => canvas({
      title: "Altitude and speed over time", note: "Scrub to preview the flight. Playback follows the aircraft along the route.",
    }) },
    /* Four segments with capture counts, for a mission that had not been planned.
       plan_battery_segments() computes the real split -- sorties sized so each fits
       inside one battery with its reserve intact. */
    { id: "plan.segments", title: "Mission Segments", flex: 2, render: () => live({
      calls: ["plan_battery_segments"],
      empty: "Plan a mission to see how it splits across batteries.",
      isEmpty: ([b]) => !(b && b.splits),
      render: ([b]) => el("div", {}, [
        table(
          [{ title: "#", key: "n", num: true },
           { title: "First capture", key: "first", num: true },
           { title: "Last capture", key: "last", num: true },
           { title: "Captures", key: "caps", num: true }],
          (b.splits || []).map((seg) => ({
            n: seg.segment, first: seg.first_capture,
            last: seg.last_capture, caps: seg.capture_count,
          })), { selectKind: "segment" }),
        /* The note says segment boundaries are estimates from capture count and to fly
           to the aircraft's own battery warning instead. Dropping it would leave a
           table of numbers that look surveyed. */
        b.note ? el("div", { class: "panel-body pad muted", text: b.note }) : null,
      ]),
    }) },
    /* Three validation results for a plan nobody checked.
       "geofence contains every capture point -- ok" is a safety check reported as
       passed without having run, which is the preflight checklist's defect on the
       planning screen. The middle line was even true by accident, which is worse: two
       thirds of the panel was right, so the whole of it read as a real check.

       Nothing validates a plan on render, and it should not -- validation is an action
       with a cost. So the panel reports what the plan really contains, and names the
       button that runs the checks. */
    { id: "plan.log", title: "Validation", flex: 2, render: () => live({
      calls: ["mission_estimates", "terrain_coverage"],
      empty: "No mission planned. Plan one, then press Validate to check it.",
      isEmpty: ([e]) => !(e && e.estimates),
      render: ([e, terrain]) => {
        const battery = (e.estimates || {}).battery || {};
        const storage = (e.estimates || {}).storage || {};
        const lines = [];
        /* Warnings the estimator produced are real findings about this plan. */
        for (const warning of battery.warnings || []) {
          lines.push({ t: "plan", text: warning, level: "warn" });
        }
        if (storage.known_camera === false) {
          lines.push({ t: "plan", text:
            "Storage is estimated for an unrecognised camera, so file sizes are assumed.",
            level: "warn" });
        }
        /* Terrain is the one that changes what altitude means, so it is stated either
           way rather than only when missing. */
        const covered = terrain && terrain.covered;
        lines.push(covered
          ? { t: "terrain", text: "Terrain model loaded; altitudes follow the ground.", level: "ok" }
          : { t: "terrain", text:
              "No terrain model loaded: planned altitudes are above a flat plane.",
              level: "warn" });
        lines.push({ t: "\u2014", text:
          "Geofence, airspace and payload checks run when you press Validate.", level: "" });
        return consoleView(lines);
      },
    }) },
  ],
};

/* ------------------------------------------------------------------- FLIGHT */

const flight = {
  id: "flight",
  title: "Flight",
  toolbar: ["Preflight", "|", "Start", "Pause", "Resume", "Capture Now", "|", "Manual Override", "RTL", "Land", "Abort"],
  left: [
    /* An M350 RTK, linked, in AUTO, armed -- with nothing connected. "Armed" is a
       statement about whether the propellers will turn. */
    { id: "fly.aircraft", title: "Aircraft", render: () => live({
      calls: ["telemetry"],
      empty: "No aircraft connected.",
      isEmpty: ([t]) => !(t && t.telemetry && t.telemetry.connected),
      render: ([t]) => {
        const tm = t.telemetry;
        return properties([
          { label: "Driver", value: tm.driver || "\u2014" },
          /* A simulated vehicle must never be mistaken for a real one on the screen
             that arms it. */
          ...(tm.is_simulated ? [{ label: "Vehicle", value: chip("SIMULATED", "warn") }] : []),
          { label: "Mode", value: chip(tm.flight_mode || "unknown", "info") },
          { label: "Armed", value: chip(tm.armed ? "yes" : "no", tm.armed ? "warn" : "ok") },
          { label: "Link", value: tm.link_quality_pct !== undefined
            ? chip(`${Math.round(tm.link_quality_pct)}%`, tm.link_quality_pct > 70 ? "ok" : "warn")
            : chip("not reported") },
        ]);
      },
    }) },
    /* Nine items, eight pre-ticked, checked against nothing.
       A checklist exists to be believed. One that is ticked before anything was
       verified is worse than no checklist at all, because an operator who would
       otherwise have checked now has a reason not to. It also read "Storage 0 GB free"
       with a tick beside it.

       Every item here is derived from what the vehicle reports, so it is now an actual
       preflight check. Items the telemetry cannot speak to are shown as unknown rather
       than passed -- an unanswered check is not a passed one. */
    { id: "fly.checklist", title: "Preflight", render: () => live({
      calls: ["telemetry"],
      empty: "No aircraft connected. Preflight is checked against the aircraft.",
      isEmpty: ([t]) => !(t && t.telemetry && t.telemetry.connected),
      render: ([t]) => {
        const tm = t.telemetry;
        /* pass / fail / unknown, never a bare tick. */
        const mark = (ok) => (ok === null ? "\u25cb" : ok ? "\u2713" : "\u2717");
        const items = [
          ["Autopilot link", tm.connected === true],
          [`GPS fix \u2014 ${["no GPS", "no fix", "2D", "3D", "RTK float", "RTK fixed"][tm.gps_fix] || "?"}`,
           tm.gps_fix >= 3],
          [`Satellites ${tm.satellites ?? 0}`, (tm.satellites ?? 0) >= 6],
          [`HDOP ${tm.hdop ?? "?"}`, tm.hdop !== undefined ? tm.hdop < 2.0 : null],
          ["Home position set", tm.home_set === true],
          [`Battery ${Math.round(tm.battery_pct ?? 0)}%`, (tm.battery_pct ?? 0) >= 50],
          ["Mission uploaded", tm.waypoint_total ? tm.waypoint_total > 0 : false],
          [`Link ${tm.link_quality_pct !== undefined ? Math.round(tm.link_quality_pct) + "%" : "unknown"}`,
           tm.link_quality_pct !== undefined ? tm.link_quality_pct > 50 : null],
          /* Not derivable from telemetry, and deliberately left open rather than
             quietly dropped: an operator has to acknowledge it. */
          ["Weather acknowledged", null],
        ];
        return tree(items.map(([label, ok], i) => ({
          id: `c${i}`, label, icon: mark(ok),
          meta: ok === null ? "unknown" : ok ? "" : "check",
        })));
      },
    }) },
    /* Two missions that do not exist, one of them marked active. */
    { id: "fly.queue", title: "Mission Queue", height: 110, grow: false, render: () => live({
      calls: ["list_mission_versions"],
      empty: "No saved missions to fly.",
      isEmpty: ([v]) => !(v && (v.versions || []).length),
      render: ([v]) => {
        const latest = new Map();
        for (const row of v.versions || []) {
          const name = row.mission_name || "(unnamed)";
          const seen = latest.get(name);
          if (!seen || (row.version_num || 0) > (seen.version_num || 0)) latest.set(name, row);
        }
        return tree([...latest.values()].map((row, i) => ({
          id: `q${i}`, label: row.mission_name || "(unnamed)", icon: "\u25b8",
          meta: `v${row.version_num ?? 1}`,
        })), { selectKind: "mission" });
      },
    }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Live flight",
    note: "Aircraft position and heading, completed and remaining route, geofence, rally points and obstacles.",
    tools: MAP_TOOLS,
    overlays: [
      /* "62 / 214 captures, AUTO, RTK fixed" over an aircraft that was not connected.
         A fix-quality chip is what an operator reads to decide whether the survey is
         worth flying at all, so inventing one is worse than an empty corner. */
      { at: "tr", node: readout({
        calls: ["telemetry"],
        empty: "No aircraft connected.",
        /* session.telemetry() returns {connected: false, reason} rather than nothing when
           there is no vehicle, so a plain null check would treat "not connected" as live
           data and render a disarmed aircraft at 0% battery with no GPS. */
        isEmpty: ([t]) => !(t && t.telemetry && t.telemetry.connected),
        render: ([t]) => {
          const tm = t.telemetry;
          /* gps_fix is the integer MAVLink reports. The difference between 4 and 5 is the
             difference between decimetre and centimetre work, so it is named exactly
             rather than flattened to good/bad. */
          const FIX = ["no GPS", "no fix", "2D fix", "3D fix", "RTK float", "RTK fixed"];
          const fix = FIX[tm.gps_fix] || `fix ${tm.gps_fix}`;
          return [
            chip(tm.flight_mode || "mode unknown"),
            chip(`${fix} \u00b7 ${tm.satellites ?? 0} sats`, tm.gps_fix >= 3 ? "ok" : "warn"),
            chip(`battery ${Math.round(tm.battery_pct ?? 0)}%`,
                 (tm.battery_pct ?? 0) < 25 ? "warn" : "ok"),
            ...(tm.waypoint_total
              ? [chip(`waypoint ${tm.waypoint_index ?? 0} / ${tm.waypoint_total}`)]
              : []),
          ];
        },
      }) },
      /* Segment progress is real, but linked_mission_progress() computes it from a
         folder of captured images the operator has to point at. Until then this says so,
         rather than claiming a sortie is on the first of four segments. */
      { at: "bl", html: "Segment progress is reported once captured images are linked to the plan." },
      { at: "br", html: COORD },
    ],
  }),
  right: [
    /* Twelve readouts of an aircraft in flight, none of them from an aircraft. */
    { id: "fly.telemetry", title: "Telemetry", render: () => live({
      calls: ["telemetry"],
      empty: "No aircraft connected.",
      isEmpty: ([t]) => !(t && t.telemetry && t.telemetry.connected),
      render: ([t]) => {
        const tm = t.telemetry;
        const n = (v, digits = 1) => (v === undefined || v === null ? null : Number(v).toFixed(digits));
        /* Only what the vehicle reported. A missing field is dropped rather than
           shown as zero: "Alt AGL 0.0 m" on a flying aircraft is a lie, and a
           readout that is absent is obviously absent. */
        const rows = [
          ["Battery", tm.battery_pct !== undefined ? `${Math.round(tm.battery_pct)}%` : null,
           (tm.battery_pct ?? 100) < 25 ? "warn" : "ok"],
          ["Voltage", n(tm.battery_v) ? `${n(tm.battery_v)} V` : null],
          ["Alt AGL", n(tm.altitude_rel_m) ? `${n(tm.altitude_rel_m)} m` : null],
          ["Alt AMSL", n(tm.altitude_abs_m) ? `${n(tm.altitude_abs_m)} m` : null],
          ["Speed", n(tm.speed_mps) ? `${n(tm.speed_mps)} m/s` : null],
          ["Heading", n(tm.heading_deg, 0) ? `${n(tm.heading_deg, 0)}\u00b0` : null],
          ["Sats", tm.satellites !== undefined ? String(tm.satellites) : null,
           (tm.satellites ?? 0) >= 6 ? "ok" : "warn"],
          ["HDOP", n(tm.hdop, 2)],
          ["Link", tm.link_quality_pct !== undefined ? `${Math.round(tm.link_quality_pct)}%` : null],
          ["Next wp", tm.distance_to_next_m !== undefined ? `${n(tm.distance_to_next_m)} m` : null],
        ].filter(([, v]) => v !== null && v !== undefined);
        return readouts(rows.map(([k, v, tone]) => ({ k, v, tone })));
      },
    }) },
    /* Two progress bars at fixed fractions -- 62 of 214 captures, 0.9 of 3.1 km --
       which moved for nobody and meant nothing. */
    { id: "fly.progress", title: "Mission Progress", height: 120, grow: false, pad: true,
      render: () => live({
        calls: ["telemetry"],
        empty: "No aircraft connected.",
        isEmpty: ([t]) => !(t && t.telemetry && t.telemetry.connected
                            && t.telemetry.waypoint_total),
        render: ([t]) => {
          const tm = t.telemetry;
          const done = tm.waypoint_index ?? 0;
          const total = tm.waypoint_total;
          return el("div", {}, [
            el("div", { class: "field" }, [
              el("label", { text: "Waypoints" }),
              el("span", { text: `${done} / ${total}` }),
            ]),
            meter(total ? done / total : 0),
          ]);
        },
      }) },
    /* A wind-gust reading, at a timestamp, from no anemometer. */
    { id: "fly.alerts", title: "Alerts", height: 100, grow: false, render: () => live({
      calls: ["list_notifications"],
      empty: "No alerts.",
      isEmpty: ([n]) => !(n && (n.notifications || []).length),
      render: ([n]) => consoleView((n.notifications || []).slice(0, 40).map((item) => ({
        t: String(item.created_utc || "").slice(11, 19) || "\u2014",
        text: item.detail ? `${item.title} — ${item.detail}` : (item.title || ""),
        level: item.level === "error" ? "error" : item.level === "warning" ? "warn" : "",
      }))),
    }) },
  ],
  bottom: [
    { id: "fly.charts", title: "Telemetry Charts", flex: 3, render: () => canvas({ title: "Altitude · speed · battery · link", note: "Rolling window over the current flight." }) },
    /* "armed", "mission started", "waypoint 31 reached", "capture 62 triggered" -- a
       flight record for a flight that never happened. The audit log is the real one. */
    { id: "fly.events", title: "Event Log", flex: 2, render: () => live({
      calls: ["audit_log"],
      empty: "No flight events recorded yet.",
      isEmpty: ([a]) => !(a && (a.events || []).length),
      render: ([a]) => consoleView((a.events || []).slice(0, 60).map((e) => ({
        t: String(e.created_at || "").slice(11, 19) || "\u2014",
        text: `${e.event_type || ""} ${e.payload_json && e.payload_json !== "{}" ? e.payload_json : ""}`.trim(),
      }))),
    }) },
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
    /* Eight filenames generated in a loop -- DJI_1000.JPG to DJI_1007.JPG -- which
       looked exactly like a real card dump and belonged to no dataset. Selecting one
       asked the application for an image that was not there. */
    { id: "ver.images", title: "Images", render: () => live({
      calls: ["list_dataset_images"],
      empty: "No dataset selected. Import a folder of images to verify them.",
      isEmpty: ([d]) => !(d && (d.images || []).length),
      render: ([d]) => tree((d.images || []).map((name, i) => ({
        id: `i${i}`, label: name, icon: "\u25a3",
      })), { selectKind: "image" }),
    }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Planned against actual",
    note: "Planned capture positions, actual positions, deviation vectors and coverage footprints.",
    tools: MAP_TOOLS,
    overlays: [
      /* "1836 matched / 3 out of tolerance / 3 missing" -- an accuracy verdict on
         control points that were never imported. This is the screen whose entire job is
         to say whether the survey met its tolerance, so a fixed answer defeats it. */
      { at: "tr", node: readout({
        calls: ["gcp_accuracy_report"],
        empty: "Import ground control points to check the survey against them.",
        isEmpty: ([g]) => !(g && g.report),
        render: ([g]) => {
          const r = g.report;
          /* accuracy_report() returns rmse_m: null when no residuals could be computed,
             and its own warnings say not to quote an accuracy in that case. Rendering
             that as "RMSE 0.000 m" would turn "this survey has no measured accuracy"
             into the best result the screen can show. */
          if (r.rmse_m === null || r.rmse_m === undefined) {
            return [chip(`${r.point_count ?? 0} control points \u00b7 no measured accuracy`, "warn")];
          }
          return [
            chip(`${r.used} of ${r.point_count} used`, "ok"),
            chip(`RMSE ${Number(r.rmse_m).toFixed(3)} m`, r.meets_survey_grade ? "ok" : "warn"),
            ...(r.outlier_count
              ? [chip(`${r.outlier_count} outlier${r.outlier_count === 1 ? "" : "s"}`, "error")]
              : []),
          ];
        },
      }) },
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
      /* Read `est.image_count`, but mission_estimates() answers {estimates: {...}} --
         so the planned count was always undefined and the "Planned" readout never
         rendered at all. The panel showed only "Captured", which looks like a panel
         with one row rather than a broken comparison, and Verification exists to make
         exactly that comparison. Found by the general key guard, not by looking. */
      render: ([imgs, est]) => {
        const planned = ((est && est.estimates) || {}).image_count;
        const captured = ((imgs && imgs.images) || []).length;
        return readouts([
          { k: "Captured", v: String(captured) },
          ...(planned ? [{ k: "Planned", v: String(planned) }] : []),
          /* The difference is the finding. Naming it costs nothing and is the number
             an operator is actually looking for. */
          ...(planned ? [{ k: "Difference", v: String(captured - planned),
                           tone: captured < planned ? "warn" : "ok" }] : []),
        ]);
      },
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
  /* Pause was here and resolves to the VEHICLE verb: pressing it on the Processing
     screen prompted "Pause the aircraft?" and sent a flight command while a
     reconstruction was running. There is no pause_job in the Api either -- a job can be
     cancelled and nothing else -- so the button had nothing behind it in both directions. */
  toolbar: ["New Job", "|", "Process", "Cancel", "|", "Add GCPs", "Set CRS", "Export Products"],
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
      isEmpty: ([log]) => !(log && (log.events || []).length),
      render: ([log]) => consoleView((log.events || []).slice(0, 10).map((e) => ({
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
  /* Validate resolves to verify_site -- "check the site against the plan" -- which is the
     Mission Planning meaning of the word. On this screen it should move a finding's
     status, which is what Accept already does. One label cannot carry two verbs. */
  toolbar: ["Run AI", "|", "Accept", "Reject", "Edit", "Flag", "|", "Open in 3D", "Generate Report"],
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
    { title: "Source image", note: "The frame the selected finding was detected in" },
    { title: "Detection", note: "Mask and confidence at full resolution" },
    { title: "3D context", note: "Back-projected onto the surface it belongs to" },
  ]),
  right: [
    /* An invented crack: 0.91 confidence, 1.42 m long, on a roof that does not exist, at
       a coordinate in Madhya Pradesh, attributed to a real model with a real digest. The
       provenance being real is what made it dangerous -- it read as a measurement with a
       audit trail. Selection is what fills this panel. */
    { id: "ai.finding", title: "Finding Details", render: () =>
      emptyState("Select a finding to see its class, severity, provenance and location.") },
  ],
  bottom: [
    /* Four invented findings used to sit here -- F-118 through F-112, on a "Roof --
       Block A" that does not exist -- with confidences to two decimal places. They
       survived the earlier sweep because that test looked for specific strings and this
       table used different ones, which is a fair argument for the sweep having been too
       narrow rather than for these having been acceptable.

       These are the project's own annotations. A finding a person drew and one a model
       proposed are different kinds of evidence, so the table says which. */
    { id: "ai.findings", title: "Findings", flex: 4, render: () => live({
      calls: ["find_annotations"],
      empty: "No findings yet. Run AI, or annotate an image.",
      isEmpty: ([found]) => !(found && (found.annotations || []).length),
      render: ([found]) => table(
        [{ title: "Finding", key: "id" }, { title: "Class", key: "cls" },
         { title: "Severity", key: "sev" }, { title: "Source", key: "src" },
         { title: "Status", key: "status" }],
        (found.annotations || []).slice(0, 200).map((a) => ({
          id: String(a.id || "").slice(0, 8),
          cls: a.label || "\u2014",
          sev: a.severity || "\u2014",
          src: a.created_by === "user" ? "human" : "model",
          status: a.status || "\u2014",
        })), { selectKind: "finding" }),
    }) },
    /* The class list was crack / corrosion / ponding -- three of the labels the sample
       table happened to use. Filtering is now done on the tags the project actually
       carries, which is what an inspector groups findings by, and the list is read from
       the project rather than written here. Minimum confidence was removed: annotations
       do not store a confidence, so the control filtered nothing. */
    { id: "ai.filters", title: "Filter by tag", flex: 1, render: () => live({
      calls: ["list_annotation_tags"],
      empty: "No tags yet. Select findings and tag them to group by elevation, "
             + "reflight or whatever the job needs.",
      isEmpty: ([tags]) => !(tags && (tags.tags || []).length),
      render: ([tags]) => tree((tags.tags || []).map((t, i) => ({
        id: `tag${i}`, label: t.tag, icon: "\u25c7", meta: String(t.count),
      })), { selectKind: "tag" }),
    }) },
  ],
};

/* ------------------------------------------------------------------ THERMAL */

const thermal = {
  id: "thermal",
  title: "Thermal",
  toolbar: ["RGB", "Thermal", "Fused", "3D Thermal", "|", "Radiometric", "Compare", "Export"],
  left: [
    /* Two blocks and two strings with module counts, for a solar site nobody surveyed.
       An array inventory is built by build_asset_inventory() from detected instances,
       and it refuses instances with no location, confidence or model digest -- because
       a module count is a claim someone invoices against. Inventing one here bypassed
       every one of those refusals. */
    { id: "th.arrays", title: "Array Inventory", render: () => live({
      calls: ["find_annotations"],
      empty: "No array inventory. Run AI over thermal imagery to detect modules, then "
             + "build an inventory from the detections.",
      isEmpty: ([f]) => !(f && (f.annotations || []).length),
      render: ([f]) => {
        /* Grouped by the tag an inspector used, since that is the only grouping this
           build really records. No tags means one flat list, not an invented hierarchy. */
        const groups = new Map();
        for (const a of f.annotations || []) {
          const key = (a.tags && a.tags[0]) || "untagged";
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(a);
        }
        return tree([...groups.entries()].map(([name, items], i) => ({
          id: `ar${i}`, label: name, icon: "\u25a6", meta: String(items.length),
          children: items.slice(0, 50).map((a, k) => ({
            id: `st${i}-${k}`, label: a.label || "(unlabelled)", icon: "\u25a4",
            meta: a.severity || "",
          })),
        })), { selectKind: "array" });
      },
    }) },
  ],
  canvas: () => canvas({
    map: true,
    title: "Thermal map",
    note: "Module boundaries, anomaly overlays and temperature labels over the array.",
    tools: MAP_TOOLS,
    overlays: [
      /* "Fused, 7 anomalies" on a project with no thermal imagery loaded. */
      { at: "tr", node: readout({
        calls: ["find_annotations"],
        empty: "No thermal anomalies recorded.",
        isEmpty: ([f]) => !(f && (f.annotations || []).length),
        render: ([f]) => {
          /* An earlier version of this filter also tested `a.temperature_c`, which
             Annotation does not have -- I added it while fixing the overlays and it was
             always undefined, so the clause did nothing. Exactly the failure this file
             keeps finding, committed by the person finding it. Annotations carry a
             label and a severity; a temperature is not among them. */
          const thermal = (f.annotations || []).filter((a) =>
            String(a.label || "").toLowerCase().includes("thermal")
            || String(a.label || "").toLowerCase().includes("hot"));
          return [chip(`${thermal.length} thermal ${thermal.length === 1 ? "finding" : "findings"}`,
                       thermal.length ? "warn" : "ok")];
        },
      }) },
      /* Ambient temperature is the reference every relative thermal reading is taken
         against, so a fixed 34.2 C would silently rescale every anomaly on screen. It
         comes from the imagery's own metadata or it is not known. */
      { at: "br", node: readout({
        calls: ["thermal_palettes"],
        empty: "Load radiometric imagery to set a palette and temperature range.",
        isEmpty: ([p]) => !(p && (p.palettes || []).length),
        render: ([p]) => [chip(`${(p.palettes || []).length} palettes available`)],
      }) },
    ],
  }),
  right: [
    /* The F-118 pattern, in its worst form.
       Module M-1-A-14, a Hot-Spot at 0.70 confidence, peak 68.4 C, delta T 34.2 K --
       none of it measured -- presented directly above the REAL provenance of a real
       model: solar_thermal_anomaly_classifier, balanced accuracy 0.724, soiling recall
       0.367 marked weak. Those three figures are true and come from the registry, which
       is precisely what made the four above them read as audited measurements.
       Truthful provenance attached to an invented reading is worse than either alone.

       Nothing in this build measures a module's peak temperature: there is a thermal
       classifier and a palette scaler, and no per-module radiometric analysis. So the
       panel shows the finding that is selected, and the model card stays -- separately,
       and about the model rather than about a module. */
    { id: "th.module", title: "Selected Module", render: () => live({
      calls: ["find_annotations", "model_status"],
      empty: "Select a thermal finding to see its class, severity and provenance.",
      isEmpty: ([f]) => !(f && (f.annotations || []).length),
      render: ([f, models]) => {
        const chosen = selection.get("array") || selection.get("finding");
        const all = f.annotations || [];
        const found = (chosen && all.find((a) => a.label === chosen.label)) || null;
        const card = ((models && models.models) || [])
          .find((m) => String(m.key || "").includes("thermal"));
        return properties([
          ...(found ? [
            { group: String(found.id || "").slice(0, 8) },
            ...reported([
              ["Class", found.label],
              ["Severity", found.severity],
              ["Status", found.status],
              ["Source", found.created_by === "user" ? "drawn by a person" : found.created_by],
              ["Image", found.source_id],
            ]),
          ] : [{ label: "Finding", value: "none selected" }]),
          ...(card ? [
            { group: "Model available for this class" },
            /* model_status() rows are {key, exists, path, spec}. The first draft read
               `present`, which is not one of them and would have printed "yes" for a
               model that is not installed -- the exact opposite of what this row is for. */
            ...reported([
              ["Model", card.key],
              ["Installed", card.exists ? "yes" : "no"],
            ]),
          ] : []),
        ]);
      },
    }) },
  ],
  bottom: [
    /* Four anomaly classes with counts and peak temperature differences. The Peak
       delta-T column is the problem: nothing in this build computes one, so the column
       could only ever have been invented, and it is the number that decides whether a
       hot-spot is a defect or a reflection. Counts by class and severity are real and
       come from the findings; the temperature column is gone. */
    { id: "th.summary", title: "Anomaly Summary", flex: 2, render: () => live({
      calls: ["find_annotations"],
      empty: "No anomalies recorded.",
      isEmpty: ([f]) => !(f && (f.annotations || []).length),
      render: ([f]) => {
        const byClass = new Map();
        for (const a of f.annotations || []) {
          const key = a.label || "(unlabelled)";
          const row = byClass.get(key) || { c: key, n: 0, severe: 0, human: 0 };
          row.n += 1;
          if (a.severity === "severe" || a.severity === "critical") row.severe += 1;
          if (a.created_by === "user") row.human += 1;
          byClass.set(key, row);
        }
        return table(
          [{ title: "Class", key: "c" }, { title: "Count", key: "n", num: true },
           { title: "Severe", key: "severe", num: true },
           /* A finding a person drew and one a model proposed are different kinds of
              evidence and a summary that merges them hides which is which. */
           { title: "Drawn by a person", key: "human", num: true }],
          [...byClass.values()], { selectKind: "anomaly" });
      },
    }) },
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
      /* The worst of the six. There is no fixed vertical accuracy and no fixed
         reporting floor: detection_floor() derives it per comparison from each survey's
         accuracy and the registration residual between them. */
      { at: "tr", html: "Detection floor is computed per comparison, from each survey's "
        + "accuracy and the registration residual between them." },
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
    /* Four aircraft that do not exist, one of them "flying". */
    { id: "fl.aircraft", title: "Aircraft", render: () => live({
      calls: ["list_fleet"],
      empty: "No aircraft registered. Add Aircraft to start a fleet.",
      isEmpty: ([f]) => !(f && (f.aircraft || []).length),
      render: ([f]) => tree((f.aircraft || []).map((a) => ({
        id: `a${a.id}`, label: a.name, icon: "\u25b2",
        /* Hours remaining until service, because that is the number that decides
           whether this airframe flies today. null means no interval is configured,
           which is not the same as "not due". */
        meta: a.hours_to_service === null ? a.status
          : a.hours_to_service <= 0 ? "service due"
          : `${a.hours_to_service} h`,
      })), { selectKind: "aircraft" }),
    }) },
    /* Two named people who do not work here, with flight hours. */
    { id: "fl.pilots", title: "Pilots", render: () => live({
      calls: ["list_fleet"],
      empty: "No pilots registered.",
      isEmpty: ([f]) => !(f && (f.pilots || []).length),
      render: ([f]) => tree((f.pilots || []).map((p) => ({
        id: `p${p.id}`, label: p.display_name || "(unnamed)", icon: "\u25c9",
        meta: `${p.flight_hours} h`,
      })), { selectKind: "pilot" }),
    }) },
  ],
  canvas: () => canvas({ map: true, title: "Fleet map", note: "Aircraft locations, assignments and operational state.", tools: MAP_TOOLS, overlays: [{ at: "br", html: COORD }] }),
  right: [
    /* A serial number, a firmware version and a service interval for an airframe
       that does not exist. "Next service in 7.4 h" is an airworthiness statement. */
    { id: "fl.details", title: "Aircraft Details", render: () => live({
      calls: ["list_fleet"],
      empty: "Select an aircraft to see its serial, firmware, hours and service state.",
      isEmpty: ([f]) => !(f && (f.aircraft || []).length),
      render: ([f]) => {
        const chosen = selection.get("aircraft");
        const all = f.aircraft || [];
        const a = (chosen && all.find((x) => `a${x.id}` === chosen.id)) || all[0];
        const due = a.hours_to_service;
        return properties([
          { group: a.name },
          ...reported([
            ["Model", a.model],
            ["Serial", a.serial_number],
            ["Firmware", a.firmware],
            ["Flight hours", a.flight_hours],
            ["Flights", a.flight_count || undefined],
          ]),
          { label: "Next service", value: due === null
            ? chip("no interval set", "warn")
            : due <= 0 ? chip("overdue", "error") : chip(`in ${due} h`, due < 10 ? "warn" : "ok") },
          { label: "State", value: chip(a.status || "unknown", "info") },
        ]);
      },
    }) },
  ],
  bottom: [
    /* Three packs with charge, cycle counts, health and cell temperatures. Nothing in
       this build reads a battery's charge or temperature -- those come off the pack over
       a link the desktop app does not have -- so the columns are the ones the record
       really holds. Inventing a temperature for a lithium pack is not a neutral
       placeholder. */
    { id: "fl.batteries", title: "Batteries", flex: 3, render: () => live({
      calls: ["list_fleet"],
      empty: "No batteries registered. Add Battery to track cycles and health.",
      isEmpty: ([f]) => !(f && (f.batteries || []).length),
      render: ([f]) => table(
        [{ title: "Serial", key: "s" }, { title: "Cycles", key: "n", num: true },
         { title: "Limit", key: "l", num: true }, { title: "Health", key: "h", num: true },
         { title: "State", key: "st" }],
        (f.batteries || []).map((b) => ({
          s: b.serial_number || `#${b.id}`,
          n: b.cycle_count,
          l: b.cycle_limit || "\u2014",
          h: b.health_pct === null ? "\u2014" : `${b.health_pct}%`,
          /* Retired is a decision someone made; over its cycle limit and still in
             service is the state worth surfacing. */
          st: b.retired ? "retired" : b.over_cycle_limit ? "past cycle limit" : "in service",
        })), { selectKind: "battery" }),
    }) },
    /* Two maintenance records for aircraft that do not exist. A maintenance log is
       the document an operator points at to say the airframe was airworthy. */
    { id: "fl.maint", title: "Maintenance", flex: 2, render: () => live({
      calls: ["list_fleet"],
      empty: "No maintenance recorded. Log Maintenance writes a record and resets the "
             + "aircraft's service clock.",
      isEmpty: ([f]) => !(f && (f.maintenance || []).length),
      render: ([f]) => consoleView((f.maintenance || []).map((m) => ({
        t: String(m.performed_at || "").slice(0, 10),
        text: `${m.aircraft} \u2014 ${m.kind}`
          + (m.description ? `: ${m.description}` : "")
          + ` (at ${m.hours_at_service} h)`,
      }))),
    }) },
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
      /* The options are the report types core/report_engine.py actually builds. They
         used to read Inspection / Survey / Progress / Thermal, none of which the engine
         has ever produced -- picking one sent a name it did not recognise and got the
         standard report regardless. Offering a choice that does not exist is worse than
         offering none, because the operator believes they made one. */
      { key: "tpl", label: "Report", value: "standard",
        options: ["standard", "engineering", "executive", "defect_only",
                  "dataset_quality", "mission_summary"] },
      { key: "fmt", label: "Format", value: "PDF", options: ["PDF", "DOCX"] },
    ], settingChanged) },
    { id: "rp.branding", title: "Branding", height: 110, grow: false, render: () => fields([
      { key: "org", label: "Organisation", value: "DEMO organisation" },
      { key: "logo", label: "Logo", value: "brand.png" },
    ], settingChanged) },
  ],
  bottom: [
    /* Three invented findings, listed as what the report would contain. A report preview
       showing findings that are not in the project is the most direct way to put an
       invented defect in front of a client. */
    { id: "rp.included", title: "Included Findings", flex: 3, render: () => live({
      calls: ["find_annotations"],
      empty: "No findings to include yet.",
      isEmpty: ([found]) => !(found && (found.annotations || []).length),
      render: ([found]) => table(
        [{ title: "Finding", key: "id" }, { title: "Class", key: "c" },
         { title: "Severity", key: "s" }, { title: "Source", key: "src" }],
        (found.annotations || []).slice(0, 100).map((a) => ({
          id: String(a.id || "").slice(0, 8),
          c: a.label || "—",
          s: a.severity || "—",
          src: a.created_by === "user" ? "human" : "model",
        }))),
    }) },
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
  /* Was ["Save", "Reset", "|", "Import", "Export"]. Save resolves to the mission verb, so
     pressing it here prompted "Mission name" and saved a mission from the Settings
     screen; Import and Export are the dataset and mission verbs for the same reason.
     Nothing in this build persists preferences, so a Save button here could only ever
     have done something else's job. */
  toolbar: ["Verify Models"],
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
    /* Four editable controls -- Units, Default CRS, Vertical datum, Angles -- that no
       code read and no Api stores. Editing one moved the input and changed nothing, and
       a preference that silently fails to apply is worse than one that is absent because
       the operator believes the setting took.

       What IS true is the coordinate system the open project's products are in, so that
       is what this shows. Editable preferences need somewhere to persist first; that is
       real work rather than a wiring fix. */
    { id: "set.detail", title: "Spatial reference", render: () => live({
      calls: ["list_layers"],
      empty: "No georeferenced products yet, so this project has no coordinate system.",
      isEmpty: ([l]) => !(l && (l.layers || []).some((x) => x.crs_epsg)),
      render: ([l]) => {
        const layers = (l.layers || []).filter((x) => x.crs_epsg);
        const codes = [...new Set(layers.map((x) => x.crs_epsg))];
        return properties([
          { label: "Project CRS", value: codes.map((c) => `EPSG:${c}`).join(", ") },
          ...reported([
            ["Georeferenced layers", layers.length],
            /* More than one CRS in a project is worth surfacing: measurements taken
               across the two are not comparable without a transform. */
            ["Mixed systems", codes.length > 1 ? "yes" : undefined],
          ]),
          { label: "Units", value: "metre (SI)" },
        ]);
      },
    }) },
  ],
  bottom: [
    /* A hardcoded six, three of whose digests were written as a literal ellipsis. Nine
       models are installed today and the list could not know that. A digest is the whole
       mechanism tying a published metric to a particular file, so showing "..." in that
       column is worse than showing nothing: it looks like the check was done. */
    { id: "set.models", title: "Installed Models", flex: 1, render: () => live({
      calls: ["verify_models"],
      empty: "No models installed.",
      isEmpty: ([report]) => !(report && (report.models || []).length),
      render: ([report]) => table(
        [{ title: "Model", key: "m" }, { title: "State", key: "s" }, { title: "Digest", key: "d" }],
        (report.models || []).map((row) => ({
          m: row.model_key,
          s: row.status,
          d: (row.actual_sha256 || row.expected_sha256 || "").slice(0, 12) || "—",
        })), { selectKind: "model" }),
    }) },
  ],
};

export const WORKSPACES = [
  home, projects, planning, flight, verification, processing,
  twin, inspection, thermal, measurements, fleet, reports, developers, settings,
];

export const WORKSPACE_BY_ID = Object.fromEntries(WORKSPACES.map((w) => [w.id, w]));
