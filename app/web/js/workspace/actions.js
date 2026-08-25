/*
 * What every toolbar button in the cockpit actually does.
 *
 * The cockpit shipped with runAction() writing "not wired to the API yet" into the
 * status bar for all seventy-five buttons across fourteen workspaces, in 11px text at
 * the bottom of the screen. The app did not look partly wired; it looked broken, which
 * is what happens when a UI is judged finished on its layout.
 *
 * Everything here that CAN reach the application does. The rest is declared, not
 * pretended: an action with no Api behind it says which capability is missing and where
 * it would live, because "nothing happened" and "this feature does not exist yet" look
 * identical to a user and mean completely different things.
 *
 * Prerequisites are checked before the call rather than after the failure. Running a
 * reconstruction with no dataset selected is a refusal the user can act on; the same
 * refusal arriving from Python three seconds later reads as a crash.
 */

import { call, connected } from "./api.js";

/** Actions that need an open project before they mean anything. */
const NEEDS_PROJECT = new Set([
  "New Mission", "Plan", "Save", "Save Template", "Export", "Export Products",
  "Process", "Start", "Run AI", "Add GCPs", "Set CRS", "Compare", "Compare Dates",
  "Change", "Volume", "Cut & Fill", "Stockpile", "Generate Report", "Export Report",
  "Export PDF", "New Report", "Share", "Archive",
]);

/** Actions that need images on disk. */
const NEEDS_DATASET = new Set(["Process", "Start", "Run AI", "Match Captures", "Open in 3D"]);

/**
 * The dispatch table.
 *
 * Each entry either names an Api method and how to build its arguments, or explains
 * what is missing. `confirm` marks the ones that start real work or touch an aircraft --
 * a misclick on Abort should not be the same gesture as a misclick on Pan.
 */
export const ACTIONS = {
  /* -- projects and data ------------------------------------------------- */
  "New Project": {
    describe: "Create a project and make it active.",
    async run(ctx) {
      const folder = await call("pick_folder");
      if (!folder || !folder.path) return { skipped: "No folder chosen." };
      const name = ctx.prompt("Project name", folder.path.split(/[\\/]/).pop() || "Survey");
      if (!name) return { skipped: "No name given." };
      const created = await call("create_project", name, folder.path);
      await call("set_active_project", created.project_id ?? created.id);
      return { message: `Project "${name}" created.`, refresh: true };
    },
  },
  Open: {
    describe: "Open one of the projects on this machine.",
    async run(ctx) {
      const listing = await call("list_projects");
      const projects = listing.projects || [];
      if (!projects.length) return { skipped: "No projects yet. Create one first." };
      const chosen = await ctx.choose("Open project", projects.map((p) => ({
        label: p.name, value: p.id, hint: p.root_dir || "",
      })));
      if (chosen === null) return { skipped: "Cancelled." };
      await call("set_active_project", chosen);
      return { message: "Project opened.", refresh: true };
    },
  },
  Import: {
    describe: "Import a folder of images as a dataset.",
    async run() {
      const folder = await call("pick_folder");
      if (!folder || !folder.path) return { skipped: "No folder chosen." };
      const result = await call("import_dataset", folder.path);
      await call("set_active_dataset", folder.path);
      return { message: `Imported ${result.image_count ?? "?"} images.`, refresh: true };
    },
  },

  /* -- mission planning --------------------------------------------------- */
  Plan: {
    describe: "Plan the mission from the current AOI and parameters.",
    async run(ctx) {
      const result = await call("plan_mission", ctx.missionOptions());
      return { message: `Planned ${result.waypoint_count ?? result.waypoints?.length ?? "?"} waypoints.`, refresh: true };
    },
  },
  "New Mission": { alias: "Plan" },
  Save: {
    describe: "Save the planned mission into the project.",
    async run(ctx) {
      const name = ctx.prompt("Mission name", "Mission");
      if (!name) return { skipped: "No name given." };
      await call("save_mission", name);
      return { message: `Saved "${name}".`, refresh: true };
    },
  },
  Export: {
    describe: "Export the mission to the flight-controller formats.",
    async run() {
      const result = await call("export_mission");
      const files = result.files || result.exported || [];
      return { message: `Exported ${files.length || "the mission"}.`, files };
    },
  },
  "Export Products": { alias: "Export" },
  Simulate: {
    describe: "Read the compiled plan back as GeoJSON.",
    async run() {
      const result = await call("mission_geojson");
      const count = (result.geojson?.features || []).length;
      return { message: `Plan has ${count} feature(s).` };
    },
  },
  Validate: {
    describe: "Check the site against the plan.",
    async run() {
      const result = await call("verify_site");
      return { message: result.summary || "Site checked." };
    },
  },

  /* -- processing. This is COLMAP. --------------------------------------- */
  Process: {
    confirm: "Start a COLMAP reconstruction? This can run for a long time.",
    describe: "Structure from motion, orthomosaic, DSM and DTM.",
    async run(ctx) {
      const result = await call("run_reconstruction", ctx.reconstructionOptions());
      return { message: "Reconstruction started.", job: result.job_id, refresh: true };
    },
  },
  Start: {
    confirm: "Run the full pipeline? Reconstruction plus analysis.",
    describe: "Reconstruction followed by defect analysis.",
    async run(ctx) {
      const result = await call("run_pipeline", ctx.reconstructionOptions());
      return { message: "Pipeline started.", job: result.job_id, refresh: true };
    },
  },
  "Run AI": { alias: "Start" },
  Cancel: {
    describe: "Cancel the running job.",
    async run(ctx) {
      const job = ctx.selectedJob();
      if (!job) return { skipped: "No job selected." };
      await call("cancel_job", job);
      return { message: `Cancelled ${job}.`, refresh: true };
    },
  },

  /* -- measurement -------------------------------------------------------- */
  Volume: {
    describe: "Volume between DSM and DTM over the selection.",
    async run(ctx) {
      const result = await call("measure_on_raster", ctx.selectionGeometry(), "volume");
      return { message: describeMeasurement(result) };
    },
  },
  "Cut & Fill": {
    describe: "Cut and fill against a reference surface.",
    async run(ctx) {
      const result = await call("measure_on_raster", ctx.selectionGeometry(), "cut_fill");
      return { message: describeMeasurement(result) };
    },
  },
  Slope: {
    describe: "Slope over the selection.",
    async run(ctx) {
      const result = await call("measure_slope", ctx.selectionGeometry());
      return { message: describeMeasurement(result) };
    },
  },
  Stockpile: {
    describe: "Inventory the assets in this project.",
    async run() {
      const result = await call("build_asset_inventory");
      return { message: `${(result.assets || []).length} asset(s).` };
    },
  },

  /* -- flight ------------------------------------------------------------- */
  Preflight: {
    describe: "What this build can and cannot do before flying.",
    async run() {
      const result = await call("capabilities");
      const missing = Object.entries(result.capabilities || result)
        .filter(([, value]) => value === false)
        .map(([name]) => name);
      return {
        message: missing.length
          ? `Unavailable: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "…" : ""}`
          : "All capabilities available.",
      };
    },
  },
  Upload: {
    confirm: "Upload the mission to the connected aircraft?",
    describe: "Send the mission over MAVLink.",
    async run() {
      await call("upload_mission");
      return { message: "Mission uploaded." };
    },
  },
  Start_Flight: { alias: "Upload" },
  Pause: { vehicle: "pause", confirm: "Pause the aircraft?" },
  Resume: { vehicle: "resume", confirm: "Resume the mission?" },
  RTL: { vehicle: "rtl", confirm: "Return to launch?" },
  Land: { vehicle: "land", confirm: "Land now?" },
  Abort: { vehicle: "abort", confirm: "ABORT the flight?" },
  "Manual Override": {
    confirm: "Take manual control?",
    describe: "Hand control to the operator.",
    async run() {
      await call("take_manual_control");
      return { message: "Manual control taken." };
    },
  },
  "Capture Now": {
    describe: "Trigger the camera.",
    async run() {
      await call("camera_command", "trigger");
      return { message: "Camera triggered." };
    },
  },
  "Load Flight": {
    describe: "Resume a flight that was interrupted.",
    async run() {
      const result = await call("check_interrupted_flight");
      return { message: result.interrupted ? "Interrupted flight found." : "No interrupted flight." };
    },
  },

  /* -- survey comparison -------------------------------------------------- */
  Compare: {
    describe: "Compare this survey against a previous one.",
    async run() {
      const result = await call("compare_surveys");
      return { message: result.summary || "Surveys compared." };
    },
  },
  "Compare Dates": { alias: "Compare" },
  Change: { alias: "Compare" },

  /* -- ground control ----------------------------------------------------- */
  "Add GCPs": {
    describe: "Import ground control points.",
    async run() {
      const file = await call("pick_file", ["csv", "txt"]);
      if (!file || !file.path) return { skipped: "No file chosen." };
      const result = await call("import_gcps", file.path);
      return { message: `Imported ${result.count ?? "?"} control points.`, refresh: true };
    },
  },
  "Set CRS": {
    describe: "Report the coordinate reference system in use.",
    async run() {
      const result = await call("check_spatial_reference");
      return { message: result.epsg ? `CRS ${result.epsg}` : "No CRS resolved." };
    },
  },

  /* -- canvas view modes -------------------------------------------------
     Local, and real: these switch what the canvas is showing rather than calling the
     application. A view toggle that made a round trip would be slower and would fail
     when disconnected, for no benefit -- what to draw is a client decision. */
  RGB: { view: "rgb", describe: "Show the RGB orthomosaic." },
  Thermal: { view: "thermal", describe: "Show the radiometric thermal layer." },
  "3D Thermal": { view: "thermal3d", describe: "Thermal draped over the surface." },
  Fused: { view: "fused", describe: "RGB and thermal together." },
  Radiometric: { view: "radiometric", describe: "Calibrated Celsius values." },
  Semantic: { view: "semantic", describe: "Semantic classes." },
  "Point Cloud": { view: "pointcloud", describe: "Show the point cloud." },
  "Textured Mesh": { view: "mesh", describe: "Show the textured mesh." },
  Profile: { view: "profile", describe: "Elevation profile along a line." },

  /* -- canvas tools ------------------------------------------------------
     Arming a tool is also local. The measurement itself goes to the Api once there is
     geometry to measure, which is why Distance and Area report through the same path as
     Volume rather than inventing a number in the browser. */
  Measure: { tool: "measure", describe: "Arm the measure tool." },
  Distance: { tool: "distance", describe: "Measure a distance on the canvas." },
  Area: { tool: "area", describe: "Measure an area on the canvas." },
  Annotate: { tool: "annotate", describe: "Draw an annotation." },
  Edit: { tool: "edit", describe: "Edit the selected geometry." },

  /* -- navigation --------------------------------------------------------- */
  "Open in 3D": { workspace: "twin", describe: "Open this in the digital twin." },
  Reset: {
    describe: "Put the panels back where they started.",
    async run(ctx) {
      ctx.resetLayout();
      return { message: "Layout reset." };
    },
  },

  /* -- layers ------------------------------------------------------------- */
  "New Folder": {
    describe: "Add a raster or vector layer from a file.",
    async run() {
      const file = await call("pick_file", ["tif", "tiff", "geojson", "json", "shp"]);
      if (!file || !file.path) return { skipped: "No file chosen." };
      await call("add_layer_from_file", file.path);
      return { message: "Layer added.", refresh: true };
    },
  },

  "Match Captures": {
    describe: "Compare what was captured against what was planned.",
    async run() {
      const result = await call("compare_survey_specifications");
      return { message: result.summary || "Captures compared against the plan." };
    },
  },

  "New Job": {
    confirm: "Queue a reconstruction for the active dataset?",
    describe: "Queue a reconstruction job.",
    async run(ctx) {
      const result = await call("run_reconstruction", ctx.reconstructionOptions());
      return { message: "Job queued.", job: result.job_id, refresh: true };
    },
  },
};


/* Actions whose capability lives in the web service rather than the desktop Api, or
   which have no implementation at all yet. Named individually so the UI can say WHICH,
   rather than shrugging. */
export const UNWIRED = {
  "Add Aircraft": "Fleet lives in the web service (services/api), not the desktop Api.",
  "Add Battery": "Fleet lives in the web service (services/api), not the desktop Api.",
  "Add Pilot": "Fleet lives in the web service (services/api), not the desktop Api.",
  "Assign Mission": "Fleet lives in the web service (services/api), not the desktop Api.",
  "Log Maintenance": "Fleet lives in the web service (services/api), not the desktop Api.",
  "Add Webhook": "Webhooks are a service capability; the desktop Api does not expose them.",
  "New Key": "API keys are issued by the web service.",
  "Copy cURL": "Needs the service base URL; the desktop app is local-first.",
  "Send Request": "Needs the service base URL; the desktop app is local-first.",
  "Install Plugin": "Plugins load from disk at startup; there is no installer yet.",
  Sync: "Sync targets a hub deployment. Nothing to sync against locally.",
  "Request Reflight": "Needs the tasking service.",
  "New Report": "Report templates exist; no desktop Api method builds one yet.",
  "Generate Report": "Report templates exist; no desktop Api method builds one yet.",
  "Export Report": "Report templates exist; no desktop Api method builds one yet.",
  "Export PDF": "Report templates exist; no desktop Api method builds one yet.",
  Generate: "Depends on the report builder above.",
  Accept: "Finding review is persisted by the web service, not the desktop Api.",
  Reject: "Finding review is persisted by the web service, not the desktop Api.",
  Flag: "Finding review is persisted by the web service, not the desktop Api.",
  Share: "Sharing issues a signed link from the web service.",
  Archive: "Archiving a project is a service operation.",
  "Save Template": "Mission templates are read-only in this build; no writer exists.",
};

function describeMeasurement(result) {
  if (!result) return "No measurement returned.";
  if (result.refused) return `Refused: ${result.refused}`;
  const parts = [];
  for (const key of ["volume_m3", "cut_m3", "fill_m3", "area_m2", "slope_deg", "mean", "value"]) {
    if (result[key] !== undefined) parts.push(`${key.replace(/_/g, " ")} ${result[key]}`);
  }
  return parts.length ? parts.join(", ") : "Measured.";
}

/** Resolve an alias chain to the entry that does the work. */
export function resolve(action) {
  let entry = ACTIONS[action];
  let guard = 0;
  while (entry && entry.alias && guard++ < 4) entry = ACTIONS[entry.alias];
  return entry;
}

/**
 * What this action needs before it can run, or null.
 *
 * Checked here rather than inside each run() so every action reports a missing
 * prerequisite the same way, and so the check cannot be forgotten when one is added.
 */
export function prerequisite(action, state) {
  if (!connected()) return "Not connected to the application.";
  if (NEEDS_PROJECT.has(action) && !state.projectOpen) return "Open a project first.";
  if (NEEDS_DATASET.has(action) && !state.datasetSelected) return "Select a dataset first.";
  return null;
}
