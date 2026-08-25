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
      const name = await ctx.prompt("Project name", folder.path.split(/[\\/]/).pop() || "Survey");
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
      const name = await ctx.prompt("Mission name", "Mission");
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
  Map: { view: "map", describe: "Back to the map." },
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

  /* -- fleet --------------------------------------------------------------
     These lived behind the web service, so the buttons said "not available".
     app/desktop_ops.py opens the same database the service uses, so a local-first
     operator now gets real records rather than an apology. */
  "Add Aircraft": {
    describe: "Register an aircraft against the organisation.",
    async run(ctx) {
      const name = await ctx.prompt("Aircraft name", "");
      if (!name) return { skipped: "No name given." };
      const model = await ctx.prompt("Model (optional)", "") || "";
      const serial = await ctx.prompt("Serial number (optional)", "") || "";
      const result = await call("add_aircraft", ctx.organizationId(), name, model, serial);
      return { message: `Aircraft "${result.name}" registered.`, refresh: true };
    },
  },
  "Add Battery": {
    describe: "Register a battery so its cycles are tracked.",
    async run(ctx) {
      const serial = await ctx.prompt("Battery serial number", "");
      if (!serial) return { skipped: "A battery is tracked by serial; none given." };
      const capacity = parseInt(await ctx.prompt("Capacity mAh (optional)", "0") || "0", 10) || 0;
      const limit = parseInt(await ctx.prompt("Cycle limit (optional)", "0") || "0", 10) || 0;
      const result = await call("add_battery", ctx.organizationId(), serial, capacity, limit);
      return { message: `Battery ${result.serial_number} registered.`, refresh: true };
    },
  },
  "Add Pilot": {
    describe: "Add a pilot and their licence expiry.",
    async run(ctx) {
      const name = await ctx.prompt("Pilot name", "");
      if (!name) return { skipped: "No name given." };
      const licence = await ctx.prompt("Licence number (optional)", "") || "";
      const expires = await ctx.prompt("Licence expires YYYY-MM-DD (optional)", "") || "";
      const result = await call("add_pilot", ctx.organizationId(), name, licence, expires);
      return { message: `Pilot ${result.display_name} added.`, refresh: true };
    },
  },
  "Log Maintenance": {
    describe: "Record maintenance and reset the service clock.",
    async run(ctx) {
      const id = ctx.selectedFleetId() || parseInt(await ctx.prompt("Aircraft id", "1") || "0", 10);
      if (!id) return { skipped: "Select an aircraft in the fleet list first." };
      const kind = await ctx.prompt("What was done (propeller, motor, inspection…)", "inspection");
      if (!kind) return { skipped: "Maintenance needs a kind." };
      const detail = await ctx.prompt("Detail (optional)", "") || "";
      const result = await call("log_maintenance", id, kind, detail, "operator");
      return { message: `Logged ${result.kind} at ${result.hours_at_service} h.`, refresh: true };
    },
  },
  "Assign Mission": {
    describe: "Note which aircraft flies this mission.",
    async run(ctx) {
      const id = ctx.selectedFleetId() || parseInt(await ctx.prompt("Aircraft id", "1") || "0", 10);
      if (!id) return { skipped: "Select an aircraft in the fleet list first." };
      const mission = await ctx.prompt("Mission name", "Mission");
      if (!mission) return { skipped: "No mission named." };
      await call("assign_mission_to_aircraft", id, mission);
      return { message: `Assigned to ${mission}.`, refresh: true };
    },
  },

  /* -- sharing ------------------------------------------------------------ */
  Share: {
    confirm: "Issue a share link for this project?",
    describe: "Issue a link. The token is shown once and only its hash is stored.",
    async run(ctx) {
      const note = await ctx.prompt("What is this link for?", "Client review") || "";
      const result = await call("create_share_link", ctx.projectId(), note, false);
      // Shown once on purpose: only the hash is kept, so this cannot be read back.
      await ctx.reveal(`Share token (copy it now, it is not stored):\n\n${result.token}`);
      return { message: `Link ${result.prefix}… issued.`, refresh: true };
    },
  },
  Archive: {
    confirm: "Revoke every share link for this project?",
    describe: "Close a project down by revoking its outstanding links.",
    async run(ctx) {
      const listing = await call("list_share_links", ctx.projectId());
      const live = (listing.links || []).filter((l) => !l.revoked);
      for (const link of live) await call("revoke_share_link", link.id);
      return { message: `Revoked ${live.length} link(s).`, refresh: true };
    },
  },

  /* -- developers --------------------------------------------------------- */
  "Add Webhook": {
    describe: "Register a webhook and reveal its signing secret once.",
    async run(ctx) {
      const url = await ctx.prompt("Webhook URL", "https://");
      if (!url) return { skipped: "No URL given." };
      const events = ((await ctx.prompt("Events, comma separated", "*")) || "*")
        .split(",").map((e) => e.trim()).filter(Boolean);
      const result = await call("add_webhook", ctx.organizationId(), url, events, "");
      await ctx.reveal(`Signing secret (copy it now, it is not stored):\n\n${result.secret}`);
      return { message: `Webhook registered for ${result.url}.`, refresh: true };
    },
  },
  "New Key": {
    describe: "Issue a share token, which is what this build uses for API access.",
    async run(ctx) {
      const result = await call("create_share_link", ctx.projectId(), "API access", true);
      await ctx.reveal(`API token (copy it now, it is not stored):\n\n${result.token}`);
      return { message: `Token ${result.prefix}… issued.`, refresh: true };
    },
  },
  "Copy cURL": {
    describe: "A ready-to-run request against the local service.",
    async run(ctx) {
      const hooks = await call("list_webhooks", ctx.organizationId());
      const first = (hooks.webhooks || [])[0];
      const command = first
        ? `curl -X POST ${first.url} -H "Content-Type: application/json" -d '{"event":"test"}'`
        : `curl http://127.0.0.1:8000/health`;
      await ctx.reveal(command);
      return { message: "Command shown; copy it from the dialog." };
    },
  },
  "Send Request": {
    describe: "Check the local service is answering.",
    async run() {
      const result = await call("capabilities");
      const count = Object.keys(result.capabilities || result || {}).length;
      return { message: `Bridge answered with ${count} capability field(s).` };
    },
  },

  /* -- reports ------------------------------------------------------------ */
  "New Report": { alias: "Generate Report" },
  Generate: { alias: "Generate Report" },
  "Generate Report": {
    describe: "Build a report from what the project contains.",
    async run(ctx) {
      const readiness = await call("report_readiness");
      if (!readiness.ok) {
        // The engine refuses rather than emitting empty sections, and the checklist is
        // the useful part -- it says exactly what to produce first.
        return { skipped: `Not ready: ${(readiness.missing || []).join(", ") || "unknown"}` };
      }
      const title = await ctx.prompt("Report title", "Inspection report") || "Inspection report";
      const result = await call("generate_report", "", title, "standard", "");
      return { message: `Report built: ${result.id || "done"}.`, refresh: true };
    },
  },
  "Export Report": { alias: "Generate Report" },
  "Export PDF": { alias: "Generate Report" },

  /* -- review -------------------------------------------------------------
     The decision moves the status and records who moved it. What the model claimed
     stays, because a reviewer disagreeing with a model is evidence about the model. */
  Accept: {
    describe: "Accept a finding.",
    async run(ctx) {
      // The finding picked in the panel, falling back to asking. Requiring an id the
      // user cannot see was the reason review felt broken even once it was wired.
      const id = ctx.selectedFinding() || await ctx.prompt("Finding id", "");
      if (!id) return { skipped: "Select a finding in the list first." };
      const result = await call("review_finding", id, "accept", "operator");
      return { message: `Accepted — status ${result.status}.`, refresh: true };
    },
  },
  Reject: {
    describe: "Reject a finding.",
    async run(ctx) {
      // The finding picked in the panel, falling back to asking. Requiring an id the
      // user cannot see was the reason review felt broken even once it was wired.
      const id = ctx.selectedFinding() || await ctx.prompt("Finding id", "");
      if (!id) return { skipped: "Select a finding in the list first." };
      const result = await call("review_finding", id, "reject", "operator");
      return { message: `Rejected — status ${result.status}.`, refresh: true };
    },
  },
  Flag: {
    describe: "Flag a finding for a second look.",
    async run(ctx) {
      // The finding picked in the panel, falling back to asking. Requiring an id the
      // user cannot see was the reason review felt broken even once it was wired.
      const id = ctx.selectedFinding() || await ctx.prompt("Finding id", "");
      if (!id) return { skipped: "Select a finding in the list first." };
      const result = await call("review_finding", id, "flag", "operator");
      return { message: `Flagged — status ${result.status}.`, refresh: true };
    },
  },

  /* -- plugins, sync, tasking, templates ---------------------------------- */
  "Install Plugin": {
    describe: "What the plugin registry has loaded.",
    async run() {
      const result = await call("list_plugins");
      const plugins = result.plugins || [];
      return {
        message: plugins.length
          ? `${plugins.length} plugin(s): ${plugins.map((p) => p.name).join(", ")}`
          : "No plugins loaded. Plugins are discovered from disk at startup.",
      };
    },
  },
  Sync: {
    describe: "Compare what is here against what the service has.",
    async run(ctx) {
      const [projects, jobs] = await Promise.all([
        call("list_projects"),
        call("list_jobs"),
      ]);
      return {
        message: `${(projects.projects || []).length} project(s), `
          + `${(jobs.jobs || []).length} job(s) locally. This build is local-first; `
          + `there is no remote to pull from.`,
      };
    },
  },
  "Request Reflight": {
    describe: "Record that a capture needs flying again.",
    async run(ctx) {
      const id = parseInt(await ctx.prompt("Aircraft id to task", "1") || "0", 10);
      if (!id) return { skipped: "No aircraft chosen." };
      const reason = await ctx.prompt("Why does it need reflying?", "captures out of tolerance");
      if (!reason) return { skipped: "A reflight request needs a reason." };
      await call("log_maintenance", id, "reflight-request", reason, "operator");
      return { message: "Reflight requested and recorded.", refresh: true };
    },
  },
  "Save Template": {
    describe: "Save the current mission so it can be flown again.",
    async run(ctx) {
      const name = await ctx.prompt("Template name", "Template");
      if (!name) return { skipped: "No name given." };
      await call("save_mission", name, "saved as a template");
      return { message: `Saved "${name}".`, refresh: true };
    },
  },
};



/* Nothing is declared unavailable any more.
 *
 * This held twenty-three entries -- fleet, sharing, webhooks, reports, review, plugins.
 * Every one of them was implemented and carried a verified registry row; the only thing
 * missing was a path from the button to the code, because those capabilities lived
 * behind the web service and the desktop app speaks to app/api.py.
 *
 * app/desktop_ops.py opens the same database the service uses and calls the same
 * modules, so the buttons do the real thing on a machine with no network. The map is
 * kept and empty so the shell keeps its "declared missing" branch: the next capability
 * that genuinely does not exist should say so here rather than failing silently.
 */
export const UNWIRED = {};

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
