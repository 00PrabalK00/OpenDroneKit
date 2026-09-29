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

import { call, tryCall, lastError, connected } from "./api.js";

/** Actions that need an open project before they mean anything. */
const NEEDS_PROJECT = new Set([
  "New Mission", "Plan", "Save", "Save Template", "Export", "Export Products",
  "Process", "Start", "Run AI", "Add GCPs", "Set CRS", "Compare", "Compare Dates",
  "Change", "Volume", "Cut & Fill", "Stockpile", "Slope", "Ponding", "Generate Report",
  "Export Report",
  "Export PDF", "New Report", "Share", "Archive",
]);

/* -- survey packs ---------------------------------------------------------- */

async function pickPath(extensions, what) {
  const picked = await call("pick_file", extensions);
  if (!picked || !picked.path) throw new Error(`No ${what} chosen.`);
  return picked.path;
}

/** "" for this project's latest semantic map, or a picked class raster and its schema. */
async function semanticSource(ctx) {
  const source = await ctx.choose("Semantic class map", [
    "This project's latest semantic map", "Choose a class raster"]);
  if (!source) throw new Error("No semantic map chosen.");
  if (source.startsWith("This project")) return ["", ""];
  const raster = await pickPath(["tif", "tiff"], "class raster");
  const schema = await call("pick_file", ["json"]);
  return [raster, (schema && schema.path) || ""];
}

async function required(ctx, label, fallback = "") {
  const value = await ctx.prompt(label, fallback);
  if (value === null || String(value).trim() === "") throw new Error(`${label} is required.`);
  return String(value).trim();
}

async function number(ctx, label, fallback = "") {
  const value = Number(await required(ctx, label, fallback));
  if (!Number.isFinite(value)) throw new Error(`${label} must be a number.`);
  return value;
}

const SURVEY_PACKS = {
  "Land GIS / encroachment": async (ctx) => {
    const [raster, schema] = await semanticSource(ctx);
    const mode = await ctx.choose("Land analysis", ["Land cover vectors", "Encroachment against a boundary"]);
    if (!mode) return { skipped: "Nothing chosen." };
    const boundary = mode.startsWith("Encroachment") ? await pickPath(["geojson", "json"], "cadastral boundary") : "";
    return call("india_land_gis", raster, schema, boundary);
  },
  "Design progress": async (ctx) => {
    const design = await pickPath(["geojson", "json"], "approved design");
    const [raster, schema] = await semanticSource(ctx);
    return call("india_design_progress", design, raster, schema);
  },
  "Canopy cover": async (ctx) => {
    const [raster, schema] = await semanticSource(ctx);
    return call("india_canopy_cover", raster, schema);
  },
  "Vegetation indices": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "multispectral raster");
    const source = await required(ctx, "Reflectance calibration source (panel / sensor record)");
    const calibrations = {};
    for (const role of ["red", "green", "red_edge", "nir"]) {
      const band = await ctx.prompt(`Band number for ${role} (blank if absent)`, "");
      if (!band) continue;
      const scale = await number(ctx, `Reflectance scale for ${role}`, "");
      const offset = await number(ctx, `Reflectance offset for ${role}`, "0");
      calibrations[role] = { band: Number(band), scale, offset, source };
    }
    return call("india_vegetation_indices", raster, calibrations);
  },
  "Stress zones": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "vegetation index raster");
    const severe = await number(ctx, "Severe below (index value)");
    const moderate = await number(ctx, "Moderate below (index value)");
    const scope = await required(ctx, "Who validated these thresholds, for which crop and stage");
    return call("india_stress_zones", raster, severe, moderate, scope);
  },
  "Plant count": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "instance mask");
    const scope = await required(ctx, "Validation scope of the instance model");
    return call("india_plant_count", raster, scope);
  },
  "Road condition": async (ctx) => {
    const centreline = await pickPath(["geojson", "json"], "road centreline");
    const findings = await pickPath(["geojson", "json"], "validated road findings");
    const [raster, schema] = await semanticSource(ctx);
    return call("india_road_condition", centreline, findings, raster, schema);
  },
  "Power corridor": (ctx) => corridor(ctx, "power"),
  "Rail corridor": (ctx) => corridor(ctx, "rail"),
  "Solar module inventory": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "module instance raster");
    const scope = await required(ctx, "Validation scope of the module model");
    const layout = await call("pick_file", ["geojson", "json"]);
    return call("india_solar_inventory", raster, scope, (layout && layout.path) || "");
  },
  "Module temperatures": async (ctx) => {
    const thermal = await pickPath(["tif", "tiff"], "radiometric thermal raster");
    const rgb = await pickPath(["tif", "tiff", "jpg", "png"], "RGB image");
    const points = await pickPath(["json", "csv"], "RGB/thermal tie points");
    const modules = await pickPath(["geojson", "json"], "module polygons");
    const by = await required(ctx, "Tie points validated by");
    const warning = await number(ctx, "Warning above module median (°C)");
    const critical = await number(ctx, "Critical above module median (°C)");
    const hot = await number(ctx, "Hot cell above module median (°C)");
    const basis = await required(ctx, "Where these thresholds come from (standard, contract)");
    return call("india_module_temperatures", thermal, rgb, points, modules, by, warning, critical, hot, basis);
  },
  "Anomaly baseline": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "raster confirmed normal");
    const by = await required(ctx, "Confirmed normal by");
    const scope = await required(ctx, "Where and when this baseline applies");
    return call("india_anomaly_baseline", [raster], by, scope);
  },
  "Find anomalies": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "feature raster");
    const baseline = await pickPath(["json"], "validated baseline");
    const sigma = await number(ctx, "Threshold (robust sigma)", "3.5");
    return call("india_find_anomalies", raster, baseline, sigma);
  },
};

async function corridor(ctx, kind) {
  const detections = await pickPath(["geojson", "json"], `${kind} detections`);
  const centreline = await pickPath(["geojson", "json"], "corridor centreline");
  const geometry = await required(ctx, "Capture geometry (for example oblique_close_range)");
  const scope = await required(ctx, "Validation scope of the detector");
  return call("india_corridor_inspection", kind, detections, centreline, geometry, scope);
}

/** A line about what a pack produced: its own status and headline numbers. */
function summarisePack(result) {
  const summary = result.summary || {};
  const parts = [];
  if (result.baseline_path) return `baseline written to ${result.baseline_path}.`;
  for (const key of ["observed_surface_coverage_percent", "surveyed_distance_m",
    "road_surface_area_m2", "module_count", "instance_count", "candidate_count", "status"]) {
    if (summary[key] !== undefined && typeof summary[key] !== "object") parts.push(`${key.replace(/_/g, " ")} ${summary[key]}`);
  }
  if (summary.categories) {
    parts.push(Object.entries(summary.categories)
      .map(([name, value]) => `${name} ${value && value.area_m2 !== undefined ? `${value.area_m2} m²` : ""}`).join(", "));
  }
  if (summary.totals) parts.push(`${Object.keys(summary.totals).length} total(s)`);
  const layers = (result.layer_ids || []).length;
  return `${parts.filter(Boolean).join("; ") || "done"}. ${layers} layer(s) added; summary ${result.summary_path}.`;
}

/** The 3D view, opened on the mesh if it is not already showing. */
async function openModel(ctx) {
  const { currentModelViewer, whenModelReady } = await import("./model3d.js");
  const shown = currentModelViewer();
  if (shown && shown.ready) return shown;
  if (!shown) ctx.showView("mesh");
  return whenModelReady();
}

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
      // The id is nested: the Api answers ok(project={...}). Reading created.id gave
      // undefined and set_active_project then failed on int(None) -- and it went
      // unnoticed because session.create_project already makes the new project active,
      // so the visible outcome was right for the wrong reason.
      const id = (created.project && created.project.id) ?? created.project_id ?? created.id;
      if (id != null) await call("set_active_project", id);
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
    describe: "Export the mission, or the products when there is no mission.",
    async run(ctx) {
      // tryCall, not call: call() THROWS when the API refuses, so a fallback written
      // after it can never run. The first version of this used call() and the mission
      // refusal propagated straight past the products branch, which is why the button
      // still said "No mission has been planned yet" on screens that had products.
      const result = await tryCall("export_mission");
      const files = (result && (result.files || result.exported)) || [];
      if (files.length) {
        return { message: `Exported ${files.length} mission file(s).`, files };
      }

      // Export sits on Projects, Digital Twin, Thermal, Fleet and Settings as well as
      // Mission Planning, and on all of those it answered "No mission has been planned
      // yet" -- true, and about something the user was not looking at. If the project has
      // products, those are the exportable thing; the mission refusal is only the right
      // answer when there is nothing else to give.
      const listed = await tryCall("list_layers");
      if (((listed && listed.layers) || []).some((l) => l.path)) {
        return ACTIONS["Export Products"].run(ctx);
      }
      return {
        message:
          lastError.get("export_mission") ||
          "Nothing to export yet: no mission has been planned and no products exist.",
      };
    },
  },
  "Export Products": {
    describe: "Reveal the georeferenced products on disk.",
    async run() {
      // This aliased to Export, which exports the MISSION. So pressing it on Processing,
      // Digital Twin, Thermal, Fleet or Settings answered "No mission has been planned
      // yet" -- a true sentence about something the user had not asked about, on screens
      // where a mission is not the subject. Seven workspaces reported that.
      //
      // The products are already georeferenced COGs written to the project directory, so
      // there is nothing to convert: revealing them IS the export.
      const listed = await tryCall("list_layers");
      const layers = (((listed && listed.layers) || [])).filter((l) => l.path);
      if (!layers.length) {
        return {
          message:
            "No products yet. Run a reconstruction and the orthomosaic, DSM, DTM and " +
            "hillshade will be written to the project folder.",
        };
      }

      const first = layers[0].path;
      const directory = first.slice(0, Math.max(first.lastIndexOf("\\"), first.lastIndexOf("/")));
      await call("open_path", directory);
      const names = layers.map((l) => l.name).join(", ");
      return { message: `${layers.length} products in ${directory}: ${names}.` };
    },
  },
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
      const result = await call("run_pipeline", ctx.pipelineOptions());
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

  /* -- measurement --------------------------------------------------------
     Every one of these used to call the Api with null: selectionGeometry was never
     assigned, and Volume asked measure_on_raster for a kind it does not have. They now
     take the area or line drawn on the map (drawing one first when there is none) and
     the project's own DSM/DTM, which the Api finds and reprojects into. */
  Volume: {
    describe: "Volume above the DTM inside the drawn area.",
    async run(ctx) {
      const polygon = ctx.requireSelection("polygon", "Volume");
      if (!polygon) return { skipped: "Draw the area to measure, then double-click." };
      const result = await call("measure_drawn", "volume", polygon);
      return { message: describeMeasurement(result), refresh: true };
    },
  },
  "Cut & Fill": {
    describe: "Cut and fill against a design level, inside the drawn area.",
    async run(ctx) {
      const polygon = ctx.requireSelection("polygon", "Cut & Fill");
      if (!polygon) return { skipped: "Draw the area to cut and fill, then double-click." };
      const level = await ctx.prompt("Design (base) elevation, metres", "");
      if (level === null || level === "" || !Number.isFinite(Number(level))) {
        return { skipped: "Cut and fill needs the design elevation to measure against." };
      }
      const result = await call("measure_drawn", "cut_fill", polygon, "",
                                { base_elevation_m: Number(level) });
      return { message: describeMeasurement(result), refresh: true };
    },
  },
  Slope: {
    describe: "Slope over the drawn area.",
    async run(ctx) {
      const polygon = ctx.requireSelection("polygon", "Slope");
      if (!polygon) return { skipped: "Draw the area to measure slope over, then double-click." };
      const result = await call("measure_drawn", "slope", polygon);
      return { message: describeMeasurement(result), refresh: true };
    },
  },
  Stockpile: {
    describe: "Measure one stockpile and write its report package.",
    async run(ctx) {
      const polygon = ctx.requireSelection("polygon", "Stockpile");
      if (!polygon) return { skipped: "Draw around the pile's toe, then double-click." };
      /* Blank means the project's DTM. The selection's lowest point is not offered as a
         base: it always over-states a pile on sloping ground. */
      const base = await ctx.prompt("Base elevation in metres (blank = use the DTM)", "");
      if (base === null) return { skipped: "Cancelled." };
      const options = base !== "" && Number.isFinite(Number(base))
        ? { base_elevation_m: Number(base) } : {};
      const result = await call("measure_drawn", "stockpile", polygon, "", options);
      return { message: `${describeMeasurement(result)} Report: ${result.report_path}`,
               refresh: true };
    },
  },
  Ponding: {
    describe: "Where water can collect on the DSM.",
    async run(ctx) {
      /* No default: without the survey's vertical accuracy a basin cannot be told from
         reconstruction noise, and find_ponding refuses rather than guess. */
      const accuracy = await ctx.prompt("Vertical accuracy of this DSM, metres", "");
      if (!accuracy || !Number.isFinite(Number(accuracy))) {
        return { skipped: "Ponding needs the survey's vertical accuracy." };
      }
      const result = await call("find_ponding", "", Number(accuracy));
      return {
        message: result.count
          ? `${result.count} depression(s) deeper than ${result.detection_floor_m} m, `
            + `${Number(result.total_volume_m3).toFixed(2)} m³ of capacity.`
          : `No depression deeper than ${result.detection_floor_m} m (the detection floor).`,
      };
    },
  },
  "Select Area": { tool: "select-polygon", draw: "polygon",
                   describe: "Draw the area Volume, Slope and Stockpile act on." },
  "Select Line": { tool: "select-line", draw: "line", describe: "Draw a line to act on." },

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

  /* -- survey comparison --------------------------------------------------
     compare_surveys was called with no arguments. Two real surveys also could never be
     compared, because no two reconstructions share a grid; the later one is now
     resampled onto the earlier one and the result says so. */
  Compare: {
    describe: "Compare this survey's DSM against an earlier one.",
    async run(ctx) {
      const surfaces = await call("project_surfaces");
      const earlier = await tryCall("pick_file", ["tif", "tiff"]);
      if (!(earlier && earlier.path)) return { skipped: "No earlier DSM chosen." };
      let later = surfaces.dsm;
      if (!later) {
        const picked = await tryCall("pick_file", ["tif", "tiff"]);
        later = picked && picked.path;
      }
      if (!later) return { skipped: "This project has no DSM to compare against." };
      const polygon = ctx.selectionVertices();
      const change = await call("compare_survey_volumes", earlier.path, later,
                                polygon.length >= 3 ? polygon : null);
      let message = `Added ${change.added_volume_m3} m³, removed `
        + `${change.removed_volume_m3} m³ (net ${change.net_volume_m3}).`;
      const warnings = change.warnings || [];
      if (warnings.length) message += ` ${warnings[warnings.length - 1]}`;

      /* Movement needs error bounds; without them it is not asked for. */
      const bounds = await ctx.prompt(
        "For movement: vertical accuracy of earlier, later survey and registration "
        + "residual in metres, comma separated (blank to skip)", "");
      const numbers = String(bounds || "").split(/[,\s]+/).filter(Boolean).map(Number);
      if (numbers.length === 3 && numbers.every(Number.isFinite)) {
        const moved = await call("compare_surveys", earlier.path, later, ...numbers);
        message += moved.moved
          ? ` ${moved.count} region(s) moved beyond the ${moved.detection_floor_m} m floor.`
          : ` No movement resolvable above ${moved.detection_floor_m} m.`;
      }
      return { message, refresh: true };
    },
  },
  "Compare Dates": { alias: "Compare" },
  Change: { alias: "Compare" },

  /* -- ground control ----------------------------------------------------- */
  "Add GCPs": {
    describe: "Import ground control points.",
    async run(ctx) {
      const file = await call("pick_file", ["csv", "txt"]);
      if (!file || !file.path) return { skipped: "No file chosen." };
      let result = await tryCall("import_gcps", file.path);
      if (!result) {
        /* A file with no CRS column is refused rather than guessed. Ask for the code
           and try again; the refusal text says why it matters. */
        const why = lastError.get("import_gcps") || "";
        if (!/CRS/i.test(why)) throw new Error(why || "The control file could not be read.");
        const epsg = await ctx.prompt("EPSG code the control coordinates are in", "");
        if (!epsg) return { skipped: "Control points need the CRS they were surveyed in." };
        result = await call("import_gcps", file.path, Number(epsg));
      }
      return { message: `Imported ${result.count ?? "?"} control points; mark each in at `
                 + "least three photographs with Mark GCP.", refresh: true };
    },
  },
  "Mark GCP": {
    describe: "Click where a control point is in a photograph.",
    async run(ctx) {
      const listed = await call("list_gcps");
      const points = listed.points || [];
      if (!points.length) return { skipped: "Import ground control points first (Add GCPs)." };
      const labels = points.map((p) => `${p.name} (${p.mark_count} mark${p.mark_count === 1 ? "" : "s"})`);
      const chosenPoint = await ctx.choose("Control point", labels);
      if (!chosenPoint) return { skipped: "No control point chosen." };
      const point = points[labels.indexOf(chosenPoint)];
      const images = ((await call("list_dataset_images")).images || [])
        .map((i) => (typeof i === "string" ? i : i.name));
      if (!images.length) return { skipped: "Select a dataset first." };
      const image = await ctx.choose(`Photograph showing ${point.name}`, images);
      if (!image) return { skipped: "No photograph chosen." };
      await ctx.showImage(image);
      const rough = await ctx.pickImagePixel(`Click ${point.name} on ${image} (Esc to cancel).`);
      if (!rough) return { skipped: "No point clicked on the photograph." };
      /* On the whole frame one screen pixel is several pixels of the photograph, which
         is metres on the ground at the far edge. The second click is on a close-up. */
      const side = Math.max(64, Math.round(Math.min(rough.width, rough.height) / 8));
      await ctx.showImage(image, [rough.x - side / 2, rough.y - side / 2, side, side]);
      const at = await ctx.pickImagePixel(`Click ${point.name} again on the close-up (Esc to cancel).`);
      if (!at) return { skipped: "No point clicked on the close-up." };
      const marked = await call("mark_gcp", point.name, image, at.x, at.y, at.width, at.height);
      const count = marked.point.mark_count;
      return { message: `${point.name} marked on ${image} at (${Math.round(at.x)}, ${Math.round(at.y)}); `
                 + `${count} mark${count === 1 ? "" : "s"}`
                 + (marked.point.usable ? "." : ` -- needs ${3 - count} more to be used.`),
               refresh: true };
    },
  },
  "GCP Report": {
    describe: "How far each control point lands from where it was surveyed.",
    async run() {
      const { report } = await call("gcp_accuracy_report");
      if (report.rmse_m === null || report.rmse_m === undefined) {
        return { message: `No measured accuracy: ${(report.warnings || []).join(" ")}` };
      }
      const kind = report.role === "check" ? "independent check" : "fit to the control";
      return { message: `GCP RMSE ${Number(report.rmse_m).toFixed(3)} m over ${report.used} of `
                 + `${report.point_count} points (${kind}; georeferenced by ${report.georeferenced_by}). `
                 + (report.warnings || []).join(" ") };
    },
  },
  "Set CRS": {
    describe: "Choose the projected CRS reconstructions are written in.",
    async run(ctx) {
      const current = await call("get_project_crs");
      const answer = await ctx.prompt(
        "EPSG code of a projected CRS (blank = automatic UTM zone)",
        current.epsg ? String(current.epsg) : "");
      if (answer === null) return { skipped: "Cancelled." };
      const trimmed = String(answer).trim();
      const result = await call("set_project_crs", trimmed === "" ? null : Number(trimmed));
      return { message: result.epsg ? `Reconstructions will be written in EPSG:${result.epsg} `
                                       + `(${result.name}).`
                                     : "Reconstructions will use the UTM zone of the imagery." };
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
  // The Poisson mesh carries no texture; it is shaded by height. The label is the
  // toolbar's, the description says what is drawn.
  "Textured Mesh": { view: "mesh", describe: "Show the reconstructed 3D mesh (shaded by height)." },
  // A line drawn on the map, sampled from the DSM. It used to be a view toggle with
  // nothing behind it: no code anywhere computed a profile.
  Profile: { tool: "profile", draw: "line", measure: "profile",
             describe: "Elevation profile along a drawn line." },

  /* -- canvas tools ------------------------------------------------------
     Arming a tool is also local. The measurement itself goes to the Api once there is
     geometry to measure, which is why Distance and Area report through the same path as
     Volume rather than inventing a number in the browser. */
  Measure: { tool: "measure", draw: "line", measure: "distance",
             describe: "Measure a distance on the map." },
  Distance: { tool: "distance", draw: "line", measure: "distance",
              describe: "Measure a distance on the map." },
  Area: { tool: "area", draw: "polygon", measure: "area",
          describe: "Measure an area on the map; it also becomes the selection." },
  Annotate: { tool: "annotate", describe: "Draw an annotation." },
  Edit: { tool: "edit", describe: "Edit the selected geometry." },

  /* -- fly to draw --------------------------------------------------------
     The pilot flies the aircraft to each corner of the site and presses a button. That
     is the whole feature, and it is the one case where a boundary comes from where the
     aircraft actually is rather than from a map someone drew at a desk -- which is why
     it exists for sites whose extent is not on any map.

     It needs a connected aircraft and nothing else. It had no button. */
  "Mark Corner": {
    describe: "Record the aircraft's current position as a boundary corner.",
    async run(ctx) {
      const note = await ctx.prompt("Note for this corner (optional)", "");
      const result = await call("mark_boundary_corner", note || "");
      return {
        message: `Corner ${result.corner} marked. `
          + (result.corner >= 3
            ? "Close Boundary will turn these into the area of interest."
            : `${3 - result.corner} more needed before a boundary can be closed.`),
        refresh: true,
      };
    },
  },
  "Close Boundary": {
    describe: "Turn the marked corners into the area of interest.",
    async run() {
      const result = await call("boundary_from_marks", true);
      /* Three points is a triangle and the minimum that encloses anything. The Api
         refuses below that, and saying so here means the operator learns it while
         still standing on site rather than afterwards. */
      return {
        message: `Boundary closed from ${result.corner_count ?? "?"} corner(s)`
          + (result.area_m2 ? `, ${(result.area_m2 / 10000).toFixed(2)} ha.` : "."),
        refresh: true,
      };
    },
  },
  "Clear Corners": {
    describe: "Discard the marked corners and start the boundary again.",
    confirm: "Discard every marked corner?",
    async run() {
      const result = await call("clear_boundary_marks");
      return { message: `Cleared. ${result.corner_count ?? 0} corners remain.`, refresh: true };
    },
  },

  /* Named "Compare Versions", not "Compare".
     ACTIONS is an object literal, so a second `Compare:` key silently REPLACES the
     first -- and the first is the survey-comparison verb that Thermal's Compare button
     and the "Compare Dates" alias both resolve to. Two working buttons would have gone
     to a mission-version diff, with no error anywhere. This is the collision this file
     already carries scars from; I reintroduced it and the duplicate-key guard below is
     what caught it. */
  "Compare Versions": {
    describe: "What changed between two saved versions of this mission.",
    async run(ctx) {
      const listed = await tryCall("mission_version_history");
      const history = (listed && listed.history) || [];
      if (history.length < 2) {
        return { skipped: "Two saved versions are needed to compare. Save again first." };
      }
      const labels = history.map((h) => `v${h.version_num}`);
      const from = await ctx.choose("Compare from", labels);
      if (!from) return { skipped: "No version chosen." };
      const to = await ctx.choose("Compare to", labels.filter((l) => l !== from));
      if (!to) return { skipped: "No second version chosen." };
      const result = await call("diff_mission_versions",
        Number(from.slice(1)), Number(to.slice(1)));
      const diff = result.diff || {};
      const changes = diff.changes || diff.fields || [];
      return {
        message: changes.length
          ? `${from} to ${to}: ${changes.length} change(s). `
            + changes.slice(0, 4).map((c) => (typeof c === "string" ? c : c.field || "")).join(", ")
          : `${from} and ${to} are identical.`,
      };
    },
  },

  /* Two points picked on the mesh in the 3D view. The picks are made by the application
     on the full-resolution mesh in its CRS; the model is measured only when its
     georeferencing is recorded. */
  "Measure in 3D": {
    describe: "Measure a length or a height between two points picked on the 3D model.",
    async run(ctx) {
      const viewer = await openModel(ctx);
      const kind = await ctx.choose("Measure", ["length", "height"]);
      if (!kind) return { skipped: "Nothing chosen." };
      viewer.clearPins();
      ctx.toast("Click the first point on the model.");
      const first = await viewer.pickNext();
      ctx.toast("Click the second point on the model.");
      const second = await viewer.pickNext();
      const result = await call("measure_in_model", viewer.scene.model_path, kind, [first, second]);
      const m = result.measurement;
      const text = kind === "height"
        ? `height ${m.height_m.toFixed(3)} m`
        : `length ${m.slope_distance_m.toFixed(3)} m (horizontal ${m.horizontal_distance_m.toFixed(3)} m, `
          + `vertical ${m.vertical_distance_m.toFixed(3)} m)`;
      return { message: `Model ${text}, EPSG:${m.crs_epsg}. Kept in the measurement history.` };
    },
  },

  /* -- thirteen rows that were verified and unreachable ---------------------
     Mission versioning, repeat surveys, boundary import, the camera and payload
     databases, the offline terrain cache, linked-sortie progress, flight log export,
     PPK input checking and job sizing. Each names its Api method in its own registry
     note and each has passing tests. None had a button. */

  "Repeat Survey": {
    describe: "Fly a saved mission again, so two surveys are comparable.",
    async run(ctx) {
      const mode = await ctx.choose("Repeat as",
        ["exact", "same-camera", "same-terrain"]);
      if (!mode) return { skipped: "No mode chosen." };
      /* "exact" is the one that makes two surveys comparable: same lines, same
         altitudes, same capture points. The others re-solve part of the plan and are
         offered because a changed camera or a new terrain model is a real reason to. */
      const result = await call("repeat_mission", null, mode === "exact" ? "exact" : mode);
      const repeat = result.repeat || {};
      return {
        message: `Repeat prepared${repeat.name ? ` as "${repeat.name}"` : ""}`
          + (repeat.differences ? `; ${repeat.differences} difference(s) from the original.` : "."),
        refresh: true,
      };
    },
  },
  "Import Boundary": {
    describe: "Load an area of interest from KML, KMZ, GeoJSON, GPX or CSV.",
    async run() {
      const picked = await tryCall("pick_file", ["kml", "kmz", "geojson", "json", "gpx", "csv"]);
      const path = picked && picked.path;
      if (!path) return { skipped: "No file chosen." };
      const result = await call("import_boundary", path);
      const ring = result.polygon || [];
      return { message: `Boundary loaded: ${ring.length} vertices.`, refresh: true };
    },
  },
  Cameras: {
    describe: "What each camera yields at a working altitude.",
    async run(ctx) {
      const listed = await tryCall("list_cameras");
      const cameras = (listed && listed.cameras) || [];
      if (!cameras.length) return { skipped: "No camera profiles available." };
      const pick = await ctx.choose("Camera", cameras.map((c) => c.name || c.key || String(c)));
      if (!pick) return { skipped: "No camera chosen." };
      const altitude = await ctx.prompt("At what altitude (m)?", "60");
      const described = await call("describe_camera", pick, Number(altitude || 60));
      const cam = described.camera || {};
      /* GSD is the number that decides whether the survey can see the defect, so it
         leads. */
      return {
        message: `${pick} at ${altitude} m: `
          + (cam.gsd_cm !== undefined ? `GSD ${Number(cam.gsd_cm).toFixed(2)} cm` : "GSD not reported")
          + (cam.footprint_m ? `, footprint ${cam.footprint_m}` : ""),
      };
    },
  },
  Payloads: {
    describe: "What fitting a payload changes about the mission.",
    async run(ctx) {
      const listed = await tryCall("list_payloads");
      const payloads = (listed && listed.payloads) || [];
      if (!payloads.length) return { skipped: "No payloads available." };
      const pick = await ctx.choose("Payload", payloads.map((p) => p.key || p.name || String(p)));
      if (!pick) return { skipped: "No payload chosen." };
      const described = await call("describe_payload", pick);
      const notes = described.plan_notes || described.notes || [];
      return {
        message: notes.length ? notes.join(" ") : `${pick} carries no planning notes.`,
      };
    },
  },
  "Cache Terrain": {
    describe: "Copy a terrain raster into the project, for flying with no connectivity.",
    async run() {
      const picked = await tryCall("pick_file", ["tif", "tiff", "hgt"]);
      const path = picked && picked.path;
      if (!path) return { skipped: "No terrain file chosen." };
      const result = await call("cache_terrain", path);
      const tile = result.tile || {};
      return {
        message: `Cached ${tile.name || path}.`
          + (tile.bounds_lonlat ? " Coverage is checked against the drawn area on Validate." : ""),
        refresh: true,
      };
    },
  },
  "Linked Progress": {
    describe: "Which survey inside a linked sortie is finished, and which is part flown.",
    async run() {
      const picked = await tryCall("pick_folder");
      const folder = picked && picked.path;
      if (!folder) return { skipped: "No folder chosen." };
      const result = await call("linked_mission_progress", folder);
      /* An overall percentage cannot tell "every survey is nearly done" from "three are
         finished and the fourth was never started", and those call for opposite actions
         on site. So both counts are reported, never a single figure. */
      return {
        message: `${result.complete_segments ?? 0} segment(s) complete, `
          + `${result.partial_segments ?? 0} part flown.`,
      };
    },
  },
  "Export Log": {
    describe: "Write the recorded flight to CSV, JSON, GPX and KML.",
    async run() {
      const result = await call("export_flight_log");
      const files = result.files || [];
      if (!files.length) return { skipped: "No flight has been recorded yet." };
      return { message: `Wrote ${files.length} file(s): ${files.join(", ")}` };
    },
  },
  "Check PPK": {
    describe: "Whether the camera events and base observations can support PPK.",
    async run(ctx) {
      const events = await tryCall("pick_file", ["csv", "txt", "mrk"]);
      if (!(events && events.path)) return { skipped: "No camera-event file chosen." };
      /* RINEX 2 observation files are named by year: base.26o, base.25o. A filter of
         "o" matched none of them. */
      const year = new Date().getFullYear();
      const yearly = Array.from({ length: 12 }, (_, i) => `${String((year - i) % 100).padStart(2, "0")}o`);
      const rinex = await tryCall("pick_file", ["obs", "rnx", "o", ...yearly]);
      if (!(rinex && rinex.path)) return { skipped: "No base observation file chosen." };
      const result = await call("check_ppk_inputs", events.path, rinex.path);
      /* It answers with what is wrong, not merely whether. This read `problems` and
         `overlap_s`, which the check never returns, so a flight half outside the base
         window was announced as "PPK inputs check out". */
      const blocking = result.blocking || [];
      const covered = `${(result.alignment || {}).covered_count ?? "?"} of ${result.events ?? "?"} events inside the base window`;
      if (!result.usable_for_ppk) {
        return { message: `PPK inputs are not usable (${covered}): ${blocking.join(" ")}` };
      }
      return { message: `PPK inputs check out: ${covered}. ${result.accuracy_statement || ""}`.trim() };
    },
  },
  "Size Job": {
    describe: "Whether this machine can finish a reconstruction of this size.",
    async run(ctx) {
      const listed = await tryCall("list_dataset_images");
      const count = ((listed && listed.images) || []).length;
      if (!count) return { skipped: "Select a dataset first." };
      const result = await call("size_reconstruction_job", count);
      /* Answered before the run rather than after the failure. A reconstruction that
         exhausts memory eight hours in has cost the whole night. */
      return {
        message: `${count} images: ${result.verdict || (result.fits ? "should fit" : "may not fit")}`
          + (result.peak_ram_gb ? `, about ${result.peak_ram_gb} GB peak RAM` : "")
          + (result.note ? `. ${result.note}` : ""),
      };
    },
  },
  History: {
    describe: "Every saved version of this mission, and what changed.",
    async run(ctx) {
      const listed = await tryCall("mission_version_history");
      const history = (listed && listed.history) || [];
      if (!history.length) return { skipped: "This mission has no saved versions yet." };
      const labels = history.map((h) =>
        `v${h.version_num}${h.summary ? ` \u2014 ${h.summary}` : ""}`);
      const pick = await ctx.choose("Restore which version?", labels);
      if (!pick) return { skipped: "No version chosen." };
      const version = history[labels.indexOf(pick)].version_num;
      /* Restoring saves the old plan as a NEW version rather than rewinding, so the
         history stays a record of what happened rather than of what is current. */
      const result = await call("restore_mission_version", version);
      return {
        message: `Restored v${version} as v${result.version?.version_num ?? "?"}.`,
        refresh: true,
      };
    },
  },

  /* -- the features that were built and could not be reached ---------------
     Seventy-two of a hundred and forty-nine Api methods are never called from the
     interface. Most of that is fine -- helpers, alternate entry points, methods the
     REST service uses. What was not fine is that it included most of what this project
     built recently: annotation tags, site markers, hazard clearance, CAD overlays,
     saved views, model clipping and thermal scaling. Each had an Api method, a core
     module and passing tests, and no control anywhere that reached it.

     The ones below reach their function today. The four after them cannot, and say so
     rather than becoming buttons that always decline. */

  Tag: {
    describe: "Group findings by elevation, reflight, client query -- whatever the job needs.",
    async run(ctx) {
      const id = ctx.selectedFinding();
      if (!id) return { skipped: "Select a finding first, then Tag." };
      const tag = await ctx.prompt("Tag", "");
      if (!tag) return { skipped: "No tag given." };
      /* tag_annotations takes a list of ids and a list of TAGS. Passing the string
         would have spread it into one tag per character. */
      const result = await call("tag_annotations", [id], [tag]);
      /* add_tags() returns `updated` (the ids it changed) and `tags`. There is no
         `tagged` count -- the first draft read one and would have printed 1 whatever
         happened, including when nothing was tagged. */
      const changed = (result.updated || []).length;
      return {
        message: changed ? `Tagged ${changed} finding(s) "${tag}".`
                         : `Already tagged "${tag}".`,
        refresh: true,
      };
    },
  },
  "Add Marker": {
    describe: "Record a hazard, a takeoff point or an access gate on the site.",
    async run(ctx) {
      const kinds = await tryCall("marker_kinds");
      /* Rows are {kind, describes}. The describes text is what tells an operator the
         difference between a hazard and an obstacle, so it is shown in the chooser. */
      const rows = (kinds && kinds.kinds) || [];
      if (!rows.length) return { skipped: "No marker kinds available." };
      const labels = rows.map((r) => `${r.kind} \u2014 ${r.describes}`);
      const picked = await ctx.choose("Marker kind", labels);
      if (!picked) return { skipped: "No kind chosen." };
      const kind = rows[labels.indexOf(picked)].kind;

      const name = await ctx.prompt("Marker name", "");
      if (!name) return { skipped: "A marker needs a name to be found again." };
      /* Clicked on the map when there is one; typed when there is not, or when the
         operator presses Esc because the coordinate is written down. */
      let point = await ctx.pickPoint(`Click where "${name}" is (Esc to type it).`);
      if (!point) {
        const lon = await ctx.prompt("Longitude", "");
        const lat = await ctx.prompt("Latitude", "");
        if (!lon || !lat) return { skipped: "A marker needs a longitude and a latitude." };
        point = [Number(lon), Number(lat)];
      }
      const result = await call("add_site_marker", name, kind, [point]);
      /* It answers with the whole marker list, not the one just added. */
      const total = (result.markers || []).length;
      return { message: `Added ${kind} "${name}". ${total} marker(s) on this site.`
                 + (result.warning ? ` ${result.warning}` : ""),
               refresh: true };
    },
  },
  /* -- survey packs -------------------------------------------------------
     The India-first packs were library code nothing could reach. One button offers
     them; each asks for exactly the inputs its pack refuses to guess (a boundary, a
     design, a calibration, a validation scope) and puts what it produced on the map. */
  "Semantic Map": {
    describe: "Classify the orthomosaic with the installed semantic model (background job).",
    async run() {
      const started = await call("semantic_map");
      const usable = (started.usable_labels || []).join(" and ");
      return { message: "Semantic map started."
                 + (usable ? ` Only ${usable} are validated for this model; treat other classes as unvalidated.` : ""),
               job: started.job_id, refresh: true };
    },
  },
  "Survey Packs": {
    describe: "Land GIS, design progress, agriculture, road/power/rail/solar and anomaly packs.",
    async run(ctx) {
      const packs = Object.keys(SURVEY_PACKS);
      const pack = await ctx.choose("Survey pack", packs);
      if (!pack) return { skipped: "No pack chosen." };
      const result = await SURVEY_PACKS[pack](ctx);
      if (!result || result.skipped) return result || { skipped: "Cancelled." };
      return { message: `${pack}: ${summarisePack(result)}`, refresh: true };
    },
  },
  "Check Hazards": {
    describe: "Report how close the planned route passes to each hazard.",
    async run(ctx) {
      const clearance = await ctx.prompt("Clearance (metres)", "30");
      if (clearance === null || clearance === "") return { skipped: "No clearance given." };
      const result = await call("check_hazards", Number(clearance));
      /* The key is `hazards`, and a NEGATIVE clearance means the route passes inside
         the hazard's own marked radius -- the most serious case, and the one that
         reads as a typo if printed as "at -21 m". */
      const found = result.hazards || [];
      const inside = found.filter((h) => h.clearance_m < 0).length;
      const breaches = found.length;
      /* Reports, never refuses. Clearance is the operator's decision and the software
         has no standing to take it. */
      return {
        message: breaches
          ? `${breaches} hazard(s) within ${clearance} m of the route`
            + (inside ? `, ${inside} of them passing INSIDE the marked radius` : "")
            + ". The route was not changed."
          : `No hazard is within ${clearance} m of the ${result.checked_waypoints} waypoint(s) checked.`,
      };
    },
  },
  "Import CAD": {
    describe: "Place a DXF, a GeoTIFF or a plain scanned image over the survey.",
    async run(ctx) {
      /* pick_file takes an EXTENSION LIST, not a title. Passing a title made pywebview
         raise "is not a valid file filter" and the exception arrived at the toolbar as
         the result of pressing the button. */
      const picked = await tryCall("pick_file", ["dxf", "tif", "tiff", "png", "jpg", "jpeg"]);
      const path = picked && picked.path;
      if (!path) return { skipped: "No file chosen." };
      const extension = path.split(".").pop().toLowerCase();

      if (extension === "tif" || extension === "tiff") {
        /* A GeoTIFF carries its own CRS: it is a layer, not something to place. Images
           used to be sent to the DXF reader, which answered "No usable geometry". */
        const added = await tryCall("add_layer_from_file", path);
        if (added && added.layer && added.layer.crs_epsg) {
          return { message: `Added ${added.layer.name} (EPSG:${added.layer.crs_epsg}).`,
                   refresh: true };
        }
      }
      if (extension !== "dxf") {
        /* A plain picture has no georeferencing: its corners are asked for, never guessed. */
        const box = await ctx.prompt("Image corners: west, south, east, north", "");
        const numbers = String(box || "").split(/[,\s]+/).filter(Boolean).map(Number);
        if (numbers.length !== 4 || !numbers.every(Number.isFinite)) {
          return { skipped: "A plain image needs its four bounds to be placed." };
        }
        const epsg = await ctx.prompt("EPSG code of those bounds", "4326");
        if (!epsg) return { skipped: "The bounds need their EPSG code." };
        const placed = await call("place_image_overlay", path, ...numbers, Number(epsg));
        return { message: `Placed ${placed.name} as a map layer.`, refresh: true };
      }

      /* Required, never guessed. An overlay placed in the wrong coordinate system lands
         somewhere plausible and wrong, and import_cad_overlay refuses without it. */
      const epsg = await ctx.prompt("Source EPSG code", "");
      if (!epsg) return { skipped: "A CAD overlay needs the EPSG code its coordinates are in." };
      const result = await call("import_cad_overlay", path, Number(epsg));
      /* `entities` is a count per entity type, and `skipped` names what could not be
         flattened. Reporting the placed count alone would present a partial import as
         a complete one. */
      const placed = Object.values(result.entities || {}).reduce((a, b) => a + b, 0);
      const offset = result.alignment && Number.isFinite(result.alignment.centre_offset_m)
        ? ` Centre ${Math.round(result.alignment.centre_offset_m)} m from the survey's.` : "";
      return {
        message: `Placed ${placed} entities from ${result.name}.${offset}`
          + (result.warning ? ` ${result.warning}` : ""),
        refresh: true,
      };
    },
  },
  "Clear Alerts": {
    describe: "Mark every notification read.",
    async run() {
      const result = await call("mark_all_notifications_read");
      return { message: `Marked ${result.cleared ?? 0} read.`, refresh: true };
    },
  },

  /* These four are implemented, tested, and cannot be driven from this interface yet.
     Each needs the 3D viewer: a camera to save, a model to cut against, and a selected
     raster to scale. Declaring that is the honest state -- a button that always answers
     "no camera pose yet" would be the same dead control this audit exists to remove. */
  "Save View": {
    describe: "Remember the 3D camera: where it stands and what it looks at.",
    async run(ctx) {
      const viewer = await openModel(ctx);
      const name = await ctx.prompt("View name", "");
      if (!name) return { skipped: "A view needs a name to be opened again." };
      const camera = viewer.getCamera();
      const saved = await call("save_view", name, camera.position, camera.target, camera.fov_deg);
      return { message: `Saved view "${name}". ${(saved.views || []).length} view(s) in this project.` };
    },
  },
  "Open View": {
    describe: "Put the 3D camera back where a saved view left it.",
    async run(ctx) {
      const listed = await call("list_views");
      const views = listed.views || [];
      if (!views.length) return { skipped: "No saved views yet. Use Save View first." };
      const name = await ctx.choose("Saved view", views.map((v) => v.name));
      if (!name) return { skipped: "No view chosen." };
      const opened = await call("open_view", name);
      const viewer = await openModel(ctx);
      viewer.setCamera(opened.view.position, opened.view.target, opened.view.fov_deg);
      return { message: `Opened view "${name}".` + (opened.warning ? ` ${opened.warning}` : "") };
    },
  },
  Clip: {
    describe: "Cut the model with a plane. Needs the interactive 3D viewer.",
    async run() {
      return { skipped: "add_plane_clip is implemented, but a cut plane has to be placed "
                        + "against a model on screen, and the 3D viewer is not built." };
    },
  },
  "Temperature Range": {
    describe: "Set the palette range. Needs a selected radiometric raster.",
    async run() {
      return { skipped: "scale_thermal is implemented, but it takes a radiometric raster "
                        + "and no thermal product is selectable in this build." };
    },
  },

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

  "Verify Models": {
    describe: "Re-hash every installed model against its recorded digest.",
    async run() {
      const report = await call("verify_models");
      const rows = report.models || [];
      const mismatched = rows.filter((r) => r.status === "mismatch");
      if (mismatched.length) {
        return {
          message: `${mismatched.length} model(s) do not match their recorded digest: `
            + mismatched.map((r) => r.model_key).join(", ")
            + ". Their published metrics describe a different file.",
        };
      }
      const verified = rows.filter((r) => r.verified).length;
      return { message: `${verified} of ${rows.length} model(s) verified against their digests.` };
    },
  },

  "Match Captures": {
    describe: "Compare what was captured against what was planned.",
    async run() {
      // compare_survey_specifications takes the two mission versions to difference.
      // This called it with none, so the button's only possible outcome was
      // "TypeError: Api.compare_survey_specifications() missing 2 required positional
      // arguments" shown to the user as the result. The versions are fetched here and
      // the honest refusal is given when there are not two of them to compare.
      const listed = await tryCall("list_mission_versions");
      const versions = ((listed && (listed.versions || listed)) || [])
        .map((v) => Number(v.version_num ?? v.version ?? 0))
        .filter((n) => n > 0)
        .sort((a, b) => a - b);

      if (versions.length < 2) {
        return {
          message:
            `Match Captures compares two saved survey versions; this project has ` +
            `${versions.length}. Save the mission again after the next flight to ` +
            `difference them.`,
        };
      }

      const [previous, latest] = versions.slice(-2);
      const result = await call("compare_survey_specifications", previous, latest);
      return {
        message:
          result.summary || `Compared survey version ${previous} against ${latest}.`,
      };
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
      const chosen = ctx.reportOptions ? ctx.reportOptions() : {};

      // DOCX only exists on the export_report path; the report engine renders HTML and
      // then PDF. Sending a Word request to the engine would produce a PDF named
      // correctly and be wrong in the one way nobody checks.
      if (chosen.format === "docx") {
        const written = await call("export_report", "docx", title,
                                   chosen.organization || "");
        return { message: `Word report written to ${written.path}.`, refresh: true };
      }

      const result = await call("generate_report", "", title,
                                chosen.reportType || "standard",
                                chosen.organization || "");
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

/** One line an operator can read, from any of the measurement results. */
export function describeMeasurement(result) {
  if (!result) return "No measurement returned.";
  if (result.refused) return `Refused: ${result.refused}`;
  const round = (v, d = 2) => Number(v).toLocaleString(undefined, { maximumFractionDigits: d });
  const parts = [];
  const m = result.measurement;
  if (m && m.value !== undefined && m.unit) {
    const unit = String(m.unit).replace("square metre", "m²").replace("metre", "m");
    parts.push(`${m.kind} ${round(m.value)} ${unit}`);
    if (result.surface_length_m !== undefined) {
      parts.push(`over the ground ${round(result.surface_length_m)} m`);
    }
  }
  const preferred = result.preferred;
  if (preferred && preferred.cut_volume_m3 !== undefined) {
    const against = preferred.reference === "plane"
      ? `a ${round(preferred.reference_elevation_m)} m plane` : `the ${preferred.reference}`;
    parts.push(`${round(preferred.cut_volume_m3)} m³ above and `
      + `${round(preferred.fill_volume_m3)} m³ below ${against}, over `
      + `${round(preferred.area_m2)} m²`);
  }
  if (result.slope) {
    parts.push(`slope mean ${round(result.slope.mean_deg, 1)}° `
      + `(${round(result.slope.mean_percent, 1)}%), plane dip `
      + `${round(result.slope.plane_dip_deg, 1)}°`);
  }
  if (result.planar_length_m !== undefined && result.elevation_min_m !== undefined) {
    parts.push(`profile ${round(result.planar_length_m)} m long, elevation `
      + `${round(result.elevation_min_m)}–${round(result.elevation_max_m)} m`);
  }
  if (result.warning) parts.push(result.warning);
  return parts.length ? parts.join("; ") + "." : "Measured.";
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
