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
import { demoEnabled } from "./demo.js";
import { armModelTool, currentViewer, queueModelTool } from "./viewer3d.js";
import { setView } from "./viewstate.js";

/* Where the REST service listens when started with its documented command. */
const REST_BASE = "http://127.0.0.1:8000";

/** Actions that need an open project before they mean anything. */
const NEEDS_PROJECT = new Set([
  "New Mission", "Plan", "Save", "Save Template", "Export", "Export Products",
  "Process", "New Job", "Start", "Run AI", "Add GCPs", "Set CRS", "Compare", "Compare Dates",
  "Change", "Volume", "Cut & Fill", "Stockpile", "Slope", "Ponding", "Generate Report",
  "Export Report", "Export PDF", "Export Word", "Open Report", "Inspection Report",
  "Record Finding", "Annotate",
  "New Report", "Share", "Revoke Links", "Archive",
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

// Run a thermal call; when the raster does not say its values are temperatures, ask
// the user for the unit and retry with their answer. Never assume a unit.
async function callAskingUnit(ctx, run) {
  try {
    return await run("");
  } catch (error) {
    if (!(error.result && error.result.needs_unit_confirmation)) throw error;
    const units = { "Degrees Celsius": "celsius", Kelvin: "kelvin" };
    const unit = await ctx.choose(`${error.message} What unit are its values in?`,
      [...Object.keys(units), "Not temperatures / not sure"]);
    if (!unit || !units[unit]) return { skipped: "The raster's unit was not confirmed; nothing was read." };
    return run(units[unit]);
  }
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
    // The pack can restrict zones to canopy pixels; the button never offered it.
    const masked = await ctx.choose("Restrict the zones to canopy?", [
      "No canopy mask", "Choose a canopy mask raster"]);
    const mask = masked && masked.startsWith("Choose") ? await pickPath(["tif", "tiff"], "canopy mask") : "";
    return call("india_stress_zones", raster, severe, moderate, scope, mask);
  },
  "Plant count": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "instance mask");
    const scope = await required(ctx, "Validation scope of the instance model");
    // A label-id mask keeps touching crowns apart; read as binary they merge into one.
    const encoding = await ctx.choose("How the mask marks plants", [
      "binary (every non-zero pixel is plant)", "label_ids (one id per plant)"]);
    if (!encoding) throw new Error("No mask encoding chosen.");
    return call("india_plant_count", raster, scope, encoding.split(" ")[0]);
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
    const withLayout = await ctx.choose("Compare with an approved layout?", [
      "No approved layout", "Choose an approved layout"]);
    const layout = withLayout && withLayout.startsWith("Choose")
      ? await pickPath(["geojson", "json"], "approved layout") : "";
    // Validated findings per module were accepted by the pack but never offered here.
    const withFindings = await ctx.choose("Attach validated module findings?", [
      "No findings", "Choose validated findings"]);
    const findings = withFindings && withFindings.startsWith("Choose")
      ? await pickPath(["geojson", "json"], "validated module findings") : "";
    return call("india_solar_inventory", raster, scope, layout, findings);
  },
  "Module temperatures": async (ctx) => {
    // Map-placed module temperatures need a georeferenced thermal source: a radiometric
    // sidecar with georeferencing, or a temperature orthomosaic GeoTIFF
    // (core/india_thermal.py). A single camera frame is refused there with the reason;
    // "Module temperatures (single frame)" below is the path for those.
    const thermal = await pickPath(["json", "tif", "tiff"],
      "georeferenced thermal source (sidecar .json or temperature GeoTIFF)");
    const rgb = await pickPath(["tif", "tiff", "jpg", "png"], "RGB image");
    const points = await pickPath(["json", "csv"], "RGB/thermal tie points");
    const modules = await pickPath(["geojson", "json"], "module polygons");
    const by = await required(ctx, "Tie points validated by");
    const warning = await number(ctx, "Warning above module median (°C)");
    const critical = await number(ctx, "Critical above module median (°C)");
    const hot = await number(ctx, "Hot cell above module median (°C)");
    const basis = await required(ctx, "Where these thresholds come from (standard, contract)");
    return callAskingUnit(ctx, (unit) => call("india_module_temperatures", thermal, rgb, points,
      modules, by, warning, critical, hot, basis, unit));
  },
  "Module temperatures (single frame)": async (ctx) => {
    // One radiometric frame (FLIR JPEG, DJI R-JPEG via the DJI Thermal SDK, FLIR
    // TLinear TIFF or sidecar) with operator-drawn module polygons in its pixel grid.
    const thermal = await pickPath(["jpg", "jpeg", "tif", "tiff", "json"], "radiometric thermal frame");
    const regions = await pickPath(["json"], "module regions in thermal pixels (.json)");
    const by = await required(ctx, "Module regions drawn by");
    const warning = await number(ctx, "Warning above module median (°C)");
    const critical = await number(ctx, "Critical above module median (°C)");
    const hot = await number(ctx, "Hot cell above module median (°C)");
    const basis = await required(ctx, "Where these thresholds come from (standard, contract)");
    return callAskingUnit(ctx, (unit) => call("india_module_temperatures_in_image", thermal,
      regions, by, warning, critical, hot, basis, unit));
  },
  "Anomaly baseline": async (ctx) => {
    // A baseline is fitted to at least two independently captured normal rasters; one
    // file was all this button could ever send, so it was refused every time.
    const rasters = [await pickPath(["tif", "tiff"], "raster confirmed normal")];
    for (;;) {
      const more = await ctx.choose(`Anomaly baseline: ${rasters.length} normal raster(s) chosen`,
        ["Add another raster confirmed normal", "Done choosing rasters"]);
      if (!more || more.startsWith("Done")) break;
      rasters.push(await pickPath(["tif", "tiff"], "raster confirmed normal"));
    }
    const by = await required(ctx, "Confirmed normal by");
    const scope = await required(ctx, "Where and when this baseline applies");
    return call("india_anomaly_baseline", rasters, by, scope);
  },
  "Find anomalies": async (ctx) => {
    const raster = await pickPath(["tif", "tiff"], "feature raster");
    const baseline = await pickPath(["json"], "validated baseline");
    const sigma = await number(ctx, "Threshold (robust sigma)", "3.5");
    return call("india_find_anomalies", raster, baseline, sigma);
  },
};

const CAPTURE_GEOMETRIES = {
  power: ["close_range_oblique", "close_range_orbit"],
  rail: ["corridor_nadir", "corridor_nadir_oblique"],
};

async function corridor(ctx, kind) {
  const detections = await pickPath(["geojson", "json"], `${kind} detections`);
  const centreline = await pickPath(["geojson", "json"], "corridor centreline");
  // The values the pack accepts (core/india_assets.py); the prompt used to suggest
  // "oblique_close_range", which it refuses.
  const geometry = await ctx.choose("Capture geometry", CAPTURE_GEOMETRIES[kind]);
  if (!geometry) throw new Error("No capture geometry chosen.");
  const scope = await required(ctx, "Validation scope of the detector");
  return call("india_corridor_inspection", kind, detections, centreline, geometry, scope);
}

/** A line about what a pack produced: its own status and headline numbers. */
function summarisePack(result) {
  const summary = result.summary || {};
  const parts = [];
  if (result.baseline_path) return `baseline written to ${result.baseline_path}.`;
  for (const key of ["observed_surface_coverage_percent", "canopy_cover_percent",
    "canopy_area_m2", "surveyed_distance_m", "road_surface_area_m2", "module_count",
    "instance_count", "count", "candidate_count", "status"]) {
    if (summary[key] !== undefined && typeof summary[key] !== "object") parts.push(`${key.replace(/_/g, " ")} ${summary[key]}`);
  }
  // Stress zones and vegetation indices had only "done" to show for themselves.
  if (summary.zones) {
    parts.push(Object.entries(summary.zones)
      .map(([name, zone]) => `${name.replace(/_/g, " ")} ${zone && zone.area_m2 !== undefined ? `${zone.area_m2} m²` : ""}`).join(", "));
  }
  if (summary.indices) {
    parts.push(Object.entries(summary.indices)
      .map(([name, index]) => `${name} ${index && index.status ? index.status : ""}`).join(", "));
  }
  if (summary.categories) {
    parts.push(Object.entries(summary.categories)
      .map(([name, value]) => `${name} ${value && value.area_m2 !== undefined ? `${value.area_m2} m²` : ""}`).join(", "));
  }
  if (summary.totals) parts.push(`${Object.keys(summary.totals).length} total(s)`);
  const layers = (result.layer_ids || []).length;
  return `${parts.filter(Boolean).join("; ") || "done"}. ${layers} layer(s) added; summary ${result.summary_path}.`;
}

/** Actions that need images on disk. */
/* Which aircraft Log Maintenance and Assign Mission act on. With a battery or pilot
   picked last, the old fallback asked for an "Aircraft id" pre-filled with 1, so one
   Enter logged maintenance against whichever aircraft happened to be number 1. The
   last-picked row now has to be an aircraft; with nothing picked the id is asked for
   with no default. */
async function pickAircraft(ctx) {
  const picked = ctx.selectedFleetRecord();
  if (picked && picked.kind !== "aircraft") {
    return { skipped: `The ${picked.kind} "${picked.name}" is selected. `
      + "Select an aircraft in the Aircraft list first." };
  }
  const selected = ctx.selectedFleetId();
  if (selected) return { id: selected };
  const typed = await ctx.prompt("Aircraft id (the number shown in the Aircraft list)", "");
  const id = Number(String(typed ?? "").trim());
  if (!typed || !Number.isInteger(id) || id <= 0) {
    return { skipped: "Select an aircraft in the fleet list first." };
  }
  return { id };
}

const NEEDS_DATASET = new Set(["Process", "New Job", "Start", "Run AI", "Match Captures",
                               "Coverage QA",
                               "Open in 3D", "Label"]);

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
      // The Api answers ok(dataset={... metadata_json}), so result.image_count was always
      // undefined and every import reported "Imported ? images".
      let meta = {};
      try {
        meta = JSON.parse((result.dataset && result.dataset.metadata_json) || "{}");
      } catch {
        meta = {};
      }
      const count = meta.image_count ?? result.image_count;
      const tagged = meta.geotagged_count;
      return {
        message: count == null
          ? `Imported ${folder.path}.`
          : `Imported ${count} image(s)${tagged != null ? `, ${tagged} geotagged` : ""}.`,
        refresh: true,
      };
    },
  },

  /* -- mission planning --------------------------------------------------- */
  Plan: {
    describe: "Plan the mission from the current AOI and parameters.",
    async run(ctx) {
      const result = await call("plan_mission", ctx.missionOptions());
      // The Api answers ok(summary={waypoints, distance_m, ...}, warnings); reading
      // result.waypoint_count printed "Planned ? waypoints" after every successful plan.
      const summary = result.summary || {};
      const warnings = result.warnings || [];
      return {
        message: `Planned ${summary.waypoints ?? "?"} waypoints, `
          + `${((summary.distance_m || 0) / 1000).toFixed(2)} km, ${summary.duration_min ?? "?"} min`
          + (summary.gsd_cm ? `, GSD ${summary.gsd_cm} cm` : "")
          + (warnings.length ? `. ${warnings.join(" ")}` : "."),
        refresh: true,
      };
    },
  },
  "New Mission": { alias: "Plan" },
  Save: {
    describe: "Save the planned mission into the project.",
    async run(ctx) {
      // Asked for a name and THEN learned there was nothing to save.
      const state = await call("get_state");
      if (!state.has_mission) return { skipped: "No mission has been planned yet. Plan one first." };
      const name = await ctx.prompt("Mission name", "Mission");
      if (!name) return { skipped: "No name given." };
      await call("save_mission", name);
      return { message: `Saved "${name}".`, refresh: true };
    },
  },
  Export: {
    describe: "Export the mission, or the products when there is no mission.",
    async run(ctx) {
      // On the Fleet screen the thing on screen is the fleet register, not a mission.
      if (ctx.workspaceId && ctx.workspaceId() === "fleet") {
        const fleet = await call("export_fleet", ctx.organizationId());
        const c = fleet.counts;
        return {
          message: `Exported the fleet register (${c.aircraft} aircraft, ${c.batteries} `
            + `batteries, ${c.pilots} pilots, ${c.maintenance} maintenance records) to `
            + `${fleet.directory}.`,
          files: Object.values(fleet.files || {}),
        };
      }
      // tryCall, not call: call() THROWS when the API refuses, so a fallback written
      // after it can never run. The first version of this used call() and the mission
      // refusal propagated straight past the products branch, which is why the button
      // still said "No mission has been planned yet" on screens that had products.
      const result = await tryCall("export_mission");
      // export_mission answers ok(written={format: path | "failed: ..."}, directory).
      // Reading result.files found nothing, so a successful export of five mission files
      // fell through to revealing the reconstruction products and said so instead.
      const written = Object.entries((result && result.written) || {});
      if (written.length) {
        const failed = written.filter(([, v]) => String(v).startsWith("failed"));
        const files = written.filter(([, v]) => !String(v).startsWith("failed")).map(([, v]) => v);
        return {
          message: `Exported ${files.length} mission file(s) to ${result.directory}`
            + (failed.length ? `; ${failed.map(([k, v]) => `${k} ${v}`).join("; ")}` : "."),
          files,
        };
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
      // The asset declaration from Run AI Settings travels with the run; without it
      // the solar and metal passes could not be reached from any screen. It builds on
      // the pipeline options, so coverage QA still gets the flight height.
      const options = ctx.inspectionOptions ? ctx.inspectionOptions() : ctx.pipelineOptions();
      const result = await call("run_pipeline", options);
      return { message: "Pipeline started.", job: result.job_id, refresh: true };
    },
  },
  "Run AI": { alias: "Start" },

  /* -- models, one image at a time ----------------------------------------
     Six installed, verified models had no caller, and corrosion grading had no Api.
     Each of these runs the real model and shows what it said beside the registry's own
     scope note, because several are valid on one kind of imagery only. */
  "Run Model": {
    describe: "Run any installed model on one image and show its output.",
    async run(ctx) {
      const status = await call("model_status");
      const options = (status.models || []).filter((m) => m.exists).map((m) => ({
        label: m.key, value: m.key, hint: (m.spec && m.spec.kind) || "",
      }));
      const key = await ctx.choose("Run which installed model?", options);
      if (!key) return { skipped: "No model chosen." };
      const image = await pickImage(ctx);
      if (!image) return { skipped: "No image chosen." };
      const result = await call("run_model", key, image);
      await ctx.showResult(`${key} \u2014 ${basename(image)}`, {
        image: result.overlay_data_uri, lines: describeModelRun(result),
      });
      return { message: `${key}: ${summariseModelRun(result)}` };
    },
  },
  "Prelabel": {
    describe: "Run an installed detector; its detections become findings to review.",
    async run(ctx) {
      const status = await call("model_status");
      const detectors = (status.models || []).filter((m) => m.exists && m.spec
        && ["onnx_yolo", "onnx_detection"].includes(m.spec.kind));
      const key = await ctx.choose("Pre-label with which detector?", detectors.map((m) => ({
        label: m.key, value: m.key, hint: (m.spec.labels || []).join(", "),
      })));
      if (!key) return { skipped: "No detector chosen." };
      const image = await pickImage(ctx);
      if (!image) return { skipped: "No image chosen." };
      const result = await call("prelabel_image", image, key);
      return {
        message: `${result.count} finding(s) from ${key} are waiting for review.`,
        refresh: true,
      };
    },
  },
  "Grade Corrosion": {
    describe: "Grade corrosion severity per pixel, or refuse without the model.",
    async run(ctx) {
      const image = await pickImage(ctx);
      if (!image) return { skipped: "No image chosen." };
      const result = await call("grade_corrosion", image);
      await ctx.showResult(`Corrosion severity \u2014 ${basename(image)}`, {
        image: result.overlay_data_uri, lines: describeModelRun(result),
      });
      return { message: `Worst grade present: ${result.output.worst_present_grade || "none"}.` };
    },
  },
  "Land Cover": {
    confirm: "Run the land-cover model over this project's orthomosaic?",
    describe: "Land-cover segmentation of the orthomosaic.",
    async run(ctx) {
      const answer = await ctx.ask("Land cover", [{
        key: "cpu", label: "If no GPU is available, run on the CPU (slow)?",
        options: ["no", "yes"], value: "no",
      }]);
      if (!answer) return { skipped: "Cancelled." };
      const result = await call("run_landcover", "", answer.cpu === "yes");
      return { message: "Land-cover segmentation started.", job: result.job_id, refresh: true };
    },
  },

  /* -- labelling and custom models ------------------------------------------ */
  Label: {
    describe: "Draw defect boxes on a dataset image for a custom model.",
    async run(ctx) {
      const listing = await call("list_dataset_images");
      const name = await ctx.choose("Label which image?", (listing.images || []).map((n) => ({
        label: n, value: n,
      })));
      if (!name) return { skipped: "No image chosen." };
      const saved = await call("list_label_boxes");
      const path = joinPath(listing.folder, name);
      const previous = (saved.entries || []).find((e) => e.path === path);
      const classText = await ctx.prompt("Classes, comma separated",
        (saved.classes && saved.classes.length ? saved.classes : ["crack", "spalling", "corrosion"]).join(", "));
      if (!classText) return { skipped: "A labeller needs at least one class." };
      const classes = classText.split(",").map((c) => c.trim()).filter(Boolean);
      const preview = await call("image_preview", name);
      const payload = await ctx.labelBoxes(`Label ${name}`, {
        imageUri: preview.data_uri, path, classes,
        existing: previous ? previous.boxes : [],
      });
      if (!payload) return { skipped: "Labels not saved." };
      const stored = await call("save_label_boxes", payload);
      return { message: `Saved ${payload.boxes.length} box(es) on ${name}. `
        + `${stored.images} image(s), ${stored.boxes} box(es) labelled in this project.` };
    },
  },
  "Build Corpus": {
    confirm: "Build a training corpus from every image labelled in this project?",
    describe: "Split the labelled boxes into a trainable corpus, or say why not.",
    async run() {
      const result = await call("build_detection_corpus", null, "");
      // counts[split] is {images, <class>: boxes}; the image count is what a split holds.
      const counts = result.counts || {};
      const images = (split) => (counts[split] && counts[split].images) || 0;
      return {
        message: `Corpus at ${result.output_dir}: ${result.total_images} image(s), `
          + `${result.total_boxes} box(es); train ${images("train")}, val ${images("val")}, `
          + `test ${images("test")} image(s).`,
      };
    },
  },
  "Train Model": {
    confirm: "Train a detector on a corpus from this project? This can take hours.",
    describe: "Train a custom detector, then deploy it once its metrics are read.",
    async run(ctx) {
      const artifacts = await call("list_training_artifacts");
      const choice = await ctx.choose("Train on which corpus, or deploy a finished run?", [
        ...(artifacts.corpora || []).map((c) => ({
          label: `Train on ${c.name}`, value: { corpus: c.path, name: c.name },
          hint: `${c.total_images} images, ${c.total_boxes} boxes: ${(c.classes || []).join(", ")}`,
        })),
        ...(artifacts.runs || []).map((r) => ({
          label: `Deploy ${r.key}`, value: { run: r.run_dir, key: r.key },
          hint: r.metrics && r.metrics.map50 != null
            ? `val mAP50 ${Number(r.metrics.map50).toFixed(3)}` : "no validation metric",
        })),
      ]);
      if (!choice) return { skipped: "Nothing chosen." };
      if (choice.run) {
        const deployed = await call("deploy_custom_model", choice.run, choice.key);
        return { message: `Deploying ${deployed.key}.`, job: deployed.job_id, refresh: true };
      }
      const settings = await ctx.ask("Training", [
        { key: "name", label: "Model name", value: "custom_detector" },
        { key: "epochs", label: "Epochs", value: "50", type: "number" },
        { key: "size", label: "Image size (px)", value: "640", type: "number" },
      ]);
      if (!settings) return { skipped: "Cancelled." };
      const started = await call("train_custom_model", settings.name, choice.corpus,
        Number(settings.epochs), Number(settings.size));
      return { message: `Training ${started.key}.`, job: started.job_id, refresh: true };
    },
  },
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

  /* -- flight -------------------------------------------------------------
     The cockpit could not connect an aircraft at all: fly.connect only switched
     workspace and there was no Connect button, so every Flight panel stayed on "No
     aircraft connected" forever. */
  Connect: {
    describe: "Connect to an aircraft over MAVLink, MAVSDK, or the simulator.",
    async run(ctx) {
      const driver = await ctx.choose("Connect vehicle", [
        { label: "MAVLink autopilot (ArduPilot / PX4 via pymavlink)", value: "pymavlink",
          hint: "real aircraft, SITL, or a Mission Planner forward" },
        { label: "PX4 via MAVSDK", value: "mavsdk", hint: "needs the mavsdk package" },
        { label: "Simulated vehicle", value: "mock", hint: "UI rehearsal; nothing is measured" },
      ]);
      if (driver === null) return { skipped: "Cancelled." };
      const fallback = driver === "mavsdk" ? "udp://:14540"
        : driver === "mock" ? "mock://" : "udpin:0.0.0.0:14550";
      const uri = await ctx.prompt("Connection (udpin:0.0.0.0:14550, tcp:127.0.0.1:5760, COM3…)", fallback);
      if (!uri) return { skipped: "No connection given." };
      const result = await call("connect_vehicle", uri, driver);
      const vehicle = result.vehicle || {};
      /* connect_vehicle answers ok with connected:false and the reason when the link
         fails, so the reason is what gets shown -- not "Connected". */
      if (vehicle.connected === false) throw new Error(vehicle.last_error || "Connection failed.");
      return {
        message: `Connected to ${uri} via ${driver}`
          + (vehicle.is_simulated ? " — SIMULATED vehicle." : "."),
        refresh: true,
      };
    },
  },
  Disconnect: {
    describe: "Close the vehicle link.",
    async run() {
      await call("disconnect_vehicle");
      return { message: "Vehicle disconnected.", refresh: true };
    },
  },
  /* Preflight listed the Python libraries this build has -- torch, colmap -- under a
     button that sits beside Start on the screen that arms an aircraft. It now runs the
     aircraft checks. The library list is still one menu item away, as Capabilities. */
  Preflight: {
    describe: "Check the aircraft and the planned mission before flying.",
    async run() {
      const result = await call("run_preflight");
      return { message: describePreflight(result), refresh: true };
    },
  },
  "Confirm Checks": {
    describe: "Record that you checked the items nothing on the aircraft can report.",
    async run(ctx) {
      const result = await call("run_preflight");
      const pending = (result.checks || []).filter((c) => c.requires_manual && !c.confirmed);
      if (!pending.length) return { skipped: "Nothing is waiting for a manual check." };
      const chosen = await ctx.choose("Confirm a check you have made yourself",
        pending.map((c) => ({ label: c.name, value: c.id, hint: c.fix_action || c.message })));
      if (chosen === null) return { skipped: "Cancelled." };
      const after = await call("run_preflight", [chosen]);
      return { message: `Confirmed. ${describePreflight(after)}`, refresh: true };
    },
  },
  Capabilities: {
    describe: "What this build can and cannot do.",
    async run() {
      const result = await call("capabilities");
      const caps = result.capabilities || result;
      const missing = Object.entries(caps)
        .filter(([, value]) => value === false)
        .map(([name]) => name);
      /* "Unavailable: dense_stereo" alone left the operator guessing why; the probe
         names the cause and the fix, so it travels with the list. */
      const why = caps.dense_stereo === false && caps.dense_stereo_reason
        ? ` Dense stereo: ${caps.dense_stereo_reason}` : "";
      return {
        message: missing.length
          ? `Unavailable: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "…" : ""}.${why}`
          : "All capabilities available.",
      };
    },
  },
  Upload: {
    confirm: "Upload the mission to the connected aircraft?",
    describe: "Send the mission, geofence and rally points over MAVLink.",
    async run() {
      /* call() throws on ok:false, so a vehicle that rejects the upload reaches the
         operator as an error. It used to say "Mission uploaded." either way. */
      const result = await call("upload_mission");
      const upload = result.upload || {};
      const extras = ["fence", "rally"]
        .filter((key) => upload[key] && upload[key].success)
        .map((key) => `${upload[key].count} ${key}`);
      return {
        message: `The vehicle accepted ${result.items ?? "?"} mission items`
          + (extras.length ? ` and ${extras.join(", ")} items.` : "."),
        refresh: true,
      };
    },
  },
  Start_Flight: { alias: "Upload" },
  /* Its own verb. The Flight toolbar's "Start" resolved to the pipeline action and
     offered to run a reconstruction from the screen that flies the aircraft. */
  "Start Mission": { vehicle: "start", confirm: "Arm and start the uploaded mission?" },
  Pause: { vehicle: "pause", confirm: "Pause the aircraft?" },
  Resume: { vehicle: "resume", confirm: "Resume the mission?" },
  RTL: { vehicle: "rth", confirm: "Return to launch?" },
  Land: { vehicle: "land", confirm: "Land now?" },
  Abort: { vehicle: "abort", confirm: "ABORT the flight?" },
  "Manual Override": {
    confirm: "Take manual control?",
    describe: "Hand control to the operator.",
    async run() {
      const result = await call("take_manual_control");
      return { message: result.note || `The aircraft is in ${result.mode || "a pilot mode"}.` };
    },
  },
  "Capture Now": {
    describe: "Trigger the camera.",
    async run() {
      /* "trigger" is not a camera action the Api knows; take_photo is. */
      const result = await call("camera_command", "take_photo");
      return { message: result.message || "The camera accepted the trigger." };
    },
  },
  /* Sat on the Verification toolbar and called check_interrupted_flight, which answers
     whether the software stopped mid-flight -- nothing to do with loading the flight to
     verify. Loading a flight here means choosing which imported image set is the flown
     one, which is what Match Captures then checks against the plan. */
  "Load Flight": {
    /* Two screens carry this button and mean different things by it. On Flight it is
       crash recovery: the record of a flight that was interrupted when this software
       last stopped. On Verification it chooses which imported flight's images Match
       Captures checks against the plan. */
    describe: "Recover an interrupted flight (Flight), or choose the flown images to verify (Verification).",
    async run(ctx) {
      if (ctx.workspaceId && ctx.workspaceId() === "verification") {
        const listed = await call("list_datasets");
        const sets = listed.datasets || [];
        if (!sets.length) return { skipped: "No flights imported yet. Use Import on Home or Projects." };
        const path = await ctx.choose("Verify which flight?", sets.map((d) => ({
          label: d.name || d.path, value: d.path, hint: d.path,
        })));
        if (!path) return { skipped: "No flight chosen." };
        await call("set_active_dataset", path);
        return { message: `Loaded ${path}. Match Captures compares it with the plan.`, refresh: true };
      }
      const result = await call("check_interrupted_flight");
      /* The Api answers `recovered`; reading `interrupted` said "No interrupted flight"
         over a record saying a mission was on the aircraft. */
      return {
        message: result.recovered
          ? (result.summary || `Interrupted flight found: ${result.state?.phase || "unknown phase"}.`)
          : "No interrupted flight.",
      };
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
  /* Called check_spatial_reference with no images at all -- "missing 1 required
     positional argument: image_paths" -- and, had it worked, set nothing. It now takes
     the output EPSG for the next reconstruction (run_reconstruction reads `epsg`),
     checks what the selected images can support in that frame, and keeps the choice. */
  "Set CRS": {
    describe: "Choose the projected CRS reconstructions are written in, and check the images.",
    async run(ctx) {
      const current = await call("get_project_crs");
      const answer = await ctx.prompt(
        "Output EPSG code of a projected CRS (blank = automatic UTM from the geotags)",
        current.epsg ? String(current.epsg) : "");
      if (answer === null) return { skipped: "Cancelled." };
      const trimmed = String(answer).trim();
      const epsg = trimmed ? Number(trimmed) : null;
      if (trimmed && !Number.isInteger(epsg)) return { skipped: `Not an EPSG code: ${trimmed}` };
      // Kept with the project (run_pipeline and run_reconstruction read it), and refused
      // when it is not a projected CRS in metres.
      const kept = await call("set_project_crs", epsg);
      ctx.setSetting("epsg", epsg || "");
      // What the selected images can support in that frame.
      let check = "";
      const listed = await tryCall("list_dataset_images");
      if (listed && listed.folder && (listed.images || []).length) {
        const sep = String(listed.folder).includes("\\") ? "\\" : "/";
        const paths = listed.images.map((name) => `${listed.folder}${sep}${name}`);
        const gcps = await tryCall("gcp_accuracy_report");
        const gcpCount = (gcps && gcps.report && gcps.report.point_count) || 0;
        const result = await call("check_spatial_reference", paths, gcpCount, epsg);
        check = ` ${result.geotagged_count}/${result.image_count} images geotagged, `
          + `${result.gcp_count} control point(s): ${result.note}`;
      }
      return {
        message: `${kept.epsg ? `Output CRS EPSG:${kept.epsg} (${kept.name})` : "Output CRS: automatic UTM"} `
          + `for the next reconstruction.${check}`,
      };
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
      /* The Api reports area_hectares; reading area_m2 dropped the area every time. */
      const hectares = result.area_hectares ?? (result.area_m2 ? result.area_m2 / 10000 : null);
      return {
        message: `Boundary closed from ${result.corner_count ?? "?"} corner(s)`
          + (hectares != null ? `, ${Number(hectares).toFixed(2)} ha.` : "."),
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

  /* Said "the interactive viewer is not built". It opens the model in the Digital Twin
     with the measure tool armed: two points picked on the model, measured in metres by
     measure_on_model through the reconstruction's georeference. With the model already
     on screen it also offers the volume the closed mesh encloses (measure_in_model),
     which refuses a mesh that does not close rather than summing it. */
  "Measure in 3D": {
    describe: "Measure between points picked on the 3D model, or the volume it encloses.",
    async run(ctx) {
      if (currentViewer()) {
        const kind = await ctx.choose("Measure in 3D", [
          { label: "Between points picked on the model", value: "picks" },
          { label: "Volume enclosed by the model", value: "volume",
            hint: "closed meshes only" },
        ]);
        if (!kind) return { skipped: "Nothing chosen." };
        if (kind === "volume") {
          const result = await call("measure_in_model", "", "volume");
          const m = result.measurement || {};
          return { message: `Model volume ${Number(m.volume_m3).toFixed(2)} m³`
            + (m.crs_epsg ? `, EPSG:${m.crs_epsg}` : "") + ". Kept in the measurement history." };
        }
      }
      queueModelTool("measure");
      setView("mesh");
      ctx.openWorkspace("twin");
      return { message: "Measure in 3D: click two points on the model." };
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
      /* files is {csv: path, json: path, …}, not an array: files.length was undefined,
         so this reported "No flight has been recorded yet" after writing four files. */
      const files = Object.values(result.files || {});
      if (!files.length) return { skipped: "No flight has been recorded yet." };
      return {
        message: (result.simulated ? "SIMULATED log — " : "")
          + `Wrote ${files.length} file(s): ${files.join(", ")}`,
      };
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
      /* Read verdict, fits and peak_ram_gb, none of which size_job returns -- so every
         dataset was reported "may not fit" beside a note that said it would. The verdict
         is can_run_as_one_job, and the note already states it with the figures. */
      const warnings = (result.warnings || []).join(" ");
      /* Below the memory one image needs, no chunk size helps; this used to recommend
         "chunks of about 1" that did not fit either. */
      const verdict = result.can_run_as_one_job ? "Fits"
        : result.fits_one_image === false ? "Cannot run here now"
        : "Does not fit as one job";
      return {
        message: `${verdict}: `
          + `${result.note || `${count} images`}${warnings ? ` ${warnings}` : ""}`,
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

  /* Save View and Clip said the 3D viewer was not built. They act on the model on screen
     (viewer3d.js): a view is the camera of that viewer, in the model's own coordinates,
     and a cut is a plane through a point picked on it. */
  "Save View": {
    describe: "Save the 3D camera as a named view, or open a saved one.",
    async run(ctx) {
      const viewer = currentViewer();
      if (!viewer) {
        return { skipped: "No 3D model on screen. Open Textured Mesh or Point Cloud in the "
                          + "Digital Twin first; a view is that camera." };
      }
      const listed = await call("list_views");
      const views = listed.views || [];
      let choice = "save";
      if (views.length) {
        choice = await ctx.choose("Views", [
          { label: "Save the current camera as a new view", value: "save" },
          ...views.map((v) => ({ label: `Open “${v.name}”${v.is_default ? " (default)" : ""}`, value: `open:${v.name}` })),
          ...views.filter((v) => !v.is_default).map((v) => ({ label: `Open with “${v.name}” by default`, value: `default:${v.name}` })),
        ]);
        if (!choice) return { skipped: "Nothing chosen." };
      }
      if (choice.startsWith("open:")) {
        const opened = await call("open_view", choice.slice(5));
        viewer.applyView(opened.view);
        const missing = opened.missing_clips || [];
        return { message: `Opened view “${opened.view.name}”.`
          + (missing.length ? ` It refers to ${missing.length} clip(s) that no longer exist.` : "") };
      }
      if (choice.startsWith("default:")) {
        await call("set_default_view", choice.slice(8));
        return { message: `“${choice.slice(8)}” now opens by default.` };
      }
      const name = await ctx.prompt("Name this view", `view ${views.length + 1}`);
      if (!name) return { skipped: "No name given." };
      const cam = viewer.cameraState();
      const clips = (await call("list_clips")).clips || [];
      await call("save_view", name, cam.position, cam.target, cam.fov_deg,
        clips.filter((c) => c.visible).map((c) => c.name), false, true, views.length === 0);
      return { message: `Saved view “${name}”${views.length === 0 ? " (opens by default)" : ""}.` };
    },
  },
  "Open View": {
    describe: "Put the 3D camera back where a saved view left it.",
    async run(ctx) {
      const viewer = currentViewer();
      if (!viewer) {
        return { skipped: "No 3D model on screen. Open Textured Mesh or Point Cloud in the "
                          + "Digital Twin first; a view is that camera." };
      }
      const listed = await call("list_views");
      const views = listed.views || [];
      if (!views.length) return { skipped: "No saved views yet. Use Save View first." };
      const name = await ctx.choose("Saved view", views.map((v) => v.name));
      if (!name) return { skipped: "No view chosen." };
      const opened = await call("open_view", name);
      viewer.applyView(opened.view);
      const missing = opened.missing_clips || [];
      return { message: `Opened view “${opened.view.name}”.`
        + (missing.length ? ` It refers to ${missing.length} clip(s) that no longer exist.` : "") };
    },
  },
  Clip: {
    describe: "Cut the model with a plane through a picked point, or manage the cuts.",
    async run(ctx) {
      const viewer = currentViewer();
      if (!viewer) {
        return { skipped: "No 3D model on screen. Open Textured Mesh or Point Cloud in the "
                          + "Digital Twin first; a cut is placed on the model." };
      }
      const clips = (await call("list_clips")).clips || [];
      const choice = await ctx.choose("Clip", [
        { label: "New cut: click a point, the part between the camera and it is removed", value: "new" },
        ...clips.map((c) => ({ label: `${c.visible ? "Hide" : "Show"} “${c.name}”`, value: `toggle:${c.name}` })),
        ...clips.map((c) => ({ label: `Remove “${c.name}”`, value: `remove:${c.name}` })),
        ...(clips.some((c) => c.visible)
          ? [{ label: "Export the model as the visible cuts leave it", value: "export" }] : []),
      ]);
      if (!choice) return { skipped: "Nothing chosen." };
      if (choice === "new") {
        viewer.setTool("clip");
        return { message: "Clip: click a point on the model." };
      }
      if (choice === "export") {
        const done = await call("export_clipped_model", "");
        return { message: `Wrote ${done.faces.toLocaleString()} of ${done.source_faces.toLocaleString()} `
          + `triangles to ${done.path}.` };
      }
      const [verb, ...rest] = choice.split(":");
      const name = rest.join(":");
      if (verb === "toggle") {
        const clip = clips.find((c) => c.name === name);
        await call("set_clip_visible", name, !(clip && clip.visible));
      } else {
        await call("remove_clip", name);
      }
      await viewer.reload();
      return { message: `${verb === "toggle" ? "Toggled" : "Removed"} “${name}”.` };
    },
  },
  /* Answered "scale_thermal is implemented, but ... no thermal product is selectable in
     this build" and did nothing. The frame is now chosen from disk: a radiometric sidecar
     (raw counts plus the camera's Planck constants) is what core/thermal reads; a thermal
     JPEG is refused by the Api with the reason, since its pixels are a palette. */
  "Temperature Range": {
    describe: "Choose the temperature range a radiometric frame is drawn in.",
    async run(ctx) {
      // A radiometric sidecar, a FLIR radiometric JPEG, a DJI R-JPEG (needs the DJI
      // Thermal SDK), a FLIR TLinear TIFF, or a temperature GeoTIFF.
      const picked = await call("pick_file", ["json", "jpg", "jpeg", "tif", "tiff"]);
      if (!picked.path) return { skipped: "No radiometric frame chosen." };
      const modes = {
        "Automatic (1st to 99th percentile)": "auto",
        "Manual (enter min and max)": "manual",
        "Anomaly (mean ± 2σ)": "anomaly",
      };
      const choice = await ctx.choose("Temperature range", Object.keys(modes));
      if (!choice) return { skipped: "No range mode chosen." };
      const mode = modes[choice];
      let low = null;
      let high = null;
      if (mode === "manual") {
        low = Number.parseFloat(await ctx.prompt("Lowest temperature °C", "") || "");
        high = Number.parseFloat(await ctx.prompt("Highest temperature °C", "") || "");
        if (!Number.isFinite(low) || !Number.isFinite(high)) {
          return { skipped: "A manual range needs a number for both ends; nothing was changed." };
        }
      }
      let result;
      try {
        result = await call("scale_thermal", picked.path, mode, low, high);
      } catch (error) {
        // A GeoTIFF that does not say what its values are: ask, never assume.
        if (!(error.result && error.result.needs_unit_confirmation)) throw error;
        const units = { "Degrees Celsius": "celsius", Kelvin: "kelvin" };
        const unit = await ctx.choose(`${error.message} What unit are its values in?`,
          [...Object.keys(units), "Not temperatures / not sure"]);
        if (!unit || !units[unit]) return { skipped: "The raster's unit was not confirmed; nothing was read." };
        result = await call("scale_thermal", picked.path, mode, low, high,
          1.0, 99.0, 2.0, "ironbow", units[unit]);
      }
      const s = result.scale;
      const text = `${s.min_c.toFixed(1)} to ${s.max_c.toFixed(1)} °C (${s.method}, ${s.palette}); `
        + `the frame spans ${s.data_min_c.toFixed(1)} to ${s.data_max_c.toFixed(1)} °C`;
      return result.warning
        ? { message: `Range ${text}. ${result.warning}`, level: "warn" }
        : { message: `Range ${text}; no pixel falls outside it.` };
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
  /* Was labelled "New Folder", which is not what it has ever done. */
  "Add Layer": {
    describe: "Add a raster or vector layer from a file.",
    async run() {
      const file = await call("pick_file", ["tif", "tiff", "geojson", "json", "shp"]);
      if (!file || !file.path) return { skipped: "No file chosen." };
      await call("add_layer_from_file", file.path);
      return { message: "Layer added.", refresh: true };
    },
  },

  /* The DJI Thermal SDK decodes DJI R-JPEGs. DJI's licence keeps it out of this build,
     so the user downloads it and points the app at the folder; this shows the state and
     saves the folder (the ODK_DJI_TSDK_DIR environment variable overrides it). */
  "Thermal SDK": {
    describe: "Show which thermal formats decode, and set the DJI Thermal SDK folder.",
    async run(ctx) {
      const status = (await call("thermal_decoder_status")).dji_thermal_sdk;
      const state = status.available
        ? `DJI Thermal SDK loaded from ${status.library}.`
        : `DJI Thermal SDK not available. ${status.reason}`;
      const choice = await ctx.choose(state, ["Choose the DJI Thermal SDK folder",
        "Where do I get it?", "Close"]);
      if (!choice || choice === "Close") return { message: state };
      if (choice.startsWith("Where")) {
        return { message: `Download the DJI Thermal SDK from ${status.download_url} (DJI's licence applies), extract it, then choose its folder with Thermal SDK.` };
      }
      const folder = await call("pick_folder");
      if (!folder || !folder.path) return { skipped: "No folder chosen." };
      const saved = (await call("set_dji_thermal_sdk_dir", folder.path)).dji_thermal_sdk;
      return saved.available
        ? { message: `DJI Thermal SDK loaded from ${saved.library}. DJI R-JPEGs now decode to °C.` }
        : { message: saved.reason, level: "warn" };
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

  /* Counted saved mission versions and diffed two of them -- a comparison of plans,
     under a button on the screen whose job is to compare the FLIGHT with its plan.
     match_captures pairs each flown image with the capture point it satisfied. */
  "Match Captures": {
    describe: "Compare what was captured against what was planned.",
    async run(ctx) {
      const radius = await ctx.prompt("Match radius (metres)", "15");
      if (radius === null || radius === "") return { skipped: "No match radius given." };
      const result = await call("match_captures", Number(radius));
      const missed = (result.missed || []).length;
      return {
        message: `${result.matched} of ${result.planned_total} planned captures matched `
          + `(${result.coverage_pct}%), ${missed} missed, ${(result.unplanned || []).length} `
          + `unplanned image(s).${missed ? " Request Reflight turns the missed points into a layer." : ""}`,
        refresh: true,
      };
    },
  },

  /* Coverage QA ran only inside the full pipeline, after reconstruction: far too late
     to send the aircraft back. It needs the height the photographs were taken above the
     ground; the images may record it (DJI), or the terrain source may give it, and
     otherwise the operator states it -- it is never assumed. */
  "Coverage QA": {
    describe: "Place the flown images on the ground and find coverage and overlap gaps.",
    async run(ctx) {
      const options = {};
      const agl = await ctx.prompt(
        "Flight height above ground (m). Leave blank if the images record it (DJI) or a "
        + "terrain model is set, or to give the ground elevation instead.", "");
      if (agl === null) return { skipped: "Coverage QA cancelled." };
      if (String(agl).trim() !== "") {
        options.flight_height_agl_m = Number(agl);
      } else {
        const ground = await ctx.prompt(
          "Ground elevation above sea level at the site (m). Leave blank to use only "
          + "the images' own height and the terrain source.", "");
        if (ground === null) return { skipped: "Coverage QA cancelled." };
        if (String(ground).trim() !== "") options.ground_elevation_m = Number(ground);
      }
      const started = await call("check_coverage", options);
      return { message: `Coverage QA started on ${started.image_count} image(s)`
                 + (started.terrain_path ? `, heights from ${started.terrain_path}.` : "."),
               job: started.job_id, refresh: true };
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
      // Passed as typed: the Api refuses a negative or non-numeric value by name. Coercing
      // here let -1 through as a cycle limit and turned "abc" silently into 0.
      const capacity = await ctx.prompt("Capacity mAh (optional)", "0");
      if (capacity === null) return { skipped: "Nothing registered." };
      const limit = await ctx.prompt("Cycle limit (optional)", "0");
      if (limit === null) return { skipped: "Nothing registered." };
      const result = await call("add_battery", ctx.organizationId(), serial, capacity, limit);
      return {
        message: `Battery ${result.serial_number} registered. Health is blank until you `
          + "measure it (Log Battery Cycle).",
        refresh: true,
      };
    },
  },
  /* Without this a registered pack never moved: Cycles stayed 0 and Health had no way
     to be entered, which is how every pack came to show the default 100%. */
  "Log Battery Cycle": {
    describe: "Add charge cycles to the selected battery and, if measured, its health.",
    async run(ctx) {
      const picked = ctx.selectedFleetRecord();
      if (!picked || picked.kind !== "battery") {
        return { skipped: "Select a battery in the Batteries list first." };
      }
      const cycles = await ctx.prompt(`Cycles to add to ${picked.name}`, "1");
      if (cycles === null) return { skipped: "Nothing recorded." };
      const health = await ctx.prompt(
        "Measured health % (leave blank if you did not measure it)", "");
      if (health === null) return { skipped: "Nothing recorded." };
      const result = await call("log_battery_cycle", ctx.organizationId(), picked.id,
        cycles, health);
      return {
        message: `${result.serial_number}: ${result.cycle_count} cycles`
          + (result.health_pct === null ? ", health not measured." : `, health ${result.health_pct}%.`),
        refresh: true,
      };
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
      const id = await pickAircraft(ctx);
      if (id.skipped) return id;
      const kind = await ctx.prompt("What was done (propeller, motor, inspection…)", "inspection");
      if (!kind) return { skipped: "Maintenance needs a kind." };
      const detail = await ctx.prompt("Detail (optional)", "") || "";
      const result = await call("log_maintenance", id.id, kind, detail, "operator");
      return { message: `Logged ${result.kind} at ${result.hours_at_service} h.`, refresh: true };
    },
  },
  "Assign Mission": {
    describe: "Note which aircraft flies this mission.",
    async run(ctx) {
      const id = await pickAircraft(ctx);
      if (id.skipped) return id;
      const mission = await ctx.prompt("Mission name", "Mission");
      if (!mission) return { skipped: "No mission named." };
      await call("assign_mission_to_aircraft", id.id, mission);
      return { message: `Assigned to ${mission}.`, refresh: true };
    },
  },

  "Remove Record": {
    describe: "Delete the selected aircraft (with its maintenance log), battery or pilot.",
    async run(ctx) {
      const picked = ctx.selectedFleetRecord();
      if (!picked) return { skipped: "Select an aircraft, battery or pilot in the fleet lists first." };
      const what = `${picked.kind} "${picked.name}"`
        + (picked.kind === "aircraft" ? " and its maintenance log" : "");
      if (!(await ctx.confirm(`Remove ${what}? This cannot be undone.`))) {
        return { skipped: "Nothing removed." };
      }
      const result = await call("remove_fleet_record", ctx.organizationId(), picked.kind, picked.id);
      return {
        message: `Removed ${result.kind} ${result.label}`
          + (result.removed_maintenance ? ` and ${result.removed_maintenance} maintenance record(s).` : "."),
        refresh: true,
      };
    },
  },
  /* Fleet's toolbar had the generic Export, which exports the MISSION: on the Fleet
     screen it answered "No mission has been planned yet". */
  "Export Fleet": {
    describe: "Write the fleet register (aircraft, batteries, pilots, maintenance) to CSV and JSON.",
    async run(ctx) {
      const result = await call("export_fleet", ctx.organizationId(), "");
      const c = result.counts || {};
      await tryCall("open_path", result.directory);
      return {
        message: `Exported ${c.aircraft || 0} aircraft, ${c.batteries || 0} batteries, `
          + `${c.pilots || 0} pilots and ${c.maintenance || 0} maintenance records to ${result.directory}`,
        files: Object.values(result.files || {}),
      };
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
      await ctx.reveal(`Share token (copy it now, it is not stored):\n\n${result.token}`
        + (result.reaches ? `\n\n${result.reaches}` : ""));
      return { message: `Link ${result.prefix}… issued.`, refresh: true };
    },
  },
  /* Was labelled "Archive" beside an Archived panel, and archived nothing: it revokes
     the project's share links, and is named for that. */
  "Revoke Links": {
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
  /* Issued a SHARE link and called it an API key. A key belongs to an account on the
     REST service; this asks for that account and issues the same token POST
     /auth/tokens would. */
  "New Key": {
    describe: "Issue a REST API token for a service account.",
    async run(ctx) {
      const answer = await ctx.form("New API key", [
        { key: "email", label: "Service account email", value: "" },
        { key: "password", label: "Password", type: "password", value: "" },
        { key: "name", label: "Key name", value: "field device" },
      ]);
      if (!answer || !answer.email) return { skipped: "No account given." };
      const result = await call("issue_api_token", answer.email, answer.password, answer.name);
      await ctx.reveal(`API key for ${result.user} (copy it now, it is not stored):\n\n${result.secret}`);
      return { message: `Key ${result.prefix}… issued to ${result.user}.`, refresh: true };
    },
  },
  /* Built its command from the first registered WEBHOOK -- a POST to someone else's
     URL -- under a button beside the endpoint list. It is the endpoint picked in the
     Endpoints panel, as a request to the REST service. */
  "Copy cURL": {
    describe: "The selected endpoint as a curl command against the REST service.",
    async run(ctx) {
      const endpoint = ctx.selectedEndpoint() || { method: "GET", path: "/health" };
      const url = `${REST_BASE}${endpoint.path}`;
      const open = endpoint.path.startsWith("/health") || endpoint.path.startsWith("/auth/");
      const auth = open ? "" : ' -H "Authorization: Bearer $ODK_TOKEN"';
      const body = ["POST", "PUT", "PATCH"].includes(endpoint.method)
        ? ' -H "Content-Type: application/json" -d \'{}\'' : "";
      await ctx.reveal(`curl -X ${endpoint.method} "${url}"${auth}${body}`, "curl command");
      return { message: `curl for ${endpoint.method} ${endpoint.path} shown; copy it from the dialog.` };
    },
  },
  /* Called capabilities() and reported how many fields the BRIDGE returned -- a request
     to this process, labelled as one to the service. */
  "Send Request": {
    describe: "Send the selected GET to the REST service and show what it answered.",
    async run(ctx) {
      const endpoint = ctx.selectedEndpoint() || { method: "GET", path: "/health" };
      if (endpoint.method !== "GET" || endpoint.path.includes("{")) {
        return { skipped: `${endpoint.method} ${endpoint.path} needs parameters or a body; `
          + "use Copy cURL. Send Request sends plain GETs only." };
      }
      const result = await call("rest_request", endpoint.path, REST_BASE);
      return { message: `${endpoint.path}: HTTP ${result.status} — ${String(result.body).slice(0, 200)}` };
    },
  },

  /* -- reports ------------------------------------------------------------ */
  "New Report": { alias: "Generate Report" },
  Generate: { alias: "Generate Report" },
  "Generate Report": {
    describe: "Build a report from what the open project contains.",
    async run(ctx) {
      const chosen = ctx.reportOptions ? ctx.reportOptions() : {};
      const formats = chosen.formats || ["html", "pdf"];
      const readiness = await call("report_readiness", "", chosen.reportType || "standard", formats);
      if (!readiness.ready) {
        // The engine refuses rather than emitting empty sections, and the checklist is
        // the useful part -- it says exactly what to produce first.
        return { skipped: `Not ready: ${(readiness.missing || []).join("; ") || "unknown"}` };
      }
      const title = await ctx.prompt("Report title", "Inspection report") || "Inspection report";
      // Word and PDF are written by the same call from the same sections: the engine
      // renders HTML and hands the blocks to the reportlab / python-docx writers.
      const result = await call("generate_report", "", title,
                                chosen.reportType || "standard",
                                chosen.author || "", chosen.organization || "",
                                chosen.client || "", chosen.logo || "", formats);
      const files = ["html", result.pdf_path ? "pdf" : "", result.docx_path ? "docx" : ""]
        .filter(Boolean).join(", ");
      // What the report says it left out travels with the confirmation, not after it.
      const caveats = (result.warnings || []).length ? ` ${result.warnings.join(" ")}` : "";
      return { message: `Report written (${files}) to ${result.output_dir}.${caveats}`, refresh: true };
    },
  },
  "Export Report": { alias: "Generate Report" },
  "Export PDF": {
    describe: "Write the report as a PDF.",
    async run(ctx) {
      return ACTIONS["Generate Report"].run({ ...ctx, reportOptions: () => ({
        ...(ctx.reportOptions ? ctx.reportOptions() : {}), formats: ["html", "pdf"] }) });
    },
  },
  "Export Word": {
    describe: "Write the report as an editable Word document.",
    async run(ctx) {
      return ACTIONS["Generate Report"].run({ ...ctx, reportOptions: () => ({
        ...(ctx.reportOptions ? ctx.reportOptions() : {}), formats: ["html", "docx"] }) });
    },
  },
  "Open Report": {
    describe: "Open the selected report, or the newest one.",
    async run(ctx) {
      const listed = await call("list_reports");
      const reports = listed.reports || [];
      if (!reports.length) return { skipped: "No reports in this project yet." };
      const id = (ctx.selectedReport && ctx.selectedReport()) || reports[0].id;
      const report = reports.find((r) => r.id === id) || reports[0];
      const kind = report.pdf_path ? "pdf" : "html";
      const opened = await call("open_report", report.id, kind);
      return { message: `Opened ${opened.opened}.` };
    },
  },
  /* Run Pipeline (Start / Run AI) writes a markdown inspection report into each run
     folder. No screen listed or opened it. This shows the chosen run's report in the
     window -- a .md often has no program associated to open it with. */
  "Inspection Report": {
    describe: "Read the inspection report a Run Pipeline run wrote.",
    async run(ctx) {
      const listed = await call("pipeline_reports");
      const runs = (listed && listed.reports) || [];
      if (!runs.length) {
        return { skipped: "No pipeline run in this project yet. Start / Run AI writes an inspection report for each run." };
      }
      let id = runs[0].id;
      if (runs.length > 1) {
        id = await ctx.choose("Which pipeline run?", runs.map((r) => ({
          label: `${r.id} (${r.image_count} image(s))`,
          value: r.id,
          hint: r.reconstruction_status === "failed" ? "reconstruction failed" : String(r.created_at || "").slice(0, 19),
        })));
        if (!id) return { skipped: "No run chosen." };
      }
      const report = await call("read_pipeline_report", id);
      await ctx.showText(`Inspection report: ${report.id}`, report.markdown);
      return { message: `Inspection report: ${report.markdown_path}` };
    },
  },
  /* The Annotate tool armed a cursor nothing read, so no finding could ever be made
     from this screen and no report could contain one an inspector had drawn. */
  /* On the Digital Twin with the model on screen, a finding is pinned to a point on the
     model (viewer3d.js, annotate_model); anywhere else it is recorded on one of the
     project's images, as Record Finding does. */
  Annotate: {
    describe: "Pin a finding to the 3D model on screen, or record one on an image.",
    async run(ctx) {
      if (currentViewer()) {
        const said = armModelTool("annotate");
        return { message: `Annotate: ${said}` };
      }
      return ACTIONS["Record Finding"].run(ctx);
    },
  },
  "Record Finding": {
    describe: "Record a finding on one of the project's images.",
    async run(ctx) {
      const image = await ctx.prompt("Image file name (from the project's datasets)", "");
      if (!image) return { skipped: "No image chosen." };
      const label = await ctx.prompt("What is it (crack, spalling, corrosion...)", "");
      if (!label) return { skipped: "A finding needs a label." };
      const severity = (await ctx.prompt("Severity: critical, high, medium, low or info",
                                          "medium") || "medium").trim().toLowerCase();
      const note = await ctx.prompt("Note (optional)", "") || "";
      const boxText = (await ctx.prompt("Box x0,y0,x1,y1 in pixels (blank = whole photo)", "")
                       || "").trim();
      let box = null;
      if (boxText) {
        box = boxText.split(/[ ,]+/).map(Number);
        if (box.length !== 4 || box.some((v) => !Number.isFinite(v))) {
          return { skipped: "The box needs four numbers: x0, y0, x1, y1." };
        }
      }
      const result = await call("record_finding", image, label, severity, note, box);
      return { message: `Recorded ${result.finding.label} (${result.finding.severity}).`,
               refresh: true };
    },
  },

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
  /* Logged a maintenance record against aircraft 1 -- not a reflight, and an error on
     any install without that aircraft. A reflight is the planned points that produced
     no image; they become a layer on the map and an entry in the audit log. */
  "Request Reflight": {
    describe: "Put the missed capture points on the map as a reflight layer.",
    async run(ctx) {
      const reason = await ctx.prompt("Why does it need reflying?", "captures missed");
      if (reason === null) return { skipped: "Cancelled." };
      const result = await call("request_reflight", reason || "");
      return { message: `${result.points} point(s) to refly, written to ${result.path} `
        + "and shown on the map as Reflight points.", refresh: true };
    },
  },
  /* On the Reports toolbar this saved a MISSION. It saves the report settings. */
  "Save Template": {
    describe: "Save the report type, formats and branding for next time.",
    async run(ctx) {
      const name = await ctx.prompt("Template name", "Template");
      if (!name) return { skipped: "No name given." };
      const chosen = ctx.reportOptions ? ctx.reportOptions() : {};
      await call("save_report_template", name, {
        report_type: chosen.reportType || "standard",
        formats: chosen.formats || ["html", "pdf"],
        organization: chosen.organization || "",
        client: chosen.client || "",
        author: chosen.author || "",
        logo_path: chosen.logo || "",
      });
      return { message: `Saved report template "${name}".`, refresh: true };
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

function basename(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function joinPath(folder, name) {
  const separator = String(folder).includes("\\") ? "\\" : "/";
  return String(folder).replace(/[\\/]+$/, "") + separator + name;
}

/** An image from the active dataset, or any file when no dataset is open. */
async function pickImage(ctx) {
  const listing = await tryCall("list_dataset_images");
  if (listing && (listing.images || []).length) {
    const name = await ctx.choose("Which image?", listing.images.map((n) => ({ label: n, value: n })));
    return name ? joinPath(listing.folder, name) : null;
  }
  const picked = await call("pick_file", ["jpg", "jpeg", "png", "tif", "tiff"]);
  return picked && picked.path ? picked.path : null;
}

function summariseModelRun(result) {
  const out = result.output || {};
  if (out.detection_count != null) return `${out.detection_count} detection(s)`;
  if (out.flagged_tile_count != null) return `${out.flagged_tile_count} of ${out.tiles} tile(s) flagged`;
  if (out.top_class) return `${out.top_class} (${Number(out.top_probability).toFixed(2)})`;
  if (out.worst_present_grade !== undefined) return `worst grade ${out.worst_present_grade || "none"}`;
  if (out.pixel_ratio != null) return `${(100 * out.pixel_ratio).toFixed(1)}% ${out.label}`;
  return "done";
}

/** What a model said, and what it is -- the digest and scope go beside every answer. */
function describeModelRun(result) {
  const out = result.output || {};
  const lines = [
    ["Model", result.model_key],
    ["Ran as", result.model_used],
    ["File digest", String(result.model_sha256 || "").slice(0, 16)],
    ["Result", summariseModelRun(result)],
  ];
  for (const d of (out.detections || []).slice(0, 12)) {
    lines.push([d.label, `confidence ${Number(d.confidence).toFixed(3)} at [${d.bbox_xyxy.join(", ")}]`]);
  }
  for (const [name, value] of Object.entries(out.class_pixel_ratio || {})) {
    lines.push([name, `${(100 * value).toFixed(2)}% of pixels`]);
  }
  if (out.probabilities) {
    const ranked = Object.entries(out.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5);
    for (const [name, value] of ranked) lines.push([name, Number(value).toFixed(3)]);
  }
  if (out.note) lines.push(["Note", out.note]);
  lines.push(["Scope", result.scope]);
  return lines;
}

/** One line an operator can act on: can they start, and if not, why not. */
function describePreflight(result) {
  if (!result || !Array.isArray(result.checks)) return "Preflight ran.";
  const blocking = result.checks.filter((c) => c.severity === "blocking" && !c.confirmed);
  const pending = result.checks.filter((c) => c.requires_manual && !c.confirmed);
  const lead = result.simulated ? "SIMULATED vehicle. " : "";
  if (blocking.length) {
    return `${lead}Not ready: ${blocking.map((c) => `${c.name} — ${c.message}`).slice(0, 2).join("; ")}`
      + (blocking.length > 2 ? ` (+${blocking.length - 2} more)` : "");
  }
  if (pending.length) {
    return `${lead}${pending.length} check(s) need your confirmation: `
      + pending.map((c) => c.name).join(", ");
  }
  return `${lead}All preflight checks pass.`;
}

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
    if (m.native_unit && result.value_m !== undefined) {
      // A raster in feet: the figure above is in its own unit, this is the metric one.
      parts.push(`= ${round(result.value_m)} ${m.kind === "area" ? "m²" : "m"}`);
    }
    if (result.surface_length_m !== undefined) {
      parts.push(`over the ground ${round(result.surface_length_m)} m`);
    } else if (result.surface_note) {
      parts.push(result.surface_note);
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
  // Demo mode answers reads from the bundled example and changes nothing, so an action
  // is refused before it can look like it worked on the example.
  if (demoEnabled()) {
    return "Demo mode shows the bundled example and cannot change anything. "
      + "Open the cockpit without ?demo=1 to act on a real project.";
  }
  if (!connected()) return "Not connected to the application.";
  if (NEEDS_PROJECT.has(action) && !state.projectOpen) return "Open a project first.";
  if (NEEDS_DATASET.has(action) && !state.datasetSelected) return "Select a dataset first.";
  return null;
}
