/* Plan: verbs new to this screen. The shared core catalogue
   (js/workspace/actions.js) is frozen; a screen unit adds verbs only here, with the
   same shape as a core entry. js/odk/registry.js merges them and refuses a duplicate key.

   Every verb ends in the Api. The tools that draw (area, no-fly zone, home) hand their
   shape to draw.js, which sends it to set_aoi / set_no_fly_zones / add_site_marker; the
   ones that shape the next plan (route, interest region, line heading) keep the drawn
   geometry in the Plan form and re-plan through the core "Plan" verb, so the plan on
   screen is always the planner's. */

import { call, lastError, tryCall } from "../../../workspace/api.js";
import { armDrawing, vertices } from "../../../workspace/draw.js";
import { fitTo } from "../../../workspace/mapview.js";
import {
  EXPORT_FORMATS, exportChoice, fmtBytes, fmtInt, fmtNum, form, settleForm, versionSummary,
} from "./model.js";
import { cancelApplying, takeRedo, takeUndo } from "./history.js";

/** Tell the Plan screen (its map and panels) about a change only it can show. */
function tell(verb, detail = {}) {
  window.dispatchEvent(new CustomEvent("odk:plan", { detail: { verb, ...detail } }));
}

/** Run another verb through the shell, as the button for it would. */
function runVerb(key) {
  window.dispatchEvent(new CustomEvent("odk:action", { detail: key }));
}

async function onPlanScreen(ctx) {
  const here = ctx.workspaceId ? String(ctx.workspaceId() || "") : "";
  if (here !== "plan" && here !== "planning") await ctx.open("plan");
}

async function hasPlan() {
  const state = await tryCall("get_state");
  return Boolean(state && state.has_mission);
}

/** Compass bearing of a drawn line, first point to last (the line's own heading). */
function bearingOf(points) {
  const [a, b] = [points[0], points[points.length - 1]];
  const rad = Math.PI / 180;
  const y = Math.sin((b[0] - a[0]) * rad) * Math.cos(b[1] * rad);
  const x = Math.cos(a[1] * rad) * Math.sin(b[1] * rad)
    - Math.sin(a[1] * rad) * Math.cos(b[1] * rad) * Math.cos((b[0] - a[0]) * rad);
  return ((Math.atan2(y, x) / rad) + 360) % 360;
}

/** Choose a saved mission and one of its versions; {name, version} or a skip reason. */
async function chooseVersion(ctx, title) {
  const listed = await call("list_mission_versions");
  const versions = ((listed && listed.versions) || []).slice()
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || ""))
      || (Number(b.version_num) || 0) - (Number(a.version_num) || 0));
  if (!versions.length) return { skipped: "No version has been saved yet. Save version first." };
  const picked = await ctx.choose(title, versions.map((v) => {
    const s = versionSummary(v);
    return {
      label: `${v.mission_name || "mission"} v${v.version_num}`,
      value: JSON.stringify([v.mission_name || "", v.version_num]),
      hint: [v.template, s.waypoints ? `${s.waypoints} wp` : "",
        v.created_at && String(v.created_at).slice(0, 16).replace("T", " ")].filter(Boolean).join(" · "),
    };
  }));
  if (picked == null) return { skipped: "No version chosen." };
  const [name, version] = JSON.parse(picked);
  return { name, version };
}

export const ACTIONS = {
  /* -- the map tools ------------------------------------------------------- */
  Select: { tool: "pan", describe: "Pan and pick on the map; stops any drawing tool." },
  "Draw Area": { tool: "polygon", describe: "Draw the area of interest on the map (set_aoi)." },
  "Draw No-fly": { tool: "exclusion", describe: "Draw a no-fly zone the planner must keep out of." },
  "Place Home": { tool: "home", describe: "Click the take-off and landing point (a site marker)." },

  "Edit Vertices": {
    describe: "Drag the corners of the area of interest; Delete removes the picked corner.",
    async run(ctx) {
      await onPlanScreen(ctx);
      const state = await call("get_state");
      if (!((state && state.aoi) || []).length) {
        return { skipped: "There is no area of interest to edit. Draw area or Import KML first." };
      }
      tell("edit-vertices");
      return { message: "Edit vertices: drag a corner, drag a midpoint to add one, Delete removes the picked corner, Esc ends." };
    },
  },

  "Rotate Flight Lines": {
    describe: "Draw a line on the map: the survey lines are re-planned along it.",
    async run(ctx) {
      await onPlanScreen(ctx);
      const state = await call("get_state");
      if (((state && state.aoi) || []).length < 3) {
        return { skipped: "Draw the area of interest before setting the line heading." };
      }
      armDrawing("line", (geometry) => {
        const points = vertices(geometry);
        if (points.length < 2) return;
        const bearing = bearingOf(points);
        form.values.line_bearing_deg = String(Math.round(bearing * 10) / 10);
        form.dirty = true;
        tell("heading", { bearing });
        runVerb("Plan");
      }, () => tell("heading-cancelled"));
      return { message: "Rotate flight lines: click two points along the heading you want; the plan is re-made along it." };
    },
  },

  "Draw Route": {
    describe: "Draw the line a linear, lateral, waypoint, solar-row or curved-facade mission flies.",
    async run(ctx) {
      await onPlanScreen(ctx);
      armDrawing("line", (geometry) => {
        form.route = vertices(geometry);
        form.dirty = true;
        tell("route");
        window.dispatchEvent(new CustomEvent("odk:say", { detail: {
          text: `Route drawn: ${form.route.length} points. Generate plans along it.`, kind: "ok" } }));
      }, () => tell("route-cancelled"));
      return { message: "Draw route: click along the line to fly. Double-click or Enter to finish." };
    },
  },

  "Draw Interest Region": {
    describe: "Draw a region a smart adaptive mission photographs in more detail.",
    async run(ctx) {
      await onPlanScreen(ctx);
      armDrawing("polygon", (geometry) => {
        form.regions = [...form.regions, vertices(geometry)];
        form.dirty = true;
        tell("regions");
        window.dispatchEvent(new CustomEvent("odk:say", { detail: {
          text: `Interest region ${form.regions.length} drawn. Generate adds its detail passes.`, kind: "ok" } }));
      }, () => tell("regions-cancelled"));
      return { message: "Interest region: click its corners. Double-click or Enter to finish." };
    },
  },

  "Clear Route": {
    describe: "Forget the drawn route and interest regions.",
    async run() {
      const had = (form.route ? 1 : 0) + form.regions.length;
      form.route = null;
      form.regions = [];
      tell("route");
      return had ? { message: "Route and interest regions cleared; the next plan will not use them." }
        : { skipped: "No route or interest region is drawn." };
    },
  },

  /* -- the area of interest ------------------------------------------------- */
  "Area From Photos": {
    describe: "Make the area of interest the hull round the survey's camera positions.",
    async run() {
      const result = await call("aoi_from_images");
      return {
        message: `Area of interest: the hull of ${result.cameras} camera positions (${result.origin}), `
          + `${result.vertices} vertices, ${fmtInt(result.area_m2)} m².`,
        refresh: true,
      };
    },
  },
  "Clear Area": {
    describe: "Remove the area of interest.",
    confirm: "Clear the area of interest? Undo puts it back.",
    async run() {
      await call("set_aoi", []);
      return { message: "Area of interest cleared.", refresh: true };
    },
  },
  "Clear No-fly": {
    describe: "Remove every no-fly zone.",
    confirm: "Clear every no-fly zone?",
    async run() {
      await call("set_no_fly_zones", []);
      return { message: "No-fly zones cleared.", refresh: true };
    },
  },
  "Undo Area Edit": {
    describe: "Put the previous area of interest back.",
    async run() {
      const previous = takeUndo();
      if (!previous) return { skipped: "Nothing to undo." };
      const result = await tryCall("set_aoi", previous.ring, previous.source || "");
      if (!result) { cancelApplying(); throw new Error(lastError.get("set_aoi") || "set_aoi refused the area."); }
      return { message: previous.ring.length ? `Area restored: ${previous.ring.length} vertices.` : "Area removed again.", refresh: true };
    },
  },
  "Redo Area Edit": {
    describe: "Re-apply the area-of-interest change that was undone.",
    async run() {
      const next = takeRedo();
      if (!next) return { skipped: "Nothing to redo." };
      const result = await tryCall("set_aoi", next.ring, next.source || "");
      if (!result) { cancelApplying(); throw new Error(lastError.get("set_aoi") || "set_aoi refused the area."); }
      return { message: next.ring.length ? `Area re-applied: ${next.ring.length} vertices.` : "Area cleared again.", refresh: true };
    },
  },
  "Facade Footprint": {
    describe: "Read the area as a building outline: reflex corners, facade passes, overhang occlusion.",
    async run(ctx) {
      const answer = await ctx.ask("Building footprint", [
        { key: "standoff", label: "Stand-off from the walls (m)", value: form.values.standoff_m || "" },
        { key: "overhang", label: "Deepest overhang or balcony (m, blank = none)", value: "" },
      ]);
      if (!answer) return { skipped: "Cancelled." };
      const standoff = Number(answer.standoff);
      if (!(standoff > 0)) return { skipped: "A stand-off in metres is needed to place the passes." };
      const result = await call("facade_footprint", standoff, Number(answer.overhang) || 0);
      const f = result.footprint || {};
      tell("facade", { segments: result.segments || [] });
      ctx.showResult("Building footprint", { lines: [
        ["Area", `${fmtInt(f.area_m2)} m²`],
        ["Corners", `${f.corner_count} (${f.reflex_count} reflex)`],
        ["Facade passes", `${result.segment_count} at ${fmtNum(result.standoff_m, 1)} m`],
        ["Overhang", result.occlusion && result.occlusion.note],
        ["Note", f.note || ""],
      ] });
      return { message: `Footprint: ${f.corner_count} corners, ${f.reflex_count} reflex, `
        + `${result.segment_count} facade passes drawn on the map.` };
    },
  },

  /* -- terrain -------------------------------------------------------------- */
  "Load Terrain": {
    describe: "Choose the terrain model (GeoTIFF, ASCII grid or CSV) the plan follows.",
    async run() {
      const picked = await tryCall("pick_file", ["tif", "tiff", "asc", "csv", "hgt"]);
      if (!picked || !picked.path) return { skipped: "No terrain file chosen; the terrain source is unchanged." };
      const result = await call("set_terrain_source", picked.path);
      const coverage = result.coverage;
      const said = coverage
        ? (coverage.covered ? " It covers the area of interest." : ` ${coverage.detail || "It does not cover the area of interest."}`)
        : "";
      return { message: `Terrain model: ${picked.path}.${said} Generate to plan with it.`, refresh: true,
        level: coverage && !coverage.covered ? "warn" : "ok" };
    },
  },
  "Clear Terrain": {
    describe: "Plan without a terrain model (heights above the take-off point).",
    async run() {
      await call("set_terrain_source", "");
      return { message: "No terrain model: heights are above the take-off point over flat ground.", refresh: true };
    },
  },
  "Terrain Details": {
    describe: "The cached terrain and whether it covers the area, for flying offline.",
    async run(ctx) {
      const cache = await call("describe_terrain_cache");
      const coverage = await tryCall("terrain_coverage");
      const lines = [
        ["Cached tiles", String(cache.tile_count ?? 0)],
        ["Size", `${fmtNum(cache.total_mb, 1)} MB`],
        ["Index", cache.index_state || "not reported"],
        ["Missing", (cache.missing || []).join(", ") || "none"],
        ["Note", cache.note || ""],
        ["Covers the area", coverage ? (coverage.covered ? "yes" : "no") : (lastError.get("terrain_coverage") || "not checked")],
      ];
      if (coverage && coverage.detail) lines.push(["Detail", coverage.detail]);
      ctx.showResult("Terrain model", { lines });
      return { message: coverage ? (coverage.covered ? "Cached terrain covers the area of interest."
        : `Terrain: ${coverage.detail}`) : `Terrain cache: ${cache.tile_count ?? 0} tile(s).`,
      level: coverage && !coverage.covered ? "warn" : "ok" };
    },
  },

  /* -- camera and payload ----------------------------------------------------- */
  "Describe Camera": {
    describe: "What the chosen camera yields at the planned altitude.",
    async run(ctx) {
      const payload = ctx.payload || {};
      const camera = payload.camera || form.camera;
      if (!camera) return { skipped: "Choose a camera first." };
      const altitude = Number(payload.altitude_m);
      const described = await call("describe_camera", camera, Number.isFinite(altitude) && altitude > 0 ? altitude : undefined);
      const c = described.camera || {};
      ctx.showResult(c.name || camera, { lines: [
        ["Sensor", `${c.sensor_w_mm} × ${c.sensor_h_mm} mm`],
        ["Focal length", `${c.focal_mm} mm`],
        ["Image", `${c.image_w_px} × ${c.image_h_px} px`],
        ["At", `${fmtNum(c.altitude_m, 1)} m`],
        ["GSD", `${fmtNum(c.gsd_cm, 2)} cm/px`],
        ["Footprint", Array.isArray(c.footprint_m) ? `${c.footprint_m[0]} × ${c.footprint_m[1]} m` : null],
        ["Profile", `${c.source || ""}${c.known === false ? " (placeholder: not this camera)" : ""}`],
        ["Note", c.warning || c.notes || ""],
      ] });
      return { message: `${c.name || camera}: GSD ${fmtNum(c.gsd_cm, 2)} cm/px at ${fmtNum(c.altitude_m, 1)} m.`,
        level: c.known === false ? "warn" : "ok" };
    },
  },
  "Add Camera": {
    describe: "Describe a camera you own, so it can be planned with.",
    async run(ctx) {
      const answer = await ctx.ask("Add a camera", [
        { key: "key", label: "Key (short id, e.g. my_m3m)" },
        { key: "name", label: "Name" },
        { key: "sensor_w_mm", label: "Sensor width (mm)" },
        { key: "sensor_h_mm", label: "Sensor height (mm)" },
        { key: "focal_mm", label: "Focal length (mm)" },
        { key: "image_w_px", label: "Image width (px)" },
        { key: "image_h_px", label: "Image height (px)" },
        { key: "thermal", label: "Thermal", options: ["no", "yes"], value: "no" },
      ]);
      if (!answer) return { skipped: "Cancelled." };
      const result = await call("add_camera", { ...answer, thermal: answer.thermal === "yes" });
      form.camera = result.camera.key;
      form.dirty = true;
      return { message: `Camera ${result.camera.name} added (${result.stored_at}) and chosen for this plan.`, refresh: true };
    },
  },
  "Camera Capabilities": {
    describe: "What the connected aircraft's camera says it can do.",
    async run(ctx) {
      const result = await call("camera_capabilities");
      const camera = result.camera || {};
      ctx.showResult("Connected camera", { lines: Object.entries(camera).map(([k, v]) => [k.replace(/_/g, " "),
        typeof v === "object" ? JSON.stringify(v) : String(v)]) });
      return { message: "Connected camera described." };
    },
  },
  "Add Payload": {
    describe: "Describe a payload you own (LiDAR, multispectral, gas sensor...).",
    async run(ctx) {
      const answer = await ctx.ask("Add a payload", [
        { key: "key", label: "Key (short id)" },
        { key: "name", label: "Name" },
        { key: "kind", label: "Kind", options: ["custom", "camera", "lidar", "multispectral", "thermal", "sensor", "dropper"], value: "custom" },
        { key: "mass_g", label: "Mass (g, optional)" },
        { key: "power_w", label: "Power (W, optional)" },
        { key: "continuous", label: "Records continuously", options: ["no", "yes"], value: "no" },
        { key: "requires_calibration", label: "Needs a calibration panel", options: ["no", "yes"], value: "no" },
        { key: "notes", label: "Notes" },
      ]);
      if (!answer) return { skipped: "Cancelled." };
      const result = await call("add_payload", { ...answer,
        continuous: answer.continuous === "yes", requires_calibration: answer.requires_calibration === "yes" });
      form.payload = result.payload.key;
      form.dirty = true;
      return { message: `Payload ${result.payload.name || result.payload.key} added and fitted to this plan.`, refresh: true };
    },
  },

  /* -- versions ------------------------------------------------------------- */
  "Load Version": {
    describe: "Put a saved version back on screen (re-planned from its saved inputs).",
    async run(ctx) {
      const chosen = await chooseVersion(ctx, "Open which saved version?");
      if (chosen.skipped) return chosen;
      const result = await call("load_mission_version", chosen.version, chosen.name);
      settleForm();
      form.template = null;
      form.camera = null;
      return {
        message: `Opened ${result.name} v${result.version_num}: ${result.summary.waypoints} waypoints`
          + (result.matches_saved ? ", as saved." : `. Differs from the saved plan: ${result.differences.join("; ")}.`),
        level: result.matches_saved ? "ok" : "warn",
        refresh: true,
      };
    },
  },
  "Compare Specifications": {
    describe: "Whether two saved surveys were flown to specifications that can be differenced.",
    async run(ctx) {
      const first = await chooseVersion(ctx, "Compare which survey...");
      if (first.skipped) return first;
      const second = await chooseVersion(ctx, `...with which? (${first.name || "mission"} v${first.version})`);
      if (second.skipped) return second;
      if (second.name !== first.name) {
        return { skipped: "Versions are numbered per mission: choose two versions of the same mission." };
      }
      const result = await call("compare_survey_specifications", first.version, second.version, first.name);
      const c = result.comparison || {};
      const lines = Object.entries(c).map(([k, v]) => [k.replace(/_/g, " "),
        Array.isArray(v) ? v.join("; ") : typeof v === "object" && v ? JSON.stringify(v) : String(v)]);
      ctx.showResult(`${first.name || "mission"} v${first.version} and v${second.version}`, { lines });
      const verdict = c.comparable ?? c.differenceable ?? c.compatible;
      return { message: verdict === undefined ? "Specifications compared." :
        (verdict ? "The two surveys can be differenced." : "The two surveys cannot be differenced as flown."),
      level: verdict === false ? "warn" : "ok" };
    },
  },

  /* -- site markers ----------------------------------------------------------- */
  "Remove Marker": {
    describe: "Remove a site marker (hazard, take-off point, gate).",
    confirm: "Remove this site marker?",
    async run(ctx) {
      let id = ctx.payload && ctx.payload.id;
      if (!id) {
        const listed = await call("list_site_markers");
        const markers = (listed.markers || []);
        if (!markers.length) return { skipped: "There are no site markers." };
        id = await ctx.choose("Remove which marker?", markers.map((m) => ({
          label: m.name, value: m.id, hint: m.kind })));
        if (!id) return { skipped: "No marker chosen." };
      }
      const result = await call("remove_site_marker", id);
      return { message: `Marker removed. ${(result.markers || []).length} marker(s) on this site.`, refresh: true };
    },
  },

  /* -- the plan's figures ----------------------------------------------------- */
  "Battery Segments": {
    describe: "Split the plan into sorties a battery can fly.",
    async run() {
      const result = await call("plan_battery_segments");
      return { message: `${result.segments} sortie(s). ${result.note || ""}`.trim(), refresh: true };
    },
  },
  "Mission Estimates": {
    describe: "Batteries, storage and duration the plan needs.",
    async run() {
      const result = await call("mission_estimates");
      const e = result.estimates || {};
      return { message: `${e.image_count} photos, ${fmtNum(e.storage && e.storage.total_gb, 2)} GB, `
        + `${e.battery && e.battery.batteries_required} battery(ies). ${e.note || ""}`.trim() };
    },
  },

  /* -- export and hand-over ----------------------------------------------------- */
  "Export Mission Files": {
    describe: "Write the plan in the formats ticked under EXPORT.",
    async run(ctx) {
      const asked = (ctx.payload && ctx.payload.formats) || [...exportChoice];
      const formats = EXPORT_FORMATS.map((f) => f.key).filter((k) => asked.includes(k));
      if (!formats.length) return { skipped: "Tick at least one format to export." };
      if (!(await hasPlan())) return { skipped: "No mission has been planned yet. Generate one first." };
      const result = await call("export_mission", formats);
      const written = Object.entries(result.written || {});
      const failed = written.filter(([, v]) => String(v).startsWith("failed"));
      const files = written.length - failed.length;
      const bytes = Object.values(result.bytes || {}).reduce((a, b) => a + b, 0);
      return {
        message: `Exported ${files} file(s), ${fmtBytes(bytes)}, to ${result.directory}`
          + (failed.length ? `; ${failed.map(([k, v]) => `${k} ${v}`).join("; ")}` : "."),
        level: failed.length ? "warn" : "ok",
        files: written.filter(([, v]) => !String(v).startsWith("failed")).map(([, v]) => v),
        refresh: true,
      };
    },
  },
  "Open Export Folder": {
    describe: "Open the folder the mission files are written to.",
    async run() {
      const listed = await call("list_mission_exports");
      await call("open_path", listed.directory);
      return { message: `Opened ${listed.directory}.` };
    },
  },
  "Send to Fly": {
    describe: "Hand the plan to Fly, with Upload ready for the connected aircraft.",
    async run(ctx) {
      if (!(await hasPlan())) return { skipped: "No mission has been planned yet. Generate one first." };
      const state = await call("get_state");
      window.odkPlanHandover = { at: Date.now(), uploadArmed: true };
      await ctx.open("fly");
      window.dispatchEvent(new CustomEvent("odk:upload-armed", { detail: { from: "plan" } }));
      const vehicle = state.vehicle || {};
      return { message: vehicle.connected
        ? "Plan sent to Fly: press Upload to send it to the aircraft."
        : "Plan sent to Fly. Connect the aircraft, then Upload." };
    },
  },

  /* -- the view --------------------------------------------------------------- */
  "Zoom to Area": {
    describe: "Fit the map to the area of interest.",
    async run(ctx) {
      const state = await call("get_state");
      const ring = (state && state.aoi) || [];
      const map = ctx.map();
      if (ring.length < 3) return { skipped: "No area of interest is drawn." };
      if (!map) return { skipped: "There is no map on this screen." };
      const xs = ring.map((p) => p[0]); const ys = ring.map((p) => p[1]);
      fitTo(map, [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
      return { message: "Zoomed to the area of interest." };
    },
  },
  "Zoom to Mission": {
    describe: "Fit the map to the planned route.",
    async run(ctx) {
      const result = await tryCall("mission_geojson");
      const map = ctx.map();
      const points = ((result && result.geojson && result.geojson.features) || [])
        .filter((f) => f.geometry && f.geometry.type === "Point").map((f) => f.geometry.coordinates);
      if (!points.length) return { skipped: "No mission has been planned yet." };
      if (!map) return { skipped: "There is no map on this screen." };
      const xs = points.map((p) => p[0]); const ys = points.map((p) => p[1]);
      fitTo(map, [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
      return { message: "Zoomed to the planned route." };
    },
  },
  "Play Route": {
    describe: "Fly the planned route on the map and the altitude profile (a preview, not a simulation of the aircraft).",
    async run(ctx) {
      await onPlanScreen(ctx);
      const result = await call("mission_geojson");
      const count = ((result.geojson && result.geojson.features) || [])
        .filter((f) => (f.properties || {}).role === "waypoint").length;
      if (!count) return { skipped: "The plan has no waypoints to play." };
      tell("play");
      return { message: `Playing the route through ${count} waypoints.` };
    },
  },
};
