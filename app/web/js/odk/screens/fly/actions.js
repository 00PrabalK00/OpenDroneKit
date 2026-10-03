/* Fly: verbs new to this screen. The shared core catalogue
   (js/workspace/actions.js) is frozen; a screen unit adds verbs only here, with the
   same shape as a core entry. js/odk/registry.js merges them and refuses a duplicate key.

   The core verbs the Fly screen also places (Connect, Disconnect, Preflight, Confirm
   Checks, Upload, Start Mission, Pause, Resume, RTL, Land, Abort, Manual Override,
   Capture Now, Export Log, Load Flight, Linked Progress, Mark Corner, Close Boundary,
   Clear Corners, Clear Alerts) keep their own behaviour. These add what the v1 screen
   needed and the cockpit never had: a connect that takes the driver and URI from the
   CONNECTION form instead of two dialogs, a per-row preflight confirmation, arm and
   disarm, the mission queue, and the two answers to an interrupted flight. Every one
   reaches the real Api and says what the vehicle answered. */

import { call } from "../../../workspace/api.js";

const DRIVER_NAMES = new Set(["pymavlink", "mavsdk", "mock", "mavlink", "px4", "sitl"]);

export const ACTIONS = {
  "Connect Vehicle": {
    describe: "Open the vehicle link with the driver and connection chosen in CONNECTION.",
    async run(ctx) {
      const { driver, uri } = ctx.payload || {};
      if (!driver || !uri) {
        return { skipped: "Choose a driver and a connection in CONNECTION, or use Connect for the dialog.", level: "warn" };
      }
      if (!DRIVER_NAMES.has(String(driver))) return { skipped: `Unknown driver ${driver}.`, level: "warn" };
      /* A link that does not open is refused (ok:false); call() throws its reason, so
         the toast says why and never "Connected". */
      const result = await call("connect_vehicle", String(uri), String(driver));
      const vehicle = result.vehicle || {};
      return {
        message: `Connected to ${uri} via ${driver}${vehicle.is_simulated ? " (SIMULATED vehicle; nothing it reports is measured)." : "."}`,
        refresh: true,
      };
    },
  },
  "Confirm Check": {
    describe: "Record that you checked one preflight item nothing on the aircraft can report.",
    async run(ctx) {
      const { id, name } = ctx.payload || {};
      if (!id) return { skipped: "No check named.", level: "warn" };
      const after = await call("run_preflight", [String(id)]);
      const row = (after.checks || []).find((c) => c.id === id);
      if (row && !row.confirmed) {
        /* Only manual checks can be confirmed by a person; the Api refuses the rest. */
        return { skipped: `${name || id} cannot be confirmed by hand: ${row.message}`, level: "warn" };
      }
      return {
        message: `Confirmed: ${name || id}. ${after.can_start ? "Preflight is ready." : `${after.blocking_count} blocking, ${(after.pending_confirmation || []).length} still to confirm.`}`,
        refresh: true,
      };
    },
  },
  Arm: { vehicle: "arm", confirm: "Arm the motors? The propellers can start turning." },
  Disarm: { vehicle: "disarm", confirm: "Disarm the motors?" },
  "Load Mission Version": {
    describe: "Make a saved mission version the one to upload and fly.",
    async run(ctx) {
      let version = ctx.payload && ctx.payload.version;
      if (version === undefined || version === null || version === "") {
        const listed = await call("list_mission_versions");
        const versions = listed.versions || [];
        if (!versions.length) return { skipped: "No saved mission versions. Save one in Plan.", level: "warn" };
        version = await ctx.choose("Fly which saved mission?", versions.map((v) => ({
          label: `v${v.version_num} · ${v.mission_name || v.template || "mission"}`,
          value: v.version_num, hint: v.note || v.created_at || "",
        })));
        if (version === null || version === undefined) return { skipped: "Cancelled." };
      }
      const result = await call("load_mission_version", Number(version));
      return {
        message: `Loaded version ${result.version} (${result.template}, ${result.waypoints} waypoints). Upload it to the aircraft before starting.`,
        refresh: true,
      };
    },
  },
  "Resume From Images": {
    describe: "Work out what is left to fly from the images already taken.",
    async run() {
      const picked = await call("pick_folder");
      const folder = picked && picked.path;
      if (!folder) return { skipped: "No folder chosen." };
      const result = await call("resume_from_images", folder);
      const progress = result.progress || {};
      const left = result.resume || {};
      const done = Array.isArray(progress.completed) ? progress.completed.length : progress.completed;
      return {
        message: `From the images: ${done ?? "an unknown number of"} capture points flown, `
          + `${left.capture_count ?? "an unknown number"} left to fly.`,
        refresh: true,
      };
    },
  },
  "Discard Interrupted Flight": {
    describe: "Forget the record of a flight this software stopped during, once it is resolved.",
    confirm: "Forget the interrupted flight? Do this only once the aircraft is accounted for.",
    async run() {
      await call("discard_interrupted_flight");
      return { message: "The interrupted flight record was discarded.", refresh: true };
    },
  },
};
