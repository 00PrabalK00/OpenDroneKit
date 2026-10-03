/* Fly: where every verb on this screen lives.

   Machine-readable placements, read by the reachability and parity tests instead of
   the old `toolbar: [` arrays. One entry per control:

     { action: "<ACTIONS key>", where: "fly.<region>", label: "<visible label>" }

   `action` must resolve in the merged registry (core ACTIONS plus ./actions.js), and
   the rendered button must carry data-action="<action>". Schema and rules:
   docs/ui/ARCHITECTURE.md, "surface.js". Every inventory row for this workspace
   (docs/ui/inventory.json) is named here, by `inventory` where the action alone does
   not say which row it satisfies. Rows with no verb (telemetry, progress, mode, the
   vehicle on the map) are read-only panels; tests/test_ui_v1_fly.py checks those. */

export const SCREEN = "fly";

export const PLACEMENTS = [
  // CONNECTION
  { action: "Connect Vehicle", where: "fly.connection", label: "Connect / Reconnect", inventory: "fly.connect-driver-uri" },
  { action: "Disconnect", where: "fly.connection", label: "Disconnect", inventory: "fly.disconnect-reconnect" },

  // PREFLIGHT (per-row acknowledge, re-run)
  { action: "Preflight", where: "fly.preflight", label: "Re-run", inventory: "fly.preflight-acknowledge-checks" },
  { action: "Confirm Check", where: "fly.preflight.row", label: "Confirm", inventory: "fly.preflight-acknowledge-checks" },

  // MISSION ("Mission loaded" row, version picker, upload)
  { action: "Upload", where: "fly.mission", label: "Upload", inventory: "fly.upload" },
  { action: "Load Mission Version", where: "fly.mission", label: "Load", inventory: "fly.mission-queue" },

  // SESSION
  { action: "Export Log", where: "fly.session", label: "Export log", inventory: "fly.export-flight-log" },
  { action: "Load Flight", where: "fly.session", label: "Load flight", inventory: "fly.load-flight-resume-interrupted" },
  { action: "Resume From Images", where: "fly.session.interrupted", label: "From images", inventory: "fly.load-flight-resume-interrupted" },
  { action: "Discard Interrupted Flight", where: "fly.session.interrupted", label: "Discard", inventory: "fly.load-flight-resume-interrupted" },
  { action: "Linked Progress", where: "fly.session", label: "Linked progress", inventory: "fly.linked-mission-progress" },
  { action: "Mark Corner", where: "fly.session", label: "Mark corner", inventory: "fly.fly-to-draw-corners" },
  { action: "Close Boundary", where: "fly.session", label: "Close boundary", inventory: "fly.fly-to-draw-corners" },
  { action: "Clear Corners", where: "fly.session", label: "Clear corners", inventory: "fly.fly-to-draw-corners" },

  // Command bar
  { action: "Arm", where: "fly.commandbar", label: "Arm / Armed", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "Start Mission", where: "fly.commandbar", label: "Start mission", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "Pause", where: "fly.commandbar", label: "Pause / Resume", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "Resume", where: "fly.commandbar", label: "Pause / Resume", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "RTL", where: "fly.commandbar", label: "Return to home", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "Abort", where: "fly.commandbar.abort", label: "Hold to abort", inventory: "fly.arm-start-pause-resume-rth-abort" },
  { action: "Land", where: "fly.commandbar.more", label: "Land now", inventory: "fly.land-manual-override-capture-now" },
  { action: "Manual Override", where: "fly.commandbar.more", label: "Take manual control", inventory: "fly.land-manual-override-capture-now" },
  { action: "Capture Now", where: "fly.commandbar.more", label: "Capture now", inventory: "fly.land-manual-override-capture-now" },
  { action: "Disarm", where: "fly.commandbar.more", label: "Disarm", inventory: "fly.arm-start-pause-resume-rth-abort" },

  // Map banner -> Details
  { action: "Clear Alerts", where: "fly.banner.details", label: "Clear alerts", inventory: "fly.alerts-and-clearing-them" },
];

/** Inventory rows the screen satisfies with a read-only panel rather than a verb. */
export const PANELS = [
  { inventory: "fly.mode-control-state", where: "fly.commandbar.mode", label: "MODE" },
  { inventory: "fly.telemetry", where: "fly.inspector.telemetry", label: "TELEMETRY" },
  { inventory: "fly.mission-progress", where: "fly.progress", label: "MISSION PROGRESS" },
  { inventory: "fly.event-log", where: "fly.inspector.events", label: "Events" },
  { inventory: "fly.session-status", where: "fly.session", label: "SESSION" },
  { inventory: "fly.vehicle-messages", where: "fly.inspector.messages", label: "Messages" },
  { inventory: "fly.mission-on-aircraft", where: "fly.map", label: "Flown / to fly" },
  { inventory: "fly.telemetry-charts-camera-live-view", where: "none", label: "Not carried: they drew no data" },
];
