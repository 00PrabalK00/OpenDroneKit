/* Home: where every verb on this screen lives.

   Machine-readable placements, read by the reachability and parity tests instead of
   the old `toolbar: [` arrays. One entry per control:

     { action: "<ACTIONS key>", where: "home.<region>", label: "<visible label>" }

   `action` resolves in the merged registry (core ACTIONS plus ./actions.js), and the
   rendered button carries data-action="<action>" inside [data-where="<where>"].
   Schema and rules: docs/ui/ARCHITECTURE.md, "surface.js". Home has no Field screen,
   so every placement is desk only. */

export const SCREEN = "home";

const desk = ["desk"];

export const PLACEMENTS = [
  // Left panel: the projects on this machine, and the machine's own state.
  { action: "New Project", where: "home.projects.footer", label: "New project", modes: desk, inventory: "home.new-project" },
  { action: "Open Project Folder", where: "home.projects.footer", label: "Open project folder", modes: desk, inventory: "home.open-project-folder" },
  { action: "Sync", where: "home.projects.status", label: "Check", modes: desk, inventory: "home.sync-a-refresh-in-reality" },

  // Project header: the open project's own verbs.
  { action: "Import", where: "home.header", label: "Import photos", modes: desk, inventory: "home.import-imagery" },
  { action: "Open", where: "home.header.more", label: "Open another project…", modes: desk, inventory: "home.open-project" },
  { action: "Share", where: "home.header.more", label: "Share read-only link…", modes: desk, inventory: "home.share-read-only-link" },
  { action: "Revoke Links", where: "home.header.more", label: "Revoke share links", modes: desk, inventory: "home.revoke-links" },

  // Summary tiles: the PHOTOS tile opens the datasets popover.
  { action: "Select Dataset", where: "home.tiles.datasets", label: "Use this dataset", modes: desk, inventory: "home.dataset-list-and-selection" },

  // Report card.
  { action: "Open Report", where: "home.report", label: "Open report", modes: desk, inventory: "home.report-card" },
  { action: "Generate Report", where: "home.report", label: "Regenerate", modes: desk, inventory: "home.report-card" },

  // Inspector: activity, notifications, exports.
  { action: "View Activity", where: "home.activity", label: "View all activity", modes: desk, inventory: "home.recent-activity-and-full-audit-log" },
  { action: "Mark Notification Read", where: "home.notifications.row", label: "Mark read", modes: desk, inventory: "home.notifications" },
  { action: "Mark All Notifications Read", where: "home.notifications", label: "Mark all read", modes: desk, inventory: "home.notifications" },
  { action: "Clear Notifications", where: "home.notifications", label: "Clear", modes: desk, inventory: "home.notifications" },
  { action: "Open Export", where: "home.exports.row", label: "Open", modes: desk, inventory: "home.exports-card" },
  { action: "Open Exports Folder", where: "home.exports", label: "Open folder", modes: desk, inventory: "home.exports-card" },
];

/* Rows whose control only changes the interface (data-control, not data-action):
   home.project-summary-tiles, home.reconstruction-card ("View log" -> Process),
   home.measurements-card ("Open in Analyse"), home.project-tags (chip -> Review filter),
   home.system-status (MODELS / BUILD rows -> Settings), home.storage-used (STORAGE row,
   project_storage). home.fleet-summary lives on Fleet and home.load-demo-workflow in
   Settings > About, as the inventory says. */
export const CONTROLS = [
  { control: "open:plan", where: "home.header", label: "New mission", inventory: "home.project-summary-tiles" },
  { control: "open:analyse", where: "home.header", label: "Open analysis", inventory: "home.measurements-card" },
  { control: "open:analyse", where: "home.map", label: "Open map", inventory: "home.project-summary-tiles" },
  { control: "open:process", where: "home.reconstruction", label: "View log", inventory: "home.reconstruction-card" },
  { control: "open:settings", where: "home.projects.status", label: "Build", inventory: "home.system-status" },
  { control: "review-tag", where: "home.header.tags", label: "tag chip", inventory: "home.project-tags" },
  { control: "panel-tab:notifications", where: "home.activity", label: "Notifications", inventory: "home.notifications" },
];
