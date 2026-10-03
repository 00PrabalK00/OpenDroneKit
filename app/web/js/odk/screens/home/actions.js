/* Home: verbs new to this screen. The shared core catalogue (js/workspace/actions.js)
   is frozen; a screen unit adds verbs only here, with the same shape as a core entry.
   js/odk/registry.js merges them and refuses a duplicate key.

   Each one reaches the Api. Payloads come from the row the button sits on (a dataset's
   folder, a notification's id, an exported file's path); the palette and the native
   menu, which have no row, fall back to the open project's newest item or ask. */

import { call } from "../../../workspace/api.js";

async function projectRoot() {
  const state = await call("get_state");
  const root = state && state.project && (state.project.root_dir || state.project.path);
  if (!root) throw new Error("No project is open.");
  return String(root);
}

export const ACTIONS = {
  "Open Project Folder": {
    describe: "Show the open project's folder in the file manager.",
    async run() {
      const root = await projectRoot();
      await call("open_path", root);
      return { message: `Opened ${root}` };
    },
  },

  "Select Dataset": {
    describe: "Make one of this project's imported photo sets the active dataset.",
    async run(ctx) {
      let folder = ctx.payload && ctx.payload.path;
      if (!folder) {
        const listed = await call("dataset_summary");
        const rows = (listed.datasets || []).filter((d) => d.exists);
        if (!rows.length) return { skipped: "No dataset on disk in this project. Import photos first." };
        folder = await ctx.choose("Active dataset", rows.map((d) => ({
          label: d.name, value: d.path, hint: d.image_count != null ? `${d.image_count} photos` : d.path,
        })));
        if (!folder) return { skipped: "Cancelled." };
      }
      await call("set_active_dataset", String(folder));
      return { message: `Active dataset: ${String(folder).split(/[\\/]/).pop()}.`, refresh: true };
    },
  },

  "View Activity": {
    describe: "Read this project's full audit log, newest first.",
    async run(ctx) {
      const { auditLine } = await import("../../menu.js");
      const log = await call("audit_log");
      const lines = (log.events || []).map(auditLine);
      await ctx.showText("Activity (newest first)", lines.join("\n") || "No events recorded for this project.");
      return { message: `${lines.length} event(s) in this project's audit log.` };
    },
  },

  "Mark Notification Read": {
    describe: "Mark one notification as read.",
    async run(ctx) {
      let id = ctx.payload && ctx.payload.id;
      if (!id) {
        const listed = await call("list_notifications", true);
        const unread = listed.notifications || [];
        if (!unread.length) return { skipped: "No unread notifications." };
        id = await ctx.choose("Mark as read", unread.map((n) => ({ label: n.title, value: n.id, hint: n.detail })));
        if (!id) return { skipped: "Cancelled." };
      }
      const result = await call("mark_notification_read", String(id));
      return { message: `Marked read. ${result.unread} unread.`, refresh: true };
    },
  },

  "Mark All Notifications Read": {
    describe: "Mark every notification in this project as read.",
    async run() {
      const result = await call("mark_all_notifications_read");
      return { message: `Marked ${result.cleared} notification(s) read.`, refresh: true };
    },
  },

  "Clear Notifications": {
    confirm: "Remove every notification in this project? The audit log keeps its own record.",
    describe: "Remove this project's notifications (the audit log is kept).",
    async run() {
      const result = await call("clear_notifications");
      return { message: `Removed ${result.removed} notification(s).`, refresh: true };
    },
  },

  "Open Export": {
    describe: "Open an exported file, or the newest export.",
    async run(ctx) {
      let path = ctx.payload && ctx.payload.path;
      if (!path) {
        const listed = await call("list_exports");
        path = ((listed.files || []).find(Boolean) || {}).path;
        if (!path) return { skipped: "Nothing has been exported from this project yet." };
      }
      await call("open_path", String(path));
      return { message: `Opened ${path}` };
    },
  },

  "Open Exports Folder": {
    describe: "Show the folder the newest export was written to.",
    async run(ctx) {
      let folder = ctx.payload && ctx.payload.folder;
      if (!folder) {
        const listed = await call("list_exports");
        folder = ((listed.files || []).find(Boolean) || {}).folder;
        if (!folder) return { skipped: "Nothing has been exported from this project yet." };
      }
      await call("open_path", String(folder));
      return { message: `Opened ${folder}` };
    },
  },
};
