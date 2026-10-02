/* Every screen module, in header-tab order. The shell (js/odk/shell.js) opens
   screens through this table; tests import it to prove each module loads. */

export const DESK_SCREENS = [
  { id: "home", label: "Home" },
  { id: "plan", label: "Plan" },
  { id: "fly", label: "Fly" },
  { id: "process", label: "Process" },
  { id: "analyse", label: "Analyse" },
  { id: "review", label: "Review" },
  { id: "fleet", label: "Fleet" },
  { id: "report", label: "Report" },
];

/** Settings is a header icon button, not a tab. */
export const SETTINGS = { id: "settings", label: "Settings" };

export const FIELD_SCREENS = [
  { id: "plan", label: "Plan" },
  { id: "fly", label: "Fly" },
  { id: "process", label: "Process" },
  { id: "analyse", label: "Analyse" },
  { id: "report", label: "Report" },
];

/** Old cockpit workspace ids, accepted by Shell.open() so ported tests keep working. */
export const WORKSPACE_ALIASES = {
  planning: "plan",
  flight: "fly",
  processing: "process",
  verification: "process",
  measurements: "analyse",
  twin: "analyse",
  thermal: "analyse",
  inspection: "review",
  reports: "report",
  projects: "home",
  developers: "settings",
};

export const PARTS = ["surface", "model", "desk", "actions"];

/** Import every screen module; resolves to { "<ws>/<part>": module }. */
export async function loadAll() {
  const out = {};
  for (const { id } of [...DESK_SCREENS, SETTINGS]) {
    for (const part of PARTS) out[`${id}/${part}`] = await import(`./${id}/${part}.js`);
  }
  for (const { id } of FIELD_SCREENS) out[`field/${id}`] = await import(`./field/${id}.js`);
  return out;
}
