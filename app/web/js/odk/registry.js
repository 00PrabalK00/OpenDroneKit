/* The one verb catalogue the v1 interface runs: the cockpit's core ACTIONS
 * (js/workspace/actions.js, frozen), the map furniture's verbs (map/actions.js) and the
 * verbs each screen adds in screens/<ws>/actions.js.
 *
 * Merged once, at import. A key that appears twice is refused with an error naming both
 * sources: two entries for one verb are two behaviours for one button, and the one
 * nobody tests is the one that ships. Everything that runs a verb -- buttons
 * (data-action), the command palette, the native menu -- resolves keys here, so a
 * screen's new verb is reachable from all three the moment it is declared.
 */

import {
  ACTIONS as CORE_ACTIONS, UNWIRED, describeMeasurement, prerequisite,
} from "../workspace/actions.js";
import { ACTIONS as HOME } from "./screens/home/actions.js";
import { ACTIONS as PLAN } from "./screens/plan/actions.js";
import { ACTIONS as FLY } from "./screens/fly/actions.js";
import { ACTIONS as PROCESS } from "./screens/process/actions.js";
import { ACTIONS as ANALYSE } from "./screens/analyse/actions.js";
import { ACTIONS as REVIEW } from "./screens/review/actions.js";
import { ACTIONS as FLEET } from "./screens/fleet/actions.js";
import { ACTIONS as REPORT } from "./screens/report/actions.js";
import { ACTIONS as SETTINGS } from "./screens/settings/actions.js";
import { ACTIONS as MAP } from "./map/actions.js";

export { UNWIRED, describeMeasurement, prerequisite };

/** Screen catalogues, in header order. */
export const SCREEN_ACTIONS = {
  home: HOME, plan: PLAN, fly: FLY, process: PROCESS, analyse: ANALYSE,
  review: REVIEW, fleet: FLEET, report: REPORT, settings: SETTINGS,
};

/**
 * Merge catalogues into one, refusing duplicates.
 * `sources` is [[name, {key: entry}]]; returns {actions, sourceOf}.
 */
export function mergeCatalogues(sources) {
  const actions = {};
  const sourceOf = {};
  for (const [name, catalogue] of sources) {
    for (const [key, entry] of Object.entries(catalogue || {})) {
      if (Object.prototype.hasOwnProperty.call(actions, key)) {
        throw new Error(`ACTIONS key "${key}" is declared twice: in ${sourceOf[key]} and in ${name}. `
          + "A screen adds verbs; it never redefines one (docs/ui/ARCHITECTURE.md).");
      }
      if (!entry || typeof entry !== "object") {
        throw new Error(`ACTIONS key "${key}" in ${name} is not an entry object.`);
      }
      actions[key] = entry;
      sourceOf[key] = name;
    }
  }
  return { actions, sourceOf };
}

const merged = mergeCatalogues([
  ["core", CORE_ACTIONS],
  // The map furniture's own verbs (Layers popover on every map), U3.
  ["map", MAP],
  ...Object.entries(SCREEN_ACTIONS).map(([ws, catalogue]) => [`screens/${ws}`, catalogue]),
]);

/** Every verb, core and screen, by key. */
export const ACTIONS = Object.freeze(merged.actions);

/** Where each key was declared: "core", "map" or "screens/<ws>". */
export const SOURCE_OF = Object.freeze(merged.sourceOf);

/** Every key, in declaration order. */
export function actionKeys() {
  return Object.keys(ACTIONS);
}

/** Whether `key` names a verb. */
export function has(key) {
  return Object.prototype.hasOwnProperty.call(ACTIONS, key);
}

/** Resolve an alias chain ("Run AI" -> "Start") to the entry that does the work. */
export function resolve(key) {
  let entry = has(key) ? ACTIONS[key] : null;
  let guard = 0;
  while (entry && entry.alias && guard++ < 4) entry = has(entry.alias) ? ACTIONS[entry.alias] : null;
  return entry;
}

/** One line describing what a verb does, for the palette. */
export function describe(key) {
  const entry = resolve(key);
  return (entry && entry.describe) || "";
}
