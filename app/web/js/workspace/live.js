/*
 * Panels that show what the application actually holds.
 *
 * The dock renders panels synchronously: `render()` returns an element and that is what
 * the user sees. So every panel that wanted project data had a choice between blocking
 * the render and baking a constant in, and they all baked a constant in. The result was
 * a cockpit that displayed "1,842 images" and "DEMO site 1" over a real 77-image project,
 * with the EXAMPLE DATA banner suppressed because the shell believed connected meant
 * measured.
 *
 * `live()` gives them the third option. It returns an element immediately, asks the
 * application in the background, and replaces the contents when the answer arrives.
 *
 * The rule it enforces is the one that matters: when there is no answer, the panel says
 * so. It does not fall back to a plausible number. An empty state is a fact about the
 * project; an invented figure is a lie that looks like a measurement, and this codebase
 * would rather show nothing.
 */

import { el } from "./primitives.js";
import { tryCall } from "./api.js";

/**
 * A panel backed by one or more API calls.
 *
 * @param {object}   spec
 * @param {string[]} spec.calls   Api method names, called in parallel.
 * @param {Function} spec.render  (results) => Element. Called only when data arrived.
 * @param {string}   spec.empty   What to say when the application has nothing to show.
 * @param {Function} [spec.isEmpty] (results) => boolean. Defaults to "every result null".
 */
export function live({ calls, render, empty, isEmpty }) {
  const host = el("div", { class: "live" });
  host.appendChild(el("div", { class: "live-wait", text: "Reading…" }));

  const settle = async () => {
    let results;
    try {
      results = await Promise.all(calls.map((name) => tryCall(name)));
    } catch {
      results = calls.map(() => null);
    }

    const nothing = isEmpty
      ? isEmpty(results)
      : results.every((r) => r === null || r === undefined);

    host.innerHTML = "";
    if (nothing) {
      host.appendChild(emptyState(empty));
      return;
    }
    try {
      host.appendChild(render(results));
    } catch (error) {
      // A renderer that throws must not leave the panel showing "Reading…" forever,
      // and must not be mistaken for an empty project either.
      host.appendChild(emptyState(`Could not read this: ${error.message}`));
    }
  };

  settle();
  return host;
}

/**
 * What a panel shows when the application has nothing for it.
 *
 * Deliberately plain. The temptation is to fill the space with something illustrative,
 * and that is exactly how the sample data ended up on screen in the first place.
 */
export function emptyState(message) {
  return el("div", { class: "empty-state" }, [
    el("p", { class: "empty-msg", text: message }),
  ]);
}

/** The active project, or null. Every panel below needs it and none should refetch it. */
export async function activeProject() {
  const state = await tryCall("get_state");
  if (!state) return null;
  return state.project || state.active_project || null;
}

/** Rows for a properties() call, dropping anything the application did not report. */
export function reported(pairs) {
  return pairs
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([label, value, unit]) =>
      unit ? { label, value: String(value), unit } : { label, value: String(value) });
}
