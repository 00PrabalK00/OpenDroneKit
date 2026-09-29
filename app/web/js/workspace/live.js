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
import { connected, tryCall, whenReady } from "./api.js";
import { demoEnabled } from "./demo.js";

/* Said instead of a panel's own empty message when there is no application to ask.
 *
 * "No projects yet. Use New Project" is a statement about the machine's projects. With
 * no bridge nobody has asked, and the cockpit in a plain browser was telling people their
 * machine was empty. */
export const NOT_CONNECTED =
  "Not connected to the application. Start OpenDroneKit to see this project's data, "
  + "or open the cockpit with ?demo=1 for the bundled example.";

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
  /* Marks everything drawn inside as having come from the application.
   *
   * The shell decides whether to show the EXAMPLE DATA banner by looking for rows that
   * are NOT inside one of these. That makes the banner self-maintaining: converting a
   * panel to live() takes it out of the synthetic count with no list to update, and a
   * new panel that renders a literal is counted from the moment it is written. */
  const host = el("div", { class: "live", "data-live": "" });
  host.appendChild(el("div", { class: "live-wait", text: "Reading…" }));

  const settle = async () => {
    /* The panels render before pywebview has injected its bridge. Asking straight away
       raced it: every call failed with "bridge is not ready" and the panel settled on
       its EMPTY state -- "No projects yet" over a machine with projects -- and nothing
       re-asked once the bridge arrived. Waiting costs nothing when it is already up. */
    if (!demoEnabled()) {
      await whenReady();
      if (!connected()) {
        host.innerHTML = "";
        host.appendChild(emptyState(NOT_CONNECTED));
        return;
      }
    }
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

  host._settle = settle;
  HOSTS.add(host);
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
  /* get_state() returns `project`. The `|| state.active_project` fallback beside it was
     dead -- no Api method has ever returned that key -- and a defensive fallback to a
     name that does not exist reads as uncertainty about the contract rather than
     robustness. */
  return state.project || null;
}

/** Rows for a properties() call, dropping anything the application did not report. */
export function reported(pairs) {
  return pairs
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([label, value, unit]) =>
      unit ? { label, value: String(value), unit } : { label, value: String(value) });
}

/* Every live panel on screen, so it can be asked again.
 *
 * A panel asked once, when its workspace opened. Pressing Import, Plan or New Project
 * changed what the application held and left every panel describing the state from before
 * the button was pressed -- "Nothing imported" beside a toast saying 24 images had been.
 * Forms are deliberately not re-rendered: only answers are re-asked, so nothing the
 * operator typed is reset. */
const HOSTS = new Set();

export function refreshLive() {
  for (const host of [...HOSTS]) {
    if (!host.isConnected) { HOSTS.delete(host); continue; }
    host._settle();
  }
  if (typeof window !== "undefined") window.dispatchEvent(new Event("odk:refresh"));
}
