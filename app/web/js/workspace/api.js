/*
 * The cockpit's connection to the real application.
 *
 * The cockpit was drawn before it had anything to draw from: every panel held a literal,
 * and the shell had no idea whether a Python bridge existed. That is why it showed a
 * populated survey on a machine with no projects at all.
 *
 * Three states, and the difference between them is the whole point:
 *
 *   connected     pywebview is present and answering. Panels show what it returns,
 *                 including "none" -- an empty project list is a real answer.
 *   disconnected  no bridge (a browser, a static preview). Panels say so and offer the
 *                 demo, rather than inventing content that looks like a survey.
 *   unwired       there IS a bridge, but this panel has no API behind it yet. Named
 *                 explicitly, because a panel that silently shows nothing is
 *                 indistinguishable from one whose answer is nothing.
 *
 * The last state is the one worth having. Several workspaces were designed against
 * capabilities that do not exist yet, and the honest thing is for the UI to under-claim
 * and say which -- not to fill the gap with a plausible number.
 */

import { demoAnswer, demoEnabled } from "./demo.js";

const PENDING = new Map();

/** Whether the Python bridge is available in this page. */
export function connected() {
  /* pywebview creates `window.pywebview.api` as an empty object first and fills it just
     before firing pywebviewready. Treating the empty object as a bridge let panels call
     methods that did not exist yet, fail, and render "no data" for the session. */
  return Boolean(typeof window !== "undefined" && window.pywebview && window.pywebview.api
    && typeof window.pywebview.api.get_state === "function");
}

/**
 * Call an Api method. Mirrors app.js so both surfaces treat the bridge identically:
 * a result carrying ok === false is an error, not data.
 */
export async function call(method, ...args) {
  // Demo mode is answered from the bundled example and never reaches Python, even when a
  // bridge is present -- otherwise the example banner sits over the real projects.
  if (demoEnabled()) return demoAnswer(method, ...args);
  if (!connected()) throw new Error("Python bridge is not ready.");
  const fn = window.pywebview.api[method];
  if (typeof fn !== "function") throw new Error(`unwired: ${method}`);
  const result = await fn(...args);
  // A refusal that explains itself in `summary` or `reason` rather than `error` -- as
  // verify_site does -- surfaced as "Unknown error", hiding the only useful sentence.
  if (result && result.ok === false) {
    const error = new Error(result.error || result.reason || result.summary || result.message
      || `${method} refused without giving a reason.`);
    // The refusal's own fields (e.g. preflight_blocked) for a caller that can act on them.
    error.result = result;
    throw error;
  }
  return result;
}

/**
 * Call and return null on failure, having recorded why.
 *
 * A panel is a bad place to throw: the dock keeps rendering and the operator sees a gap
 * where the telemetry should be, with nothing saying the call failed.
 */
export async function tryCall(method, ...args) {
  try {
    return await call(method, ...args);
  } catch (error) {
    lastError.set(method, String(error.message || error));
    return null;
  }
}

export const lastError = new Map();

/**
 * Which of these Api methods actually exist on the bridge.
 *
 * Used to decide between "no data" and "not wired yet" without guessing, so a panel can
 * state which of the two it is looking at.
 */
export function available(methods) {
  if (!connected()) return new Set();
  return new Set(methods.filter((name) => typeof window.pywebview.api[name] === "function"));
}

/**
 * Resolve once the bridge is ready, or immediately if it already is.
 *
 * pywebview fires `pywebviewready` after the page loads; a cockpit that reads the API on
 * DOMContentLoaded races it and concludes there is no bridge.
 */
export function whenReady(timeoutMs = 4000) {
  const key = "ready";
  if (PENDING.has(key)) return PENDING.get(key);
  const promise = new Promise((resolve) => {
    if (connected()) return resolve(true);
    if (typeof window === "undefined") return resolve(false);
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    window.addEventListener("pywebviewready", () => done(true), { once: true });
    // Resolving false rather than hanging: a browser has no bridge and never will, and
    // a cockpit that waits forever shows an empty frame with no explanation.
    setTimeout(() => done(connected()), timeoutMs);
  });
  PENDING.set(key, promise);
  // A "not ready" answer is not kept: a bridge that arrives late must be seen by the
  // next caller, not answered "no bridge" from the cache for the rest of the session.
  promise.then((ready) => { if (!ready) PENDING.delete(key); });
  return promise;
}
