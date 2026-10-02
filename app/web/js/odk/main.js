/* Entry point for app/web/odk.html (the v1 interface, opened with ODK_UI=odk or
 * --ui odk).
 *
 *   odk.html            the frame. Until the shell unit (U2) lands this is a blank
 *                       frame: header with the brand, empty panels, empty footer.
 *   odk.html?gallery    every design-system component, in desk and field mode.
 *   ?mode=field|desk    force a mode for this load (not persisted).
 *
 * The mode lives on <html data-mode>, persisted in localStorage under MODE_KEY.
 */

import { el } from "./components.js";

export const MODE_KEY = "odk.mode";
const MODES = new Set(["desk", "field"]);

function storedMode() {
  try { return localStorage.getItem(MODE_KEY); } catch { return null; }
}

export function setMode(mode, { persist = true } = {}) {
  if (!MODES.has(mode)) throw new Error(`unknown mode ${mode}`);
  document.documentElement.setAttribute("data-mode", mode);
  if (persist) { try { localStorage.setItem(MODE_KEY, mode); } catch { /* private window */ } }
}

/** A frame with nothing in it: the regions the shell will fill, and the brand. */
export function mountBlankFrame(root) {
  const header = el("header", { class: "odk-frame-header", "data-region": "header" }, [
    el("span", { class: "odk-brand" }, [
      el("img", { src: "assets/brand/mark-dark.svg", alt: "" }), "OpenDroneKit",
    ]),
  ]);
  const frame = el("div", { class: "odk-frame", "data-frame": "blank" }, [
    header,
    el("aside", { class: "odk-frame-left", "data-region": "left", "aria-label": "Panel" }),
    el("main", { class: "odk-frame-main", "data-region": "main" }),
    el("aside", { class: "odk-frame-inspector", "data-region": "inspector", "aria-label": "Inspector" }),
    el("footer", { class: "odk-frame-footer", "data-region": "footer" }),
  ]);
  root.replaceChildren(frame);
  return frame;
}

async function boot() {
  const params = new URLSearchParams(window.location.search);
  const forced = params.get("mode");
  if (MODES.has(forced)) setMode(forced, { persist: false });
  else setMode(MODES.has(storedMode()) ? storedMode() : "desk", { persist: false });

  const root = document.getElementById("odk-root");
  if (params.has("gallery")) {
    const { mountGallery } = await import("./gallery.js");
    mountGallery(root);
  } else {
    mountBlankFrame(root);
  }
  window.__odkReady = true;
  window.dispatchEvent(new Event("odk:ready"));
}

boot().catch((error) => {
  window.__odkBootError = String(error && error.stack || error);
  console.error(error);
});
