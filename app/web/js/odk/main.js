/* Entry point for app/web/odk.html (the v1 interface, opened with ODK_UI=odk or
 * --ui odk).
 *
 *   odk.html            the shell (js/odk/shell.js): frame, screens, palette, menu.
 *   odk.html?gallery    every design-system component, in desk and field mode.
 *   ?mode=field|desk    force a mode for this load (not persisted).
 *
 * The mode lives on <html data-mode>, persisted in localStorage under MODE_KEY by
 * Shell.setMode(). window.odkShell is the mounted shell, for the browser harnesses.
 */

export const MODE_KEY = "odk.mode"; // the same key shell.js persists under
const MODES = new Set(["desk", "field"]);

function storedMode() {
  try { return localStorage.getItem(MODE_KEY); } catch { return null; }
}

async function boot() {
  const params = new URLSearchParams(window.location.search);
  const forced = params.get("mode");
  const mode = MODES.has(forced) ? forced : MODES.has(storedMode()) ? storedMode() : "desk";
  document.documentElement.setAttribute("data-mode", mode);

  const root = document.getElementById("odk-root");
  if (params.has("gallery")) {
    const { mountGallery } = await import("./gallery.js");
    mountGallery(root);
  } else {
    const { Shell } = await import("./shell.js");
    const shell = new Shell(root, { mode });
    window.odkShell = shell;
    await shell.ready;
  }
  window.__odkReady = true;
  window.dispatchEvent(new Event("odk:ready"));
}

boot().catch((error) => {
  window.__odkBootError = String((error && error.stack) || error);
  console.error(error);
});
