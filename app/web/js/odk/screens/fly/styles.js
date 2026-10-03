/* Fly: the screen's own stylesheet (docs/ui/ARCHITECTURE.md: screen CSS lives in
   screens/<ws>/<ws>.css). Linked once, from this module's own folder, so no shared page
   has to list it. */

export function ensureFlyStyles() {
  if (typeof document === "undefined" || document.querySelector("link[data-odk-fly-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./fly.css", import.meta.url).href;
  link.setAttribute("data-odk-fly-css", "");
  document.head.appendChild(link);
}
