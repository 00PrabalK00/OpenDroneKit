/*
 * Which view the canvas is currently showing.
 *
 * A module rather than a shell property because the canvas is built by workspaces.js,
 * which the shell does not pass arguments to. Recording the mode on the shell and
 * re-rendering was not enough: the canvas rebuilt itself identically, so the view
 * buttons reported success and the screen never moved.
 */

let current = "map";
const listeners = new Set();

export function currentView() {
  return current;
}

export function setView(view) {
  if (!view || view === current) return false;
  current = view;
  listeners.forEach((fn) => fn(current));
  return true;
}

export function onViewChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/* The image the user picked, if any.
 *
 * Kept beside the view mode rather than inside the shell, because the canvas is built by
 * workspaces.js which the shell does not pass arguments to -- the same reason the view
 * mode lives here.
 */
let picked = null;

export function currentImage() {
  return picked;
}

export function setImage(image) {
  picked = image || null;
  listeners.forEach((fn) => fn(current));
}
