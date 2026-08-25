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
