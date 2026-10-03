/* Plan: undo and redo for the area of interest.

   Every change to the area -- drawn, edited, imported, outlined from the cameras,
   cleared, loaded with a saved version -- reaches the Api first (set_aoi and friends),
   so the history is kept as what the Api reported after each change: the screen calls
   observe() with the ring mission_overview returned. Undo puts the previous ring back
   through set_aoi with its own source, so the Api, the map and every panel agree on
   what is current. Nothing here draws or plans. */

const undo = [];
const redo = [];
let current = null;      // {ring, source} last reported by the Api
let applying = false;    // the next observed ring is ours (undo/redo), not a new edit

const LIMIT = 50;

const key = (entry) => JSON.stringify(entry ? entry.ring : null);

/** Record the area the Api reports now. A different ring is a new step. */
export function observe(aoi) {
  const next = aoi && aoi.set ? { ring: aoi.polygon || [], source: aoi.source || "" } : { ring: [], source: "" };
  if (current && key(current) === key(next)) { current = next; return; }
  if (current && !applying) {
    undo.push(current);
    if (undo.length > LIMIT) undo.shift();
    redo.length = 0;
  }
  applying = false;
  current = next;
}

export function canUndo() { return undo.length > 0; }
export function canRedo() { return redo.length > 0; }

/** The ring to put back for an undo, or null; the caller sends it to set_aoi. */
export function takeUndo() {
  if (!undo.length) return null;
  const previous = undo.pop();
  if (current) redo.push(current);
  applying = true;
  return previous;
}

export function takeRedo() {
  if (!redo.length) return null;
  const next = redo.pop();
  if (current) undo.push(current);
  applying = true;
  return next;
}

/** A failed set_aoi leaves the Api where it was: the step is not consumed. */
export function cancelApplying() { applying = false; }
