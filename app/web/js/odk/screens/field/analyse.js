/* Field Analyse: the Field Daylight renderer for the Analyse workspace. It imports the
   same ../analyse/model.js as the desk renderer, so both modes show the same Api values.
   A stub until the Field unit lands: it renders nothing, rather than anything invented. */

export { CALLS } from "../analyse/model.js";

export const SCREEN = "analyse";
export const MODE = "field";

export function render(regions /* { left, main } */, _context) {
  for (const node of Object.values(regions || {})) {
    if (node && node.replaceChildren) node.replaceChildren();
  }
}
