/* Field Fly: the Field Daylight renderer for the Fly workspace. It imports the
   same ../fly/model.js as the desk renderer, so both modes show the same Api values.
   A stub until the Field unit lands: it renders nothing, rather than anything invented. */

export { CALLS } from "../fly/model.js";

export const SCREEN = "fly";
export const MODE = "field";

export function render(regions /* { left, main } */, _context) {
  for (const node of Object.values(regions || {})) {
    if (node && node.replaceChildren) node.replaceChildren();
  }
}
