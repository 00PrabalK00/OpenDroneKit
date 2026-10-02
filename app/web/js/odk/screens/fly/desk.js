/* Fly: desk renderer. Draws into the frame regions the shell hands it
   (left, main, inspector) using js/odk/components.js only. A stub until the Fly
   workspace unit lands: it renders nothing, rather than anything invented. */

export const SCREEN = "fly";

export function render(regions /* { left, main, inspector } */, _context) {
  for (const node of Object.values(regions || {})) {
    if (node && node.replaceChildren) node.replaceChildren();
  }
}
