/* Process: data. The live() calls and the values derived from them, shared by the
   desk renderer (./desk.js) and, where one exists, the Field renderer.

   Every figure a screen shows must come from a call listed here: no literal number
   may be rendered from screens/** (docs/ui/ARCHITECTURE.md, "Honesty rules"). */

/** Api methods this screen reads. Each entry: { method, args }. */
export const CALLS = [];

/** Turn the Api results (keyed by method) into what the renderers draw. */
export function derive(results) {
  return { ...results };
}
