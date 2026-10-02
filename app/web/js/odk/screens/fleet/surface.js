/* Fleet: where every verb on this screen lives.

   Machine-readable placements, read by the reachability and parity tests instead of
   the old `toolbar: [` arrays. One entry per control:

     { action: "<ACTIONS key>", where: "fleet.<region>", label: "<visible label>" }

   `action` must resolve in the merged registry (core ACTIONS plus ./actions.js), and
   the rendered button must carry data-action="<action>". Schema and rules:
   docs/ui/ARCHITECTURE.md, "surface.js". Filled by the Fleet workspace unit; every
   inventory row for this workspace (docs/ui/inventory.json) must end up here. */

export const SCREEN = "fleet";

export const PLACEMENTS = [];
