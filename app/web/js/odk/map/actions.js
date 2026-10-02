/* Verbs the map's own furniture runs (the Layers popover on every map).
 *
 * The core catalogue (js/workspace/actions.js) is frozen, and these belong to no single
 * screen: the Layers popover is on every map. js/odk/registry.js merges them as the
 * "map" source, with the same duplicate check as the screens' catalogues, so the buttons
 * carry data-action like any other verb and the palette and menu reach them too.
 *
 * Both read their target from ctx.payload (the popover passes the row it was pressed
 * on) and fall back to what the shell can tell them, so they also work from the palette.
 */

import { call } from "../../workspace/api.js";
import { visibleBounds } from "../../workspace/mapview.js";

/* The deepest level the offline cache asks for. Past z19 Esri has no imagery at most
   sites (js/basemap-depth.js), so a deeper request only costs tiles that never arrive. */
const CACHE_MAX_ZOOM = 19;
const CACHE_LEVELS = 3;

export const ACTIONS = {
  "Remove Layer": {
    describe: "Take a layer off the project's layer list. The file on disk is kept.",
    async run(ctx) {
      const layer = (ctx.payload && ctx.payload.layer)
        || (typeof ctx.selectedLayer === "function" ? ctx.selectedLayer() : null);
      if (!layer || !layer.id) return { skipped: "Choose a layer to remove." };
      const name = layer.name || layer.id;
      const sure = typeof ctx.confirm === "function"
        ? await ctx.confirm(`Remove "${name}" from the layer list? The file on disk is kept.`)
        : true;
      if (!sure) return { skipped: `${name} kept.` };
      const result = await call("remove_layer", layer.id);
      if (!result.removed) return { error: `${name} is no longer in the layer list.` };
      return { message: `Removed ${name} from the layer list. The file is still on disk.`, refresh: true };
    },
  },

  "Cache Basemap View": {
    describe: "Download the satellite basemap for the area on screen, for use with no network.",
    async run(ctx) {
      const view = (ctx.payload && ctx.payload.view) || visibleBounds();
      if (!view) return { skipped: "Open a map to choose the area to cache." };
      const minZoom = Math.max(0, Math.min(CACHE_MAX_ZOOM, Math.floor(view.zoom)));
      const maxZoom = Math.min(CACHE_MAX_ZOOM, minZoom + CACHE_LEVELS);
      const result = await call("cache_basemap_area", view.west, view.south, view.east, view.north,
        minZoom, maxZoom, "satellite");
      if (typeof ctx.watchJob === "function" && result.job_id) ctx.watchJob(result.job_id);
      return { message: `Caching ${result.tiles} satellite tiles (z${minZoom}–${maxZoom}) for offline use.`,
               job: result.job_id };
    },
  },
};
