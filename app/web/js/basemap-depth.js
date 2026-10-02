/* How deep the Esri basemaps really go, here.
 *
 * Esri's tiled services answer a request for a level they have no imagery at with
 * HTTP 200 and a grey "Map data not yet available" picture. MapLibre cannot tell that
 * apart from imagery, so with the satellite source declaring z19 (or z23) it painted
 * the placeholder over the whole view as soon as the operator zoomed past what Esri
 * holds for the site -- at Aukerman, anything deeper than z19 -- instead of enlarging
 * the last real tile. The real depth changes from place to place, so no fixed maxzoom
 * is right everywhere.
 *
 * Two things here, used by both the classic window (js/map.js) and the cockpit
 * (js/workspace/mapview.js):
 *
 *  - `esri://` tile URLs. Each tile's bytes are checked against the placeholder
 *    (by size and SHA-256) before MapLibre sees them; a placeholder is replaced by the
 *    covering part of the nearest ancestor tile that has imagery, enlarged -- exactly
 *    what MapLibre shows past a source's maxzoom. The placeholder never reaches the map.
 *    (`blankTile=false`, which makes the service answer 404, cannot be used from a
 *    page: the 404 carries no CORS header, so the browser reports "Failed to fetch"
 *    and the map would take a coverage gap for a lost connection.)
 *  - `track(map)`. Asks the service's tilemap (…/MapServer/tilemap/{z}/{row}/{col}/{w}/{h},
 *    which lists which tiles exist) about the tiles on screen and caps the basemap
 *    source's maxzoom at the deepest level that has imagery for the whole view, so the
 *    map enlarges the deepest real tile rather than asking for tiles that do not exist.
 *    Answers are cached per tile; with no network nothing is asked and nothing changes.
 */
(function (global) {
  'use strict';

  const PROTOCOL = 'esri';
  // What the error says when a level has no imagery. Deliberately free of the words
  // the offline detectors look for ("tile", "fetch", "network", "Failed"): this is the
  // service being honest about coverage, not the machine being offline.
  const NO_IMAGERY = 'No imagery at this level here; showing the deepest level that has it.';
  // Before the service has said anything about a place, cap here: Esri World Imagery
  // holds z17 almost everywhere on land, and a lower cap only costs sharpness briefly.
  const SAFE_DEPTH = 17;
  const MAX_WINDOW = 32;

  const known = new Map();
  const inflight = new Map();

  function esriUrl(httpsUrl) {
    return httpsUrl.replace(/^https?:\/\//, PROTOCOL + '://');
  }

  /* The MapServer root of an esri:// (or https) tile template, or null. */
  function serviceRoot(template) {
    const match = /^(?:esri|https?):\/\/([^/]+\/arcgis\/rest\/services\/.+?\/MapServer)\/tile\//i
      .exec(String(template || ''));
    return match ? 'https://' + match[1] : null;
  }

  // Esri's "Map data not yet available" tile: World_Imagery and World_Topo_Map, 256x256
  // JPEG, 2521 bytes, observed 2026-09-29 (also core/tile_cache.PLACEHOLDER_SHA256).
  const PLACEHOLDERS = new Map([
    [2521, '9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858'],
  ]);
  const stats = { tiles: 0, placeholders: 0, enlarged: 0, holes: 0 };

  async function sha256(buffer) {
    const subtle = global.crypto && global.crypto.subtle;
    if (!subtle) return null;
    const digest = await subtle.digest('SHA-256', buffer);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  async function isPlaceholder(buffer) {
    const expected = PLACEHOLDERS.get(buffer.byteLength);
    if (!expected) return false;
    return (await sha256(buffer)) === expected;
  }

  async function fetchTile(url, signal) {
    const response = await fetch(url, { signal, credentials: 'omit' });
    if (!response.ok) throw new Error(`Tile request failed: HTTP ${response.status} ${url}`);
    return response.arrayBuffer();
  }

  /* The quarter (or sixteenth, ...) of an ancestor tile that covers this tile, enlarged
     to a full tile: what MapLibre shows past a source's maxzoom, done here for a tile
     the service answered with its placeholder. The pixels are the ancestor's own. */
  async function enlargeAncestor(url, z, x, y, signal) {
    if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') return null;
    for (let up = 1; up <= Math.min(z, 8); up += 1) {
      const pz = z - up;
      const px = x >> up;
      const py = y >> up;
      const parentUrl = url.replace(`/tile/${z}/${y}/${x}`, `/tile/${pz}/${py}/${px}`);
      const body = await fetchTile(parentUrl, signal);
      if (await isPlaceholder(body)) continue;
      const n = 1 << up;
      const size = 256 / n;
      const bitmap = await createImageBitmap(new Blob([body]), (x - px * n) * size, (y - py * n) * size,
                                             size, size, { resizeWidth: 256, resizeHeight: 256,
                                                           resizeQuality: 'high' });
      const canvas = new OffscreenCanvas(256, 256);
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      const blob = await canvas.convertToBlob({ type: 'image/png' });
      return blob.arrayBuffer();
    }
    return null;
  }

  let registered = false;
  function registerProtocol(maplibregl) {
    if (registered || !maplibregl || typeof maplibregl.addProtocol !== 'function') return registered;
    maplibregl.addProtocol(PROTOCOL, async (params, abortController) => {
      const url = params.url.replace(new RegExp('^' + PROTOCOL + '://'), 'https://');
      const signal = abortController && abortController.signal;
      const body = await fetchTile(url, signal);
      stats.tiles += 1;
      if (!(await isPlaceholder(body))) return { data: body };
      // No imagery at this level here. Never hand the placeholder to the map.
      stats.placeholders += 1;
      const at = /\/tile\/(\d+)\/(\d+)\/(\d+)/.exec(url);
      const enlarged = at ? await enlargeAncestor(url, Number(at[1]), Number(at[3]), Number(at[2]), signal) : null;
      if (enlarged) {
        stats.enlarged += 1;
        return { data: enlarged };
      }
      stats.holes += 1;
      const error = new Error(NO_IMAGERY);
      error.odkNoImagery = true;
      throw error;
    });
    registered = true;
    return true;
  }

  function isNoImagery(event) {
    const error = event && event.error;
    return Boolean(error && (error.odkNoImagery || String(error.message || '') === NO_IMAGERY));
  }

  function tileX(lon, z) {
    const n = 2 ** z;
    return Math.max(0, Math.min(n - 1, Math.floor(((lon + 180) / 360) * n)));
  }

  function tileY(lat, z) {
    const n = 2 ** z;
    const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
    const r = (clamped * Math.PI) / 180;
    return Math.max(0, Math.min(n - 1, Math.floor(((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * n)));
  }

  /* The tiles covering [west, south, east, north] at level z, as a window no larger
     than MAX_WINDOW each way (centred, when the view is bigger than that). */
  function tileWindow(bounds, z) {
    let x0 = tileX(bounds[0], z);
    let x1 = tileX(bounds[2], z);
    let y0 = tileY(bounds[3], z);
    let y1 = tileY(bounds[1], z);
    if (x1 < x0) x1 = x0; // across the antimeridian: judge the western part
    if (x1 - x0 + 1 > MAX_WINDOW) { const c = (x0 + x1) >> 1; x0 = c - MAX_WINDOW / 2; x1 = x0 + MAX_WINDOW - 1; }
    if (y1 - y0 + 1 > MAX_WINDOW) { const c = (y0 + y1) >> 1; y0 = c - MAX_WINDOW / 2; y1 = y0 + MAX_WINDOW - 1; }
    return { x0, x1, y0, y1 };
  }

  function key(root, z, x, y) { return `${root}|${z}|${x}|${y}`; }

  /* Every tile of the window at level z exists (true), one does not (false), or the
     service could not be asked (null). */
  async function levelCovered(root, z, win, fetchImpl) {
    const missing = () => {
      for (let y = win.y0; y <= win.y1; y += 1) {
        for (let x = win.x0; x <= win.x1; x += 1) if (!known.has(key(root, z, x, y))) return true;
      }
      return false;
    };
    if (missing()) {
      const w = win.x1 - win.x0 + 1;
      const h = win.y1 - win.y0 + 1;
      const url = `${root}/tilemap/${z}/${win.y0}/${win.x0}/${w}/${h}`;
      let pending = inflight.get(url);
      if (!pending) {
        pending = (async () => {
          const response = await fetchImpl(url, { credentials: 'omit' });
          if (!response.ok) throw new Error(`tilemap HTTP ${response.status}`);
          const body = await response.json();
          const loc = body.location || {};
          const data = body.data || [];
          for (let r = 0; r < (loc.height || 0); r += 1) {
            for (let c = 0; c < (loc.width || 0); c += 1) {
              known.set(key(root, z, loc.left + c, loc.top + r), data[r * loc.width + c] ? 1 : 0);
            }
          }
        })().finally(() => inflight.delete(url));
        inflight.set(url, pending);
      }
      try {
        await pending;
      } catch (_err) {
        return null;
      }
    }
    for (let y = win.y0; y <= win.y1; y += 1) {
      for (let x = win.x0; x <= win.x1; x += 1) {
        // A tile the tilemap did not describe is not assumed to exist.
        if (known.get(key(root, z, x, y)) !== 1) return false;
      }
    }
    return true;
  }

  /* The deepest level, no deeper than `ceiling`, at which every tile in view exists.
     Looks two levels past the current zoom so zooming in finds the cap already set.
     null when the service could not be asked. */
  async function deepestLevel(root, bounds, zoom, ceiling, fetchImpl) {
    const top = Math.min(ceiling, Math.max(0, Math.round(zoom) + 2));
    const floor = Math.max(0, top - 8);
    for (let z = top; z > floor; z -= 1) {
      const covered = await levelCovered(root, z, tileWindow(bounds, z), fetchImpl);
      if (covered === null) return null;
      if (covered) return z;
    }
    return floor;
  }

  /* Keep `map`'s basemap source capped at the depth the service really has here.

     `ceilingOf()` answers the provider's own ceiling for the current basemap, or null
     when the basemap is not an Esri tiled service (street, offline cache). The cap is
     set on the live source object, which MapLibre reads on every update, so nothing is
     reloaded and nothing flickers. */
  function track(map, ceilingOf, options) {
    const opts = options || {};
    const sourceId = opts.sourceId || 'basemap';
    const fetchImpl = opts.fetch || ((url, init) => global.fetch(url, init));
    let generation = 0;
    let timer = null;

    const apply = (depth) => {
      const source = map.getSource && map.getSource(sourceId);
      if (!source || typeof depth !== 'number') return;
      map._odkBasemapDepth = depth;
      if (source.maxzoom !== depth) {
        source.maxzoom = depth;
        // MapLibre recomputes which tiles it needs only when the view or the style
        // changes; setMaxZoom (to the same value) is the public call that marks the
        // sources for recomputation, so the new cap takes effect without a move.
        map.setMaxZoom(map.getMaxZoom());
      }
    };

    const update = async () => {
      timer = null;
      const source = map.getSource && map.getSource(sourceId);
      const tiles = source && (source.tiles || (source._options && source._options.tiles));
      const root = serviceRoot(tiles && tiles[0]);
      const ceiling = root ? ceilingOf() : null;
      if (!root || ceiling == null) return;
      if (map._odkBasemapDepth == null && source.maxzoom > SAFE_DEPTH) apply(Math.min(ceiling, SAFE_DEPTH));
      const mine = ++generation;
      const b = map.getBounds();
      const depth = await deepestLevel(root, [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()],
                                       map.getZoom(), ceiling, fetchImpl);
      if (mine !== generation || depth == null) return;
      // Imagery all the way to two levels past the view says nothing about deeper
      // levels, so the cap is not pulled below the usual depth for that: zooming in
      // stays sharp while the next answer is on its way. A shallower answer is a real
      // limit found here, and is used as it is.
      const top = Math.min(ceiling, Math.max(0, Math.round(map.getZoom()) + 2));
      const cap = depth >= top ? Math.max(depth, Math.min(ceiling, SAFE_DEPTH)) : depth;
      apply(Math.max(0, Math.min(ceiling, cap)));
    };

    const schedule = (delay) => {
      if (timer) return;
      timer = setTimeout(update, delay);
    };

    map.on('moveend', () => schedule(0));
    map.on('zoom', () => schedule(200));
    // A style change (a basemap switch) brings a new source at its declared depth.
    map.on('styledata', () => {
      const source = map.getSource && map.getSource(sourceId);
      if (source && map._odkBasemapDepth != null && source.maxzoom !== map._odkBasemapDepth) {
        map._odkBasemapDepth = null;
      }
      schedule(0);
    });
    map.on('load', () => schedule(0));
    return { update, deepestLevel: () => map._odkBasemapDepth };
  }

  const api = {
    PROTOCOL, NO_IMAGERY, SAFE_DEPTH, esriUrl, serviceRoot, registerProtocol, isNoImagery,
    isPlaceholder, tileWindow, deepestLevel, track, stats, _known: known,
  };
  global.ODKBasemapDepth = api;
})(typeof window !== 'undefined' ? window : globalThis);
