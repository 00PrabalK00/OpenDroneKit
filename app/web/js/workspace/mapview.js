/* A real MapLibre map inside a workspace canvas, or an honest placeholder.
 *
 * The framework does not depend on this. A canvas is a mounting point, and whether a
 * live map, a 3D viewport or a placeholder occupies it is a per-workspace decision --
 * which is why adding this changes no other file.
 *
 * Two failure modes are handled rather than hidden, because both are ordinary in the
 * field. MapLibre may be absent (a stripped deployment). The network may be absent,
 * which for an offline-first product is not a failure at all -- it is Tuesday. In both
 * cases the canvas says which of the two happened instead of showing an empty grey
 * rectangle that could mean anything.
 */

import { el } from "./primitives.js";

/* Style built here rather than imported from js/map.js: that module belongs to the hub
 * and carries its own draw tooling and event wiring. Borrowing it would couple two UIs
 * that should be free to change independently. */
const OFFLINE_STYLE = {
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": "#0d1117" } }],
};

const SATELLITE_STYLE = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Esri World Imagery",
    },
  },
  layers: [{ id: "basemap", type: "raster", source: "basemap" }],
};

export function mapAvailable() {
  return typeof window !== "undefined" && typeof window.maplibregl !== "undefined";
}

/**
 * Mount a map into a canvas element.
 *
 * Returns the map instance, or null when one could not be created. Null is a legitimate
 * answer -- callers keep their placeholder rather than assuming a map appeared.
 */
export function mountMap(container, {
  centre = [77.4022, 23.2591],
  zoom = 15,
  offline = false,
  onReady,
} = {}) {
  if (!mapAvailable()) {
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip warn", text: "map library not loaded" }),
    ]));
    return null;
  }

  const holder = el("div", { style: "position:absolute;inset:0" });
  container.insertBefore(holder, container.firstChild);

  let map;
  try {
    map = new window.maplibregl.Map({
      container: holder,
      style: offline ? OFFLINE_STYLE : SATELLITE_STYLE,
      center: centre,
      zoom,
      attributionControl: false,
    });
  } catch (error) {
    // A map that cannot construct must not take the workspace down with it: the panels
    // around it are still useful, and a blank canvas beats a blank application.
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip error", text: `map failed: ${error.message}` }),
    ]));
    return null;
  }

  map.addControl(new window.maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-right");

  // Tile errors are reported once, not per tile: a site with no connectivity would
  // otherwise fill the console with hundreds of identical failures and bury anything
  // else. Offline is the expected state for this product, so it is stated calmly.
  let reportedOffline = false;
  map.on("error", (event) => {
    const message = String(event?.error?.message || "");
    if (!/tile|fetch|network|Failed/i.test(message) || reportedOffline) return;
    reportedOffline = true;
    container.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip warn", text: "basemap tiles unavailable — offline" }),
    ]));
  });

  map.on("load", () => {
    const placeholder = container.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    if (onReady) onReady(map);
  });

  // A live coordinate readout, because a survey interface that cannot tell you where
  // the cursor is has left something out.
  const readout = el("div", { class: "canvas-overlay br", html: "—" });
  container.appendChild(readout);
  map.on("mousemove", (event) => {
    readout.innerHTML =
      `${event.lngLat.lat.toFixed(5)} N &nbsp; ${event.lngLat.lng.toFixed(5)} E ` +
      `&nbsp; z${map.getZoom().toFixed(1)}`;
  });

  return map;
}

/** Draw a mission's flight lines and capture points onto a mounted map. */
export function showMission(map, { line = [], captures = [] } = {}) {
  if (!map || !map.isStyleLoaded()) return false;

  if (line.length > 1) {
    const id = "odk-mission-line";
    if (map.getSource(id)) map.removeLayer(id), map.removeSource(id);
    map.addSource(id, {
      type: "geojson",
      data: { type: "Feature", geometry: { type: "LineString", coordinates: line } },
    });
    map.addLayer({
      id, type: "line", source: id,
      paint: { "line-color": "#3b82f6", "line-width": 1.5, "line-opacity": 0.9 },
    });
  }

  if (captures.length) {
    const id = "odk-captures";
    if (map.getSource(id)) map.removeLayer(id), map.removeSource(id);
    map.addSource(id, {
      type: "geojson",
      data: {
        type: "FeatureCollection",
        features: captures.map((c, index) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: c },
          properties: { seq: index + 1 },
        })),
      },
    });
    map.addLayer({
      id, type: "circle", source: id,
      paint: {
        "circle-radius": 2.5,
        "circle-color": "#22d3ee",
        "circle-stroke-width": 0.5,
        "circle-stroke-color": "#0d1015",
      },
    });
  }
  return true;
}
