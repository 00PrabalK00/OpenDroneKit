/* MapLibre GL map surface: basemaps, drawing, mission and raster layers, measurement.
 *
 * Geometry drawn here is the real input to mission planning. There is no placeholder
 * survey area: if nothing is drawn, planning reports that instead of inventing one.
 */

(function (global) {
  'use strict';

  const EARTH_RADIUS_M = 6371008.8;

  /* Raster basemaps.

     `maxzoom` is the deepest zoom the provider actually serves. MapLibre overzooms
     past it by scaling the last real tile, so setting it correctly is what turns a
     grey screen at z20 into a usable, if soft, picture. Setting it too high asks the
     provider for tiles that do not exist, which is what made the topographic layer
     look broken.

     `offline` is served from this deployment's own tile cache. It is not a blank
     background: whatever has been cached for the working area is shown, and the rest
     is honestly empty rather than pretending coverage exists. */
  const OFFLINE_TILE_URL = 'http://127.0.0.1:8000/tiles/{z}/{x}/{y}.png';

  const BASEMAPS = {
    satellite: {
      label: 'Satellite',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Esri, Maxar, Earthstar Geographics',
      // Esri imagery reaches z23 over most populated areas.
      maxzoom: 23
    },
    street: {
      label: 'Street',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      attribution: '&copy; OpenStreetMap contributors',
      maxzoom: 19
    },
    topo: {
      label: 'Topographic',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Esri, USGS, NOAA',
      // The topographic service stops at z19; asking beyond it returns nothing, which
      // is why this looked broken when it was set to match satellite.
      maxzoom: 19
    },
    terrain: {
      label: 'Terrain (hillshade)',
      tiles: ['https://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Esri, Airbus DS, USGS, NGA',
      maxzoom: 16
    },
    usgs_topo: {
      label: 'USGS Topo (US)',
      tiles: ['https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'USGS The National Map',
      maxzoom: 16
    },
    offline: {
      label: 'Offline cache',
      tiles: [OFFLINE_TILE_URL],
      attribution: 'Locally cached tiles',
      maxzoom: 20,
      offline: true
    }
  };

  function basemapStyle(key) {
    const spec = BASEMAPS[key] || BASEMAPS.offline;
    if (!spec.tiles) {
      return {
        version: 8,
        sources: {},
        layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#0d1117' } }]
      };
    }
    return {
      version: 8,
      sources: {
        basemap: {
          type: 'raster',
          tiles: spec.tiles,
          tileSize: 256,
          // The provider's real limit. MapLibre scales the last real tile beyond it
          // rather than requesting tiles that do not exist.
          maxzoom: spec.maxzoom || 19,
          attribution: spec.attribution
        }
      },
      layers: [
        { id: 'background', type: 'background', paint: { 'background-color': '#0d1117' } },
        { id: 'basemap', type: 'raster', source: 'basemap' }
      ]
    };
  }

  function haversine(a, b) {
    const toRad = Math.PI / 180;
    const dLat = (b[1] - a[1]) * toRad;
    const dLon = (b[0] - a[0]) * toRad;
    const lat1 = a[1] * toRad;
    const lat2 = b[1] * toRad;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
    return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  function pathLength(coords) {
    let total = 0;
    for (let i = 1; i < coords.length; i += 1) total += haversine(coords[i - 1], coords[i]);
    return total;
  }

  /* Planar area on a local equirectangular projection. Accurate well past the scale
     of any single inspection site. */
  function ringArea(ring) {
    if (ring.length < 3) return 0;
    const lat0 = ring.reduce((s, p) => s + p[1], 0) / ring.length;
    const lon0 = ring.reduce((s, p) => s + p[0], 0) / ring.length;
    const kx = Math.cos((lat0 * Math.PI) / 180) * ((Math.PI * EARTH_RADIUS_M) / 180);
    const ky = (Math.PI * EARTH_RADIUS_M) / 180;
    const xs = ring.map((p) => (p[0] - lon0) * kx);
    const ys = ring.map((p) => (p[1] - lat0) * ky);
    let area = 0;
    for (let i = 0; i < ring.length; i += 1) {
      const j = (i + 1) % ring.length;
      area += xs[i] * ys[j] - xs[j] * ys[i];
    }
    return Math.abs(area) / 2;
  }

  function boundsOf(coords) {
    let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
    coords.forEach((c) => {
      west = Math.min(west, c[0]); east = Math.max(east, c[0]);
      south = Math.min(south, c[1]); north = Math.max(north, c[1]);
    });
    return Number.isFinite(west) ? [[west, south], [east, north]] : null;
  }

  function flatten(geojson) {
    const out = [];
    const walk = (node) => {
      if (!node) return;
      if (Array.isArray(node)) {
        if (node.length >= 2 && typeof node[0] === 'number' && typeof node[1] === 'number') out.push(node);
        else node.forEach(walk);
        return;
      }
      if (node.type === 'FeatureCollection') node.features.forEach(walk);
      else if (node.type === 'Feature') walk(node.geometry);
      else if (node.coordinates) walk(node.coordinates);
    };
    walk(geojson);
    return out;
  }

  class OdkMap {
    constructor(container, options) {
      this.opts = options || {};
      this.basemap = 'satellite';
      this.tool = 'pan';
      this.measurePoints = [];
      this.rasterLayers = new Map();
      this.vectorLayers = new Map();

      this.map = new maplibregl.Map({
        container: container,
        // Tests, embedded deployments and air-gapped sites may provide a complete
        // local style.  The default remains satellite for the normal Hub UI.
        style: this.opts.style || basemapStyle(this.basemap),
        center: this.opts.center || [0, 20],
        zoom: this.opts.zoom || 2,
        attributionControl: { compact: true }
      });

      this.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
      this.map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-right');

      this.draw = new MapboxDraw({
        displayControlsDefault: false,
        controls: {},
        styles: drawStyles()
      });
      this.map.addControl(this.draw);

      this.map.on('load', () => {
        this._initOverlays();
        if (this.opts.onReady) this.opts.onReady();
      });

      this.map.on('mousemove', (e) => {
        if (this.opts.onCursor) this.opts.onCursor(e.lngLat.lng, e.lngLat.lat);
      });

      this.map.on('draw.create', (e) => this._onDrawChange(e));
      this.map.on('draw.update', (e) => this._onDrawChange(e));
      this.map.on('draw.delete', (e) => this._onDrawChange(e));
      this.map.on('click', (e) => this._onClick(e));
    }

    /* Overlay sources are created once; the style reload on basemap change re-adds them. */
    _initOverlays() {
      const empty = { type: 'FeatureCollection', features: [] };

      this._addSourceOnce('mission', empty);
      this._addLayerOnce({
        id: 'mission-line', type: 'line', source: 'mission',
        filter: ['==', ['geometry-type'], 'LineString'],
        paint: { 'line-color': '#3d9df6', 'line-width': 2.2, 'line-opacity': 0.95 }
      });
      this._addLayerOnce({
        id: 'mission-points', type: 'circle', source: 'mission',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 2.2, 18, 5],
          'circle-color': '#ffd166', 'circle-stroke-color': '#1b2430', 'circle-stroke-width': 1
        }
      });

      this._addSourceOnce('measure', empty);
      this._addLayerOnce({
        id: 'measure-line', type: 'line', source: 'measure',
        paint: { 'line-color': '#e7b04a', 'line-width': 2, 'line-dasharray': [2, 1.5] }
      });
      this._addLayerOnce({
        id: 'measure-points', type: 'circle', source: 'measure',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: { 'circle-radius': 4, 'circle-color': '#e7b04a' }
      });

      this.rasterLayers.forEach((entry, id) => this._mountRaster(id, entry));
      this.vectorLayers.forEach((entry, id) => this._mountVector(id, entry));
    }

    _addSourceOnce(id, data) {
      if (!this.map.getSource(id)) this.map.addSource(id, { type: 'geojson', data: data });
    }

    _addLayerOnce(spec) {
      if (!this.map.getLayer(spec.id)) this.map.addLayer(spec);
    }

    setBasemap(key) {
      if (!BASEMAPS[key]) return;
      this.basemap = key;
      const drawn = this.draw.getAll();
      this.map.setStyle(basemapStyle(key));
      this.map.once('styledata', () => {
        this._initOverlays();
        // A style swap wipes drawn features, so they are restored explicitly.
        if (drawn && drawn.features.length) this.draw.set(drawn);
        if (this._missionData) this._setData('mission', this._missionData);
      });
    }

    setTool(tool) {
      this.tool = tool;
      this.measurePoints = [];
      this._setData('measure', { type: 'FeatureCollection', features: [] });
      if (tool === 'aoi' || tool === 'nofly') this.draw.changeMode('draw_polygon');
      else if (tool === 'delete') this.draw.trash();
      else this.draw.changeMode('simple_select');
      if (this.opts.onToolChange) this.opts.onToolChange(tool);
    }

    _onClick(e) {
      if (this.tool !== 'measure-distance' && this.tool !== 'measure-area') return;
      this.measurePoints.push([e.lngLat.lng, e.lngLat.lat]);
      const pts = this.measurePoints;
      const features = pts.map((p) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: p }, properties: {} }));
      if (pts.length >= 2) {
        const line = this.tool === 'measure-area' ? pts.concat([pts[0]]) : pts;
        features.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: line }, properties: {} });
      }
      this._setData('measure', { type: 'FeatureCollection', features: features });

      if (this.opts.onMeasure) {
        if (this.tool === 'measure-area' && pts.length >= 3) {
          const area = ringArea(pts);
          this.opts.onMeasure(`Area: ${area >= 10000 ? (area / 10000).toFixed(3) + ' ha' : area.toFixed(1) + ' m²'} (${pts.length} vertices)`);
        } else if (pts.length >= 2) {
          const d = pathLength(pts);
          this.opts.onMeasure(`Distance: ${d >= 1000 ? (d / 1000).toFixed(3) + ' km' : d.toFixed(1) + ' m'} (${pts.length} points)`);
        } else {
          this.opts.onMeasure('Click another point to measure.');
        }
      }
    }

    /* Drawn polygons are tagged by the tool that made them, so AOI and no-fly stay
       distinguishable after the fact. */
    _onDrawChange(event) {
      (event.features || []).forEach((feature) => {
        if (feature.geometry.type === 'Polygon' && !feature.properties.odkRole) {
          const role = this.tool === 'nofly' ? 'nofly' : 'aoi';
          this.draw.setFeatureProperty(feature.id, 'odkRole', role);
        }
      });
      if (this.opts.onGeometryChange) this.opts.onGeometryChange(this.getGeometry());
    }

    getGeometry() {
      const all = this.draw.getAll();
      const aoi = [];
      const nofly = [];
      all.features.forEach((feature) => {
        if (feature.geometry.type !== 'Polygon') return;
        const ring = feature.geometry.coordinates[0];
        if (feature.properties.odkRole === 'nofly') nofly.push(ring);
        else aoi.push(ring);
      });
      return { aoi: aoi.length ? aoi[aoi.length - 1] : null, allAoi: aoi, nofly: nofly };
    }

    clearRole(role) {
      const all = this.draw.getAll();
      all.features.forEach((feature) => {
        const featureRole = feature.properties.odkRole || 'aoi';
        if (featureRole === role) this.draw.delete(feature.id);
      });
      if (this.opts.onGeometryChange) this.opts.onGeometryChange(this.getGeometry());
    }

    _setData(id, data) {
      const source = this.map.getSource(id);
      if (source) source.setData(data);
    }

    setMission(geojson) {
      this._missionData = geojson || { type: 'FeatureCollection', features: [] };
      this._setData('mission', this._missionData);
    }

    addRaster(id, dataUri, coordinates, opacity) {
      const entry = { dataUri: dataUri, coordinates: coordinates, opacity: opacity == null ? 1 : opacity };
      this.rasterLayers.set(id, entry);
      this._mountRaster(id, entry);
    }

    _mountRaster(id, entry) {
      const sourceId = `raster-${id}`;
      if (this.map.getLayer(sourceId)) this.map.removeLayer(sourceId);
      if (this.map.getSource(sourceId)) this.map.removeSource(sourceId);
      this.map.addSource(sourceId, { type: 'image', url: entry.dataUri, coordinates: entry.coordinates });
      this.map.addLayer({
        id: sourceId, type: 'raster', source: sourceId,
        paint: { 'raster-opacity': entry.opacity, 'raster-fade-duration': 0 }
      }, this.map.getLayer('mission-line') ? 'mission-line' : undefined);
    }

    addVector(id, geojson, opacity) {
      const entry = { geojson: geojson, opacity: opacity == null ? 1 : opacity };
      this.vectorLayers.set(id, entry);
      this._mountVector(id, entry);
    }

    _mountVector(id, entry) {
      const sourceId = `vector-${id}`;
      ['-point', '-line', '-fill'].forEach((suffix) => {
        if (this.map.getLayer(sourceId + suffix)) this.map.removeLayer(sourceId + suffix);
      });
      if (this.map.getSource(sourceId)) this.map.removeSource(sourceId);
      this.map.addSource(sourceId, { type: 'geojson', data: entry.geojson });
      this.map.addLayer({
        id: sourceId + '-fill', type: 'fill', source: sourceId,
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'fill-color': '#35c98b', 'fill-opacity': 0.18 * entry.opacity }
      });
      this.map.addLayer({
        id: sourceId + '-line', type: 'line', source: sourceId,
        filter: ['!=', ['geometry-type'], 'Point'],
        paint: { 'line-color': '#35c98b', 'line-width': 1.6, 'line-opacity': entry.opacity }
      });
      this.map.addLayer({
        id: sourceId + '-point', type: 'circle', source: sourceId,
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-radius': 3.4, 'circle-color': '#35c98b',
          'circle-stroke-width': 1, 'circle-stroke-color': '#12161c', 'circle-opacity': entry.opacity
        }
      });
    }

    setLayerVisible(id, visible) {
      const value = visible ? 'visible' : 'none';
      [`raster-${id}`, `vector-${id}-fill`, `vector-${id}-line`, `vector-${id}-point`].forEach((layerId) => {
        if (this.map.getLayer(layerId)) this.map.setLayoutProperty(layerId, 'visibility', value);
      });
    }

    setLayerOpacity(id, opacity) {
      if (this.map.getLayer(`raster-${id}`)) this.map.setPaintProperty(`raster-${id}`, 'raster-opacity', opacity);
      if (this.map.getLayer(`vector-${id}-fill`)) this.map.setPaintProperty(`vector-${id}-fill`, 'fill-opacity', 0.18 * opacity);
      if (this.map.getLayer(`vector-${id}-line`)) this.map.setPaintProperty(`vector-${id}-line`, 'line-opacity', opacity);
      if (this.map.getLayer(`vector-${id}-point`)) this.map.setPaintProperty(`vector-${id}-point`, 'circle-opacity', opacity);
    }

    removeLayer(id) {
      this.rasterLayers.delete(id);
      this.vectorLayers.delete(id);
      [`raster-${id}`, `vector-${id}-fill`, `vector-${id}-line`, `vector-${id}-point`].forEach((layerId) => {
        if (this.map.getLayer(layerId)) this.map.removeLayer(layerId);
      });
      [`raster-${id}`, `vector-${id}`].forEach((sourceId) => {
        if (this.map.getSource(sourceId)) this.map.removeSource(sourceId);
      });
    }

    fitTo(geojsonOrBounds, padding) {
      let bounds = geojsonOrBounds;
      if (geojsonOrBounds && geojsonOrBounds.type) bounds = boundsOf(flatten(geojsonOrBounds));
      else if (Array.isArray(geojsonOrBounds) && geojsonOrBounds.length === 4) {
        bounds = [[geojsonOrBounds[0], geojsonOrBounds[1]], [geojsonOrBounds[2], geojsonOrBounds[3]]];
      }
      if (!bounds) return false;
      this.map.fitBounds(bounds, { padding: padding == null ? 60 : padding, duration: 600, maxZoom: 19 });
      return true;
    }

    fitToCoords(coords, padding) {
      const bounds = boundsOf(coords);
      if (!bounds) return false;
      this.map.fitBounds(bounds, { padding: padding == null ? 60 : padding, duration: 600, maxZoom: 19 });
      return true;
    }

    /* Move to a search result. A bounding box frames the place at its true extent;
       a bare point has no extent, so it gets a working zoom rather than a guess at one. */
    goToPlace(place, options) {
      if (!place || !Number.isFinite(place.lon) || !Number.isFinite(place.lat)) return false;
      const zoom = (options && options.zoom) || 16;
      if (Array.isArray(place.bounds) && place.bounds.length === 4) {
        const [west, south, east, north] = place.bounds;
        // A point-like result can carry a degenerate box; fitBounds would over-zoom.
        if (Math.abs(east - west) > 1e-6 && Math.abs(north - south) > 1e-6) {
          this.map.fitBounds([[west, south], [east, north]], { padding: 80, duration: 800, maxZoom: 18 });
          return true;
        }
      }
      this.map.flyTo({ center: [place.lon, place.lat], zoom, duration: 800 });
      return true;
    }

    /* A transient marker so the operator can see what the search actually matched. */
    markPlace(place) {
      if (!place) return;
      this.clearPlaceMarker();
      const element = document.createElement('div');
      element.className = 'place-marker';
      element.title = place.name || '';
      this._placeMarker = new maplibregl.Marker({ element })
        .setLngLat([place.lon, place.lat])
        .addTo(this.map);
    }

    clearPlaceMarker() {
      if (this._placeMarker) {
        this._placeMarker.remove();
        this._placeMarker = null;
      }
    }
  }

  /* Draw styling: AOI in blue, no-fly in red, matching the toolbar semantics. */
  function drawStyles() {
    const noFly = ['==', ['get', 'user_odkRole'], 'nofly'];
    return [
      {
        id: 'gl-draw-polygon-fill', type: 'fill', filter: ['all', ['==', '$type', 'Polygon']],
        paint: {
          'fill-color': ['case', noFly, '#e5645f', '#3d9df6'],
          'fill-opacity': ['case', ['==', ['get', 'active'], 'true'], 0.3, 0.16]
        }
      },
      {
        id: 'gl-draw-polygon-stroke', type: 'line', filter: ['all', ['==', '$type', 'Polygon']],
        paint: {
          'line-color': ['case', noFly, '#e5645f', '#3d9df6'],
          'line-width': 2,
          'line-dasharray': ['case', noFly, ['literal', [2, 1.5]], ['literal', [1, 0]]]
        }
      },
      {
        id: 'gl-draw-line', type: 'line', filter: ['all', ['==', '$type', 'LineString']],
        paint: { 'line-color': '#3d9df6', 'line-width': 2 }
      },
      {
        id: 'gl-draw-vertex', type: 'circle',
        filter: ['all', ['==', 'meta', 'vertex'], ['==', '$type', 'Point']],
        paint: { 'circle-radius': 4.5, 'circle-color': '#ffffff', 'circle-stroke-color': '#3d9df6', 'circle-stroke-width': 2 }
      },
      {
        id: 'gl-draw-midpoint', type: 'circle', filter: ['all', ['==', 'meta', 'midpoint']],
        paint: { 'circle-radius': 3, 'circle-color': '#8b98a9' }
      },
      {
        id: 'gl-draw-point', type: 'circle',
        filter: ['all', ['==', '$type', 'Point'], ['!=', 'meta', 'vertex'], ['!=', 'meta', 'midpoint']],
        paint: { 'circle-radius': 4, 'circle-color': '#ffd166' }
      }
    ];
  }

  global.OdkMap = OdkMap;
  global.OdkMapUtils = { BASEMAPS: BASEMAPS, haversine: haversine, pathLength: pathLength, ringArea: ringArea, boundsOf: boundsOf, flatten: flatten };
})(window);
