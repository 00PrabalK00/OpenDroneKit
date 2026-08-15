/* Shared browser-side annotation shaping. The API validates the same contract again. */
(function (global) {
  'use strict';

  const TYPES = ['point', 'line', 'polygon', 'rectangle', 'circle', 'freehand', 'text'];
  const SEVERITIES = ['critical', 'high', 'medium', 'low', 'info'];
  const STATUSES = ['open', 'in_review', 'resolved', 'dismissed'];

  function clone(value) { return JSON.parse(JSON.stringify(value)); }

  function point(value, name) {
    if (!Array.isArray(value) || value.length < 2) throw new Error(`${name} needs two coordinates.`);
    const result = [Number(value[0]), Number(value[1])];
    if (!result.every(Number.isFinite)) throw new Error(`${name} coordinates must be finite.`);
    return result;
  }

  function rectangle(geometry) {
    if (!geometry || geometry.type !== 'Polygon' || !Array.isArray(geometry.coordinates[0])) {
      throw new Error('Rectangle drawing requires a Polygon gesture.');
    }
    const positions = geometry.coordinates[0].map((value) => point(value, 'Rectangle vertex'));
    const xs = positions.map((value) => value[0]), ys = positions.map((value) => value[1]);
    const west = Math.min(...xs), east = Math.max(...xs), south = Math.min(...ys), north = Math.max(...ys);
    if (!(east > west && north > south)) throw new Error('Rectangle gesture has no area.');
    return { type: 'Polygon', coordinates: [[
      [west, south], [east, south], [east, north], [west, north], [west, south]
    ]] };
  }

  function normaliseGeometry(annotationType, input, radiusM) {
    if (!TYPES.includes(annotationType)) throw new Error(`Unsupported annotation type: ${annotationType}.`);
    const geometry = clone(input || {});
    if (annotationType === 'rectangle') return rectangle(geometry);
    if (['point', 'circle', 'text'].includes(annotationType)) {
      if (geometry.type !== 'Point') throw new Error(`${annotationType} annotations require a Point.`);
      geometry.coordinates = point(geometry.coordinates, 'Point');
      if (annotationType === 'circle') {
        const radius = Number(radiusM);
        if (!Number.isFinite(radius) || radius <= 0) throw new Error('Circle radius must be positive metres.');
        geometry.radius_m = radius;
      }
      return geometry;
    }
    if (['line', 'freehand'].includes(annotationType)) {
      if (geometry.type !== 'LineString' || !Array.isArray(geometry.coordinates) || geometry.coordinates.length < 2) {
        throw new Error(`${annotationType} annotations require a two-point LineString.`);
      }
      geometry.coordinates = geometry.coordinates.map((value) => point(value, 'Line vertex'));
      return geometry;
    }
    if (geometry.type !== 'Polygon' || !Array.isArray(geometry.coordinates[0])) {
      throw new Error('Polygon annotations require a Polygon.');
    }
    const ring = geometry.coordinates[0].map((value) => point(value, 'Polygon vertex'));
    if (ring.length < 3) throw new Error('Polygon annotation needs at least three vertices.');
    if (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1]) {
      ring.push(ring[0].slice());
    }
    geometry.coordinates = [ring];
    return geometry;
  }

  function buildAnnotation(metadata, geometry) {
    const source = metadata || {};
    if (!TYPES.includes(source.annotation_type)) throw new Error('Choose an annotation type.');
    if (!SEVERITIES.includes(source.severity)) throw new Error('Choose a valid annotation severity.');
    if (!STATUSES.includes(source.status)) throw new Error('Choose a valid annotation status.');
    if (!String(source.label || '').trim()) throw new Error('Annotation label is required.');
    return {
      source_type: String(source.source_type || 'map'),
      source_id: String(source.source_id || 'viewer2d'),
      annotation_type: source.annotation_type,
      geometry: normaliseGeometry(source.annotation_type, geometry, source.radius_m),
      crs_epsg: source.crs_epsg == null ? null : Number(source.crs_epsg),
      label: String(source.label).trim(),
      severity: source.severity,
      status: source.status,
      note: String(source.note || ''),
      include_in_report: source.include_in_report !== false
    };
  }

  const exported = { TYPES, SEVERITIES, STATUSES, buildAnnotation, normaliseGeometry };
  global.ODKHubAnnotations = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
