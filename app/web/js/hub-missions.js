/* Mission preview/playback derived from the persisted compiled-plan API response. */
(function (global) {
  'use strict';

  function finite(value, name) {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${name} must be finite.`);
    return number;
  }

  function validateSimulation(payload) {
    if (!payload || payload.type !== 'odk-mission-simulation') throw new Error('Not an OpenDroneKit mission simulation.');
    if (payload.source !== 'persisted_compiled_plan') throw new Error('Simulation must identify the persisted compiled plan as its source.');
    if (!Array.isArray(payload.timeline) || !payload.timeline.length) throw new Error('Compiled plan contains no playback frames.');
    const timeline = payload.timeline.map((row, index) => {
      if (!Array.isArray(row.position) || row.position.length < 3) throw new Error(`Frame ${index + 1} has no 3D position.`);
      return Object.assign({}, row, {
        time_s: finite(row.time_s, `frame ${index + 1} time`),
        position: row.position.slice(0, 3).map((value) => finite(value, `frame ${index + 1} position`))
      });
    });
    const terrain = payload.terrain || { status: 'unavailable', samples: [], reason: 'Terrain was not described.' };
    if (terrain.status !== 'available') terrain.samples = [];
    return Object.assign({}, payload, { timeline, terrain });
  }

  function localPositions(timeline) {
    const origin = timeline[0].position;
    const latScale = 111320;
    const lonScale = latScale * Math.cos(origin[1] * Math.PI / 180);
    return timeline.map((row) => ({
      x: (row.position[0] - origin[0]) * lonScale,
      y: (row.position[1] - origin[1]) * latScale,
      z: row.position[2], row
    }));
  }

  class MissionSimulator {
    constructor(canvas, onFrame) {
      if (!canvas || typeof canvas.getContext !== 'function') throw new Error('Mission simulation needs a canvas.');
      this.canvas = canvas; this.context = canvas.getContext('2d'); this.onFrame = onFrame || null;
      this.simulation = null; this.index = 0;
    }

    load(payload) {
      this.simulation = validateSimulation(payload); this.points = localPositions(this.simulation.timeline);
      this.index = 0; this.render(); return this.simulation;
    }

    setFrame(index) {
      if (!this.simulation) throw new Error('Load a compiled mission simulation first.');
      this.index = Math.max(0, Math.min(this.points.length - 1, Number(index) || 0));
      this.render(); return this.simulation.timeline[this.index];
    }

    render() {
      if (!this.simulation) return;
      const canvas = this.canvas, ctx = this.context;
      const width = Math.max(1, canvas.clientWidth || canvas.width || 720);
      const height = Math.max(1, canvas.clientHeight || canvas.height || 420);
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#071018'; ctx.fillRect(0, 0, width, height);
      const xs = this.points.map((point) => point.x), ys = this.points.map((point) => point.y), zs = this.points.map((point) => point.z);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys), minZ = Math.min(...zs);
      const span = Math.max(1, maxX - minX, maxY - minY); const scale = Math.min(width, height) * 0.68 / span;
      const project = (point) => [
        width / 2 + (point.x - (minX + maxX) / 2 - point.y * 0.28) * scale,
        height * 0.72 - (point.y - (minY + maxY) / 2) * scale * 0.48 - (point.z - minZ) * scale * 0.36
      ];
      // No ground polygon is drawn when terrain is unavailable. A blank background is
      // honest; a flat plane would look like surveyed ground.
      ctx.lineWidth = 2; ctx.strokeStyle = '#36d1dc'; ctx.beginPath();
      this.points.forEach((point, index) => { const p = project(point); if (!index) ctx.moveTo(...p); else ctx.lineTo(...p); }); ctx.stroke();
      ctx.fillStyle = '#f7c948'; this.points.filter((point) => point.row.capture).forEach((point) => { const p = project(point); ctx.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3); });
      const active = project(this.points[this.index]); ctx.fillStyle = '#ff5c77'; ctx.beginPath(); ctx.arc(active[0], active[1], 6, 0, Math.PI * 2); ctx.fill();
      const frame = this.simulation.timeline[this.index];
      if (this.onFrame) this.onFrame(frame, this.simulation);
    }
  }

  const exported = { validateSimulation, localPositions, MissionSimulator };
  global.ODKHubMissions = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
