/*
 * The Digital Twin's 3D view: the project's own mesh or point cloud, orbitable, with
 * points picked on it for measuring, annotating, cutting and saving views.
 *
 * "Textured Mesh" and "Point Cloud" used to show a 2D raster, and Save View, Clip and
 * Measure in 3D each answered "the 3D viewer is not built" -- while the reconstruction,
 * the saved-view store and the clip store were all on disk and tested. This is that
 * viewer. It draws what model_scene returns and nothing else; every number it shows
 * (a length, an area, a coordinate in the CRS) is computed by the application from the
 * picked points, and a refusal from the application is shown as one.
 *
 * Coordinates: the scene arrives relative to `scene.center` in the model FILE's frame,
 * which is the frame saved views and clips are stored in. Everything handed back to the
 * application is converted back to that frame first.
 *
 * No 3D library: WebGL directly, so the viewer works offline with nothing vendored.
 */

import { call } from "./api.js";

const state = {
  viewer: null,
  refusal: "",
  pendingTool: null,
  measurements: [],
  hooks: {
    toast: () => {},
    prompt: async () => null,
    changed: () => {},
  },
};

/* The tools, what they collect, and what the operator is told. */
const TOOLS = {
  orbit: { picks: 0, hint: "Drag to orbit, right-drag or Shift-drag to pan, wheel to zoom." },
  select: { picks: 1, hint: "Click a point on the model to read its coordinates." },
  measure: { picks: 2, hint: "Click two points on the model to measure between them." },
  distance: { picks: 2, hint: "Click two points on the model to measure between them." },
  height: { picks: 2, hint: "Click two points on the model; the vertical difference is measured." },
  area: { picks: 0, hint: "Click the corners on the model. Double-click or Enter to finish." },
  annotate: { picks: 1, hint: "Click the spot on the model to pin a finding to." },
  clip: { picks: 1, hint: "Click a point on the model. Everything between the camera and that point is cut away." },
};

/* Canvas buttons are labelled for a person; these are the tools they arm. */
const TOOL_BY_TITLE = {
  Orbit: "orbit", Zoom: "orbit", Select: "select", Measure: "measure", Annotate: "annotate",
};

export function configureViewer(hooks) {
  Object.assign(state.hooks, hooks);
}

/** The viewer on screen, or null. */
export function currentViewer() {
  const v = state.viewer;
  return v && v.canvas.isConnected ? v : null;
}

/** Measurements taken in 3D this session, newest first. */
export function modelMeasurements() {
  return state.measurements.slice();
}

/** Arm a tool on the viewer on screen; remembered for the next one if there is none. */
export function armModelTool(tool) {
  if (!TOOLS[tool]) return null;
  const viewer = currentViewer();
  if (!viewer) {
    state.pendingTool = tool;
    return null;
  }
  viewer.setTool(tool);
  return TOOLS[tool].hint;
}

/** Ask for a tool to be armed as soon as a 3D view is mounted. */
export function queueModelTool(tool) {
  state.pendingTool = TOOLS[tool] ? tool : null;
}

/* ---------------------------------------------------------------- mounting */

/** The canvas buttons of a 3D canvas (Orbit, Select, Measure, Annotate, Fit). */
export function answerCanvasTool(title) {
  const viewer = currentViewer();
  if (!viewer) {
    return `there is no 3D model on screen${state.refusal ? `: ${state.refusal}` : "."}`;
  }
  if (title === "Fit") { viewer.fit(); return "Fitted the model to the view."; }
  const tool = TOOL_BY_TITLE[title];
  return tool ? armModelTool(tool) : null;
}

/** Load the project's `kind` ("mesh" or "pointcloud") into `root`. */
export async function mountViewer(root, kind, { fallback = null } = {}) {
  const holder = document.createElement("div");
  holder.className = "viewer3d";
  const message = document.createElement("div");
  message.className = "placeholder";
  message.innerHTML = `<div><div class="big">Loading the ${kind === "mesh" ? "mesh" : "point cloud"}…</div></div>`;
  root.querySelectorAll(".placeholder").forEach((node) => node.remove());
  root.appendChild(message);

  let answer;
  try {
    try {
      answer = await call("model_scene", kind);
    } catch (first) {
      // The Digital Twin opens on the mesh; a run that meshed nothing still has a cloud.
      if (!fallback) throw first;
      answer = await call("model_scene", fallback);
      kind = fallback;
    }
    state.refusal = "";
  } catch (error) {
    state.refusal = String(error.message || error);
    message.innerHTML = "";
    const box = document.createElement("div");
    box.innerHTML = `<div class="big"></div><div class="small"></div>`;
    box.querySelector(".big").textContent = `No ${kind === "mesh" ? "mesh" : "point cloud"} to show`;
    box.querySelector(".small").textContent = String(error.message || error);
    message.appendChild(box);
    return null;
  }
  if (!root.isConnected) return null;
  message.remove();

  const canvas = document.createElement("canvas");
  canvas.className = "viewer3d-gl";
  const overlay = document.createElement("canvas");
  overlay.className = "viewer3d-overlay";
  holder.append(canvas, overlay);
  root.appendChild(holder);

  let viewer;
  try {
    viewer = new ModelViewer(canvas, overlay, answer.scene);
  } catch (error) {
    holder.remove();
    const box = document.createElement("div");
    box.className = "placeholder";
    box.innerHTML = `<div><div class="big">The 3D view cannot start</div><div class="small"></div></div>`;
    box.querySelector(".small").textContent = String(error.message || error);
    root.appendChild(box);
    return null;
  }
  state.viewer = viewer;
  viewer.onChanged = () => state.hooks.changed();

  const chip = document.createElement("div");
  chip.className = "canvas-overlay bl";
  const text = document.createElement("span");
  text.className = "chip";
  text.dataset.role = "scene-summary";
  text.textContent = describeScene(answer.scene);
  chip.appendChild(text);
  root.appendChild(chip);

  if (answer.default_view) viewer.applyView(answer.default_view);
  if (state.pendingTool) {
    const tool = state.pendingTool;
    state.pendingTool = null;
    viewer.setTool(tool);
    state.hooks.toast(TOOLS[tool].hint, "ok");
  }
  return viewer;
}

function describeScene(scene) {
  const what = scene.kind === "mesh"
    ? `${scene.face_count.toLocaleString()} triangles`
    : `${scene.vertex_count.toLocaleString()} points`;
  const total = scene.kind === "mesh" ? scene.source_face_count : scene.source_vertex_count;
  const parts = [`${scene.source} — ${what}`];
  if (scene.reduction) parts.push(`reduced for display from ${total.toLocaleString()} (${scene.reduction})`);
  if (!scene.colors) parts.push("coloured by height");
  if ((scene.clips_applied || []).length) parts.push(`clipped: ${scene.clips_applied.join(", ")}`);
  parts.push(scene.georeferenced
    ? `EPSG:${scene.georeferenced.epsg}, georeference RMSE ${scene.georeferenced.rmse_m.toFixed(2)} m`
    : "not georeferenced — SfM units, measurements refused");
  return parts.join(" · ");
}

/* ------------------------------------------------------------------ maths */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0]);
}

function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let s = 0;
      for (let k = 0; k < 4; k += 1) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

function decode(b64, Type) {
  if (!b64) return null;
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Type(bytes.buffer);
}

/* ------------------------------------------------------------------ viewer */

export class ModelViewer {
  constructor(canvas, overlay, scene) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.scene = scene;
    const gl = canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true })
      || canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL is unavailable in this window, so the model cannot be drawn.");
    this.gl = gl;
    this.webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
    if (!this.webgl2 && !gl.getExtension("OES_element_index_uint") && scene.indices) {
      throw new Error("This WebGL cannot index more than 65,535 vertices; the mesh cannot be drawn.");
    }
    this.tool = "orbit";
    this.picks = [];
    this.marks = [];
    this.fov = 50;
    this.up = norm(scene.up || [0, 0, 1]);
    // Two axes perpendicular to up, for yaw.
    const seed = Math.abs(this.up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    this.axisA = norm(cross(this.up, seed));
    this.axisB = cross(this.up, this.axisA);
    this.load(scene);
    this.wire();
    this.fit();
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(canvas);
  }

  load(scene) {
    this.scene = scene;
    this.center = scene.center;
    this.positions = decode(scene.positions, Float32Array);
    const n = this.positions.length / 3;
    let colors = decode(scene.colors, Uint8Array);
    if (!colors) colors = heightColors(this.positions, this.up);
    this.indices = decode(scene.indices, Uint32Array);
    const gl = this.gl;
    this.program = this.program || buildProgram(gl, this.webgl2);
    const buffer = (data, target = gl.ARRAY_BUFFER) => {
      const b = gl.createBuffer();
      gl.bindBuffer(target, b);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return b;
    };
    this.buffers = {
      position: buffer(this.positions),
      color: buffer(colors),
      normal: buffer(this.indices ? vertexNormals(this.positions, this.indices) : new Float32Array(n * 3)),
      index: this.indices ? buffer(this.indices, gl.ELEMENT_ARRAY_BUFFER) : null,
    };
    this.count = this.indices ? this.indices.length : n;
  }

  /** Re-read the scene from the application, keeping the camera. */
  async reload() {
    const answer = await call("model_scene", this.scene.kind);
    const camera = this.cameraState();
    this.load(answer.scene);
    this.applyView(camera);
    const chip = this.canvas.parentElement && this.canvas.parentElement.parentElement
      && this.canvas.parentElement.parentElement.querySelector('[data-role="scene-summary"]');
    if (chip) chip.textContent = describeScene(answer.scene);
  }

  fit() {
    this.target = [0, 0, 0];
    this.distance = Math.max(1e-3, (this.scene.radius || 1) / Math.tan((this.fov * Math.PI) / 360) * 1.1);
    this.yaw = 0.6;
    this.pitch = 0.7;
    this.render();
  }

  /* -- camera -- */

  eye() {
    const c = Math.cos(this.pitch);
    const dir = add(add(scale(this.axisA, c * Math.cos(this.yaw)), scale(this.axisB, c * Math.sin(this.yaw))),
      scale(this.up, Math.sin(this.pitch)));
    return add(this.target, scale(dir, this.distance));
  }

  matrices() {
    const width = this.canvas.clientWidth || 640;
    const height = this.canvas.clientHeight || 400;
    const r = this.scene.radius || 1;
    const near = Math.max(this.distance - r * 2, this.distance * 0.01, 1e-4);
    const far = this.distance + r * 3;
    const view = lookAt(this.eye(), this.target, this.up);
    const proj = perspective((this.fov * Math.PI) / 180, width / height, near, far);
    return { mvp: multiply(proj, view), width, height };
  }

  /** Camera in the model file's frame: what save_view stores. */
  cameraState() {
    return {
      position: add(this.eye(), this.center),
      target: add(this.target, this.center),
      fov_deg: this.fov,
    };
  }

  /** Open a saved view (file-frame position and target). */
  applyView(view) {
    const target = sub(view.target, this.center);
    const offset = sub(sub(view.position, this.center), target);
    const distance = Math.hypot(offset[0], offset[1], offset[2]);
    if (!(distance > 0)) return;
    const d = scale(offset, 1 / distance);
    this.target = target;
    this.distance = distance;
    this.pitch = Math.asin(Math.max(-1, Math.min(1, dot(d, this.up))));
    this.yaw = Math.atan2(dot(d, this.axisB), dot(d, this.axisA));
    if (view.fov_deg) this.fov = Number(view.fov_deg);
    this.render();
  }

  /** Unit vector the camera looks along, in the file frame (frame rotation is none). */
  viewDirection() {
    return norm(sub(this.target, this.eye()));
  }

  /* -- drawing -- */

  render() {
    const gl = this.gl;
    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round((this.canvas.clientWidth || 640) * pixelRatio));
    const height = Math.max(1, Math.round((this.canvas.clientHeight || 400) * pixelRatio));
    for (const c of [this.canvas, this.overlay]) {
      if (c.width !== width || c.height !== height) { c.width = width; c.height = height; }
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.05, 0.07, 0.1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const p = this.program;
    gl.useProgram(p.program);
    const attr = (name, b, size, type, normalized) => {
      const loc = p.attribs[name];
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
    };
    attr("a_position", this.buffers.position, 3, gl.FLOAT, false);
    attr("a_color", this.buffers.color, 3, gl.UNSIGNED_BYTE, true);
    attr("a_normal", this.buffers.normal, 3, gl.FLOAT, false);
    const { mvp } = this.matrices();
    gl.uniformMatrix4fv(p.uniforms.u_mvp, false, mvp);
    gl.uniform3fv(p.uniforms.u_light, new Float32Array(scale(this.viewDirection(), -1)));
    gl.uniform1f(p.uniforms.u_lit, this.indices ? 1 : 0);
    gl.uniform1f(p.uniforms.u_size, Math.max(1.5, 2 * pixelRatio));
    if (this.indices) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
    } else {
      gl.drawArrays(gl.POINTS, 0, this.count);
    }
    this.drawOverlay();
  }

  project(local, mvp, width, height) {
    const [x, y, z] = local;
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0) return null;
    return [(cx / cw * 0.5 + 0.5) * width, (1 - (cy / cw * 0.5 + 0.5)) * height, cw];
  }

  drawOverlay() {
    const ctx = this.overlay.getContext("2d");
    const pixelRatio = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const { mvp, width, height } = this.matrices();
    const shapes = [...this.marks, ...(this.picks.length ? [{ points: this.picks, live: true }] : [])];
    for (const shape of shapes) {
      const screen = shape.points.map((pt) => this.project(pt.local, mvp, width, height)).filter(Boolean);
      ctx.strokeStyle = shape.live ? "#ffd166" : "#4cc9f0";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2;
      if (screen.length > 1) {
        ctx.beginPath();
        screen.forEach(([sx, sy], i) => (i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
        if (shape.closed) ctx.closePath();
        ctx.stroke();
      }
      for (const [sx, sy] of screen) { ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill(); }
      if (shape.label && screen.length) {
        const [lx, ly] = screen[screen.length - 1];
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(shape.label, lx + 8, ly - 8);
      }
    }
  }

  /* -- picking -- */

  /** The model vertex nearest the camera within a few pixels of (x, y), or null. */
  pick(x, y) {
    const { mvp, width, height } = this.matrices();
    const pos = this.positions;
    const radius = 8;
    let best = -1;
    let bestDepth = Infinity;
    let bestDist = Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      const s = this.project([pos[i], pos[i + 1], pos[i + 2]], mvp, width, height);
      if (!s) continue;
      const d = Math.hypot(s[0] - x, s[1] - y);
      if (d > radius) continue;
      // Nearest the camera first, so a click lands on the front surface, not through it.
      if (s[2] < bestDepth - 1e-6 || (Math.abs(s[2] - bestDepth) <= 1e-6 && d < bestDist)) {
        best = i; bestDepth = s[2]; bestDist = d;
      }
    }
    if (best < 0) return null;
    const local = [pos[best], pos[best + 1], pos[best + 2]];
    return { local, file: add(local, this.center) };
  }

  setTool(tool) {
    this.tool = TOOLS[tool] ? tool : "orbit";
    this.picks = [];
    this.overlay.style.cursor = this.tool === "orbit" ? "grab" : "crosshair";
    this.render();
  }

  wire() {
    const el = this.overlay;
    let drag = null;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, moved: false, pan: e.button === 2 || e.shiftKey };
      try { el.setPointerCapture(e.pointerId); } catch (_) { /* synthetic events */ }
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const eye = this.eye();
        const forward = norm(sub(this.target, eye));
        const right = norm(cross(forward, this.up));
        const upScreen = cross(right, forward);
        const perPixel = (2 * this.distance * Math.tan((this.fov * Math.PI) / 360)) / (this.canvas.clientHeight || 400);
        this.target = add(this.target, add(scale(right, -dx * perPixel), scale(upScreen, dy * perPixel)));
      } else {
        this.yaw -= dx * 0.008;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.008));
      }
      this.render();
    });
    el.addEventListener("pointerup", (e) => {
      const was = drag;
      drag = null;
      if (was && !was.moved && e.button === 0) this.click(e);
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.distance = Math.max(1e-4, this.distance * Math.exp(e.deltaY * 0.001));
      this.render();
    }, { passive: false });
    el.addEventListener("dblclick", (e) => { e.preventDefault(); if (this.tool === "area") this.finish(); });
    el.tabIndex = 0;
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.finish();
      if (e.key === "Escape") { this.picks = []; this.setTool("orbit"); }
    });
  }

  /** A click at client coordinates, as an operator's would land. */
  async click(event) {
    if (this.tool === "orbit") return;
    const rect = this.overlay.getBoundingClientRect();
    const hit = this.pick(event.clientX - rect.left, event.clientY - rect.top);
    if (!hit) {
      state.hooks.toast("That click is not on the model. Click on a visible point or surface.", "warn");
      return;
    }
    this.picks.push(hit);
    this.render();
    const need = TOOLS[this.tool].picks;
    if (need && this.picks.length >= need) await this.finish();
  }

  /** Complete what the armed tool collects, in the application. */
  async finish() {
    const tool = this.tool;
    const picks = this.picks;
    this.picks = [];
    const say = state.hooks.toast;
    try {
      if (tool === "select") {
        const p = picks[0].file;
        say(`Model point ${p.map((v) => v.toFixed(3)).join(", ")} (model file coordinates).`, "ok");
      } else if (["measure", "distance", "height", "area"].includes(tool)) {
        if (tool === "area" && picks.length < 3) {
          say(`Area: needs at least 3 points; ${picks.length} clicked. Nothing was measured.`, "warn");
          return;
        }
        const kind = tool === "area" ? "area" : tool === "height" ? "height" : "length";
        const { measurement: m } = await call("measure_on_model", kind, picks.map((p) => p.file));
        const text = kind === "area"
          ? `Area ${m.surface_area_m2.toFixed(2)} m² on the surface, ${m.planimetric_area_m2.toFixed(2)} m² in plan`
          : kind === "height"
            ? `Height ${Math.abs(m.vertical_distance_m).toFixed(2)} m`
            : `Length ${m.slope_distance_m.toFixed(2)} m (horizontal ${m.horizontal_distance_m.toFixed(2)} m, vertical ${m.vertical_distance_m.toFixed(2)} m)`;
        const full = `${text}; georeference RMSE ${m.georeference_rmse_m} m`;
        this.marks.push({ points: picks, closed: kind === "area", label: text.split(" (")[0] });
        state.measurements.unshift({ kind, text: full, measurement: m });
        say(`${full}.`, "ok");
        if (this.onChanged) this.onChanged();
      } else if (tool === "annotate") {
        const label = await state.hooks.prompt("Finding at this point", "Crack");
        if (!label) { say("Annotate: no label given; nothing was saved.", "warn"); return; }
        const { annotation } = await call("annotate_model", picks[0].file, label, "medium", "", this.scene.source);
        this.marks.push({ points: picks, label: annotation.label });
        say(`Pinned "${annotation.label}" to the model; it is listed with the project's findings.`, "ok");
        if (this.onChanged) this.onChanged();
      } else if (tool === "clip") {
        const name = await state.hooks.prompt("Name this cut", "section");
        if (!name) { say("Clip: no name given; the model was not cut.", "warn"); return; }
        // The plane faces away from the camera through the picked point: what lies
        // between the camera and the point is removed, which is what a section is.
        const n = this.viewDirection();
        const heading = (Math.atan2(n[1], n[0]) * 180) / Math.PI;
        const pitch = (-Math.asin(Math.max(-1, Math.min(1, n[2]))) * 180) / Math.PI;
        await call("add_plane_clip", name, picks[0].file, heading, pitch, 0);
        this.setTool("orbit");
        await this.reload();
        say(`Cut "${name}" applied. Clip lists, hides or removes cuts.`, "ok");
        if (this.onChanged) this.onChanged();
      }
    } catch (error) {
      say(`${tool[0].toUpperCase()}${tool.slice(1)}: ${error.message || error}`, "error");
    } finally {
      this.render();
    }
  }
}

/* ------------------------------------------------------------------ GL setup */

function buildProgram(gl, webgl2) {
  const vs = webgl2 ? `#version 300 es
    in vec3 a_position; in vec3 a_color; in vec3 a_normal;
    uniform mat4 u_mvp; uniform vec3 u_light; uniform float u_lit; uniform float u_size;
    out vec3 v_color;
    void main() {
      gl_Position = u_mvp * vec4(a_position, 1.0);
      gl_PointSize = u_size;
      float shade = u_lit > 0.5 ? 0.35 + 0.65 * abs(dot(normalize(a_normal + 1e-6), u_light)) : 1.0;
      v_color = a_color * shade;
    }` : `
    attribute vec3 a_position; attribute vec3 a_color; attribute vec3 a_normal;
    uniform mat4 u_mvp; uniform vec3 u_light; uniform float u_lit; uniform float u_size;
    varying vec3 v_color;
    void main() {
      gl_Position = u_mvp * vec4(a_position, 1.0);
      gl_PointSize = u_size;
      float shade = u_lit > 0.5 ? 0.35 + 0.65 * abs(dot(normalize(a_normal + 1e-6), u_light)) : 1.0;
      v_color = a_color * shade;
    }`;
  const fs = webgl2 ? `#version 300 es
    precision mediump float; in vec3 v_color; out vec4 color;
    void main() { color = vec4(v_color, 1.0); }` : `
    precision mediump float; varying vec3 v_color;
    void main() { gl_FragColor = vec4(v_color, 1.0); }`;
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const attribs = {};
  for (const name of ["a_position", "a_color", "a_normal"]) attribs[name] = gl.getAttribLocation(program, name);
  const uniforms = {};
  for (const name of ["u_mvp", "u_light", "u_lit", "u_size"]) uniforms[name] = gl.getUniformLocation(program, name);
  return { program, attribs, uniforms };
}

function vertexNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3; const b = indices[i + 1] * 3; const c = indices[i + 2] * 3;
    const e1 = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const e2 = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    const n = cross(e1, e2);
    for (const v of [a, b, c]) { normals[v] += n[0]; normals[v + 1] += n[1]; normals[v + 2] += n[2]; }
  }
  return normals;
}

/* A model with no colours of its own is shaded by height, and the chip says so. */
function heightColors(positions, up) {
  const n = positions.length / 3;
  const heights = new Float32Array(n);
  let lo = Infinity; let hi = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const h = positions[i * 3] * up[0] + positions[i * 3 + 1] * up[1] + positions[i * 3 + 2] * up[2];
    heights[i] = h; if (h < lo) lo = h; if (h > hi) hi = h;
  }
  const span = hi - lo || 1;
  const colors = new Uint8Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const t = (heights[i] - lo) / span;
    colors[i * 3] = Math.round(40 + 200 * t);
    colors[i * 3 + 1] = Math.round(120 + 80 * (1 - Math.abs(t - 0.5) * 2));
    colors[i * 3 + 2] = Math.round(220 - 180 * t);
  }
  return colors;
}
