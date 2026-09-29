/*
 * The 3D model view: the project's reconstructed mesh, in its CRS, on a WebGL canvas.
 *
 * Three things were implemented in the Api and unreachable because the cockpit had no
 * 3D camera: measuring inside the model, saved views, and seeing the mesh at all. This
 * is the smallest viewer that gives them something real to act on -- orbit, pan, zoom,
 * a click that lands on the surface, and a camera that can be written down and put back.
 *
 * Picking is done by the application on the full-resolution mesh, not here: the browser
 * holds a thinned copy for display, and a point picked on that would be off by whatever
 * the thinning removed. The view only turns a click into a ray.
 *
 * Coordinates arrive relative to `scene.origin` because UTM eastings do not survive a
 * float32 buffer; every position this module hands back is absolute CRS metres.
 */

import { call } from "./api.js";
import { el } from "./primitives.js";

let active = null;
const waiting = new Set();

/** The viewer currently on screen, or null. */
export function currentModelViewer() {
  return active && active.canvas.isConnected ? active : null;
}

/** Resolves with the viewer once a model is loaded (rejects with why it cannot be). */
export function whenModelReady(timeoutMs = 20000) {
  const viewer = currentModelViewer();
  if (viewer && viewer.ready) return Promise.resolve(viewer);
  if (viewer && viewer.failure) return Promise.reject(new Error(viewer.failure));
  return new Promise((resolve, reject) => {
    const entry = { resolve, reject };
    waiting.add(entry);
    setTimeout(() => {
      if (waiting.delete(entry)) reject(new Error("The 3D model did not load. Open the Mesh view first."));
    }, timeoutMs);
  });
}

function settleWaiting(viewer) {
  for (const entry of [...waiting]) {
    waiting.delete(entry);
    if (viewer.ready) entry.resolve(viewer);
    else entry.reject(new Error(viewer.failure || "The 3D model could not be shown."));
  }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

const VERTEX = `
attribute vec3 a_position; attribute vec3 a_normal;
uniform mat4 u_mvp; uniform float u_zmin; uniform float u_zspan;
varying float v_light; varying float v_height;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
  v_light = 0.35 + 0.65 * abs(dot(normalize(a_normal), normalize(vec3(0.4, -0.3, 0.85))));
  v_height = clamp((a_position.z - u_zmin) / max(u_zspan, 1e-6), 0.0, 1.0);
}`;
const FRAGMENT = `
precision mediump float;
varying float v_light; varying float v_height;
void main() {
  vec3 low = vec3(0.30, 0.42, 0.55); vec3 high = vec3(0.86, 0.82, 0.70);
  gl_FragColor = vec4(mix(low, high, v_height) * v_light, 1.0);
}`;

class ModelViewer {
  constructor(root) {
    this.root = root;
    this.canvas = el("canvas", { class: "canvas-product model3d" });
    this.canvas.style.cursor = "grab";
    this.canvas.style.touchAction = "none";
    root.appendChild(this.canvas);
    this.caption = el("span", { class: "chip", text: "Loading the 3D model…" });
    root.appendChild(el("div", { class: "canvas-overlay bl" }, [this.caption]));
    this.pinLayer = el("div", { class: "model3d-pins" });
    Object.assign(this.pinLayer.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: 4 });
    root.appendChild(this.pinLayer);
    this.pins = [];
    this.fov = 50;
    this.target = [0, 0, 0];
    this.yaw = Math.PI / 6;
    this.pitch = 0.6;
    this.distance = 100;
    this.ready = false;
    this.failure = "";
    this.pickers = [];
  }

  async load() {
    let scene;
    try {
      scene = await call("model_scene");
    } catch (error) {
      this.fail(error.message || String(error));
      return;
    }
    const gl = typeof this.canvas.getContext === "function"
      ? this.canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true }) : null;
    if (!gl) { this.fail("WebGL is not available, so the model cannot be drawn."); return; }
    const wide = gl.getExtension("OES_element_index_uint");
    if (!wide && scene.positions.length / 3 > 65535) {
      this.fail("This graphics driver cannot index a mesh this large."); return;
    }
    this.gl = gl;
    this.scene = scene;
    this.program = this.compile(gl);
    const positions = new Float32Array(scene.positions);
    const indices = wide ? new Uint32Array(scene.indices) : new Uint16Array(scene.indices);
    const normals = new Float32Array(positions.length);
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3];
      const pa = [positions[a], positions[a + 1], positions[a + 2]];
      const n = cross(sub([positions[b], positions[b + 1], positions[b + 2]], pa),
        sub([positions[c], positions[c + 1], positions[c + 2]], pa));
      for (const k of [a, b, c]) { normals[k] += n[0]; normals[k + 1] += n[1]; normals[k + 2] += n[2]; }
    }
    this.buffers = {
      position: this.upload(gl.ARRAY_BUFFER, positions),
      normal: this.upload(gl.ARRAY_BUFFER, normals),
      index: this.upload(gl.ELEMENT_ARRAY_BUFFER, indices),
    };
    this.indexType = wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    this.count = indices.length;
    let zmin = Infinity; let zmax = -Infinity; let reach = 0;
    for (let i = 0; i < positions.length; i += 3) {
      zmin = Math.min(zmin, positions[i + 2]); zmax = Math.max(zmax, positions[i + 2]);
      reach = Math.max(reach, Math.hypot(positions[i], positions[i + 1], positions[i + 2]));
    }
    this.zmin = zmin; this.zspan = zmax - zmin;
    this.distance = Math.max(1, reach * 2.2);
    this.wire();
    this.ready = true;
    const thinned = scene.shown_face_count < scene.face_count
      ? ` (${scene.shown_face_count.toLocaleString()} of ${scene.face_count.toLocaleString()} faces drawn; picks use all)`
      : "";
    this.caption.textContent = `Mesh — EPSG:${scene.epsg}${thinned}`;
    this.render();
    settleWaiting(this);
  }

  fail(message) {
    this.failure = message;
    this.caption.textContent = message;
    this.caption.className = "chip warn";
    settleWaiting(this);
  }

  compile(gl) {
    const shader = (type, source) => {
      const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s); return s;
    };
    const program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    return program;
  }

  upload(kind, data) {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    gl.bindBuffer(kind, buffer);
    gl.bufferData(kind, data, gl.STATIC_DRAW);
    return buffer;
  }

  /* -- camera ------------------------------------------------------------ */

  eye() {
    const c = Math.cos(this.pitch);
    return add(this.target, scale([c * Math.sin(this.yaw), -c * Math.cos(this.yaw), Math.sin(this.pitch)], this.distance));
  }

  basis() {
    const eye = this.eye();
    const forward = norm(sub(this.target, eye));
    let right = cross(forward, [0, 0, 1]);
    if (Math.hypot(...right) < 1e-6) right = [1, 0, 0];
    right = norm(right);
    const up = cross(right, forward);
    return { eye, forward, right, up };
  }

  size() {
    const box = this.canvas.getBoundingClientRect();
    return { box, width: Math.max(1, box.width), height: Math.max(1, box.height) };
  }

  matrix() {
    const { eye, forward, right, up } = this.basis();
    const { width, height } = this.size();
    const f = 1 / Math.tan((this.fov * Math.PI) / 360);
    const aspect = width / height;
    const near = Math.max(0.01, this.distance / 1000);
    const far = this.distance * 20 + 1000;
    const view = [
      right[0], up[0], -forward[0], 0,
      right[1], up[1], -forward[1], 0,
      right[2], up[2], -forward[2], 0,
      -dot(right, eye), -dot(up, eye), dot(forward, eye), 1,
    ];
    const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0];
    const out = new Array(16).fill(0);
    for (let c = 0; c < 4; c += 1) for (let r = 0; r < 4; r += 1) {
      for (let k = 0; k < 4; k += 1) out[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];
    }
    return out;
  }

  render() {
    if (!this.ready) return;
    const gl = this.gl;
    const { width, height } = this.size();
    const ratio = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * ratio)) this.canvas.width = Math.round(width * ratio);
    if (this.canvas.height !== Math.round(height * ratio)) this.canvas.height = Math.round(height * ratio);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.05, 0.063, 0.082, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(this.program);
    const bind = (name, buffer) => {
      const at = gl.getAttribLocation(this.program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(at);
      gl.vertexAttribPointer(at, 3, gl.FLOAT, false, 0, 0);
    };
    bind("a_position", this.buffers.position);
    bind("a_normal", this.buffers.normal);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "u_mvp"), false, new Float32Array(this.matrix()));
    gl.uniform1f(gl.getUniformLocation(this.program, "u_zmin"), this.zmin);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_zspan"), this.zspan);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    gl.drawElements(gl.TRIANGLES, this.count, this.indexType, 0);
    this.drawPins();
  }

  /** The camera, in absolute CRS metres, as save_view stores it. */
  getCamera() {
    const origin = this.scene.origin;
    return { position: add(this.eye(), origin), target: add(this.target, origin), fov_deg: this.fov };
  }

  /** Put the camera back where a saved view says it was. */
  setCamera(position, target, fovDeg) {
    const origin = this.scene.origin;
    this.target = sub(target, origin);
    const offset = sub(sub(position, origin), this.target);
    this.distance = Math.max(0.01, Math.hypot(...offset));
    this.pitch = Math.asin(Math.max(-1, Math.min(1, offset[2] / this.distance)));
    this.yaw = Math.atan2(offset[0], -offset[1]);
    if (Number.isFinite(Number(fovDeg))) this.fov = Number(fovDeg);
    this.render();
  }

  /** Screen position of an absolute CRS point, or null when it is behind the camera. */
  project(point) {
    const { eye, forward, right, up } = this.basis();
    const rel = sub(sub(point, this.scene.origin), eye);
    const z = dot(rel, forward);
    if (z <= 0) return null;
    const { box, width, height } = this.size();
    const t = Math.tan((this.fov * Math.PI) / 360);
    const nx = dot(rel, right) / (z * t * (width / height));
    const ny = dot(rel, up) / (z * t);
    return { x: box.left + ((nx + 1) / 2) * width, y: box.top + ((1 - ny) / 2) * height };
  }

  /** The ray under a screen position, in absolute CRS metres. */
  ray(clientX, clientY) {
    const { eye, forward, right, up } = this.basis();
    const { box, width, height } = this.size();
    const t = Math.tan((this.fov * Math.PI) / 360);
    const nx = ((clientX - box.left) / width) * 2 - 1;
    const ny = 1 - ((clientY - box.top) / height) * 2;
    const direction = norm(add(forward, add(scale(right, nx * t * (width / height)), scale(up, ny * t))));
    return { origin: add(eye, this.scene.origin), direction };
  }

  /* -- interaction --------------------------------------------------------- */

  wire() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", (event) => {
      drag = { x: event.clientX, y: event.clientY, x0: event.clientX, y0: event.clientY, pan: event.shiftKey || event.button === 2 };
      if (this.canvas.setPointerCapture && event.pointerId !== undefined) {
        try { this.canvas.setPointerCapture(event.pointerId); } catch (_) { /* synthetic */ }
      }
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x; const dy = event.clientY - drag.y;
      drag.x = event.clientX; drag.y = event.clientY;
      if (drag.pan) {
        const { right, up } = this.basis();
        const k = this.distance / Math.max(200, this.size().height);
        this.target = add(this.target, add(scale(right, -dx * k), scale(up, dy * k)));
      } else {
        this.yaw -= dx * 0.008;
        this.pitch = Math.max(-1.45, Math.min(1.5, this.pitch + dy * 0.008));
      }
      this.render();
    });
    this.canvas.addEventListener("pointerup", () => { drag = null; });
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.max(0.05, this.distance * Math.exp(event.deltaY * 0.001));
      this.render();
    }, { passive: false });
    // A click that did not move is a pick when an action is waiting for one.
    this.canvas.addEventListener("click", async (event) => {
      const picker = this.pickers.shift();
      if (!picker) return;
      const { origin, direction } = this.ray(event.clientX, event.clientY);
      try {
        const hit = await call("pick_model_point", this.scene.model_path, origin, direction);
        this.pins.push(hit.point);
        this.drawPins();
        picker.resolve(hit.point);
      } catch (error) {
        picker.reject(error);
      }
    });
  }

  /** Resolve with the CRS point under the next click on the model. */
  pickNext() {
    return new Promise((resolve, reject) => this.pickers.push({ resolve, reject }));
  }

  clearPins() { this.pins = []; this.drawPins(); }

  drawPins() {
    this.pinLayer.innerHTML = "";
    const box = this.root.getBoundingClientRect();
    for (const pin of this.pins) {
      const at = this.project(pin);
      if (!at) continue;
      const dot_ = el("div");
      Object.assign(dot_.style, { position: "absolute", width: "8px", height: "8px", borderRadius: "50%",
        background: "#f59e0b", border: "1px solid #0d1015",
        left: `${at.x - box.left - 4}px`, top: `${at.y - box.top - 4}px` });
      this.pinLayer.appendChild(dot_);
    }
  }
}

/** Put the project's model on a canvas root. */
export function mountModel(root) {
  root.querySelectorAll(".placeholder").forEach((node) => node.remove());
  const viewer = new ModelViewer(root);
  active = viewer;
  viewer.load().catch((error) => viewer.fail(error.message || String(error)));
  return viewer;
}
