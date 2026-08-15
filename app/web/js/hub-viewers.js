/* Local-first Hub viewer primitives: provider normalization, 3D and point streams. */
(function (global) {
  'use strict';

  function finiteNumber(value, label) {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${label} must be finite.`);
    return number;
  }

  function normaliseProvider(provider) {
    const kind = String(provider.kind || 'xyz').toLowerCase();
    if (!['xyz', 'wms', 'wmts'].includes(kind)) throw new Error('Map provider kind must be xyz, wms, or wmts.');
    const url = String(provider.url || '').trim();
    if (!/^https?:\/\//i.test(url) && !url.startsWith('/')) throw new Error('Map provider URL must be HTTP(S) or root-relative.');
    const maxzoom = Math.max(0, Math.min(24, Number(provider.maxzoom || 19)));
    return {
      kind,
      name: String(provider.name || kind.toUpperCase()),
      url,
      layers: String(provider.layers || ''),
      format: String(provider.format || 'image/png'),
      attribution: String(provider.attribution || ''),
      maxzoom
    };
  }

  function providerTileUrl(input) {
    const provider = normaliseProvider(input);
    if (provider.kind === 'xyz') return provider.url;
    if (provider.kind === 'wmts') {
      return provider.url
        .replace(/\{TileMatrix\}/gi, '{z}')
        .replace(/\{TileCol\}/gi, '{x}')
        .replace(/\{TileRow\}/gi, '{y}');
    }
    const separator = provider.url.includes('?') ? '&' : '?';
    const query = [
      'SERVICE=WMS', 'REQUEST=GetMap', 'VERSION=1.1.1',
      `LAYERS=${encodeURIComponent(provider.layers)}`,
      `FORMAT=${encodeURIComponent(provider.format)}`,
      'TRANSPARENT=TRUE', 'SRS=EPSG:3857', 'WIDTH=256', 'HEIGHT=256',
      'BBOX={bbox-epsg-3857}'
    ].join('&');
    return `${provider.url}${separator}${query}`;
  }

  function mapLibreRasterSource(provider) {
    const normalized = normaliseProvider(provider);
    return {
      type: 'raster',
      tiles: [providerTileUrl(normalized)],
      tileSize: 256,
      maxzoom: normalized.maxzoom,
      attribution: normalized.attribution
    };
  }

  function parsePointChunk(payload) {
    if (!payload || typeof payload !== 'object') throw new Error('Point chunk must be a JSON object.');
    let positions = [];
    let colors = [];
    if (Array.isArray(payload.positions)) {
      positions = payload.positions.map((value, index) => finiteNumber(value, `positions[${index}]`));
      colors = Array.isArray(payload.colors)
        ? payload.colors.map((value, index) => finiteNumber(value, `colors[${index}]`))
        : [];
    } else if (Array.isArray(payload.points)) {
      payload.points.forEach((point, index) => {
        if (!Array.isArray(point) || point.length < 3) throw new Error(`points[${index}] needs x, y, z.`);
        positions.push(
          finiteNumber(point[0], `points[${index}].x`),
          finiteNumber(point[1], `points[${index}].y`),
          finiteNumber(point[2], `points[${index}].z`)
        );
        if (point.length >= 6) colors.push(
          finiteNumber(point[3], `points[${index}].r`) / 255,
          finiteNumber(point[4], `points[${index}].g`) / 255,
          finiteNumber(point[5], `points[${index}].b`) / 255
        );
      });
    } else {
      throw new Error('Point chunk needs positions or points.');
    }
    if (positions.length % 3 !== 0) throw new Error('Flat positions length must be divisible by three.');
    const count = positions.length / 3;
    if (colors.length && colors.length !== positions.length) throw new Error('Colors must contain one RGB triplet per point.');
    if (!colors.length) {
      for (let index = 0; index < count; index += 1) colors.push(0.35, 0.75, 1.0);
    }
    return { positions, colors, count };
  }

  function parsePointManifest(payload) {
    if (!payload || payload.type !== 'odk-point-cloud-manifest') {
      throw new Error('Point cloud manifest type must be odk-point-cloud-manifest.');
    }
    if (!Array.isArray(payload.chunks) || !payload.chunks.length) {
      throw new Error('Point cloud manifest needs at least one chunk URL.');
    }
    return {
      type: payload.type,
      crs_epsg: payload.crs_epsg == null ? null : Number(payload.crs_epsg),
      units: String(payload.units || 'unknown'),
      chunks: payload.chunks.map((item) => typeof item === 'string' ? { url: item } : { ...item }),
      point_count: payload.point_count == null ? null : Number(payload.point_count)
    };
  }

  function parseScene(payload) {
    if (!payload || payload.type !== 'odk-scene-3d') throw new Error('3D scene type must be odk-scene-3d.');
    const geometry = parsePointChunk({ positions: payload.vertices, colors: payload.colors });
    const indices = Array.isArray(payload.indices)
      ? payload.indices.map((value, index) => {
        const number = finiteNumber(value, `indices[${index}]`);
        if (number < 0 || !Number.isInteger(number)) throw new Error('Scene indices must be non-negative integers.');
        return number;
      })
      : [];
    return {
      type: payload.type,
      positions: geometry.positions,
      colors: geometry.colors,
      indices,
      primitive: indices.length ? 'triangles' : 'points',
      units: String(payload.units || 'unknown'),
      crs_epsg: payload.crs_epsg == null ? null : Number(payload.crs_epsg),
      coordinate_frame: String(payload.coordinate_frame || 'unknown'),
      origin_xy: Array.isArray(payload.origin_xy) && payload.origin_xy.length === 2
        ? payload.origin_xy.map((value, index) => finiteNumber(value, `origin_xy[${index}]`))
        : null,
      overlays: Array.isArray(payload.overlays) ? payload.overlays : []
    };
  }

  function parseThermalMap(payload) {
    if (!payload || payload.type !== 'odk-thermal-map') {
      throw new Error('Thermal map type must be odk-thermal-map.');
    }
    if (Number(payload.schema_version) !== 1) throw new Error('Unsupported thermal map schema version.');
    const width = Number(payload.width), height = Number(payload.height);
    if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
      throw new Error('Thermal map width and height must be positive integers.');
    }
    const epsg = Number(payload.crs_epsg);
    if (!Number.isInteger(epsg) || epsg < 1) {
      throw new Error('Thermal map needs an explicit EPSG coordinate reference system.');
    }
    if (String(payload.unit || '').toLowerCase() !== 'celsius') {
      throw new Error('Thermal map unit must be celsius; palette indices are not temperatures.');
    }
    if (!Array.isArray(payload.transform) || payload.transform.length !== 6) {
      throw new Error('Thermal map needs a six-value affine transform.');
    }
    const transform = payload.transform.map((value, index) => finiteNumber(value, `transform[${index}]`));
    if (Math.abs(transform[0] * transform[4] - transform[1] * transform[3]) < 1e-15) {
      throw new Error('Thermal map affine transform is degenerate.');
    }
    if (!Array.isArray(payload.values) || payload.values.length !== width * height) {
      throw new Error('Thermal map needs exactly width x height temperature values.');
    }
    const values = payload.values.map((value, index) => {
      if (value == null) return NaN;
      return finiteNumber(value, `values[${index}]`);
    });
    const finite = values.filter(Number.isFinite);
    if (!finite.length) throw new Error('Thermal map contains no valid temperature measurements.');

    let registration = null;
    if (payload.registration != null) {
      const source = payload.registration;
      const rgbWidth = Number(source.rgb_width), rgbHeight = Number(source.rgb_height);
      const residual = Number(source.residual_px);
      if (source.validated !== true || String(source.target || '').toLowerCase() !== 'rgb') {
        throw new Error('RGB/thermal registration must be explicitly validated for an RGB target.');
      }
      if (!String(source.method || '').trim() || !String(source.validated_by || '').trim()) {
        throw new Error('RGB/thermal registration must name its method and validator.');
      }
      if (!Number.isInteger(rgbWidth) || rgbWidth < 1 || !Number.isInteger(rgbHeight) || rgbHeight < 1) {
        throw new Error('RGB/thermal registration needs positive RGB dimensions.');
      }
      if (!Number.isFinite(residual) || residual < 0) {
        throw new Error('RGB/thermal registration residual_px must be finite and non-negative.');
      }
      registration = {
        validated: true, target: 'rgb', method: String(source.method),
        validated_by: String(source.validated_by), rgb_width: rgbWidth,
        rgb_height: rgbHeight, residual_px: residual
      };
    }
    return {
      type: payload.type, schema_version: 1, width, height, crs_epsg: epsg,
      transform, unit: 'celsius', values,
      min_c: Math.min(...finite), max_c: Math.max(...finite),
      source: String(payload.source || ''),
      temperature_source: String(payload.temperature_source || 'unspecified'),
      interpolated: payload.interpolated === true,
      registration
    };
  }

  function thermalColorRgb(value, minimum, maximum) {
    if (!Number.isFinite(value)) return [0, 0, 0, 0];
    const span = Math.max(1e-12, maximum - minimum);
    const t = Math.max(0, Math.min(1, (value - minimum) / span));
    const stops = [
      [0.00, 18, 22, 88], [0.25, 32, 126, 184], [0.50, 56, 210, 168],
      [0.75, 246, 218, 72], [1.00, 215, 38, 56]
    ];
    for (let index = 1; index < stops.length; index += 1) {
      if (t <= stops[index][0]) {
        const left = stops[index - 1], right = stops[index];
        const mix = (t - left[0]) / (right[0] - left[0]);
        return [
          Math.round(left[1] + (right[1] - left[1]) * mix),
          Math.round(left[2] + (right[2] - left[2]) * mix),
          Math.round(left[3] + (right[3] - left[3]) * mix), 255
        ];
      }
    }
    return [215, 38, 56, 255];
  }

  function thermalImagePixels(input) {
    const map = input.type === 'odk-thermal-map' && Array.isArray(input.values) && input.min_c != null
      ? input : parseThermalMap(input);
    const data = [];
    map.values.forEach((value) => data.push(...thermalColorRgb(value, map.min_c, map.max_c)));
    return { width: map.width, height: map.height, data };
  }

  function sampleThermalMap(input, x, y) {
    const map = input.type === 'odk-thermal-map' && input.min_c != null ? input : parseThermalMap(input);
    const [a, b, c, d, e, f] = map.transform;
    const determinant = a * e - b * d;
    const dx = finiteNumber(x, 'sample x') - c, dy = finiteNumber(y, 'sample y') - f;
    const column = Math.floor((e * dx - b * dy) / determinant);
    const row = Math.floor((-d * dx + a * dy) / determinant);
    if (row < 0 || column < 0 || row >= map.height || column >= map.width) return null;
    const value = map.values[row * map.width + column];
    return Number.isFinite(value) ? value : null;
  }

  function projectThermalOntoScene(sceneInput, thermalInput) {
    const scene = Array.isArray(sceneInput.positions) ? sceneInput : parseScene(sceneInput);
    const thermal = thermalInput.min_c != null ? thermalInput : parseThermalMap(thermalInput);
    if (!Number.isInteger(scene.crs_epsg)) throw new Error('3D scene needs an explicit EPSG CRS for thermal projection.');
    if (scene.crs_epsg !== thermal.crs_epsg) throw new Error('3D scene and thermal map CRS values do not match.');
    let origin = [0, 0];
    if (scene.coordinate_frame === 'local_projected') {
      if (!scene.origin_xy) throw new Error('A local_projected 3D scene needs origin_xy.');
      origin = scene.origin_xy;
    } else if (scene.coordinate_frame !== 'projected') {
      throw new Error('3D scene coordinate_frame must be projected or local_projected.');
    }

    const colors = [], temperatures = [];
    let sampled = 0;
    for (let index = 0; index < scene.positions.length; index += 3) {
      const value = sampleThermalMap(
        thermal, scene.positions[index] + origin[0], scene.positions[index + 1] + origin[1]
      );
      temperatures.push(value);
      if (value == null) colors.push(0.18, 0.2, 0.24);
      else {
        const color = thermalColorRgb(value, thermal.min_c, thermal.max_c);
        colors.push(color[0] / 255, color[1] / 255, color[2] / 255); sampled += 1;
      }
    }
    if (!sampled) throw new Error('No 3D vertices overlap valid measured thermal cells.');
    return {
      ...scene, colors, temperatures_c: temperatures,
      thermal_projection: {
        source: thermal.source, crs_epsg: thermal.crs_epsg,
        sampled_vertices: sampled, total_vertices: scene.positions.length / 3,
        min_c: thermal.min_c, max_c: thermal.max_c,
        sampling: 'nearest_measured_cell', interpolated: false
      }
    };
  }

  function parseDigitalTwin(payload) {
    if (!payload || typeof payload !== 'object') throw new Error('Digital twin index must be a JSON object.');
    const source = payload.artifacts || {};
    const artifacts = [];
    if (Array.isArray(source)) {
      source.forEach((item, index) => artifacts.push({ id: String(item.id || `artifact-${index + 1}`), ...item }));
    } else {
      Object.entries(source).forEach(([kind, value]) => {
        if (Array.isArray(value)) value.forEach((path, index) => artifacts.push({ id: `${kind}-${index + 1}`, kind, path }));
        else if (value && typeof value === 'object') artifacts.push({ id: kind, kind, ...value });
        else if (value) artifacts.push({ id: kind, kind, path: value });
      });
    }
    return {
      id: String(payload.id || payload.project_id || 'digital-twin'),
      name: String(payload.name || 'Digital twin'),
      artifacts,
      surveys: Array.isArray(payload.surveys) ? payload.surveys : [],
      annotations: Array.isArray(payload.annotations) ? payload.annotations : [],
      defects: Array.isArray(payload.defects) ? payload.defects : [],
      crs_epsg: payload.crs_epsg == null ? null : Number(payload.crs_epsg)
    };
  }

  function identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        for (let k = 0; k < 4; k += 1) out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
      }
    }
    return out;
  }

  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2);
    const range = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (near + far) * range, -1, 0, 0, 2 * near * far * range, 0]);
  }

  function transformMatrix(yaw, pitch, distance, target) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return new Float32Array([
      cy, sy * sp, sy * cp, 0,
      0, cp, -sp, 0,
      -sy, cy * sp, cy * cp, 0,
      -target[0], -target[1], -target[2] - distance, 1
    ]);
  }

  class OdkWebGLViewer {
    constructor(canvas) {
      if (!canvas || typeof canvas.getContext !== 'function') throw new Error('A canvas element is required.');
      this.canvas = canvas;
      this.gl = canvas.getContext('webgl2', { antialias: true });
      if (!this.gl) throw new Error('WebGL2 is unavailable in this browser.');
      this.yaw = 0.5;
      this.pitch = -0.45;
      this.distance = 8;
      this.target = [0, 0, 0];
      this.clipZ = -1e20;
      this.pointSize = 2;
      this.geometry = null;
      this._program = this._createProgram();
      this._wireControls();
      this.resize();
    }

    _shader(type, source) {
      const shader = this.gl.createShader(type);
      this.gl.shaderSource(shader, source);
      this.gl.compileShader(shader);
      if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) throw new Error(this.gl.getShaderInfoLog(shader));
      return shader;
    }

    _createProgram() {
      const gl = this.gl;
      const vertex = this._shader(gl.VERTEX_SHADER, `#version 300 es
        in vec3 a_position; in vec3 a_color; uniform mat4 u_mvp; uniform float u_clipZ;
        uniform float u_pointSize; out vec3 v_color; out float v_visible;
        void main(){ v_color=a_color; v_visible=step(u_clipZ,a_position.z);
          gl_Position=u_mvp*vec4(a_position,1.0); gl_PointSize=u_pointSize;
          if(v_visible<0.5) gl_Position=vec4(2.0,2.0,2.0,1.0); }`);
      const fragment = this._shader(gl.FRAGMENT_SHADER, `#version 300 es
        precision highp float; in vec3 v_color; in float v_visible; out vec4 outColor;
        void main(){ if(v_visible<0.5) discard; outColor=vec4(v_color,1.0); }`);
      const program = gl.createProgram();
      gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      return program;
    }

    _wireControls() {
      let drag = null;
      this.canvas.addEventListener('pointerdown', (event) => { drag = [event.clientX, event.clientY]; this.canvas.setPointerCapture(event.pointerId); });
      this.canvas.addEventListener('pointermove', (event) => {
        if (!drag) return;
        this.yaw += (event.clientX - drag[0]) * 0.008;
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (event.clientY - drag[1]) * 0.008));
        drag = [event.clientX, event.clientY]; this.render();
      });
      this.canvas.addEventListener('pointerup', () => { drag = null; });
      this.canvas.addEventListener('wheel', (event) => {
        event.preventDefault(); this.distance = Math.max(0.05, this.distance * Math.exp(event.deltaY * 0.001)); this.render();
      }, { passive: false });
      this.canvas.tabIndex = 0;
      this.canvas.addEventListener('keydown', (event) => {
        const step = Math.max(0.05, this.distance * 0.04);
        const moves = { w: [0, step, 0], s: [0, -step, 0], a: [-step, 0, 0], d: [step, 0, 0], q: [0, 0, -step], e: [0, 0, step] };
        const move = moves[event.key.toLowerCase()];
        if (move) { this.target = this.target.map((value, index) => value + move[index]); this.render(); }
      });
    }

    resize() {
      const width = Math.max(1, this.canvas.clientWidth || this.canvas.width || 640);
      const height = Math.max(1, this.canvas.clientHeight || this.canvas.height || 400);
      if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
      this.gl.viewport(0, 0, width, height);
      this.render();
    }

    loadScene(input) {
      const scene = input.type === 'odk-scene-3d' && !Array.isArray(input.positions)
        ? parseScene(input) : input;
      this.geometry = scene;
      this._upload(scene.positions, scene.colors, scene.indices || []);
      this.render();
      return scene;
    }

    _upload(positions, colors, indices) {
      const gl = this.gl;
      this.vertexCount = positions.length / 3;
      this.indexCount = indices.length;
      this.mode = indices.length ? gl.TRIANGLES : gl.POINTS;
      this.positionBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
      this.colorBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
      if (indices.length) { this.indexBuffer = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW); }
    }

    setClipping(z) { this.clipZ = finiteNumber(z, 'clipping z'); this.render(); }

    render() {
      if (!this.gl) return;
      const gl = this.gl;
      gl.clearColor(0.025, 0.04, 0.065, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
      if (!this.geometry) return;
      gl.useProgram(this._program);
      const position = gl.getAttribLocation(this._program, 'a_position');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer); gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
      const color = gl.getAttribLocation(this._program, 'a_color');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer); gl.enableVertexAttribArray(color); gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 0, 0);
      const projection = perspective(Math.PI / 3, this.canvas.width / this.canvas.height, 0.01, 100000);
      const view = transformMatrix(this.yaw, this.pitch, this.distance, this.target);
      gl.uniformMatrix4fv(gl.getUniformLocation(this._program, 'u_mvp'), false, multiply(projection, view));
      gl.uniform1f(gl.getUniformLocation(this._program, 'u_clipZ'), this.clipZ);
      gl.uniform1f(gl.getUniformLocation(this._program, 'u_pointSize'), this.pointSize);
      if (this.indexCount) { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer); gl.drawElements(this.mode, this.indexCount, gl.UNSIGNED_INT, 0); }
      else gl.drawArrays(this.mode, 0, this.vertexCount);
    }

    async loadPointManifest(input, fetcher, onProgress) {
      const manifest = parsePointManifest(input);
      const request = fetcher || global.fetch.bind(global);
      const positions = [], colors = [];
      for (let index = 0; index < manifest.chunks.length; index += 1) {
        const response = await request(manifest.chunks[index].url);
        if (!response.ok) throw new Error(`Point chunk ${index + 1} returned HTTP ${response.status}.`);
        const chunk = parsePointChunk(await response.json());
        positions.push(...chunk.positions); colors.push(...chunk.colors);
        this.loadScene({ type: 'odk-scene-3d', vertices: positions, colors, indices: [], units: manifest.units, crs_epsg: manifest.crs_epsg });
        if (onProgress) onProgress({ loaded_chunks: index + 1, total_chunks: manifest.chunks.length, loaded_points: positions.length / 3 });
      }
      return { loaded_chunks: manifest.chunks.length, loaded_points: positions.length / 3 };
    }
  }

  class OdkThermalCanvas {
    constructor(canvas, options) {
      if (!canvas || typeof canvas.getContext !== 'function') throw new Error('A thermal canvas element is required.');
      this.canvas = canvas;
      this.context = canvas.getContext('2d');
      if (!this.context) throw new Error('A 2D canvas context is required for thermal maps.');
      this.map = null;
      this.sourceCanvas = null;
      this.zoom = 1; this.panX = 0; this.panY = 0;
      if (!options || options.interactive !== false) this._wireControls();
    }

    _wireControls() {
      let drag = null;
      this.canvas.addEventListener('pointerdown', (event) => {
        drag = [event.clientX, event.clientY]; this.canvas.setPointerCapture(event.pointerId);
      });
      this.canvas.addEventListener('pointermove', (event) => {
        if (!drag) return;
        this.panX += event.clientX - drag[0]; this.panY += event.clientY - drag[1];
        drag = [event.clientX, event.clientY]; this.render();
      });
      this.canvas.addEventListener('pointerup', () => { drag = null; });
      this.canvas.addEventListener('wheel', (event) => {
        event.preventDefault(); this.zoom = Math.max(0.25, Math.min(32, this.zoom * Math.exp(-event.deltaY * 0.001)));
        this.render();
      }, { passive: false });
    }

    load(input) {
      this.map = input.min_c != null ? input : parseThermalMap(input);
      const pixels = thermalImagePixels(this.map);
      this.sourceCanvas = document.createElement('canvas');
      this.sourceCanvas.width = pixels.width; this.sourceCanvas.height = pixels.height;
      const context = this.sourceCanvas.getContext('2d');
      const image = context.createImageData(pixels.width, pixels.height);
      image.data.set(new Uint8ClampedArray(pixels.data)); context.putImageData(image, 0, 0);
      this.zoom = 1; this.panX = 0; this.panY = 0; this.render();
      return this.map;
    }

    setView(view) {
      this.zoom = Math.max(0.25, Math.min(32, finiteNumber(view.zoom, 'thermal zoom')));
      this.panX = finiteNumber(view.panX, 'thermal pan x');
      this.panY = finiteNumber(view.panY, 'thermal pan y');
      this.render();
    }

    viewState() { return { zoom: this.zoom, panX: this.panX, panY: this.panY }; }

    _size() {
      const width = Math.max(1, this.canvas.clientWidth || this.canvas.width || 640);
      const height = Math.max(1, this.canvas.clientHeight || this.canvas.height || 400);
      if (this.canvas.width !== width) this.canvas.width = width;
      if (this.canvas.height !== height) this.canvas.height = height;
      return [width, height];
    }

    destination() {
      if (!this.map) return null;
      const [width, height] = this._size();
      const scale = Math.min(width / this.map.width, height / this.map.height) * this.zoom;
      const drawWidth = this.map.width * scale, drawHeight = this.map.height * scale;
      return { x: (width - drawWidth) / 2 + this.panX, y: (height - drawHeight) / 2 + this.panY,
        width: drawWidth, height: drawHeight, scale };
    }

    render() {
      const [width, height] = this._size();
      this.context.clearRect(0, 0, width, height);
      if (!this.map || !this.sourceCanvas) return;
      const destination = this.destination();
      this.context.imageSmoothingEnabled = false;
      this.context.drawImage(this.sourceCanvas, destination.x, destination.y,
        destination.width, destination.height);
    }

    temperatureAtClient(clientX, clientY) {
      if (!this.map) return null;
      const rect = this.canvas.getBoundingClientRect(), destination = this.destination();
      const column = Math.floor((clientX - rect.left - destination.x) / destination.scale);
      const row = Math.floor((clientY - rect.top - destination.y) / destination.scale);
      if (row < 0 || column < 0 || row >= this.map.height || column >= this.map.width) return null;
      const value = this.map.values[row * this.map.width + column];
      return Number.isFinite(value) ? value : null;
    }
  }

  function validateRgbThermalRegistration(map, rgbWidth, rgbHeight) {
    if (!map.registration) {
      throw new Error('RGB/thermal comparison requires explicit validated registration metadata.');
    }
    if (map.registration.rgb_width !== rgbWidth || map.registration.rgb_height !== rgbHeight) {
      throw new Error(
        `Registered RGB size is ${map.registration.rgb_width}x${map.registration.rgb_height}, ` +
        `but the selected image is ${rgbWidth}x${rgbHeight}.`
      );
    }
    return map.registration;
  }

  class OdkThermalComparison {
    constructor(wrapper, rgbCanvas, thermalCanvas) {
      if (!wrapper || !rgbCanvas || !thermalCanvas) throw new Error('Thermal comparison needs a wrapper and two canvases.');
      this.wrapper = wrapper; this.rgbCanvas = rgbCanvas;
      this.rgbContext = rgbCanvas.getContext('2d');
      this.thermalViewer = new OdkThermalCanvas(thermalCanvas, { interactive: false });
      this.mode = 'side-by-side'; this.opacity = 0.65; this.swipe = 0.5;
      this.zoom = 1; this.panX = 0; this.panY = 0;
      this.rgbSource = null; this.map = null; this.registration = null;
      this._wireControls(); this._applyMode();
    }

    _wireControls() {
      let drag = null;
      [this.rgbCanvas, this.thermalViewer.canvas].forEach((canvas) => {
        canvas.addEventListener('pointerdown', (event) => { drag = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
        canvas.addEventListener('pointermove', (event) => {
          if (!drag) return;
          this.panX += event.clientX - drag[0]; this.panY += event.clientY - drag[1];
          drag = [event.clientX, event.clientY]; this.render();
        });
        canvas.addEventListener('pointerup', () => { drag = null; });
        canvas.addEventListener('wheel', (event) => {
          event.preventDefault(); this.zoom = Math.max(0.25, Math.min(32, this.zoom * Math.exp(-event.deltaY * 0.001)));
          this.render();
        }, { passive: false });
      });
    }

    load(rgbSource, thermalInput) {
      const map = thermalInput.min_c != null ? thermalInput : parseThermalMap(thermalInput);
      const width = Number(rgbSource.naturalWidth || rgbSource.videoWidth || rgbSource.width);
      const height = Number(rgbSource.naturalHeight || rgbSource.videoHeight || rgbSource.height);
      this.registration = validateRgbThermalRegistration(map, width, height);
      this.rgbSource = rgbSource; this.map = map; this.thermalViewer.load(map);
      this.zoom = 1; this.panX = 0; this.panY = 0; this.render();
      return this.registration;
    }

    setMode(mode) {
      if (!['side-by-side', 'swipe', 'overlay'].includes(mode)) throw new Error('Thermal comparison mode must be side-by-side, swipe, or overlay.');
      this.mode = mode; this._applyMode(); this.render();
    }

    setOpacity(value) {
      this.opacity = Math.max(0, Math.min(1, finiteNumber(value, 'thermal opacity')));
      this._applyMode();
    }

    setSwipe(value) {
      this.swipe = Math.max(0, Math.min(1, finiteNumber(value, 'thermal swipe')));
      this._applyMode();
    }

    setView(view) {
      this.zoom = Math.max(0.25, Math.min(32, finiteNumber(view.zoom, 'comparison zoom')));
      this.panX = finiteNumber(view.panX, 'comparison pan x');
      this.panY = finiteNumber(view.panY, 'comparison pan y'); this.render();
    }

    viewState() { return { zoom: this.zoom, panX: this.panX, panY: this.panY }; }

    _applyMode() {
      this.wrapper.dataset.mode = this.mode;
      this.wrapper.classList.remove('side-by-side', 'swipe', 'overlay');
      this.wrapper.classList.add(this.mode);
      const thermal = this.thermalViewer.canvas;
      thermal.style.opacity = this.mode === 'overlay' ? String(this.opacity) : '1';
      thermal.style.clipPath = this.mode === 'swipe'
        ? `inset(0 ${Math.round((1 - this.swipe) * 10000) / 100}% 0 0)` : 'none';
    }

    render() {
      if (!this.rgbSource || !this.map) return;
      const width = Math.max(1, this.rgbCanvas.clientWidth || this.rgbCanvas.width || 640);
      const height = Math.max(1, this.rgbCanvas.clientHeight || this.rgbCanvas.height || 400);
      if (this.rgbCanvas.width !== width) this.rgbCanvas.width = width;
      if (this.rgbCanvas.height !== height) this.rgbCanvas.height = height;
      this.rgbContext.clearRect(0, 0, width, height);
      const scale = Math.min(width / this.map.width, height / this.map.height) * this.zoom;
      const drawWidth = this.map.width * scale, drawHeight = this.map.height * scale;
      const x = (width - drawWidth) / 2 + this.panX, y = (height - drawHeight) / 2 + this.panY;
      this.rgbContext.imageSmoothingEnabled = false;
      this.rgbContext.drawImage(this.rgbSource, x, y, drawWidth, drawHeight);
      this.thermalViewer.setView({ zoom: this.zoom, panX: this.panX, panY: this.panY });
      this._applyMode();
    }
  }

  const exported = {
    OdkWebGLViewer,
    OdkThermalCanvas,
    OdkThermalComparison,
    mapLibreRasterSource,
    normaliseProvider,
    parseDigitalTwin,
    parsePointChunk,
    parsePointManifest,
    parseScene,
    parseThermalMap,
    projectThermalOntoScene,
    providerTileUrl,
    sampleThermalMap,
    thermalColorRgb,
    thermalImagePixels,
    validateRgbThermalRegistration
  };
  global.ODKHubViewers = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
