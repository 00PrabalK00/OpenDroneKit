/* OpenDroneKit Hub panels. All network access follows an explicit Connect action. */
(function () {
  'use strict';

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const state = {
    api: null,
    orgId: null,
    projectId: null,
    projects: [],
    assets: [],
    layerMap: null,
    customProviders: [],
    viewer2d: null,
    sceneViewer: null,
    pointViewer: null,
    thermalMap: null,
    thermalViewer: null,
    thermalComparison: null,
    thermalRgb: null,
    thermalScene: null,
    thermalModelViewer: null,
    twin: null
  };

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }

  function status(message, kind) {
    const target = $('#hub-status');
    target.textContent = message;
    target.className = kind || '';
  }

  function record(title, details, action) {
    return `<article class="record"><header><strong>${escapeHtml(title)}</strong>${action || ''}</header><small>${escapeHtml(details)}</small></article>`;
  }

  function timeline(when, text) {
    return `<div class="timeline-item"><time>${escapeHtml(when || 'undated')}</time><span>${escapeHtml(text)}</span></div>`;
  }

  async function readJsonFile(input) {
    const file = input.files && input.files[0];
    if (!file) throw new Error('Select a JSON file first.');
    try { return JSON.parse(await file.text()); }
    catch (error) { throw new Error(`${file.name} is not valid JSON: ${error.message}`); }
  }

  async function apiAction(label, operation) {
    if (!state.api) { status('Connect to the API first.', 'warning'); return null; }
    try {
      const result = await operation();
      status(`${label} complete.`, 'success');
      return result;
    } catch (error) {
      status(`${label}: ${error.message}`, 'error');
      return null;
    }
  }

  function switchPanel(name) {
    $$('.hub-nav button').forEach((button) => button.classList.toggle('active', button.dataset.panel === name));
    $$('.hub-panel').forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${name}`));
    if (name === 'maps') initLayerMap();
    if (name === 'viewer2d') initViewer2d();
    if (name === 'viewer3d') initSceneViewer();
    if (name === 'thermal') initThermalViewers();
    if (name === 'pointcloud') initPointViewer();
    setTimeout(() => {
      if (state.layerMap) state.layerMap.resize();
      if (state.viewer2d) state.viewer2d.map.resize();
      if (state.sceneViewer) state.sceneViewer.resize();
      if (state.pointViewer) state.pointViewer.resize();
      if (state.thermalViewer) state.thermalViewer.render();
      if (state.thermalComparison) state.thermalComparison.render();
      if (state.thermalModelViewer) state.thermalModelViewer.resize();
    }, 0);
  }

  async function connect() {
    try {
      state.orgId = Number($('#hub-org-id').value);
      if (!Number.isInteger(state.orgId) || state.orgId < 1) throw new Error('Organization id must be positive.');
      state.api = new ODKHubApi.HubApi($('#hub-api-url').value, $('#hub-token').value);
      localStorage.setItem('odk-hub-api', $('#hub-api-url').value);
      localStorage.setItem('odk-hub-org', String(state.orgId));
      const [projects, assets, tiles] = await Promise.all([
        state.api.listProjects(state.orgId), state.api.listAssets(state.orgId), state.api.tileStatus()
      ]);
      state.projects = projects;
      state.assets = assets;
      renderProjects();
      renderAssets();
      renderTileStatus(tiles);
      $('#metric-projects').textContent = String(projects.length);
      $('#metric-assets').textContent = String(assets.length);
      $('#metric-tiles').textContent = String(tiles.total_tiles || 0);
      const chip = $('#hub-connection-state');
      chip.textContent = 'connected'; chip.classList.remove('muted');
      await loadTileProviders();
      await renderOrganizationActivity();
      status('Connected to the OpenDroneKit API.', 'success');
    } catch (error) {
      state.api = null;
      const chip = $('#hub-connection-state');
      chip.textContent = 'not connected'; chip.classList.add('muted');
      status(`Connection failed: ${error.message}`, 'error');
    }
  }

  async function loadProjects() {
    const rows = await apiAction('Load projects', () => state.api.listProjects(state.orgId));
    if (rows) { state.projects = rows; renderProjects(); $('#metric-projects').textContent = String(rows.length); }
  }

  function renderProjects() {
    const holder = $('#project-list');
    holder.classList.toggle('empty', !state.projects.length);
    holder.innerHTML = state.projects.length ? state.projects.map((project) => record(
      project.name,
      `${project.client || 'no client'} · ${project.address || 'no site'} · ${project.project_type || 'untyped'} · ${project.tags || 'no tags'}`,
      `<button data-project-id="${Number(project.id)}">Open</button>`
    )).join('') : 'No projects returned by the organization.';
    holder.querySelectorAll('[data-project-id]').forEach((button) => {
      button.onclick = () => selectProject(Number(button.dataset.projectId));
    });
  }

  async function selectProject(projectId) {
    state.projectId = projectId;
    $$('#project-list .record').forEach((row) => row.classList.toggle('selected', Number(row.querySelector('[data-project-id]').dataset.projectId) === projectId));
    const [jobs, missions, members, audit] = await Promise.all([
      apiAction('Load project jobs', () => state.api.listJobs(projectId)),
      apiAction('Load project missions', () => state.api.listMissions(projectId)),
      apiAction('Load organization members', () => state.api.listMembers(state.orgId)),
      apiAction('Load organization audit', () => state.api.auditLog(state.orgId))
    ]);
    const jobRows = jobs || [];
    const missionRows = missions || [];
    $('#metric-jobs').textContent = String(jobRows.length);
    const memberRows = Array.isArray(members) ? members : (members && members.members) || [];
    $('#project-members').innerHTML = memberRows.length
      ? memberRows.map((member) => record(member.email || member.user_id, member.role || 'member')).join('')
      : 'No member rows returned.';
    const auditRows = Array.isArray(audit) ? audit : (audit && (audit.events || audit.items)) || [];
    const relevantAudit = auditRows.filter((item) => String(item.resource || '').includes(`project:${projectId}`));
    const activity = [
      ...jobRows.map((item) => ({ when: item.created_at, text: `Job ${item.id}: ${item.kind} — ${item.status} ${item.percent || 0}%` })),
      ...missionRows.map((item) => ({ when: item.created_at, text: `Mission ${item.name}: ${item.template}, ${item.waypoint_count} waypoints` })),
      ...relevantAudit.map((item) => ({ when: item.created_at, text: `${item.action} (${item.resource})` }))
    ].sort((a, b) => String(b.when || '').localeCompare(String(a.when || '')));
    $('#project-activity').innerHTML = activity.length
      ? activity.map((item) => timeline(item.when, item.text)).join('')
      : 'No project activity returned.';
  }

  async function renderOrganizationActivity() {
    const audit = await apiAction('Load organization activity', () => state.api.auditLog(state.orgId));
    if (!audit) return;
    const rows = Array.isArray(audit) ? audit : (audit.events || audit.items || []);
    $('#overview-activity').innerHTML = rows.length
      ? rows.slice(0, 20).map((item) => timeline(item.created_at, `${item.action || 'activity'} · ${item.resource || ''}`)).join('')
      : 'No activity returned.';
  }

  async function createProject(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const payload = Object.fromEntries(data.entries());
    const created = await apiAction('Create project', () => state.api.createProject(state.orgId, payload));
    if (created) { event.currentTarget.reset(); await loadProjects(); await selectProject(created.id); }
  }

  async function loadAssets() {
    const rows = await apiAction('Load assets', () => state.api.listAssets(state.orgId));
    if (rows) { state.assets = rows; renderAssets(); $('#metric-assets').textContent = String(rows.length); }
  }

  function renderAssets() {
    const holder = $('#asset-list');
    holder.classList.toggle('empty', !state.assets.length);
    holder.innerHTML = state.assets.length ? state.assets.map((asset) => {
      const location = asset.longitude == null ? 'geometry/location unavailable' : `${asset.latitude}, ${asset.longitude} · EPSG:${asset.crs_epsg}`;
      return record(asset.name, `${asset.asset_type} · ${location}`);
    }).join('') : 'No assets returned by the organization.';
  }

  async function createAsset(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const payload = Object.fromEntries(data.entries());
    for (const field of ['longitude', 'latitude']) payload[field] = payload[field] === '' ? null : Number(payload[field]);
    payload.crs_epsg = 4326;
    if (payload.geometry) {
      try { payload.geometry = JSON.parse(payload.geometry); }
      catch (error) { status(`Asset geometry: ${error.message}`, 'error'); return; }
    } else payload.geometry = null;
    const created = await apiAction('Create asset', () => state.api.createAsset(state.orgId, payload));
    if (created) { event.currentTarget.reset(); await loadAssets(); }
  }

  const STANDARD_PROVIDERS = {
    street: { kind: 'xyz', name: 'OpenStreetMap', url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', maxzoom: 19, attribution: 'OpenStreetMap contributors' },
    satellite: { kind: 'xyz', name: 'Satellite', url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', maxzoom: 23, attribution: 'Esri, Maxar, Earthstar Geographics' },
    terrain: { kind: 'xyz', name: 'Terrain', url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}', maxzoom: 16, attribution: 'Esri' },
    offline: { kind: 'xyz', name: 'Offline cache', url: '/tiles/{z}/{x}/{y}.png', maxzoom: 20, attribution: 'Locally cached tiles' }
  };

  function styleForProvider(provider) {
    return { version: 8, sources: { base: ODKHubViewers.mapLibreRasterSource(provider) }, layers: [
      { id: 'background', type: 'background', paint: { 'background-color': '#071018' } },
      { id: 'base', type: 'raster', source: 'base' }
    ] };
  }

  function initLayerMap() {
    if (state.layerMap) return;
    state.layerMap = new maplibregl.Map({
      container: 'hub-map', style: styleForProvider(STANDARD_PROVIDERS.street), center: [77.59, 12.97], zoom: 5
    });
    state.layerMap.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    state.layerMap.on('style.load', mountCustomProviders);
  }

  function mountCustomProviders() {
    if (!state.layerMap || !state.layerMap.isStyleLoaded()) return;
    state.customProviders.forEach((provider, index) => {
      const id = `custom-${index}`;
      if (!state.layerMap.getSource(id)) state.layerMap.addSource(id, ODKHubViewers.mapLibreRasterSource(provider));
      if (!state.layerMap.getLayer(id)) state.layerMap.addLayer({ id, type: 'raster', source: id, paint: { 'raster-opacity': 0.8 } });
    });
    renderMapLayerList();
  }

  function renderMapLayerList() {
    $('#map-layer-list').innerHTML = state.customProviders.map((provider, index) =>
      record(provider.name, `${provider.kind.toUpperCase()} · ${provider.url}`, `<button data-remove-provider="${index}">Remove</button>`)
    ).join('');
    $$('[data-remove-provider]').forEach((button) => {
      button.onclick = () => { state.customProviders.splice(Number(button.dataset.removeProvider), 1); state.layerMap.setStyle(styleForProvider(STANDARD_PROVIDERS[$('#hub-basemap').value])); };
    });
  }

  function addMapProvider() {
    try {
      const provider = ODKHubViewers.normaliseProvider({
        name: $('#provider-name').value, kind: $('#provider-kind').value,
        url: $('#provider-url').value, layers: $('#provider-layers').value,
        attribution: $('#provider-attribution').value
      });
      ODKHubViewers.providerTileUrl(provider);
      state.customProviders.push(provider);
      mountCustomProviders();
      status(`${provider.kind.toUpperCase()} layer added.`, 'success');
    } catch (error) { status(`Map provider: ${error.message}`, 'error'); }
  }

  async function loadTileProviders() {
    const payload = await apiAction('Load tile providers', () => state.api.tileProviders());
    if (!payload) return;
    const select = $('#tile-provider');
    select.innerHTML = Object.entries(payload.providers || {}).map(([name, spec]) =>
      `<option value="${escapeHtml(name)}">${escapeHtml(name)} (max z${Number(spec.max_zoom)})</option>`
    ).join('');
  }

  function renderTileStatus(payload) {
    const total = Number(payload.total_tiles || 0);
    const rows = Object.entries(payload.providers || {});
    $('#tile-cache-status').innerHTML = [
      record(total > 0 ? 'Offline coverage present' : 'Offline coverage unavailable', `${total} real tiles · ${Number(payload.bytes || 0)} bytes`),
      ...rows.map(([name, item]) => record(name, `${item.tiles} tiles · zooms ${(item.zoom_levels || []).join(', ') || 'none'}`))
    ].join('');
    $('#metric-tiles').textContent = String(total);
  }

  async function cacheTiles(event) {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.currentTarget).entries());
    const request = {
      provider: data.provider,
      west: Number(data.west), south: Number(data.south), east: Number(data.east), north: Number(data.north),
      min_zoom: Number(data.min_zoom), max_zoom: Number(data.max_zoom)
    };
    const job = await apiAction('Start tile cache', () => state.api.cacheTiles(request, state.orgId));
    if (!job) return;
    $('#tile-cache-job').textContent = `Caching ${job.tiles} tiles…`;
    const poll = async () => {
      const progress = await apiAction('Poll tile cache', () => state.api.cacheStatus(job.job_id));
      if (!progress) return;
      $('#tile-cache-job').textContent = `${progress.percent}% · ${progress.fetched} fetched · ${progress.failed} unavailable`;
      if (!progress.done) setTimeout(poll, 750);
      else {
        const summary = await state.api.tileStatus(); renderTileStatus(summary);
        status(progress.error ? `Tile cache failed: ${progress.error}` : 'Tile cache finished.', progress.error ? 'error' : 'success');
      }
    };
    setTimeout(poll, 300);
  }

  function initViewer2d() {
    if (state.viewer2d) return;
    state.viewer2d = new OdkMap('viewer2d-map', {
      center: [77.59, 12.97], zoom: 4,
      onMeasure: (message) => { $('#viewer2d-result').textContent = message; },
      onGeometryChange: () => { $('#viewer2d-result').textContent = 'Annotation geometry changed locally.'; }
    });
  }

  function coordinateBounds(geojson) {
    const points = [];
    const walk = (value) => {
      if (Array.isArray(value) && value.length >= 2 && typeof value[0] === 'number' && typeof value[1] === 'number') points.push(value);
      else if (Array.isArray(value)) value.forEach(walk);
      else if (value && typeof value === 'object') { if (value.coordinates) walk(value.coordinates); if (value.features) value.features.forEach(walk); if (value.geometry) walk(value.geometry); }
    };
    walk(geojson);
    if (!points.length) return null;
    return [Math.min(...points.map((p) => p[0])), Math.min(...points.map((p) => p[1])), Math.max(...points.map((p) => p[0])), Math.max(...points.map((p) => p[1]))];
  }

  async function loadViewerLayer(input, id, opacity) {
    try {
      initViewer2d();
      const payload = await readJsonFile(input);
      if (payload.type !== 'FeatureCollection') throw new Error('2D viewer accepts GeoJSON FeatureCollection files.');
      state.viewer2d.removeLayer(id);
      state.viewer2d.addVector(id, payload, opacity);
      const bounds = coordinateBounds(payload); if (bounds) state.viewer2d.fitTo(bounds);
      status(`${id} loaded with ${(payload.features || []).length} features.`, 'success');
    } catch (error) { status(`2D layer: ${error.message}`, 'error'); }
  }

  function initSceneViewer() {
    if (state.sceneViewer) return;
    try { state.sceneViewer = new ODKHubViewers.OdkWebGLViewer($('#scene-canvas')); }
    catch (error) { $('#scene-status').textContent = error.message; status(error.message, 'error'); }
  }

  function initPointViewer() {
    if (state.pointViewer) return;
    try { state.pointViewer = new ODKHubViewers.OdkWebGLViewer($('#point-canvas')); }
    catch (error) { $('#point-status').textContent = error.message; status(error.message, 'error'); }
  }

  async function loadSceneFile() {
    try {
      initSceneViewer(); if (!state.sceneViewer) return;
      const scene = ODKHubViewers.parseScene(await readJsonFile($('#scene-file')));
      state.sceneViewer.loadScene(scene);
      $('#scene-status').textContent = `${scene.positions.length / 3} vertices · ${scene.primitive} · ${scene.units}`;
    } catch (error) { $('#scene-status').textContent = error.message; status(`3D scene: ${error.message}`, 'error'); }
  }

  async function loadPointManifest() {
    try {
      initPointViewer(); if (!state.pointViewer) return;
      const manifest = ODKHubViewers.parsePointManifest(await readJsonFile($('#point-manifest-file')));
      $('#point-status').textContent = `0 of ${manifest.chunks.length} chunks`;
      await state.pointViewer.loadPointManifest(manifest, null, (progress) => {
        $('#point-status').textContent = `${progress.loaded_chunks} of ${progress.total_chunks} chunks · ${progress.loaded_points} points`;
      });
    } catch (error) { $('#point-status').textContent = error.message; status(`Point cloud: ${error.message}`, 'error'); }
  }

  function initThermalViewers() {
    if (!state.thermalViewer) {
      try {
        state.thermalViewer = new ODKHubViewers.OdkThermalCanvas($('#thermal-map-canvas'));
        $('#thermal-map-canvas').addEventListener('mousemove', (event) => {
          const value = state.thermalViewer.temperatureAtClient(event.clientX, event.clientY);
          if (value != null) $('#thermal-map-status').textContent = `${value.toFixed(2)} °C · EPSG:${state.thermalMap.crs_epsg}`;
        });
      } catch (error) { $('#thermal-map-status').textContent = error.message; }
    }
    if (!state.thermalComparison) {
      try {
        state.thermalComparison = new ODKHubViewers.OdkThermalComparison(
          $('#thermal-comparison'), $('#thermal-rgb-canvas'), $('#thermal-comparison-canvas')
        );
      } catch (error) { $('#thermal-comparison-status').textContent = error.message; }
    }
    if (!state.thermalModelViewer) {
      try { state.thermalModelViewer = new ODKHubViewers.OdkWebGLViewer($('#thermal-model-canvas')); }
      catch (error) { $('#thermal-model-status').textContent = error.message; }
    }
  }

  function readImageFile(input) {
    const file = input.files && input.files[0];
    if (!file) return Promise.reject(new Error('Select an RGB image first.'));
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file), image = new Image();
      image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error(`${file.name} is not a readable image.`)); };
      image.src = url;
    });
  }

  function loadThermalComparisonIfReady() {
    if (!state.thermalMap || !state.thermalRgb || !state.thermalComparison) return;
    const registration = state.thermalComparison.load(state.thermalRgb, state.thermalMap);
    $('#thermal-comparison-status').textContent =
      `${registration.method} · residual ${registration.residual_px.toFixed(2)} px · validated by ${registration.validated_by}`;
  }

  async function loadThermalMap() {
    try {
      initThermalViewers();
      state.thermalMap = ODKHubViewers.parseThermalMap(await readJsonFile($('#thermal-map-file')));
      state.thermalViewer.load(state.thermalMap);
      $('#thermal-map-status').textContent =
        `${state.thermalMap.min_c.toFixed(2)}–${state.thermalMap.max_c.toFixed(2)} °C · EPSG:${state.thermalMap.crs_epsg} · ${state.thermalMap.interpolated ? 'interpolated source declared' : 'measured cells'}`;
      try { loadThermalComparisonIfReady(); }
      catch (error) { $('#thermal-comparison-status').textContent = error.message; }
      status('Radiometric thermal map loaded locally.', 'success');
    } catch (error) { $('#thermal-map-status').textContent = error.message; status(`Thermal map: ${error.message}`, 'error'); }
  }

  async function loadThermalRgb() {
    try {
      initThermalViewers(); state.thermalRgb = await readImageFile($('#thermal-rgb-file'));
      loadThermalComparisonIfReady(); status('Registered RGB comparison image loaded.', 'success');
    } catch (error) { $('#thermal-comparison-status').textContent = error.message; status(`Thermal comparison: ${error.message}`, 'error'); }
  }

  async function loadThermalScene() {
    try {
      state.thermalScene = ODKHubViewers.parseScene(await readJsonFile($('#thermal-scene-file')));
      $('#thermal-model-status').textContent = `${state.thermalScene.positions.length / 3} vertices ready for CRS-checked projection.`;
    } catch (error) { $('#thermal-model-status').textContent = error.message; status(`Thermal 3D scene: ${error.message}`, 'error'); }
  }

  function projectThermalModel() {
    try {
      initThermalViewers();
      if (!state.thermalMap || !state.thermalScene) throw new Error('Load both a thermal map and a 3D scene first.');
      const model = ODKHubViewers.projectThermalOntoScene(state.thermalScene, state.thermalMap);
      state.thermalModelViewer.loadScene(model);
      const evidence = model.thermal_projection;
      $('#thermal-model-status').textContent =
        `${evidence.sampled_vertices}/${evidence.total_vertices} vertices carry nearest measured-cell temperatures · ${evidence.min_c.toFixed(2)}–${evidence.max_c.toFixed(2)} °C`;
      status('Thermal measurements projected onto the 3D scene.', 'success');
    } catch (error) { $('#thermal-model-status').textContent = error.message; status(`Thermal 3D model: ${error.message}`, 'error'); }
  }

  function renderTwin(twin) {
    $('#twin-identity').innerHTML = record(twin.name, `${twin.id} · ${twin.crs_epsg ? `EPSG:${twin.crs_epsg}` : 'CRS not stated'}`);
    $('#twin-artifacts').innerHTML = twin.artifacts.length
      ? twin.artifacts.map((item) => record(item.kind || item.id, item.path || item.url || 'embedded artifact')).join('')
      : 'No artifacts indexed.';
    $('#twin-evidence').innerHTML = [
      record('Surveys', twin.surveys.length), record('Annotations', twin.annotations.length), record('Defects', twin.defects.length)
    ].join('');
    $('#survey-timeline').innerHTML = twin.surveys.length ? twin.surveys
      .slice().sort((a, b) => String(a.date || a.captured_at || '').localeCompare(String(b.date || b.captured_at || '')))
      .map((survey) => {
        const change = survey.measured_change
          ? ` · measured change: ${JSON.stringify(survey.measured_change)}`
          : ' · no measured change artifact';
        return timeline(survey.date || survey.captured_at, `${survey.name || survey.id || 'survey'}${change}`);
      }).join('') : 'No dated surveys are indexed.';
  }

  async function loadDigitalTwin() {
    try {
      state.twin = ODKHubViewers.parseDigitalTwin(await readJsonFile($('#digital-twin-file')));
      renderTwin(state.twin); status(`Digital twin ${state.twin.name} loaded locally.`, 'success');
    } catch (error) { status(`Digital twin: ${error.message}`, 'error'); }
  }

  function wire() {
    $$('.hub-nav button').forEach((button) => { button.onclick = () => switchPanel(button.dataset.panel); });
    $('#hub-connect').onclick = connect;
    $('#refresh-projects').onclick = loadProjects;
    $('#refresh-assets').onclick = loadAssets;
    $('#project-form').onsubmit = createProject;
    $('#asset-form').onsubmit = createAsset;
    $('#hub-basemap').onchange = (event) => { initLayerMap(); state.layerMap.setStyle(styleForProvider(STANDARD_PROVIDERS[event.target.value])); };
    $('#add-map-provider').onclick = addMapProvider;
    $('#tile-cache-form').onsubmit = cacheTiles;
    $('#viewer2d-a').onchange = (event) => loadViewerLayer(event.target, 'comparison-a', 1 - Number($('#comparison-slider').value) / 100);
    $('#viewer2d-b').onchange = (event) => loadViewerLayer(event.target, 'comparison-b', Number($('#comparison-slider').value) / 100);
    $('#comparison-slider').oninput = (event) => {
      if (!state.viewer2d) return;
      const value = Number(event.target.value) / 100;
      state.viewer2d.setLayerOpacity('comparison-a', 1 - value);
      state.viewer2d.setLayerOpacity('comparison-b', value);
    };
    $$('[data-viewer-tool]').forEach((button) => { button.onclick = () => { initViewer2d(); state.viewer2d.setTool(button.dataset.viewerTool); }; });
    $('#scene-file').onchange = loadSceneFile;
    $('#scene-clip').oninput = (event) => { initSceneViewer(); if (state.sceneViewer) state.sceneViewer.setClipping(Number(event.target.value)); };
    $('#point-manifest-file').onchange = loadPointManifest;
    $('#point-size').oninput = (event) => { initPointViewer(); if (state.pointViewer) { state.pointViewer.pointSize = Number(event.target.value); state.pointViewer.render(); } };
    $('#thermal-map-file').onchange = loadThermalMap;
    $('#thermal-rgb-file').onchange = loadThermalRgb;
    $('#thermal-scene-file').onchange = loadThermalScene;
    $('#project-thermal-model').onclick = projectThermalModel;
    $('#thermal-comparison-mode').onchange = (event) => { initThermalViewers(); state.thermalComparison.setMode(event.target.value); };
    $('#thermal-opacity').oninput = (event) => { initThermalViewers(); state.thermalComparison.setOpacity(Number(event.target.value) / 100); };
    $('#thermal-swipe').oninput = (event) => { initThermalViewers(); state.thermalComparison.setSwipe(Number(event.target.value) / 100); };
    $('#digital-twin-file').onchange = loadDigitalTwin;
    window.addEventListener('resize', () => {
      if (state.layerMap) state.layerMap.resize();
      if (state.viewer2d) state.viewer2d.map.resize();
      if (state.sceneViewer) state.sceneViewer.resize();
      if (state.pointViewer) state.pointViewer.resize();
      if (state.thermalViewer) state.thermalViewer.render();
      if (state.thermalComparison) state.thermalComparison.render();
      if (state.thermalModelViewer) state.thermalModelViewer.resize();
    });
  }

  const savedApi = localStorage.getItem('odk-hub-api'); if (savedApi) $('#hub-api-url').value = savedApi;
  const savedOrg = localStorage.getItem('odk-hub-org'); if (savedOrg) $('#hub-org-id').value = savedOrg;
  wire();
})();
