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
    annotations: [],
    annotationSelection: new Set(),
    missions: [],
    missionSimulation: null,
    missionPlayer: null,
    missionTimer: null,
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
    if (name === 'missions') initMissionSimulator();
    setTimeout(() => {
      if (state.layerMap) state.layerMap.resize();
      if (state.viewer2d) state.viewer2d.map.resize();
      if (state.sceneViewer) state.sceneViewer.resize();
      if (state.pointViewer) state.pointViewer.resize();
      if (state.thermalViewer) state.thermalViewer.render();
      if (state.thermalComparison) state.thermalComparison.render();
      if (state.thermalModelViewer) state.thermalModelViewer.resize();
      if (state.missionPlayer) state.missionPlayer.render();
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
    state.missions = missionRows;
    renderMissionChoices();
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
    await loadAnnotations();
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
      onGeometryChange: () => { $('#viewer2d-result').textContent = 'Viewer geometry changed locally.'; },
      onAnnotation: (payload) => { persistAnnotation(payload); }
    });
  }

  function annotationFeature(row) {
    return {
      type: 'Feature', geometry: row.geometry,
      properties: {
        id: row.id, annotation_type: row.annotation_type, label: row.label,
        severity: row.severity, status: row.status, radius_m: row.geometry.radius_m || null
      }
    };
  }

  function renderAnnotations() {
    const holder = $('#annotation-list');
    holder.classList.toggle('empty', !state.annotations.length);
    holder.innerHTML = state.annotations.length ? state.annotations.map((row) => {
      const claim = row.machine_claims && row.machine_claims[0];
      const provenance = claim
        ? `${escapeHtml(claim.model_key)} @ ${escapeHtml(String(claim.model_sha256).slice(0, 12))} &middot; score ${Number(claim.confidence).toFixed(3)} &middot; ${row.machine_claims.length} immutable claim${row.machine_claims.length === 1 ? '' : 's'}`
        : 'Human-drawn annotation';
      const review = row.origin === 'model' ? `<div class="annotation-actions">
        <label><input type="checkbox" data-annotation-select="${Number(row.id)}" ${state.annotationSelection.has(Number(row.id)) ? 'checked' : ''} /> Select</label>
        <button data-annotation-action="accept" data-annotation-id="${Number(row.id)}">Accept</button>
        <button data-annotation-action="edit" data-annotation-id="${Number(row.id)}">Edit</button>
        <button data-annotation-action="reclassify" data-annotation-id="${Number(row.id)}">Reclassify</button>
        <button data-annotation-action="split" data-annotation-id="${Number(row.id)}">Split</button>
      </div>` : '';
      return `<article class="annotation-review-card" data-origin="${escapeHtml(row.origin)}">
        <header><strong>${escapeHtml(row.annotation_type)} &middot; ${escapeHtml(row.label)}</strong><span>${escapeHtml(row.severity)} &middot; ${escapeHtml(row.status)}</span></header>
        <small>${provenance} &middot; review: ${escapeHtml(row.review_action)}</small>${review}
      </article>`;
    }).join('') : 'No annotations saved for this project.';
    if (state.viewer2d) {
      state.viewer2d.removeLayer('saved-annotations');
      const mapped = state.annotations.filter((row) => row.source_type === 'map');
      if (mapped.length) state.viewer2d.addVector('saved-annotations', {
        type: 'FeatureCollection', features: mapped.map(annotationFeature)
      }, 1);
    }
  }

  async function loadAnnotations() {
    if (!state.api || !state.projectId) return;
    const rows = await apiAction('Load annotations', () => state.api.listAnnotations(state.projectId));
    if (rows) {
      state.annotations = rows;
      const ids = new Set(rows.map((row) => Number(row.id)));
      state.annotationSelection = new Set([...state.annotationSelection].filter((id) => ids.has(id)));
      renderAnnotations();
    }
  }

  async function runPrelabel(event) {
    event.preventDefault();
    if (!state.api || !state.projectId) { status('Connect and select a project first.', 'warning'); return; }
    const input = $('#prelabel-image'); const file = input.files && input.files[0];
    if (!file) { status('Select a real inspection image first.', 'warning'); return; }
    const batch = await apiAction('Installed-model pre-label', () => state.api.prelabelAnnotations(
      state.projectId, file, $('#prelabel-model').value, $('#prelabel-severity').value
    ));
    if (!batch) return;
    state.annotations.push(...batch.prelabels); renderAnnotations();
    $('#viewer2d-result').textContent = `${batch.finding_count} pre-labels from ${batch.model.model_key} @ ${batch.model.model_sha256.slice(0, 12)}; all await review.`;
  }

  function parsePromptJson(message, initial) {
    const text = window.prompt(message, JSON.stringify(initial));
    if (text == null) return null;
    try { return JSON.parse(text); }
    catch (error) { status(`Review JSON: ${error.message}`, 'error'); return null; }
  }

  async function handleAnnotationReview(event) {
    const button = event.target.closest('[data-annotation-action]');
    if (!button) return;
    const id = Number(button.dataset.annotationId);
    const row = state.annotations.find((item) => Number(item.id) === id);
    if (!row) return;
    const action = button.dataset.annotationAction;
    let operation;
    if (action === 'accept') {
      operation = () => state.api.reviewAnnotation(id, { action: 'accept' });
    } else if (action === 'edit') {
      const geometry = parsePromptJson('Edit the visible GeoJSON geometry. The model claim remains unchanged.', row.geometry);
      if (!geometry) return;
      const label = window.prompt('Edit the visible label.', row.label);
      if (label == null || !label.trim()) return;
      operation = () => state.api.reviewAnnotation(id, { action: 'edit', geometry, label: label.trim() });
    } else if (action === 'reclassify') {
      const label = window.prompt('Replacement human-reviewed class:', row.label);
      if (label == null || !label.trim()) return;
      operation = () => state.api.reviewAnnotation(id, { action: 'reclassify', label: label.trim() });
    } else if (action === 'split') {
      const geometries = parsePromptJson(
        'Enter a JSON array with two or more replacement GeoJSON geometries.', [row.geometry, row.geometry]
      );
      if (!Array.isArray(geometries) || geometries.length < 2) {
        status('Split needs at least two geometries.', 'warning'); return;
      }
      operation = () => state.api.splitAnnotation(id, { parts: geometries.map((geometry, index) => ({
        annotation_type: row.annotation_type, geometry, label: `${row.label} ${index + 1}`,
        severity: row.severity, status: 'open', note: `Split from annotation ${row.id}`
      })) });
    }
    const result = await apiAction(`Annotation ${action}`, operation);
    if (result) await loadAnnotations();
  }

  function handleAnnotationSelection(event) {
    const input = event.target.closest('[data-annotation-select]');
    if (!input) return;
    const id = Number(input.dataset.annotationSelect);
    if (input.checked) state.annotationSelection.add(id); else state.annotationSelection.delete(id);
  }

  async function mergeSelectedAnnotations() {
    const ids = [...state.annotationSelection];
    if (ids.length < 2) { status('Select at least two machine annotations to merge.', 'warning'); return; }
    const parents = ids.map((id) => state.annotations.find((row) => Number(row.id) === id)).filter(Boolean);
    if (parents.length !== ids.length) { status('Refresh annotations before merging.', 'warning'); return; }
    const geometry = parsePromptJson('Enter the reviewed GeoJSON geometry for the merged annotation.', parents[0].geometry);
    if (!geometry) return;
    const label = window.prompt('Merged human-reviewed label:', parents[0].label);
    if (label == null || !label.trim()) return;
    const merged = await apiAction('Merge annotations', () => state.api.mergeAnnotations(state.projectId, {
      annotation_ids: ids, annotation_type: parents[0].annotation_type, geometry,
      label: label.trim(), severity: parents[0].severity, status: 'open',
      note: `Merged from annotations ${ids.join(', ')}`
    }));
    if (merged) { state.annotationSelection.clear(); await loadAnnotations(); }
  }

  function renderMissionChoices() {
    const select = $('#mission-select');
    select.innerHTML = '<option value="">Select a mission</option>' + state.missions.map((mission) =>
      `<option value="${Number(mission.id)}">${escapeHtml(mission.name)} · v${Number(mission.version)} · ${Number(mission.waypoint_count)} points</option>`
    ).join('');
  }

  function initMissionSimulator() {
    if (state.missionPlayer) return;
    try {
      state.missionPlayer = new ODKHubMissions.MissionSimulator($('#mission-simulation-canvas'), (frame, simulation) => {
        $('#mission-frame').textContent = `${frame.time_s.toFixed(1)} s · ${frame.position[2].toFixed(1)} m · gimbal ${frame.gimbal_pitch_deg.toFixed(1)}° · ${frame.capture ? 'capture' : 'transit'}`;
        $('#mission-terrain').textContent = simulation.terrain.status === 'available'
          ? `${simulation.terrain.model_type} · ${simulation.terrain.source}`
          : simulation.terrain.reason;
        $('#mission-battery').textContent = frame.battery_pct == null
          ? simulation.battery.reason : `${frame.battery_pct.toFixed(1)}% · ${simulation.battery.basis}`;
      });
    } catch (error) { $('#mission-simulation-status').textContent = error.message; }
  }

  async function loadMissionSimulation() {
    const missionId = Number($('#mission-select').value);
    if (!missionId) { status('Select a mission first.', 'warning'); return; }
    const payload = await apiAction('Load compiled mission simulation', () => state.api.getMissionSimulation(missionId));
    if (!payload) return;
    initMissionSimulator(); state.missionSimulation = state.missionPlayer.load(payload);
    $('#mission-timeline').max = String(payload.timeline.length - 1); $('#mission-timeline').value = '0';
    $('#mission-simulation-status').textContent = `${payload.timeline.length} compiled frames · ${payload.capture_points.length} capture points · source ${payload.source}`;
  }

  function toggleMissionPlayback() {
    if (!state.missionSimulation) { status('Load a compiled mission first.', 'warning'); return; }
    if (state.missionTimer) {
      clearInterval(state.missionTimer); state.missionTimer = null;
      $('#play-mission-simulation').textContent = 'Play'; return;
    }
    $('#play-mission-simulation').textContent = 'Pause';
    state.missionTimer = setInterval(() => {
      const slider = $('#mission-timeline'); let next = Number(slider.value) + 1;
      if (next >= state.missionSimulation.timeline.length) next = 0;
      slider.value = String(next); state.missionPlayer.setFrame(next);
    }, 150);
  }

  async function createMissionShare(event) {
    event.preventDefault();
    if (!state.projectId) { status('Select a project first.', 'warning'); return; }
    const form = new FormData(event.currentTarget);
    const created = await apiAction('Create secure mission preview', () => state.api.createShare(state.projectId, {
      expires_in_days: Number(form.get('expires_in_days')),
      password: String(form.get('password') || ''), include_missions: true,
      include_defects: false, allow_download: false,
      note: 'Mission preview created in Hub'
    }));
    if (!created) return;
    const apiUrl = `${state.api.baseUrl}/public/shares/${encodeURIComponent(created.url_token)}`;
    $('#mission-share-result').classList.remove('empty');
    $('#mission-share-result').innerHTML = record(
      'View-only link created',
      `${apiUrl} · token shown once · ${created.password_protected ? 'password protected' : 'no password'}`
    );
  }

  async function persistAnnotation(payload) {
    if (!state.api || !state.projectId) {
      status('Connect and select a project before drawing an annotation.', 'warning');
      return;
    }
    const created = await apiAction(
      'Save annotation', () => state.api.createAnnotation(state.projectId, payload)
    );
    if (created) {
      state.annotations.push(created); renderAnnotations();
      $('#viewer2d-result').textContent = `${created.annotation_type} saved as annotation ${created.id}.`;
    }
  }

  function startAnnotation(event) {
    event.preventDefault();
    if (!state.api || !state.projectId) {
      status('Connect and select a project before drawing an annotation.', 'warning'); return;
    }
    initViewer2d();
    const data = new FormData(event.currentTarget);
    const type = String(data.get('annotation_type'));
    state.viewer2d.setAnnotationMetadata({
      source_type: 'map', source_id: `project:${state.projectId}:viewer2d`, crs_epsg: 4326,
      annotation_type: type, severity: String(data.get('severity')),
      status: String(data.get('status')), label: String(data.get('label')),
      note: String(data.get('note') || ''), radius_m: Number(data.get('radius_m')),
      include_in_report: true
    });
    state.viewer2d.setTool(`annotation-${type}`);
    $('#viewer2d-result').textContent = `Draw the ${type} annotation on the map.`;
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
    $('#annotation-form').onsubmit = startAnnotation;
    $('#refresh-annotations').onclick = loadAnnotations;
    $('#prelabel-form').onsubmit = runPrelabel;
    $('#merge-annotations').onclick = mergeSelectedAnnotations;
    $('#annotation-list').onclick = handleAnnotationReview;
    $('#annotation-list').onchange = handleAnnotationSelection;
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
    $('#load-mission-simulation').onclick = loadMissionSimulation;
    $('#play-mission-simulation').onclick = toggleMissionPlayback;
    $('#mission-timeline').oninput = (event) => {
      if (state.missionPlayer) state.missionPlayer.setFrame(Number(event.target.value));
    };
    $('#mission-share-form').onsubmit = createMissionShare;
    window.addEventListener('resize', () => {
      if (state.layerMap) state.layerMap.resize();
      if (state.viewer2d) state.viewer2d.map.resize();
      if (state.sceneViewer) state.sceneViewer.resize();
      if (state.pointViewer) state.pointViewer.resize();
      if (state.thermalViewer) state.thermalViewer.render();
      if (state.thermalComparison) state.thermalComparison.render();
      if (state.thermalModelViewer) state.thermalModelViewer.resize();
      if (state.missionPlayer) state.missionPlayer.render();
    });
  }

  const savedApi = localStorage.getItem('odk-hub-api'); if (savedApi) $('#hub-api-url').value = savedApi;
  const savedOrg = localStorage.getItem('odk-hub-org'); if (savedOrg) $('#hub-org-id').value = savedOrg;
  wire();
})();
