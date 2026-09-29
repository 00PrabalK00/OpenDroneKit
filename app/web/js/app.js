/* Application controller: wires the DOM and the native menu to the Python API.
 *
 * Every control here calls a real backend method. Where the backend cannot do
 * something (no vehicle, no models, no CUDA) the UI shows that reason rather than a
 * cosmetic value.
 */

(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const state = {
    map: null,
    layers: [],
    jobPollers: new Map(),
    attributeRows: [],
    ready: false
  };

  /* ---------- helpers ---------- */

  function log(message) {
    const pre = $('#console-log');
    const stamp = new Date().toLocaleTimeString();
    pre.textContent += `[${stamp}] ${message}\n`;
    pre.scrollTop = pre.scrollHeight;
  }

  function status(message) {
    $('#status-message').textContent = message;
    log(message);
  }

  async function call(method, ...args) {
    if (!window.pywebview || !window.pywebview.api) throw new Error('Python bridge is not ready.');
    const fn = window.pywebview.api[method];
    if (!fn) throw new Error(`Unknown API method: ${method}`);
    const result = await fn(...args);
    if (result && result.ok === false) throw new Error(result.error || 'Unknown error');
    return result;
  }

  /* Runs an API call and reports failure in the status bar instead of throwing into
     an event handler where it would be swallowed. */
  async function tryCall(method, ...args) {
    try {
      return await call(method, ...args);
    } catch (err) {
      status(`${method}: ${err.message}`);
      return null;
    }
  }

  function modal(title, bodyHtml, actions) {
    $('#modal-title').textContent = title;
    $('#modal-body').innerHTML = bodyHtml;
    const holder = $('#modal-actions');
    holder.innerHTML = '';
    (actions || [{ label: 'Close' }]).forEach((action) => {
      const button = document.createElement('button');
      button.textContent = action.label;
      if (action.primary) button.className = 'primary';
      button.onclick = async () => {
        if (action.onClick) {
          const keepOpen = await action.onClick();
          if (keepOpen === true) return;
        }
        closeModal();
      };
      holder.appendChild(button);
    });
    $('#modal-backdrop').classList.remove('hidden');
  }

  function closeModal() {
    $('#modal-backdrop').classList.add('hidden');
  }

  function kv(rows) {
    return rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  }

  function flag(value) {
    return value ? '<b class="yes">yes</b>' : '<b class="no">no</b>';
  }

  /* ---------- layer tree ---------- */

  async function refreshLayers() {
    const result = await tryCall('list_layers');
    if (!result) return;
    state.layers = result.layers;
    const tree = $('#layer-tree');
    tree.innerHTML = '';

    if (!state.layers.length) {
      tree.innerHTML = '<li><span class="layer-meta">No layers. Import a raster or run a reconstruction.</span></li>';
      return;
    }

    state.layers.forEach((layer) => {
      const li = document.createElement('li');

      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = layer.visible;
      check.onchange = async () => {
        await tryCall('set_layer_visible', layer.id, check.checked);
        state.map.setLayerVisible(layer.id, check.checked);
      };

      const name = document.createElement('div');
      const warning = layer.metadata && layer.metadata.error;
      name.className = 'layer-name';
      name.title = warning || layer.path;
      name.innerHTML = `${layer.name}${warning ? ' <span class="layer-warn">!</span>' : ''}` +
        `<div class="layer-meta">${layer.kind}${layer.crs_epsg ? ' · EPSG:' + layer.crs_epsg : ' · no CRS'}</div>`;
      name.onclick = () => zoomToLayer(layer);

      const opacity = document.createElement('input');
      opacity.type = 'range';
      opacity.min = '0';
      opacity.max = '1';
      opacity.step = '0.05';
      opacity.value = String(layer.opacity);
      opacity.oninput = async () => {
        const value = parseFloat(opacity.value);
        state.map.setLayerOpacity(layer.id, value);
        await tryCall('set_layer_opacity', layer.id, value);
      };

      const remove = document.createElement('button');
      remove.className = 'layer-remove';
      remove.textContent = '×';
      remove.title = `Remove ${layer.name} from the project`;
      remove.onclick = async (event) => {
        event.stopPropagation();
        await tryCall('remove_layer', layer.id);
        state.map.removeLayer(layer.id);
        await refreshLayers();
      };

      li.append(check, name, opacity, remove);
      tree.appendChild(li);
    });

    await Promise.all(state.layers.map(mountLayer));
  }

  async function mountLayer(layer) {
    if (layer.kind === 'vector') {
      if (state.map.vectorLayers.has(layer.id)) return;
      const result = await tryCall('read_vector_layer', layer.id);
      if (result) {
        state.map.addVector(layer.id, result.geojson, layer.opacity);
        showAttributes(result.geojson, layer.name);
      }
      return;
    }
    if (state.map.rasterLayers.has(layer.id)) return;
    if (!layer.crs_epsg) return;
    const result = await tryCall('raster_preview', layer.id);
    if (result) state.map.addRaster(layer.id, result.data_uri, result.coordinates, layer.opacity);
  }

  function zoomToLayer(layer) {
    if (layer.bounds_lonlat && layer.bounds_lonlat.length === 4) {
      state.map.fitTo(layer.bounds_lonlat);
      status(`Zoomed to ${layer.name}`);
    } else {
      status(`${layer.name} has no usable extent${layer.crs_epsg ? '' : ' (no CRS)'}.`);
    }
  }

  /* ---------- attribute table ---------- */

  function showAttributes(geojson, title) {
    const features = (geojson && geojson.features) || [];
    const tableHead = $('#attr-table thead');
    const tableBody = $('#attr-table tbody');
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';
    state.attributeRows = features;

    if (!features.length) {
      tableHead.innerHTML = `<tr><th>${title || 'Attributes'}</th></tr>`;
      tableBody.innerHTML = '<tr><td>No features</td></tr>';
      return;
    }

    const keys = Array.from(new Set(features.flatMap((f) => Object.keys(f.properties || {})))).slice(0, 12);
    tableHead.innerHTML = '<tr>' + keys.map((k) => `<th>${k}</th>`).join('') + '</tr>';

    features.slice(0, 500).forEach((feature, index) => {
      const tr = document.createElement('tr');
      tr.innerHTML = keys.map((k) => `<td>${formatCell((feature.properties || {})[k])}</td>`).join('');
      // Selecting a row flies the map to the feature: the two-way link that makes an
      // attribute table useful rather than decorative.
      tr.onclick = () => {
        $$('#attr-table tbody tr').forEach((row) => row.classList.remove('selected'));
        tr.classList.add('selected');
        const coords = OdkMapUtils.flatten(feature);
        if (coords.length) state.map.fitToCoords(coords, 220);
        status(`Feature ${index + 1} of ${features.length}`);
      };
      tableBody.appendChild(tr);
    });
    switchTab('attributes');
  }

  function formatCell(value) {
    if (value == null) return '';
    if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(4);
    return String(value);
  }

  /* ---------- jobs ---------- */

  function pollJob(jobId, onDone) {
    if (state.jobPollers.has(jobId)) return;
    const timer = setInterval(async () => {
      const result = await tryCall('job_status', jobId);
      if (!result) {
        clearInterval(timer);
        state.jobPollers.delete(jobId);
        return;
      }
      renderJobs();
      const job = result.job;
      if (['done', 'failed', 'cancelled'].includes(job.status)) {
        clearInterval(timer);
        state.jobPollers.delete(jobId);
        status(`${job.name}: ${job.status}${job.error ? ' — ' + job.error : ''}`);
        refreshReportInfo();
        if (job.status === 'done' && onDone) onDone(job);
      } else {
        $('#status-message').textContent = `${job.name} ${job.percent}% — ${job.message}`;
      }
    }, 900);
    state.jobPollers.set(jobId, timer);
    switchTab('jobs');
  }

  async function renderJobs() {
    const result = await tryCall('list_jobs');
    if (!result) return;
    const holder = $('#jobs-list');
    const jobs = result.jobs.slice().reverse();
    if (!jobs.length) {
      holder.innerHTML = '<div class="info">No jobs have run yet.</div>';
      return;
    }
    holder.innerHTML = jobs.map((job) => `
      <div class="job ${job.status}">
        <div class="job-head">
          <b>${job.name}</b>
          <span>${job.status} · ${job.percent}%</span>
        </div>
        <div class="job-bar"><div style="width:${job.percent}%"></div></div>
        <div class="job-msg">${job.error || job.message || ''}</div>
        ${job.status === 'running'
          ? `<button class="job-cancel" data-job="${job.id}">Cancel</button>`
          : ''}
      </div>`).join('');

    /* A reconstruction can run for hours. The backend has supported cooperative
       cancellation all along; without this button there was no way to reach it. */
    holder.querySelectorAll('.job-cancel').forEach((button) => {
      button.onclick = async () => {
        button.disabled = true;
        button.textContent = 'Cancelling...';
        await tryCall('cancel_job', button.dataset.job);
        await renderJobs();
      };
    });
  }

  /* ---------- mission ---------- */

  /* A blank number box is "not given" (null), never 0: a stand-off of 0 or a gimbal of
     NaN is a value the planner would act on. */
  function numberOrNull(selector) {
    const raw = ($(selector) && $(selector).value || '').trim();
    if (raw === '') return null;
    const value = parseFloat(raw);
    return Number.isFinite(value) ? value : null;
  }

  function missionOptions(geometry) {
    const g = geometry || (state.map ? state.map.getGeometry() : {});
    const options = {
      template: $('#mission-template').value,
      altitude_m: parseFloat($('#mission-altitude').value),
      speed_m_s: parseFloat($('#mission-speed').value),
      front_overlap_pct: parseFloat($('#mission-front').value),
      side_overlap_pct: parseFloat($('#mission-side').value),
      gimbal_tilt_deg: numberOrNull('#mission-gimbal'),
      dwell_s: numberOrNull('#mission-dwell'),
      standoff_m: numberOrNull('#mission-standoff'),
      min_altitude_m: parseFloat($('#mission-minalt').value),
      max_altitude_m: parseFloat($('#mission-maxalt').value),
      rth_altitude_m: parseFloat($('#mission-rthalt').value),
      wind_speed_m_s: parseFloat($('#mission-wind').value),
      terrain_follow_mode: $('#mission-terrain-mode').value,
      path_lonlat: g.route || null,
      interest_regions: g.interest || [],
      // Mission-type inputs; each is read only by the type that needs it.
      thermal_camera: $('#mission-thermal-camera').value || null,
      target_gsd_cm: numberOrNull('#mission-thermal-gsd'),
      rgb_camera: $('#mission-rgb-camera').value || '',
      payload_key: $('#mission-ms-payload').value || null,
      calibration_panel_lonlat: $('#mission-ms-panel').value.trim() || null,
      accept_relative_values: $('#mission-ms-relative').checked,
      pylon_elements: $('#mission-pylon-elements').value.trim(),
      dome_radius_m: numberOrNull('#mission-dome-radius'),
      dome_base_altitude_m: numberOrNull('#mission-dome-base'),
      box_altitude_levels_m: $('#mission-box-levels').value.trim() || null,
      linked_version_ids: $('#mission-linked-ids').value.trim() || null
    };
    return options;
  }

  /* The template list comes from the planner, so every mission type it can plan is
     offered -- the static list here showed twelve of twenty-five. */
  async function loadMissionTemplates() {
    const result = await tryCall('mission_templates');
    if (!result) return;
    const select = $('#mission-template');
    const current = select.value || 'grid';
    const labels = result.labels || {};
    const all = [...(result.templates || []), ...(result.special_templates || [])];
    all.sort((a, b) => (a === 'grid' ? -1 : b === 'grid' ? 1 : String(labels[a] || a).localeCompare(labels[b] || b)));
    select.innerHTML = all.map((t) => `<option value="${t}">${labels[t] || t}</option>`).join('');
    select.value = all.includes(current) ? current : 'grid';
    showTemplateOptions();

    const cameras = await tryCall('list_cameras');
    const list = (cameras && (cameras.cameras || [])) || [];
    const thermal = list.filter((c) => c.thermal);
    $('#mission-thermal-camera').innerHTML = thermal.map((c) => `<option value="${c.key}">${c.name || c.key}</option>`).join('');
    $('#mission-rgb-camera').innerHTML = '<option value="">none</option>' +
      list.filter((c) => !c.thermal).map((c) => `<option value="${c.key}">${c.name || c.key}</option>`).join('');
    const payloads = await tryCall('list_payloads');
    const ms = ((payloads && payloads.payloads) || []).filter((p) => p.kind === 'multispectral');
    $('#mission-ms-payload').innerHTML = ms.map((p) => `<option value="${p.key}">${p.name || p.key}</option>`).join('');
  }

  function showTemplateOptions() {
    const template = $('#mission-template').value;
    $$('#mission-template-options [data-for]').forEach((node) => {
      node.style.display = node.dataset.for.split(/\s+/).includes(template) ? '' : 'none';
    });
  }

  async function planMission() {
    const geometry = state.map.getGeometry();
    if (!geometry.aoi) {
      status('Draw an area of interest first — click "Draw AOI" and click out a polygon.');
      return;
    }
    await tryCall('set_aoi', geometry.aoi);
    await tryCall('set_no_fly_zones', geometry.nofly);

    $('#btn-plan').disabled = true;
    status('Planning mission...');
    const result = await tryCall('plan_mission', missionOptions(geometry));
    $('#btn-plan').disabled = false;
    if (!result) return;

    state.map.setMission(result.geojson);
    const s = result.summary;
    const adjustments = s.adjustments || {};
    $('#mission-summary').innerHTML =
      kv([
        ['Waypoints', s.waypoints],
        ['Distance', (s.distance_m / 1000).toFixed(2) + ' km'],
        ['Duration', s.duration_min.toFixed(1) + ' min'],
        ['GSD', s.gsd_cm.toFixed(2) + ' cm/px'],
        ['Altitude', s.altitude_m.toFixed(0) + ' m'],
        ['Speed', (s.speed_m_s || 0).toFixed(1) + ' m/s'],
        ['Altitudes', s.altitude_frame === 'amsl' ? 'absolute (AMSL)' : 'relative to home']
      ]) +
      kv([
        ['Geofence clamps', adjustments.geofence_projections || 0],
        ['No-fly pushes', adjustments.no_fly_projections || 0],
        ['Detour points', adjustments.obstacle_detours || 0],
        ['Altitude clamps', adjustments.altitude_clamps || 0]
      ]) +
      (result.warnings || []).map((w) => `<span class="warn">${w}</span>`).join('');

    status(`Mission planned: ${s.waypoints} waypoints, ${s.duration_min.toFixed(1)} min.`);
    state.map.fitTo(result.geojson);
  }

  /* ---------- vehicle ---------- */

  function updateVehicleChip(vehicle) {
    const chip = $('#vehicle-chip');
    if (!vehicle || !vehicle.connected) {
      chip.className = 'chip chip-muted';
      chip.textContent = 'No vehicle';
      return;
    }
    chip.className = vehicle.is_simulated ? 'chip chip-sim' : 'chip chip-live';
    chip.textContent = vehicle.is_simulated ? `SIMULATED (${vehicle.driver})` : `LIVE ${vehicle.driver}`;
  }

  function connectDialog() {
    modal('Connect Vehicle', `
      <div class="form">
        <label>Driver
          <select id="dlg-driver">
            <option value="pymavlink">MAVLink (real autopilot: ArduPilot / PX4)</option>
            <option value="mavsdk">PX4 via MAVSDK</option>
            <option value="mock">Mock (simulated, for UI testing)</option>
          </select>
        </label>
        <label>Connection URI
          <input id="dlg-uri" value="udpin:0.0.0.0:14550" />
        </label>
        <div class="info">
          Examples: <b>udpin:0.0.0.0:14550</b> (SITL / telemetry),
          <b>tcp:127.0.0.1:5760</b> (SITL direct), <b>COM3</b> or <b>/dev/ttyUSB0</b> (radio).
        </div>
      </div>`,
      [
        { label: 'Cancel' },
        {
          label: 'Connect', primary: true, onClick: async () => {
            const driver = $('#dlg-driver').value;
            const uri = $('#dlg-uri').value.trim();
            const result = await tryCall('connect_vehicle', uri, driver);
            if (result) {
              updateVehicleChip(result.vehicle);
              status(result.vehicle.connected
                ? `Connected to ${uri} via ${driver}.`
                : `Connection failed: ${result.vehicle.last_error}`);
              if (result.vehicle.connected) { startTelemetry(); switchTab('telemetry'); }
            }
          }
        }
      ]);
  }

  let telemetryTimer = null;

  function startTelemetry() {
    if (telemetryTimer) clearInterval(telemetryTimer);
    telemetryTimer = setInterval(async () => {
      const result = await tryCall('telemetry');
      if (!result) return;
      const data = result.telemetry;
      const grid = $('#telemetry-grid');
      const chip = $('#vehicle-chip');
      /* A lost link keeps its last values on screen -- where the aircraft was last
         heard is what the pilot needs -- under a status row that says they are old. */
      if (!data.connected && !data.link_lost) {
        grid.innerHTML = `<div><span>status</span><b>${data.reason || 'disconnected'}</b></div>`;
        return;
      }
      if (data.link_lost) {
        chip.className = 'chip chip-sim';
        chip.textContent = `LINK LOST (${Math.round(data.telemetry_age_s || 0)} s)`;
      } else if (chip.textContent.startsWith('LINK LOST')) {
        updateVehicleChip({ connected: true, driver: data.driver, is_simulated: data.is_simulated });
      }
      const hidden = new Set(['connected', 'timestamp', 'stale', 'link_lost', 'reason']);
      const head = data.link_lost ? `<div><span>status</span><b class="no">${data.reason}</b></div>` : '';
      /* Only what the vehicle reported: a field it never sent is absent from the
         payload, so it is absent here, rather than shown as 0.000. */
      grid.innerHTML = head + Object.entries(data)
        .filter(([k]) => !hidden.has(k))
        .map(([k, v]) => `<div><span>${k}</span><b>${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}</b></div>`)
        .join('');
    }, 1000);
  }

  /* ---------- preflight ---------- */

  async function runPreflight(acknowledged) {
    const result = await tryCall('run_preflight', acknowledged || []);
    const list = $('#preflight-list');
    if (!result) return;
    const mark = (c) => (c.confirmed || c.severity === 'pass') ? '<b class="yes">pass</b>'
      : c.severity === 'blocking' ? '<b class="no">BLOCKS</b>' : '<b>check</b>';
    list.innerHTML =
      `<div><span>${result.simulated ? 'SIMULATED vehicle — ' : ''}Ready to start</span>${flag(result.can_start)}</div>` +
      result.checks.map((c) => `<div><span>${c.name}: ${c.message}` +
        (c.requires_manual && !c.confirmed
          ? ` <button class="mini-btn" data-confirm="${c.id}">I checked this</button>` : '') +
        `</span>${mark(c)}</div>`).join('');
    $$('#preflight-list [data-confirm]').forEach((button) => {
      button.onclick = () => runPreflight([button.dataset.confirm]);
    });
    status(result.can_start ? 'Preflight: ready to start.'
      : `Preflight: ${result.blocking_count} blocking, ${result.pending_confirmation.length} to confirm.`);
  }

  /* ---------- fleet ---------- */

  const ORG = 1;  // the local organisation a desktop install records against

  async function refreshFleet() {
    const result = await tryCall('list_fleet', ORG);
    const list = $('#fleet-list');
    if (!result) return;
    const rows = (items, fmt) => items.length ? items.map(fmt).join('') : '<div><span>none registered</span></div>';
    list.innerHTML =
      '<div><b>Aircraft</b></div>' + rows(result.aircraft, (a) =>
        `<div><span>#${a.id} ${a.name} ${a.model || ''}</span><b>${a.flight_hours} h · ${a.status}` +
        `${a.hours_to_service == null ? '' : ` · service in ${a.hours_to_service} h`}</b></div>`) +
      '<div><b>Batteries</b></div>' + rows(result.batteries, (b) =>
        `<div><span>${b.serial_number}</span><b>${b.cycle_count}${b.cycle_limit ? ` / ${b.cycle_limit}` : ''} cycles` +
        `${b.over_cycle_limit ? ' — over limit' : ''}</b></div>`) +
      '<div><b>Pilots</b></div>' + rows(result.pilots, (p) =>
        `<div><span>${p.display_name} ${p.licence_number || ''}</span><b>${p.licence_expires_on || 'no expiry recorded'}</b></div>`);
  }

  function fleetForm(title, fields, submit) {
    modal(title, `<div class="form">${fields.map(([id, label, value]) =>
      `<label>${label}<input id="${id}" value="${value || ''}" /></label>`).join('')}</div>`, [
      { label: 'Cancel' },
      {
        label: 'Save', primary: true, onClick: async () => {
          const values = Object.fromEntries(fields.map(([id]) => [id, $(`#${id}`).value.trim()]));
          const result = await submit(values);
          if (result) { status(`${title}: saved.`); await refreshFleet(); }
        }
      }
    ]);
  }

  function wireFleet() {
    $('#btn-fleet-aircraft').onclick = () => fleetForm('Add aircraft',
      [['f-name', 'Name'], ['f-model', 'Model'], ['f-serial', 'Serial']],
      (v) => tryCall('add_aircraft', ORG, v['f-name'], v['f-model'], v['f-serial']));
    $('#btn-fleet-battery').onclick = () => fleetForm('Add battery',
      [['f-serial', 'Serial'], ['f-cap', 'Capacity mAh', '0'], ['f-limit', 'Cycle limit', '0']],
      (v) => tryCall('add_battery', ORG, v['f-serial'], parseInt(v['f-cap'] || '0', 10) || 0,
                     parseInt(v['f-limit'] || '0', 10) || 0));
    $('#btn-fleet-pilot').onclick = () => fleetForm('Add pilot',
      [['f-name', 'Name'], ['f-licence', 'Licence number'], ['f-expires', 'Licence expires (YYYY-MM-DD)']],
      (v) => tryCall('add_pilot', ORG, v['f-name'], v['f-licence'], v['f-expires']));
    $('#btn-fleet-maintenance').onclick = () => fleetForm('Log maintenance',
      [['f-id', 'Aircraft id'], ['f-kind', 'What was done', 'inspection'], ['f-detail', 'Detail']],
      (v) => tryCall('log_maintenance', parseInt(v['f-id'] || '0', 10), v['f-kind'], v['f-detail'], 'operator'));
    $('#btn-fleet-assign').onclick = () => fleetForm('Assign mission',
      [['f-id', 'Aircraft id'], ['f-mission', 'Mission name', 'Mission']],
      (v) => tryCall('assign_mission_to_aircraft', parseInt(v['f-id'] || '0', 10), v['f-mission']));
  }

  /* ---------- tabs and tools ---------- */

  function switchTab(name) {
    $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
    $$('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${name}`));
    if (name === 'fleet') refreshFleet();
    if (name === 'preflight') runPreflight();
  }

  function setTool(tool) {
    $$('.tool[data-tool]').forEach((button) => button.classList.toggle('active', button.dataset.tool === tool));
    state.map.setTool(tool);
    const hints = {
      pan: 'Pan mode.',
      aoi: 'Click to place area-of-interest vertices; double-click to finish.',
      nofly: 'Click to place no-fly zone vertices; double-click to finish.',
      'measure-distance': 'Click points to measure distance.',
      'measure-area': 'Click points to measure area.',
      delete: 'Selected geometry deleted.'
    };
    status(hints[tool] || tool);
  }

  /* ---------- reports ----------
     Everything here reads and writes the open project: the report is built from its
     datasets, saved missions, findings, reconstruction and audit trail, and written
     into <project>/reports. Before this the default window had no way to produce a
     report at all. */

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  const REPORT_TYPES = ['standard', 'engineering', 'executive', 'defect_only',
    'dataset_quality', 'mission_summary'];

  async function refreshReportInfo() {
    const [found, reports, notes] = await Promise.all([
      tryCall('find_annotations'), tryCall('list_reports'), tryCall('list_notifications', true)]);
    const findings = (found && found.annotations) || [];
    const unreviewed = findings.filter((a) => a.status === 'open' || a.status === 'in_review').length;
    const latest = reports && reports.reports && reports.reports[0];
    $('#report-info').innerHTML = kv([
      ['Findings', `${findings.length} (${unreviewed} not reviewed)`],
      ['Reports', reports ? String(reports.reports.length) : '—'],
      ['Latest', latest ? esc(latest.title) : 'none yet']
    ]);
    const unread = notes ? notes.unread : 0;
    $('#btn-notifications').textContent = unread ? `Notifications (${unread})` : 'Notifications';
  }

  function readinessHtml(r) {
    const rows = [];
    (r.missing || []).forEach((m) => rows.push(`<div><span>Missing</span><b class="no">${esc(m)}</b></div>`));
    (r.warnings || []).forEach((m) => rows.push(`<div><span>Warning</span><b>${esc(m)}</b></div>`));
    (r.notes || []).forEach((m) => rows.push(`<div><span>Note</span><b>${esc(m)}</b></div>`));
    // A refusal says so in words beside its reasons, before Generate is pressed, rather
    // than leaving the missing items to be read as advice.
    if (r.ready === false) {
      rows.unshift('<div><span>Not ready</span><b class="no">Generate will refuse until the missing items exist.</b></div>');
    }
    if (!rows.length) rows.push('<div><span>Ready</span><b class="yes">Nothing missing.</b></div>');
    return rows.join('');
  }

  function reportLinks(report) {
    const kinds = [['html', report.html_path], ['pdf', report.pdf_path], ['docx', report.docx_path]]
      .filter(([, path]) => path);
    return kinds.map(([kind]) =>
      `<button data-open-report="${esc(report.id)}" data-kind="${kind}">${kind.toUpperCase()}</button>`)
      .join(' ') + ` <button data-open-report="${esc(report.id)}" data-kind="folder">Folder</button>`;
  }

  function wireReportLinks() {
    $$('#modal-body button[data-open-report]').forEach((button) => {
      button.onclick = async () => {
        const result = await tryCall('open_report', button.dataset.openReport, button.dataset.kind);
        if (result) status(`Opened ${result.opened}`);
      };
    });
  }

  async function generateReportDialog() {
    const saved = await tryCall('report_templates');
    const last = (saved && saved.templates && saved.templates[0]) || {};
    const options = REPORT_TYPES.map((t) =>
      `<option value="${t}" ${t === (last.report_type || 'standard') ? 'selected' : ''}>${t}</option>`).join('');
    modal('Generate Report', `
      <div class="form">
        <label>Title<input id="rp-title" value="${esc(last.title || 'Inspection report')}" /></label>
        <label>Type<select id="rp-type">${options}</select></label>
        <label>Formats
          <span><input type="checkbox" id="rp-pdf" checked /> PDF
                <input type="checkbox" id="rp-docx" /> Word</span></label>
        <label>Organisation<input id="rp-org" value="${esc(last.organization || '')}" /></label>
        <label>Client<input id="rp-client" value="${esc(last.client || '')}" /></label>
        <label>Prepared by<input id="rp-author" value="${esc(last.author || '')}" /></label>
        <label>Logo file<span class="file-row"><input id="rp-logo" value="${esc(last.logo_path || '')}" />
          <button class="mini-btn" id="rp-logo-pick">...</button></span></label>
        <label>Summary (optional)<textarea id="rp-summary" rows="2"></textarea></label>
        <label>Recommendations, one per line (optional)<textarea id="rp-recs" rows="3"></textarea></label>
      </div>
      <div class="info" id="rp-readiness">Checking what the report needs…</div>`,
      [{ label: 'Cancel' }, {
        label: 'Generate', primary: true, onClick: async () => {
          const formats = ['html'];
          if ($('#rp-pdf').checked) formats.push('pdf');
          if ($('#rp-docx').checked) formats.push('docx');
          const settings = {
            title: $('#rp-title').value.trim() || 'Inspection report',
            report_type: $('#rp-type').value,
            organization: $('#rp-org').value.trim(),
            client: $('#rp-client').value.trim(),
            author: $('#rp-author').value.trim(),
            logo_path: $('#rp-logo').value.trim(),
            formats
          };
          status('Generating report…');
          const result = await tryCall('generate_report', '', settings.title, settings.report_type,
            settings.author, settings.organization, settings.client, settings.logo_path, formats,
            $('#rp-summary').value.trim(), $('#rp-recs').value);
          if (!result) return true;
          await tryCall('save_report_template', 'last used', settings);
          await refreshReportInfo();
          status(`Report written to ${result.output_dir}`);
          showReportResult(result);
          return true;
        }
      }]);
    const check = async () => {
      const formats = ['html'];
      if ($('#rp-pdf').checked) formats.push('pdf');
      if ($('#rp-docx').checked) formats.push('docx');
      const r = await tryCall('report_readiness', '', $('#rp-type').value, formats);
      if (r && $('#rp-readiness')) $('#rp-readiness').innerHTML = readinessHtml(r);
    };
    $('#rp-type').onchange = check;
    $('#rp-pdf').onchange = check;
    $('#rp-docx').onchange = check;
    $('#rp-logo-pick').onclick = async () => {
      const picked = await tryCall('pick_file', ['png', 'jpg', 'jpeg']);
      if (picked && picked.path) $('#rp-logo').value = picked.path;
    };
    await check();
  }

  function showReportResult(result) {
    modal('Report Written', `<div class="info">${kv([
      ['Title', esc(result.title)],
      ['Type', esc(result.report_type)],
      ['Findings', String(result.findings)],
      ['Folder', esc(result.output_dir)]
    ])}${(result.warnings || []).map((w) => `<div><span>Warning</span><b>${esc(w)}</b></div>`).join('')}
      <div><span>Open</span><b>${reportLinks(result)}</b></div></div>`);
    wireReportLinks();
  }

  async function listReportsDialog() {
    const result = await tryCall('list_reports');
    if (!result) return;
    const rows = result.reports.map((r) =>
      `<div><span>${esc(r.created_at.slice(0, 19))} · ${esc(r.title)} (${esc(r.report_type)})</span>
       <b>${reportLinks(r)}</b></div>`).join('');
    modal('Generated Reports', `<div class="info">${rows || 'No reports generated in this project yet.'}</div>`);
    wireReportLinks();
  }

  async function exportFindings(format) {
    const saved = await tryCall('report_templates');
    const last = (saved && saved.templates && saved.templates[0]) || {};
    const result = await tryCall('export_report', format, last.title || 'Inspection findings',
      last.organization || '', last.client || '', last.author || '', '', last.logo_path || '');
    if (!result) return;
    status(`${format.toUpperCase()} with ${result.findings} finding(s) written to ${result.path}`);
    modal('Findings Exported', `<div class="info">${kv([['File', esc(result.path)],
      ['Findings', String(result.findings)]])}</div>`,
      [{ label: 'Close' }, { label: 'Open', primary: true, onClick: () => tryCall('open_path', result.path) }]);
    await refreshReportInfo();
  }

  async function recordFindingDialog() {
    const datasets = await tryCall('list_datasets');
    const hint = datasets && datasets.datasets && datasets.datasets.length
      ? `Images from: ${datasets.datasets.map((d) => esc(d.name)).join(', ')}`
      : 'Import an imagery folder first (File > Import Imagery Folder).';
    modal('Record Finding', `
      <div class="form">
        <label>Image (file name or path)<span class="file-row"><input id="fd-image" placeholder="DSC00229.JPG" />
          <button class="mini-btn" id="fd-pick">...</button></span></label>
        <label>What is it<input id="fd-label" placeholder="crack, spalling, corrosion…" /></label>
        <label>Severity<select id="fd-severity">
          <option>critical</option><option>high</option><option selected>medium</option>
          <option>low</option><option>info</option></select></label>
        <label>Box x0,y0,x1,y1 in pixels (optional)<input id="fd-bbox" placeholder="whole photo" /></label>
        <label>Note<textarea id="fd-note" rows="2"></textarea></label>
      </div><div class="info">${hint}</div>`,
      [{ label: 'Cancel' }, {
        label: 'Record', primary: true, onClick: async () => {
          const bboxText = $('#fd-bbox').value.trim();
          let bbox = null;
          if (bboxText) {
            bbox = bboxText.split(/[ ,]+/).map(Number);
            if (bbox.length !== 4 || bbox.some((v) => !Number.isFinite(v))) {
              status('The box needs four numbers: x0, y0, x1, y1.');
              return true;
            }
          }
          const result = await tryCall('record_finding', $('#fd-image').value.trim(),
            $('#fd-label').value.trim(), $('#fd-severity').value, $('#fd-note').value.trim(), bbox);
          if (!result) return true;
          status(`Finding recorded: ${result.finding.label} (${result.finding.severity}).`);
          await refreshReportInfo();
        }
      }]);
    $('#fd-pick').onclick = async () => {
      const picked = await tryCall('pick_file', ['jpg', 'jpeg', 'png', 'tif', 'tiff']);
      if (picked && picked.path) $('#fd-image').value = picked.path;
    };
  }

  async function reviewDialog() {
    const result = await tryCall('find_annotations');
    if (!result) return;
    const rows = result.annotations.map((a) => {
      const source = a.created_by === 'user' ? 'inspector' : esc(a.created_by);
      const image = esc(String(a.source_id || '').split(/[\\/]/).pop());
      return `<div><span>${esc(a.label)} · ${esc(a.severity)} · ${image} · ${source}
        <br/><small>${esc(a.status)} — ${esc((a.note || '').split('\n')[0])}</small></span>
        <b><button data-review="${esc(a.id)}" data-decision="accept">Accept</button>
           <button data-review="${esc(a.id)}" data-decision="reject">Reject</button>
           <button data-review="${esc(a.id)}" data-decision="flag">Flag</button></b></div>`;
    }).join('');
    modal('Findings and Review', `<div class="info">${rows || 'No findings recorded in this project.'}</div>`);
    $$('#modal-body button[data-review]').forEach((button) => {
      button.onclick = async () => {
        const done = await tryCall('review_finding', button.dataset.review, button.dataset.decision, 'operator');
        if (done) {
          status(`Finding ${done.label}: ${done.status}.`);
          await refreshReportInfo();
          await reviewDialog();
        }
      };
    });
  }

  async function provenanceDialog() {
    const result = await tryCall('verify_provenance');
    if (!result) return;
    const rows = Object.entries(result.results || {}).map(([name, r]) =>
      `<div><span>${esc(name)}</span><b class="${r.ok ? 'yes' : 'no'}">${esc(r.status)}</b></div>`).join('');
    modal('Provenance', `<div class="info">${kv([
      ['Folder', esc(result.directory)],
      ['Verified', `${result.verified.length} of ${result.total}`],
      ['Unattributed', String(result.unattributed.length)],
      ['Modified', String(result.modified.length)]
    ])}${rows}</div>`,
      result.unattributed.length
        ? [{ label: 'Close' }, {
          label: 'Record now', onClick: async () => {
            const done = await tryCall('record_provenance');
            if (done) status(`Recorded provenance for ${done.artifact_count} file(s).`);
          }
        }]
        : [{ label: 'Close' }]);
  }

  async function notificationsDialog() {
    const result = await tryCall('list_notifications');
    if (!result) return;
    const rows = result.notifications.map((n) =>
      `<div><span>${esc((n.created_utc || '').slice(0, 19))} · ${esc(n.title)}${n.read ? '' : ' •'}</span>
       <b class="${n.level === 'error' ? 'no' : ''}">${esc(n.detail || '')}</b></div>`).join('');
    modal('Notifications', `<div class="info">${rows || 'No notifications.'}</div>`,
      [{ label: 'Close' }, {
        label: 'Mark all read', onClick: async () => {
          await tryCall('mark_all_notifications_read');
          await refreshReportInfo();
        }
      }]);
  }

  /* ---------- menu dispatch ---------- */

  const menuHandlers = {
    'project.new': () => modal('New Project', `
        <div class="form">
          <label>Name<input id="dlg-name" placeholder="Bridge inspection 2026" /></label>
          <label>Root folder (optional)<input id="dlg-root" placeholder="leave blank for default" /></label>
        </div>`,
      [{ label: 'Cancel' }, {
        label: 'Create', primary: true, onClick: async () => {
          const name = $('#dlg-name').value.trim();
          if (!name) { status('Project name cannot be empty.'); return true; }
          const result = await tryCall('create_project', name, $('#dlg-root').value.trim());
          if (result) { await refreshProject(); status(`Project "${name}" created.`); }
        }
      }]),

    'project.open': async () => {
      const result = await tryCall('list_projects');
      if (!result) return;
      const rows = result.projects.map((p) =>
        `<div><span>${p.name}</span><b><button data-pid="${p.id}">Open</button></b></div>`).join('');
      modal('Open Project', `<div class="info">${rows || 'No projects yet.'}</div>`);
      $$('#modal-body button[data-pid]').forEach((button) => {
        button.onclick = async () => {
          await tryCall('set_active_project', parseInt(button.dataset.pid, 10));
          await refreshProject();
          await refreshLayers();
          closeModal();
        };
      });
    },

    'project.reveal': async () => {
      const result = await tryCall('get_project');
      if (result && result.project && result.project.root_dir) await tryCall('open_path', result.project.root_dir);
      else status('This project has no root folder set.');
    },

    'data.import_imagery': () => pickDataset(),
    'data.import_terrain': async () => {
      const picked = await tryCall('pick_file', ['Terrain (*.tif;*.tiff;*.asc;*.csv)']);
      if (picked && picked.path) {
        const result = await tryCall('set_terrain_source', picked.path);
        if (result) { $('#mission-terrain').value = result.path; status('Terrain source set. Replan to apply.'); }
      }
    },
    'data.import_vector': () => importLayer(['GeoJSON (*.geojson;*.json)']),
    'data.import_raster': () => importLayer(['GeoTIFF (*.tif;*.tiff)']),

    'mission.plan': () => planMission(),
    'mission.clear_aoi': () => { state.map.clearRole('aoi'); tryCall('set_aoi', []); status('Area of interest cleared.'); },
    'mission.clear_nofly': () => { state.map.clearRole('nofly'); tryCall('set_no_fly_zones', []); status('No-fly zones cleared.'); },
    'mission.settings': () => switchTab('console'),
    'mission.save': async () => {
      const result = await tryCall('save_mission', 'mission', 'Saved from menu');
      if (result) status(`Mission version ${result.version && result.version.version_num} saved.`);
    },
    'mission.history': async () => {
      const result = await tryCall('list_mission_versions');
      if (!result) return;
      const rows = result.versions.map((v) => `<div><span>v${v.version_num} · ${v.created_at}</span><b>${v.note || ''}</b></div>`).join('');
      modal('Mission History', `<div class="info">${rows || 'No saved mission versions.'}</div>`);
    },
    'mission.export_all': () => exportMission(null),
    'mission.export': (payload) => exportMission([payload.format]),

    'fly.connect': () => connectDialog(),
    'fly.disconnect': async () => {
      const result = await tryCall('disconnect_vehicle');
      if (result) { updateVehicleChip(result.vehicle); status('Vehicle disconnected.'); }
    },
    'fly.upload': async () => {
      /* tryCall returns null on ok:false and puts the vehicle's refusal in the status
         bar; only an acknowledged upload reaches the success line. */
      const result = await tryCall('upload_mission');
      if (!result) return;
      const upload = result.upload || {};
      const extras = ['fence', 'rally'].filter((k) => upload[k] && upload[k].success)
        .map((k) => `${upload[k].count} ${k}`);
      status(`Vehicle accepted ${result.items} mission items${extras.length ? ` and ${extras.join(', ')} items` : ''}.`);
    },
    'fly.command': async (payload) => {
      const result = await tryCall('vehicle_command', payload.command);
      if (result) status(`Command ${payload.command}: ${(result.result && result.result.message) || 'accepted'}`);
    },

    'analysis.pipeline': async () => {
      const result = await tryCall('run_pipeline', { engine: $('#recon-engine').value });
      if (result) {
        status('Pipeline started.');
        pollJob(result.job_id, (job) => {
          refreshLayers();
          // Say what the run produced, including the parts it did not: a failed
          // reconstruction and heuristic regions left out of the findings.
          const r = job.result || {};
          const parts = [`${r.findings_created || 0} model finding(s) to review`];
          if (r.heuristic_regions_not_recorded) {
            parts.push(`${r.heuristic_regions_not_recorded} heuristic region(s) not recorded as findings`);
          }
          if (r.reconstruction_status === 'failed') parts.push('reconstruction failed (detection results kept)');
          status(`Pipeline done: ${parts.join('; ')}.`);
        });
      }
    },
    'analysis.models': async () => {
      const result = await tryCall('model_status');
      if (!result) return;
      const esc = (text) => String(text == null ? '' : text).replace(/[&<>"]/g,
        (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
      const rows = result.models.map((m) =>
        `<div><span>${esc(m.key)}</span><b class="${m.exists ? 'yes' : 'no'}">${m.exists ? 'available' : 'missing'}</b></div>`).join('');
      // Per detector, not one global count: nine models installed said nothing about
      // metal and solar detection being heuristic on this machine.
      const detectors = (result.detectors || []).map((d) =>
        `<div><span>${esc(d.detector)}${d.model_key ? ` (${esc(d.model_key)})` : ''}</span>` +
        `<b class="${d.backend === 'model' ? 'yes' : 'no'}">${d.backend === 'heuristic' ? 'HEURISTIC' : esc(d.backend)}</b></div>`).join('');
      const heuristic = result.heuristic_detectors || [];
      modal('Model Manager',
        `<div class="info"><div><span>Available</span><b>${result.available} of ${result.total}</b></div>${rows}</div>` +
        `<div class="info"><div><span>Detector</span><b>Backend</b></div>${detectors}</div>` +
        (heuristic.length
          ? `<div class="info">${esc(heuristic.join(', '))}: no trained model answers on this machine, so these use classical image rules. Their results are labelled <b>heuristic</b> and carry no confidence.</div>`
          : ''));
    },

    'recon.run': (payload) => runReconstruction(payload.engine),
    'recon.settings': () => status('Reconstruction settings are in the right dock.'),
    'recon.reveal': async () => {
      const result = await tryCall('get_project');
      if (result && result.project && result.project.root_dir) await tryCall('open_path', result.project.root_dir);
    },

    'view.panel': (payload) => switchTab(payload.panel === 'layers' ? 'attributes' : payload.panel),
    'view.basemap': (payload) => { $('#basemap-select').value = payload.basemap; chooseBasemap(payload.basemap); },
    'view.cache_tiles': () => cacheVisibleMap(),
    'view.zoom_aoi': () => {
      const geometry = state.map.getGeometry();
      if (geometry.aoi) state.map.fitToCoords(geometry.aoi);
      else status('No area of interest drawn.');
    },
    'view.zoom_mission': () => {
      if (state.map._missionData) state.map.fitTo(state.map._missionData);
      else status('No mission planned.');
    },

    'tools.measure': (payload) => setTool(payload.mode === 'area' ? 'measure-area' : 'measure-distance'),
    'tools.capabilities': () => showCapabilities(true),
    'tools.audit': async () => {
      const result = await tryCall('audit_log', 100);
      if (!result) return;
      // The payload is what the event was about; showing only its type said that
      // something happened without saying what.
      const rows = result.events.map((e) => {
        let detail = '';
        try {
          detail = Object.entries(JSON.parse(e.payload_json || '{}'))
            .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(', ');
        } catch (err) { detail = e.payload_json || ''; }
        return `<div><span>${esc(e.created_at)} · ${esc(e.event_type)}</span><b>${esc(detail)}</b></div>`;
      }).join('');
      modal('Audit Log', `<div class="info">${rows || 'No events recorded.'}</div>`,
        [{ label: 'Close' }, { label: 'Export CSV', onClick: () => menuHandlers['report.audit_export']() }]);
    },

    'report.generate': () => generateReportDialog(),
    'report.export': (payload) => exportFindings(payload.format || 'pdf'),
    'report.list': () => listReportsDialog(),
    'report.finding': () => recordFindingDialog(),
    'report.review': () => reviewDialog(),
    'report.provenance': () => provenanceDialog(),
    'report.notifications': () => notificationsDialog(),
    'report.audit_export': async () => {
      const result = await tryCall('export_audit_log', 'csv');
      if (result) status(`Audit log (${result.events} events) written to ${result.path}`);
    },

    'help.about': () => modal('About OpenDroneKit',
      `<div class="info">
        <div><span>Purpose</span><b>Offline drone inspection &amp; geospatial toolkit</b></div>
        <div><span>Planning</span><b>16 mission templates with geofence, no-fly, terrain and wind</b></div>
        <div><span>Reconstruction</span><b>COLMAP SfM with georeferenced GeoTIFF output</b></div>
        <div><span>Export</span><b>QGC .plan, WPL, DJI WPML, Litchi, KML, GeoJSON</b></div>
      </div>`),
    'help.shortcuts': () => modal('Keyboard Shortcuts',
      `<div class="info">
        <div><span>Esc</span><b>Pan mode / close dialog</b></div>
        <div><span>A</span><b>Draw area of interest</b></div>
        <div><span>N</span><b>Draw no-fly zone</b></div>
        <div><span>M</span><b>Measure distance</b></div>
        <div><span>Delete</span><b>Remove selected geometry</b></div>
        <div><span>Ctrl+P</span><b>Plan mission</b></div>
      </div>`)
  };

  async function importLayer(filters) {
    const picked = await tryCall('pick_file', filters);
    if (!picked || !picked.path) return;
    const result = await tryCall('add_layer_from_file', picked.path);
    if (!result) return;
    await refreshLayers();
    // Frame what was just imported. The map stayed at the world view, so an imported
    // raster was a dot nobody could find -- and an area of interest drawn at that zoom
    // covered a continent.
    const added = (state.layers || []).find((l) => l.id === result.layer.id) || result.layer;
    if (added.bounds_lonlat && added.bounds_lonlat.length === 4) {
      state.map.fitTo(added.bounds_lonlat);
      status(`Layer added and framed: ${added.name}`);
    } else {
      status(`Layer added: ${added.name}. It has no usable extent${added.crs_epsg ? '' : ' (no CRS)'}, so the map did not move.`);
    }
  }

  async function pickDataset() {
    const picked = await tryCall('pick_folder');
    if (!picked || !picked.path) return;
    status('Reading dataset EXIF...');
    const result = await tryCall('import_dataset', picked.path);
    if (!result) return;
    $('#dataset-path').value = picked.path;
    await tryCall('set_active_dataset', picked.path);
    const meta = result.dataset.metadata || {};
    $('#dataset-info').innerHTML = kv([
      ['Images', meta.image_count || 0],
      ['Geotagged', `${meta.geotagged_count || 0} / ${meta.image_count || 0}`],
      ['Camera', meta.camera || 'unknown'],
      ['Suggested CRS', meta.suggested_epsg ? 'EPSG:' + meta.suggested_epsg : 'none']
    ]);
    await refreshLayers();
    if (meta.bounds_lonlat) state.map.fitTo(meta.bounds_lonlat);
    status(`Dataset imported: ${meta.image_count} images, ${meta.geotagged_count} geotagged.`);
  }

  async function runReconstruction(engine) {
    const result = await tryCall('run_reconstruction', {
      engine: engine || $('#recon-engine').value,
      profile: $('#recon-profile').value
    });
    if (!result) return;
    status('Reconstruction started. Progress is in the Jobs tab.');
    pollJob(result.job_id, async (job) => {
      await refreshLayers();
      // Frame the new orthomosaic, which is what the operator ran this to see.
      const ortho = (state.layers || []).find((l) => /ortho/i.test(l.name) && l.bounds_lonlat && l.bounds_lonlat.length === 4);
      if (ortho) state.map.fitTo(ortho.bounds_lonlat);
      const payload = job.result || {};
      const warnings = (payload.warnings || []).map((w) => `<div><span>warning</span><b>${w}</b></div>`).join('');
      modal('Reconstruction Complete', `<div class="info">${kv([
        ['Engine', payload.engine],
        ['Registered', `${payload.registered_images} / ${payload.frame_count}`],
        ['Points', (payload.total_points || 0).toLocaleString()],
        ['Reprojection error', payload.reprojection_error_px ? payload.reprojection_error_px.toFixed(2) + ' px' : 'n/a'],
        ['CRS', payload.crs_epsg ? 'EPSG:' + payload.crs_epsg : 'not georeferenced'],
        ['Geo RMSE', payload.geo_rmse_m ? payload.geo_rmse_m.toFixed(2) + ' m' : 'n/a'],
        ['Ground sample', payload.ground_sample_distance_m ? payload.ground_sample_distance_m.toFixed(3) + ' m/px' : 'n/a']
      ])}${warnings}</div>`);
    });
  }

  /* ---------- place search ---------- */

  /* Searching is the one action that can reach outside this machine, so it only ever
     runs on an explicit submit -- never as-you-type -- and the provider note tells the
     operator where the query went. */
  function wirePlaceSearch() {
    const input = $('#place-search');
    const button = $('#btn-place-search');
    const list = $('#place-results');
    if (!input || !button || !list) return;

    function hide() {
      list.hidden = true;
      list.innerHTML = '';
    }

    function render(result) {
      const places = result.results || [];
      const esc = (text) => String(text == null ? '' : text).replace(/[&<>"]/g,
        (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
      if (!places.length) {
        list.innerHTML = `<li class="search-empty">${esc(result.error || 'No matching place found.')}</li>`;
        list.hidden = false;
        return;
      }
      // Offline, the remote search did not happen; say so above what was found locally.
      const warning = result.error ? `<li class="search-empty">${esc(result.error)}</li>` : '';
      list.innerHTML = warning + places.map((place, index) => `
        <li data-index="${index}" title="${esc(place.name)}">
          <b>${esc(place.name)}</b>
          <span>${esc(place.kind || place.source || '')} · ${place.lat.toFixed(5)}, ${place.lon.toFixed(5)}</span>
        </li>`).join('') +
        `<li class="search-note">${esc(result.provider_note || '')}</li>`;
      list.hidden = false;

      list.querySelectorAll('li[data-index]').forEach((row) => {
        row.onclick = () => {
          const place = places[Number(row.dataset.index)];
          state.map.goToPlace(place);
          state.map.markPlace(place);
          status(`Moved to ${place.name}`);
          hide();
        };
      });
    }

    async function run() {
      const query = input.value.trim();
      if (!query) { hide(); return; }
      button.disabled = true;
      const previous = button.textContent;
      button.textContent = '...';
      try {
        const result = await tryCall('search_places', query, '');
        if (result) render(result);
      } finally {
        button.disabled = false;
        button.textContent = previous;
      }
    }

    button.onclick = run;
    input.onkeydown = (event) => {
      if (event.key === 'Enter') { event.preventDefault(); run(); }
      if (event.key === 'Escape') { hide(); input.blur(); }
    };
    document.addEventListener('click', (event) => {
      if (!list.contains(event.target) && event.target !== input && event.target !== button) hide();
    });
  }

  async function exportMission(formats) {
    const result = await tryCall('export_mission', formats, '');
    if (!result) return;
    const rows = Object.entries(result.written)
      .map(([k, v]) => `<div><span>${k}</span><b>${v.startsWith('failed') ? v : v.split(/[\\/]/).pop()}</b></div>`).join('');
    modal('Mission Exported', `<div class="info">${rows}<div><span>Folder</span><b>${result.directory}</b></div></div>`,
      [{ label: 'Open Folder', onClick: () => tryCall('open_path', result.directory) }, { label: 'Close', primary: true }]);
  }

  async function showCapabilities(asModal) {
    const result = await tryCall('capabilities');
    if (!result) return;
    const c = result.capabilities;
    const html = kv([
      ['pycolmap', flag(c.pycolmap)],
      ['CUDA (COLMAP)', flag(c.pycolmap_cuda)],
      // The reason matters as much as the flag: a CUDA build that cannot run on this
      // GPU used to read "yes" here and then fail every dense run.
      ['Dense MVS', flag(c.dense_stereo) + (c.dense_stereo_reason
        ? ` <small class="muted">${String(c.dense_stereo_reason).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]))}</small>`
        : '')],
      ['Open3D', flag(c.open3d)],
      ['rasterio', flag(c.geo && c.geo.rasterio)],
      ['pyproj', flag(c.geo && c.geo.pyproj)],
      ['pymavlink', flag(c.pymavlink)],
      ['Torch CUDA', flag(c.cuda)],
      ['GPU', c.gpu || 'none']
    ]);
    $('#capabilities').innerHTML = html;
    if (asModal) modal('Environment Capabilities', `<div class="info">${html}</div>`);
  }

  async function refreshProject() {
    const result = await tryCall('get_project');
    if (!result) return;
    const project = result.project || {};
    $('#project-info').innerHTML = kv([
      ['Name', project.name || 'unnamed'],
      ['Id', project.id],
      ['Root', project.root_dir || 'default'],
      ['Sync', project.sync_status || 'offline']
    ]);
    if ($('#report-info')) await refreshReportInfo();
  }

  /* ---------- offline basemap ---------- */

  function chooseBasemap(key) {
    state.map.setBasemap(key);
    if (key === 'offline') reportOfflineCache();
  }

  /* Say what the offline basemap can show, so an empty cache reads as empty rather
     than as a broken map. */
  async function reportOfflineCache() {
    const result = await tryCall('basemap_cache_status');
    if (!result) return;
    const sat = (result.providers || {}).satellite;
    status(sat && sat.tiles
      ? `Offline cache: ${sat.tiles} satellite tiles at zoom ${sat.zoom_levels.join(', ')}. Uncached areas stay empty.`
      : 'The offline tile cache is empty. While online, frame the site and press "Cache view".');
  }

  /* Download the tiles for what is on screen, a few zoom levels deeper, so the map
     still shows this site with no signal. Explicit: panning never downloads. */
  async function cacheVisibleMap() {
    const view = state.map.visibleBounds();
    const minZoom = Math.max(0, Math.floor(view.zoom));
    const maxZoom = Math.min(19, minZoom + 3);
    const result = await tryCall('cache_basemap_area', view.west, view.south, view.east, view.north, minZoom, maxZoom, 'satellite');
    if (!result) return;
    status(`Caching ${result.tiles} satellite tiles (z${minZoom}-${maxZoom}) for offline use…`);
    pollJob(result.job_id, (job) => {
      const r = job.result || {};
      status(`Offline cache: ${r.fetched || 0} downloaded, ${r.already_cached || 0} already cached, ` +
             `${r.failed || 0} failed of ${r.requested || result.tiles}.` +
             (r.failed ? ' Failed tiles stay empty on the offline map.' : ''));
    });
  }

  /* ---------- bootstrap ---------- */

  function wireEvents() {
    $$('.tool[data-tool]').forEach((button) => { button.onclick = () => setTool(button.dataset.tool); });
    $$('.tab').forEach((tab) => { tab.onclick = () => switchTab(tab.dataset.tab); });

    $('#basemap-select').onchange = (e) => chooseBasemap(e.target.value);
    $('#btn-cache-tiles').onclick = cacheVisibleMap;
    wirePlaceSearch();
    $('#btn-zoom-aoi').onclick = menuHandlers['view.zoom_aoi'];
    $('#btn-zoom-mission').onclick = menuHandlers['view.zoom_mission'];
    $('#btn-plan').onclick = planMission;
    $('#mission-template').onchange = showTemplateOptions;
    $('#btn-connect').onclick = connectDialog;
    $('#btn-upload').onclick = menuHandlers['fly.upload'];
    $('#btn-preflight').onclick = () => switchTab('preflight');
    wireFleet();
    $('#btn-dataset').onclick = pickDataset;
    $('#btn-terrain').onclick = menuHandlers['data.import_terrain'];
    $('#btn-add-layer').onclick = () => importLayer(['Geodata (*.tif;*.tiff;*.geojson;*.json)']);
    $('#btn-recon').onclick = () => runReconstruction($('#recon-engine').value);
    $('#btn-pipeline').onclick = menuHandlers['analysis.pipeline'];
    $('#btn-report').onclick = menuHandlers['report.generate'];
    $('#btn-finding').onclick = menuHandlers['report.finding'];
    $('#btn-review').onclick = menuHandlers['report.review'];
    $('#btn-reports').onclick = menuHandlers['report.list'];
    $('#btn-notifications').onclick = menuHandlers['report.notifications'];
    $('#modal-backdrop').onclick = (e) => { if (e.target.id === 'modal-backdrop') closeModal(); };

    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'Escape') { closeModal(); setTool('pan'); }
      else if (e.key.toLowerCase() === 'a') setTool('aoi');
      else if (e.key.toLowerCase() === 'n') setTool('nofly');
      else if (e.key.toLowerCase() === 'm') setTool('measure-distance');
      else if (e.key === 'Delete') setTool('delete');
      else if (e.ctrlKey && e.key.toLowerCase() === 'p') { e.preventDefault(); planMission(); }
    });
  }

  function initMap() {
    state.map = new OdkMap('map', {
      onReady: () => { state.ready = true; status('Map ready. Draw an area of interest to plan a mission.'); },
      onBasemapFallback: (message) => { $('#basemap-select').value = 'offline'; status(message); reportOfflineCache(); },
      onCursor: (lon, lat) => {
        $('#status-coords').textContent = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
        $('#status-scale').textContent = `z${state.map.map.getZoom().toFixed(1)}`;
      },
      onMeasure: (text) => { $('#status-message').textContent = `${text} (map estimate; double-click to measure on the survey)`; },
      /* The running figure above is a spherical estimate from the basemap. On
         double-click the shape goes to the application, which measures it on the
         project's own orthomosaic/DSM -- with the over-the-ground length for a line --
         and keeps it in the project's measurement history. */
      onMeasureDone: async (kind, points) => {
        const result = await tryCall('measure_drawn', kind, points);
        if (!result) return;
        const m = result.measurement || {};
        const unit = kind === 'area' ? 'm²' : 'm';
        const ground = result.surface_length_m !== undefined
          ? `, ${Number(result.surface_length_m).toFixed(2)} m over the ground`
          : (result.surface_note ? ` (${result.surface_note})` : '');
        status(`${kind === 'area' ? 'Area' : 'Distance'} on the survey: `
          + `${Number(result.value_m ?? m.value).toFixed(2)} ${unit}${ground} (EPSG:${m.epsg}). Saved to measurements.`);
      },
      // The zoom readout changed only when the cursor moved, so after "zoom to layer"
      // it still said z2 over a map at z17.
      onMove: (zoom) => { $('#status-scale').textContent = `z${zoom.toFixed(1)}`; },
      onGeometryChange: async (geometry) => {
        if (geometry.aoi) {
          const result = await tryCall('set_aoi', geometry.aoi);
          if (result) status(`Area of interest: ${result.vertices} vertices, ${(result.area_m2 / 10000).toFixed(3)} ha, UTM EPSG:${result.suggested_epsg}`);
        }
        await tryCall('set_no_fly_zones', geometry.nofly);
      }
    });
  }

  window.odk = {
    onMenu: (message) => {
      const handler = menuHandlers[message.action];
      if (handler) handler(message.payload || {});
      else status(`Unhandled menu action: ${message.action}`);
    }
  };

  /* Start once the Python bridge exists.

     pywebview fires 'pywebviewready' once. If it fires before this script has added
     its listener -- which happens when the page loads slowly, measured with the network
     unreachable -- a listener alone never runs: the bridge is there, and the window
     sits on "Ready" with no map for the whole session. So check for the bridge first,
     listen as well, and start exactly once. */
  let started = false;
  async function start() {
    if (started || !window.pywebview || !window.pywebview.api
        || typeof window.pywebview.api.get_state !== 'function') return;
    started = true;
    initMap();
    wireEvents();
    setTool('pan');
    await refreshProject();
    await loadMissionTemplates();
    await showCapabilities(false);
    await refreshLayers();
    await renderJobs();
    status('OpenDroneKit ready.');
  }
  window.addEventListener('pywebviewready', start);
  start();
})();
