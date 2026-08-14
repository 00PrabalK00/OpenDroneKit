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

  function missionOptions() {
    return {
      template: $('#mission-template').value,
      altitude_m: parseFloat($('#mission-altitude').value),
      speed_m_s: parseFloat($('#mission-speed').value),
      front_overlap_pct: parseFloat($('#mission-front').value),
      side_overlap_pct: parseFloat($('#mission-side').value),
      gimbal_tilt_deg: parseFloat($('#mission-gimbal').value),
      dwell_s: parseFloat($('#mission-dwell').value),
      min_altitude_m: parseFloat($('#mission-minalt').value),
      max_altitude_m: parseFloat($('#mission-maxalt').value),
      rth_altitude_m: parseFloat($('#mission-rthalt').value),
      wind_speed_m_s: parseFloat($('#mission-wind').value)
    };
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
    const result = await tryCall('plan_mission', missionOptions());
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
        ['Altitude', s.altitude_m.toFixed(0) + ' m']
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
            <option value="mavlink">MAVLink (real autopilot)</option>
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
              if (result.vehicle.connected) startTelemetry();
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
      if (!data.connected) {
        grid.innerHTML = `<div><span>status</span><b>${data.reason || 'disconnected'}</b></div>`;
        return;
      }
      grid.innerHTML = Object.entries(data)
        .filter(([k]) => k !== 'connected')
        .map(([k, v]) => `<div><span>${k}</span><b>${typeof v === 'number' ? v.toFixed(3) : v}</b></div>`)
        .join('');
    }, 1000);
  }

  /* ---------- tabs and tools ---------- */

  function switchTab(name) {
    $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
    $$('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${name}`));
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
      const result = await tryCall('upload_mission');
      if (result) status(`Uploaded ${result.items} mission items.`);
    },
    'fly.command': async (payload) => {
      const result = await tryCall('vehicle_command', payload.command);
      if (result) status(`Command ${payload.command}: ${JSON.stringify(result.result)}`);
    },

    'analysis.pipeline': async () => {
      const result = await tryCall('run_pipeline', { engine: $('#recon-engine').value });
      if (result) { status('Pipeline started.'); pollJob(result.job_id, () => refreshLayers()); }
    },
    'analysis.models': async () => {
      const result = await tryCall('model_status');
      if (!result) return;
      const rows = result.models.map((m) =>
        `<div><span>${m.key}</span><b class="${m.exists ? 'yes' : 'no'}">${m.exists ? 'available' : 'missing'}</b></div>`).join('');
      modal('Model Manager',
        `<div class="info"><div><span>Available</span><b>${result.available} of ${result.total}</b></div>${rows}</div>` +
        (result.available === 0
          ? '<div class="info">No trained weights are installed, so detection falls back to classical image processing. Results are labelled <b>heuristic</b>, not AI.</div>'
          : ''));
    },

    'recon.run': (payload) => runReconstruction(payload.engine),
    'recon.settings': () => status('Reconstruction settings are in the right dock.'),
    'recon.reveal': async () => {
      const result = await tryCall('get_project');
      if (result && result.project && result.project.root_dir) await tryCall('open_path', result.project.root_dir);
    },

    'view.panel': (payload) => switchTab(payload.panel === 'layers' ? 'attributes' : payload.panel),
    'view.basemap': (payload) => { $('#basemap-select').value = payload.basemap; state.map.setBasemap(payload.basemap); },
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
      const rows = result.events.map((e) => `<div><span>${e.created_at} · ${e.event_type}</span><b></b></div>`).join('');
      modal('Audit Log', `<div class="info">${rows || 'No events recorded.'}</div>`);
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
    if (result) { await refreshLayers(); status(`Layer added: ${result.layer.name}`); }
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

  async function exportMission(formats) {
    const result = await tryCall('export_mission', formats, '');
    if (!result) return;
    const rows = Object.entries(result.written)
      .map(([k, v]) => `<div><span>${k}</span><b>${v.startsWith('failed') ? v : 'written'}</b></div>`).join('');
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
      ['Dense MVS', flag(c.dense_stereo)],
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
  }

  /* ---------- bootstrap ---------- */

  function wireEvents() {
    $$('.tool[data-tool]').forEach((button) => { button.onclick = () => setTool(button.dataset.tool); });
    $$('.tab').forEach((tab) => { tab.onclick = () => switchTab(tab.dataset.tab); });

    $('#basemap-select').onchange = (e) => state.map.setBasemap(e.target.value);
    $('#btn-zoom-aoi').onclick = menuHandlers['view.zoom_aoi'];
    $('#btn-zoom-mission').onclick = menuHandlers['view.zoom_mission'];
    $('#btn-plan').onclick = planMission;
    $('#btn-connect').onclick = connectDialog;
    $('#btn-upload').onclick = menuHandlers['fly.upload'];
    $('#btn-dataset').onclick = pickDataset;
    $('#btn-terrain').onclick = menuHandlers['data.import_terrain'];
    $('#btn-add-layer').onclick = () => importLayer(['Geodata (*.tif;*.tiff;*.geojson;*.json)']);
    $('#btn-recon').onclick = () => runReconstruction($('#recon-engine').value);
    $('#btn-pipeline').onclick = menuHandlers['analysis.pipeline'];
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
      onCursor: (lon, lat) => {
        $('#status-coords').textContent = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
        $('#status-scale').textContent = `z${state.map.map.getZoom().toFixed(1)}`;
      },
      onMeasure: (text) => { $('#status-message').textContent = text; },
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

  window.addEventListener('pywebviewready', async () => {
    initMap();
    wireEvents();
    setTool('pan');
    await refreshProject();
    await showCapabilities(false);
    await refreshLayers();
    await renderJobs();
    status('OpenDroneKit ready.');
  });
})();
