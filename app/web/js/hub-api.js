/* Dependency-free REST adapter for the browser Hub and Node integration tests. */
(function (global) {
  'use strict';

  class HubApiError extends Error {
    constructor(status, message, payload) {
      super(message);
      this.name = 'HubApiError';
      this.status = status;
      this.payload = payload;
    }
  }

  class HubApi {
    constructor(baseUrl, token) {
      const base = String(baseUrl || '').trim().replace(/\/$/, '');
      if (!/^https?:\/\//i.test(base)) throw new Error('API URL must start with http:// or https://.');
      this.baseUrl = base;
      this.token = String(token || '').trim();
    }

    async request(method, path, payload) {
      const headers = { Accept: 'application/json' };
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      if (payload !== undefined) headers['Content-Type'] = 'application/json';
      let response;
      try {
        response = await fetch(`${this.baseUrl}${path}`, {
          method,
          headers,
          body: payload === undefined ? undefined : JSON.stringify(payload)
        });
      } catch (error) {
        throw new HubApiError(null, `OpenDroneKit API is unavailable: ${error.message}`, null);
      }
      const text = await response.text();
      let body = null;
      if (text) {
        try { body = JSON.parse(text); } catch (_) { body = text; }
      }
      if (!response.ok) {
        const detail = body && typeof body === 'object' ? body.detail : body;
        throw new HubApiError(response.status, detail || response.statusText, body);
      }
      return body;
    }

    async requestForm(path, form, method) {
      const headers = { Accept: 'application/json' };
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      let response;
      try {
        response = await fetch(`${this.baseUrl}${path}`, { method: method || 'POST', headers, body: form });
      } catch (error) {
        throw new HubApiError(null, `OpenDroneKit API is unavailable: ${error.message}`, null);
      }
      const text = await response.text();
      let body = null;
      if (text) { try { body = JSON.parse(text); } catch (_) { body = text; } }
      if (!response.ok) {
        const detail = body && typeof body === 'object' ? body.detail : body;
        throw new HubApiError(response.status, detail || response.statusText, body);
      }
      return body;
    }

    // Sign-in returns a session token; the caller keeps it in this instance only.
    async login(email, password) {
      const body = await this.request('POST', '/auth/login', { email, password });
      this.token = body.access_token;
      return body;
    }
    async register(email, password, organizationName) {
      const body = await this.request('POST', '/auth/register', {
        email, password, organization_name: organizationName || ''
      });
      this.token = body.access_token;
      return body;
    }
    me() { return this.request('GET', '/auth/me'); }
    listOrganizations() { return this.request('GET', '/organizations'); }
    listProjects(orgId) { return this.request('GET', `/organizations/${orgId}/projects`); }
    createProject(orgId, project) { return this.request('POST', `/organizations/${orgId}/projects`, project); }
    getProject(projectId) { return this.request('GET', `/projects/${projectId}`); }
    listAssets(orgId) { return this.request('GET', `/organizations/${orgId}/assets`); }
    createAsset(orgId, asset) { return this.request('POST', `/organizations/${orgId}/assets`, asset); }
    listJobs(projectId) { return this.request('GET', `/projects/${projectId}/jobs`); }
    submitJob(projectId, job) { return this.request('POST', `/projects/${projectId}/jobs`, job); }
    getJob(jobId) { return this.request('GET', `/jobs/${jobId}`); }
    listAiJobs(projectId) { return this.request('GET', `/projects/${projectId}/ai-jobs`); }
    createAiJob(projectId, job) { return this.request('POST', `/projects/${projectId}/ai-jobs`, job); }
    cancelJob(jobId) { return this.request('POST', `/jobs/${jobId}/cancel`); }
    cancelAiJob(jobId) { return this.request('POST', `/ai-jobs/${jobId}/cancel`); }
    aiModels() { return this.request('GET', '/ai-models'); }
    listDatasets(projectId) { return this.request('GET', `/projects/${projectId}/datasets`); }
    createDataset(projectId, dataset) { return this.request('POST', `/projects/${projectId}/datasets`, dataset); }
    beginUpload(datasetId, upload) { return this.request('POST', `/datasets/${datasetId}/uploads`, upload); }
    uploadChunk(uploadId, index, blob) {
      const form = new FormData();
      form.append('chunk', blob, `${index}.part`);
      return this.requestForm(`/uploads/${encodeURIComponent(uploadId)}/chunks/${index}`, form, 'PUT');
    }
    completeUpload(uploadId) { return this.request('POST', `/uploads/${encodeURIComponent(uploadId)}/complete`); }

    // Chunked, checksummed upload of one file. The checksum lets the server refuse a
    // damaged transfer; without Web Crypto (a non-secure origin) it is sent empty and
    // the server says checksum_verified=false rather than pretending.
    async uploadFile(datasetId, file, onProgress, chunkSize) {
      const size = chunkSize || 4 * 1024 * 1024;
      let sha256 = '';
      const subtle = global.crypto && global.crypto.subtle;
      if (subtle) {
        const digest = await subtle.digest('SHA-256', await file.arrayBuffer());
        sha256 = Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
      }
      const session = await this.beginUpload(datasetId, {
        filename: file.name, total_bytes: file.size, sha256, chunk_size: size
      });
      const total = session.missing_chunks.length;
      for (const index of session.missing_chunks) {
        await this.uploadChunk(session.upload_id, index, file.slice(index * size, (index + 1) * size));
        if (onProgress) onProgress(index + 1, total);
      }
      return this.completeUpload(session.upload_id);
    }

    async downloadArtifact(jobId, ref) {
      const headers = {};
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      const path = String(ref).split('/').map(encodeURIComponent).join('/');
      let response;
      try {
        response = await fetch(`${this.baseUrl}/jobs/${jobId}/artifacts/${path}`, { headers });
      } catch (error) {
        throw new HubApiError(null, `OpenDroneKit API is unavailable: ${error.message}`, null);
      }
      if (!response.ok) {
        let detail = response.statusText;
        try { detail = (await response.json()).detail || detail; } catch (_) { /* not JSON */ }
        throw new HubApiError(response.status, detail, null);
      }
      return response.blob();
    }
    listMissions(projectId) { return this.request('GET', `/projects/${projectId}/missions`); }
    getMissionPlan(missionId) { return this.request('GET', `/missions/${missionId}/plan`); }
    getMissionSimulation(missionId) { return this.request('GET', `/missions/${missionId}/simulation`); }
    createShare(projectId, options) { return this.request('POST', `/projects/${projectId}/shares`, options); }
    listShares(projectId) { return this.request('GET', `/projects/${projectId}/shares`); }
    listDefects(projectId) { return this.request('GET', `/projects/${projectId}/defects`); }
    listAnnotations(projectId) { return this.request('GET', `/projects/${projectId}/annotations`); }
    createAnnotation(projectId, annotation) { return this.request('POST', `/projects/${projectId}/annotations`, annotation); }
    updateAnnotation(annotationId, patch) { return this.request('PATCH', `/annotations/${annotationId}`, patch); }
    reviewAnnotation(annotationId, review) { return this.request('POST', `/annotations/${annotationId}/review`, review); }
    mergeAnnotations(projectId, merge) { return this.request('POST', `/projects/${projectId}/annotations/merge`, merge); }
    splitAnnotation(annotationId, split) { return this.request('POST', `/annotations/${annotationId}/split`, split); }
    prelabelAnnotations(projectId, file, modelKey, severity) {
      const form = new FormData();
      form.append('image', file, file.name);
      form.append('model_key', modelKey || 'structural_multiclass_detector');
      form.append('severity', severity || 'info');
      return this.requestForm(`/projects/${projectId}/annotations/prelabel`, form);
    }
    deleteAnnotation(annotationId) { return this.request('DELETE', `/annotations/${annotationId}`); }
    listMembers(orgId) { return this.request('GET', `/organizations/${orgId}/members`); }
    auditLog(orgId) { return this.request('GET', `/organizations/${orgId}/audit`); }
    tileProviders() { return this.request('GET', '/tiles/providers'); }
    tileStatus() { return this.request('GET', '/tiles/status'); }
    cacheTiles(request, orgId) {
      const suffix = orgId ? `?organization_id=${encodeURIComponent(orgId)}` : '';
      return this.request('POST', `/tiles/cache${suffix}`, request);
    }
    cacheStatus(jobId) { return this.request('GET', `/tiles/cache/${encodeURIComponent(jobId)}`); }
  }

  const exported = { HubApi, HubApiError };
  global.ODKHubApi = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
