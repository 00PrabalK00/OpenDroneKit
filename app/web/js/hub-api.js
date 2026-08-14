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

    listProjects(orgId) { return this.request('GET', `/organizations/${orgId}/projects`); }
    createProject(orgId, project) { return this.request('POST', `/organizations/${orgId}/projects`, project); }
    getProject(projectId) { return this.request('GET', `/projects/${projectId}`); }
    listAssets(orgId) { return this.request('GET', `/organizations/${orgId}/assets`); }
    createAsset(orgId, asset) { return this.request('POST', `/organizations/${orgId}/assets`, asset); }
    listJobs(projectId) { return this.request('GET', `/projects/${projectId}/jobs`); }
    listMissions(projectId) { return this.request('GET', `/projects/${projectId}/missions`); }
    listDefects(projectId) { return this.request('GET', `/projects/${projectId}/defects`); }
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
