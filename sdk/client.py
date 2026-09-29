"""Small dependency-free REST client for OpenDroneKit projects, assets and jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


# The statuses the API actually reports for a job that will not change again. "done"
# is the server's word for success; waiting on anything else never returns.
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class JobResult:
    payload: dict[str, Any]

    @property
    def id(self) -> int:
        return int(self.payload["id"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class OpenDroneKitClient:
    """Synchronous REST client with explicit URL, token and timeout configuration."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_s: float = 30.0,
    ) -> None:
        base = base_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")
        self.base_url = base
        self.token = token.strip()
        self.timeout_s = float(timeout_s)

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        suffix = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{suffix}"
        if query:
            url += "?" + urlencode({key: value for key, value in query.items() if value is not None})
        body = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "User-Agent": "OpenDroneKit-SDK/1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read()
            try:
                error_payload = json.loads(raw.decode("utf-8")) if raw else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = raw.decode("utf-8", errors="replace")
            detail = error_payload.get("detail") if isinstance(error_payload, dict) else None
            raise ApiError(exc.code, str(detail or exc.reason), error_payload) from exc
        except URLError as exc:
            raise ApiError(None, f"OpenDroneKit API is unavailable: {exc.reason}") from exc

    def list_projects(self, organization_id: int) -> list[dict[str, Any]]:
        return list(self.request("GET", f"/organizations/{organization_id}/projects"))

    def create_project(self, organization_id: int, **project: Any) -> dict[str, Any]:
        return dict(
            self.request("POST", f"/organizations/{organization_id}/projects", payload=project)
        )

    def list_assets(self, organization_id: int) -> list[dict[str, Any]]:
        return list(self.request("GET", f"/organizations/{organization_id}/assets"))

    def create_asset(self, organization_id: int, **asset: Any) -> dict[str, Any]:
        return dict(
            self.request("POST", f"/organizations/{organization_id}/assets", payload=asset)
        )

    def list_jobs(self, project_id: int) -> list[JobResult]:
        return [
            JobResult(dict(row))
            for row in self.request("GET", f"/projects/{project_id}/jobs")
        ]

    def submit_job(
        self,
        project_id: int,
        *,
        kind: str,
        dataset_id: int | None = None,
        engine: str = "auto",
        profile: str = "standard",
        options: Mapping[str, Any] | None = None,
    ) -> JobResult:
        return JobResult(
            dict(
                self.request(
                    "POST",
                    f"/projects/{project_id}/jobs",
                    payload={
                        "kind": kind,
                        "dataset_id": dataset_id,
                        "engine": engine,
                        "profile": profile,
                        "options": dict(options or {}),
                    },
                )
            )
        )

    def get_job(self, job_id: int) -> JobResult:
        return JobResult(dict(self.request("GET", f"/jobs/{job_id}")))

    def cancel_job(self, job_id: int) -> JobResult:
        return JobResult(dict(self.request("POST", f"/jobs/{job_id}/cancel")))

    def job_log(self, job_id: int) -> dict[str, Any]:
        return dict(self.request("GET", f"/jobs/{job_id}/log"))

    def download_artifact(self, job_id: int, ref: str) -> bytes:
        """Bytes of one file a job listed in its ``artifacts``."""
        request = Request(
            # Refs are storage paths: keep the slashes, escape spaces and the rest.
            f"{self.base_url}/jobs/{int(job_id)}/artifacts/{quote(ref, safe='/')}",
            headers={"User-Agent": "OpenDroneKit-SDK/1",
                     **({"Authorization": f"Bearer {self.token}"} if self.token else {})},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except HTTPError as exc:
            raise ApiError(exc.code, str(exc.reason)) from exc
        except URLError as exc:
            raise ApiError(None, f"OpenDroneKit API is unavailable: {exc.reason}") from exc

    def submit_ai_job(
        self,
        project_id: int,
        *,
        task: str,
        model_key: str,
        input_ref: str,
        dataset_id: int | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> JobResult:
        return JobResult(dict(self.request(
            "POST", f"/projects/{project_id}/ai-jobs",
            payload={"task": task, "model_key": model_key, "input_ref": input_ref,
                     "dataset_id": dataset_id, "parameters": dict(parameters or {})},
        )))

    def get_ai_job(self, job_id: int) -> JobResult:
        return JobResult(dict(self.request("GET", f"/ai-jobs/{job_id}")))

    def wait_for_ai_job(self, job_id: int, *, timeout_s: float = 3600.0,
                        poll_interval_s: float = 1.0) -> JobResult:
        return self._wait(self.get_ai_job, job_id, timeout_s, poll_interval_s)

    def wait_for_job(
        self,
        job_id: int,
        *,
        timeout_s: float = 3600.0,
        poll_interval_s: float = 1.0,
    ) -> JobResult:
        return self._wait(self.get_job, job_id, timeout_s, poll_interval_s)

    def _wait(self, fetch: Any, job_id: int, timeout_s: float,
              poll_interval_s: float) -> JobResult:
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("timeout_s and poll_interval_s must be positive.")
        deadline = time.monotonic() + timeout_s
        while True:
            job = fetch(job_id)
            if job.terminal:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout_s} seconds.")
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))
