"""Reconstruction on a self-hosted OpenDroneKit server (``--recon-mode cloud``).

Cloud mode used to run locally and add a warning, then was refused outright. This is
the real thing: the imagery is uploaded to an OpenDroneKit server project through the
resumable, checksummed upload API, a reconstruction job is submitted and followed, and
every file the job recorded is downloaded into the local output folder, so the rest of
the pipeline (measurements, report, map layers) works on the server's products exactly
as it would on a local run.

It only ever runs when an operator names an endpoint. Nothing here is reached by the
local engines, and nothing is sent anywhere without ``--recon-mode cloud`` and
``--recon-cloud-endpoint``. Every step that cannot complete is an error naming the step,
never a local run presented as a remote one.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import uuid

from .errors import OperationCancelled
from .reconstruction import SUPPORTED_EXTENSIONS, ReconstructionResult

TOKEN_ENV = "ODK_CLOUD_TOKEN"
CHUNK_BYTES = 8 * 1024 * 1024
_PROJECT_URL = re.compile(r"^(?P<base>https?://[^?#]*?)/projects/(?P<project>\d+)/?$")


class RemoteReconstructionError(RuntimeError):
    """A cloud run that could not complete, with the step that failed in the message."""


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """``http://host:8000/projects/3`` -> (``http://host:8000``, 3).

    The project is part of the endpoint because a job belongs to a project, and naming
    it explicitly means a run can never land in whichever project happens to be first.
    """
    value = str(endpoint or "").strip()
    match = _PROJECT_URL.match(value)
    if not match:
        raise ValueError(
            f"Cloud endpoint {value!r} is not an OpenDroneKit server project URL. Expected "
            "http(s)://<server>/projects/<project id>, for example "
            "http://10.0.0.5:8000/projects/3."
        )
    return match.group("base").rstrip("/"), int(match.group("project"))


def check_cloud_request(mode: str | None, endpoint: str | None, token: str | None = None) -> None:
    """Validate a cloud request before any work, so a bad one fails in seconds.

    Imagery is only ever uploaded to the server the operator names; there is no default
    service, so cloud mode without an endpoint is refused rather than guessed.
    """
    if str(mode or "local").strip().lower() != "cloud":
        return
    if not str(endpoint or "").strip():
        raise ValueError(
            "--recon-mode cloud needs --recon-cloud-endpoint, the URL of a project on a "
            "self-hosted OpenDroneKit server (http(s)://<server>/projects/<id>). This build is "
            "offline-first: imagery is uploaded only to a server you name, never to a default one."
        )
    parse_endpoint(str(endpoint))
    if not str(token or os.environ.get(TOKEN_ENV, "")).strip():
        raise ValueError(
            f"--recon-mode cloud needs an API token for that server: pass --recon-cloud-token or "
            f"set {TOKEN_ENV}."
        )


class RemoteReconstructor:
    """Runs reconstruction as a job on an OpenDroneKit server and fetches the products."""

    def __init__(
        self,
        endpoint: str,
        *,
        token: str = "",
        engine: str = "auto",
        profile: str = "standard",
        poll_interval_s: float = 1.0,
        timeout_s: float = 60.0,
        max_wait_s: float = 12 * 3600.0,
    ) -> None:
        self.base_url, self.project_id = parse_endpoint(endpoint)
        self.token = str(token or os.environ.get(TOKEN_ENV, "")).strip()
        if not self.token:
            raise ValueError(
                f"Cloud reconstruction needs an API token for {self.base_url}: pass "
                f"--recon-cloud-token or set {TOKEN_ENV}. Create one on the server with "
                "POST /auth/tokens."
            )
        self.engine = str(engine or "auto").strip().lower()
        self.profile = str(profile or "standard")
        self.poll_interval_s = float(poll_interval_s)
        self.timeout_s = float(timeout_s)
        self.max_wait_s = float(max_wait_s)

    # ------------------------------------------------------------------ http

    def _call(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        body: bytes | None = None,
        content_type: str = "",
        step: str,
        raw: bool = False,
    ) -> Any:
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "OpenDroneKit-Pipeline/1"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                data = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (ValueError, AttributeError):
                pass
            error = RemoteReconstructionError(
                f"{step} failed on {self.base_url}: HTTP {exc.code} {detail}".rstrip()
            )
            error.status = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RemoteReconstructionError(
                f"{step} failed: the OpenDroneKit server at {self.base_url} is unreachable ({reason})."
            ) from exc
        if raw:
            return data
        return json.loads(data.decode("utf-8")) if data else None

    def _upload(self, dataset_id: int, image: Path, step: str) -> dict[str, Any]:
        content = image.read_bytes()
        begun = self._call(
            "POST", f"/datasets/{dataset_id}/uploads", step=step,
            payload={"filename": image.name, "total_bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest(), "chunk_size": CHUNK_BYTES},
        )
        upload_id = begun["upload_id"]
        for index in begun["missing_chunks"]:
            chunk = content[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
            boundary = uuid.uuid4().hex
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunk\"; "
                f"filename=\"{index}.part\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode("ascii") + chunk + f"\r\n--{boundary}--\r\n".encode("ascii")
            self._call("PUT", f"/uploads/{upload_id}/chunks/{index}", body=body,
                       content_type=f"multipart/form-data; boundary={boundary}", step=step)
        completed = self._call("POST", f"/uploads/{upload_id}/complete", step=step)
        if not completed or not completed.get("checksum_verified"):
            raise RemoteReconstructionError(f"{step}: the server did not verify the checksum of {image.name}.")
        return completed

    # ------------------------------------------------------------------ run

    def reconstruct(
        self,
        image_dir: str | Path,
        crack_mask_dir: str | Path | None = None,
        output_dir: str | Path = "reconstruction",
        profile: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        **_: Any,
    ) -> ReconstructionResult:
        def emit(percent: int, message: str) -> None:
            if should_cancel is not None and should_cancel():
                raise OperationCancelled("Reconstruction cancelled")
            if progress_callback is not None:
                progress_callback(int(max(0, min(100, percent))), message)

        profile = str(profile or self.profile)
        images = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found in: {image_dir}")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        emit(1, f"Connecting to {self.base_url}")
        self._call("GET", "/auth/me", step="Authenticating")
        dataset = self._call(
            "POST", f"/projects/{self.project_id}/datasets", step="Creating the server dataset",
            payload={"name": f"{Path(image_dir).name} {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z",
                     "description": "Uploaded by run_pipeline.py --recon-mode cloud"},
        )
        dataset_id = int(dataset["id"])
        uploaded: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            emit(2 + int(28 * index / len(images)), f"Uploading {index + 1}/{len(images)}: {image.name}")
            done = self._upload(dataset_id, image, step=f"Uploading {image.name}")
            uploaded.append({"file": image.name, "bytes": done["bytes"], "sha256": done["sha256"]})

        emit(30, "Submitting the reconstruction job")
        job = self._call(
            "POST", f"/projects/{self.project_id}/jobs", step="Submitting the job",
            payload={"kind": "reconstruction", "dataset_id": dataset_id,
                     "engine": self.engine, "profile": profile},
        )
        job_id = int(job["id"])
        request_record = {
            "server": self.base_url, "project_id": self.project_id, "dataset_id": dataset_id,
            "job_id": job_id, "engine": self.engine, "profile": profile,
            "submitted_utc": datetime.now(timezone.utc).isoformat(), "uploaded": uploaded,
        }
        request_path = out_dir / "cloud_request.json"
        request_path.write_text(json.dumps(request_record, indent=2), encoding="utf-8")

        deadline = time.monotonic() + self.max_wait_s
        last = ""
        while job["status"] not in {"done", "failed", "cancelled"}:
            if should_cancel is not None and should_cancel():
                try:
                    self._call("POST", f"/jobs/{job_id}/cancel", step="Cancelling the server job")
                    note = f"server job {job_id} asked to stop"
                except RemoteReconstructionError as exc:
                    note = f"server job {job_id} may still be running: {exc}"
                raise OperationCancelled(f"Reconstruction cancelled ({note})")
            if time.monotonic() > deadline:
                raise RemoteReconstructionError(
                    f"Server job {job_id} did not finish within {self.max_wait_s:.0f} s; it is "
                    f"still {job['status']} on {self.base_url}."
                )
            time.sleep(self.poll_interval_s)
            job = self._call("GET", f"/jobs/{job_id}", step="Polling the job")
            message = f"Server job {job_id}: {job.get('message') or job['status']}"
            if message != last:
                emit(30 + int(0.6 * int(job.get("percent") or 0)), message)
                last = message
        response_path = out_dir / "cloud_response.json"
        response_path.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
        if job["status"] != "done":
            raise RemoteReconstructionError(
                f"Server job {job_id} {job['status']}: {job.get('error') or job.get('message') or 'no reason given'}"
            )

        refs = [str(ref) for ref in job.get("artifacts") or []]
        if not refs:
            raise RemoteReconstructionError(f"Server job {job_id} finished but recorded no output files.")
        local = self._download(job_id, refs, out_dir, emit)
        summary_ref = next((ref for ref in refs if PurePosixPath(ref).name == "reconstruction_result.json"), None)
        if summary_ref is None:
            raise RemoteReconstructionError(
                f"Server job {job_id} recorded no reconstruction_result.json, so its products "
                "cannot be identified. The server is older than this client."
            )
        payload = json.loads(local[summary_ref].read_text(encoding="utf-8"))
        result = self._localise(payload, local)
        result.execution_mode_requested = "cloud"
        result.execution_mode_used = "cloud"
        result.cloud_request_path = str(request_path)
        result.cloud_response_path = str(response_path)
        result.warnings = list(result.warnings) + [
            f"Reconstructed on {self.base_url} as job {job_id} (project {self.project_id}, "
            f"dataset {dataset_id}, engine {job.get('engine') or self.engine}); "
            f"{len(local)} files downloaded to this output folder."
        ]
        emit(100, "Cloud reconstruction complete")
        return result

    def _download(
        self, job_id: int, refs: list[str], out_dir: Path, emit: Callable[[int, str], None]
    ) -> dict[str, Path]:
        """Fetch every recorded file, keeping its layout relative to the job folder."""
        parents = [PurePosixPath(ref).parent.parts for ref in refs]
        common: tuple[str, ...] = parents[0]
        for parts in parents[1:]:
            size = 0
            while size < min(len(common), len(parts)) and common[size] == parts[size]:
                size += 1
            common = common[:size]
        local: dict[str, Path] = {}
        for index, ref in enumerate(refs):
            if PurePosixPath(ref).is_absolute() or re.match(r"^[A-Za-z]:", ref):
                raise RemoteReconstructionError(
                    f"Server job {job_id} recorded server file paths instead of downloadable "
                    "references, so its products cannot be fetched. Update the server."
                )
            relative = PurePosixPath(*PurePosixPath(ref).parts[len(common):])
            if ".." in relative.parts:
                raise RemoteReconstructionError(f"Refusing artifact reference {ref!r}: it escapes the output folder.")
            emit(90 + int(9 * index / len(refs)), f"Downloading {relative}")
            try:
                data = self._call("GET", f"/jobs/{job_id}/artifacts/{quote(ref)}",
                                  step=f"Downloading {relative}", raw=True)
            except RemoteReconstructionError as exc:
                if getattr(exc, "status", None) in {404, 405} and not local:
                    raise RemoteReconstructionError(
                        f"Server job {job_id} finished, but {self.base_url} does not serve job "
                        f"artifacts (GET /jobs/{{id}}/artifacts/{{ref}}), so the products stay on "
                        f"the server. Update the server. ({exc})"
                    ) from exc
                raise
            target = out_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            local[ref] = target
        return local

    @staticmethod
    def _localise(payload: dict[str, Any], local: dict[str, Path]) -> ReconstructionResult:
        """Point every *_path in the server's result at the downloaded copy.

        A path the server reported but did not record as an artifact is cleared rather
        than left pointing at a file on another machine.
        """
        by_suffix = sorted(local.items(), key=lambda item: -len(item[0]))

        def to_local(value: Any) -> Any:
            if not isinstance(value, str) or not value:
                return value
            normalised = value.replace("\\", "/")
            for ref, path in by_suffix:
                if normalised == ref or normalised.endswith("/" + ref):
                    return str(path)
            return ""

        names = {f.name for f in fields(ReconstructionResult)}
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in names:
                continue
            if key.endswith(("_path", "_dir")):
                value = to_local(value)
            elif key.endswith("_paths") and isinstance(value, list):
                value = [v for v in (to_local(item) for item in value) if v]
            kwargs[key] = value
        return ReconstructionResult(**kwargs)
