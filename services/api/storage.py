"""Object storage, behind one interface.

Two backends: the local filesystem, and anything speaking S3 (MinIO, AWS S3, Ceph).
Which one runs is chosen by ODK_STORAGE_BACKEND, so a self-hosted deployment can keep
every byte on its own disks and a larger one can move to object storage without any
calling code changing.

The containment check in `LocalStorage` is not a nicety. Keys reach this layer from
client-supplied filenames and dataset identifiers, and a key containing `..` would
otherwise read or write anywhere the process can reach.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from .paths import storage_root


class StorageError(RuntimeError):
    """Raised when a key is unusable or a backend is misconfigured."""


class Storage(Protocol):
    """Implement this to add a backend; no calling code needs to change."""

    name: str

    def put(self, key: str, data: bytes | BinaryIO) -> str: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def url(self, key: str, expires_s: int = 3600) -> str: ...


def _normalise_key(key: str) -> str:
    """Reject a key that could escape its root before it reaches any backend.

    Checked here rather than per-backend so a new backend cannot forget it.
    """
    text = str(key or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        raise StorageError("Storage key must not be empty.")
    parts = [segment for segment in text.split("/") if segment not in {"", "."}]
    if any(segment == ".." for segment in parts):
        raise StorageError(f"Storage key must not contain a parent reference: {key!r}")
    if not parts:
        raise StorageError(f"Storage key resolves to nothing: {key!r}")
    return "/".join(parts)


@dataclass
class LocalStorage:
    """Files under a root directory. The default, and what self-hosting uses."""

    root: Path | None = None
    name: str = "local"

    def _base(self) -> Path:
        return Path(self.root) if self.root is not None else storage_root() / "objects"

    def _path(self, key: str) -> Path:
        base = self._base()
        target = (base / _normalise_key(key)).resolve()
        # Belt and braces: the key was already checked, but a symlink inside the root
        # could still point outward.
        if not target.is_relative_to(base.resolve()):
            raise StorageError(f"Storage key escapes the storage root: {key!r}")
        return target

    def put(self, key: str, data: bytes | BinaryIO) -> str:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            target.write_bytes(bytes(data))
        else:
            with target.open("wb") as handle:
                shutil.copyfileobj(data, handle)
        return str(target)

    def get(self, key: str) -> bytes:
        target = self._path(key)
        if not target.exists():
            raise StorageError(f"No object at {key!r}")
        return target.read_bytes()

    def delete(self, key: str) -> bool:
        target = self._path(key)
        if not target.exists():
            return False
        target.unlink()
        return True

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).exists()
        except StorageError:
            return False

    def list(self, prefix: str = "") -> list[str]:
        base = self._base()
        if not base.exists():
            return []
        keys = [
            path.relative_to(base).as_posix()
            for path in base.rglob("*") if path.is_file()
        ]
        return sorted(key for key in keys if key.startswith(prefix))

    def url(self, key: str, expires_s: int = 3600) -> str:
        """A file:// path, not a signed URL.

        Local storage has no signing authority, so `expires_s` is meaningless here.
        Returning a plain path and saying so is better than implying an access grant
        that does not exist.
        """
        return self._path(key).as_uri()


@dataclass
class S3Storage:
    """Any S3-compatible endpoint: MinIO for self-hosting, AWS S3, Ceph."""

    bucket: str
    endpoint_url: str = ""
    region: str = "us-east-1"
    name: str = "s3"

    def _client(self):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - environment guard
            raise StorageError("S3 storage needs boto3: pip install boto3") from exc
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url or None,
            region_name=self.region,
            aws_access_key_id=os.environ.get("ODK_S3_ACCESS_KEY") or None,
            aws_secret_access_key=os.environ.get("ODK_S3_SECRET_KEY") or None,
        )

    def put(self, key: str, data: bytes | BinaryIO) -> str:
        normalised = _normalise_key(key)
        client = self._client()
        if isinstance(data, (bytes, bytearray)):
            client.put_object(Bucket=self.bucket, Key=normalised, Body=bytes(data))
        else:
            client.upload_fileobj(data, self.bucket, normalised)
        return f"s3://{self.bucket}/{normalised}"

    def get(self, key: str) -> bytes:
        response = self._client().get_object(Bucket=self.bucket, Key=_normalise_key(key))
        return response["Body"].read()

    def delete(self, key: str) -> bool:
        self._client().delete_object(Bucket=self.bucket, Key=_normalise_key(key))
        return True

    def exists(self, key: str) -> bool:
        try:
            self._client().head_object(Bucket=self.bucket, Key=_normalise_key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[str]:
        paginator = self._client().get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)

    def url(self, key: str, expires_s: int = 3600) -> str:
        """A genuinely signed, time-limited URL."""
        return self._client().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": _normalise_key(key)},
            ExpiresIn=int(expires_s),
        )


BACKENDS = {"local": LocalStorage, "s3": S3Storage}


def build_storage(backend: str = "", **kwargs) -> Storage:
    """Construct the configured backend.

    An unrecognised name raises rather than falling back to local: silently storing
    data somewhere other than where the operator configured is worse than refusing.
    """
    name = (backend or os.environ.get("ODK_STORAGE_BACKEND", "local")).strip().lower()
    if name == "local":
        return LocalStorage(**kwargs)
    if name == "s3":
        bucket = kwargs.pop("bucket", None) or os.environ.get("ODK_S3_BUCKET", "")
        if not bucket:
            raise StorageError("S3 storage requires ODK_S3_BUCKET.")
        return S3Storage(
            bucket=bucket,
            endpoint_url=kwargs.pop("endpoint_url", None) or os.environ.get("ODK_S3_ENDPOINT", ""),
            region=kwargs.pop("region", None) or os.environ.get("ODK_S3_REGION", "us-east-1"),
            **kwargs,
        )
    raise StorageError(
        f"Unknown storage backend {name!r}. Available: {', '.join(sorted(BACKENDS))}."
    )


def describe_storage() -> dict[str, object]:
    """What the deployment is actually using, for the health endpoint."""
    name = os.environ.get("ODK_STORAGE_BACKEND", "local").strip().lower()
    if name == "s3":
        return {
            "backend": "s3",
            "bucket": os.environ.get("ODK_S3_BUCKET", ""),
            "endpoint": os.environ.get("ODK_S3_ENDPOINT", "(aws default)"),
            "signed_urls": True,
        }
    return {
        "backend": "local",
        "path": str(storage_root()),
        "signed_urls": False,
        "note": "Local storage cannot sign URLs; url() returns a file:// path.",
    }
