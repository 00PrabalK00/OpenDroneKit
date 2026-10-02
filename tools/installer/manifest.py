"""The install manifest: every payload file and its sha256, plus how the build was made.

The build writes ``install_manifest.json`` at the root of the staged tree; the installer
ships it; ``main.py --self-test`` re-hashes the installed files against it. Files the
application creates after installation (logs, caches) are not in the manifest and are
not reported -- only files that were shipped and are now missing or different.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MANIFEST_NAME = "install_manifest.json"
SCHEMA = 1
REQUIRED_KEYS = ("schema", "app", "version", "python", "signed", "files")


def sha256_file(path: Path, attempts: int = 7) -> str:
    """sha256 of a file. A failed open or read is retried (a fresh hash each time): on a
    busy external drive, or a machine whose memory other processes have exhausted,
    opening one of ~30k files occasionally fails (EINVAL, WinError 1450 "insufficient
    system resources") and succeeds later. Backoff doubles from 0.5 s (~30 s in all).
    A file that keeps failing raises."""
    for attempt in range(1, attempts + 1):
        hasher = hashlib.sha256()
        try:
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            return hasher.hexdigest()
        except (OSError, MemoryError):
            if attempt == attempts or not Path(path).is_file():
                raise
            time.sleep(0.5 * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def iter_payload(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            yield path


def build_manifest(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    files = {}
    total = 0
    for path in iter_payload(root):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files[rel] = {"sha256": sha256_file(path), "size": size}
        total += size
    manifest = {"schema": SCHEMA, **meta, "file_count": len(files), "total_bytes": total,
                "files": files}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise ValueError when the manifest is not one this code wrote, or is unsafe."""
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest lacks {missing}")
    if manifest["schema"] != SCHEMA:
        raise ValueError(f"unknown manifest schema {manifest['schema']!r}")
    if not isinstance(manifest["signed"], bool):
        raise ValueError("manifest 'signed' must be true or false")
    for rel, entry in manifest["files"].items():
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or ":" in rel:
            raise ValueError(f"unsafe path in manifest: {rel!r}")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"bad sha256 for {rel!r}")


def forbidden_files(manifest: dict[str, Any], patterns: Iterable[str]) -> list[str]:
    """Payload files whose name matches a forbidden pattern (weights, secrets, SDKs)."""
    pats = [p.lower() for p in patterns]
    hits = []
    for rel in manifest["files"]:
        name = PurePosixPath(rel).name.lower()
        if any(fnmatch.fnmatch(name, p) for p in pats):
            hits.append(rel)
    return hits


def write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    target = Path(root) / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return target


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def verify_tree(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Shipped files that are missing or whose bytes changed. Empty means intact."""
    problems = []
    for rel, entry in manifest["files"].items():
        path = Path(root) / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            problems.append(f"changed: {rel}")
    return problems
