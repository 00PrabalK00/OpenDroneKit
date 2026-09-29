"""Filesystem roots for uploaded and derived data.

Deliberately separate from the pluggable object-storage backend: these are the local
paths the API needs before any backend is configured, and keeping them here means the
upload path does not change shape when object storage is introduced.
"""

from __future__ import annotations

import os
from pathlib import Path

def hub_data_dir() -> Path:
    """Per-user home for a server started without explicit paths.

    Defaults used to be relative to the working directory, so launching the API from
    the repository wrote a database and uploads into the source tree, and two launches
    from different directories silently saw different data. ODK_HUB_HOME overrides it.
    """
    override = os.environ.get("ODK_HUB_HOME", "").strip()
    if override:
        base = Path(override).expanduser()
    else:
        from core.settings import user_config_dir  # noqa: PLC0415

        base = user_config_dir() / "hub"
    base.mkdir(parents=True, exist_ok=True)
    return base


def legacy_or(legacy: Path, preferred: Path) -> Path:
    """The old cwd-relative location if it holds data and the new one does not yet."""
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def storage_root() -> Path:
    """Root for uploads and derived artifacts, from ODK_STORAGE_PATH."""
    configured = os.environ.get("ODK_STORAGE_PATH", "").strip()
    root = Path(configured).expanduser() if configured else legacy_or(
        Path("odk_storage"), hub_data_dir() / "storage")
    root.mkdir(parents=True, exist_ok=True)
    return root


def to_storage_ref(path: str | Path) -> str | None:
    """A path inside the storage root as a portable, root-relative reference.

    Artifacts are recorded this way so a job's outputs stay addressable when the
    storage root moves (a new mount, a restored backup) and never leak a server path.
    Returns None for anything outside the root.
    """
    root = storage_root().resolve()
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def from_storage_ref(ref: str) -> Path:
    """Resolve a recorded artifact reference back to a file path.

    Older rows stored absolute paths; those are returned unchanged.
    """
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    return storage_root() / candidate


def within_root(candidate: Path) -> bool:
    """Whether a resolved path is genuinely inside the storage root.

    Used to refuse a key that escapes via `..` or a symlink. Client-supplied names
    reach the filesystem, so containment is checked rather than assumed.
    """
    root = storage_root().resolve()
    try:
        return candidate.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False
