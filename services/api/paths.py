"""Filesystem roots for uploaded and derived data.

Deliberately separate from the pluggable object-storage backend: these are the local
paths the API needs before any backend is configured, and keeping them here means the
upload path does not change shape when object storage is introduced.
"""

from __future__ import annotations

from pathlib import Path

def storage_root() -> Path:
    """Root for uploads and derived artifacts, from ODK_STORAGE_PATH.

    Unset, it is `<data root>/storage` -- absolute, so the service and the desktop shell
    agree on it regardless of the working directory either was started from.
    """
    from core.paths import storage_dir

    root = storage_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


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
