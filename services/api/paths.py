"""Filesystem roots for uploaded and derived data.

Deliberately separate from the pluggable object-storage backend: these are the local
paths the API needs before any backend is configured, and keeping them here means the
upload path does not change shape when object storage is introduced.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "./odk_storage"


def storage_root() -> Path:
    """Root for uploads and derived artifacts, from ODK_STORAGE_PATH."""
    root = Path(os.environ.get("ODK_STORAGE_PATH", DEFAULT_ROOT)).expanduser()
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
