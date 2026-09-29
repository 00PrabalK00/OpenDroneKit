"""Where OpenDroneKit keeps its state, as absolute paths.

There used to be three roots and two of them were relative to the working directory:
the desktop project store (`./final_toolkit_outputs/app_state/projects.db`), the
fleet/share/webhook database (`sqlite:///./opendronekit.db`) and the report engine's
workspace (`<repo parent>/opendrone_workspace`). Launching from a shortcut, a terminal
somewhere else, or a script with a different cwd created a fresh empty store there, so
the user's projects appeared to vanish -- nothing was lost, it was simply being looked for
in the wrong folder.

Now there is one data root, resolved once to an absolute path:

    ODK_DATA_DIR            if set (relative values are resolved against the cwd once)
    <toolkit>/final_toolkit_outputs   otherwise

The default is the folder the launcher has always written to when double-clicked from
Explorer (which starts it in the toolkit folder), so existing projects stay where they
are. Everything else hangs off it.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "ODK_DATA_DIR"


def toolkit_root() -> Path:
    """The repository / install folder: the parent of `core/`."""
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """The single absolute root for every piece of local application state."""
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    root = Path(configured).expanduser() if configured else toolkit_root() / "final_toolkit_outputs"
    return root.resolve()


def app_state_dir() -> Path:
    """The desktop project store and its settings."""
    return data_root() / "app_state"


def projects_dir() -> Path:
    """Default parent folder for new desktop projects."""
    return data_root() / "projects"


def engine_workspace_dir() -> Path:
    """The report engine's project index and report output (core/project.py)."""
    return data_root() / "workspace"


def missions_dir() -> Path:
    """Default mission output when no project folder is given."""
    return data_root() / "missions"


def service_sqlite_path() -> Path:
    """The SQLite fallback for the service database (fleet, shares, webhooks)."""
    return data_root() / "opendronekit.db"


def service_sqlite_url() -> str:
    return "sqlite:///" + service_sqlite_path().as_posix()


def storage_dir() -> Path:
    """Uploads and derived artefacts of the Hub service (ODK_STORAGE_PATH overrides)."""
    configured = os.environ.get("ODK_STORAGE_PATH", "").strip()
    return Path(configured).expanduser().resolve() if configured else data_root() / "storage"


def tile_cache_dir() -> Path:
    """Offline basemap tiles: `<provider>/<z>/<x>/<y>.png`, shared by shell and Hub."""
    return storage_dir() / "tiles"


def logs_dir() -> Path:
    return data_root() / "logs"


def describe() -> dict[str, str]:
    """Every location, for diagnostics and the About/Environment panels."""
    return {
        "data_root": str(data_root()),
        "project_store": str(app_state_dir() / "projects.db"),
        "projects": str(projects_dir()),
        "report_workspace": str(engine_workspace_dir()),
        "service_database": str(service_sqlite_path()),
        "storage": str(storage_dir()),
        "tile_cache": str(tile_cache_dir()),
        "logs": str(logs_dir()),
    }
