"""Building the desktop shell does not load the processing stack.

`core/__init__` imported every submodule eagerly, so constructing Shell() -- which only
needs the project store and the flight log -- pulled in skimage, rasterio, boto3 and the
detection pipeline first: 7-11 s of nothing on screen on a cold start. The package now
resolves its names lazily; this pins that the window's critical path stays light.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HEAVY = ["skimage", "rasterio", "boto3", "core.detection", "core.pipeline", "core.survey_intelligence"]


def test_building_the_shell_leaves_the_heavy_stack_unloaded(tmp_path) -> None:
    code = (
        "import sys, json, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "t = time.perf_counter()\n"
        "from app.shell import Shell\n"
        "Shell()\n"
        f"print(json.dumps({{'loaded': [m for m in {HEAVY!r} if m in sys.modules],"
        " 'seconds': time.perf_counter() - t}))\n"
    )
    env = dict(os.environ, ODK_DATA_DIR=str(tmp_path / "data"))
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["loaded"] == [], f"Shell() imported {result['loaded']}"


def test_public_core_names_still_resolve() -> None:
    """Lazy is only acceptable if every name the package exported still arrives."""
    import core

    for name in core.__all__:
        assert getattr(core, name) is not None, name
    from core import detect_cracks, get_manager, run_pipeline  # noqa: F401
    import core.detection

    assert core.detection.detect_cracks is detect_cracks
