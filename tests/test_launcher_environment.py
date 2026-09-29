"""The app a user launches must say what it is missing, not fail a button later.

An audit found Fleet working under the test environment and broken in the one
OpenDroneKit.bat actually launches: that conda environment had no SQLAlchemy, and the
first sign of it was a Fleet save failing with "No module named 'sqlalchemy'". The
launcher now checks requirements.txt before starting and offers to install what is
missing, and a feature whose package is absent says which package and how to add it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools import check_environment

ROOT = Path(__file__).resolve().parents[1]


def _requirements(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestTheEnvironmentCheck:
    def test_an_absent_distribution_is_listed(self, tmp_path):
        req = _requirements(tmp_path, "# comment", "pytest>=8.0",
                            "odk-no-such-package-anywhere>=1.0  # absent")
        assert check_environment.missing(req) == ["odk-no-such-package-anywhere>=1.0"]
        assert check_environment.main(["--requirements", str(req)]) == 1

    def test_an_environment_with_everything_passes(self, tmp_path):
        req = _requirements(tmp_path, "pytest>=8.0", "PyYAML>=6.0", "PyYAML>=6.0")
        assert check_environment.missing(req) == []
        assert check_environment.main(["--requirements", str(req)]) == 0

    def test_a_gpu_build_counts_as_the_package(self, tmp_path, monkeypatch):
        """onnxruntime-gpu provides onnxruntime; offering the CPU wheel on top of it
        would break the GPU install."""
        installed = {"onnxruntime-gpu"}
        monkeypatch.setattr(check_environment, "_installed", lambda name: name in installed)
        req = _requirements(tmp_path, "onnxruntime>=1.17")
        assert check_environment.missing(req) == []

    def test_the_real_requirements_file_parses(self):
        names = dict(check_environment.requirements())
        assert "sqlalchemy" in names and names["sqlalchemy"] == "SQLAlchemy>=2.0"

    def test_the_launcher_runs_the_check_before_the_app(self):
        bat = (ROOT / "OpenDroneKit.bat").read_text(encoding="utf-8")
        check = bat.index(r"tools\check_environment.py")
        assert check < bat.index("pythonw")
        assert "--install" in bat


class TestAMissingPackageIsNamed:
    def test_fleet_without_sqlalchemy_says_so(self, tmp_path):
        """Drive the real Api in a fresh interpreter where SQLAlchemy cannot be imported,
        as in the launcher's environment the audit found."""
        script = (
            "import sys, json\n"
            "sys.modules['sqlalchemy'] = None\n"
            "from app.api import Api\n"
            "from app.session import AppSession\n"
            "from app.store import ProjectStore\n"
            f"api = Api(AppSession(ProjectStore(r'{tmp_path / 'p.db'}')))\n"
            "print(json.dumps(api.add_aircraft(1, 'ODK-01', 'Quad', 'SN-1')))\n"
        )
        env = {"ODK_DATABASE_URL": f"sqlite:///{(tmp_path / 'fleet.db').as_posix()}"}
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True,
                             text=True, env={**os.environ, **env}, timeout=120)
        assert out.returncode == 0, out.stderr
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result["ok"] is False
        assert "sqlalchemy is not installed" in result["error"]
        assert "check_environment.py --install" in result["error"]
