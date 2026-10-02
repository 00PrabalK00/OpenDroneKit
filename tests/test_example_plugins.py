"""The two example plugins, installed with ``pip install -e`` and driven through the SDK.

They are installed (editable, offline, no dependencies) into a scratch prefix rather than
the shared environment, and that prefix is added as a site directory -- which processes
the editable ``.pth`` hook exactly as a normal install would. Discovery then goes through
``importlib.metadata`` like any installed plugin.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import opendronekit as odk
import opendronekit.plugins as plugins_module
from opendronekit.plugins import PluginHost, PluginRegistry

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "plugins"
PACKAGES = ("odk-csv-exporter", "odk-strip-template")

# A real place: the Aukerman park survey site (Ohio), where the sample dataset was flown.
CENTRELINE = [[-81.75380, 41.30390], [-81.75220, 41.30440], [-81.75060, 41.30480]]


def install_editable(prefix: Path) -> Path:
    for package in PACKAGES:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
             "--no-index", "--disable-pip-version-check", "-q", "-e",
             str(EXAMPLES / package), "--prefix", str(prefix)],
            check=True, capture_output=True, text=True, timeout=300)
    return Path(sysconfig.get_path("purelib", vars={"base": str(prefix),
                                                    "platbase": str(prefix)}))


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("plugin_prefix")
    try:
        purelib = install_editable(prefix)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment issue
        pytest.fail(f"pip install -e failed: {exc.stderr}")
    saved_path, saved_meta = list(sys.path), list(sys.meta_path)
    site.addsitedir(str(purelib))
    importlib.invalidate_caches()
    yield purelib
    sys.path[:] = saved_path
    sys.meta_path[:] = saved_meta
    for name in [m for m in sys.modules if m.startswith(("odk_csv_exporter", "odk_strip_template"))]:
        del sys.modules[name]
    importlib.invalidate_caches()


@pytest.fixture
def host(installed, monkeypatch):
    fresh = PluginHost(PluginRegistry())
    monkeypatch.setattr(plugins_module, "host", fresh)
    return fresh


@pytest.fixture
def project(tmp_path, host):
    with odk.Workspace(tmp_path / "odk-data") as ws:
        yield ws.create_project("Plugin test")


def _metres_between(a, b):
    lat0 = math.radians((a[1] + b[1]) / 2)
    dx = math.radians(b[0] - a[0]) * 6378137.0 * math.cos(lat0)
    dy = math.radians(b[1] - a[1]) * 6378137.0
    return math.hypot(dx, dy)


def _inside(point, polygon):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class TestInstalledEditable:
    def test_both_packages_are_editable_installs(self, installed):
        for dist in ("odk_csv_exporter", "odk_strip_template"):
            info = next(installed.glob(f"{dist}-0.1.0.dist-info"))
            direct = json.loads((info / "direct_url.json").read_text(encoding="utf-8"))
            assert direct["dir_info"].get("editable") is True
            assert "opendronekit.plugins" in (info / "entry_points.txt").read_text(encoding="utf-8")

    def test_discovery_finds_three_plugins_in_two_packages(self, host):
        report = host.load()
        names = {(s.kind.value, s.name, s.version) for s in report.loaded}
        assert {("exporter", "waypoints_csv", "0.1.0"),
                ("report_section", "mission_waypoints", "0.1.0"),
                ("mission_type", "strip_survey", "0.1.0")} <= names
        assert {m.name for m in report.manifests} >= {"odk-csv-exporter", "odk-strip-template"}
        assert not [f for f in report.failures if "odk" in f.source]

    def test_the_sdk_lists_the_plugin_template(self, project):
        assert "plugin:strip_survey" in project.workspace.mission_templates()
        listed = project.workspace.capabilities()["plugins"]["loaded"]
        assert {"waypoints_csv", "mission_waypoints", "strip_survey"} <= {p["name"] for p in listed}


class TestStripTemplate:
    def test_plans_a_strip_of_the_requested_width_with_the_real_planner(self, project):
        plan = project.plan_mission([], template="plugin:strip_survey",
                                    centreline_lonlat=CENTRELINE, width_m=50, altitude_m=55)
        assert plan.summary.plugin == {"kind": "mission_type", "name": "strip_survey",
                                       "version": "0.1.0"}
        assert plan.summary.template == "grid" and plan.summary.waypoints > 0
        polygon = [list(p[:2]) for p in plan.plan["polygon"]]
        # The strip's ends are 50 m across (left vertex i <-> right vertex mirrored).
        n = len(CENTRELINE)
        assert _metres_between(polygon[0], polygon[2 * n - 1]) == pytest.approx(50, abs=0.5)
        assert _metres_between(polygon[n - 1], polygon[n]) == pytest.approx(50, abs=0.5)
        # The planner's options came from the plugin, the caller's altitude won.
        assert plan.plan["front_overlap_pct"] == 80.0 and plan.plan["altitude_m"] == 55.0
        # Planned image positions fall inside (or on the turn-around margin of) the strip.
        inside = sum(_inside(w[:2], polygon) for w in plan.plan["waypoints"])
        assert inside / len(plan.plan["waypoints"]) > 0.5

    def test_a_plugin_that_cannot_plan_is_a_plugin_error_not_a_grid(self, project):
        with pytest.raises(odk.PluginError) as caught:
            project.plan_mission([], template="plugin:strip_survey", width_m=50)
        assert "centreline_lonlat" in caught.value.reason and caught.value.stage == "call"


class TestCsvExporterAndReportSection:
    def test_the_csv_holds_exactly_the_planned_waypoints(self, project, tmp_path):
        plan = project.plan_mission([], template="plugin:strip_survey",
                                    centreline_lonlat=CENTRELINE, width_m=40)
        written = plan.export(["qgc_plan", "waypoints_csv"], directory=tmp_path / "out",
                              stem="strip")
        assert written["qgc_plan"][0].name == "strip.plan"
        csv_path = written["waypoints_csv"][0]
        assert csv_path.name == "strip.waypoints.csv"
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(plan.plan["waypoints"])
        for row, wp in zip(rows, plan.plan["waypoints"]):
            assert float(row["longitude"]) == pytest.approx(wp[0], abs=1e-8)
            assert float(row["latitude"]) == pytest.approx(wp[1], abs=1e-8)
            assert float(row["altitude_m"]) == pytest.approx(wp[2], abs=0.01)

    def test_an_unknown_format_is_refused_by_name(self, project, tmp_path):
        plan = project.plan_mission([], template="plugin:strip_survey",
                                    centreline_lonlat=CENTRELINE, width_m=40)
        with pytest.raises(odk.PluginError, match="no exporter plugin named 'shapefile'"):
            plan.export(["shapefile"], directory=tmp_path)

    def test_the_report_section_appears_in_a_real_report(self, project):
        plan = project.plan_mission([], template="plugin:strip_survey",
                                    centreline_lonlat=CENTRELINE, width_m=40)
        plan.save("Strip A")
        report = project.generate_report(title="Strip survey", report_type="mission_summary",
                                         plugin_sections=["mission_waypoints", "not_installed"])
        html = report.html_path.read_text(encoding="utf-8")
        assert "Mission Waypoints (plugin)" in html
        assert "Strip A" in html
        assert "mission_waypoints 0.1.0" in html
        # The missing plugin left its section out, said so, and did not stop the report.
        assert "not_installed (plugin section not included)" in html
        assert any("not_installed" in w for w in report.warnings)
