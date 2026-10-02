"""The SDK and plugin documentation is current, complete and runs.

* docs/sdk/reference.md is generated from the code and must match it;
* every public class, function, method and property it lists has a docstring;
* the quickstart in docs/sdk/README.md runs as written;
* docs/plugins.md lists the real kinds, contract members and failure stages;
* the example in PLUGIN_GUIDE.md registers and exports a real plan;
* relative links in these pages point at files that exist.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import gen_sdk_reference as gen
from tools import run_tutorial

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_the_reference_matches_the_code():
    result = subprocess.run([sys.executable, str(REPO / "tools" / "gen_sdk_reference.py"), "--check"],
                            capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_reference_covers_the_whole_public_api():
    import opendronekit
    import opendronekit.plugins as plugins

    text = (DOCS / "sdk" / "reference.md").read_text(encoding="utf-8")
    for module, names in ((opendronekit, opendronekit.__all__), (plugins, plugins.__all__)):
        for name in names:
            if name in {"registry", "load_plugins", "plugin_status", "PLUGIN_API_LEVEL",
                        "PluginError"} and module is opendronekit:
                continue                       # documented once, under opendronekit.plugins
            if name == "registry":
                continue                       # the module-level default registry object
            assert f"`{module.__name__}.{name}" in text, f"{module.__name__}.{name} undocumented"


def _public_callables():
    import enum

    import opendronekit
    import opendronekit.plugins as plugins

    for module, names in ((opendronekit, opendronekit.__all__), (plugins, plugins.__all__)):
        for name in names:
            value = getattr(module, name)
            if inspect.isclass(value):
                if not issubclass(value, enum.Enum):
                    yield f"{module.__name__}.{name}", value
                for member_name, member in gen._members(value):
                    fn = member.fget if isinstance(member, property) else member
                    yield f"{module.__name__}.{name}.{member_name}", fn
            elif inspect.isfunction(value):
                yield f"{module.__name__}.{name}", value


def test_every_public_name_has_a_docstring():
    missing = sorted({qual for qual, obj in _public_callables() if not gen._doc(obj)})
    assert not missing, f"undocumented public API: {missing}"


def test_the_quickstart_runs(tmp_path):
    out = run_tutorial.run(DOCS / "sdk" / "README.md", tmp_path,
                           env={"ODK_TUTORIAL_DIR": str(tmp_path)}, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    import opendronekit

    assert f"OpenDroneKit SDK {opendronekit.__version__} | plugin API 1.1" in text
    plan = re.search(r"MissionPlan\(template='grid', waypoints=(\d+), distance_m=([\d.]+)", text)
    assert plan and int(plan.group(1)) > 10 and float(plan.group(2)) > 500
    assert (tmp_path / "missions" / "quickstart.plan").stat().st_size > 0
    assert (tmp_path / "missions" / "quickstart.kml").stat().st_size > 0
    assert "refused: Unknown mission template 'spiral_of_doom'" in text
    assert "refused: This project has no georeferenced raster to measure on." in text


def test_plugins_md_matches_the_real_contracts():
    from opendronekit.plugins import CONTRACTS, PluginKind

    text = (DOCS / "plugins.md").read_text(encoding="utf-8")
    for kind in PluginKind:
        assert f"`{kind.value}`" in text, f"kind {kind.value} missing from plugins.md"
    for kind, members in CONTRACTS.items():
        section = text.split(f"(`{kind.value}`)", 1)
        assert len(section) == 2, f"no section for {kind.value}"
        body = section[1].split("\n### ", 1)[0]
        for member in members:
            assert member in body, f"{kind.value} section does not show {member}"
    sources = "\n".join((REPO / p).read_text(encoding="utf-8")
                        for p in ("sdk/plugins.py", "opendronekit/plugins.py"))
    for stage in re.findall(r"^\| `([a-z]+)`(?: / `([a-z]+)`)*", text, re.M):
        for name in filter(None, stage):
            assert f'"{name}"' in sources, f"documented failure stage {name} does not exist"
    for block in run_tutorial.blocks(text):
        if block.lang == "python":
            compile(block.code, "plugins.md", "exec")


def test_the_plugin_guide_example_exports_a_real_plan(tmp_path):
    import opendronekit as odk
    from opendronekit.plugins import PluginKind

    code = next(b.code for b in run_tutorial.blocks((DOCS / "PLUGIN_GUIDE.md").read_text(
        encoding="utf-8")) if b.lang == "python" and "class WaypointTextExporter" in b.code)
    namespace: dict = {}
    exec(compile(code, "PLUGIN_GUIDE.md", "exec"), namespace)
    exporter = namespace["registry"].create(PluginKind.EXPORTER, "waypoints_txt")
    with odk.Workspace(tmp_path / "data") as ws:
        plan = ws.create_project("Guide").plan_mission(
            [[-81.7535, 41.3036], [-81.7505, 41.3036], [-81.7505, 41.3050], [-81.7535, 41.3050]],
            template="grid", altitude_m=80)
    out = Path(exporter.export(plan.plan, tmp_path / ("m" + exporter.suffix)))
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == plan.summary.waypoints
    lon, lat, alt = map(float, rows[0].split())
    assert -81.76 < lon < -81.74 and 41.30 < lat < 41.31 and alt > 0


@pytest.mark.parametrize("page", ["sdk/README.md", "sdk/reference.md", "plugins.md",
                                  "PLUGIN_GUIDE.md", "tutorials/README.md"])
def test_relative_links_resolve(page):
    path = DOCS / page
    for link in re.findall(r"\]\((?!https?:|#)([^)#\s]+)", path.read_text(encoding="utf-8")):
        assert (path.parent / link).exists(), f"{page} links to missing {link}"
