"""The tutorials in docs/tutorials run as written, on the open sample datasets.

Each test extracts the tutorial's code with ``tools/run_tutorial.py`` (``python`` blocks
are executed in order in a fresh interpreter, ``file=`` blocks are written to disk), does
the shell step the tutorial tells the reader to do, and checks the output against what the
data must give. The sample data is the reference copy in odk-testdata (``ODK_TESTDATA_DIR``),
laid out exactly as ``tools/fetch_sample_data.py`` writes it; the copy is checked against
the pinned hashes first, so the tutorial sees the same bytes a reader downloads.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tools import fetch_sample_data as fsd
from tools import run_tutorial

REPO = Path(__file__).resolve().parents[1]
TUTORIALS = REPO / "docs" / "tutorials"
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "E:/Shit was cluittering/Projects/odk-testdata"))
ALL = sorted(TUTORIALS.glob("[0-9][0-9]-*.md"))


def _env(samples: Path, workdir: Path) -> dict[str, str]:
    return {"ODK_SAMPLE_DATA": str(samples), "ODK_TUTORIAL_DIR": str(workdir)}


def _check(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, (result.stdout[-3000:] + "\n" + result.stderr[-6000:])
    return result.stdout


@pytest.fixture(scope="module")
def aukerman_8(tmp_path_factory) -> Path:
    """``fetch odm_aukerman --limit 8``: the licence and the first 8 photographs."""
    reference = TESTDATA / "odm_aukerman"
    if not reference.is_dir():
        pytest.skip(f"Aukerman reference copy {reference} not present")
    samples = tmp_path_factory.mktemp("samples")
    for entry in fsd.select(fsd.read_manifest("odm_aukerman"), 8):
        target = samples / "odm_aukerman" / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / entry.path, target)
    check = fsd.verify("odm_aukerman", samples / "odm_aukerman", limit=8)
    assert len(check["ok"]) == 9 and not check["missing"] and not check["mismatched"]
    return samples


@pytest.mark.parametrize("tutorial", ALL, ids=[p.stem for p in ALL])
def test_every_tutorial_is_complete_python(tutorial):
    blocks = run_tutorial.blocks(tutorial.read_text(encoding="utf-8"))
    code = [b for b in blocks if b.lang == "python" and not b.file]
    assert code, f"{tutorial.name} has no runnable code"
    compile("\n".join(b.code for b in code), tutorial.name, "exec")
    for block in blocks:
        if block.file and block.file.endswith(".py"):
            compile(block.code, block.file, "exec")
    text = tutorial.read_text(encoding="utf-8")
    for link in re.findall(r"\]\((?!https?:)([^)#]+)", text):
        assert (tutorial.parent / link).exists(), f"{tutorial.name} links to missing {link}"


def test_there_are_three_tutorials_on_open_data():
    names = [p.name for p in ALL]
    assert names == ["01-aukerman-end-to-end.md", "02-crack-detection.md", "03-custom-plugin.md"]
    catalog = fsd.load_catalog()
    for name in ("odm_aukerman", "road_damage_rdd2022_subset"):
        assert catalog[name]["fetch"] != "restricted"
        assert catalog[name]["licence"] in {"CC0-1.0", "CC-BY-4.0"}


def _colmap_available() -> bool:
    from core.reconstruction_colmap import engine_capabilities

    caps = engine_capabilities(probe_dense=False)
    return bool(caps.get("colmap_binary") or caps.get("pycolmap"))


def test_tutorial_1_aukerman_end_to_end(aukerman_8, tmp_path):
    if not _colmap_available():
        pytest.skip("COLMAP is not installed; the tutorial's reconstruction would be refused")
    out = _check(run_tutorial.run(TUTORIALS / "01-aukerman-end-to-end.md", tmp_path,
                                  env=_env(aukerman_8, tmp_path)))
    assert "8 photos, 8 geotagged" in out
    assert "camera: SONY DSC-WX220" in out
    waypoints = int(re.search(r"^(\d+) waypoints, ", out, re.M).group(1))
    assert waypoints > 10
    assert re.search(r"qgc_plan -> \['mission_\w+\.plan'\]", out)
    assert re.search(r"kml -> \['mission_\w+\.kml'\]", out)
    registered, frames = map(int, re.search(r"(\d+)/(\d+) photos registered", out).groups())
    assert frames == 8 and registered >= 3
    assert "georeferenced: True | EPSG 32617" in out
    # 0.0004 deg of longitude at the site's latitude, and 0.0004 x 0.0004 deg: the values
    # are measured on the orthomosaic, so they may only differ by the georeferencing.
    lat = 41.3045
    metres_per_deg_lon = math.radians(1) * 6378137.0 * math.cos(math.radians(lat))
    expected_d = 0.0004 * metres_per_deg_lon
    distance = float(re.search(r"^distance: ([\d.]+) m$", out, re.M).group(1))
    assert abs(distance - expected_d) / expected_d < 0.03
    area = float(re.search(r"^area: ([\d.]+) m2$", out, re.M).group(1))
    expected_a = expected_d * 0.0004 * math.radians(1) * 6367000.0
    assert abs(area - expected_a) / expected_a < 0.05
    report = Path(re.search(r"^report: (.+report\.html)$", out, re.M).group(1))
    assert report.is_file() and "Aukerman park survey" in report.read_text(encoding="utf-8")
    assert "refused: No project 'No such project'" in out


def test_tutorial_2_crack_detection(tmp_path):
    rdd = TESTDATA / "road_damage_rdd2022_subset"
    if not rdd.is_dir():
        pytest.skip(f"RDD2022 reference copy {rdd} not present")
    photo = "China_Drone/train/images/China_Drone_000011.jpg"
    pinned = {e.path: e.sha256 for e in fsd.read_manifest("road_damage_rdd2022_subset")}
    assert fsd.sha256_of(rdd / photo) == pinned[photo]
    from core.models import model_status

    has_model = bool(model_status("crack_segmentation").get("exists"))
    out = _check(run_tutorial.run(TUTORIALS / "02-crack-detection.md", tmp_path,
                                  env=_env(TESTDATA, tmp_path)))
    source = "model" if has_model else "heuristic"
    assert f"source: {source} |" in out
    if has_model:
        assert re.search(r"weights: crack_segmentation [0-9a-f]{16}\.\.\.", out)
    assert "annotated cracks: [('D00', 292, 98, 340, 256)]" in out
    mask = Path(re.search(r"^mask: (.+crack_mask\.png)$", out, re.M).group(1))
    assert mask.is_file() and mask.parent.name == source
    coverage = re.search(rf"^{source}: (\d+)% of the annotated crack's length", out, re.M)
    assert coverage and int(coverage.group(1)) >= 50
    assert re.search(r"^heuristic: \d+% of the annotated crack's length", out, re.M)
    assert "refused: A photograph has no scale of its own" in out


def _install_editable(package: Path, prefix: Path) -> Path:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                    "--no-index", "--disable-pip-version-check", "-q", "-e", str(package),
                    "--prefix", str(prefix)], check=True, capture_output=True, text=True,
                   timeout=300)
    return Path(sysconfig.get_path("purelib", vars={"base": str(prefix), "platbase": str(prefix)}))


def test_tutorial_3_custom_plugin(aukerman_8, tmp_path):
    tutorial = TUTORIALS / "03-custom-plugin.md"
    written, _ = run_tutorial.materialise(tutorial.read_text(encoding="utf-8"), tmp_path)
    assert {p.name for p in written} == {"pyproject.toml", "__init__.py"}
    # The tutorial's shell step, `pip install -e odk-geojson-path`, into a scratch prefix
    # (not the shared environment); the script then sees it as an installed site.
    purelib = _install_editable(tmp_path / "odk-geojson-path", tmp_path / "prefix")
    preamble = f"import site\nsite.addsitedir({str(purelib)!r})\n"
    out = _check(run_tutorial.run(tutorial, tmp_path, env=_env(aukerman_8, tmp_path),
                                  preamble=preamble))
    assert "plugin API level: 1.1" in out
    assert "loaded: exporter geojson_path 0.1.0 (package odk-geojson-path)" in out
    vertices, waypoints = map(int, re.search(
        r"wrote aukerman\.path\.geojson with (\d+) vertices; the plan has (\d+) waypoints",
        out).groups())
    assert vertices == waypoints > 2
    assert "loaded: ['geojson_path']" in out
    assert re.search(r"skipped: broken = odk_no_such_module:manifest at load - "
                     r"ModuleNotFoundError", out)
    assert "refused: Plugin exporter:no_such_format failed at lookup" in out
