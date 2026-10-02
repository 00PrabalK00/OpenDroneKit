"""Real dense runs with the binaries on this machine, skipped (with the reason) without them.

    ODK_DENSE_COLMAP   colmap.exe whose patch-match runs on this GPU (COLMAP 4.0.4 on RTX 50)
    ODK_OPENMVS_DIR    folder of the extracted OpenMVS release
    ODK_SURVEY_IMAGERY folder of Aukerman frames (see tests/conftest.py)

The first ten Aukerman frames, downscaled to 1600 px, go through the full pipeline. The
assertions are about provenance: the products must come from the named engine's own
output file, the capability report must agree with what ran, and the run must not carry
a "skipped" warning. See docs/DENSE_GPU.md for the full 77-image runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import survey_imagery_dir
from core import openmvs
from core import reconstruction_colmap as rc


def _frames(root: Path) -> Path:
    frames = sorted(survey_imagery_dir().glob("*.JPG"))[:10]
    if len(frames) < 10:
        pytest.skip(f"Survey imagery not present at {survey_imagery_dir()}.")
    from PIL import Image

    images = root / "images"
    images.mkdir()
    for frame in frames:
        with Image.open(frame) as image:
            image.resize((1600, round(image.height * 1600 / image.width))).save(
                images / frame.name, quality=92, exif=image.info.get("exif", b""))
    return images


@pytest.fixture(scope="module")
def openmvs_run(tmp_path_factory):
    folder = os.environ.get(openmvs.ENV_VAR, "").strip()
    if not folder or openmvs.find_tools(folder) is None:
        pytest.skip(f"Set {openmvs.ENV_VAR} to an extracted OpenMVS release.")
    if not rc.colmap_available():
        pytest.skip("pycolmap is not available in this environment.")
    root = tmp_path_factory.mktemp("openmvs")
    images = _frames(root)
    caps = rc.engine_capabilities()
    engine = rc.ColmapReconstructor(profile="fast_preview", dense=True, dense_engine="openmvs",
                                    use_gpu=False)
    result = engine.reconstruct(image_dir=images, output_dir=root / "out")
    return caps, engine, result, root / "out"


def test_openmvs_is_reported_available(openmvs_run):
    caps, _engine, _result, _out = openmvs_run
    assert caps["openmvs"] is True, caps["openmvs_reason"]
    assert caps["dense_engines"]["openmvs"] is True


def test_openmvs_products_come_from_its_own_cloud(openmvs_run):
    caps, engine, result, out = openmvs_run
    warnings = " ".join(result.warnings)
    assert "Dense MVS skipped" not in warnings and "Dense MVS failed" not in warnings, warnings
    report = engine.dense_report
    assert report["engine"] == "openmvs" and report["ran"] is True
    assert report["gpu"] is caps["openmvs_gpu"]
    cloud = Path(report["dense_cloud"])
    assert cloud == out / "colmap" / "openmvs" / "scene_dense.ply" and cloud.is_file()
    header = cloud.read_bytes()[:600].decode("ascii", "replace")
    count = int(header.split("element vertex ")[1].split()[0])
    assert count == result.dense_point_count > 0
    ply = (out / "reconstruction.ply").read_bytes()[:400].decode("ascii", "replace")
    assert int(ply.split("element vertex ")[1].split()[0]) == count


def test_openmvs_writes_a_textured_mesh(openmvs_run):
    _caps, _engine, result, _out = openmvs_run
    assert result.textured_mesh_obj_path and Path(result.textured_mesh_obj_path).is_file()
    assert result.texture_image_path and Path(result.texture_image_path).is_file()
    obj = Path(result.textured_mesh_obj_path).read_text("utf-8", "replace")
    assert "\nvt " in obj and "\nf " in obj


def test_the_densify_command_disabled_tower_mode(openmvs_run):
    _caps, _engine, _result, out = openmvs_run
    logs = sorted((out / "colmap" / "openmvs").glob("DensifyPointCloud-*.log"))
    assert logs, "OpenMVS writes a log per run"
    assert "--tower-mode 0" in logs[-1].read_text("utf-8", "replace")


@pytest.fixture(scope="module")
def colmap_dense_capability():
    binary = os.environ.get("ODK_DENSE_COLMAP", "").strip()
    if not binary or not Path(binary).is_file():
        pytest.skip("Set ODK_DENSE_COLMAP to a COLMAP build whose patch-match runs on this GPU.")
    return binary


def test_the_dense_colmap_build_passes_the_probe(colmap_dense_capability):
    result = rc.native_dense_probe(colmap_dense_capability)
    assert result["ok"] is True, result["reason"]
