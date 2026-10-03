"""Relative depth (Depth Anything V2 Small): unitless by construction, refused when absent.

What is pinned here:

* refusal -- an unregistered key, a missing file, a file whose sha256 is not the
  registry's, and a model of the wrong kind all raise ModelRunRefused; nothing falls
  back to a guess;
* preprocessing -- both network sides are multiples of 14, short side ~518, aspect kept;
* units -- every result says "relative (unitless)" and no result, file or Api reply has
  a field that names a metric unit, distance, clearance, crack depth or volume;
* real inference (skipped without the weights) through run_registered_model and the Api;
* PARITY of the int8 graph against fp32 on 20 photos of bridge_inspection_h20t_boga_cayi
  /B1_Wide (skipped without the weights or the photos). This is parity, not accuracy;
* agreement with SfM depth, computed only when ODK_DEPTH_EVAL_RECON names a COLMAP
  sparse model (and ODK_DEPTH_EVAL_IMAGES its photos).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
import pytest

from core import depth as depth_mod
from core.depth import (
    DEFAULT_DEPTH_MODEL,
    UNITS,
    DepthResult,
    network_size,
    preprocess,
    save_depth,
)
from core.model_runner import ModelRunRefused, run_registered_model
from core.models import get_model_spec, resolve_model_path

KEY = DEFAULT_DEPTH_MODEL
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "") or "E:/Shit was cluittering/Projects/odk-testdata")
B1_WIDE = TESTDATA / "bridge_inspection_h20t_boga_cayi" / "B1_Wide"

# Any key naming a metric quantity. "relative" results must never carry one.
METRIC_FIELD = re.compile(
    r"(metre|meter|_m$|^m$|_mm$|_cm$|distance|clearance|crack_depth|volume|"
    r"(height|depth|scale|range)_m(_|$)|absolute)", re.IGNORECASE)


def _keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _keys(v)


def _assert_no_metric_fields(payload):
    bad = [k for k in _keys(payload) if METRIC_FIELD.search(k)]
    assert not bad, f"metric-sounding fields in a relative-depth result: {bad}"


def _weights() -> Path | None:
    path = resolve_model_path(KEY)
    return path if path is not None and path.is_file() else None


needs_weights = pytest.mark.skipif(
    _weights() is None,
    reason=f"{KEY} weights not installed (set ODK_MODEL_WEIGHTS_DIR, or download the "
           "v0.1.0-beta release asset relative_depth_dav2_small.onnx)")


# --------------------------------------------------------------------------- registry

def test_registry_row_is_relative_and_pinned():
    spec = get_model_spec(KEY)
    assert spec is not None
    assert spec.kind == "onnx_relative_depth"
    assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
    row = json.loads((Path(__file__).resolve().parents[1] / "models" / "model_registry.json")
                     .read_text(encoding="utf-8"))["models"][KEY]
    assert row["output_semantics"] == "relative"
    assert row["units"] == UNITS
    assert row["licence"]["weights"] == "Apache-2.0"
    # Only Small is Apache-2.0; Base/Large/Giant are CC BY-NC 4.0 and must never be swapped in.
    assert "small" in row["source"]["repo"].lower()
    assert row["source"]["original"] == "depth-anything/Depth-Anything-V2-Small"
    for f in row["download"]["files"]:
        assert f["name"] == Path(row["path"]).name and f["sha256"] == spec.sha256
    metric = row["metric"]
    if metric != "unmeasured":
        assert metric["measured_by"] in {"opendronekit", "publisher"}
        assert metric["split"] and metric.get("n")
    for pub in row["publisher_metrics"]:
        assert pub["measured_by"] == "publisher" and pub["source_url"]
    _assert_no_metric_fields({k: v for k, v in row.items()
                              if k not in {"description", "status_note"}})


# --------------------------------------------------------------------------- refusal

def test_unregistered_key_is_refused():
    with pytest.raises(ModelRunRefused):
        depth_mod.relative_depth(np.zeros((32, 32, 3), np.uint8), "no_such_depth_model")


def test_missing_weights_are_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("ODK_MODEL_WEIGHTS_DIR", str(tmp_path / "empty"))
    if resolve_model_path(KEY).is_file():
        pytest.skip("weights are installed inside the checkout's models/ folder")
    with pytest.raises(ModelRunRefused, match="cannot run"):
        depth_mod.relative_depth(np.zeros((64, 64, 3), np.uint8), KEY)
    with pytest.raises(ModelRunRefused):
        run_registered_model(np.zeros((64, 64, 3), np.uint8), KEY)


def test_wrong_bytes_are_refused(monkeypatch, tmp_path):
    root = tmp_path / "weights"
    spec = get_model_spec(KEY)
    target = root / spec.path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"these are not the registered weights")
    monkeypatch.setenv("ODK_MODEL_WEIGHTS_DIR", str(root))
    if (Path(__file__).resolve().parents[1] / "models" / spec.path).is_file():
        pytest.skip("weights are installed inside the checkout's models/ folder")
    with pytest.raises(ModelRunRefused, match="not the file that was registered"):
        depth_mod.relative_depth(np.zeros((64, 64, 3), np.uint8), KEY)


def test_wrong_kind_is_refused(monkeypatch):
    from core import models

    real = models.get_model_spec

    def as_classifier(key):
        spec = real(key)
        if key == KEY and spec is not None:
            from dataclasses import replace
            return replace(spec, kind="onnx_classification")
        return spec

    monkeypatch.setattr("core.models.get_model_spec", as_classifier)
    monkeypatch.setattr("core.model_runner.get_model_spec", as_classifier)
    monkeypatch.setattr("core.model_runner.model_status",
                        lambda key: {"exists": True, "path": __file__})
    with pytest.raises(ModelRunRefused, match="not a relative-depth model"):
        depth_mod.relative_depth(np.zeros((64, 64, 3), np.uint8), KEY)


# --------------------------------------------------------------------------- shapes

@pytest.mark.parametrize("hw", [(3000, 4000), (4000, 3000), (512, 640), (1080, 1920),
                                (518, 518), (100, 37), (7, 9)])
def test_network_sides_are_multiples_of_14(hw):
    h, w = hw
    nh, nw = network_size(h, w)
    assert nh % 14 == 0 and nw % 14 == 0
    assert min(nh, nw) == 518 or abs(min(nh, nw) - 518) <= 7
    assert abs((nw / nh) - (w / h)) / (w / h) < 0.05
    blob = preprocess(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))
    assert blob.shape == (1, 3, nh, nw) and blob.dtype == np.float32


def test_preprocess_is_imagenet_normalised_rgb():
    img = np.zeros((28, 28, 3), np.uint8)
    img[..., 2] = 255  # pure red in BGR
    blob = preprocess(img)
    r = (1.0 - 0.485) / 0.229
    g = (0.0 - 0.456) / 0.224
    assert np.allclose(blob[0, 0].mean(), r, atol=1e-3)
    assert np.allclose(blob[0, 1].mean(), g, atol=1e-3)


# --------------------------------------------------------------------------- units

def _fake_result(h=60, w=80) -> DepthResult:
    inv = np.tile(np.linspace(0.1, 5.0, w, dtype=np.float32), (h, 1))
    return DepthResult(inverse_depth=inv, overlay=depth_mod.colourise(inv),
                       model_key=KEY, model_sha256="0" * 64, model_used="onnx:x.onnx",
                       scope="test", network_input_hw=network_size(h, w))


def test_units_label_present_and_no_metre_field(tmp_path):
    result = _fake_result()
    summary = result.summary()
    assert summary["units"] == "relative (unitless)"
    assert summary["output_semantics"] == "relative"
    assert summary["colorbar"]["title"] == "nearer / farther (relative)"
    _assert_no_metric_fields(summary)
    assert not any(METRIC_FIELD.search(f) for f in DepthResult.__dataclass_fields__)

    files = save_depth(result, tmp_path, "img")
    meta = json.loads(Path(files["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["units"] == "relative (unitless)"
    assert "relative" in meta["semantics"] and "not distances" in meta["semantics"]
    _assert_no_metric_fields(meta)
    png = cv2.imread(files["depth_png16_path"], cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint16 and png.shape == (60, 80)
    assert np.load(files["depth_npy_path"]).dtype == np.float32


def test_no_api_converts_relative_depth_to_metric_quantities():
    """No function in core.depth, or Api method on it, offers metres/clearance/volume."""
    names = [n for n in dir(depth_mod) if not n.startswith("__")]
    assert not [n for n in names if METRIC_FIELD.search(n) or "metric" in n.lower()]
    from app.api import Api

    depth_methods = [n for n in dir(Api) if "depth" in n.lower()]
    assert "relative_depth" in depth_methods
    assert not [n for n in depth_methods if METRIC_FIELD.search(n) or "metric" in n.lower()
                or "crack" in n.lower()]


# --------------------------------------------------------------------------- real inference

@needs_weights
def test_real_inference_through_the_runner():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (300, 420, 3), dtype=np.uint8), (0, 0), 3)
    result = run_registered_model(img, KEY)
    assert result.kind == "onnx_relative_depth"
    assert result.model_sha256 == get_model_spec(KEY).sha256
    assert result.output["units"] == UNITS
    assert result.overlay is not None and result.overlay.shape[0] == 300
    _assert_no_metric_fields(result.to_dict())

    direct = depth_mod.relative_depth(img, KEY)
    assert direct.inverse_depth.shape == (300, 420)
    assert direct.inverse_depth.dtype == np.float32
    assert np.isfinite(direct.inverse_depth).all()


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    created = instance.create_project("depth", str(tmp_path / "project"))
    assert created["ok"], created
    return instance


def test_api_refuses_a_missing_image(api):
    reply = api.relative_depth(str(Path(__file__).parent / "no_such_image.jpg"))
    assert reply["ok"] is False


@needs_weights
def test_api_saves_npy_png16_and_relative_metadata(api, tmp_path):
    img_path = tmp_path / "bridge.png"
    rng = np.random.default_rng(2)
    cv2.imwrite(str(img_path), cv2.GaussianBlur(
        rng.integers(0, 255, (200, 260, 3), dtype=np.uint8), (0, 0), 3))
    reply = api.relative_depth(str(img_path))
    assert reply["ok"], reply
    assert reply["units"] == "relative" and reply["units_label"] == UNITS
    assert reply["overlay_data_uri"].startswith("data:image/")
    npy = Path(reply["depth_npy_path"])
    assert npy.parent.name == "depth" and npy.parent.parent.name == "analysis"
    assert npy.name == "bridge.npy" and np.load(npy).shape == (200, 260)
    assert cv2.imread(reply["depth_png16_path"], cv2.IMREAD_UNCHANGED).dtype == np.uint16
    meta = json.loads(Path(reply["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["units"] == UNITS and "Not metres" in meta["warning"]
    _assert_no_metric_fields(reply)


# --------------------------------------------------------------------------- parity

def _int8_path() -> Path | None:
    fp32 = _weights()
    if fp32 is None:
        return None
    cand = fp32.with_name("relative_depth_dav2_small_int8.onnx")
    return cand if cand.is_file() else None


@pytest.mark.skipif(_int8_path() is None or not B1_WIDE.is_dir(),
                    reason="needs both ONNX variants and odk-testdata/bridge_inspection_"
                           "h20t_boga_cayi/B1_Wide")
def test_int8_vs_fp32_parity_on_20_bridge_photos():
    """PARITY, not accuracy: does int8 reproduce fp32 on 20 real inspection photos?

    Measured 2026-10-03 on raw network-resolution maps: Spearman rho mean 0.978 (min
    0.862), AbsRel after scale-shift alignment mean 0.031 (max 0.068), and int8 was slower
    on this CPU (5.3 s vs 2.2 s). int8 is not at parity, so the release ships fp32. The bounds below catch a broken graph or a
    preprocessing regression, not a quality claim.
    """
    from tools.eval_relative_depth import list_images, run_parity

    report = run_parity(_weights(), _int8_path(), list_images(B1_WIDE, 20))
    print("PARITY", json.dumps({k: v for k, v in report.items() if k != "per_image"}))
    assert report["n_images"] == 20
    assert report["spearman_rho_mean"] > 0.95
    assert report["absrel_aligned_mean"] < 0.06


# --------------------------------------------------------------------------- SfM agreement

@pytest.mark.skipif(not os.environ.get("ODK_DEPTH_EVAL_RECON"),
                    reason="set ODK_DEPTH_EVAL_RECON to a COLMAP sparse model (and "
                           "ODK_DEPTH_EVAL_IMAGES to its photos) to measure agreement")
@needs_weights
def test_agreement_with_sfm_depth():
    pytest.importorskip("pycolmap")
    from tools.eval_relative_depth import run_sfm

    images = Path(os.environ.get("ODK_DEPTH_EVAL_IMAGES", ""))
    max_images = int(os.environ.get("ODK_DEPTH_EVAL_MAX", "10"))
    report = run_sfm(_weights(), Path(os.environ["ODK_DEPTH_EVAL_RECON"]), images,
                     max_images=max_images)
    print("SFM AGREEMENT", json.dumps({k: v for k, v in report.items() if k != "per_image"}))
    assert report["n_images"] > 0
    assert 0.0 <= report["delta1_mean"] <= 1.0


def test_sfm_agreement_math_on_synthetic_points():
    """The scale-shift fit in inverse depth recovers an affine copy exactly, and a
    flat prediction scores no better than the flat-scene baseline."""
    pytest.importorskip("scipy")
    from tools.eval_relative_depth import sample_network_map, sfm_agreement_image

    rng = np.random.default_rng(0)
    z = rng.uniform(20.0, 60.0, 400)
    exact = sfm_agreement_image(3.0 * (1.0 / z) + 0.7, z, np.random.default_rng(1))
    assert exact["absrel"] < 1e-9 and exact["delta1"] == 1.0
    assert exact["spearman_rho_eval"] > 0.999
    flat = sfm_agreement_image(np.full_like(z, 2.0) + 1e-9 * rng.standard_normal(z.size), z,
                               np.random.default_rng(1))
    assert flat["absrel"] >= 0.9 * flat["baseline_flat_absrel"]

    net = np.arange(12, dtype=np.float32).reshape(3, 4)
    # COLMAP coordinates (corner origin) of a 30 x 40 photo onto the 3 x 4 grid's centres.
    got = sample_network_map(net, np.array([5.0, 35.0]), np.array([5.0, 25.0]), (30, 40))
    assert np.allclose(got, [0.0, 11.0], atol=1e-6)
