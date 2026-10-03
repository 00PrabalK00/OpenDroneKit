"""Interactive segmentation (U1): click or box to mask, then area on a georeferenced ortho.

What is pinned here:

* a missing model, or a file that does not hash to the registry's sha256, is refused
  with the reason, and the other model is never substituted for it;
* run_registered_model refuses the promptable kind ("needs a prompt") instead of inventing
  one;
* the preprocessing and the mask -> polygon step are exact (shoelace area of the polygon
  equals the mask's pixel count);
* the real graphs run, with the I/O shapes the code assumes, and the in-memory cut of the
  EfficientSAM graph reproduces the whole graph;
* ONNX-vs-PyTorch PARITY (docs/holdout/interactive_segmentation_parity.json, 20 photos)
  describes the files the registry ships, and is reported as parity, not accuracy;
* the box-prompt mIoU in the registry is the one measured on the NeonTreeEvaluation
  manual-delineation subset, with N stated, by OpenDroneKit;
* save_segmented_finding measures on B1_Aug22.tif (EPSG:32636) exactly pixel count x
  |pixel area|, and refuses a raster with no CRS.

Tests that need weights or test data skip with the reason when ODK_MODEL_WEIGHTS_DIR or
ODK_TESTDATA_DIR is not set or does not hold them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core import interactive_segmentation as S
from core.model_runner import ModelRunRefused, run_registered_model

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "models" / "model_registry.json"
PARITY = REPO / "docs" / "holdout" / "interactive_segmentation_parity.json"
NEONTREE = REPO / "docs" / "holdout" / "interactive_segmentation_neontree.json"
RELEASE_BASE = "https://github.com/00PrabalK00/OpenDroneKit/releases/download/v0.1.0-beta/"
KEYS = (S.SAM21_KEY, S.EFFICIENTSAM_KEY)

WEIGHTS_ENV = os.environ.get("ODK_MODEL_WEIGHTS_DIR", "").strip()
TESTDATA_ENV = os.environ.get("ODK_TESTDATA_DIR", "").strip()


def _rows() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]


def _weights_present(key: str) -> bool:
    row = _rows()[key]
    paths = [row["path"]] + [c for c in row.get("companions", []) if c.endswith(".onnx")]
    return bool(WEIGHTS_ENV) and all((Path(WEIGHTS_ENV) / p).is_file() for p in paths)


def needs_weights(*keys: str):
    if not WEIGHTS_ENV:
        return pytest.mark.skip(reason="ODK_MODEL_WEIGHTS_DIR is not set")
    missing = [k for k in keys if not _weights_present(k)]
    return pytest.mark.skipif(bool(missing),
                              reason=f"weights not under ODK_MODEL_WEIGHTS_DIR: {missing}")


def needs_testdata(relative: str):
    if not TESTDATA_ENV:
        return pytest.mark.skip(reason="ODK_TESTDATA_DIR is not set")
    return pytest.mark.skipif(not (Path(TESTDATA_ENV) / relative).exists(),
                              reason=f"{relative} is not under ODK_TESTDATA_DIR")


@pytest.fixture
def no_weights(monkeypatch, tmp_path):
    """A weights folder with nothing in it, and no local copies either."""
    monkeypatch.setenv("ODK_MODEL_WEIGHTS_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr("core.models.models_root", lambda: tmp_path / "models_local")
    monkeypatch.setattr("core.models.registry_path", lambda: REGISTRY)
    S.clear_cache()
    return tmp_path


@pytest.fixture
def blob_image() -> np.ndarray:
    """A bright disc on a dark textured background: an unambiguous object to prompt."""
    rng = np.random.default_rng(0)
    image = (rng.normal(40, 8, (480, 640, 3))).clip(0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:480, :640]
    disc = (xx - 300) ** 2 + (yy - 220) ** 2 <= 90 ** 2
    image[disc] = (40, 180, 230)
    return image


# ── Registry rows ─────────────────────────────────────────────────────────────

class TestRegistryRows:
    @pytest.mark.parametrize("key", KEYS)
    def test_row_declares_kind_licence_source_and_semantics(self, key):
        row = _rows()[key]
        assert row["kind"] == S.PROMPT_SEGMENTATION_KIND
        assert row["path"].startswith(f"{key}/")
        assert row["output_semantics"] == "mask"
        assert row["status"] == "installed" and row["status_note"]
        lic = row["licence"]
        assert lic["weights"] == "Apache-2.0" and lic["use"] == "commercial"
        assert lic["attribution"] and lic["data"]
        assert (REPO / lic["text"]).is_file()
        assert "Apache License" in (REPO / lic["text"]).read_text(encoding="utf-8")
        assert len(row["source"]["revision"]) == 40
        assert (REPO / "docs" / "model_cards" / f"{key}.md").is_file()
        assert (REPO / "docs" / "model_licences" / f"{key}.md").is_file()

    @pytest.mark.parametrize("key", KEYS)
    def test_downloads_are_flat_release_assets_with_digests(self, key):
        row = _rows()[key]
        files = row["download"]["files"]
        assert files
        names = {f["name"] for f in files}
        assert Path(row["path"]).name in names
        for f in files:
            assert f["url"] == RELEASE_BASE + f["name"]
            assert f["name"].startswith(key)
            assert len(f["sha256"]) == 64
        by_name = {f["name"]: f["sha256"] for f in files}
        assert by_name[Path(row["path"]).name] == row["sha256"]
        for companion, digest in (row.get("companion_sha256") or {}).items():
            assert by_name[Path(companion).name] == digest

    @pytest.mark.parametrize("key", KEYS)
    def test_metrics_are_attributed(self, key):
        row = _rows()[key]
        for metric in row["publisher_metrics"]:
            assert metric["measured_by"] == "publisher" and metric["source_url"]
        metric = row["metric"]
        if metric != "unmeasured":
            assert metric["measured_by"] == "opendronekit"
            assert metric["n"] > 0 and metric["split"]
            assert "defect" not in metric["name"].lower()


# ── Refusals ──────────────────────────────────────────────────────────────────

class TestRefusals:
    @pytest.mark.parametrize("key", KEYS)
    def test_a_missing_model_is_refused_with_where_to_get_it(self, no_weights, blob_image, key):
        with pytest.raises(ModelRunRefused) as info:
            S.segment(blob_image, box=[200, 120, 400, 320], model_key=key)
        message = str(info.value)
        assert key in message and "v0.1.0-beta" in message

    def test_the_other_model_is_never_substituted(self, no_weights, tmp_path, monkeypatch,
                                                  blob_image):
        # SAM 2.1 asked for; only EfficientSAM's (fake) file exists. Refused, not swapped.
        root = tmp_path / "empty"
        (root / S.EFFICIENTSAM_KEY).mkdir(parents=True)
        (root / S.EFFICIENTSAM_KEY / f"{S.EFFICIENTSAM_KEY}.onnx").write_bytes(b"x")
        with pytest.raises(ModelRunRefused) as info:
            S.segment(blob_image, box=[200, 120, 400, 320], model_key=S.SAM21_KEY)
        assert S.SAM21_KEY in str(info.value)
        assert S.EFFICIENTSAM_KEY not in str(info.value)

    def test_a_file_that_is_not_the_registered_one_is_refused(self, no_weights, tmp_path,
                                                              blob_image):
        root = tmp_path / "empty" / S.EFFICIENTSAM_KEY
        root.mkdir(parents=True)
        (root / f"{S.EFFICIENTSAM_KEY}.onnx").write_bytes(b"not the published weights")
        with pytest.raises(ModelRunRefused) as info:
            S.segment(blob_image, box=[200, 120, 400, 320], model_key=S.EFFICIENTSAM_KEY)
        assert "not the file that was registered" in str(info.value)

    def test_a_swapped_decoder_is_refused(self, no_weights, tmp_path, blob_image, monkeypatch):
        # The encoder verifies; the decoder beside it does not. Still refused.
        row = _rows()[S.SAM21_KEY]
        root = tmp_path / "empty"
        enc = root / row["path"]
        enc.parent.mkdir(parents=True)
        enc.write_bytes(b"encoder")
        (root / row["companions"][0]).write_bytes(b"some other decoder")
        import hashlib

        from core import models

        real_spec = models.get_model_spec

        def spec(key):
            got = real_spec(key)
            if key == S.SAM21_KEY:
                from dataclasses import replace

                return replace(got, sha256=hashlib.sha256(b"encoder").hexdigest())
            return got

        monkeypatch.setattr(models, "get_model_spec", spec)
        with pytest.raises(ModelRunRefused) as info:
            S.segment(blob_image, box=[200, 120, 400, 320], model_key=S.SAM21_KEY)
        assert "mask_decoder" in str(info.value) and "not the registered file" in str(info.value)

    @pytest.mark.parametrize("key", KEYS)
    def test_run_registered_model_needs_a_prompt(self, key):
        with pytest.raises(ModelRunRefused) as info:
            run_registered_model(np.zeros((64, 64, 3), np.uint8), key)
        assert "needs a prompt" in str(info.value)

    def test_prompts_are_checked(self):
        with pytest.raises(ValueError, match="include point or a box"):
            S._normalise_prompts([[10, 10, 0]], None, (100, 100))
        with pytest.raises(ValueError, match="outside the image"):
            S._normalise_prompts([[150, 10, 1]], None, (100, 100))
        with pytest.raises(ValueError, match="no area"):
            S._normalise_prompts([], [10, 10, 10, 50], (100, 100))


class TestModelChoice:
    def test_default_is_sam21_on_cuda_and_efficientsam_on_cpu(self, monkeypatch):
        monkeypatch.setattr(S, "cuda_available", lambda: True)
        assert S.choose_model_key("") == S.SAM21_KEY
        monkeypatch.setattr(S, "cuda_available", lambda: False)
        assert S.choose_model_key("") == S.EFFICIENTSAM_KEY

    def test_the_user_can_override(self, monkeypatch):
        monkeypatch.setattr(S, "cuda_available", lambda: False)
        assert S.choose_model_key(S.SAM21_KEY) == S.SAM21_KEY


# ── Exact geometry ────────────────────────────────────────────────────────────

class TestGeometry:
    def test_polygon_area_equals_pixel_count(self):
        mask = np.zeros((60, 80), bool)
        mask[5:30, 10:50] = True
        mask[12:16, 20:24] = False          # a hole: filled
        mask[40:55, 0:6] = True             # a smaller region: dropped
        mask[20:25, 50:60] = True           # attached to the main region
        ring, area, dropped, filled = S.mask_to_polygon(mask)
        assert ring[0] == ring[-1]
        assert area == int(S.filled_region(mask).sum())
        assert S.shoelace(ring) == pytest.approx(area)
        assert dropped == 1 and filled == 16

    def test_polygon_on_the_image_edge(self):
        mask = np.zeros((40, 40), bool)
        mask[0:10, 0:40] = True
        ring, area, _, _ = S.mask_to_polygon(mask)
        assert area == 400 and S.shoelace(ring) == pytest.approx(400)
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        assert min(xs) == 0 and max(xs) == 40 and min(ys) == 0 and max(ys) == 10

    def test_coordinates_map_through_the_square_resize(self):
        assert S.to_model_space((400, 150), (300, 800)) == pytest.approx((512.0, 512.0))
        x = S.preprocess_sam21(np.zeros((300, 800, 3), np.uint8))
        assert x.shape == (1, 3, 1024, 1024)
        # ImageNet normalisation of black, channel by channel (RGB).
        assert x[0, :, 0, 0] == pytest.approx(-np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]), rel=1e-5)
        y = S.preprocess_efficientsam(np.full((300, 800, 3), 255, np.uint8))
        assert y.shape == (1, 3, 1024, 1024) and y.max() == pytest.approx(1.0)

    def test_window_snaps_and_contains_the_prompts(self):
        col, row, w, h = S.prompt_window([[3000, 2000, 1]], None, (5000, 6000))
        assert w == h == 1024 and col % 512 == 0 and row % 512 == 0
        assert col <= 3000 < col + w and row <= 2000 < row + h
        again = S.prompt_window([[3100, 2050, 1], [2950, 1990, 0]], None, (5000, 6000))
        assert again == (col, row, w, h), "nearby clicks must reuse the cached window"
        col, row, w, h = S.prompt_window([], [100, 100, 2100, 900], (5000, 6000))
        assert col <= 100 and col + w >= 2100 and w == 3072

    def test_raster_vertices_move_to_pixel_centres(self):
        assert S.polygon_for_raster_measurement([[0, 0], [2, 0], [2, 2], [0, 0]]) == \
            [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5]]


# ── Real inference ────────────────────────────────────────────────────────────

class TestRealInference:
    @needs_weights(S.SAM21_KEY)
    def test_sam21_io_shapes_and_a_box(self, blob_image):
        emb = S.encode(blob_image, S.SAM21_KEY)
        assert [a.shape for a in emb.arrays] == [(1, 32, 256, 256), (1, 64, 128, 128),
                                                (1, 256, 64, 64)]
        result = S.predict(emb, box=[200, 120, 400, 320])
        disc = np.zeros(blob_image.shape[:2], bool)
        yy, xx = np.mgrid[:480, :640]
        disc[(xx - 300) ** 2 + (yy - 220) ** 2 <= 90 ** 2] = True
        inter = (result.mask & disc).sum() / (result.mask | disc).sum()
        assert inter > 0.85
        assert len(result.candidate_scores) == 3 and 0 < result.score <= 1.2
        assert result.model_key == S.SAM21_KEY
        assert result.model_sha256 == _rows()[S.SAM21_KEY]["sha256"]
        # Second prompt on the same image reuses the embedding.
        assert S.encode(blob_image, S.SAM21_KEY).computed is False

    @needs_weights(S.EFFICIENTSAM_KEY)
    def test_efficientsam_points_and_box(self, blob_image):
        emb = S.encode(blob_image, S.EFFICIENTSAM_KEY)
        assert emb.arrays[0].shape == (1, 3, 1024, 1024)
        assert emb.arrays[1].shape == (1, 256, 64, 64), "the graph was not cut at the encoder"
        result = S.predict(emb, points=[[300, 220, 1], [40, 40, 0]], box=[200, 120, 400, 320])
        assert result.area_px > 0.8 * np.pi * 90 ** 2
        assert result.model_key == S.EFFICIENTSAM_KEY
        with pytest.raises(ValueError, match="at most 6"):
            S.predict(emb, points=[[300, 220, 1]] * 5, box=[200, 120, 400, 320])

    @needs_weights(S.EFFICIENTSAM_KEY)
    def test_efficientsam_cut_reproduces_the_whole_graph(self, blob_image):
        model = S.verified_model(S.EFFICIENTSAM_KEY)
        blob = S.preprocess_efficientsam(blob_image)
        coords = np.array([[[[300.0, 220.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]],
                          np.float32) * np.array([1024 / 640, 1024 / 480], np.float32)
        labels = np.array([[[1, -1, -1, -1, -1, -1]]], np.float32)
        whole = S._session(model.graph, prefer_cuda=False)
        names = [i.name for i in whole.get_inputs()]
        w_masks, w_iou = whole.run(None, {names[0]: blob, names[1]: coords, names[2]: labels})[:2]
        enc, dec = S._efficientsam_sessions(model.graph, prefer_cuda=False)
        emb = enc.run(None, {enc.get_inputs()[0].name: blob})[0]
        dn = [i.name for i in dec.get_inputs()]
        c_masks, c_iou = dec.run(None, {dn[0]: emb, dn[1]: coords, dn[2]: labels})[:2]
        assert np.abs(w_iou - c_iou).max() < 1e-4
        assert np.abs(w_masks - c_masks).max() < 1e-3

    @needs_weights(*KEYS)
    def test_parity_report_describes_the_shipped_files(self):
        report = json.loads(PARITY.read_text(encoding="utf-8"))
        assert report["measurement"].startswith("PARITY")
        assert report["n_images"] == 20
        rows = _rows()
        for name, key in (("sam21_fp32", S.SAM21_KEY), ("efficientsam_fp32", S.EFFICIENTSAM_KEY)):
            got = report["results"][name]
            assert got["model_file_sha256"] == rows[key]["sha256"]
            assert got["n_comparisons"] == 60
            assert got["mask_iou_mean"] >= rows[key]["parity"]["mask_iou_mean_min_accepted"]
            assert rows[key]["parity"]["mask_iou_mean"] == got["mask_iou_mean"]

    def test_registry_metric_is_the_neontree_measurement(self):
        report = json.loads(NEONTREE.read_text(encoding="utf-8"))
        rows = _rows()
        for name, key in (("sam21_fp32", S.SAM21_KEY), ("efficientsam_fp32", S.EFFICIENTSAM_KEY)):
            metric = rows[key]["metric"]
            got = report["results"][name]
            assert metric["value"] == got["miou"] and metric["n"] == got["n_crowns"]
            assert metric["measured_by"] == "opendronekit"
            assert got["model_file_sha256"] == rows[key]["sha256"]
            assert "box prompt" in metric["split"]


# ── The Api, end to end ───────────────────────────────────────────────────────

@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    created = instance.create_project("segment", str(tmp_path / "project"))
    assert created["ok"], created
    return instance


ORTHO = "bridge_inspection_h20t_boga_cayi/orthophoto/B1_Aug22.tif"


class TestApi:
    @needs_testdata(ORTHO)
    def test_area_on_a_projected_ortho_is_pixel_count_times_pixel_area(self, api):
        import rasterio

        ortho = Path(TESTDATA_ENV) / ORTHO
        with rasterio.open(ortho) as src:
            assert src.crs.to_epsg() == 32636
            pixel_area = abs(src.transform.a * src.transform.e)
        # A mask on the real ortho, standing in for a model's: an irregular blob with a
        # notch, placed in the raster's frame.
        mask = np.zeros((400, 400), bool)
        yy, xx = np.mgrid[:400, :400]
        mask[(xx - 200) ** 2 / 150 ** 2 + (yy - 180) ** 2 / 90 ** 2 <= 1] = True
        mask[150:200, 180:260] = False
        ring, count, _, _ = S.mask_to_polygon(mask)
        polygon = [[x + 2000, y + 2100] for x, y in ring]
        result = api.save_segmented_finding(
            str(ortho), polygon, "Spalling (outline)", measure=True,
            model_key=S.SAM21_KEY, model_sha256="0" * 64, score=0.91)
        assert result["ok"], result
        assert result["measurement"]["unit"] == "m2"
        assert result["measurement"]["value"] == pytest.approx(count * pixel_area, rel=1e-6, abs=1e-3)
        assert result["measurement"]["crs_epsg"] == 32636
        finding = result["finding"]
        assert finding["created_by"] == "user" and "sam-assisted" in finding["tags"]
        assert finding["model_key"] == S.SAM21_KEY and finding["model_sha256"] == "0" * 64
        assert finding["annotation_type"] == "polygon"
        history = api.list_measurements()["measurements"]
        kinds = {m["measurement_type"]: m for m in history}
        assert kinds["area"]["unit"] == "m2"
        assert kinds["area"]["value"] == pytest.approx(count * pixel_area, rel=1e-6, abs=1e-3)
        assert kinds["area"]["details"]["annotation_id"] == finding["id"]

    def test_a_raster_with_no_crs_is_refused_and_nothing_is_saved(self, api, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        path = tmp_path / "no_crs.tif"
        with rasterio.open(path, "w", driver="GTiff", height=50, width=50, count=3,
                           dtype="uint8", transform=from_origin(0, 50, 1, 1)) as dst:
            dst.write(np.zeros((3, 50, 50), np.uint8))
        result = api.save_segmented_finding(str(path), [[5, 5], [30, 5], [30, 30]], "x",
                                            model_key=S.SAM21_KEY, model_sha256="0" * 64)
        assert result["ok"] is False
        assert "no coordinate reference system" in result["error"]
        from core.annotations import list_annotations

        assert list_annotations(Path(api._session.project_root())) == []

    def test_a_plain_photo_gives_pixel_area_labelled_as_pixels(self, api, tmp_path):
        import cv2

        path = tmp_path / "photo.jpg"
        cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))
        result = api.save_segmented_finding(str(path), [[10, 10], [60, 10], [60, 40], [10, 40]],
                                            "Rust stain", model_key=S.EFFICIENTSAM_KEY,
                                            model_sha256="1" * 64, score=0.7)
        assert result["ok"], result
        assert result["measurement"] == {"kind": "area", "value": 1500.0, "unit": "px2",
                                         "units_label": "square pixels: this photo has no ground scale",
                                         "area_px": 1500.0}
        assert result["measurement_ids"] == {}
        assert result["finding"]["confidence"] == pytest.approx(0.7)

    def test_saving_without_model_provenance_is_refused(self, api, tmp_path):
        import cv2

        path = tmp_path / "photo.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), np.uint8))
        result = api.save_segmented_finding(str(path), [[1, 1], [20, 1], [20, 20]], "x")
        assert result["ok"] is False and "segment_prompt" in result["error"]

    def test_segment_prompt_refuses_a_missing_model(self, api, no_weights, tmp_path):
        import cv2

        path = tmp_path / "photo.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), np.uint8))
        result = api.segment_prompt(str(path), [[25, 25, 1]], model_key=S.EFFICIENTSAM_KEY)
        assert result["ok"] is False and S.EFFICIENTSAM_KEY in result["error"]

    @needs_weights(S.EFFICIENTSAM_KEY)
    def test_segment_then_save_on_a_photo(self, api, tmp_path, blob_image):
        import cv2

        path = tmp_path / "blob.png"
        cv2.imwrite(str(path), blob_image)
        seg = api.segment_prompt(str(path), [[300, 220, 1]], [200, 120, 400, 320],
                                 model_key=S.EFFICIENTSAM_KEY)
        assert seg["ok"], seg
        assert seg["model_key"] == S.EFFICIENTSAM_KEY and seg["status"] == "suggestion"
        assert seg["mask_png_data_uri"].startswith("data:image/png;base64,")
        saved = api.save_segmented_finding(str(path), seg["polygon_px"], "Disc")
        assert saved["ok"], saved
        assert saved["finding"]["model_sha256"] == _rows()[S.EFFICIENTSAM_KEY]["sha256"]
        assert saved["measurement"]["value"] == pytest.approx(seg["area_px"])

    @needs_weights(S.EFFICIENTSAM_KEY)
    @needs_testdata(ORTHO)
    def test_a_one_band_ortho_is_refused_for_segmentation(self, api):
        result = api.segment_prompt(str(Path(TESTDATA_ENV) / ORTHO), [[2200, 2300, 1]],
                                    model_key=S.EFFICIENTSAM_KEY)
        assert result["ok"] is False and "RGB" in result["error"]
