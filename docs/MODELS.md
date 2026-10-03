# Pretrained models: integration points and UI hooks

Each pretrained model in the v0.1.0-beta registry ships under its own licence (see
`docs/model_licences/<key>.md` and `LICENSES/<key>.txt`); MIT covers the code only. A
missing model, or a file whose sha256 differs from the registry's, is refused with the
reason; there is no fallback.

The web UI (`app/web/**`) is being rebuilt and is not edited by the model units. The
sections below are the contract the UI rebuild wires to.

## Interactive segmentation: interactive_sam21_tiny (SAM 2.1 Hiera-Tiny) and interactive_efficientsam_ti (EfficientSAM-Ti)

Click or box to get a mask on a photo or an orthomosaic window; the person edits it, saves
it as a finding, and gets area in m² and perimeter in m when the raster is georeferenced
and projected. Both models are **promptable, not detectors**: they mask whatever they are
pointed at and know nothing about defects. The mask is a suggestion; saving it is the
person's decision. The score is the model's own predicted IoU, uncalibrated, and is never
shown as a measurement.

Model choice: `model_key=""` takes the default, SAM 2.1-T when onnxruntime can really open
a CUDA session (a probe graph is opened, not just the provider list read), EfficientSAM-Ti
otherwise. The user can pick either. A missing or mismatching model is refused; the other
one is never substituted, and every reply names the model that ran (`model_key`,
`model_sha256`, `model_used`).

Back end:

| Layer | Entry point | Notes |
|---|---|---|
| Core | `core.interactive_segmentation.encode(image_bgr, model_key="") -> Embedding` | Verifies the graph sha256 (and the SAM 2.1 decoder's, from `companion_sha256`) before running. Runs the image encoder once; cached per (image sha256, model sha256), LRU of 6. EfficientSAM's single graph is cut in memory at the encoder output. |
| Core | `core.interactive_segmentation.predict(embedding, points=[(x, y, 1\|0)], box=None) -> MaskResult` | Points and box in the image's pixels. Returns `mask`, `score` (predicted IoU), `candidate_scores`, `polygon_px` (largest region, holes filled, traced on pixel edges so its area equals the pixel count), `area_px`, `model_key`, `model_sha256`, `model_used`. EfficientSAM takes at most 6 points (a box counts as 2). |
| Core | `core.interactive_segmentation.segment_source(path, points, box, model_key)` | Photos: whole image. Rasters (`.tif/.tiff/.vrt/.img`): reads an RGB window with rasterio around the prompts (≥1024 px at native GSD, snapped to a 512 grid so later clicks reuse the cached embedding, up to 4096 px) and offsets the polygon back to raster pixels. Adds `polygon_crs` and `polygon_lonlat` when the raster has a CRS. One-band rasters (thermal, DSM) are refused. |
| Core | `core.model_runner.run_registered_model(image, key)` | Kind `onnx_prompt_segmentation` is refused with "needs a prompt; use Segment (Api.segment_prompt)". |
| Api | `Api.segment_prompt(source_path, points, box=None, model_key="")` | Reply: `polygon_px`, `area_px`, `score`, `score_kind: "predicted_iou"`, `score_note`, `candidate_scores`, `model_key`, `model_sha256`, `model_used`, `default_model_key`, `model_choice` (`user`/`default`), `output_semantics: "mask"`, `status: "suggestion"`, `mask_png_data_uri` (transparent PNG of the window), `mask_origin_px`, `mask_hw`, `window_px`, `georeferenced`, `crs_epsg`, `polygon_crs`/`polygon_lonlat` (rasters with a CRS), `embedding_cached`, `components_dropped`, `holes_filled_px`. On refusal: `ok: false`, `error`. |
| Api | `Api.save_segmented_finding(source_path, polygon_px, label, severity="info", measure=True, model_key="", model_sha256="", score=None, note="")` | Creates a `polygon` annotation via `core.annotations.create_annotation`: `created_by="user"`, status `open`, tag `sam-assisted`, `model_key`, `model_sha256`, `confidence` = predicted IoU, and a note saying the outline was model-suggested and user-accepted. Provenance defaults to the last `segment_prompt` on that source; without any it refuses. With `measure=True` on a raster the area and perimeter go through `measure_on_raster` (`area`, `perimeter`), `_planar_value_in_metres` and `_record_measurement`, linked by `annotation_id`; `NotGeoreferenced`/`NotProjected` refuse and **nothing is saved**. On a plain photo the area is returned in `px2` with "square pixels: this photo has no ground scale", and is not added to the measurement history. |

Pixel conventions: prompts are pixel indices; `polygon_px` vertices are pixel corners
(pixel (c, r) covers [c, c+1] × [r, r+1]). `save_segmented_finding` shifts the vertices by
-0.5 before `measure_on_raster`, which treats integers as pixel centres, so the m² equals
pixel count × |pixel area| exactly (pinned on `B1_Aug22.tif`, EPSG:32636).

UI hooks (for the rebuild; not wired by this unit, which does not edit `app/web/**`):

- A **Segment** tool in the workspace actions (beside Run Model / Prelabel in
  `app/web/js/workspace/actions.js`) and in the image and ortho viewers:
  - left click adds an include point `[x, y, 1]`; shift-click adds an exclude point
    `[x, y, 0]`; drag draws a box `[x0, y0, x1, y1]`;
  - every change calls `api.segment_prompt(path, points, box, modelKey)` and draws
    `mask_png_data_uri` at `mask_origin_px` (or `polygon_px` as an outline);
  - **Enter** accepts the current mask; **Backspace** clears the prompts;
  - show `model_key` (and "default" vs "chosen") beside the mask, and the score only as
    "model's predicted IoU (uncalibrated)", never as confidence in a defect;
  - a model picker lists the two keys; when `ok` is false, show `error` as is (it says
    which model is missing and where to download it).
- **Save as finding** asks for a label and severity, then calls
  `api.save_segmented_finding(path, polygon_px, label, severity, true)` and shows
  `measurement.value` with its `unit` (m2 or px2). Show `measurement.units_label` for px2.
  On a refusal (no CRS, geographic CRS) offer "save without measuring"
  (`measure=false`).
- Users may edit the polygon before saving; pass the edited vertices. The finding still
  carries the model key and sha256 and the `sam-assisted` tag.
- Not a Review item: the person drew and accepted the outline. Do not run these models in
  bulk or present their output as detections.
