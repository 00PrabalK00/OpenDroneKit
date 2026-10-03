# Pretrained models: integration points and UI hooks

Each pretrained model in the v0.1.0-beta registry ships under its own licence (see
`docs/model_licences/<key>.md` and `LICENSES/<key>.txt`); MIT covers the code only. A
missing model, or a file whose sha256 differs from the registry's, is refused with the
reason; there is no fallback.

The web UI (`app/web/**`) is being rebuilt and is not edited by the model units. The
sections below are the contract the UI rebuild wires to.

## relative_depth_dav2_small (Depth Anything V2 Small): relative depth overlay

Unitless relative inverse depth from one photo. **Never metres**: no clearance, crack
depth, height or volume can be derived from it, and no API offers one.

Back end:

| Layer | Entry point | Notes |
|---|---|---|
| Core | `core.depth.relative_depth(image_bgr, model_key="relative_depth_dav2_small") -> DepthResult` | Verifies the file's sha256 against the registry before running; raises `ModelRunRefused` otherwise. Short side resized to ~518, both sides multiples of 14, ImageNet normalisation, bicubic upsample back to the photo's size. |
| Core | `core.model_runner.run_registered_model(image, "relative_depth_dav2_small")` | Kind `onnx_relative_depth`. Returns a `ModelRunResult` whose `output` is the JSON summary and whose `overlay` is the colourised map with its colour bar. `Api.run_model` therefore also works for this key. |
| Core | `core.depth.save_depth(result, dir, stem)` | Writes `<stem>.npy` (float32), `<stem>_relative16.png` (16-bit per-image display stretch, 0 = farthest, 65535 = nearest), `<stem>_overlay.png`, `<stem>.json` (states the values are relative). |
| Api | `Api.relative_depth(image_path, model_key="relative_depth_dav2_small")` | Saves under `<project>/analysis/depth/`. Reply: `ok`, `units: "relative"`, `units_label: "relative (unitless)"`, `output_semantics: "relative"`, `semantics`, `colorbar` (`title`, `high`, `low`), `overlay_data_uri`, `depth_npy_path`, `depth_png16_path`, `overlay_path`, `metadata_path`, `model_key`, `model_sha256`, `model_used`, `scope`, `image_hw`, `network_input_hw`. On refusal: `ok: false`, `error` with the reason. |

UI hook (for the rebuild; not wired by this unit):

- A **"Relative Depth"** action on a selected photo (beside "Run Model" in the workspace
  actions) calls `api.relative_depth(path)` and shows `overlay_data_uri`.
- The colour bar is labelled with `colorbar.title` = "nearer / farther (relative)", with
  `colorbar.high` at the top and `colorbar.low` at the bottom. **No numbers appear on it**,
  and no value is ever shown in metres or any other unit. Hover read-outs, if any, may show
  only "nearer"/"farther" ordering.
- Show `scope` and the `units_label` next to the overlay. When `ok` is false, show
  `error` as is (it explains what is missing and where to get it).
- Do not route this output into measurement tools (clearance, volume, crack width/depth,
  profile). It is not a finding and does not go to Review.

Measurement (details in `docs/model_cards/relative_depth_dav2_small.md`, evidence in
`docs/holdout/relative_depth_dav2_small.json`):

- Registry `metric`: agreement with SfM depth, ODM bellus, N=106, AbsRel 0.029 after
  scale-shift alignment (flat-scene baseline 0.063), `measured_by: "opendronekit"`. Also
  ODM sheffield_cross, N=172, AbsRel 0.012 (baseline 0.082). Agreement with
  photogrammetry, not accuracy.
- Publisher metrics (KITTI, NYU-D, DA-2K) are kept separately in `publisher_metrics`.
- Re-measure: `ODK_DEPTH_EVAL_RECON=<colmap sparse dir> ODK_DEPTH_EVAL_IMAGES=<photos>
  pytest tests/test_relative_depth.py -k sfm` or `python tools/eval_relative_depth.py sfm ...`.
