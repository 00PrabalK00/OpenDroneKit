# Models Directory

This folder is the canonical location for inference model assets used by the toolkit.

## Layout

```text
models/
  model_registry.json
  structural/
  solar/
  legacy/
    checkpoints/
  manifests/
```

## Required Behavior

1. Add model files under the correct category folder.
2. Register model metadata in `model_registry.json`.
3. Use registry keys from CLI/UI config:
   - `structural_multiclass_detector`
   - `solar_defect_detector`

## Pretrained Preset Slots

The registry already includes preset keys you can fill with converted ONNX files:

- Structural:
  - `structural_multiclass_detector`
  - `structural_codebrim_detector`
  - `structural_concrete_yolov8_baseline`
- Solar:
  - `solar_defect_detector`
  - `solar_pv_multidefect_detector`
- Legacy segmentation:
  - `legacy_swin_unet`
  - `legacy_crack_segmentation_unet`

Drop files at the configured `path` under `models/` and switch keys in UI/CLI.

## Legacy Migration

The toolkit attempts to migrate:

`CConCrack_SwinUNet_Final/checkpoints/best_model.pth`

to:

`models/legacy/swin_unet_best_model.pth`

if the legacy source exists.

To archive additional legacy checkpoints into this folder, run:

```bash
python training/scripts/sync_legacy_models.py
```

Runtime code uses files from `final_toolkit/models/` only.

## Supported Model Runtime (Current Build)

Current model-backed inference path is ONNX via OpenCV DNN (`onnx_yolo` kind).

If a model key is registered but the file is missing, the toolkit automatically falls back to classical heuristics.

## Provenance

Artifact source mapping (with hashes) is tracked in:

`models/manifests/model_provenance.json`
