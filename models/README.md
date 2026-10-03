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

To train, export, and install a model into this folder, use the training rig:

```bash
python -m training.datasets.download crack        # fetch source data
python -m training.datasets.prepare crack         # normalise to trainer layout
python -m training.train_seg --config training/configs/crack_segformer_b5.yaml
python -m training.export_onnx --run training/runs/crack_segformer_b5 --kind seg
python -m training.register --run training/runs/crack_segformer_b5 --key crack_segmentation
python -m training.register --list                # what is installed, with real sha256
```

`register.py` refuses to install an export whose ONNX-vs-torch parity check failed,
and records the true sha256, source run, training datasets, and licences in
`manifests/model_provenance.json`.

Runtime code resolves weights through `model_registry.json` relative to this folder.

## Supported Model Runtime (Current Build)

Every installed model is an ONNX file. The registry `kind` says how it is read:
`onnx_yolo` / `onnx_detection` (detectors), `onnx_segmentation`,
`onnx_segmentation_multiclass`, `onnx_classification` and `onnx_shared_semantic`.
`pytorch_checkpoint` entries are legacy and awaiting weights.

If a model key is registered but its file is missing, nothing falls back silently and
nothing heuristic is ever presented as a model's answer:

- **Solar, metal (corrosion) and corrosion-severity analysis refuse.** The result row
  says `model_used=refused` with the reason, and the report and the finished-job
  message name it. There is no heuristic for these questions.
- **Crack and structural detection** can still run their classical image rules, but the
  result is labelled `model_used=heuristic` (or `onnx_fallback_heuristic` when the model
  failed at inference), and heuristic regions are **never** turned into findings, scored,
  or put in a report: they stay in the run summary as screening output.
- A file whose sha256 does not match the registry is reported as a mismatch by the
  identity check (`Api.verify_models`), never as verified.

Nine of the fifteen registry entries are `installed` (weights measured, recorded by
sha256); the other six are `awaiting_weights`.

## Provenance

Artifact source mapping (with hashes) is tracked in:

`models/manifests/model_provenance.json`
