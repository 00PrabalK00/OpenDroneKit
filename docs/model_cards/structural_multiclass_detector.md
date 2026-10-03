# Model card: structural_multiclass_detector

| | |
|---|---|
| Registry key | `structural_multiclass_detector` |
| Release asset | `structural_multiclass_detector.onnx` (v0.1.0-beta) |
| sha256 | `359817c663f4cb7772f44e4370095c116efb9d0123478bc58aa618884380c445` |
| Size | 227,714,515 bytes |
| Weights licence | AGPL-3.0 AND LicenseRef-CODEBRIM (non-commercial research and educational use) |
| Use | non-commercial (also AGPL-3.0; see the OWNER GATE in `WEIGHTS_LICENSES.md`) |
| Output | detection; a review aid: every result goes to Review for a person to confirm |

## Task

Multi-class structural defect detection on concrete bridge imagery. Classes: `Efflorescence`, `CorrosionStain`, `Crack`, `Spallation`, `ExposedBars`, `Background`.

## Architecture and input

Ultralytics YOLO11x fine-tuned at 640 px (run `structural_yolo11x`) on the CODEBRIM data via the Roboflow export `codebrim_structural` (2,169 train / 486 val / 106 test images). ONNX opset 17; ONNX/PyTorch parity max abs diff 1.0e-03; loads in `cv2.dnn`.

Input size 640 px; registry kind `onnx_yolo`; score threshold 0.25.

## Training data and licence

- Data: CODEBRIM (Goethe University / FIAS), Zenodo record 2620293, licence 'Other (Non-Commercial)'; used via the Roboflow export codebrim_structural
- Attribution: Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos and Visvanathan Ramesh, 'Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset', CVPR 2019. Fine-tuned from Ultralytics YOLO11x (AGPL-3.0).
- Full terms: `docs/model_licences/structural_multiclass_detector.md`, `LICENSES/structural_multiclass_detector.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | structural_det test (106 images, 743 defect boxes), IoU 0.5, letterbox decoder |
| F1 / recall at score 0.25 (the general threshold) | 0.392 / 0.339 |
| F1 / recall at score 0.15 (the pre-label threshold) | 0.398 / 0.409 |

## Known weaknesses

- F1 near 0.4 and recall 0.34 to 0.41: most defects are missed or mislocalised. Every detection goes to human review.
- `Background` is a non-defect label (`non_defect_labels`) and is never a finding.
- Not measured on Indian sites.
