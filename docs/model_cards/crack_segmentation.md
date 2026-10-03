# Model card: crack_segmentation

| | |
|---|---|
| Registry key | `crack_segmentation` |
| Release asset | `crack_segmentation.onnx` (v0.1.0-beta) |
| sha256 | `3391eca15d43bcc2810c3fcfe7e9db9f99dddb5de4633d5963c705c8e1cc0c23` |
| Size | 340,011,769 bytes |
| Weights licence | LicenseRef-NVIDIA-SegFormer (NVIDIA Source Code License for SegFormer, non-commercial: research or evaluation only) |
| Use | non-commercial |
| Output | mask; a review aid: every result goes to Review for a person to confirm |

## Task

Binary crack segmentation (one class, `crack`) for locating crack extent on close-range concrete and pavement imagery. It is not a triage model: `crack_presence_classifier` should gate which tiles reach it.

## Architecture and input

NVIDIA SegFormer-B5 (`nvidia/mit-b5`) fine-tuned at 1024 px for 40 epochs (run `crack_segformer_b5`). Corpus `crack_seg`: 8,439 train / 1,568 val / 1,946 test samples. ONNX/PyTorch parity max abs diff 1.2e-07; loads in `cv2.dnn`. **Shipped decision threshold 0.85.**

Input size 1024 px; registry kind `onnx_segmentation`; score threshold 0.85.

## Training data and licence

- Data: crack_segmentation_kaggle (lakshaymiddha/crack-segmentation-dataset, Kaggle licence 'Unknown'; aggregates CRACK500, GAPs384, CFD, AEL, CrackTree200, DeepCrack, Rissbilder, Volker and others), DeepCrack (non-commercial research and educational use only), CrackForest/CFD (non-commercial research only). GAPs: academic use only, commercial use prohibited. None forbids distributing a trained model; see docs/model_licences/crack_segmentation.md
- Attribution: See docs/model_licences/crack_segmentation.md for the citation of every source dataset. Fine-tuned from NVIDIA SegFormer nvidia/mit-b5, NVIDIA Source Code License for SegFormer.
- Full terms: `docs/model_licences/crack_segmentation.md`, `LICENSES/crack_segmentation.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | validation (1,568 samples) |
| IoU, best during training | 0.6063 |
| IoU at the shipped threshold 0.85 | 0.5962 (against 0.5582 at the previous 0.25; the threshold was chosen by sweep on the validation split) |

## Known weaknesses

- The threshold was chosen on the same validation split that reports 0.5962, so that figure is optimistic; the 1,946-sample test split has no recorded measurement.
- At 0.85 thin and faint cracks are missed, so an empty mask is not evidence of a sound surface.
- Crack WIDTH is not measured: width at survey GSD is usually below one pixel.
- The weaker of the two crack models.
- Not measured on Indian sites.
