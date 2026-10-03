# Model card: corrosion_severity_segmentation

| | |
|---|---|
| Registry key | `corrosion_severity_segmentation` |
| Release asset | `corrosion_severity_segmentation.onnx` (v0.1.0-beta) |
| sha256 | `edcecd84cc6839d76b7db3e5a6475ec694636fc6cffe545ca4718c7147b5002f` |
| Size | 109,917,813 bytes |
| Weights licence | LicenseRef-NVIDIA-SegFormer (NVIDIA Source Code License for SegFormer, non-commercial: research or evaluation only) |
| Use | non-commercial |
| Output | mask; a review aid: every result goes to Review for a person to confirm |

## Task

Four-grade corrosion condition-state segmentation (`good`, `fair`, `poor`, `severe`). The graph ends in a softmax over four channels and the answer is the argmax; it is not a binary mask, so `score_threshold` does not apply as it does to the binary segmenters.

## Architecture and input

NVIDIA SegFormer-B2 (`nvidia/mit-b2`) fine-tuned at 512 px for 80 epochs with log-inverse class weighting (run `corrosion_segformer_b2_cs`). 440 prepared images: 330 train / 66 val / 44 test. ONNX agrees with torch on every one of 262,144 pixel decisions (max abs prob diff 2.7e-06); loads in `cv2.dnn`.

Input size 512 px; registry kind `onnx_segmentation_multiclass`; score threshold 0.5.

## Training data and licence

- Data: Corrosion Condition State Semantic Segmentation Dataset (Virginia Tech), CC0 1.0
- Attribution: Bianchi, E. & Hebdon, M., Corrosion Condition State Semantic Segmentation Dataset, University Libraries, Virginia Tech, doi:10.7294/16624663.v2 (CC0 1.0; credit given as courtesy). Fine-tuned from NVIDIA SegFormer nvidia/mit-b2, NVIDIA Source Code License for SegFormer.
- Full terms: `docs/model_licences/corrosion_severity_segmentation.md`, `LICENSES/corrosion_severity_segmentation.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | validation (66 images) |
| Mean IoU | 0.5769 |
| Pixel accuracy | 0.8497 |
| Per-class IoU | good 0.871, fair 0.498, poor 0.450, severe 0.489 |
| Severe recall | 0.788 (514,827 of 653,071 severe pixels); 124,241 lost to 'poor', one grade down |

## Known weaknesses

- The classes are an ordinal scale and errors are almost all between adjacent grades: a reported grade can be one step off.
- 440 images from a single source: a first indication, not a settled measurement.
- Not tested on Indian infrastructure imagery.
- The 44-image test split has no recorded measurement; the figures above are validation.
