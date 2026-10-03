# Model card: rail_corridor_segmentation

| | |
|---|---|
| Registry key | `rail_corridor_segmentation` |
| Release asset | `rail_corridor_segmentation.onnx` (v0.1.0-beta) |
| sha256 | `1885757000e89f396c16ead68ccc4c8ce14a46f593e42abdabcbdeba6312ada1` |
| Size | 109,908,496 bytes |
| Weights licence | LicenseRef-NVIDIA-SegFormer (NVIDIA Source Code License for SegFormer, non-commercial: research or evaluation only) |
| Use | non-commercial |
| Output | mask; a review aid: every result goes to Review for a person to confirm |

## Task

Binary segmentation of the rail CORRIDOR surface in UAV imagery (one class, `rail_corridor`). Rail lines are not a class.

## Architecture and input

NVIDIA SegFormer-B2 (`nvidia/mit-b2`) fine-tuned at 512 px for 60 epochs (run `rail_corridor_segformer_b2`). ONNX/PyTorch parity max abs diff 2.4e-07; loads in `cv2.dnn`.

Input size 512 px; registry kind `onnx_segmentation`; score threshold 0.5.

## Training data and licence

- Data: UAV-RSOD, V1 segmentation archive (Rail Corridor aspect), CC BY 4.0
- Attribution: R S, Rampriya (2024), UAV Railroad Images for Segmentation and Obstacle Detection, Zenodo, doi:10.5281/zenodo.12606374, CC BY 4.0. Fine-tuned from NVIDIA SegFormer nvidia/mit-b2, NVIDIA Source Code License for SegFormer.
- Full terms: `docs/model_licences/rail_corridor_segmentation.md`, `LICENSES/rail_corridor_segmentation.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | validation |
| IoU | 0.681 |
| Dice | 0.810 |
| Precision / recall | 0.752 / 0.878 |

## Known weaknesses

- About 315 distinct Indian UAV images in the corpus, so 0.681 is a first indication rather than a settled figure.
- Not tested outside the flights it was trained on.
- The Rail Lines aspect of the source archive was excluded on purpose; a rails model would be a separate run.
