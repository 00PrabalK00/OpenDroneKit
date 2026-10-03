# Model card: rail_obstacle_detector

| | |
|---|---|
| Registry key | `rail_obstacle_detector` |
| Release asset | `rail_obstacle_detector.onnx` (v0.1.0-beta) |
| sha256 | `7a53c8b9eaeabf956a1f2fc3a25ca3ff39f6217c5cb3a3b744f54e017e8ced23` |
| Size | 101,514,362 bytes |
| Weights licence | AGPL-3.0 |
| Use | copyleft |
| Output | detection; a review aid: every result goes to Review for a person to confirm |

## Task

Object detection of discrete obstacles lying on or beside railway track, in UAV corridor imagery. Six classes: `Barrel`, `Boulder`, `Branch`, `IronRod`, `Jerrycan`, `Person`.

## Architecture and input

Ultralytics YOLO11l fine-tuned at 640 px for 60 epochs on a Kaggle P100 (run `kaggle_v7/rail`). ONNX opset 17; ONNX/PyTorch parity max abs diff 4.9e-04; loads in `cv2.dnn` (`models/manifests/model_provenance.json`).

Input size 640 px; registry kind `onnx_yolo`; score threshold 0.25.

## Training data and licence

- Data: UAV Railroad Images for Segmentation and Obstacle Detection (UAV-RSOD), V2 obstacle detection archive, CC BY 4.0
- Attribution: R S, Rampriya (2024), UAV Railroad Images for Segmentation and Obstacle Detection, Zenodo, doi:10.5281/zenodo.12606374, CC BY 4.0. Fine-tuned from Ultralytics YOLO11l (AGPL-3.0).
- Full terms: `docs/model_licences/rail_obstacle_detector.md`, `LICENSES/rail_obstacle_detector.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | validation, 273 held-out images |
| mAP50 | 0.824 |
| mAP50-95 | 0.493 |
| Precision / recall | 0.762 / 0.743 |
| Per-class mAP50-95 | Barrel 0.590, Boulder 0.532, IronRod 0.531, Person 0.493, Jerrycan 0.418, Branch 0.395 |

## Known weaknesses

- Recall 0.743: roughly one obstacle in four is missed. An empty result means the model found nothing, not that the corridor is safe. Never present it as clearing a track.
- Person is mid-table at 0.493 mAP50-95.
- Trained on 1,338 corridor images of discrete objects. It does not detect track geometry faults, ballast condition, missing fasteners or vegetation encroachment.
- Not measured on Indian railway imagery.
