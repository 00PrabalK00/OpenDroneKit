# Licence: powerline_ttpla_yolo11s_seg

| | |
|---|---|
| Weights licence | **AGPL-3.0-or-later** |
| Full text | [`LICENSES/powerline_ttpla_yolo11s_seg.txt`](../../LICENSES/powerline_ttpla_yolo11s_seg.txt) |
| Licence source | https://huggingface.co/ethan0502/ttpla-yolo11-seg |
| Use | copyleft: commercial use allowed, but anyone who distributes the weights, or offers them to users over a network in a modified program, must provide the corresponding source under AGPL-3.0 |
| Released file | `powerline_ttpla_yolo11s_seg.onnx`, 40,513,985 bytes, sha256 `28c83ee06096f8a8854acf782e7006b2d7b1009f3c01a7098c1e48b41b5ce766` |
| Corresponding source (AGPL) | https://github.com/ethan0502/ttpla-yolo11-seg |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence, as
published by their authors.

## Attribution

Weights: ethan0502/ttpla-yolo11-seg, a fine-tune of Ultralytics yolo11s-seg (AGPL-3.0). Data: Abdelfattah, Wang, Wang, 'TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines', ACCV 2020.

## Provenance of the file

- `ethan0502/ttpla-yolo11-seg` @ `b77643ae3b3b2a95b777d0e92e2da6a8744c23a5`, `yolo11s_seg_ft640.onnx`. The released file is that file **byte for byte**, renamed only; its sha256 equals the HF LFS id.
- Corresponding source for the AGPL: https://github.com/ethan0502/ttpla-yolo11-seg (training, export and deployment scripts; split files in `third_party/ttpla_splits`).

## Training data

TTPLA, Apache-2.0 (R3ab/ttpla_dataset). Images are not redistributed. (https://github.com/R3ab/ttpla_dataset)

Results in OpenDroneKit say "geometry only, not clearance": a 2-D mask locates a cable in the image and certifies no distance.
