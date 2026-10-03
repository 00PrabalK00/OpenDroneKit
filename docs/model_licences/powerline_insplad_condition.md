# Licence: powerline_insplad_condition

| | |
|---|---|
| Weights licence | **CC-BY-NC-3.0** |
| Full text | [`LICENSES/powerline_insplad_condition.txt`](../../LICENSES/powerline_insplad_condition.txt) |
| Licence source | https://huggingface.co/ACloudCenter/yolo11s-insplad-powerline |
| Use | **non-commercial use only** (CC BY-NC 3.0) |
| Released file | `powerline_insplad_condition.onnx`, 81,905,176 bytes, sha256 `66635a28e0fe34a74ce273536226720baefd92359dab222d3711c0868a138831` |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence, as
published by their authors.

## Attribution

Weights: ACloudCenter/yolo11s-insplad-powerline (code: github.com/Josh-E-S/powerline-inspection-demo). Data: Vieira-e-Silva, Felix, Simoes, Teichrieb, 'InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images', International Journal of Remote Sensing 44(23), 2023.

## Provenance of the file

- `ACloudCenter/yolo11s-insplad-powerline` @ `e444141f5e91f26acc7ecc809b462bc32b2f1d17`: `classifier.onnx` (sha256 `35088a6a...47ea`) + `classifier.onnx.data` (sha256 `3aa24397...6d50`), both equal to the HF LFS ids, and `classifier_classes.json`.
- OpenDroneKit folded the external data into one file (`onnx.save_model(..., save_as_external_data=False)`). Outputs are identical to the two-file original (max abs diff 0.0 on 8 random inputs).

## Training data

InsPLAD, CC BY-NC 3.0 (Vieira-e-Silva et al., IJRS 2023) (https://github.com/andreluizbvs/InsPLAD)

## OWNER GATE: InsPLAD licence conflict (open, not resolved here)

The weights' publisher licenses them **CC BY-NC 3.0** ("non-commercial use only ... these
weights are derivative works of the InsPLAD dataset"); the InsPLAD dataset's own LICENSE is
the CC BY-NC 3.0 legal code (verified). Ultralytics states that YOLO11 fine-tunes are
**AGPL-3.0**, and the detector's ONNX metadata says so. AGPL-3.0 does not allow the extra
non-commercial restriction, so no single set of terms satisfies both notices.

Neither licence forbids redistribution outright, so under the owner's rule the files are
staged and **flagged non-commercial with both notices** in `LICENSES/powerline_insplad_condition.txt`. The owner
must confirm this explicitly in `WEIGHTS_LICENSES.md` (Unit 6 assembles it). If the answer is
no, drop `powerline_insplad_detector`, `powerline_insplad_detector_1280` and
`powerline_insplad_condition`; TTPLA is then the only power-line model.

Scope per the publisher: research and demonstration, not operational decisions. Results in OpenDroneKit say "non-commercial; research/demonstration per publisher".
