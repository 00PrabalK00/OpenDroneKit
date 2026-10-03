# Licence: powerline_insplad_detector_1280

| | |
|---|---|
| Weights licence | **CC-BY-NC-3.0** |
| Full text | [`LICENSES/powerline_insplad_detector_1280.txt`](../../LICENSES/powerline_insplad_detector_1280.txt) |
| Licence source | https://huggingface.co/ACloudCenter/yolo11s-insplad-powerline |
| Use | **non-commercial use only** (CC BY-NC 3.0) |
| Released file | `powerline_insplad_detector_1280.onnx`, 38,457,356 bytes, sha256 `485c730a83d5162984f2731ae8baeb24f2970002d7cd5132a92f628db5cd3aa2` |
| Also claimed | AGPL-3.0 (Ultralytics), see the owner gate below |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence, as
published by their authors.

## Attribution

Weights: ACloudCenter/yolo11s-insplad-powerline (code: github.com/Josh-E-S/powerline-inspection-demo). Data: Vieira-e-Silva, Felix, Simoes, Teichrieb, 'InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images', International Journal of Remote Sensing 44(23), 2023.

## Provenance of the file

- `ACloudCenter/yolo11s-insplad-powerline` @ `e444141f5e91f26acc7ecc809b462bc32b2f1d17`, `detector-1280-fp32.onnx`, byte for byte (sha256 equals the HF LFS id), renamed only.

## Training data

InsPLAD, CC BY-NC 3.0 (Vieira-e-Silva et al., IJRS 2023) (https://github.com/andreluizbvs/InsPLAD)

## OWNER GATE: InsPLAD licence conflict (open, not resolved here)

The weights' publisher licenses them **CC BY-NC 3.0** ("non-commercial use only ... these
weights are derivative works of the InsPLAD dataset"); the InsPLAD dataset's own LICENSE is
the CC BY-NC 3.0 legal code (verified). Ultralytics states that YOLO11 fine-tunes are
**AGPL-3.0**, and the detector's ONNX metadata says so. AGPL-3.0 does not allow the extra
non-commercial restriction, so no single set of terms satisfies both notices.

Neither licence forbids redistribution outright, so under the owner's rule the files are
staged and **flagged non-commercial with both notices** in `LICENSES/powerline_insplad_detector_1280.txt`. The owner
must confirm this explicitly in `WEIGHTS_LICENSES.md` (Unit 6 assembles it). If the answer is
no, drop `powerline_insplad_detector`, `powerline_insplad_detector_1280` and
`powerline_insplad_condition`; TTPLA is then the only power-line model.

Scope per the publisher: research and demonstration, not operational decisions. Results in OpenDroneKit say "non-commercial; research/demonstration per publisher".
