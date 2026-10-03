# Model card: solar_cell_defect_detector

| | |
|---|---|
| Registry key | `solar_cell_defect_detector` |
| Release asset | `solar_cell_defect_detector.onnx` (v0.1.0-beta) |
| sha256 | `a57676a7141baba3605fa81dd3a817405ee28522aff9edbc3e74e12755c6783b` |
| Size | 101,783,319 bytes |
| Weights licence | AGPL-3.0 AND LicenseRef-PVEL-AD (research and academic use only) |
| Use | non-commercial (also AGPL-3.0; see the OWNER GATE in `WEIGHTS_LICENSES.md`) |
| Output | detection; a review aid: every result goes to Review for a person to confirm |

## Task

Electroluminescence (EL) cell-level defect detection for module manufacturing and lab inspection imagery. Classes: `black_core`, `finger`, `crack`, `star_crack`, `thick_line`, `horizontal_dislocation`, `vertical_dislocation`, `short_circuit`.

## Architecture and input

Ultralytics YOLO11l fine-tuned at 1024 px for 60 epochs on PVEL-AD (run `pvel_ad_yolo11l`). Label order is the checkpoint order; re-validation must use these labels, not a freshly prepared data.yaml.

Input size 1024 px; registry kind `onnx_detection`; score threshold 0.25.

## Training data and licence

- Data: PVEL-AD (Hebei University of Technology and Beihang University), released through a signed request form: 'The datasets can ONLY be used for research and academic purposes.'
- Attribution: Binyi Su, Zhong Zhou, Haiyong Chen, 'PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic Cell Anomaly Detection', IEEE Trans. Ind. Inform., doi:10.1109/TII.2022.3162846. Fine-tuned from Ultralytics YOLO11l (AGPL-3.0).
- Full terms: `docs/model_licences/solar_cell_defect_detector.md`, `LICENSES/solar_cell_defect_detector.txt`, `WEIGHTS_LICENSES.md`.

## Metrics (measured by OpenDroneKit, as recorded in the registry)

Quoted from `models/model_registry.json`; no new measurement was made for this release.

| | Value |
|---|---|
| Split | validation |
| mAP50 | 0.8835 |
| mAP50-95 | 0.5434 |
| Precision / recall | 0.812 / 0.856 |
| Per-class mAP50-95 | short_circuit 0.918, black_core 0.847, finger 0.540, horizontal_dislocation 0.504, thick_line 0.485, star_crack 0.393, vertical_dislocation 0.335, crack 0.325 |

## Known weaknesses

- NOT an aerial or thermal model: never apply it to drone photographs of installed panels.
- Four PVEL-AD classes (printing_error, corner, fragment, scratch) had under 35 boxes each and were dropped; the model cannot see them.
- Validation figures only; no separate test measurement is recorded.
