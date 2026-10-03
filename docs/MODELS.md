# Pretrained models: integration points and UI hooks

Each pretrained model in the v0.1.0-beta registry ships under its own licence (see
`docs/model_licences/<key>.md` and `LICENSES/<key>.txt`); MIT covers the code only. A
missing model, or a file whose sha256 differs from the registry's, is refused with the
reason; there is no fallback.

The web UI (`app/web/**`) is being rebuilt and is not edited by the model units. The
sections below are the contract the UI rebuild wires to.

## Infrastructure detectors (Unit 5)

| Key | Kind | What it returns | Licence use |
|---|---|---|---|
| `road_damage_rfdetr_small` | `onnx_detr` | Boxes: longitudinal, transverse, alligator crack, pothole | commercial (Apache-2.0) |
| `powerline_ttpla_yolo11s_seg` | `onnx_yolo_seg` | Boxes plus instance masks: Line, Tower | copyleft (AGPL-3.0) |
| `powerline_insplad_detector` | `onnx_yolo` + `condition_model` | Boxes: 17 power-line assets, int8 at 640 (CPU default) | non-commercial (CC BY-NC 3.0; AGPL also claimed: owner gate) |
| `powerline_insplad_detector_1280` | `onnx_yolo` + `condition_model` | Same detector, fp32 at 1280 (GPU or patient CPU) | non-commercial (as above) |
| `powerline_insplad_condition` | `onnx_condition_classifier` | Second stage only: condition of a detected component | non-commercial (CC BY-NC 3.0) |

Every result is a pre-label for review, never a measurement. TTPLA results carry
"TTPLA: geometry only, not clearance". InsPLAD results carry "Non-commercial;
research/demonstration use per the publisher".

Back end:

| Layer | Entry point | Notes |
|---|---|---|
| Core | `core.infrastructure_detectors.load_model(key, expected_kinds=...)` | Resolves the file, checks its sha256 against the registry (`verify_model_identity` must say `verified`), opens an onnxruntime CPU session. Missing file, missing digest or mismatch: `ModelRunRefused` with the reason. Unlike `detection._load_onnx_net`, nothing returns an empty list. Registry labels must equal the ONNX `names` metadata when the graph has it. |
| Core | `core.infrastructure_detectors.run_detr(image, model, threshold)` | RF-DETR: RGB, bilinear stretch to 512, ImageNet normalisation; sigmoid, top-300 over queries x classes, normalised cxcywh -> xyxy, no NMS; the head's fifth (no-object) slot is never reported. |
| Core | `core.infrastructure_detectors.run_yolo_seg(image, model, threshold, iou)` | YOLO11-seg: the shared `detection.letterbox_image`, per-class NMS, masks = coefficients @ protos (logit > 0, i.e. sigmoid > 0.5), cropped to the box, upsampled, un-letterboxed. Each detection adds `mask_area_px` and `mask_polygons_px` (source-image pixels). |
| Core | `core.powerline.run_powerline(image, detector_key=..., condition_key=...)` | InsPLAD two-stage: detector, then the condition classifier on crops of the 5 defect-prone assets (short side 256, centre 224, ImageNet), logits masked to that asset's `asset__condition` classes before the softmax. Each eligible detection gets `condition: {label, class, probability, probabilities, model_key, model_sha256}`. A missing condition model leaves `condition: None` and says why in `condition_refused`. |
| Core | `core.model_runner.run_registered_model(image, key)` | Dispatches all of the above by kind (and by the row's `condition_model`). `output`: `detections`, `detection_count`, `confidence_kind`, `usage_note` (licence use plus the row's `result_note`); TTPLA adds `mask_note`; InsPLAD adds `condition_model`, `condition_refused`. The overlay draws boxes, labels and masks. The condition classifier alone is refused (it is a second stage). |
| App | `app.desktop_ops.prelabel_image(root, project, image, key)` | Accepts every kind in `core.detection.DETECTION_KINDS` (now including `onnx_detr`, `onnx_yolo_seg`). Writes one finding per detection: `status="in_review"`, `include_in_report=False`, `created_by="model"`, `model_key`, `model_sha256`, `confidence`; tags `model-prelabel`, `licence:<use>`, and `condition:<label>` when a condition was suggested; the note carries the condition's probability and model digest, the mask area, and the row's `result_note`. |
| App | `app.desktop_ops.findings_from_detections(..., tags=, note_suffix=)` | Extended (not forked) with optional tags and a note suffix. `findings_from_model_result(root, project, image, result)` builds both from the registry row. |
| Api | `Api.model_status()` | Each model row now has `is_detector` (kind in `DETECTION_KINDS`), `licence_use` and `output_semantics`. |
| Api | `Api.prelabel_image(image_path, model_key)` | Unchanged signature; works for the five keys above (not the condition classifier). Reply adds `usage_note`. |
| Api | `Api.run_model(model_key, image_path)` | Unchanged; returns the masks/conditions in `output` and the overlay. |
| Api | `Api.infrastructure_models()` | The five rows with `identity` (verified / missing / mismatch / unrecorded), `can_run`, `licence`, `metric` (with `measured_by`), `result_note`. |
| Api | `Api.powerline_inspection(image_path, model_key="powerline_insplad_detector", prelabel=True)` | Two-stage run; with `prelabel` the detections become in-review findings. Reply: the run, `findings`, `count`, `overlay_data_uri`. |

UI hooks (for the rebuild; not wired by this unit):

- **Prelabel menu** (`app/web/js/workspace/actions.js`, "Prelabel", around line 721)
  still hard-codes `["onnx_yolo","onnx_detection"]`. Replace it with
  `models.filter(m => m.exists && m.is_detector)` from `api.model_status()`. Without that
  change the three new detector kinds work through the Api but do not appear in that
  menu. (`powerline_insplad_detector*` are `onnx_yolo`, so they already appear.)
- **Licence badge** beside each model in "Run Model" and "Prelabel", from `licence_use`:
  `commercial` -> "Commercial OK", `copyleft` -> "Copyleft (AGPL)", `non-commercial` ->
  "Non-commercial". Link to `docs/model_licences/<key>.md`.
- **Result panel**: show `usage_note` verbatim with every run. For TTPLA draw
  `mask_polygons_px` and never label a mask with a distance. For InsPLAD show the condition
  as "suggested: <label> (p=...)" on the box, and `condition_refused` when present.
- **Review**: findings arrive `in_review`; the condition is in the note and the
  `condition:<label>` tag. Nothing reaches a report until a person accepts it.
- **Metric display**: show `metric` with its `split` and `measured_by`
  ("measured by OpenDroneKit" or "publisher reports ... on ..."); never unlabelled.
