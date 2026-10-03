# Pretrained models: integration points and UI hooks

Each pretrained model in the v0.1.0-beta registry ships under its own licence (see
`docs/model_licences/<key>.md` and `LICENSES/<key>.txt`); MIT covers the code only. A
missing model, or a file whose sha256 differs from the registry's, is refused with the
reason; there is no fallback.

The web UI (`app/web/**`) is being rebuilt and is not edited by the model units. The
sections below are the contract the UI rebuild wires to.

## open_vocab_grounding_dino_tiny and vision_language_florence2_base_ft: suggestions into Review

Text-prompted object search (Grounding DINO Tiny) and captions, OCR and phrase grounding
(Florence-2-base-ft). Both are **zero-shot**: neither was trained on any OpenDroneKit
inspection class. Every box is a **suggestion** that a person confirms or rejects in
Review; every caption or OCR line is an **unverified machine description**. Nothing from
either model reaches a report until a person accepts it.

Back end:

| Layer | Entry point | Notes |
|---|---|---|
| Core | `core.open_vocab.detect_by_text(image_bgr, prompts, model_key="open_vocab_grounding_dino_tiny", box_threshold=0.35, text_threshold=0.25, tile=None)` | Verifies the sha256 of the graph **and of every companion** (tokenizer, configs) against `download.files` before loading; raises `ModelRunRefused` on a missing file, a mismatch or an empty prompt. Caption "a. b. c.", refused over 256 tokens. Frames over 2000 px are tiled (1024 px, 20 % overlap) and merged with per-label NMS. Returns `detections` (`label`, `phrase`, `confidence`, `bbox_xyxy`), `confidence_kind: "zero_shot_score"`, `output_semantics: "suggestion"`, `model_key`, `model_sha256`, `model_used`. |
| Core | `core.vision_language.florence(image_bgr, task, text="", model_key="vision_language_florence2_base_ft")` | Tasks `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>` (needs `text`), `<OCR>`, `<OCR_WITH_REGION>`. Same verification and refusal. Returns `text`, `regions` (boxes in pixels, or OCR quads), `raw`, `unverified: true`, `metric: "unmeasured"`, `note`. |
| Core | `core.model_runner.run_registered_model(image, key)` | Refuses both kinds with "needs a text prompt", pointing at Suggest (text) / Describe image. `Api.run_model` therefore refuses them too, with that reason. |
| Ops | `app.desktop_ops.findings_from_detections(..., tags=None, note=None, unscored=False)` | New optional parameters; defaults unchanged. |
| Api | `Api.suggest_objects(image_path, prompts, model_key="open_vocab_grounding_dino_tiny")` | `prompts` is a list or a string like `"insulator. bird nest."`. Each box becomes a finding: `status: "in_review"`, `include_in_report: false`, `created_by: "model"`, tags `["suggestion", "zero-shot", <key>]`, note "Zero-shot suggestion from <key>: score is uncalibrated; confirm or reject.", `confidence` = the zero-shot score, model key and sha256. Reply: `count`, `findings`, `message` ("N suggestion(s) waiting for review"), `confidence_kind`, `model_key`, `model_sha256`. With `model_key="vision_language_florence2_base_ft"` it runs phrase grounding instead; those boxes carry `confidence: null` (the model gives no score). |
| Api | `Api.describe_image(image_path, task="<CAPTION>", text="", model_key="vision_language_florence2_base_ft")` | Text tasks store one image-level `text` annotation (label "machine description (unverified)", `created_by: "model"`, `in_review`, out of reports, tags `machine-description`, `unverified`, <key>). Box tasks store unscored suggestions as above. Reply: `text`, `regions`, `unverified: true`, `metric: "unmeasured"`, `note`, `annotation`, `findings`, `model_key`, `model_sha256`. |

UI hooks (for the rebuild; not wired by this unit, `app/web/**` untouched):

- **"Suggest (text)"** in the workspace actions, beside "Prelabel"
  (`app/web/js/workspace/actions.js`): ask for the prompts (free text, e.g.
  "insulator. bird nest. rust stain.") and an image, call
  `api.suggest_objects(image, prompts)`, and report `message`
  ("N suggestions waiting for review") with `refresh: true`. Offer the Florence-2 key as an
  alternative model only in an advanced choice.
- **"Describe image"**: choose a task (Caption, Detailed caption, Read text (OCR), Read
  text with regions, Objects, Find phrase), pick an image, call
  `api.describe_image(image, task, text)` and show `text` with the label
  **"Machine description, unverified. Accuracy unmeasured by OpenDroneKit."** Never show
  it as a measurement or a finding.
- **Review list badge**: a finding whose `tags` contain `suggestion` shows a
  **"suggestion"** badge (and "zero-shot"); its score is shown as "score (uncalibrated)",
  never as a percentage or a probability. A `text` annotation tagged `unverified` shows
  an **"unverified"** badge.
- "Prelabel" keeps its detector-only filter (`onnx_yolo`, `onnx_detection`); the two new
  kinds are not detectors and must not be added to it.
- When `ok` is false, show `error` as is (it names the missing file or the mismatch and
  how to download the model).

Measured numbers and publisher metrics: `docs/model_cards/open_vocab_grounding_dino_tiny.md`
and `docs/model_cards/vision_language_florence2_base_ft.md`.
