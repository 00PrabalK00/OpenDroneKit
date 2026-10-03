# Models: licences, identity and UI hooks

Each model is a row in `models/model_registry.json`. OpenDroneKit's MIT licence covers the
code only; each set of weights ships under its own terms. `WEIGHTS_LICENSES.md` is
generated from the registry and `docs/model_licences/*.md` by
`python -m tools.build_weights_licences`, and `tests/test_release_weight_urls.py` fails when
it is stale.

## Registry fields (licence plumbing)

All optional; a row without them reads as empty, never as "cleared".

| Field | Shape | Read by |
|---|---|---|
| `licence` | `{weights, weights_url, data, data_url, use: commercial \| copyleft \| non-commercial, attribution, redistribution: allowed \| held \| forbidden, held_reason, copyleft, owner_gate}` | `core.models.licence_summary`, `Api.model_status`, the release test, the licence builder |
| `source` | `{repo, revision, files, base_model, base_model_url}` | model cards |
| `companions` | `["<key>/<file>", ...]` | `core.models.model_identity` (hashes the primary plus every companion) |
| `metric` | `{value, name, split, measured_by: opendronekit \| publisher, source_url}` or `"unmeasured"` | cards, `WEIGHTS_LICENSES.md` |
| `output_semantics` | `suggestion \| relative \| detection \| mask \| classification` | `Api.model_status` |
| `download` | `{release, license, files: [{name, url, sha256, bytes}]}` | the first-run downloader (roadmap/installer) |

### Identity of multi-file models

A row without `companions` keeps its digest exactly as before (`sha256_onnx_model`: the
graph plus any external-data sidecars). A row with `companions` is hashed by
`core.models.sha256_model_files`: the primary file, then each companion sorted by file name,
each contributing `name + NUL + bytes`. A missing companion makes the model `missing`, and a
swapped one makes it `mismatch`; either way the runner refuses.

## Api

`Api.model_status()` returns, for each registry row, the existing status fields plus:

| Key | Meaning |
|---|---|
| `licence_use` | `commercial`, `copyleft`, `non-commercial`, or `""` when the row has no licence block |
| `licence_badge` | `Commercial OK`, `Copyleft AGPL`, `Non-commercial`, or `""` (no badge, never a guessed one) |
| `licence_weights` | the weights licence string |
| `licence_terms` | repo path of the licence page, `docs/model_licences/<key>.md` |
| `redistribution` | `allowed`, `held` or `forbidden` |
| `output_semantics` | as in the registry; `suggestion` means a person must confirm every result |

## UI hooks (for the UI rebuild; `app/web/**` was not edited here)

- **Run Model** and **Prelabel** model pickers: show `licence_badge` beside each model name,
  linking to `licence_terms`. No badge when it is empty. A `Non-commercial` badge should
  carry the tooltip "Research, evaluation or educational use only".
- When `output_semantics == "suggestion"`, label results "Suggestion: confirm in Review" and
  never as findings until a person accepts them.
- Prelabel currently filters detector kinds to `["onnx_yolo", "onnx_detection"]`; the badge
  applies to whatever rows that filter keeps.
- A model whose status is not `verified`/`installed` must show the registry's refusal reason;
  the app refuses rather than guesses.

## Release state of the in-house weights (v0.1.0-beta)

| Key | Decision | Badge |
|---|---|---|
| `crack_presence_classifier` | published | Commercial OK |
| `solar_thermal_anomaly_classifier` | published | Commercial OK |
| `rail_obstacle_detector` | published (AGPL-3.0) | Copyleft AGPL |
| `rail_corridor_segmentation` | published (NVIDIA SegFormer NC) | Non-commercial |
| `corrosion_severity_segmentation` | published (NVIDIA SegFormer NC) | Non-commercial |
| `crack_segmentation` | published (NVIDIA SegFormer NC, data notes) | Non-commercial |
| `structural_multiclass_detector` | published (CODEBRIM NC + AGPL; OWNER GATE) | Non-commercial |
| `solar_cell_defect_detector` | published (PVEL-AD research-only + AGPL; OWNER GATE) | Non-commercial |
| `shared_semantic_landcover` | **held**: CC BY-SA 4.0 and CC BY-NC-SA 4.0 data are incompatible; fix is a retrain without the NC regions | Non-commercial (not published) |
