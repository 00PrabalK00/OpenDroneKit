# Architecture

The shape of the system, and the one rule that explains most of its odd decisions.

## The rule

**Refusal over fabrication.** Every capability either produces a result it can account
for, or refuses and says why. Nothing degrades quietly into a plausible answer.

This is not a style preference. A drone survey produces numbers that people act on:
a volume that gets invoiced, a crack width that decides a repair, a corridor declared
clear. A wrong number that looks right is worse than no number, because no number
prompts a question and a wrong one does not.

Consequences visible throughout the codebase:

- A reconstruction without geotags or control reports that it has **arbitrary scale**,
  rather than emitting distances nobody should trust.
- Ponding and deformation require a stated vertical accuracy and refuse to report
  anything smaller than twice it.
- An asset inventory refuses instances that carry no model digest, because a count
  nobody can audit is indistinguishable from a guess.
- Demo artefacts are stamped `synthetic: true` recursively, so a single finding lifted
  out of a demo still declares itself.

## Layers

```
  app/         Desktop shell and the Api facade. One method per capability.
  mission/     Mission planning. Pure geometry, no I/O, no autopilot.
  core/        Measurement, reconstruction, detection, change, packs.
  services/    Multi-user web API (FastAPI), workers, database.
  training/    Corpus preparation, trainers, ONNX export, registration.
  sdk/         Plugin system and client library.
  tools/       Operational scripts: feature status, Kaggle kernels, SITL.
  docs/        This, plus the feature registry.
```

Dependencies point downwards: `app` uses `mission` and `core`; `core` does not import
`app`. `mission` imports nothing from the rest of the toolkit, which is why the mission
engine can be used as an independent library (`mp.engine.independent`).

## The feature registry

`docs/features/registry.py` lists all 167 specified capabilities. Each row carries a
*claimed* status and a list of tests.

**Status is computed, never asserted.** `tools/feature_status.py` runs the tests and
takes the lower of what the row claims and what the evidence supports:

```
not_started → in_progress → implemented → verified
```

A row claiming `verified` whose tests fail is reported at the lower level. A row whose
tests pass but which is honestly incomplete stays where its author put it — several
rows sit at `in_progress` with a note explaining exactly what is missing, and that note
is part of the row.

```bash
python tools/feature_status.py            # summary
python tools/feature_status.py --markdown # rewrite docs/FEATURES.md
python tools/feature_status.py --strict   # non-zero exit on any downgrade
```

## Mission planning

A mission is compiled, not assembled by hand:

```
  template (grid, facade, orbit, mapping_3d, …)
      ↓  _mission_primitives()
  MissionPrimitive[]        kind + params
      ↓  _compile_primitive()
  _CapturePose[]            x, y, alt, yaw, gimbal pitch, trigger, dwell
      ↓
  MissionPlan               waypoints, recipe, autopilot commands, estimates
```

One trap is worth knowing: `_compile_primitive` normalises a primitive's kind through
the *mission template* alias table, and that table maps anything unrecognised to
`"grid"`. A new primitive kind therefore does not fail — it silently compiles as a full
nadir grid. Kinds that are not templates are listed in `_NON_TEMPLATE_PRIMITIVE_KINDS`
and matched before normalisation.

## Models

Models are ONNX, loaded through `cv2.dnn` at runtime. The path from a trained
checkpoint to a usable model is deliberately gated:

```
  train  →  export_onnx  →  parity check  →  register  →  digest recorded
```

The parity check compares torch and ONNX outputs on the same input and **fails the
export** if they disagree beyond tolerance. It also confirms `cv2.dnn` can read the
graph, because onnxruntime accepting a file says nothing about the runtime that will
actually load it.

Registration records a sha256. Identity matters more than existence: a model swapped
after registration keeps every number from the run that measured a different file.

## Flight

`core/mission_planner_bridge.py` speaks MAVLink. Two things to know:

- **Mission sequence 0 is reserved for home.** ArduPilot overwrites whatever is stored
  there. The bridge prepends a home slot on upload and strips it on download, so
  callers see the plan they sent. Telemetry's waypoint index is reported in the same
  plan numbering.
- Telemetry callbacks run on the listener thread. A subscriber that raises is caught,
  because losing the listener does not look like a crash — the numbers simply stop
  moving, which reads as a quiet aircraft.

SITL (`docs/SITL.md`) is how flight code is verified against real ArduPilot rather than
against a mock. This is not ceremony: a mock stores what it is given, and only a real
autopilot has an opinion about sequence 0.

## Two-agent development

Parts of this repository are developed in parallel by two agents with file-level lane
ownership (`app/web/**`, `services/**`, `tests/sitl/**`, `tools/sitl/**` belong to one
lane). If you are automating changes here, respect the lanes or you will produce
conflicting edits to the same registry rows.
