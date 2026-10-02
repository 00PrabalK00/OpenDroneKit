# OpenDroneKit Python SDK

`opendronekit` is the public, versioned Python API of OpenDroneKit. It drives the same
code as the desktop application's buttons: a script and the application plan the same
missions, refuse the same requests for the same reasons, and write the same files.

| | |
|---|---|
| Package | `opendronekit` (in this repository; typed, ships `py.typed`) |
| Version | `opendronekit.__version__`, semantic versioning |
| Plugin API | `opendronekit.plugins`, contract level `PLUGIN_API_LEVEL` ([plugins.md](../plugins.md)) |
| Reference | [reference.md](reference.md), generated from the code's docstrings |
| Tutorials | [docs/tutorials](../tutorials/README.md) |

## What it covers

| Task | Call |
|---|---|
| Open or create a project | `Workspace(data_dir)`, `.create_project(name)`, `.open_project(name_or_id)`, `.list_projects()` |
| What this machine can do | `Workspace.capabilities()`, `Workspace.mission_templates()` |
| Plan and export a mission | `Project.plan_mission(aoi, template=..., **options)` -> `MissionPlan`; `.export(formats)`, `.save()` |
| Import photographs | `Project.import_dataset(folder)` |
| Reconstruct | `Project.reconstruct(engine="auto", ...)` (waits for the job) |
| Detect | `Project.detect_cracks(image)`, `Project.run_model(key, image)`, `Project.detect(image, detector=plugin)` |
| Measure | `Project.measure_map(kind, lonlat)`, `Project.measure_image(...)`, `Project.measure_crack(...)` |
| Report | `Project.report_readiness()`, `Project.generate_report(...)` |
| Plugins | `load_plugins()`, `plugin_status()`, `opendronekit.plugins` |

## Install

The SDK is part of the OpenDroneKit repository. Set up the environment as in
[INSTALLATION.md](../INSTALLATION.md) and run Python from the repository root (or put it on
`PYTHONPATH`). Reconstruction needs COLMAP; detection with a trained model needs that
model installed. Without them the calls are refused with the reason; nothing is
substituted.

## Quickstart

This plans a survey over part of Aukerman park (Ohio) and exports it for QGroundControl and
Google Earth. It needs no data and no network, and it runs as written:
`tests/test_sdk_docs.py` executes the `python` blocks on this page.

```python
import os
import tempfile
from pathlib import Path

import opendronekit as odk

print("OpenDroneKit SDK", odk.__version__, "| plugin API", odk.PLUGIN_API_LEVEL)

work = Path(os.environ.get("ODK_TUTORIAL_DIR") or tempfile.mkdtemp(prefix="odk-sdk-"))
with odk.Workspace(work / "odk-data") as ws:
    caps = ws.capabilities()
    print("COLMAP:", caps.get("colmap_binary") or ("pycolmap" if caps.get("pycolmap") else "not found"))
    print("templates:", ", ".join(ws.mission_templates()[:6]), "...")

    project = ws.create_project("Quickstart")
    field = [[-81.7535, 41.3036], [-81.7505, 41.3036], [-81.7505, 41.3050], [-81.7535, 41.3050]]
    plan = project.plan_mission(field, template="grid", altitude_m=80)
    print(plan)

    written = plan.export(["qgc_plan", "kml"], directory=work / "missions", stem="quickstart")
    for fmt, files in written.items():
        print(f"{fmt}: {files[0].name} ({files[0].stat().st_size} bytes)")
```

`plan.summary` holds the planner's figures (`waypoints`, `distance_m`, `duration_min`,
`gsd_cm`, `warnings`); `plan.plan` is the full plan as a dict and `plan.geojson` the map
geometry.

## Refusals are exceptions with reasons

Every call returns a real result or raises:

* `odk.Refused`: OpenDroneKit declined and said why (`exc.reason`, the same text the
  application shows; `exc.payload` holds any structured detail).
* `odk.JobFailed`: a background job (a reconstruction) failed or was cancelled; `exc.job`
  is the job record.
* `odk.PluginError`: a plugin could not be found, built or run (`exc.plugin`, `exc.stage`).

All three derive from `odk.OdkError`. Nothing comes back "empty but successful" in place of
a refusal:

```python
with odk.Workspace(work / "odk-data") as ws:
    project = ws.open_project("Quickstart")
    try:
        project.plan_mission(field, template="spiral_of_doom")
    except odk.Refused as exc:
        print("refused:", exc.reason[:60], "...")
    try:
        project.measure_map("distance", field[:2])
    except odk.Refused as exc:
        print("refused:", exc.reason)
```

## Results are typed

Calls return frozen dataclasses (`MissionSummary`, `Dataset`, `Reconstruction`,
`CrackDetection`, `Detections`, `Measurement`, `Report`) with the fields a script normally
needs and `raw`, the full payload, for anything else. Results say where they came from:
`Reconstruction.georeferenced` is true only when the model was anchored to a real CRS;
`CrackDetection.source` is `"model"` or `"heuristic"`; detections carry the model key and
the sha256 of the weights that ran; measurements carry their caveats.

## Versioning and stability

* `opendronekit.__version__` follows semantic versioning. The public API is everything in
  `opendronekit.__all__` and `opendronekit.plugins.__all__`, as listed in
  [reference.md](reference.md). Removing or incompatibly changing any of it bumps the major
  version; additions bump the minor version.
* The plugin contracts are versioned separately (`PLUGIN_API_LEVEL`), so a plugin keeps
  loading across SDK releases until the plugin API's major version changes.
* `app`, `core`, `mission` and `sdk` are the implementation behind the SDK. Import them at
  your own risk: they change without notice.
* The reference is regenerated with `python tools/gen_sdk_reference.py`; the test suite
  fails if it is out of date.

## Threading

A `Workspace` serialises its calls (one in-process API instance serves all its projects),
so it is safe to share between threads, but its calls do not run in parallel.
