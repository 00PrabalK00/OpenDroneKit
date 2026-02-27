# OpenDroneKit

<p align="center">
  <img src="logo.png" alt="OpenDroneKit Logo" width="220" />
</p>

Offline-first drone inspection toolkit for mission planning, data capture QA, defect analytics, reconstruction, and reporting.

## What This Project Does

- Plans repeatable inspection missions (grid, facade, corridor, tower, solar, waypoints, linked missions).
- Applies mission safety constraints (geofence, no-fly zones, altitude bands, standoff, RTH rules).
- Supports terrain-aware planning (AGL/AMSL mode, terrain source input, slope-normal camera behavior, wind-aware speed adjustments).
- Runs post-flight analysis (coverage QA, crack/metal/structural/solar defect detection, crack propagation).
- Builds reconstruction artifacts (point cloud, mesh, orthomosaic, DSM/DTM, digital twin metadata).
- Generates project reports from run outputs.

## UI Workflow

- `Projects`: create/select project, offline/sync status.
- `Missions`: map-based mission design + export.
- `Fly`: preflight and run controls.
- `Data`: dataset import and quick review.
- `Analysis`: operations, full pipeline, crack/propagation, reconstruction, defect models.
- `Reports`: standard/advanced report generation.

## Quick Start

```bash
python main.py
```

CLI pipeline:

```bash
python run_pipeline.py --images <dataset_dir> --output final_toolkit_outputs
```

## Repository Layout

```text
main.py
run_pipeline.py
mission/
core/
ui/
models/
training/
```

## Model Assets And Licensing

Model files and provenance:

- Registry: `models/model_registry.json`
- Provenance map: `models/manifests/model_provenance.json`

Current externally sourced model origins used in this repo:

1. `training/sources/GBH-YOLOv5`  
   License file present: `GPL-3.0` (`training/sources/GBH-YOLOv5/LICENSE`)
2. `training/sources/crack-segmentation`  
   License file present: `MIT` (`training/sources/crack-segmentation/LICENSE`)
3. `training/sources/Concrete-defect-detection`  
   No license file found in the cloned source tree at time of integration.

Important compliance note:

- If a model source has no explicit license, do not assume redistribution rights.
- Before distributing binaries, Docker images, or public model weights, verify and document rights for each model file.
- For uncertain model assets, replace them with models that have explicit, compatible licenses.

## Outputs

By default, outputs are written under:

```text
final_toolkit_outputs/
```

Typical run artifacts:

- `summary.json`
- `inspection_report.md`
- reconstruction outputs (`.ply`, mesh, orthomosaic, DSM/DTM, digital twin metadata)

## Notes

- Cloud reconstruction mode currently logs request/response and uses local fallback in this build.
- GUI flight execution is simulated workflow control, not direct autopilot flight control.
