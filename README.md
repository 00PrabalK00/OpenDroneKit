# OpenDroneKit

<img src="logo.png" alt="OpenDroneKit" width="180" />

Offline-first drone inspection and geospatial toolkit: plan repeatable inspection
missions, fly them over MAVLink, then turn the imagery into georeferenced
reconstruction products and defect analytics.

Built for GIS / geomatics work — outputs land in a real coordinate reference system
and open directly in QGIS.

## What it does

**Mission planning** — 16 templates (grid, double grid, corridor, facade, tower, solar,
orbit, panorama, 360 bubble, waypoints, roof, linear, lateral, magnetic, linked, adaptive)
with real constraint geometry: ray-cast geofence containment, no-fly polygons with
segment-level detour insertion, altitude bands, standoff, and RTH rules.

**Terrain-aware planning** — AGL/AMSL follow modes with terrain loaded from GeoTIFF,
ESRI ASCII grid, CSV samples, or a fitted plane. Slope-normal gimbal attitude and
crosswind/headwind speed penalties.

**Flight** — MAVLink upload carrying yaw, gimbal pitch, dwell, and camera triggers, plus
geofence and rally points. Live telemetry, arm / start / pause / RTL / abort.

**Reconstruction** — COLMAP structure-from-motion with bundle adjustment, camera
intrinsics from an EXIF sensor database, and georeferencing solved as a RANSAC Helmert
similarity between recovered camera centres and image geotags. Outputs a
Cloud-Optimized GeoTIFF orthomosaic, DSM, DTM, and hillshade in the automatically
selected UTM zone, plus a point cloud, Poisson mesh, and camera-track GeoJSON.

**Analysis** — coverage QA, crack / structural / solar / corrosion detection, Paris-law
crack propagation, asset health scoring, and back-projection of 2-D defects onto the
reconstructed surface to produce georeferenced defect polygons with square-metre areas.

**Export** — QGroundControl `.plan`, QGC WPL `.waypoints`, DJI WPML `.kmz`, Litchi CSV,
KML, GeoJSON, and Cloud-Optimized GeoTIFF.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

The desktop shell uses the operating system's built-in webview (Edge WebView2 on
Windows), with a native menu bar and a MapLibre GL map canvas. There is no bundled
browser and no Qt dependency.

Tests:

```bash
python -m pytest
```

The suite covers the geospatial measurement path against surfaces with analytically
known answers, the mission exporters, the job lifecycle, dataset preparation, and a
set of regression tests for fabrications that were removed (synthetic point-cloud
densification, an ignored outbound cloud request, a heuristic reported as a model).

Command-line pipeline:

```bash
python run_pipeline.py --images <dataset_dir> --engine colmap --output final_toolkit_outputs
```

Useful flags: `--engine {auto,colmap,custom}`, `--dense` / `--no-dense`, `--epsg <code>`,
`--recon-profile {fast_preview,standard,inspection_high_accuracy}`.

## Repository layout

```
main.py                  desktop entry point
run_pipeline.py          CLI pipeline
app/                     desktop shell (window, menu, JS bridge, web UI, SQLite store)
core/                    geo, reconstruction, detection, propagation, reporting, flight
mission/                 mission planner and exporters
models/                  model registry and provenance
training/                dataset fetchers and training scripts
```

## Current capabilities and limits

This section states what actually works in this build.

| Area | State |
|---|---|
| Mission planning and constraints | Working, verified against drawn geometry |
| Mission export (5 formats) | Working; QGC WPL round-trips through pymavlink |
| COLMAP SfM + georeferencing | Working. Verified on the OpenDroneMap Aukerman survey: 77/77 images registered, 1.27 px mean reprojection error, geo RMSE 1.22 m, EPSG:32617 |
| Georeferenced COG outputs | Working (orthomosaic, DSM, DTM, hillshade) |
| Dense MVS | **Requires a CUDA COLMAP build.** The pycolmap wheels are CPU-only, so dense stereo is skipped and raster resolution is reduced to what the sparse cloud supports. The run reports this explicitly. |
| Trained defect models | **Two installed, two still missing.** Crack segmentation and multi-class structural detection run through real ONNX weights and report `model_used` as `onnx:<file>`. Solar and corrosion have no usable weights, so those paths still fall back to classical image processing and report `heuristic`, never as AI. |
| Crack segmentation quality | **Measured on a held-out test split:** SegFormer-B5 @1024 reaches IoU 0.637, precision 0.811, recall 0.748 at the registered decision threshold of 0.85. The classical baseline it replaced reaches IoU 0.045 on the same data, and the earlier B2 model reached 0.515. |
| Structural detection quality | **Measured:** YOLO11x on CODEBRIM, mAP50 0.417 / mAP50-95 0.201. Per class: ExposedBars 0.330, Spallation 0.306, CorrosionStain 0.254, Efflorescence 0.193, Crack 0.124. Empty results are reported as empty rather than filled with invented findings. |
| Solar and corrosion models | **Not shipped.** Detectors were trained (solar mAP50 0.262, corrosion mAP50 0.242) and passed the ONNX parity gate, but are deliberately unregistered: their trained classes do not match the schema those registry keys declare, and wiring them in would emit findings labelled `Clear` or `non-corrosion` into a pipeline expecting defect classes. |
| Model identity | The registry records the digest of the file whose metrics were measured, and the installed file is hashed and compared against it at load. A replaced file is reported as a mismatch; a model with no recorded digest is reported as unrecorded, never as verified. |
| Training rig | Working end to end: download → prepare → train → ONNX export with a torch-parity check → register with real sha256. Verified at full scale: the crack model is a 40-epoch run on a rented 4090, the structural and detector runs were trained on Kaggle. |
| Volume / measurement / risk stages | Working against the real DSM/DTM. Volume verified to 0.00 m³ error against an analytic test surface. |
| Cloud reconstruction | Not implemented. Requesting it runs locally and reports that it did. No data leaves the machine: the custom engine previously POSTed the imagery's absolute path and image count to a configured endpoint and then discarded the reply, which has been removed. |
| MAVLink upload | Working and verified against a MAVLink peer: mission, geofence, and rally upload each use the request/ack transfer protocol and land in the correct `MAV_MISSION_TYPE` slot. Gimbal, yaw, dwell, and camera-trigger items survive a full upload/download round trip. Requires MAVLink 2 (fence and rally are v2-only). |
| GUI flight control | Real MAVLink commands when a MAVLink driver is connected. A `mock` driver is also selectable and is always labelled SIMULATED in the UI. |
| `smart_adaptive` template | Genuinely adaptive when given interest regions or a prior survey's defect points: those get a finer cross-hatched pass at lower altitude. With no such input it emits a uniform grid and labels its poses `grid_uniform_no_interest_input` rather than implying adaptivity. |
| FEniCSx phase-field propagation | Not available — no `cracksim` solver ships here. Reports unavailable explicitly. The Paris-law model is implemented and is the supported path. |

## Model assets and licensing

`models/model_registry.json` lists the model keys the pipeline looks for.
`models/manifests/model_provenance.json` records what is actually installed.

**At present no weights ship with this repository.** A previous manifest listed four
ONNX targets sharing a single checksum — one concrete-defect model copied under solar
and CODEBRIM filenames. Those entries have been retracted rather than corrected.

To train real weights, see `training/`:

```bash
python -m training.datasets.download --list     # catalogue and credential status
python -m training.datasets.download public     # no account needed
python -m training.datasets.download crack solar structural corrosion

python -m training.datasets.prepare --list      # task catalogue
python -m training.datasets.prepare crack       # normalise to trainer layout

python -m training.train_seg --config training/configs/crack_segformer_b5.yaml
python -m training.train_det --config training/configs/structural_yolo11x.yaml

python -m training.export_onnx --run training/runs/crack_segformer_b5 --kind seg
python -m training.register --run training/runs/crack_segformer_b5 --key crack_segmentation
python -m training.register --list              # installed models with real sha256
```

Kaggle datasets need `KAGGLE_API_TOKEN` (or `~/.kaggle/access_token`); Roboflow needs
`ROBOFLOW_API_KEY`. Each dataset's licence is recorded in the catalogue and written to
`training/data/manifest.json` on download.

Prepared corpora normalise to one of three layouts (binary-mask segmentation, folder
classification, YOLO detection) with deterministic splits: a dataset that ships its own
train/test division keeps it, and everything else is bucketed by a salted hash of the
sample id, so re-running prepare can never move a sample from val into train and
quietly invalidate an earlier metric.

Both trainers checkpoint and resume every epoch, because Vast.ai interruptible
instances get reclaimed without warning and Kaggle sessions are capped at nine hours.

`export_onnx.py` fails the run if the ONNX graph disagrees with torch by more than
1e-3, and additionally checks the graph loads under `cv2.dnn` — the runtime
`core/detection.py` actually uses. `register.py` refuses to install a model whose
parity check failed.

Licensing rules for anything produced here:

- Record the true source, licence, and sha256 of every shipped weight file.
- Do not redistribute weights whose source licence is absent or incompatible.
- ELPV is CC BY-NC-SA 4.0, so models trained on it must not be shipped commercially.

## Outputs

Written under `final_toolkit_outputs/` by default:

```
summary.json                 run manifest
inspection_report.md         generated report
reconstruction.ply           point cloud
mesh.ply / mesh.obj          Poisson surface
orthomosaic.tif              georeferenced COG
dsm.tif / dtm.tif            elevation COGs in metres
dsm_hillshade.tif            shaded relief
camera_track.geojson         camera positions with GPS residuals
geo_anchor.json              similarity transform and fit quality
digital_twin.json            artifact index and run metadata
measurements.json            defect areas/lengths in m2/m, terrain statistics
volume_estimation.json       cut/fill against DTM, plane, and lowest-point references
risk_scoring.json            per-defect risk ranking with an action for each
health_scoring.json          asset integrity score and grade
```

Measurements and volumes need the georeferenced rasters, so they are only produced
by the COLMAP engine. When they are absent the report states that rather than
printing zeros — a defect area in square metres means nothing unless something
actually measured it.
