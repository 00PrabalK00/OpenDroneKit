# OpenDroneKit v0.1.0-beta

The first public beta of OpenDroneKit: an MIT-licensed, offline-first desktop toolkit for
drone survey and inspection work on Windows. It is a **beta**. Version 1.0 is still being
built, and this release exists so people outside the project can try it and tell us
what breaks.

**Do not fly real aircraft with this beta.** Mission exports and flight control are
tested against file specifications, a MAVLink peer and ArduPilot SITL, not on your
aircraft at your site.

## Highlights

- **Plan** grid, double-grid, 3D mapping, facade, orbit, corridor, solar, tower and
  other missions, with terrain following from a DEM and warnings that say what each
  plan assumed.
- **Export** to QGroundControl (`.plan`, `.waypoints`), DJI WPML (`.kmz`), Litchi
  (`.csv`), KML and GeoJSON, carrying altitudes, yaw, gimbal pitch and camera triggers.
- **Reconstruct** with COLMAP or the built-in engine into georeferenced orthomosaic, DSM
  and DTM GeoTIFFs. On the public OpenDroneMap Aukerman survey: 77/77 images registered,
  1.27 px mean reprojection error, 1.22 m geo RMSE.
- **Measure** distance, area, volume, cut/fill, stockpile, slope, profile and ponding on
  your own rasters, and lengths on photographs with a stated scale. Measurements refuse
  when the model has no real-world scale.
- **Detect** with nine trained ONNX models (cracks, corrosion severity, solar cell
  defects, solar thermal anomalies, rail obstacles and corridor, structural defects,
  land cover). Every result names its model and sha256; each model's weak classes are
  written in its registry entry.
- **Report**: PDF and Word deliverables from the open project, or a checklist of what is
  missing.
- **Offline and private**: no account, no telemetry. The UI is served on 127.0.0.1 only.

## Known limitations

- Dense point clouds need a CUDA build of COLMAP. Without one, reconstruction is sparse
  and the run says so.
- Model weights are a separate download, verified against the sha256 in
  `models/model_registry.json`. Weights carry the terms of the datasets they were trained
  on, which can be stricter than MIT.
- Model quality varies and is stated, not hidden: `structural_multiclass_detector` reaches
  mAP50 0.417; `crack_segmentation` misses thin and faint cracks at its shipped threshold;
  `shared_semantic_landcover` is usable for vegetation and building only (two of six
  classes); `rail_obstacle_detector` misses about one obstacle in four. An empty result
  is not proof of absence.
- DJI R-JPEG thermal images need the DJI Thermal SDK, which is not bundled (DJI licence).
- Flight control is verified against ArduPilot SITL in CI only, never against real
  hardware in this release.
- The installer is Windows x64 only. Linux and macOS users run from source.
- If the installer's file name ends in `-UNSIGNED`, it has no code-signing certificate
  and Windows SmartScreen will warn. Check its SHA-256 below before running it.

## Install

1. Download `OpenDroneKit-<version>-win64-setup.exe` below.
2. Check its hash: `certutil -hashfile OpenDroneKit-<version>-win64-setup.exe SHA256`.
   It must match the value in **Checksums**.
3. Run it (per-user install, no administrator rights needed).
4. Get the model weights as described in [docs/BETA_TESTING.md](docs/BETA_TESTING.md).

From source: `git checkout v0.1.0-beta`, `conda env create -f environment.yml`, then
`OpenDroneKit.bat`. See [docs/INSTALLATION.md](docs/INSTALLATION.md).

**Testing it?** Follow the 30-minute script in
[docs/BETA_TESTING.md](docs/BETA_TESTING.md) and file a **Beta feedback** issue.

## Checksums

<!-- Fill from the built files before publishing. Do not publish with placeholders. -->

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `OpenDroneKit-<version>-win64-setup.exe` | `<size>` | `<sha256>` |
| `<weights archive or per-model files>` | `<size>` | `<sha256>` |

Per-model weight digests are also in `models/model_registry.json` at the `v0.1.0-beta` tag.
