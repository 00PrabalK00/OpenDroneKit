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
  and DTM GeoTIFFs. On the public OpenDroneMap Aukerman survey COLMAP
  registers all 77 images, georeferenced to UTM zone 17N (EPSG:32617).
- **Measure** distance, area, volume, cut/fill, stockpile, slope, profile and ponding on
  your own rasters, and lengths on photographs with a stated scale. Measurements refuse
  when the model has no real-world scale.
- **Detect** with registered ONNX models. Two sets of weights are published with this
  release: `crack_presence_classifier` (crack / no crack on concrete) and
  `solar_thermal_anomaly_classifier` (thermal PV module anomalies). Every result names
  its model and sha256. The app installs a weight file only if it matches the sha256 in
  the registry.
- **Report**: PDF and Word deliverables from the open project, or a checklist of what is
  missing.
- **Offline and private**: no account, no telemetry. The UI is served on 127.0.0.1 only.

## Known limitations

- Dense point clouds need a CUDA build of COLMAP. Without one, reconstruction is sparse
  and the run says so.
- Only two of the nine trained models have published weights. The other seven
  (structural multiclass, crack segmentation, solar cell defects, rail obstacles, rail
  corridor, corrosion severity, land cover) are not distributed: their base models or
  training data are AGPL-3.0, non-commercial or research-only. See
  [WEIGHTS_LICENSES.md](https://github.com/00PrabalK00/OpenDroneKit/blob/v0.1.0-beta/WEIGHTS_LICENSES.md). Those detectors say they have no model.
- Weights are separate downloads, not in the installer. They are MIT with training-data
  attribution (CC BY 4.0, MIT) kept in WEIGHTS_LICENSES.md.
- Published model quality, from the validation split (not a separate test set):
  `crack_presence_classifier` misses about one cracked tile in fourteen and was trained
  on close-range tiles, not survey-altitude imagery; `solar_thermal_anomaly_classifier`
  misses about two in three soiled modules and one hot spot in three. Model cards are in
  [docs/model_cards/](https://github.com/00PrabalK00/OpenDroneKit/tree/v0.1.0-beta/docs/model_cards). An empty result is not proof of absence.
- DJI R-JPEG thermal images need the DJI Thermal SDK, which is not bundled (DJI licence).
- Flight control is verified against ArduPilot SITL in CI only, never against real
  hardware in this release.
- The installer is Windows x64 only. Linux and macOS users run from source.
- If the installer's file name contains `-UNSIGNED`, it has no code-signing certificate
  and Windows SmartScreen will warn. Check its SHA-256 below before running it.

## Install

1. Download the installer below (`OpenDroneKit-0.1.0b0-win64-setup-UNSIGNED.exe`).
2. Check its hash: `certutil -hashfile <installer file name> SHA256`.
   It must match the value in **Checksums**.
3. Run it (per-user install, no administrator rights needed).
4. Run **Download AI models** from the Start menu. It fetches the two published weight
   files and checks each against its recorded sha256.

From source: `git checkout v0.1.0-beta`, `conda env create -f environment.yml`, then
`OpenDroneKit.bat`. See [docs/INSTALLATION.md](https://github.com/00PrabalK00/OpenDroneKit/blob/v0.1.0-beta/docs/INSTALLATION.md).

**Testing it?** Follow the 30-minute script in
[docs/BETA_TESTING.md](https://github.com/00PrabalK00/OpenDroneKit/blob/v0.1.0-beta/docs/BETA_TESTING.md) and file a **Beta feedback** issue.

## Checksums

<!-- Fill from the built files before publishing. Do not publish with placeholders. -->

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `OpenDroneKit-0.1.0b0-win64-setup-UNSIGNED.exe` | `<size>` | `<sha256>` |
| `crack_presence_classifier.onnx` | 44,701,863 | `92a4d1425b60a8ff96aa31de0ae404325f5a1ab47edc01f5b3acef061da34d74` |
| `solar_thermal_anomaly_cls.onnx` | 44,721,172 | `6933a09abff99bf8d4e2619a50a64498432d2877b9abd55ccab7a87ac357fd42` |

The installer row comes from the build's `SHA256SUMS.txt`. The two weight rows are the
registry values in `models/model_registry.json` at the `v0.1.0-beta` tag; re-hash the
files you upload and confirm they match before publishing.
