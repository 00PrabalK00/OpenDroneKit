# Changelog

All notable changes to OpenDroneKit are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). OpenDroneKit
has not made a tagged release yet, so everything is under **Unreleased** until the first
version is cut; the project intends to use [Semantic Versioning](https://semver.org/)
from then on.

## [Unreleased]

The largest round of work so far: every area of the application was audited for features
that were broken, unreachable from the UI, or producing fabricated output, and each was
fixed or made to refuse with a reason. Feature status in `docs/FEATURES.md` is computed
from passing tests, as before.

### Added

- **Licence**: OpenDroneKit is released under the MIT License (`LICENSE`).
- **Brand**: a new quad mark and wordmark (light, dark and mono SVG masters in
  `app/web/assets/brand/`), with every favicon, app icon, toolbar mark and Hub header
  generated from them by `tools/brand_assets.py`. The desktop window uses the app icon.
- **Launcher and environment**: `environment.yml` defines the reference `odk` conda
  environment (Python 3.11). `python main.py --check-deps` names missing or too-old
  packages, and `OpenDroneKit.bat` runs it before every start and offers to install them.
- **Explicit UI choice**: `--ui classic|cockpit` and `ODK_UI` (case-insensitive) choose
  the interface; an unknown value is refused visibly, even under `pythonw`.
- **Offline basemap**: the desktop shell serves cached tiles from disk, *Cache view*
  downloads the visible area and three deeper zoom levels, and the map switches to the
  cache when online basemaps are unreachable and says so on the map.
- **In-process Hub**: the desktop shell starts the Hub REST service on loopback, and the
  browser Hub gains a sign-in / create-account dialog and a Data and jobs panel.
- **Measurement from the map**: distance, area, perimeter, profile, slope, volume,
  cut/fill and stockpile measured on the project's own rasters from drawn geometry, kept
  in the project's measurement history; survey-to-survey comparison across different grids.
- **Ground control**: GCPs are stored with the project, marked on photographs, and used
  to georeference COLMAP runs (three or more marked points), with a fit/check report and
  leave-one-out residuals.
- **Custom engine**: real incremental structure-from-motion (SIFT matching, PnP
  registration, triangulation, bundle adjustment) with georeferencing and COG products,
  replacing the earlier frame strip.
- **Cloud reconstruction**: `--recon-mode cloud` uploads to a self-hosted OpenDroneKit
  server through the resumable, sha256-verified upload API, follows the job, and
  downloads its products. It needs an explicit endpoint and token.
- **Hub jobs**: detection, measurement and report job kinds have workers; an AI
  inference worker (`python -m services.worker.ai_worker`); webhooks fire; artifacts are
  served by storage reference.
- **Flight**: a MAVSDK driver with live telemetry and mission/fence/rally upload; both
  UIs fly the aircraft through the real `Api`; flight logs mark simulated samples.
- **Missions**: dome and box inspection have their own geometry; exact terrain following
  over DEM triangulation; per-template gimbal and dwell defaults; every template the
  planner accepts is listed in the application.
- **Cockpit**: the Digital Twin draws the project's model in 3D with picking, in-model
  measurement and saved views; Coverage QA runs on flown images; Temperature Range reads
  radiometric frames; the inspection report opens when a pipeline job finishes; AI
  labelled boxes can be trained into a deployed custom model from inside the app.
- **Reports**: PDF and Word deliverables built from the open project, with distinct
  report types; refused report types say what is missing.
- **India survey packs** run from the cockpit with all their inputs.
- **Docs**: the dense-MVS guide for RTX 50-series GPUs, measured on the development
  machine; the installation guide describes the launcher, data root and offline cache.

### Changed

- All application state lives under one absolute data root (`ODK_DATA_DIR`, default
  `final_toolkit_outputs/`), so the app finds its projects whichever folder it starts in.
  Tests run against an isolated data root.
- The Hub API allows loopback origins on any port so the desktop shell can reach it
  (`ODK_CORS_LOOPBACK=0` disables this).
- Flight commands wait for `COMMAND_ACK` and report accepted, denied or unacknowledged;
  abort holds position (BRAKE/LOITER) before falling back to pause; the flight state is
  "flying" only once telemetry shows armed and AUTO; `MissionManager` defaults to no
  drone rather than a silent simulator.
- Mission exporters carry the plan's poses, yaw, gimbal pitch, dwell, speed, fences and
  limits; the WPL file reserves sequence 0 for home; Litchi missions split at 99
  waypoints; AMSL plans export absolute altitudes.
- Structure missions (facade, wind turbine, dome and the other structure templates) keep their stand-off along the whole
  path and face the structure.
- Preflight: Start Mission is held to the checklist, and an override is recorded.
- Detection-job findings are recorded with severity `unassessed` until a reviewer grades
  them; model detections become reviewable findings, heuristics do not.
- The crack presence classifier is a recorded second opinion, not a default gate.
- Health scoring is an uncalibrated severity index; heuristic or refused solar and metal
  passes no longer move it.
- Core package exports resolve lazily, so the window opens without loading the
  processing stack.

### Fixed

- Run AI, Run Full Pipeline, Size Job, New Job and the Reports workspace complete end
  to end, report progress, and can be cancelled.
- Every installed model is reachable from AI Inspection; a custom model deployment no
  longer breaks grayscale image reads app-wide; the semantic map runs on machines where
  CUDA is listed but does not load.
- Dense-stereo capability is probed by running it, names the real cause (for example a
  CUDA toolkit newer than the driver), does not call a JIT-compiling COLMAP broken, and
  every capabilities screen shows the reason. Dense runs no longer label their fused
  cloud sparse.
- Offline map: tiles are labelled by their real type, a burst of tile requests on a busy
  machine is no longer refused, and zooming past the deepest cached level enlarges real
  cached tiles instead of requesting tiles that were never cached.
- Share links, webhooks and API keys work on a fresh install; New Key issues a real API
  token.
- Fleet: adding aircraft, batteries and pilots, maintenance, mission assignment and
  export all work; a failed read says so instead of showing an empty list.
- Reports: an empty project is refused a report on every path; a failed reconstruction
  is reported; the Markdown and HTML reports agree on what a run did.
- Terrain: every raster format is checked, ESRI grids are cached with their `.prj`, and
  projected-coordinate CSVs and feet DEMs are refused.
- Measurements judge true scale from the CRS itself; Web Mercator and feet CRSs are
  refused for surface analyses; over-the-ground length is withheld across DSM holes;
  defects meeting a DSM hole are placed rather than dropped.
- Coverage QA places images from EXIF GPS and camera geometry, and refuses a
  non-numeric height in words.
- The SDK's `wait_for_job` no longer hangs on finished jobs; artifact names with spaces
  download.
- The classic window no longer sticks on "Ready" with no map when the network is
  unreachable at start.
- A Compare no longer makes the open project unreadable to the UI.

### Removed

- Fabricated outputs, each replaced by a measurement or a refusal with a reason:
  failure probabilities from a hard-coded hazard table, forecast-driven health scores,
  the hard-coded coverage layout and ORB-ratio overlap flag, the frame-strip
  "orthomosaic" at an assumed overlap, pseudo-confidence and whole-frame boxes from the
  classical detector, hard-coded `medium` severity on model findings, preflight checks
  hard-coded to pass, and crack length guessed from mask pixels.
- The old PNG logo masters.

### Security

- Reconstruction results served as artifacts no longer disclose absolute server paths.
- Cloud reconstruction uploads nothing without an explicit endpoint and token.
- The desktop UI server and the in-process Hub bind to `127.0.0.1` only.
- Sharing records name the desktop project by id, never its folder path.

[Unreleased]: https://github.com/00PrabalK00/OpenDroneKit/commits/main
